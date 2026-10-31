"""ManualPick: an asymmetric object projected into noisy, CTF-flipped synthetic micrographs. Picks on one micrograph
train templates that find the particles of the others (suggestions and Pick all), the reviewed micrograph keeps
exactly its picks, the rejected suggestions set the threshold, and Extract reads the job like a Pick job's output.
Also: the 2D class averages recover an image's orientation (rotation / shift conventions)."""
import json
import os

import numpy as np
import pytest
import torch

from celeris2d import gpu
from celeris2d import manualpick as mp
from celeris2d import picking as pk
from celeris2d.project import Project

APIX, D, NOISE = 2.0, 60.0, 10.0
needs_gpu = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")


def _volume(b=40):
    c = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    v = torch.zeros(b, b, b)
    for p, s in (((0, 0, 0), 5.0), ((9, 3, 0), 3.0), ((-4, -9, 4), 3.5), ((2, 6, -8), 3.0)):
        v += torch.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) / (2 * s * s))
    return v.numpy()


def _micrograph(rng, proj, defocus, H=1024, W=1024):
    from celeris2d.engine import Optics
    mic = np.zeros((H, W), np.float32)
    pos = []
    for gy in range(80, H - 80, 105):
        for gx in range(80, W - 80, 105):
            y, x = gy + rng.integers(-15, 16), gx + rng.integers(-15, 16)
            im = np.rot90(proj[rng.integers(len(proj))], rng.integers(4)).copy()
            b = im.shape[0]
            mic[y - b // 2:y - b // 2 + b, x - b // 2:x - b // 2 + b] += im
            pos.append((x * APIX, y * APIX))
    F = torch.fft.rfft2(torch.as_tensor(mic))
    F = F * pk.ctf_sign(H, W, APIX, defocus, defocus, 0.0, Optics(300.0, 2.7, 0.1), "cpu")    # phases flipped
    mic = torch.fft.irfft2(F, s=(H, W)).numpy()
    mic = -mic / mic.std() + rng.normal(0, NOISE, mic.shape).astype(np.float32)        # dark protein, SNR ~ 0.1
    return mic.astype(np.float32), np.array(pos)


def _project(tmp_path, n_mics=4):
    """Import + MotionCtf jobs with synthetic micrographs (no movies), and the true particle positions."""
    from celeris2d.preprocess import CTFFit
    prj = Project(str(tmp_path / "prj"))
    imp = prj.new_job("Import", dict(movies="synthetic"), [], "syn")
    rng = np.random.default_rng(0)
    proj, _ = pk.templates_from_map(_volume(), APIX, n_dirs=200)
    mc = prj.new_job("MotionCtf", dict(apix=APIX, dose_total=50.0, bin=1.0, kv=300.0, cs_mm=2.7, amp_contrast=0.1,
                                       patch_ctf="off", movies=imp.id), [imp], "syn")
    truth = {}
    for i in range(1, n_mics + 1):
        with prj._db() as c:
            c.execute("INSERT INTO movies VALUES (?,?,?,?,?,?,?)", (i, f"/syn/mic{i:02d}.tif", f"mic{i:02d}", "",
                                                                     f"{i:064x}", imp.id, 0.0))
        df = 12000.0 + 1500 * i
        mic, pos = _micrograph(rng, proj, df)
        path = os.path.join(mc.data_path, "Micrographs", f"mic{i:02d}.mrc")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prj.write_mrc(path, mic, APIX, "float32")
        prj.add_micrograph(mc, i, path, APIX, mic.shape, CTFFit(df, df, 0.0, 6.0, 0.0, {}))
        truth[f"mic{i:02d}"] = pos
    prj.set_status(imp, "finished", dict(movies=n_mics))
    prj.set_status(mc, "finished", dict(micrographs=n_mics))
    return prj, mc, truth


def test_class_averages_align():
    """Rotated, shifted noisy copies of one asymmetric image average back to it (up to an overall rotation)."""
    rng = np.random.default_rng(1)
    proj, _ = pk.templates_from_map(_volume(), APIX, n_dirs=4)
    ref = torch.as_tensor(proj[1][::-1][:39, :39].copy())                       # odd box
    n = 200
    ang = torch.as_tensor(rng.uniform(0, 360, n), dtype=torch.float32)
    dy, dx = (torch.as_tensor(rng.integers(-2, 3, n), dtype=torch.float32) for _ in range(2))
    X = mp._shift(mp._rot(ref[None].expand(n, -1, -1), ang), dy, dx) + torch.randn(n, 39, 39) * 0.5 * ref.std()
    refs, cls, score = mp.class_averages(X, 1, 15.0)
    t = refs[0]
    best = max(float(torch.corrcoef(torch.stack([mp._rot(t[None], torch.tensor([a]))[0].flatten(),
                                                 ref.flatten()]))[0, 1]) for a in range(0, 360, 3))
    assert best > 0.9, best


@needs_gpu
def test_manualpick_flow(tmp_path):
    from celeris2d import pipeline as pl
    from celeris2d.jobs import new_job
    from celeris2d.scheduler import queue_job
    prj, mc, truth = _project(tmp_path)
    j = queue_job(prj, "ManualPick", {"micrographs": mc.id}, {"diameter_a": str(D), "suggest": "2"})
    assert j.status == "waiting"                                       # the user picks first, not the queue
    rng = np.random.default_rng(2)
    mine = truth["mic01"][:40] + rng.normal(0, 3.0, (40, 2))           # 40 hand picks, a few A off
    mp.set_picks(prj, j, "mic01", picks=[[x, y, 0] for x, y in mine], state="reviewed")
    info = mp.train(prj, j)
    assert info["n"] >= 2 and info["threshold"] is not None and info["missing"] <= 2
    got = mp.suggest(prj, j, after="mic01")
    assert got == ["mic02", "mic03"]
    mi = mp.micrographs(prj, j)
    row = mi[mi["stem"] == "mic02"].iloc[0]
    sg = mp.suggestions(prj, j, row)
    th = mp.load_state(j)["threshold"]
    xy = np.asarray(sg["xy"])[(np.asarray(sg["score"]) >= th) & (np.asarray(sg["ok"]) > 0)]
    mpk, mref = pk.match(xy, truth["mic02"], 0.3 * D)
    assert mref.mean() > 0.85 and mpk.mean() > 0.8, (mref.mean(), mpk.mean())
    # the user removes a few false suggestions on mic02 (negatives) and accepts the rest; retrain uses both
    bad = xy[~mpk][:5]
    mp.set_picks(prj, j, "mic02", picks=[[x, y, 1] for x, y in xy[mpk]], rejected=bad.tolist(), state="reviewed")
    info = mp.train(prj, j)
    assert info["micrographs"] == 2 and info["accepted"] == int(mpk.sum())
    # mic03 not checked: a pick of the user's and a removed suggestion there stay through Pick all
    row3 = mi[mi["stem"] == "mic03"].iloc[0]
    s3 = mp.suggestions(prj, j, row3)
    gone = np.asarray(s3["xy"])[int(np.argmax(s3["score"]))]
    mp.set_picks(prj, j, "mic03", picks=[[60.0, 60.0, 0]], rejected=[gone.tolist()], state="suggested")
    # Pick all, through the queue's run function
    mp.queue_run(prj, j)
    assert prj.job(j.id).status == "queued"
    mp.run_manualpick(prj, j)
    j = prj.job(j.id)
    assert j.status == "finished"
    p1 = prj.picks(j, 1)
    assert p1["keep"].sum() == 40
    assert np.allclose(np.sort(p1["xy_a"][:, 0]), np.sort(np.round(mine[:, 0], 1)), atol=0.06)    # as picked
    for stem, mid in (("mic03", 3), ("mic04", 4)):
        p = prj.picks(j, mid)
        mpk, mref = pk.match(p["xy_a"][p["keep"]], truth[stem], 0.3 * D)
        assert mref.mean() > 0.85 and mpk.mean() > 0.8, (stem, mref.mean(), mpk.mean())
    p3 = prj.picks(j, 3)
    k3 = p3["xy_a"][p3["keep"]]
    assert np.hypot(*(k3 - [60.0, 60.0]).T).min() < 0.1                 # the user's pick
    assert np.hypot(*(k3 - gone).T).min() > 0.3 * D                      # the removed suggestion
    summ = prj.summary_of(j)
    assert summ["reviewed"] == 2 and summ["micrographs"] == 4 and j.find("templates.png")
    # Extract takes it as a Pick job's output
    ex = new_job(prj, "Extract", {"picks": j.id}, {"particle_a": str(D)})
    pl.run(prj, [ex])
    n = sum(int(prj.picks(j, m)["keep"].sum()) for m in range(1, 5))
    assert prj.summary_of(ex)["particles"] == n
    # the micrograph panel sees the picks of an Extract made from it
    from celeris2d.micview import family
    assert family(prj, ex)["pick"].id == j.id
    json.dumps(mp.load_state(j))
    # templates off, the score range, then a new particle size: templates dropped, picks kept, label follows
    assert mp.set_off(prj, j, [1, 1]) == [1]
    assert mp.set_threshold(prj, j, upper=0.9)[1] == 0.9
    n_before = len(mp.load_state(j)["mics"])
    mp.set_settings(prj, j, diameter_a=D + 20, templates=3)
    j = prj.job(j.id)
    st = mp.load_state(j)
    assert not mp.has_templates(j) and st["threshold"] is None and st["threshold_hi"] is None and st["off"] == []
    assert len(st["mics"]) == n_before and mp.settings(j).templates == 3 and mp.settings(j).diameter_a == D + 20
    assert j.label == f"d{D + 20:g}"


def test_auto_keep():
    c = pk.Candidates(xy_a=np.zeros((5, 2)), score=np.array([0.2, 0.4, 0.6, 0.8, 0.9]), local_std=np.ones(5),
                      local_mean=np.zeros(5), template=np.array([0, 1, 0, 1, 0]))
    ok = np.array([1, 1, 1, 1, 0], bool)
    assert mp.auto_keep(c, ok, 0.3, None, []).tolist() == [0, 1, 1, 1, 0]
    assert mp.auto_keep(c, ok, 0.3, 0.7, []).tolist() == [0, 1, 1, 0, 0]       # dark contamination above the range
    assert mp.auto_keep(c, ok, 0.3, None, [1]).tolist() == [0, 0, 1, 0, 0]      # template 1 switched off


def test_threshold():
    pos = np.array([0.5, 0.6, 0.7, 0.8, -np.inf])
    assert 0.52 < mp.choose_threshold(pos, []) < 0.54                # 10th percentile of the picks found
    t = mp.choose_threshold(pos, [0.3, 0.35, 0.4, 0.45])           # between the worst pick and the best reject
    assert 0.45 < t < 0.5
    t = mp.choose_threshold(pos, [0.3, 0.35, 0.4, 0.55])           # one reject among the picks: above it
    assert 0.55 < t < 0.6
    assert mp.choose_threshold(np.array([-np.inf]), []) is None


@needs_gpu
def test_manualpick_api_and_picks_only(tmp_path):
    """The dashboard endpoints (state, micrograph, picks, train, suggest, templates, run), and a job finished with the
    user's picks only (no templates: the other micrographs get none)."""
    from fastapi.testclient import TestClient
    from celeris2d.webapp import create
    prj, mc, truth = _project(tmp_path, n_mics=3)
    c = TestClient(create(str(tmp_path / "web"), project_dir=prj.root))
    r = c.post("/api/project/new", json={"type": "ManualPick", "inputs": {"micrographs": mc.id},
                                         "settings": {"diameter_a": str(D), "suggest": "1"}})
    assert r.status_code == 200 and r.json()["status"] == "waiting"
    j = r.json()["job"]
    st = c.get("/api/project/manualpick/state", params={"job": j}).json()
    assert [m["stem"] for m in st["mics"]] == ["mic01", "mic02", "mic03"] and st["version"] == 0
    m = c.get("/api/project/manualpick/mic", params={"job": j, "stem": "mic01"}).json()
    assert m["bin"] == 1 and m["shape"] == [1024, 1024] and m["picks"] == []
    assert c.post("/api/project/manualpick/train", json={"job": j}).status_code == 400       # no picks yet
    picks = [[float(x), float(y), 0] for x, y in truth["mic01"][:30]]
    assert c.post("/api/project/manualpick/picks", json={"job": j, "stem": "mic01", "picks": picks,
                                                          "state": "reviewed"}).json()["picks"] == 30
    tr = c.post("/api/project/manualpick/train", json={"job": j}).json()
    assert tr["version"] == 1 and len(tr["pos"]) == 30
    assert c.get("/api/project/manualpick/templates.png", params={"job": j}).headers["content-type"] == "image/png"
    assert c.post("/api/project/manualpick/suggest", json={"job": j, "after": "mic01"}).json()["stems"] == ["mic02"]
    m = c.get("/api/project/manualpick/mic", params={"job": j, "stem": "mic02"}).json()
    assert m["state"] == "suggested" and m["suggest"]["version"] == 1 and len(m["suggest"]["xy"]) > 50
    assert c.post("/api/project/manualpick/threshold", json={"job": j, "threshold": 0.5}).json()["threshold"] == 0.5
    assert c.post("/api/project/manualpick/run", json={"job": j}).json()["status"] == "queued"
    assert c.post("/api/project/manualpick/picks", json={"job": j, "stem": "mic01", "picks": []}).status_code == 400
    # picks only: a second job, no templates
    j2 = mp.manualpick_job(prj, mc, mp.ManualPickSettings(D))
    mp.set_picks(prj, j2, "mic02", picks=picks[:5], state="reviewed")
    mp.queue_run(prj, j2)
    mp.run_manualpick(prj, j2)
    assert [int(prj.picks(j2, k)["keep"].sum()) for k in (1, 2, 3)] == [0, 5, 0]
    assert "picks" in c.get("/api/project").json()["jobs"][-1]["key"]
    # no diameter yet: the picker measures it first; training waits for it, the label follows it
    from celeris2d import scheduler
    j3 = mp.manualpick_job(prj, mc, mp.ManualPickSettings())
    assert j3.status == "waiting" and j3.label == ""
    with pytest.raises(ValueError, match="measure the particle"):
        mp.train(prj, j3)
    assert c.post("/api/project/settings", json={"job": j3.name, "settings": {"diameter_a": D}}).json()["status"] == "waiting"
    j3 = prj.job(j3.id)
    assert j3.params["diameter_a"] == D and j3.label == f"d{D:g}" and mp.sized(j3).diameter_a == D
