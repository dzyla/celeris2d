"""Project store, picking and extraction on a synthetic micrograph with known particle positions (needs a CUDA GPU;
a few seconds). Checks the job layout (descriptive job folders, heavy data apart, lineage, rename, purge / rebuild,
Trash), particle identity (keys decode, uids stable across extraction jobs), that the blob picker finds the
particles, that binned and unbinned extractions of the same picks hold the same particles, that a selection
extracts only its particles, and that the RELION / cryoSPARC exports and the job folder itself read back as
particle sources.

    pytest -q tests/test_project.py
"""
import json
import os

import numpy as np
import pytest
import torch

from celeris2d import picking as pk
from celeris2d import pipeline as pl
from celeris2d import preprocess as pp
from celeris2d.export import export
from celeris2d.project import Project, parse_key
from celeris2d.sources import open_source

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

APIX, DIAM = 2.0, 120.0


def _micrograph(rng, n=60, size=1536):
    """Dark Gaussian blobs (cryo contrast) on noise, at least 1.3 diameters apart, away from the edges."""
    yy, xx = np.mgrid[:size, :size]
    pos = []
    while len(pos) < n:
        p = rng.uniform(DIAM / APIX, size - DIAM / APIX, 2)
        if all(np.hypot(*(p - q)) > 1.3 * DIAM / APIX for q in pos):
            pos.append(p)
    img = np.zeros((size, size), np.float32)
    s = DIAM / APIX / 4
    for x, y in pos:
        img -= np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
    return img + 1.5 * rng.standard_normal(img.shape).astype(np.float32), np.array(pos) * APIX


def test_jobs_picking_extraction_exports(tmp_path):
    rng = np.random.default_rng(0)
    movies = []
    for i in range(2):                       # stand-in movie files: only their bytes (fingerprint) are used
        f = tmp_path / "raw" / f"grid{i}" / f"mov_{i:03d}.tif"
        f.parent.mkdir(parents=True)
        f.write_bytes(rng.bytes(4096))
        movies.append(str(f))
    prj = Project(str(tmp_path / "prj"))
    imp = prj.import_movies(movies)
    assert imp.name == "001_import_raw" and imp.status == "finished"
    assert sorted(prj.movies()["subdir"]) == ["grid0", "grid1"]

    mc = pl.motion_job(prj, imp, pl.MotionSettings(apix=APIX, dose_total=40, figures="none"))
    truth, ctf = {}, pp.CTFFit(15000.0, 15000.0, 0.0, 6.0, 1.0, {})
    for mid in prj.movies()["id"]:
        img, pos = _micrograph(rng)
        mv = prj.movie(mid)
        path = mc.per_movie("Micrographs", mv, ".mrc")
        prj.write_mrc(path, img, APIX, "float16")
        prj.add_micrograph(mc, mid, path, APIX, img.shape, ctf)
        truth[mid] = pos
    assert os.path.exists(os.path.join(prj.root, "data", "002", "Micrographs", "grid1", "mov_001.mrc"))
    pl.finish(prj, mc)

    pj = pl.pick_job(prj, mc, pl.PickSettings(diameter_a=DIAM, figures="none"))
    ex = pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM, box_a=1.5 * DIAM))
    eb = pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM, box_a=1.5 * DIAM, particle_apix=4 * APIX),
                        label="bin4")
    pl.run(prj, [pj, ex])
    pl.run(prj, [eb])
    assert [mc.name, pj.name, ex.name, eb.name] == ["002_motionctf_2apx", "003_pick_blob120",
                                                    "004_extract_180A_2apx_f16", "005_extract_bin4"]
    assert prj.job("bin4").id == prj.job("005").id == prj.job("jobs/005_extract_bin4/").id == 5
    assert prj.lineage(eb) == [eb.name, pj.name, mc.name, imp.name]
    assert ex.params["box"] == pk.suggest_box(1.5 * DIAM, APIX) and eb.params["box"] == pk.suggest_box(1.5 * DIAM, 4 * APIX)

    # the picker finds the particles, positions within a few A
    for mid, pos in truth.items():
        z = prj.picks(pj, mid)
        got, hit = pk.match(z["xy_a"][z["keep"]], pos, 0.25 * DIAM)
        assert hit.mean() > 0.95, f"recall {hit.mean():.2f}"

    # identity: keys decode, the same particle has the same uid in both extractions
    a, b = prj.particles(ex), prj.particles(eb)
    assert len(a) == len(b) and set(a["uid"]) == set(b["uid"])
    a["key"] = prj.particle_keys(a)
    stem, x, y = parse_key(a["key"].iloc[0])
    assert stem == a["stem"].iloc[0] and abs(x - a["x_a"].iloc[0]) < 0.06 and abs(y - a["y_a"].iloc[0]) < 0.06
    assert prj.find(int(a["uid"].iloc[0]))["key"] == a["key"].iloc[0]
    assert len(prj.find(a["key"].iloc[0])["extractions"]) == 2

    # nothing foreign written by default; exports on request; all read back with the same particles
    assert not os.path.exists(os.path.join(ex.path, "particles.star"))
    star, cs = export(prj, ex, "relion"), export(prj, ex, "cryosparc")
    export(prj, pj, "relion")
    export(prj, mc, "relion")
    for src in (open_source(star), open_source(cs), open_source(ex.path)):
        assert src.N == len(a) and src.box == ex.params["box"] and abs(src.apix - APIX) < 1e-6
        idx, ims = next(iter(src.chunks()))
        assert abs(float(ims[:, 20:-20, 20:-20].mean())) > 0.05      # white blobs in the centre
    sb = open_source(eb.path)
    assert sb.box == eb.params["box"] and abs(sb.apix - 4 * APIX) < 1e-6

    # tables are sharded and hold numbers only; rethreshold reads them in one pass
    assert os.path.exists(os.path.join(ex.path, "particles", "index.json"))
    assert not any(v.startswith(("<U", "|O")) for v in prj.table(ex, "particles").index["columns"].values())
    th = pl.rethreshold(prj, pj, threshold=0.99, label="strict")
    assert prj.summary_of(th)["picks"] < prj.summary_of(pj)["picks"]
    prj.delete_job(th)

    # a selection (uids, e.g. good classes of a 2D run on the binned extraction) extracts only its particles
    sel = prj.select(eb, b["uid"].to_numpy()[::3], label="every3rd")
    es = pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM, box_a=1.5 * DIAM, select=sel.id))
    assert es.name == "008_extract_180A_2apx_f16_sel007" and sel.name == "007_select_every3rd"
    pl.run(prj, [es])
    assert set(prj.particles(es)["uid"]) == set(prj.selected_uids(sel))
    assert prj.find(int(prj.selected_uids(sel)[0]))["selected_in"] == [sel.name]

    # rename (references are by number), purge + rebuild the stacks, summaries for people and agents
    ex = prj.rename(ex, "full")
    assert ex.name == "004_extract_full" and prj.dependents(pj) == ["004_extract_full", "005_extract_bin4",
                                                                    "008_extract_180A_2apx_f16_sel007"]
    before = open_source(ex.path).read_all()
    assert prj.purge(ex) > 0 and prj.disk().set_index("job").loc[ex.name, "purgeable_gb"] == 0
    pl.rebuild(prj, ex)
    after = open_source(ex.path).read_all()
    assert np.allclose(before, after, atol=1e-2)
    js = json.load(open(os.path.join(ex.path, "summary.json")))
    assert js["particles"] == len(a)
    assert "| 005_extract_bin4 | finished | 003 |" in open(os.path.join(prj.root, "JOBS.md")).read()

    # delete -> Trash (job folder and its data), refused while used; numbers are never reused
    with pytest.raises(ValueError):
        prj.delete_job(pj)
    prj.delete_job(es)
    assert os.path.isdir(os.path.join(prj.root, "Trash", es.name, "data"))
    assert not os.path.exists(es.data_path)
    assert pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM)).id == 9


def _synthetic_project(tmp_path, rng):
    movies = []
    for i in range(2):
        f = tmp_path / "raw" / f"mov_{i:03d}.tif"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rng.bytes(4096))
        movies.append(str(f))
    prj = Project(str(tmp_path / "prj"))
    imp = prj.import_movies(movies)
    mc = pl.motion_job(prj, imp, pl.MotionSettings(apix=APIX, dose_total=40, figures="none"))
    ctf = pp.CTFFit(15000.0, 15000.0, 0.0, 6.0, 1.0, {})
    for mid in prj.movies()["id"]:
        img, _ = _micrograph(rng)
        path = mc.per_movie("Micrographs", prj.movie(mid), ".mrc")
        prj.write_mrc(path, img, APIX, "float16")
        prj.add_micrograph(mc, mid, path, APIX, img.shape, ctf)
    pl.finish(prj, mc)
    pj = pl.pick_job(prj, mc, pl.PickSettings(diameter_a=DIAM, figures="none"))
    ex = pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM, box_a=1.5 * DIAM, particle_apix=2 * APIX))
    pl.run(prj, [pj, ex])
    return prj, pj, ex


def test_class2d_selection_recentre(tmp_path):
    """Class2D on an extraction, a class selection that keeps the rejected particles too, and a re-centring
    extraction from known shifts (centre = old centre + shift, uid kept, near-duplicates merged)."""
    from celeris2d import jobs
    rng = np.random.default_rng(1)
    prj, pj, ex = _synthetic_project(tmp_path, rng)
    c = jobs.class2d_job(prj, ex, jobs.Class2DSettings(n_classes=3, max_res_a=12))
    s = jobs.run_class2d(prj, c, log=lambda *a: None)
    assert c.name == "005_class2d_K3_on004" and s["particles"] == len(prj.particles(ex))
    t = prj.table(c, "particles").read()
    assert set(t) >= {"uid", "cls", "psi", "shift_x_a", "shift_y_a", "maxprob"} and t["cls"].max() < 3
    assert os.path.exists(os.path.join(c.path, "classes.mrcs")) and os.path.exists(os.path.join(c.path, "gallery.png"))
    k = int(np.bincount(t["cls"]).argmax()) + 1
    sel = jobs.select_classes(prj, c, [k])
    kept, rej = prj.selected_uids(sel), prj.selected_uids(sel, rejected=True)
    assert len(np.intersect1d(kept, rej)) == 0 and len(kept) + len(rej) == len(t["uid"])
    assert sel.params["classes"] == [k] and sel.params["rejected_classes"] == [i for i in (1, 2, 3) if i != k]
    for f in ("selection.png", "selected_classes.mrcs", "rejected_classes.mrcs"):
        assert os.path.exists(os.path.join(sel.path, f)), f

    # an alignment job with known shifts: particle 1 is moved onto particle 0 (a duplicate after re-centring)
    e = prj.particles(ex)
    al = prj.new_job("Class2D", dict(particles=ex.id), [ex], "known_shifts")
    sx, sy = np.full(len(e), 6.0, np.float32), np.full(len(e), -4.0, np.float32)
    sx[1] = e["x_a"].iloc[0] - e["x_a"].iloc[1] + 6.0
    sy[1] = e["y_a"].iloc[0] - e["y_a"].iloc[1] - 4.0
    prj.table(al, "particles").write(dict(uid=e["uid"].to_numpy(), cls=np.zeros(len(e), np.int16), psi=np.zeros(len(e), np.float32),
                                          shift_x_a=sx, shift_y_a=sy, maxprob=np.linspace(1, 0.5, len(e)).astype(np.float32)),
                                     e["movie_id"].to_numpy())
    prj.set_status(al, "finished", {})
    rc = pl.extract_job(prj, pj, pl.ExtractSettings(particle_a=DIAM, box_a=1.5 * DIAM, recenter=al.id))
    pl.run(prj, [rc])
    assert rc.name.endswith("_rc007")
    r = prj.particles(rc).set_index("uid")
    old = e.set_index("uid")
    assert len(r) == len(e) - 1 and e["uid"].iloc[1] not in r.index        # the lower-scoring duplicate is gone
    u = r.index.intersection(old.index)
    assert np.allclose(r.loc[u, "x_a"], old.loc[u, "x_a"] + 6.0, atol=0.01)
    assert np.allclose(r.loc[u, "y_a"], old.loc[u, "y_a"] - 4.0, atol=0.01)
    assert np.allclose(r.loc[u, "dx_a"], 6.0, atol=0.01) and np.allclose(r.loc[u, "dy_a"], -4.0, atol=0.01)
