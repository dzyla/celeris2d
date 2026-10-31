"""Symmetry expansion as a job (SymExpand) on a simulated C2 particle with flexible subunits (needs a CUDA GPU).

A small rigid core on the 2-fold axis carries two copies of one subunit, related by the 2-fold; each subunit turns
about its own hinge by an independent random rotation in every particle (up to 25 deg), so the C2 consensus blurs
both. Then:
  - SymExpand C2 of the consensus: one row per (uid, copy), copy k at A S_k, the particle's shift and half; RELION
    export like relion_particle_symmetry_expand; C1 refused;
  - LocalRefine on the expansion (no subtraction), focused on subunit 0: copy 0 looks at subunit 0, copy 1 at
    subunit 1 (both in the mask's frame) - each image loaded once, the subunit better than in the consensus;
  - Subtract on the SymExpand job = Subtract with its own sym_expand (same images); Class3D on the SymExpand job =
    Class3D with sym_expand (same rows and classes); Refine3D takes the expansion too.

    pytest -q tests/test_symexpand.py
"""
import os
import sys

import numpy as np
import pytest
import torch

from celeris2d import gpu
from celeris2d import abinitio as ab
from celeris2d.engine import TWO_PI, Optics, ctf_2d, freqs

sys.path.insert(0, os.path.dirname(__file__))
from test_subtract import _rand_rot, _rel_err, _volume  # noqa: E402

pytestmark = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")

BOX, APIX, N, NOISE = 64, 3.0, 1600, 1.5
C_U, HINGE = np.array([14.0, 0, 0]), np.array([4.0, 0, 0])              # subunit 0 centre, its hinge (voxels x y z)
_r = np.random.default_rng(11)
BLOBS_U = [tuple(C_U + _r.uniform(-5.0, 5.0, 3)) + (_r.uniform(0.7, 1.1), _r.uniform(0.8, 1.3)) for _ in range(16)]
CORE = [tuple(_r.uniform(-3.5, 3.5, 3)) + (_r.uniform(0.9, 1.3), _r.uniform(0.6, 1.0)) for _ in range(5)]
MAX_TURN, ENVELOPE_B = 25.0, 200.0
DEFOCUS = (9000.0, 14000.0, 19000.0, 24000.0)
S1 = ab.symmetry_ops("C2")[1]                                             # the 2-fold (about z)


def _rot_blobs(blobs, R):
    out = []
    for x, y, z, s, a in blobs:
        p = R @ torch.tensor([x, y, z], dtype=torch.float32)
        out.append((float(p[0]), float(p[1]), float(p[2]), s, a))
    return out


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from celeris2d import jobs
    from celeris2d import pipeline as pl
    from celeris2d import preprocess as pp
    from celeris2d.project import Project
    dev = gpu.default()
    tmp = tmp_path_factory.mktemp("symexpand")
    g = torch.Generator().manual_seed(9)
    core = CORE + _rot_blobs(CORE, S1)                                   # C2-symmetric core
    VU, VC = ab.Volume(BOX, dev), ab.Volume(BOX, dev)
    VU.set_real(_volume(BLOBS_U).to(dev))
    VC.set_real(_volume(core).to(dev))
    rot, psi = torch.rand(N, generator=g) * 360 - 180, torch.rand(N, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(N, generator=g) * 2 - 1))
    A = ab.euler_to_matrix(rot, tilt, psi)
    t = (torch.rand(N, 2, generator=g) - 0.5) * 6.0
    h = torch.tensor(HINGE, dtype=torch.float32) * APIX
    Rk = [_rand_rot(N, MAX_TURN, g), _rand_rot(N, MAX_TURN, g)]          # subunit k about its hinge
    Sk = [torch.eye(3), S1]
    # subunit k at rest is S_k U; turned: S_k (R x + h - R h) -> pose A S_k R, shift t + (A S_k (h - R h))_xy
    A_true = [A @ Sk[k] @ Rk[k] for k in (0, 1)]
    t_true = [t + (A @ Sk[k] @ (h[None, :, None] - Rk[k] @ h[None, :, None]))[:, :2, 0] for k in (0, 1)]
    mic = torch.arange(N) % len(DEFOCUS)
    ctf_p = torch.zeros(N, 6)
    ctf_p[:, 0] = ctf_p[:, 1] = torch.tensor(DEFOCUS)[mic]
    ctf_p[:, 4], ctf_p[:, 5] = ENVELOPE_B, 1
    optics = Optics(300.0, 2.7, 0.1)
    fy, fx = freqs(BOX, dev)
    ph = TWO_PI / (BOX * APIX)

    def img(V, A_, t_):
        c = ctf_2d(ctf_p.to(dev), optics, fy, fx, BOX * APIX)
        e = torch.exp(-1j * ph * (fx[None, None, :] * t_[:, 0, None, None].to(dev) + fy[None, :, None] * t_[:, 1, None, None].to(dev)))
        return torch.fft.irfft2(V.project(A_.to(dev)) * c * e, s=(BOX, BOX), norm="forward").cpu()
    x = img(VC, A, t) + img(VU, A_true[0], t_true[0]) + img(VU, A_true[1], t_true[1])
    x = x / float(x.std())
    x = (x + NOISE * torch.randn(x.shape, generator=g)).numpy()

    prj = Project(str(tmp / "prj"))
    movies = []
    for i in range(len(DEFOCUS)):
        f = tmp / "raw" / f"mov_{i:03d}.tif"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(np.random.default_rng(i).bytes(4096))
        movies.append(str(f))
    imp = prj.import_movies(movies)
    mc = pl.motion_job(prj, imp, pl.MotionSettings(apix=APIX / 2, bin=2, dose_total=40, figures="none"))
    mids = list(prj.movies()["id"])
    for i, mid in enumerate(mids):
        prj.add_micrograph(mc, mid, mc.per_movie("Micrographs", prj.movie(mid), ".mrc"), APIX, (4096, 4096),
                           pp.CTFFit(DEFOCUS[i], DEFOCUS[i], 0.0, 6.0, 1.0, {}))
    prj.set_status(mc, "finished", {})
    ex = prj.new_job("Extract", dict(micrographs=mc.id, box=BOX, out_apix=APIX, box_a_used=BOX * APIX, particle_a=120.0,
                                     dark=True), [mc], "sim")
    uid = np.zeros(N, np.int64)
    for i, mid in enumerate(mids):
        sel = np.flatnonzero(mic.numpy() == i)
        k = np.arange(len(sel))
        xy = np.stack([300.0 + 300.0 * (k % 40), 300.0 + 300.0 * (k // 40)], 1)
        uid[sel] = prj.add_particles(ex, mid, xy, x[sel], APIX, columns=dict(score=np.ones(len(sel))))
    prj.consolidate(ex)
    prj.set_status(ex, "finished", dict(particles=N, box=BOX))
    half = (torch.randperm(N, generator=g) % 2).numpy()
    st = prj.new_job("Refine3D", dict(particles=ex.id, sym="C2"), [ex], "truth_rest")
    rr, tt, pp_ = (v.numpy() for v in ab.matrix_to_euler(A))
    prj.table(st, "particles").write(dict(uid=uid, rot=rr, tilt=tt, psi=pp_, shift_x_a=t[:, 0].numpy(),
                                          shift_y_a=t[:, 1].numpy(), half=half.astype(np.uint8),
                                          fit=np.ones(N, np.float32), used=np.ones(N, bool)), mic.numpy() + mids[0])
    rest = _volume(core) + _volume(BLOBS_U) + _volume(_rot_blobs(BLOBS_U, S1))
    ab._save_mrc(st.out("map_nomask.mrc"), rest, APIX)
    prj.set_status(st, "finished", dict(resolution_a=12.0))
    cons = jobs.refine3d_job(prj, ex, st, st, jobs.Refine3DSettings(sym="C2", n_iter=4, res_start_a=12.0, drop_frac=0.0))
    jobs.run_refine3d(prj, cons, log=lambda *a: None)
    mj = prj.new_job("MaskCreate", dict(map=cons.id), [cons], "subunit0")
    rgrid = torch.arange(BOX, dtype=torch.float32) - BOX // 2
    z_, y_, x_ = torch.meshgrid(rgrid, rgrid, rgrid, indexing="ij")
    d = torch.sqrt((x_ - C_U[0]) ** 2 + y_ ** 2 + z_ ** 2)
    mask = torch.clamp((10.5 - d) / 2.5, 0, 1)
    mask = 0.5 - 0.5 * torch.cos(np.pi * mask)
    ab._save_mrc(mj.out("mask.mrc"), mask, APIX)
    prj.set_status(mj, "finished", {})
    return dict(prj=prj, ex=ex, cons=cons, mask=mj, mask_vol=mask, uid=uid, half=half, A_true=A_true)


def _by_uid(uid_all, uids):
    o = np.argsort(uid_all)
    return o[np.searchsorted(uid_all[o], uids)]


def _A(t):
    return ab.euler_to_matrix(torch.tensor(t["rot"]), torch.tensor(t["tilt"]), torch.tensor(t["psi"]))


def test_symexpand_rows_and_export(sim):
    import pandas as pd
    from celeris2d import export, jobs
    from celeris2d.star import read_star
    from celeris2d.symexpand import SymExpandSettings, run_symexpand, symexpand_job
    prj, cons = sim["prj"], sim["cons"]
    with pytest.raises(ValueError, match="nothing to expand"):
        symexpand_job(prj, cons, None, SymExpandSettings(sym="C1"))
    with pytest.raises(ValueError, match="never detected"):
        symexpand_job(prj, cons, None, SymExpandSettings())
    se = jobs.new_job(prj, "SymExpand", {"poses": cons.id}, {"sym": "C2"})
    summ = run_symexpand(prj, se, log=lambda *a: None)
    sim["se"] = se
    assert summ["particles"] == N and summ["copies"] == 2 and summ["rows"] == 2 * N and se.find("symexpand.png")
    assert prj.aligned_particles(se).id == sim["ex"].id
    t = prj.table(se, "particles").read()
    c = prj.table(cons, "particles").read()
    assert sorted(set(t["copy"].astype(int))) == [0, 1] and len(np.unique(t["uid"])) == N
    i = _by_uid(c["uid"], t["uid"])
    A_c = _A(c)[i]
    S = ab.symmetry_ops("C2")[torch.as_tensor(t["copy"].astype(np.int64))]
    assert torch.allclose(_A(t), A_c @ S, atol=1e-4)                     # copy k at A S_k
    assert np.allclose(t["shift_x_a"], c["shift_x_a"][i]) and np.array_equal(t["half"], c["half"][i])
    # RELION export: a row per copy, the particle's image, the expanded angles, origin = -shift, subset = half + 1
    path = export.export(prj, se, "relion")
    b = read_star(path)
    df = b["particles"] if isinstance(b, dict) else b
    assert len(df) == 2 * N and set(df["celCopy"].astype(int)) == {0, 1}
    names = df.groupby(df.index // 2)["rlnImageName"].nunique()
    assert (names == 1).all()                                            # copies of a particle share its image
    o = np.lexsort((t["copy"], i))
    assert np.allclose(df["rlnAngleTilt"].astype(float), t["tilt"][o], atol=1e-3)
    assert np.allclose(df["rlnOriginXAngst"].astype(float), -t["shift_x_a"][o], atol=1e-3)
    assert set(df["rlnRandomSubset"].astype(int)) == {1, 2}
    assert isinstance(df, pd.DataFrame)


def test_local_refine_on_expansion(sim):
    from celeris2d.subtract import LocalRefineSettings, local_refine_job, run_local_refine
    prj = sim["prj"]
    if "se" not in sim:
        pytest.skip("needs the SymExpand job")
    lj = local_refine_job(prj, sim["ex"], sim["se"], sim["mask"], None,
                          LocalRefineSettings(n_iter=8, drop_frac=0.0, max_angle_deg=30.0))
    summ = run_local_refine(prj, lj, log=lambda *a: None)
    n_used = summ["particles"] // 2
    assert summ["images"] == n_used and summ["particles"] == 2 * n_used >= 2 * N - 20     # n images, 2n rows
    t = prj.table(lj, "particles").read()
    assert sorted(set(t["copy"].astype(int))) == [0, 1]
    i = _by_uid(sim["uid"], t["uid"])
    assert np.array_equal(t["half"].astype(int), sim["half"][i])
    k = t["copy"].astype(np.int64)
    A_true = torch.stack([sim["A_true"][0], sim["A_true"][1]])[torch.as_tensor(k), torch.as_tensor(i)]
    se = prj.table(sim["se"], "particles").read()
    j = np.lexsort((se["copy"], se["uid"]))
    o = np.lexsort((t["copy"], t["uid"]))
    e_loc = _rel_err(_A(t)[o], A_true[o])
    e_cons = _rel_err(_A(se)[j], A_true[o])
    print(f"\n[test] expanded C2 subunit: focused resolution {summ['resolution_a']} A vs consensus "
          f"{summ['consensus_region_resolution_a']} A (same mask); pose error median {e_loc:.2f} vs {e_cons:.2f} deg; "
          f"{summ['images']:,} images for {summ['particles']:,} rows, GPU peak {summ['gpu_peak_gb']} GB, "
          f"{summ['wall_seconds']} s")
    assert summ["resolution_a"] < summ["consensus_region_resolution_a"], summ
    assert e_loc < 0.7 * e_cons, (e_loc, e_cons)


def test_subtract_on_symexpand_is_own_expansion(sim):
    import mrcfile
    from celeris2d.subtract import SubtractSettings, run_subtract, subtract_job
    prj = sim["prj"]
    if "se" not in sim:
        pytest.skip("needs the SymExpand job")
    with pytest.raises(ValueError, match="already symmetry-expanded"):
        subtract_job(prj, sim["se"], sim["se"], sim["mask"], SubtractSettings(sym_expand="C2"))
    out = {}
    for nm, (p, q, x) in dict(chain=(sim["se"], sim["se"], "C1"), own=(sim["ex"], sim["cons"], "C2")).items():
        sj = subtract_job(prj, p, q, sim["mask"], SubtractSettings(sym_expand=x, box=40))
        summ = run_subtract(prj, sj, log=lambda *a: None)
        assert summ["images"] == 2 * N and summ["copies"] == 2 and summ["sym_expand"] == "C2", summ
        P = prj.particles(sj)
        imgs = np.stack([mrcfile.mmap(prj.abs(s_), permissive=True).data[k_] for s_, k_ in zip(P["stack"], P["slice"])])
        o = np.lexsort((P["copy"].to_numpy(), P["uid"].to_numpy()))
        out[nm] = (imgs[o].astype(np.float32), P.iloc[o])
    (a, Pa), (b, Pb) = out["chain"], out["own"]
    assert np.array_equal(Pa["uid"].to_numpy(), Pb["uid"].to_numpy())
    assert np.allclose(Pa["rot"].to_numpy(), Pb["rot"].to_numpy(), atol=1e-3)
    assert np.abs(a - b).max() < 1e-3 * np.abs(b).max(), np.abs(a - b).max()


def test_class3d_on_symexpand_is_own_expansion(sim):
    from celeris2d import jobs
    from celeris2d.class3d import run_class3d
    prj = sim["prj"]
    if "se" not in sim:
        pytest.skip("needs the SymExpand job")
    with pytest.raises(ValueError, match="already symmetry-expanded"):
        jobs.new_job(prj, "Class3D", {"particles": sim["ex"].id, "poses": sim["se"].id}, {"sym_expand": "C2"})
    z = {}
    for nm, (q, x) in dict(chain=(sim["se"], ""), own=(sim["cons"], "C2")).items():
        cj = jobs.new_job(prj, "Class3D", {"particles": sim["ex"].id, "poses": q.id},
                          {"n_classes": 2, "sym_expand": x, "n_iter": 4, "res_a": 12.0, "mask": str(sim["mask"].id)})
        run_class3d(prj, cj, log=lambda *a: None)
        t = prj.table(cj, "particles").read()
        o = np.lexsort((t["copy"], t["uid"]))
        z[nm] = (t["uid"][o], t["copy"][o], t["cls"][o], t["rot"][o])
    assert np.array_equal(z["chain"][0], z["own"][0]) and np.array_equal(z["chain"][1], z["own"][1])
    assert np.allclose(z["chain"][3], z["own"][3], atol=1e-3)
    # class numbers are arbitrary (two near-equal classes can come out in either order): compare up to a relabelling
    from scipy.optimize import linear_sum_assignment
    a_, b_ = z["chain"][2], z["own"][2]
    k_ = int(max(a_.max(), b_.max())) + 1
    cm_ = np.zeros((k_, k_))
    np.add.at(cm_, (a_, b_), 1)
    r_, c_ = linear_sum_assignment(-cm_)
    agree = float(cm_[r_, c_].sum() / len(a_))
    print(f"\n[test] Class3D: SymExpand input vs its own sym_expand: {agree * 100:.2f}% of copies in the same class")
    assert agree > 0.99, agree


def test_refine3d_on_symexpand(sim):
    from celeris2d import jobs
    prj = sim["prj"]
    if "se" not in sim:
        pytest.skip("needs the SymExpand job")
    with pytest.raises(ValueError, match="symmetry-expanded poses"):     # (e.g. HeteroRefine: refused)
        jobs._posed_particles(prj, sim["cons"], sim["ex"], sim["se"])
    r = jobs.new_job(prj, "Refine3D", {"particles": sim["ex"].id, "poses": sim["se"].id}, {"n_iter": 1, "drop_frac": 0.0})
    summ = jobs.run_refine3d(prj, r, log=lambda *a: None)
    assert summ["particles"] == 2 * summ["images"] and summ["images"] >= N - 10, summ
    t = prj.table(r, "particles").read()
    assert sorted(set(t["copy"].astype(int))) == [0, 1]
