"""Particle subtraction, symmetry expansion and focused local refinement on a simulated two-domain particle with a
known answer (needs a CUDA GPU; ~30 s).

Domain A (large) is rigid; domain B (small) turns about a hinge between them by a different random rotation in every
particle (up to 12 deg), so a consensus refinement (dominated by A) blurs B. The particles are written into a project
(Extract job with CTF from four micrographs of different defocus), refined (Refine3D), then:
  - Subtract keeps B (a mask on B): the subtracted, re-centred images must match B's clean projections (A gone),
    and the job keeps uids, halves and poses (shifts follow the re-centring);
  - LocalRefine on the subtracted particles, focus mask on B: B's resolution in the mask must beat the consensus
    half maps' in the same mask, and B's poses must be closer to the truth than the consensus poses;
  - symmetry expansion: C2 copies keep the particle's uid and half, carry their own pose and region centre, and a
    poses job with one pose per particle is refused for them.

    pytest -q tests/test_subtract.py
"""

import numpy as np
import pytest
import torch

from celeris2d import abinitio as ab
from celeris2d.engine import TWO_PI, Optics, ctf_2d, freqs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

BOX, APIX, N, NOISE = 64, 3.0, 2000, 1.5
C_A, C_B, HINGE = np.array([-8.0, 0, 0]), np.array([14.0, 0, 0]), np.array([0.0, 0, 0])   # voxels (x, y, z)
_r = np.random.default_rng(3)
BLOBS_A = [tuple(C_A + _r.uniform(-7.0, 7.0, 3)) + (_r.uniform(0.8, 1.6), _r.uniform(0.6, 1.2)) for _ in range(20)]
BLOBS_B = [tuple(_r.uniform(-5.0, 5.0, 3)) + (_r.uniform(0.7, 1.1), _r.uniform(0.8, 1.3)) for _ in range(16)]  # B-centred
MAX_TURN, ENVELOPE_B = 25.0, 200.0             # B turns up to this (deg) about the hinge; data envelope (A^2)
DEFOCUS = (9000.0, 14000.0, 19000.0, 24000.0)


def _volume(blobs, b=BOX):
    r = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    v = torch.zeros(b, b, b)
    for cx, cy, cz, s, a in blobs:
        v += a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    return v


def _rand_rot(n, max_deg, g):
    ax = torch.randn(n, 3, generator=g)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = torch.deg2rad(torch.rand(n, generator=g) * max_deg)
    K = torch.zeros(n, 3, 3)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K = K - K.transpose(1, 2)
    s, c = ang.sin()[:, None, None], ang.cos()[:, None, None]
    return torch.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _rel_err(A_est, A_true):
    """Median angle (deg) between estimated and true orientations after removing their mean offset (the map's frame
    is defined only up to a small common rotation)."""
    G = A_est.transpose(1, 2).double() @ A_true.double()
    U, _, V = torch.linalg.svd(G.sum(0))
    M = U @ V
    if torch.det(M) < 0:
        U[:, -1] *= -1
        M = U @ V
    tr = (M.T[None] @ G).diagonal(dim1=-2, dim2=-1).sum(-1)
    return float(torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).median())


def _band_corr(x, y, kmax):
    """Correlation of image stacks x, y (n, b, b) over frequencies up to kmax (cycles/box), pooled."""
    b = x.shape[-1]
    fy, fx = freqs(b, "cpu")
    keep = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) <= kmax
    X, Y = torch.fft.rfft2(torch.as_tensor(x)), torch.fft.rfft2(torch.as_tensor(y))
    X, Y = X[:, keep], Y[:, keep]
    return float((X.conj() * Y).real.sum() / ((X.abs() ** 2).sum() * (Y.abs() ** 2).sum()).sqrt())


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from celeris2d import jobs
    from celeris2d import pipeline as pl
    from celeris2d import preprocess as pp
    from celeris2d.project import Project
    dev = "cuda"
    tmp = tmp_path_factory.mktemp("subtract")
    g = torch.Generator().manual_seed(5)
    VA, VB = ab.Volume(BOX, dev), ab.Volume(BOX, dev)
    VA.set_real(_volume(BLOBS_A).to(dev))
    VB.set_real(_volume(BLOBS_B).to(dev))
    rot, psi = torch.rand(N, generator=g) * 360 - 180, torch.rand(N, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(N, generator=g) * 2 - 1))
    A = ab.euler_to_matrix(rot, tilt, psi)
    R = _rand_rot(N, MAX_TURN, g)                                  # B about the hinge, per particle
    t = (torch.rand(N, 2, generator=g) - 0.5) * 6.0            # A
    h, cB = torch.tensor(HINGE, dtype=torch.float32) * APIX, torch.tensor(C_B, dtype=torch.float32) * APIX
    A_B = A @ R
    t_B = t + (A @ (h[None, :, None] + R @ (cB - h)[None, :, None]))[:, :2, 0]
    mic = torch.arange(N) % len(DEFOCUS)
    ctf_p = torch.zeros(N, 6)
    ctf_p[:, 0] = ctf_p[:, 1] = torch.tensor(DEFOCUS)[mic]
    ctf_p[:, 5] = 1
    ctf_p[:, 4] = ENVELOPE_B                                   # signal envelope (unknown to the jobs, as in real data)
    optics = Optics(300.0, 2.7, 0.1)
    fy, fx = freqs(BOX, dev)
    ph = TWO_PI / (BOX * APIX)

    def img(V, A_, t_):
        c = ctf_2d(ctf_p.to(dev), optics, fy, fx, BOX * APIX)
        e = torch.exp(-1j * ph * (fx[None, None, :] * t_[:, 0, None, None].to(dev) + fy[None, :, None] * t_[:, 1, None, None].to(dev)))
        return torch.fft.irfft2(V.project(A_.to(dev)) * c * e, s=(BOX, BOX), norm="forward").cpu()
    a, b_ = img(VA, A, t), img(VB, A_B, t_B)
    sc = float((a + b_).std())
    a, b_ = a / sc, b_ / sc
    noise = NOISE * torch.randn(a.shape, generator=g)
    x = (a + b_ + noise).numpy()

    # a project: 4 micrographs (defocus), one Extract job holding the particles, a start poses job
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
    st = prj.new_job("Refine3D", dict(particles=ex.id, sym="C1"), [ex], "truth_A")
    rr, tt, pp_ = (v.numpy() for v in ab.matrix_to_euler(A))
    prj.table(st, "particles").write(dict(uid=uid, rot=rr, tilt=tt, psi=pp_, shift_x_a=t[:, 0].numpy(),
                                          shift_y_a=t[:, 1].numpy(), half=half.astype(np.uint8),
                                          fit=np.ones(N, np.float32), used=np.ones(N, bool)), mic.numpy() + mids[0])
    ab._save_mrc(st.out("map_nomask.mrc"), _volume(BLOBS_A) + _volume([(bx + C_B[0], by, bz, s, w) for bx, by, bz, s, w in BLOBS_B]), APIX)
    prj.set_status(st, "finished", dict(resolution_a=12.0))
    # consensus refinement (global poses, dominated by A)
    cons = jobs.refine3d_job(prj, ex, st, st, jobs.Refine3DSettings(n_iter=4, res_start_a=12.0, drop_frac=0.0,
                                                                  save_iterations=True))
    jobs.run_refine3d(prj, cons, log=lambda *a: None)
    # the mask on B (a MaskCreate job holding a soft sphere about B's rest position)
    mj = prj.new_job("MaskCreate", dict(map=cons.id), [cons], "B")
    rgrid = torch.arange(BOX, dtype=torch.float32) - BOX // 2
    z_, y_, x_ = torch.meshgrid(rgrid, rgrid, rgrid, indexing="ij")
    d = torch.sqrt((x_ - C_B[0]) ** 2 + y_ ** 2 + z_ ** 2)
    mask = torch.clamp((11.5 - d) / 2.5, 0, 1)
    mask = 0.5 - 0.5 * torch.cos(np.pi * mask)
    ab._save_mrc(mj.out("mask.mrc"), mask, APIX)
    prj.set_status(mj, "finished", {})
    return dict(prj=prj, ex=ex, cons=cons, mask=mj, mask_vol=mask, uid=uid, a=a.numpy(), b=b_.numpy(), noise=noise.numpy(), A=A,
                A_B=A_B, t_B=t_B, half=half, mids=mids)


def _recentred(stack, d_px, nb):
    """Images moved by -d (px, the job's re-centring) and cut to nb (as Subtract does)."""
    b = stack.shape[-1]
    fy, fx = freqs(b, "cpu")
    X = torch.fft.rfft2(torch.as_tensor(stack), norm="forward")
    d = torch.as_tensor(d_px, dtype=torch.float32)
    X = X * torch.exp(2j * np.pi / b * (fx[None, None, :] * d[:, 0, None, None] + fy[None, :, None] * d[:, 1, None, None]))
    o = b // 2 - nb // 2
    return torch.fft.irfft2(X, s=(b, b), norm="forward")[:, o:o + nb, o:o + nb].numpy()


def test_subtract_keeps_domain_b(sim):
    import mrcfile
    from celeris2d.subtract import SubtractSettings, run_subtract, subtract_job
    prj = sim["prj"]
    sj = subtract_job(prj, sim["ex"], sim["cons"], sim["mask"], SubtractSettings(box=40))
    summ = run_subtract(prj, sj, log=lambda *a: None)
    sim["sub"] = sj
    assert summ["images"] == summ["particles"] == N and summ["box"] == 40 and not summ["warnings"], summ
    assert 0.7 < summ["scale_median"] < 1.4, summ["scale_median"]
    assert sj.find("subtract.png")
    P = prj.particles(sj)
    assert set(P.columns) >= {"uid", "x_a", "y_a", "rot", "tilt", "psi", "shift_x_a", "shift_y_a", "half", "sub_scale"}
    assert int(P["box"].iloc[0]) == 40 and np.all(P["stack"].str.startswith(f"data/{sj.id:03d}/"))
    order = np.argsort(sim["uid"])
    i = order[np.searchsorted(sim["uid"][order], P["uid"].to_numpy())]            # simulated particle of each row
    assert np.array_equal(np.sort(P["uid"]), np.sort(sim["uid"]))
    assert np.array_equal(P["half"].to_numpy().astype(int), sim["half"][i])     # halves of the refinement kept
    E = prj.particles(sim["ex"]).set_index("uid").loc[P["uid"]]
    d = np.stack([P["x_a"].to_numpy() - E["x_a"].to_numpy(), P["y_a"].to_numpy() - E["y_a"].to_numpy()], 1)
    # the re-centred box holds B's centre: its projected position is the new box centre (within the pose error)
    cB = torch.tensor(C_B * APIX, dtype=torch.float32)
    A_c = ab.euler_to_matrix(torch.tensor(P["rot"].to_numpy()), torch.tensor(P["tilt"].to_numpy()), torch.tensor(P["psi"].to_numpy()))
    pos = (A_c @ cB)[:, :2].numpy() + np.stack([P["shift_x_a"], P["shift_y_a"]], 1)
    assert np.median(np.linalg.norm(pos, axis=1)) < 1.0, np.median(np.linalg.norm(pos, axis=1))
    # images: subtracted minus the (known) noise, against B's and A's clean images moved and cut the same way
    y = np.concatenate([mrcfile.open(prj.abs(s), permissive=True).data[P["slice"].to_numpy()[P["stack"].to_numpy() == s]]
                        for s in P["stack"].unique()]).astype(np.float32)
    y_rows = np.concatenate([np.flatnonzero(P["stack"].to_numpy() == s) for s in P["stack"].unique()])
    y = y[np.argsort(y_rows)]
    nb = 40
    dpx = d / APIX
    a_c, b_c, n_c = (_recentred(sim[k][i], dpx, nb) for k in ("a", "b", "noise"))
    x_c = a_c + b_c + n_c
    kmax = nb * APIX / 8.0                                           # to 8 A (the consensus map is at ~6-8 A)
    cb_sub, ca_sub = _band_corr(y - n_c, b_c, kmax), _band_corr(y - n_c, a_c, kmax)
    cb_raw = _band_corr(x_c - n_c, b_c, kmax)
    print(f"\n[test] subtracted - noise vs B: {cb_sub:.3f} (before: {cb_raw:.3f}); vs A: {ca_sub:.3f}; "
          f"scale median {summ['scale_median']}, {summ['seconds']} s")
    assert cb_sub > 0.9 and cb_sub > cb_raw + 0.2 and abs(ca_sub) < 0.25, (cb_sub, cb_raw, ca_sub)


def test_local_refine_improves_domain_b(sim):
    from celeris2d.subtract import LocalRefineSettings, local_refine_job, run_local_refine
    prj = sim["prj"]
    if "sub" not in sim:
        pytest.skip("needs the subtraction test's job")
    lj = local_refine_job(prj, sim["sub"], sim["cons"], sim["mask"], None,
                          LocalRefineSettings(n_iter=8, drop_frac=0.0, max_angle_deg=30.0))
    summ = run_local_refine(prj, lj, log=lambda *a: None)
    for f in ("map.mrc", "half1_unfil.mrc", "mask.mrc", "focus_fsc.png", "fsc.png"):
        assert lj.find(f), f
    uid = sim["uid"]
    o = np.argsort(uid)

    cB = torch.tensor(C_B * APIX, dtype=torch.float32)

    def err(job):
        """Median orientation error vs B's true pose (deg) and error of B's projected centre (A) in the micrograph:
        the job's box centre (aligned_particles) + its shift + (A c_B)_xy vs the extraction centre + B's true one."""
        t = prj.table(job, "particles").read()
        i = o[np.searchsorted(uid[o], t["uid"])]
        A_est = ab.euler_to_matrix(torch.tensor(t["rot"]), torch.tensor(t["tilt"]), torch.tensor(t["psi"]))
        box_c = prj.particles(prj.aligned_particles(job)).set_index("uid").loc[t["uid"], ["x_a", "y_a"]].to_numpy()
        ex_c = prj.particles(sim["ex"]).set_index("uid").loc[t["uid"], ["x_a", "y_a"]].to_numpy()
        est = box_c + np.stack([t["shift_x_a"], t["shift_y_a"]], 1) + (A_est @ cB)[:, :2].numpy()
        d = np.linalg.norm(est - (ex_c + sim["t_B"][i].numpy()), axis=1)
        return _rel_err(A_est, sim["A_B"][i]), t, i, float(np.median(d))
    e_cons, _, _, d_cons = err(sim["cons"])
    e_loc, t, i, d_loc = err(lj)
    assert d_loc < 0.6 * d_cons and d_loc < 2.0, (d_loc, d_cons)       # shifts back in the consensus frame
    assert np.array_equal(t["half"].astype(int), sim["half"][i])             # gold-standard halves kept
    # B's signal kept in the maps (unfiltered half-map average vs the truth, B at rest, in the B mask): per shell
    # the regression of the map on the truth relative to low resolution (blurring by pose errors attenuates it)
    import mrcfile
    B = torch.as_tensor(_volume([(x + C_B[0], y, z, s, w) for x, y, z, s, w in BLOBS_B]))
    m = sim["mask_vol"]
    k = torch.fft.fftfreq(BOX) * BOX
    sh = torch.round(torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + torch.arange(BOX // 2 + 1.0)[None, None] ** 2))
    sh = sh.long().clamp(max=BOX // 2).reshape(-1)
    FB = torch.fft.rfftn(B * m).reshape(-1)
    kept = {}
    for nm, j in (("local", lj), ("consensus", sim["cons"])):
        v = 0
        for h in (1, 2):
            with mrcfile.open(j.find(f"half{h}_unfil.mrc"), permissive=True) as f:
                v = v + 0.5 * torch.as_tensor(np.asarray(f.data, np.float32))
        FV = torch.fft.rfftn(v * m).reshape(-1)
        beta = torch.zeros(BOX // 2 + 1).index_add_(0, sh, (FV * FB.conj()).real) / \
            torch.zeros(BOX // 2 + 1).index_add_(0, sh, FB.abs() ** 2).clamp_min(1e-12)
        band = slice(int(BOX * APIX / 12), int(BOX * APIX / 8) + 1)             # 12-8 A
        kept[nm] = float(beta[band].mean() / beta[2:5].mean())
    print(f"\n[test] domain B: half-map resolution {summ['resolution_a']} A local vs {summ['consensus_region_resolution_a']} "
          f"A consensus (same mask); signal kept at 12-8 A {kept['local']:.2f} vs {kept['consensus']:.2f}; "
          f"pose error median {e_loc:.2f} vs {e_cons:.2f} deg, centre {d_loc:.2f} vs {d_cons:.2f} A; {summ['seconds']} s")
    assert summ["resolution_a"] < summ["consensus_region_resolution_a"], summ
    assert kept["local"] > 1.3 * kept["consensus"], kept
    assert e_loc < 0.6 * e_cons, (e_loc, e_cons)


def test_symmetry_expansion_copies(sim):
    from celeris2d import jobs
    from celeris2d.subtract import SubtractSettings, run_subtract, subtract_job
    prj = sim["prj"]
    sj = subtract_job(prj, sim["ex"], sim["cons"], sim["mask"], SubtractSettings(sym_expand="C2", subtract=False, box=40))
    summ = run_subtract(prj, sj, log=lambda *a: None)
    assert summ["images"] == 2 * N and summ["copies"] == 2
    t = prj.table(sj, "particles").read()
    assert sorted(set(t["copy"].astype(int))) == [0, 1] and len(np.unique(t["uid"])) == N
    for k in ("half",):                                              # copies of a particle stay in its half set
        u0, u1 = (t["copy"] == 0), (t["copy"] == 1)
        o0, o1 = np.argsort(t["uid"][u0]), np.argsort(t["uid"][u1])
        assert np.array_equal(t[k][u0][o0], t[k][u1][o1])
    # copy 1 = pose x the 2-fold: the same particle, its other symmetry position
    A0 = ab.euler_to_matrix(torch.tensor(t["rot"][u0][o0]), torch.tensor(t["tilt"][u0][o0]), torch.tensor(t["psi"][u0][o0]))
    A1 = ab.euler_to_matrix(torch.tensor(t["rot"][u1][o1]), torch.tensor(t["tilt"][u1][o1]), torch.tensor(t["psi"][u1][o1]))
    assert torch.allclose(A0 @ ab.symmetry_ops("C2")[1], A1, atol=1e-4)
    with pytest.raises(ValueError, match="symmetry-expanded"):        # one pose per particle cannot start the copies
        jobs._posed_particles(prj, sj, sj, sim["cons"])
    src, poses, t_start, half, n_dup, _ = jobs._posed_particles(prj, sj, sj, sj)
    assert src.N == 2 * N and n_dup == 0 and np.allclose(t_start, np.stack([poses["shift_x_a"], poses["shift_y_a"]], 1), atol=1e-3)


def test_refine_writes_sharpened_map(sim):
    """Refine3D sharpen=True (default): the refinement sharpens its own half maps with its own mask (no separate job):
    sharpened map, post-processing figure and local resolution in the job, the numbers in its summary."""
    prj, cons = sim["prj"], sim["cons"]
    rj = prj.job(cons.id)
    summ = prj.summary_of(rj)
    assert summ["sharpened"]["resolution_a"] > 0 and "b_factor" in summ["sharpened"]
    assert rj.find("sharpened.mrc") and rj.find("postprocess.png") and rj.find("localres.mrc")
    assert not [i for i in prj.jobs("Sharpen").id if prj.job(int(i)).params.get("refine") == cons.id]

def test_refine_saves_iterations(sim):
    """save_iterations=True: each iteration's FSC curve (data/..._iterations.json) and every particle's angle / shift
    change and fit per iteration (particle table columns); poses settle, so the last iteration moves particles less
    than the first. Off by default: no such columns."""
    import json
    prj, cons = sim["prj"], sim["cons"]
    rj = prj.job(cons.id)
    it = json.load(open(rj.find("iterations.json")))
    assert len(it["fsc"]) == 4 and len(it["fsc"][0]) == 1 and len(it["shell_res_a"]) == len(it["fsc"][0][0])
    t = prj.table(rj, "particles").read()
    for k in range(1, 5):
        assert {f"d_angle_it{k:02d}", f"d_shift_it{k:02d}", f"fit_it{k:02d}"} <= set(t), sorted(t)
    a1, a4 = (np.median(t[f"d_angle_it{k:02d}"].astype(np.float32)) for k in (1, 4))
    assert a4 < a1, (a1, a4)
    from celeris2d.jobs import Refine3DSettings
    assert not Refine3DSettings().save_iterations
