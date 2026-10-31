"""Ab initio 3D on a simulated particle with a known answer (needs a CUDA GPU; ~1 min).

An asymmetric blob model is projected at random orientations with random shifts and noise, written as an .mrcs
stack, classified in 2D and reconstructed ab initio through the normal Session path. The estimated orientations
must agree with the true ones up to one global rotation (either hand: ab initio cannot tell), and the map's
reprojections must match the 2D class averages.

    pytest -q tests
"""
import math
import os

import mrcfile
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from celeris2d import gpu
from celeris2d import abinitio as ab
from celeris2d.engine import rot_grid, upsample2
from celeris2d.session import Session

pytestmark = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")

BOX, APIX, N, NOISE = 64, 4.0, 4000, 1.5
_r = np.random.default_rng(7)
# an asymmetric object: 14 blobs of different sizes and weights spread through a sphere (few smooth blobs carry so
# little information that wrong orientations reproject almost as well, and they are often nearly symmetric)
BLOBS = [tuple(_r.uniform(-13, 13, 3)) + (_r.uniform(1.8, 3.5), _r.uniform(0.6, 1.2)) for _ in range(14)]


def _volume(blobs, b=BOX):
    r = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    v = torch.zeros(b, b, b)
    for cx, cy, cz, s, a in blobs:
        v += a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    return v


def _c3(blobs):
    out = []
    for k in range(3):
        a = 2 * np.pi * k / 3
        out += [(np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y, z, s, w) for x, y, z, s, w in blobs]
    return out


def _simulate(tmp_path_factory, blobs):
    dev = gpu.default()
    g = torch.Generator().manual_seed(4)
    V = ab.Volume(BOX, dev)
    V.set_real(_volume(blobs).to(dev))
    rot, psi = torch.rand(N, generator=g) * 360 - 180, torch.rand(N, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(N, generator=g) * 2 - 1))
    A = ab.euler_to_matrix(rot, tilt, psi)
    img = torch.fft.irfft2(V.project(A.to(dev)), s=(BOX, BOX), norm="forward")
    img = torch.roll(img, shifts=(1, -1), dims=(1, 2))                   # a small common offset
    img = img / img.std() + NOISE * torch.randn(img.shape, generator=g).to(dev)
    path = tmp_path_factory.mktemp("abinitio") / "particles.mrcs"
    with mrcfile.new(str(path)) as m:
        m.set_data(img.cpu().numpy().astype(np.float32))
        m.voxel_size = APIX
    return str(path), A


@pytest.fixture(scope="module")
def particles(tmp_path_factory):
    return _simulate(tmp_path_factory, BLOBS)


# pseudo-C3: a C3 object plus one blob that breaks the symmetry
_rp = np.random.default_rng(11)
PSEUDO_C3 = _c3([tuple(_rp.uniform(-12, 12, 3)) + (_rp.uniform(2.0, 3.5), _rp.uniform(0.6, 1.2)) for _ in range(5)]) + [
    (9.0, -6.0, 8.0, 3.0, 1.0)]


def _consistency(A_est, A_true):
    """Largest fraction of particles whose orientation error is one common rotation (within 10 deg), either hand."""
    M = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    best = 0.0
    for Ae in (A_est, M @ A_est @ M):
        Q = Ae.transpose(1, 2) @ A_true
        for G in Q[torch.randperm(len(Q), generator=torch.Generator().manual_seed(0))[:200]]:
            tr = (G.T[None] @ Q).diagonal(dim1=-2, dim2=-1).sum(-1)
            d = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
            best = max(best, float((d < 10).float().mean()))
    return best


def test_ab_initio_recovers_orientations(tmp_path, particles):
    path, A_true = particles
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(path, diameter=200, max_res=12, gpus="0")
    s.classify("k12", K=12, n_iter=15)
    summ = s.abinitio("k12_3d", "k12", n_starts=4, sgd_iter=12, n_iter=6, check_starts=2, drop_frac=0.0)
    out = os.path.join(s.workdir, "k12_3d")
    for f in ("map.mrc", "map_nomask.mrc", "half1_unfil.mrc", "fsc.png", "views.png", "reprojections.png",
              "particles.star", "summary.json"):
        assert os.path.isfile(os.path.join(out, f)), f
    pz = np.load(os.path.join(out, "poses.npz"))
    A_est = ab.euler_to_matrix(torch.tensor(pz["rot"]), torch.tensor(pz["tilt"]), torch.tensor(pz["psi"]))
    frac = _consistency(A_est, A_true[torch.as_tensor(pz["idx"])])
    assert frac >= 0.8, f"only {frac:.2f} of orientations consistent with the truth"
    assert np.median(summ["reprojection_cc"]) > 0.8, summ["reprojection_cc"]
    from celeris2d.star import read_star
    df = read_star(os.path.join(out, "particles.star"))["particles"]
    assert {"rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst"} <= set(df.columns)
    assert len(df) == summ["particles_used"]
    assert summ["relaxed_from"] is None and "suggested_sym" not in summ, "symmetry is the user's choice only"


def test_relax_from_user_symmetry(tmp_path, tmp_path_factory):
    """relax_from (user-given): a pseudo-C3 particle solved with C3 imposed, then refined in C1 over the C3-related
    poses, so the blob that breaks the symmetry can be placed. Measured 0.60 of orientations within 10 deg of the
    truth without symmetry (a symmetry-related copy counts as wrong); direct C1 reached 0.77-0.89 here, so
    relax_from is for particles that are symmetric apart from small features (J341, C3: 89-90 % vs 47-72 %)."""
    path, A_true = _simulate(tmp_path_factory, PSEUDO_C3)
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(path, diameter=200, max_res=12, gpus="0")
    s.classify("k12", K=12, n_iter=15)
    summ = s.abinitio("k12_3d", "k12", n_starts=4, sgd_iter=12, n_iter=6, drop_frac=0.0, relax_from="C3")
    assert summ["relaxed_from"] == "C3" and "suggested_sym" not in summ
    pz = np.load(os.path.join(s.workdir, "k12_3d", "poses.npz"))
    A_est = ab.euler_to_matrix(torch.tensor(pz["rot"]), torch.tensor(pz["tilt"]), torch.tensor(pz["psi"]))
    frac = _consistency(A_est, A_true[torch.as_tensor(pz["idx"])])
    assert frac >= 0.5, f"only {frac:.2f} of orientations consistent with the truth"


def test_projector_conventions():
    """A 2D in-plane rotation of a projection equals rotz composed on the 3D orientation; Euler round trip."""
    dev = gpu.default()
    V = ab.Volume(BOX, dev)
    V.set_real(_volume(BLOBS).to(dev))
    A = ab.euler_to_matrix(torch.tensor([30.0]), torch.tensor([70.0]), torch.tensor([10.0]))
    P = V.project(A.to(dev))
    a = torch.tensor([40.0])
    img = F.grid_sample(upsample2(P, BOX)[:, None], rot_grid(a.to(dev), BOX, 2 * BOX), align_corners=False)[:, 0]
    X = torch.fft.rfft2(img, norm="forward")
    Q = V.project((ab.rotz(a) @ A).to(dev))
    cc = (X.conj() * Q).real.sum() / (X.abs().pow(2).sum() * Q.abs().pow(2).sum()).sqrt()
    assert cc > 0.99
    r, t, p = ab.matrix_to_euler(A)
    assert torch.allclose(ab.euler_to_matrix(r, t, p), A, atol=1e-5)


def test_working_box_fits_memory(monkeypatch):
    class Fake:
        dev, bmax, box_a = torch.device(gpu.default()), 256, 400.0
    monkeypatch.setattr(ab, "_free", lambda dev: 6e9)
    b = ab.working_box(Fake(), n_acc=2, n_ft=2, log=lambda *a: None)
    assert b < 256 and (2 * 160 + 2 * 64 + 320) * b ** 3 + 1.5e9 <= 0.85 * 6e9


def test_volume_padding_and_gridding():
    """set_real / real round trip in every mode; gridding correction brings pad 2 projections closer to a pad 4
    reference, and pad 1 (the big-box fallback) stays within its known error."""
    dev = gpu.default()
    vol = _volume(BLOBS).to(dev)
    for pad in (1, 2):
        for gc in (False, True):
            V = ab.Volume(BOX, dev, pad=pad, grid_correct=gc)
            V.set_real(vol)
            assert torch.allclose(V.real(), vol, atol=1e-4 * float(vol.abs().max())), (pad, gc)
    A = ab.euler_to_matrix(torch.tensor([10.0, 80.0]), torch.tensor([30.0, 120.0]), torch.tensor([0.0, 45.0])).to(dev)

    def proj(pad, gc):
        V = ab.Volume(BOX, dev, pad=pad, grid_correct=gc)
        V.set_real(vol)
        return V.project(A)
    ref = proj(4, True)
    err = lambda p: float((p - ref).norm() / ref.norm())
    assert err(proj(2, True)) < err(proj(2, False)) and err(proj(2, True)) < 0.05
    assert err(proj(1, True)) < 0.3


def test_optics_fit_recovers_magnification():
    """optics.fit_optics: particles simulated with a known anisotropic magnification (0.5 % along 30 deg) are fitted
    back to it (gradient through the projection's sampling coordinates)."""
    from celeris2d import optics as op
    from celeris2d.engine import TWO_PI, Optics, freqs
    dev, b, apix, n = gpu.default(), BOX, APIX, 1500
    g = torch.Generator().manual_seed(5)
    vols = []
    for _ in range(2):
        V = ab.Volume(b, dev)
        V.set_real(_volume(BLOBS).to(dev))
        vols.append(V)
    A = ab.euler_to_matrix(torch.rand(n, generator=g) * 360, torch.rad2deg(torch.arccos(torch.rand(n, generator=g) * 2 - 1)),
                           torch.rand(n, generator=g) * 360)
    t = torch.zeros(n, 2)
    half = (torch.arange(n) % 2)
    eps, phi = 0.005, math.radians(30)
    R = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    M = (R @ torch.diag(torch.tensor([1 + eps, 1 - eps])) @ R.T).to(dev)
    optics = Optics(300, 2.7, 0.1)
    p = torch.tensor([[15000.0, 15000.0, 0.0, 0.0, 0.0, 1.0]]).expand(n, 6).contiguous()
    fy, fx = freqs(b, dev)
    kx, ky = fx[None, :].expand(b, -1), fy[:, None].expand(-1, len(fx))
    for v in vols:
        v.mag = M
    proj = vols[0].project(A.to(dev))
    for v in vols:
        v.mag = None
    c = op.ctf_points(p.to(dev), optics, (M[0, 0] * kx + M[0, 1] * ky).reshape(1, -1), (M[1, 0] * kx + M[1, 1] * ky).reshape(1, -1),
                      b * apix).reshape(n, b, -1)
    data = proj * c
    data = data + 2.0 * data.abs().mean() * torch.complex(torch.randn(data.shape, generator=g).to(dev),
                                                           torch.randn(data.shape, generator=g).to(dev))

    class Clf:
        pass
    clf = Clf()
    clf.dev, clf.box_a, clf.optics, clf.ctf_p = dev, b * apix, optics, p
    clf._fetch = lambda j, d, b_: data[j.to(dev)]
    fit = op.fit_optics(clf, torch.arange(n), A, t, half, vols, res_hi_a=2.5 * apix, res_lo_a=20 * apix, n_max=n,
                        fit_odd=False, fit_even=False, log=lambda *a: None)
    assert np.allclose(np.array(fit.mag), M.cpu().numpy(), atol=5e-4), (fit.mag, M)


def test_optics_fit_recovers_aberrations():
    """optics.fit_optics: particles simulated with known odd (coma, trefoil) and even (spherical, 4-fold) aberration
    phases are fitted back to them (magnification fixed: fitted together, even astigmatism-like terms trade with
    anisotropic magnification - 0.20 for 0.30 here)."""
    from celeris2d import optics as op
    from celeris2d.engine import Optics, freqs
    dev, b, apix, n = gpu.default(), BOX, APIX, 1500
    g = torch.Generator().manual_seed(6)
    vols = []
    for _ in range(2):
        V = ab.Volume(b, dev)
        V.set_real(_volume(BLOBS).to(dev))
        vols.append(V)
    A = ab.euler_to_matrix(torch.rand(n, generator=g) * 360, torch.rad2deg(torch.arccos(torch.rand(n, generator=g) * 2 - 1)),
                           torch.rand(n, generator=g) * 360)
    t = torch.zeros(n, 2)
    half = (torch.arange(n) % 2)
    optics = Optics(300, 2.7, 0.1)
    p = torch.tensor([[15000.0, 15000.0, 0.0, 0.0, 0.0, 1.0]]).expand(n, 6).contiguous()
    fy, fx = freqs(b, dev)
    kx, ky = fx[None, :].expand(b, -1), fy[:, None].expand(-1, len(fx))
    res_hi = 2.5 * apix
    s_max = 1.0 / res_hi
    odd = torch.tensor([0.6, -0.3, 0.2, 0.0], device=dev)
    even = torch.tensor([0.4, 0.0, 0.3, 0.0, 0.0], device=dev)
    proj = vols[0].project(A.to(dev))
    c = op.ctf_points(p.to(dev), optics, kx.reshape(1, -1), ky.reshape(1, -1), b * apix, odd, even, s_max).reshape(n, b, -1)
    data = proj * c
    data = data + 1.0 * data.abs().mean() * torch.complex(torch.randn(data.shape, generator=g).to(dev),
                                                           torch.randn(data.shape, generator=g).to(dev))

    class Clf:
        pass
    clf = Clf()
    clf.dev, clf.box_a, clf.optics, clf.ctf_p = dev, b * apix, optics, p
    clf._fetch = lambda j, d, b_: data[j.to(dev)]
    fit = op.fit_optics(clf, torch.arange(n), A, t, half, vols, res_hi_a=res_hi, res_lo_a=20 * apix, n_max=n,
                        fit_mag=False, log=lambda *a: None)
    assert np.allclose(fit.odd, odd.cpu().numpy(), atol=0.1), fit.odd
    assert np.allclose(fit.even, even.cpu().numpy(), atol=0.1), fit.even


def test_distinct_candidates():
    """distinct_candidates: a near neighbour and a symmetry-equivalent copy of the best pose are suppressed; a pose far
    away and the same orientation at a distant shift are kept."""
    A0 = ab.euler_to_matrix(torch.tensor([10.0]), torch.tensor([40.0]), torch.tensor([70.0]))[0]
    ops = ab.symmetry_ops("C4").float()
    near = ab.euler_to_matrix(torch.tensor([11.0]), torch.tensor([40.5]), torch.tensor([70.0]))[0]
    far = ab.euler_to_matrix(torch.tensor([100.0]), torch.tensor([120.0]), torch.tensor([10.0]))[0]
    A = torch.stack([A0, near, A0 @ ops[1], far, A0])[None]
    t = torch.tensor([[[0.0, 0], [0.5, 0], [0, 0], [0, 0], [20, 0]]])
    assert ab.distinct_candidates(A, t, ops, 3, 4.0, 3.0).tolist() == [[0, 3, 4]]
    assert ab.distinct_candidates(A, t, ops, 4, 4.0, 3.0).tolist() == [[0, 3, 4, 0]]      # padded with the best


def test_symmetrize_accumulators_matches_symmetric_insertion():
    """Inserting each image once and symmetrising the accumulators (refine_particles at pad 2) gives the same
    reconstruction as inserting it at every symmetry-related pose. (Pad 1: 17 % apart, the resampled accumulators
    lose a little at low and gain at high frequency, so refine_particles keeps the per-operator insertion there.)"""
    pad = 2
    dev = gpu.default()
    vol = _volume(_c3(BLOBS[:5])).to(dev)
    ops = ab.symmetry_ops("C3").to(dev)
    g = torch.Generator().manual_seed(3)
    n = 300
    A = ab.euler_to_matrix(torch.rand(n, generator=g) * 360, torch.rand(n, generator=g) * 180,
                           torch.rand(n, generator=g) * 360).to(dev)
    V = ab.Volume(BOX, dev, pad=2)
    V.set_real(vol)
    x = V.project(A)
    w = torch.ones(x.shape, device=dev)
    maps = []
    for sym_after in (False, True):
        R = ab.Volume(BOX, dev, pad=pad, grid_correct=True)
        R.insert(x, w, A, kmax=BOX / 2 - 1, ops=None if sym_after else ops)
        if sym_after:
            R.symmetrize(ops)
        ab._regularise_solve(R, 0.02)
        maps.append(R.real())
    rel = lambda a, b_: float((a - b_).norm() / b_.norm())
    assert rel(maps[1], maps[0]) < 0.05, rel(maps[1], maps[0])
    assert rel(maps[1], vol) < 1.1 * rel(maps[0], vol) + 0.01


def test_model_fsc_not_cut_at_classification_resolution(tmp_path, particles):
    """3D classes (n_models > 1): each class's half maps go to Nyquist, so its FSC falls and stays down. The J20 run
    (classes filtered to 10 A) showed curves that fell to ~0.1 at the filter edge and rose to ~0.2 behind it: the
    halves were reconstructed only 4 shells past res_a and the mask spread the low shells beyond. Classified here at
    40 A, the old code's class FSC rose back to 0.27-0.44 after crossing 0.143; now it stays within +-0.05."""
    import json
    path, _ = particles
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(path, diameter=200, max_res=12, gpus="0")
    s.classify("k12", K=12, n_iter=15)
    s.abinitio("k12_m2", "k12", n_starts=4, sgd_iter=12, n_iter=6, check_starts=2, drop_frac=0.0, n_models=2,
               class3d_res=40.0, class3d_iter=6)
    curves = json.load(open(os.path.join(s.workdir, "k12_m2", "fsc.json")))["curves"]
    for k in ("class1_sphere_masked", "class2_sphere_masked"):
        c = np.asarray(curves[k])
        cross = int(np.argmax(c < 0.143))
        assert cross > 0 and c[cross:].max() < 0.143, (k, c.round(2).tolist())


@pytest.mark.parametrize("sym", ["T", "O", "I"])
def test_polyhedral_symmetrize_is_as_accurate_as_insertion(sym):
    """T / O / I: resampling the accumulators once recovers a map with the group's symmetry about as well as
    inserting every image at all |G| poses (errors vs the truth, box 32, 300 images: T 0.130 vs 0.120, O equal - its
    operators permute the grid axes - I 0.128 vs 0.174)."""
    dev = gpu.default()
    ops = ab.symmetry_ops(sym).double()
    blobs = [tuple((S @ torch.tensor([x, y, z], dtype=torch.float64)).tolist()) + (s * 0.8, w)
             for S in ops for x, y, z, s, w in BLOBS[:2]]
    vol = _volume(blobs).to(dev)
    ops = ops.float().to(dev)
    g = torch.Generator().manual_seed(5)
    n = 300
    A = ab.euler_to_matrix(torch.rand(n, generator=g) * 360, torch.rand(n, generator=g) * 180,
                           torch.rand(n, generator=g) * 360).to(dev)
    V = ab.Volume(BOX, dev, pad=2)
    V.set_real(vol)
    x = V.project(A)
    w = torch.ones(x.shape, device=dev)
    err = []
    for sym_after in (False, True):
        R = ab.Volume(BOX, dev, pad=2, grid_correct=True)
        R.insert(x, w, A, kmax=BOX / 2 - 1, ops=None if sym_after else ops)
        if sym_after:
            R.symmetrize(ops)
        ab._regularise_solve(R, 0.02)
        err.append(float((R.real() - vol).norm() / vol.norm()))
    assert err[1] < 1.2 * err[0] + 0.01, err
