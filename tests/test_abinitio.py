"""Ab initio 3D on a simulated particle with a known answer (needs a CUDA GPU; ~1 min).

An asymmetric blob model is projected at random orientations with random shifts and noise, written as an .mrcs
stack, classified in 2D and reconstructed ab initio through the normal Session path. The estimated orientations
must agree with the true ones up to one global rotation (either hand: ab initio cannot tell), and the map's
reprojections must match the 2D class averages.

    pytest -q tests
"""
import os

import mrcfile
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from celeris2d import abinitio as ab
from celeris2d.engine import rot_grid, upsample2
from celeris2d.session import Session

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

BOX, APIX, N, NOISE = 48, 5.0, 3000, 2.0
BLOBS = [(-8, 0, 0, 3.5, 1.0), (6, 3, 0, 3.0, 1.0), (0, -9, 4, 2.5, 0.9), (2, 6, -8, 3.0, 1.1), (-3, 2, 9, 2.0, 0.8)]


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


@pytest.fixture(scope="module")
def particles(tmp_path_factory):
    dev = "cuda"
    g = torch.Generator().manual_seed(4)
    V = ab.Volume(BOX, dev)
    V.set_real(_volume(BLOBS).to(dev))
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
    assert summ["suggested_sym"] == "C1", summ["symmetry"]


def test_symmetry_detection():
    dev = "cuda"
    for blobs, want in ((_c3(BLOBS[:3]), "C3"), (BLOBS, "C1")):
        rows, sugg, ops = ab.detect_symmetry(_volume(blobs).to(dev), APIX, 200)
        assert sugg == want, (want, rows[:4])
        assert len(ops) == (3 if want == "C3" else 1)


def test_projector_conventions():
    """A 2D in-plane rotation of a projection equals rotz composed on the 3D orientation; Euler round trip."""
    dev = "cuda"
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
        dev, bmax, box_a = torch.device("cuda"), 256, 400.0
    monkeypatch.setattr(ab, "_free", lambda dev: 6e9)
    b = ab.working_box(Fake(), n_acc=2, n_ft=2, log=lambda *a: None)
    assert b < 256 and (2 * 160 + 2 * 64 + 320) * b ** 3 + 1.5e9 <= 0.85 * 6e9
