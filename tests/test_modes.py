"""Modes of variation (modes.py) on simulated particles with a known answer (needs a CUDA GPU; ~15 s).

A blob particle with a domain of three blobs is projected at random orientations with CTF, random shifts and noise
(SNR ~0.4 per pixel, as in test_abinitio). (a) Continuous: the domain is translated by u x 24 A, u uniform in
[-1, 1]; the first latent coordinate must follow u (|r| > 0.9). (b) Discrete: two states 48 A apart (u = -1 / +1); a
two-cluster mixture on the latent coordinates must recover them. The particles are given at their true poses, as a
consensus refinement would give them; the mean and the components are solved from the particles alone.

Measured (RTX 5090, 12 iterations): (a) |r(z_1, u)| 0.98, |r(z_2, u^2)| 0.93 (the second component is the second
harmonic of a non-linear translation, not noise), halves |cc| 0.95 / 0.86 / 0.39; (b) 100 % of particles in the right
cluster; no motion: halves |cc| <= 0.04 (the noise floor sd is ~4 % of |mean|). ~10 s for the file.

    pytest -q tests/test_modes.py
    CELERIS2D_FIG_DIR=/some/dir pytest -q tests/test_modes.py     # keep the figures and GIFs to look at
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from celeris2d import gpu
from celeris2d import abinitio as ab
from celeris2d import modes as md
from celeris2d.engine import Optics, ctf_2d, freqs, ft_resize

pytestmark = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")

BOX, APIX, N, NOISE = 64, 4.0, 3000, 1.5
_r = np.random.default_rng(7)
BASE = [tuple(_r.uniform(-13, 13, 3)) + (_r.uniform(1.8, 3.5), _r.uniform(0.6, 1.2)) for _ in range(14)]
# the domain: three blobs on one side of the particle; it moves along y (volume coordinates, px)
DOMAIN = [(17.0, 2.0, 0.0, 3.0, 1.2), (15.0, -4.0, 3.0, 2.6, 1.0), (19.0, -1.0, -4.0, 2.6, 1.0)]
MOVE = torch.tensor([0.0, 6.0, 0.0])          # px at u = 1 (24 A)


def _volume(blobs, b=BOX):
    r = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    v = torch.zeros(b, b, b)
    for cx, cy, cz, s, a in blobs:
        v += a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    return v


def _simulate(u: torch.Tensor, seed: int = 4):
    """Particle spectra (n, b, bh) on the GPU and a stand-in for the 2D engine (clf) that serves them, with the true
    poses: the domain translated by u x MOVE (as a phase ramp on its projection: an in-plane shift by A @ s)."""
    dev, n = gpu.default(), len(u)
    g = torch.Generator().manual_seed(seed)
    Vb, Vd = ab.Volume(BOX, dev), ab.Volume(BOX, dev)
    Vb.set_real(_volume(BASE).to(dev))
    Vd.set_real(_volume(DOMAIN).to(dev))
    rot, psi = torch.rand(n, generator=g) * 360 - 180, torch.rand(n, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(n, generator=g) * 2 - 1))
    A = ab.euler_to_matrix(rot, tilt, psi)
    t = (torch.rand(n, 2, generator=g) - 0.5) * 8.0                       # A
    df = 10000 + 15000 * torch.rand(n, generator=g)
    ctf_p = torch.stack([df, df * 1.02, torch.rand(n, generator=g) * 180, torch.zeros(n), torch.zeros(n),
                         torch.ones(n)], 1)
    optics = Optics(300.0, 2.7, 0.1)
    fy, fx = freqs(BOX, dev)
    box_a = BOX * APIX
    d = (A @ MOVE)[:, :2] * u[:, None]                                    # in-plane shift of the domain, px
    data = torch.empty((n, BOX, BOX // 2 + 1), dtype=torch.complex64, device=dev)
    for i in range(0, n, 500):
        s = slice(i, i + 500)
        AA, dd, tt = A[s].to(dev), d[s].to(dev), t[s].to(dev)
        ramp = torch.exp(-2j * np.pi / BOX * (fx[None, None, :] * dd[:, 0, None, None] + fy[None, :, None] * dd[:, 1, None, None]))
        e_t = torch.exp(-2j * np.pi / box_a * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
        ctf = ctf_2d(ctf_p[s].to(dev), optics, fy, fx, box_a)
        data[s] = (Vb.project(AA) + Vd.project(AA) * ramp) * ctf * e_t
    img = torch.fft.irfft2(data, s=(BOX, BOX), norm="forward")
    img = img / img.std() + NOISE * torch.randn(img.shape, generator=g).to(dev)
    data = torch.fft.rfft2(img, norm="forward")
    clf = SimpleNamespace(dev=torch.device(dev), box_a=box_a, optics=optics, ctf_p=ctf_p, bmax=BOX, N=n,
                          p=SimpleNamespace(diameter_a=240.0))
    clf._fetch = lambda j, dv, b: ft_resize(data[j.to(dv)], b)
    return clf, A, t


def _solve(clf, A, t, **kw):
    return md.solve(clf, torch.arange(clf.N), A, t, n_comp=3, filter_res_a=12.0, n_iter=12, log=lambda *a: None, **kw)


def _halves(clf, A, t, mask=None):
    """Independent solves on the two halves of the particles: their modes."""
    h0, h1 = torch.arange(0, clf.N, 2), torch.arange(1, clf.N, 2)
    r0 = md.solve(clf, h0, A[h0], t[h0], n_comp=3, filter_res_a=12.0, n_iter=12, log=lambda *a: None, mask=mask)
    r1 = md.solve(clf, h1, A[h1], t[h1], n_comp=3, filter_res_a=12.0, n_iter=12, log=lambda *a: None,
                  mask=r0["mask"])
    return r0["basis"], r1["basis"]


def _out(tmp_path, sub):
    d = os.path.join(os.environ.get("CELERIS2D_FIG_DIR") or str(tmp_path), sub)
    os.makedirs(d, exist_ok=True)
    return lambda name: os.path.join(d, name)


def test_continuous_motion(tmp_path):
    u = torch.rand(N, generator=torch.Generator().manual_seed(1)) * 2 - 1
    clf, A, t = _simulate(u)
    res = _solve(clf, A, t)
    z = res["z"].numpy()
    r = abs(np.corrcoef(z[:, 0], u.numpy())[0, 1])
    print(f"continuous: |r(z1, u)| = {r:.3f}, sd / |mean| = {(res['sd'] / res['mean_norm']).numpy().round(4)}")
    assert r > 0.9, r
    # a translation of a blob several times its width is not linear in u: the second component is its second
    # harmonic (u^2), not noise
    r2 = abs(np.corrcoef(z[:, 1], u.numpy() ** 2)[0, 1])
    print(f"continuous: |r(z2, u^2)| = {r2:.3f}, sd of the posterior means {z.std(0).round(3)}")
    assert r2 > 0.7, r2
    assert z[:, 0].std() < 1.25, z.std(0)       # an EM fixed point has var(E[z]) <= 1 (0.83 at 30 iterations)
    rep = md.half_reproducibility(res["basis"], *_halves(clf, A, t, res["mask"]), res["box_a"])
    print(f"continuous: halves |cc| {rep['cc']}, agree to {rep['res_a']} A, subspace overlap {rep['subspace_overlap']}")
    assert rep["cc"][0] > 0.8, rep
    # outputs: series frames + ChimeraX scripts, figures, GIFs
    mean, basis, sd = res["mean"].cpu().numpy(), res["basis"].cpu().numpy(), res["sd"].numpy()
    mask = res["mask"].cpu().numpy()
    sd_rel = (res["sd"] / res["mean_norm"]).tolist()
    shown = md.display_res(rep["cc"], rep["res_a"], 12.0)
    out = _out(tmp_path, "continuous")
    o = md.mode_outputs(out, mean, basis, sd, mask, res["apix"], sd_rel, rep["cc"], shown, 12.0,
                        "synthetic: a domain moving with u", n_frames=5)
    assert len(o["series"]) == 3 and len(o["series"][0]["frames"]) == 5 and os.path.isfile(o["series"][0]["cxc"])
    assert os.path.isfile(out("mode1.gif")) and os.path.getsize(out("mode1.gif")) < 1.5e6
    md.variance_figure(out("variance.png"), sd_rel, rep["cc"], res["history"], "synthetic: a domain moving with u")
    md.latent_figure(out("latent.png"), z, "synthetic: a domain moving with u")
    assert all(os.path.isfile(out(f)) for f in ("variance.png", "latent.png", "what_moves.png", "frames.png",
                                                "projections.png"))


def test_two_states(tmp_path):
    state = (torch.rand(N, generator=torch.Generator().manual_seed(2)) < 0.4).long()
    clf, A, t = _simulate(state.float() * 2 - 1, seed=5)
    res = _solve(clf, A, t)
    z = res["z"].numpy()
    g = md.gmm(z[:, :2], 2, seed=0)
    acc = max((g["labels"] == state.numpy()).mean(), (g["labels"] != state.numpy()).mean())
    print(f"two states: |r(z1, state)| = {abs(np.corrcoef(z[:, 0], state.numpy())[0, 1]):.3f}, "
          f"clusters agree with the states for {acc * 100:.1f}% of particles")
    assert acc > 0.9, acc
    # the gold-standard reconstruction of each cluster
    half = (np.arange(N) % 2)
    rec = md.reconstruct_groups(clf, torch.arange(N), A, t, g["labels"], 2, half, log=lambda *a: None)
    assert all(r is not None and r["resolution_a"] < 20 for r in rec), [r and r["resolution_a"] for r in rec]
    out = _out(tmp_path, "two_states")
    names = ["cluster 1", "cluster 2"]
    cen = np.array([z[g["labels"] == k].mean(0) for k in range(2)])
    md.latent_figure(out("latent.png"), z, "synthetic: two states", labels=g["labels"], centres=cen, names=names)
    md.groups_figure(out("map_views.png"), [r["map"].cpu().numpy() for r in rec], names,
                     [int((g["labels"] == k).sum()) for k in range(2)], [r["resolution_a"] for r in rec],
                     "synthetic: two states", res["mask"].cpu().numpy())
    assert os.path.isfile(out("map_views.png"))


def test_noise_only():
    """No motion: whatever the components hold is fitted noise, and it does not reproduce between halves."""
    clf, A, t = _simulate(torch.zeros(N), seed=6)
    h = md.compare_bases(*_halves(clf, A, t))
    print(f"no motion: halves |cc| {h['cc']}, subspace overlap {h['subspace_overlap']}")
    assert max(h["cc"]) < 0.3, h
