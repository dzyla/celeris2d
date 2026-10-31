"""MapScore on simulated particles (needs a CUDA GPU; a few seconds): projections of a known map in noise must
fit it far more often than structureless images with the same power spectrum and than projections of another
object, and a phase-randomised control must pass at about the chance rate. Measured: 95 % of the particles fit,
0 % of pure-noise images, 36 % of another compact object of the same size (at 16 A two globular blob objects look
alike: the score tells particle-like density from empty ice / noise attractors, not one protein from another)."""
import numpy as np
import pytest
import torch

from celeris2d import gpu
from celeris2d import abinitio as ab
from celeris2d.engine import Optics
from celeris2d.mapscore import Scorer, class_shares

pytestmark = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")

BOX, APIX, N = 64, 4.0, 600


def _blobs(seed, n=14):
    r = np.random.default_rng(seed)
    return [tuple(r.uniform(-13, 13, 3)) + (r.uniform(1.8, 3.5), r.uniform(0.6, 1.2)) for _ in range(n)]


def _volume(blobs, b=BOX):
    x = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(x, x, x, indexing="ij")
    v = torch.zeros(b, b, b)
    for cx, cy, cz, s, a in blobs:
        v += a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    return v


def _particles(vol, n, noise, g):
    V = ab.Volume(BOX, gpu.default())
    V.set_real(vol.to(gpu.default()))
    rot, psi = torch.rand(n, generator=g) * 360 - 180, torch.rand(n, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(n, generator=g) * 2 - 1))
    img = torch.fft.irfft2(V.project(ab.euler_to_matrix(rot, tilt, psi).to(gpu.default())), s=(BOX, BOX), norm="forward")
    img = img / img.std()
    return img + noise * torch.randn(img.shape, generator=g).to(gpu.default())


def test_mapscore_separates_particles_from_junk():
    g = torch.Generator().manual_seed(3)
    vol = _volume(_blobs(7))
    good = _particles(vol, N, 3.0, g)                                  # SNR ~ 0.1
    other = _particles(_volume(_blobs(21)), N, 3.0, g)                 # another object, same noise
    noise = 3.0 * torch.randn(N, BOX, BOX, generator=g).to(gpu.default())        # empty ice
    sc = Scorer(vol.numpy(), APIX, BOX, APIX, Optics(300.0, 2.7, 0.1, no_ctf=True), diameter_a=120, res_a=16,
                n_dirs=150, psi_step=10, shift_a=8)
    ctf = torch.zeros(N, 6)
    out = {}
    for name, im in (("good", good), ("other", other), ("noise", noise)):
        x, r = sc.prepare(im, ctf, torch.Generator().manual_seed(1))
        out[name] = (sc.score(x).cpu().numpy(), sc.score(r).cpu().numpy())
    thr = np.quantile(np.concatenate([out[k][1] for k in out]), 0.99)
    fit = {k: float(np.mean(v[0] > thr)) for k, v in out.items()}
    ctrl = float(np.mean(np.concatenate([out[k][1] for k in out]) > thr))
    assert fit["good"] > 0.5, fit
    assert fit["noise"] < 0.05 and fit["other"] < 0.5 * fit["good"], fit
    assert abs(ctrl - 0.01) < 0.01
    cls = np.repeat([0, 1, 2], N)
    sh = class_shares(cls, np.concatenate([out[k][0] > thr for k in ("good", "other", "noise")]), 0.01)
    assert sh[0]["lo"] > 0.03 and sh[2]["lo"] < 0.03, sh


AMP = 6.0


def test_decoy_rejects_junk_blobs():
    """High-contrast junk of the particle's size (ice / contamination blobs, a quarter of the stack) correlates with
    projections of any object that size: by the map score alone 23 % of it fits. By the map score minus the score
    against a structureless decoy (the map's envelope, phases random beyond 80 A) none does, 48 % of the particles do
    (decoy at 40 A: 6 %, too close to this blobby map)."""
    from celeris2d.jobs import decoy_map
    g = torch.Generator().manual_seed(5)
    vol = _volume(_blobs(7))
    good = _particles(vol, N, 3.0, g)
    yy = torch.arange(BOX, dtype=torch.float32)[:, None] - BOX / 2
    xx = torch.arange(BOX, dtype=torch.float32)[None, :] - BOX / 2
    NB = N // 4                                                      # junk a minority, as in real stacks
    c = (torch.rand(NB, 2, generator=g) - 0.5) * 8
    rad = 10 + 6 * torch.rand(NB, generator=g)                        # 80-128 A across, the particle's size
    d = ((xx - c[:, 0, None, None]) ** 2 + (yy - c[:, 1, None, None]) ** 2).sqrt()
    blob = torch.sigmoid((rad[:, None, None] - d) / 1.5) * AMP       # an ice / contamination blob
    blob = blob.to(gpu.default()) + 3.0 * torch.randn(NB, BOX, BOX, generator=g).to(gpu.default())
    kw = dict(diameter_a=120, res_a=16, n_dirs=150, psi_step=10, shift_a=8)
    opt = Optics(300.0, 2.7, 0.1, no_ctf=True)
    sc = Scorer(vol.numpy(), APIX, BOX, APIX, opt, **kw)
    sd = Scorer(decoy_map(vol.to(gpu.default()), BOX * APIX, 80.0, 144.0).cpu().numpy(), APIX, BOX, APIX, opt, **kw)
    ctf = torch.zeros(N, 6)
    out = {}
    for name, im in (("good", good), ("blob", blob)):
        x, r = sc.prepare(im, ctf[:len(im)], torch.Generator().manual_seed(1))
        out[name] = [sc.score(x).cpu().numpy(), sc.score(r).cpu().numpy(), sd.score(x).cpu().numpy(),
                     sd.score(r).cpu().numpy()]
    ctrl = np.concatenate([v[1] for v in out.values()])
    dctrl = ctrl - np.concatenate([v[3] for v in out.values()])
    t_map, t_dec = np.quantile(ctrl, 0.99), np.quantile(dctrl, 0.99)
    by_map = {k: float(np.mean(v[0] > t_map)) for k, v in out.items()}
    by_dec = {k: float(np.mean(v[0] - v[2] > t_dec)) for k, v in out.items()}
    assert by_map["blob"] > 0.15, by_map                    # the failure the decoy is for
    assert by_dec["good"] > 0.3 and by_dec["blob"] < 0.02, by_dec
