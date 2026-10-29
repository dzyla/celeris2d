"""Movie preprocessing on simulated data with a known answer (needs a CUDA GPU; a few seconds).

Motion: frames are shifted copies of one image with Poisson counts; the recovered shifts must match the true ones.
CTF: white noise filtered by a known astigmatic CTF plus background; the fit must recover defocus and angle.

    pytest -q tests/test_preprocess.py
"""
import math

import numpy as np
import pytest
import torch

from celeris2d import preprocess as pp
from celeris2d.engine import Optics, freqs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_motion_recovers_shifts():
    dev, n, b, apix = "cuda", 20, 1024, 1.0
    g = torch.Generator(device=dev).manual_seed(0)
    img = torch.randn(b, b, generator=g, device=dev)
    ky, kx = pp._grid(b, b, apix, dev)
    F0 = torch.fft.rfft2(img) * torch.exp(-200 * (ky ** 2 + kx ** 2))      # smooth structure down to ~10 A
    t = torch.linspace(0, 1, n, device=dev)
    true = torch.stack([12 * t + 2 * torch.sin(6 * t), -8 * t ** 2], 1)    # drifting, decelerating path (A)
    true = true - true.mean(0)
    frames = torch.fft.irfft2(F0[None] * pp._shift_phase(ky, kx, true[:, 1], true[:, 0]), s=(b, b))
    frames = frames / frames.std() * 0.5 + 1.0                             # ~1 count per pixel per frame
    counts = torch.poisson(frames.clamp_min(0), generator=g).to(torch.uint8).cpu().numpy()
    mic = pp.motion_correct(counts, None, apix, dose_per_frame=1.5, align_apix=2.0, patches=0)
    err = np.abs(mic.shifts + true.cpu().numpy()).max()                 # the correction undoes the drift
    assert err < 0.1, f"shift error {err:.2f} A\n{mic.shifts}\n{true.cpu().numpy()}"


def _frc_band(a, b, apix, lo, hi):
    fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
    ky, kx = pp._grid(a.shape[0], a.shape[1], apix, a.device)
    k = torch.sqrt(ky ** 2 + kx ** 2)
    m = (k >= 1 / lo) & (k <= 1 / hi)
    return float((fa * fb.conj()).real[m].sum() / torch.sqrt(fa.abs().pow(2)[m].sum() * fb.abs().pow(2)[m].sum()))


def test_local_motion_recovers_signal():
    """Frames warped by a smooth field that grows with time (4 A at the edges): the patch model must bring the
    even/odd-frame FRC (independent noise, signal only) back to the level without local motion."""
    dev, b, n, apix = "cuda", 2048, 30, 1.0
    g = torch.Generator(device=dev).manual_seed(0)
    ky, kx = pp._grid(b, b, apix, dev)
    truth = torch.fft.irfft2(torch.fft.rfft2(torch.randn(b, b, generator=g, device=dev))
                             * torch.exp(-20 * (ky ** 2 + kx ** 2)), s=(b, b))
    truth = truth / truth.std()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, b, device=dev), torch.linspace(-1, 1, b, device=dev), indexing="ij")
    out = {}
    for amp in (0.0, 4.0):
        frames = []
        for t in torch.linspace(0, 1, n).tolist():
            dx = 10 * t + amp * t * (xx + 0.5 * yy * yy)
            dy = -6 * t + amp * t * (-yy + 0.7 * xx * yy)
            grid = torch.stack([xx - dx * 2 / b, yy - dy * 2 / b], -1)[None]
            fr = torch.nn.functional.grid_sample(truth[None, None], grid, mode="bicubic", padding_mode="reflection",
                                                 align_corners=True)[0, 0]
            frames.append(torch.poisson((1.0 + 0.3 * fr).clamp_min(0), generator=g).to(torch.uint8))
        frames = torch.stack(frames).cpu().numpy()
        for p in (0, 5):
            mic = pp.motion_correct(frames, None, apix, dose_per_frame=0.01, patches=p)
            out[amp, p] = _frc_band(*mic.halves, apix, 6.0, 3.5)
    assert out[4.0, 0] < out[0.0, 0] - 0.1, out                 # the local motion hurts a whole-frame alignment
    assert out[4.0, 5] > out[0.0, 0] - 0.02, out                # and the patch model recovers it
    assert out[0.0, 5] > out[0.0, 0] - 0.01, out                # without harm when there is none


def test_ctf_recovers_defocus():
    dev, b, apix = "cuda", 2048, 1.3
    optics = Optics(300, 2.7, 0.1)
    du, dv, ang = 15400.0, 14900.0, 35.0
    fy, fx = freqs(b, dev)
    p = torch.tensor([[du, dv, ang, 0.0, 0.0, 1.0]], device=dev)
    from celeris2d.engine import ctf_2d
    c = ctf_2d(p, optics, fy, fx, b * apix)[0]
    k = torch.sqrt((fy[:, None] / (b * apix)) ** 2 + (fx[None] / (b * apix)) ** 2)
    g = torch.Generator(device=dev).manual_seed(1)
    sig = torch.fft.rfft2(torch.randn(b, b, generator=g, device=dev)) * c * torch.exp(-60 * k ** 2)
    noise = torch.fft.rfft2(torch.randn(b, b, generator=g, device=dev)) * (0.6 + 0.4 * torch.exp(-200 * k ** 2))
    mic = torch.fft.irfft2(sig + noise, s=(b, b))
    fit = pp.estimate_ctf(mic, apix, optics)
    d_ang = abs(fit.angle - ang) % 180
    assert abs(fit.defocus_u - du) < 100 and abs(fit.defocus_v - dv) < 100, fit
    assert min(d_ang, 180 - d_ang) < 5, fit
    assert fit.fit_res_a < 6, fit


def test_dose_weights_keep_noise_level():
    k = torch.linspace(0.01, 0.5, 50)
    w = pp.dose_weights(40, 1.4, k)
    assert torch.allclose(w.pow(2).sum(0), torch.full_like(k, 40.0), rtol=1e-4)
    assert bool((w[-1] < w[0]).all())                    # later frames count less, most at high frequency
    assert float(w[-1, -1] / w[0, -1]) < float(w[-1, 0] / w[0, 0])
    assert math.isfinite(float(w.sum()))
