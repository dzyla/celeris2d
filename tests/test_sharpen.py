"""Sharpen (RELION PostProcess) on synthetic half maps with a known B-factor and noise (CPU, ~1 s): the Guinier fit
recovers B, the corrected masked FSC is not raised by a tight mask, and a mask that cuts the particle is flagged.

    pytest -q tests/test_sharpen.py
"""
import numpy as np
import torch

from celeris2d.sharpen import SharpenSettings, postprocess

N, APIX, B = 96, 1.0, 100.0


def _maps(seed=0, noise=0.6):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N)[:, None, None] ** 2 + np.fft.fftfreq(N)[None, :, None] ** 2 + \
        np.fft.rfftfreq(N)[None, None, :] ** 2
    s2 = k / APIX ** 2
    # random-phase "protein" with a flat amplitude spectrum, B-factor decay, inside a 30 A sphere
    blob = np.fft.irfftn(np.exp(2j * np.pi * rng.random(s2.shape)) * np.exp(-B * s2 / 4), s=(N,) * 3, axes=(0, 1, 2))
    z, y, x = np.mgrid[:N, :N, :N] - N // 2
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    sphere = np.clip((30 - r) / 4 + 0.5, 0, 1)
    v = blob * sphere
    v /= v.std()
    hs = [v + noise * rng.standard_normal(v.shape) * v.std() for _ in range(2)]
    mask = np.clip((34 - r) / 6 + 0.5, 0, 1).astype(np.float32)
    return [h.astype(np.float32) for h in hs], mask, r


def test_b_factor_recovered_and_mask_checks():
    (h1, h2), mask, r = _maps()
    out = postprocess(h1, h2, mask, APIX, SharpenSettings(fit_from_a=8.0, resolution_a=3.0), dev="cpu")
    assert abs(-out["b_factor"] - B) < 0.25 * B, out["b_factor"]
    assert not out["warnings"], out["warnings"]
    # the same half maps with pure, independent noise in each: the corrected FSC must not be raised by the mask
    rng = np.random.default_rng(5)
    n1, n2 = (rng.standard_normal(h1.shape).astype(np.float32) for _ in range(2))
    tight = np.clip((20 - r) / 2 + 0.5, 0, 1).astype(np.float32)
    o2 = postprocess(n1, n2, tight, APIX, SharpenSettings(resolution_a=4.0), dev="cpu")
    assert abs(o2["fsc"]["corrected"][5:].mean()) < 0.05
    # a mask smaller than the particle is reported
    o3 = postprocess(h1, h2, tight, APIX, SharpenSettings(resolution_a=3.0), dev="cpu")
    assert any("outside the mask" in w for w in o3["warnings"]), o3["warnings"]
