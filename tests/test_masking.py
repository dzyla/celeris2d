"""Solvent masks (masking.py) on a synthetic map with known density: the automatic mask holds all of it and little
else, the threshold comes from the half-map noise, and the user choices (segments, include / exclude regions,
inversion) do what they say.

    pytest -q tests/test_masking.py
"""
import numpy as np
import torch

from celeris2d import masking as mk

N, APIX = 96, 2.0


def _maps(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = (torch.arange(N, dtype=torch.float32) + 0.5) * APIX
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    blob = lambda cx, cy, cz, s: torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    v = blob(70, 96, 96, 12) + blob(122, 96, 96, 12)                       # two domains 52 A apart, along x
    noise = lambda: 0.05 * torch.randn(v.shape, generator=g)
    return v, v + noise(), v + noise()


def test_auto_mask_holds_the_density():
    v, h1, h2 = _maps()
    info = {}
    m = mk.auto_mask(0.5 * (h1 + h2), APIX, res_a=8.0, k_sigma=5.0, halves=(h1, h2), info=info)
    dens = v > 0.1 * v.max()
    assert float(m[dens].min()) > 0.99                                    # nothing of the particle cut
    assert float(m.mean()) < 0.25                                         # and not the whole box
    assert info["threshold_sigma"] >= 4.9 and info["pieces_kept"] == 1


def test_user_choices():
    v, h1, h2 = _maps()
    avg = 0.5 * (h1 + h2)
    m_all, _, lab = mk.make_mask(avg, APIX, lowpass_a=10.0, threshold=0.2, n_segments=4)
    assert lab is not None and int(lab.max()) >= 2
    # the segment holding the left domain's centre only
    s_left = int(lab[48, 48, 35])
    m_left, _, _ = mk.make_mask(avg, APIX, lowpass_a=10.0, threshold=0.2, n_segments=4, segments=(s_left,))
    assert float(m_left[48, 48, 35]) > 0.99 and float(m_left[48, 48, 61]) < 0.01
    # exclude the right domain with a sphere, include an empty region with a box
    m_ex, _, _ = mk.make_mask(avg, APIX, lowpass_a=10.0, threshold=0.2, n_segments=0,
                              exclude=[mk.parse_region("sphere 122 96 96 30")],
                              include=[mk.parse_region("box 20 20 20 30 30 30")])
    assert float(m_ex[48, 48, 61]) < 0.01 and float(m_ex[48, 48, 35]) > 0.99 and float(m_ex[12, 12, 12]) > 0.99
    m_inv, _, _ = mk.make_mask(avg, APIX, lowpass_a=10.0, threshold=0.2, n_segments=0, invert=True)
    assert torch.allclose(m_inv, 1 - m_all, atol=1e-6)


def test_mask_report_flags_a_cut():
    v, h1, h2 = _maps()
    m, _, _ = mk.make_mask(0.5 * (h1 + h2), APIX, lowpass_a=10.0, threshold=0.2, n_segments=0,
                           exclude=[mk.parse_region("sphere 122 96 96 30")])
    r = mk.mask_report(m, APIX, 0.5 * (h1 + h2))
    assert r["density_outside"] > 0.2
