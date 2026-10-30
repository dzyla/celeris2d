"""HeteroRefine "decoy" references: the first map's phases randomised beyond decoy_res_a - same low-resolution
envelope and power, no structure finer than that (CPU, < 1 s)."""
import numpy as np
import torch

from celeris2d.jobs import decoy_map


def _shell_cc(a, b, box_a, lo_a, hi_a):
    Fa, Fb = torch.fft.rfftn(a), torch.fft.rfftn(b)
    n = a.shape[0]
    k = torch.fft.fftfreq(n) * n
    kz = torch.arange(n // 2 + 1, dtype=torch.float32)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)
    m = (kk >= box_a / lo_a) & (kk < box_a / hi_a)
    return float((Fa[m] * Fb[m].conj()).real.sum() / torch.sqrt((Fa[m].abs() ** 2).sum() * (Fb[m].abs() ** 2).sum()))


def test_decoy_keeps_envelope_not_structure():
    r = np.random.default_rng(0)
    n, apix = 64, 3.0
    x = torch.arange(n, dtype=torch.float32) - n // 2
    z, y, x = torch.meshgrid(x, x, x, indexing="ij")
    v = torch.zeros(n, n, n)
    for _ in range(30):
        c = r.uniform(-12, 12, 3)
        v += torch.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * 1.5 ** 2))
    d = decoy_map(v, n * apix, 30.0, 0.9 * n * apix, seed=1)
    assert _shell_cc(v, d, n * apix, 200, 40) > 0.9                  # envelope kept
    assert abs(_shell_cc(v, d, n * apix, 15, 8)) < 0.2               # structure gone
    assert 0.5 < float(d.std() / v.std()) < 2.0                      # same scale
    d2 = decoy_map(v, n * apix, 30.0, 0.9 * n * apix, seed=2)
    assert abs(_shell_cc(d, d2, n * apix, 15, 8)) < 0.2              # two decoys differ
