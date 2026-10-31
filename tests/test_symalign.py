"""symalign: a D2 object turned by a known rotation is put back into the symmetry frame (n-fold on z, 2-fold on x,
up to the group's own operators), and apply_frame keeps every projection: the map sampled at R r with poses A @ R
gives the images of the original map with poses A."""
import math

import numpy as np
import torch

from celeris2d import gpu
from celeris2d import abinitio as ab
from celeris2d.symalign import apply_frame, sym_frame


def _blob(b, centres, sig=3.0):
    c = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    v = torch.zeros(b, b, b)
    for p in centres:
        v += torch.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) / (2 * sig ** 2))
    return v


def _d2_points():
    base = [(9, 4, 6), (5, -7, 3), (12, 2, -2)]          # asymmetric unit; D2 ops: x,y,z 2-folds
    out = []
    for (x, y, z) in base:
        out += [(x, y, z), (-x, -y, z), (x, -y, -z), (-x, y, -z)]
    return out


def test_sym_frame_recovers_d2_axes():
    b = 48
    dev = gpu.default()
    v0 = _blob(b, _d2_points())
    Rt = ab.euler_to_matrix(torch.tensor([23.0]), torch.tensor([37.0]), torch.tensor([71.0]))[0]
    v = ab._rotate_volumes(v0.to(dev), Rt[None].to(dev))[0].cpu()          # the object in an arbitrary frame
    R, info = sym_frame(v, 2.0, "D2", res_a=6.0, nb=48, n_axes=1500, log=lambda *a: None)
    back = ab._rotate_volumes(v.to(dev), R[None].to(dev))[0].cpu()
    # in the found frame the D2 operators hold
    for S in ab.symmetry_ops("D2")[1:]:
        w = ab._rotate_volumes(back.to(dev), S[None].float().to(dev))[0].cpu()
        cc = float(torch.corrcoef(torch.stack([back.flatten(), w.flatten()]))[0, 1])
        assert cc > 0.95, (cc, info)
    assert info["scores"]["C2 about its axis"] > 0.95


def test_apply_frame_keeps_projections():
    b = 40
    dev = gpu.default()
    rng = np.random.default_rng(0)
    v = _blob(b, [tuple(rng.uniform(-10, 10, 3)) for _ in range(6)])
    A = ab.euler_to_matrix(torch.rand(5) * 360, torch.rand(5) * 180, torch.rand(5) * 360).float()
    R = ab.euler_to_matrix(torch.tensor([40.0]), torch.tensor([65.0]), torch.tensor([-20.0]))[0].float()
    v2, A2 = apply_frame(v, A, R)
    p = []
    for vol, AA in ((v, A), (v2, A2)):
        V = ab.Volume(b, dev)
        V.set_real(vol.to(dev))
        p.append(torch.fft.irfft2(V.project(AA.to(dev)), s=(b, b)).cpu())
    cc = [float(torch.corrcoef(torch.stack([p[0][i].flatten(), p[1][i].flatten()]))[0, 1]) for i in range(5)]
    assert min(cc) > 0.97, cc
