"""Local resolution from two half maps (blocres / RELION LocalRes style), on the GPU.

At points of a grid (stride px) inside the mask, both half maps are cut out in a cube of window_a and multiplied by a
smooth (Hann) sphere; the FSC of the two windows, shell by shell, crosses 0.143 at the local resolution (linear
interpolation between shells). The coarse grid is interpolated to every voxel. Checked on synthetic half maps whose
resolution changes across the box (tests/test_localres.py: each region within 6% of its true resolution;
with a window of 5x the resolution the estimate is 5-7% too fine, at 6-7x 2-3%).

    localres(h1, h2, mask, apix, res_global) -> dict(map (b^3, A), filtered (b^3), window_a, stride, stats)

The window: about 7x the global resolution (at least 20 px and 20 A): smaller windows give noisier FSC curves (few
Fourier voxels per shell), larger ones average over a bigger region. The filtered map is the average of the half
maps low-passed at each voxel to its local resolution (blend of maps low-passed at a ladder of resolutions).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _window(d: int, dev) -> torch.Tensor:
    r = torch.arange(d, device=dev, dtype=torch.float32) - (d - 1) / 2
    rr = torch.sqrt(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2) / (d / 2)
    return torch.where(rr < 1, 0.5 + 0.5 * torch.cos(math.pi * rr), torch.zeros_like(rr))


def default_window(res_global: float, apix: float, box: int) -> int:
    d = max(20, int(round(max(7 * res_global, 20.0) / apix)))
    d = min(d, box // 2)
    return d + (d % 2)


def localres(h1: np.ndarray | torch.Tensor, h2: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None,
             apix: float, res_global: float, window_px: int = 0, stride: int = 0, thr: float = 0.143,
             filtered: bool = True, dev=None, batch: int = 2048) -> dict:
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    h1 = torch.as_tensor(np.asarray(h1), dtype=torch.float32, device=dev)
    h2 = torch.as_tensor(np.asarray(h2), dtype=torch.float32, device=dev)
    b = h1.shape[0]
    m = torch.ones_like(h1) if mask is None else torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=dev)
    d = window_px or default_window(res_global, apix, b)
    st = stride or max(2, d // 6)
    win = _window(d, dev).reshape(-1)
    # Fourier shells of the window (rfftn of d^3)
    k = torch.fft.fftfreq(d, device=dev) * d
    kz = torch.arange(d // 2 + 1, device=dev, dtype=torch.float32)
    sh = torch.round(torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)).long().reshape(-1)
    nsh = d // 2 + 1
    keep = sh < nsh
    sh = sh[keep]
    onehot = F.one_hot(sh, nsh).float()                                       # (nvox_kept, nsh)
    # grid points (coarse) whose window centre lies inside the mask
    g = torch.arange(0, b, st, device=dev)
    nc = len(g)
    gz, gy, gx = torch.meshgrid(g, g, g, indexing="ij")
    cen = torch.stack([gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)], 1)
    inside = m[cen[:, 0], cen[:, 1], cen[:, 2]] > 0.5
    # windows must stay in the box: centres within d/2 of the edge are clamped
    lo = d // 2
    cc = cen[inside].clamp(lo, b - d + lo)
    o = torch.arange(d, device=dev) - lo
    oz, oy, ox = torch.meshgrid(o, o, o, indexing="ij")
    off = (oz * b * b + oy * b + ox).reshape(-1)
    f1, f2 = h1.reshape(-1), h2.reshape(-1)
    res = torch.full((len(cc),), float("nan"), device=dev)
    for i in range(0, len(cc), batch):
        c = cc[i:i + batch]
        base = c[:, 0] * b * b + c[:, 1] * b + c[:, 2]
        ix = base[:, None] + off[None]
        p1 = (f1[ix] * win).reshape(-1, d, d, d)
        p2 = (f2[ix] * win).reshape(-1, d, d, d)
        F1 = torch.fft.rfftn(p1, dim=(1, 2, 3)).reshape(len(c), -1)[:, keep]
        F2 = torch.fft.rfftn(p2, dim=(1, 2, 3)).reshape(len(c), -1)[:, keep]
        num = (F1 * F2.conj()).real @ onehot
        den = torch.sqrt(((F1.abs() ** 2) @ onehot) * ((F2.abs() ** 2) @ onehot)).clamp_min(1e-30)
        fsc = num / den                                                        # (B, nsh)
        below = fsc[:, 1:] < thr
        below[:, -1] = True
        s1 = below.float().argmax(1) + 1                                       # first shell below thr
        fa = fsc.gather(1, (s1 - 1)[:, None]).squeeze(1)
        fb = fsc.gather(1, s1[:, None]).squeeze(1)
        frac = ((fa - thr) / (fa - fb).clamp_min(1e-6)).clamp(0, 1)
        sf = (s1 - 1).float() + frac
        crossed = fb < thr
        sf = torch.where(crossed, sf, torch.full_like(sf, nsh - 1.0))
        res[i:i + batch] = d * apix / sf.clamp_min(1.0)
    nyq = 2 * apix
    res = res.clamp(nyq, d * apix)
    coarse = torch.full((nc ** 3,), float("nan"), device=dev)
    coarse[inside] = res
    fill = float(res.median()) if len(res) else res_global
    coarse = torch.nan_to_num(coarse, nan=fill).reshape(nc, nc, nc)
    # trilinear interpolation to every voxel (coarse point i sits at voxel i * st)
    x = torch.arange(b, device=dev, dtype=torch.float32) / st / max(nc - 1, 1) * 2 - 1
    zz, yy, xx = torch.meshgrid(x, x, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], -1)[None]                                # grid_sample wants (x, y, z)
    full = F.grid_sample(coarse[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    vals = full[m > 0.5]
    q = torch.quantile(vals[torch.randperm(len(vals), device=dev)[:1_000_000]], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=dev)) \
        if len(vals) else torch.full((5,), float("nan"))
    hist_edges = np.round(np.linspace(nyq, min(d * apix, 4 * max(res_global, nyq)), 41), 3)
    hist = np.histogram(vals.cpu().numpy(), bins=hist_edges)[0]
    stats = dict(median_a=round(float(q[2]), 2), p05_a=round(float(q[0]), 2), p25_a=round(float(q[1]), 2),
                 p75_a=round(float(q[3]), 2), p95_a=round(float(q[4]), 2), window_a=round(d * apix, 1),
                 window_px=d, stride_px=st, points=int(len(res)), threshold=thr,
                 at_nyquist=round(float((vals <= nyq * 1.02).float().mean()) if len(vals) else 0.0, 4),
                 hist_edges_a=hist_edges.tolist(), hist_counts=hist.tolist())
    out = dict(map=full, stats=stats)
    if filtered:
        out["filtered"] = local_filter(0.5 * (h1 + h2), full, apix)
    return out


def local_filter(v: torch.Tensor, lres: torch.Tensor, apix: float, n_levels: int = 16) -> torch.Tensor:
    """Each voxel of v low-passed to its local resolution: maps low-passed at a ladder of resolutions (cosine edge),
    linearly blended between the two levels bracketing the voxel's value."""
    b = v.shape[0]
    dev = v.device
    k = torch.fft.fftfreq(b, device=dev)
    kz = torch.fft.rfftfreq(b, device=dev)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2) / apix   # 1/A
    Fv = torch.fft.rfftn(v)
    lo, hi = float(lres.min()), float(lres.max())
    levels = np.geomspace(max(lo, 2 * apix), max(hi, lo * 1.01), n_levels)
    out = torch.zeros_like(v)
    wsum = torch.zeros_like(v)
    for i, r in enumerate(levels):
        edge = 1 / r
        w = 0.5 + 0.5 * torch.cos(math.pi * ((kk - edge) / (0.15 * edge)).clamp(0, 1))
        w = torch.where(kk <= edge, torch.ones_like(kk), w)
        lp = torch.fft.irfftn(Fv * w, s=v.shape)
        # hat weight of this level for each voxel (in log resolution)
        lr = torch.log(lres.clamp_min(1e-3))
        c = math.log(r)
        left = math.log(levels[i - 1]) if i > 0 else c - 1e9
        right = math.log(levels[i + 1]) if i + 1 < n_levels else c + 1e9
        wt = torch.where(lr <= c, ((lr - left) / max(c - left, 1e-9)).clamp(0, 1),
                         ((right - lr) / max(right - c, 1e-9)).clamp(0, 1))
        if i == 0:
            wt = torch.where(lr <= c, torch.ones_like(wt), wt)
        if i == n_levels - 1:
            wt = torch.where(lr >= c, torch.ones_like(wt), wt)
        out += wt * lp
        wsum += wt
    return out / wsum.clamp_min(1e-6)
