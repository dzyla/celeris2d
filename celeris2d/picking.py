"""Particle picking and extraction on the GPU.

Blob picking: the micrograph is Fourier-binned to ~diameter/16 per pixel, phase-flipped with its CTF, inverted
if particles are dark (cryo), band-passed, and correlated with a soft disk of the particle diameter (normalised
cross-correlation over a window 1.4x the disk, so contrast alone does not score). Local maxima at least
min_dist apart are candidates; each carries features (score, local standard deviation, large-scale mean) used to
threshold and to reject contamination (carbon, ice). The candidates and their scores are kept, so a threshold can
change without picking again.

    cand = blob_candidates(mic, apix, diameter_a=160, ctf=fit, optics=optics)
    keep = select(cand)                       # automatic threshold + contamination filter
    imgs = extract(mic, apix, cand.xy_a[keep], box=suggest_box(160 * 1.5, apix))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .preprocess import _grid, _shift_phase, crop_half

NICE = sorted({2 ** a * 3 ** b * 5 ** c for a in range(1, 11) for b in range(3) for c in range(3)
               if 32 <= 2 ** a * 3 ** b * 5 ** c <= 1024})


def box_for(particle_a: float, defocus_a: float, res_a: float = 4.0, wavelength_a: float = 0.0197) -> float:
    """Box edge (A) that holds the particle and its CTF-delocalised signal up to res_a: a point of the particle
    spreads by lambda * defocus / d at resolution d, on both sides; at least 1.5 x the particle.
    T20S (150 A long, 1.7 um defocus, 4 A): 320 A."""
    return max(1.5 * particle_a, particle_a + 2 * wavelength_a * defocus_a / res_a)


def suggest_box(box_a: float, apix: float) -> int:
    """Smallest FFT-friendly even box (px) of at least box_a."""
    n = math.ceil(box_a / apix)
    return next(b for b in NICE if b >= n and b % 2 == 0)


def ctf_sign(ny: int, nx: int, apix: float, du: float, dv: float, ang_deg: float, optics, dev) -> torch.Tensor:
    """sign(CTF) on an rfft2 grid (ny, nx//2+1), RELION convention (as engine.ctf_2d)."""
    ky, kx = _grid(ny, nx, apix, dev)
    s2 = ky * ky + kx * kx
    a = torch.atan2(ky.expand_as(s2), kx.expand_as(s2))
    df = 0.5 * (du + dv) + 0.5 * (du - dv) * torch.cos(2 * (a - math.radians(ang_deg)))
    lam, cs = optics.wavelength, optics.cs_mm * 1e7
    q0 = optics.q0
    chi = math.pi * lam * df * s2 - 0.5 * math.pi * cs * lam ** 3 * s2 * s2 + math.atan(q0 / math.sqrt(1 - q0 * q0))
    return torch.sign(torch.sin(chi))


def prepare(mic: torch.Tensor, apix: float, diameter_a: float, ctf=None, optics=None, dark: bool = True,
            work_apix: float | None = None):
    """Binned, phase-flipped, band-passed micrograph with white particles: (image, its pixel size)."""
    ny, nx = mic.shape
    wa = max(apix, work_apix or diameter_a / 16)
    my, mx = int(round(ny * apix / wa / 2)) * 2, int(round(nx * apix / wa / 2)) * 2
    wa = apix * ny / my
    F = crop_half(torch.fft.rfft2(mic.float(), norm="forward"), my, mx)
    if ctf is not None and optics is not None:
        F = F * ctf_sign(my, mx, wa, ctf.defocus_u, ctf.defocus_v, ctf.angle, optics, mic.device)
    ky, kx = _grid(my, mx, wa, mic.device)
    k2 = ky * ky + kx * kx
    hp = 1 - torch.exp(-k2 * (2 * diameter_a) ** 2)                 # remove gradients beyond ~2 diameters
    lp = torch.exp(-k2 * (diameter_a / 8) ** 2)                      # smooth below ~diameter/8
    F = F * hp * lp * (-1.0 if dark else 1.0)
    F[0, 0] = 0
    img = torch.fft.irfft2(F, s=(my, mx), norm="forward")
    return img / img.std(), wa


@dataclass
class Candidates:
    xy_a: np.ndarray            # (n, 2) position in A from the micrograph origin (x = column, y = row)
    score: np.ndarray           # normalised cross-correlation with the disk
    local_std: np.ndarray       # standard deviation in the window, relative to the micrograph median
    local_mean: np.ndarray      # mean over ~2 diameters, in micrograph standard deviations
    work: dict = field(default_factory=dict)   # the binned image and score map, for figures


def _conv(img_ft, ker, shape):
    """Circular convolution of an image (its rfft2) with a centred kernel of the same shape."""
    return torch.fft.irfft2(img_ft * torch.fft.rfft2(torch.fft.ifftshift(ker)), s=shape)


def blob_candidates(mic: torch.Tensor, apix: float, diameter_a: float, ctf=None, optics=None, dark: bool = True,
                    min_dist_frac: float = 0.8, keep_maps: bool = False) -> Candidates:
    img, wa = prepare(mic, apix, diameter_a, ctf, optics, dark)
    dev, (h, w) = img.device, img.shape
    yy = (torch.arange(h, device=dev) - h // 2).float()[:, None] * wa
    xx = (torch.arange(w, device=dev) - w // 2).float()[None, :] * wa
    rr = torch.sqrt(yy * yy + xx * xx)
    r = diameter_a / 2
    disk = torch.sigmoid((r - rr) / (0.08 * r))
    win = (rr <= 1.4 * r).float()
    nwin = float(win.sum())
    t = (disk - (disk * win).sum() / nwin) * win
    t = t / t.norm()
    Fi = torch.fft.rfft2(img)
    num = _conv(Fi, t, (h, w))
    mu = _conv(Fi, win, (h, w)) / nwin
    var = (_conv(torch.fft.rfft2(img * img), win, (h, w)) / nwin - mu * mu).clamp_min(1e-6)
    ncc = num / torch.sqrt(var * nwin)
    big = (rr <= diameter_a).float()                                # ~2 diameters across
    mean_big = _conv(Fi, big, (h, w)) / float(big.sum())
    k = 2 * max(1, int(round(min_dist_frac * diameter_a / wa / 2))) + 1
    peak = (ncc == torch.nn.functional.max_pool2d(ncc[None, None], k, 1, k // 2)[0, 0]) & (ncc > 0)
    e = int(math.ceil(r / wa))
    peak[:e], peak[-e:], peak[:, :e], peak[:, -e:] = False, False, False, False
    yx = torch.nonzero(peak)
    sd = var.sqrt()
    s = ncc[yx[:, 0], yx[:, 1]]
    o = torch.argsort(s, descending=True)
    yx, s = yx[o], s[o]
    # sub-pixel position: parabola through the score map
    def para(m, c, p):
        d = m - 2 * c + p
        return torch.where(d.abs() > 1e-9, 0.5 * (m - p) / d, torch.zeros_like(d)).clamp(-0.5, 0.5)
    y, x = yx[:, 0], yx[:, 1]
    oy = para(ncc[y - 1, x], s, ncc[y + 1, x])
    ox = para(ncc[y, x - 1], s, ncc[y, x + 1])
    xy = torch.stack([(x.float() + ox) * wa, (y.float() + oy) * wa], 1)
    med = float(sd[e:-e, e:-e].median())
    work = dict(image=img.cpu().numpy(), score=ncc.cpu().numpy(), apix=wa) if keep_maps else {}
    return Candidates(xy.cpu().numpy(), s.cpu().numpy(), (sd[y, x] / med).cpu().numpy(),
                      mean_big[y, x].cpu().numpy(), work)


def auto_threshold(score: np.ndarray) -> float:
    """Generous score threshold: the mode of the candidate scores (smoothed histogram). Below it the candidates
    are mostly noise maxima; above it particles take over, mixed with some background that 2D classification
    removes. EMPIAR-10025 (dense T20S, 120 A blob): ~14k picks on 19 micrographs, 80 % of the 10k particles of
    cryoSPARC's refined set (whose own blob/template picks were cleaned by 2D)."""
    if len(score) < 20:
        return float(np.min(score)) if len(score) else 0.0
    h, e = np.histogram(score, bins=40)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    i = int(np.argmax(h))
    return float(0.5 * (e[i] + e[i + 1]))


def select(c: Candidates, threshold: float | None = None, max_std: float = 2.0, max_mean: float = 3.0):
    """Indices of candidates kept: score >= threshold (auto if None), and not in high-variance (carbon edge, ice
    crystal) or extreme-mean (thick ice, carbon) regions."""
    th = auto_threshold(c.score) if threshold is None else threshold
    ok = (c.score >= th) & (c.local_std <= max_std) & (np.abs(c.local_mean) <= max_mean)
    return np.flatnonzero(ok)


def extract(mic: torch.Tensor, apix: float, xy_a: np.ndarray, box: int, out_apix: float | None = None,
            dark: bool = True, diameter_a: float | None = None) -> np.ndarray:
    """Particles (n, b, b) float32 cut at the (sub-pixel) positions, white protein, normalised to zero mean and
    unit standard deviation over the background (outside 0.55 diameter, or outside 0.8 of the box radius).
    out_apix > apix: Fourier-binned to that pixel size (box is then in output pixels)."""
    dev = mic.device
    f = (out_apix or apix) / apix
    bi = int(math.ceil(box * f / 2)) * 2                                   # box at the micrograph pixel size
    h = bi // 2
    m = torch.nn.functional.pad(mic.float()[None, None], (h + 2,) * 4, mode="reflect")[0, 0]
    x, y = np.asarray(xy_a)[:, 0] / apix, np.asarray(xy_a)[:, 1] / apix
    x0, y0 = np.round(x).astype(int), np.round(y).astype(int)
    out = torch.empty(len(x), box, box, device=dev)
    ky, kx = _grid(bi, bi, 1.0, dev)
    yy = (torch.arange(box, device=dev) - box // 2).float()
    rr = torch.sqrt(yy[:, None] ** 2 + yy[None, :] ** 2)
    rb = (0.55 * diameter_a / (out_apix or apix)) if diameter_a else 0.8 * box / 2
    bg = rr > min(rb, 0.9 * box / 2)
    W = m.shape[1]
    grid = (torch.arange(bi, device=dev)[:, None] * W + torch.arange(bi, device=dev)[None, :]).view(1, bi, bi)
    base = torch.as_tensor((y0 + 2) * W + (x0 + 2), device=dev)
    flat = m.reshape(-1)
    for i in range(0, len(x), 256):
        sl = slice(i, i + 256)
        idx = range(i, min(i + 256, len(x)))
        p = flat[base[sl, None, None] + grid]                             # all boxes of the chunk in one gather
        dx = torch.as_tensor(x[sl] - x0[sl], device=dev, dtype=torch.float32)
        dy = torch.as_tensor(y[sl] - y0[sl], device=dev, dtype=torch.float32)
        F = torch.fft.rfft2(p, norm="forward") * _shift_phase(ky, kx, -dy, -dx)
        if bi != box:
            F = crop_half(F, box, box)
        p = torch.fft.irfft2(F, s=(box, box), norm="forward") * (-1.0 if dark else 1.0)
        b = p[:, bg]
        p = (p - b.mean(1)[:, None, None]) / b.std(1).clamp_min(1e-6)[:, None, None]
        out[i:i + len(idx)] = p
    return out.cpu().numpy()


def match(xy_a: np.ndarray, ref_a: np.ndarray, radius_a: float):
    """Greedy one-to-one matching of picks to reference positions within radius: (matched pick mask,
    matched reference mask)."""
    from scipy.spatial import cKDTree
    mp, mr = np.zeros(len(xy_a), bool), np.zeros(len(ref_a), bool)
    if not len(xy_a) or not len(ref_a):
        return mp, mr
    d, j = cKDTree(ref_a).query(xy_a, k=1, distance_upper_bound=radius_a)
    for i in np.argsort(d):
        if np.isfinite(d[i]) and not mr[j[i]]:
            mp[i], mr[j[i]] = True, True
    return mp, mr
