"""Automatic soft masks from two half maps (Sharpen; usable for anything that needs a particle mask).

    auto_mask(h1, h2, apix) -> (mask (n^3, 0..1), info)

The half maps' average and half difference, both low-passed to lowpass_a: the difference holds only noise,
filtered exactly like the average, so its spread sets the threshold where the density stands clear of noise
(sigma_level x its standard deviation, never below min_frac of the maximum). Pieces much smaller than the particle
(noise, detached blobs: below keep_frac of the largest piece) are dropped; the rest is grown by dilate_a and given a
cosine edge of soft_a, both measured with a Euclidean distance (round outline, not the blocky one of repeated 3^3
dilations). A soft edge of about 6-8 px or more keeps the mask from correlating the half maps by itself.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _lowpass(v: torch.Tensor, apix: float, res_a: float) -> torch.Tensor:
    n = v.shape[0]
    k = torch.fft.fftfreq(n, device=v.device) * n
    kz = torch.arange(n // 2 + 1, device=v.device, dtype=torch.float32)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2) / (n * apix)
    edge = 1 / res_a
    w = 0.5 + 0.5 * torch.cos(math.pi * ((kk - edge) / (0.3 * edge)).clamp(0, 1))
    w = torch.where(kk <= edge, torch.ones_like(w), w)
    return torch.fft.irfftn(torch.fft.rfftn(v) * w, s=v.shape)


def auto_mask(h1, h2, apix: float, lowpass_a: float = 15.0, sigma_level: float = 5.0, min_frac: float = 0.05,
              keep_frac: float = 0.05, dilate_a: float = 3.0, soft_a: float = 0.0, dev=None) -> tuple[np.ndarray, dict]:
    """soft_a 0: max(8 A, 6 px)."""
    from scipy import ndimage
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    t1 = torch.as_tensor(np.asarray(h1), dtype=torch.float32, device=dev)
    t2 = torch.as_tensor(np.asarray(h2), dtype=torch.float32, device=dev)
    avg = _lowpass(0.5 * (t1 + t2), apix, lowpass_a)
    dif = _lowpass(0.5 * (t1 - t2), apix, lowpass_a)
    noise = float(dif.std())
    vmax = float(avg.max())
    thr = max(sigma_level * noise, min_frac * vmax)
    binary = (avg > thr).cpu().numpy()
    lab, n_lab = ndimage.label(binary)
    if n_lab > 1:
        size = np.bincount(lab.ravel())[1:]
        keep = np.flatnonzero(size >= keep_frac * size.max()) + 1
        binary = np.isin(lab, keep)
    soft_a = soft_a or max(8.0, 6 * apix)
    d = ndimage.distance_transform_edt(~binary) * apix                     # A from the thresholded density
    x = np.clip((d - dilate_a) / soft_a, 0, 1)
    mask = (0.5 + 0.5 * np.cos(np.pi * x)).astype(np.float32)
    info = dict(lowpass_a=lowpass_a, threshold=round(thr, 6), threshold_sigma=round(thr / max(noise, 1e-30), 2),
                threshold_frac_max=round(thr / max(vmax, 1e-30), 4), pieces=int(n_lab),
                pieces_kept=int(len(np.unique(lab[binary]))) if n_lab else 0, dilate_a=dilate_a,
                soft_a=round(soft_a, 2), volume_fraction=round(float(mask.mean()), 4),
                core_fraction=round(float(binary.mean()), 4))
    return mask, info
