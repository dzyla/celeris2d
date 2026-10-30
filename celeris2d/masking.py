"""Solvent masks for refinement and post-processing.

auto_mask: the map low-passed (to ~1.5 x its resolution, 8-15 A), a threshold set from the solvent itself (mean + k
noise standard deviations of the low-passed map outside the particle; never below a small fraction of the peak), the
connected particle density kept (specks of noise and detached blobs below a share of the largest piece dropped),
then grown by `extend_a` and given a cosine edge `soft_a` along the exact (Euclidean) distance from it. Built on a
grid of ~lowpass/4 voxels (a mask has no finer detail) and interpolated to the map's grid, so it is cheap enough to
be recomputed every refinement iteration (dynamic masking, as cryoSPARC).

make_mask: the same with user choices on top: an explicit threshold, regions (spheres, boxes) to include or exclude,
segments to keep (segment_map: the masked density split around its strongest peaks; numbered in a figure), inversion.

mask_report: what a mask holds: volume, share of the box, the low-passed map's density outside it (a mask that cuts
the particle), its own correlation of phase-randomised half maps (a mask too sharp to trust), and the masked FSC.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _lowpass(v: torch.Tensor, apix: float, res_a: float) -> torch.Tensor:
    """Smooth low-pass (cosine edge over 10 % of the frequency) of a real map to res_a."""
    n = v.shape[0]
    k = torch.fft.fftfreq(n, device=v.device) / apix
    kz = torch.fft.rfftfreq(n, device=v.device) / apix
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)
    k0 = 1.0 / res_a
    w = torch.clamp((kk - 0.9 * k0) / (0.2 * k0), 0, 1)
    f = 0.5 * (1 + torch.cos(math.pi * w))
    return torch.fft.irfftn(torch.fft.rfftn(v) * f, s=v.shape)


def _resize(v: torch.Tensor, n: int) -> torch.Tensor:
    """Fourier crop / pad of a real cube to n^3 (same box in A)."""
    b = v.shape[0]
    if b == n:
        return v
    ft = torch.fft.fftshift(torch.fft.fftn(v))
    if n < b:
        o = (b - n) // 2
        ft = ft[o:o + n, o:o + n, o:o + n]
    else:
        z = torch.zeros((n,) * 3, dtype=ft.dtype, device=v.device)
        o = (n - b) // 2
        z[o:o + b, o:o + b, o:o + b] = ft
        ft = z
    return torch.fft.ifftn(torch.fft.ifftshift(ft)).real * (n / b) ** 3


def solvent_stats(lp: torch.Tensor, apix: float, diameter_a: float | None) -> tuple[float, float]:
    """Mean and robust standard deviation of a low-passed map in its solvent: outside the particle's sphere
    (diameter_a; default 70 % of the box) and inside the box's inscribed sphere (the corners hold nothing measured)."""
    n = lp.shape[0]
    r = torch.arange(n, device=lp.device, dtype=torch.float32) - n // 2
    rr = torch.sqrt(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2) * apix
    r_in = 0.5 * (diameter_a if diameter_a else 0.7 * n * apix)
    sel = (rr > r_in) & (rr < 0.5 * n * apix)
    if int(sel.sum()) < 1000:
        sel = rr > 0.4 * n * apix
    x = lp[sel]
    med = x.median()
    mad = (x - med).abs().median()
    return float(med), float(1.4826 * mad)


def _components(m: np.ndarray, keep_frac: float) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= keep_frac * sizes.max()) + 1
    return np.isin(lab, keep)


def _soften(core: np.ndarray, vox: float, extend_a: float, soft_a: float) -> np.ndarray:
    """Distance (A) from the binary core -> 1 within extend_a, cosine to 0 over soft_a beyond it."""
    d = ndimage.distance_transform_edt(~core) * vox
    x = np.clip((d - extend_a) / max(soft_a, 1e-6), 0, 1)
    return (0.5 * (1 + np.cos(np.pi * x))).astype(np.float32)


def _grid(n: int, apix: float, lowpass_a: float) -> tuple[int, float]:
    """Working grid for building a mask: voxels of ~lowpass/4, at most the map's."""
    nb = min(n, max(32, int(math.ceil(n * apix / (lowpass_a / 4) / 2)) * 2))
    return nb, n * apix / nb


def noise_sigma(h1: torch.Tensor, h2: torch.Tensor, nb: int, vox: float, lp_a: float) -> float:
    """Noise standard deviation of the low-passed average of two half maps: (h1 - h2) / 2 is pure noise with the
    statistics of the average, everywhere (also inside the particle, and in a map already masked to a sphere, whose
    solvent is zero), restricted to where the halves are not both zero."""
    d = _lowpass(_resize(0.5 * (h1.float() - h2.float()), nb), vox, lp_a)
    a = _resize(0.5 * (h1.float() + h2.float()), nb)
    sel = a.abs() > 1e-12 * a.abs().max()
    x = d[sel] if int(sel.sum()) > 1000 else d.reshape(-1)
    return float(1.4826 * (x - x.median()).abs().median())


def auto_mask(vmap: torch.Tensor, apix: float, res_a: float | None = None, lowpass_a: float | None = None,
              k_sigma: float = 6.0, min_frac: float = 0.03, extend_a: float = 3.0, soft_a: float = 6.0,
              keep_frac: float = 0.05, diameter_a: float | None = None, halves=None,
              info: dict | None = None) -> torch.Tensor:
    """Soft solvent mask of a real map (n^3, apix). res_a: the map's resolution (sets the low-pass to 1.5 x res_a,
    within 8-15 A; lowpass_a overrides). Threshold: k_sigma noise standard deviations above the solvent level, the
    noise from the half maps (halves=(h1, h2): their difference) or else from the solvent outside the particle; at
    least min_frac of the low-passed map's peak. info (dict): filled with the numbers used."""
    n = vmap.shape[0]
    lp_a = lowpass_a or (min(15.0, max(8.0, 1.5 * res_a)) if res_a else 12.0)
    nb, vox = _grid(n, apix, lp_a)
    v = _resize(vmap.float(), nb)
    lp = _lowpass(v, vox, lp_a)
    mu, sd = solvent_stats(lp, vox, diameter_a)
    if halves is not None:
        mu, sd = 0.0, noise_sigma(halves[0], halves[1], nb, vox, lp_a)
    peak = float(lp.max())
    thr = max(mu + k_sigma * sd, mu + min_frac * (peak - mu))
    core = _components((lp > thr).cpu().numpy(), keep_frac)
    soft = torch.as_tensor(_soften(core, vox, extend_a, soft_a), device=vmap.device)
    out = F.interpolate(soft[None, None], size=(n, n, n), mode="trilinear", align_corners=False)[0, 0] if nb != n else soft
    if info is not None:
        info.update(lowpass_a=round(lp_a, 2), threshold=thr, threshold_sigma=round((thr - mu) / max(sd, 1e-12), 2),
                    threshold_of_peak=round((thr - mu) / max(peak - mu, 1e-12), 4), extend_a=extend_a, soft_a=soft_a,
                    core_volume_a3=float(core.sum() * vox ** 3), working_grid=nb)
    return out.clamp(0, 1)


# ------------------------------------------------------------------------------------------------ user masks
def _region(n: int, apix: float, shape: dict, dev) -> torch.Tensor:
    """Boolean region of a sphere {"sphere": [x, y, z, r]} or box {"box": [x0, y0, z0, x1, y1, z1]} in A, map
    coordinates from the box corner (as ChimeraX shows them for a map at the origin), x fastest (array [z, y, x])."""
    c = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) * apix
    z, y, x = c[:, None, None], c[None, :, None], c[None, None, :]
    if "sphere" in shape:
        cx, cy, cz, r = shape["sphere"]
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r
    if "box" in shape:
        x0, y0, z0, x1, y1, z1 = shape["box"]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
    raise ValueError(f"region {shape}: use {{'sphere': [x, y, z, r]}} or {{'box': [x0, y0, z0, x1, y1, z1]}} (A)")


def segment_map(vmap: torch.Tensor, apix: float, core: np.ndarray, vox: float, lp: torch.Tensor, n_max: int = 20,
                min_sep_a: float = 20.0) -> np.ndarray:
    """Split a binary core (working grid) into segments around the strongest peaks of the low-passed map lp that are
    at least min_sep_a apart (every core voxel joins its nearest peak): labels 1..k, 0 outside."""
    a = lp.cpu().numpy()
    a = np.where(core, a, -np.inf)
    size = max(3, int(round(min_sep_a / vox)) | 1)
    peaks = (a == ndimage.maximum_filter(a, size=size)) & core
    idx = np.argwhere(peaks)
    if not len(idx):
        return core.astype(np.int32)
    idx = idx[np.argsort(-a[tuple(idx.T)])][:n_max]
    seeds = np.zeros(core.shape, np.int32)
    seeds[tuple(idx.T)] = np.arange(1, len(idx) + 1)
    _, ind = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    lab = seeds[tuple(ind)]
    return np.where(core, lab, 0).astype(np.int32)


def make_mask(vmap: torch.Tensor, apix: float, res_a: float | None = None, lowpass_a: float | None = None,
              threshold: float | None = None, k_sigma: float = 6.0, extend_a: float = 3.0, soft_a: float = 6.0,
              keep_frac: float = 0.05, include=(), exclude=(), segments=(), n_segments: int = 12,
              invert: bool = False, diameter_a: float | None = None):
    """A mask with user choices: threshold (map value of the low-passed map; None: automatic as auto_mask), regions
    to add (include: always in) or remove (exclude), segments to keep (1-based numbers of segment_map; empty: all),
    inversion (the mask of everything else, e.g. for signal subtraction). Returns (mask (n^3), info, segment labels
    on the map's grid (n^3 int) or None)."""
    n = vmap.shape[0]
    dev = vmap.device
    lp_a = lowpass_a or (min(15.0, max(8.0, 1.5 * res_a)) if res_a else 12.0)
    nb, vox = _grid(n, apix, lp_a)
    lp = _lowpass(_resize(vmap.float(), nb), vox, lp_a)
    mu, sd = solvent_stats(lp, vox, diameter_a)
    peak = float(lp.max())
    thr = threshold if threshold is not None else max(mu + k_sigma * sd, mu + 0.03 * (peak - mu))
    core = _components((lp > thr).cpu().numpy(), keep_frac)
    labels = None
    if segments or n_segments:
        labels = segment_map(vmap, apix, core, vox, lp, n_max=n_segments)
        if segments:
            core = np.isin(labels, np.asarray(segments, int))
    for s in exclude:
        core &= ~_region(nb, vox, s, dev).cpu().numpy()
    for s in include:
        core |= _region(nb, vox, s, dev).cpu().numpy()
    soft = torch.as_tensor(_soften(core, vox, extend_a, soft_a), device=dev)
    out = F.interpolate(soft[None, None], size=(n, n, n), mode="trilinear", align_corners=False)[0, 0] if nb != n else soft
    out = out.clamp(0, 1)
    if invert:
        out = 1 - out
    if labels is not None:
        lab_t = torch.as_tensor(labels, device=dev, dtype=torch.float32)[None, None]
        labels = F.interpolate(lab_t, size=(n, n, n), mode="nearest")[0, 0].to(torch.int32)
    info = dict(lowpass_a=round(lp_a, 2), threshold=float(thr), threshold_sigma=round((thr - mu) / max(sd, 1e-12), 2),
                extend_a=extend_a, soft_a=soft_a, core_volume_a3=float(core.sum() * vox ** 3), working_grid=nb,
                segments_kept=list(segments), include=list(include), exclude=list(exclude), invert=invert)
    return out, info, labels


def mask_report(mask: torch.Tensor, apix: float, vmap: torch.Tensor | None = None, h1=None, h2=None,
                mw_kda: float | None = None) -> dict:
    """Numbers that say whether a mask is right: its volume (sum, and voxels fully inside), the volume a protein of
    mw_kda occupies (1.21 A^3/Da), the share of the 15 A low-passed map's strong density beyond its soft edge (a mask
    that cuts the particle), and with half maps: masked / corrected FSC and the randomised-phase FSC at the corrected
    resolution (the mask's own correlation; high = too sharp an edge)."""
    out = dict(volume_a3=float(mask.sum()) * apix ** 3, full_volume_a3=float((mask > 0.99).sum()) * apix ** 3,
               box_share=float(mask.mean()))
    if mw_kda:
        out["protein_volume_a3"] = mw_kda * 1000 * 1.21
        out["volume_over_protein"] = out["volume_a3"] / out["protein_volume_a3"]
    if vmap is not None:
        lp = _lowpass(vmap.float(), apix, 15.0)
        dens = lp > 0.15 * lp.max()
        out["density_outside"] = float((dens & (mask < 0.05)).sum() / dens.sum().clamp_min(1))
    if h1 is not None and h2 is not None:
        from .abinitio import corrected_masked_fsc, resolution_from_fsc
        n = h1.shape[0]
        f_c, f_m, f_r, f_un, k_r = corrected_masked_fsc(h1, h2, mask, parts=True)
        r_c = resolution_from_fsc(f_c, n * apix)
        i = min(int(round(n * apix / r_c)), len(f_r) - 1)
        out.update(resolution_corrected_a=r_c, resolution_masked_a=resolution_from_fsc(f_m, n * apix),
                   resolution_unmasked_a=resolution_from_fsc(f_un, n * apix), randomised_fsc_at_res=float(f_r[i]),
                   fsc_corrected=f_c)
    return out
