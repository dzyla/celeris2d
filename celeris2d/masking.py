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
    """Fourier crop of a real cube to n^3 <= its size (same box in A, values kept): real FFT, the low frequencies
    taken by index (a full complex FFT and shifted copies were 2 GB at 512^3)."""
    b = v.shape[0]
    if b == n:
        return v
    if n > b:
        raise ValueError("_resize only crops")
    ft = torch.fft.rfftn(v)
    h = n // 2
    keep = torch.cat([torch.arange(h, device=v.device), torch.arange(b - h, b, device=v.device)])
    ft = ft[keep][:, keep][:, :, :h + 1]
    return torch.fft.irfftn(ft, s=(n, n, n)) * (n / b) ** 3


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


def _components(m: np.ndarray, keep_frac: float, info: dict | None = None) -> np.ndarray:
    """The connected pieces of a binary map at least keep_frac of the largest one."""
    lab, n = ndimage.label(m)
    if info is not None:
        info.update(pieces=int(n), pieces_kept=int(n))
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= keep_frac * sizes.max()) + 1
    if info is not None:
        info["pieces_kept"] = int(len(keep))
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


def noise_sigma(h1: torch.Tensor, h2: torch.Tensor | None, nb: int, vox: float, lp_a: float) -> float:
    """Noise standard deviation of the low-passed average of two half maps: (h1 - h2) / 2 is pure noise with the
    statistics of the average, everywhere (also inside the particle, and in a map already masked to a sphere, whose
    solvent is zero), restricted to where the halves are not both zero. h2 None: h1 is that half difference."""
    d = _lowpass(_resize(h1.float() if h2 is None else 0.5 * (h1.float() - h2.float()), nb), vox, lp_a)
    a = d if h2 is None else _resize(0.5 * (h1.float() + h2.float()), nb)
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
    if halves is not None:          # (h1, h2), or (half difference, None)
        mu, sd = 0.0, noise_sigma(halves[0], halves[1], nb, vox, lp_a)
    peak = float(lp.max())
    thr = max(mu + k_sigma * sd, mu + min_frac * (peak - mu))
    core = _components((lp > thr).cpu().numpy(), keep_frac, info)
    soft = torch.as_tensor(_soften(core, vox, extend_a, soft_a), device=vmap.device)
    out = F.interpolate(soft[None, None], size=(n, n, n), mode="trilinear", align_corners=False)[0, 0] if nb != n else soft
    if info is not None:
        info.update(lowpass_a=round(lp_a, 2), threshold=thr, threshold_sigma=round((thr - mu) / max(sd, 1e-12), 2),
                    threshold_of_peak=round((thr - mu) / max(peak - mu, 1e-12), 4),
                    threshold_frac_max=round(thr / max(peak, 1e-30), 4), extend_a=extend_a, soft_a=soft_a,
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


# ------------------------------------------------------------------------------------------------ MaskCreate job
from dataclasses import dataclass  # noqa: E402

from . import gpu
from .project import has_classes  # noqa: E402


@dataclass
class MaskCreateSettings:
    cls: int = 0                    # a HeteroRefine / Class3D / ModeMaps: the class (1-based) whose map is masked
    lowpass_a: float = 0.0          # low-pass before thresholding (A); 0: 1.5 x the map's resolution, within 8-15 A
    threshold: float = 0.0          # threshold on the low-passed map (map units); 0: automatic, sigma_level x the noise
    sigma_level: float = 5.0        # automatic threshold: noise standard deviations (from the half-map difference)
    extend_a: float = 3.0           # the thresholded density grown by this (A) ...
    soft_a: float = 6.0             # ... then given a cosine edge this wide (A)
    keep_frac: float = 0.05         # connected pieces smaller than this share of the largest are dropped
    n_segments: int = 12            # the density split into this many segments around its strongest peaks (figure)
    segments: tuple = ()            # keep only these segments (numbers in the figure); empty: all
    include: tuple = ()             # regions always in: "sphere x y z r" or "box x0 y0 z0 x1 y1 z1" (A, from the box corner)
    exclude: tuple = ()             # regions taken out, same form (e.g. a detergent belt, a flexible domain)
    invert: bool = False            # the mask of everything else (e.g. for signal subtraction)


def parse_region(text: str) -> dict:
    """'sphere x y z r' or 'box x0 y0 z0 x1 y1 z1' (A; commas allowed) -> the dict make_mask takes."""
    parts = text.replace(",", " ").split()
    kind, vals = parts[0].lower(), [float(v) for v in parts[1:]]
    if kind == "sphere" and len(vals) == 4:
        return {"sphere": vals}
    if kind == "box" and len(vals) == 6:
        return {"box": vals}
    raise ValueError(f"region {text!r}: use 'sphere x y z r' or 'box x0 y0 z0 x1 y1 z1' (A)")


def mask_job(project, source, s: MaskCreateSettings, label: str = ""):
    from .jobspec import params_of
    src = project.job(source)
    if has_classes(src) and not s.cls:
        raise ValueError(f"{src.name}: choose the class to mask (cls)")
    for r in tuple(s.include) + tuple(s.exclude):
        parse_region(r)
    tag = "auto" if not (s.segments or s.include or s.exclude or s.threshold) else "custom"
    return project.new_job("MaskCreate", dict(params_of(s), map=src.id), [src],
                           label or f"{tag}_on{src.id:03d}" + (f"c{s.cls}" if s.cls else ""))


def _source_maps(project, src, cls: int):
    """(average map, half 1, half 2 or None, apix, resolution) of a job's map."""
    import mrcfile
    rj = project.job(src.params["refine"]) if src.type == "Sharpen" else src
    pre = f"class{cls:02d}_" if has_classes(rj) else ""

    def rd(name):
        p = rj.find(pre + name)
        if p is None:
            return None, None
        with mrcfile.open(p, permissive=True) as f:
            return torch.as_tensor(np.asarray(f.data, np.float32)), float(f.voxel_size.x)
    h1, apix = rd("half1_unfil.mrc")
    h2, _ = rd("half2_unfil.mrc")
    if h1 is not None and h2 is not None:
        avg = 0.5 * (h1 + h2)
    else:
        avg, apix = rd("map_nomask.mrc")
        if avg is None:
            avg, apix = rd("map.mrc")
        h1 = h2 = None
    summ = project.summary_of(rj)
    res = summ.get("resolution_a") or 8.0
    return avg, h1, h2, apix, float(res)


def run_mask(project, job, log=print) -> dict:
    import time as _t
    from .abinitio import _save_mrc
    from .jobspec import settings_of
    job = project.job(job)
    s = settings_of(job, MaskCreateSettings)
    src = project.job(job.params["map"])
    t0 = _t.time()
    dev = gpu.default()
    avg, h1, h2, apix, res = _source_maps(project, src, s.cls)
    avg = avg.to(dev)
    halves = None if h1 is None else (h1.to(dev), h2.to(dev))
    n = avg.shape[0]
    lp_a = s.lowpass_a or min(15.0, max(8.0, 1.5 * res))
    nb, vox = _grid(n, apix, lp_a)
    if s.threshold:
        thr = float(s.threshold)
    else:                                                # automatic: as auto_mask, noise from the half maps
        lp = _lowpass(_resize(avg.float(), nb), vox, lp_a)
        mu, sd = (0.0, noise_sigma(halves[0], halves[1], nb, vox, lp_a)) if halves else solvent_stats(lp, vox, None)
        thr = max(mu + s.sigma_level * sd, mu + 0.03 * (float(lp.max()) - mu))
    m, info, labels = make_mask(avg, apix, lowpass_a=lp_a, threshold=thr, extend_a=s.extend_a, soft_a=s.soft_a,
                                keep_frac=s.keep_frac, include=[parse_region(r) for r in s.include],
                                exclude=[parse_region(r) for r in s.exclude], segments=tuple(s.segments),
                                n_segments=s.n_segments, invert=s.invert)
    if not s.threshold:                                  # the automatic threshold, in the noise sigma that set it
        info["threshold_sigma"] = s.sigma_level
    _save_mrc(job.out("mask.mrc"), m, apix)
    seg_info = []
    if labels is not None:
        _save_mrc(job.out("segments.mrc"), labels.float(), apix)
        lab = labels.cpu().numpy()
        c = (np.arange(n) + 0.5) * apix
        for k in range(1, int(lab.max()) + 1):
            z, y, x = np.nonzero(lab == k)
            if len(z):
                seg_info.append(dict(segment=k, volume_a3=round(float(len(z) * apix ** 3)), centre_a=[
                    round(float(c[x].mean()), 1), round(float(c[y].mean()), 1), round(float(c[z].mean()), 1)]))
    rep = mask_report(m, apix, avg, *(halves or (None, None)))
    summ = dict(source=src.name, apix=round(apix, 4), box=n, volume_a3=round(rep["volume_a3"]),
                box_share=round(rep["box_share"], 4), density_outside=round(rep["density_outside"], 4),
                threshold=info["threshold"], threshold_sigma=info["threshold_sigma"], lowpass_a=info["lowpass_a"],
                extend_a=s.extend_a, soft_a=s.soft_a, segments=seg_info, segments_kept=list(s.segments),
                include=list(s.include), exclude=list(s.exclude), invert=s.invert, seconds=round(_t.time() - t0, 1))
    if halves:
        summ.update(resolution_corrected_a=round(rep["resolution_corrected_a"], 2),
                    resolution_unmasked_a=round(rep["resolution_unmasked_a"], 2),
                    randomised_fsc_at_res=round(rep["randomised_fsc_at_res"], 3))
    warnings = []
    if rep["density_outside"] > 0.02 and not (s.segments or s.exclude or s.invert):
        warnings.append(f"{rep['density_outside'] * 100:.1f} % of the low-passed density is outside the mask")
    if halves and rep["randomised_fsc_at_res"] > 0.2:
        warnings.append("the mask alone correlates phase-randomised half maps: widen soft_a")
    summ["warnings"] = warnings
    _figure(job.out("mask.png"), avg.cpu().numpy(), m.cpu().numpy(), None if labels is None else labels.cpu().numpy(),
            apix, f"{job.name} · {src.name} · {rep['volume_a3'] / 1e6:.2f} M Å³ · threshold {info['threshold_sigma']} σ")
    project.set_status(job, "finished", summ)
    log(f"[mask] {job.name}: {rep['volume_a3'] / 1e6:.2f} M A^3 ({rep['box_share'] * 100:.1f}% of the box), "
        f"threshold {info['threshold_sigma']} sd, {len(seg_info)} segments" +
        (f", corrected FSC {summ['resolution_corrected_a']} A" if halves else ""))
    return summ


def _figure(path, vmap, mask, labels, apix, title_):
    from .plotstyle import themed
    themed(_draw)(path, vmap, mask, labels, apix, title_)


def _draw(path, vmap, mask, labels, apix, title_):
    """Top: central sections of the map with the mask's 0.5 contour (x, y, z). Bottom: projections of the
    segments (each a colour, its number at its centre) - the numbers `segments` keeps."""
    from .plotstyle import GRAY, T, norm_img, plt, title
    n = vmap.shape[0]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.05, hspace=0.12)
    ext = [0, n * apix, 0, n * apix]
    secs = [(vmap[n // 2], mask[n // 2], "z section (x right, y up)"),
            (vmap[:, n // 2], mask[:, n // 2], "y section (x right, z up)"),
            (vmap[:, :, n // 2], mask[:, :, n // 2], "x section (y right, z up)")]
    for a, (im, mk, nm) in zip(ax[0], secs):
        a.imshow(norm_img(im, 1, 99.5), cmap=GRAY, origin="lower", extent=ext)
        a.contour(mk, levels=[0.5], colors=[T.series[1]], linewidths=1.2, origin="lower", extent=ext)
        a.set_title(nm, fontsize=9, fontweight="normal", loc="center")
        a.set_xticks([])
        a.set_yticks([])
    if labels is not None and labels.max() > 0:
        from matplotlib.colors import ListedColormap
        k = int(labels.max())
        cols = plt.get_cmap("tab20")(np.arange(k) % 20)
        cmap = ListedColormap(np.vstack([[0, 0, 0, 0], cols]))
        for i, (a, axis, nm) in enumerate(zip(ax[1], (0, 1, 2), ("along z", "along y", "along x"))):
            # the segment nearest the viewer along the axis: the last non-zero label
            lab = np.moveaxis(labels, axis, 0)
            nz = lab > 0
            last = np.where(nz.any(0), n - 1 - np.argmax(nz[::-1], axis=0), 0)
            top = np.take_along_axis(lab, last[None], 0)[0]
            a.imshow(norm_img(vmap.sum(axis), 1, 99.5), cmap=GRAY, origin="lower", extent=ext)
            a.imshow(top, cmap=cmap, vmin=0, vmax=k, origin="lower", extent=ext, alpha=0.55, interpolation="nearest")
            for s in range(1, k + 1):
                yy, xx = np.nonzero(top == s)
                if len(yy):
                    a.text((xx.mean() + 0.5) * apix, (yy.mean() + 0.5) * apix, str(s), color=T.ink, fontsize=10,
                           fontweight="bold", ha="center", va="center")
            a.set_title(f"segments, seen {nm}", fontsize=9, fontweight="normal", loc="center")
            a.set_xticks([])
            a.set_yticks([])
    else:
        for a in ax[1]:
            a.axis("off")
    title(fig, "Mask", title_)
    fig.savefig(path)
    plt.close(fig)
