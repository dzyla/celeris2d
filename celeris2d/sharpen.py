"""Sharpen job (RELION PostProcess): the half maps of a Refine3D job -> a sharpened map to look at side chains, with
the diagnostics needed to trust it.

    FSC          unmasked, masked, masked with randomised phases beyond the shell where the unmasked FSC drops
                 below 0.8, and the corrected masked FSC (FSC_m - FSC_rand) / (1 - FSC_rand): the resolution
                 (0.143) is read from the corrected curve, so a mask that creates correlation by itself does not
                 count.
    B-factor     Guinier fit, ln(spherically averaged amplitude) vs 1/d^2, between fit_from_a (10 A) and the
                 resolution (RELION's automatic B-factor); or given. RELION's sign: negative sharpens.
    filter       map x exp(-B s^2 / 4) x sqrt(2 FSC / (1 + FSC)) (Rosenthal & Henderson 2003, corrected FSC),
                 low-passed at the resolution with a cosine edge of edge_shells shells.
    outputs      sharpened.mrc (whole box), sharpened_masked.mrc, postprocess.png (FSC curves, Guinier plot,
                 central slices with the mask outline), summary: resolutions (corrected / masked / unmasked, 0.143
                 and 0.5), B-factor and fit quality, mask coverage and warnings (mask too tight, mask cutting
                 density, poor Guinier fit, single-shell FSC dips).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from .plotstyle import GRAY, STATUS, T, grid_y, image_axes, plt, themed, title, top, warnings_text
from .jobspec import params_of, settings_of
from .project import CLASS_JOBS, Job, Project, logged


@dataclass
class SharpenSettings:
    VERSION: ClassVar[int] = 2      # 2: mask "" no longer means the refinement's mask ("auto" is the default)
    b_factor: float = 0.0           # applied B-factor (A^2, RELION sign: negative sharpens); 0: automatic (Guinier fit)
    fit_from_a: float = 10.0        # Guinier fit from this resolution to the map's resolution
    resolution_a: float = 0.0       # low-pass (A); 0: the corrected masked FSC at 0.143
    mask: str = "auto"              # "auto": made from the half maps (masks.auto_mask); "refine": the refinement's;
                                    # else a MaskCreate job (number) or a mask .mrc (path, absolute or relative to the project)
    edge_shells: int = 2            # cosine edge of the low-pass (Fourier shells)
    fsc_weight: bool = True         # weight each shell by sqrt(2 FSC / (1 + FSC)) (RELION)
    cls: int = 0                    # a HeteroRefine / Class3D job: the class (1-based) to sharpen
    local_res: bool = True          # local resolution (localres.py): localres.mrc + localres_filtered.mrc + figure
    local_window_a: float = 0.0     # its window (0: ~7x the resolution)

    @classmethod
    def migrate(cls, params: dict, from_version: int) -> dict:
        if from_version < 2 and not params.get("mask"):   # version 1 used the refinement's mask by default
            params = dict(params, mask="refine")
        return params


def sharpen_job(project: Project, refine, s: SharpenSettings, label: str = "") -> Job:
    """Sharpen the half maps of a Refine3D job (or of class s.cls of a HeteroRefine / Class3D job)."""
    r = project.job(refine)
    if r.type not in ("Refine3D", *CLASS_JOBS):
        raise ValueError(f"{r.name} is not a Refine3D, HeteroRefine or Class3D job")
    if r.type in CLASS_JOBS and not s.cls:
        raise ValueError(f"{r.name}: choose the class to sharpen (cls)")
    return project.new_job("Sharpen", dict(params_of(s), refine=r.id), [r],
                           label or (f"on{r.id:03d}" + (f"c{s.cls}" if s.cls else "")))


def _read(path):
    import mrcfile
    with mrcfile.open(path, permissive=True) as f:
        return np.array(f.data, np.float32), float(f.voxel_size.x)


def _write(path, v, apix):
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(v, np.float32))
        m.voxel_size = apix


def _shells(n: int, dev):
    k = torch.fft.fftfreq(n, device=dev) * n
    kz = torch.arange(n // 2 + 1, device=dev, dtype=torch.float32)
    return torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)


def _res(c: np.ndarray, box_a: float, thr: float) -> float:
    for i in range(1, len(c)):
        if np.isnan(c[i]):              # past the half maps' content: as far as they go
            return box_a / max(i - 1, 1)
        if c[i] < thr:
            return box_a / i
    return box_a / (len(c) - 1)


def _dips(c: np.ndarray, box_a: float, upto: int) -> list[float]:
    """Resolutions (A) of single shells whose FSC is far below both neighbours (an artifact, not signal)."""
    out = []
    for i in range(2, min(upto, len(c) - 1)):
        if min(c[i - 1], c[i + 1]) - c[i] > 0.15:
            out.append(round(box_a / i, 2))
    return out


def content_end(h: torch.Tensor, kk: torch.Tensor) -> int:
    """Last Fourier shell with content: a refinement reconstructs to a limit a little beyond its resolution, the
    shells after it are (near) zero - a 100-fold fall of the power within 4 shells, which no real map shows."""
    n = h.shape[0]
    nsh = n // 2 + 1
    sh = torch.round(kk).long().clamp(max=nsh - 1).reshape(-1)
    pw = torch.zeros(nsh, device=h.device).index_add_(0, sh, (torch.fft.rfftn(h).abs() ** 2).reshape(-1).float())
    p = (pw / torch.bincount(sh, minlength=nsh).float().clamp_min(1)).cpu().numpy()
    for i in range(8, nsh - 4):
        if p[i + 4] < 0.01 * p[i]:      # the last shell above 1% of the power before the fall
            return i + max(j for j in range(4) if p[i + j] >= 0.01 * p[i])
    return nsh - 1


def postprocess(h1: np.ndarray, h2: np.ndarray, mask: np.ndarray, apix: float, s: SharpenSettings, dev=None) -> dict:
    """The sharpening of two half maps (n^3, same box) with a soft mask; returns maps, curves and numbers."""
    from .abinitio import corrected_masked_fsc
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    t1, t2, m = (torch.as_tensor(x, device=dev) for x in (h1, h2, mask))
    n = t1.shape[0]
    box_a = n * apix
    kk = _shells(n, dev)
    sh = torch.round(kk).long()
    nsh = n // 2 + 1
    k_end = content_end(t1, kk)          # the refinement reconstructs a little beyond its resolution: zeros after
    f_c, f_m, f_r, f_un, k_r = corrected_masked_fsc(t1, t2, m, parts=True)   # the one the refinements use
    f_un, f_m, f_r, f_c = (x[:nsh].cpu().numpy().copy() for x in (f_un, f_m, f_r, f_c))
    for c in (f_un, f_m, f_r, f_c):     # shells the half maps hold nothing in: no FSC (mask leakage only)
        c[k_end + 1:] = np.nan
    res = s.resolution_a or _res(f_c, box_a, 0.143)
    k_res = box_a / res                                               # shell of the low-pass

    full = 0.5 * (t1 + t2)
    F = torch.fft.rfftn(full * m)
    shc = sh.clamp(max=nsh - 1).reshape(-1)
    pw = torch.zeros(nsh, device=dev).index_add_(0, shc, (F.abs() ** 2).reshape(-1).float())
    cnt = torch.zeros(nsh, device=dev).index_add_(0, shc, torch.ones_like(shc, dtype=torch.float32))
    amp = torch.sqrt(pw / cnt.clamp_min(1)).cpu().numpy()
    s2 = (np.arange(nsh) / box_a) ** 2
    lo, hi_ = int(np.ceil(box_a / s.fit_from_a)), int(np.floor(k_res))
    fit = dict(from_a=s.fit_from_a, to_a=round(res, 2), shells=max(0, hi_ - lo + 1), r2=float("nan"))
    if hi_ - lo + 1 >= 3:
        x, y = s2[lo:hi_ + 1], np.log(np.maximum(amp[lo:hi_ + 1], 1e-30))
        slope, icpt = np.polyfit(x, y, 1)
        pred = slope * x + icpt
        fit.update(r2=float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)), slope=float(slope),
                   intercept=float(icpt))
        b_auto = 4 * slope                                            # ln A = c - B s^2 / 4: applied B = -B = 4 slope
    else:
        b_auto = 0.0
    b = s.b_factor or b_auto
    s2v = (kk / box_a) ** 2
    filt = torch.exp(-b * s2v / 4)
    if s.fsc_weight:
        fc = torch.as_tensor(np.clip(f_c, 0, 1), device=dev)
        filt = filt * torch.sqrt(2 * fc / (1 + fc)).nan_to_num(0)[sh.clamp(max=nsh - 1)]
    w = max(1, s.edge_shells)
    edge = torch.clamp((kk - k_res) / w + 0.5, 0, 1)
    filt = filt * 0.5 * (1 + torch.cos(np.pi * edge))                 # 1 below k_res - w/2, 0 beyond k_res + w/2
    sharp = torch.fft.irfftn(torch.fft.rfftn(full) * filt, s=full.shape)
    Fs = torch.fft.rfftn(sharp * m)
    pws = torch.zeros(nsh, device=dev).index_add_(0, shc, (Fs.abs() ** 2).reshape(-1).float())
    amp_s = torch.sqrt(pws / cnt.clamp_min(1)).cpu().numpy()

    # mask checks: the share of the (15 A low-passed) map's density outside the mask, and the mask's own correlation
    lp = torch.fft.irfftn(torch.fft.rfftn(full) * (kk / box_a < 1 / 15.0), s=full.shape)
    dens = lp > 0.15 * lp.max()
    outside = float((dens & (m < 0.05)).sum() / dens.sum().clamp_min(1))     # beyond the soft edge
    i_res = min(int(round(k_res)), nsh - 1)
    warnings = []
    if f_r[i_res] > 0.2:
        warnings.append(f"the mask alone correlates the randomised half maps (FSC {f_r[i_res]:.2f} at {res:.2f} A): "
                        f"too tight or too sharp an edge; the corrected FSC accounts for it")
    if outside > 0.02:
        warnings.append(f"{outside * 100:.1f} % of the low-passed density lies outside the mask: it cuts the particle")
    if fit["shells"] < 5 or not fit["r2"] > 0.8:
        warnings.append(f"Guinier fit on {fit['shells']} shells, r^2 {fit['r2']:.2f}: the B-factor is uncertain (set it)")
    dips = _dips(f_un, box_a, i_res)
    if dips:
        warnings.append(f"single-shell drops of the unmasked FSC at {dips} A: a reconstruction artifact, not signal")
    return dict(sharp=sharp.cpu().numpy(), mask=m.cpu().numpy(), apix=apix, box_a=box_a,
                fsc=dict(unmasked=f_un, masked=f_m, randomised=f_r, corrected=f_c), randomised_from_a=box_a / max(k_r, 1),
                amp=amp, amp_sharp=amp_s, s2=s2, fit=fit, b_factor=float(b), b_auto=float(b_auto), resolution_a=float(res),
                resolution_masked_a=_res(f_m, box_a, 0.143), resolution_unmasked_a=_res(f_un, box_a, 0.143),
                resolution_05_a=_res(f_c, box_a, 0.5), mask_fraction=float(m.mean()), density_outside_mask=outside,
                fsc_dips_a=dips, warnings=warnings, content_to_a=round(box_a / max(k_end, 1), 2))


@logged
def run_sharpen(project: Project, job, log=print) -> dict:
    job = project.job(job)
    s = settings_of(job, SharpenSettings)
    rj = project.job(job.params["refine"])
    t0 = time.time()
    pre = f"class{s.cls:02d}_" if rj.type in CLASS_JOBS else ""
    h1, apix = _read(rj.find(f"{pre}half1_unfil.mrc"))
    h2, _ = _read(rj.find(f"{pre}half2_unfil.mrc"))
    mask_info = {}
    if s.mask in ("auto", ""):
        from .masks import auto_mask
        mask, mask_info = auto_mask(h1, h2, apix)
        mpath = job.out("mask.mrc")
        _write(mpath, mask, apix)
        mapix = apix
        log(f"[sharpen] automatic mask: threshold {mask_info['threshold_sigma']} sd of the noise "
            f"({mask_info['threshold_frac_max']:.3f} of the maximum), {mask_info['pieces_kept']} of "
            f"{mask_info['pieces']} pieces kept, +{mask_info['dilate_a']} A, soft edge {mask_info['soft_a']} A, "
            f"{mask_info['volume_fraction'] * 100:.1f}% of the box")
    else:
        if s.mask == "refine":
            mpath = rj.find(f"{pre}mask.mrc")
            mask, mapix = _read(mpath)
        else:                                        # a MaskCreate job or an .mrc path, resampled to the half maps
            from .jobs import load_mask
            mpath = s.mask
            mask, mapix = load_mask(project, s.mask, h1.shape[0], apix), apix
    if mask.shape != h1.shape or abs(mapix - apix) > 1e-3 * apix:
        raise ValueError(f"mask {mpath}: {mask.shape} at {mapix:.4f} A/px, half maps {h1.shape} at {apix:.4f} A/px")
    r = postprocess(h1, h2, np.clip(mask, 0, 1), apix, s)
    _write(job.out("sharpened.mrc"), r["sharp"], apix)
    _write(job.out("sharpened_masked.mrc"), r["sharp"] * r["mask"], apix)
    src_name = rj.name + (f" class {s.cls}" if pre else "")
    figure(job.out("postprocess.png"), r, f"{job.name} (from {src_name})")
    from .diagnostics import write_fsc
    write_fsc(job.out("fsc.json")[:-5], r["fsc"], r["box_a"], resolution_a=round(r["resolution_a"], 3),
              main="corrected", title=f"{job.name} (from {src_name})")
    with open(job.out("guinier.json"), "w") as f:
        json.dump(dict(s2_inv_a2=r["s2"].round(6).tolist(), ln_amplitude=np.log(np.maximum(r["amp"], 1e-30)).round(4).tolist(),
                       ln_amplitude_sharpened=np.log(np.maximum(r["amp_sharp"], 1e-30)).round(4).tolist(),
                       fit={k: v for k, v in r["fit"].items()}, b_factor=r["b_factor"], b_auto=r["b_auto"]), f)
    lres = None
    if s.local_res:
        lres = local_resolution(job, h1, h2, r, apix, s, f"{job.name} (from {src_name})", log)
        r["warnings"] += lres.pop("warnings", [])
    for w in r["warnings"]:
        log(f"[sharpen] warning: {w}")
    summ = dict(resolution_a=round(r["resolution_a"], 2), resolution_masked_uncorrected_a=round(r["resolution_masked_a"], 2),
                resolution_unmasked_a=round(r["resolution_unmasked_a"], 2), resolution_fsc05_a=round(r["resolution_05_a"], 2),
                b_factor=round(r["b_factor"], 1), b_factor_auto=round(r["b_auto"], 1),
                guinier_fit=dict(from_a=r["fit"]["from_a"], to_a=r["fit"]["to_a"], shells=r["fit"]["shells"],
                                 r2=round(r["fit"]["r2"], 3) if r["fit"]["r2"] == r["fit"]["r2"] else None),
                randomised_phases_from_a=round(r["randomised_from_a"], 2), mask=project.rel(mpath), mask_auto=mask_info or None, content_to_a=r["content_to_a"],
                mask_fraction=round(r["mask_fraction"], 4), density_outside_mask=round(r["density_outside_mask"], 4),
                fsc_dips_a=r["fsc_dips_a"], warnings=r["warnings"], apix=round(apix, 4), box=int(h1.shape[0]),
                fsc_corrected=[round(float(x), 4) for x in r["fsc"]["corrected"]], map=project.rel(job.out("sharpened.mrc")),
                **({"local_resolution": lres} if lres else {}), seconds=round(time.time() - t0, 1))
    project.set_status(job, "finished", summ)
    log(f"[sharpen] {summ['resolution_a']} A (corrected masked FSC 0.143), B {summ['b_factor']} A^2")
    return summ


def local_resolution(job: Job, h1, h2, r: dict, apix: float, s: SharpenSettings, title_: str, log=print) -> dict:
    """Local resolution of the half maps inside the mask (localres.py): localres.mrc (A per voxel), localres_filtered.mrc
    (the B-sharpened map low-passed at every voxel to its local resolution), localres.png, localres.json."""
    from . import plots3d
    from .localres import localres
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = r["mask"]
    window = int(round(s.local_window_a / apix)) if s.local_window_a else 0
    out = localres(h1, h2, m, apix, r["resolution_a"], window_px=window, filtered=False, dev=dev)
    lr = out["map"]
    full = 0.5 * (torch.as_tensor(h1, device=dev) + torch.as_tensor(h2, device=dev))
    kk = _shells(full.shape[0], dev) / (full.shape[0] * apix)
    sharpB = torch.fft.irfftn(torch.fft.rfftn(full) * torch.exp(-r["b_factor"] * kk ** 2 / 4), s=full.shape)
    from .localres import local_filter
    filt = local_filter(sharpB, lr, apix)
    lr_np = lr.cpu().numpy()
    _write(job.out("localres.mrc"), lr_np, apix)
    _write(job.out("localres_filtered.mrc"), filt.cpu().numpy(), apix)
    st = out["stats"]
    with open(job.out("localres.json"), "w") as f:
        json.dump(st, f)
    plots3d.localres_figure(job.out("localres.png"), r["sharp"], lr_np, m, r["resolution_a"], st, title_)
    warnings = []
    if st["at_nyquist"] > 0.05:
        nyq = 2 * apix
        if r["resolution_a"] > 1.3 * nyq:
            warnings.append(f"local resolution reaches Nyquist ({nyq:.2f} A) in {st['at_nyquist'] * 100:.0f}% of the mask "
                            f"although the map is at {r['resolution_a']:.2f} A: the half maps may share noise "
                            "(duplicate particles, or halves that are not independent)")
        else:
            warnings.append(f"the map ({r['resolution_a']:.2f} A) is close to Nyquist ({nyq:.2f} A): in "
                            f"{st['at_nyquist'] * 100:.0f}% of the mask the local resolution is only known to be "
                            f"<= {nyq:.2f} A (refine at a smaller pixel size to measure it)")
    log(f"[sharpen] local resolution: median {st['median_a']} A (5-95%: {st['p05_a']}-{st['p95_a']} A), window "
        f"{st['window_a']} A, {time.time() - t0:.1f}s")
    keep = {k: st[k] for k in ("median_a", "p05_a", "p25_a", "p75_a", "p95_a", "window_a", "at_nyquist")}
    keep.update(map=os.path.relpath(job.out("localres.mrc"), job.root),
                filtered=os.path.relpath(job.out("localres_filtered.mrc"), job.root), seconds=round(time.time() - t0, 1),
                warnings=warnings)
    return keep


@themed
def figure(path: str, r: dict, title_: str):
    """FSC curves (corrected, masked, unmasked, phase-randomised), Guinier plot with the fit, central slices of the
    sharpened map with the mask outline."""
    from .plots3d import _fsc_axes
    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], left=0.06, right=0.98, top=top(fig, 0.8),
                          bottom=0.04 + 0.025 * len(r["warnings"]), hspace=0.32, wspace=0.28)
    ax = fig.add_subplot(gs[0, :2])
    curves = [("corrected masked", r["fsc"]["corrected"]), ("masked, uncorrected", r["fsc"]["masked"]),
              ("unmasked", r["fsc"]["unmasked"]),
              (f"phases randomised beyond {r['randomised_from_a']:.1f} Å", r["fsc"]["randomised"])]
    styles = [(T.series[0], "-"), (T.series[1], "-"), (T.series[2], "-"), (T.ink3, ":")]
    _fsc_axes(ax, curves, r["box_a"], (len(r["fsc"]["corrected"]) - 1) / r["box_a"], styles)
    for d in r["fsc_dips_a"]:
        ax.axvline(1 / d, color=STATUS["warning"], lw=0.8)
    ax.set_title("Fourier shell correlation")
    ax = fig.add_subplot(gs[0, 2])
    s2, f = r["s2"], r["fit"]
    sel = (s2 > 0) & (s2 <= (1.15 / r["resolution_a"]) ** 2)
    a0 = np.log(np.maximum(r["amp"][sel], 1e-30))
    a1 = np.log(np.maximum(r["amp_sharp"][sel], 1e-30))
    ax.plot(s2[sel], a0 - a0[0], color=T.series[0], lw=1.6, label="map (masked)")
    ax.plot(s2[sel], a1 - a1[0], color=T.series[1], lw=1.6, label="sharpened")
    if "slope" in f:
        x = np.array([1 / f["from_a"] ** 2, 1 / f["to_a"] ** 2])
        ax.axvspan(*x, color=T.well, zorder=0)
        ax.plot(x, f["slope"] * x + f["intercept"] - a0[0], "--", color=T.ink, lw=1.2,
                label=f"fit {f['from_a']:.0f}-{f['to_a']:.1f} Å: B {r['b_auto']:.0f} Å², r² {f['r2']:.2f}")
    ax.set_xlabel("1/d² (1/Å²)")
    ax.set_ylabel("ln amplitude (relative)")
    ax.set_title(f"Guinier plot · applied B {r['b_factor']:.0f} Å²")
    ax.legend(loc="lower left")
    grid_y(ax)
    v, m = r["sharp"], r["mask"]
    c = v.shape[0] // 2
    lo_, hi = np.percentile(v, [1, 99.9])
    for i, (sl, ms, name) in enumerate([(v[c], m[c], "z"), (v[:, c], m[:, c], "y"), (v[:, :, c], m[:, :, c], "x")]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(sl, cmap=GRAY, vmin=lo_, vmax=hi, origin="lower")
        ax.contour(ms, levels=[0.5], colors=T.series[2], linewidths=0.8)
        ax.set_title(f"sharpened · central {name} section · mask outline", fontsize=9, fontweight="normal")
        image_axes(ax)
    title(fig, f"Sharpened map · {r['resolution_a']:.2f} Å · B {r['b_factor']:.0f} Å²",
          f"{title_} · corrected masked FSC 0.143 {r['resolution_a']:.2f} Å · masked {r['resolution_masked_a']:.2f} Å · "
          f"unmasked {r['resolution_unmasked_a']:.2f} Å · FSC 0.5 {r['resolution_05_a']:.2f} Å · mask "
          f"{r['mask_fraction'] * 100:.1f}% of the box")
    warnings_text(fig, r["warnings"])
    fig.savefig(path)
    plt.close(fig)
