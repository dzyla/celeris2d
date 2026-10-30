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

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .project import Job, Project


@dataclass
class SharpenSettings:
    b_factor: float = 0.0           # applied B-factor (A^2, RELION sign: negative sharpens); 0: automatic (Guinier fit)
    fit_from_a: float = 10.0        # Guinier fit from this resolution to the map's resolution
    resolution_a: float = 0.0       # low-pass (A); 0: the corrected masked FSC at 0.143
    mask: str = ""                  # mask .mrc (path, absolute or relative to the project); "": the refinement's
    edge_shells: int = 2            # cosine edge of the low-pass (Fourier shells)
    fsc_weight: bool = True


def sharpen_job(project: Project, refine, s: SharpenSettings, label: str = "") -> Job:
    r = project.job(refine)
    if r.type != "Refine3D":
        raise ValueError(f"{r.name} is not a Refine3D job")
    return project.new_job("Sharpen", dict(asdict(s), refine=r.id), [r], label or f"on{r.id:03d}")


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


def postprocess(h1: np.ndarray, h2: np.ndarray, mask: np.ndarray, apix: float, s: SharpenSettings, dev=None) -> dict:
    """The sharpening of two half maps (n^3, same box) with a soft mask; returns maps, curves and numbers."""
    from .abinitio import fsc_maps
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    t1, t2, m = (torch.as_tensor(x, device=dev) for x in (h1, h2, mask))
    n = t1.shape[0]
    box_a = n * apix
    kk = _shells(n, dev)
    sh = torch.round(kk).long()
    nsh = n // 2 + 1
    f_un = fsc_maps(t1, t2)
    f_m = fsc_maps(t1 * m, t2 * m)
    below = torch.nonzero(f_un < 0.8)
    k_r = int(below[0]) if len(below) else nsh - 1
    g = torch.Generator(device=dev).manual_seed(0)
    rnd = []
    for h in (t1, t2):
        ft = torch.fft.rfftn(h)
        ph = torch.rand(ft.shape, generator=g, device=dev) * 2 * np.pi
        rnd.append(torch.fft.irfftn(torch.where(kk >= k_r, ft.abs() * torch.exp(1j * ph), ft), s=h.shape) * m)
    f_r = fsc_maps(*rnd)
    f_c = f_m.clone()
    hi = torch.arange(len(f_c), device=dev) >= k_r
    f_c[hi] = (f_m[hi] - f_r[hi]) / (1 - f_r[hi]).clamp_min(1e-6)
    f_un, f_m, f_r, f_c = (x[:nsh].cpu().numpy() for x in (f_un, f_m, f_r, f_c))
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
                fsc_dips_a=dips, warnings=warnings)


def run_sharpen(project: Project, job, log=print) -> dict:
    job = project.job(job)
    s = SharpenSettings(**{k: v for k, v in job.params.items() if k in SharpenSettings.__dataclass_fields__})
    rj = project.job(job.params["refine"])
    t0 = time.time()
    h1, apix = _read(os.path.join(rj.path, "half1_unfil.mrc"))
    h2, _ = _read(os.path.join(rj.path, "half2_unfil.mrc"))
    mpath = project.abs(s.mask) if s.mask else os.path.join(rj.path, "mask.mrc")
    mask, mapix = _read(mpath)
    if mask.shape != h1.shape or abs(mapix - apix) > 1e-3 * apix:
        raise ValueError(f"mask {mpath}: {mask.shape} at {mapix:.4f} A/px, half maps {h1.shape} at {apix:.4f} A/px")
    r = postprocess(h1, h2, np.clip(mask, 0, 1), apix, s)
    _write(os.path.join(job.path, "sharpened.mrc"), r["sharp"], apix)
    _write(os.path.join(job.path, "sharpened_masked.mrc"), r["sharp"] * r["mask"], apix)
    figure(os.path.join(job.path, "postprocess.png"), r, f"{job.name} (from {rj.name})")
    for w in r["warnings"]:
        log(f"[sharpen] warning: {w}")
    summ = dict(resolution_a=round(r["resolution_a"], 2), resolution_masked_uncorrected_a=round(r["resolution_masked_a"], 2),
                resolution_unmasked_a=round(r["resolution_unmasked_a"], 2), resolution_fsc05_a=round(r["resolution_05_a"], 2),
                b_factor=round(r["b_factor"], 1), b_factor_auto=round(r["b_auto"], 1),
                guinier_fit=dict(from_a=r["fit"]["from_a"], to_a=r["fit"]["to_a"], shells=r["fit"]["shells"],
                                 r2=round(r["fit"]["r2"], 3) if r["fit"]["r2"] == r["fit"]["r2"] else None),
                randomised_phases_from_a=round(r["randomised_from_a"], 2), mask=project.rel(mpath),
                mask_fraction=round(r["mask_fraction"], 4), density_outside_mask=round(r["density_outside_mask"], 4),
                fsc_dips_a=r["fsc_dips_a"], warnings=r["warnings"], apix=round(apix, 4), box=int(h1.shape[0]),
                fsc_corrected=[round(float(x), 4) for x in r["fsc"]["corrected"]], map="sharpened.mrc",
                seconds=round(time.time() - t0, 1))
    project.set_status(job, "finished", summ)
    log(f"[sharpen] {summ['resolution_a']} A (corrected masked FSC 0.143), B {summ['b_factor']} A^2")
    return summ


def figure(path: str, r: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.2), dpi=110)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :2])
    k = np.arange(len(r["fsc"]["corrected"])) / r["box_a"]
    styles = [("unmasked", "#767a86", "-"), ("masked", "#3987e5", "-"), ("randomised", "#e34948", ":"),
              ("corrected", "#1a1a1a", "-")]
    labels = dict(unmasked="unmasked", masked="masked (uncorrected)",
                  randomised=f"masked, phases randomised beyond {r['randomised_from_a']:.1f} A", corrected="corrected masked")
    for key, c, ls in styles:
        ax.plot(k[1:], r["fsc"][key][1:], ls, color=c, lw=1.6 if key == "corrected" else 1.1, label=labels[key])
    ax.axhline(0.143, color="#999", lw=0.8, ls="--")
    ax.axhline(0.5, color="#ccc", lw=0.8, ls="--")
    ax.axvline(1 / r["resolution_a"], color="#1a1a1a", lw=0.8, ls="--")
    for d in r["fsc_dips_a"]:
        ax.axvline(1 / d, color="#eda100", lw=0.8)
    ax.set_xlim(0, k[-1])
    ax.set_ylim(-0.1, 1.02)
    ax.set_xlabel("resolution (1/A)")
    ax.set_ylabel("FSC")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-6), lambda x: 1 / np.maximum(x, 1e-6)))
    sec.set_xticks([20, 10, 6, 5, 4, 3.5, 3, 2.5])
    sec.set_xlabel("A")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.set_title(f"{r['resolution_a']:.2f} A (corrected, 0.143)   masked {r['resolution_masked_a']:.2f}   "
                 f"unmasked {r['resolution_unmasked_a']:.2f}   0.5: {r['resolution_05_a']:.2f} A", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    s2, f = r["s2"], r["fit"]
    sel = (s2 > 0) & (s2 <= (1.15 / r["resolution_a"]) ** 2)
    a0 = np.log(np.maximum(r["amp"][sel], 1e-30))
    a1 = np.log(np.maximum(r["amp_sharp"][sel], 1e-30))
    ax.plot(s2[sel], a0 - a0[0], color="#3987e5", lw=1.2, label="map (masked)")
    ax.plot(s2[sel], a1 - a1[0], color="#1a1a1a", lw=1.2, label="sharpened")
    if "slope" in f:
        x = np.array([1 / f["from_a"] ** 2, 1 / f["to_a"] ** 2])
        ax.plot(x, f["slope"] * x + f["intercept"] - a0[0], "--", color="#e34948", lw=1,
                label=f"fit: B {r['b_auto']:.0f} A$^2$, r$^2$ {f['r2']:.2f}")
        ax.axvspan(*x, color="#e34948", alpha=0.06)
    ax.set_xlabel("1/d$^2$ (1/A$^2$)")
    ax.set_ylabel("ln amplitude (relative)")
    ax.set_title(f"Guinier: applied B {r['b_factor']:.0f} A$^2$", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    v, m = r["sharp"], r["mask"]
    c = v.shape[0] // 2
    for i, (sl, ms, name) in enumerate([(v[c], m[c], "z"), (v[:, c], m[:, c], "y"), (v[:, :, c], m[:, :, c], "x")]):
        ax = fig.add_subplot(gs[1, i])
        lo_, hi = np.percentile(v, [1, 99.9])
        ax.imshow(sl, cmap="gray", vmin=lo_, vmax=hi, origin="lower")
        ax.contour(ms, levels=[0.5], colors="#1baf7a", linewidths=0.8)
        ax.set_title(f"sharpened, central {name} slice (mask outline)", fontsize=9)
        ax.axis("off")
    if r["warnings"]:
        fig.text(0.01, 0.005, "\n".join("warning: " + w for w in r["warnings"]), fontsize=8, color="#b5330f", va="bottom")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05 if r["warnings"] else 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
