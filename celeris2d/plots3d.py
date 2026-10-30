"""Figures of the 3D jobs (ab initio, Refine3D, HeteroRefine, Sharpen, local resolution), in the common style of
plotstyle.py (dark and light variants of every figure). Numbers behind each figure are written next to it by
diagnostics.py (FSC as JSON / STAR / EMDB XML, angular distribution as JSON).
"""
from __future__ import annotations

import math
import os

import numpy as np
from matplotlib.colors import LogNorm

from .plotstyle import GRAY, STATUS, T, grid_y, image_axes, norm_img, note, plt, res_axis, themed, title, top, \
    warnings_text


def crossing(f: np.ndarray, thr: float = 0.143):
    """First shell (fractional, linear interpolation) where f drops below thr."""
    for i in range(1, len(f)):
        if f[i] < thr:
            return i - 1 + (f[i - 1] - thr) / max(f[i - 1] - f[i], 1e-9)
    return float(len(f) - 1)


def _fsc_axes(ax, curves, box_a, nyq, styles=None, thr=0.143):
    """FSC curves on one axes: x in 1/A labelled in A, 0.143 line, crossing markers, legend with the crossings."""
    ax.axhline(thr, color=T.ink3, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.5, color=T.grid, lw=0.8, zorder=1)
    ax.axhline(0, color=T.axis, lw=1, zorder=1)
    for i, (lab, f) in enumerate(curves):
        f = np.asarray(f, float)
        x = np.arange(len(f)) / box_a
        c, ls = (styles[i] if styles else (T.series[i % 3], "-"))
        k = crossing(f, thr)
        xr = k / box_a
        inside = 0 < xr < nyq
        first = next((i for i in range(1, len(f)) if f[i] < thr), None)   # as reported (abinitio.resolution_from_fsc)
        lab_r = "" if ls != "-" else (f"   {box_a / first:.2f} Å" if inside and first else "   ≤ Nyquist")
        ax.plot(x[1:], f[1:], color=c, lw=2 if ls == "-" else 1.6, ls=ls, zorder=3, label=lab + lab_r)
        if inside and ls == "-":
            ax.plot([xr], [thr], "o", ms=7, color=c, mec=T.surface, mew=2, zorder=4)
    res_axis(ax, nyquist=nyq)
    ax.set_ylim(-0.08, 1.04)
    ax.set_yticks([0, 0.143, 0.5, 1.0], ["0", "0.143", "0.5", "1"])
    ax.set_ylabel("Fourier shell correlation")
    leg = ax.legend(loc="lower left", title=f"FSC {thr:g} at", title_fontsize=8.5, alignment="left",
                    facecolor=T.surface, framealpha=0.9, frameon=True, edgecolor="none")
    leg.get_title().set_color(T.ink3)


@themed
def fsc_figure(path: str, curves: list[tuple[str, np.ndarray]], box_a: float, apix: float, title_: str,
               heading: str = "Half-map FSC", styles=None, warnings=()):
    """curves: (label, FSC per shell of a box_a box) - at most 3 solid series (+ dotted helpers via styles)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    fig.subplots_adjust(left=0.1, right=0.97, top=top(fig, 0.75), bottom=0.14 + 0.05 * len(warnings))
    _fsc_axes(ax, curves, box_a, 1 / (2 * apix), styles)
    title(fig, heading, title_)
    warnings_text(fig, list(warnings))
    fig.savefig(path)
    plt.close(fig)


def _sky(rot_deg, tilt_deg, nlon=72, nlat=36):
    lon = np.deg2rad(((np.asarray(rot_deg) + 180) % 360) - 180)
    lat = np.deg2rad(90 - np.asarray(tilt_deg))
    le = np.linspace(-np.pi, np.pi, nlon + 1)
    se = np.linspace(-1, 1, nlat + 1)                     # bins equal in sin(lat) = equal area
    H, _, _ = np.histogram2d(lon, np.sin(lat), bins=[le, se])
    return H, le, se


def _sky_axes(fig, rect, H, le, se, vmax=None):
    ax = fig.add_axes(rect, projection="mollweide")
    Hm = np.ma.masked_equal(H.T, 0)
    m = ax.pcolormesh(le, np.arcsin(se), Hm, cmap=T.density, norm=LogNorm(vmin=1, vmax=max(2, vmax or H.max())),
                      shading="flat", rasterized=True)
    ax.set_facecolor(T.well)
    ax.grid(True, color=T.line, lw=0.6)
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=7.5, colors=T.ink3)
    return ax, m


@themed
def views_figure(path: str, rot_deg: np.ndarray, tilt_deg: np.ndarray, title_: str, stats: dict | None = None):
    """Angular distribution: viewing directions on an equal-area (Mollweide) map, longitude = rot, latitude =
    90 - tilt, particles per equal-area bin on a log scale. stats (diagnostics.angular_stats) go in the subtitle."""
    H, le, se = _sky(rot_deg, tilt_deg)
    fig = plt.figure(figsize=(7.4, 4.3))
    ax, m = _sky_axes(fig, [0.03, 0.07, 0.8, 0.72], H, le, se)
    cax = fig.add_axes([0.86, 0.2, 0.018, 0.5])
    cb = fig.colorbar(m, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=T.ink3, labelsize=8)
    cb.set_label("particles per bin (log)", color=T.ink2, fontsize=8.5)
    s = stats or {}
    sub = [title_, f"{len(np.asarray(rot_deg)):,} particles"]
    if "scf" in s:
        sub.append(f"SCF {s['scf']:.2f} (1 = uniform Fourier sampling)")
    if "occupied" in s:
        sub.append(f"{s['occupied'] * 100:.0f}% of directions occupied")
    if "top10_share" in s:
        sub.append(f"busiest 10% hold {s['top10_share'] * 100:.0f}%")
    title(fig, "Angular distribution", " · ".join(sub))
    warnings_text(fig, s.get("warnings", []))
    fig.savefig(path)
    plt.close(fig)


@themed
def views_grid_figure(path: str, groups: list[tuple[str, np.ndarray, np.ndarray, dict]], title_: str):
    """Angular distribution per class: (label, rot, tilt, stats) per panel, shared colour scale."""
    n = len(groups)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    pw, ph = 4.0, 2.45
    fig = plt.figure(figsize=(cols * pw + 0.5, rows * ph + 0.85))
    hs = [_sky(r, t) for _, r, t, _ in groups]
    vmax = max([h[0].max() for h in hs] + [2])
    H0 = fig.get_figheight()
    for i, ((lab, r, t, st), (H, le, se)) in enumerate(zip(groups, hs)):
        rr, c = divmod(i, cols)
        x0 = (0.35 + c * pw) / (cols * pw + 0.5)
        y0 = 1 - (0.8 + (rr + 1) * ph) / H0
        _, m = _sky_axes(fig, [x0, y0 + 0.28 / H0, (pw - 0.35) / (cols * pw + 0.5), (ph - 0.5) / H0], H, le, se, vmax)
        txt = f"{lab} · {len(np.asarray(r)):,}" + (f" · SCF {st['scf']:.2f}" if st and "scf" in st else "")
        fig.text(x0, y0 + 0.08 / H0, txt, fontsize=8.5, color=T.ink2)
    title(fig, "Angular distribution per class", f"{title_} · particles per equal-area bin, shared log scale")
    fig.savefig(path)
    plt.close(fig)


@themed
def reprojections_figure(path: str, classes: np.ndarray, reproj: np.ndarray, pops: np.ndarray, labels: list,
                         title_: str):
    """Each class average next to the map's projection at the orientation assigned to it, sorted by size."""
    b = classes.shape[-1]
    r = np.arange(b) - b // 2
    circ = (r[:, None] ** 2 + r[None, :] ** 2) < (0.45 * b) ** 2
    cc = []
    for c, p in zip(classes, reproj):
        x, y = c[circ] - c[circ].mean(), p[circ] - p[circ].mean()
        cc.append(float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-12)))
    order = np.argsort(-pops)
    n = len(order)
    cols = min(6, n)
    rows = int(math.ceil(n / cols))
    cw, ch = 2.15, 1.32
    W, H = cols * cw + 0.2, rows * ch + 0.75
    fig = plt.figure(figsize=(W, H))
    y_top = 1 - 0.65 / H
    for j, k in enumerate(order):
        rr, c = divmod(j, cols)
        x0 = (0.1 + c * cw) / W
        y0 = y_top - (rr + 1) * ch / H
        w, h = (cw * 0.47) / W, (ch * 0.8) / H
        for q, im in enumerate((classes[k], reproj[k])):
            ax = fig.add_axes([x0 + q * (w + 0.004), y0 + 0.17 * ch / H, w, h])
            ax.imshow(norm_img(im), cmap=GRAY)
            ax.set_axis_off()
        ccol = T.ink if cc[k] >= 0.7 else (T.ink2 if cc[k] >= 0.4 else STATUS["serious"])
        fig.text(x0, y0 + 0.02 * ch / H, f"{labels[k]} · {int(pops[k]):,}", fontsize=7.5, color=T.ink2)
        fig.text(x0 + 2 * w, y0 + 0.02 * ch / H, f"cc {cc[k]:.2f}" + (" ⚠" if cc[k] < 0.4 else ""), fontsize=7.5,
                 color=ccol, ha="right", fontweight="bold")
    title(fig, "Class averages vs map reprojections",
          f"{title_} · left: 2D class average · right: projection of the 3D map at the class's orientation · "
          f"median cc {np.median(cc):.2f}")
    fig.savefig(path)
    plt.close(fig)
    return cc


@themed
def map_views_figure(path: str, maps: list, labels: list, title_: str, masks: list | None = None):
    """Per map: projections along z, y, x and central sections (mask outline on the sections when given)."""
    n = len(maps)
    H = 1.95 * n + 0.85
    fig, axs = plt.subplots(n, 6, figsize=(10.5, H), squeeze=False)
    fig.subplots_adjust(left=0.03 if n > 1 else 0.01, right=0.99, top=1 - 0.85 / H, bottom=0.02, wspace=0.04,
                        hspace=0.18)
    for i, (v, lab) in enumerate(zip(maps, labels)):
        b = v.shape[0]
        ims = [v.sum(0), v.sum(1), v.sum(2), v[b // 2], v[:, b // 2], v[:, :, b // 2]]
        ms = None
        if masks is not None and masks[i] is not None:
            mk = masks[i]
            ms = [None, None, None, mk[b // 2], mk[:, b // 2], mk[:, :, b // 2]]
        names = ["projection z", "projection y", "projection x", "section z", "section y", "section x"]
        for j, (im, nm) in enumerate(zip(ims, names)):
            ax = axs[i, j]
            ax.imshow(norm_img(im), cmap=GRAY)
            if ms is not None and ms[j] is not None:
                ax.contour(ms[j], levels=[0.5], colors=T.series[2], linewidths=0.8)
            image_axes(ax)
            ax.set_title(nm if i == 0 else "", fontsize=8, color=T.ink3, fontweight="normal", loc="center")
            if j == 0 and n > 1:
                ax.set_ylabel(lab, color=T.ink2, fontsize=9)
    title(fig, "Map", title_)
    fig.savefig(path)
    plt.close(fig)


@themed
def refine_history_figure(path: str, history: list[dict], title_: str, classes: bool = False):
    """Per iteration of a refinement: resolution (FSC 0.143) and the alignment limit, angular step, share of
    particles whose pose changed; with classes, the class sizes per iteration (stacked)."""
    it = np.array([h["it"] for h in history])
    ncol = 4 if classes else 3
    fig, axs = plt.subplots(1, ncol, figsize=(4.0 * ncol, 3.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.8), bottom=0.16, wspace=0.32)
    a = axs[0]
    if classes:
        R = np.array([[np.nan if r is None else r for r in h.get("res", [np.nan] * len(h["pops"]))]
                      for h in history], float)
        for m in range(min(3, R.shape[1])):
            a.plot(it, R[:, m], "o-", ms=3.5, color=T.series[m], label=f"class {m + 1}")
        best = np.nanmin(R[-1]) if np.isfinite(R[-1]).any() else np.nan
        final = f"best class {best:.2f} Å" + (" · first 3 classes shown" if R.shape[1] > 3 else "")
    else:
        R = np.array([(h["res"][0] if isinstance(h.get("res"), list) else h.get("res", np.nan)) or np.nan
                      for h in history], float)
        a.plot(it, R, "o-", ms=3.5, color=T.series[0], label="FSC 0.143")
        final = f"final {R[-1]:.2f} Å"
    a.plot(it, [h["align_res"] for h in history], "o--", ms=3, lw=1.4, color=T.ink3, label="alignment limit")
    a.invert_yaxis()
    a.set(title=f"Resolution (finer ↑) · {final}", xlabel="iteration", ylabel="Å")
    a.legend(loc="lower right")
    grid_y(a)
    a = axs[1]
    a.plot(it, [h["step"] for h in history], "o-", ms=3.5, color=T.series[0])
    a.set(title="Angular step of the local search", xlabel="iteration", ylabel="degrees")
    a.set_ylim(bottom=0)
    grid_y(a)
    a = axs[2]
    a.plot(it, [h["moved"] * 100 for h in history], "o-", ms=3.5, color=T.series[0], label="pose changed")
    if classes:
        a.plot(it, [h.get("switched", 0) * 100 for h in history], "o-", ms=3.5, color=T.series[1],
               label="class changed")
        a.legend(loc="upper right")
    a.set(title="Convergence", xlabel="iteration", ylabel="% of particles")
    a.set_ylim(bottom=0)
    grid_y(a)
    if classes:
        a = axs[3]
        P = np.array([h["pops"] for h in history], float)
        P = P / P.sum(1, keepdims=True) * 100
        bottom = np.zeros(len(it))
        K = P.shape[1]
        for m in range(K):
            a.bar(it, P[:, m], bottom=bottom, width=0.8, color=T.series[m % 8], edgecolor=T.surface, linewidth=1)
            if P[-1, m] >= 6:                    # direct label on the last bar
                a.text(it[-1], bottom[-1] + P[-1, m] / 2, f"{m + 1}", ha="center", va="center", fontsize=8,
                       color="#0b0b0b" if m % 8 == 3 else "#ffffff", fontweight="bold")
            bottom += P[:, m]
        a.set(title="Class sizes (class number on the last bar)", xlabel="iteration", ylabel="% of particles",
              ylim=(0, 100))
    from matplotlib.ticker import MaxNLocator
    for a in axs:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    tot = sum(h.get("seconds", 0) for h in history[-1:])
    title(fig, "Refinement progress", title_ + (f" · {tot:.0f} s" if tot else ""))
    fig.savefig(path)
    plt.close(fig)


@themed
def class3d_progress_figure(path: str, history: list[dict], post: np.ndarray, title_: str):
    """3D classification without alignment: class sizes per iteration (share of the particles, stacked; class
    number at the right of each band), convergence (particles that changed class, mean of the largest posterior)
    and the final posterior distribution (each particle's largest posterior; near 1 = a clear assignment)."""
    from matplotlib.ticker import MaxNLocator
    it = np.array([h["it"] for h in history])
    P = np.array([h["pops"] for h in history], float)
    P = P / np.maximum(P.sum(1, keepdims=True), 1) * 100
    K = P.shape[1]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.6))
    fig.subplots_adjust(left=0.06, right=0.97, top=top(fig, 0.8), bottom=0.16, wspace=0.3)
    a = axs[0]
    bottom = np.zeros(len(it))
    for m in range(K):
        a.bar(it, P[:, m], bottom=bottom, width=0.8, color=T.series[m % 8], edgecolor=T.surface, linewidth=1)
        if P[-1, m] >= 5:
            a.text(it[-1] + 0.55, bottom[-1] + P[-1, m] / 2, f"{m + 1}", ha="left", va="center", fontsize=8,
                   color=T.ink2)
        bottom += P[:, m]
    a.set(title="Class sizes (class number at the right)", xlabel="iteration", ylabel="% of particles", ylim=(0, 100))
    a.set_xlim(it[0] - 0.6, it[-1] + 1.4)
    a = axs[1]
    a.plot(it, [h["changed"] * 100 for h in history], "o-", ms=3.5, color=T.series[0], label="changed class (%)")
    a.set(title="Convergence", xlabel="iteration", ylabel="% of particles")
    a.set_ylim(bottom=0)
    grid_y(a)
    a2 = a.twinx()
    a2.plot(it, [h["max_posterior"] for h in history], "o--", ms=3, lw=1.4, color=T.series[1],
            label="mean max posterior")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("mean max posterior", color=T.ink2)
    a2.spines["right"].set_visible(True)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper right")
    a = axs[2]
    mp = np.asarray(post).max(1)
    a.hist(mp, bins=np.linspace(1.0 / max(K, 1), 1, 41), color=T.series[0])
    a.set(title="Largest posterior per particle", xlabel="posterior of the best class", ylabel="particles")
    note(a, f"median {np.median(mp):.2f}\n> 0.9: {(mp > 0.9).mean() * 100:.0f}%\n< 0.6: {(mp < 0.6).mean() * 100:.0f}%",
         "upper left")
    grid_y(a)
    for a in axs[:2]:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    tot = history[-1].get("seconds", 0) if history else 0
    title(fig, "3D classification", title_ + (f" · {len(history)} iterations · {tot:.0f} s" if tot else ""))
    fig.savefig(path)
    plt.close(fig)


@themed
def particle_stats_figure(path: str, fit: np.ndarray, used: np.ndarray, shift_a: np.ndarray, moved_deg: np.ndarray,
                          title_: str, cls: np.ndarray | None = None):
    """Per-particle diagnostics: fit to the map (normalised correlation) of particles used and left out, distance of
    the aligned centre from the box centre, angle between the starting and the final orientation."""
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.8), bottom=0.16, wspace=0.28)
    a = axs[0]
    bins = np.linspace(np.percentile(fit, 0.5), np.percentile(fit, 99.5), 60)
    a.hist(fit[used], bins=bins, color=T.series[0], label=f"used ({int(used.sum()):,})")
    if (~used).any():
        a.hist(fit[~used], bins=bins, color=T.series[1], label=f"left out ({int((~used).sum()):,})")
    a.set(title="Fit to the map", xlabel="normalised correlation with the projection", ylabel="particles")
    a.legend(loc="upper right")
    a = axs[1]
    d = np.hypot(shift_a[:, 0], shift_a[:, 1])
    a.hist(d, bins=60, color=T.series[0])
    note(a, f"median {np.median(d):.1f} Å\n95% below {np.percentile(d, 95):.1f} Å", "upper right")
    a.set(title="Aligned centre from box centre", xlabel="Å", ylabel="particles")
    a = axs[2]
    if moved_deg is not None and len(moved_deg):
        a.hist(np.clip(moved_deg, 0, 30), bins=np.linspace(0, 30, 61), color=T.series[0])
        note(a, f"median {np.median(moved_deg):.1f}°\n> 10°: {(moved_deg > 10).mean() * 100:.1f}%", "upper right")
    a.set(title="Orientation change from the start", xlabel="degrees (clipped at 30)", ylabel="particles")
    for a in axs:
        grid_y(a)
    title(fig, "Particles", title_)
    fig.savefig(path)
    plt.close(fig)


@themed
def class_summary_figure(path: str, maps: list, labels: list, pops: list, res: list, title_: str):
    """Heterogeneous refinement: one row of projections / central sections per class, with size and resolution."""
    n = len(maps)
    H = 1.75 * n + 0.85
    fig, axs = plt.subplots(n, 4, figsize=(8.4, H), squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.85 / H, bottom=0.02, wspace=0.04, hspace=0.12)
    tot = max(1, sum(pops))
    for i, (v, lab) in enumerate(zip(maps, labels)):
        b = v.shape[0]
        for j, im in enumerate([v.sum(0), v.sum(1), v[b // 2], v[:, b // 2]]):
            ax = axs[i, j]
            ax.imshow(norm_img(im), cmap=GRAY)
            image_axes(ax)
            if i == 0:
                ax.set_title(["projection z", "projection y", "section z", "section y"][j], fontsize=8,
                             color=T.ink3, fontweight="normal", loc="center")
        r = res[i]
        fig.text(0.012, 1 - (0.85 + 1.75 * i + 0.75) / H, f"{lab}\n{pops[i]:,} ({pops[i] / tot * 100:.0f}%)\n"
                 + (f"{r:.2f} Å" if r == r and r is not None else "merged"), fontsize=9, color=T.ink, va="center")
    title(fig, "3D classes", title_)
    fig.savefig(path)
    plt.close(fig)


def write_all(outdir: str, *, class_imgs, reproj, pops, class_labels, maps, map_labels, fsc_curves, box_a, apix,
              rot, tilt, title, view_stats=None):
    fsc_figure(os.path.join(outdir, "fsc.png"), fsc_curves, box_a, apix, title)
    views_figure(os.path.join(outdir, "views.png"), rot, tilt, title, view_stats)
    cc = reprojections_figure(os.path.join(outdir, "reprojections.png"), class_imgs, reproj, pops, class_labels, title)
    map_views_figure(os.path.join(outdir, "map_views.png"), maps, map_labels, title)
    return cc



@themed
def localres_figure(path: str, vol: np.ndarray, lres: np.ndarray, mask: np.ndarray, res_global: float, stats: dict,
                    title_: str):
    """Central sections of the map coloured by local resolution inside the mask (blue finer than the global
    resolution, red coarser, gray at it) and the distribution of local resolution in the mask."""
    from matplotlib.colors import TwoSlopeNorm
    b = vol.shape[0]
    c = b // 2
    lo, hi = stats["p05_a"], stats["p95_a"]
    lo, hi = min(lo, res_global * 0.98), max(hi, res_global * 1.02)
    norm = TwoSlopeNorm(vcenter=res_global, vmin=lo, vmax=hi)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, left=0.02, right=0.98, top=top(fig, 0.8), bottom=0.22, wspace=0.22,
                          width_ratios=[1, 1, 1, 1.25])
    g = norm_img(vol, 1, 99.9)
    for i, (sl, lr, mk, nm) in enumerate([(g[c], lres[c], mask[c], "z"), (g[:, c], lres[:, c], mask[:, c], "y"),
                                          (g[:, :, c], lres[:, :, c], mask[:, :, c], "x")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(sl, cmap=GRAY, origin="lower")
        over = np.ma.masked_where(mk < 0.5, lr)
        im = ax.imshow(over, cmap=T.diverging_cmap, norm=norm, alpha=0.55, origin="lower")
        image_axes(ax)
        ax.set_title(f"central {nm} section", fontsize=9, fontweight="normal", loc="center", color=T.ink3)
    ax = fig.add_subplot(gs[0, 3])
    e = np.asarray(stats["hist_edges_a"])
    h = np.asarray(stats["hist_counts"], float)
    ctr = 0.5 * (e[1:] + e[:-1])
    ax.bar(ctr, h / max(h.sum(), 1) * 100, width=np.diff(e) * 0.92, color=T.diverging_cmap(norm(ctr)))
    ax.axvline(res_global, color=T.ink, lw=1, ls="--")
    ax.text(res_global, ax.get_ylim()[1] * 0.97, f" global {res_global:.2f} Å", color=T.ink, fontsize=8.5, va="top")
    ax.set(title="Local resolution in the mask", xlabel="Å", ylabel="% of mask voxels")
    grid_y(ax)
    cax = fig.add_axes([0.08, 0.13, 0.5, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=T.ink3, labelsize=8)
    cb.set_label("local resolution (Å) · blue finer than the global value, red coarser", color=T.ink2, fontsize=8.5)
    title(fig, "Local resolution", f"{title_} · median {stats['median_a']:.2f} Å · 5–95% {stats['p05_a']:.2f}–"
                                   f"{stats['p95_a']:.2f} Å · window {stats['window_a']:.0f} Å · FSC {stats['threshold']}")
    fig.savefig(path)
    plt.close(fig)


@themed
def nu_figure(path: str, history: list[dict], cutoff_map: np.ndarray | None, mask: np.ndarray | None,
              vmap: np.ndarray | None, candidates_a: list, title_: str):
    """Non-uniform refinement: per iteration the cutoffs chosen inside the mask (median, 5-95 % band) beside the
    global FSC resolution; the share of the mask filtered at the global resolution or finer and at twice it or
    coarser; how well the two halves' cutoff maps agree (r); the last iteration's share per candidate cutoff; and
    sections of the last cutoff map through the mask's centre (inside the mask, over the map)."""
    hs = [h for h in history if "nu" in h]
    fig = plt.figure(figsize=(16, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.3)
    fig.subplots_adjust(left=0.05, right=0.97, top=top(fig, 0.8), bottom=0.07)
    it = np.array([h["it"] for h in hs])
    nu = [h["nu"] for h in hs]
    a = fig.add_subplot(gs[0, 0:2])
    if hs:
        a.fill_between(it, [x["p05_a"] for x in nu], [x["p95_a"] for x in nu], color=T.series[0], alpha=0.25,
                       lw=0, label="cutoffs 5–95 %")
        a.plot(it, [x["median_a"] for x in nu], "o-", ms=3.5, color=T.series[0], label="median cutoff")
        a.plot(it, [x["res_a"] for x in nu], "o--", ms=3, lw=1.4, color=T.ink3, label="global FSC resolution")
        a.invert_yaxis()
        a.legend(loc="lower right")
    a.set(title="Local cutoffs in the mask (finer ↑)", xlabel="iteration", ylabel="Å")
    grid_y(a)
    a = fig.add_subplot(gs[0, 2])
    if hs:
        a.plot(it, [x["at_global_pct"] for x in nu], "o-", ms=3.5, color=T.series[0], label="≤ 1.05 × global")
        a.plot(it, [x["coarse_pct"] for x in nu], "o-", ms=3.5, color=T.series[1], label="≥ 2 × global")
        a.legend(loc="best")
    a.set(title="Share of the mask", xlabel="iteration", ylabel="% of mask voxels", ylim=(0, 100))
    grid_y(a)
    a = fig.add_subplot(gs[0, 3])
    if hs:
        a.plot(it, [x["halves_r"] for x in nu], "o-", ms=3.5, color=T.series[0])
    a.set(title="Halves agree (r of cutoff maps)", xlabel="iteration", ylabel="Pearson r", ylim=(-0.05, 1.05))
    grid_y(a)
    from matplotlib.ticker import MaxNLocator
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    a = fig.add_subplot(gs[1, 0])
    if hs:
        sh = nu[-1]["share_pct"]
        x = np.arange(len(candidates_a))
        a.bar(x, sh, width=0.8, color=T.series[0])
        a.set_xticks(x[::2], [f"{c:.1f}" for c in candidates_a[::2]], rotation=45, fontsize=7.5)
        note(a, f"median {nu[-1]['median_a']:.2f} Å\nglobal {nu[-1]['res_a']:.2f} Å", "upper right")
    a.set(title="Last iteration: cutoffs (half 1)", xlabel="cutoff (Å)", ylabel="% of mask voxels")
    grid_y(a)
    if cutoff_map is not None and mask is not None:
        m = mask > 0.5
        b = cutoff_map.shape[0]
        idx = np.argwhere(m)
        cz, cy, cx = (np.round(idx.mean(0)).astype(int) if len(idx) else (b // 2,) * 3)
        lo, hi = np.percentile(cutoff_map[m], [2, 98]) if m.any() else (0, 1)
        secs = [("section z", cutoff_map[cz], m[cz], None if vmap is None else vmap[cz]),
                ("section y", cutoff_map[:, cy], m[:, cy], None if vmap is None else vmap[:, cy]),
                ("section x", cutoff_map[:, :, cx], m[:, :, cx], None if vmap is None else vmap[:, :, cx])]
        im = None
        for j, (nm, c, mk, v) in enumerate(secs):
            a = fig.add_subplot(gs[1, 1 + j])
            if v is not None:
                a.imshow(norm_img(v), cmap=GRAY, alpha=0.6)
            im = a.imshow(np.where(mk, c, np.nan), cmap=T.density.reversed(), vmin=lo, vmax=hi, alpha=0.85)
            image_axes(a)
            a.set_title(nm, fontsize=9, color=T.ink3, fontweight="normal")
        cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.025, pad=0.01)
        cb.set_label("local cutoff (Å)", color=T.ink2)
    title(fig, "Non-uniform refinement", title_)
    fig.savefig(path)
    plt.close(fig)
