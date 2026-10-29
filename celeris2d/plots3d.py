"""Figures of an ab initio run (PNG, dark theme matching the dashboard).

Colours: dashboard surface/ink tokens; categorical series use the first three validated slots (blue, orange, aqua,
dark-surface steps), sequential density a single blue ramp from the surface to light blue.
"""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

SURFACE, INK, INK2, INK3, LINE = "#16181f", "#f0efec", "#a0a3ad", "#767a86", "#2a2e3a"
SERIES = ["#3987e5", "#d95926", "#199e70"]          # categorical slots 1-3 (dark steps), validated all-pairs
DENSITY = LinearSegmentedColormap.from_list("density", ["#1d2029", "#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"])
GRAY = LinearSegmentedColormap.from_list("em", ["#000000", "#ffffff"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": LINE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 9, "lines.solid_capstyle": "round",
    })


def _title(fig, title, sub):
    """Title and subtitle at fixed distances (inches) from the top, whatever the figure height."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.1 / h, title, fontsize=12, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.012, 1 - 0.36 / h, sub, fontsize=9, color=INK2, va="top")


def _crossing(f: np.ndarray, thr: float = 0.143):
    """First shell (fractional, linear interpolation) where f drops below thr."""
    for i in range(1, len(f)):
        if f[i] < thr:
            return i - 1 + (f[i - 1] - thr) / max(f[i - 1] - f[i], 1e-9)
    return float(len(f) - 1)


def fsc_figure(path: str, curves: list[tuple[str, np.ndarray]], box_a: float, apix: float, title: str):
    """curves: (label, FSC per shell); x axis in 1/A with resolution (A) tick labels; 0.143 crossings labelled."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.14)
    nyq = 1 / (2 * apix)
    ax.axhline(0.143, color=INK3, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(nyq, 0.143 + 0.025, "0.143", color=INK3, fontsize=8, ha="right")
    ax.axhline(0, color=LINE, lw=1, zorder=1)
    for i, (lab, f) in enumerate(curves[:3]):
        f = np.asarray(f, float)
        x = np.arange(len(f)) / box_a
        c = SERIES[i]
        k = _crossing(f)
        xr = k / box_a
        ax.plot(x[1:], f[1:], color=c, lw=2, zorder=3,
                label=f"{lab}   {box_a / k:.1f} Å" if 0 < xr < nyq else f"{lab}   ≤ Nyquist")
        if 0 < xr < nyq:
            ax.plot([xr], [0.143], "o", ms=7, color=c, mec=SURFACE, mew=2, zorder=4)
    ticks = [r for r in (40, 20, 15, 10, 8, 6, 5, 4, 3.5, 3, 2.5) if 1 / r <= nyq * 1.001]
    ax.set_xticks([1 / r for r in ticks], [f"{r:g}" for r in ticks])
    ax.set_xlim(0, nyq)
    ax.set_ylim(-0.08, 1.04)
    ax.set_yticks([0, 0.143, 0.5, 1.0], ["0", "", "0.5", "1"])
    ax.set_xlabel("resolution (Å)")
    ax.set_ylabel("Fourier shell correlation")
    ax.yaxis.grid(True, color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", title="FSC 0.143 at", title_fontsize=8.5, alignment="left")
    leg.get_title().set_color(INK3)
    _title(fig, "Half-map FSC", title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def views_figure(path: str, rot_deg: np.ndarray, tilt_deg: np.ndarray, title: str):
    """Viewing directions on an equal-area (Mollweide) sky map: longitude = rot, latitude = 90 - tilt;
    density per equal-area bin on a log scale."""
    _style()
    lon = np.deg2rad(((rot_deg + 180) % 360) - 180)
    lat = np.deg2rad(90 - tilt_deg)
    nlon, nlat = 72, 36
    le = np.linspace(-np.pi, np.pi, nlon + 1)
    se = np.linspace(-1, 1, nlat + 1)                     # bins equal in sin(lat) = equal area
    H, _, _ = np.histogram2d(lon, np.sin(lat), bins=[le, se])
    occ = (H > 0).mean()
    fig = plt.figure(figsize=(7.2, 4.1))
    ax = fig.add_axes([0.03, 0.08, 0.8, 0.74], projection="mollweide")
    Hm = np.ma.masked_equal(H.T, 0)
    m = ax.pcolormesh(le, np.arcsin(se), Hm, cmap=DENSITY, norm=LogNorm(vmin=1, vmax=max(2, H.max())),
                      shading="flat", rasterized=True)
    ax.set_facecolor("#1d2029")
    ax.grid(True, color=LINE, lw=0.6)
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=7.5, colors=INK3)
    cax = fig.add_axes([0.86, 0.2, 0.018, 0.5])
    cb = fig.colorbar(m, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK3, labelsize=8)
    cb.set_label("particles per bin", color=INK2, fontsize=8.5)
    nz = H[H > 0]
    _title(fig, "Viewing directions", f"{title} · {occ * 100:.0f}% of the sphere occupied · densest bin "
                                     f"{nz.max() / np.median(nz):.0f}× the median occupied bin")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _norm_img(im):
    lo, hi = np.percentile(im, [1, 99.5])
    return np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1)


def reprojections_figure(path: str, classes: np.ndarray, reproj: np.ndarray, pops: np.ndarray, labels: list,
                         title: str):
    """Each class average next to the map's projection at the orientation assigned to it, sorted by size."""
    _style()
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
    fig = plt.figure(figsize=(cols * cw + 0.2, rows * ch + 0.75))
    top = 1 - 0.65 / (rows * ch + 0.75)
    for j, k in enumerate(order):
        rr, c = divmod(j, cols)
        x0 = (0.1 + c * cw) / (cols * cw + 0.2)
        y0 = top - (rr + 1) * ch / (rows * ch + 0.75)
        w, h = (cw * 0.47) / (cols * cw + 0.2), (ch * 0.8) / (rows * ch + 0.75)
        for q, im in enumerate((classes[k], reproj[k])):
            ax = fig.add_axes([x0 + q * (w + 0.004), y0 + 0.17 * ch / (rows * ch + 0.75), w, h])
            ax.imshow(_norm_img(im), cmap=GRAY, interpolation="lanczos")
            ax.set_axis_off()
        ccol = SERIES[2] if cc[k] >= 0.7 else (INK2 if cc[k] >= 0.4 else SERIES[1])
        fig.text(x0, y0 + 0.02 * ch / (rows * ch + 0.75), f"{labels[k]} · {int(pops[k]):,}", fontsize=7.5, color=INK2)
        fig.text(x0 + 2 * w, y0 + 0.02 * ch / (rows * ch + 0.75), f"cc {cc[k]:.2f}", fontsize=7.5, color=ccol,
                 ha="right", fontweight="bold")
    _title(fig, "Class averages vs map reprojections",
           f"{title} · left: 2D class average · right: projection of the 3D map at the class's orientation · "
           f"median cc {np.median(cc):.2f}")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cc


def map_views_figure(path: str, maps: list, labels: list, title: str):
    """Per map: projections along z, y, x and central sections."""
    _style()
    n = len(maps)
    fig, axs = plt.subplots(n, 6, figsize=(10.5, 1.95 * n + 0.85), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.85 / (1.95 * n + 0.85), bottom=0.02, wspace=0.04, hspace=0.18)
    for i, (v, lab) in enumerate(zip(maps, labels)):
        b = v.shape[0]
        ims = [v.sum(0), v.sum(1), v.sum(2), v[b // 2], v[:, b // 2], v[:, :, b // 2]]
        names = ["projection z", "projection y", "projection x", "section z", "section y", "section x"]
        for j, (im, nm) in enumerate(zip(ims, names)):
            ax = axs[i, j]
            ax.imshow(_norm_img(im), cmap=GRAY, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(nm if i == 0 else "", fontsize=8, color=INK3, fontweight="normal", loc="center")
            if j == 0 and n > 1:
                ax.set_ylabel(lab, color=INK2, fontsize=9)
    _title(fig, "Map", title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_all(outdir: str, *, class_imgs, reproj, pops, class_labels, maps, map_labels, fsc_curves, box_a, apix,
              rot, tilt, title):
    fsc_figure(os.path.join(outdir, "fsc.png"), fsc_curves, box_a, apix, title)
    views_figure(os.path.join(outdir, "views.png"), rot, tilt, title)
    cc = reprojections_figure(os.path.join(outdir, "reprojections.png"), class_imgs, reproj, pops, class_labels, title)
    map_views_figure(os.path.join(outdir, "map_views.png"), maps, map_labels, title)
    return cc
