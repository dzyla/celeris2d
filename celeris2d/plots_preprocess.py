"""Control figures of movie preprocessing (PNG, dark theme of plots3d / the dashboard).

movie_figure: one micrograph - frame drift, local motion model vs patch measurements, the micrograph, the power
spectrum (data | fitted model), the equal-phase CTF profile, the ring-by-ring fit significance, the even/odd-frame
FRC and the dose weights. summary_figure: all micrographs against a reference (cryoSPARC) - defocus, astigmatism,
CTF fit resolution, motion. fsc_figure: half-map FSC of reconstructions from different micrograph sets.
"""
from __future__ import annotations

import math

import numpy as np

import torch

from .plots3d import DENSITY, GRAY, INK, INK2, INK3, LINE, SERIES, _style, plt


def _res_axis(ax, k):
    """Label a frequency axis (1/A) in Angstrom."""
    ticks, gap = [], 0.07 * (k[-1] - k[0])                         # keep labels apart on the linear k axis
    for r in (50, 30, 20, 15, 10, 8, 6, 5, 4, 3.5, 3, 2.5):
        if k[0] <= 1 / r <= k[-1] and (not ticks or 1 / r - 1 / ticks[-1] >= gap):
            ticks.append(r)
    ax.set_xticks([1 / r for r in ticks])
    ax.set_xticklabels([f"{r:g}" for r in ticks])
    ax.set_xlabel("resolution (Å)")


def _note(ax, text, top=True):
    ax.text(0.98, 0.96 if top else 0.04, text, transform=ax.transAxes, ha="right", va="top" if top else "bottom",
            color=INK2, fontsize=8.5)


def movie_figure(path: str, mic, fit, title: str = "", frc_eo=None):
    """mic: preprocess.Micrograph, fit: preprocess.CTFFit (of the plain sum), frc_eo: (k, frc) of the even/odd
    sums or None."""
    _style()
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.2))
    (a_tr, a_loc, a_img, a_ps), (a_1d, a_cc, a_eo, a_dw) = axs
    sh = mic.shifts
    n = len(sh)

    # global drift: the shift applied to each frame, frame order as colour
    a_tr.plot(sh[:, 0], sh[:, 1], color=LINE, lw=1.5, zorder=1)
    sc = a_tr.scatter(sh[:, 0], sh[:, 1], c=np.arange(1, n + 1), cmap=DENSITY, vmin=-0.4 * n, vmax=n, s=22,
                      zorder=2, edgecolors="none")                 # ramp starts above the surface: frame 1 visible
    a_tr.annotate("frame 1", sh[0], xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=8)
    a_tr.annotate(f"frame {n}", sh[-1], xytext=(6, -10), textcoords="offset points", color=INK2, fontsize=8)
    a_tr.set_aspect("equal", "datalim")
    a_tr.set_xlabel("x shift (Å)")
    a_tr.set_ylabel("y shift (Å)")
    path_a = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).sum())
    a_tr.set_title("Whole-frame motion")
    _note(a_tr, f"path {path_a:.1f} Å\nfirst→last {np.linalg.norm(sh[-1] - sh[0]):.1f} Å")
    cb = fig.colorbar(sc, ax=a_tr, fraction=0.04, pad=0.02)
    cb.set_label("frame", color=INK2)
    cb.ax.set_ylim(1, n)

    # local motion: residual trajectories at the patch centres (measured dots, model lines), magnified
    loc = getattr(mic, "local", None)
    if loc:
        cen, meas = loc["centres"], loc["measured"]
        mag = 0.15 / max(float(np.abs(meas).max()), 1e-3)                  # longest excursion ~0.15 of the field
        mod = loc.get("model")
        for p in range(len(cen)):
            cx, cy = cen[p]
            a_loc.scatter(cx + meas[p, :, 0] * mag, cy + meas[p, :, 1] * mag, s=4, color=SERIES[0], alpha=0.55,
                          edgecolors="none", label="measured (patch)" if p == 0 else None)
            if mod is not None:
                a_loc.plot(cx + mod[p, :, 0] * mag, cy + mod[p, :, 1] * mag, color=SERIES[1], lw=1.6,
                           label="model" if p == 0 else None)
            a_loc.plot([cx], [cy], marker="+", color=INK3, ms=6)
        a_loc.set_xlim(-1.15, 1.15)
        a_loc.set_ylim(1.15, -1.15)
        a_loc.set_aspect("equal")
        a_loc.set_xticks([])
        a_loc.set_yticks([])
        a_loc.legend(loc="lower left", bbox_to_anchor=(0, -0.02), fontsize=8, frameon=False, ncol=2)
        bar = 2.0 if np.abs(meas).max() > 2 else 1.0                  # scale bar in A of shift
        a_loc.plot([0.55, 0.55 + bar * mag], [1.09, 1.09], color=INK, lw=2)
        a_loc.text(0.55 + bar * mag + 0.03, 1.09, f"{bar:g} Å shift", color=INK2, fontsize=8, va="center")
        a_loc.set_title(f"Local motion (after whole-frame); model rms {loc['rms_a']:.2f} Å")
    else:
        a_loc.axis("off")
        a_loc.text(0.5, 0.5, "whole-frame motion only", ha="center", va="center", color=INK3,
                   transform=a_loc.transAxes)

    # micrograph, binned to ~10 A pixels for display
    m = mic.sum_dw
    b = max(1, int(round(10 / mic.apix)))
    t = torch.nn.functional.avg_pool2d(m[None, None].float(), b)[0, 0].cpu().numpy()
    lo, hi = np.percentile(t, [1, 99])
    a_img.imshow(t, cmap=GRAY, vmin=lo, vmax=hi)
    a_img.set_xticks([])
    a_img.set_yticks([])
    a_img.set_title("Dose-weighted sum")
    a_img.set_xlabel(f"{m.shape[1]}×{m.shape[0]} px at {mic.apix:.3f} Å/px, shown binned ×{b}")

    # power spectrum: left half data (background removed), right half fitted model; fit resolution circle
    pr = fit.profile
    d2, m2 = np.fft.fftshift(pr["data2d"], 0), np.fft.fftshift(pr["model2d"], 0)
    T = d2.shape[0]
    full = np.concatenate([d2[::-1, :0:-1], m2[:, :-1]], 1)          # Hermitian mirror for the left half
    full = np.concatenate([np.full((T, 1), np.nan), full], 1)[:, :T]
    v = np.nanpercentile(np.abs(d2), 98)
    a_ps.imshow(full, cmap=GRAY, vmin=-v, vmax=v, extent=[-0.5 / pr["apix"], 0.5 / pr["apix"]] * 2)
    th = np.linspace(0, 2 * math.pi, 200)
    r = 1 / fit.fit_res_a
    a_ps.plot(r * np.cos(th), r * np.sin(th), color=SERIES[1], lw=1, ls="--")
    a_ps.set_xticks([])
    a_ps.set_yticks([])
    a_ps.set_title("Power spectrum: data | fit")
    a_ps.set_xlabel(f"background removed; circle: fit resolution {fit.fit_res_a:.2f} Å")

    # equal-phase 1D profile
    k1, dd, mm = pr["k1d"], pr["data1d"], pr["model1d"]
    a_1d.plot(k1, dd, color=SERIES[0], lw=1.2, label="spectrum − background")
    a_1d.plot(k1, mm, color=SERIES[1], lw=1.6, label="envelope × CTF²")
    a_1d.axvline(1 / fit.fit_res_a, color=INK3, lw=1, ls="--")
    _res_axis(a_1d, k1)
    a_1d.set_ylabel("normalised amplitude")
    a_1d.set_title("CTF fit (equal-phase average)")
    a_1d.legend(loc="center right", fontsize=8, frameon=False)
    _note(a_1d, f"defocus U {fit.defocus_u:,.0f} Å\nV {fit.defocus_v:,.0f} Å, angle {fit.angle:.1f}°\n"
                f"explained variance {fit.score:.3f}")

    # ring-by-ring correlation with the significance threshold
    kc, cc, nn = pr["k"], pr["cc"], pr["n"]
    a_cc.plot(kc, cc, color=SERIES[0], lw=1.6, marker="o", ms=3, label="correlation per ring pair")
    a_cc.plot(kc, pr["z"] / np.sqrt(nn), color=INK3, lw=1, ls=":", label=f"significance ({pr['z']:g}/√pixels)")
    a_cc.axvline(1 / fit.fit_res_a, color=INK3, lw=1, ls="--")
    a_cc.axhline(0, color=LINE, lw=1)
    _res_axis(a_cc, kc)
    a_cc.set_ylabel("correlation")
    a_cc.set_title("Thon-ring fit significance")
    a_cc.legend(loc="upper right", fontsize=8, frameon=False)
    a_cc.text(0.02, 0.04, f"fit resolution {fit.fit_res_a:.2f} Å", transform=a_cc.transAxes, color=INK, fontsize=8.5)

    # even / odd frame sums
    if frc_eo is not None:
        k, f = frc_eo
        sel = k >= 1 / 60
        a_eo.plot(k[sel], f[sel], color=SERIES[0], lw=1.6)
        a_eo.axhline(0, color=LINE, lw=1)
        _res_axis(a_eo, k[sel])
        a_eo.set_ylabel("FRC")
        a_eo.set_title("Even vs odd frames (alignment check)")
        _note(a_eo, "independent noise:\nhigher = better aligned")
    else:
        a_eo.axis("off")

    # dose weights
    from .preprocess import dose_weights
    dpf = getattr(mic, "dose_per_frame", None)
    if dpf:
        for j, r in enumerate((20.0, 6.0, 3.0)):
            w = dose_weights(n, dpf, torch.tensor([1 / r]))[:, 0].numpy()
            a_dw.plot(np.arange(1, n + 1), w, color=SERIES[j], lw=1.6, label=f"{r:g} Å")
        a_dw.axhline(1, color=LINE, lw=1)
        a_dw.set_xlabel("frame")
        a_dw.set_ylabel("weight (Σw² = frames)")
        a_dw.set_title("Dose weights")
        a_dw.legend(loc="center right", fontsize=8, frameon=False, title="frequency", title_fontsize=8)
        _note(a_dw, f"{dpf:.2f} e/Å² per frame\n{dpf * n:.1f} e/Å² total")
    else:
        a_dw.axis("off")

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_figure(path: str, rows: list[dict], ref_name: str = "cryoSPARC"):
    """rows: preprocess_eval.py JSON lines. Ours vs the reference per micrograph."""
    _style()
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    a_df, a_as, a_fr, a_mo = axs
    rp = [r for r in rows if "cryosparc_particles" in r]

    def scatter(ax, x, y, xl, yl, title, fmt):
        x, y = np.asarray(x, float), np.asarray(y, float)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=LINE, lw=1)
        ax.scatter(x, y, s=30, color=SERIES[0], edgecolors=SERIES[0], linewidths=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        d = y - x
        _note(ax, f"n = {len(x)}\nmedian |Δ| {fmt(np.median(np.abs(d)))}\nmean Δ {fmt(d.mean())}, max |Δ| "
                  f"{fmt(np.abs(d).max())}")

    mo = [(r["ours"]["du"] + r["ours"]["dv"]) / 2 for r in rp]
    mr = [(r["cryosparc_particles"]["du"] + r["cryosparc_particles"]["dv"]) / 2 for r in rp]
    scatter(a_df, np.array(mr) / 1e4, np.array(mo) / 1e4, f"{ref_name}: median of particles (µm)", "ours (µm)",
            "Mean defocus", lambda v: f"{v * 1e4:.0f} Å")
    ao = [r["ours"]["du"] - r["ours"]["dv"] for r in rp]
    ar = [r["cryosparc_particles"]["du"] - r["cryosparc_particles"]["dv"] for r in rp]
    scatter(a_as, ar, ao, f"{ref_name} (Å)", "ours (Å)", "Astigmatism (U − V)", lambda v: f"{v:.0f} Å")
    rc = [r for r in rows if "cryosparc_mic" in r]
    key = "ours_dw" if "ours_dw" in rc[0] else "ours"
    scatter(a_fr, [r["cryosparc_mic"]["fit_res"] for r in rc], [r[key]["fit_res"] for r in rc],
            f"{ref_name} micrograph (Å)", "ours (Å)", "CTF fit resolution (same fitter)", lambda v: f"{v:.2f} Å")
    pl = np.array([r["path_len_a"] for r in rows])
    order = np.argsort(pl)
    a_mo.barh(np.arange(len(pl)), pl[order], color=SERIES[0], height=0.75)
    a_mo.set_yticks(np.arange(len(pl)))
    a_mo.set_yticklabels([rows[i]["movie"][-15:] for i in order], fontsize=7)
    a_mo.set_xlabel("whole-frame motion path (Å)")
    a_mo.set_title("Motion per movie")
    t = np.median([[r["read_s"], r["motion_s"], r["ctf_s"]] for r in rows], 0)
    _note(a_mo, f"median per movie:\nread {t[0]:.1f} s, motion {t[1]:.1f} s\nCTF {t[2]:.1f} s", top=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fsc_figure(path: str, curves: dict, apix: float, box: int, title: str = "Half-map FSC"):
    """curves: name -> FSC per shell (shell i at i / (box * apix) 1/A); the 0.143 crossing is noted per curve."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    lines = []
    for j, (name, f) in enumerate(curves.items()):
        f = np.asarray(f)
        k = np.arange(len(f)) / (box * apix)
        below = np.flatnonzero((f < 0.143) & (k > 0))
        res = 1 / k[below[0]] if len(below) else 1 / k[-1]
        ax.plot(k[1:], f[1:], color=SERIES[j % 3], lw=1.8, label=f"{name}: {res:.2f} Å")
        lines.append(res)
    ax.axhline(0.143, color=INK3, lw=1, ls=":")
    ax.axhline(0, color=LINE, lw=1)
    _res_axis(ax, np.arange(1, len(f)) / (box * apix))
    ax.set_ylabel("FSC")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
