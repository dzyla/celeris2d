"""Control figures of movie preprocessing (PNG, common style of plotstyle.py: dark and light variants).

movie_figure: one micrograph - frame drift, local motion model vs patch measurements, the micrograph, the power
spectrum (data | fitted model), the equal-phase CTF profile, the ring-by-ring fit significance, the even/odd-frame
FRC and the local defocus (patch CTF: regional offsets and the fitted plane). summary_figure: all micrographs against a reference (cryoSPARC) - defocus, astigmatism,
CTF fit resolution, motion. fsc_figure: half-map FSC of reconstructions from different micrograph sets.
"""
from __future__ import annotations

import math

import numpy as np

import torch

from .engine import release_cache

from .plotstyle import GRAY, STATUS, T, plt, themed


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
            color=T.ink2, fontsize=8.5)


@themed
def movie_figure(path: str, mic, fit, title: str = "", frc_eo=None):
    """mic: preprocess.Micrograph, fit: preprocess.CTFFit (of the plain sum), frc_eo: (k, frc) of the even/odd
    sums or None."""
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.2))
    (a_tr, a_loc, a_img, a_ps), (a_1d, a_cc, a_eo, a_dw) = axs
    sh = mic.shifts
    n = len(sh)

    # global drift: the shift applied to each frame, frame order as colour
    a_tr.plot(sh[:, 0], sh[:, 1], color=T.line, lw=1.5, zorder=1)
    sc = a_tr.scatter(sh[:, 0], sh[:, 1], c=np.arange(1, n + 1), cmap=T.density, vmin=-0.4 * n, vmax=n, s=22,
                      zorder=2, edgecolors="none")                 # ramp starts above the surface: frame 1 visible
    a_tr.annotate("frame 1", sh[0], xytext=(6, 4), textcoords="offset points", color=T.ink2, fontsize=8)
    a_tr.annotate(f"frame {n}", sh[-1], xytext=(6, -10), textcoords="offset points", color=T.ink2, fontsize=8)
    a_tr.set_aspect("equal", "datalim")
    a_tr.set_xlabel("x shift (Å)")
    a_tr.set_ylabel("y shift (Å)")
    path_a = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).sum())
    a_tr.set_title("Whole-frame motion")
    _note(a_tr, f"path {path_a:.1f} Å\nfirst→last {np.linalg.norm(sh[-1] - sh[0]):.1f} Å")
    cb = fig.colorbar(sc, ax=a_tr, fraction=0.04, pad=0.02)
    cb.set_label("frame", color=T.ink2)
    cb.ax.set_ylim(1, n)

    # local motion: residual trajectories at the patch centres (measured dots, model lines), magnified
    loc = getattr(mic, "local", None)
    if loc:
        cen, meas = loc["centres"], loc["measured"]
        mag = 0.15 / max(float(np.abs(meas).max()), 1e-3)                  # longest excursion ~0.15 of the field
        mod = loc.get("model")
        for p in range(len(cen)):
            cx, cy = cen[p]
            a_loc.scatter(cx + meas[p, :, 0] * mag, cy + meas[p, :, 1] * mag, s=4, color=T.series[0], alpha=0.55,
                          edgecolors="none", label="measured (patch)" if p == 0 else None)
            if mod is not None:
                a_loc.plot(cx + mod[p, :, 0] * mag, cy + mod[p, :, 1] * mag, color=T.series[1], lw=1.6,
                           label="model" if p == 0 else None)
            a_loc.plot([cx], [cy], marker="+", color=T.ink3, ms=6)
        a_loc.set_xlim(-1.15, 1.15)
        a_loc.set_ylim(1.15, -1.15)
        a_loc.set_aspect("equal")
        a_loc.set_xticks([])
        a_loc.set_yticks([])
        a_loc.legend(loc="lower left", bbox_to_anchor=(0, -0.02), fontsize=8, frameon=False, ncol=2)
        bar = 2.0 if np.abs(meas).max() > 2 else 1.0                  # scale bar in A of shift
        a_loc.plot([0.55, 0.55 + bar * mag], [1.09, 1.09], color=T.ink, lw=2)
        a_loc.text(0.55 + bar * mag + 0.03, 1.09, f"{bar:g} Å shift", color=T.ink2, fontsize=8, va="center")
        a_loc.set_title(f"Local motion (after whole-frame); model rms {loc['rms_a']:.2f} Å")
    else:
        a_loc.axis("off")
        a_loc.text(0.5, 0.5, "whole-frame motion only", ha="center", va="center", color=T.ink3,
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
    ns = d2.shape[0]                                                 # (not T: that is the theme)
    full = np.concatenate([d2[::-1, :0:-1], m2[:, :-1]], 1)          # Hermitian mirror for the left half
    full = np.concatenate([np.full((ns, 1), np.nan), full], 1)[:, :ns]
    v = np.nanpercentile(np.abs(d2), 98)
    a_ps.imshow(full, cmap=GRAY, vmin=-v, vmax=v, extent=[-0.5 / pr["apix"], 0.5 / pr["apix"]] * 2)
    th = np.linspace(0, 2 * math.pi, 200)
    r = 1 / fit.fit_res_a
    a_ps.plot(r * np.cos(th), r * np.sin(th), color=T.series[1], lw=1, ls="--")
    a_ps.set_xticks([])
    a_ps.set_yticks([])
    a_ps.set_title("Power spectrum: data | fit")
    a_ps.set_xlabel(f"background removed; circle: fit resolution {fit.fit_res_a:.2f} Å")

    # equal-phase 1D profile
    k1, dd, mm = pr["k1d"], pr["data1d"], pr["model1d"]
    a_1d.plot(k1, dd, color=T.series[0], lw=1.2, label="spectrum − background")
    a_1d.plot(k1, mm, color=T.series[1], lw=1.6, label="envelope × CTF²")
    a_1d.axvline(1 / fit.fit_res_a, color=T.ink3, lw=1, ls="--")
    _res_axis(a_1d, k1)
    a_1d.set_ylabel("normalised amplitude")
    a_1d.set_title("CTF fit (equal-phase average)")
    a_1d.legend(loc="center right", fontsize=8, frameon=False)
    _note(a_1d, f"defocus U {fit.defocus_u:,.0f} Å\nV {fit.defocus_v:,.0f} Å, angle {fit.angle:.1f}°\n"
                f"explained variance {fit.score:.3f}")

    # ring-by-ring correlation with the significance threshold
    kc, cc, nn = pr["k"], pr["cc"], pr["n"]
    a_cc.plot(kc, cc, color=T.series[0], lw=1.6, marker="o", ms=3, label="correlation per ring pair")
    a_cc.plot(kc, pr["z"] / np.sqrt(nn), color=T.ink3, lw=1, ls=":", label=f"significance ({pr['z']:g}/√pixels)")
    a_cc.axvline(1 / fit.fit_res_a, color=T.ink3, lw=1, ls="--")
    a_cc.axhline(0, color=T.line, lw=1)
    _res_axis(a_cc, kc)
    a_cc.set_ylabel("correlation")
    a_cc.set_title("Thon-ring fit significance")
    a_cc.legend(loc="upper right", fontsize=8, frameon=False)
    a_cc.text(0.02, 0.04, f"fit resolution {fit.fit_res_a:.2f} Å", transform=a_cc.transAxes, color=T.ink, fontsize=8.5)

    # even / odd frame sums
    if frc_eo is not None:
        k, f = frc_eo
        sel = k >= 1 / 60
        a_eo.plot(k[sel], f[sel], color=T.series[0], lw=1.6)
        a_eo.axhline(0, color=T.line, lw=1)
        _res_axis(a_eo, k[sel])
        a_eo.set_ylabel("FRC")
        a_eo.set_title("Even vs odd frames (alignment check)")
        _note(a_eo, "independent noise:\nhigher = better aligned")
    else:
        a_eo.axis("off")

    # local defocus (patch CTF): each region's offset from the micrograph's defocus, the fitted plane as contours
    pz = getattr(fit, "patch", None)
    if pz and pz.get("offsets_a"):
        g = int(pz.get("grid") or round(math.sqrt(len(pz["offsets_a"]))))
        cx, cy = pz["centre_a"]
        off = np.asarray(pz["offsets_a"], float)
        if len(off) == g * g:
            lim = max(50.0, float(np.abs(off).max()))
            xe, ye = np.linspace(-cx, cx, g + 1) / 1e4, np.linspace(-cy, cy, g + 1) / 1e4      # um
            m = a_dw.pcolormesh(xe, ye, off.reshape(g, g), cmap=T.diverging_cmap, vmin=-lim, vmax=lim)
            for (x, y), o in zip(np.asarray(pz["regions_a"]), off):
                a_dw.text(x / 1e4, y / 1e4, f"{o:+.0f}", ha="center", va="center", fontsize=8, color=T.ink)
            a, bx, by = pz["plane"]
            X, Y = np.meshgrid(np.linspace(-cx, cx, 50), np.linspace(-cy, cy, 50))
            cs = a_dw.contour(X / 1e4, Y / 1e4, a + bx * X + by * Y, levels=7, colors=[T.ink2], linewidths=0.8)
            a_dw.clabel(cs, fontsize=7, fmt="%.0f")
            a_dw.set_aspect("equal")
            a_dw.invert_yaxis()                               # as the micrograph is shown: first row at the top
            a_dw.set_ylabel("y (µm)")
            cb = a_dw.figure.colorbar(m, ax=a_dw, fraction=0.046, pad=0.02)
            cb.set_label("defocus offset (Å)", color=T.ink2)
        state = "applied" if pz.get("applied") else ("significant, not applied" if pz.get("significant")
                                                      else "not significant")
        a_dw.set_title(f"Local defocus: tilt {pz['tilt_deg']:.2f}°, {state}")
        a_dw.set_xlabel(f"x (µm) · regions ±{pz.get('residual_a', float('nan')):.0f} Å about the plane")
    else:
        a_dw.axis("off")

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=T.ink, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@themed
def summary_figure(path: str, rows: list[dict], ref_name: str = "cryoSPARC"):
    """rows: preprocess_eval.py JSON lines. Ours vs the reference per micrograph."""
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    a_df, a_as, a_fr, a_mo = axs
    rp = [r for r in rows if "cryosparc_particles" in r]

    def scatter(ax, x, y, xl, yl, title, fmt):
        x, y = np.asarray(x, float), np.asarray(y, float)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=T.line, lw=1)
        ax.scatter(x, y, s=30, color=T.series[0], edgecolors=T.series[0], linewidths=0)
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
    a_mo.barh(np.arange(len(pl)), pl[order], color=T.series[0], height=0.75)
    a_mo.set_yticks(np.arange(len(pl)))
    a_mo.set_yticklabels([rows[i]["movie"][-15:] for i in order], fontsize=7)
    a_mo.set_xlabel("whole-frame motion path (Å)")
    a_mo.set_title("Motion per movie")
    t = np.median([[r["read_s"], r["motion_s"], r["ctf_s"]] for r in rows], 0)
    _note(a_mo, f"median per movie:\nread {t[0]:.1f} s, motion {t[1]:.1f} s\nCTF {t[2]:.1f} s", top=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@themed
def fsc_figure(path: str, curves: dict, apix: float, box: int, title: str = "Half-map FSC"):
    """curves: name -> FSC per shell (shell i at i / (box * apix) 1/A); the 0.143 crossing is noted per curve."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    lines = []
    for j, (name, f) in enumerate(curves.items()):
        f = np.asarray(f)
        k = np.arange(len(f)) / (box * apix)
        below = np.flatnonzero((f < 0.143) & (k > 0))
        res = 1 / k[below[0]] if len(below) else 1 / k[-1]
        ax.plot(k[1:], f[1:], color=T.series[j % 3], lw=1.8, label=f"{name}: {res:.2f} Å")
        lines.append(res)
    ax.axhline(0.143, color=T.ink3, lw=1, ls=":")
    ax.axhline(0, color=T.line, lw=1)
    _res_axis(ax, np.arange(1, len(f)) / (box * apix))
    ax.set_ylabel("FSC")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@themed
def picks_figure(path: str, cand, keep: np.ndarray, diameter_a: float, threshold: float, title: str = "",
                 ref_xy_a: np.ndarray | None = None):
    """Picks on the binned, phase-flipped micrograph used for picking (circles: kept picks; optional reference
    positions as dots), the score map, and the candidate score distribution with the threshold."""
    w = cand.work
    fig, (a_img, a_sc, a_h) = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    img, wa = w["image"], w["apix"]
    lo, hi = np.percentile(img, [1, 99])
    a_img.imshow(img, cmap=GRAY, vmin=lo, vmax=hi)
    th = np.linspace(0, 2 * math.pi, 40)
    r = diameter_a / 2 / wa
    xy = cand.xy_a[keep] / wa
    for x, y in xy:
        a_img.plot(x + r * np.cos(th), y + r * np.sin(th), color=T.series[0], lw=0.8)
    if ref_xy_a is not None:
        a_img.scatter(ref_xy_a[:, 0] / wa, ref_xy_a[:, 1] / wa, s=6, color=T.series[1], edgecolors="none",
                      label="reference particles")
        a_img.legend(loc="lower left", bbox_to_anchor=(0, -0.09), fontsize=8, frameon=False)
    a_img.set_xticks([])
    a_img.set_yticks([])
    a_img.set_title(f"Picks: {len(keep)} of {len(cand.score)} candidates")
    a_img.set_xlabel(f"binned to {wa:.1f} Å/px, phase-flipped; circles: {diameter_a:g} Å")
    sm = w["score"]
    v = np.percentile(np.abs(sm), 99.5)
    a_sc.imshow(sm, cmap=T.density, vmin=0, vmax=v)
    a_sc.set_xticks([])
    a_sc.set_yticks([])
    a_sc.set_title("Blob score (normalised cross-correlation)")
    h, e = np.histogram(cand.score, bins=40)
    a_h.bar(0.5 * (e[1:] + e[:-1]), h, width=0.9 * (e[1] - e[0]), color=T.series[0])
    a_h.axvline(threshold, color=T.series[1], lw=1.6)
    a_h.text(threshold, max(h) * 0.97, f"  threshold {threshold:.3f}", color=T.ink, fontsize=8.5, va="top")
    a_h.set_xlabel("score")
    a_h.set_ylabel("candidates")
    a_h.set_title("Score distribution")
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=T.ink, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@themed
def size_survey(path: str, project, mic_job, movie_ids, diameters_a=(80, 120, 160, 220, 300), n_gallery: int = 24,
                title: str = ""):
    """Particle size from the first micrographs, for an unknown sample, as one figure an agent or a person can read:
    top, each micrograph phase-flipped and low-passed (~15 A) with a 500 A scale bar and circles of the candidate
    diameters; then per candidate diameter, the best blob picks of all these micrographs (box 2 diameters) and their
    rotational average with its radial profile (where the profile meets the background is the particle's edge).
    A diameter that is right shows whole, separated particles in the gallery and an average whose edge sits near
    the drawn circle. Returns a dict per diameter: picks per micrograph, median score, edge of the average (A)."""
    import torch
    from . import picking as pk
    from .pipeline import _load_mic, optics_of
    optics = optics_of(project, mic_job)
    movie_ids = list(movie_ids)
    mics = [_load_mic(project, mic_job, int(m)) for m in movie_ids]
    nd = len(diameters_a)
    fig = plt.figure(figsize=(16, 3.4 + 2.3 * nd), dpi=100)
    gs = fig.add_gridspec(1 + nd, max(len(mics), 1) * 2, height_ratios=[1.5] + [1] * nd, hspace=0.35, wspace=0.08)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, nd))
    for i, (img, apix, ctf) in enumerate(mics):
        v, wa = pk.prepare(img, apix, 120.0, ctf, optics, dark=True, work_apix=max(apix, 5.0))
        v = v.cpu().numpy()
        h, w = v.shape
        s = int(min(h, w, 2000 / wa))                                   # central ~2000 A
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = v[y0:y0 + s, x0:x0 + s]
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        lo, hi = np.percentile(crop, [1, 99.5])
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi)
        ax.plot([s * 0.05, s * 0.05 + 500 / wa], [s * 0.95] * 2, color="w", lw=3)
        ax.text(s * 0.05, s * 0.92, "500 A", color="w", fontsize=8)
        x = s * 0.03
        for d, c in zip(diameters_a, colors):                          # the candidate sizes side by side, top left
            r = d / 2 / wa
            ax.add_patch(plt.Circle((x + r, s * 0.03 + r), r, fill=False, color=c, lw=1.5))
            x += 2 * r + s * 0.015
        ax.set_title(f"micrograph {movie_ids[i]} (defocus {(ctf.defocus_u + ctf.defocus_v) / 2 / 1e4:.2f} um)",
                     fontsize=8)
        ax.axis("off")
    out = {}
    for j, d in enumerate(diameters_a):
        box = int(np.ceil(2 * d / 4.0 / 2)) * 2                          # gallery at ~4 A/px
        ims, scores, per_mic = [], [], []
        for img, apix, ctf in mics:
            c = pk.blob_candidates(img, apix, d, ctf, optics, True)
            k = pk.select(c)                                              # indices, best first
            per_mic.append(len(k))
            top = k[: max(1, 4 * n_gallery // len(mics))]
            if len(top):
                ims.append(pk.extract(img, apix, c.xy_a[top], box, out_apix=max(apix, 2 * d / box), dark=True,
                                      diameter_a=d))
                scores.append(c.score[top])
        ims = np.concatenate(ims) if ims else np.zeros((0, box, box))
        sc = np.concatenate(scores) if scores else np.zeros(0)
        o = np.argsort(-sc)
        ims = ims[o]
        # gallery
        ng = min(n_gallery, len(ims))
        sub = gs[1 + j, :-2].subgridspec(2, max(1, (n_gallery + 1) // 2), wspace=0.03, hspace=0.03)
        for g in range(ng):
            ax = fig.add_subplot(sub[g % 2, g // 2])
            im = ims[g]
            lo, hi = np.percentile(im, [1, 99])
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax.add_patch(plt.Circle((box / 2, box / 2), box / 4, fill=False, color=colors[j], lw=0.8))
            ax.axis("off")
        # rotational average and radial profile
        avg = ims[: max(1, len(ims))].mean(0) if len(ims) else np.zeros((box, box))
        yy, xx = np.indices(avg.shape) - box / 2
        rr = np.hypot(yy, xx) * (2 * d / box)                           # A
        rb = np.arange(0, d, 2 * d / box)
        prof = np.array([avg[(rr >= a) & (rr < a + 2 * d / box)].mean() for a in rb])
        bgv = prof[-max(2, len(prof) // 6):].mean()
        pk_ = prof.max() - bgv
        edge = float(rb[np.flatnonzero(prof - bgv > 0.15 * pk_)[-1]] * 2) if pk_ > 0 else float("nan")
        ax = fig.add_subplot(gs[1 + j, -2])
        ax.imshow(avg, cmap="gray")
        ax.add_patch(plt.Circle((box / 2, box / 2), box / 4, fill=False, color=colors[j], lw=1))
        ax.axis("off")
        ax.set_title(f"average of {len(ims)}", fontsize=8)
        ax = fig.add_subplot(gs[1 + j, -1])
        ax.plot(rb, prof, color=colors[j])
        ax.axhline(bgv, color=T.ink3, lw=0.8)
        ax.axvline(d / 2, color=colors[j], ls="--", lw=0.8)
        if edge == edge:
            ax.axvline(edge / 2, color=STATUS["critical"], lw=0.8)
        ax.set_xlabel("radius (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"d={d:.0f} A: {np.mean(per_mic):.0f} picks/mic, edge ~{edge:.0f} A", fontsize=8)
        fig.text(0.005, 1 - (1.5 + j + 0.5) / (1.5 + nd) * 0.93 - 0.04, f"{d:.0f} A", fontsize=10, color=colors[j],
                 rotation=90, va="center")
        out[float(d)] = dict(picks_per_micrograph=float(np.mean(per_mic)),
                             median_score=round(float(np.median(sc)), 3) if len(sc) else None,
                             average_edge_a=round(edge, 1) if edge == edge else None)
    fig.suptitle(title or "particle size survey", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    release_cache()
    return out


@themed
def overview_figure(path: str, panels: list[tuple[str, str, np.ndarray]], title: str, sub: str = ""):
    """Job-wide overview: one panel per quantity (title, unit, value per micrograph in processing order), each a dot
    per micrograph with its median line and the 5-95% range in a note."""
    from .plotstyle import grid_y, note, title as title_block, top
    n = len(panels)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    W, H = max(5.6, 4.4 * cols), 2.9 * rows + 0.8
    fig, axs = plt.subplots(rows, cols, figsize=(W, H), squeeze=False)
    # margins in inches (as fractions they left no room for tick and axis labels in a one-panel figure)
    fig.subplots_adjust(left=0.8 / W, right=1 - 0.2 / W, top=top(fig, 0.75), bottom=0.55 / H, wspace=0.3,
                        hspace=0.55)
    for a in axs.ravel()[n:]:
        a.axis("off")
    for a, (name, unit, v) in zip(axs.ravel(), panels):
        v = np.asarray(v, float)
        x = np.arange(len(v))
        ok = np.isfinite(v)
        a.plot(x[ok], v[ok], "o", ms=3 if len(v) < 500 else 1.5, color=T.series[0], alpha=0.8, mew=0)
        if ok.any():
            med = float(np.median(v[ok]))
            a.axhline(med, color=T.ink2, lw=1, ls="--")
            lo, hi = np.percentile(v[ok], [5, 95])
            fmt = (lambda q: f"{q:,.0f}") if max(abs(lo), abs(hi)) >= 1000 else (lambda q: f"{q:.3g}")
            note(a, f"median {fmt(med)} {unit}\n5–95% {fmt(lo)}–{fmt(hi)}", "upper right")
        a.set(title=name, xlabel="micrograph (processing order)", ylabel=unit)
        grid_y(a)
    title_block(fig, title, sub)
    fig.savefig(path)
    plt.close(fig)


@themed
def shifts_figure(path: str, d: np.ndarray, per_mic: np.ndarray, title: str, diameter_a: float | None = None):
    """Re-centring shifts (n, 2) in A (aligned centre - pick): magnitude histogram, 2D density (with the pick
    diameter's quarter as a reference circle) and the median per micrograph."""
    from .plotstyle import grid_y, note, title as title_block, top
    r = np.linalg.norm(d, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1.1]))
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.75), bottom=0.14, wspace=0.45)
    a = ax[0]
    a.hist(r, bins=60, color=T.series[0])
    for q, ls in ((50, "--"), (95, ":")):
        v = np.percentile(r, q)
        a.axvline(v, color=T.ink2, lw=1, ls=ls)
        a.text(v, a.get_ylim()[1] * 0.95, f" {q}%: {v:.1f} Å", color=T.ink2, fontsize=8, va="top")
    a.set(xlabel="shift from the pick (Å)", ylabel="particles", title="Shift magnitude")
    grid_y(a)
    a = ax[1]
    lim = float(np.percentile(r, 99.5)) * 1.1 + 1
    h = a.hexbin(d[:, 0], d[:, 1], gridsize=40, extent=(-lim, lim, -lim, lim), cmap=T.density, mincnt=1)
    a.axhline(0, color=T.line, lw=0.8)
    a.axvline(0, color=T.line, lw=0.8)
    if diameter_a:
        th = np.linspace(0, 2 * math.pi, 200)
        a.plot(diameter_a / 4 * np.cos(th), diameter_a / 4 * np.sin(th), color=T.ink3, lw=1, ls="--")
    m = d.mean(0)
    a.plot([0, m[0]], [0, m[1]], color=T.series[1], lw=2)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dx (Å)", ylabel="dy (Å)", aspect="equal",
          title="2D distribution")
    note(a, f"mean ({m[0]:.1f}, {m[1]:.1f}) Å" + (f"\ncircle: ¼ pick diameter" if diameter_a else ""), "upper left")
    fig.colorbar(h, ax=a, fraction=0.046, pad=0.02).set_label("particles")
    a = ax[2]
    a.plot(np.arange(len(per_mic)), per_mic, "o", color=T.series[0], ms=4)
    a.axhline(np.median(per_mic), color=T.ink2, lw=1, ls="--")
    a.set(xlabel="micrograph (processing order)", ylabel="median shift (Å)", title="Per micrograph")
    grid_y(a)
    title_block(fig, "Re-centring shifts", title)
    fig.savefig(path)
    plt.close(fig)
