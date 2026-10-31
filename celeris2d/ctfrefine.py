"""Per-particle defocus refinement (RELION CtfRefine / cryoSPARC local CTF refinement) on a Refine3D job.

Every particle is compared with the projection of its own half map (gold standard: the half it was aligned against,
as in polishing) at its refined pose; only its defocus changes. The CTF phase is chi = pi lambda df s^2 - ... , so a
defocus offset D adds pi lambda D s^2: the score of every candidate offset comes from one precomputed phase. Score:
noise-weighted normalised correlation of the particle with the CTF(D)-modulated projection over a band (default
20 A to the refinement's resolution), offsets -range..range on a coarse grid, then a fine grid around the best, then
a parabola through the best three. The astigmatism (U - V, angle) stays the micrograph's.

A CtfRefine job is a particle job (like an extraction or a polishing) whose particles table carries per-particle
defocus_u / defocus_v / defocus_angle; it points at the stacks of the particles it refined (no copy). Every job that
reads particles (Refine3D, HeteroRefine, Polish, Class2D, exports) takes the per-particle values from it
(Project.particles). Per micrograph, a plane is fitted to the offsets (ice tilt / thickness gradient); plane=True
uses the plane instead of the per-particle values (less noise for small particles).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .project import PARTICLE_JOBS, Job, Project


@dataclass
class CtfRefineSettings:
    range_a: float = 1500.0         # defocus offsets searched: -range..+range (A)
    step_a: float = 25.0            # coarse grid step (A); then a fine grid of step/10 around the best, then a parabola
    res_lo_a: float = 20.0          # band used: from this resolution ...
    res_hi_a: float = 0.0           # ... to this one (0: the refinement's resolution, sphere-masked FSC 0.143)
    plane: bool = False             # use the per-micrograph plane fit of the offsets instead of the per-particle values
    shrink: bool = True             # per-particle offsets pulled toward their micrograph's plane by how well each is
                                    # measured (the sharpness of its score peak; empirical Bayes): raw per-particle
                                    # offsets scattered 174 A about the planes on INX-6 and cost 0.04 A (3.61 -> 3.65)
    batch: int = 256                # particles per GPU batch (fewer if memory is short)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def ctf_refine_job(project: Project, refine, s: CtfRefineSettings, label: str = "") -> Job:
    rj = project.job(refine)
    if rj.type != "Refine3D":
        raise ValueError(f"{rj.name} is not a Refine3D job")
    ex = project.job(rj.params["particles"])
    if ex.type not in PARTICLE_JOBS:
        raise ValueError(f"{rj.name}: its particles ({ex.name}) are not a particle job")
    e = ex.params
    keep = {k: e[k] for k in ("picks", "micrographs", "box", "out_apix", "box_a_used", "particle_a", "dark") if k in e}
    return project.new_job("CtfRefine", dict(asdict(s), refine=rj.id, particles=ex.id, **keep), [rj, ex],
                           label or f"defocus{'_plane' if s.plane else ''}_on{rj.id:03d}")


def _plane_fit(xy: np.ndarray, d: np.ndarray, n_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Robust plane d ~ a + b x + c y (iteratively reweighted, Huber-like): (coefficients, prediction)."""
    X = np.c_[np.ones(len(d)), xy]
    w = np.ones(len(d))
    coef = np.zeros(3)
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(X * sw[:, None], d * sw, rcond=None)[0]
        r = d - X @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
    return coef, X @ coef


def run_ctf_refine(project: Project, job, log=print) -> dict:
    from . import abinitio as ab
    from .engine import TWO_PI, freqs, release_cache
    from .jobs import _session, particle_source, settings_of
    from .polish import _half_maps
    job = project.job(job)
    s = settings_of(job, CtfRefineSettings)
    rj, ex = project.job(job.params["refine"]), project.job(job.params["particles"])
    rs = project.summary_of(rj)
    res_hi = s.res_hi_a or float(rs.get("resolution_a", 4.0))
    poses = project.table(rj, "particles").read()
    if "half" not in poses:
        raise ValueError(f"{rj.name} has no half sets (made by an older version: run the refinement again)")
    src = particle_source(project, ex)
    uid = src.parts["celUid"].to_numpy(np.int64)
    o = np.argsort(poses["uid"])
    pos = np.searchsorted(poses["uid"][o], uid).clip(0, len(o) - 1)
    has = poses["uid"][o[pos]] == uid
    src.subset(np.flatnonzero(has))
    row = o[pos[has]]
    src.path = f"{project.root}#{ex.id:03d}#{job.id:03d}"
    t0 = time.time()
    ses, _ = _session(project, job, src, 0.0, max(res_hi * 0.95, 2 * src.apix), s.gpus, log)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    clf.ensure_polarity()
    dev, b, box_a = clf.dev, clf.bmax, clf.box_a
    n = clf.N
    A = ab.euler_to_matrix(torch.as_tensor(poses["rot"][row]), torch.as_tensor(poses["tilt"][row]),
                           torch.as_tensor(poses["psi"][row])).float().to(dev)
    t = torch.as_tensor(np.stack([poses["shift_x_a"][row], poses["shift_y_a"][row]], 1)).float().to(dev)
    half = torch.as_tensor(poses["half"][row].astype(np.int64), device=dev)
    vols = _half_maps(project, rj, b, box_a / b, res_hi, dev)                  # masked, FSC-weighted, band box
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= box_a / s.res_lo_a) & (kr <= box_a / res_hi)
    fyb, fxb = fy[:, None].expand(b, b // 2 + 1)[band], fx[None, :].expand(b, b // 2 + 1)[band]
    s2 = (fyb ** 2 + fxb ** 2) / box_a ** 2                                    # (F,) 1/A^2
    ang = torch.atan2(fyb, fxb)
    shell = torch.round(kr[band]).long()
    herm = torch.where(fxb > 0, 2.0, 1.0)                                       # rfft half plane: count twice
    o_ = clf.optics
    lam, cs = o_.wavelength, o_.cs_mm * 1e7
    k3 = math.atan(o_.q0 / math.sqrt(1 - o_.q0 ** 2))
    ph = TWO_PI / box_a
    P0 = clf.ctf_p.to(dev).float()                                              # (n, 6) du, dv, angle, phase, B, scale

    def batch_terms(i, j):
        """Per particle of rows i..j: Q = Re(conj x . P e_t), R = |P|^2, X2 = |x|^2, chi0, envelope (all (nb, F))."""
        idx = torch.arange(i, j)
        x = clf._fetch(idx, dev, b)[:, band]
        A_, hh = A[i:j], half[i:j]
        Pm = torch.empty_like(x)
        for h in (0, 1):
            k = torch.nonzero(hh == h).squeeze(1)
            if len(k):
                Pm[k] = vols[h].project(A_[k])[:, band]
        tt = t[i:j]
        Pm = Pm * torch.exp(-1j * ph * (fxb * tt[:, 0, None] + fyb * tt[:, 1, None]))
        p = P0[i:j]
        du, dv, da = p[:, 0, None], p[:, 1, None], torch.deg2rad(p[:, 2, None])
        df = 0.5 * (du + dv) + 0.5 * (du - dv) * torch.cos(2 * (ang - da))
        chi0 = math.pi * lam * df * s2 - 0.5 * math.pi * cs * lam ** 3 * s2 * s2 + k3 + torch.deg2rad(p[:, 3, None])
        env = p[:, 5, None] * torch.exp(-p[:, 4, None] * s2 / 4)
        return (x.conj() * Pm).real, Pm.abs() ** 2, x.abs() ** 2, chi0, env, x, Pm

    # noise per shell: residual power at the starting defocus (a subset)
    nsh = int(shell.max()) + 1
    rp, rc = torch.zeros(nsh, device=dev), torch.zeros(nsh, device=dev)
    for i in range(0, min(n, 2000), s.batch):
        Q, R, X2, chi0, env, x, Pm = batch_terms(i, min(i + s.batch, n))
        c = env * torch.sin(chi0)
        a = (Q * c).sum(1, keepdim=True) / (R * c * c).sum(1, keepdim=True).clamp_min(1e-20)   # amplitude scale
        res = (x - a * c * Pm).abs() ** 2
        rp.index_add_(0, shell, (res * herm).sum(0))
        rc.index_add_(0, shell, herm * len(x))
    w = (herm / (rp / rc.clamp_min(1)).clamp_min(1e-20)[shell])                 # (F,)

    coarse = torch.arange(-s.range_a, s.range_a + 0.5 * s.step_a, s.step_a, device=dev)
    fine_step = s.step_a / 10
    fine_off = torch.arange(-1.0 * s.step_a, 1.0 * s.step_a + 0.5 * fine_step, fine_step, device=dev)
    delta = torch.zeros(n, device=dev)
    cc0 = torch.zeros(n, device=dev)
    cc1 = torch.zeros(n, device=dev)
    curv = torch.zeros(n, device=dev)            # sharpness of each score peak: -d2 score / d(offset)^2 (per A^2)
    at_edge = torch.zeros(n, dtype=torch.bool, device=dev)
    F = int(band.sum())
    free = ab._free(dev)
    nd = 8                                                                       # offsets evaluated at once
    bt = int(max(16, min(s.batch, free * 0.3 / (nd * F * 4 * 4))))

    def scores(Q, R, X2, chi0, env, D, wv=None):
        """Normalised weighted correlation (nb, len(D)) for defocus offsets D (nb, nd) or (nd,); wv: pixel weights."""
        wv = w if wv is None else wv
        out = []
        D = D if D.dim() == 2 else D[None].expand(len(Q), -1)
        for k in range(0, D.shape[1], nd):
            c = env[:, None] * torch.sin(chi0[:, None] + math.pi * lam * D[:, k:k + nd, None] * s2)   # (nb, nd, F)
            num = (wv * Q)[:, None].mul(c).sum(-1)
            den = (wv * R)[:, None].mul(c * c).sum(-1)
            out.append(num / (den * (wv * X2).sum(1, keepdim=True)).sqrt().clamp_min(1e-20))
        return torch.cat(out, 1)

    def search(Q, R, X2, chi0, env, wv=None):
        """Best offset per particle (coarse grid, fine grid, parabola), whether it is at the search edge, and the
        sharpness of the score peak."""
        sc = scores(Q, R, X2, chi0, env, coarse, wv)
        best = coarse[sc.argmax(1)]
        edge = (best <= coarse[0] + 1e-3) | (best >= coarse[-1] - 1e-3)
        Df = best[:, None] + fine_off[None]
        sf = scores(Q, R, X2, chi0, env, Df, wv)
        k = sf.argmax(1).clamp(1, len(fine_off) - 2)
        ar = torch.arange(len(Q), device=dev)
        y0, y1, y2 = sf[ar, k - 1], sf[ar, k], sf[ar, k + 1]
        den = y0 - 2 * y1 + y2
        frac = torch.where(den < 0, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-1, 1)
        return Df[ar, k] + frac * fine_step, edge, (-den).clamp_min(0) / fine_step ** 2

    # two random halves of the band's pixels: the offsets they give apart measure each particle's noise (shrink)
    split = s.shrink and not s.plane
    if split:
        g_ = torch.Generator().manual_seed(1)
        ma = (torch.rand(int(band.sum()), generator=g_) < 0.5).float().to(dev)
        wa, wb = w * ma, w * (1 - ma)
        da, db = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    for i in range(0, n, bt):
        j = min(i + bt, n)
        Q, R, X2, chi0, env, _, _ = batch_terms(i, j)
        delta[i:j], at_edge[i:j], curv[i:j] = search(Q, R, X2, chi0, env)
        if split:
            da[i:j], ea, _ = search(Q, R, X2, chi0, env, wa)
            db[i:j], eb, _ = search(Q, R, X2, chi0, env, wb)
            at_edge[i:j] |= ea | eb
        cc0[i:j] = scores(Q, R, X2, chi0, env, torch.zeros(1, device=dev))[:, 0]
        cc1[i:j] = scores(Q, R, X2, chi0, env, delta[i:j, None])[:, 0]
    seconds = time.time() - t0
    delta, cc0, cc1, at_edge, curv = (v.cpu().numpy() for v in (delta, cc0, cc1, at_edge, curv))
    gap = (da - db).cpu().numpy() if split else None

    # per micrograph: robust plane of the offsets over the particle positions
    ctr = np.stack([src.parts["rlnCoordinateX"].to_numpy(float), src.parts["rlnCoordinateY"].to_numpy(float)], 1)
    ctr = ctr * float(project.micrographs(job.params["micrographs"])["apix"].iloc[0])
    mov = src.parts["celMovie"].to_numpy(np.int64)
    plane = np.zeros_like(delta)
    per = []
    for m in np.unique(mov):
        sel = np.flatnonzero((mov == m) & ~at_edge)
        if len(sel) >= 8:
            coef, pred = _plane_fit((ctr[sel] - ctr[sel].mean(0)) / 1e4, delta[sel])
            plane[mov == m] = coef[0] + ((ctr[mov == m] - ctr[sel].mean(0)) / 1e4) @ coef[1:]
            r = delta[sel] - pred
            per.append(dict(movie=int(m), n=int(len(sel)), median_a=float(np.median(delta[sel])),
                            tilt_a_per_um=[float(coef[1]), float(coef[2])],
                            scatter_a=float(1.4826 * np.median(np.abs(r - np.median(r))))))
        else:
            plane[mov == m] = np.median(delta[mov == m]) if (mov == m).any() else 0.0
    shrunk = None
    if s.shrink and not s.plane:
        shrunk, sh_info = _shrink(delta, plane, curv, at_edge, gap)
        log(f"[ctf] offsets about the micrograph planes: real spread {sh_info['tau_a']:.0f} A, measurement noise "
            f"{sh_info['noise_median_a']:.0f} A (median particle): each offset keeps {sh_info['weight_median'] * 100:.0f}% "
            f"(median) of its distance from the plane")
    use = plane if s.plane else (shrunk if shrunk is not None else delta)
    use = np.where(at_edge, plane, use)                                        # an offset at the search edge is not trusted

    # the particles table: every particle of the refined particle job, per-particle CTF (refined where it had a pose)
    tab = project.table(ex, "particles").read()
    allp = project.particles(ex, ["uid"])
    ou = np.argsort(allp["uid"].to_numpy())
    du_all = allp["defocus_u"].to_numpy(np.float64)[ou]
    dv_all = allp["defocus_v"].to_numpy(np.float64)[ou]
    an_all = allp["defocus_angle"].to_numpy(np.float64)[ou]
    tu = np.asarray(tab["uid"], np.int64)
    k_all = np.searchsorted(allp["uid"].to_numpy()[ou], tu)
    du, dv, an = du_all[k_all], dv_all[k_all], an_all[k_all]
    d_all = np.full(len(tu), np.nan, np.float32)
    gain = np.full(len(tu), np.nan, np.float32)
    ok_ = np.argsort(tu)
    hit = ok_[np.searchsorted(tu[ok_], uid[has])]
    d_all[hit] = use
    gain[hit] = cc1 - cc0
    du = du + np.nan_to_num(d_all)
    dv = dv + np.nan_to_num(d_all)
    cols = {k: v for k, v in tab.items() if k != "movie"}
    cols.update(defocus_u=du.astype(np.float32), defocus_v=dv.astype(np.float32), defocus_angle=an.astype(np.float32),
                ctf_delta_a=d_all, ctf_cc_gain=gain)
    project.table(job, "particles").write(cols, tab["movie"])
    with project._db() as c:                                                    # the same stacks as the refined job
        c.execute("INSERT OR REPLACE INTO chunks SELECT ?, movie_id, table_path, stack_path, n, box, apix, ? FROM chunks "
                  "WHERE job_id=?", (job.id, time.time(), ex.id))
    project.consolidate(job)

    scat = np.array([p["scatter_a"] for p in per]) if per else np.zeros(1)
    summ = dict(particles=int(len(tu)), refined=int(len(use)), seconds=round(seconds, 1), band_a=[s.res_lo_a, round(res_hi, 2)],
                mode="plane" if s.plane else "particle, shrunk to the plane" if shrunk is not None else "particle",
                **({"shrink": sh_info} if shrunk is not None else {}),
                delta_median_a=round(float(np.median(use)), 1), delta_abs_median_a=round(float(np.median(np.abs(use))), 1),
                delta_p05_p95_a=[round(float(np.percentile(use, 5)), 1), round(float(np.percentile(use, 95)), 1)],
                at_search_edge=round(float(at_edge.mean()), 4),
                per_particle_scatter_about_plane_a=round(float(np.median(scat)), 1),
                micrograph_median_delta_a=[round(p["median_a"], 1) for p in per],
                cc_gain_mean=round(float(np.mean(cc1 - cc0)), 5), cc_improved_share=round(float(np.mean(cc1 > cc0)), 4))
    _figure(job.out("ctfrefine.png"), use, delta, plane, mov, per, cc1 - cc0,
            f"{job.name} · {len(use):,} particles of {rj.name} · band {s.res_lo_a:g}-{res_hi:.2f} Å")
    project.set_status(job, "finished", summ)
    ses.clf = None
    release_cache()
    log(f"[ctf] {job.name}: {len(use):,} particles, offset median {summ['delta_median_a']:+.0f} A, |offset| median "
        f"{summ['delta_abs_median_a']:.0f} A, scatter about the micrograph planes {summ['per_particle_scatter_about_plane_a']:.0f} A, "
        f"{summ['cc_improved_share'] * 100:.0f}% fit better ({seconds:.0f} s)")
    return summ


def _shrink(delta, plane, curv, at_edge, gap=None, n_bins: int = 10):
    """Empirical Bayes: residual r = offset - plane = real variation (variance tau^2) + measurement noise. The noise
    variance of a particle is c / curvature (its score peak's sharpness), c calibrated by `gap`, the difference of
    the offsets from two random halves of the band's pixels (each half has twice the noise variance of the whole, so
    var(gap) = 4 x the noise variance); tau^2 = what the residuals' robust variance holds beyond that noise, in bins of
    the curvature. Each offset is then plane + tau^2 / (tau^2 + noise) r: the plane when the variation is noise, the
    particle's own value when it is measured sharply. (The curvature alone, uncalibrated, put beta-gal's noise above
    its whole scatter - 42 vs 29 A - and every offset on the plane: 3.01 A vs 2.87 A with the particles' own.)"""
    ok = ~at_edge & (curv > 0)
    r = delta - plane
    x = 1.0 / np.maximum(curv, 1e-12)
    mad2 = lambda v: (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2
    xs, vr, vn = [], [], []
    if ok.sum() >= 50 * n_bins:
        edges = np.quantile(x[ok], np.linspace(0, 1, n_bins + 1))
        for a, b in zip(edges[:-1], edges[1:]):
            sel = ok & (x >= a) & (x <= b)
            if sel.sum() >= 20:
                xs.append(float(np.median(x[sel])))
                vr.append(mad2(r[sel]))
                vn.append(mad2(gap[sel]) / 4 if gap is not None else np.nan)
    if len(xs) < 3:                                     # too few particles to tell: the plane
        noise, tau2 = np.full(len(r), np.inf), 0.0
    else:
        xs, vr, vn = np.array(xs), np.array(vr), np.array(vn)
        if gap is not None:                             # noise = c x, c from the split halves (least squares, origin)
            c = float(np.dot(xs, vn) / np.dot(xs, xs))
        else:                                           # uncalibrated: the residual variance's slope in x
            c = max(0.0, float(np.polyfit(xs, vr, 1)[0]))
        tau2 = float(max(0.0, np.median(vr - c * xs)))
        noise = c * x
    wgt = np.where(ok, tau2 / np.maximum(tau2 + noise, 1e-12), 0.0)
    out = plane + wgt * r
    return out, dict(tau_a=round(float(np.sqrt(tau2)), 1),
                     noise_median_a=round(float(np.sqrt(np.median(noise[ok]))) if ok.any() and np.isfinite(noise).all() else 0.0, 1),
                     weight_median=round(float(np.median(wgt[ok])) if ok.any() else 0.0, 3),
                     calibrated=gap is not None)


def _figure(path, use, delta, plane, mov, per, gain, title_):
    from .plotstyle import themed
    themed(_draw)(path, use, delta, plane, mov, per, gain, title_)


def _draw(path, use, delta, plane, mov, per, gain, title_):
    from .plotstyle import T, grid_y, note, plt, title
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.14, wspace=0.3)
    a = ax[0]
    lim = float(np.percentile(np.abs(delta), 99)) * 1.1 + 10
    a.hist(np.clip(delta, -lim, lim), bins=80, color=T.series[0])
    a.axvline(0, color=T.ink3, lw=1, ls="--")
    a.set(xlabel="defocus offset from the micrograph's (Å)", ylabel="particles", title="Per-particle offsets")
    note(a, f"median {np.median(delta):+.0f} Å\n|offset| median {np.median(np.abs(delta)):.0f} Å", "upper right")
    grid_y(a)
    a = ax[1]
    ms = np.unique(mov)
    for i, m in enumerate(ms):
        d = delta[mov == m]
        lo, med, hi = np.percentile(d, [25, 50, 75])
        a.plot([i, i], [lo, hi], color=T.series[0], lw=3, alpha=0.6)
        a.plot([i], [med], "o", color=T.series[0], ms=5)
    a.axhline(0, color=T.ink3, lw=1, ls="--")
    a.set(xlabel="micrograph", ylabel="offset (Å)", title="Per micrograph (median, 25-75 %)")
    grid_y(a)
    a = ax[2]
    a.plot(plane, delta, "o", ms=1.5, color=T.series[0], alpha=0.4)
    v = np.percentile(np.r_[plane, delta], [1, 99])
    a.plot(v, v, color=T.ink3, lw=1, ls="--")
    a.set(xlim=v, ylim=v, xlabel="micrograph plane fit (Å)", ylabel="per-particle offset (Å)",
          title="Per-particle vs plane (ice tilt)")
    if per:
        note(a, f"scatter about plane {np.median([p['scatter_a'] for p in per]):.0f} Å (median)\n"
                f"{(gain > 0).mean() * 100:.0f}% of particles fit better", "upper left")
    title(fig, "Per-particle defocus refinement", title_)
    fig.savefig(path)
    plt.close(fig)
