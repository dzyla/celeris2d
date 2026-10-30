"""Curate job: which micrographs of a MotionCtf job go on to picking (cryoSPARC's Curate Exposures, RELION's
micrograph selection). Each micrograph is judged by its CTF fit (resolution the Thon rings were fitted to, defocus,
astigmatism, fit score, tilt of the local-defocus plane) and its motion (total drift, local residual): fixed limits
where a value has a physical meaning; drift / local motion are limited only when asked (a value, or motion_outliers:
median + outlier_mad x MAD), since motion that cost resolution already shows in the CTF fit.

    cur = curate_job(project, mc, CurateSettings(max_ctf_fit_a=6))
    run_curate(project, cur)                                    # table, figure, summary with the limits used
    pick = pipeline.pick_job(project, mc, PickSettings(..., curate=cur.id))   # only accepted micrographs are picked

The limits are resolved when the job runs and stored in its summary; micrographs added later (more movies, live
collection) are judged against the same limits when the pipeline reaches them (accepted()).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .jobspec import params_of, settings_of
from .plotstyle import STATUS, T, grid_y, note, plt, themed, title, top
from .project import Job, Project, logged


@dataclass
class CurateSettings:
    max_ctf_fit_a: float = 8.0          # reject a CTF fitted only to worse than this (A): Thon rings fade early (thick
                                        # ice, drift, carbon); 0: no limit
    min_defocus_a: float = 3000.0       # reject nearer to focus than this (A): too little low-resolution contrast to pick
    max_defocus_a: float = 40000.0      # reject further from focus than this (A)
    max_astigmatism_a: float = 2000.0   # reject a larger defocus difference (A) between the CTF's axes
    min_ctf_score: float = 0.0          # reject a lower CTF fit score (0: no limit)
    max_ctf_tilt_deg: float = 0.0       # reject a steeper local-defocus plane (patch CTF; deg; 0: no limit)
    max_drift_a: float = 0.0            # reject a longer total drift (A); 0: automatic (with motion_outliers) or none
    max_local_rms_a: float = 0.0        # reject a larger local (patch) motion residual (A); 0: automatic or none
    motion_outliers: bool = False       # automatic drift / local-motion limits (median + outlier_mad x MAD). Off by
                                        # default: motion that cost resolution shows in the CTF fit, which already
                                        # limits; dz_2025_10 (1,000 movies), the automatic limits (41 A drift) rejected
                                        # 34 micrographs only for motion although their CTF fit was 5.45 A (median;
                                        # accepted 4.25 A) - e.g. 411 A of drift, 6.3 A fit, particles plainly visible
    outlier_mad: float = 5.0            # automatic limits: this many MADs above the median


# statistic: (column, the settings that limit it from below / above)
_LIMITS = dict(ctf_fit_a=(None, "max_ctf_fit_a"), defocus_a=("min_defocus_a", "max_defocus_a"),
               astigmatism_a=(None, "max_astigmatism_a"), ctf_score=("min_ctf_score", None),
               ctf_tilt_deg=(None, "max_ctf_tilt_deg"), drift_a=(None, "max_drift_a"),
               local_rms_a=(None, "max_local_rms_a"))
_AUTO = ("max_drift_a", "max_local_rms_a")
_LABEL = dict(ctf_fit_a="CTF fit resolution (Å)", defocus_a="defocus (Å)", astigmatism_a="astigmatism (Å)",
              ctf_score="CTF fit score", ctf_tilt_deg="local-defocus tilt (deg)", drift_a="total drift (Å)",
              local_rms_a="local motion residual (Å)")


def micrograph_table(project: Project, mc) -> pd.DataFrame:
    """One row per micrograph of a MotionCtf job: movie_id, stem and the statistics judged."""
    mi = project.micrographs(mc)
    meta = [json.loads(m or "{}") if isinstance(m, str) else (m or {}) for m in mi["meta"]]

    def g(k):
        return np.array([np.nan if m.get(k) is None else float(m.get(k)) for m in meta], float)
    return pd.DataFrame(dict(movie_id=mi["movie_id"].to_numpy(), stem=mi["stem"].to_numpy(),
                             ctf_fit_a=mi["ctf_fit_res"].to_numpy(float),
                             defocus_a=(mi["defocus_u"] + mi["defocus_v"]).to_numpy(float) / 2,
                             astigmatism_a=(mi["defocus_u"] - mi["defocus_v"]).abs().to_numpy(float),
                             ctf_score=g("ctf_score"), ctf_tilt_deg=g("ctf_tilt_deg"), drift_a=g("motion_path_a"),
                             local_rms_a=g("local_rms_a")))


def limits(s: CurateSettings, df: pd.DataFrame) -> dict:
    """Resolved limits {statistic: [low or None, high or None]}; automatic ones from the data in df."""
    out = {}
    for col, (lo, hi) in _LIMITS.items():
        a = getattr(s, lo) if lo else 0.0
        b = getattr(s, hi) if hi else 0.0
        if hi in _AUTO and not b and s.motion_outliers:
            v = df[col].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) >= 10:
                med = float(np.median(v))
                b = med + s.outlier_mad * max(1.4826 * float(np.median(np.abs(v - med))), 1e-3 * abs(med), 1e-6)
        out[col] = [float(a) if a else None, float(b) if b else None]
    return out


def judge(df: pd.DataFrame, lim: dict) -> tuple[np.ndarray, list[str]]:
    """(accepted mask, reason per micrograph: '' or the failed criteria). A missing value never rejects."""
    bad = {}
    for col, (lo, hi) in lim.items():
        v = df[col].to_numpy(float)
        f = np.zeros(len(v), bool)
        if lo is not None:
            f |= np.isfinite(v) & (v < lo)
        if hi is not None:
            f |= np.isfinite(v) & (v > hi)
        bad[col] = f
    reasons = [", ".join(c for c in bad if bad[c][i]) for i in range(len(df))]
    return np.array([not r for r in reasons], bool), reasons


def curate_job(project: Project, micrographs, s: CurateSettings, label: str = "") -> Job:
    mc = project.job(micrographs)
    if mc.type != "MotionCtf":
        raise ValueError(f"{mc.name} is not a MotionCtf job")
    return project.new_job("Curate", dict(params_of(s), micrographs=mc.id), [mc],
                           label or f"ctf{s.max_ctf_fit_a:g}A_on{mc.id:03d}")


@logged
def run_curate(project: Project, job, log=print) -> dict:
    job = project.job(job)
    s = settings_of(job, CurateSettings)
    df = micrograph_table(project, job.params["micrographs"])
    if not len(df):
        raise ValueError("no micrographs yet")
    lim = limits(s, df)
    ok, why = judge(df, lim)
    df["accepted"], df["reason"] = ok, why
    df.to_csv(job.out("micrographs.csv"), index=False)
    by = {c: int(sum(c in r.split(", ") for r in why)) for c in _LIMITS}
    by = {c: n for c, n in by.items() if n}
    _figure(job.out("curate.png"), df, lim, f"{job.name}: {int(ok.sum()):,} of {len(df):,} micrographs accepted")
    summ = dict(micrographs=len(df), accepted=int(ok.sum()), rejected=int((~ok).sum()), rejected_by=by,
                limits={c: v for c, v in lim.items() if v != [None, None]},
                warnings=[f"only {ok.mean() * 100:.0f}% of micrographs accepted"] if ok.mean() < 0.5 else [])
    log(f"[curate] {summ['accepted']:,} of {len(df):,} accepted; rejected by {by}; limits {summ['limits']}")
    project.set_status(job, "finished", summ)
    return summ


_cache: dict = {}


def accepted(project: Project, job, movie_id: int) -> bool:
    """Is this micrograph accepted by a Curate job (with the limits stored when it ran; also for micrographs that
    were not there then)?"""
    job = project.job(job)
    if job.id not in _cache:
        summ = project.summary_of(job)
        if job.status != "finished" or "limits" not in summ:
            raise ValueError(f"{job.name} has not run yet")
        lim = {c: summ["limits"].get(c, [None, None]) for c in _LIMITS}
        _cache.clear()
        _cache[job.id] = (lim, {})
    lim, seen = _cache[job.id]
    if movie_id not in seen:
        df = micrograph_table(project, job.params["micrographs"])
        ok, _ = judge(df, lim)
        seen.update(zip(df["movie_id"].tolist(), ok.tolist()))
    return bool(seen.get(movie_id, True))


@themed
def _figure(path, df: pd.DataFrame, lim: dict, title_: str):
    cols = [c for c in _LIMITS if np.isfinite(df[c].to_numpy(float)).any()]
    n = len(cols)
    nc = min(4, n)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.9 * nr + 0.6), squeeze=False)
    title(fig, "Micrograph curation", title_)
    fig.subplots_adjust(top=top(fig, 0.9), bottom=0.1, left=0.06, right=0.98, hspace=0.6, wspace=0.3)
    ok = df["accepted"].to_numpy(bool)
    for ax, c in zip(axes.flat, cols):
        v = df[c].to_numpy(float)
        f = np.isfinite(v)
        bins = np.histogram_bin_edges(v[f], bins=40)
        ax.hist([v[f & ok], v[f & ~ok]], bins=bins, stacked=True, color=[T.series[0], STATUS["critical"]],
                label=["accepted", "rejected"])
        for x in lim[c]:
            if x is not None:
                ax.axvline(x, color=STATUS["warning"], lw=1.2, ls="--")
        ax.set(xlabel=_LABEL[c], ylabel="micrographs")
        r = int(sum(c in str(w).split(", ") for w in df["reason"]))
        if r:
            note(ax, f"{r} rejected")
        grid_y(ax)
    axes.flat[0].legend(loc="upper left", fontsize=8, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.savefig(path)
