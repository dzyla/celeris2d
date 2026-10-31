"""Select particles by their values: the numbers a particle carries (pick score, defocus, CTF fit of its micrograph,
the change CtfRefine made, a classification's probability, ...) of any particle job - an extraction, a Select (also
of another Select), a CtfRefine / Polish job, a classification - as histograms, and a Select job of the particles
inside the chosen ranges. Used by the dashboard's "Select particles by their values" page (webapp /api/project/values,
/api/project/filter)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .project import PARTICLE_JOBS, Job, Project

# columns that are bookkeeping, not a property worth selecting on
SKIP = {"uid", "slice", "movie_id", "box", "apix", "mic_apix", "mic_ny", "mic_nx", "stack", "mic_path", "stem",
        "subdir", "copy", "half"}
NAMES = {"score": "pick score", "local_std": "local contrast (sd)", "defocus_u": "defocus U (Å)",
         "defocus_v": "defocus V (Å)", "defocus": "defocus (Å)", "defocus_angle": "astigmatism angle (°)",
         "ctf_fit_res": "CTF fit of the micrograph (Å)", "x_a": "x on the micrograph (Å)",
         "y_a": "y on the micrograph (Å)", "dx_a": "re-centring x (Å)", "dy_a": "re-centring y (Å)",
         "ctf_delta_a": "CtfRefine defocus change (Å)", "ctf_cc_gain": "CtfRefine fit gain",
         "cls": "class (0-based)", "maxprob": "class probability", "track_rms_a": "Polish track rms (Å)",
         "track_mean_a": "Polish centre shift (Å)", "astigmatism": "astigmatism (Å)"}


def values(project: Project, source) -> pd.DataFrame:
    """One row per particle of the source job: uid and every numeric property (the extraction's columns and its
    micrographs', and the source's own table where it has more, e.g. a classification's class and probability)."""
    src = project.job(source)
    ex = src if src.type in PARTICLE_JOBS else project.ancestor(src, PARTICLE_JOBS)
    if ex is None:
        raise ValueError(f"{src.name} does not descend from particles")
    df = project.particles(ex)
    if "defocus_u" in df and "defocus_v" in df:
        df["defocus"] = 0.5 * (df["defocus_u"] + df["defocus_v"])
        df["astigmatism"] = (df["defocus_u"] - df["defocus_v"]).abs()
    if src.type == "Select":
        df = df[df["uid"].isin(project.selected_uids(src))]
    elif src.id != ex.id:
        own = project.table(src, "particles")
        cols = [c for c in own.columns if c not in df.columns or c == "uid"]
        t = own.read(cols)
        o = pd.DataFrame({k: v for k, v in t.items() if k != "movie"}).drop_duplicates("uid")
        df = df.merge(o, on="uid", how="inner")
    keep = ["uid"] + [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])
                      and df[c].notna().any() and df[c].nunique() > 1]
    return df[keep].reset_index(drop=True)


def histograms(df: pd.DataFrame, bins: int = 40) -> list[dict]:
    """Per column: name, label, min, max, and counts in `bins` equal bins (robust range: 0.1-99.9 %, so one wild value
    does not squash the histogram; the slider still reaches the extremes)."""
    out = []
    for c in df.columns:
        if c == "uid":
            continue
        v = df[c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.1, 99.9])
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        h, e = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
        out.append(dict(name=c, label=NAMES.get(c, c.replace("_", " ")), min=float(v.min()), max=float(v.max()),
                        lo=float(lo), hi=float(hi), counts=h.tolist(), integer=bool(np.all(v == np.round(v)))))
    return out


def mask(df: pd.DataFrame, rules: list[dict]) -> np.ndarray:
    """Particles inside every rule's range ({col, min, max}; a missing bound is open)."""
    keep = np.ones(len(df), bool)
    for r in rules:
        v = df[r["col"]].to_numpy(np.float64)
        if r.get("min") is not None:
            keep &= v >= float(r["min"])
        if r.get("max") is not None:
            keep &= v <= float(r["max"])
    return keep


def select_by_values(project: Project, source, rules: list[dict], label: str = "") -> Job:
    """A Select job of the source's particles inside the ranges (the others kept as its rejected particles)."""
    src = project.job(source)
    rules = [r for r in rules if r.get("min") is not None or r.get("max") is not None]
    if not rules:
        raise ValueError("set a range on at least one value")
    df = values(project, src)
    bad = [r["col"] for r in rules if r["col"] not in df.columns]
    if bad:
        raise ValueError(f"{src.name} has no value {bad}")
    keep = mask(df, rules)
    if not keep.any():
        raise ValueError("no particle is inside these ranges")
    fmt = lambda x: "-" if x is None else f"{float(x):.4g}"
    note = "; ".join(f"{NAMES.get(r['col'], r['col'])} {fmt(r.get('min'))} to {fmt(r.get('max'))}" for r in rules)
    job = project.select(src, df["uid"].to_numpy()[keep], label=label or f"values_of{src.id:03d}",
                         note=f"{int(keep.sum()):,} of {len(df):,} particles of {src.name}: {note}",
                         rejected=df["uid"].to_numpy()[~keep])
    project.update_params(job, rules=rules)
    return job
