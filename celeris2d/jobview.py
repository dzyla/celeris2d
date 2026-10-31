"""What the dashboard shows on the Import, Pick and Extract job pages - each job's own numbers, not the micrographs'
(those are MotionCtf's panel, micview.py): the imported movies (count, format, frames, sizes, when they were
recorded), the picks (per micrograph, scores kept / rejected, against defocus) and the particles (per micrograph,
box, a random handful as images). Read-only, computed in the dashboard's request threads.
"""
from __future__ import annotations

import base64
import io
import math
import os
from collections import Counter

import numpy as np

from .project import Project


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _header(path: str) -> dict:
    """Frames, size and pixel size of one movie from its header (TIFF: the page count and first page)."""
    p = path.lower()
    try:
        if p.endswith((".mrc", ".mrcs")):
            import warnings
            import mrcfile
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mrcfile.open(path, header_only=True, permissive=True) as m:
                    h = m.header
                    return dict(frames=int(h.nz), nx=int(h.nx), ny=int(h.ny), apix=round(float(m.voxel_size.x), 4) or None,
                                mode=int(h.mode))
        if p.endswith((".tif", ".tiff")):
            import tifffile
            with tifffile.TiffFile(path) as t:
                pg = t.pages[0]
                return dict(frames=len(t.pages), ny=int(pg.shape[0]), nx=int(pg.shape[1]), dtype=str(pg.dtype),
                            compression=str(getattr(pg.compression, "name", pg.compression)))
    except Exception:                                 # noqa: BLE001 - shown as unknown
        pass
    return {}


def import_info(project: Project, job) -> dict:
    """An Import job: how many movies, where, in what format and size, the first movie's header, and when they were
    recorded (file times: a timeline and the rate) - the session at a glance."""
    j = project.job(job)
    mv = project.movies(j)
    paths = mv["path"].tolist()
    t, size, missing = [], 0, 0
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            missing += 1
            continue
        t.append(st.st_mtime)
        size += st.st_size
    t = np.sort(np.asarray(t, np.float64))
    ext = Counter(os.path.splitext(p)[1].lower() for p in paths)
    sub = Counter(mv["subdir"].fillna("").tolist())
    head = _header(paths[0]) if paths else {}
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out = dict(job=j.name, movies=len(paths), missing=missing, bytes=size, formats=dict(ext),
               folders=[dict(name=k or "(top folder)", n=v) for k, v in sub.most_common(30)],
               root=j.params.get("root") or (os.path.commonpath(paths) if paths else ""),
               first=os.path.basename(paths[0]) if paths else None, header=head,
               recorded_from=float(t[0]) if len(t) else None, recorded_to=float(t[-1]) if len(t) else None,
               hours=span / 3600, per_hour=(len(t) - 1) / span * 3600 if span > 0 else None,
               created=j.params.get("run_started") or (float(mv["added"].min()) if len(mv) else None),
               skipped=project.summary_of(j).get("skipped", []))
    step = max(1, len(t) // 4000)                     # the timeline, thinned for the browser
    out["times"] = t[::step].round(1).tolist()
    out["times_step"] = step
    if len(t) > 2:                                    # gaps longer than 10x the median interval: grid changes, pauses
        d = np.diff(t)
        med = float(np.median(d))
        gaps = np.nonzero(d > max(10 * med, 300))[0]
        out["median_interval_s"] = med
        out["pauses"] = [dict(after=int(i + 1), minutes=round(float(d[i]) / 60, 1)) for i in gaps[:50]]
    return _clean(out)


def pick_stats(project: Project, job) -> dict:
    """A Pick job: per micrograph the candidates and picks (and its defocus), and the candidates' scores, kept and
    rejected (sampled), for the histograms."""
    j = project.job(job)
    tab = project.pick_table(j)
    mics = project.micrographs(j.params["micrographs"])[["movie_id", "defocus_u", "defocus_v", "ctf_fit_res"]]
    tab = tab.merge(mics, on="movie_id", how="left")
    rows = [dict(i=i + 1, stem=r.stem, candidates=int(r.n_candidates), picks=int(r.n_kept),
                 threshold=None if r.threshold is None or not np.isfinite(r.threshold) else round(float(r.threshold), 4),
                 defocus_a=None if not np.isfinite(r.defocus_u) else round(0.5 * (r.defocus_u + r.defocus_v)),
                 ctf_fit_a=None if not np.isfinite(r.ctf_fit_res) else round(float(r.ctf_fit_res), 2))
            for i, r in enumerate(tab.sort_values("created").itertuples())]
    out = dict(job=j.name, rows=rows, diameter_a=j.params.get("diameter_a"),
               method="templates" if j.params.get("templates") or j.params.get("template_file") else "blob")
    try:
        z = project.table(j, "picks").read(["score", "keep", "local_std"])
    except Exception:                                 # noqa: BLE001 - nothing picked yet
        z = None
    if z is not None and len(z["score"]):
        g = np.random.default_rng(0)
        idx = g.permutation(len(z["score"]))[:40000]
        out.update(score=np.round(z["score"][idx], 4).tolist(), keep=z["keep"][idx].astype(int).tolist(),
                   local_std=np.round(z["local_std"][idx], 3).tolist() if "local_std" in z else None,
                   candidates=int(len(z["score"])), kept=int(z["keep"].sum()))
    return _clean(out)


def extract_stats(project: Project, job) -> dict:
    """An Extract job: particles per micrograph, the box and pixel size, scores and defocus of the particles."""
    j = project.job(job)
    with project._db() as c:
        got = c.execute("SELECT v.stem, k.n, k.box, k.apix, k.created FROM chunks k JOIN movies v ON v.id = k.movie_id "
                        "WHERE k.job_id=? ORDER BY k.created", (j.id,)).fetchall()
    rows = [dict(i=i + 1, stem=s, particles=int(n)) for i, (s, n, _, _, _) in enumerate(got)]
    box = got[0][2] if got else j.params.get("box")
    apix = got[0][3] if got else j.params.get("out_apix")
    out = dict(job=j.name, rows=rows, box=box, apix=apix, box_a=None if not box or not apix else round(box * apix, 1),
               particles=int(sum(r["particles"] for r in rows)), dtype=j.params.get("dtype"),
               particle_a=j.params.get("particle_a"), recentred=bool(j.params.get("recenter")),
               select=j.params.get("select") or None)
    try:
        t = project.table(j, "particles")
        cols = [k for k in ("score", "defocus_u", "defocus_v", "dx_a", "dy_a") if k in t.columns]
        z = t.read(cols) if cols else {}
    except Exception:                                 # noqa: BLE001 - nothing extracted yet
        z = {}
    if z and len(next(iter(z.values()))):
        g = np.random.default_rng(0)
        n = len(next(iter(z.values())))
        idx = g.permutation(n)[:40000]
        if "score" in z:
            out["score"] = np.round(z["score"][idx], 4).tolist()
        if "defocus_u" in z:
            out["defocus_a"] = np.round(0.5 * (z["defocus_u"][idx] + z["defocus_v"][idx])).tolist()
        if "dx_a" in z:                               # re-centred: how far each particle moved from its pick
            out["shift_a"] = np.round(np.hypot(z["dx_a"][idx], z["dy_a"][idx]), 1).tolist()
    return _clean(out)


def particle_gallery(project: Project, job, n: int = 60, seed: int = 0, lowpass_a: float = 0.0) -> dict:
    """A random handful of an Extract job's particles as small PNGs (each normalised; low-passed to lowpass_a when
    given), with the micrograph and score of each - to see what was boxed."""
    import mrcfile
    j = project.job(job)
    df = project.particles(j, ["uid", "slice", "score"] if "score" in project.table(j, "particles").columns else ["uid", "slice"])
    if not len(df):
        return dict(job=j.name, tiles=[])
    g = np.random.default_rng(seed)
    pick = df.iloc[np.sort(g.permutation(len(df))[:n])]
    tiles = []
    from PIL import Image
    for stack, grp in pick.groupby("stack"):
        with mrcfile.mmap(project.abs(stack), permissive=True, mode="r") as m:
            data = m.data
            for r in grp.itertuples():
                a = np.asarray(data[int(r.slice)] if data.ndim == 3 else data, np.float32)
                if lowpass_a and r.apix:
                    from scipy import ndimage
                    a = ndimage.gaussian_filter(a, 0.1874 * lowpass_a / float(r.apix))
                lo, hi = np.percentile(a, [1, 99])
                im = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))
                if im.width < 96:
                    im = im.resize((96, 96), Image.BICUBIC)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                tiles.append(dict(src="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                                  stem=r.stem, uid=int(r.uid), score=None if not hasattr(r, "score") else float(r.score),
                                  defocus_a=round(0.5 * (float(r.defocus_u) + float(r.defocus_v)))))
    return _clean(dict(job=j.name, tiles=tiles, box=int(pick["box"].iloc[0]), apix=float(pick["apix"].iloc[0]),
                       total=int(len(df))))
