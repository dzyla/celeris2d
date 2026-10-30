"""Processing jobs from movies to particles (docs/project-design.md). Each job type is independent: a job reads
its inputs from the jobs named in its parameters, writes into its own folder and skips movies it already has, so
jobs can run again with other settings (a new job each time), continue as movies arrive, or resume after a crash.

    imp = project.import_movies(paths)                        jobs/001_import_<movie folder>
    mc = motion_job(project, imp, MotionSettings(...))        jobs/002_motionctf_1.32apx
    pk = pick_job(project, mc, PickSettings(...))             jobs/003_pick_blob120
    ex = extract_job(project, pk, ExtractSettings(...))       jobs/004_extract_336A_1.32apx_f16
    run(project, [mc, pk, ex])            all three over the movies, overlapped (the live / fast path)
    sel = project.select(ex, good_uids)                       jobs/006_select_...
    extract_job(project, pk, ExtractSettings(..., select=sel.id))   only the selected particles, another box
    draw_figure(project, job, movie_id)   control figure of one micrograph (made again: ~3 s), on demand
    purge(job) / rebuild(project, job)    drop / remake a job's micrographs or particle stacks

The fast path keeps the current micrograph and picks in GPU memory between jobs, decodes the next movies in
worker processes into shared memory and writes files / draws figures on background threads.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from . import picking as pk
from . import preprocess as pp
from .engine import Optics
from .project import PARTICLE_JOBS, Job, Project, particle_uids


@dataclass
class MotionSettings:
    apix: float                         # movie pixel size (A), super-resolution if bin >= 2
    dose_total: float                   # e/A^2 over all frames
    bin: float = 1.0                    # Fourier binning of the frames (2: super-resolution -> physical pixels)
    gain: str = ""
    kv: float = 300.0
    cs_mm: float = 2.7
    amp_contrast: float = 0.1
    patches: int = 5                    # local motion grid (0: whole-frame only)
    flip_y: bool = False                # mirror the frames in y after reading. RELION reads TIFF movies y-flipped
                                        # relative to the file's row order: on the RELION 5 tutorial movies its
                                        # micrographs are the y-mirror of ours (cc 0.32 vs -0.01), with the gain
                                        # applied as is. True gives RELION's micrographs, pick coordinates,
                                        # astigmatism angles and map hand; any gain is then oriented to the flipped frames
    dtype: str = "float16"              # micrographs on disk: float16 (half the space) or float32
    figures: str | int = "outliers"     # control figures: all | outliers (+ the first micrograph) | none | N (the
                                        # first N micrographs, drawn as they are processed, + the outliers at the end)


@dataclass
class PickSettings:
    diameter_a: float                   # blob (the particle's short dimension)
    dark: bool = True                   # particles darker than the background (cryo); False for negative stain
    min_dist_frac: float = 0.5           # x diameter: T20S recall 90.8% (0.4: 97.2% at 1.6x the picks; 0.6: 76%)
    threshold: float | None = None      # None: automatic per micrograph (generous; 2D cleans up)
    max_std: float = 2.0                # contamination filters (picking.select)
    max_mean: float = 3.0
    figures: str | int = "outliers"     # all | outliers (+ the first micrograph) | none | N (first N live + outliers)


@dataclass
class ExtractSettings:
    particle_a: float                   # longest particle dimension (A)
    box_a: float = 0.0                  # box edge in A; 0: particle + CTF delocalisation at 2 um defocus and box_res_a
    box_res_a: float = 5.0              # (picking.box_for; T20S, 160 A: 318 A -> 250 px at 1.315 A)
    particle_apix: float = 0.0          # Fourier-bin particles to this pixel size (0: the micrograph's)
    dtype: str = "float16"
    dark: bool = True
    select: int = 0                     # a Select job: extract only its particles (by uid)
    recenter: int = 0                   # a Class2D / InitialModel / Refine3D job: extract its particles, each centred
                                        # at its aligned centre (centre in that job's extraction + its shift)
    dedup_a: float = -1.0               # with recenter: of particles whose new centres are closer than this (A) only
                                        # the best-scoring is kept (-1: half the pick diameter, the particle's short dimension; 0: keep all)
    export: tuple = ()                  # ("relion", "cryosparc"): write particles.star / particles.cs at the end


def optics_of(project: Project, job: Job) -> Optics:
    """Optics of the MotionCtf job a job descends from."""
    j = project.ancestor(job, "MotionCtf")
    if j is None:
        raise ValueError(f"{job.name} has no MotionCtf ancestor")
    return Optics(j.params["kv"], j.params["cs_mm"], j.params["amp_contrast"])


# ------------------------------------------------------------------------------------- creating jobs
def motion_job(project: Project, movies: Job, s: MotionSettings, label: str = "") -> Job:
    return project.new_job("MotionCtf", dict(asdict(s), movies=movies.id), [movies],
                           label or f"{s.apix * s.bin:.3g}apx" + ("" if s.patches else "_global"))


def pick_job(project: Project, micrographs: Job, s: PickSettings, label: str = "") -> Job:
    return project.new_job("Pick", dict(asdict(s), micrographs=micrographs.id), [micrographs],
                           label or f"blob{s.diameter_a:g}" + ("" if s.threshold is None else f"_th{s.threshold:g}"))


def extract_job(project: Project, picks: Job, s: ExtractSettings, label: str = "") -> Job:
    """The box (px) is fixed when the job is made, so every micrograph of the job has the same one. Label by
    default: box (A), pixel size, precision, and the selection it extracts."""
    mic_job = project.job(picks.params["micrographs"])
    m = mic_job.params
    mic_apix = m["apix"] * m["bin"]
    out_apix = max(s.particle_apix, mic_apix) if s.particle_apix else mic_apix
    box_a = s.box_a or pk.box_for(s.particle_a, 20000.0, max(s.box_res_a, 2 * out_apix),
                                  optics_of(project, mic_job).wavelength)
    box = pk.suggest_box(box_a, out_apix)
    out_apix = pk.binned_apix(box, mic_apix, out_apix)          # exact (the source box is an even number of pixels)
    label = label or (f"{box * out_apix:.0f}A_{out_apix:.3g}apx_{'f16' if s.dtype == 'float16' else 'f32'}"
                      + (f"_sel{s.select:03d}" if s.select else "") + (f"_rc{s.recenter:03d}" if s.recenter else ""))
    return project.new_job("Extract", dict(asdict(s), picks=picks.id, micrographs=mic_job.id, box=box,
                                           out_apix=out_apix, box_a_used=round(box * out_apix, 1)),
                           [picks] + ([s.select] if s.select else []) + ([s.recenter] if s.recenter else []), label)


def _recentred(project: Project, job: Job) -> dict:
    """movie id -> (uid, new centre (n, 2), pick position (n, 2), score) of a re-centring extraction: the
    alignment job's particles (used ones), centre = centre in the extraction they were aligned on + shift."""
    rj = project.job(job.params["recenter"])
    al = project.table(rj, "particles").read()
    if "used" in al:
        al = {k: v[al["used"]] for k, v in al.items()}
    ex0 = project.aligned_particles(rj)
    e = project.table(ex0, "particles").read()
    o = np.argsort(e["uid"])
    pos = np.searchsorted(e["uid"][o], al["uid"])
    ok = pos < len(o)
    ok[ok] = e["uid"][o[pos[ok]]] == al["uid"][ok]
    r = o[pos[ok]]
    cx, cy = e["x_a"][r].astype(np.float64), e["y_a"][r].astype(np.float64)
    px = cx - (e["dx_a"][r] if "dx_a" in e else 0)
    py = cy - (e["dy_a"][r] if "dy_a" in e else 0)
    centre = np.stack([cx + al["shift_x_a"][ok], cy + al["shift_y_a"][ok]], 1)
    score = al["maxprob"][ok] if "maxprob" in al else (al["fit"][ok] if "fit" in al else e["score"][r])
    mv, uid = e["movie"][r], al["uid"][ok]
    out = {}
    for m in np.unique(mv):
        i = np.flatnonzero(mv == m)
        out[int(m)] = (uid[i], centre[i], np.stack([px[i], py[i]], 1), score[i])
    return out


def _dedup(xy: np.ndarray, score: np.ndarray, dist: float) -> np.ndarray:
    return pk.dedup(xy, score, dist)


# ------------------------------------------------------------------------------- one movie of one job
_gain_cache: dict = {}
_mem: dict = {}           # movie_id -> (micrograph on the GPU, apix, CTF): stages of one movie in one run
_mem_picks: dict = {}     # (pick job id, movie_id) -> picks
_n_live: dict = {}        # job id -> figures drawn while processing (figures = N)


def _live_figure(job: Job, figures) -> bool:
    """Draw this micrograph's figure now? figures = "all", or a number N while fewer than N have been drawn (the
    count starts from the figures already in the job's folder, so a resumed job does not draw N more)."""
    if figures == "all":
        return True
    try:
        n = int(figures)
    except (TypeError, ValueError):
        return False
    if job.id not in _n_live:
        d = job.file("figures")
        _n_live[job.id] = sum(f.endswith(".png") and not f.endswith(".light.png")
                              for f in (os.listdir(d) if os.path.isdir(d) else []))
    if _n_live[job.id] >= n:
        return False
    _n_live[job.id] += 1
    return True
_sel_cache: dict = {}     # Select job id -> uids
_rc_cache: dict = {}      # re-centring Extract job id -> per-movie particles


class _Writer(ThreadPoolExecutor):
    """The I/O thread: one worker (catalog order kept); check() raises the first failed write."""
    def __init__(self):
        super().__init__(1)
        self.futs = []

    def submit(self, *a, **k):
        f = super().submit(*a, **k)
        self.futs.append(f)
        return f

    def check(self):
        done = [f for f in self.futs if f.done()]
        self.futs = [f for f in self.futs if not f.done()]
        for f in done:
            if f.exception() is not None:
                raise RuntimeError("writing results failed") from f.exception()


def _release(item):
    """Free a shared-memory block, or the frames of a reader queue item (i, (shm, frames)): unlinked first (the
    /dev/shm name goes now), then closed if no array still views it (else when the last view is collected)."""
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
        item = item[1][0]
    if item is None or isinstance(item, (BaseException, tuple)):
        return
    for f in (item.unlink, item.close):
        try:
            f()
        except (FileNotFoundError, BufferError):
            pass


def _release_name(name: str):
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _io(executor, fn, *args, **kw):
    """Run a file write (+ its catalog entry) on the I/O thread if there is one: the GPU loop does not wait."""
    return executor.submit(fn, *args, **kw) if executor else fn(*args, **kw)


def _load_mic(project: Project, mic_job: Job, movie_id: int):
    import mrcfile
    if movie_id in _mem:
        return _mem[movie_id]
    m = project.micrographs(mic_job).set_index("movie_id").loc[movie_id]
    with mrcfile.open(project.abs(m["path"]), permissive=True) as f:
        img = torch.as_tensor(np.asarray(f.data, dtype=np.float32), device="cuda")
    ctf = pp.CTFFit(m["defocus_u"], m["defocus_v"], m["defocus_angle"], m["ctf_fit_res"], 0.0, {})
    return img, float(m["apix"]), ctf


def _cpu(mic):
    """The micrograph fields a figure needs, off the GPU (figures are drawn on another thread)."""
    import copy
    m = copy.copy(mic)
    m.sum, m.sum_dw = mic.sum.cpu(), mic.sum_dw.cpu()
    m.halves = ()
    return m


def _motion(project: Project, job: Job, mv: dict, frames=None, log=print, keep_frames: bool = False, ctf: bool = True):
    s = MotionSettings(**{k: v for k, v in job.params.items() if k in MotionSettings.__dataclass_fields__})
    t = {}
    t0 = time.time()
    fr = pp.read_movie(mv["path"]) if frames is None else frames
    if s.flip_y:
        fr = np.ascontiguousarray(fr[:, ::-1])
    t["read"] = time.time() - t0
    gain = None
    if s.gain:
        if s.gain not in _gain_cache:
            _gain_cache[s.gain] = pp.orient_gain(pp.read_gain(s.gain), fr, log=log, return_name=True)
            project.update_params(job, gain_orientation=_gain_cache[s.gain][1])
        gain = _gain_cache[s.gain][0]
    t0 = time.time()
    mic = pp.motion_correct(fr, gain, s.apix, s.dose_total / len(fr), bin=s.bin, patches=s.patches,
                            keep_frames=keep_frames, kv=s.kv)
    optics = Optics(s.kv, s.cs_mm, s.amp_contrast)
    fit = pp.estimate_ctf(mic.sum, mic.apix, optics) if ctf else None
    torch.cuda.synchronize()
    t["motion_ctf"] = time.time() - t0
    return s, fr, mic, fit, t


def process_movie_motion(project: Project, job: Job, movie_id: int, frames: np.ndarray | None = None, log=print,
                         figures: ThreadPoolExecutor | None = None, io: ThreadPoolExecutor | None = None):
    """Gain, motion correction, dose weighting and CTF of one movie into a MotionCtf job."""
    if project.done(job, movie_id):
        return None
    mv = project.movie(movie_id)
    s, fr, mic, fit, t = _motion(project, job, mv, frames, log)
    sh = mic.shifts
    stats = dict(frames=len(fr), motion_path_a=round(float(np.linalg.norm(np.diff(sh, axis=0), axis=1).sum()), 2),
                 local_rms_a=round(mic.local["rms_a"], 3) if mic.local else None, ctf_score=round(fit.score, 4),
                 seconds={k: round(v, 2) for k, v in t.items()})
    _mem.clear()
    _mem[movie_id] = (mic.sum_dw, mic.apix, fit)
    data, shape = mic.sum_dw.cpu().numpy(), tuple(mic.sum_dw.shape)
    mpath = job.per_movie("Micrographs", mv, ".mrc")

    def write():                                     # files first, then the catalog entry that says they exist
        project.write_mrc(mpath, data, mic.apix, s.dtype)
        st = dict(stats, movie=mv["path"], micrograph=project.rel(mpath), apix=mic.apix, shape=shape,
                  defocus_u=fit.defocus_u, defocus_v=fit.defocus_v, angle=fit.angle, fit_res_a=fit.fit_res_a,
                  shifts=sh.tolist(), local=None if not mic.local else
                  dict(centres=mic.local["centres"].tolist(), rms_a=mic.local["rms_a"]))
        json.dump(st, open(job.per_movie("Stats", mv, ".json"), "w"))
        project.add_micrograph(job, movie_id, mpath, mic.apix, shape, fit, meta=stats)
    _io(io, write)
    if _live_figure(job, s.figures):
        from .plots_preprocess import movie_figure
        args = (job.figure(mv), _cpu(mic), fit)
        kw = dict(title=mv["stem"], frc_eo=pp.frc(*mic.halves, mic.apix, n_shell=120))
        figures.submit(movie_figure, *args, **kw) if figures else movie_figure(*args, **kw)
    log(f"[{job.name}] {mv['stem']}: drift {stats['motion_path_a']:.0f} A, defocus "
        f"{(fit.defocus_u + fit.defocus_v) / 2:.0f} A, CTF fit {fit.fit_res_a:.1f} A; "
        + ", ".join(f"{k} {v:.1f} s" for k, v in t.items()))
    return stats


def process_movie_pick(project: Project, job: Job, movie_id: int, log=print,
                       figures: ThreadPoolExecutor | None = None):
    if project.done(job, movie_id):
        return None
    s = PickSettings(**{k: v for k, v in job.params.items() if k in PickSettings.__dataclass_fields__})
    mic_job = project.job(job.params["micrographs"])
    img, apix, ctf = _load_mic(project, mic_job, movie_id)
    live = _live_figure(job, s.figures)
    c = pk.blob_candidates(img, apix, s.diameter_a, ctf, optics_of(project, mic_job), s.dark, s.min_dist_frac,
                           keep_maps=live)
    th = pk.auto_threshold(c.score) if s.threshold is None else s.threshold
    keep = np.zeros(len(c.score), bool)
    keep[pk.select(c, th, s.max_std, s.max_mean)] = True
    cols = dict(xy_a=c.xy_a, score=c.score, local_std=c.local_std, local_mean=c.local_mean)
    _mem_picks.clear()
    _mem_picks[(job.id, movie_id)] = dict(cols, keep=keep)
    project.add_picks(job, movie_id, cols, keep, th)
    if live:
        from .plots_preprocess import picks_figure
        mv = project.movie(movie_id)
        args = (job.figure(mv), c, np.flatnonzero(keep), s.diameter_a, th)
        figures.submit(picks_figure, *args, title=mv["stem"]) if figures else picks_figure(*args, title=mv["stem"])
    return dict(candidates=len(keep), kept=int(keep.sum()), threshold=th)


def process_movie_extract(project: Project, job: Job, movie_id: int, io: ThreadPoolExecutor | None = None):
    if project.done(job, movie_id):
        return None
    p = job.params
    pick_job_ = project.job(p["picks"])
    img, apix, ctf = _load_mic(project, project.job(p["micrographs"]), movie_id)
    out_apix = p["out_apix"]
    if p.get("recenter"):                            # the aligned particles at their aligned centres
        if job.id not in _rc_cache:
            _rc_cache.clear()
            _rc_cache[job.id] = _recentred(project, job)
        if movie_id not in _rc_cache[job.id]:
            imgs = np.zeros((0, p["box"], p["box"]), np.float32)
            _io(io, project.add_particles, job, movie_id, np.zeros((0, 2)), imgs, out_apix, dtype=p["dtype"],
                uid=np.zeros(0, np.int64), pick_xy_a=np.zeros((0, 2)), columns=dict(score=np.zeros(0)))
            return dict(particles=0)
        uid, xy, pick, score = _rc_cache[job.id][movie_id]
        d = p.get("dedup_a", -1.0)
        k = _dedup(xy, score, pick_job_.params["diameter_a"] / 2 if d < 0 else d)
        ny, nx = img.shape
        k &= (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < nx * apix) & (xy[:, 1] < ny * apix)
        imgs = pk.extract(img, apix, xy[k], p["box"], out_apix=out_apix if out_apix > apix * 1.001 else None,
                          dark=p["dark"], diameter_a=pick_job_.params["diameter_a"])
        _io(io, project.add_particles, job, movie_id, xy[k], imgs, out_apix, columns=dict(score=score[k]),
            dtype=p["dtype"], uid=uid[k], pick_xy_a=pick[k])
        return dict(particles=int(k.sum()), duplicates=int((~k).sum()))
    z = _mem_picks.get((pick_job_.id, movie_id)) or project.picks(pick_job_, movie_id)
    k = z["keep"].copy()
    if p.get("select"):                              # only the selected particles (by uid)
        if p["select"] not in _sel_cache:
            _sel_cache.clear()
            _sel_cache[p["select"]] = project.selected_uids(p["select"])
        uid = particle_uids(project.movie(movie_id)["fingerprint"], z["xy_a"])
        k &= np.isin(uid, _sel_cache[p["select"]])
    imgs = pk.extract(img, apix, z["xy_a"][k], p["box"], out_apix=out_apix if out_apix > apix * 1.001 else None,
                      dark=p["dark"], diameter_a=pick_job_.params["diameter_a"])
    _io(io, project.add_particles, job, movie_id, z["xy_a"][k], imgs, out_apix,
        columns=dict(score=z["score"][k], local_std=z["local_std"][k]), dtype=p["dtype"])
    return dict(particles=int(k.sum()))


# ------------------------------------------------------------------------------------------ jobs over movies
def rethreshold(project: Project, picks: Job, threshold: float | None = None, max_std: float = 2.0,
                max_mean: float = 3.0, label: str = "") -> Job:
    """A new Pick job with other thresholds on the same candidates (no picking)."""
    lab = label or (f"blob{picks.params['diameter_a']:g}_th" + ("auto" if threshold is None else f"{threshold:g}")
                    + (f"_std{max_std:g}" if max_std != 2.0 else "") + (f"_mean{max_mean:g}" if max_mean != 3.0 else ""))
    new = project.new_job("Pick", dict(picks.params, threshold=threshold, max_std=max_std, max_mean=max_mean,
                                       rethreshold_of=picks.id, figures="none"), [picks], lab)
    z = project.table(picks, "picks").read()         # all candidates at once (rows grouped by micrograph)
    keep = np.zeros(len(z["score"]), bool)
    ids, start = np.unique(z["movie"], return_index=True)
    order = np.argsort(start)
    ids, start = ids[order], start[order]
    end = np.append(start[1:], len(keep))
    rows = []
    for m, a, b in zip(ids, start, end):
        c = pk.Candidates(np.stack([z["x_a"][a:b], z["y_a"][a:b]], 1), z["score"][a:b], z["local_std"][a:b],
                          z["local_mean"][a:b])
        th = pk.auto_threshold(c.score) if threshold is None else threshold
        keep[a + pk.select(c, th, max_std, max_mean)] = True
        rows.append((new.id, int(m), "picks", int(b - a), int(keep[a:b].sum()), float(th), time.time()))
    project.table(new, "picks").write({k: (keep if k == "keep" else v) for k, v in z.items() if k != "movie"},
                                      z["movie"])
    with project._db() as c:
        c.executemany("INSERT OR REPLACE INTO picks VALUES (?,?,?,?,?,?,?)", rows)
    finish(project, new)
    return new


def finish(project: Project, job: Job, status: str = "finished") -> dict:
    """Consolidated table, summary.json, exports the job asked for; status."""
    if status == "finished" and job.params.get("purged"):
        if all(project.done(job, int(m)) for m in _job_movies(project, job)):
            project.update_params(job, purged=False)
    if job.type in ("MotionCtf", "Pick", "Extract"):
        project.consolidate(job)
    fg = job.params.get("figures")
    if job.type in ("MotionCtf", "Pick") and fg not in ("all", "none") and status == "finished":  # outliers / N
        mc = project.ancestor(job, "MotionCtf")
        st = project.micrograph_stats(mc)
        stems = {o for v in st.values() if isinstance(v, dict) for o in v.get("outliers", [])}
        mi = project.micrographs(mc)
        ids = mi["movie_id"][mi["stem"].isin(stems)].tolist()[:10]
        for i in ([int(mi["movie_id"].iloc[0])] if len(mi) else []) + ids:
            try:
                draw_figure(project, job, i)
            except Exception as e:                    # a figure never fails a job
                print(f"[{job.name}] figure for movie {i}: {e!r}")
    if job.type == "MotionCtf":
        summ = project.micrograph_stats(job)
    elif job.type == "Pick":
        t = project.pick_table(job)
        summ = dict(micrographs=len(t), candidates=int(t["n_candidates"].sum()), picks=int(t["n_kept"].sum()),
                    picks_per_micrograph=round(float(t["n_kept"].mean()), 1) if len(t) else 0)
    elif job.type == "Extract":
        with project._db() as c:
            n, k = c.execute("SELECT SUM(n), COUNT(*) FROM chunks WHERE job_id=?", (job.id,)).fetchone()
        p = job.params
        summ = dict(micrographs=k, particles=n, box=p["box"], apix=round(p["out_apix"], 4),
                    box_a=p["box_a_used"], dtype=p["dtype"])
        if p.get("select"):
            summ["selection"] = project.job(p["select"]).name
        if p.get("recenter") and status == "finished":
            try:
                summ.update(recenter_report(project, job))
            except Exception as e:                    # a figure never fails a job
                print(f"[{job.name}] re-centring report: {e!r}")
        if job.params.get("export"):
            from .export import export
            for fmt in job.params["export"]:
                summ[f"export_{fmt}"] = project.rel(export(project, job, fmt))
    else:
        summ = {}
    if status == "finished" and job.type in ("MotionCtf", "Pick", "Extract"):
        try:
            overview(project, job)
        except Exception as e:                        # a figure never fails a job
            print(f"[{job.name}] overview figure: {e!r}")
    project.set_status(job, status, summ)
    return summ


def recenter_report(project: Project, job) -> dict:
    """A re-centred extraction: how far each particle's box moved from its pick (dx_a, dy_a: aligned centre - pick).
    recenter_shifts.png (magnitudes, 2D distribution, median per micrograph) and the numbers for the summary. A
    systematic mean shift means the picker is off-centre for this particle; a long tail, picks on the particle's edge
    or on neighbours (duplicates merge)."""
    from .plots_preprocess import shifts_figure
    job = project.job(job)
    t = project.table(job, "particles").read(["dx_a", "dy_a"])
    d = np.stack([t["dx_a"], t["dy_a"]], 1).astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    stems = project.movies().set_index("id")["stem"]
    per = pd.DataFrame(dict(movie=t["movie"], r=r)).groupby("movie")["r"].median()
    out = dict(recenter_shift_median_a=round(float(np.median(r)), 2), recenter_shift_p95_a=round(float(np.percentile(r, 95)), 2),
               recenter_shift_mean_a=[round(float(x), 2) for x in d.mean(0)], recenter_shift_max_a=round(float(r.max()), 1))
    shifts_figure(os.path.join(job.path, "recenter_shifts.png"), d, per.to_numpy(),
                  f"{job.name} · {len(r):,} particles re-centred on {project.job(job.params['recenter']).name}",
                  project.job(job.params["picks"]).params.get("diameter_a"))
    pd.DataFrame(dict(stem=stems.loc[per.index].to_numpy(), median_shift_a=per.to_numpy())).to_csv(
        os.path.join(job.path, "recenter_shifts.csv"), index=False, float_format="%.3g")
    return out


def overview(project: Project, job) -> str:
    """Every micrograph of a MotionCtf / Pick / Extract job at a glance: overview.png (per-micrograph values in
    processing order) and the numbers as micrographs.csv. Redrawn whenever the job finishes (also after new movies)."""
    from .plots_preprocess import overview_figure
    job = project.job(job)
    if job.type == "MotionCtf":
        mi = project.micrographs(job)
        meta = [json.loads(m) if m else {} for m in mi["meta"]]
        df = pd.DataFrame(dict(stem=mi["stem"], defocus_a=(mi["defocus_u"] + mi["defocus_v"]) / 2,
                               astigmatism_a=(mi["defocus_u"] - mi["defocus_v"]).abs(), ctf_fit_a=mi["ctf_fit_res"],
                               drift_a=[m.get("motion_path_a", np.nan) for m in meta],
                               local_rms_a=[m.get("local_rms_a", np.nan) for m in meta],
                               ctf_score=[m.get("ctf_score", np.nan) for m in meta]))
        panels = [("Defocus", "Å", df.defocus_a), ("Astigmatism", "Å", df.astigmatism_a),
                  ("CTF fit resolution", "Å", df.ctf_fit_a), ("Total drift", "Å", df.drift_a),
                  ("Local motion (rms)", "Å", df.local_rms_a), ("CTF fit score", "", df.ctf_score)]
    elif job.type == "Pick":
        t = project.pick_table(job)
        df = pd.DataFrame(dict(stem=t["stem"], candidates=t["n_candidates"], picks=t["n_kept"], threshold=t["threshold"]))
        panels = [("Picks per micrograph", "picks", df.picks), ("Candidates per micrograph", "candidates",
                                                                  df.candidates), ("Score threshold", "", df.threshold)]
    else:
        with project._db() as c:
            ch = pd.read_sql("SELECT c.movie_id, c.n, v.stem FROM chunks c JOIN movies v ON v.id = c.movie_id "
                             "WHERE c.job_id=? ORDER BY c.movie_id", c, params=(job.id,))
        df = pd.DataFrame(dict(stem=ch["stem"], particles=ch["n"]))
        panels = [("Particles per micrograph", "particles", df.particles)]
    if not len(df):
        return ""
    df.to_csv(os.path.join(job.path, "micrographs.csv"), index=False, float_format="%.4g")
    path = os.path.join(job.path, "overview.png")
    overview_figure(path, panels, f"{job.type}: all micrographs", f"{job.name} · {len(df):,} micrographs")
    return path


def _decode_shm(path: str):
    """Decode a movie in a worker process straight into shared memory: (name, shape, dtype). TIFF decoding is
    Python-heavy (tifffile iterates over strips); in another process it does not compete with the GPU loop."""
    from multiprocessing import shared_memory
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            ser = tf.series[0]
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(ser.shape)) * np.dtype(ser.dtype).itemsize)
            out = np.ndarray(ser.shape, ser.dtype, buffer=shm.buf)
            ser.asarray(out=out, maxworkers=8)
    else:
        a = pp.read_movie(path)
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        out = np.ndarray(a.shape, a.dtype, buffer=shm.buf)
        out[...] = a
    res = (shm.name, out.shape, out.dtype.str)
    del out
    shm.close()
    return res


def _noop():
    return None


def _decode_pool(n: int):
    """Worker processes for movie decoding (spawn: no CUDA or thread state is inherited). Spawned workers would
    import the caller's main script again (running a script without a __main__ guard a second time, or failing
    for stdin / python -c); they only need this module, so the script is hidden from them while they start, and
    they are all started here."""
    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor
    main = sys.modules["__main__"]
    f = getattr(main, "__file__", None)
    hide = f is not None
    if hide:
        del main.__file__
    try:
        pool = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"))
        for fu in [pool.submit(_noop) for _ in range(n)]:
            fu.result()
    finally:
        if hide:
            main.__file__ = f
    return pool


def run(project: Project, jobs: list[Job], movie_ids=None, log=print, stop: threading.Event | None = None,
        progress=None, finish_jobs: bool = True) -> dict:
    """A chain of jobs (MotionCtf [-> Pick [-> Extract]]) over movies (default: all movies of the MotionCtf job's
    Import job), overlapped: worker processes decode the next movies while the GPU works on the current one, the
    micrograph and picks stay on the GPU between the jobs, files and figures are written on other threads."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import shared_memory
    by = {j.type: j for j in jobs}
    mc, pj, ej = by.get("MotionCtf"), by.get("Pick"), by.get("Extract")
    if movie_ids is None:                            # the movies of the Import job the chain descends from
        movie_ids = project.movies(project.ancestor(jobs[0], "Import"))["id"].tolist()
    todo = [i for i in movie_ids if not all(project.done(j, i) for j in jobs)]
    paths = project.movies().set_index("id")["path"]
    q: queue.Queue = queue.Queue(maxsize=2)
    pool = _decode_pool(2) if mc is not None else None
    halt = threading.Event()                         # the GPU loop has stopped (error): the reader stops too
    futs: dict = {}

    def put(x) -> bool:
        while not halt.is_set():
            try:
                q.put(x, timeout=0.5)
                return True
            except queue.Full:
                pass
        _release(x)
        return False

    def reader():                                    # keeps two movies decoding ahead, in order
        try:
            for n, i in enumerate(todo):
                for k in todo[n:n + 2]:
                    if k not in futs and mc is not None and not project.done(mc, k):
                        futs[k] = pool.submit(_decode_shm, paths[k])
                if (stop is not None and stop.is_set()) or halt.is_set():
                    break
                if i in futs:
                    name, shape, dt = futs.pop(i).result()
                    shm = shared_memory.SharedMemory(name=name)
                    if not put((i, (shm, np.ndarray(shape, np.dtype(dt), buffer=shm.buf)))):
                        return
                elif not put((i, None)):
                    return
            put(None)
        except BaseException as e:                   # the GPU loop raises it instead of waiting forever
            put(e)

    threading.Thread(target=reader, daemon=True).start()
    t0, n = time.time(), 0
    tm = dict(wait_for_reader=0.0, motion_ctf=0.0, pick=0.0, extract=0.0)

    def lap(k, t):
        tm[k] += time.time() - t
        return time.time()

    cur: list = [None]
    try:
        with ThreadPoolExecutor(1) as figs, _Writer() as io:     # one writer: catalog order kept
            while True:
                io.check()
                t = time.time()
                item = q.get()
                t = lap("wait_for_reader", t)
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise RuntimeError("reading movies failed") from item
                i, sf = item
                cur[0] = None if sf is None else sf[0]
                if mc is not None:
                    process_movie_motion(project, mc, i, frames=None if sf is None else sf[1], log=log,
                                         figures=figs, io=io)
                    t = lap("motion_ctf", t)
                del sf, item                             # frames are on the GPU by now: free the shared block
                _release(cur[0])
                cur[0] = None
                if pj is not None:
                    process_movie_pick(project, pj, i, log=log, figures=figs)
                    t = lap("pick", t)
                if ej is not None:
                    process_movie_extract(project, ej, i, io=io)
                    t = lap("extract", t)
                n += 1
                if progress:
                    progress(n, len(todo))
            t = time.time()
        io.check()                                       # a failed write (e.g. a full disk) fails the run
        tm["writes_after_last"] = time.time() - t
    finally:
        halt.set()
        _release(cur[0])
        while True:                                      # blocks the reader decoded but the loop did not use
            try:
                _release(q.get_nowait())
            except queue.Empty:
                break
        _mem.clear()
        _mem_picks.clear()
        _sel_cache.clear()
        _rc_cache.clear()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            for f in list(futs.values()):
                if f.done() and not f.cancelled() and f.exception() is None:
                    _release_name(f.result()[0])
    if finish_jobs:
        for j in jobs:
            finish(project, j)
    return dict(processed=n, of=len(todo), seconds=round(time.time() - t0, 1),
                per_movie_s=round((time.time() - t0) / max(n, 1), 2),
                stage_s_per_movie={k: round(v / max(n, 1), 2) for k, v in tm.items()})


def draw_figure(project: Project, job, movie_id: int) -> str:
    """Control figure of one micrograph of a MotionCtf or Pick job (jobs/<job>/figures/<stem>.png), drawn now if
    it is not there: motion + CTF are computed again from the movie (~3 s warm), picks from the stored micrograph."""
    job = project.job(job)
    mv = project.movie(movie_id)
    path = job.figure(mv)
    if os.path.exists(path):
        return path
    if job.type == "MotionCtf":
        from .plots_preprocess import movie_figure
        s, fr, mic, fit, t = _motion(project, job, mv, log=lambda *a: None)
        movie_figure(path, _cpu(mic), fit, title=mv["stem"], frc_eo=pp.frc(*mic.halves, mic.apix, n_shell=120))
    elif job.type == "Pick":
        from .plots_preprocess import picks_figure
        s = PickSettings(**{k: v for k, v in job.params.items() if k in PickSettings.__dataclass_fields__})
        mic_job = project.job(job.params["micrographs"])
        img, apix, ctf = _load_mic(project, mic_job, movie_id)
        c = pk.blob_candidates(img, apix, s.diameter_a, ctf, optics_of(project, mic_job), s.dark, s.min_dist_frac,
                               keep_maps=True)
        z = project.picks(job, movie_id)
        # the stored candidates are the same picking; their kept mask (another threshold after rethreshold)
        th = float(z["threshold"])
        keep = np.flatnonzero(pk.match(c.xy_a, z["xy_a"][z["keep"]], 1.0)[0])
        picks_figure(path, c, keep, s.diameter_a, th, title=mv["stem"])
    else:
        raise ValueError(f"no per-micrograph figures for {job.type} jobs")
    return path


def rebuild(project: Project, job, log=print) -> dict:
    """Make a purged job's micrographs / particle stacks again from its recorded parameters (its input jobs must
    still have their data: rebuild those first)."""
    job = check_rebuild(project, job)
    return run(project, [job], log=log)


def _job_movies(project: Project, job: Job) -> list[int]:
    from .project import PER_MOVIE
    with project._db() as c:
        return [r[0] for r in c.execute(f"SELECT movie_id FROM {PER_MOVIE[job.type]} WHERE job_id=?", (job.id,))]


def check_rebuild(project: Project, job) -> Job:
    """The job, if rebuild can make its files again: nothing is changed before the run, which writes each movie's
    files again and replaces its catalog row; a purged job's movie counts as done once its file exists, and the
    purged flag is cleared when every movie is (finish). A failed rebuild leaves the job purged and can be repeated."""
    from .project import REBUILDABLE
    job = project.job(job)
    if job.type not in REBUILDABLE:
        raise ValueError(f"nothing to rebuild for {job.type} jobs")
    if not job.params.get("purged"):
        raise ValueError(f"{job.name} was not purged")
    if project.is_running(job):
        raise ValueError(f"{job.name} is running")
    for i in job.inputs:
        if project.job(i).params.get("purged"):
            raise ValueError(f"{job.name} needs {project.job(i).name}, which is purged: rebuild that first")
    return job
