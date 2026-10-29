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
    draw_figure(project, job, movie_id)   control figure of one micrograph (made again: 2 s), on demand
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
import torch

from . import picking as pk
from . import preprocess as pp
from .engine import Optics
from .project import Job, Project, particle_uids


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
    dtype: str = "float16"              # micrographs on disk: float16 (half the space) or float32
    figures: str = "outliers"           # control figures: all | outliers (+ the first micrograph) | none


@dataclass
class PickSettings:
    diameter_a: float                   # blob (the particle's short dimension)
    dark: bool = True                   # particles darker than the background (cryo); False for negative stain
    min_dist_frac: float = 0.8
    threshold: float | None = None      # None: automatic per micrograph (generous; 2D cleans up)
    max_std: float = 2.0                # contamination filters (picking.select)
    max_mean: float = 3.0
    figures: str = "outliers"           # all | outliers (+ the first micrograph) | none


@dataclass
class ExtractSettings:
    particle_a: float                   # longest particle dimension (A)
    box_a: float = 0.0                  # box edge in A; 0: particle + CTF delocalisation at 2 um defocus and box_res_a
    box_res_a: float = 5.0              # (picking.box_for; T20S, 160 A: 318 A -> 250 px at 1.315 A)
    particle_apix: float = 0.0          # Fourier-bin particles to this pixel size (0: the micrograph's)
    dtype: str = "float16"
    dark: bool = True
    select: int = 0                     # a Select job: extract only its particles (by uid)
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
    label = label or (f"{box * out_apix:.0f}A_{out_apix:.3g}apx_{'f16' if s.dtype == 'float16' else 'f32'}"
                      + (f"_sel{s.select:03d}" if s.select else ""))
    return project.new_job("Extract", dict(asdict(s), picks=picks.id, micrographs=mic_job.id, box=box,
                                           out_apix=out_apix, box_a_used=round(box * out_apix, 1)),
                           [picks] + ([s.select] if s.select else []), label)


# ------------------------------------------------------------------------------- one movie of one job
_gain_cache: dict = {}
_mem: dict = {}           # movie_id -> (micrograph on the GPU, apix, CTF): stages of one movie in one run
_mem_picks: dict = {}     # (pick job id, movie_id) -> picks
_sel_cache: dict = {}     # Select job id -> uids


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


def _motion(project: Project, job: Job, mv: dict, frames=None, log=print):
    s = MotionSettings(**{k: v for k, v in job.params.items() if k in MotionSettings.__dataclass_fields__})
    t = {}
    t0 = time.time()
    fr = pp.read_movie(mv["path"]) if frames is None else frames
    t["read"] = time.time() - t0
    gain = None
    if s.gain:
        if s.gain not in _gain_cache:
            _gain_cache[s.gain] = pp.orient_gain(pp.read_gain(s.gain), fr, log=log, return_name=True)
            project.update_params(job, gain_orientation=_gain_cache[s.gain][1])
        gain = _gain_cache[s.gain][0]
    t0 = time.time()
    mic = pp.motion_correct(fr, gain, s.apix, s.dose_total / len(fr), bin=s.bin, patches=s.patches)
    optics = Optics(s.kv, s.cs_mm, s.amp_contrast)
    fit = pp.estimate_ctf(mic.sum, mic.apix, optics)
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
    if s.figures == "all":
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
    c = pk.blob_candidates(img, apix, s.diameter_a, ctf, optics_of(project, mic_job), s.dark, s.min_dist_frac,
                           keep_maps=s.figures == "all")
    th = pk.auto_threshold(c.score) if s.threshold is None else s.threshold
    keep = np.zeros(len(c.score), bool)
    keep[pk.select(c, th, s.max_std, s.max_mean)] = True
    cols = dict(xy_a=c.xy_a, score=c.score, local_std=c.local_std, local_mean=c.local_mean)
    _mem_picks.clear()
    _mem_picks[(job.id, movie_id)] = dict(cols, keep=keep)
    project.add_picks(job, movie_id, cols, keep, th)
    if s.figures == "all":
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
    z = _mem_picks.get((pick_job_.id, movie_id)) or project.picks(pick_job_, movie_id)
    k = z["keep"].copy()
    if p.get("select"):                              # only the selected particles (by uid)
        if p["select"] not in _sel_cache:
            _sel_cache.clear()
            _sel_cache[p["select"]] = project.selected_uids(p["select"])
        uid = particle_uids(project.movie(movie_id)["fingerprint"], z["xy_a"])
        k &= np.isin(uid, _sel_cache[p["select"]])
    out_apix = p["out_apix"]
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
    kept = 0
    for r in project.pick_table(picks).itertuples():
        z = project.picks(picks, r.movie_id)
        c = pk.Candidates(z["xy_a"], z["score"], z["local_std"], z["local_mean"])
        th = pk.auto_threshold(c.score) if threshold is None else threshold
        keep = np.zeros(len(c.score), bool)
        keep[pk.select(c, th, max_std, max_mean)] = True
        project.add_picks(new, r.movie_id, {k: z[k] for k in ("xy_a", "score", "local_std", "local_mean")}, keep, th)
        kept += int(keep.sum())
    finish(project, new)
    return new


def finish(project: Project, job: Job, status: str = "finished") -> dict:
    """Consolidated table, summary.json, exports the job asked for; status."""
    if job.type in ("MotionCtf", "Pick", "Extract"):
        project.consolidate(job)
    if job.type in ("MotionCtf", "Pick") and job.params.get("figures") == "outliers" and status == "finished":
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
        if job.params.get("export"):
            from .export import export
            for fmt in job.params["export"]:
                summ[f"export_{fmt}"] = project.rel(export(project, job, fmt))
    else:
        summ = {}
    project.set_status(job, status, summ)
    return summ


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
    pool = ProcessPoolExecutor(2, mp_context=mp.get_context("spawn"))

    def reader():                                    # keeps two movies decoding ahead, in order
        futs = {}
        for n, i in enumerate(todo):
            for k in todo[n:n + 2]:
                if k not in futs and mc is not None and not project.done(mc, k):
                    futs[k] = pool.submit(_decode_shm, paths[k])
            if stop is not None and stop.is_set():
                break
            if i in futs:
                name, shape, dt = futs.pop(i).result()
                shm = shared_memory.SharedMemory(name=name)
                q.put((i, (shm, np.ndarray(shape, np.dtype(dt), buffer=shm.buf))))
            else:
                q.put((i, None))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    t0, n = time.time(), 0
    tm = dict(wait_for_reader=0.0, motion_ctf=0.0, pick=0.0, extract=0.0)

    def lap(k, t):
        tm[k] += time.time() - t
        return time.time()

    try:
        with ThreadPoolExecutor(1) as figs, ThreadPoolExecutor(1) as io:     # one writer: catalog order kept
            while True:
                t = time.time()
                item = q.get()
                t = lap("wait_for_reader", t)
                if item is None:
                    break
                i, sf = item
                if mc is not None:
                    process_movie_motion(project, mc, i, frames=None if sf is None else sf[1], log=log,
                                         figures=figs, io=io)
                    t = lap("motion_ctf", t)
                if sf is not None:                       # frames are on the GPU by now: free the shared block
                    shm = sf[0]
                    del sf, item
                    shm.close()
                    shm.unlink()
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
        tm["writes_after_last"] = time.time() - t
    finally:
        _mem.clear()
        _mem_picks.clear()
        _sel_cache.clear()
        pool.shutdown()
    if finish_jobs:
        for j in jobs:
            finish(project, j)
    return dict(processed=n, of=len(todo), seconds=round(time.time() - t0, 1),
                per_movie_s=round((time.time() - t0) / max(n, 1), 2),
                stage_s_per_movie={k: round(v / max(n, 1), 2) for k, v in tm.items()})


def draw_figure(project: Project, job, movie_id: int) -> str:
    """Control figure of one micrograph of a MotionCtf or Pick job (jobs/<job>/figures/<stem>.png), drawn now if
    it is not there: motion + CTF are computed again from the movie (~2 s), picks from the stored micrograph."""
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
    job = project.job(job)
    if job.type not in ("MotionCtf", "Extract"):
        raise ValueError(f"nothing to rebuild for {job.type} jobs")
    project.forget(job)
    project.update_params(job, purged=False)
    return run(project, [job], log=log)
