"""Particle-level jobs of a project (docs/project-design.md): 2D classification, class selection, ab initio,
3D refinement. Movie-level jobs (motion, picking, extraction) are in pipeline.py.

    c = class2d_job(project, extract, Class2DSettings(n_classes=50)); run_class2d(project, c)
    s = select_classes(project, c, [1, 4, 7])                   # the user's (or agent's) choice
    c2 = class2d_job(project, s, ...)                           # 2D again on the selection, and so on
    m = initial_model_job(project, c2, classes=[...], sym="C1"); run_initial_model(project, m)
    e = extract_job(project, picks, ExtractSettings(..., recenter=m.id))     # re-centred on the 3D shifts
    r = refine3d_job(project, e, poses=m, reference=m, Refine3DSettings(sym="D7")); run_refine3d(project, r)

Per-particle results are sharded tables (tables.py) keyed by uid, with shifts in A in the model convention of the
engine (the particle's centre is at +shift from the box centre: re-centring moves the extraction centre by
+shift; RELION origin = -shift): Class2D `particles` = uid, cls (0-based), psi, shift_x_a, shift_y_a, maxprob;
InitialModel / Refine3D `particles` = uid, rot, tilt, psi (RELION Euler, image = A @ volume), shift_x_a,
shift_y_a, used.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch

from .engine import release_cache
from .jobspec import Input, JobSpec, _n, params_of, settings_of, warn_mark
from .project import CLASS_JOBS, PARTICLE_JOBS, Job, Project, logged


# ------------------------------------------------------------------------------------------ particle sources
def particle_source(project: Project, ref):
    """The particles a job works on: an Extract job's, or a Select / Class2D job's (their uids, in the Extract job
    they descend from). Returns a source whose parts carry celUid / celMovie."""
    from .export import job_source
    job = project.job(ref)
    if job.type in PARTICLE_JOBS:
        src = job_source(job.path)
        src.path = f"{project.root}#{job.id:03d}"
        return src
    ex = project.ancestor(job, PARTICLE_JOBS)
    if job.type == "Select":
        uids = project.selected_uids(job)
    else:                   # (a SymExpand job: every copy of a particle is the same image)
        uids = np.unique(project.table(job, "particles").read(["uid"])["uid"])
    src = job_source(ex.path, uids=uids)
    src.path = f"{project.root}#{ex.id:03d}#{job.id:03d}"    # by number: a renamed job keeps its saved 2D results
    return src


def match_rows(uid_t, copy_t, uid_q, copy_q=None) -> np.ndarray:
    """Rows of a table (uid_t, copy_t) holding each queried particle (uid_q, copy_q); -1 where none. copy: the
    symmetry-expansion index of a Subtract (or Class3D) job's copies (None: not expanded). A table without copies
    holds the one value of every copy of its particles (matched on uid only); a query without copies takes the
    particle's lowest copy in the table (copy 0, unless the table was filtered, e.g. to a Class3D class)."""
    uid_t, uid_q = np.asarray(uid_t, np.int64), np.asarray(uid_q, np.int64)
    if copy_t is None or copy_q is None:
        o = np.argsort(uid_t, kind="stable") if copy_t is None else np.lexsort((np.asarray(copy_t, np.int64), uid_t))
        pos = np.searchsorted(uid_t[o], uid_q).clip(0, max(len(o) - 1, 0))
        if not len(o):
            return np.full(len(uid_q), -1, np.int64)
        return np.where(uid_t[o[pos]] == uid_q, o[pos], -1)
    import pandas as pd
    idx = pd.MultiIndex.from_arrays([uid_t, np.asarray(copy_t, np.int64)])
    return np.asarray(idx.get_indexer(pd.MultiIndex.from_arrays([uid_q, np.asarray(copy_q, np.int64)])), np.int64)


def _centres(project: Project, particle_job, uids: np.ndarray, copies: np.ndarray | None = None) -> np.ndarray:
    """Box centres (n, 2) in A of these particles in a particle job (by uid, and copy for a symmetry-expanded
    Subtract job; a Select job: its extraction's)."""
    pj = project.job(particle_job)
    if pj.type not in PARTICLE_JOBS:
        pj = project.ancestor(pj, PARTICLE_JOBS)
    tab = project.table(pj, "particles")
    t = tab.read(["uid", "x_a", "y_a"] + (["copy"] if "copy" in tab.columns else []))
    row = match_rows(t["uid"], t.get("copy"), uids, copies)
    if (row < 0).any():
        raise ValueError(f"{pj.name} lacks some of the particles")
    return np.stack([t["x_a"][row], t["y_a"][row]], 1).astype(np.float64)


def duplicate_distance(project: Project, particle_job) -> float:
    """Two particles whose aligned centres are closer than this are one particle picked twice: half the picking
    diameter (the particle's short dimension; two distinct particles cannot be that close)."""
    pick = project.ancestor(particle_job, "Pick")
    return 0.5 * pick.params["diameter_a"] if pick else project.job(particle_job).params.get("particle_a", 0) / 4


def drop_duplicates(centres: np.ndarray, movie: np.ndarray, fit: np.ndarray, dist: float) -> np.ndarray:
    """Mask of particles kept: per micrograph, of aligned centres closer than dist only the best-fitting one. Must run
    before particles are split into gold-standard halves - two copies of one particle in different halves inflate
    the FSC."""
    from .picking import dedup
    keep = np.ones(len(centres), bool)
    for m in np.unique(movie):
        i = np.flatnonzero(movie == m)
        keep[i] = dedup(centres[i], fit[i], dist)
    return keep


def _session(project: Project, job: Job, src, diameter_a, max_res_a, gpus, log, workdir=None, **params):
    from .session import Session
    s = Session(workdir or os.path.join(job.path, "session"), log=log)
    _migrate_session(s.workdir, src.path)
    info = s.load(src, diameter=diameter_a or None, max_res=max_res_a, gpus=gpus, **params)
    return s, info


def _ids(key: str) -> list[str]:
    """Job numbers in a session input key: '<root>#004#005' (now) or '<root>/jobs/004_extract_x#005_class2d_y' (older
    versions, which broke when a job was renamed)."""
    return [os.path.basename(k)[:3] for k in key.split("#")[1:]] + ([os.path.basename(key.split("#")[0])[:3]]
                                                                     if "/jobs/" in key.split("#")[0] else [])


def _migrate_session(workdir: str, key: str):
    """Results a job's session saved under an older input key (folder names) are re-keyed to the current one (job
    numbers), so they are restored."""
    if not os.path.isdir(workdir):
        return
    for fn in os.listdir(workdir):
        f = os.path.join(workdir, fn, "summary.json") if os.path.isdir(os.path.join(workdir, fn)) else \
            os.path.join(workdir, fn)
        if not (f.endswith(".result.json") or f.endswith("summary.json")) or not os.path.isfile(f):
            continue
        meta = json.load(open(f))
        old = meta.get("input")
        if isinstance(old, str) and old != key and "#" in old and sorted(_ids(old)) == sorted(_ids(key)):
            meta["input"] = key
            json.dump(meta, open(f, "w"), indent=1, default=str)


def _write_results(project: Project, job: Job, src, idx, cols: dict, name="particles"):
    parts = src.parts.iloc[np.asarray(idx)]
    if "celCopy" in parts:                           # symmetry-expanded particles: one row per (uid, copy)
        cols = dict(cols, copy=parts["celCopy"].to_numpy(np.int16))
    project.table(job, name).write(dict(uid=parts["celUid"].to_numpy(np.int64), **cols),
                                   parts["celMovie"].to_numpy(np.int64))


# ------------------------------------------------------------------------------------------------ Class2D
@dataclass
class Class2DSettings:
    n_classes: int = 50             # number of classes (K)
    max_res_a: float = 0.0          # finest resolution used (0: Nyquist of the particles, but not finer than 6 A)
    diameter_a: float = 0.0         # mask (0: 1.25 x the extraction's particle size)
    seeds: int = 1                  # > 1: independent repeats, per-particle P(good) after selection
    gpus: str = "auto"              # GPUs to use: "auto" (all free ones) or a list like "0,2"


def class2d_job(project: Project, particles, s: Class2DSettings, label: str = "") -> Job:
    p = project.job(particles)
    return project.new_job("Class2D", dict(params_of(s), particles=p.id), [p], label or f"K{s.n_classes}_on{p.id:03d}")


@logged
def run_class2d(project: Project, job, log=print) -> dict:
    job = project.job(job)
    s = settings_of(job, Class2DSettings)
    src = particle_source(project, job.params["particles"])
    max_res = s.max_res_a or max(6.0, 2 * src.apix)
    t0 = time.time()
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log)
    ses.classify("2d", K=s.n_classes, seeds=s.seeds)
    r = ses.results["2d"]
    _write_results(project, job, src, r.idx, dict(cls=r.labels.astype(np.int16), psi=r.psi.astype(np.float32),
                                                  shift_x_a=r.offsets[:, 0].astype(np.float32),
                                                  shift_y_a=r.offsets[:, 1].astype(np.float32),
                                                  maxprob=r.maxprob.astype(np.float32)))
    project.write_mrc(job.out("classes.mrcs"), r.images, ses.clf.box_a / r.images.shape[-1])
    shutil.copy(r.gallery_path, job.out("gallery.png"))
    light = os.path.splitext(r.gallery_path)[0] + ".light.png"
    if os.path.exists(light):
        shutil.copy(light, job.out("gallery.light.png"))
    summ = dict(particles=int(len(r.idx)), classes=int(len(r.pops)), mask_a=round(ses.clf.p.diameter_a, 1),
                max_res_a=round(max_res, 2), seconds=round(time.time() - t0, 1),
                table=[dict(cls=k + 1, particles=int(r.pops[k]), conf=round(float(r.conf[k]), 3),
                            res_a=round(float(r.res_a[k]), 1) if np.isfinite(r.res_a[k]) else None)
                       for k in np.argsort(-r.pops)])
    project.set_status(job, "finished", summ)
    ses.clf = None
    release_cache()
    return summ


def select_classes(project: Project, class2d, classes: list[int], label: str = "", note: str = "") -> Job:
    """A Select job with the particles of these classes (1-based numbers, as in the gallery) of a Class2D job (or a
    HeteroRefine / Class3D: its table `particles` has the same uid / cls columns)."""
    c = project.job(class2d)
    t = project.table(c, "particles").read(["uid", "cls"])
    keep = np.isin(t["cls"], np.asarray(classes, int) - 1)
    chosen = sorted(int(k) for k in classes)
    K = len(project.summary_of(c).get("table", [])) or int(t["cls"].max()) + 1
    job = project.select(c, t["uid"][keep], label=label or f"{len(classes)}cls_of{c.id:03d}",
                         note=note or f"classes {chosen} of {c.name}", rejected=t["uid"][~keep])
    rejected = [k for k in range(1, K + 1) if k not in chosen]
    project.update_params(job, classes=chosen, rejected_classes=rejected)
    cm = c.find("classes.mrcs")
    if cm:
        import mrcfile
        with mrcfile.open(cm, permissive=True) as f:
            imgs, apix = np.asarray(f.data, np.float32), float(f.voxel_size.x)
        pops = np.bincount(t["cls"], minlength=len(imgs))
        for name, ks in (("selected_classes.mrcs", chosen), ("rejected_classes.mrcs", rejected)):
            if ks:
                project.write_mrc(job.out(name), imgs[np.asarray(ks) - 1], apix)
        _selection_figure(job.out("selection.png"), imgs, pops, set(chosen),
                          f"{job.name}: {int(keep.sum()):,} kept ({len(chosen)} classes), "
                          f"{int((~keep).sum()):,} rejected ({len(rejected)} classes) of {c.name}")
    return job


def _selection_figure(path, imgs, pops, chosen, title_, ncol=10):
    from .plotstyle import themed
    themed(_draw_selection)(path, imgs, pops, chosen, title_, ncol)


def _draw_selection(path, imgs, pops, chosen, title_, ncol):
    """Class images by population, kept ones framed in the 'good' status colour, rejected ones dimmed and framed in
    'critical' (each also marked ✓ / ✕, so the colour is not the only cue)."""
    from .plotstyle import GRAY, STATUS, T, norm_img, plt, title
    order = np.argsort(-pops)
    nrow = (len(order) + ncol - 1) // ncol
    H = nrow * 1.45 + 0.8
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.3, H), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.8 / H, bottom=0.01, wspace=0.08, hspace=0.3)
    for a in axs.ravel():
        a.axis("off")
    for a, k in zip(axs.ravel(), order):
        ok = k + 1 in chosen
        a.imshow(norm_img(imgs[k], 1, 99), cmap=GRAY, alpha=1.0 if ok else 0.4)
        a.axis("on")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(STATUS["good"] if ok else STATUS["critical"])
            sp.set_linewidth(2.5)
        a.set_title(f"{'✓' if ok else '✕'} {k + 1} · {pops[k]:,}", fontsize=7.5, fontweight="normal",
                    color=T.ink if ok else T.ink3, loc="center")
    title(fig, "Class selection", title_)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- InitialModel
@dataclass
class InitialModelSettings:
    sym: str = "C1"                 # the user's choice; never detected
    classes: tuple = ()             # 1-based classes of the Class2D job (empty: all populated classes)
    n_models: int = 1               # > 1: that many maps, particles classified among them (3D classes)
    mask: str = "auto"              # solvent mask of the particle refinement: "auto" (dynamic, below 15 A) or "none"
                                    # (the particle sphere only)
    seed: int = 0                   # random seed
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def _as_classified(project: Project, c: Job, src):
    """Restrict src (the particles a Class2D job was given) to those it classified, in the order it saw them: an
    extraction that kept growing afterwards (more movies, live collection) would otherwise no longer match the 2D
    result the job saved (its session is keyed on the particle count)."""
    u = project.table(c, "particles").read(["uid"])["uid"]
    have = src.parts["celUid"].to_numpy(np.int64)
    if len(have) == len(u):
        return
    keep = np.flatnonzero(np.isin(have, u))
    if len(keep) != len(u) or not np.array_equal(have[keep], np.asarray(u, np.int64)):
        raise ValueError(f"{c.name}: its particles changed since it ran (not a superset in the same order): run the "
                         "Class2D job again")
    src.subset(keep)


def initial_model_job(project: Project, class2d, s: InitialModelSettings, label: str = "") -> Job:
    c = project.job(class2d)
    return project.new_job("InitialModel", dict(params_of(s), class2d=c.id), [c], label or f"{s.sym}_from{c.id:03d}")


@logged
def run_initial_model(project: Project, job, log=print) -> dict:
    """Ab initio (abinitio.run) from the Class2D job's classes and their particles; the 2D run is reloaded from that
    job's session (same particles, same settings), so nothing is classified again."""
    from . import abinitio as ab
    job = project.job(job)
    s = settings_of(job, InitialModelSettings)
    c = project.job(job.params["class2d"])
    cs = settings_of(c, Class2DSettings)
    src = particle_source(project, c.params["particles"])
    _as_classified(project, c, src)
    ses, _ = _session(project, c, src, cs.diameter_a, project.summary_of(c)["max_res_a"], s.gpus, log,
                      workdir=os.path.join(c.path, "session"))
    if "2d" not in ses.results:
        raise ValueError(f"{c.name}: its 2D result was not found in {c.name}/session (made by an older version: run "
                         f"the Class2D job again)")
    r = ses.results["2d"]
    mask = None
    if s.classes:
        mask = np.zeros(len(r.pops), bool)
        mask[np.asarray(s.classes, int) - 1] = True
    # duplicates (2D-aligned centres closer than duplicate_distance) are left out before the half split
    pj = project.aligned_particles(c)
    covered = r.labels >= 0
    rows = np.asarray(r.idx)[covered]
    uid = src.parts["celUid"].to_numpy(np.int64)[rows]
    centres = _centres(project, pj, uid) + np.asarray(r.offsets, np.float64)[covered]
    dist = duplicate_distance(project, pj)
    keep = drop_duplicates(centres, src.parts["celMovie"].to_numpy(np.int64)[rows], np.asarray(r.maxprob)[covered],
                           dist)
    n_dup = int((~keep).sum())
    if n_dup:
        log(f"[3d] {n_dup} duplicate particles left out (2D-aligned centres < {dist:.0f} A apart; best fit kept)")
        r = copy.copy(r)
        r.labels = np.asarray(r.labels).copy()
        r.labels[np.flatnonzero(covered)[~keep]] = -1
    if str(s.mask).lower() not in ("auto", "dynamic", "none", "sphere"):
        raise ValueError(f"InitialModel mask {s.mask!r}: 'auto' or 'none' (there is no starting map for a static mask)")
    prm = ab.AbinitioParams(sym=s.sym, n_models=s.n_models, seed=s.seed,
                            solvent_mask="auto" if str(s.mask).lower() in ("auto", "dynamic") else "sphere")
    summ = ab.run(ses.clf, r, job.path, prm, classes=mask, log=log, src=None)
    z = np.load(os.path.join(job.path, "poses.npz"))
    _write_results(project, job, src, z["idx"], dict(
        rot=z["rot"].astype(np.float32), tilt=z["tilt"].astype(np.float32), psi=z["psi"].astype(np.float32),
        shift_x_a=z["shift_a"][:, 0].astype(np.float32), shift_y_a=z["shift_a"][:, 1].astype(np.float32),
        used=z["used"].astype(bool), model=z["model"].astype(np.int16)))
    os.remove(os.path.join(job.path, "poses.npz"))                     # the sharded table replaces it
    job.organize()                                                     # ab.run writes flat: into the job layout
    keep = {k: summ[k] for k in ("particles", "particles_used", "resolution_a", "resolution_solvent_mask_a",
                                 "resolution_unmasked_a", "apix", "box", "seconds_total")}
    keep.update(sym=s.sym, map=project.rel(job.find("map.mrc")), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1))
    project.set_status(job, "finished", keep)
    ses.clf = None
    release_cache()
    return keep


def _sharpen_job_maps(project: Project, job: Job, s, classes=(0,), log=print) -> dict:
    """The job's own sharpening and local resolution (settings sharpen / local_res / mtf / mtf_apix) of each map
    (classes: 0 for a single map, else the 1-based classes). A failure is logged, not fatal: the refinement stands."""
    if not getattr(s, "sharpen", False):
        return {}
    from .sharpen import sharpen_in_job
    out = {}
    for c in classes:
        try:
            r = sharpen_in_job(project, job, cls=c, mtf=s.mtf, mtf_apix=s.mtf_apix, local_res=s.local_res, log=log)
            out[str(c) if c else "map"] = {k: r[k] for k in ("resolution_a", "resolution_fsc05_a", "b_factor",
                                                             "guinier_fit", "local_resolution", "map", "warnings")
                                           if k in r}
        except Exception as e:                       # noqa: BLE001 - reported, the refinement's own results stand
            log(f"[sharpen] {job.name}{f' class {c}' if c else ''}: not sharpened ({e})")
            out[str(c) if c else "map"] = dict(error=str(e))
    return out if classes != (0,) else out.get("map", {})


# ---------------------------------------------------------------------------------------------- Refine3D
@dataclass
class Refine3DSettings:
    sym: str = "C1"                 # point-group symmetry, the user's choice (C1, Cn, Dn, T, O, I)
    n_iter: int = 10                # iterations of local orientation search
    res_start_a: float = 0.0        # first alignment resolution (0: the reference job's resolution, at least 8 A)
    max_res_a: float = 0.0          # finest resolution loaded (0: Nyquist)
    n_local: int = 1                # local search rounds per iteration
    min_step_deg: float = 0.5       # finest angular step (deg); the step halves once < 30% of particles move
    drop_frac: float = 0.1          # share of worst-fitting particles left out of each reconstruction
    global_iters: int = 0           # first iterations that also search all orientations (for starts with pose errors)
    global_res_a: float = 12.0      # resolution of that search
    diameter_a: float = 0.0         # particle diameter for the mask (0: from the extraction)
    classes: tuple = ()             # poses from a HeteroRefine: only the particles of these 1-based classes
    mask: str = "auto"              # solvent mask during refinement: "auto" (dynamic: made from the map every iteration
                                    # below 15 A, masking.auto_mask), "static" (made once from the starting map, then
                                    # fixed), "none" (the soft particle sphere only), or a MaskCreate job / .mrc path
    grid_correct: bool = False      # gridding correction of the 3D spectra (more accurate projections and maps)
    reference_map: str = ""         # start from this map (.mrc) instead of the reference job's (low-passed to res_start_a,
                                    # default 30 A; poses from `poses`, so add global_iters when its frame differs)
    nu: bool = False                # non-uniform regularisation (cryoSPARC NU-refine's idea): each region of the
                                    # references low-passed at its own resolution, chosen by cross-validation
    nu_strict: bool = True          # ... the cutoffs of each half from two folds of its own particles (gold standard)
    particle_scale: bool = False    # a contrast scale per particle (cryoSPARC's per-particle scale), fitted every
                                    # iteration, used in the search and the reconstruction
    wiener: float = 0.0             # > 0: half maps regularised by the FSC (Wiener / MAP, RELION's tau2 term with this
                                    # fudge factor, 1 = RELION's default); 0: a fixed small regulariser
    pad: int = 0                    # padding of the 3D spectra: 0 automatic (2 if it fits the GPU, else 1), 1 or 2
    align_symmetry: bool = True     # sym differs from the start job's: first rotate the map and the poses so the
                                    # group's axes (n-fold on z, 2-fold on x) lie on the map's (symalign.py); the
                                    # symmetry itself is the user's choice, the scores say how well it holds
    sharpen: bool = True            # sharpen the final half maps with the job's own mask and compute their local
                                    # resolution (sharpened.mrc, postprocess.png, localres.mrc ...; a few seconds), as a
                                    # Sharpen job with mask "refine" would - no separate job needed
    local_res: bool = True          # ... including the local resolution
    mtf: str = ""                   # ... detector MTF (RELION .star) for the sharpening; empty: none
    mtf_apix: float = 0.0           # ... the pixel size (A) the MTF refers to (0: the map's)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def refine3d_job(project: Project, particles, poses, reference, s: Refine3DSettings, label: str = "") -> Job:
    """particles: an Extract job (e.g. re-centred, full box); poses: the job whose per-particle orientations start
    the refinement (InitialModel / Refine3D, matched by uid); reference: the job whose map.mrc starts it."""
    p, q = project.job(particles), project.job(poses)
    r = project.job(reference) if reference not in (None, "", 0) else q
    return project.new_job("Refine3D", dict(params_of(s), particles=p.id, poses=q.id, reference=r.id),
                           [p, q] + ([r] if r.id != q.id else []), label or f"{s.sym}_on{p.id:03d}")


def map_to_box(vol: np.ndarray, apix: float, b: int, apix_new: float) -> torch.Tensor:
    """A map resampled to pixel size apix_new (Fourier crop / pad) and padded / cropped to a b^3 box, centred."""
    n = vol.shape[0]
    n2 = int(round(n * apix / apix_new / 2)) * 2
    F = torch.fft.fftshift(torch.fft.fftn(torch.as_tensor(vol, dtype=torch.float32)))
    G = torch.zeros((n2,) * 3, dtype=torch.complex64)
    m = min(n, n2)
    a, c = (n - m) // 2, (n2 - m) // 2
    G[c:c + m, c:c + m, c:c + m] = F[a:a + m, a:a + m, a:a + m]
    v = torch.fft.ifftn(torch.fft.ifftshift(G)).real * (n2 / n) ** 3
    out = torch.zeros((b,) * 3)
    m = min(b, n2)
    a, c = (n2 - m) // 2, (b - m) // 2
    out[c:c + m, c:c + m, c:c + m] = v[a:a + m, a:a + m, a:a + m]
    return out


def _scale_to_data(clf, idx, A, t, vol: torch.Tensor, b: int, res_a: float, n: int = 500) -> float:
    """Least-squares scale of the map's CTF-modulated projections to particles at their poses (frequencies up to
    res_a): a map from another box or pixel size has other intensity units than these particles."""
    from .abinitio import Volume, _to_box
    from .engine import TWO_PI, ctf_2d, freqs, release_cache
    dev, box_a = clf.dev, clf.box_a
    # on the smallest box that holds the band (the data are cropped alike): a pad-2 volume at box 512 is 8.6 GB
    b = min(b, int(math.ceil(2 * box_a / res_a / 2)) * 2 + 8)
    v = Volume(b, dev)
    v.set_real(_to_box(vol.to(dev), b, dev) * (b / vol.shape[0]) ** 3)     # crop keeping the map's values
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= 2) & (kr <= box_a / res_a)
    fyb, fxb = fy[:, None].expand_as(kr)[band], fx[None, :].expand_as(kr)[band]
    sel = torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:n]
    j = idx[sel]
    x = clf._fetch(j, dev, b)[:, band]
    ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
    tt = t[sel].to(dev)
    P = v.project(A[sel].to(dev), band) * ctf * torch.exp(-1j * TWO_PI / box_a * (fxb * tt[:, 0, None] + fyb * tt[:, 1, None]))
    return float((x.conj() * P).real.sum() / (P.abs() ** 2).sum().clamp_min(1e-20))


def _posed_particles(project: Project, job: Job, pj: Job, qj: Job, classes=(), log=print, expand: bool = False):
    """The particles of particle job pj that have poses in job qj (optionally only those in these 1-based classes of a
    HeteroRefine / Class3D), duplicates removed. Returns (source, poses (dict of arrays, one per row), starting shifts
    (n, 2) A, halves or None, number of duplicates, duplicate distance), one row per particle (or per image of
    symmetry-expanded particles, e.g. an expanded Subtract job: (uid, copy) matched).
    expand: poses with several copies per uid (a SymExpand job, or a refinement of one) on particles without copies
    give one row per copy - the source holds every image once, and a seventh value img (rows,) is the source index of
    each row's image (refine_particles / classify take it as idx / rows: n images in memory, not n x copies).
    Without expand such poses are refused for a SymExpand job, else each particle takes its lowest copy."""
    # every particle with a pose, also those the poses job left out of its reconstruction: the refinement leaves out
    # its own worst-fitting share. Taking only the "used" ones compounded: InitialModel -> Refine3D -> Refine3D
    # (polished) -> Refine3D (CTF-refined) kept 4,317 -> 3,885 -> 3,496 -> 3,146 of beta-gal's particles.
    poses = project.table(qj, "particles").read()
    if classes:
        if "cls" not in poses:
            raise ValueError(f"{qj.name} has no 3D classes")
        sel = np.isin(poses["cls"], np.asarray(classes, int) - 1)
        poses = {k: v[sel] for k, v in poses.items()}
    src = particle_source(project, pj)
    uid = src.parts["celUid"].to_numpy(np.int64)
    cp = src.parts["celCopy"].to_numpy(np.int64) if "celCopy" in src.parts else None
    pc = poses.get("copy")
    if cp is not None and pc is None and (cp > 0).any():
        raise ValueError(f"{pj.name} holds symmetry-expanded copies of each particle: take the poses from it (or from "
                         f"a refinement of it), not from {qj.name}, whose poses are one per particle")
    expanded = pc is not None and bool((np.asarray(pc) > 0).any())
    if expanded and cp is None and not expand and qj.type == "SymExpand":
        raise ValueError(f"{job.type} does not take symmetry-expanded poses ({qj.name}): LocalRefine, Refine3D, "
                         f"Subtract and Class3D do")
    grow = expand and expanded and cp is None
    if grow:                        # every pose row of a particle (its copies, by copy number), the image shared
        pu = np.asarray(poses["uid"], np.int64)
        o = np.lexsort((np.asarray(pc, np.int64), pu))
        lo, hi = np.searchsorted(pu[o], uid, "left"), np.searchsorted(pu[o], uid, "right")
        ok = hi > lo
        if not ok.any():
            raise ValueError(f"none of the particles of {pj.name} have poses in {qj.name}")
        src.subset(np.flatnonzero(ok))
        lo, cnt = lo[ok], (hi - lo)[ok]
        img = np.repeat(np.arange(len(lo)), cnt)
        row = o[np.repeat(lo, cnt) + np.arange(len(img)) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    else:
        rows = match_rows(poses["uid"], pc, uid, cp)
        ok = rows >= 0
        if not ok.any():
            raise ValueError(f"none of the particles of {pj.name} have poses in {qj.name}")
        src.subset(np.flatnonzero(ok))
        row = rows[ok]
        cp = None if cp is None else cp[ok]
        img = np.arange(len(row))
    # starting shift = aligned centre (centre in the extraction the poses were found on + their shift) - centre in
    # this extraction: right for any pair of extractions (re-centred on these poses, on other ones, or not at all)
    c_new = _centres(project, pj, poses["uid"][row], cp)
    c_al = _centres(project, project.aligned_particles(qj), poses["uid"][row],
                    None if pc is None else pc[row]) + np.stack([poses["shift_x_a"][row], poses["shift_y_a"][row]], 1)
    t_start = (c_al - c_new).astype(np.float32)
    # gold-standard halves: those of the refinement the poses come from (a particle stays in the half it was aligned
    # / polished against); duplicates (aligned centres closer than duplicate_distance) are left out, best fit kept
    half = poses["half"][row].astype(np.int64) if "half" in poses else None
    fit = poses["fit"][row] if "fit" in poses else (poses["maxprob"][row] if "maxprob" in poses else np.ones(len(row)))
    dist = duplicate_distance(project, pj)
    mov = src.parts["celMovie"].to_numpy(np.int64)[img]
    if cp is None and not grow:
        keep = drop_duplicates(c_al, mov, fit, dist)
    else:   # symmetry copies share the particle's centre: duplicates are judged per particle, its copies go together
        u_ = poses["uid"][row]
        first = np.unique(u_, return_index=True)[1]
        k1 = drop_duplicates(c_al[first], mov[first], fit[first], dist)
        keep = np.isin(u_, u_[first][k1])
    n_dup = int((~keep).sum()) if not grow else len(np.unique(img[~keep]))
    if n_dup:
        log(f"[3d] {n_dup} duplicate particles left out (aligned centres < {dist:.0f} A apart; best fit kept)")
        row, t_start, img = row[keep], t_start[keep], img[keep]
        half = None if half is None else half[keep]
        kept_img = np.unique(img)
        src.subset(kept_img)
        img = np.searchsorted(kept_img, img)
    src.path = f"{project.root}#{pj.id:03d}#{job.id:03d}"
    out = (src, {k: v[row] for k, v in poses.items()}, t_start, half, n_dup, dist)
    return out + (img,) if expand else out


def load_mask(project: Project, ref: str, b: int, apix: float) -> np.ndarray:
    """A mask for a map of box b at apix: a MaskCreate job (its mask.mrc) or an .mrc path (absolute or relative to
    the project), resampled to that grid and clipped to 0..1."""
    import mrcfile
    path = ref if str(ref).endswith(".mrc") else project.job(ref).find("mask.mrc")
    path = path if os.path.isabs(path) else project.abs(path)
    with mrcfile.open(path, permissive=True) as f:
        m, ap = np.asarray(f.data, np.float32), float(f.voxel_size.x)
    return np.clip(map_to_box(m, ap, b, apix).cpu().numpy(), 0, 1)


MASK_MODES = ("auto", "static", "none")


def solvent_mask_arg(project: Project, mode: str, b: int, apix: float, refs: list, res_a: float, diameter_a: float,
                     log=print):
    """A refinement's mask setting -> refine_particles(solvent_mask=...): "auto" (dynamic: made from each model's map
    every iteration below 15 A), "static" (made once from each starting reference with masking.auto_mask, then
    fixed), "none" (no solvent mask: the soft particle sphere only; "sphere" is the same), or a MaskCreate job /
    .mrc path (that mask, fixed, for every model). refs: the starting maps (real, b^3), one per model."""
    m = str(mode or "auto").strip()
    if m.lower() in ("auto", "dynamic"):
        return "auto"
    if m.lower() in ("none", "sphere", "off", ""):
        return None
    if m.lower() == "static":
        from .masking import auto_mask
        out = []
        for r in refs:
            out.append(auto_mask(torch.as_tensor(r).float(), apix, res_a=max(res_a, 2.2 * apix), diameter_a=diameter_a))
        log(f"[3d] static solvent mask from the starting map{'s' if len(out) > 1 else ''}: "
            + ", ".join(f"{float(x.mean()) * 100:.1f}%" for x in out) + " of the box")
        return out if len(out) > 1 else out[0]
    try:
        return torch.as_tensor(load_mask(project, m, b, apix))
    except Exception as e:
        raise ValueError(f"mask {mode!r}: not one of {MASK_MODES} and not a MaskCreate job / .mrc file ({e})") from e


def map_job(project: Project, ref) -> Job:
    """The job whose map goes with a poses job: a Subtract or SymExpand job's poses are those of the refinement whose
    map it used (expanded copies: that map's symmetry copies), so that refinement's."""
    j = project.job(ref)
    while j.type in ("Subtract", "SymExpand"):
        j = project.job(j.params["poses"])
    return j


def _reference_map(rj: Job, cls: int = 0) -> tuple[np.ndarray, float, str]:
    """A job's map before masking: a HeteroRefine / Class3D class (1-based cls; 0: the class with most particles),
    else map_nomask.mrc / map.mrc."""
    import mrcfile
    if rj.type in CLASS_JOBS:
        if not cls:
            sp = os.path.join(rj.path, "summary.json")
            cl = json.load(open(sp)).get("classes", []) if os.path.isfile(sp) else []
            cls = max(cl, key=lambda c: c.get("particles", 0))["cls"] if cl else 1
        names = [f"class{cls:02d}_map_nomask.mrc"]
    else:
        names = ["map_nomask.mrc", "map.mrc"]
    path = next((rj.find(n) for n in names if rj.find(n)), None)
    if path is None:
        raise ValueError(f"{rj.name} has no {names[0]}")
    with mrcfile.open(path, permissive=True) as f:
        return np.array(f.data, np.float32), float(f.voxel_size.x), path


@logged
def run_refine3d(project: Project, job, log=print) -> dict:
    from . import abinitio as ab
    job = project.job(job)
    s = settings_of(job, Refine3DSettings)
    pj, qj, rj = (project.job(job.params[k]) for k in ("particles", "poses", "reference"))
    rj = map_job(project, rj)
    # symmetry-expanded poses (a SymExpand job, or a refinement of one): one row per copy, each image loaded once
    src, poses, t_start, half, n_dup, dist, img = _posed_particles(project, job, pj, qj, tuple(s.classes), log,
                                                                   expand=qj.type not in CLASS_JOBS)
    max_res = s.max_res_a or 2.0 * src.apix
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log, nyquist_margin=1.0, max_c64_gb=1.5)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    dev, b, box_a = clf.dev, clf.bmax, clf.box_a
    A = ab.euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                           torch.as_tensor(poses["psi"])).float()
    t = torch.as_tensor(t_start).float()
    ref_cls = 0
    if rj.type in CLASS_JOBS:       # the class the particles come from (the largest of them)
        cl = poses["cls"] if "cls" in poses else np.zeros(len(t_start), int)
        ref_cls = int(np.bincount(cl).argmax()) + 1
    ref, ref_apix, _ = _reference_map(rj, ref_cls)
    rsum = project.summary_of(rj)
    r_ref = rsum.get("resolution_a", 8.0) if rj.type not in CLASS_JOBS else \
        next((c["resolution_a"] for c in rsum.get("classes", []) if c["cls"] == ref_cls), None) or 8.0
    if s.reference_map:                              # an external map (e.g. a deposited one), low-passed to start
        import mrcfile
        with mrcfile.open(s.reference_map if os.path.isabs(s.reference_map) else project.abs(s.reference_map),
                          permissive=True) as f:
            ref, ref_apix = np.array(f.data, np.float32), float(f.voxel_size.x)
        r_ref = 30.0
        log(f"[3d] reference {s.reference_map} ({ref.shape[0]} px at {ref_apix:.3f} A), from {s.res_start_a or 30:g} A")
    res0 = s.res_start_a or max(8.0, float(r_ref))
    vol = map_to_box(ref, ref_apix, b, box_a / b)
    start_sym = str(map_job(project, qj).params.get("sym") or "C1").upper()
    sym_frame = None
    if s.align_symmetry and s.sym.upper() != "C1" and s.sym.upper() != start_sym:
        from .symalign import apply_frame, sym_frame as _frame
        R, sym_frame = _frame(vol, box_a / b, s.sym, log=log)
        vol, A = apply_frame(vol, A, R)
        sym_frame["from"] = start_sym
    idx = torch.as_tensor(img)
    n_rows = len(img)
    vol *= _scale_to_data(clf, idx, A, t, vol, b, max(res0, 10.0))
    init = vol.to(dev)                                  # the real map (a padded Volume at box 512 would be 8.6 GB)
    t0 = time.time()
    A_start = A.clone()
    sm = solvent_mask_arg(project, s.mask, b, box_a / b, [init], res0, clf.p.diameter_a, log=log)
    maps, A, t, z, fscs, res, vols = ab.refine_particles(clf, idx, A, t, init, sym=s.sym, n_iter=s.n_iter, step_min=s.min_step_deg,
                                                         res_start=res0, n_local=s.n_local, drop_frac=s.drop_frac,
                                                         global_at=tuple(range(1, s.global_iters + 1)),
                                                         global_res=s.global_res_a, box=b,
                                                         half=None if half is None else torch.as_tensor(half),
                                                         solvent_mask=sm, grid_correct=s.grid_correct,
                                                         pad=s.pad or None, nu=s.nu, nu_strict=s.nu_strict,
                                                         particle_scale=s.particle_scale, wiener=s.wiener, log=log)
    seconds = time.time() - t0
    apix = box_a / b
    h1, h2 = vols[0][0].real(), vols[0][1].real()
    smask = ab.refine_particles.last_mask
    if smask is None:                                   # sphere during refinement: a solvent mask for the report
        from .masking import auto_mask
        smask = auto_mask(maps[0], apix, res_a=float(res[0]), k_sigma=5.0, halves=(h1, h2), diameter_a=clf.p.diameter_a)
    fsc_sol = ab.corrected_masked_fsc(h1, h2, smask)
    fsc_unm = ab.fsc(vols[0][0], vols[0][1])
    fsc_msk = ab.fsc_maps(h1 * smask, h2 * smask)
    for name, v in (("map.mrc", maps[0] * smask), ("map_nomask.mrc", maps[0]), ("mask.mrc", smask),
                    ("half1_unfil.mrc", h1), ("half2_unfil.mrc", h2)):
        ab._save_mrc(job.out(name), v, apix)
    rot, tilt, psi = ab.matrix_to_euler(A.cpu())
    fit = ab.refine_particles.last_score.cpu().numpy().astype(np.float32)
    used = (ab.refine_particles.last_weights > 0).numpy()
    _write_results(project, job, src, img, dict(
        rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32), psi=psi.numpy().astype(np.float32),
        shift_x_a=t[:, 0].cpu().numpy().astype(np.float32), shift_y_a=t[:, 1].cpu().numpy().astype(np.float32),
        fit=fit, half=ab.refine_particles.last_half.numpy().astype(np.uint8), used=used,
        **({"scale": ab.refine_particles.last_scale.numpy().astype(np.float32)} if s.particle_scale else {}),
        **({"copy": poses["copy"].astype(np.int16)} if "copy" in poses and "celCopy" not in src.parts else {})))
    res_sol = ab.resolution_from_fsc(fsc_sol, box_a)
    tag = f"{job.name} · {n_rows:,} particles · {s.sym}"
    report = refinement_report(job, tag, s.sym, box_a, apix, A_start, A, t, fit, used,
                               ab.refine_particles.last_history,
                               fsc=dict(corrected=fsc_sol, unmasked=fsc_unm, masked=fsc_msk, sphere_masked=fscs[0]),
                               resolution_a=res_sol, maps=[maps[0].cpu().numpy()], masks=[smask.cpu().numpy()])
    if sym_frame:
        report["warnings"] = sym_frame.pop("warnings", []) + report.get("warnings", [])
    nu_summ = {}
    if s.nu and ab.refine_particles.last_nu_map is not None:   # non-uniform filter: cutoff map and its statistics
        from . import plots3d
        cmap_ = ab.refine_particles.last_nu_map.float()
        ab._save_mrc(job.out("nu_cutoff.mrc"), cmap_, apix)
        last = ab.refine_particles.last_nu
        plots3d.nu_figure(job.out("nu.png"), ab.refine_particles.last_history, cmap_.numpy(), smask.cpu().numpy(),
                          maps[0].cpu().numpy(), last["candidates_a"], tag)
        nu_summ = dict(nu={k: last[k] for k in ("p05_a", "median_a", "p95_a", "at_global_pct", "coarse_pct",
                                                 "halves_r", "strict", "window_a")})
    sharp = _sharpen_job_maps(project, job, s, log=log)
    summ = dict(particles=int(n_rows), images=int(clf.N), particles_used=int(used.sum()), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), sym=s.sym, symmetry_frame=sym_frame,
                resolution_a=round(float(res[0]), 2),
                resolution_solvent_mask_a=round(res_sol, 2),
                resolution_unmasked_a=round(ab.resolution_from_fsc(fsc_unm, box_a), 2),
                apix=round(apix, 4), box=b, res_start_a=round(res0, 1), seconds=round(seconds, 1), **report,
                fsc_solvent=[round(float(x), 4) for x in fsc_sol.cpu()], map=project.rel(job.find("map.mrc")), **nu_summ,
                **({"sharpened": sharp} if sharp else {}))
    project.set_status(job, "finished", summ)
    ses.clf = None
    release_cache()
    return summ


# ------------------------------------------------------------------------------------------- HeteroRefine
@dataclass
class HeteroRefineSettings:
    references: tuple = ()          # maps, one per class: job numbers ("12") or "job:class" ("25:2" = InitialModel
                                    # model 2, "30:3" = HeteroRefine class 3); a reference can be given several times;
                                    # "decoy": a junk class (cryoSPARC's decoy references) - the first reference with
                                    # its phases randomised beyond decoy_res_a (its size and power, no structure), so
                                    # particles that fit no structure gather there; use with global_search. Which
                                    # class ends up clean is decided by the data, not by the reference it started
                                    # from: judge the classes by their maps and resolution. dz_2025_10, 153k
                                    # unselected particles, references (ab initio map, decoy, decoy): one class of
                                    # 35k holds 76 % of the particles that fit the map (MapScore) at 34 % fit; three
                                    # copies of the map instead: best class 51 % at 18 %, the rest split over two
    sym: str = "C1"                 # point-group symmetry, the user's choice
    n_iter: int = 12                # iterations
    res_start_a: float = 20.0       # first alignment / comparison resolution
    max_res_a: float = 0.0          # finest resolution loaded (0: Nyquist)
    global_search: bool = True      # first iteration: every class searches all orientations (references in their own
                                    # frames, e.g. separate ab initio runs); off: all start from the poses' orientation
    global_res_a: float = 15.0      # resolution of that search (A)
    n_local: int = 1                # local search rounds per iteration
    drop_frac: float = 0.0          # share of worst-fitting particles left out of each reconstruction
    soft: bool = False              # soft: every particle in every class, weighted by its probability (RELION);
                                    # hard: in its best class only (each particle's probabilities are written either way)
    diameter_a: float = 0.0         # particle diameter for the mask (0: from the extraction)
    mask: str = "auto"              # solvent mask during refinement, per class: "auto" (dynamic: from each class's map
                                    # every iteration below 15 A), "static" (once from each reference), "none" (the
                                    # particle sphere only), or a MaskCreate job / .mrc path (one mask for every class)
    seed: int = 0                   # random seed (noise added to repeated references)
    decoy_res_a: float = 30.0       # "decoy" references keep the first reference only to this resolution (A)
    sharpen: bool = True            # sharpen the final half maps with the job's own mask and compute their local
                                    # resolution (sharpened.mrc, postprocess.png, localres.mrc ...; a few seconds), as a
                                    # Sharpen job with mask "refine" would - no separate job needed
    local_res: bool = True          # ... including the local resolution
    mtf: str = ""                   # ... detector MTF (RELION .star) for the sharpening; empty: none
    mtf_apix: float = 0.0           # ... the pixel size (A) the MTF refers to (0: the map's)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def hetero_refine_job(project: Project, particles, poses, s: HeteroRefineSettings, label: str = "") -> Job:
    """Heterogeneous refinement (cryoSPARC's, RELION's 3D classification with alignment): particles are split among K
    reference maps, each class refined with its own poses, half maps and FSC. particles: an Extract / Polish / Select
    job; poses: the job whose orientations and shifts start it (InitialModel, Refine3D, HeteroRefine)."""
    if len(s.references) < 2:
        raise ValueError("heterogeneous refinement needs at least two reference maps")
    p, q = project.job(particles), project.job(poses)
    refs = [project.job(str(r).split(":")[0]) for r in s.references if str(r) != "decoy"]
    ins = [p, q] + [r for i, r in enumerate(refs) if r.id not in [x.id for x in [p, q] + refs[:i]]]
    return project.new_job("HeteroRefine", dict(params_of(s), references=[str(r) for r in s.references], particles=p.id,
                                                poses=q.id), ins, label or f"K{len(s.references)}_on{p.id:03d}")


def _reference(project: Project, ref: str):
    """(map, apix, description) of a reference given as 'job' or 'job:class'."""
    import mrcfile
    jid, _, k = str(ref).partition(":")
    rj = project.job(jid)
    if rj.type in CLASS_JOBS:
        return (*_reference_map(rj, int(k or 1))[:2], f"{rj.name} class {k or 1}")
    if k and rj.type == "InitialModel":
        path = rj.find(f"map_nomask_{int(k)}.mrc")
        if not path:
            raise ValueError(f"{rj.name} has no model {k}")
        with mrcfile.open(path, permissive=True) as f:
            return np.array(f.data, np.float32), float(f.voxel_size.x), f"{rj.name} model {k}"
    v, a, _ = _reference_map(rj)
    return v, a, rj.name


def decoy_map(vol: torch.Tensor, box_a: float, res_a: float, diameter_a: float, seed: int = 0) -> torch.Tensor:
    """A junk reference: vol's Fourier phases randomised beyond res_a (its low-resolution envelope and power spectrum
    kept, no structure), inside a soft sphere of the particle diameter."""
    from .abinitio import sphere_mask
    b = vol.shape[0]
    F = torch.fft.rfftn(vol.float())
    k = torch.fft.fftfreq(b, device=vol.device) * b
    kz = torch.arange(b // 2 + 1, device=vol.device, dtype=torch.float32)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)
    g = torch.Generator().manual_seed(seed)
    ph = torch.exp(2j * np.pi * torch.rand(F.shape, generator=g)).to(vol.device)
    F = torch.where(kk > box_a / res_a, F.abs() * ph, F)
    out = torch.fft.irfftn(F, s=vol.shape)
    return out * sphere_mask(b, 0.5 * diameter_a * b / box_a, 3.0, vol.device).to(out.dtype)


def _amplitude(vol: torch.Tensor, box_a: float, lo_a: float, hi_a: float) -> float:
    """Mean Fourier amplitude of a map between two resolutions (the scale of maps of one object agrees there)."""
    b = vol.shape[0]
    k = torch.fft.fftfreq(b, device=vol.device) * b
    kz = torch.arange(b // 2 + 1, device=vol.device, dtype=torch.float32)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)
    band = (kk >= box_a / lo_a) & (kk <= box_a / hi_a)
    return float(torch.fft.rfftn(vol).abs()[band].mean())


@logged
def run_hetero_refine(project: Project, job, log=print) -> dict:
    """Each class starts from its reference low-passed to res_start_a. With global_search, every particle first
    searches all orientations against every class (at global_res_a) and goes to the class it fits best; then local
    refinement (abinitio.refine_particles with K models): each iteration refines the pose in every class and moves
    the particle to the class it fits best, all classes compared at the same resolution (the worst one), so a
    class with more particles does not win particles just for being less noisy. A class left with < 1% of the
    particles is merged into the others."""
    from . import abinitio as ab
    from . import diagnostics as dg
    job = project.job(job)
    s = settings_of(job, HeteroRefineSettings)
    pj, qj = project.job(job.params["particles"]), project.job(job.params["poses"])
    src, poses, t_start, half, n_dup, dist = _posed_particles(project, job, pj, qj, (), log)
    max_res = s.max_res_a or 2.0 * src.apix
    t0 = time.time()
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log, nyquist_margin=1.0, max_c64_gb=1.5)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    dev, box_a, n = clf.dev, clf.box_a, clf.N
    K = len(s.references)
    b = ab.working_box(clf, n_acc=2 * K, n_ft=2 * K, log=log)
    apix = box_a / b
    A0 = ab.euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                            torch.as_tensor(poses["psi"])).float()
    t = torch.as_tensor(t_start).float()
    idx = torch.arange(n)
    if half is None:
        half = (torch.randperm(n, generator=torch.Generator().manual_seed(1)) % 2).numpy()
    half_t = torch.as_tensor(half).long()
    # references: same box and pixel size; the first scaled to the particles at their poses, the others to the
    # first's Fourier amplitudes at low resolution (they may be in other frames, so poses cannot scale them)
    vols, names = [], []
    if str(s.references[0]) == "decoy":
        raise ValueError("the first reference must be a map (decoys are made from it)")
    for i, r in enumerate(s.references):
        if str(r) == "decoy":
            vols.append(decoy_map(vols[0], box_a, s.decoy_res_a, clf.p.diameter_a, seed=s.seed + 11 * i))
            names.append(f"decoy ({s.decoy_res_a:g} A)")
            continue
        v, a, nm = _reference(project, r)
        vols.append(map_to_box(v, a, b, apix))
        names.append(nm)
    vols[0] = vols[0] * _scale_to_data(clf, idx, A0, t, vols[0], b, max(s.res_start_a, 10.0))
    a0 = _amplitude(vols[0], box_a, 2 * s.res_start_a, s.res_start_a)
    for i in range(1, K):
        vols[i] = vols[i] * (a0 / max(_amplitude(vols[i], box_a, 2 * s.res_start_a, s.res_start_a), 1e-20))
    inits = []
    g = torch.Generator().manual_seed(s.seed + 3)
    for i, v in enumerate(vols):
        V = ab.Volume(b, dev, pad=ab.auto_pad(b, K, dev), grid_correct=True)
        vv = v.to(dev)
        if any(str(s.references[j]) == str(s.references[i]) for j in range(i)):
            # a repeated reference: a little independent noise beyond res_start so the copies can separate
            vv = vv + 0.05 * vv.std() * torch.randn(vv.shape, generator=g).to(dev)
        V.set_real(vv)
        ab.lowpass(V, s.res_start_a, box_a)
        inits.append(V)
    A = A0[:, None].expand(n, K, 3, 3).contiguous()
    z0 = None
    if s.global_search:
        g_res = max(s.global_res_a, s.res_start_a * 0.75)
        step = ab.global_step(g_res, clf.p.diameter_a)
        lls = []
        Ag = []
        for i, V in enumerate(inits):
            Ai, ll, _ = ab.global_search(clf, idx, A0, t, [V, V], half_t, g_res, step, box=b, log=log)
            Ag.append(Ai.cpu())
            lls.append(ll.cpu())
        A = torch.stack(Ag, 1)
        z0 = torch.stack(lls, 1).argmax(1)
        log(f"[hetero] global search at {g_res:.0f} A ({step:.1f} deg): per class {torch.bincount(z0, minlength=K).tolist()}")
    A_init = A.clone()
    sm = solvent_mask_arg(project, s.mask, b, box_a / b, [V.real() for V in inits], s.res_start_a,
                          clf.p.diameter_a, log=log)
    maps, A, t, z, fscs, res, hv = ab.refine_particles(clf, idx, A, t, inits, z=z0, sym=s.sym, n_iter=s.n_iter,
                                                       res_start=s.res_start_a, n_local=s.n_local,
                                                       drop_frac=s.drop_frac, box=b, half=half_t, soft=s.soft,
                                                       solvent_mask=sm, log=log)
    dyn_masks = ab.refine_particles.last_masks
    seconds = time.time() - t0
    fit = ab.refine_particles.last_score.cpu().numpy().astype(np.float32)
    used = (ab.refine_particles.last_weights > 0).numpy()
    post = ab.refine_particles.last_post.numpy()
    if not s.soft and ab.refine_particles.last_ll is not None:   # hard: the posteriors the last pass's fits imply
        post = torch.softmax(ab.refine_particles.last_ll.nan_to_num(-1e30, neginf=-1e30), 1).numpy()
    # orientation change during the local refinement: from the pose each particle started with in its final class
    A_start = A_init[torch.arange(n), z.cpu()]
    z = z.cpu().numpy()
    rot, tilt, psi = ab.matrix_to_euler(A.cpu())
    _write_results(project, job, src, np.arange(n), dict(
        rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32), psi=psi.numpy().astype(np.float32),
        shift_x_a=t[:, 0].cpu().numpy().astype(np.float32), shift_y_a=t[:, 1].cpu().numpy().astype(np.float32),
        cls=z.astype(np.int16), fit=fit, half=np.asarray(half).astype(np.uint8), used=used,
        **{f"prob_{k + 1}": post[:, k].astype(np.float32) for k in range(K)}))
    table, fsc, masks, res_cls = [], {}, [], []
    for k in range(K):
        nk = int((z == k).sum())
        h1, h2 = hv[k][0].real().nan_to_num(0), hv[k][1].real().nan_to_num(0)
        smask = ((dyn_masks[k] if dyn_masks[k] is not None else ab.solvent_mask(maps[k], apix)) if nk
                 else torch.zeros_like(maps[k]))
        f_c = ab.corrected_masked_fsc(h1, h2, smask) if nk else torch.zeros(b // 2 + 1, device=dev)
        fsc[f"class{k + 1}"] = fscs[k]
        fsc[f"class{k + 1}_corrected"] = f_c
        r_c = ab.resolution_from_fsc(f_c, box_a) if nk else float("nan")
        res_cls.append(r_c)
        if nk:
            dg.write_fsc(job.out(f"class{k + 1:02d}_fsc.json")[:-5],
                         dict(corrected=f_c.cpu().numpy(), unmasked=ab.fsc_maps(h1, h2).cpu().numpy(),
                              masked=ab.fsc_maps(h1 * smask, h2 * smask).cpu().numpy(),
                              sphere_masked=fscs[k].cpu().numpy()), box_a, resolution_a=round(r_c, 3),
                         title=f"{job.name} class {k + 1}")
        masks.append(smask.cpu().numpy())
        for nm, v in (("map", maps[k] * smask), ("map_nomask", maps[k]), ("mask", smask), ("half1_unfil", h1),
                      ("half2_unfil", h2)):
            ab._save_mrc(job.out(f"class{k + 1:02d}_{nm}.mrc"), v, apix)
        table.append(dict(cls=k + 1, reference=names[k], particles=nk, share=round(nk / n, 4),
                          weight=round(float(post[:, k].sum()), 1),
                          confident=int(((z == k) & (post.max(1) > 0.9)).sum()),
                          resolution_a=round(res[k], 2) if res[k] == res[k] else None,
                          resolution_solvent_mask_a=round(r_c, 2) if r_c == r_c else None))
    tag = f"{job.name} · {n:,} particles · {K} classes · {s.sym}"
    labels = [f"class {k + 1}" for k in range(K)]
    report = refinement_report(job, tag, s.sym, box_a, apix, A_start, A, t, fit, used,
                               ab.refine_particles.last_history, fsc=fsc,
                               resolution_a=float(np.nanmin(res_cls)) if np.isfinite(res_cls).any() else float("nan"),
                               maps=[m.cpu().numpy() for m in maps], masks=masks, cls=z, class_labels=labels,
                               class_res=res_cls)
    cv = {c["cls"]: c for c in report.pop("class_views", [])}
    for row in table:
        row["scf"] = cv.get(row["cls"], {}).get("scf")
    sharp = _sharpen_job_maps(project, job, s, classes=tuple(r_["cls"] for r_ in table if r_["particles"] >= 100),
                              log=log)
    for row in table:                                 # each class's sharpened resolution / B-factor in its row
        sr = sharp.get(str(row["cls"]), {})
        if "resolution_a" in sr:
            row.update(sharpened_resolution_a=sr["resolution_a"], b_factor=sr["b_factor"])
    summ = dict(particles=int(n), classes=table, duplicates_removed=n_dup, duplicate_distance_a=round(dist, 1),
                sym=s.sym, apix=round(apix, 4), box=b, seconds=round(seconds, 1), **report,
                **({"sharpened": sharp} if sharp else {}))
    project.set_status(job, "finished", summ)
    log("[hetero] " + ", ".join(f"class {r['cls']}: {r['particles']} ({r['resolution_solvent_mask_a']} A)" for r in table))
    ses.clf = None
    release_cache()
    return summ


def refinement_report(job, tag: str, sym: str, box_a: float, apix: float, A_start, A, t, fit, used, history,
                      fsc: dict, resolution_a: float, maps: list, masks: list, cls=None, class_labels=None,
                      class_res=None) -> dict:
    """Diagnostics every 3D refinement writes (Refine3D, HeteroRefine): FSC curves (fsc.png + fsc.json / .star /
    .xml), angular distribution (views.png + views.json, angdist.bild for ChimeraX; per class with classes), progress
    per iteration
    (progress.png), per-particle fit / shift / orientation change (particles.png), map projections and sections with
    the mask (map_views.png). job: a Job (files go to job.out(...)) or a folder. Returns the summary entries (SCF,
    coverage, warnings, history)."""
    out_ = job.out if hasattr(job, "out") else (lambda n: os.path.join(job, n))
    from . import diagnostics as dg
    from . import plots3d
    from .abinitio import matrix_to_euler
    fsc_np = {k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in fsc.items()}
    dg.write_fsc(out_("fsc.json")[:-5], fsc_np, box_a, resolution_a=round(resolution_a, 3), title=tag)
    curves = [(lab, fsc_np[k]) for k, lab in (("corrected", "solvent mask, corrected"), ("sphere_masked", "soft sphere"),
                                              ("unmasked", "no mask")) if k in fsc_np]
    if cls is not None:
        curves = [(class_labels[i], fsc_np[f"class{i + 1}"]) for i in range(len(class_labels)) if f"class{i + 1}" in fsc_np]
    rot, tilt, _ = (x.numpy() for x in matrix_to_euler(A.cpu()))
    stats = dg.angular_stats(A.cpu(), sym)
    dg.write_views(out_("views.json"), rot, tilt, stats)
    radius = dg.particle_radius(masks[0] if masks else None, apix, box_a)
    dg.write_angdist_bild(out_("angdist.bild"), rot, tilt, box_a, radius)
    warnings = list(stats["warnings"])
    plots3d.fsc_figure(out_("fsc.png"), curves, box_a, apix, tag)
    plots3d.views_figure(out_("views.png"), rot, tilt, tag, stats)
    out = dict(scf=stats["scf"], view_coverage=dict(occupied=stats["occupied"], top10_share=stats["top10_share"]))
    if cls is not None:
        groups, per = [], []
        for i, lab in enumerate(class_labels):
            sel = cls == i
            if sel.sum() >= 10:
                st = dg.angular_stats(A.cpu()[torch.as_tensor(sel)], sym)
                groups.append((lab, rot[sel], tilt[sel], st))
                dg.write_angdist_bild(out_(f"class{i + 1:02d}_angdist.bild"), rot[sel], tilt[sel], box_a, radius)
                per.append(dict(cls=i + 1, scf=st["scf"], occupied=st["occupied"]))
        plots3d.views_grid_figure(out_("views_classes.png"), groups, tag)
        out["class_views"] = per
    if history:
        plots3d.refine_history_figure(out_("progress.png"), history, tag, classes=cls is not None)
        out["history"] = history
    moved = dg.orientation_change(A_start.cpu(), A.cpu(), sym) if A_start is not None else None
    tt = t.cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    plots3d.particle_stats_figure(out_("particles.png"), np.asarray(fit), np.asarray(used, bool), tt,
                                  moved, tag)
    if moved is not None:
        out["orientation_change_deg_median"] = round(float(np.median(moved)), 2)
        out["orientation_change_over_10deg"] = round(float((moved > 10).mean()), 4)
        if (moved > 10).mean() > 0.3:
            warnings.append(f"{(moved > 10).mean() * 100:.0f}% of particles moved more than 10 degrees from their "
                            "starting orientation: the start poses did not fit this map (global search may help)")
    if cls is None:
        plots3d.map_views_figure(out_("map_views.png"), maps, ["map"], tag, masks)
    else:
        plots3d.class_summary_figure(out_("map_views.png"), maps, class_labels,
                                     [int((cls == i).sum()) for i in range(len(maps))], class_res, tag)
    out["warnings"] = warnings
    d = os.path.dirname(out_("fsc.png"))
    out["figures"] = sorted(f for f in os.listdir(d) if f.endswith(".png") and ".light." not in f)
    return out


# ---------------------------------------------------------------------------------------------- registry
_PARTICLES = ("Extract", "Polish", "CtfRefine", "Subtract", "Select", "SymExpand")  # (SymExpand: its particles once)
_POSES = ("InitialModel", "Refine3D", "HeteroRefine", "Class3D", "LocalRefine", "Subtract", "SymExpand")
_POSES_CLS = _POSES + ("ModeMaps",)     # also clusters / bins along the modes of a Modes3D job (classes=)


def _res_key(s: dict) -> str:
    return (f"{_n(s.get('resolution_a'), 2)} Å {s.get('sym', '')}" + warn_mark(s)) if s.get("resolution_a") else ""


def _hetero_key(s: dict) -> str:
    c = s.get("classes") or []
    if not c:
        return ""
    best = min((x.get("resolution_solvent_mask_a") or x.get("resolution_a") or float("inf")) for x in c)
    return f"K{len(c)} · best {_n(best, 2)} Å" + warn_mark(s) if best < float("inf") else f"K{len(c)}"


def _class3d_key(s: dict) -> str:
    c = s.get("classes") or []
    if not c or not isinstance(c, list):
        return ""
    n = sum(x.get("particles", 0) for x in c) or 1
    big = max(c, key=lambda x: x.get("particles", 0))
    res = [x.get("resolution_solvent_mask_a") for x in c if x.get("resolution_solvent_mask_a")]
    return (f"K{len(c)} · largest {big.get('particles', 0) / n * 100:.0f}%"
            + (f" · best {_n(min(res), 2)} Å" if res else "") + warn_mark(s))


def registry() -> dict[str, JobSpec]:
    """Every particle job type (see jobspec.py): rerun, the dashboard's New job form, MCP new_job / job_types, the
    job list's key numbers and the reports all read this table."""
    from .class3d import Class3DSettings, class3d_job, run_class3d
    from .ctfrefine import CtfRefineSettings, ctf_refine_job, run_ctf_refine
    from .curate import CurateSettings, curate_job, run_curate
    from .masking import MaskCreateSettings, mask_job, run_mask
    from .polish import PolishSettings, polish_job, run_polish
    from .sharpen import SharpenSettings, run_sharpen, sharpen_job
    from .mapscore import MapScoreSettings, mapscore_job, run_mapscore
    from .subtract import (LocalRefineSettings, SubtractSettings, local_refine_job, run_local_refine, run_subtract,
                           subtract_job)
    from .symexpand import POSE_JOBS, SymExpandSettings, run_symexpand, symexpand_job
    from . import modes as md
    specs = [
        JobSpec("Curate", "micrographs kept for picking: limits on CTF fit, defocus, astigmatism and motion",
                CurateSettings, (Input("micrographs", ("MotionCtf",), doc="the motion-corrected, CTF-fitted micrographs"),),
                curate_job, run_curate,
                key=lambda s: f"{_n(s.get('accepted'))} of {_n(s.get('micrographs'))} mics" + warn_mark(s),
                figures=("curate.png",)),
        JobSpec("Class2D", "2D classification of the particles of an extraction or selection", Class2DSettings,
                (Input("particles", _PARTICLES, doc="particles to classify"),), class2d_job, run_class2d,
                key=lambda s: f"{_n(s.get('particles'))} ptcls · K{s.get('classes', '')}",
                figures=("gallery.png",)),
        JobSpec("InitialModel", "ab initio 3D from 2D classes and their particles", InitialModelSettings,
                (Input("class2d", ("Class2D",), doc="the 2D classification (classes chosen in settings)"),),
                initial_model_job, run_initial_model, key=_res_key,
                figures=("map_views.png", "fsc.png", "views.png", "reprojections.png")),
        JobSpec("Refine3D", "local 3D refinement with gold-standard halves", Refine3DSettings,
                (Input("particles", _PARTICLES, doc="particles (e.g. re-centred, full box)"),
                 Input("poses", _POSES_CLS, doc="starting orientations, matched by particle uid"),
                 Input("reference", _POSES_CLS, optional=True, doc="starting map (default: the poses job's)")),
                refine3d_job, run_refine3d, key=_res_key,
                figures=("map_views.png", "fsc.png", "postprocess.png", "localres.png", "views.png", "progress.png",
                         "particles.png", "nu.png")),
        JobSpec("HeteroRefine", "heterogeneous refinement: K maps, particles sorted among them (hard or soft)",
                HeteroRefineSettings,
                (Input("particles", _PARTICLES, doc="particles"),
                 Input("poses", tuple(x for x in _POSES_CLS if x != "SymExpand"),
                       doc="starting orientations, matched by particle uid")),
                hetero_refine_job, run_hetero_refine, key=_hetero_key,
                figures=("map_views.png", "views_classes.png", "fsc.png", "progress.png")),
        JobSpec("Class3D", "3D classification without alignment: K maps at the consensus poses (optionally focused "
                "on a mask, symmetry-expanded)", Class3DSettings,
                (Input("particles", _PARTICLES, doc="particles"),
                 Input("poses", _POSES, doc="the consensus orientations and shifts, matched by particle uid (kept fixed)"),
                 Input("reference", _POSES, optional=True,
                       doc="consensus map for init=consensus and signal subtraction (default: the poses job's)")),
                class3d_job, run_class3d, key=_class3d_key,
                figures=("map_views.png", "progress.png", "fsc.png", "views_classes.png")),
        JobSpec("Polish", "per-particle motion tracks and per-frame dose weights from a refinement", PolishSettings,
                (Input("refine", ("Refine3D",), doc="the refinement (its particles and map)"),),
                polish_job, run_polish,
                key=lambda s: f"{_n(s.get('particles'))} ptcls · {s.get('box', '')} px", figures=("polish.png", "tracks.png")),
        JobSpec("CtfRefine", "per-particle defocus against the particle's own half map (gold standard)",
                CtfRefineSettings, (Input("refine", ("Refine3D",), doc="the refinement (its particles, poses, maps)"),),
                ctf_refine_job, run_ctf_refine,
                key=lambda s: f"{_n(s.get('refined'))} ptcls · |Δdf| {_n(s.get('delta_abs_median_a'))} Å",
                figures=("ctfrefine.png",)),
        JobSpec("MaskCreate", "a mask from a map: automatic, or with a chosen threshold, regions and segments",
                MaskCreateSettings,
                (Input("map", ("Refine3D", *CLASS_JOBS, "InitialModel", "Sharpen", "LocalRefine"),
                       doc="the job whose map is masked (its half maps set the automatic threshold)"),),
                mask_job, run_mask,
                key=lambda s: (f"{_n((s.get('volume_a3') or 0) / 1e6, 2)} M Å³" + warn_mark(s)) if s.get("volume_a3") else "",
                figures=("mask.png",)),
        JobSpec("Subtract", "particle subtraction: the map outside a mask removed from each particle at its pose, "
                "optional symmetry expansion, re-centring on the mask and a smaller box", SubtractSettings,
                (Input("particles", _PARTICLES, doc="the particles to subtract"),
                 Input("poses", ("Refine3D", "HeteroRefine", "Class3D", "LocalRefine", "SymExpand"),
                       doc="the refinement whose poses, half sets and map are used (a SymExpand job: per copy)"),
                 Input("mask", ("MaskCreate",), doc="the region to keep (made on that refinement's map)")),
                subtract_job, run_subtract,
                key=lambda s: f"{_n(s.get('images'))} imgs · {s.get('box', '')} px"
                              + (f" · {s.get('sym_expand')}x" if (s.get("copies") or 1) > 1 else "") + warn_mark(s),
                figures=("subtract.png",)),
        JobSpec("LocalRefine", "focused local refinement: a masked region refined with limited searches around the "
                "input poses, rotations about the mask centre", LocalRefineSettings,
                (Input("particles", _PARTICLES, doc="particles (e.g. a Subtract job's)"),
                 Input("poses", _POSES, doc="starting poses (a SymExpand job: every copy, each image loaded once; a "
                       "Subtract job's for its expanded copies)"),
                 Input("mask", ("MaskCreate",), doc="the focus mask"),
                 Input("reference", _POSES, optional=True, doc="starting map (default: the poses job's)")),
                local_refine_job, run_local_refine,
                key=lambda s: (f"{_n(s.get('resolution_a'), 2)} Å focused (consensus "
                               f"{_n(s.get('consensus_region_resolution_a'), 2)})" + warn_mark(s))
                if s.get("resolution_a") else "",
                figures=("focus_fsc.png", "map_views.png", "fsc.png", "views.png", "progress.png", "particles.png")),
        JobSpec("SymExpand", "symmetry expansion: every particle once per operator of a point group, at its "
                "symmetry-related pose (no images written)", SymExpandSettings,
                (Input("poses", POSE_JOBS, doc="the refinement whose poses are expanded"),
                 Input("particles", _PARTICLES, optional=True, doc="the particles (default: the refinement's)")),
                symexpand_job, run_symexpand,
                key=lambda s: f"{_n(s.get('particles'))} ptcls x {s.get('copies', '')} ({s.get('sym', '')})"
                if s.get("rows") else "",
                figures=("symexpand.png",)),
        JobSpec("Sharpen", "post-processing: corrected masked FSC, B-factor, local resolution", SharpenSettings,
                (Input("refine", ("Refine3D", "LocalRefine", *CLASS_JOBS),
                       doc="the refinement (a HeteroRefine / Class3D / ModeMaps: set cls)"),),
                sharpen_job, run_sharpen,
                key=lambda s: (f"{_n(s.get('resolution_a'), 2)} Å · B {_n(s.get('b_factor'))}" + warn_mark(s))
                if s.get("resolution_a") else "",
                figures=("postprocess.png", "localres.png")),
        JobSpec("MapScore", "particles vs a 3D map: per-particle fit against a phase-randomised control, per 2D class "
                "share that fits (an objective class selection proposal)", MapScoreSettings,
                (Input("particles", _PARTICLES + ("Class2D",), doc="particles; a Class2D job also gets per-class shares"),
                 Input("map", _POSES, doc="the map the particles are scored against")),
                mapscore_job, run_mapscore,
                key=lambda s: f"{_n(s.get('fit', 0) * 100, 1)} % fit · {len(s.get('proposed_classes') or [])} classes"
                if s.get("particles") else "",
                figures=("mapscore.png",)),
        JobSpec("Modes3D", "modes of variation at fixed poses: the main ways the density changes between particles, "
                "each particle's coordinate along them, a volume series per mode", md.Modes3DSettings,
                (Input("particles", ("Extract", "Polish", "CtfRefine", "Select"),   # not Subtract: rows per (uid, copy)
                       doc="particles"),
                 Input("poses", ("Refine3D", "HeteroRefine"), doc="the consensus refinement (poses, shifts, halves)"),
                 Input("mask", ("MaskCreate",), optional=True,
                       doc="region the modes may use (default: the refinement's solvent mask)")),
                md.modes3d_job, md.run_modes3d, key=md.key,
                figures=("what_moves.png", "projections.png", "frames.png", "variance.png", "latent.png")),
        JobSpec("ModeMaps", "maps along the modes of a Modes3D job: a series per mode, reconstructions in bins along a "
                "mode, or of clusters of the particles' coordinates (classes for Select / Refine3D)",
                md.ModeMapsSettings,
                (Input("modes3d", ("Modes3D",), doc="the Modes3D job"),),
                md.mode_maps_job, md.run_mode_maps, key=md.maps_key,
                figures=("map_views.png", "latent.png", "fsc.png", "what_moves.png", "projections.png", "frames.png")),
    ]
    return {sp.type: sp for sp in specs}


def new_job(project: Project, job_type: str, inputs: dict, settings: dict | None = None, label: str = "") -> Job:
    """Make a job of any registered type from input job references and settings values (checked and converted
    against the settings dataclass; unknown names are an error). Returns the job; spec.run(project, job) runs it."""
    from .jobspec import coerce
    spec = registry().get(job_type)
    if spec is None:
        raise ValueError(f"unknown job type {job_type!r} (known: {sorted(registry())})")
    args = []
    for i in spec.inputs:
        ref = inputs.get(i.name)
        if ref in (None, "", 0):
            if not i.optional:
                raise ValueError(f"{job_type}: input {i.name!r} is required ({'/'.join(i.types)} job)")
            args.append(None)
            continue
        j = project.job(ref)
        if j.type not in i.types:
            raise ValueError(f"{job_type}: input {i.name!r} must be a {'/'.join(i.types)} job, not {j.name}")
        args.append(j.id)
    extra = set(inputs) - {i.name for i in spec.inputs}
    if extra:
        raise ValueError(f"{job_type}: unknown inputs {sorted(extra)}")
    s = spec.settings(**coerce(spec.settings, settings or {}))
    return spec.make(project, *args, s, label)


# ------------------------------------------------------------------------------------------------- rerun
def rerun(project: Project, ref, log=print):
    """Run a job again, RELION-style: a particle job (any type in registry()) becomes a new job with the same
    parameters and inputs (the old one stays, for comparison); a movie job (MotionCtf, Pick, Extract) carries on with
    the movies it does not have yet (e.g. new ones of a live Import). Returns (job, run) where run() does the work
    (call it on a worker thread)."""
    from . import pipeline as pl
    old = project.job(ref)
    if old.type in ("MotionCtf", "Pick", "Extract"):
        project.set_status(old, "running")
        return old, lambda: pl.run(project, [old], log=log)
    spec = registry().get(old.type)
    if spec is None:
        raise ValueError(f"{old.type} jobs are not run again (make a new one)")
    j = spec.make(project, *(old.params.get(i.name) for i in spec.inputs), settings_of(old, spec.settings))
    return j, lambda: spec.run(project, j, log=log)
