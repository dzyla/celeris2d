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
import os
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .project import PARTICLE_JOBS, Job, Project


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
    else:
        uids = project.table(job, "particles").read(["uid"])["uid"]
    src = job_source(ex.path, uids=uids)
    src.path = f"{project.root}#{ex.id:03d}#{job.id:03d}"    # by number: a renamed job keeps its saved 2D results
    return src


def _centres(project: Project, particle_job, uids: np.ndarray) -> np.ndarray:
    """Box centres (n, 2) in A of these particles in a particle job (by uid)."""
    t = project.table(particle_job, "particles").read(["uid", "x_a", "y_a"])
    o = np.argsort(t["uid"])
    pos = np.searchsorted(t["uid"][o], uids).clip(0, len(o) - 1)
    if not (t["uid"][o[pos]] == uids).all():
        raise ValueError(f"{project.job(particle_job).name} lacks some of the particles")
    return np.stack([t["x_a"][o[pos]], t["y_a"][o[pos]]], 1).astype(np.float64)


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


def _session(project: Project, job: Job, src, diameter_a, max_res_a, gpus, log, workdir=None):
    from .session import Session
    s = Session(workdir or os.path.join(job.path, "session"), log=log)
    _migrate_session(s.workdir, src.path)
    info = s.load(src, diameter=diameter_a or None, max_res=max_res_a, gpus=gpus)
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
    project.table(job, name).write(dict(uid=parts["celUid"].to_numpy(np.int64), **cols),
                                   parts["celMovie"].to_numpy(np.int64))


# ------------------------------------------------------------------------------------------------ Class2D
@dataclass
class Class2DSettings:
    n_classes: int = 50
    max_res_a: float = 0.0          # finest resolution used (0: Nyquist of the particles, but not finer than 6 A)
    diameter_a: float = 0.0         # mask (0: 1.25 x the extraction's particle size)
    seeds: int = 1                  # > 1: independent repeats, per-particle P(good) after selection
    gpus: str = "auto"


def class2d_job(project: Project, particles, s: Class2DSettings, label: str = "") -> Job:
    p = project.job(particles)
    return project.new_job("Class2D", dict(asdict(s), particles=p.id), [p], label or f"K{s.n_classes}_on{p.id:03d}")


def run_class2d(project: Project, job, log=print) -> dict:
    job = project.job(job)
    s = Class2DSettings(**{k: v for k, v in job.params.items() if k in Class2DSettings.__dataclass_fields__})
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
    project.write_mrc(os.path.join(job.path, "classes.mrcs"), r.images, ses.clf.box_a / r.images.shape[-1])
    shutil.copy(r.gallery_path, os.path.join(job.path, "gallery.png"))
    summ = dict(particles=int(len(r.idx)), classes=int(len(r.pops)), mask_a=round(ses.clf.p.diameter_a, 1),
                max_res_a=round(max_res, 2), seconds=round(time.time() - t0, 1),
                table=[dict(cls=k + 1, particles=int(r.pops[k]), conf=round(float(r.conf[k]), 3),
                            res_a=round(float(r.res_a[k]), 1) if np.isfinite(r.res_a[k]) else None)
                       for k in np.argsort(-r.pops)])
    project.set_status(job, "finished", summ)
    ses.clf = None
    torch.cuda.empty_cache()
    return summ


def select_classes(project: Project, class2d, classes: list[int], label: str = "", note: str = "") -> Job:
    """A Select job with the particles of these classes (1-based numbers, as in the gallery) of a Class2D job."""
    c = project.job(class2d)
    t = project.table(c, "particles").read(["uid", "cls"])
    keep = np.isin(t["cls"], np.asarray(classes, int) - 1)
    chosen = sorted(int(k) for k in classes)
    K = len(project.summary_of(c).get("table", [])) or int(t["cls"].max()) + 1
    job = project.select(c, t["uid"][keep], label=label or f"{len(classes)}cls_of{c.id:03d}",
                         note=note or f"classes {chosen} of {c.name}", rejected=t["uid"][~keep])
    rejected = [k for k in range(1, K + 1) if k not in chosen]
    project.update_params(job, classes=chosen, rejected_classes=rejected)
    cm = os.path.join(c.path, "classes.mrcs")
    if os.path.exists(cm):
        import mrcfile
        with mrcfile.open(cm, permissive=True) as f:
            imgs, apix = np.asarray(f.data, np.float32), float(f.voxel_size.x)
        pops = np.bincount(t["cls"], minlength=len(imgs))
        for name, ks in (("selected_classes.mrcs", chosen), ("rejected_classes.mrcs", rejected)):
            if ks:
                project.write_mrc(os.path.join(job.path, name), imgs[np.asarray(ks) - 1], apix)
        _selection_figure(os.path.join(job.path, "selection.png"), imgs, pops, set(chosen),
                          f"{job.name}: {int(keep.sum()):,} kept ({len(chosen)} classes), "
                          f"{int((~keep).sum()):,} rejected ({len(rejected)} classes) of {c.name}")
    return job


def _selection_figure(path, imgs, pops, chosen, title, ncol=10):
    """Class images by population, kept ones framed green, rejected ones red and dimmed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-pops)
    nrow = (len(order) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.3, nrow * 1.45 + 0.4), dpi=100, squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, k in zip(axs.ravel(), order):
        im = imgs[k]
        lo, hi = np.percentile(im, [1, 99])
        ok = k + 1 in chosen
        a.imshow(im, cmap="gray", vmin=lo, vmax=hi, alpha=1.0 if ok else 0.45)
        a.axis("on")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#2e9e44" if ok else "#c8372d")
            sp.set_linewidth(3)
        a.set_title(f"{k + 1}  ({pops[k]:,})", fontsize=7, color="black" if ok else "#c8372d")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- InitialModel
@dataclass
class InitialModelSettings:
    sym: str = "C1"                 # the user's choice; never detected
    classes: tuple = ()             # 1-based classes of the Class2D job (empty: all populated classes)
    n_models: int = 1
    seed: int = 0
    gpus: str = "auto"


def initial_model_job(project: Project, class2d, s: InitialModelSettings, label: str = "") -> Job:
    c = project.job(class2d)
    return project.new_job("InitialModel", dict(asdict(s), class2d=c.id), [c], label or f"{s.sym}_from{c.id:03d}")


def run_initial_model(project: Project, job, log=print) -> dict:
    """Ab initio (abinitio.run) from the Class2D job's classes and their particles; the 2D run is reloaded from that
    job's session (same particles, same settings), so nothing is classified again."""
    from . import abinitio as ab
    job = project.job(job)
    s = InitialModelSettings(**{k: v for k, v in job.params.items() if k in InitialModelSettings.__dataclass_fields__})
    c = project.job(job.params["class2d"])
    cs = Class2DSettings(**{k: v for k, v in c.params.items() if k in Class2DSettings.__dataclass_fields__})
    src = particle_source(project, c.params["particles"])
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
    prm = ab.AbinitioParams(sym=s.sym, n_models=s.n_models, seed=s.seed)
    summ = ab.run(ses.clf, r, job.path, prm, classes=mask, log=log, src=None)
    z = np.load(os.path.join(job.path, "poses.npz"))
    _write_results(project, job, src, z["idx"], dict(
        rot=z["rot"].astype(np.float32), tilt=z["tilt"].astype(np.float32), psi=z["psi"].astype(np.float32),
        shift_x_a=z["shift_a"][:, 0].astype(np.float32), shift_y_a=z["shift_a"][:, 1].astype(np.float32),
        used=z["used"].astype(bool), model=z["model"].astype(np.int16)))
    os.remove(os.path.join(job.path, "poses.npz"))                     # the sharded table replaces it
    keep = {k: summ[k] for k in ("particles", "particles_used", "resolution_a", "resolution_solvent_mask_a",
                                 "resolution_unmasked_a", "apix", "box", "seconds_total")}
    keep.update(sym=s.sym, map="map.mrc", duplicates_removed=n_dup, duplicate_distance_a=round(dist, 1))
    project.set_status(job, "finished", keep)
    ses.clf = None
    torch.cuda.empty_cache()
    return keep


# ---------------------------------------------------------------------------------------------- Refine3D
@dataclass
class Refine3DSettings:
    sym: str = "C1"
    n_iter: int = 10
    res_start_a: float = 0.0        # first alignment resolution (0: the reference job's resolution, at least 8 A)
    max_res_a: float = 0.0          # finest resolution loaded (0: Nyquist)
    n_local: int = 1
    drop_frac: float = 0.1
    global_iters: int = 0           # first iterations that also search all orientations (for starts with pose errors)
    global_res_a: float = 12.0      # resolution of that search
    diameter_a: float = 0.0
    gpus: str = "auto"


def refine3d_job(project: Project, particles, poses, reference, s: Refine3DSettings, label: str = "") -> Job:
    """particles: an Extract job (e.g. re-centred, full box); poses: the job whose per-particle orientations start
    the refinement (InitialModel / Refine3D, matched by uid); reference: the job whose map.mrc starts it."""
    p, q, r = project.job(particles), project.job(poses), project.job(reference)
    return project.new_job("Refine3D", dict(asdict(s), particles=p.id, poses=q.id, reference=r.id),
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
    from .abinitio import Volume
    from .engine import TWO_PI, ctf_2d, freqs
    dev, box_a = clf.dev, clf.box_a
    v = Volume(b, dev)
    v.set_real(vol.to(dev))
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


def run_refine3d(project: Project, job, log=print) -> dict:
    import mrcfile
    from . import abinitio as ab
    job = project.job(job)
    s = Refine3DSettings(**{k: v for k, v in job.params.items() if k in Refine3DSettings.__dataclass_fields__})
    pj, qj, rj = (project.job(job.params[k]) for k in ("particles", "poses", "reference"))
    poses = project.table(qj, "particles").read()
    if "used" in poses:
        poses = {k: v[poses["used"]] for k, v in poses.items()}
    src = particle_source(project, pj)
    uid = src.parts["celUid"].to_numpy(np.int64)
    order = np.argsort(poses["uid"])
    pos = np.searchsorted(poses["uid"][order], uid)
    ok = pos < len(order)
    ok[ok] = poses["uid"][order[pos[ok]]] == uid[ok]
    src.subset(np.flatnonzero(ok))
    row = order[pos[ok]]
    # starting shift = aligned centre (centre in the extraction the poses were found on + their shift) - centre in
    # this extraction: right for any pair of extractions (re-centred on these poses, on other ones, or not at all)
    c_new = _centres(project, pj, poses["uid"][row])
    c_al = _centres(project, project.aligned_particles(qj), poses["uid"][row]) + np.stack(
        [poses["shift_x_a"][row], poses["shift_y_a"][row]], 1)
    t_start = (c_al - c_new).astype(np.float32)
    # gold-standard halves: those of the refinement the poses come from (a particle stays in the half it was aligned
    # / polished against); duplicates (aligned centres closer than duplicate_distance) are left out, best fit kept
    half = poses["half"][row].astype(np.int64) if "half" in poses else None
    fit = poses["fit"][row] if "fit" in poses else (poses["maxprob"][row] if "maxprob" in poses else np.ones(len(row)))
    dist = duplicate_distance(project, pj)
    keep = drop_duplicates(c_al, src.parts["celMovie"].to_numpy(np.int64), fit, dist)
    n_dup = int((~keep).sum())
    if n_dup:
        log(f"[refine3d] {n_dup} duplicate particles left out (aligned centres < {dist:.0f} A apart; best fit kept)")
        src.subset(np.flatnonzero(keep))
        row, t_start = row[keep], t_start[keep]
        half = None if half is None else half[keep]
    src.path = f"{project.root}#{pj.id:03d}#{job.id:03d}"
    max_res = s.max_res_a or 2.0 * src.apix
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    dev, b, box_a = clf.dev, clf.bmax, clf.box_a
    A = ab.euler_to_matrix(torch.as_tensor(poses["rot"][row]), torch.as_tensor(poses["tilt"][row]),
                           torch.as_tensor(poses["psi"][row])).float()
    t = torch.as_tensor(t_start).float()
    with mrcfile.open(os.path.join(rj.path, "map_nomask.mrc" if os.path.exists(os.path.join(rj.path, "map_nomask.mrc"))
                                   else "map.mrc"), permissive=True) as f:
        ref, ref_apix = np.array(f.data, np.float32), float(f.voxel_size.x)
    res0 = s.res_start_a or max(8.0, float(project.summary_of(rj).get("resolution_a", 8.0)))
    vol = map_to_box(ref, ref_apix, b, box_a / b)
    idx = torch.arange(clf.N)
    vol *= _scale_to_data(clf, idx, A, t, vol, b, max(res0, 10.0))
    init = ab.Volume(b, dev)
    init.set_real(vol.to(dev))
    t0 = time.time()
    maps, A, t, z, fscs, res, vols = ab.refine_particles(clf, idx, A, t, init, sym=s.sym, n_iter=s.n_iter,
                                                         res_start=res0, n_local=s.n_local, drop_frac=s.drop_frac,
                                                         global_at=tuple(range(1, s.global_iters + 1)),
                                                         global_res=s.global_res_a, box=b,
                                                         half=None if half is None else torch.as_tensor(half), log=log)
    apix = box_a / b
    h1, h2 = vols[0][0].real(), vols[0][1].real()
    smask = ab.solvent_mask(maps[0], apix)
    fsc_sol = ab.corrected_masked_fsc(h1, h2, smask)
    fsc_unm = ab.fsc(vols[0][0], vols[0][1])
    for name, v in (("map.mrc", maps[0] * smask), ("map_nomask.mrc", maps[0]), ("mask.mrc", smask),
                    ("half1_unfil.mrc", h1), ("half2_unfil.mrc", h2)):
        ab._save_mrc(os.path.join(job.path, name), v, apix)
    rot, tilt, psi = ab.matrix_to_euler(A.cpu())
    _write_results(project, job, src, np.arange(clf.N), dict(
        rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32), psi=psi.numpy().astype(np.float32),
        shift_x_a=t[:, 0].cpu().numpy().astype(np.float32), shift_y_a=t[:, 1].cpu().numpy().astype(np.float32),
        fit=ab.refine_particles.last_score.cpu().numpy().astype(np.float32),
        half=ab.refine_particles.last_half.numpy().astype(np.uint8)))
    _fsc_figure(os.path.join(job.path, "fsc.png"), box_a,
                [("soft sphere mask", fscs[0].cpu().numpy()), ("no mask", fsc_unm.cpu().numpy()),
                 ("solvent mask", fsc_sol.cpu().numpy())], f"{job.name}: {clf.N:,} particles, {s.sym}")
    summ = dict(particles=int(clf.N), duplicates_removed=n_dup, duplicate_distance_a=round(dist, 1), sym=s.sym, resolution_a=round(float(res[0]), 2),
                resolution_solvent_mask_a=round(ab.resolution_from_fsc(fsc_sol, box_a), 2),
                resolution_unmasked_a=round(ab.resolution_from_fsc(fsc_unm, box_a), 2),
                apix=round(apix, 4), box=b, res_start_a=round(res0, 1), seconds=round(time.time() - t0, 1),
                fsc_solvent=[round(float(x), 4) for x in fsc_sol.cpu()], map="map.mrc")
    project.set_status(job, "finished", summ)
    ses.clf = None
    torch.cuda.empty_cache()
    return summ


def _fsc_figure(path, box_a, curves, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for lab, c in curves:
        k = np.arange(len(c)) / box_a
        ax.plot(k[1:], c[1:], label=lab)
    ax.axhline(0.143, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("resolution (A)")
    ticks = [t for t in (30, 15, 10, 7, 5, 4, 3.5, 3, 2.5) if 1 / t <= (len(curves[0][1]) - 1) / box_a]
    ax.set_xticks([1 / t for t in ticks], [f"{t:g}" for t in ticks])
    ax.set_ylabel("FSC")
    ax.set_ylim(-0.05, 1.02)
    ax.legend(frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- rerun
def rerun(project: Project, ref, log=print):
    """Run a job again, RELION-style: a particle job (Class2D, InitialModel, Refine3D, Polish) becomes a new job with
    the same parameters and inputs (the old one stays, for comparison); a movie job (MotionCtf, Pick, Extract)
    carries on with the movies it does not have yet (e.g. new ones of a live Import). Returns (job, run) where
    run() does the work (call it on a worker thread)."""
    from . import pipeline as pl
    from .polish import PolishSettings, polish_job, run_polish
    old = project.job(ref)
    q = old.params

    def pick(cls):
        return cls(**{k: v for k, v in q.items() if k in cls.__dataclass_fields__})
    if old.type == "Class2D":
        j = class2d_job(project, q["particles"], pick(Class2DSettings))
        return j, lambda: run_class2d(project, j, log=log)
    if old.type == "InitialModel":
        s = pick(InitialModelSettings)
        s.classes = tuple(s.classes)
        j = initial_model_job(project, q["class2d"], s)
        return j, lambda: run_initial_model(project, j, log=log)
    if old.type == "Refine3D":
        j = refine3d_job(project, q["particles"], q["poses"], q["reference"], pick(Refine3DSettings))
        return j, lambda: run_refine3d(project, j, log=log)
    if old.type == "Polish":
        j = polish_job(project, q["refine"], pick(PolishSettings))
        return j, lambda: run_polish(project, j, log=log)
    if old.type in ("MotionCtf", "Pick", "Extract"):
        project.set_status(old, "running")
        return old, lambda: pl.run(project, [old], log=log)
    raise ValueError(f"{old.type} jobs are not run again (make a new one)")
