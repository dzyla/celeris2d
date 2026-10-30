"""Symmetry expansion as a job of its own (RELION relion_particle_symmetry_expand, cryoSPARC Symmetry Expansion).

SymExpand writes no images: its table "particles" has one row per (uid, copy), copy k at pose A S_k for the operators
S_k of the user's point group (never detected; C1 refused), the particle's shift (relative to the box centres of its
particle job, as the refinement's) and its gold-standard half - every copy of a particle stays in that particle's half,
or the FSC is inflated. The copies are the same image: jobs that take a SymExpand job as poses (LocalRefine, Refine3D,
Class3D, Subtract) load each image once and give refine_particles / classify the image of every row
(jobs._posed_particles(expand=True): n images in memory, not n x copies). Subtract on a SymExpand job subtracts per
copy (copy k keeps the masked region at its k-th symmetry position: then each copy is an image of its own), and its own
sym_expand goes through expand_poses below, as Class3D's.

    j = symexpand_job(project, poses=53, particles=None, s=SymExpandSettings(sym="D2")); run_symexpand(project, j)
    l = local_refine_job(project, 52, j, mask, None, LocalRefineSettings())     # expanded, not subtracted
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .jobspec import params_of, settings_of
from .project import CLASS_JOBS, Job, Project, logged

POSE_JOBS = ("Refine3D", "HeteroRefine", "Class3D", "LocalRefine")


def expand_poses(A: torch.Tensor, sym: str):
    """Orientations A (n, 3, 3) expanded by the operators S_k of a point group: (n S, 3, 3) poses A_i S_k, row i S + k
    (a particle's copies next to each other, by copy number), the copy number k and the particle i of each row."""
    from .abinitio import symmetry_ops
    ops = symmetry_ops(sym).to(A.dtype)
    n, S = len(A), len(ops)
    return (A[:, None] @ ops[None]).reshape(-1, 3, 3), np.tile(np.arange(S), n), np.repeat(np.arange(n), S)


@dataclass
class SymExpandSettings:
    sym: str = ""                   # point group to expand by (the user's choice, required, never detected; C1 is no
                                    # expansion): every particle once per operator S_k, at pose A S_k
    classes: tuple = ()             # poses from a HeteroRefine / Class3D: only the particles of these 1-based classes
                                    # (empty: all; the class is kept per row)
    seed: int = 0                   # half split for poses without gold-standard halves (a particle's copies share it)


def symexpand_job(project: Project, poses, particles, s: SymExpandSettings, label: str = "") -> Job:
    """poses: the refinement whose poses are expanded (Refine3D, HeteroRefine, Class3D, LocalRefine); particles: the
    particles (default: that refinement's)."""
    from .abinitio import symmetry_ops
    q = project.job(poses)
    if q.type not in POSE_JOBS:
        raise ValueError(f"{q.name}: the poses come from a {', '.join(POSE_JOBS)} job")
    sym = str(s.sym or "").strip().upper()
    if not sym:
        raise ValueError("SymExpand: set sym, the point group to expand by (it is never detected)")
    if len(symmetry_ops(sym)) < 2:
        raise ValueError(f"SymExpand: {sym} has one operator - nothing to expand")
    p =project.job(particles) if particles not in (None, "", 0) else project.job(q.params["particles"])
    ins = [p] + ([q] if q.id != p.id else [])
    tag = f"{sym}" + (f"_cls{'-'.join(str(c) for c in s.classes)}" if s.classes else "")
    return project.new_job("SymExpand", dict(params_of(s), sym=sym, poses=q.id, particles=p.id), ins,
                           label or f"{tag}_on{q.id:03d}")


@logged
def run_symexpand(project: Project, job, log=print) -> dict:
    from . import abinitio as ab
    from . import diagnostics as dg
    from .jobs import _posed_particles, map_job
    from .plots3d import views_grid_figure
    job = project.job(job)
    s = settings_of(job, SymExpandSettings)
    t0 = time.time()
    pj, qj = project.job(job.params["particles"]), project.job(job.params["poses"])
    tab = project.table(qj, "particles")
    if "copy" in tab.columns and (np.asarray(tab.read(["copy"])["copy"]) > 0).any():
        raise ValueError(f"{qj.name} is already symmetry-expanded")
    src, poses, t, half, n_dup, dist = _posed_particles(project, job, pj, qj, tuple(s.classes), log)
    n = src.N
    A0 = ab.euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                            torch.as_tensor(poses["psi"])).float()
    A, copy, i = expand_poses(A0, s.sym)
    S = len(A) // max(n, 1)
    if half is None:
        half = (torch.randperm(n, generator=torch.Generator().manual_seed(s.seed + 1)) % 2).numpy()
        log(f"[symexpand] {qj.name} has no half sets: a random split (seed {s.seed}), a particle's copies together")
    rot, tilt, psi = ab.matrix_to_euler(A)
    fit = poses["fit"] if "fit" in poses else np.ones(n, np.float32)
    cols = dict(copy=copy.astype(np.int16), rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32),
                psi=psi.numpy().astype(np.float32), shift_x_a=t[i, 0].astype(np.float32),
                shift_y_a=t[i, 1].astype(np.float32), half=np.asarray(half)[i].astype(np.uint8),
                fit=np.asarray(fit, np.float32)[i], used=np.ones(len(i), bool))
    if "cls" in poses and qj.type in CLASS_JOBS:
        cols["cls"] = np.asarray(poses["cls"])[i].astype(np.int16)
    parts = src.parts
    project.table(job, "particles").write(dict(uid=parts["celUid"].to_numpy(np.int64)[i], **cols),
                                          parts["celMovie"].to_numpy(np.int64)[i])
    st0, st1 = dg.angular_stats(A0, "C1"), dg.angular_stats(A, "C1")
    r0, t0_, p0 = ab.matrix_to_euler(A0)
    views_grid_figure(job.out("symexpand.png"),
                      [("before", r0.numpy(), t0_.numpy(), st0), (f"{s.sym} expanded (x{S})", rot.numpy(), tilt.numpy(), st1)],
                      f"{job.name} · poses of {qj.name}",
                      heading="Symmetry expansion: viewing directions")
    summ = dict(particles=int(n), copies=int(S), rows=int(len(A)), sym=s.sym, poses=qj.name, particles_job=pj.name,
                map=map_job(project, qj).name, classes=list(s.classes), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), scf_before=st0.get("scf"), scf_after=st1.get("scf"),
                seconds=round(time.time() - t0, 1), warnings=[])
    project.set_status(job, "finished", summ)
    log(f"[symexpand] {job.name}: {n:,} particles x {S} ({s.sym}) = {len(A):,} rows, {summ['seconds']} s")
    return summ
