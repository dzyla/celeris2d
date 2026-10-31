"""Particle signal subtraction, symmetry expansion and focused local refinement (RELION Subtract + symmetry expansion
+ Refine3D with a mask and local searches; cryoSPARC Particle Subtraction, Symmetry Expansion, Local Refinement).

Subtract (a particle job). Inputs: particles, a poses job (Refine3D / HeteroRefine / LocalRefine) whose map is
subtracted, and a MaskCreate job: the region to KEEP. For every particle at its pose, with its CTF, the projection of
the map outside the mask is subtracted:

    particle - a_i g(k) CTF(k) e^{-i 2 pi k.t} P_A[map (1 - mask)](k)

The map is each particle's own half map (gold standard, as polishing and CTF refinement do: the half it was aligned
against), times the refinement's solvent mask, FSC-weighted and low-passed to the refinement's resolution (noise beyond
it is not subtracted). Scale: g(k) per shell (one curve for all particles, least squares of the full map's projections
against the particles: the map's units and B-factor are not the data's) and a_i per particle (least squares of the
projection of the whole map to the particle over a band, as cryoSPARC's per-particle scale; RELION's subtraction
scales by group / per-particle normalisation). Options: re-centre each new box on the mask's centre at the particle's
pose, and cut a smaller box from it (the region alone). Particles keep their uid (so their selections and their
gold-standard half), the table carries their poses (rot, tilt, psi, shifts relative to the new box centre, half), so
a Subtract job is also a poses job; any job that takes particles takes it (Class2D, Refine3D, HeteroRefine, Class3D).

Symmetry expansion (Subtract's sym_expand, the user's point group, never detected): every particle once per operator
S_k of the group, at pose A S_k. With a mask on one subunit, copy k keeps subunit S_k of the particle, re-centred on
it, the rest subtracted - each copy is a different image, so expansion is done where images are made. The copies keep
the particle's uid (lineage, selections, its half: all copies of a particle stay in one half set, or the FSC is
inflated); the table adds `copy` (0..S-1) and (uid, copy) is the row's key (jobs.match_rows). Copies share the
particle's centre, so duplicates are judged per uid and a particle's copies are kept or dropped together. Poses of
expanded particles must come from the Subtract job itself (or a refinement of it): a poses job with one pose per uid
cannot tell the copies apart (refused). Expansion without new images is the SymExpand job (symexpand.py): Subtract
takes it as poses and subtracts per copy, LocalRefine refines its copies on the original images.

LocalRefine (a 3D job): local refinement of the region in a focus mask, in the frame centred on the mask's centre
(the reference map and the mask are moved by -c, each pose's shift by +(A c)_xy), so orientations turn about the
mask's centre, not the box centre (cryoSPARC's fulcrum); the box can be the small subtracted box. The mask restricts
the references and the FSC (corrected for the mask, RELION-style); searches are limited to max_angle_deg / max_shift_a
around the input poses (the shift limit is on the mask centre's projected position); gold-standard halves are those of
the input poses. Maps are written back in the consensus frame and box (they overlay the input map), and the summary
compares the region's resolution with that of the consensus half maps in the same mask.

    s = subtract_job(project, particles, refine, mask, SubtractSettings(sym_expand="D2", box=128)); run_subtract(...)
    l = local_refine_job(project, s, s, mask, None, LocalRefineSettings()); run_local_refine(project, l)
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from .jobspec import params_of, settings_of
from .project import PARTICLE_JOBS, Job, Project, class_table, has_classes, logged


# --------------------------------------------------------------------------------------------- geometry
def mask_geometry(mask: np.ndarray, apix: float) -> tuple[np.ndarray, float]:
    """Centre of a mask (x, y, z in A from the box centre voxel n//2, the mask-weighted mean) and its radius (A,
    farthest voxel > 0.5 from that centre, the soft edge's half-width added)."""
    m = np.clip(np.asarray(mask, np.float64), 0, 1)
    n = m.shape[0]
    r = (np.arange(n) - n // 2) * apix
    w = m.sum()
    if w <= 0:
        raise ValueError("the mask is empty")
    c = np.array([(m.sum((0, 1)) * r).sum(), (m.sum((0, 2)) * r).sum(), (m.sum((1, 2)) * r).sum()]) / w
    z, y, x = np.nonzero(m > 0.5)
    if not len(z):
        z, y, x = np.nonzero(m > 0.5 * m.max())
    d = np.sqrt((r[x] - c[0]) ** 2 + (r[y] - c[1]) ** 2 + (r[z] - c[2]) ** 2)
    soft = 0.5 * apix * max(1, int(((m > 0.02) & (m < 0.98)).sum() ** (1 / 3) / 4))
    return c, float(d.max() + soft)


def _roll_crop(v: torch.Tensor, c_vox, b: int) -> torch.Tensor:
    """The b^3 box of a map v (n^3) centred on voxel n//2 + c_vox (z, y, x integers): the map moved by -c_vox, cut."""
    n = v.shape[0]
    v = torch.roll(v, shifts=tuple(-int(x) for x in c_vox), dims=(0, 1, 2))
    o = n // 2 - b // 2
    return v[o:o + b, o:o + b, o:o + b].contiguous()


def _uncrop_roll(v: torch.Tensor, c_vox, n: int) -> torch.Tensor:
    """Inverse of _roll_crop: a b^3 map placed back in an n^3 box at voxel n//2 + c_vox."""
    b = v.shape[0]
    out = torch.zeros((n, n, n), dtype=v.dtype, device=v.device)
    o = n // 2 - b // 2
    out[o:o + b, o:o + b, o:o + b] = v
    return torch.roll(out, shifts=tuple(int(x) for x in c_vox), dims=(0, 1, 2))


def _refinement_halves(project: Project, rj: Job, cls: int = 0):
    """Half maps, solvent mask, pixel size, FSC (per shell of a fsc_box_a box), fsc_box_a and resolution of a
    refinement: Refine3D / LocalRefine, or class cls (1-based) of a HeteroRefine."""
    import json
    import mrcfile
    pre = f"class{cls:02d}_" if has_classes(rj) else ""
    out = []
    for name in ("half1_unfil.mrc", "half2_unfil.mrc", "mask.mrc"):
        p = rj.find(pre + name)
        if p is None:
            raise ValueError(f"{rj.name} has no {pre + name}")
        with mrcfile.open(p, permissive=True) as f:
            out.append(np.asarray(f.data, np.float32))
            apix = float(f.voxel_size.x)
    summ = project.summary_of(rj)
    if has_classes(rj):
        d = json.load(open(rj.find(f"class{cls:02d}_fsc.json")))
        fsc = np.asarray([0.0 if x is None else x for x in d["curves"]["corrected"]], np.float64)
        box_a = float(d["box_a"])
        res = next((c.get("resolution_solvent_mask_a") or c.get("resolution_a") for c in class_table(summ)
                    if c["cls"] == cls), None) or float(d.get("resolution_a") or 8.0)
    else:
        fsc = np.asarray(summ["fsc_solvent"], np.float64)
        box_a = float(summ.get("fsc_box_a") or summ["box"] * summ["apix"])
        res = float(summ.get("resolution_solvent_mask_a") or summ.get("resolution_a") or 8.0)
    return out[0], out[1], out[2], apix, fsc, box_a, float(res)


def _fsc_weighted(v: np.ndarray, apix: float, b: int, apix_b: float, fsc: np.ndarray, fsc_box_a: float,
                  res_a: float, dev) -> torch.Tensor:
    """A map resampled to b^3 at apix_b, weighted by sqrt(2 FSC / (1 + FSC)) and cut at res_a (real, on dev)."""
    from .jobs import map_to_box
    v = map_to_box(v, apix, b, apix_b).to(dev)
    k = torch.fft.fftfreq(b, device=dev) * b
    kr = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2) / (b * apix_b)   # 1/A
    f_ = np.interp((kr * fsc_box_a).cpu().numpy(), np.arange(len(fsc)), np.clip(fsc, 0, 1), right=0.0)
    w = torch.as_tensor(np.sqrt(2 * f_ / (1 + f_)), device=dev, dtype=torch.float32) * (kr <= 1 / res_a)
    return torch.fft.ifftn(torch.fft.fftn(v) * w).real


# --------------------------------------------------------------------------------------------- Subtract
@dataclass
class SubtractSettings:
    sym_expand: str = "C1"          # symmetry expansion: each particle once per operator of this point group (the
                                    # user's choice, never detected; C1: none). Copy k keeps the masked region at its
                                    # k-th symmetry position (own image and pose; uid kept, column `copy`)
    subtract: bool = True           # subtract the map outside the mask (off: only expand / re-centre / re-box)
    recentre: bool = True           # centre each new box on the mask's centre at the particle's pose
    box: int = 0                    # new box (px, even; cut around the new centre); 0: the particles' box
    res_a: float = 0.0              # the map is low-passed to this before projection (A; 0: the refinement's resolution)
    scale: str = "particle"         # "particle": a per-shell scale (all particles) and a per-particle scale of the map's
                                    # projection to each image; "shell": the per-shell scale only
    res_lo_a: float = 40.0          # band of the scale fits: from this resolution (A) to res_a
    dtype: str = "float16"          # precision of the new particle stacks
    batch: int = 256                # particles per GPU batch (fewer if memory is short)
    gpus: str = "auto"              # GPU to use: "auto" (the one with most free memory) or a number like "1"


def subtract_job(project: Project, particles, poses, mask, s: SubtractSettings, label: str = "") -> Job:
    """particles: an Extract / Polish / CtfRefine / Select / Subtract / SymExpand job; poses: the job whose poses and
    map are used (Refine3D, HeteroRefine, Class3D, LocalRefine; or a SymExpand job: every copy, subtracted per copy);
    mask: a MaskCreate job, the region to keep."""
    p, q, m = project.job(particles), project.job(poses), project.job(mask)
    if q.type not in ("Refine3D", "HeteroRefine", "Class3D", "LocalRefine", "SymExpand"):
        raise ValueError(f"{q.name}: the poses and map come from a Refine3D, HeteroRefine, Class3D, LocalRefine or "
                         f"SymExpand job")
    if m.type != "MaskCreate":
        raise ValueError(f"{m.name} is not a MaskCreate job")
    if q.type == "SymExpand" and s.sym_expand.upper() != "C1":
        raise ValueError(f"{q.name} is already symmetry-expanded ({q.params.get('sym')}): set sym_expand to C1")
    ex = p if p.type in PARTICLE_JOBS else project.ancestor(p, PARTICLE_JOBS)
    if ex is None:
        raise ValueError(f"{p.name} holds no particles")
    from .abinitio import symmetry_ops
    symmetry_ops(s.sym_expand)                                      # a group it knows (Cn, Dn)
    if s.box and (s.box % 2 or s.box > int(ex.params.get("box", 1 << 30))):
        raise ValueError(f"box {s.box}: an even number of pixels, at most the particles' box")
    e = ex.params
    keep = {k: e[k] for k in ("picks", "micrographs", "box", "out_apix", "box_a_used", "particle_a", "dark") if k in e}
    if s.box:
        keep.update(box=s.box, box_a_used=round(s.box * float(e.get("out_apix", 1.0)), 2))
    xs = str(q.params.get("sym")).upper() if q.type == "SymExpand" else s.sym_expand.upper()
    tag = ("sub" if s.subtract else "rebox") + (f"_{xs}x" if xs != "C1" else "") + (f"_{s.box}px" if s.box else "")
    return project.new_job("Subtract", dict(params_of(s), particles=p.id, poses=q.id, mask=m.id, source=ex.id, **keep),
                           [x for i, x in enumerate([p, q, m]) if x.id not in [y.id for y in [p, q, m][:i]]],
                           label or f"{tag}_mask{m.id:03d}_on{q.id:03d}")


def _band_volumes(project: Project, rj: Job, cls_list, keep_mask_ref, b: int, apix: float, res_a: float | None,
                  dev, log=print):
    """Per class and half: (whole map, map outside the keep mask) as _BandVolumes projecting at box b / apix, and the
    resolution used. The keep mask is resampled to the refinement's grid."""
    from .abinitio import Volume, auto_pad
    from .jobs import load_mask
    from .polish import _BandVolume
    vols, used_res, info = {}, {}, {}
    for k in cls_list:
        h1, h2, smask, ap_r, fsc, fsc_box_a, res_r = _refinement_halves(project, rj, k)
        res = float(res_a or res_r)
        bb = min(b, int(math.ceil(2 * b * apix / res / 2)) * 2 + 8)
        apb = b * apix / bb
        keep = load_mask(project, keep_mask_ref, h1.shape[0], ap_r)
        pad = auto_pad(bb, 2 * len(cls_list), dev)        # 4 spectra per class (no accumulators)
        out = []
        for h in (h1, h2):
            pair = []
            for part in (h * smask, h * smask * (1 - keep)):
                v = _fsc_weighted(part, ap_r, bb, apb, fsc, fsc_box_a, res, dev)
                vol = Volume(bb, dev, pad=pad, grid_correct=pad == 1)
                vol.set_real(v)
                pair.append(_BandVolume(vol, b))
            out.append(pair)
        vols[k] = out
        used_res[k] = res
        info[k] = dict(band_box=bb, pad=pad, refinement_res_a=round(res_r, 2))
        log(f"[subtract] map{'' if k == 0 else f' class {k}'}: half maps at {res:.2f} A, band box {bb} (pad {pad})")
    return vols, used_res, info


def _rows(project: Project, job: Job):
    """The images to subtract and the rows made from them. Returns P (DataFrame, one row per image: stack, slice,
    centre, CTF, uid ...; sorted by micrograph and stack slice) and per row (the rows of an image next to each other, in
    image order): img (index into P), A (n, 3, 3), t (n, 2) A relative to the image's box centre, half, cls (1-based;
    0: not a class job), fit, copy (or None). One row per image, except for poses with several copies per particle
    (a SymExpand job): a row per copy, the image shared."""
    from .abinitio import euler_to_matrix
    from .jobs import _centres, map_job, match_rows
    pj, qj = project.job(job.params["particles"]), project.job(job.params["poses"])
    rj = map_job(project, qj)
    ex = project.job(job.params["source"])
    P = project.particles(ex)
    if pj.type not in PARTICLE_JOBS:
        uids = project.selected_uids(pj) if pj.type == "Select" else project.table(pj, "particles").read(["uid"])["uid"]
        P = P[np.isin(P["uid"].to_numpy(np.int64), np.asarray(uids, np.int64))]
    poses = project.table(qj, "particles").read()
    if "half" not in poses:
        raise ValueError(f"{qj.name} has no half sets (made by an older version: run the refinement again)")
    cp = P["copy"].to_numpy(np.int64) if "copy" in P else None
    pc = poses.get("copy")
    pu = np.asarray(poses["uid"], np.int64)
    if cp is None and pc is not None and (np.asarray(pc) > 0).any():      # expanded poses: every copy of an image
        o = np.lexsort((np.asarray(pc, np.int64), pu))
        uid = P["uid"].to_numpy(np.int64)
        lo, hi = np.searchsorted(pu[o], uid, "left"), np.searchsorted(pu[o], uid, "right")
        ok = hi > lo
        lo, cnt = lo[ok], (hi - lo)[ok]
        img = np.repeat(np.arange(len(lo)), cnt)
        r = o[np.repeat(lo, cnt) + np.arange(len(img)) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
        copy = np.asarray(pc, np.int64)[r]
    else:
        rows = match_rows(pu, pc, P["uid"].to_numpy(np.int64), cp)
        ok = rows >= 0
        r = rows[ok]
        img = np.arange(len(r))
        copy = None if cp is None else cp[ok]
    if not ok.any():
        raise ValueError(f"none of the particles of {pj.name} have poses in {qj.name}")
    P = P[ok].reset_index(drop=True)
    c_box = P[["x_a", "y_a"]].to_numpy(np.float64)[img]
    c_al = _centres(project, project.aligned_particles(qj), pu[r], None if pc is None else np.asarray(pc)[r]) \
        + np.stack([poses["shift_x_a"][r], poses["shift_y_a"][r]], 1)
    A = euler_to_matrix(torch.as_tensor(poses["rot"][r]), torch.as_tensor(poses["tilt"][r]),
                        torch.as_tensor(poses["psi"][r])).float()
    cls = (poses["cls"][r].astype(np.int64) + 1) if (has_classes(rj) and "cls" in poses) else np.zeros(len(r), np.int64)
    fit = poses["fit"][r].astype(np.float32) if "fit" in poses else np.ones(len(r), np.float32)
    o = np.lexsort((P["slice"].to_numpy(), P["movie_id"].to_numpy()))              # by micrograph, then stack slice
    P = P.iloc[o].reset_index(drop=True)
    inv = np.empty(len(o), np.int64)
    inv[o] = np.arange(len(o))
    img = inv[img]
    ro = np.argsort(img, kind="stable")                                            # rows follow their images
    return P, img[ro], A[torch.as_tensor(ro)], (c_al - c_box)[ro].astype(np.float32), \
        poses["half"][r][ro].astype(np.int64), cls[ro], fit[ro], None if copy is None else copy[ro]


@logged
def run_subtract(project: Project, job, log=print) -> dict:
    from . import abinitio as ab
    from .cli import pick_devices
    from .engine import TWO_PI, ctf_2d, freqs, gpu_free, release_cache
    from .jobs import load_mask, map_job
    from .pipeline import optics_of
    import mrcfile
    job = project.job(job)
    s = settings_of(job, SubtractSettings)
    t_start = time.time()
    qj, mj = project.job(job.params["poses"]), project.job(job.params["mask"])
    rj = map_job(project, qj)
    P, img0, A0, t0_, half0, cls0, fit0, copy0 = _rows(project, job)
    n0 = len(P)                                           # images read (particles, or images of expanded particles)
    b = int(P["box"].iloc[0])
    apix = float(P["apix"].iloc[0])
    if (P["box"] != b).any():
        raise ValueError("particles of different box sizes")
    box_a = b * apix
    nb_out = int(s.box or b)
    copies_in = copy0 is not None and bool((copy0 > 0).any())
    S = len(ab.symmetry_ops(s.sym_expand))
    if S > 1 and copies_in:
        raise ValueError(f"{project.job(job.params['particles']).name} is already symmetry-expanded")
    dev = torch.device(pick_devices(s.gpus, 1)[0] if s.gpus != "auto" else
                       f"cuda:{max(range(torch.cuda.device_count()), key=lambda i: gpu_free(i))}")
    torch.cuda.set_device(dev)
    # the region kept: centre and size in the refinement's frame
    with mrcfile.open(rj.find(("class01_" if rj.type == "HeteroRefine" else "") + "half1_unfil.mrc"), permissive=True) as f:
        n_r, ap_r = f.data.shape[0], float(f.voxel_size.x)
    keep_r = load_mask(project, mj.id, n_r, ap_r)
    c_mask, radius = mask_geometry(keep_r, ap_r)
    log(f"[subtract] {n0:,} images of {project.job(job.params['particles']).name} with poses in {qj.name}; mask "
        f"{mj.name}: centre ({c_mask[0]:.1f}, {c_mask[1]:.1f}, {c_mask[2]:.1f}) A from the map centre, radius "
        f"{radius:.0f} A; {len(img0) * S:,} rows ({S} per pose row); box {b} -> {nb_out} px at {apix:.3f} A")
    # rows: one per (image, copy); own expansion (sym_expand) as SymExpand's: row j S + k = pose row j at A_j S_k
    if S > 1:
        from .symexpand import expand_poses
        A, copy, j_ = expand_poses(A0, s.sym_expand)
        src_i, t, half_r, cls_r, fit_r = img0[j_], t0_[j_], half0[j_], cls0[j_], fit0[j_]
    else:
        A, copy, src_i, t, half_r, cls_r, fit_r = A0, copy0, img0, t0_, half0, cls0, fit0
    off = np.searchsorted(src_i, np.arange(n0 + 1))                 # rows of image i: off[i]:off[i + 1]
    S_eff = int(copy.max()) + 1 if copy is not None else 1
    c_t = torch.as_tensor(c_mask, dtype=torch.float32)
    p_off = ((A @ c_t)[:, :2].numpy() + t) if s.recentre else np.zeros_like(t)     # region's centre in the box (A)
    n = len(A)
    lim = 0.5 * nb_out * apix
    outside = float(np.mean(np.abs(p_off).max(1) + min(radius, lim) > 0.5 * box_a))
    if outside > 0.01:
        log(f"[subtract] warning: for {outside * 100:.1f}% of the particles the region reaches past the original box")
    if s.recentre and radius > lim:
        log(f"[subtract] warning: the region (radius {radius:.0f} A) is larger than the new box ({2 * lim:.0f} A)")
    # CTF and optics
    optics = optics_of(project, project.job(job.params["source"]))
    ctf_p = torch.zeros((n0, 6))
    ctf_p[:, 0] = torch.as_tensor(P["defocus_u"].to_numpy(np.float64))
    ctf_p[:, 1] = torch.as_tensor(P["defocus_v"].to_numpy(np.float64))
    ctf_p[:, 2] = torch.as_tensor(P["defocus_angle"].to_numpy(np.float64))
    ctf_p[:, 5] = 1.0
    cls_list = sorted(set(int(c) for c in cls_r))
    vols, used_res, vinfo = _band_volumes(project, rj, cls_list, mj.id, b, apix, s.res_a or None, dev, log)
    res_hi = max(used_res.values())
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= box_a / s.res_lo_a) & (kr <= box_a / res_hi)
    shell = torch.round(kr).long().clamp(max=b // 2)
    nsh = b // 2 + 1
    herm = torch.where(fx[None, :].expand_as(kr) > 0, 2.0, 1.0)
    ph = TWO_PI / box_a
    free = gpu_free(dev)
    bt = int(max(16, min(s.batch, free * 0.3 / (b * (b // 2 + 1) * 8 * 14))))

    def read_movie(rows):
        """Images (len(rows), b, b) of these particle rows (one stack file)."""
        st = project.abs(P["stack"].iloc[rows[0]])
        with mrcfile.mmap(st, permissive=True, mode="r") as f:
            d = f.data if f.data.ndim == 3 else f.data[None]
            return np.asarray(d[P["slice"].to_numpy()[rows]], np.float32)

    def models(X, ii, rr):
        """For rows rr (expanded) of the images X (of particles ii): CTF x shift x projection of the whole map and of
        the map outside the mask, each particle at its own half (and class)."""
        A_ = A[rr].to(dev)
        ctf = ctf_2d(ctf_p[ii].to(dev), optics, fy, fx, box_a)
        tt = torch.as_tensor(t[rr], device=dev)
        e = torch.exp(-1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
        Mf = torch.empty((len(rr), b, b // 2 + 1), dtype=torch.complex64, device=dev)
        Mo = torch.empty_like(Mf)
        hh, cc = half_r[rr], cls_r[rr]
        for k in cls_list:
            for h in (0, 1):
                sel = np.flatnonzero((hh == h) & (cc == k))
                if len(sel):
                    si = torch.as_tensor(sel, device=dev)
                    Mf[si] = vols[k][h][0].project(A_[si])
                    Mo[si] = vols[k][h][1].project(A_[si])
        f = ctf * e
        return Mf * f, Mo * f

    def fft(img):
        return torch.fft.rfft2(torch.as_tensor(img, device=dev), norm="forward")

    # ---- per-shell scale g(k), noise and the spread of per-particle scales, from a subset of whole micrographs
    rng = np.random.default_rng(0)
    movies = P["movie_id"].to_numpy()
    um = np.unique(movies)
    cache, taken = [], 0
    for m in rng.permutation(um):
        ii = np.flatnonzero(movies == m)
        cache.append((fft(read_movie(ii)).cpu(), ii))
        taken += off[ii[-1] + 1] - off[ii[0]]
        if taken >= 2000:
            break

    def subset():
        for Xc, ii in cache:
            X = Xc.to(dev)
            rr = np.arange(off[ii[0]], off[ii[-1] + 1])
            jj = src_i[rr] - ii[0]
            for i in range(0, len(rr), bt):
                r_, j_ = rr[i:i + bt], jj[i:i + bt]
                Xr = X[torch.as_tensor(j_, device=dev)]
                yield Xr, models(Xr, src_i[r_], r_)[0]

    def per_shell(v):                                             # (nb, b, bh) -> (nsh,) sums over the half plane
        return torch.zeros(nsh, device=dev, dtype=torch.float64).index_add_(
            0, shell.reshape(-1), (v * herm).sum(0).reshape(-1).double())
    num = den = 0
    for Xr, Mf in subset():                                       # g(k): least squares over all particles (a = 1)
        num = num + per_shell((Xr.conj() * Mf).real)
        den = den + per_shell(Mf.abs() ** 2)
    k_lo, k_hi = int(math.ceil(box_a / s.res_lo_a)), int(box_a / res_hi)
    g = (num / den.clamp_min(1e-30)).float()
    gs = g.clone()
    for k in range(1, nsh - 1):                                   # 3-shell running mean
        gs[k] = g[max(0, k - 1):k + 2].mean()
    g = gs.clamp(min=0)
    g[:max(1, k_lo)] = g[max(1, k_lo)]                            # below the band: the band's first shell
    g[0] = g[1]
    g[k_hi + 1:] = 0
    G = g[shell]
    rp = rc = 0
    for Xr, Mf in subset():                                       # noise: residual power per shell
        rp = rp + per_shell((Xr - Mf * G).abs() ** 2)
        rc = rc + per_shell(torch.ones_like(Xr.real))
    sig2 = (rp / rc.clamp_min(1)).float().clamp_min(1e-30)
    w = (band / sig2[shell]) * herm                               # noise-weighted band for the per-particle fits
    # per-particle scales are noisy (variance 1 / sum w |M|^2): shrunk towards the mean by their reliability
    # (empirical Bayes: true spread tau^2 = var(a) - mean noise variance)
    a_s, v_s = [], []
    for Xr, Mf in subset():
        M_ = Mf * G
        d_ = ((M_.abs() ** 2) * w).sum((1, 2)).clamp_min(1e-30)
        a_s.append(((Xr.conj() * M_).real * w).sum((1, 2)) / d_)
        v_s.append(1.0 / d_)
    a_s, v_s = torch.cat(a_s), torch.cat(v_s)
    a_mean = float(a_s.median())
    tau2 = max(float(a_s.var()) - float(v_s.mean()), 0.0)
    del cache
    log(f"[subtract] per-shell scale from {taken:,} images: {float(g[k_lo]):.3f} at {s.res_lo_a:.0f} A, "
        f"{float(g[(k_lo + k_hi) // 2]):.3f} at {box_a / max((k_lo + k_hi) // 2, 1):.1f} A, {float(g[k_hi]):.3f} at "
        f"{box_a / max(k_hi, 1):.1f} A; per-particle scale {a_mean:.3f}, spread {math.sqrt(tau2):.3f} (sd), "
        f"estimate noise {math.sqrt(float(v_s.mean())):.3f} (sd)")

    # ---- every particle: subtract, re-centre, cut, write per micrograph
    o0 = b // 2 - nb_out // 2
    a_all = np.zeros(n, np.float32)
    cc_all = np.zeros(n, np.float32)
    examples = []
    pool = ThreadPoolExecutor(2)
    jobs_ = [(m, np.flatnonzero(movies == m)) for m in um]
    fut = {i: pool.submit(read_movie, jobs_[i][1]) for i in range(min(2, len(jobs_)))}
    done = 0
    for q_, (m, ii) in enumerate(jobs_):
        img = fut.pop(q_).result()
        if q_ + 2 < len(jobs_):
            fut[q_ + 2] = pool.submit(read_movie, jobs_[q_ + 2][1])
        X = fft(img)
        rr = np.arange(off[ii[0]], off[ii[-1] + 1])
        jj = src_i[rr] - ii[0]
        outs = []
        for i in range(0, len(rr), bt):
            r_, j_ = rr[i:i + bt], jj[i:i + bt]
            Xr = X[torch.as_tensor(j_, device=dev)]
            Mf, Mo = models(Xr, src_i[r_], r_)
            Mf, Mo = Mf * G, Mo * G
            if s.scale == "particle":
                d_ = ((Mf.abs() ** 2) * w).sum((1, 2)).clamp_min(1e-30)
                a = ((Xr.conj() * Mf).real * w).sum((1, 2)) / d_
                lam = tau2 / (tau2 + 1.0 / d_)
                a = (a_mean + lam * (a - a_mean)).clamp(min=0)
            else:
                a = torch.ones(len(r_), device=dev)
            cc = (((Xr.conj() * Mf).real * w).sum((1, 2))
                  / (((Xr.abs() ** 2) * w).sum((1, 2)) * ((Mf.abs() ** 2) * w).sum((1, 2))).sqrt().clamp_min(1e-30))
            a_all[r_], cc_all[r_] = a.cpu().numpy(), cc.cpu().numpy()
            Y = Xr - a[:, None, None] * Mo if s.subtract else Xr
            if s.recentre:
                pp = torch.as_tensor(p_off[r_], device=dev, dtype=torch.float32)
                Y = Y * torch.exp(1j * ph * (fx[None, None, :] * pp[:, 0, None, None] + fy[None, :, None] * pp[:, 1, None, None]))
            y = torch.fft.irfft2(Y, s=(b, b), norm="forward")[:, o0:o0 + nb_out, o0:o0 + nb_out]
            outs.append(y.cpu().numpy())
            if len(examples) < 4 and done == 0:
                x0 = torch.fft.irfft2(Xr[:4], s=(b, b), norm="forward").cpu().numpy()
                mo = torch.fft.irfft2(a[:4, None, None] * Mo[:4], s=(b, b), norm="forward").cpu().numpy()
                for e_ in range(min(4, len(r_))):
                    examples.append((x0[e_], mo[e_], y[e_].cpu().numpy(), p_off[r_[e_]] / apix))
        imgs = np.concatenate(outs)
        c_new = P[["x_a", "y_a"]].to_numpy(np.float64)[src_i[rr]] + p_off[rr]
        pick = P[["x_a", "y_a"]].to_numpy(np.float64)[src_i[rr]]
        if "dx_a" in P:
            pick = pick - P[["dx_a", "dy_a"]].to_numpy(np.float64)[src_i[rr]]
        Ar = A[rr]
        rot, tilt, psi = ab.matrix_to_euler(Ar)
        cols = dict(score=cc_all[rr], fit=fit_r[rr], sub_scale=a_all[rr], half=half_r[rr].astype(np.uint8),
                    rot=rot.numpy(), tilt=tilt.numpy(), psi=psi.numpy(),
                    shift_x_a=(t[rr, 0] - p_off[rr, 0]).astype(np.float32),
                    shift_y_a=(t[rr, 1] - p_off[rr, 1]).astype(np.float32),
                    defocus_u=P["defocus_u"].to_numpy(np.float32)[src_i[rr]],
                    defocus_v=P["defocus_v"].to_numpy(np.float32)[src_i[rr]],
                    defocus_angle=P["defocus_angle"].to_numpy(np.float32)[src_i[rr]])
        if copy is not None:
            cols["copy"] = copy[rr].astype(np.float32)
        if has_classes(rj):
            cols["cls"] = (cls_r[rr] - 1).astype(np.float32)
        project.add_particles(job, int(m), c_new, imgs, apix, dtype=s.dtype,
                              uid=P["uid"].to_numpy(np.int64)[src_i[rr]], pick_xy_a=pick, columns=cols)
        done += 1
        if done % 50 == 0 or done == len(jobs_):
            log(f"[subtract] {done}/{len(jobs_)} micrographs, {(time.time() - t_start) / done:.2f} s each")
    pool.shutdown(cancel_futures=True)
    project.consolidate(job)
    seconds = time.time() - t_start
    region_a = 2 * radius
    n_part = len(np.unique(P["uid"].to_numpy()))
    xs = str(qj.params.get("sym")).upper() if qj.type == "SymExpand" else s.sym_expand.upper()
    # the particle size downstream jobs mask with (job_source: 1.25 x this, within the box): the region when re-centred
    project.update_params(job, particle_a=round(region_a if s.recentre else job.params.get("particle_a", region_a), 1),
                          box=nb_out, box_a_used=round(nb_out * apix, 2))
    figure(job.out("subtract.png"), examples, a_all, g.cpu().numpy(), box_a, k_lo, k_hi, nb_out,
           f"{job.name} · {n:,} images ({n_part:,} particles x {S_eff}) · mask {mj.name} · map {rj.name}")
    summ = dict(particles=int(n_part), images=int(n), copies=int(S_eff), sym_expand=xs, box=int(nb_out),
                apix=round(apix, 4), box_a=round(nb_out * apix, 1), subtracted=bool(s.subtract),
                recentred=bool(s.recentre), mask_centre_a=[round(float(x), 1) for x in c_mask],
                region_diameter_a=round(region_a, 1), map=rj.name, map_res_a=round(res_hi, 2),
                scale_mode=s.scale, scale_median=round(float(np.median(a_all)), 4),
                scale_spread_sd=round(math.sqrt(tau2), 4), scale_estimate_noise_sd=round(math.sqrt(float(v_s.mean())), 4),
                scale_p05_p95=[round(float(np.percentile(a_all, 5)), 4), round(float(np.percentile(a_all, 95)), 4)],
                shell_scale=[round(float(x), 4) for x in g.cpu()],
                fit_cc_median=round(float(np.median(cc_all)), 4), region_outside_box_share=round(outside, 4),
                seconds=round(seconds, 1), dtype=s.dtype, band_volumes=vinfo)
    warnings = []
    if outside > 0.01:
        warnings.append(f"for {outside * 100:.1f}% of the images the region reaches past the original box (wrapped)")
    if s.recentre and radius > lim:
        warnings.append(f"the region (radius {radius:.0f} A) is larger than the new box ({2 * lim:.0f} A)")
    if np.median(a_all) <= 0:
        warnings.append("the map's projections do not fit the particles (scale <= 0): check the poses job")
    summ["warnings"] = warnings
    project.set_status(job, "finished", summ)
    release_cache()
    log(f"[subtract] {job.name}: {n:,} images ({n_part:,} particles x {S_eff}), box {nb_out}, scale median "
        f"{summ['scale_median']} ({summ['scale_p05_p95'][0]}-{summ['scale_p05_p95'][1]}), {seconds:.0f} s")
    return summ


def figure(path, examples, a_all, g, box_a, k_lo, k_hi, nb_out, title_):
    from .plotstyle import themed
    themed(_draw)(path, examples, a_all, g, box_a, k_lo, k_hi, nb_out, title_)


def _draw(path, examples, a_all, g, box_a, k_lo, k_hi, nb_out, title_):
    """Examples (particle, model subtracted, result with the new box), the per-particle scales, the per-shell scale."""
    from .plotstyle import GRAY, T, grid_y, image_axes, norm_img, note, plt, title, top
    ne = min(3, len(examples))
    fig = plt.figure(figsize=(14, 2.6 * max(ne, 1) + 0.9))
    gs = fig.add_gridspec(max(ne, 1), 5, left=0.02, right=0.98, top=top(fig, 0.8), bottom=0.08, wspace=0.35, hspace=0.25)
    for r in range(ne):
        x0, mo, y, p = examples[r]
        b = x0.shape[0]
        lo, hi = np.percentile(x0, [1, 99.5])
        for c, (im, lab) in enumerate(((x0, "particle"), (mo, "subtracted model"), (y, f"result ({nb_out} px)"))):
            a = fig.add_subplot(gs[r, c])
            a.imshow(np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1) if c < 2 else norm_img(im), cmap=GRAY)
            image_axes(a)
            if c == 0:
                o = b // 2 - nb_out // 2
                a.add_patch(plt.Rectangle((o + p[0] - 0.5, o + p[1] - 0.5), nb_out, nb_out, fill=False,
                                          ec=T.series[1], lw=1.2))
            if r == 0:
                a.set_title(lab, fontsize=9)
    a = fig.add_subplot(gs[:, 3])
    v = a_all[np.isfinite(a_all)]
    if len(v):
        a.hist(np.clip(v, *np.percentile(v, [0.5, 99.5])), bins=60, color=T.series[0])
        note(a, f"median {np.median(v):.3f}", "upper right")
    a.set(xlabel="per-particle scale", ylabel="images", title="Scale of the map to each image")
    grid_y(a)
    a = fig.add_subplot(gs[:, 4])
    k = np.arange(len(g))
    sel = (k >= 1) & (k <= k_hi + 2)
    a.plot(k[sel] / box_a, g[sel], color=T.series[0])
    a.axvspan(k_lo / box_a, k_hi / box_a, color=T.ink3, alpha=0.12, lw=0)
    a.set(xlabel="spatial frequency (1/Å)", ylabel="scale", title="Per-shell scale (band shaded)")
    grid_y(a)
    title(fig, "Particle subtraction", title_)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- LocalRefine
@dataclass
class LocalRefineSettings:
    sym: str = "C1"                 # symmetry imposed on the focused region, the user's choice (expanded copies: C1)
    n_iter: int = 8                 # iterations of local search and reconstruction
    max_angle_deg: float = 20.0     # orientations searched only within this angle of the input pose (deg)
    max_shift_a: float = 10.0       # the mask centre's projected position searched only within this distance of the
                                    # input's (A)
    res_start_a: float = 0.0        # first alignment resolution (0: the input map's resolution, at least 8 A)
    step_deg: float = 0.0           # first angular step (deg; 0: from res_start and the region's size); halves once
                                    # < 30% of particles move
    min_step_deg: float = 0.3       # finest angular step (deg)
    n_local: int = 1                # local search: (2 n + 1) steps per rotation axis
    drop_frac: float = 0.1          # share of worst-fitting particles left out of each reconstruction
    max_res_a: float = 0.0          # finest resolution loaded (0: Nyquist)
    diameter_a: float = 0.0         # mask diameter for the particle images (0: from the particles - the region's size
                                    # for a re-centred Subtract job)
    sharpen: bool = True            # sharpen the final half maps with the job's own mask and compute their local
                                    # resolution (as a Sharpen job with mask "refine"; no separate job needed)
    local_res: bool = True          # ... including the local resolution
    mtf: str = ""                   # ... detector MTF (RELION .star) for the sharpening; empty: none
    mtf_apix: float = 0.0           # ... the pixel size (A) the MTF refers to (0: the map's)
    save_iterations: bool = False   # keep every iteration's FSC curves (data/..._iterations.json) and each particle's
                                    # angle / shift change and fit per iteration (particle table columns d_angle_itNN,
                                    # d_shift_itNN, fit_itNN): convergence, particles that keep jumping
    reference_class: int = 0        # the reference job has classes (3D classification, multi-class ab initio, ...): start
                                    # from this class's map (1-based); 0: the class most of the particles are in
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def local_refine_job(project: Project, particles, poses, mask, reference, s: LocalRefineSettings,
                     label: str = "") -> Job:
    """particles: any particle job (a Subtract job's subtracted, re-centred ones; or the original particles);
    poses: the job whose poses start it (Refine3D, HeteroRefine, LocalRefine, or the Subtract job - required for
    expanded copies); mask: the focus mask (a MaskCreate job on the consensus map); reference: the job whose map starts
    it (default: the poses job's map)."""
    p, q, m = project.job(particles), project.job(poses), project.job(mask)
    if m.type != "MaskCreate":
        raise ValueError(f"{m.name} is not a MaskCreate job")
    r = project.job(reference) if reference not in (None, "", 0) else q
    ins = [p, q, m] + ([r] if r.id not in (p.id, q.id, m.id) else [])
    return project.new_job("LocalRefine", dict(params_of(s), particles=p.id, poses=q.id, mask=m.id, reference=r.id),
                           [x for i, x in enumerate(ins) if x.id not in [y.id for y in ins[:i]]],
                           label or f"{s.sym}_mask{m.id:03d}_on{p.id:03d}")


def _first_below(f: torch.Tensor, thr: float) -> int:
    below = torch.nonzero(f[1:] < thr)
    return int(below[0]) + 1 if len(below) else len(f) - 1


def _halves_frame(rj: Job, cls: int, n_big: int, apix: float, c_vox, b: int):
    """A refinement's half maps resampled to apix in an n_big box, moved to the focus frame, cut to b^3 (or None)."""
    import mrcfile
    from .jobs import map_to_box
    pre = f"class{cls:02d}_" if has_classes(rj) else ""
    out = []
    for h in (1, 2):
        p = rj.find(f"{pre}half{h}_unfil.mrc")
        if p is None:
            return None
        with mrcfile.open(p, permissive=True) as f:
            v, ap = np.asarray(f.data, np.float32), float(f.voxel_size.x)
        out.append(_roll_crop(map_to_box(v, ap, n_big, apix), c_vox, b))
    return out


@logged
def run_local_refine(project: Project, job, log=print) -> dict:
    from . import abinitio as ab
    from .engine import release_cache
    from .jobs import (_posed_particles, _reference_map, _scale_to_data, _session, _write_results, iteration_columns,
                       load_mask, map_job, map_to_box, refinement_report)
    job = project.job(job)
    s = settings_of(job, LocalRefineSettings)
    pj, qj, mj = (project.job(job.params[k]) for k in ("particles", "poses", "mask"))
    rj = map_job(project, job.params.get("reference") or qj)
    t_wall = time.time()
    for d_ in range(torch.cuda.device_count() if torch.cuda.is_initialized() else 0):   # (a fresh process: from 0)
        torch.cuda.reset_peak_memory_stats(d_)
    # symmetry-expanded poses (a SymExpand job): one row per copy, the image of each row loaded once (img)
    src, poses, t_start, half, n_dup, dist, img = _posed_particles(project, job, pj, qj, (), log, expand=True)
    max_res = s.max_res_a or 2.0 * src.apix
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log, nyquist_margin=1.0, max_c64_gb=1.5)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    dev, b, box_a = clf.dev, clf.bmax, clf.box_a
    apix = box_a / b
    n = len(img)
    ref_cls = 0
    if has_classes(rj) and s.reference_class:
        ref_cls = int(s.reference_class)
    elif has_classes(rj):
        cl = poses["cls"] if "cls" in poses else np.zeros(n, int)
        ref_cls = int(np.bincount(cl).argmax()) + 1
    ref, ref_apix, _ = _reference_map(rj, ref_cls)
    n_big = max(b, int(math.ceil(ref.shape[0] * ref_apix / apix / 2)) * 2)
    vol_big = map_to_box(ref, ref_apix, n_big, apix)
    mask_big = torch.as_tensor(load_mask(project, mj.id, n_big, apix))
    c_a, radius = mask_geometry(mask_big.numpy(), apix)
    c_vox = tuple(int(round(x)) for x in (c_a[2] / apix, c_a[1] / apix, c_a[0] / apix))     # z, y, x voxels
    c_xyz = torch.tensor([c_vox[2], c_vox[1], c_vox[0]], dtype=torch.float32) * apix         # A, on the voxel grid
    fmask = _roll_crop(mask_big, c_vox, b)
    lost = 1 - float(fmask.sum() / mask_big.sum().clamp_min(1e-9))
    warnings = []
    if lost > 0.01:
        warnings.append(f"{lost * 100:.1f}% of the focus mask lies outside the {b} px box around its centre")
    init = _roll_crop(vol_big, c_vox, b) * fmask
    rsum = project.summary_of(rj)
    r_ref = rsum.get("resolution_a", 8.0) if not has_classes(rj) else \
        next((c["resolution_a"] for c in class_table(rsum) if c["cls"] == ref_cls), None) or 8.0
    res0 = s.res_start_a or max(8.0, float(r_ref))
    A = ab.euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                           torch.as_tensor(poses["psi"])).float()
    # the focus frame: the mask's centre is the origin; its projection is at t + (A c)_xy in each image
    t = torch.as_tensor(t_start).float() + (A @ c_xyz)[:, :2]
    log(f"[local] {n:,} particles{f' ({clf.N:,} images)' if n > clf.N else ''} of {pj.name}, poses of {qj.name}, map of {rj.name}; focus mask {mj.name}: centre "
        f"({c_xyz[0]:.1f}, {c_xyz[1]:.1f}, {c_xyz[2]:.1f}) A, radius {radius:.0f} A; box {b} at {apix:.3f} A; "
        f"limits {s.max_angle_deg:g} deg / {s.max_shift_a:g} A; mask centre {float(t.norm(dim=1).median()):.1f} A "
        f"from the box centre (median)")
    idx = torch.as_tensor(img)
    init = init * _scale_to_data(clf, idx, A, t, init, b, max(res0, 10.0))
    # first step: a turn that moves the region's edge by a quarter of the starting resolution (shift steps follow the
    # angular step: one step moves the edge by one shift step)
    step0 = s.step_deg or min(max(0.5, math.degrees(res0 / (4 * radius))), s.max_angle_deg / 4)
    t0 = time.time()
    A_start, t_in = A.clone(), t.clone()
    maps, A, t, z, fscs, res, vols = ab.refine_particles(
        clf, idx, A, t, init.to(dev), sym=s.sym, n_iter=s.n_iter, res_start=res0, step_start=step0,
        step_min=s.min_step_deg, n_local=s.n_local, drop_frac=s.drop_frac, box=b,
        half=None if half is None else torch.as_tensor(half), solvent_mask=fmask.to(dev),
        max_angle=s.max_angle_deg, max_shift=s.max_shift_a, shift_per_deg=radius * math.pi / 180,
        iter_stats=s.save_iterations, log=log)
    seconds = time.time() - t0
    h1, h2 = vols[0][0].real(), vols[0][1].real()
    fm = fmask.to(dev)
    fsc_unm = ab.fsc(vols[0][0], vols[0][1])
    # the consensus half maps in the same frame, box and mask: the region's resolution before. Both FSCs are corrected
    # from one shell, where the first of the two masked FSCs drops below 0.8: the unmasked FSC (RELION's choice of
    # shell) of the consensus is that of the whole particle, sharp where the region is not, and left the mask's own
    # correlation in the region uncorrected (synthetic test: a blurred domain read 6.7 A instead of ~8 A)
    cons = _halves_frame(rj, ref_cls, n_big, apix, c_vox, b)
    pairs = [(h1, h2)] + ([(cons[0].to(dev), cons[1].to(dev))] if cons is not None else [])
    k_r = min(_first_below(ab.fsc_maps(x * fm, y * fm), 0.8) for x, y in pairs)
    # shells past a half map's content (a refinement reconstructs a little beyond its resolution) get no FSC: the
    # masked FSC there is the mask's leakage only (as Sharpen)
    from .sharpen import _shells, content_end
    kk = _shells(b, dev)

    def masked_fsc(x, y):
        f = ab.corrected_masked_fsc(x, y, fm, k_r=k_r)
        k_end = content_end(x, kk)
        f[k_end + 1:] = float("nan")
        return f, ab.resolution_from_fsc(f[:k_end + 1], box_a)
    fsc_loc, res_loc = masked_fsc(h1, h2)
    fsc_cons, res_cons = None, float("nan")
    if cons is not None:
        fsc_cons, res_cons = masked_fsc(*pairs[1])
        # a mask whose edge cuts through the consensus density (the region of a larger particle) correlates the
        # consensus halves past their resolution, which phase randomisation does not remove (dz_2025_10, half of the
        # D2 particle, 5 A edge: corrected FSC back to 0.78 at 2.7 A); subtracted particles have no density there
        fc = fsc_cons.nan_to_num(0).cpu().numpy()
        k0 = int(np.argmax(fc < 0.143)) if (fc < 0.143).any() else len(fc)
        if k0 < len(fc) and fc[k0:].max() > 0.3:
            warnings.append(f"the consensus FSC in the focus mask rises again to {fc[k0:].max():.2f} at "
                            f"{box_a / (k0 + int(np.argmax(fc[k0:]))):.1f} A: the mask's edge cuts through density "
                            f"(a softer edge, or a mask that follows the density, gives a cleaner comparison)")
    del pairs
    # maps back in the consensus frame and box
    fw = torch.sqrt((2 * fsc_loc / (1 + fsc_loc)).clamp(0, 1).nan_to_num(0))
    avg = ab.Volume(b, dev, pad=vols[0][0].pad, ft=0.5 * (vols[0][0].ft + vols[0][1].ft), pre=vols[0][0].pre,
                    grid_correct=vols[0][0].grid_correct)
    avg.filter_shells(fw)
    nomask = avg.real()
    del avg
    back = lambda v: _uncrop_roll(v.float().cpu(), c_vox, n_big)
    for name, v in (("map.mrc", nomask * fm), ("map_nomask.mrc", nomask), ("half1_unfil.mrc", h1),
                    ("half2_unfil.mrc", h2)):
        ab._save_mrc(job.out(name), back(v), apix)
    ab._save_mrc(job.out("mask.mrc"), mask_big, apix)
    # poses in the consensus frame, relative to the particles' box centres
    A_c = A.cpu()
    t_out = t.cpu() - (A_c @ c_xyz)[:, :2]
    rot, tilt, psi = ab.matrix_to_euler(A_c)
    fit = ab.refine_particles.last_score.cpu().numpy().astype(np.float32)
    used = (ab.refine_particles.last_weights > 0).numpy()
    _write_results(project, job, src, img, dict(
        rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32), psi=psi.numpy().astype(np.float32),
        shift_x_a=t_out[:, 0].numpy().astype(np.float32), shift_y_a=t_out[:, 1].numpy().astype(np.float32),
        fit=fit, half=ab.refine_particles.last_half.numpy().astype(np.uint8), used=used,
        **({"copy": poses["copy"].astype(np.int16)} if "copy" in poses and "celCopy" not in src.parts else {}),
        **(iteration_columns(job, box_a) if s.save_iterations else {})))
    from . import diagnostics as dg
    moved = dg.orientation_change(A_start, A_c, "C1")
    dshift = (t.cpu() - t_in).norm(dim=1).numpy()
    tag = f"{job.name} · {n:,} particles · focus {mj.name} · {s.sym}"
    report = refinement_report(job, tag, s.sym, box_a, apix, A_start, A_c, t.cpu(), fit, used,
                               ab.refine_particles.last_history,
                               fsc=dict(corrected=fsc_loc, unmasked=fsc_unm),
                               resolution_a=res_loc, maps=[(nomask * fm).cpu().numpy()], masks=[fmask.numpy()])
    from .plots3d import fsc_figure
    curves = [("local refinement, focus mask", fsc_loc.cpu().numpy())]
    if fsc_cons is not None:
        curves.append(("consensus, focus mask", fsc_cons.cpu().numpy()))
    fsc_figure(job.out("focus_fsc.png"), curves, box_a, apix, tag, heading="Focused region: corrected masked FSC")
    warnings += report.pop("warnings", [])
    import resource
    gpu_peak = max(torch.cuda.max_memory_allocated(d_) for d_ in range(torch.cuda.device_count()))
    summ = dict(particles=int(n), images=int(clf.N), particles_used=int(used.sum()), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), sym=s.sym, resolution_a=round(res_loc, 2),
                resolution_solvent_mask_a=round(res_loc, 2),
                resolution_unmasked_a=round(ab.resolution_from_fsc(fsc_unm, box_a), 2),
                consensus_region_resolution_a=round(res_cons, 2) if res_cons == res_cons else None,
                consensus=rj.name, mask_centre_a=[round(float(x), 1) for x in c_xyz], region_radius_a=round(radius, 1),
                apix=round(apix, 4), box=n_big, local_box=b, fsc_box_a=round(box_a, 3), res_start_a=round(res0, 1),
                angle_change_deg_median=round(float(np.median(moved)), 2),
                shift_change_a_median=round(float(np.median(dshift)), 2),
                at_angle_limit=round(float((moved > 0.9 * s.max_angle_deg).mean()), 4),
                at_shift_limit=round(float((dshift > 0.9 * s.max_shift_a).mean()), 4),
                seconds=round(seconds, 1), wall_seconds=round(time.time() - t_wall, 1),
                gpu_peak_gb=round(gpu_peak / 1e9, 2),
                host_peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                **report, warnings=warnings,
                fsc_solvent=[None if x != x else round(float(x), 4) for x in fsc_loc.cpu()],
                fsc_consensus_region=None if fsc_cons is None else [None if x != x else round(float(x), 4)
                                                                    for x in fsc_cons.cpu()],
                map=project.rel(job.find("map.mrc")))
    from .jobs import _sharpen_job_maps
    sharp = _sharpen_job_maps(project, job, s, log=log)               # with the focus mask
    if sharp:
        summ["sharpened"] = sharp
    project.set_status(job, "finished", summ)
    log(f"[local] {job.name}: focused region {res_loc:.2f} A (consensus in the same mask {res_cons:.2f} A), orientation "
        f"change median {summ['angle_change_deg_median']} deg, shift {summ['shift_change_a_median']} A, {seconds:.0f} s")
    ses.clf = None
    release_cache()
    return summ
