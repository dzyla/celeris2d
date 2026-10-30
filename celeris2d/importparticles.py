"""Particles from another program into a project: a RELION .star or a cryoSPARC .cs / .csg (any source open_source
reads) becomes the same three jobs our own chain makes - Import (the micrographs, registered by path; the files need
not be here), MotionCtf (their pixel size, shape and CTF; params imported=True) and Extract (the particles, copied into
the project's data folder as white-protein stacks, optionally Fourier-binned, with per-particle defocus; imported=True).
Every particle job (Class2D, Select, InitialModel, Refine3D, ...) then takes the Extract job as it takes ours.

Micrographs and coordinates are required (rlnMicrographName + rlnCoordinateX/Y, or cryoSPARC location/*): duplicate
removal, re-centring and the particle keys need them. Imported jobs cannot be purged or rebuilt (their source is
outside the project). Poses in the input (alignments3D, rlnAngle*) are not imported."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .project import Job, Project, logged


@dataclass
class ImportParticlesSettings:
    path: str = ""                  # RELION particles .star, or cryoSPARC particles .cs / .csg (e.g. an extraction's
                                    # J20_particles.csg): micrograph names and coordinates are required
    particle_a: float = 0.0         # longest particle dimension (A): the 2D mask is 1.25 x this (0: 60 % of the box)
    particle_apix: float = 0.0      # Fourier-bin the particles to this pixel size when copying (0: keep the input's)
    invert: str = "auto"            # auto (decided from the data) | yes (protein dark in the input) | no
    project_dir: str = ""           # where the input's relative image paths start (empty: found from the file)


def import_job(project: Project, s: ImportParticlesSettings, label: str = "") -> Job:
    if not s.path:
        raise ValueError("path: a .star, .cs or .csg file of particles")
    base = os.path.splitext(os.path.basename(s.path))[0]
    return project.new_job("ImportParticles", dict(asdict(s)), [], label or base)


@logged
def run_import(project: Project, job, log=print) -> dict:
    """The queued ImportParticles job: the import (its Import / MotionCtf / Extract jobs) and a summary naming them."""
    job = project.job(job)
    p = job.params
    ex = import_particles(project, p["path"], particle_a=p["particle_a"], particle_apix=p["particle_apix"],
                          invert=p["invert"], project_dir=p["project_dir"] or None, label=job.label, log=log)
    mc = project.job(ex.params["micrographs"])
    summ = dict(project.summary_of(ex), particles_job=ex.name, micrographs_job=mc.name)
    project.set_status(job, "finished", summ)
    return summ


def _micrographs(src) -> pd.DataFrame:
    """Per particle: micrograph path (absolute where it can be resolved), its pixel size and shape (ny, nx), and the
    particle's position in A."""
    p = src.parts
    if not {"rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"} <= set(p.columns):
        raise ValueError("the particles have no micrograph names / coordinates (needed for duplicate removal, "
                         "re-centring and particle identity)")
    if getattr(src, "kind", "") == "cryosparc":
        cs = src.cs
        mic_apix = cs["location/micrograph_psize_A"].astype(np.float64)
        shp = cs["location/micrograph_shape"].astype(np.int64)
        ny, nx = shp[:, 0], shp[:, 1]
    else:
        mic_apix = np.full(len(p), float(getattr(src, "_mic_apix", src.apix)))
        ny = nx = None
    name = p["rlnMicrographName"].astype(str).to_numpy()
    path = np.array([n if os.path.isabs(n) else os.path.join(src.root, n) for n in name], dtype=object)
    df = pd.DataFrame(dict(mic=path, mic_apix=mic_apix, x_a=p["rlnCoordinateX"].to_numpy(float) * mic_apix,
                           y_a=p["rlnCoordinateY"].to_numpy(float) * mic_apix))
    if ny is None:                              # RELION: the header if the micrograph is here, else the extent of the picks
        ny, nx = np.zeros(len(p), np.int64), np.zeros(len(p), np.int64)
        for m, g in df.groupby("mic").groups.items():
            shape = None
            if os.path.exists(m):
                import mrcfile
                with mrcfile.mmap(m, mode="r", permissive=True) as f:
                    shape = f.data.shape[-2:]
            if shape is None:
                a = df.loc[g, "mic_apix"].iloc[0]
                shape = (int(df.loc[g, "y_a"].max() / a) + 1, int(df.loc[g, "x_a"].max() / a) + 1)
            ny[g], nx[g] = shape
    df["ny"], df["nx"] = ny, nx
    return df


def _flip_sign(src, rows: np.ndarray, radius_a: float) -> bool:
    """Is protein dark in these images? The phase-flipped average of a sample (centred particles: a blob) is compared
    between the centre and a ring at the particle radius, as the 2D engine does."""
    from .engine import ctf_2d, freqs
    rows = np.sort(rows)
    imgs = np.concatenate([src._read(fi, rows[src.file_idx[rows] == fi])[1]
                           for fi in np.unique(src.file_idx[rows])])
    order = np.concatenate([rows[src.file_idx[rows] == fi] for fi in np.unique(src.file_idx[rows])])
    b = imgs.shape[-1]
    fy, fx = freqs(b, "cpu")
    ft = torch.fft.rfft2(torch.from_numpy(imgs), norm="forward")
    c = ctf_2d(torch.as_tensor(src.ctf[order], dtype=torch.float32), src.optics, fy, fx, b * src.apix)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    avg = (torch.sign(c) * ft).mean(0) * torch.exp(-0.5 * (kr / (b * src.apix / 30.0)) ** 2)   # ~30 A low-pass
    img = torch.fft.irfft2(avg, s=(b, b), norm="forward")
    r = torch.arange(b, dtype=torch.float32) - b // 2                 # the particle sits at the box centre
    rr = torch.sqrt(r[:, None] ** 2 + r[None, :] ** 2) * src.apix
    centre = img[rr < 0.3 * radius_a].mean()
    ring = img[(rr > 0.9 * radius_a) & (rr < 1.1 * radius_a)].mean()
    return bool(centre < ring)


def _bin(imgs: np.ndarray, b_out: int) -> np.ndarray:
    """Fourier crop (n, b, b) -> (n, b_out, b_out); spectra with norm="forward", so values keep their scale."""
    from .engine import ft_resize
    if imgs.shape[-1] == b_out:
        return imgs
    ft = torch.fft.rfft2(torch.from_numpy(imgs), norm="forward")
    return torch.fft.irfft2(ft_resize(ft, b_out), s=(b_out, b_out), norm="forward").numpy()


def import_particles(project: Project, path: str, particle_a: float = 0.0, particle_apix: float = 0.0,
                     invert: str = "auto", project_dir: str | None = None, dtype: str = "float16", label: str = "",
                     log=print) -> Job:
    """Import a particle set (see the module docstring). particle_a: longest particle dimension (A; the 2D mask is
    1.25 x this; 0: 60 % of the box). particle_apix: Fourier-bin to this pixel size (0: keep). invert: "auto" (from the
    data), "yes" (protein dark in the input) or "no". Returns the Extract job."""
    from .pipeline import ExtractSettings, finish
    from .sources import open_source
    t0 = time.time()
    path = os.path.abspath(path)
    src = open_source(path, project_dir)
    if src.optics.no_ctf:
        raise ValueError(f"{path}: no CTF parameters")
    m = _micrographs(src)
    box_a = src.box * src.apix
    particle_a = particle_a or 0.6 * box_a
    b_out = src.box
    if particle_apix and particle_apix > src.apix * 1.001:
        b_out = max(16, int(round(src.box * src.apix / particle_apix / 2)) * 2)
    out_apix = box_a / b_out
    name = label or os.path.basename(os.path.dirname(path)) or "particles"
    mic_paths = list(dict.fromkeys(m["mic"]))
    root = os.path.commonpath([os.path.dirname(p) for p in mic_paths]) if mic_paths else "/"
    imp = project.import_movies(mic_paths, root=root, params=dict(particles_from=path), label=f"{name}_micrographs")
    mv = project.movies(imp).set_index("path")
    o = src.optics
    mc = project.new_job("MotionCtf", dict(imported=True, source=path, kv=o.voltage_kv, cs_mm=o.cs_mm,
                                           amp_contrast=o.q0, apix=float(m["mic_apix"].iloc[0]), bin=1.0,
                                           figures="none"), [imp], f"{name}_imported")
    s = ExtractSettings(particle_a=particle_a, box_a=box_a, particle_apix=out_apix if b_out != src.box else 0.0,
                        dtype=dtype)
    ex = project.new_job("Extract", dict(asdict(s), imported=True, source=path, micrographs=mc.id, picks=0, box=b_out,
                                         out_apix=out_apix, box_a_used=round(box_a, 1)), [mc],
                         f"{name}_{box_a:.0f}A_{out_apix:.3g}apx")
    groups = list(m.groupby("mic", sort=False).groups.items())
    if invert == "auto":
        rng = np.random.default_rng(0)
        pick = [np.asarray(g) for _, g in rng.permutation(np.array(groups, dtype=object))[:200]]
        sample = np.concatenate(pick)[:3000] if pick else np.zeros(0, np.int64)
        flip = _flip_sign(src, sample, particle_a / 2)
        log(f"[import] protein {'dark in the input: inverting' if flip else 'already white'} (from the data)")
    else:
        flip = invert == "yes"
    ctf = src.ctf
    score = np.zeros(src.N, np.float32)                 # the pick score where the input has one
    if "rlnAutopickFigureOfMerit" in src.parts:
        score = src.parts["rlnAutopickFigureOfMerit"].to_numpy(np.float32)
    elif getattr(src, "kind", "") == "cryosparc" and "pick_stats/ncc_score" in src.cs.dtype.names:
        score = src.cs["pick_stats/ncc_score"].astype(np.float32)

    def read(mic, rows):
        rows = np.asarray(rows)
        out = np.empty((len(rows), src.box, src.box), np.float32)
        for fi in np.unique(src.file_idx[rows]):
            k = np.flatnonzero(src.file_idx[rows] == fi)
            out[k] = src._read(fi, rows[k])[1]
        out = _bin(out, b_out)
        return mic, rows, -out if flip else out

    n, last = 0, time.time()
    with ThreadPoolExecutor(8) as pool:
        futs = [pool.submit(read, mic, rows) for mic, rows in groups[:16]]
        nxt = 16
        while futs:
            mic, rows, imgs = futs.pop(0).result()
            if nxt < len(groups):
                futs.append(pool.submit(read, *groups[nxt]))
                nxt += 1
            mid = int(mv.loc[mic, "id"])
            r0 = m.iloc[rows[0]]
            du, dv, an = (float(np.median(ctf[rows, k])) for k in range(3))
            project.add_micrograph(mc, mid, mic, float(r0["mic_apix"]), (int(r0["ny"]), int(r0["nx"])),
                                   SimpleNamespace(defocus_u=du, defocus_v=dv, angle=an, fit_res_a=None), {})
            project.add_particles(ex, mid, m.iloc[rows][["x_a", "y_a"]].to_numpy(), imgs, out_apix,
                                  columns=dict(score=score[rows], defocus_u=ctf[rows, 0], defocus_v=ctf[rows, 1],
                                               defocus_angle=ctf[rows, 2]),
                                  dtype=dtype)
            n += len(rows)
            if time.time() - last > 30:
                last = time.time()
                log(f"[import] {n:,} / {src.N:,} particles")
    finish(project, mc)
    summ = finish(project, ex)
    log(f"[import] {n:,} particles of {len(groups):,} micrographs, {b_out} px at {out_apix:.3f} A/px "
        f"({time.time() - t0:.0f} s)")
    project.set_status(ex, "finished", dict(summ, imported_from=path, inverted=flip, seconds=round(time.time() - t0, 1)))
    return ex
