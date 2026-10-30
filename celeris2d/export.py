"""RELION / cryoSPARC files for a job, on request (docs/project-design.md, "Compatibility").

    export(project, job, "relion")      Import: movies.star; MotionCtf: corrected_micrographs.star +
                                        micrographs_ctf.star; Pick: autopick.star + Coordinates/; Extract:
                                        particles.star
    export(project, job, "cryosparc")   Extract: particles.cs (blob, CTF, location; uid = Celeris2D particle uid)

Paths in .star files are relative to the project directory: run RELION there. Particles are stored with white
protein (RELION's --invert_contrast). cryoSPARC needs blob/sign -1 for them: dz_2025_10 particles imported with +1
refined to an inverted map (skew -5.5) at 7 A; with -1 to 4.76 A, positive density (cryoSPARC v5.0.6 NU refinement).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .project import PARTICLE_JOBS, Job, Project, _atomic
from .star import write_star


def _optics(project: Project, job: Job, image_apix: float | None = None, image_size: int | None = None):
    from .pipeline import optics_of
    o = optics_of(project, job) if job.type != "Import" else None
    mc = next((project.job(d) for d in project.lineage(job) if project.job(d).type == "MotionCtf"), None)
    p = mc.params if mc else {}
    row = dict(rlnOpticsGroupName="opticsGroup1", rlnOpticsGroup=1,
               rlnMicrographOriginalPixelSize=p.get("apix", 1.0),
               rlnVoltage=o.voltage_kv if o else p.get("kv", 300.0), rlnSphericalAberration=o.cs_mm if o else 2.7,
               rlnAmplitudeContrast=o.q0 if o else 0.1)
    if mc is not None:
        row["rlnMicrographPixelSize"] = p["apix"] * p["bin"]
    if image_apix:
        row.update(rlnImagePixelSize=image_apix, rlnImageSize=image_size, rlnImageDimensionality=2,
                   rlnCtfDataAreCtfPremultiplied=0)
    return pd.DataFrame([row])


def _star(path, blocks):
    _atomic(path, lambda t: write_star(t, blocks))
    return path


def export(project: Project, job, fmt: str) -> str:
    job = job if isinstance(job, Job) else project.job(job)
    fmt = fmt.lower()
    if fmt == "relion":
        return dict(Import=_relion_movies, MotionCtf=_relion_micrographs, Pick=_relion_picks,
                    Extract=_relion_particles, Polish=_relion_particles, Subtract=_relion_particles,
                    SymExpand=_relion_symexpand)[job.type](project, job)
    if fmt == "cryosparc":
        if job.type not in PARTICLE_JOBS:
            raise ValueError("cryoSPARC export is for Extract / Polish jobs (cryoSPARC imports movies / micrographs itself)")
        return _cryosparc_particles(project, job)
    raise ValueError(fmt)


def _relion_movies(project, job):
    mv = project.movies(job)
    return _star(job.out("movies.star"),
                 {"optics": _optics(project, job), "movies": pd.DataFrame(
                     dict(rlnMicrographMovieName=mv["path"], rlnOpticsGroup=1))})


def _relion_micrographs(project, job):
    mi = project.micrographs(job)
    import json
    meta = [json.loads(m) for m in mi["meta"]]
    base = dict(rlnMicrographName=mi["path"], rlnOpticsGroup=1,
                rlnAccumMotionTotal=[m.get("motion_path_a", 0) / (job.params["apix"] * job.params["bin"]) for m in meta])
    opt = _optics(project, job)
    _star(job.out("corrected_micrographs.star"), {"optics": opt, "micrographs": pd.DataFrame(base)})
    ctf = pd.DataFrame(dict(base, rlnDefocusU=mi["defocus_u"], rlnDefocusV=mi["defocus_v"],
                            rlnCtfAstigmatism=mi["defocus_u"] - mi["defocus_v"], rlnDefocusAngle=mi["defocus_angle"],
                            rlnCtfFigureOfMerit=[m.get("ctf_score", 0) for m in meta],
                            rlnCtfMaxResolution=mi["ctf_fit_res"]))
    return _star(job.out("micrographs_ctf.star"), {"optics": opt, "micrographs": ctf})


def _relion_picks(project, job):
    mics = project.micrographs(job.params["micrographs"]).set_index("movie_id")
    rows = []
    for r in project.pick_table(job).itertuples():
        z = project.picks(job, r.movie_id)
        k = z["keep"]
        m = mics.loc[r.movie_id]
        cpath = os.path.join(job.path, "Coordinates", r.subdir or "", r.stem + "_autopick.star")
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        _star(cpath, {"": pd.DataFrame(dict(rlnCoordinateX=z["xy_a"][k, 0] / m["apix"],
                                            rlnCoordinateY=z["xy_a"][k, 1] / m["apix"],
                                            rlnAutopickFigureOfMerit=z["score"][k], rlnClassNumber=0,
                                            rlnAnglePsi=-999.0))})
        rows.append(dict(rlnMicrographName=m["path"], rlnMicrographCoordinates=project.rel(cpath)))
    return _star(job.out("autopick.star"), {"coordinate_files": pd.DataFrame(rows)})


def relion_particle_blocks(project, job) -> dict:
    p = project.particles(job)
    part = pd.DataFrame({
        "rlnCoordinateX": p["x_a"] / p["mic_apix"], "rlnCoordinateY": p["y_a"] / p["mic_apix"],
        "rlnImageName": [f"{i + 1:06d}@{s}" for i, s in zip(p["slice"], p["stack"])],
        "rlnMicrographName": p["mic_path"], "rlnOpticsGroup": 1,
        "rlnDefocusU": p["defocus_u"], "rlnDefocusV": p["defocus_v"], "rlnDefocusAngle": p["defocus_angle"],
        "rlnCtfMaxResolution": p["ctf_fit_res"], "rlnAutopickFigureOfMerit": p["score"]})
    part["celUid"] = p["uid"].to_numpy()
    part["celMovie"] = p["movie_id"].to_numpy()
    if "copy" in p:                                  # symmetry-expanded copies (Subtract): (uid, copy) is the key
        part["celCopy"] = p["copy"].to_numpy().astype(np.int64)
    return {"optics": _optics(project, job, float(p["apix"].iloc[0]), int(p["box"].iloc[0])), "particles": part}


def _relion_particles(project, job):
    b = relion_particle_blocks(project, job)
    b["particles"] = b["particles"].drop(columns=[c for c in ("celUid", "celMovie", "celCopy") if c in b["particles"]])
    return _star(job.out("particles.star"), b)


def _relion_symexpand(project, job):
    """A SymExpand job as relion_particle_symmetry_expand writes it: one row per copy, the particle's image (same
    rlnImageName), the expanded angles (rlnAngleRot/Tilt/Psi, image = A @ volume as RELION), rlnOriginX/YAngst = -shift
    (relative to the particle job's box centres), rlnRandomSubset = the particle's half + 1; celCopy keeps the copy
    number (RELION ignores the column; (uid, copy) is the row's key)."""
    from .jobs import match_rows
    ex = project.aligned_particles(job)
    b = relion_particle_blocks(project, ex)
    part = b["particles"]
    t = project.table(job, "particles").read()
    row = match_rows(part["celUid"].to_numpy(np.int64), None, t["uid"])
    if (row < 0).any():
        raise ValueError(f"{ex.name} lacks particles of {job.name}")
    o = np.lexsort((t["copy"], row))                    # by image, then copy (relion_particle_symmetry_expand's order)
    df = part.iloc[row[o]].reset_index(drop=True)
    df["rlnAngleRot"], df["rlnAngleTilt"], df["rlnAnglePsi"] = t["rot"][o], t["tilt"][o], t["psi"][o]
    df["rlnOriginXAngst"], df["rlnOriginYAngst"] = -t["shift_x_a"][o], -t["shift_y_a"][o]
    df["rlnRandomSubset"] = t["half"][o].astype(int) + 1
    if "cls" in t:
        df["rlnClassNumber"] = t["cls"][o].astype(int) + 1
    df["celCopy"] = t["copy"][o].astype(np.int64)
    b["particles"] = df.drop(columns=[c for c in ("celUid", "celMovie") if c in df])
    return _star(job.out("particles.star"), b)


def job_source(job_dir: str, uids=None, **kw):
    """Particles of an Extract job folder as a particle source (RelionSource over the project's own table; no
    file is written). Its parts carry celUid (the Celeris2D particle uid) and celMovie. uids: only these particles
    (e.g. a Select job's)."""
    from .sources import RelionSource
    job_dir = os.path.abspath(job_dir).rstrip("/")
    root = os.path.dirname(os.path.dirname(job_dir))
    project = Project(root)
    job = project.job(os.path.relpath(job_dir, root))
    if job.type not in PARTICLE_JOBS:
        raise SystemExit(f"{job.dir} is not an Extract or Polish job")
    src = RelionSource(relion_particle_blocks(project, job), project_dir=root, **kw)
    src.kind = "celeris2d"
    src.path = job_dir
    if uids is not None:
        keep = np.flatnonzero(np.isin(src.parts["celUid"].to_numpy(), np.asarray(uids, np.int64)))
        src.subset(keep)
    # mask: the particle (longest dimension) plus 25 %, so its ends stay clear of the soft edge
    src.mask_diameter_a = min(1.25 * job.params["particle_a"], 0.95 * job.params["box_a_used"])
    return src


def _cryosparc_particles(project, job):
    from .pipeline import optics_of
    p = project.particles(job)
    o = optics_of(project, job)
    n = len(p)
    blen = max(len(project.abs(s)) for s in p["stack"].unique()) + 1
    mlen = max(len(project.abs(s)) for s in p["mic_path"].unique()) + 1
    dt = [("uid", "<u8"), ("blob/path", f"S{blen}"), ("blob/idx", "<u4"), ("blob/shape", "<u4", (2,)),
          ("blob/psize_A", "<f4"), ("blob/sign", "<f4"), ("blob/import_sig", "<u8"),
          ("ctf/type", "S1"), ("ctf/exp_group_id", "<u4"), ("ctf/accel_kv", "<f4"), ("ctf/cs_mm", "<f4"),
          ("ctf/amp_contrast", "<f4"), ("ctf/df1_A", "<f4"), ("ctf/df2_A", "<f4"), ("ctf/df_angle_rad", "<f4"),
          ("ctf/phase_shift_rad", "<f4"), ("ctf/scale", "<f4"), ("ctf/scale_const", "<f4"), ("ctf/shift_A", "<f4", (2,)),
          ("ctf/tilt_A", "<f4", (2,)), ("ctf/trefoil_A", "<f4", (2,)), ("ctf/tetra_A", "<f4", (4,)),
          ("ctf/anisomag", "<f4", (4,)), ("ctf/bfactor", "<f4"),
          ("location/micrograph_uid", "<u8"), ("location/exp_group_id", "<u4"), ("location/micrograph_path", f"S{mlen}"),
          ("location/micrograph_shape", "<u4", (2,)), ("location/micrograph_psize_A", "<f4"),
          ("location/center_x_frac", "<f4"), ("location/center_y_frac", "<f4")]
    a = np.zeros(n, dtype=dt)
    a["uid"] = p["uid"].to_numpy().astype(np.uint64)
    a["blob/path"] = [project.abs(s).encode() for s in p["stack"]]
    a["blob/idx"] = p["slice"]
    a["blob/shape"] = np.stack([p["box"], p["box"]], 1)
    a["blob/psize_A"] = p["apix"]
    a["blob/sign"] = -1.0                                   # white protein: cryoSPARC -1 (see the module docstring)
    a["ctf/type"] = b"i"
    a["ctf/accel_kv"], a["ctf/cs_mm"], a["ctf/amp_contrast"] = o.voltage_kv, o.cs_mm, o.q0
    a["ctf/df1_A"], a["ctf/df2_A"] = p["defocus_u"], p["defocus_v"]
    a["ctf/df_angle_rad"] = np.radians(p["defocus_angle"].to_numpy(float))
    a["ctf/scale"] = 1.0
    a["ctf/scale_const"] = 0.0
    a["ctf/anisomag"] = (1, 0, 0, 1)
    a["location/micrograph_uid"] = p["movie_id"].to_numpy().astype(np.uint64)
    a["location/micrograph_path"] = [project.abs(s).encode() for s in p["mic_path"]]
    a["location/micrograph_shape"] = np.stack([p["mic_ny"], p["mic_nx"]], 1)
    a["location/micrograph_psize_A"] = p["mic_apix"]
    a["location/center_x_frac"] = p["x_a"] / (p["mic_nx"] * p["mic_apix"])
    a["location/center_y_frac"] = p["y_a"] / (p["mic_ny"] * p["mic_apix"])
    path = job.out("particles.cs")
    _atomic(path, lambda t: np.save(open(t, "wb"), a))
    return path
