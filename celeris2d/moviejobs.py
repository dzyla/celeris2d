"""The movie jobs (Import, MotionCtf, Pick, Extract) as job types for the dashboard's New job form and the queue, so
a project can be made from movies to a map in the dashboard alone. They run through pipeline.run (the scheduler
runs a queued MotionCtf -> Pick -> Extract chain as one overlapped run); jobs.registry() keeps the particle jobs.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from .jobspec import Input, JobSpec, MOVIE_KEYS
from .project import Job, Project


@dataclass
class ImportSettings:
    movies: str                         # the movies: a folder (all .tif / .tiff / .mrc / .mrcs / .eer in it and
                                        # below) or a pattern such as /data/session1/**/*.tif; Windows paths work
    root: str = ""                      # the folder whose sub-folders are kept in output names (default: the
                                        # movies' common parent)


MOVIE_EXT = (".tif", ".tiff", ".mrc", ".mrcs", ".eer")


def _wsl(path: str) -> str:
    """A Windows path typed into the dashboard on WSL: E:\\x -> /mnt/e/x."""
    p = os.path.expanduser(str(path).strip())
    if os.name != "nt" and len(p) > 2 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


def movie_paths(movies: str) -> list[str]:
    p = _wsl(movies)
    if os.path.isdir(p):
        out = [os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs if f.lower().endswith(MOVIE_EXT)]
    else:
        out = [f for f in glob.glob(p, recursive=True) if os.path.isfile(f)]
    out = sorted(f for f in out if not os.path.basename(f).lower().startswith(("gain", "norm", "defect")))
    if not out:
        raise ValueError(f"no movies at {movies}")
    return out


def truncated(path: str) -> bool:
    """An MRC movie shorter than its header says (an interrupted download: EMPIAR-10291 had 9 of 300); other
    formats are not checked here (the movie pipeline skips any movie it cannot read)."""
    if not path.lower().endswith((".mrc", ".mrcs")):
        return False
    import warnings
    import mrcfile
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mrcfile.open(path, header_only=True, permissive=True) as m:
                h = m.header
                bpp = {0: 1, 1: 2, 2: 4, 6: 2, 12: 2}.get(int(h.mode), 4)
                need = 1024 + int(h.nsymbt) + int(h.nx) * int(h.ny) * int(h.nz) * bpp
    except (OSError, ValueError):
        return True
    return os.path.getsize(path) < need


def import_job(project: Project, s: ImportSettings, label: str = "") -> Job:
    """Registers the movies at once: the job is finished when made (nothing to queue). Truncated MRC files are left
    out, named in the job's warnings."""
    paths = movie_paths(s.movies)
    bad = [p for p in paths if truncated(p)]
    paths = [p for p in paths if p not in bad]
    if not paths:
        raise ValueError(f"every movie at {s.movies} is truncated or unreadable")
    j = project.import_movies(paths, root=_wsl(s.root) if s.root else "", params=dict(movies=s.movies),
                              label=label)
    if bad:
        summ = dict(project.summary_of(j), skipped=[os.path.basename(p) for p in bad],
                    warnings=[f"{len(bad)} movie files are truncated or unreadable and were left out: "
                              + ", ".join(os.path.basename(p) for p in bad[:12]) + (" ..." if len(bad) > 12 else "")])
        project.write_summary(j, summ)
    return j


def _run_movies(project: Project, job, log=print):
    from .pipeline import run
    return run(project, [job], log=log)


def movie_registry() -> dict[str, JobSpec]:
    from . import pipeline as pl

    def motion(project, movies, s, label=""):
        return pl.motion_job(project, project.job(movies), s, label)

    def pick(project, micrographs, s, label=""):
        return pl.pick_job(project, project.job(micrographs), s, label)

    def extract(project, picks, s, label=""):
        return pl.extract_job(project, project.job(picks), s, label)

    specs = [
        JobSpec("Import", "register movies (they stay where they are)", ImportSettings, (),
                lambda project, s, label="": import_job(project, s, label), lambda project, job, log=print: {},
                key=MOVIE_KEYS["Import"]),
        JobSpec("MotionCtf", "motion correction, dose weighting and CTF fit of every movie", pl.MotionSettings,
                (Input("movies", ("Import",), doc="the movies"),), motion, _run_movies, key=MOVIE_KEYS["MotionCtf"]),
        JobSpec("Pick", "particle picking: a blob, 2D class averages or map projections as templates",
                pl.PickSettings, (Input("micrographs", ("MotionCtf",), doc="the micrographs"),), pick, _run_movies,
                key=MOVIE_KEYS["Pick"]),
        JobSpec("Extract", "particle images from the picks (box from the particle size and CTF delocalisation)",
                pl.ExtractSettings, (Input("picks", ("Pick",), doc="the picks"),), extract, _run_movies,
                key=MOVIE_KEYS["Extract"]),
    ]
    return {sp.type: sp for sp in specs}
