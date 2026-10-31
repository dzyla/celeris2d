"""Project store (docs/project-design.md): jobs in one chronological list of descriptive folders, heavy per-micrograph
data apart from them (purgeable, rebuildable), a SQLite catalog as the index.

    myproject/
      project.db  project.json  JOBS.md
      jobs/004_extract_336A_1.32apx_f16/    job.json, summary.json, particles.npz, figures, exports (small)
      data/004/Particles/<session>/<stem>.mrcs + .npz                   heavy / per-micrograph, by job number
      Trash/                                                           deleted jobs

A job is referred to by its number (4, "004"), its folder name, or its label (the descriptive part, which can be
renamed: nothing refers to it). Jobs record their inputs by number.

Particle identity, two forms of the same thing:
  key = "<movie stem>:<x A>:<y A>" (e.g. 14sep05c_00024sq_00003hl_00002es:4234.5:1663.2): readable, decodes on its
        own (parse_key), stored with every particle;
  uid = first 63 bits of SHA-256(movie fingerprint, position in A rounded to 0.1 A): the same particle has the
        same uid in every extraction job, in any project and after re-processing, so a selection (a Select job: a
        list of uids) applies to any extraction of the same picks. One-way; the catalog maps it back.
Movie fingerprint = SHA-256 of the file name, size and its first and last MiB.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tables import ShardedTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY, type TEXT, name TEXT UNIQUE, label TEXT, params TEXT, inputs TEXT, status TEXT,
    created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE, stem TEXT, subdir TEXT, fingerprint TEXT, import_job INTEGER,
    added REAL);
CREATE TABLE IF NOT EXISTS micrographs (
    job_id INTEGER, movie_id INTEGER, path TEXT, apix REAL, ny INTEGER, nx INTEGER,
    defocus_u REAL, defocus_v REAL, defocus_angle REAL, ctf_fit_res REAL, meta TEXT, created REAL,
    PRIMARY KEY (job_id, movie_id));
CREATE TABLE IF NOT EXISTS picks (
    job_id INTEGER, movie_id INTEGER, path TEXT, n_candidates INTEGER, n_kept INTEGER, threshold REAL,
    created REAL, PRIMARY KEY (job_id, movie_id));
CREATE TABLE IF NOT EXISTS chunks (
    job_id INTEGER, movie_id INTEGER, table_path TEXT, stack_path TEXT, n INTEGER, box INTEGER, apix REAL,
    created REAL, PRIMARY KEY (job_id, movie_id));
"""

JOB_TYPES = ("Import", "ImportParticles", "MotionCtf", "Curate", "Pick", "Extract", "Select", "Class2D", "InitialModel", "Refine3D", "HeteroRefine",
             "Polish", "CtfRefine", "Sharpen", "MaskCreate", "MapScore", "Class3D", "Subtract", "LocalRefine", "SymExpand",
             "Modes3D", "ModeMaps")
PER_MOVIE = dict(MotionCtf="micrographs", Pick="picks", Extract="chunks", Polish="chunks", Subtract="chunks")
HEAVY = dict(MotionCtf=(".mrc",), Extract=(".mrcs",), Polish=(".mrcs",), Subtract=(".mrcs",))   # purgeable files of a job's data folder
REBUILDABLE = ("MotionCtf", "Extract")         # purged files can be made again from the recorded parameters
PARTICLE_JOBS = ("Extract", "Polish", "CtfRefine", "Subtract")   # jobs that hold particle images (a "particles" table
                                                      # + stacks; a CtfRefine job's are those of the job it refined; a
                                                      # Subtract job's table also holds poses, and a symmetry-expanded one
                                                      # one row per (uid, copy))
CLASS_JOBS = ("HeteroRefine", "Class3D", "ModeMaps")    # 3D jobs with several maps: classNN_map.mrc ..., table
                                                        # "particles" with cls (ModeMaps: clusters / bins along modes)


def has_classes(job) -> bool:
    """A job whose result is several 3D classes (classNN_map.mrc ..., a summary class_table, a cls column in its
    particle table): the CLASS_JOBS, and an InitialModel with n_models > 1 - except one finished before that layout
    (map_1.mrc ..., no class files), which stays a single-map job (its map.mrc: the largest model)."""
    if job.type in CLASS_JOBS:
        return True
    if job.type != "InitialModel" or int((job.params or {}).get("n_models") or 1) <= 1:
        return False
    return job.status != "finished" or job.find("class01_map.mrc") is not None


def class_table(summary: dict) -> list:
    """A multi-class job's per-class rows (cls 1-based, particles, share, resolution_a, resolution_solvent_mask_a,
    ...): summary["class_table"]; jobs made before it kept the table under "classes"."""
    t = summary.get("class_table")
    if t is None and isinstance(summary.get("classes"), list):
        t = summary["classes"]
    return list(t or [])


def movie_fingerprint(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    if not os.path.exists(path):                     # an imported particle set's micrograph that is not here: its path
        h.update(path.encode())
        return h.hexdigest()
    size = os.path.getsize(path)
    h.update(os.path.basename(path).encode())
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def particle_keys(stem: str, xy_a: np.ndarray) -> np.ndarray:
    """Readable, reversible particle keys: movie stem and position in A (0.1 A)."""
    return np.array([f"{stem}:{x:.1f}:{y:.1f}" for x, y in np.round(np.asarray(xy_a, dtype=np.float64), 1)])


def parse_key(key: str) -> tuple[str, float, float]:
    """(movie stem, x A, y A) of a particle key."""
    stem, x, y = key.rsplit(":", 2)
    return stem, float(x), float(y)


def particle_uids(fingerprint: str, xy_a: np.ndarray) -> np.ndarray:
    """uids (int64, >= 0) of particles at positions xy_a (n, 2) in A on the movie with this fingerprint."""
    out = np.empty(len(xy_a), dtype=np.int64)
    for i, (x, y) in enumerate(np.round(np.asarray(xy_a, dtype=np.float64), 1)):
        d = hashlib.sha256(f"{fingerprint}:{x:.1f}:{y:.1f}".encode()).digest()
        out[i] = int.from_bytes(d[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
    return out


def _atomic(path: str, write):
    """write(tmp_path), then rename onto path: readers see the old file or the complete new one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    tmp = f"{base}.tmp{os.getpid()}{ext}"
    write(tmp)
    os.replace(tmp, path)


def _save_table(path: str, df: pd.DataFrame):
    """A DataFrame as an .npz of columns (text columns as unicode arrays), written atomically."""
    cols = {k: df[k].to_numpy() if pd.api.types.is_numeric_dtype(df[k]) else df[k].astype(str).to_numpy(dtype=str)
            for k in df.columns}
    _atomic(path, lambda t: np.savez(t, **cols))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")[:60]


def _du(path: str) -> int:
    n = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                n += os.path.getsize(fp)
    return n


def _version() -> str:
    try:
        from importlib.metadata import version
        v = version("celeris2d")
    except Exception:
        v = "?"
    try:
        h = subprocess.run(["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        h = ""
    return f"{v} {h}".strip()


@dataclass
class Job:
    id: int
    type: str
    name: str                   # folder name: NNN_type_label
    label: str                  # descriptive part (renamable)
    params: dict
    inputs: list                # job numbers this job was made from
    status: str
    root: str

    @property
    def dir(self) -> str:       # relative to the project
        return f"jobs/{self.name}"

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.dir)

    @property
    def data_path(self) -> str:
        return os.path.join(self.root, "data", f"{self.id:03d}")

    def file(self, *parts) -> str:
        """A file in the job folder (small results)."""
        p = os.path.join(self.path, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def per_movie(self, kind: str, movie: dict, ext: str) -> str:
        """data/NNN/<kind>/<movie subdir>/<stem><ext> (per-micrograph data)."""
        p = os.path.join(self.data_path, kind, movie["subdir"] or "", movie["stem"] + ext)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def figure(self, movie: dict) -> str:
        return self.file("figures", "movies", self.prefix + (movie["subdir"] or "").replace(os.sep, "_")
                         + "_" * bool(movie["subdir"]) + movie["stem"] + ".png")

    # result files: <job>/maps/j012_map.mrc, <job>/figures/j012_fsc.png (+ .light.png), <job>/data/j012_fsc.star,
    # per-micrograph figures in <job>/figures/movies/, <job>/j012_run.log - the job number in every name, so a file copied elsewhere (or listed by RELION next to
    # another job's) is never ambiguous. job.json / summary.json (the job's record) and the particle tables stay as is.
    @property
    def prefix(self) -> str:
        return f"j{self.id:03d}_"

    def out(self, name: str) -> str:
        """Where a result file named `name` goes (folder by kind, job number in front); the folder is created."""
        sub = _subdir(name)
        p = os.path.join(self.path, sub, self.prefix + name) if sub else os.path.join(self.path, self.prefix + name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def find(self, name: str) -> str:
        """A result file by its plain name: the current layout, else the flat layout of older jobs (or None)."""
        p = os.path.join(self.path, _subdir(name) or "", self.prefix + name)
        if os.path.exists(p):
            return p
        old = os.path.join(self.path, name)
        return old if os.path.exists(old) else None

    def organize(self):
        """Move result files a function wrote flat into the job folder into the layout of out()."""
        for f in sorted(os.listdir(self.path)):
            src = os.path.join(self.path, f)
            if os.path.isfile(src) and f not in ("job.json", "summary.json") and not f.startswith(self.prefix) \
                    and os.path.splitext(f)[1].lower() in _SUBDIR:
                os.replace(src, self.out(f))

    @property
    def log_path(self) -> str:
        return os.path.join(self.path, self.prefix + "run.log")


_SUBDIR = {".mrc": "maps", ".mrcs": "maps", ".bild": "maps", ".png": "figures", ".gif": "figures", ".json": "data",
           ".star": "data", ".xml": "data", ".csv": "data", ".cs": "data", ".csg": "data", ".npz": "data", ".txt": "data",
           ".log": ""}


def _subdir(name: str) -> str:
    """The folder of a result file: by extension; a figure's data (x.plot.json, figjson.py) stays next to x.png."""
    if name.lower().endswith(".plot.json"):
        return "figures"
    return _SUBDIR.get(os.path.splitext(name)[1].lower(), "data")


class run_log:
    """with run_log(jobs, log) as lg: lg(...) writes each line, time-stamped, to every job's run log (j012_run.log,
    read live by the dashboard) and passes it on to log; the start, the end and an error with its traceback too."""

    def __init__(self, jobs, log=print, what: str = ""):
        self.jobs, self.log, self.what = list(jobs), log, what

    def __enter__(self):
        self.fh = []
        for j in self.jobs:
            os.makedirs(j.path, exist_ok=True)
            self.fh.append(open(j.log_path, "a", buffering=1))
        self.t0 = time.time()
        self._write(f"started{': ' + self.what if self.what else ''} (pid {os.getpid()} on {socket.gethostname()})")
        return self

    def _write(self, text: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S ")
        for f in self.fh:
            f.write(stamp + text + "\n")

    def __call__(self, *m):
        self._write(" ".join(str(x) for x in m))
        self.log(*m)

    def __exit__(self, et, e, tb):
        import traceback
        if et is None:
            self._write(f"finished in {time.time() - self.t0:.1f} s")
        else:
            self._write(f"FAILED after {time.time() - self.t0:.1f} s: {e!r}\n"
                        + "".join(traceback.format_exception(et, e, tb)))
        for f in self.fh:
            f.close()
        return False


def _write_text(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def logged(run):
    """A job's run function (project, job, ..., log=) also writes its log to the job's run log (see run_log)."""
    import functools

    @functools.wraps(run)
    def wrap(project, job, *a, log=print, **k):
        job = project.job(job)
        with run_log([job], log, run.__name__) as lg:
            return run(project, job, *a, log=lg, **k)
    return wrap

    def __str__(self):
        return self.name


def fits_float16(data: np.ndarray) -> bool:
    """float16 holds the data: every value below its maximum (65504) and the mean small enough that its steps
    (~mean / 1024) stay below a fifth of the standard deviation. Raw-count micrographs do not (CryoPPP EMPIAR-10017,
    Falcon: mean 70,867, std 2,742 - written as float16 they were 98 % inf)."""
    x = np.asarray(data)
    if not np.issubdtype(x.dtype, np.floating) or x.dtype == np.float16:
        return True
    v = x[::max(1, x.size // 1_000_000)] if x.ndim == 1 else x.reshape(-1)[::max(1, x.size // 1_000_000)]
    m, sd = float(np.mean(v, dtype=np.float64)), float(np.std(v, dtype=np.float64))
    return bool(np.isfinite(x).all() and float(np.abs(x).max()) < 6.0e4 and abs(m) < 200 * max(sd, 1e-30))


class Project:
    """A processing project directory (one writer process; any number of readers)."""

    def __init__(self, root: str, name: str = ""):
        self.root = os.path.abspath(root)
        os.makedirs(os.path.join(self.root, "jobs"), exist_ok=True)
        with self._db() as c:
            c.executescript(SCHEMA)
        pj = os.path.join(self.root, "project.json")
        if not os.path.exists(pj):
            with open(pj, "w") as f:
                json.dump(dict(name=name or os.path.basename(self.root), created=time.time(),
                               celeris2d=_version()), f, indent=1)

    def _db(self):
        c = sqlite3.connect(os.path.join(self.root, "project.db"), timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    def abs(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    # ------------------------------------------------------------------ jobs
    def new_job(self, type: str, params: dict | None = None, inputs=(), label: str = "") -> Job:
        """A new job folder jobs/NNN_<type>_<label> (NNN: project-wide counter, never reused)."""
        if type not in JOB_TYPES:
            raise ValueError(f"job type {type!r} not in {JOB_TYPES}")
        inputs = [self.job(j).id for j in inputs]
        label = _slug(label)
        params = dict(params or {}, run_pid=os.getpid(), run_host=socket.gethostname())     # is_running
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")        # the write lock before reading the counter: jobs created by several
            n = (c.execute("SELECT MAX(id) FROM jobs").fetchone()[0] or 0) + 1      # processes at once get distinct ids
            name = f"{n:03d}_{type.lower()}" + (f"_{label}" if label else "")
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                      (n, type, name, label, json.dumps(params, default=str), json.dumps(inputs), "running",
                       time.time(), time.time()))
        job = self.job(n)
        os.makedirs(job.path, exist_ok=True)
        self._write_job_json(job)
        self.write_index()
        return job

    def job(self, ref) -> Job:
        """By number (4, "004"), folder name ("004_extract_..."), path ("jobs/004_..."), label, or Job."""
        if isinstance(ref, Job):
            ref = ref.id
        if isinstance(ref, str):
            ref = os.path.basename(ref.rstrip("/"))
            if ref.isdigit():
                ref = int(ref)
        with self._db() as c:
            if isinstance(ref, (int, np.integer)):
                r = c.execute("SELECT * FROM jobs WHERE id=?", (int(ref),)).fetchone()
            else:
                m = re.match(r"^(\d+)_", ref)
                r = c.execute("SELECT * FROM jobs WHERE name=? OR label=? OR id=? ORDER BY id DESC",
                              (ref, ref, int(m.group(1)) if m else -1)).fetchone()
        if r is None:
            raise KeyError(f"no job {ref!r}")
        return Job(r[0], r[1], r[2], r[3], json.loads(r[4]), json.loads(r[5]), r[6], self.root)

    def jobs(self, type: str | None = None, include_deleted: bool = False) -> pd.DataFrame:
        with self._db() as c:
            q = "SELECT id, type, name, label, status, inputs, params, created, updated FROM jobs WHERE 1=1"
            q += "" if include_deleted else " AND status != 'deleted'"
            return pd.read_sql(q + (" AND type=?" if type else "") + " ORDER BY id", c, params=(type,) if type else ())

    def latest(self, type: str, status: str | None = None) -> Job | None:
        j = self.jobs(type)
        if status:
            j = j[j["status"] == status]
        return self.job(int(j["id"].iloc[-1])) if len(j) else None

    def set_status(self, job: Job, status: str, summary: dict | None = None, pid: int | None = None):
        """running / finished / failed / aborted; summary: the job's key numbers (summary.json). A running job
        records the process running it (is_running): this one, or `pid` (a process started for it)."""
        if status == "running":
            job.params.update(run_pid=pid or os.getpid(), run_host=socket.gethostname(),
                              run_started=job.params.get("run_started") if pid is None and job.params.get("run_started")
                              and job.params.get("run_pid") == os.getpid() else time.time())   # started for it: kept
            with self._db() as c:
                c.execute("UPDATE jobs SET params=? WHERE id=?", (json.dumps(job.params, default=str), job.id))
        with self._db() as c:
            c.execute("UPDATE jobs SET status=?, updated=? WHERE id=?", (status, time.time(), job.id))
        job.status = status
        if summary is not None:
            self.write_summary(job, summary)
        self._write_job_json(job)
        self.write_index()

    def update_params(self, job: Job, **kw):
        job.params.update(kw)
        with self._db() as c:
            c.execute("UPDATE jobs SET params=?, updated=? WHERE id=?",
                      (json.dumps(job.params, default=str), time.time(), job.id))
        self._write_job_json(job)

    def rename(self, ref, label: str) -> Job:
        """New descriptive label (the folder becomes NNN_<type>_<label>); references are by number, so nothing
        breaks. RELION / cryoSPARC exports point into data/, which does not move."""
        job = self.job(ref)
        self._not_running(job, "renamed")
        label = _slug(label)
        name = f"{job.id:03d}_{job.type.lower()}" + (f"_{label}" if label else "")
        os.rename(job.path, os.path.join(self.root, "jobs", name))
        with self._db() as c:
            c.execute("UPDATE jobs SET name=?, label=?, updated=? WHERE id=?", (name, label, time.time(), job.id))
        job = self.job(job.id)
        self._write_job_json(job)
        self.write_index()
        return job

    def is_running(self, job: Job) -> bool:
        """Status running and its process alive (a process that died leaves a job marked running; on this host that
        is detected, on another host the job counts as running)."""
        if job.status != "running":
            return False
        pid, host = job.params.get("run_pid"), job.params.get("run_host")
        if pid is None or host != socket.gethostname():
            return pid is not None
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True

    def reap_stale(self) -> list[str]:
        """Jobs marked running whose process on this host is gone (crashed, killed, the machine restarted): marked
        failed, so the job list does not show them running for ever. Returns their names."""
        out = []
        for r in self.jobs().itertuples():
            if r.status == "running":
                j = self.job(int(r.id))
                if not self.is_running(j):
                    summ = dict(self.summary_of(j), error="the process ended without finishing (crashed or killed)")
                    self.set_status(j, "failed", summ)
                    out.append(j.name)
        return out

    # ------------------------------------------------------------------ final jobs (publication report)
    def final_marks(self) -> dict:
        """{job id: {"note", "marked"}} of the jobs marked final (final.json in the project): the datasets / maps a
        publication reports; report.py draws the flowchart and the methods from their lineage."""
        p = os.path.join(self.root, "final.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return {int(k): v for k, v in json.load(f).items()}

    def mark_final(self, ref, final: bool = True, note: str = "") -> dict:
        job = self.job(ref)
        marks = self.final_marks()
        if final:
            marks[job.id] = dict(note=note, marked=time.time())
        else:
            marks.pop(job.id, None)
        _atomic(os.path.join(self.root, "final.json"),
                lambda t: _write_text(t, json.dumps({str(k): v for k, v in sorted(marks.items())}, indent=1)))
        return marks

    def _not_running(self, job: Job, what: str):
        if self.is_running(job):
            raise ValueError(f"{job.name} is running: it cannot be {what} now")

    def write_summary(self, job: Job, summary: dict):
        _atomic(os.path.join(job.path, "summary.json"),
                lambda t: json.dump(summary, open(t, "w"), indent=1, default=str))

    def summary_of(self, ref) -> dict:
        p = os.path.join(self.job(ref).path, "summary.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def _write_job_json(self, job: Job):
        d = dict(id=job.id, type=job.type, name=job.name, label=job.label, status=job.status,
                 inputs={i: self.job(i).name for i in job.inputs}, params=job.params, data=self.rel(job.data_path),
                 celeris2d=_version(), updated=time.time())
        _atomic(os.path.join(job.path, "job.json"), lambda t: _write_text(t, json.dumps(d, indent=1, default=str)))

    def write_index(self):
        """JOBS.md: one line per job, readable without any tool."""
        from .jobspec import key_of
        try:
            from .jobs import registry
            reg = registry()
        except ImportError:                              # no torch: the movie jobs' key numbers only
            reg = {}
        rows = ["| job | status | inputs | key numbers |", "|---|---|---|---|"]
        for r in self.jobs().itertuples():
            key = key_of(r.type, self.summary_of(int(r.id)), reg).replace("|", "/")
            ins = ", ".join(f"{i:03d}" for i in json.loads(r.inputs))
            rows.append(f"| {r.name} | {r.status} | {ins} | {key} |")
        _atomic(os.path.join(self.root, "JOBS.md"), lambda t: _write_text(t, "\n".join(rows) + "\n"))

    def lineage(self, ref) -> list[str]:
        """The job and every job it was made from (folder names), nearest first."""
        out, todo = [], [self.job(ref).id]
        while todo:
            j = self.job(todo.pop(0))
            if j.name not in out:
                out.append(j.name)
                todo += j.inputs
        return out

    def ancestor(self, ref, type) -> Job | None:
        """The nearest job of this type (or of one of these types) the job descends from (or the job itself)."""
        types = (type,) if isinstance(type, str) else tuple(type)
        return next((self.job(n) for n in self.lineage(ref) if self.job(n).type in types), None)

    def aligned_particles(self, ref) -> Job:
        """The particle job (Extract / Polish) whose box centres an alignment job's shifts are relative to: a
        Refine3D / HeteroRefine / Class3D / Modes3D / ModeMaps job's particles, or the extraction a
        Class2D (or the Class2D of an InitialModel) was run on."""
        job = self.job(ref)
        if job.type == "Subtract":                  # its poses are relative to its own (re-centred) boxes
            return job
        if job.type in ("Refine3D", "LocalRefine", "SymExpand", "Modes3D", *CLASS_JOBS):
            src = self.job(job.params["particles"])
            return src if src.type in PARTICLE_JOBS else self.ancestor(src, PARTICLE_JOBS)
        if job.type == "InitialModel":
            job = self.job(job.params["class2d"])
        if job.type != "Class2D":
            raise ValueError(f"{job.name} has no particle alignments")
        src = self.job(job.params["particles"])
        return src if src.type in PARTICLE_JOBS else self.ancestor(src, PARTICLE_JOBS)

    def dependents(self, ref) -> list[str]:
        i = self.job(ref).id
        return [r.name for r in self.jobs().itertuples() if i in json.loads(r.inputs)]

    def delete_job(self, ref, force: bool = False) -> str:
        """Move a job (folder and data) to Trash/. Refuses if other jobs used it, unless force."""
        job = self.job(ref)
        self._not_running(job, "deleted")
        deps = self.dependents(job)
        if deps and not force:
            raise ValueError(f"{job.name} is an input of {deps}")
        dst = os.path.join(self.root, "Trash", job.name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(job.path, dst)
        if os.path.isdir(job.data_path):
            shutil.move(job.data_path, os.path.join(dst, "data"))
        with self._db() as c:
            for t in ("micrographs", "picks", "chunks"):
                c.execute(f"DELETE FROM {t} WHERE job_id=?", (job.id,))
            c.execute("UPDATE jobs SET status='deleted', updated=? WHERE id=?", (time.time(), job.id))
        self.write_index()
        return dst

    def empty_trash(self) -> int:
        p = os.path.join(self.root, "Trash")
        n = _du(p) if os.path.isdir(p) else 0
        shutil.rmtree(p, ignore_errors=True)
        return n

    def disk(self) -> pd.DataFrame:
        """Space per job (folder + data), how much of it is purgeable (micrographs / particle stacks, rebuildable
        from the recorded parameters), and which jobs use it. Biggest first."""
        rows = []
        for r in self.jobs().itertuples():
            j = self.job(int(r.id))
            heavy = sum(os.path.getsize(f) for ext in HEAVY.get(j.type, ())
                        for f in glob.glob(os.path.join(j.data_path, "**", "*" + ext), recursive=True))
            rows.append(dict(job=j.name, status=j.status, gb=round((_du(j.path) + _du(j.data_path)) / 1e9, 3),
                             purgeable_gb=round(heavy / 1e9, 3), purged=bool(j.params.get("purged")),
                             used_by=", ".join(self.dependents(j))))
        return pd.DataFrame(rows).sort_values("gb", ascending=False, ignore_index=True)

    def purge(self, ref) -> float:
        """Delete a job's heavy files (micrographs / particle stacks) and keep everything else; rebuild() makes
        them again from the recorded parameters. Returns GB freed."""
        job = self.job(ref)
        if job.type not in REBUILDABLE or job.params.get("imported"):
            raise ValueError(f"{job.name}: {'imported' if job.params.get('imported') else job.type} jobs cannot be "
                             "purged (rebuild could not make their files again)")
        self._not_running(job, "purged")
        n = 0
        for ext in HEAVY.get(job.type, ()):
            for f in glob.glob(os.path.join(job.data_path, "**", "*" + ext), recursive=True):
                n += os.path.getsize(f)
                os.remove(f)
        self.update_params(job, purged=True)
        self.write_index()
        return n / 1e9

    def write_mrc(self, path: str, data: np.ndarray, apix: float, dtype: str = "float32") -> str:
        """MRC/MRCS in float32 (mode 2) or float16 (mode 12: RELION >= 4, cryoSPARC >= 4; half the space). float16
        only where the data fit it (fits_float16), else float32. Returns the dtype written."""
        import mrcfile
        if dtype == "float16" and not fits_float16(data):
            dtype = "float32"

        def w(t):
            with mrcfile.new(t, overwrite=True) as m:
                m.set_data(np.ascontiguousarray(data, dtype=np.float16 if dtype == "float16" else np.float32))
                m.voxel_size = apix
        _atomic(path, w)
        return dtype

    # ------------------------------------------------------------------ movies (Import jobs)
    def import_movies(self, paths: list[str], root: str = "", job: Job | None = None, params: dict | None = None,
                      label: str = "", live: bool = False) -> Job:
        """Register movies (a new Import job, or more movies for a running one: live acquisition, live=True keeps
        it running). root: the folder their sub-directories are taken relative to (default: their common parent).
        label: default the root folder's name."""
        paths = [os.path.abspath(p) for p in paths]
        if job is None:
            root = os.path.abspath(root or os.path.commonpath([os.path.dirname(p) for p in paths]))
            job = self.new_job("Import", dict(params or {}, root=root), label=label or os.path.basename(root))
        root = job.params["root"]
        with self._db() as c:
            have = {r[0] for r in c.execute("SELECT path FROM movies")}
            for p in paths:
                if p in have:
                    continue
                sub = os.path.dirname(os.path.relpath(p, root))
                sub = "" if sub.startswith("..") else sub
                c.execute("INSERT INTO movies (path, stem, subdir, fingerprint, import_job, added) VALUES (?,?,?,?,?,?)",
                          (p, os.path.basename(p).split(".")[0], sub, movie_fingerprint(p), job.id, time.time()))
        mv = self.movies(job)
        _save_table(os.path.join(job.path, "movies.npz"), mv)
        summ = dict(movies=len(mv), root=root)
        if live:
            self.write_summary(job, summ)
        else:
            self.set_status(job, "finished", summ)
        return job

    def movies(self, job=None) -> pd.DataFrame:
        with self._db() as c:
            if job is None:
                return pd.read_sql("SELECT * FROM movies ORDER BY id", c)
            return pd.read_sql("SELECT * FROM movies WHERE import_job=? ORDER BY id", c, params=(self.job(job).id,))

    def movie(self, movie_id: int) -> dict:
        with self._db() as c:
            c.row_factory = sqlite3.Row
            return dict(c.execute("SELECT * FROM movies WHERE id=?", (int(movie_id),)).fetchone())

    def done(self, job: Job, movie_id: int) -> bool:
        """Has this job its output for the movie? (A purged job's: only if its file is there again.)"""
        with self._db() as c:
            if not job.params.get("purged"):
                return c.execute(f"SELECT 1 FROM {PER_MOVIE[job.type]} WHERE job_id=? AND movie_id=?",
                                 (job.id, int(movie_id))).fetchone() is not None
            col = dict(micrographs="path, 1", chunks="stack_path, n").get(PER_MOVIE[job.type])
            r = c.execute(f"SELECT {col} FROM {PER_MOVIE[job.type]} WHERE job_id=? AND movie_id=?",
                          (job.id, int(movie_id))).fetchone() if col else None
            return r is not None and (r[1] == 0 or os.path.exists(self.abs(r[0])))

    def forget(self, job: Job, movie_ids=None):
        """Drop a job's catalog rows (all, or for some movies) so that a run makes them again (rebuild)."""
        with self._db() as c:
            if movie_ids is None:
                c.execute(f"DELETE FROM {PER_MOVIE[job.type]} WHERE job_id=?", (job.id,))
            else:
                c.executemany(f"DELETE FROM {PER_MOVIE[job.type]} WHERE job_id=? AND movie_id=?",
                              [(job.id, int(m)) for m in movie_ids])

    # ------------------------------------------------------------------ micrographs (MotionCtf jobs)
    def add_micrograph(self, job: Job, movie_id: int, path: str, apix: float, shape, ctf=None,
                       meta: dict | None = None):
        row = (job.id, int(movie_id), self.rel(path), apix, int(shape[0]), int(shape[1]),
               ctf.defocus_u if ctf else None, ctf.defocus_v if ctf else None, ctf.angle if ctf else None,
               ctf.fit_res_a if ctf else None, json.dumps(meta or {}, default=str), time.time())
        with self._db() as c:
            c.execute("INSERT OR REPLACE INTO micrographs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)

    def micrographs(self, job) -> pd.DataFrame:
        job = self.job(job)
        with self._db() as c:
            return pd.read_sql("SELECT m.*, v.stem, v.subdir, v.fingerprint, v.path AS movie_path FROM micrographs m "
                               "JOIN movies v ON v.id = m.movie_id WHERE m.job_id=? ORDER BY m.movie_id", c,
                               params=(job.id,))

    # ------------------------------------------------------------------ picks (Pick jobs)
    def table(self, job, name: str) -> ShardedTable:
        """A job's sharded per-particle table (picks, particles, results) in its folder."""
        return ShardedTable(os.path.join(self.job(job).path, name))

    def add_picks(self, job: Job, movie_id: int, columns: dict, keep: np.ndarray, threshold: float):
        """All candidates of one micrograph (xy_a (n, 2) and per-candidate features) and which are kept."""
        cols = {k: np.asarray(v) for k, v in columns.items() if k != "xy_a"}
        xy = np.asarray(columns["xy_a"], np.float32).reshape(-1, 2)
        self.table(job, "picks").append(movie_id, dict(x_a=xy[:, 0], y_a=xy[:, 1], keep=np.asarray(keep, bool),
                                                       **{k: v.astype(np.float32) for k, v in cols.items()}))
        with self._db() as c:
            c.execute("INSERT OR REPLACE INTO picks VALUES (?,?,?,?,?,?,?)",
                      (job.id, int(movie_id), "picks", len(keep), int(np.sum(keep)), threshold, time.time()))

    def picks(self, job, movie_id: int) -> dict:
        """Candidates of one micrograph: xy_a (n, 2), features, keep, threshold."""
        job = self.job(job)
        z = self.table(job, "picks").read(movies=[movie_id])
        with self._db() as c:
            th = c.execute("SELECT threshold FROM picks WHERE job_id=? AND movie_id=?", (job.id, int(movie_id))).fetchone()
        out = {k: v for k, v in z.items() if k not in ("x_a", "y_a", "movie")}
        out["xy_a"] = np.stack([z["x_a"], z["y_a"]], 1)
        out["threshold"] = np.float32(th[0] if th else np.nan)
        return out

    def pick_table(self, job) -> pd.DataFrame:
        job = self.job(job)
        with self._db() as c:
            return pd.read_sql("SELECT p.*, v.stem, v.subdir FROM picks p JOIN movies v ON v.id = p.movie_id "
                               "WHERE job_id=? ORDER BY movie_id", c, params=(job.id,))

    # ------------------------------------------------------------------ particles (Extract jobs)
    def add_particles(self, job: Job, movie_id: int, xy_a: np.ndarray, images: np.ndarray, apix: float,
                      columns: dict | None = None, dtype: str = "float32", uid: np.ndarray | None = None,
                      pick_xy_a: np.ndarray | None = None) -> np.ndarray:
        """Particles of one micrograph: centres (n, 2) in A, images (n, b, b) (white protein), extra numeric columns
        (pick score, ...), stored as float32 or float16. uid / pick_xy_a: identity of re-centred particles (the
        uid and key of their original pick; default: from xy_a). Returns their uids."""
        mv = self.movie(movie_id)
        xy = np.asarray(xy_a, np.float64).reshape(-1, 2)
        pick = xy if pick_xy_a is None else np.asarray(pick_xy_a, np.float64).reshape(-1, 2)
        uid = particle_uids(mv["fingerprint"], pick) if uid is None else np.asarray(uid, np.int64)
        stack = job.per_movie("Particles", mv, ".mrcs")
        self.write_mrc(stack, images, apix, dtype)
        cols = dict(uid=uid, x_a=xy[:, 0].astype(np.float32), y_a=xy[:, 1].astype(np.float32),
                    slice=np.arange(len(uid), dtype=np.uint32))
        if pick_xy_a is not None:                    # re-centred: offset from the pick (key = stem:pick position)
            cols.update(dx_a=(xy[:, 0] - pick[:, 0]).astype(np.float32), dy_a=(xy[:, 1] - pick[:, 1]).astype(np.float32))
        cols.update({k: np.asarray(v, np.float32) for k, v in (columns or {}).items()})
        self.table(job, "particles").append(movie_id, cols)
        with self._db() as c:
            c.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                      (job.id, int(movie_id), "particles", self.rel(stack), len(uid), int(images.shape[-1]),
                       apix, time.time()))
        return uid

    def particles(self, job, columns=None, movies=None) -> pd.DataFrame:
        """Particles of an Extract job (all, or of some movie ids), one row each: the job's per-particle columns
        (uid, x_a, y_a, slice, score, ...) and, joined from the micrograph and movie tables: movie_id, stem,
        stack (path of its .mrcs), box, apix, CTF, micrograph path / pixel size / shape. key: particle_keys()."""
        job = self.job(job)
        t = self.table(job, "particles").read(columns, movies)
        df = pd.DataFrame({k: v for k, v in t.items() if k != "movie"})
        df["movie_id"] = t["movie"].astype(np.int64)
        with self._db() as c:
            ch = pd.read_sql("SELECT movie_id, stack_path AS stack, box, apix FROM chunks WHERE job_id=?", c,
                             params=(job.id,))
        mics = self.micrographs(job.params["micrographs"])[
            ["movie_id", "stem", "subdir", "defocus_u", "defocus_v", "defocus_angle", "ctf_fit_res", "path", "apix",
             "ny", "nx"]].rename(columns=dict(path="mic_path", apix="mic_apix", ny="mic_ny", nx="mic_nx"))
        per = [k for k in ("defocus_u", "defocus_v", "defocus_angle") if k in df]
        if not per and "defocus_u" in self.table(job, "particles").columns:   # not among the columns asked for
            d = self.table(job, "particles").read(["defocus_u", "defocus_v", "defocus_angle"], movies)
            per = ["defocus_u", "defocus_v", "defocus_angle"]
            for k in per:
                df[k] = d[k]
        if per:                                       # per-particle CTF (CtfRefine, or carried through a polishing)
            mics = mics.drop(columns=per)
            for k in per:
                df[k] = df[k].astype(np.float64)
        return df.merge(ch, on="movie_id", how="left").merge(mics, on="movie_id", how="left")

    def particle_keys(self, df: pd.DataFrame) -> np.ndarray:
        """Keys (stem:x:y of the original pick) of particle rows from particles()."""
        x = df["x_a"].to_numpy(np.float64) - (df["dx_a"].to_numpy(np.float64) if "dx_a" in df else 0)
        y = df["y_a"].to_numpy(np.float64) - (df["dy_a"].to_numpy(np.float64) if "dy_a" in df else 0)
        return np.array([f"{s}:{a:.1f}:{b:.1f}" for s, a, b in zip(df["stem"], np.round(x, 1), np.round(y, 1))])

    # ------------------------------------------------------------------ selections (Select jobs)
    def select(self, source, uids, label: str = "", note: str = "", inputs=(), rejected=None, job=None) -> Job:
        """A Select job: a set of particle uids (e.g. the good classes of a 2D run on a binned extraction). It
        applies to any extraction of the same picks (extract with select=<this job>). rejected: the uids that were
        considered and left out, kept as the table "rejected" (selected_uids(job, rejected=True)). job: an existing
        Select job waiting for the choice (scheduler.waiting_select) to fill instead of a new one."""
        uids = np.unique(np.asarray(uids, dtype=np.int64))
        src = self.job(source)
        if job is None:
            job = self.new_job("Select", dict(source=src.id, particles=len(uids), note=note), [source, *inputs], label)
        else:
            job = self.job(job)
            job.params.pop("waiting", None)
            self.update_params(job, source=src.id, particles=len(uids), note=note)
            self.set_status(job, "running")
        ex = src if src.type in ("Select", *PARTICLE_JOBS) else self.job(self.ancestor(src, PARTICLE_JOBS))
        p = self.table(ex, "particles").read(["uid"])
        o = np.argsort(p["uid"])
        su = p["uid"][o]

        def write(name, u):
            pos = np.searchsorted(su, u)
            hit = pos < len(o)
            hit[hit] = su[pos[hit]] == u[hit]
            mv = np.zeros(len(u), np.uint32)
            mv[hit] = p["movie"][o[pos[hit]]]       # grouped by micrograph like every particle table
            self.table(job, name).write(dict(uid=u), mv)

        write("particles", uids)
        summ = dict(particles=len(uids), source=src.name, note=note)
        if rejected is not None:
            rej = np.setdiff1d(np.asarray(rejected, dtype=np.int64), uids)
            write("rejected", rej)
            summ["rejected"] = len(rej)
        self.set_status(job, "finished", summ)
        return job

    def selected_uids(self, ref, rejected: bool = False) -> np.ndarray:
        return self.table(ref, "rejected" if rejected else "particles").read(["uid"])["uid"]

    def consolidate(self, job) -> str:
        """End of a job: per-particle tables merge their remaining staged micrographs into shards; a MotionCtf job
        writes micrographs.npz (one row per micrograph: what other programs or agents read without the catalog)."""
        job = self.job(job)
        if job.type == "MotionCtf":
            path = os.path.join(job.path, "micrographs.npz")
            _save_table(path, self.micrographs(job).drop(columns=["meta"]))
            return path
        name = "picks" if job.type == "Pick" else "particles"
        self.table(job, name).close()
        return os.path.join(job.path, name)

    # ------------------------------------------------------------------ overview (for people and agents)
    def summary(self) -> dict:
        return dict(root=self.root, movies=len(self.movies()),
                    jobs=[dict(job=r.name, status=r.status, inputs=[self.job(i).name for i in json.loads(r.inputs)],
                               summary=self.summary_of(int(r.id))) for r in self.jobs().itertuples()])

    def micrograph_stats(self, job) -> dict:
        """Per-micrograph statistics of a MotionCtf job: median, range and outliers (> 4 MAD from the median)."""
        mi = self.micrographs(job)
        if not len(mi):
            return dict(micrographs=0)
        meta = [json.loads(m) for m in mi["meta"]]
        cols = dict(defocus_a=(mi["defocus_u"] + mi["defocus_v"]).to_numpy(float) / 2,
                    astigmatism_a=(mi["defocus_u"] - mi["defocus_v"]).to_numpy(float),
                    ctf_fit_res_a=mi["ctf_fit_res"].to_numpy(float),      # (none for imported micrographs)
                    drift_a=np.array([m.get("motion_path_a", np.nan) for m in meta], dtype=float))
        st = dict(micrographs=len(mi), apix=round(float(mi["apix"].iloc[0]), 4))
        for k, v in cols.items():
            if not np.isfinite(v).any():
                continue
            med = float(np.nanmedian(v))
            mad = float(np.nanmedian(np.abs(v - med))) * 1.4826 + 1e-9
            st[k] = dict(median=round(med, 2), min=round(float(np.nanmin(v)), 2), max=round(float(np.nanmax(v)), 2),
                         outliers=mi["stem"][np.abs(v - med) > 4 * mad].tolist())
        # plausibility: a pixel size 2x too small makes every defocus 4x too small (EMPIAR-10291's headers say 0.616 A
        # for 1.232 A K2 images: 0.3-0.7 um fitted for 1.2-2.8 um; refinements stalled at the first CTF zero, ~10 A)
        warnings = []
        if "defocus_a" in st and st["defocus_a"]["median"] < 6000:
            warnings.append(f"median defocus {st['defocus_a']['median'] / 1e4:.2f} um is unusually low: check the pixel "
                            f"size ({st['apix']} A here; 2x too small gives 4x too little defocus) unless a phase plate "
                            f"was used")
        if "ctf_fit_res_a" in st and st["ctf_fit_res_a"]["median"] < 2.2 * st["apix"]:
            warnings.append(f"CTF fits reach {st['ctf_fit_res_a']['median']} A, near Nyquist ({2 * st['apix']:.2f} A): "
                            f"the micrographs may be binned less than the pixel size says")
        if warnings:
            st["warnings"] = warnings
        return st

    def find(self, key_or_uid) -> dict:
        """Everything about one particle, by key (stem:x:y) or uid: every extraction and selection holding it."""
        hits, sel = [], []
        if isinstance(key_or_uid, str):                  # key -> its movie, then the uid of that pick
            stem, x, y = parse_key(key_or_uid)
            mv = self.movies()
            mv = mv[mv["stem"] == stem]
            if not len(mv):
                return {}
            uid = int(particle_uids(mv["fingerprint"].iloc[0], np.array([[x, y]]))[0])
        else:
            uid = int(key_or_uid)
        for r in self.jobs().itertuples():
            if r.type not in PARTICLE_JOBS:
                continue
            t = self.table(int(r.id), "particles")
            u = t.read(["uid"])
            i = np.flatnonzero(u["uid"] == uid)
            if len(i):
                p = self.particles(int(r.id), movies=[int(u["movie"][i[0]])])
                p = p[p["uid"] == uid]
                p["key"] = self.particle_keys(p)
                hits += [dict(row, job=r.name) for row in p.to_dict("records")]
        if not hits:
            return {}
        for r in self.jobs("Select").itertuples():
            if uid in set(self.selected_uids(int(r.id)).tolist()):
                sel.append(r.name)
        return dict(uid=uid, key=str(hits[0]["key"]), extractions=hits, selected_in=sel)
