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
import sqlite3
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

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

JOB_TYPES = ("Import", "MotionCtf", "Pick", "Extract", "Select", "Class2D", "InitialModel")
PER_MOVIE = dict(MotionCtf="micrographs", Pick="picks", Extract="chunks")
HEAVY = dict(MotionCtf=(".mrc",), Extract=(".mrcs",))       # purgeable files of a job's data folder


def movie_fingerprint(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
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
        return self.file("figures", (movie["subdir"] or "").replace(os.sep, "_") + "_" * bool(movie["subdir"])
                         + movie["stem"] + ".png")

    def __str__(self):
        return self.name


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
        with self._db() as c:
            n = (c.execute("SELECT MAX(id) FROM jobs").fetchone()[0] or 0) + 1
            name = f"{n:03d}_{type.lower()}" + (f"_{label}" if label else "")
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                      (n, type, name, label, json.dumps(params or {}, default=str), json.dumps(inputs), "running",
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

    def set_status(self, job: Job, status: str, summary: dict | None = None):
        """running / finished / failed / aborted; summary: the job's key numbers (summary.json)."""
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
        label = _slug(label)
        name = f"{job.id:03d}_{job.type.lower()}" + (f"_{label}" if label else "")
        os.rename(job.path, os.path.join(self.root, "jobs", name))
        with self._db() as c:
            c.execute("UPDATE jobs SET name=?, label=?, updated=? WHERE id=?", (name, label, time.time(), job.id))
        job = self.job(job.id)
        self._write_job_json(job)
        self.write_index()
        return job

    def write_summary(self, job: Job, summary: dict):
        _atomic(os.path.join(job.path, "summary.json"),
                lambda t: json.dump(summary, open(t, "w"), indent=1, default=str))

    def summary_of(self, ref) -> dict:
        p = os.path.join(self.job(ref).path, "summary.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def _write_job_json(self, job: Job):
        d = dict(id=job.id, type=job.type, name=job.name, label=job.label, status=job.status,
                 inputs={i: self.job(i).name for i in job.inputs}, params=job.params, data=self.rel(job.data_path),
                 celeris2d=_version(), updated=time.time())
        _atomic(os.path.join(job.path, "job.json"), lambda t: json.dump(d, open(t, "w"), indent=1, default=str))

    def write_index(self):
        """JOBS.md: one line per job, readable without any tool."""
        rows = ["| job | status | inputs | key numbers |", "|---|---|---|---|"]
        for r in self.jobs().itertuples():
            s = self.summary_of(int(r.id))
            key = ", ".join(f"{k} {v}" if not isinstance(v, dict) else f"{k} {v['median']:g}"
                            for k, v in s.items()
                            if isinstance(v, (int, float, str)) or (isinstance(v, dict) and "median" in v))[:200]
            ins = ", ".join(f"{i:03d}" for i in json.loads(r.inputs))
            rows.append(f"| {r.name} | {r.status} | {ins} | {key} |")
        _atomic(os.path.join(self.root, "JOBS.md"), lambda t: open(t, "w").write("\n".join(rows) + "\n"))

    def lineage(self, ref) -> list[str]:
        """The job and every job it was made from (folder names), nearest first."""
        out, todo = [], [self.job(ref).id]
        while todo:
            j = self.job(todo.pop(0))
            if j.name not in out:
                out.append(j.name)
                todo += j.inputs
        return out

    def ancestor(self, ref, type: str) -> Job | None:
        """The nearest job of this type the job descends from (or the job itself)."""
        return next((self.job(n) for n in self.lineage(ref) if self.job(n).type == type), None)

    def dependents(self, ref) -> list[str]:
        i = self.job(ref).id
        return [r.name for r in self.jobs().itertuples() if i in json.loads(r.inputs)]

    def delete_job(self, ref, force: bool = False) -> str:
        """Move a job (folder and data) to Trash/. Refuses if other jobs used it, unless force."""
        job = self.job(ref)
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
        n = 0
        for ext in HEAVY.get(job.type, ()):
            for f in glob.glob(os.path.join(job.data_path, "**", "*" + ext), recursive=True):
                n += os.path.getsize(f)
                os.remove(f)
        self.update_params(job, purged=True)
        self.write_index()
        return n / 1e9

    def write_mrc(self, path: str, data: np.ndarray, apix: float, dtype: str = "float32"):
        """MRC/MRCS in float32 (mode 2) or float16 (mode 12: RELION >= 4, cryoSPARC >= 4; half the space)."""
        import mrcfile

        def w(t):
            with mrcfile.new(t, overwrite=True) as m:
                m.set_data(np.ascontiguousarray(data, dtype=np.float16 if dtype == "float16" else np.float32))
                m.voxel_size = apix
        _atomic(path, w)

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
        """Has this job its output for the movie?"""
        with self._db() as c:
            return c.execute(f"SELECT 1 FROM {PER_MOVIE[job.type]} WHERE job_id=? AND movie_id=?",
                             (job.id, int(movie_id))).fetchone() is not None

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
    def add_picks(self, job: Job, movie_id: int, columns: dict, keep: np.ndarray, threshold: float):
        """All candidates of one micrograph (xy_a and features) and which are kept."""
        path = job.per_movie("Candidates", self.movie(movie_id), ".npz")
        _atomic(path, lambda t: np.savez(t, keep=np.asarray(keep, bool), threshold=threshold,
                                         **{k: np.asarray(v) for k, v in columns.items()}))
        with self._db() as c:
            c.execute("INSERT OR REPLACE INTO picks VALUES (?,?,?,?,?,?,?)",
                      (job.id, int(movie_id), self.rel(path), len(keep), int(np.sum(keep)), threshold, time.time()))

    def picks(self, job, movie_id: int) -> dict:
        job = self.job(job)
        with self._db() as c:
            r = c.execute("SELECT path FROM picks WHERE job_id=? AND movie_id=?", (job.id, int(movie_id))).fetchone()
        z = np.load(self.abs(r[0]))
        return {k: z[k] for k in z.files}

    def pick_table(self, job) -> pd.DataFrame:
        job = self.job(job)
        with self._db() as c:
            return pd.read_sql("SELECT p.*, v.stem, v.subdir FROM picks p JOIN movies v ON v.id = p.movie_id "
                               "WHERE job_id=? ORDER BY movie_id", c, params=(job.id,))

    # ------------------------------------------------------------------ particles (Extract jobs)
    def add_particles(self, job: Job, movie_id: int, xy_a: np.ndarray, images: np.ndarray, apix: float,
                      columns: dict | None = None, dtype: str = "float32") -> np.ndarray:
        """Particles of one micrograph: positions (n, 2) in A, images (n, b, b) (white protein), extra columns
        (pick score, ...), stored as float32 or float16. Returns their uids."""
        mv = self.movie(movie_id)
        uid = particle_uids(mv["fingerprint"], xy_a)
        stack = job.per_movie("Particles", mv, ".mrcs")
        self.write_mrc(stack, images, apix, dtype)
        table = job.per_movie("Particles", mv, ".npz")
        _atomic(table, lambda t: np.savez(t, uid=uid, key=particle_keys(mv["stem"], xy_a), x_a=np.asarray(xy_a)[:, 0],
                                          y_a=np.asarray(xy_a)[:, 1],
                                          **{k: np.asarray(v) for k, v in (columns or {}).items()}))
        with self._db() as c:
            c.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                      (job.id, int(movie_id), self.rel(table), self.rel(stack), len(uid), int(images.shape[-1]),
                       apix, time.time()))
        return uid

    def particles(self, job) -> pd.DataFrame:
        """All particles of an Extract job, one row each: uid, key, position, extra columns, stack + slice, and the
        CTF of its micrograph (from the MotionCtf job the extraction used)."""
        job = self.job(job)
        with self._db() as c:
            ch = pd.read_sql("SELECT * FROM chunks WHERE job_id=? ORDER BY movie_id", c, params=(job.id,))
        mics = self.micrographs(job.params["micrographs"]).set_index("movie_id")
        parts = []
        for r in ch.itertuples():
            z = np.load(self.abs(r.table_path))
            df = pd.DataFrame({k: z[k] for k in z.files})
            df["movie_id"], df["stack"], df["slice"] = r.movie_id, r.stack_path, np.arange(len(df))
            df["box"], df["apix"] = r.box, r.apix
            m = mics.loc[r.movie_id]
            for k in ("defocus_u", "defocus_v", "defocus_angle", "ctf_fit_res", "stem"):
                df[k] = m[k]
            df["mic_path"], df["mic_apix"], df["mic_ny"], df["mic_nx"] = m["path"], m["apix"], m["ny"], m["nx"]
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # ------------------------------------------------------------------ selections (Select jobs)
    def select(self, source, uids, label: str = "", note: str = "", inputs=()) -> Job:
        """A Select job: a set of particle uids (e.g. the good classes of a 2D run on a binned extraction). It
        applies to any extraction of the same picks (extract with select=<this job>)."""
        uids = np.unique(np.asarray(uids, dtype=np.int64))
        job = self.new_job("Select", dict(source=self.job(source).id, particles=len(uids), note=note),
                           [source, *inputs], label)
        _save_table(os.path.join(job.path, "particles.npz"), pd.DataFrame(dict(uid=uids)))
        self.set_status(job, "finished", dict(particles=len(uids), source=self.job(source).name, note=note))
        return job

    def selected_uids(self, ref) -> np.ndarray:
        return np.load(os.path.join(self.job(ref).path, "particles.npz"))["uid"]

    def consolidate(self, job) -> str:
        """One table for the whole job in its folder (micrographs.npz / picks.npz / particles.npz): fast to load at
        any size, and what other programs or agents read without the catalog."""
        job = self.job(job)
        if job.type == "MotionCtf":
            df, name = self.micrographs(job).drop(columns=["meta"]), "micrographs.npz"
        elif job.type == "Pick":
            rows = []
            for r in self.pick_table(job).itertuples():
                z = self.picks(job, r.movie_id)
                k = z["keep"]
                rows.append(pd.DataFrame(dict(movie_id=r.movie_id, stem=r.stem, x_a=z["xy_a"][k, 0], y_a=z["xy_a"][k, 1],
                                              score=z["score"][k])))
            df, name = (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), "picks.npz"
        elif job.type == "Extract":
            df, name = self.particles(job), "particles.npz"
        else:
            raise ValueError(job.type)
        path = os.path.join(job.path, name)
        _save_table(path, df)
        return path

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
        cols = dict(defocus_a=(mi["defocus_u"] + mi["defocus_v"]).to_numpy() / 2,
                    astigmatism_a=(mi["defocus_u"] - mi["defocus_v"]).to_numpy(),
                    ctf_fit_res_a=mi["ctf_fit_res"].to_numpy(),
                    drift_a=np.array([m.get("motion_path_a", np.nan) for m in meta], dtype=float))
        st = dict(micrographs=len(mi), apix=round(float(mi["apix"].iloc[0]), 4))
        for k, v in cols.items():
            if not np.isfinite(v).any():
                continue
            med = float(np.nanmedian(v))
            mad = float(np.nanmedian(np.abs(v - med))) * 1.4826 + 1e-9
            st[k] = dict(median=round(med, 2), min=round(float(np.nanmin(v)), 2), max=round(float(np.nanmax(v)), 2),
                         outliers=mi["stem"][np.abs(v - med) > 4 * mad].tolist())
        return st

    def find(self, key_or_uid) -> dict:
        """Everything about one particle, by key (stem:x:y) or uid: every extraction and selection holding it."""
        hits, sel = [], []
        for r in self.jobs().itertuples():
            if r.type == "Extract":
                p = self.particles(int(r.id))
                if not len(p):
                    continue
                h = p[p["key"] == key_or_uid] if isinstance(key_or_uid, str) else p[p["uid"] == int(key_or_uid)]
                hits += [dict(row, job=r.name) for row in h.to_dict("records")]
        if not hits:
            return {}
        uid = int(hits[0]["uid"])
        for r in self.jobs("Select").itertuples():
            if uid in set(self.selected_uids(int(r.id)).tolist()):
                sel.append(r.name)
        return dict(uid=uid, key=str(hits[0]["key"]), extractions=hits, selected_in=sel)
