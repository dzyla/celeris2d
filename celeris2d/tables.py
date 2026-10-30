"""Sharded per-particle tables (docs/project-design.md, "Particle tables").

A table is a directory:

    index.json                  columns (name -> dtype), shards: file, rows, movie ids and their row ranges
    shard_00000.npz ...         numbers only, uncompressed, one column per member; immutable once written
    staging/<movie id>.npz      micrographs added since the last full shard (live jobs)

Rows are grouped by micrograph. append() writes a micrograph to staging; once `shard_size` micrographs are staged
they are merged into a new shard (never rewritten afterwards); close() merges the rest. Reads see shards and
staging. A crash between writing a shard and removing its staging files is harmless: the index says which movies
are in shards, and staging files of those are ignored.

Why this layout (measured, 10,000 micrographs x 500 particles, cold reads): 40 shards read in 0.13 s from NVMe and
1.6 s over NFS on 1 GbE, against 2.4 s and 7.3 s for 10,000 per-micrograph files (and ~2 min estimated on a local
HDD: one seek per file). Text (names, paths) is not stored per particle: it is 6x the bytes of the numbers.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def _atomic_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_npz(path, cols):
    base = path[:-4]
    tmp = f"{base}.tmp{os.getpid()}.npz"
    np.savez(tmp, **cols)
    os.replace(tmp, path)


class ShardedTable:
    def __init__(self, path: str, shard_size: int = 250):
        self.path = path
        self.shard_size = shard_size
        os.makedirs(os.path.join(path, "staging"), exist_ok=True)
        self._ix = os.path.join(path, "index.json")
        if os.path.exists(self._ix):
            with open(self._ix) as f:
                self.index = json.load(f)
        else:
            self.index = dict(columns={}, shards=[], shard_size=shard_size)
            _atomic_json(self._ix, self.index)

    # ------------------------------------------------------------------ writing
    def _check(self, cols: dict):
        n = {len(v) for v in cols.values()}
        if len(n) != 1:
            raise ValueError(f"columns of different lengths: { {k: len(v) for k, v in cols.items()} }")
        new = False
        for k, v in cols.items():
            dt = np.asarray(v).dtype
            if dt.kind in "OUS":
                raise TypeError(f"column {k}: text is not stored per particle (keep it in the micrograph table)")
            known = self.index["columns"].get(k)
            if known is None:
                self.index["columns"][k] = dt.str
                new = True
            elif known != dt.str:
                raise TypeError(f"column {k}: {dt.str}, table has {known}")
        if new:                                      # readers of a live table see the columns before any shard
            _atomic_json(self._ix, self.index)

    def append(self, movie_id: int, cols: dict, flush: bool = True):
        """Rows of one micrograph (all columns the same length; an empty micrograph is recorded too)."""
        cols = {k: np.asarray(v) for k, v in cols.items()}
        self._check(cols)
        _atomic_npz(os.path.join(self.path, "staging", f"{int(movie_id):08d}.npz"), cols)
        if flush and len(self._staged()) >= self.shard_size:
            self.flush()

    def write(self, cols: dict, movie: np.ndarray):
        """A whole table at once (results of a job over all particles): rows are grouped by `movie` (stable
        order within a micrograph) and split into shards. Returns the row order used (indices into the input)."""
        cols = {k: np.asarray(v) for k, v in cols.items()}
        movie = np.asarray(movie)
        self._check(cols)
        order = np.argsort(movie, kind="stable")
        mv = movie[order]
        ids, starts = np.unique(mv, return_index=True)
        ends = np.append(starts[1:], len(mv))
        for s in range(0, len(ids), self.shard_size):
            a, b = starts[s], ends[min(s + self.shard_size, len(ids)) - 1]
            self._new_shard({k: v[order[a:b]] for k, v in cols.items()}, ids[s:s + self.shard_size],
                            starts[s:s + self.shard_size] - a, ends[s:s + self.shard_size] - starts[s:s + self.shard_size])
        return order

    def _staged(self) -> list[tuple[int, str]]:
        done = self.movies_in_shards()
        out = []
        for f in sorted(glob.glob(os.path.join(self.path, "staging", "*.npz"))):
            m = int(os.path.basename(f)[:-4])
            if m not in done:
                out.append((m, f))
        return out

    def _new_shard(self, cols, movie_ids, starts, counts):
        k = len(self.index["shards"])
        name = f"shard_{k:05d}.npz"
        _atomic_npz(os.path.join(self.path, name), cols)
        rows = int(len(next(iter(cols.values())))) if cols else 0
        self.index["shards"].append(dict(file=name, rows=rows, movies=[int(m) for m in movie_ids],
                                         start=[int(s) for s in starts], count=[int(c) for c in counts]))
        _atomic_json(self._ix, self.index)

    def flush(self, final: bool = False):
        """Merge staged micrographs into shards: full shards only, or everything if final."""
        staged = self._staged()
        while len(staged) >= self.shard_size or (final and staged):
            part, staged = staged[:self.shard_size], staged[self.shard_size:]
            parts = [dict(np.load(f)) for _, f in part]
            counts = [len(next(iter(p.values()))) if p else 0 for p in parts]
            names = list(self.index["columns"])
            cols = {k: np.concatenate([p[k] for p in parts if k in p]) if any(k in p for p in parts)
                    else np.zeros(0, self.index["columns"][k]) for k in names}
            self._new_shard(cols, [m for m, _ in part], np.cumsum([0] + counts[:-1]), counts)
            for _, f in part:
                os.remove(f)

    def close(self):
        self.flush(final=True)

    # ------------------------------------------------------------------ reading
    def movies_in_shards(self) -> set[int]:
        return {m for s in self.index["shards"] for m in s["movies"]}

    def movies(self) -> list[int]:
        """Movie ids in the table (shards, then staging), in row order."""
        return [m for s in self.index["shards"] for m in s["movies"]] + [m for m, _ in self._staged()]

    def __len__(self):
        return sum(s["rows"] for s in self.index["shards"]) + sum(
            len(next(iter(np.load(f).values()))) if np.load(f).files else 0 for _, f in self._staged())

    @property
    def columns(self) -> list[str]:
        return list(self.index["columns"])

    def read(self, columns=None, movies=None) -> dict:
        """Columns (default all) of all rows, or of the given movie ids (only the shards holding them are read).
        Adds `movie` (the movie id of every row) unless the table has that column already."""
        cols = list(columns or self.index["columns"])
        want = None if movies is None else {int(m) for m in np.atleast_1d(movies)}
        out = {k: [] for k in cols}
        mv = []
        for s in self.index["shards"]:
            ms = s["movies"]
            if want is not None and not want.intersection(ms):
                continue
            with np.load(os.path.join(self.path, s["file"])) as z:
                data = {k: z[k] for k in cols if k in z.files}
            for i, m in enumerate(ms):
                if want is not None and m not in want:
                    continue
                a, n = s["start"][i], s["count"][i]
                for k in cols:
                    out[k].append(data[k][a:a + n] if k in data else np.zeros(n, self.index["columns"][k]))
                mv.append(np.full(n, m, np.uint32))
        for m, f in self._staged():
            if want is not None and m not in want:
                continue
            with np.load(f) as z:
                n = len(z[z.files[0]]) if z.files else 0
                for k in cols:
                    out[k].append(z[k] if k in z.files else np.zeros(n, self.index["columns"][k]))
            mv.append(np.full(n, m, np.uint32))
        res = {k: np.concatenate(v) if v else np.zeros(0, self.index["columns"].get(k, "<f4")) for k, v in out.items()}
        if "movie" not in res:
            res["movie"] = np.concatenate(mv) if mv else np.zeros(0, np.uint32)
        return res
