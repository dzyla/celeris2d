"""Particle inputs: RELION .star, cryoSPARC .cs/.csg, or a plain .mrc/.mrcs stack.

Every source exposes: N, box, apix, ctf (N x 6: dfU, dfV [A], astig angle [deg], phase shift [deg],
B-factor, scale), optics, parts (RELION-style DataFrame used for the output star), blocks (star blocks),
chunks() -> (row indices, float32 images) streamed one stack file at a time, and a polarity_hint string.
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import mrcfile
import numpy as np
import pandas as pd

from .engine import Optics
from .star import read_star


class _StackReader:
    """Threaded, bounded read-ahead over (file, slice) image addresses.
    Particles are grouped by stack file once (file_list + per-particle file_idx), never by rescanning."""

    def pick_positions(self):
        return None
    root: str
    file_list: list       # unique relative paths
    file_idx: np.ndarray  # per particle: index into file_list
    slices: np.ndarray    # per particle: 0-based index within the file
    _parts = None
    _extra_blocks: dict = {}

    def _index_files(self, files):
        codes, uniq = pd.factorize(np.asarray(files, dtype=object))
        self.file_list, self.file_idx = list(uniq), codes.astype(np.int32)

    def _read(self, fi, rows):
        with mrcfile.mmap(os.path.join(self.root, self.file_list[fi]), mode="r", permissive=True) as m:
            d = m.data if m.data.ndim == 3 else m.data[None]
            return rows, np.asarray(d[self.slices[rows]], dtype=np.float32)

    def chunks(self, threads: int = 8, shuffle_seed=None):
        """Yield (row_indices, float32 images) per stack file; reads run ahead in a thread pool.
        With shuffle_seed the files come in random order, so any prefix is a random sample."""
        order = np.argsort(self.file_idx, kind="stable")
        counts = np.bincount(self.file_idx, minlength=len(self.file_list))
        groups = [(fi, g) for fi, g in enumerate(np.split(order, np.cumsum(counts)[:-1])) if len(g)]
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(groups)
        with ThreadPoolExecutor(threads) as ex:
            pending = deque(ex.submit(self._read, fi, g) for fi, g in groups[: 2 * threads])
            for nxt in groups[2 * threads:] + [None] * len(pending):
                rows_imgs = pending.popleft().result()
                if nxt is not None:
                    pending.append(ex.submit(self._read, *nxt))
                yield rows_imgs

    def read_all(self) -> np.ndarray:
        out = np.empty((self.N, self.box, self.box), dtype=np.float32)
        for rows, d in self.chunks():
            out[rows] = d
        return out

    @property
    def parts(self) -> pd.DataFrame:
        """RELION-style particle table (built lazily: only needed when writing the output star)."""
        if self._parts is None:
            self._parts = self._build_parts()
        return self._parts

    @property
    def blocks(self) -> dict:
        return {**self._extra_blocks, "particles": self.parts}

    def image_names(self) -> list:
        return [f"{i + 1:06d}@{os.path.join(self.root, self.file_list[f])}"
                for i, f in zip(self.slices, self.file_idx)]

    def subset(self, idx: np.ndarray):
        """Restrict to particles idx (sorted indices into the current table)."""
        self.file_idx, self.slices, self.ctf = self.file_idx[idx], self.slices[idx], self.ctf[idx]
        if self._parts is not None:
            self._parts = self._parts.iloc[idx].reset_index(drop=True)
        if hasattr(self, "cs"):
            self.cs = self.cs[idx]
            self.subsetted = True
        self.N = len(idx)

    def _probe(self):
        with mrcfile.mmap(os.path.join(self.root, self.file_list[0]), mode="r", permissive=True) as m:
            return m.data.shape[-1], float(m.voxel_size.x)


def _find_root(start_dir: str, rel: str) -> str:
    d = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(d, rel)):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            raise SystemExit(f"cannot locate {rel}; pass --project-dir")
        d = nd


# --------------------------------------------------------------------------- RELION
class RelionSource(_StackReader):
    kind = "relion"

    def __init__(self, star_path: str, project_dir=None, no_ctf=False, apix=None):
        blocks = read_star(star_path)
        parts = blocks.pop("particles", None)
        optics = blocks.get("optics")
        if parts is None:  # RELION 3.0 style
            parts = blocks.pop(next(iter(blocks)))
        self._parts, self._extra_blocks = parts, blocks
        names = parts["rlnImageName"].str.split("@", expand=True)
        self.slices = names[0].astype(int).to_numpy() - 1
        self._index_files(names[1].to_numpy())
        self.root = project_dir or _find_root(os.path.dirname(star_path), self.file_list[0])
        self.N = len(parts)
        self.box, apix_hdr = self._probe()

        def col(name, default):
            if name in parts:
                return parts[name].to_numpy(dtype=np.float64)
            if optics is not None and name in optics:
                og = parts["rlnOpticsGroup"].to_numpy() if "rlnOpticsGroup" in parts else np.ones(self.N, int)
                lut = dict(zip(optics["rlnOpticsGroup"], optics[name]))
                return np.array([lut[g] for g in og], dtype=np.float64)
            return np.full(self.N, default, dtype=np.float64)

        self.apix = apix or float(col("rlnImagePixelSize", col("rlnDetectorPixelSize", apix_hdr)[0])[0])
        if optics is not None and "rlnCtfDataAreCtfPremultiplied" in optics \
                and int(optics["rlnCtfDataAreCtfPremultiplied"].iloc[0]):
            raise SystemExit("CTF-premultiplied particles are not supported")
        no_ctf = no_ctf or "rlnDefocusU" not in parts
        self.ctf = np.stack([col("rlnDefocusU", 1e4), col("rlnDefocusV", 1e4), col("rlnDefocusAngle", 0),
                             col("rlnPhaseShift", 0), col("rlnCtfBfactor", 0), col("rlnCtfScalefactor", 1)], 1)
        self.optics = Optics(col("rlnVoltage", 300)[0], col("rlnSphericalAberration", 2.7)[0],
                             col("rlnAmplitudeContrast", 0.1)[0], no_ctf=no_ctf)
        self.polarity_hint = "RELION extraction (normally --invert_contrast: white)"
        self._mic_apix = float(col("rlnMicrographPixelSize", col("rlnMicrographOriginalPixelSize", self.apix)[0])[0])

    def pick_positions(self):
        """(micrograph code per particle, pick position in A) or None."""
        p = self.parts
        if not {"rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"} <= set(p.columns):
            return None
        codes = pd.factorize(p["rlnMicrographName"])[0]
        return codes, p[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(float) * self._mic_apix


# --------------------------------------------------------------------------- cryoSPARC
def read_cs(path: str) -> np.ndarray:
    """Load a .cs file, or a .csg group (merging the fields of all its metafiles row-wise)."""
    return read_cs_groups(path)[0]


def read_cs_groups(path: str):
    """(merged array, {result group: absolute metafile path})."""
    if path.endswith(".cs"):
        arr = np.load(path)
        return arr, {g: os.path.abspath(path) for g in {n.split("/")[0] for n in arr.dtype.names if "/" in n}}
    import yaml
    with open(path) as fh:
        grp = yaml.safe_load(fh)
    base = os.path.dirname(os.path.abspath(path))
    groups = {}
    for gname, res in grp["results"].items():
        groups[gname] = os.path.abspath(os.path.join(base, res["metafile"].lstrip(">")))
    cache = {mf: np.load(mf) for mf in set(groups.values())}
    first = cache[next(iter(groups.values()))]
    n = len(first)
    for mf, arr in cache.items():
        if len(arr) == n and np.array_equal(arr["uid"], first["uid"]):
            continue
        # same particles in another order (e.g. extraction jobs): reorder rows by uid
        o_first, o_arr = np.argsort(first["uid"]), np.argsort(arr["uid"])
        if len(arr) != n or not np.array_equal(arr["uid"][o_arr], first["uid"][o_first]):
            raise SystemExit(f"{path}: metafiles describe different particles ({os.path.basename(mf)})")
        perm = np.empty(n, np.int64)
        perm[o_first] = o_arr
        cache[mf] = arr[perm]
    # each result group comes only from its own metafile (as in cryoSPARC); other files may carry
    # stale copies of the same fields
    descr = [("uid", "<u8")]
    for g, mf in groups.items():
        descr += [d for d in cache[mf].dtype.descr if d[0].startswith(g + "/")]
    out = np.empty(n, dtype=descr)
    out["uid"] = first["uid"]
    for g, mf in groups.items():
        for f in cache[mf].dtype.names:
            if f.startswith(g + "/"):
                out[f] = cache[mf][f]
    return out, groups


class CryosparcSource(_StackReader):
    kind = "cryosparc"

    def __init__(self, path: str, project_dir=None, no_ctf=False, apix=None):
        cs, self.cs_groups = read_cs_groups(path)
        self.subsetted = False
        if "blob/path" not in cs.dtype.names:
            raise SystemExit(f"{path} has no particle blob (image) fields")
        self.cs = cs
        self.N = len(cs)
        codes, uniq = pd.factorize(cs["blob/path"])
        self.file_list, self.file_idx = [u.decode() for u in uniq], codes.astype(np.int32)
        self.slices = cs["blob/idx"].astype(np.int64)
        # blob paths are relative to the cryoSPARC project directory (parent of the Jxx job dir)
        self.root = project_dir or _find_root(os.path.dirname(os.path.abspath(path)), self.file_list[0])
        self.box = int(cs["blob/shape"][0][0])
        self.apix = apix or float(cs["blob/psize_A"][0])
        f = cs.dtype.names
        has_ctf = "ctf/df1_A" in f
        no_ctf = no_ctf or not has_ctf

        def fld(name, default, conv=lambda v: v):
            return conv(cs[name].astype(np.float64)) if name in f else np.full(self.N, default, np.float64)

        self.ctf = np.stack([fld("ctf/df1_A", 1e4), fld("ctf/df2_A", 1e4), fld("ctf/df_angle_rad", 0, np.degrees),
                             fld("ctf/phase_shift_rad", 0, np.degrees), fld("ctf/bfactor", 0),
                             fld("ctf/scale", 1)], 1)
        kv, csmm, q0 = (fld(n, d)[0] for n, d in (("ctf/accel_kv", 300), ("ctf/cs_mm", 2.7), ("ctf/amp_contrast", 0.1)))
        for n in ("ctf/accel_kv", "ctf/cs_mm", "ctf/amp_contrast"):
            if n in f and np.ptp(cs[n]) > 1e-3:
                print(f"[warn] {n} varies between particles; using the first value")
        self.optics = Optics(kv, csmm, q0, no_ctf=no_ctf)
        sign = cs["blob/sign"][0] if "blob/sign" in f else None
        self.polarity_hint = f"cryoSPARC blob/sign={sign:+.0f}" if sign is not None else None

        self._extra_blocks = {"optics": pd.DataFrame({
            "rlnOpticsGroupName": ["opticsGroup1"], "rlnOpticsGroup": [1], "rlnVoltage": [kv],
            "rlnSphericalAberration": [csmm], "rlnAmplitudeContrast": [q0], "rlnImagePixelSize": [self.apix],
            "rlnImageSize": [self.box], "rlnImageDimensionality": [2], "rlnCtfDataAreCtfPremultiplied": [0]})}
        self._has_ctf = has_ctf

    def pick_positions(self):
        f = self.cs.dtype.names
        if "location/micrograph_uid" not in f:
            return None
        cs = self.cs
        shp = cs["location/micrograph_shape"].astype(float)
        ps = cs["location/micrograph_psize_A"].astype(float)
        xy = np.stack([cs["location/center_x_frac"] * shp[:, 1] * ps, cs["location/center_y_frac"] * shp[:, 0] * ps], 1)
        return pd.factorize(cs["location/micrograph_uid"])[0], xy

    def _build_parts(self):
        """RELION view of the same particles (for the output .star)."""
        cs, f = self.cs, self.cs.dtype.names
        df = {"rlnImageName": self.image_names()}
        if "location/micrograph_path" in f:
            df["rlnMicrographName"] = np.char.decode(cs["location/micrograph_path"].astype(bytes))
            shp = cs["location/micrograph_shape"]  # (ny, nx)
            df["rlnCoordinateX"] = cs["location/center_x_frac"] * shp[:, 1]
            df["rlnCoordinateY"] = cs["location/center_y_frac"] * shp[:, 0]
        if self._has_ctf:
            df.update(rlnDefocusU=self.ctf[:, 0], rlnDefocusV=self.ctf[:, 1], rlnDefocusAngle=self.ctf[:, 2],
                      rlnPhaseShift=self.ctf[:, 3], rlnCtfBfactor=self.ctf[:, 4], rlnCtfScalefactor=self.ctf[:, 5])
        df["rlnOpticsGroup"] = np.ones(self.N, dtype=int)
        return pd.DataFrame(df)


# --------------------------------------------------------------------------- plain stack
class StackSource(_StackReader):
    kind = "stack"

    def __init__(self, path: str, project_dir=None, no_ctf=True, apix=None):
        path = os.path.abspath(path)
        with mrcfile.mmap(path, mode="r", permissive=True) as m:
            n = m.data.shape[0] if m.data.ndim == 3 else 1
            self.box = m.data.shape[-1]
            self.apix = apix or float(m.voxel_size.x)
        if not self.apix or self.apix <= 0:
            raise SystemExit("pixel size missing from the stack header; pass --apix")
        self.N, self.root = n, "/"
        self.file_list, self.file_idx = [path.lstrip("/")], np.zeros(n, dtype=np.int32)
        self.slices = np.arange(n)
        self._extra_blocks = {"optics": pd.DataFrame({
            "rlnOpticsGroupName": ["opticsGroup1"], "rlnOpticsGroup": [1], "rlnImagePixelSize": [self.apix],
            "rlnImageSize": [self.box], "rlnImageDimensionality": [2], "rlnCtfDataAreCtfPremultiplied": [0]})}
        self.ctf = np.tile([1e4, 1e4, 0, 0, 0, 1], (n, 1)).astype(np.float64)
        self.optics = Optics(300.0, 2.7, 0.1, no_ctf=True)
        self.polarity_hint = None

    def _build_parts(self):
        return pd.DataFrame({"rlnImageName": self.image_names(), "rlnOpticsGroup": np.ones(self.N, dtype=int)})


def open_source(path: str, project_dir=None, no_ctf=False, apix=None):
    if path.endswith((".cs", ".csg")):
        return CryosparcSource(path, project_dir, no_ctf, apix)
    if path.endswith((".mrcs", ".mrc")):
        return StackSource(path, project_dir, True, apix)
    return RelionSource(path, project_dir, no_ctf, apix)


def load_particles(path: str, project_dir=None):
    """Convenience for small data sets / scripts: everything in memory."""
    src = open_source(path, project_dir)
    return src.blocks, src.parts, src.read_all(), src.ctf, src.apix, src.optics


def find_duplicates(mic, xy_a, shift_a, score, min_dist_a):
    """Boolean mask of duplicates: within one micrograph, aligned centres (pick + fitted shift) closer than
    min_dist_a; the higher-scoring particle of each close pair is kept (greedy, best first)."""
    from scipy.spatial import cKDTree
    centre = np.asarray(xy_a, dtype=np.float64) + shift_a
    span = np.ptp(centre, axis=0).max() + 10 * min_dist_a + 1.0
    pts = centre + np.stack([np.asarray(mic, dtype=np.float64) * span, np.zeros(len(mic))], 1)  # mics far apart
    pairs = cKDTree(pts).query_pairs(min_dist_a, output_type="ndarray")
    dup = np.zeros(len(mic), bool)
    if len(pairs) == 0:
        return dup
    rank = np.empty(len(score), np.int64)
    rank[np.argsort(-np.asarray(score), kind="stable")] = np.arange(len(score))
    i, j = pairs[:, 0], pairs[:, 1]
    better = np.where(rank[i] < rank[j], i, j)
    worse = np.where(rank[i] < rank[j], j, i)
    order = np.argsort(rank[better], kind="stable")  # a particle's own fate is settled before its pairs are used
    for bt, wr in zip(better[order], worse[order]):
        if not dup[bt]:
            dup[wr] = True
    return dup
