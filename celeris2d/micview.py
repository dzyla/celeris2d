"""Micrographs of a movie job for the dashboard, while it runs and after (cryoSPARC Live-like): the per-micrograph
numbers so far (mic_rows: CTF, motion, picks) and, per micrograph, preview files the dashboard's worker makes as
micrographs come in (warm: data/NNN/Preview/<stem>.png raw binned micrograph, .ps.png spectrum beside the fitted CTF,
.json the 1D CTF fit, global and local motion, patch defocus) - the page filters and draws from them. Read-only
for the jobs' own files; the job never waits for any of this.
"""
from __future__ import annotations

import functools
import glob
import io
import json
import math
import os

import numpy as np

from .project import Project


def _mc_job(project: Project, job):
    j = project.job(job)
    return j if j.type == "MotionCtf" else project.ancestor(j, "MotionCtf")


def _stats_dir(project: Project, mc) -> str:
    return os.path.join(project.root, "data", f"{mc.id:03d}", "Stats")


def _stats_files(project: Project, mc) -> list[str]:
    """Every micrograph's stats file: Stats/<movie subdir>/<stem>.json (Project.per_movie; the subdir is the movies'
    folder under the import root, often one level or more)."""
    return glob.glob(os.path.join(_stats_dir(project, mc), "**", "*.json"), recursive=True)


def _stats_path(project: Project, mc, stem: str) -> str:
    p = os.path.join(_stats_dir(project, mc), stem + ".json")
    if os.path.exists(p):
        return p
    mv = project.movies()
    sub = mv[mv["stem"] == stem]["subdir"]
    if len(sub) and sub.iloc[0]:
        return os.path.join(_stats_dir(project, mc), sub.iloc[0], stem + ".json")
    return p


def _ctf_row(s: dict) -> dict:
    du, dv = s.get("defocus_u"), s.get("defocus_v")
    r = dict(defocus_a=None if du is None else round(0.5 * (du + dv)),
             astigmatism_a=None if du is None else round(abs(du - dv)), angle=s.get("angle"),
             ctf_fit_a=None if s.get("fit_res_a") is None else round(float(s["fit_res_a"]), 2),
             ctf_score=s.get("ctf_score"), drift_a=s.get("motion_path_a"), local_rms_a=s.get("local_rms_a"),
             tilt_deg=s.get("ctf_tilt_deg"), frames=s.get("frames"))
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}


def family(project: Project, job) -> dict:
    """The movie jobs shown together: the MotionCtf job, its Pick job and that Pick job's Extract job (the job asked
    about, or the latest one; a running one first) - motion, CTF and picks of each micrograph on one panel."""
    j = project.job(job)
    mc = _mc_job(project, j)
    out = dict(mc=mc, pick=None, extract=None)
    if mc is None:
        return out
    jobs = project.jobs()

    def latest(typ, key, parent):
        c = [project.job(int(r.id)) for r in jobs[jobs["type"] == typ].itertuples()]
        c = [x for x in c if int(x.params.get(key, -1)) == parent.id and x.status not in ("cancelled", "deleted")]
        c.sort(key=lambda x: (x.status == "running", x.id))
        return c[-1] if c else None
    out["pick"] = j if j.type in ("Pick", "ManualPick") else (
        project.job(j.params["picks"]) if j.type == "Extract" else latest("Pick", "micrographs", mc))
    if out["pick"] is not None:
        out["extract"] = j if j.type == "Extract" else latest("Extract", "picks", out["pick"])
    return out


def mic_rows(project: Project, job) -> list[dict]:
    """One row per micrograph done so far, in processing order: stem, its CTF / motion numbers and - from the Pick and
    Extract jobs of the same chain (family) - candidates, picks, particles."""
    fam = family(project, job)
    mc = fam["mc"]
    if mc is None:
        return []
    rows = {}
    for p in _stats_files(project, mc):
        try:
            with open(p) as fh:
                s = json.load(fh)
        except (OSError, ValueError):                 # being written
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        rows[stem] = dict(stem=stem, t=os.path.getmtime(p), **_ctf_row(s))
    if fam["pick"] is not None:
        for r in project.pick_table(fam["pick"]).itertuples():
            if r.stem in rows:
                rows[r.stem].update(candidates=int(r.n_candidates), picks=int(r.n_kept))
    if fam["extract"] is not None:
        with project._db() as c:
            got = c.execute("SELECT v.stem, k.n FROM chunks k JOIN movies v ON v.id = k.movie_id WHERE k.job_id=?",
                            (fam["extract"].id,)).fetchall()
        for stem, n in got:
            if stem in rows:
                rows[stem]["particles"] = int(n)
    out = sorted(rows.values(), key=lambda r: r["t"])
    for i, r in enumerate(out):
        r["i"] = i + 1
        r["t"] = round(r["t"], 1)
    return out


def _micrograph(project: Project, mc, stem: str):
    with open(_stats_path(project, mc, stem)) as fh:
        s = json.load(fh)
    import mrcfile
    with mrcfile.open(project.abs(s["micrograph"]), permissive=True) as f:
        img = np.asarray(f.data, np.float32)
    return img, float(s.get("apix") or mc.params["apix"]), s


def _bin(img: np.ndarray, size: int):
    """Block-mean binning to <= size pixels on the long side."""
    f = max(1, int(math.ceil(max(img.shape) / size)))
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3)), f


def _png(a: np.ndarray, lo_hi=(0.5, 99.5)) -> "Image":
    from PIL import Image
    lo, hi = np.percentile(a, lo_hi)
    return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8))


@functools.lru_cache(maxsize=64)
def mic_png(root: str, job_id: int, stem: str, size: int = 1024, picks: bool = True, lowpass_a: float = 20.0) -> bytes:
    """One micrograph for display: binned to `size`, low-passed to `lowpass_a`; a Pick job's candidates drawn as
    circles (kept: blue, rejected: faint)."""
    from PIL import ImageDraw
    from .plots_preprocess import display_blur
    project = Project(root)
    j = project.job(job_id)
    mc = _mc_job(project, j)
    img, apix, _ = _micrograph(project, mc, stem)
    small, f = _bin(img, size)
    small = display_blur(small, apix * f, lowpass_a)
    im = _png(small).convert("RGB")
    pj = family(project, j)["pick"]
    if picks and pj is not None:
        mv = project.movies()
        mid = mv[mv["stem"] == stem]["id"]
        if len(mid):
            try:
                p = project.picks(pj, int(mid.iloc[0]))
            except Exception:                         # noqa: BLE001 - not picked yet
                p = None
            if p is not None and len(p["xy_a"]):
                d = ImageDraw.Draw(im)
                r = 0.5 * float(pj.params.get("diameter_a", 150)) / (apix * f)
                keep = np.asarray(p.get("keep", np.ones(len(p["xy_a"]), bool)), bool)
                for (x, y), k in zip(np.asarray(p["xy_a"]) / (apix * f), keep):
                    if not k:
                        d.ellipse([x - r, y - r, x + r, y + r], outline=(120, 120, 120), width=1)
                for (x, y), k in zip(np.asarray(p["xy_a"]) / (apix * f), keep):
                    if k:
                        d.ellipse([x - r, y - r, x + r, y + r], outline=(57, 135, 229), width=2)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


@functools.lru_cache(maxsize=32)
def _spectrum(root: str, mc_id: int, stem: str, tile: int = 512):
    """The micrograph's tile power spectrum (amplitudes) as the full centred plane, cached: the 2D picture and the
    1D fit plot share it. Computed in the dashboard (its request threads), never in the job's process."""
    import torch
    from .preprocess import power_spectrum
    project = Project(root)
    mc = project.job(mc_id)
    img, apix, s = _micrograph(project, mc, stem)
    ps = power_spectrum(torch.as_tensor(img), apix, tile).sqrt().numpy()
    return _full_plane(ps, tile), apix, s, mc


def _flat(full: np.ndarray, tile: int, apix: float):
    """The spectrum less its smooth radial background (rings stand out); radius map."""
    from scipy import ndimage
    yy, xx = np.indices((tile, tile)) - tile // 2
    r = np.hypot(yy, xx)
    ri = np.round(r).astype(int)
    prof = ndimage.median(full, labels=ri, index=np.arange(ri.max() + 1))
    return full - ndimage.uniform_filter1d(prof, 15)[ri], r, ri


def _ctf2(s: dict, mc, tile: int, apix: float) -> np.ndarray | None:
    import torch
    from .engine import Optics, ctf_2d
    if s.get("defocus_u") is None:
        return None
    p = torch.zeros(1, 6)
    p[0, 0], p[0, 1], p[0, 2], p[0, 5] = s["defocus_u"], s["defocus_v"], s["angle"], 1.0
    opt = Optics(mc.params["kv"], mc.params["cs_mm"], mc.params["amp_contrast"])
    g = torch.as_tensor(np.arange(tile) - tile // 2, dtype=torch.float32)
    return ctf_2d(p, opt, g, g, tile * apix)[0].pow(2).numpy()


def _preview(project: Project, mc, stem: str, kind: str) -> str:
    d = os.path.join(project.root, "data", f"{mc.id:03d}", "Preview")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem + kind)


def _cached(project: Project, mc, stem: str, kind: str, make):
    """A preview file of one micrograph (the dashboard's worker makes them as micrographs come in, see warm()):
    read when it is there and newer than the micrograph's stats, else made and written now."""
    p = _preview(project, mc, stem, kind)
    st = _stats_path(project, mc, stem)
    if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(st):
        with open(p, "rb") as fh:
            data = fh.read()
        return json.loads(data) if kind.endswith(".json") else data
    out = make()
    data = json.dumps(out).encode() if kind.endswith(".json") else out
    with open(p + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(p + ".tmp", p)
    return out


def detail(root: str, job_id: int, stem: str) -> dict:
    project = Project(root)
    mc = _mc_job(project, project.job(job_id))
    return _cached(project, mc, stem, ".json", lambda: _clean_json(mic_detail(root, mc.id, stem)))


def spectrum_png(root: str, job_id: int, stem: str) -> bytes:
    project = Project(root)
    mc = _mc_job(project, project.job(job_id))
    return _cached(project, mc, stem, ".ps.png", lambda: ps_png(root, mc.id, stem))


def raw_png(root: str, job_id: int, stem: str) -> bytes:
    """The micrograph binned to <= 1024 px, unfiltered, 8 bits (median +- 4 sd): the page low-passes and stretches it
    itself (the filter slider), and draws the picks."""
    project = Project(root)
    mc = _mc_job(project, project.job(job_id))

    def make():
        img, apix, _ = _micrograph(project, mc, stem)
        small, _ = _bin(img, 1024)
        m, sd = float(np.median(small)), float(small.std())
        from PIL import Image
        a = np.clip((small - (m - 4 * sd)) / max(8 * sd, 1e-12), 0, 1)
        buf = io.BytesIO()
        Image.fromarray((a * 255).astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
    return _cached(project, mc, stem, ".png", make)


def picks_of(project: Project, job, stem: str) -> dict:
    """The candidates of one micrograph of the chain's Pick job: centres (A), kept or not, the blob diameter."""
    fam = family(project, job)
    pj = fam["pick"]
    if pj is None:
        return dict(xy=[], keep=[])
    mv = project.movies()
    mid = mv[mv["stem"] == stem]["id"]
    try:
        p = project.picks(pj, int(mid.iloc[0]))
    except Exception:                                 # noqa: BLE001 - not picked yet
        return dict(xy=[], keep=[], diameter_a=pj.params.get("diameter_a"))
    keep = np.asarray(p.get("keep", np.ones(len(p["xy_a"]), bool)), bool)
    sc = p.get("score")
    return dict(xy=np.round(np.asarray(p["xy_a"], np.float32), 1).tolist(), keep=keep.astype(int).tolist(),
                score=None if sc is None else np.round(np.asarray(sc, np.float32), 4).tolist(),
                diameter_a=pj.params.get("diameter_a"), job=pj.name)


def warm(project: Project, budget_s: float = 4.0) -> int:
    """The dashboard's preview worker: makes the preview files of micrographs that have none yet - the running
    MotionCtf jobs first, newest micrographs first - for at most budget_s seconds. Returns how many it made."""
    import time
    t0, n = time.time(), 0
    jobs = project.jobs()
    mcs = [project.job(int(r.id)) for r in jobs[jobs["type"] == "MotionCtf"].itertuples()]
    mcs.sort(key=lambda j: (j.status != "running", -j.id))
    for mc in mcs:
        stats = sorted(_stats_files(project, mc), key=os.path.getmtime, reverse=True)
        for st in stats[:400]:
            stem = os.path.splitext(os.path.basename(st))[0]
            if all(os.path.exists(_preview(project, mc, stem, k)) for k in (".json", ".ps.png", ".png")):
                continue
            try:
                detail(project.root, mc.id, stem)
                spectrum_png(project.root, mc.id, stem)
                raw_png(project.root, mc.id, stem)
                n += 1
            except Exception:                         # noqa: BLE001 - being written; next pass
                pass
            if time.time() - t0 > budget_s:
                return n
    return n


def _clean_json(d):
    if isinstance(d, dict):
        return {k: _clean_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean_json(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return float(d) if math.isfinite(d) else None
    if isinstance(d, np.integer):
        return int(d)
    return d


@functools.lru_cache(maxsize=64)
def mic_detail(root: str, job_id: int, stem: str, tile: int = 512) -> dict:
    """Data for the per-micrograph plots (drawn by the browser): the 1D CTF fit - rotational average of the
    spectrum (background subtracted) and of the fitted CTF^2 scaled to it, against resolution - and the motion
    (global frame shifts; patch shifts when there are any)."""
    from scipy import ndimage
    project = Project(root)
    mc = _mc_job(project, project.job(job_id))
    full, apix, s, _ = _spectrum(root, mc.id, stem, tile)
    flat, r, ri = _flat(full, tile, apix)
    n = tile // 2
    idx = np.arange(1, n)
    meas = ndimage.mean(flat, labels=ri, index=idx)
    k = idx / (tile * apix)                                        # 1/A
    loc = s.get("local") or None
    out = dict(stem=stem, res_a=[round(1 / x, 3) for x in k], measured=np.round(meas, 5).tolist(),
               fit_res_a=s.get("fit_res_a"), defocus_u=s.get("defocus_u"), defocus_v=s.get("defocus_v"),
               angle=s.get("angle"), frames=s.get("frames"), shifts=s.get("shifts"), apix=apix,
               shape=s.get("shape"), local=loc, ctf_patch=s.get("ctf_patch"),
               drift_a=s.get("motion_path_a"), local_rms_a=s.get("local_rms_a"))
    c2 = _ctf2(s, mc, tile, apix)
    if c2 is not None:
        fit = ndimage.mean(c2, labels=ri, index=idx)
        band = (k >= 1 / 30) & (k <= 1 / max(float(s.get("fit_res_a") or 6), 2.2 * apix))
        if band.sum() > 5:                      # scaled to the spectrum's local ring amplitude (its envelope)
            w = 15
            mf = ndimage.uniform_filter1d(fit, w)
            sf = np.sqrt(ndimage.uniform_filter1d((fit - mf) ** 2, w)) + 1e-9
            mm = ndimage.uniform_filter1d(meas, w)
            sm = np.sqrt(ndimage.uniform_filter1d((meas - mm) ** 2, w))
            out["fit"] = np.round(mm + (fit - mf) / sf * sm, 5).tolist()
    return {k_: (None if isinstance(v, float) and not math.isfinite(v) else v) for k_, v in out.items()}


@functools.lru_cache(maxsize=64)
def ps_png(root: str, job_id: int, stem: str, tile: int = 512) -> bytes:
    """The micrograph's power spectrum (left half, background subtracted) beside the CTF^2 of its fit (right half),
    centred and zoomed to 1.25 x the fit resolution, with a ring at the fit resolution: Thon rings that continue across
    the middle line = a good fit."""
    from PIL import ImageDraw
    from scipy import ndimage
    project = Project(root)
    mc = _mc_job(project, project.job(job_id))
    full, apix, s, _ = _spectrum(root, mc.id, stem, tile)
    flat, r, _ = _flat(full, tile, apix)
    flat = ndimage.gaussian_filter(flat, 1.0)
    band = (r > tile * apix / 30) & (r < tile * apix / 4)
    lo, hi = np.percentile(flat[band], [1, 99])
    left = np.clip((flat - lo) / max(hi - lo, 1e-12), 0, 1)
    out = left
    c2 = _ctf2(s, mc, tile, apix)
    if c2 is not None:
        out = np.where((np.arange(tile) >= tile // 2)[None, :], c2, left)
    # zoomed to where the rings are: 1.25 x the fit resolution's radius (at most Nyquist)
    rc = int(min(tile // 2, 1.25 * tile * apix / max(float(s.get("fit_res_a") or 0), 5.0)))
    out = out[tile // 2 - rc:tile // 2 + rc, tile // 2 - rc:tile // 2 + rc]
    from PIL import Image
    im = _png(out, (0, 100)).convert("RGB").resize((512, 512), Image.BICUBIC)
    z = 512 / (2 * rc)
    d = ImageDraw.Draw(im)
    d.line([(256, 0), (256, 512)], fill=(60, 60, 60), width=1)
    if s.get("fit_res_a"):
        rr = tile * apix / float(s["fit_res_a"]) * z
        if rr < 256:
            d.ellipse([256 - rr, 256 - rr, 256 + rr, 256 + rr], outline=(250, 178, 25), width=1)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _full_plane(half: np.ndarray, tile: int) -> np.ndarray:
    """A real image's rfft2 half-plane (tile, tile//2+1) as the full, centred plane (tile, tile)."""
    full = np.empty((tile, tile), half.dtype)
    full[:, :tile // 2 + 1] = half
    # F(-ky, -kx) = conj F(ky, kx): column -kx comes from column kx at row -ky
    idx = (-np.arange(tile)) % tile
    full[:, tile // 2 + 1:] = half[idx][:, 1:tile // 2][:, ::-1]
    return np.fft.fftshift(full)
