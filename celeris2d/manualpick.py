"""Manual picking with a picker trained from the picks (a boxer, like EMAN's, on the GPU). The user picks particles by
hand on a few micrographs; train() turns those picks into a few 2D templates (crops from the binned, phase-flipped
micrographs, centred and rotationally aligned into reference-free class averages, ~1 s); suggest() picks the next
micrographs with them (picking.template_candidates: every candidate keeps its score, so the threshold moves without
picking again); the user corrects the suggestions (removed ones are negatives that set the threshold at the next
training), retrains, and finally runs the job, which picks every micrograph through the queue - the micrographs the
user reviewed keep exactly their picks. The output is a Pick job's: a "picks" table and catalog rows (Extract reads it).

    j = manualpick_job(project, mc, ManualPickSettings(diameter_a=130))     status "waiting" (for the user)
    set_picks(project, j, stem, picks=[[x_a, y_a, 0], ...])                 by hand (the dashboard does this)
    train(project, j); suggest(project, j)                                    templates, then the next micrographs
    scheduler.queue (status queued) -> run_manualpick(project, j)            every micrograph

Interactive state (dashboard process): jobs/NNN/data/jNNN_manualpick.json - per micrograph (movie id) the user's
picks [x_a, y_a, src] (src 0: clicked, 1: an accepted suggestion), the rejected suggestions [x_a, y_a], and its state
("suggested": suggestions not yet checked, "reviewed": picks final) and its painted zones ([x_a, y_a, r_a, +1 excluded /
-1 not], in order); the lowest template score (threshold) and the power range (power_lo / power_hi: cryoSPARC's
Inspect Picks power score - the local standard deviation of the band-passed micrograph over its median; high:
carbon edges, ice, contamination, aggregates; low: empty ice); excluded zones found by themselves (zone_maps: cells
of half a diameter denser than ~8 diameters around them by more than zone_dark robust standard deviations - ethane /
ice balls, carbon edges - or over ~4 diameters denser than the micrograph's median by more than zone_mean - carbon,
thick ice); templates switched off ("off": their candidates
are never picked, and the picks that made them - data/jNNN_members.json, each training pick's template - are "junk":
left out of the next trainings and scored as negatives, until put back); the templates' version and training numbers. Every pick trains (clicked and accepted, on any
micrograph), whatever the micrograph's state. Templates: maps/jNNN_templates.mrcs (white protein at the picker's working pixel size, diameter/20).
Suggestions: data/NNN/Suggest/<stem>.npz (all candidates of a micrograph: xy_a, score, ok, template version).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import gpu
from . import picking as pk
from .plotstyle import themed
from .preprocess import _grid, _shift_phase
from .project import Job, Project, logged


@dataclass
class ManualPickSettings:
    diameter_a: float = 0.0             # particle diameter (A): the circle drawn around each pick, the templates'
                                        # window and the spacing of picks (as a Pick job's); 0: measured in the picker
                                        # first (scheduler.change_settings)
    dark: bool = True                   # particles darker than the background (cryo); False for negative stain
    templates: int = 4                  # templates trained from the picks: 2D classes of them (one per ~10 picks
                                        # while there are few)
    suggest: int = 4                    # micrographs picked with the templates per Suggest, for you to check
    min_dist_frac: float = 0.5          # x diameter: the closest two picks may be (template picker and Pick all)


MAX_STD, MAX_MEAN = 2.0, 3.0            # contamination filters of picking.select (the Pick job's defaults); MAX_STD is
                                        # the default power_hi here
ZONE_DARK, ZONE_MEAN = 6.0, 5.0         # default zones (robust sd): contamination, carbon / thick ice - see zone_maps
MATCH = 0.3                             # x diameter: a pick and a candidate this close are the same particle
MAX_TRAIN = 3000                        # picks used for templates at most (random subset beyond)


def manualpick_label(s: ManualPickSettings) -> str:
    return f"d{s.diameter_a:g}" if s.diameter_a > 0 else ""


def sized(job: Job) -> ManualPickSettings:
    """The job's settings; an error while it has no particle diameter."""
    s = settings(job)
    if not s.diameter_a > 0:
        raise ValueError(f"{job.name}: measure the particle first (drag across one on a micrograph)")
    return s


def manualpick_job(project: Project, micrographs, s: ManualPickSettings, label: str = "") -> Job:
    """A ManualPick job, waiting for the user (the dashboard's picker); Pick all queues it."""
    mc = project.job(micrographs)
    j = project.new_job("ManualPick", dict(asdict(s), micrographs=mc.id), [mc], label or manualpick_label(s))
    project.set_status(j, "waiting", dict(waiting="pick particles by hand, train the picker, check its suggestions, "
                                                  "then Pick all"))
    return j


# ------------------------------------------------------------------------------------------- interactive state
_locks: dict = {}
_glock = threading.Lock()


def lock(job: Job) -> threading.Lock:
    """One writer per job in the dashboard process (picks saved while a training reads them)."""
    with _glock:
        return _locks.setdefault(job.path, threading.RLock())


def _state_path(job: Job) -> str:
    return job.out("manualpick.json")


def load_state(job: Job) -> dict:
    p = _state_path(job)
    st = dict(mics={}, threshold=None, power_lo=None, power_hi=MAX_STD, zone_dark=ZONE_DARK, zone_mean=ZONE_MEAN,
              off=[], junk={},
              junk_base={}, version=0, train=None, t_edit=0.0)
    if os.path.exists(p):
        with open(p) as fh:
            st.update(json.load(fh))
    return st


def save_state(job: Job, st: dict):
    p = _state_path(job)
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(st, fh)
    os.replace(tmp, p)


def settings(job: Job) -> ManualPickSettings:
    from .jobspec import settings_of
    return settings_of(job, ManualPickSettings)


def micrographs(project: Project, job: Job):
    return project.micrographs(project.job(job.params["micrographs"]))


def movie_of(project: Project, job: Job, stem: str) -> int:
    mi = micrographs(project, job)
    r = mi[mi["stem"] == stem]
    if not len(r):
        raise KeyError(f"no micrograph {stem} in {project.job(job.params['micrographs']).name}")
    return int(r["movie_id"].iloc[0])


def _editable(job: Job):
    if job.status in ("queued", "running"):
        raise ValueError(f"{job.name} is {job.status}: picks can change when it has finished (or stop it)")


def set_picks(project: Project, job, stem: str, picks=None, rejected=None, state: str | None = None,
              zones=None) -> dict:
    """Store one micrograph's picks ([x_a, y_a, src]), rejected suggestions ([x_a, y_a]), state and painted zones
    ([x_a, y_a, r_a, +1 / -1])."""
    job = project.job(job)
    _editable(job)
    mid = str(movie_of(project, job, stem))
    with lock(job):
        st = load_state(job)
        m = st["mics"].setdefault(mid, dict(picks=[], rejected=[], state=""))
        if picks is not None:
            m["picks"] = [[round(float(p[0]), 1), round(float(p[1]), 1), int(p[2]) if len(p) > 2 else 0] for p in picks]
        if rejected is not None:
            m["rejected"] = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in rejected]
        if zones is not None:
            m["zones"] = [[round(float(z[0]), 1), round(float(z[1]), 1), round(float(z[2]), 1), 1 if z[3] > 0 else -1]
                          for z in zones]
        if state is not None:
            if state not in ("", "suggested", "reviewed"):
                raise ValueError(f"state {state!r}: '', suggested or reviewed")
            m["state"] = state
        m["t"] = st["t_edit"] = time.time()
        if not m["picks"] and not m["rejected"] and not m["state"] and not m.get("zones"):
            st["mics"].pop(mid)
        save_state(job, st)
    return m


_KEEP = object()


def set_threshold(project: Project, job, threshold=_KEEP, power_lo=_KEEP, power_hi=_KEEP, zone_dark=_KEEP,
                  zone_mean=_KEEP) -> dict:
    """The filters of the template picks: lowest score, power range, zones by density (None: no limit / off)."""
    job = project.job(job)
    with lock(job):
        st = load_state(job)
        for k, v in (("threshold", threshold), ("power_lo", power_lo), ("power_hi", power_hi),
                     ("zone_dark", zone_dark), ("zone_mean", zone_mean)):
            if v is not _KEEP:
                st[k] = None if v is None else float(v)
        st["t_edit"] = time.time()
        save_state(job, st)
    return {k: st.get(k) for k in ("threshold", "power_lo", "power_hi", "zone_dark", "zone_mean")}


def _members(job: Job) -> dict:
    """The last training's picks per template: {movie id: [[x_a, y_a, template], ...]}."""
    p = job.find("members.json")
    if p is None:
        return {}
    with open(p) as fh:
        return json.load(fh)


def _junk_of(base: dict, members: dict, off) -> dict:
    """Junk picks: those carried from earlier trainings plus the members of the templates now off."""
    off = {int(k) for k in off}
    out = {k: [list(q) for q in v] for k, v in base.items()}
    for mid, ps in members.items():
        add = [[q[0], q[1]] for q in ps if int(q[2]) in off]
        if add:
            out.setdefault(mid, []).extend(add)
    return {k: v for k, v in out.items() if v}


def _is_junk(xy: np.ndarray, junk) -> np.ndarray:
    if not junk or not len(xy):
        return np.zeros(len(xy), bool)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(junk, np.float64).reshape(-1, 2)).query(xy, distance_upper_bound=1.0)
    return np.isfinite(d)


def set_off(project: Project, job, off) -> list:
    """Templates switched off (0-based): candidates they found best are not picked (e.g. one that averaged ice), and
    the picks that made them leave the next trainings (negatives instead)."""
    job = project.job(job)
    with lock(job):
        st = load_state(job)
        st["off"] = sorted({int(k) for k in off})
        st["junk"] = _junk_of(st.get("junk_base") or {}, _members(job), st["off"])
        st["t_edit"] = time.time()
        save_state(job, st)
    return st["off"]


def put_back(project: Project, job) -> int:
    """Every junk pick back into training (templates stay as they are, none off)."""
    job = project.job(job)
    with lock(job):
        st = load_state(job)
        n = sum(len(v) for v in (st.get("junk") or {}).values())
        st.update(junk={}, junk_base={}, off=[], t_edit=time.time())
        save_state(job, st)
    return n


def has_templates(job: Job) -> bool:
    return job.find("templates.mrcs") is not None


def set_settings(project: Project, job, diameter_a=None, templates=None) -> ManualPickSettings:
    """The particle size and the number of templates, at any time but while Pick all runs. A new size drops the
    templates (they are made at the particle's size: train again) and the score range; the picks stay."""
    job = project.job(job)
    _editable(job)
    old = settings(job)
    kw = {}
    if templates is not None:
        kw["templates"] = int(np.clip(int(templates), 1, 50))
    if diameter_a is not None:
        d = float(diameter_a)
        if not 10 <= d <= 5000:
            raise ValueError(f"particle diameter {d:g} A: between 10 and 5000 A")
        kw["diameter_a"] = round(d, 1)
    if not kw:
        return old
    project.update_params(job, **kw)
    if "diameter_a" in kw and kw["diameter_a"] != old.diameter_a:
        with lock(job):
            for nm in ("templates.mrcs", "templates.png", "members.json"):
                p = job.find(nm)
                if p:
                    os.remove(p)
            st = load_state(job)
            st.update(train=None, threshold=None, off=[], junk_base=st.get("junk") or {},
                      t_edit=time.time())
            save_state(job, st)
        job = project.job(job.id)
        if job.label == manualpick_label(old) and job.status != "running":
            job = project.rename(job, manualpick_label(settings(job)))
    return settings(project.job(job.id))


# ------------------------------------------------------------------------------------------------ micrographs
_mics: OrderedDict = OrderedDict()      # (project, MotionCtf job, movie id) -> (micrograph as stored, apix, CTF)
_mics_lock = threading.Lock()


def load_mic(project: Project, mc: Job, row, dev) -> tuple[torch.Tensor, float, object]:
    """A micrograph (catalog row) on dev, with its pixel size and CTF; the last few are kept in host memory (training
    reads the same micrographs again)."""
    import mrcfile
    from .preprocess import CTFFit
    key = (project.root, mc.id, int(row.movie_id))
    with _mics_lock:
        hit = _mics.get(key)
        if hit is not None:
            _mics.move_to_end(key)
    if hit is None:
        with mrcfile.open(project.abs(row.path), permissive=True) as f:
            a = np.array(f.data)
        hit = (a, float(row.apix), CTFFit(row.defocus_u, row.defocus_v, row.defocus_angle, row.ctf_fit_res, 0.0, {}))
        with _mics_lock:
            _mics[key] = hit
            while len(_mics) > 8:
                _mics.popitem(last=False)
    return torch.as_tensor(hit[0], device=dev).float(), hit[1], hit[2]


def _device():
    return gpu.default()


def _gpu_or_cpu(fn):
    """fn(dev) on the GPU, or on the CPU when the GPU has no room (a job of the queue may hold it)."""
    try:
        return fn(_device())
    except gpu.OutOfMemoryError:
        gpu.empty_cache()
        return fn("cpu")


def _work_apix(s: ManualPickSettings) -> float:
    return s.diameter_a / 20                     # picking.template_candidates' working pixel size


# ------------------------------------------------------------------------------- templates from picks (2D classes)
def _rot(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """(n, b, b) each rotated by its own angle (deg, bilinear) about the centre; b odd: about pixel b // 2, the
    centre the FFT shifts use too."""
    a = torch.deg2rad(ang.float().to(x.device))
    c, s = torch.cos(a), torch.sin(a)
    th = torch.zeros(len(a), 2, 3, device=x.device)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = c, -s, s, c
    g = torch.nn.functional.affine_grid(th, (len(a), 1, *x.shape[-2:]), align_corners=False)
    return torch.nn.functional.grid_sample(x[:, None].float(), g, align_corners=False, padding_mode="zeros")[:, 0]


def _shift(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """(n, b, b) moved by (+dx, +dy) px (Fourier, circular)."""
    b = x.shape[-1]
    ky, kx = _grid(b, b, 1.0, x.device)
    return torch.fft.irfft2(torch.fft.rfft2(x) * _shift_phase(ky, kx, dy.float(), dx.float()), s=(b, b))


def _lowpass(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    b = x.shape[-1]
    ky, kx = _grid(b, b, 1.0, x.device)
    return torch.fft.irfft2(torch.fft.rfft2(x) * torch.exp(-2 * math.pi ** 2 * sigma_px ** 2 * (ky * ky + kx * kx)),
                            s=(b, b))


def _radial(x: torch.Tensor) -> torch.Tensor:
    """Rotational average of one image (b, b) about its centre pixel."""
    b = x.shape[-1]
    c = torch.arange(b, device=x.device) - b // 2
    r = torch.sqrt(c[:, None] ** 2 + c[None, :] ** 2).round().long()
    prof = torch.zeros(int(r.max()) + 1, device=x.device).index_add_(0, r.flatten(), x.flatten())
    cnt = torch.bincount(r.flatten(), minlength=len(prof)).clamp_min(1)
    return (prof / cnt)[r]


def _centre(refs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Each image moved so the centre of mass of its positive density inside the mask is on the centre pixel."""
    b = refs.shape[-1]
    c = (torch.arange(b, device=refs.device) - b // 2).float()
    w = (refs * mask).clamp_min(0)
    tot = w.sum((1, 2)).clamp_min(1e-9)
    cy, cx = (w * c[:, None]).sum((1, 2)) / tot, (w * c[None, :]).sum((1, 2)) / tot
    return _shift(refs, -cy, -cx)


def _align(X: torch.Tensor, refs: torch.Tensor, angles: torch.Tensor, mask: torch.Tensor, smax: float):
    """Best reference, angle and shift of every image: X ~ shift_t(rot_a(ref_k)), |t| <= smax px. Returns k, angle
    (deg), dy, dx, score (correlation with the masked, normalised reference)."""
    from .engine import gpu_free
    K, A, b = len(refs), len(angles), X.shape[-1]
    R = _rot(refs.repeat_interleave(A, 0), angles.repeat(K)) * mask
    R = R - (R.sum((1, 2), keepdim=True) / mask.sum()) * mask
    R = R / R.flatten(1).norm(dim=1).clamp_min(1e-9)[:, None, None]
    FR = torch.fft.rfft2(R).conj()
    idx = torch.arange(b, device=X.device)
    circ = (idx + b // 2) % b - b // 2                               # circular index -> signed shift
    allowed = (circ[:, None] ** 2 + circ[None, :] ** 2) <= smax * smax
    per = K * A * b * b * 4 * 4                                       # bytes per image (maps + spectra)
    free = gpu_free(X.device) if gpu.is_gpu(X.device) else 2 << 30
    chunk = int(np.clip(free // 4 // per, 8, 512))
    out = []
    for i in range(0, len(X), chunk):
        cc = torch.fft.irfft2(torch.fft.rfft2(X[i:i + chunk])[:, None] * FR[None], s=(b, b))
        cc = cc.masked_fill(~allowed, -float("inf"))
        v, j = cc.flatten(1).max(1)
        out.append((v, j))
    v = torch.cat([o[0] for o in out])
    j = torch.cat([o[1] for o in out])
    m, pix = j // (b * b), j % (b * b)
    return m // A, angles[m % A], circ[pix // b].float(), circ[pix % b].float(), v


def class_averages(X: torch.Tensor, n_classes: int, radius_px: float, n_iter: int = 10, seed: int = 0):
    """Reference-free 2D classes of picked images (n, b, b), b odd, white protein: centred on their rotational
    average, then k-means in the aligned space (every image to every class at every in-plane angle and a shift
    within a fifth of the diameter; each class re-centred on its centre of mass). Returns the classes (K, b, b),
    each image's class and its score; classes ordered by size."""
    dev, (n, b, _) = X.device, X.shape
    X = X - X.mean((1, 2), keepdim=True)
    X = X / X.std((1, 2), keepdim=True).clamp_min(1e-6)
    c = (torch.arange(b, device=dev) - b // 2).float()
    rr = torch.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    mask = torch.sigmoid((1.2 * radius_px - rr) / 1.0)                # the template window (picking: 0.6 diameter)
    smax = max(1.0, 0.4 * radius_px)
    zero = torch.zeros(1, device=dev)
    ref = _radial(X.mean(0))
    for _ in range(3):                                                # centring on the rotational average
        _, _, dy, dx, _ = _align(X, ref[None], zero, mask, smax)
        ref = _radial(_shift(X, -dy, -dx).mean(0))
    _, _, dy, dx, _ = _align(X, ref[None], zero, mask, smax)
    Xc = _shift(X, -dy, -dx)
    K = max(1, min(int(n_classes), n // 10 if n >= 20 else 1))
    step = float(np.clip(math.degrees(1.0 / max(radius_px, 1.0)), 3.0, 12.0))      # a pixel at the edge
    angles = torch.arange(0, 360, step, device=dev)
    g = torch.Generator().manual_seed(seed)
    if K == 1:
        refs = Xc.mean(0, keepdim=True)
    else:                                                             # seeds: K images, as different as possible
        first = int(torch.randint(n, (1,), generator=g))
        pick = [first]
        flat = (_lowpass(Xc, 1.0) * mask).flatten(1)
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sim = flat @ flat[first]
        for _ in range(K - 1):
            j = int(torch.argmin(sim))
            pick.append(j)
            sim = torch.maximum(sim, flat @ flat[j])
        refs = _lowpass(Xc[pick], 1.5)
    cls = torch.zeros(n, dtype=torch.long, device=dev)
    score = torch.zeros(n, device=dev)
    for it in range(n_iter):
        k, a, dy, dx, score = _align(X, refs, angles, mask, smax)
        Y = _rot(_shift(X, -dy, -dx), -a)
        new = torch.zeros_like(refs)
        new.index_add_(0, k, Y)
        cnt = torch.bincount(k, minlength=K)
        for e in torch.nonzero(cnt == 0).flatten().tolist():                 # an empty class: the worst-fitting image
            w = int(torch.argmin(score))
            new[e], cnt[e], score[w] = Y[w], 1, float("inf")
        refs = _centre(_lowpass(new / cnt[:, None, None].clamp_min(1), 0.5), mask)
        changed = int((k != cls).sum()) if it else n
        cls = k
        if it >= 3 and changed <= max(1, n // 200):
            break
    o = torch.argsort(torch.bincount(cls, minlength=K), descending=True)
    rank = torch.empty_like(o)
    rank[o] = torch.arange(K, device=dev)
    return refs[o], rank[cls], score


def _crop(img: torch.Tensor, xy_px: np.ndarray, b: int) -> torch.Tensor:
    """(n, b, b) boxes (b odd) of img centred on the nearest pixels of xy_px (x, y); zero outside the image."""
    h = b // 2
    p = torch.nn.functional.pad(img, (h, h, h, h))
    x = np.clip(np.round(xy_px[:, 0]).astype(int), 0, img.shape[1] - 1)
    y = np.clip(np.round(xy_px[:, 1]).astype(int), 0, img.shape[0] - 1)
    return torch.stack([p[yy:yy + b, xx:xx + b] for xx, yy in zip(x, y)]) if len(x) else img.new_zeros(0, b, b)


def _scores_at(cand: pk.Candidates, xy_a: np.ndarray, radius_a: float) -> np.ndarray:
    """Best score within radius of each position: from the score map when the candidates kept it (a particle a
    stronger neighbour suppressed as a candidate still scores: nearest candidates missed 15 % of accepted suggestions
    on INX-6), else the nearest candidate's (-inf: none)."""
    from scipy.spatial import cKDTree
    out = np.full(len(xy_a), -np.inf, np.float32)
    if not len(xy_a):
        return out
    sm = cand.work.get("score") if cand.work else None
    if sm is not None:
        wa, (h, w) = cand.work["apix"], sm.shape
        r = int(math.ceil(radius_a / wa))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = dy * dy + dx * dx <= (radius_a / wa) ** 2
        dy, dx = dy[disk], dx[disk]
        p = np.round(np.asarray(xy_a, np.float64).reshape(-1, 2) / wa).astype(int)
        yy, xx = p[:, 1:2] + dy[None], p[:, 0:1] + dx[None]
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        v = np.where(ok, sm[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)], -np.inf)
        return v.max(1).astype(np.float32)
    if not len(cand.xy_a):
        return out
    d, j = cKDTree(cand.xy_a).query(np.asarray(xy_a, np.float64).reshape(-1, 2), k=1, distance_upper_bound=radius_a)
    ok = np.isfinite(d)
    out[ok] = cand.score[j[ok]]
    return out


def choose_threshold(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Score threshold from the scores of the user's picks (pos; -inf: not a candidate at all) and of the suggestions
    the user removed (neg): with a few negatives, the lowest threshold of the best TPR - FPR (generous: 2D cleans
    up); without, the 10th percentile of the picks' scores (keeps ~90 % of what the user would pick)."""
    p = np.asarray(pos, float)
    pf = p[np.isfinite(p)]
    if not len(pf):
        return None
    n = np.asarray(neg, float)
    n = n[np.isfinite(n)]
    if len(n) >= 3:
        cand = np.unique(np.concatenate([pf, n]))
        tpr = (p[None] >= cand[:, None]).mean(1)
        fpr = (n[None] >= cand[:, None]).mean(1)
        j = tpr - fpr
        t = float(cand[np.flatnonzero(j >= j.max() - 1e-9)[0]])
        lower = cand[cand < t]
        return 0.5 * (t + float(lower.max())) if len(lower) else t - 1e-4
    return float(np.quantile(pf, 0.1)) - 1e-4


def _candidates(img, apix, ctf, optics, s: ManualPickSettings, tm: np.ndarray, ta: float, batch: int = 96,
                keep_maps: bool = False):
    """Template candidates of one micrograph and which pass the contamination filters and lie inside its real area."""
    c = pk.template_candidates(img, apix, s.diameter_a, tm, ta, ctf, optics, s.dark, s.min_dist_frac, batch=batch,
                               keep_maps=keep_maps)
    x0, x1, y0, y1 = pk.valid_bounds(img)
    x, y = c.xy_a[:, 0] / apix, c.xy_a[:, 1] / apix
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    ok = inside & (np.abs(c.local_mean) <= MAX_MEAN)                 # the power range is the user's (auto_keep)
    return c, ok, inside


def zone_maps(img: torch.Tensor, apix: float, s: ManualPickSettings) -> dict:
    """Density maps of a micrograph on cells of half a diameter, in robust standard deviations (more density: higher;
    padded edges 0), that find zones to leave out. dark_z: each cell over the mean of ~8 diameters around it -
    contamination (ethane / ice balls), carbon edges; the band-passed power did not find these (a ball's inside is
    flat). mean_z: over ~4 diameters, against the micrograph's median - carbon, thick ice (one-sided: where carbon
    covers most of a micrograph the median is carbon's and nothing is left out)."""
    D = s.diameter_a
    cell = D / 2
    ny, nx = img.shape
    gy, gx = max(1, int(round(ny * apix / cell))), max(1, int(round(nx * apix / cell)))
    m = torch.nn.functional.adaptive_avg_pool2d(img[None, None].float(), (gy, gx))[0, 0] * (-1.0 if s.dark else 1.0)

    def smooth(a, h):
        h = max(1, min(h, (min(gy, gx) - 1) // 2))
        return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(a[None, None], (h, h, h, h), mode="replicate"),
                                              2 * h + 1, 1)[0, 0]

    def robust_z(a):
        med = a.median()
        mad = (a - med).abs().median() * 1.4826
        return ((a - med) / mad.clamp_min(1e-9 * float(a.abs().max()) + 1e-12)).cpu().numpy().astype(np.float32)

    out = dict(cell_a=float(cell), dark_z=robust_z(m - smooth(m, int(round(4 * D / cell)))),
               mean_z=robust_z(smooth(m, int(round(2 * D / cell)))))
    x0, x1, y0, y1 = pk.valid_bounds(img)                             # padded edges: not a zone of their own
    ys, xs = (np.arange(gy) + 0.5) * cell / apix, (np.arange(gx) + 0.5) * cell / apix
    edge = ~(((ys >= y0) & (ys < y1))[:, None] & ((xs >= x0) & (xs < x1))[None, :])
    out["dark_z"][edge], out["mean_z"][edge] = 0.0, 0.0
    return out


def zone_floor(zm: dict, xy_a: np.ndarray) -> float:
    """The lowest contamination limit of a micrograph: twice the 90th percentile of dark_z at its particles (the
    candidates picked) - contamination is far denser than particles. Real cryo particles sit near 0 (dz_2021_08_18:
    median 0), high-contrast ones (negative stain, the synthetic test's: median 6) would be zones of their own."""
    xy = np.asarray(xy_a, np.float64).reshape(-1, 2)
    if len(xy) < 10:
        return 0.0
    d, c = zm["dark_z"], zm["cell_a"]
    v = d[np.clip((xy[:, 1] / c).astype(int), 0, d.shape[0] - 1), np.clip((xy[:, 0] / c).astype(int), 0, d.shape[1] - 1)]
    return float(max(0.0, 2 * np.quantile(v, 0.9)))


def zone_mask(zm: dict, zone_dark, zone_mean, floor: float = 0.0) -> np.ndarray:
    """Cells left out: dark_z above zone_dark (at least floor: zone_floor) or mean_z above zone_mean (None: that test
    off), grown by one cell."""
    bad = np.zeros(zm["mean_z"].shape, bool)
    if zone_dark is not None:
        bad |= zm["dark_z"] > max(zone_dark, floor)
    if zone_mean is not None:
        bad |= zm["mean_z"] > zone_mean
    if bad.any():
        from scipy.ndimage import binary_dilation
        bad = binary_dilation(bad, np.ones((3, 3), bool))
    return bad


def in_zones(xy_a: np.ndarray, zm: dict | None, bad: np.ndarray | None, painted=()) -> np.ndarray:
    """Which points lie in an excluded zone: the cells found by themselves, then the painted strokes in order (+1
    excludes, -1 frees - also a cell found by itself)."""
    xy = np.asarray(xy_a, np.float64).reshape(-1, 2)
    out = np.zeros(len(xy), bool)
    if zm is not None and bad is not None and len(xy):
        c = zm["cell_a"]
        i = np.clip((xy[:, 1] / c).astype(int), 0, bad.shape[0] - 1)
        j = np.clip((xy[:, 0] / c).astype(int), 0, bad.shape[1] - 1)
        out = bad[i, j].copy()
    for x, y, r, sg in painted or ():
        hit = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2 < r * r
        out[hit] = sg > 0
    return out


def templates_of(job: Job):
    """The job's trained templates (n, b, b) and their pixel size, or (None, None)."""
    import mrcfile
    p = job.find("templates.mrcs")
    if p is None:
        return None, None
    with mrcfile.open(p, permissive=True) as f:
        return np.array(f.data, np.float32).reshape(-1, *f.data.shape[-2:]), float(f.voxel_size.x)


def train(project: Project, job, log=print) -> dict:
    """Templates from the picks of the reviewed micrographs (2D classes), and a threshold from how templates score the
    user's picks and the suggestions the user removed. The picks are scored by templates made from the other half of
    the picks (in-sample they score ~0.03 higher than new particles: the templates' noise matches their own: the
    threshold came out too strict, 70 % recall on the synthetic test against 98 % half-set). ~1-5 s."""
    from .pipeline import optics_of
    job = project.job(job)
    _editable(job)
    s = sized(job)
    mc = project.job(job.params["micrographs"])
    optics = optics_of(project, mc)
    t0 = time.time()
    with lock(job):
        st = load_state(job)
    mi = micrographs(project, job).set_index("movie_id", drop=False)
    # every pick trains: clicked or accepted, on reviewed micrographs and on ones with suggestions still to check
    # picks of switched-off templates ("junk") leave training and are scored as negatives
    junk = st.get("junk") or {}
    rev, jxy = {}, {}
    for k, v in sorted(st["mics"].items(), key=lambda kv: int(kv[0])):
        if not v.get("picks") or int(k) not in mi.index:
            continue
        pa = np.asarray(v["picks"], np.float64)
        bad = _is_junk(pa[:, :2], junk.get(k))
        if bad.any():
            jxy[int(k)] = pa[bad, :2]
        if (~bad).any():
            rev[int(k)] = dict(v, picks=[p for p, b in zip(v["picks"], bad) if not b])
    n_junk = sum(len(v) for v in jxy.values())
    n_picks = sum(len(v["picks"]) for v in rev.values())
    n_accepted = sum(sum(1 for p in v["picks"] if len(p) > 2 and p[2]) for v in rev.values())
    if n_picks < 5:
        raise ValueError(f"{n_picks} picks so far: pick at least 5 particles (better 20-50) on a micrograph first")
    wa_target = _work_apix(s)
    rng = np.random.default_rng(0)
    xy = {m: np.asarray(v["picks"], np.float64)[:, :2] for m, v in rev.items()}
    half, off = {}, 0                                                 # alternating halves over all picks
    for m, p in xy.items():
        half[m] = (off + np.arange(len(p))) % 2
        off += len(p)
    use = {m: (rng.random(len(p)) < MAX_TRAIN / n_picks) if n_picks > MAX_TRAIN else np.ones(len(p), bool)
           for m, p in xy.items()}                                    # beyond MAX_TRAIN: the same share of each
    split = int(sum(u.sum() for u in use.values())) >= 20

    members = {}

    def run(dev):
        members.clear()
        patches, halves, wa = [], [], None
        for m, p in xy.items():
            img, apix, ctf = load_mic(project, mc, mi.loc[m], dev)
            im, wa = pk.prepare(img, apix, s.diameter_a, ctf, optics, s.dark, work_apix=wa_target)
            b = 2 * int(math.ceil(0.8 * s.diameter_a / wa)) + 1      # 1.6 diameters, odd (centre pixel b // 2)
            patches.append(_crop(im, p[use[m]] / wa, b))
            halves.append(half[m][use[m]])
            del img
        X, H = torch.cat(patches), np.concatenate(halves)
        rad = 0.5 * s.diameter_a / wa
        refs, cls, _ = class_averages(X, s.templates, rad)
        cl, off_ = cls.cpu().numpy(), 0
        for m, p in xy.items():                                       # each training pick's template
            q = p[use[m]]
            members[str(m)] = [[round(float(a), 1), round(float(b_), 1), int(c_)]
                               for (a, b_), c_ in zip(q, cl[off_:off_ + len(q)])]
            off_ += len(q)
        tm = refs[:, :-1, :-1].cpu().numpy()                          # even box, centre pixel kept at b // 2
        counts = torch.bincount(cls, minlength=len(refs)).cpu().numpy()
        th_tm = [class_averages(X[torch.as_tensor(H == h, device=X.device)], s.templates, rad)[0][:, :-1, :-1]
                 .cpu().numpy() for h in (0, 1)] if split else [tm, tm]
        # the threshold: the picks scored by the other half's templates, the removed suggestions by both (mean)
        pos, neg = [], []
        for k, v in st["mics"].items():
            m = int(k)
            if m not in mi.index or not (m in rev or v.get("rejected") or m in jxy):
                continue
            img, apix, ctf = load_mic(project, mc, mi.loc[m], dev)
            cs = [_candidates(img, apix, ctf, optics, s, t, wa, batch=32, keep_maps=True)[0]
                  for t in (th_tm if split else th_tm[:1])]
            cs = cs if split else cs * 2                                 # [scored by half 0, by half 1]
            if m in rev:
                sc = [_scores_at(c, xy[m], MATCH * s.diameter_a) for c in cs]
                pos.append(np.where(half[m] == 0, sc[1], sc[0]))
            r = np.concatenate([np.asarray(v.get("rejected") or [], np.float64).reshape(-1, 2),
                                jxy.get(m, np.zeros((0, 2)))])
            if len(r):
                neg.append(np.mean([_scores_at(c, r, MATCH * s.diameter_a) for c in cs[:2]], 0))
            del img
        return tm, wa, counts, np.concatenate(pos) if pos else np.zeros(0), np.concatenate(neg) if neg else np.zeros(0)
    tm, wa, counts, pos, neg = _gpu_or_cpu(run)
    th = choose_threshold(pos, neg)
    project.write_mrc(job.out("templates.mrcs"), tm.astype(np.float32), wa, "float32")
    with open(job.out("members.json"), "w") as fh:
        json.dump(members, fh)
    with lock(job):
        st = load_state(job)
        st["version"] = int(st.get("version", 0)) + 1
        st["off"] = []                                                # new templates, new numbering
        st["junk_base"] = st.get("junk") or {}                        # junk stays junk until put back
        if th is not None:
            st["threshold"] = th
        info = dict(version=st["version"], n=len(tm), counts=counts.tolist(), apix=round(wa, 4), box=int(tm.shape[-1]),
                    picks=int(sum(u.sum() for u in use.values())), accepted=n_accepted, junk=n_junk,
                    micrographs=len(xy),
                    half_sets=split,
                    rejected=int(np.isfinite(neg).sum()), missing=int((~np.isfinite(pos)).sum()),
                    pos=np.round(pos[np.isfinite(pos)], 4).tolist(), neg=np.round(neg[np.isfinite(neg)], 4).tolist(),
                    threshold=th, seconds=round(time.time() - t0, 2), t=time.time())
        st["train"] = info
        save_state(job, st)
    log(f"[{job.name}] {len(tm)} templates from {info['picks']} picks ({n_accepted} accepted, {n_junk} junk left out) "
        f"on {len(xy)} micrographs "
        f"({counts.tolist()}), threshold {th if th is None else round(th, 3)}, {info['seconds']} s")
    return info


def suggest(project: Project, job, stems=None, after: str = "", n: int | None = None, log=print) -> list[str]:
    """Pick micrographs with the templates: the given stems, else the next n (settings.suggest) not reviewed and
    without suggestions of the current templates, after `after` in catalog order. Candidates (all, with scores) go
    to data/NNN/Suggest/<stem>.npz; the micrographs become "suggested" (reviewed ones stay reviewed)."""
    from .pipeline import optics_of
    job = project.job(job)
    _editable(job)
    s = sized(job)
    tm, ta = templates_of(job)
    if tm is None:
        raise ValueError("no templates yet (or the particle size changed): Train first")
    mc = project.job(job.params["micrographs"])
    optics = optics_of(project, mc)
    mi = micrographs(project, job)
    with lock(job):
        st = load_state(job)
    ver = int(st.get("version", 0))
    th0 = st.get("threshold") if st.get("threshold") is not None else 0.0
    prange = (st.get("power_lo"), st.get("power_hi", MAX_STD))
    if stems:
        rows = mi[mi["stem"].isin(list(stems))]
    else:
        order = list(mi["stem"])
        start = order.index(after) + 1 if after in order else 0
        pick = []
        for stem in order[start:] + order[:start]:
            r = mi[mi["stem"] == stem].iloc[0]
            m = st["mics"].get(str(int(r.movie_id)), {})
            if m.get("state") == "reviewed" or suggestion_version(project, job, r) == ver:
                continue
            pick.append(stem)
            if len(pick) >= (n or s.suggest):
                break
        rows = mi[mi["stem"].isin(pick)]
    done = []

    def run(dev):
        done.clear()
        for r in rows.itertuples():
            img, apix, ctf = load_mic(project, mc, r, dev)
            c, ok, _ = _candidates(img, apix, ctf, optics, s, tm, ta, batch=32)
            zm = zone_maps(img, apix, s)
            floor = zone_floor(zm, c.xy_a[auto_keep(c, ok, th0, st.get("off") or [], prange)])
            del img
            path = job.per_movie("Suggest", project.movie(r.movie_id), ".npz")
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, xy_a=c.xy_a.astype(np.float32), score=c.score.astype(np.float32), ok=ok,
                         template=c.template.astype(np.int16), version=np.int32(ver),
                         power=c.local_std.astype(np.float32), zone_dark=zm["dark_z"].astype(np.float16),
                         zone_mean=zm["mean_z"].astype(np.float16), zone_cell=np.float32(zm["cell_a"]),
                         zone_floor=np.float32(floor))
            os.replace(tmp, path)
            done.append(r)
    t0 = time.time()
    _gpu_or_cpu(run)
    with lock(job):
        st = load_state(job)
        for r in done:
            m = st["mics"].setdefault(str(int(r.movie_id)), dict(picks=[], rejected=[], state=""))
            if m.get("state") != "reviewed":
                m["state"] = "suggested"
        save_state(job, st)
    log(f"[{job.name}] suggestions on {len(done)} micrographs in {time.time() - t0:.1f} s")
    return [r.stem for r in done]


def _suggest_path(project: Project, job: Job, row) -> str:
    return os.path.join(job.data_path, "Suggest", row.subdir or "", row.stem + ".npz")


def suggestion_version(project: Project, job: Job, row) -> int:
    """Template version of a micrograph's suggestions (0: none)."""
    p = _suggest_path(project, job, row)
    if not os.path.exists(p):
        return 0
    try:
        with np.load(p) as z:
            return int(z["version"])
    except (OSError, ValueError, KeyError):
        return 0


def suggestions(project: Project, job: Job, row) -> dict | None:
    p = _suggest_path(project, job, row)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        out = dict(xy=np.round(z["xy_a"], 1).tolist(), score=np.round(z["score"], 4).tolist(),
                   ok=z["ok"].astype(int).tolist(), version=int(z["version"]),
                   template=z["template"].astype(int).tolist() if "template" in z else None,
                   power=np.round(z["power"], 3).tolist() if "power" in z else None, zones=None)
        if "zone_dark" in z:                      # files of older versions: no zones, the power cut was in "ok"
            out["zones"] = dict(cell_a=float(z["zone_cell"]), shape=list(z["zone_dark"].shape),
                                floor=round(float(z["zone_floor"]), 2),
                                dark_z=np.round(z["zone_dark"].astype(np.float32), 2).ravel().tolist(),
                                mean_z=np.round(z["zone_mean"].astype(np.float32), 2).ravel().tolist())
        return out


def auto_keep(c, ok, lo: float, off, power=(None, None), zone=None) -> np.ndarray:
    """Template candidates picked: past the filters, score >= lo, power (local_std) within the range, not of a
    template switched off, not in an excluded zone (zone: bool per candidate)."""
    keep = ok & (c.score >= lo)
    if power[0] is not None:
        keep &= c.local_std >= power[0]
    if power[1] is not None:
        keep &= c.local_std <= power[1]
    if off:
        keep &= ~np.isin(c.template, np.asarray(off, int))
    if zone is not None:
        keep &= ~zone
    return keep


# ---------------------------------------------------------------------------------------- the job: Pick all
@themed
def templates_figure(path: str, tm: np.ndarray, counts, apix: float, title: str = ""):
    """The trained templates (white protein) with how many picks each was averaged from."""
    import matplotlib.pyplot as plt
    from .plotstyle import GRAY, T
    n = len(tm)
    fig, axs = plt.subplots(1, n, figsize=(1.6 * n + 0.4, 2.2), squeeze=False)
    for a, t, c in zip(axs[0], tm, counts):
        lo, hi = np.percentile(t, [1, 99.5])
        a.imshow(t, cmap=GRAY, vmin=lo, vmax=hi)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(T.line)
        a.set_title(f"{c} picks", fontsize=9, color=T.ink2, fontweight="normal")
    fig.suptitle(title or f"Templates ({apix:.2f} Å/px)", fontsize=10.5, color=T.ink, x=0.02, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@logged
def run_manualpick(project: Project, job, log=print) -> dict:
    """Pick every micrograph: the reviewed ones keep the user's picks as they are, the others are picked with the
    trained templates at the chosen threshold (no templates: the reviewed micrographs only). Output as a Pick job's
    (picks table, catalog rows), made anew at every run."""
    from concurrent.futures import ThreadPoolExecutor
    from .pipeline import optics_of
    job = project.job(job)
    s = sized(job)
    mc = project.job(job.params["micrographs"])
    optics = optics_of(project, mc)
    st = load_state(job)
    tm, ta = templates_of(job)
    th = st.get("threshold")
    off = st.get("off") or []
    prange = (st.get("power_lo"), st.get("power_hi", MAX_STD))
    zdark, zmean = st.get("zone_dark", ZONE_DARK), st.get("zone_mean", ZONE_MEAN)
    rev = {int(k): v for k, v in st["mics"].items() if v.get("state") == "reviewed"}
    # micrographs not reviewed with some picks or removals of the user's (suggestions not checked, or a Pick all
    # since): those, plus the template picks around them
    part = {int(k): v for k, v in st["mics"].items() if v.get("state") != "reviewed" and (v.get("picks") or
                                                                                         v.get("rejected"))}
    if tm is None and not rev:
        raise ValueError("no picks and no templates: pick particles on some micrographs (and Train) first")
    if tm is not None and th is None:
        th = pk.auto_threshold(np.asarray((st.get("train") or {}).get("pos") or [0.0]))
    shutil.rmtree(os.path.join(job.path, "picks"), ignore_errors=True)       # a new output each run
    project.forget(job)
    mi = micrographs(project, job)
    dev = _device()
    log(f"{len(mi)} micrographs: {len(rev)} reviewed (kept as picked), "
        + (f"the others with {len(tm) - len(off)} of {len(tm)} templates, score >= {th:.3f}, power "
           f"{prange[0]} - {prange[1]}, zones at contamination > {zdark} / carbon > {zmean}" if tm is not None else "the others left empty"))
    t0, n_manual, n_auto = time.time(), 0, 0
    rows = list(mi.itertuples())
    need = rows if tm is not None else []                         # without templates no micrograph is read

    def read(r):
        return load_mic(project, mc, r, "cpu")
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(read, need[0]) if need else None
        k = 0
        for i, r in enumerate(rows):
            mid = int(r.movie_id)
            m = rev.get(mid) or (part.get(mid) if tm is None else None)
            if tm is not None:
                img, apix, ctf = fut.result()
                k += 1
                fut = ex.submit(read, need[k]) if k < len(need) else None
                gi = img.to(dev)
                c, ok, inside = _candidates(gi, apix, ctf, optics, s, tm, ta)
                zm = zone_maps(gi, apix, s)
                del img, gi
            elif m is None:                                           # nothing to pick it with
                cols = dict(xy_a=np.zeros((0, 2)), score=[], local_std=[], local_mean=[], inside=[], template=[],
                            manual=[])
                project.add_picks(job, mid, cols, np.zeros(0, bool), np.nan)
                continue
            if m is not None:                                         # the user's picks, as they are
                xy = np.asarray([p[:2] for p in m["picks"]], np.float64).reshape(-1, 2)
                if tm is not None and len(c.xy_a) and len(xy):
                    from scipy.spatial import cKDTree
                    d, j = cKDTree(c.xy_a).query(xy, k=1, distance_upper_bound=MATCH * s.diameter_a)
                    hit = np.isfinite(d)
                    jj = np.where(hit, j, 0)
                    score = np.where(hit, c.score[jj], th)
                    lstd = np.where(hit, c.local_std[jj], 1.0)
                    lmean = np.where(hit, c.local_mean[jj], 0.0)
                    tmpl = np.where(hit, c.template[jj], -1)
                else:
                    score, lstd, lmean, tmpl = (np.full(len(xy), v, np.float32) for v in (1.0, 1.0, 0.0, -1.0))
                cols = dict(xy_a=xy, score=score, local_std=lstd, local_mean=lmean, inside=np.ones(len(xy)),
                            template=tmpl, manual=np.ones(len(xy)))
                keep = np.ones(len(xy), bool)
                n_manual += len(xy)
                thr = np.nan if tm is None else th
            else:
                painted = (st["mics"].get(str(mid)) or {}).get("zones")
                floor = zone_floor(zm, c.xy_a[auto_keep(c, ok, th, off, prange)])
                zone = in_zones(c.xy_a, zm, zone_mask(zm, zdark, zmean, floor), painted)
                keep = auto_keep(c, ok, th, off, prange, zone)
                manual = np.zeros(len(c.score))
                cols = dict(xy_a=c.xy_a, score=c.score, local_std=c.local_std, local_mean=c.local_mean,
                            inside=inside, template=c.template, manual=manual)
                pm = part.get(mid)
                if pm is not None:                            # as the picker shows it: removed ones out, picks in
                    if len(c.xy_a):
                        from scipy.spatial import cKDTree
                        tree = cKDTree(c.xy_a)
                        for q in pm.get("rejected") or []:
                            for jj in tree.query_ball_point(q[:2], MATCH * s.diameter_a):
                                keep[jj] = False
                        for q in pm.get("picks") or []:
                            for jj in tree.query_ball_point(q[:2], s.min_dist_frac * s.diameter_a):
                                keep[jj] = False
                    xy = np.asarray([q[:2] for q in pm.get("picks") or []], np.float64).reshape(-1, 2)
                    if len(xy):
                        n1 = len(xy)
                        cols = {k: np.concatenate([np.asarray(v), add]) for (k, v), add in zip(cols.items(), (
                            xy, np.full(n1, th), np.ones(n1), np.zeros(n1), np.ones(n1), np.full(n1, -1), np.ones(n1)))}
                        keep = np.concatenate([keep, np.ones(n1, bool)])
                        n_manual += n1
                n_auto += int(keep.sum()) - (len(pm.get("picks") or []) if pm is not None else 0)
                thr = th
            project.add_picks(job, mid, cols, keep, thr)
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                el = time.time() - t0
                log(f"{i + 1}/{len(rows)} micrographs, {el / (i + 1):.2f} s each")
                project.write_summary(job, dict(project.summary_of(job), progress=dict(movies_done=i + 1,
                                                                                       movies_to_do=len(rows))))
    project.table(job, "picks").close()
    with lock(job):                                                   # unchecked suggestions: Pick all picked those
        st = load_state(job)
        for v in st["mics"].values():
            if v.get("state") == "suggested":
                v["state"] = ""
        st["mics"] = {k: v for k, v in st["mics"].items() if v.get("state") or v.get("picks") or v.get("rejected")}
        save_state(job, st)
    t = project.pick_table(job)
    summ = dict(micrographs=len(t), picks=int(t["n_kept"].sum()), candidates=int(t["n_candidates"].sum()),
                picks_per_micrograph=round(float(t["n_kept"].mean()), 1) if len(t) else 0, reviewed=len(rev),
                manual_picks=n_manual, template_picks=n_auto, templates=0 if tm is None else len(tm),
                threshold=None if tm is None else round(float(th), 4),
                power_range=list(prange), zone_dark=zdark, zone_mean=zmean,
                templates_off=len(off), seconds=round(time.time() - t0, 1),
                t_run=time.time())
    try:                                                              # a figure never fails a job
        from .plots_preprocess import overview_figure
        overview_figure(job.out("overview.png"), [("Picks per micrograph", "picks", t["n_kept"].to_numpy())],
                        "ManualPick: all micrographs", f"{job.name} · {len(t):,} micrographs · {len(rev)} reviewed")
        if tm is not None:
            templates_figure(job.out("templates.png"), tm, (st.get("train") or {}).get("counts") or [0] * len(tm), ta)
    except Exception as e:                                            # noqa: BLE001
        log(f"figures: {e!r}")
    project.set_status(job, "finished", summ)
    log(f"{summ['picks']:,} picks ({n_manual:,} yours, {n_auto:,} from the templates) in {summ['seconds']} s")
    return summ


def queue_run(project: Project, job) -> Job:
    """Pick all through the queue: the job becomes queued (scheduler: one GPU job at a time, in its own process)."""
    job = project.job(job)
    _editable(job)
    st = load_state(job)
    if not st.get("version") and not any(v.get("state") == "reviewed" for v in st["mics"].values()):
        raise ValueError("no picks yet: pick particles on some micrographs first")
    project.set_status(job, "queued", {k: v for k, v in project.summary_of(job).items()
                                       if k not in ("error", "waiting")})
    return job

