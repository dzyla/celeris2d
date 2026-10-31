"""3D polishing (a Polish job): per-particle motion tracks and per-frame weights from the movies, against a
refined map.

For every particle of a Refine3D job, the frames of its movie are cut at the particle's position in each raw
frame (refined centre minus the frame's motion-correction shift there: whole-frame shift + local motion model).
Each particle then gets a smooth track d(t) = c0 + c1 (t - 1/2) + c2 ((t - 1/2)^2 - 1/12), fitted by maximising the
Gaussian log-likelihood of its frames given the CTF-modulated projection of its own half map (gold standard)
under a Gaussian prior on c0, c1, c2. The frames, moved back along the track, are summed with per-frame,
per-frequency weights a_i(k) / sigma2_i(k): a_i(k) = s_i exp(-B_i k^2 / 4) is the signal of frame i relative to
the reference (fitted on the particles, as RELION's per-frame B-factors) and sigma2_i(k) its noise power. The sums
are the polished particles: a particle job like an extraction (same uids), centred at the refined centre plus the
track's mean, so a Refine3D on them starts from the poses of the refinement with zero shifts.

    j = polish_job(project, refine, PolishSettings()); run_polish(project, j)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import gpu
from .clock import since
from . import picking as pk
from . import preprocess as pp
from .engine import ctf_2d, freqs
from .jobspec import params_of, settings_of
from .project import Job, PARTICLE_JOBS, Project, logged


@dataclass
class PolishSettings:
    max_res_a: float = 0.0          # finest frequency used for tracks and weights (0: the refinement's resolution)
    sigma_c0_a: float = 1.0         # prior on the track: offset of the mean position from the refined centre (A)
    sigma_c1_a: float = 2.0         # linear drift over the whole movie (A)
    sigma_c2_a: float = 4.0         # curvature coefficient (A): ~0.17 x this at the first and last frames
    stats_movies: int = 100         # movies used to fit the per-frame weights (random)
    batch: int = 24                 # particles per GPU batch
    dtype: str = "float16"         # precision of the polished particle stacks
    seed: int = 0                   # random seed (movies used for the weights)
    min_resolution_a: float = 4.0   # refused unless the refinement reaches this (A): per-particle tracks need
                                    # high-resolution signal; first improve the map (classes, symmetry); 0: no check


def polish_job(project: Project, refine, s: PolishSettings, label: str = "") -> Job:
    rj = project.job(refine)
    res = project.summary_of(rj).get("resolution_solvent_mask_a") or project.summary_of(rj).get("resolution_a")
    if s.min_resolution_a and (res is None or res > s.min_resolution_a):
        raise ValueError(f"{rj.name} is at {res} A, worse than {s.min_resolution_a} A: polishing needs a "
                         f"high-resolution map (improve it first - classes, symmetry - or set min_resolution_a 0)")
    ex = project.job(rj.params["particles"])
    if ex.type not in PARTICLE_JOBS:
        raise ValueError(f"{rj.name}: its particles ({ex.name}) are not an extraction")
    e = ex.params
    keep = {k: e[k] for k in ("picks", "micrographs", "box", "out_apix", "box_a_used", "particle_a", "dark")}
    return project.new_job("Polish", dict(params_of(s), refine=rj.id, particles=ex.id, **keep), [rj, ex],
                           label or f"{e['box_a_used']:.0f}A_{e['out_apix']:.3g}apx_on{rj.id:03d}")


# --------------------------------------------------------------------------------------------- helpers
class _BandVolume:
    """A Volume built on a small box (Nyquist just past the band used) whose projections are zero-padded to the
    frame box bi: the same spectra (norm='forward': cropping / padding is exact resampling), at (b/bi)^3 of the
    memory. Two pad-2 volumes at bi = 360 held 6 GB and peaked at 18.7 GB while being built (GPU overflow)."""

    def __init__(self, vol, bi: int):
        self.vol, self.bi = vol, bi

    def project(self, A: torch.Tensor) -> torch.Tensor:
        P = self.vol.project(A)                                       # (n, b, b//2+1)
        b, bi = self.vol.b, self.bi
        if b == bi:
            return P
        P = P * (bi / b)                    # a projection sums along the ray: its amplitude scales with the box
        out = torch.zeros((len(P), bi, bi // 2 + 1), dtype=P.dtype, device=P.device)
        h = b // 2
        out[:, :h, :b // 2 + 1] = P[:, :h]
        out[:, bi - h:, :b // 2 + 1] = P[:, b - h:]
        return out


def _half_maps(project: Project, rj: Job, bi: int, apix: float, max_res: float, dev) -> list:
    """The refinement's two half maps, masked, FSC-weighted and low-passed to max_res, for projection at box bi and
    pixel size apix (built on the smallest box whose Nyquist is a little beyond max_res: _BandVolume)."""
    import mrcfile
    from .abinitio import Volume
    from .jobs import map_to_box
    bi_full, apix_full = bi, apix
    bi = min(bi_full, int(math.ceil(2 * bi_full * apix_full / max_res / 2)) * 2 + 8)
    apix = bi_full * apix_full / bi                                   # same box in A, coarser pixels
    out = []
    summ = project.summary_of(rj)
    with mrcfile.open(rj.find("mask.mrc"), permissive=True) as f:
        mask = np.asarray(f.data, np.float32)
    fsc = np.asarray(summ["fsc_solvent"], np.float64)
    box_a_r = summ["box"] * summ["apix"]
    for h in (1, 2):
        with mrcfile.open(rj.find(f"half{h}_unfil.mrc"), permissive=True) as f:
            v, ap = np.asarray(f.data, np.float32), float(f.voxel_size.x)
        v = map_to_box(v * mask, ap, bi, apix).to(dev)
        k = torch.fft.fftfreq(bi, device=dev) * bi
        kr = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2) / (bi * apix)  # 1/A
        kk = (kr * box_a_r).cpu().numpy()                                          # shell of the refinement box
        f_ = np.interp(kk, np.arange(len(fsc)), np.clip(fsc, 0, 1), right=0.0)
        w = torch.as_tensor(np.sqrt(2 * f_ / (1 + f_)), device=dev, dtype=torch.float32)
        w = w * (kr <= 1 / max_res)
        v = torch.fft.ifftn(torch.fft.fftn(v) * w).real
        vol = Volume(bi, dev)
        vol.set_real(v)
        out.append(_BandVolume(vol, bi_full))
    return out


def _cut(R: torch.Tensor, pos_px: torch.Tensor, bi: int) -> torch.Tensor:
    """Boxes (B, n, bi, bi) of frames R (n, ny, nx) centred at pos_px (B, n, 2) (x, y in pixels): integer cut
    (edges clamped) and the sub-pixel rest as a phase ramp. Returns their spectra (B, n, bi, bi//2+1)."""
    n, ny, nx = R.shape
    B = pos_px.shape[0]
    dev = R.device
    h = bi // 2
    i0 = torch.round(pos_px).long()
    r = torch.arange(bi, device=dev) - h
    out = torch.empty((B, n, bi, bi), device=dev)
    for i in range(n):
        ys = (i0[:, i, 1, None] + r[None]).clamp(0, ny - 1)                       # (B, bi)
        xs = (i0[:, i, 0, None] + r[None]).clamp(0, nx - 1)
        out[:, i] = R[i][ys[:, :, None], xs[:, None, :]]
    ky, kx = pp._grid(bi, bi, 1.0, dev)
    d = pos_px - i0.float()                                                         # (B, n, 2)
    X = torch.fft.rfft2(out, norm="forward")
    X *= torch.exp(2j * math.pi * (ky * d[..., 1, None, None] + kx * d[..., 0, None, None]))
    return X


def _basis(n: int, dev) -> torch.Tensor:
    t = torch.linspace(0, 1, n, device=dev) - 0.5
    return torch.stack([torch.ones_like(t), t, t * t - 1.0 / 12], 1)               # (n, 3)


# --------------------------------------------------------------------------------------------- the job
def _particles(project: Project, job: Job):
    from .abinitio import euler_to_matrix
    from .export import job_source
    rj, ex = project.job(job.params["refine"]), project.job(job.params["particles"])
    r = project.table(rj, "particles").read()
    e = project.table(ex, "particles").read(["uid", "x_a", "y_a"])
    o = np.argsort(e["uid"])
    pos = np.searchsorted(e["uid"][o], r["uid"]).clip(0, len(o) - 1)
    ok = e["uid"][o[pos]] == r["uid"]
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} particles of {rj.name} are not in {ex.name}")
    j = o[pos]
    src = job_source(ex.path)
    su = src.parts["celUid"].to_numpy(np.int64)
    so = np.argsort(su)
    ctf = src.ctf[so[np.searchsorted(su[so], r["uid"])]]
    A = euler_to_matrix(torch.as_tensor(r["rot"]), torch.as_tensor(r["tilt"]), torch.as_tensor(r["psi"])).float()
    centre = np.stack([e["x_a"][j] + r["shift_x_a"], e["y_a"][j] + r["shift_y_a"]], 1)
    pick = np.stack([e["x_a"][j], e["y_a"][j]], 1)
    ex_t = project.table(ex, "particles")
    if "dx_a" in ex_t.columns:                                     # the pick position, kept through re-centring
        d = ex_t.read(["dx_a", "dy_a"])
        pick = pick - np.stack([d["dx_a"][j], d["dy_a"][j]], 1)
    if "half" not in r:
        raise ValueError(f"{rj.name} has no half sets (made by an older version: run the refinement again)")
    half = r["half"]
    return dict(uid=r["uid"], movie=e["movie"][j].astype(np.int64), centre=centre, pick=pick, A=A, half=half,
                ctf=torch.as_tensor(np.asarray(ctf, np.float32)), optics=src.optics,
                per_particle_ctf="defocus_u" in ex_t.columns)


def _movie_frames(project: Project, mc: Job, movie_id: int, frames=None):
    from .pipeline import _motion
    mv = project.movie(movie_id)
    s, fr, mic, _, _ = _motion(project, mc, mv, frames, log=lambda *a: None, keep_frames=True, ctf=False)
    return mic


def _frame_positions(mic, centre_a: np.ndarray, dev) -> torch.Tensor:
    """Position (B, n, 2) in pixels, in every raw frame, of what the aligned sum shows at centre_a (B, 2) (A)."""
    F = mic.frames_ft
    n, ny, nh = F.shape
    nx = 2 * (nh - 1)
    c = torch.as_tensor(centre_a, device=dev, dtype=torch.float32)
    s = torch.as_tensor(mic.shifts, device=dev, dtype=torch.float32)[None].expand(len(c), n, 2)
    if mic.model is not None:
        xy = torch.stack([c[:, 0] / mic.apix / nx * 2 - 1, c[:, 1] / mic.apix / ny * 2 - 1], 1)
        s = s + mic.model(xy, torch.linspace(0, 1, n, device=dev))
    return (c[:, None] - s) / mic.apix


@logged
def run_polish(project: Project, job, log=print) -> dict:
    from multiprocessing import shared_memory
    from .pipeline import _decode_pool, _decode_shm
    job = project.job(job)
    p = job.params
    s = PolishSettings(**{k: v for k, v in p.items() if k in PolishSettings.__dataclass_fields__})
    rj, mc = project.job(p["refine"]), project.job(p["micrographs"])
    dev = torch.device(gpu.default())
    t_start = time.time()
    P = _particles(project, job)
    movies = np.unique(P["movie"])
    mic_apix = mc.params["apix"] * mc.params["bin"]
    out_apix, box = p["out_apix"], p["box"]
    bi = pk.source_box(box, mic_apix, out_apix)                                     # box at the frame pixel size
    out_apix = bi * mic_apix / box              # exact (extractions by older versions stored the requested one)
    box_a = bi * mic_apix
    max_res = s.max_res_a or max(2.5 * mic_apix, float(project.summary_of(rj)["resolution_a"]))
    vols = _half_maps(project, rj, bi, mic_apix, max_res, dev)
    fy, fx = freqs(bi, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)                           # cycles / box
    shell = torch.round(kr).long().clamp(max=bi // 2)
    NS = bi // 2 + 1
    band = (kr >= 1) & (kr <= box_a / max_res)
    herm = torch.where(fx[None, :].expand_as(kr) > 0, 2.0, 1.0)                    # rfft half plane: count twice
    sgn = -1.0 if p.get("dark", True) else 1.0
    log(f"[polish] {len(P['uid']):,} particles on {len(movies)} micrographs; frames cut at {bi} px "
        f"({mic_apix:.3f} A/px), tracks and weights up to {max_res:.2f} A")

    pool = _decode_pool(2)

    def frames_of(ids):
        """Decoded movies in order, two decoding ahead."""
        futs = {}
        for n_, m in enumerate(ids):
            for k in ids[n_:n_ + 2]:
                if k not in futs:
                    futs[k] = pool.submit(_decode_shm, project.movie(int(k))["path"])
            name, shape, dt = futs.pop(m).result()
            shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray(shape, np.dtype(dt), buffer=shm.buf).copy()   # no view may outlive the segment
            shm.close()
            shm.unlink()
            yield m, arr

    def batches(mic, sel):
        """Per batch of the movie's particles: (indices, frame spectra X (B, n, bi, bh), reference C*P (B, bi, bh))."""
        R = torch.fft.irfft2(mic.frames_ft, norm="forward")                        # (n, ny, nx) real frames
        for i in range(0, len(sel), s.batch):
            j = sel[i:i + s.batch]
            pos = _frame_positions(mic, P["centre"][j], dev)
            X = _cut(R, pos, bi) * sgn
            A = P["A"][j].to(dev)
            hv = torch.as_tensor(P["half"][j], device=dev)
            ref = torch.zeros((len(j), bi, bi // 2 + 1), dtype=torch.complex64, device=dev)
            for h in (0, 1):
                k = torch.nonzero(hv == h).squeeze(1)
                if len(k):
                    ref[k] = vols[h].project(A[k])
            ref *= ctf_2d(P["ctf"][j].to(dev), P["optics"], fy, fx, box_a)
            yield j, X, ref
        del R

    # ---- pass 1: per-frame signal a_i(k) and noise sigma2_i(k), on a random subset of movies
    rng = np.random.default_rng(s.seed)
    stat_ids = np.sort(rng.choice(movies, min(len(movies), s.stats_movies), replace=False))
    num = den = pw = None
    cnt = torch.zeros(NS, device=dev).index_add_(0, shell.reshape(-1), herm.reshape(-1))
    t0 = time.time()
    n_fr = None
    for m, fr in frames_of(list(stat_ids)):
        mic = _movie_frames(project, mc, int(m), fr)
        if mic.frames_ft is None or mic.frames_ft.shape[0] < 2:
            raise ValueError(f"Polish needs movies: {project.movie(int(m))['stem']} is a single frame (a micrograph "
                             f"aligned and summed elsewhere, e.g. CryoPPP / EMPIAR micrograph sets), so there are no "
                             f"frames to track particles in")
        n_fr = mic.frames_ft.shape[0]
        if num is None:
            num = torch.zeros((n_fr, NS), device=dev, dtype=torch.float64)
            pw = torch.zeros((n_fr, NS), device=dev, dtype=torch.float64)
            den = torch.zeros(NS, device=dev, dtype=torch.float64)
        for j, X, ref in batches(mic, np.flatnonzero(P["movie"] == m)):
            num.index_add_(1, shell.reshape(-1), ((X * ref.conj()[:, None]).real * herm).sum(0).reshape(n_fr, -1).double())
            pw.index_add_(1, shell.reshape(-1), ((X.abs() ** 2) * herm).sum(0).reshape(n_fr, -1).double())
            den.index_add_(0, shell.reshape(-1), ((ref.abs() ** 2) * herm).sum(0).reshape(-1).double())
        del mic
    a_raw = (num / den.clamp_min(1e-30)).float()                                     # (n, NS)
    nparts = float(np.isin(P["movie"], stat_ids).sum())
    sig2 = (pw / (nparts * cnt.double().clamp_min(1))).float()                       # noise power per coefficient
    # parametric fit a_i(k) = s_i exp(-B_i k^2 / 4) on shells from 1/30 A to max_res, weighted by den
    k_a = torch.arange(NS, device=dev, dtype=torch.float32) / box_a                  # 1/A
    fit_sh = (k_a >= 1 / 30.0) & (k_a <= 1 / max_res)
    Bs = torch.arange(-100.0, 2000.0, 5.0, device=dev)
    E = torch.exp(-Bs[:, None] * (k_a[fit_sh] ** 2)[None] / 4)                       # (nB, K)
    wgt = den[fit_sh].float()
    y = a_raw[:, fit_sh]                                                             # (n, K)
    sc = (y[:, None] * E[None] * wgt).sum(-1) / ((E ** 2) * wgt).sum(-1)[None]       # (n, nB) best scale per B
    err = ((y[:, None] - sc[..., None] * E[None]) ** 2 * wgt).sum(-1)
    bi_best = err.argmin(1)
    B_fr = Bs[bi_best]
    s_fr = sc[torch.arange(n_fr, device=dev), bi_best]
    B_rel = B_fr - B_fr.min()
    a_mod = (s_fr[:, None] * torch.exp(-B_fr[:, None] * k_a[None] ** 2 / 4)).clamp_min(0)      # (n, NS)
    w = a_mod / sig2.clamp_min(1e-30)                                                # matched filter per frame
    # noise power of the weighted sum = that of a plain sum, at every frequency
    w = w * torch.sqrt((sig2.sum(0) / ((w ** 2) * sig2).sum(0).clamp_min(1e-30)))[None]
    log(f"[polish] frame weights from {len(stat_ids)} micrographs ({nparts:,.0f} particles, {since(t0):.0f}s): "
        f"relative B {', '.join(f'{float(b):.0f}' for b in B_rel[::max(1, n_fr // 8)])} ... A^2, "
        f"scale {float(s_fr[0] / s_fr.max()):.2f} (first) {float(s_fr[n_fr // 2] / s_fr.max()):.2f} (middle) "
        f"{float(s_fr[-1] / s_fr.max()):.2f} (last) of the best frame")

    # ---- pass 2: tracks and polished sums
    basis = _basis(n_fr, dev)                                                        # (n, 3)
    prior = torch.tensor([s.sigma_c0_a, s.sigma_c1_a, s.sigma_c2_a], device=dev) ** -2
    ky_b, kx_b = fy[:, None].expand_as(kr)[band] / box_a, fx[None, :].expand_as(kr)[band] / box_a    # 1/A
    wl = (a_mod / sig2.clamp_min(1e-30))[:, shell[band]] * herm[band][None]          # (n, F) likelihood weights
    lo = kr[band] <= box_a / max(10.0, 2 * max_res)
    ky_f, kx_f = (fy[:, None].expand_as(kr) / box_a), (fx[None, :].expand_as(kr) / box_a)
    wsum = w[:, shell]                                                               # (n, bi, bh)
    yy = (torch.arange(box, device=dev) - box // 2).float()
    rr = torch.sqrt(yy[:, None] ** 2 + yy[None, :] ** 2)
    bg = rr > min(0.55 * p["particle_a"] / out_apix, 0.9 * box / 2)
    done = 0
    t0 = time.time()
    for m, fr in frames_of(list(movies)):
        if project.done(job, int(m)):
            continue
        mic = _movie_frames(project, mc, int(m), fr)
        sel = np.flatnonzero(P["movie"] == m)
        imgs, new_c, trk, mvd, coef = [], [], [], [], []
        for j, X, ref in batches(mic, sel):
            G = (X[..., band] * ref[:, None][..., band].conj()) * wl[None]           # (B, n, F)
            c = torch.zeros((len(j), 3, 2), device=dev, requires_grad=True)
            opt = torch.optim.Adam([c], lr=0.15)
            for it in range(120):
                use = lo if it < 50 else slice(None)
                d = torch.einsum("nk,bkx->bnx", basis, c)                           # (B, n, 2) A
                ph = 2 * math.pi * (d[..., 0, None] * kx_b[use] + d[..., 1, None] * ky_b[use])
                ll = (G[..., use].real * torch.cos(ph) - G[..., use].imag * torch.sin(ph)).sum((1, 2))
                loss = -(ll - 0.5 * (c ** 2 * prior[None, :, None]).sum((1, 2))).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                d = torch.einsum("nk,bkx->bnx", basis, c)
                trk.append((d - d.mean(1, keepdim=True)).pow(2).sum(-1).mean(1).sqrt().cpu().numpy())
                mvd.append(d.mean(1).norm(dim=-1).cpu().numpy())
                coef.append(c.detach().reshape(len(j), 6).cpu().numpy())
                # frames moved back along the track (by -d), weighted, summed
                S = (X * torch.exp(2j * math.pi * (d[..., 0, None, None] * kx_f + d[..., 1, None, None] * ky_f))
                     * wsum[None]).sum(1)
                if bi != box:
                    S = pp.crop_half(S, box, box)
                im = torch.fft.irfft2(S, s=(box, box), norm="forward")
                b = im[:, bg]
                im = (im - b.mean(1)[:, None, None]) / b.std(1).clamp_min(1e-6)[:, None, None]
                imgs.append(im.cpu().numpy())
                new_c.append(P["centre"][j] + d.mean(1).cpu().numpy())
        if len(sel):
            imgs, new_c = np.concatenate(imgs), np.concatenate(new_c)
            trk, mvd = np.concatenate(trk).astype(np.float32), np.concatenate(mvd).astype(np.float32)
            coef = np.concatenate(coef).astype(np.float32)
        else:
            imgs, new_c = np.zeros((0, box, box), np.float32), np.zeros((0, 2))
            trk, mvd = np.zeros(0, np.float32), np.zeros(0, np.float32)
            coef = np.zeros((0, 6), np.float32)
        cols = dict(score=np.zeros(len(sel), np.float32), track_rms_a=trk, track_mean_a=mvd,
                    half=P["half"][sel].astype(np.uint8),
                    **{f"track_c{k}{ax}": coef[:, 2 * k + i] for k in range(3) for i, ax in enumerate("xy")})
        if P["per_particle_ctf"]:                    # refined defocus (CtfRefine) goes on with the polished particles
            c_ = P["ctf"][sel].numpy()
            cols.update(defocus_u=c_[:, 0], defocus_v=c_[:, 1], defocus_angle=c_[:, 2])
        project.add_particles(job, int(m), new_c, imgs, out_apix, dtype=s.dtype, uid=P["uid"][sel],
                              pick_xy_a=P["pick"][sel], columns=cols)
        done += 1
        del mic
        if done % 10 == 0 or done == len(movies):
            log(f"[polish] {done}/{len(movies)} micrographs, {(time.time() - t0) / done:.1f} s each")
    pool.shutdown(cancel_futures=True)
    project.consolidate(job)
    tt = project.table(job, "particles").read(["track_rms_a", "track_mean_a"])
    tr, mv_ = tt["track_rms_a"], tt["track_mean_a"]
    np.savez(job.out("frame_weights.npz"), B=B_fr.cpu().numpy(), scale=s_fr.cpu().numpy(),
             a_raw=a_raw.cpu().numpy(), a_model=a_mod.cpu().numpy(), sigma2=sig2.cpu().numpy(),
             weights=w.cpu().numpy(), apix=mic_apix, box=bi)
    # SNR of the weighted sum relative to the plain sum, per shell: (sum a^2 / s2) / ((sum a)^2 / sum s2)
    am, sg = a_mod.double(), sig2.double().clamp_min(1e-30)
    gain = ((am ** 2 / sg).sum(0) / ((am.sum(0) ** 2) / sg.sum(0)).clamp_min(1e-30)).float().cpu().numpy()
    per_mic = {}
    for mid, v in zip(np.asarray(tt["movie"]), np.asarray(tr)):
        per_mic.setdefault(int(mid), []).append(float(v))
    mic_rms = np.array([np.median(per_mic[k]) for k in sorted(per_mic)]) if per_mic else np.zeros(0)
    title_ = f"{job.name} · {len(P['uid']):,} particles · {len(movies)} micrographs · {n_fr} frames"
    _figure(job.out("polish.png"), B_rel.cpu().numpy(), (s_fr / s_fr.max()).cpu().numpy(), w.cpu().numpy(), box_a,
            tr, title_, a_raw=a_raw.cpu().numpy(), a_mod=a_mod.cpu().numpy(), gain=gain, moved=mv_,
            mic_rms=mic_rms, max_res=max_res)
    tracks_figure(project, job, n_fr, title_)
    k_gain = {f"{r_:g}": round(float(gain[int(round(box_a / r_))]), 3) for r_ in (20, 8, 5, 4, 3)
              if int(round(box_a / r_)) < len(gain) and r_ >= max_res}
    summ = dict(particles=int(len(P["uid"])), micrographs=int(len(movies)), box=box, apix=out_apix,
                box_a=round(box * out_apix, 1), max_res_a=round(max_res, 2),
                frame_B_rel=[round(float(b), 1) for b in B_rel], frame_scale=[round(float(x), 3) for x in s_fr / s_fr.max()],
                track_rms_a_median=round(float(np.median(tr)), 3) if len(tr) else None,
                centre_moved_a_median=round(float(np.median(mv_)), 3) if len(mv_) else None,
                snr_gain_by_res_a=k_gain,
                micrograph_track_rms_a=dict(median=round(float(np.median(mic_rms)), 3),
                                            p95=round(float(np.percentile(mic_rms, 95)), 3)) if len(mic_rms) else None,
                seconds=round(time.time() - t_start, 1), dtype=s.dtype)
    project.set_status(job, "finished", summ)
    return summ


def _figure(path, B_rel, scale, w, box_a, track_rms, title_, **extra):
    from .plotstyle import themed
    themed(_draw)(path, B_rel, scale, w, box_a, track_rms, title_, **extra)


def _draw(path, B_rel, scale, w, box_a, track_rms, title_, a_raw=None, a_mod=None, gain=None, moved=None,
          mic_rms=None, max_res=None):
    """Top: per-frame B-factor and scale, the frame weights over frame x resolution, and the SNR gain of the weighted
    sum over the plain sum per resolution. Bottom: the measured per-frame signal at a few resolutions with the fitted
    model, the particle tracks (rms about their mean position), how far particle centres moved, and the median track
    per micrograph (a micrograph far above the others: drift the patch model missed, or a bad movie)."""
    from .plotstyle import T, grid_y, note, plt, title, top
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.6))
    fig.subplots_adjust(left=0.05, right=0.98, top=top(fig, 0.8), bottom=0.08, wspace=0.34, hspace=0.5)
    n = len(B_rel)
    fr = np.arange(1, n + 1)
    a = ax[0, 0]
    a.plot(fr, B_rel, "o-", ms=3, color=T.series[0])
    a.set(title="Relative B-factor per frame", xlabel="frame", ylabel="Å²")
    a = ax[0, 1]
    a.plot(fr, scale, "o-", ms=3, color=T.series[0])
    a.set(title="Scale per frame", xlabel="frame", ylabel="relative")
    a = ax[0, 2]
    nk = w.shape[1]
    k_hi = min(nk - 1, int(box_a / max_res) if max_res else nk - 1)
    W = w[:, 1:k_hi + 1] / np.maximum(w[:, 1:k_hi + 1].max(0, keepdims=True), 1e-30)
    im = a.imshow(W.T, origin="lower", aspect="auto", cmap=T.density, extent=(0.5, n + 0.5, 1, k_hi),
                  vmin=0, vmax=1)
    ticks = [r for r in (30, 15, 8, 5, 4, 3, 2.5) if 1 <= box_a / r <= k_hi]
    a.set_yticks([box_a / r for r in ticks], [f"{r:g}" for r in ticks])
    a.set(title="Frame weights (each resolution scaled to 1)", xlabel="frame", ylabel="resolution (Å)")
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
    a = ax[0, 3]
    if gain is not None:
        k = np.arange(1, k_hi + 1)
        a.plot(k, gain[1:k_hi + 1], color=T.series[0])
        a.axhline(1, color=T.ink3, lw=1, ls="--")
        a.set_xticks([box_a / r for r in ticks], [f"{r:g}" for r in ticks])
        i5 = int(round(box_a / 5))
        if i5 <= k_hi:
            note(a, f"x{gain[i5]:.2f} at 5 Å", "upper left")
    a.set(title="SNR gain of the weighting", xlabel="resolution (Å)", ylabel="weighted / plain sum")
    a = ax[1, 0]
    if a_raw is not None:
        for q, res in enumerate((20, 8, 5, 3)):
            i = int(round(box_a / res))
            if i <= k_hi:
                a.plot(fr, a_raw[:, i], "o", ms=3, color=T.series[q], label=f"{res} Å")
                a.plot(fr, a_mod[:, i], "-", lw=1.2, color=T.series[q])
        a.legend(loc="upper right")
    a.set(title="Signal per frame: measured (dots), model", xlabel="frame", ylabel="correlation with the map")
    a = ax[1, 1]
    a.hist(track_rms, bins=60, color=T.series[0])
    if len(track_rms):
        note(a, f"median {np.median(track_rms):.2f} Å", "upper right")
    a.set(title="Particle tracks", xlabel="rms about the mean position (Å)", ylabel="particles")
    a = ax[1, 2]
    if moved is not None and len(moved):
        a.hist(moved, bins=60, color=T.series[0])
        note(a, f"median {np.median(moved):.2f} Å", "upper right")
    a.set(title="Centre moved by the track", xlabel="Å", ylabel="particles")
    a = ax[1, 3]
    if mic_rms is not None and len(mic_rms):
        x = np.arange(len(mic_rms))
        a.plot(x, mic_rms, "o", ms=3 if len(x) < 500 else 1.5, color=T.series[0], mew=0)
        med = float(np.median(mic_rms))
        a.axhline(med, color=T.ink2, lw=1, ls="--")
        note(a, f"median {med:.2f} Å, 95 % {np.percentile(mic_rms, 95):.2f} Å", "upper right")
    a.set(title="Median track per micrograph", xlabel="micrograph", ylabel="rms (Å)")
    for a in ax.ravel():
        if a is not ax[0, 2]:
            grid_y(a)
    title(fig, "Polishing", title_)
    fig.savefig(path)
    plt.close(fig)


def tracks_figure(project: Project, job, n_frames: int, title_: str = "", n_mic: int = 2):
    """tracks.png of a Polish job from what it stores: the first n_mic micrographs (MotionCtf sums, low-passed), the
    particles' polished positions and their track coefficients (track_cKX, K = 0..2, X = x / y on _basis)."""
    from .pipeline import _load_mic
    from .plots_preprocess import display_blur
    job = project.job(job)
    mc = project.job(job.params["micrographs"])
    t = project.table(job, "particles").read()
    if "track_c0x" not in t or not len(t["movie"]):
        return None
    basis = _basis(n_frames, "cpu").numpy()                                         # (n, 3)
    examples = []
    for m in list(dict.fromkeys(np.asarray(t["movie"]).tolist()))[:n_mic]:
        sel = np.flatnonzero(t["movie"] == m)
        C = np.stack([np.stack([t[f"track_c{k}{ax}"][sel] for ax in "xy"], -1) for k in range(3)], 1)   # (p, 3, 2)
        tr = np.einsum("nk,pkx->pnx", basis, C)                                    # (p, n, 2) A
        img, apix, _ = _load_mic(project, mc, int(m))
        f_ = max(1, int(round(4.0 / apix)))                                         # ~4 A/px for the figure
        im = torch.nn.functional.avg_pool2d(img.float()[None, None], f_)[0, 0].cpu().numpy()
        xy = np.stack([t["x_a"][sel], t["y_a"][sel]], 1) if "x_a" in t else None
        if xy is None:
            continue
        examples.append(dict(stem=project.movie(int(m))["stem"], image=display_blur(im, apix * f_, 20.0),
                             apix=apix * f_, centre=xy - tr.mean(1), tracks=tr))
    if not examples:
        return None
    from .plotstyle import themed
    themed(_draw_tracks)(job.out("tracks.png"), examples, title_ or job.name)
    return job.out("tracks.png")


def _draw_tracks(path, examples, title_, mag: float = 0.0):
    """Particle tracks on the first micrographs (low-passed): every particle's path over the frames, magnified mag
    times (0: so that the median path is ~35 pixels of the figure), the start marked."""
    from matplotlib.collections import LineCollection
    from .plotstyle import GRAY, T, image_axes, plt, title, top
    if not mag:
        L = np.concatenate([np.abs(np.diff(e["tracks"], axis=1)).sum((1, 2)) for e in examples])     # path length (A)
        mag = float(np.clip(np.round(35 * examples[0]["apix"] / max(float(np.median(L)), 1e-3), -1), 10, 1000))
    k = len(examples)
    fig, ax = plt.subplots(1, k, figsize=(8.2 * k, 8.4), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=top(fig, 0.85), bottom=0.04, wspace=0.05)
    for a, e in zip(ax[0], examples):
        img, ap = e["image"], e["apix"]
        lo, hi = np.percentile(img, [0.5, 99.5])
        a.imshow(img, cmap=GRAY, vmin=lo, vmax=hi)
        c = e["centre"] / ap                                              # (n, 2) px
        tr = e["tracks"]                                                  # (n, frames, 2) A
        pts = c[:, None] + (tr - tr[:, :1]) * mag / ap
        a.add_collection(LineCollection(pts, colors=T.series[1], linewidths=1.4, alpha=0.95))
        a.plot(pts[:, 0, 0], pts[:, 0, 1], "o", ms=2.4, color=T.series[2], mew=0)
        a.set_xlim(0, img.shape[1])
        a.set_ylim(img.shape[0], 0)
        image_axes(a)
        rms = np.sqrt(((tr - tr.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))
        a.set_title(f"{e['stem']} · {len(c)} particles · median track {np.median(rms):.2f} Å", fontsize=10)
    title(fig, f"Particle tracks (x{mag:g}, start dotted)", title_)
    fig.savefig(path)
    plt.close(fig)
