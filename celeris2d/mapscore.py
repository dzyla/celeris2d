"""MapScore job: how well each particle fits a 3D map, against a control that has the particle's own power spectrum
but no structure. Every particle is phase-flipped, low-passed and compared with projections of the map over all
orientations (normalised correlation inside the particle circle, best over directions x in-plane angles x small
shifts); the same is done for a phase-randomised copy of it. Each is also compared with a decoy: the map's envelope
without its structure (phases random beyond `decoy_res_a`). A particle "fits" when its map score minus its decoy
score beats the controls' quantile `q` (a share 1 - q of structureless images passes by chance): high-contrast junk
(carbon edges, ice) correlates with any object of the right size, a particle more with the map than with the decoy.

At single-particle SNR a single score separates poorly (dz_2025_10 at 16 A: 34 % of 2D-selected particles vs 2.5 % of
the "honeycomb" classes above the 99th percentile of controls), so the job also reports, per 2D class of the
particles' Class2D job, the share that fits with its binomial interval: a class of real particles sits far above the
chance rate, a noise-attractor class at it - an objective class selection where class-average scores against map
projections are fooled by round, textured averages. Measured with the map score alone (before the decoy):
dz_2025_10 (152,932 template picks, Class2D K100 vs the D2 map at
20 A, 6 min on one RTX PRO 4000): particles in the classes selected by class-average scores fit at 28.6 %, the rest
at 2.7 %; classes such as two at 84 % and 70 % fit had been left out. J20 (198k particles, the whole stack, against
its own ab initio map): particles of clean 2D classes fit at 40 %, of noisy ones at 15 %; the class proposal keeps
97 % of the clean particles and removes 81 % of the noisy ones (3D classification at fixed poses: 80 % / 56 %).
Decisions stay the user's: the job proposes, Select decides.

At 16-20 A the score tells particle-like density of the right size from empty ice / noise, not one protein from
another (synthetic: 95 % of particles, 0 % of noise, 36 % of another globular object of the same size fit).

    ms = mapscore_job(project, particles=45, map_job=53, s=MapScoreSettings())
    run_mapscore(project, ms)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import gpu
from .clock import since
from .engine import ctf_2d, gpu_free
from .preprocess import crop_half
from .jobspec import params_of, settings_of
from .plotstyle import STATUS, T, grid_y, note, plt, themed, title, top
from .project import Job, Project, logged


@dataclass
class MapScoreSettings:
    res_a: float = 20.0             # compare particles and projections to this resolution (A): low resolution carries
                                    # the shape; finer adds noise and time
    n_dirs: int = 150               # projection directions over the whole sphere
    psi_step: float = 10.0          # in-plane angle step (deg)
    shift_a: float = 10.0           # largest shift searched (A) around the box centre
    diameter_a: float = 0.0         # circle the correlation is taken in (A); 0: the extraction's particle size
    quantile: float = 0.99          # a particle fits when it beats this quantile of the phase-randomised controls
    decoy_res_a: float = 80.0       # also score against a structureless decoy (the map's phases kept to this
                                    # resolution - its envelope - random beyond, same power); a particle fits by how
                                    # much more it correlates with the map than with the decoy (0: by the map score
                                    # alone). Carbon edges / ice blobs correlate with any envelope of the right size.
                                    # dz_2025_10 K100 (3,000 movies, 300 per class, classes by my reading of the
                                    # gallery), share that fits, protein views / edge + ice / other: map score alone
                                    # 5.9 / 21 / 0.5 % (three edge classes had topped the list at 65-81 %); map - decoy
                                    # at 40 A 35 / 0.4 / 1.6 %, 60 A 47 / 0.4 / 2.3, 80 A 47 / 0.4 / 2.1, 120 A 27 /
                                    # 0.1 / 0.7. Synthetic (test_mapscore): 40 A 6 %, 60 A 35 %, 80 A 48 %, 120 A 25 %
    propose_over_chance: float = 10.0   # propose a 2D class when the lower 95 % bound of its share that fits exceeds
                                    # this x the chance rate ...
    propose_relative: float = 0.4   # ... and this x the median share of the 10 best classes (>= 100 particles): the
                                    # share of real particles that fits depends on the SNR. J20 (198k, K50, my reading
                                    # of its gallery: 21 noisy classes), clean particles kept / noisy removed, map -
                                    # decoy: 0.4 100 / 91 %, 0.5 94 / 100 %, 0.6 85 / 100 % (map score alone: 0.4 100 /
                                    # 70, 0.5 97 / 81, 0.6 97 / 96; 3D classification at the consensus poses, 4 models,
                                    # best two kept: 80 / 56). dz_2025_10 K100 (460k): 19 classes / 100k particles, all
                                    # protein views; five more views fit at 17-26 % (bar 27 %)
    max_particles: int = 0          # score a random subset of this many (0: all); class shares need ~200 per class
    seed: int = 0                   # control phases and subset
    gpus: str = "auto"              # GPU to use: "auto" or an index


def _norm(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    x = (x - (x * m).sum((-2, -1), keepdim=True) / m.sum()) * m
    return x / x.flatten(1).norm(dim=1).clamp_min(1e-9)[:, None, None]


class Scorer:
    """Projections of one map, prepared once; score(images, ctf) for batches of particles."""

    def __init__(self, vol: np.ndarray, vol_apix: float, box: int, apix: float, optics, diameter_a: float,
                 res_a: float = 20.0, n_dirs: int = 150, psi_step: float = 10.0, shift_a: float = 10.0, dev=None):
        from .picking import _resample, _rotate2d, templates_from_map
        dev = dev or gpu.default()
        self.dev, self.optics, self.box, self.apix = dev, optics, box, apix
        wa = max(apix, res_a / 2.2)                                   # working pixel: Nyquist a little past res_a
        self.bw = max(16, int(round(box * apix / wa / 2)) * 2)
        self.wa = box * apix / self.bw
        self.box_a = box * apix
        pr, pa = templates_from_map(vol, vol_apix, n_dirs, wa=self.wa)
        pr = _resample(torch.as_tensor(pr, device=dev), pa, self.wa)
        o = (pr.shape[-1] - self.bw) // 2
        pr = pr[:, o:o + self.bw, o:o + self.bw] if o >= 0 else torch.nn.functional.pad(pr, (-o,) * 4)
        pr = _rotate2d(pr, torch.arange(0, 360, psi_step))
        b = self.bw
        self.fy = torch.fft.fftfreq(b, device=dev) * b
        self.fx = torch.fft.rfftfreq(b, device=dev) * b
        self.lp = ((self.fy[:, None] ** 2 + self.fx[None, :] ** 2).sqrt() / self.box_a <= 1 / res_a).float()
        r = (torch.arange(b, device=dev) - b // 2).float()
        self.m = ((r[:, None] ** 2 + r[None, :] ** 2).sqrt() * self.wa <= diameter_a / 2).float()
        self.Ft = torch.fft.rfft2(_norm(torch.fft.irfft2(torch.fft.rfft2(pr) * self.lp, s=(b, b)), self.m)).conj()
        self.s = max(1, int(round(shift_a / self.wa)))
        free = gpu_free(torch.device(dev))
        self.chunk = int(max(1, min(64, 0.3 * free / (len(self.Ft) * b * b * 4 * 3))))

    def prepare(self, imgs: torch.Tensor, ctf: torch.Tensor, gen: torch.Generator):
        """Particles (n, box, box) and their CTF (n, 6) -> (phase-flipped, low-passed, normalised images; the same
        with random phases), both at the working box."""
        b = self.bw
        F = torch.fft.rfft2(imgs.float().to(self.dev), norm="forward")
        F = crop_half(F, b, b) if b < self.box else F
        F = F * torch.sign(ctf_2d(ctf.float().to(self.dev), self.optics, self.fy, self.fx, self.box_a)) * self.lp
        ph = torch.rand(F.shape, generator=gen, device="cpu").to(self.dev)
        rnd = F.abs() * torch.exp(2j * math.pi * ph)
        return (_norm(torch.fft.irfft2(F, s=(b, b), norm="forward"), self.m),
                _norm(torch.fft.irfft2(rnd, s=(b, b), norm="forward"), self.m))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Best normalised correlation of each prepared image over all projections and shifts up to +-s px."""
        b, s, out = self.bw, self.s, []
        for i in range(0, len(x), self.chunk):
            c = torch.fft.irfft2(self.Ft[None] * torch.fft.rfft2(x[i:i + self.chunk])[:, None], s=(b, b))
            c = torch.roll(c, (s, s), (-2, -1))[..., :2 * s + 1, :2 * s + 1]
            out.append(c.amax((1, 2, 3)))
        return torch.cat(out)


def class_shares(cls: np.ndarray, fits: np.ndarray, chance: float) -> list[dict]:
    """Per class: particles, share that fits, its 95 % Wilson interval, and the share over the chance rate."""
    out = []
    for k in np.unique(cls):
        f = fits[cls == k]
        n, p = len(f), float(f.mean())
        z = 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        out.append(dict(cls=int(k) + 1, n=n, fit=round(p, 4), lo=round(c - h, 4), hi=round(c + h, 4),
                        over_chance=round(p / max(chance, 1e-9), 2)))
    return out


def mapscore_job(project: Project, particles, map_job, s: MapScoreSettings, label: str = "") -> Job:
    pj, mj = project.job(particles), project.job(map_job)
    return project.new_job("MapScore", dict(params_of(s), particles=pj.id, map=mj.id), [pj, mj],
                           label or f"on{pj.id:03d}_map{mj.id:03d}")


@logged
def run_mapscore(project: Project, job, log=print) -> dict:
    from .jobs import _reference_map, map_job, particle_source
    job = project.job(job)
    s = settings_of(job, MapScoreSettings)
    pj, mj = project.job(job.params["particles"]), project.job(job.params["map"])
    t0 = time.time()
    src = particle_source(project, pj)
    uid = src.parts["celUid"].to_numpy(np.int64)
    rng = np.random.default_rng(s.seed)
    if s.max_particles and s.max_particles < len(uid):
        keep = np.sort(rng.choice(len(uid), s.max_particles, replace=False))
        src.subset(keep)
        uid = uid[keep]
    vol, vapix, _ = _reference_map(map_job(project, mj))              # a Subtract job: its refinement's map
    diam = s.diameter_a or getattr(src, "mask_diameter_a", 0.0) / 1.25 or 0.6 * src.box * src.apix
    dev = gpu.device(int(s.gpus)) if s.gpus not in ("auto", "") else gpu.default()
    sc = Scorer(vol, vapix, src.box, src.apix, src.optics, diam, s.res_a, s.n_dirs, s.psi_step, s.shift_a, dev)
    log(f"[mapscore] {len(uid):,} particles of {pj.name} vs {mj.name}: {len(sc.Ft):,} projections, working box "
        f"{sc.bw} ({sc.wa:.1f} A/px), {s.res_a:g} A, circle {diam:.0f} A")
    sd = None
    if s.decoy_res_a > 0:
        from .jobs import decoy_map
        vt = torch.as_tensor(np.asarray(vol), dtype=torch.float32, device=dev)
        dec = decoy_map(vt, vt.shape[0] * vapix, s.decoy_res_a, 1.2 * diam, s.seed).cpu().numpy()
        sd = Scorer(dec, vapix, src.box, src.apix, src.optics, diam, s.res_a, s.n_dirs, s.psi_step, s.shift_a, dev)
        del vt
    gen = torch.Generator().manual_seed(s.seed)
    score, ctrl, dscore, dctrl = (np.zeros(len(uid), np.float32) for _ in range(4))
    done = 0
    for idx, im in src.chunks():
        x, r = sc.prepare(torch.as_tensor(im), torch.as_tensor(src.ctf[idx]), gen)
        score[idx] = sc.score(x).cpu().numpy()
        ctrl[idx] = sc.score(r).cpu().numpy()
        if sd is not None:
            dscore[idx] = sd.score(x).cpu().numpy()
            dctrl[idx] = sd.score(r).cpu().numpy()
        done += len(idx)
        if done % 20000 < len(idx):
            log(f"[mapscore] {done:,}/{len(uid):,} ({since(t0):.0f} s)")
    val, cval = (score - dscore, ctrl - dctrl) if sd is not None else (score, ctrl)     # what is thresholded
    thr = float(np.quantile(cval, s.quantile))
    fits = val > thr
    chance = 1 - s.quantile
    cols = dict(uid=uid, score=score, control=ctrl, fits=fits)
    if sd is not None:
        cols.update(decoy=dscore, control_decoy=dctrl)
    shares = []
    cls_job = pj if pj.type == "Class2D" else None
    if cls_job is not None:
        t = project.table(cls_job, "particles").read(["uid", "cls"])
        lut = dict(zip(t["uid"].tolist(), t["cls"].tolist()))
        cls = np.array([lut.get(int(u), -1) for u in uid])
        cols["cls"] = cls
        shares = class_shares(cls[cls >= 0], fits[cls >= 0], chance)
    project.table(job, "particles").write(cols, src.parts["celMovie"].to_numpy(np.int64))
    top = sorted((c["fit"] for c in shares if c["n"] >= 100), reverse=True)[:10]
    bar = max(s.propose_over_chance * chance, s.propose_relative * float(np.median(top)) if top else 0.0)
    proposed = [c["cls"] for c in shares if c["lo"] > bar]
    _figure(job.out("mapscore.png"), val, cval, thr, shares, bar, sd is not None, f"{job.name}: {fits.mean() * 100:.1f} % of "
            f"{len(uid):,} particles fit {mj.name} (chance {chance * 100:.0f} %)")
    summ = dict(particles=len(uid), fit=round(float(fits.mean()), 4), chance=chance, threshold=round(thr, 4),
                score_median=round(float(np.median(val)), 4), control_median=round(float(np.median(cval)), 4),
                decoy_res_a=s.decoy_res_a if sd is not None else None,
                seconds=round(time.time() - t0, 1), classes=shares, proposal_bar=round(bar, 4),
                proposed_classes=proposed,
                proposed_particles=int(sum(c["n"] for c in shares if c["cls"] in proposed)))
    log(f"[mapscore] {summ['fit'] * 100:.1f} % fit (chance {chance * 100:.0f} %), median score {summ['score_median']} "
        f"vs control {summ['control_median']}; {len(proposed)} classes proposed; {summ['seconds']} s")
    project.set_status(job, "finished", summ)
    return summ


@themed
def _figure(path, score, ctrl, thr, shares, bar, decoy, title_):
    n = 2 if shares else 1
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 3.6), squeeze=False)
    title(fig, "Particles vs map", title_)
    fig.subplots_adjust(top=top(fig, 0.82), bottom=0.16, left=0.07, right=0.98, wspace=0.25)
    ax = axes[0, 0]
    bins = np.histogram_bin_edges(np.concatenate([score, ctrl]), bins=60)
    ax.hist(ctrl, bins=bins, color=T.series[1], alpha=0.7, label="phase-randomised control")
    ax.hist(score, bins=bins, color=T.series[0], alpha=0.7, label="particles")
    ax.axvline(thr, color=STATUS["warning"], lw=1.2, ls="--")
    ax.set(xlabel="correlation with the map minus with a structureless decoy" if decoy else
           "best correlation with a map projection", ylabel="particles")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    grid_y(ax)
    if shares:
        ax = axes[0, 1]
        sh = sorted(shares, key=lambda c: -c["fit"])
        x = np.arange(len(sh))
        f = np.array([c["fit"] for c in sh])
        ax.bar(x, f, color=[T.series[0] if c["lo"] > bar else T.series[1] for c in sh], width=0.8)
        ax.errorbar(x, f, yerr=[f - [c["lo"] for c in sh], [c["hi"] for c in sh] - f], fmt="none",
                    ecolor=T.ink3, lw=0.8)
        ax.axhline(bar, color=STATUS["warning"], lw=1.0, ls="--")
        ax.set(xlabel="2D class (sorted)", ylabel="share of particles that fit", xticks=[])
        note(ax, f"{sum(c['lo'] > bar for c in sh)} of {len(sh)} classes proposed")
        grid_y(ax)
    fig.savefig(path)
