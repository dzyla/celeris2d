"""GPU maximum-likelihood 2D classification (PyTorch).

Model (per particle n, Fourier space, rfft2 half-plane, norm="forward"):
    X_n = C_n * shift_t( rot_psi( A_k ) ) + noise,   noise ~ CN(0, sigma2(|f|))

E-step: exhaustive search over classes k, in-plane rotations psi and a local
translation window around each particle's previous best offset. Rotations are
applied to the references (2x Fourier-upsampled, bilinear); translations are
evaluated exactly (sub-pixel) as separable DFT phase ramps, so the whole search
is two batched complex GEMMs per particle batch.

M-step: exact marginalisation over all (k, psi, t). The weighted back-shift is
the transpose of the E-step GEMMs; CTF^2 weights are accumulated on a 2x
oversampled Fourier grid and rotated back with the numerators. Classes are
Wiener-filtered with a data-driven per-shell signal power tau2 (RELION-like
tau2_fudge), masked, and re-centred on their centre of mass each iteration.
"""
from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Fourier / geometry helpers. All spectra are rfft2 with norm="forward", which
# makes Fourier values invariant to the sampling (box) for a fixed field of view,
# so cropping a spectrum is exactly Fourier down-sampling.
# --------------------------------------------------------------------------- #
def release_cache():
    """Return PyTorch's cached GPU blocks, after every stream has finished: a background stream (the particle loader)
    may still use blocks the cache holds, and with expandable segments the release unmaps them without waiting (an
    illegal memory access in the loader's kernels: EMPIAR-10291 ab initio while particles streamed in)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cap_memory(dev):
    """Hard limit for this process on a CUDA device: the card less the headroom gpu_free plans around (and at most
    CELERIS2D_VRAM_GB). Past it PyTorch frees its cache and retries, then raises out-of-memory: the driver never
    spills GPU memory into system RAM (WSL does, and everything slows down many times)."""
    dev = torch.device(dev)
    if dev.type != "cuda":
        return
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(dev).total_memory
    used_by_others = total - torch.cuda.mem_get_info(dev)[0] - torch.cuda.memory_reserved(dev)
    limit = total - max(int(0.1 * total), int(1.5e9)) - max(0, used_by_others)
    cap = os.environ.get("CELERIS2D_VRAM_GB")
    if cap:
        limit = min(limit, int(float(cap) * 1e9))
    torch.cuda.set_per_process_memory_fraction(max(0.2, limit / total), dev)


def gpu_free(dev) -> int:
    """Memory (bytes) this process may still plan to use on dev: what the device has free, less a headroom of 10 % of
    the card (at least 1.5 GB) that is never planned for (the display, other programs, allocations that are not
    sized from free memory: a Refine3D sized from all free memory held 15.6 of 16 GB), and capped by
    CELERIS2D_VRAM_GB (a budget for everything this process holds)."""
    dev = torch.device(dev)
    free, total = torch.cuda.mem_get_info(dev)
    free = max(0, free - max(int(0.1 * total), int(1.5e9)))
    cap = os.environ.get("CELERIS2D_VRAM_GB")
    if cap:
        free = min(free, max(0, int(float(cap) * 1e9) - torch.cuda.memory_reserved(dev)))
    return int(free)


def freqs(b: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed row frequencies (b,) and column frequencies (b//2+1,), in cycles/box."""
    fy = torch.fft.fftfreq(b, d=1.0 / b, device=device)
    fx = torch.arange(b // 2 + 1, device=device, dtype=torch.float32)
    return fy, fx


def ft_resize(ft: torch.Tensor, b_new: int) -> torch.Tensor:
    """Crop or zero-pad half spectra (..., b, b//2+1) -> (..., b_new, b_new//2+1)."""
    b = ft.shape[-2]
    if b == b_new:
        return ft
    h = min(b, b_new) // 2
    out = ft.new_zeros(ft.shape[:-2] + (b_new, b_new // 2 + 1))
    out[..., :h, : h + 1] = ft[..., :h, : h + 1]
    out[..., -h:, : h + 1] = ft[..., -h:, : h + 1]
    return out


def rot_grid(angles_deg: torch.Tensor, b_out: int, b_in: int) -> torch.Tensor:
    """grid_sample grid that rotates content counter-clockwise (x=col, y=row) by angle,
    about pixel b/2, reading from a (b_in/b_out)-times oversampled input."""
    dev = angles_deg.device
    c = b_out // 2
    r = torch.arange(b_out, device=dev, dtype=torch.float32) - c
    y, x = torch.meshgrid(r, r, indexing="ij")
    a = torch.deg2rad(angles_deg)[:, None, None]
    ca, sa = torch.cos(a), torch.sin(a)
    xs = ca * x + sa * y  # source = R(-a) @ (x, y)
    ys = -sa * x + ca * y
    s = b_in / b_out
    u = (xs * s + b_in // 2 + 0.5) / b_in * 2 - 1
    v = (ys * s + b_in // 2 + 0.5) / b_in * 2 - 1
    return torch.stack([u, v], dim=-1)


def upsample2(ft: torch.Tensor, b: int) -> torch.Tensor:
    """Half spectra at box b -> real images at box 2b (sinc interpolation)."""
    return torch.fft.irfft2(ft_resize(ft, 2 * b), s=(2 * b, 2 * b), norm="forward")


def half_to_full_centered(h: torch.Tensor, b: int) -> torch.Tensor:
    """Real, centro-symmetric half-plane maps (..., b, b//2+1) -> centred full plane (..., b, b)."""
    full = h.new_zeros(h.shape[:-2] + (b, b))
    full[..., : b // 2 + 1] = h
    rows = (-torch.arange(b, device=h.device)) % b
    full[..., b // 2 + 1 :] = h[..., rows, :][..., 1 : b // 2].flip(-1)
    return torch.fft.fftshift(full, dim=(-2, -1))


def soft_circle(b: int, radius_px: float, edge_px: float, device) -> torch.Tensor:
    r = torch.arange(b, device=device, dtype=torch.float32) - b // 2
    rr = torch.sqrt(r[:, None] ** 2 + r[None, :] ** 2)
    m = torch.clamp((radius_px + edge_px - rr) / max(edge_px, 1e-3), 0, 1)
    return 0.5 - 0.5 * torch.cos(math.pi * m)


# --------------------------------------------------------------------------- #
@dataclass
class Optics:
    voltage_kv: float
    cs_mm: float
    q0: float
    no_ctf: bool = False   # images without CTF information: CTF == 1 everywhere

    @property
    def wavelength(self) -> float:
        v = self.voltage_kv * 1e3
        return 12.2643247 / math.sqrt(v * (1 + v * 0.978466e-6))


def _host_available() -> float:
    """Bytes of host RAM available for new allocations (Linux MemAvailable; unknown -> unlimited)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return float("inf")


def ctf_2d(p: torch.Tensor, optics: Optics, fy: torch.Tensor, fx: torch.Tensor, box_a: float,
           ang_offset_deg: torch.Tensor | None = None) -> torch.Tensor:
    """RELION-convention CTF on a half-plane grid.

    p: (n, 6) = defocusU, defocusV [A], defocus angle [deg], phase shift [deg], B-factor, scale.
    ang_offset_deg: (n,) rotation of the CTF frame (used to express a particle's CTF in a
    rotated class frame: astig angle -> angle - offset).
    """
    if optics.no_ctf:
        return torch.ones((p.shape[0], len(fy), len(fx)), device=p.device)
    lam, cs = optics.wavelength, optics.cs_mm * 1e7
    k3 = math.atan(optics.q0 / math.sqrt(1 - optics.q0 ** 2))
    sy = (fy / box_a)[:, None]
    sx = (fx / box_a)[None, :]
    s2 = sx * sx + sy * sy
    ang = torch.atan2(sy.expand_as(s2), sx.expand_as(s2))
    du, dv, da, ph, bf, sc = (p[:, i, None, None] for i in range(6))
    da = torch.deg2rad(da if ang_offset_deg is None else da - ang_offset_deg[:, None, None])
    df = 0.5 * (du + dv) + 0.5 * (du - dv) * torch.cos(2 * (ang - da))
    chi = math.pi * lam * df * s2 - 0.5 * math.pi * cs * lam ** 3 * s2 * s2 + k3 + torch.deg2rad(ph)
    out = sc * torch.sin(chi)
    if torch.any(bf != 0):
        out = out * torch.exp(-bf * s2 / 4)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class Params:
    n_classes: int = 20
    n_iter: int = 25
    diameter_a: float = 200.0
    max_res_a: float = 6.0          # final E/M-step resolution limit
    max_c64_gb: float = 0.0         # particle spectra larger than this (GB as complex64) are stored as fp16 even if they
                                    # fit (0: no limit); the 3D jobs set 1.5 to leave the GPU to their volumes
    nyquist_margin: float = 1.1     # max_res_a is at least this x Nyquist: 2D never classifies at Nyquist (the last shells
                                    # only carry interpolation/aliasing artefacts); 3D refinements use 1.0 (their FSC
                                    # decides; beta-gal at 3.1 A was capped by 2.2 x 1.24 A/px = 2.7 A)
    init_res_a: float = 30.0        # starting resolution limit (frequency marching)
    march_frac: float = 0.6         # fraction of iterations used to reach max_res
    psi_step_min: float = 2.0
    psi_factor: float = 1.0         # in-plane step = psi_factor * degrees(res / diameter)
    trans_div: float = 4.0          # translation step = res / trans_div
    psi_step_max: float = 15.0
    max_shift_a: float = 16.0       # local translation search radius around previous offset
    max_trans_half: int = 4         # cap on translation grid half-width (4 -> 9 x 9 positions)
    max_offset_a: float = -1.0      # cap on accumulated offset (-1 = diameter / 4)
    precenter_a: float = 0.0        # first visit of a particle: centre it within this radius on the rotational
                                    # average of the mean class before the local search (-1 = 0.15 x diameter, 0 = off).
                                    # The local window (max_shift_a) cannot reach off-centre picks: on NS J75 an
                                    # exhaustive E-step vs cryoSPARC's final classes drops F1 0.61 -> 0.42 at +-19 A
    # (+-38 A: 0.635; rotation step, alignment resolution and particle mask cost little)
    local_refine: bool = True       # hard pose: coarse global search at 2x the angular step, then for each particle's
                                    # best classes a local search at refine_split x finer angles and half-step shifts
                                    # around the coarse optimum; class sums by direct rotation at the fine pose. Every
                                    # iteration, not only full passes: with reseed_until 0.9, 4 seeds, single 5090,
                                    # J1798 F1/AP vs J1814 0.507/0.434 -> 0.531/0.449 in 120 -> 61 s, NS J75 0.452/0.392
                                    # -> 0.443/0.393 (seed sd 0.012) in 86 -> 52 s. Full passes only: 89 s and 69 s.
    refine_split: int = 4           # fine angle step = coarse step / refine_split, searched over +-1 coarse step
    refine_classes: int = 2         # classes per particle refined locally (the rest keep their coarse score)
    tau2_fudge: float = 2.0
    batch_per_class: int = 500      # online-EM subset size per class for the early iterations (cryoSPARC-like)
    n_full_iter: int = -1           # final iterations over all particles (-1 = auto from N, then extended)
    full_converge: float = 0.10     # auto: add full passes while more than this fraction changes class
    max_full_iter: int = 3          # auto: at most this many full passes in total. 6 doubled the time on 0.5-1.2M particles
                                    # for no measurable separation gain (J2473 vs J2474 F1 0.464 -> 0.469, J461 vs J466
                                    # 0.600 -> 0.611; seed noise +-0.03): what keeps changing are near-identical views
    online_decay: float = 0.0       # online EM: running-statistics decay (0 = fresh batch each iter; 0.8 hurt)
    anneal_t0: float = 1.0          # likelihood temperature at iter 1 (1 = no annealing)
    reseed_frac: float = 0.25       # reseed classes holding < frac * N/K particles (early iters)
    reseed_until: float = 0.9       # ... during the first reseed_until * n_iter iterations, never in full passes.
                                    # 0.7 left 4-7 of 50 classes empty on J1798 (they empty out once reseeding stops);
                                    # 0.9: 1-3, more distinct views. Reseeding late only below 0.02 N/K: no better
    reseed_full: bool = True        # keep reseeding during the full passes (all but the last two). Off, a class that
                                    # empties in the full passes stays empty (J1798: pop[min] 181 -> 0 at the first
                                    # full pass, 2 empty classes per run)
    reseed_mode: str = "split"      # "split": the largest class is split along the first principal component of its
                                    # members (see _split_class); "noise": a copy of the largest class + 30 % noise.
                                    # A noise copy explains the same particles as its parent, loses, is Wiener-
                                    # filtered to a blob and dies once reseeding stops (J1798 classes 5, 6).
                                    # split + reseed_full + recenter_detrend vs the old defaults, 3 seeds, one RTX
                                    # PRO 4000: empty classes dz01 1.0 -> 0, J1798 2.0 -> 0, tiny (< 2 % N/K) dz01
                                    # 2.3 -> 0, dz03 0.3 -> 0; F1 vs curated sets J75 0.421 -> 0.428, J1798 0.525
                                    # -> 0.527, RELION tutorial 0.845 -> 0.846; seed agreement and time unchanged
                                    # (J75 46 s, J1798 170 -> 178 s within its 158-201 s spread)
    center_classes: bool = True
    recenter_detrend: bool = True   # re-centre on the density after removing the best-fit plane inside the mask. A
                                    # class of a background gradient (ice / carbon edge; Krios set dz01) was being
                                    # re-centred by 40-55 A every iteration (other classes 2 A) until its particles'
                                    # offsets hit max_offset_a and it emptied; with the plane removed 4-14 A and the
                                    # edge classes persist as stable junk classes. Neutral on the curated sets.
    seed: int = 0
    batch_size: int = 0             # particles per GPU batch (0 = auto)
    polarity: int = 0               # 0 = auto-detect, +1 keep, -1 invert contrast
    march_margin: float = 0.0       # alignment (E-step) limit: shells beyond the current SSNR=1 resolution.
                                    # >0 lets alignment chase noise on low-SNR data (honeycomb attractors)
    recon_margin: float = 10.0      # class reconstruction may go this many shells further (Wiener decides)
    polish: bool = False            # last iteration aligns out to the reconstruction limit, classes locked (sharper on
                                    # high-SNR data; scrambles poses on low-SNR data such as J43 -> opt-in;
                                    # classes are converged, so noise gets one pass, not a feedback loop)
    normalize: bool = True          # per-particle background mean/std normalisation
    mstep_fp16: bool = True         # fp16 tensor-core M-step GEMMs
    frc_reg: bool = True            # Wiener filter from odd/even half-set FRC (not from a class's own power)
    particle_scale: bool = False    # E-step: fit a per-particle amplitude for every hypothesis (contrast varies
                                    # between particles, e.g. stain thickness); otherwise dim particles prefer dim classes.
                                    # NS J75 vs J83: single E-step recall 0.13 -> 0.28, but full runs worse (F1 0.457 -> 0.414)
    log_duplicates: bool = False    # diagnostic: log the class pairs whose odd/even halves agree across classes as well as
                                    # within them. Not used to merge: J341's distinct views (ratio 0.97-0.99) and J1798's
                                    # near-identical lattice classes (0.99-1.00) overlap, so no threshold separates them
    gold_standard: bool = False     # each half-set is aligned only to classes built from its own half, so the
                                    # half-set FRC (Wiener filter) cannot mistake a shared noise pattern for signal.
                                    # In practice the two half-classes drift apart and classes collapse: dz01 22 of
                                    # 50 empty, est_res 13 -> 17 A; RELION tutorial F1 0.845 -> 0.830, 4 empty
    sigma_init_factor: float = 1.0  # >1: noise inflated at the start, annealed to 1 (cryoSPARC-style); off: hurt LoG separation
    sigma_anneal_start: int = 2
    sigma_anneal_iters: int = 15
    sigma_decay: float = 0.7        # online EM: running noise estimate = decay * old + (1 - decay) * batch
    recenter_thresh: float = 0.2    # re-centre on the centre of mass of density above this fraction of max
    solvent_flatten: bool = False   # mask each class by a soft envelope of its own density (flat solvent). Off by
                                    # default: on weak/small particles the envelope takes in the surrounding noise
                                    # and grows a honeycomb (J461/J2473); helps clean, well-separated views
    solvent_thresh: float = 0.08    # envelope core: 20 A low-passed density above this fraction of max
    solvent_dilate_a: float = 20.0  # envelope extends this far beyond the core (A)
    notch_thresh: float = 2.5       # zero Fourier pixels whose mean power is > this x their shell mean
                                    # (fixed-frame detector/extraction artefacts; 0 = off)
    noise_model: str = "coloured"   # "coloured": per-Fourier-pixel noise in the particle frame (down-weights
                                    # detector/extraction artefacts); "radial": one value per shell. Flat (white)
                                    # E-step weights agreed best with cryoSPARC's own classes in an exhaustive search,
                                    # but in full runs: NS J75 F1 0.442 -> 0.421 (estRes 16 -> 23 A), J43 0.451 -> 0.450,
                                    # RELION tutorial 0.849 -> 0.849 -> not kept
    init_classes: str = ""          # optional .mrc/.mrcs of starting class images (reference-based start)
    spill_dir: str = ""             # where spectra go when they fit neither on the GPU nor in host RAM (a disk-backed
                                    # file read through the page cache; "" = $TMPDIR); a fast local SSD is best
    hard_pose: bool | None = None   # best pose per class, soft over classes (cryoSPARC 'force max'); None = auto
    class_prior: bool | None = None # class populations as prior (False = uniform); None = auto
    # auto: >= 100k particles -> hard pose + uniform prior (resists noise attractors on big low-SNR sets);
    # smaller sets -> soft marginalisation + class prior (RELION-like; keeps confidence informative)


class Classifier2D:
    def __init__(self, source, prm: Params, devices=("cuda",), log=print):
        """source: object with N, box, apix, ctf (N x 6), optics, chunks() -> (rows, images)
        and optionally polarity_hint (+1 protein white / -1 protein dark)."""
        self.p, self.optics, self.log = prm, source.optics, log
        self.devs = [torch.device(d) for d in devices]
        self.dev = self.devs[0]  # master device: references, reconstruction
        for d in self.devs:
            cap_memory(d)
        self.N, B, apix = source.N, source.box, source.apix
        self.B_full, self.apix_full = B, apix
        self.box_a = B * apix
        self.rng = torch.Generator(device="cpu").manual_seed(prm.seed)
        torch.manual_seed(prm.seed)
        self._auto_full, self._auto_iter = prm.n_full_iter < 0, prm.n_iter < 0
        self._auto_hard, self._auto_prior = prm.hard_pose is None, prm.class_prior is None

        # never classify right at Nyquist: the last shells only carry interpolation/aliasing artefacts
        lim = 2 * prm.nyquist_margin * apix
        if prm.max_res_a < lim - 1e-6:
            log(f"[res] --max-res {prm.max_res_a:.1f} A is at Nyquist ({2 * apix:.1f} A); using {lim:.1f} A")
            prm.max_res_a = lim
        # working (maximum) box: Nyquist at max_res, even, never above the input box
        bmax = int(math.ceil(2 * self.box_a / prm.max_res_a / 2.0)) * 2 + 4
        self.bmax = min(B, bmax + (bmax % 2))
        self.apix_max = self.box_a / self.bmax
        self.radius_a = prm.diameter_a / 2
        if prm.max_offset_a < 0:
            prm.max_offset_a = prm.diameter_a / 4
        if prm.precenter_a < 0:
            prm.precenter_a = min(0.15 * prm.diameter_a, prm.max_offset_a)
        self._mdt = torch.float16 if prm.mstep_fp16 else torch.float32
        # per-particle metadata/state lives on the host; batches are shipped to whichever GPU is free
        self.ctf_p = torch.as_tensor(np.asarray(source.ctf), dtype=torch.float32)

        nsh = self.bmax // 2 + 1
        fy, fx = freqs(self.bmax, self.dev)
        self.shell_max = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)).long().clamp_(max=nsh - 1)
        self.nsh = nsh
        self.polarity_hint = getattr(source, "polarity_hint", None)
        self._load(source)
        self.offsets = torch.zeros(self.N, 2)  # (x, y) in A, model shift t
        self.psi = torch.zeros(self.N)
        self.cls = torch.zeros(self.N, dtype=torch.long)
        self.maxprob = torch.zeros(self.N)
        self.loglik = torch.zeros(self.N)
        self.nsig = torch.zeros(self.N, dtype=torch.long)
        self.seen = torch.zeros(self.N, dtype=torch.bool)
        self._polarity_done = False
        self.stop_requested = False  # set from another thread: finish the current iteration, then return
        self.stopped = False
        self.init_images = None  # optional (k, bmax, bmax) starting classes for the next run()
        self.notch = None        # (bmax, bmax//2+1) 0/1 mask of fixed-frame artefact pixels
        self._set_active(None)

    def _set_active(self, active):
        """Restrict classification to host indices `active` (None = all) and reset the run state.
        The loaded spectra are shared, so repeated runs on subsets (deep2D rounds) cost no I/O."""
        p = self.p
        self.active = torch.arange(self.N) if active is None else torch.as_tensor(np.asarray(active), dtype=torch.long)
        self.active_mask = torch.zeros(self.N, dtype=torch.bool)
        self.active_mask[self.active] = True
        self.Na = len(self.active)
        if self._auto_full:  # full passes are what costs time on big data sets
            p.n_full_iter = 6 if self.Na <= 50_000 else 3 if self.Na <= 500_000 else 2
        if self._auto_iter:  # big data sets get more (cheap) online-EM iterations
            p.n_iter = 25 if self.Na <= 100_000 else 40
        big = self.Na >= 100_000
        if self._auto_hard:
            p.hard_pose = big
        if self._auto_prior:
            p.class_prior = not big
        self.offsets[self.active] = 0
        self.seen[self.active] = False
        self.S_num = self.S_den = self.S_pop = None  # running (online-EM) sufficient statistics, class frame
        self.S_numh = self.S_denh = None              # the same per half-set (gold-standard alignment)
        self.classes_h = None                         # (2, K, bm, bh) half-set classes (gold-standard)
        self._last_enc = float('nan')
        self._last_nbad, self._bad_logged = 0, False
        self.prior = torch.full((p.n_classes,), 1.0 / p.n_classes, device=self.dev)
        self.class_res = torch.full((p.n_classes,), float(p.init_res_a))
        # per-particle class posterior (marginal over poses) of the last pass that saw the particle
        self.cpost = torch.zeros((self.N, p.n_classes), dtype=torch.float16)
        self.est_res = p.init_res_a
        self.align_cap = None
        self.history = []

    # ------------------------------------------------------------------ data
    def _load(self, source):
        """Start streaming particles in the background: normalise, mask, FFT, Fourier-crop to bmax and
        store fp16/complex64 spectra (master GPU if they fit, else pinned host RAM). Stack files are read
        in random order, so the particles available at any moment are a random sample; classification
        starts once a first batch is in and only full passes wait for the rest."""
        B, bm = self.B_full, self.bmax
        bh = bm // 2 + 1
        c64 = self.N * bm * bh * 8
        free = gpu_free(self.dev) if self.dev.type == "cuda" else 0
        if c64 < 0.35 * free and not (self.p.max_c64_gb and c64 > self.p.max_c64_gb * 1e9):
            self.half, where = False, self.dev
        elif c64 / 2 < 0.6 * free:
            self.half, where = True, self.dev
        else:
            self.half, where = True, torch.device("cpu")
        self.store_scale = float(B)  # forward-FFT coefficients ~ 1/B for unit-variance images: keeps fp16 O(1)
        shape = (self.N, bm, bh, 2) if self.half else (self.N, bm, bh)
        self._spill = None
        if where.type == "cpu" and c64 / 2 > 0.6 * _host_available():
            # too big for RAM as well: a disk-backed array (the OS keeps what fits in its page cache)
            d = self.p.spill_dir or tempfile.gettempdir()
            fd, self._spill = tempfile.mkstemp(prefix="celeris2d_", suffix=".f16", dir=d)
            os.close(fd)
            arr = np.memmap(self._spill, dtype=np.float16, mode="w+", shape=shape)
            os.unlink(self._spill)  # space is freed when the array is (the open mapping keeps the file alive)
            self.data = torch.from_numpy(arr)
            self._spill_arr = arr
        else:
            self.data = torch.empty(shape, dtype=torch.float16 if self.half else torch.complex64, device=where,
                                    pin_memory=where.type == "cpu" and self.dev.type == "cuda")
        self.flip = False
        self._loaded = []                 # list of host index arrays, in load order
        self._n_loaded = 0
        self._load_lock = threading.Lock()
        self._load_err = None
        self._load_t0 = time.time()
        kind = ("fp16 " if self.half else "") + ("on " + str(where) if where.type == "cuda" else
                                                 f"in a disk-backed file in {os.path.dirname(self._spill)} (larger "
                                                 f"than RAM; slower passes)" if self._spill else "in pinned host RAM")
        self.log(f"[load] streaming {self.N} particles, box {B} -> working box {bm} ({self.apix_max:.3f} A/px), "
                 f"spectra {kind} ({c64 / (2e9 if self.half else 1e9):.1f} GB)")
        self._loader = threading.Thread(target=self._load_worker, args=(source,), daemon=True)
        self._loader.start()

    def _load_worker(self, source):
        try:
            B, bm = self.B_full, self.bmax
            dev = self.dev
            r = torch.arange(B, device=dev, dtype=torch.float32) - B // 2
            rr = torch.sqrt(r[:, None] ** 2 + r[None, :] ** 2)
            rad_px = self.radius_a / self.apix_full
            mask = soft_circle(B, rad_px, 0.05 * B, dev)
            bg = rr > rad_px
            if bg.sum() < 0.05 * B * B:  # mask fills the box: normalise on the outer ring instead
                bg = rr > 0.8 * (B / 2)
            stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
            next_log = 0
            with torch.cuda.device(dev), torch.cuda.stream(stream):
                for rows, imgs in source.chunks(shuffle_seed=self.p.seed):
                    rows_t = torch.as_tensor(rows, device=self.data.device)
                    for i in range(0, len(rows), 2048):
                        x = torch.as_tensor(imgs[i:i + 2048], device=dev, dtype=torch.float32)
                        xb = x[:, bg]
                        if self.p.normalize:
                            x = (x - xb.mean(1)[:, None, None]) / xb.std(1).clamp_min(1e-6)[:, None, None]
                        ft = ft_resize(torch.fft.rfft2(x * mask, norm="forward"), bm)
                        if self.half:
                            ft = torch.view_as_real(ft * self.store_scale).half()
                        self.data[rows_t[i:i + 2048]] = ft.to(self.data.device)
                    if stream is not None:
                        stream.synchronize()  # rows are only published once their spectra are written
                    with self._load_lock:
                        self._loaded.append(torch.as_tensor(rows))
                        self._n_loaded += len(rows)
                    if self._n_loaded >= next_log:
                        self.log(f"[load] {self._n_loaded}/{self.N} particles "
                                 f"({time.time() - self._load_t0:.0f}s)")
                        next_log += max(50_000, self.N // 10)
            self.log(f"[load] all {self.N} particles loaded in {time.time() - self._load_t0:.1f}s")
        except BaseException as e:  # surfaced in the main thread by _wait_loaded
            self._load_err = e

    def _wait_loaded(self, n: int):
        """Block until at least n particles are available (n >= N waits for everything)."""
        n = min(n, self.N)
        t0 = time.time()
        while self._n_loaded < n:
            if self._load_err is not None:
                raise RuntimeError("particle loading failed") from self._load_err
            if not self._loader.is_alive() and self._n_loaded < n:
                raise RuntimeError("particle loader stopped early")
            time.sleep(0.05)
        if time.time() - t0 > 1:
            self.log(f"[load] waited {time.time() - t0:.1f}s for {n} particles")

    def _pool(self) -> torch.Tensor:
        """Host indices of the active particles loaded so far."""
        with self._load_lock:
            parts = list(self._loaded)
        pool = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
        return pool[self.active_mask[pool]]

    def _sample(self, n: int) -> torch.Tensor:
        """Random sample of n active, loaded particles (waits until enough are loaded)."""
        n = min(n, self.Na)
        pool = self._pool()
        while len(pool) < n and self._n_loaded < self.N:
            self._wait_loaded(self._n_loaded + 1)
            pool = self._pool()
        return pool[torch.randperm(len(pool), generator=self.rng)[:n]]

    def _fetch(self, idx: torch.Tensor, dev, b: int) -> torch.Tensor:
        """Particle spectra (complex64) for host indices idx, on device dev, cropped to box b."""
        d = self.data
        x = d[idx.to(d.device)] if d.device.type == "cuda" else d[idx]
        x = x.to(dev, non_blocking=True)
        if self.half:
            x = torch.view_as_complex(x.float()) / self.store_scale
        if self.flip:
            x = -x
        if self.notch is not None:
            x = x * self.notch.to(x.device)
        return ft_resize(x, b)

    def ensure_polarity(self, n_wait: int = 20000):
        """Decide the contrast once per loaded dataset, before particle data is used. Everything that reads
        particles without starting a 2D run (ab initio from a restored 2D result) must call this: a negative-stain
        stack (J85) used as stored gives a 3D map with dark protein."""
        if not self._polarity_done:
            self._wait_loaded(n_wait)
            self._check_polarity(self.polarity_hint)
            self._polarity_done = True

    def _check_polarity(self, hint):
        """Protein must be white. Decide from the unaligned CTF-weighted average of loaded particles:
        centred picks give a positive central blob. --invert / --no-invert override."""
        force = self.p.polarity
        if force:
            flip = force < 0
            how = "forced"
        else:
            bm = self.bmax
            fy, fx = freqs(bm, self.dev)
            sub = self._sample(20000)
            num = torch.zeros((bm, bm // 2 + 1), dtype=torch.complex64, device=self.dev)
            den = torch.zeros((bm, bm // 2 + 1), device=self.dev)
            for j in range(0, len(sub), 2048):
                idx = sub[j:j + 2048]
                c = ctf_2d(self.ctf_p[idx].to(self.dev), self.optics, fy, fx, self.box_a)
                num += (c * self._fetch(idx, self.dev, bm)).sum(0)
                den += (c * c).sum(0)
            kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
            ft = num / (den + 0.1 * den.mean()) * torch.exp(-0.5 * (kr / (self.box_a / 30.0)) ** 2)
            img = torch.fft.irfft2(ft, s=(bm, bm), norm="forward")
            r = torch.arange(bm, device=self.dev, dtype=torch.float32) - bm // 2
            rr = torch.sqrt(r[:, None] ** 2 + r[None, :] ** 2) * self.apix_max
            centre = img[rr < 0.3 * self.radius_a].mean()
            ring = img[(rr > 0.9 * self.radius_a) & (rr < 1.1 * self.radius_a)].mean()
            flip = bool(centre < ring)
            how = f"auto (centre-ring contrast {float(centre - ring):+.3g}"
            how += f"; metadata: {hint})" if hint else ")"
        self.flip = flip
        self.log(f"[polarity] {how}: {'inverting to make protein white' if flip else 'protein already white'}")

    def _find_notch(self):
        """Fourier pixels whose mean power across particles is far above their shell average are artefacts
        fixed in the micrograph frame (detector/gain stripes, extraction): real particle signal is spread
        evenly over a shell by random in-plane orientations. Aligning on them turns noise picks into
        hexagonal 'honeycomb' classes, so they are zeroed in every particle (alignment and averages)."""
        self._notch_done = True
        if self.p.notch_thresh <= 0:
            return
        P = self._initial_noise()
        sh = self.shell_max
        rad = self._shell_mean(P, sh)
        ratio = P / rad[sh].clamp_min(1e-30)
        bad = (ratio > self.p.notch_thresh) & (sh >= 2) & (sh <= self.bmax // 2 - 1)
        if bad.any():
            self.notch = (~bad).float()
            fy, fx = freqs(self.bmax, self.dev)
            where = [f"({int(fy[r])},{int(fx[c])}) {float(ratio[r, c]):.1f}x {self.box_a / float(sh[r, c]):.1f}A"
                     for r, c in torch.nonzero(bad)[:6].tolist()]
            self.log(f"[notch] {int(bad.sum())} artefact Fourier pixels zeroed: " + ", ".join(where))

    def _initial_noise(self) -> torch.Tensor:
        """Mean power spectrum of (up to 20k) loaded particles -> initial per-shell noise."""
        sub = self._sample(20000)
        acc = torch.zeros((self.bmax, self.bmax // 2 + 1), device=self.dev)
        for j in range(0, len(sub), 2048):
            acc += (self._fetch(sub[j:j + 2048], self.dev, self.bmax).abs() ** 2).sum(0)
        return acc / len(sub)

    def _shell_mean(self, m2d: torch.Tensor, shell: torch.Tensor, weight=None) -> torch.Tensor:
        w = torch.ones_like(m2d) if weight is None else weight
        num = torch.zeros(self.nsh, device=self.dev).index_add_(0, shell.flatten(), (m2d * w).flatten())
        den = torch.zeros(self.nsh, device=self.dev).index_add_(0, shell.flatten(), w.flatten())
        return num / den.clamp_min(1e-12)

    def _sigma_factor(self, it):
        p = self.p
        a = np.clip(1.0 - (it - p.sigma_anneal_start) / max(p.sigma_anneal_iters, 1), 0.0, 1.0)
        return 1.0 + (p.sigma_init_factor - 1.0) * a

    # ------------------------------------------------------------- schedule
    def _iter_settings(self, it: int) -> dict:
        p = self.p
        n_march = max(1, int(round(p.march_frac * p.n_iter)) - 1)
        a = min(1.0, it / n_march)
        inv = (1 - a) / p.init_res_a + a / p.max_res_a
        # never march more than a few shells beyond the resolution the data currently support
        inv_sched = inv
        inv = min(inv_sched, 1.0 / self.est_res + p.march_margin / self.box_a)
        res = max(p.max_res_a, 1.0 / inv)                        # alignment limit
        full_pass = it >= p.n_iter - p.n_full_iter or p.batch_per_class <= 0
        polish = p.polish and it == p.n_iter - 1 and it > 0
        if full_pass and self.align_cap is not None and not polish:
            # alignment depends on per-PARTICLE signal; a full pass has far more particles per class, which
            # inflates the class SSNR estimate. Keep the limit the constant-size online batches established.
            res = max(res, self.align_cap)
        elif not full_pass:
            self.align_cap = res
        klim = self.box_a / res
        b_al = min(self.bmax, int(math.ceil(klim)) * 2 + 4)
        inv_rec = max(inv, min(inv_sched, 1.0 / self.est_res + p.recon_margin / self.box_a))
        res_rec = max(p.max_res_a, 1.0 / inv_rec)                # reconstruction limit (>= alignment)
        if polish:
            res = res_rec
            b_al = min(self.bmax, int(math.ceil(self.box_a / res)) * 2 + 4)
        klim_rec = self.box_a / res_rec
        b = min(self.bmax, int(math.ceil(klim_rec)) * 2 + 4)
        klim = self.box_a / res
        step = float(np.clip(p.psi_factor * math.degrees(res / p.diameter_a), p.psi_step_min, p.psi_step_max))
        n_rot = int(math.ceil(360.0 / step))
        tstep = res / p.trans_div
        # local search around the previous offset, at most 9 x 9 positions: at high resolution the step is fine and
        # +-max_shift would mean 11 x 11 or more (cost ~ nt^2); larger offsets still accumulate over iterations
        nt = min(int(math.floor(p.max_shift_a / tstep)), p.max_trans_half)
        temp = 1.0 + (p.anneal_t0 - 1.0) * max(0.0, 1.0 - it / max(1, n_march))
        full = it >= p.n_iter - p.n_full_iter or p.batch_per_class <= 0
        if full and it > p.n_iter - p.n_full_iter and it > 0:
            nt = min(nt, 2)  # every particle already has an offset from the previous full pass: refine locally
        n_sub = self.Na if full else min(self.Na, p.batch_per_class * p.n_classes)
        return dict(polish=polish, res=res, klim=klim, b_al=b_al, res_rec=res_rec, klim_rec=klim_rec, b=b, n_rot=n_rot, psi=torch.arange(n_rot, device=self.dev) * (360.0 / n_rot),
                    tgrid=torch.arange(-nt, nt + 1, device=self.dev, dtype=torch.float32) * tstep,
                    temp=temp, n_sub=n_sub)

    # ------------------------------------------------------- initialisation
    def init_classes(self):
        """Each class = CTF-corrected, unaligned average of a random particle subset."""
        K, bm = self.p.n_classes, self.bmax
        self.ensure_polarity(max(20000, K * 400))
        if not getattr(self, "_notch_done", False):
            self._find_notch()
        p2 = self._initial_noise()                                     # initial noise = data power (2D)
        nw = None if self.notch is None else self.notch.to(self.dev)
        self.sigma2_emp = self._shell_mean(p2, self.shell_max, nw)     # radial, for reconstruction
        if nw is not None:  # notched pixels hold no data: give them their shell's noise, never ~0
            p2 = torch.where(nw > 0, p2, self.sigma2_emp[self.shell_max])
        self.noise2d = p2.clone()                                      # particle frame, bmax half-plane
        fy, fx = freqs(bm, self.dev)
        perm = self._sample(K * min(self.Na // K, 1000))
        num = torch.zeros((K, bm, bm // 2 + 1), dtype=torch.complex64, device=self.dev)
        den = torch.zeros((K, bm, bm // 2 + 1), device=self.dev)
        for k in range(K):
            idx = perm[k::K]
            for j in range(0, len(idx), 1024):
                sub = idx[j:j + 1024]
                c = ctf_2d(self.ctf_p[sub].to(self.dev), self.optics, fy, fx, self.box_a)
                num[k] += (c * self._fetch(sub, self.dev, bm)).sum(0)
                den[k] += (c * c).sum(0)
        ft = num / (den + 0.5 * den.mean())
        kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        ft = ft * torch.exp(-0.5 * (kr / (self.box_a / self.p.init_res_a)) ** 2)
        ref, rescale = None, True
        if self.init_images is not None:  # in-memory templates (deep2D), consumed by this run
            ref = torch.as_tensor(np.asarray(self.init_images, dtype=np.float32), device=self.dev)
            self.init_images, rescale = None, False  # same run sampling and scale already
        elif self.p.init_classes:
            import mrcfile
            ref = torch.as_tensor(mrcfile.read(self.p.init_classes).astype(np.float32), device=self.dev)
        if ref is not None:
            if ref.ndim == 2:
                ref = ref[None]
            # Fourier-resample to this run's sampling (same field of view assumed), match power to the data
            ft_ref = ft_resize(torch.fft.rfft2(ref, norm="forward"), bm)[:K]
            scale = (ft.abs() ** 2).sum() / (ft_ref.abs() ** 2).sum().clamp_min(1e-30) * len(ft_ref) / K
            ft[: len(ft_ref)] = ft_ref * (scale.sqrt() if rescale else 1.0)
            self.log(f"[init] {len(ft_ref)} of {K} starting classes from templates")
        self.classes = self._postprocess(ft, bm, center=False)[0]
        # both half-sets start from the same (low-resolution) classes and then evolve independently
        self.classes_h = self.classes[None].repeat(2, 1, 1, 1) if self.p.gold_standard else None
        self.S_numh = self.S_denh = None

    # --------------------------------------------------------- references
    def _make_refs(self, b: int, psi: torch.Tensor, klim: float, classes=None) -> torch.Tensor:
        """Rotated class spectra, layout (b, M, bh) with m = k * R + r."""
        K, R = self.p.n_classes, len(psi)
        up = upsample2(ft_resize(self.classes if classes is None else classes, b), b)  # (K, 2b, 2b)
        grid = rot_grid(psi, b, 2 * b)
        out = torch.empty((R, K, b, b), device=self.dev)
        for r0 in range(0, R, 64):
            g = grid[r0:r0 + 64]
            out[r0:r0 + 64] = F.grid_sample(up[None].expand(len(g), -1, -1, -1), g, mode="bilinear",
                                            padding_mode="zeros", align_corners=False)
        ft = torch.fft.rfft2(out.transpose(0, 1).reshape(K * R, b, b), norm="forward")
        fy, fx = freqs(b, self.dev)
        ft = ft * (fy[:, None] ** 2 + fx[None, :] ** 2 <= klim ** 2)
        return ft.permute(1, 0, 2).contiguous()

    # ------------------------------------------------------------ iterate
    def iterate(self, it: int):
        p, K = self.p, self.p.n_classes
        s = self._iter_settings(it)
        refine = (p.local_refine and p.hard_pose and not s["polish"] and s["n_rot"] >= 16
                  and p.refine_split >= 2)
        if refine:  # coarse global stage at twice the angular step; the fine stage covers +-1 coarse step
            R_c = int(math.ceil(s["n_rot"] / 2))
            s["n_rot"], s["psi"] = R_c, torch.arange(R_c, device=self.dev) * (360.0 / R_c)
        # alignment box (E-step, noise model) and reconstruction box (M-step, classes) are separate
        b, b_al, R, klim, klim_rec = s["b"], s["b_al"], s["n_rot"], s["klim"], s["klim_rec"]
        bh, bh_a, M = b // 2 + 1, b_al // 2 + 1, K * R
        t_start = time.time()
        fy, fx = freqs(b, self.dev)
        kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        fmask = ((kr <= klim_rec) & (kr > 0)).float()                  # reconstruction support
        shell = torch.round(kr).long().clamp_(max=self.nsh - 1)
        fy_a, fx_a = freqs(b_al, self.dev)
        kr_a = torch.sqrt(fy_a[:, None] ** 2 + fx_a[None, :] ** 2)
        fmask_a = ((kr_a <= klim) & (kr_a > 0)).float()                # alignment support
        shell_a = torch.round(kr_a).long().clamp_(max=self.nsh - 1)
        hw = torch.ones(bh_a, device=self.dev)
        hw[0] = hw[-1] = 0.5
        rows_a = (fy_a.long() % self.bmax)                   # alignment-box frequencies on the bmax grid
        if self.notch is not None:  # zeroed artefact pixels carry no data: leave them out of the likelihood
            fmask_a = fmask_a * self.notch.to(self.dev)[rows_a][:, :bh_a]
        if p.noise_model == "coloured":
            noise = self.noise2d[rows_a][:, :bh_a]
        else:
            noise = self.sigma2_emp[shell_a]
        w8 = fmask_a * hw[None, :] / (noise * self._sigma_factor(it)).clamp_min(1e-30)  # (b_al, bh_a)

        if self.classes_h is not None:  # one reference set per half-set
            refs_h = torch.stack([self._make_refs(b_al, s["psi"], klim, self.classes_h[h]) for h in (0, 1)])
        else:
            refs_h = self._make_refs(b_al, s["psi"], klim)[None]        # (H, b_al, M, bh_a) complex
        tg = s["tgrid"]
        T1 = len(tg)
        ph = TWO_PI / self.box_a
        ang = ph * fy[:, None] * tg[None, :]
        ctx = dict(refs=refs_h, r2w=(refs_h.abs() ** 2).permute(0, 2, 1, 3).reshape(len(refs_h), M, -1), w8=w8, tg=tg, psi=s["psi"],
                   Ex=torch.exp(-1j * ph * fx_a[:, None] * tg[None, :]).to(torch.complex64),   # (bh_a, T1)
                   EyT=torch.exp(-1j * ph * tg[:, None] * fy_a[None, :]).to(torch.complex64),  # (T1, b_al)
                   Exr=torch.exp(-1j * ph * fx[:, None] * tg[None, :]).to(torch.complex64),    # (bh, T1) M-step
                   E2=torch.cat([torch.cos(ang), torch.sin(ang)], 0).to(self._mdt),           # (2b, T1): conj(Ey)
                   logprior=torch.log(self.prior.clamp_min(1e-8)).repeat_interleave(R),       # (M,)
                   b=b, b_al=b_al, R=R, M=M, T1=T1, temp=s["temp"], lock_class=s["polish"])
        if refine:
            S = p.refine_split
            st = float(tg[1] - tg[0]) if T1 > 1 else s["res"] / p.trans_div
            sg = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=self.dev) * st      # fine shifts around the coarse one
            cls_h = self.classes_h if self.classes_h is not None else self.classes[None]
            ctx.update(refine=True, C=max(1, min(p.refine_classes, K)),
                       up_al=torch.stack([upsample2(ft_resize(ch, b_al), b_al) for ch in cls_h]),  # (H, K, 2ba, 2ba)
                       dpsi=(torch.arange(2 * S + 1, device=self.dev) - S) * (360.0 / R / S), sg=sg,
                       kmask_a=((kr_a <= klim).float()), logprior_k=torch.log(self.prior.clamp_min(1e-8)),
                       Exf=torch.exp(-1j * ph * fx_a[:, None] * sg[None, :]).to(torch.complex64),   # (bh_a, S)
                       EyfT=torch.exp(-1j * ph * sg[:, None] * fy_a[None, :]).to(torch.complex64))  # (S, b_al)
        if p.precenter_a > 0:  # rotationally averaged mean class, noise-weighted (no half-plane weights: irfft2)
            mc_ = ft_resize(self.classes, b_al).mean(0).real
            rad = torch.zeros(self.nsh, device=self.dev).index_add_(0, shell_a.flatten(), mc_.flatten())
            cnt = torch.zeros(self.nsh, device=self.dev).index_add_(0, shell_a.flatten(), torch.ones_like(mc_).flatten())
            ctx["blob"] = (rad / cnt.clamp_min(1))[shell_a] * w8 / hw[None, :]

        n_sub = s["n_sub"]
        if n_sub < self.Na:
            idx_all = self._sample(n_sub)
            n_sub = len(idx_all)
        else:
            self._wait_loaded(self.N)
            idx_all = self.active
        bn = p.batch_size if p.batch_size > 0 else self._batch_budget(b, b_al, M, T1, len(refs_h))
        ctx["mc"] = max(16, int(0.4e9 / (b * bn * T1 * 8)))  # reference chunk for the GEMMs
        old_cls = self.cls[idx_all].clone()
        old_psi, old_off = self.psi[idx_all].clone(), self.offsets[idx_all].clone()
        seen_before = self.seen[idx_all].clone()

        def finish(h, tot):
            """Half-set h done: its share of the expected residual power (noise model), then its sums rotated
            into the class frame, so the large per-rotation accumulators of one half are freed before the next."""
            if "num" in tot:  # local refinement: sums are in the class frame already, residual computed directly
                return tot.pop("res"), (torch.fft.rfft2(tot.pop("num"), norm="forward"), tot.pop("den"))
            nr, dm2 = tot.pop("numer"), tot.pop("dm2")
            numer = torch.complex(nr[..., :bh], nr[..., bh:])
            del nr
            refs_m = refs_h[min(h, len(refs_h) - 1)].permute(1, 0, 2)
            dm1 = ft_resize(dm2.view(M, 2 * b, b + 1)[:, ::2, ::2], b_al)               # (M, b_al, bh_a)
            cross = (refs_m * torch.conj(ft_resize(numer.permute(1, 0, 2), b_al))).real.sum(0)
            r = (refs_m.abs() ** 2 * dm1).sum(0) - 2 * cross
            del dm1, cross
            return r, self._rotate_back(numer, dm2, s["psi"], b)

        halves, sumx2, pop = self._run_pass(idx_all, ctx, bn, finish)
        if self._last_nbad > 0.5 * n_sub:
            # a pass in which most likelihoods are non-finite carries no information; rebuilding the classes from it
            # would leave K identical empty classes that never separate again (NS J75 seed 2, it 36). Keep the state.
            self.cls[idx_all], self.psi[idx_all], self.offsets[idx_all] = old_cls, old_psi, old_off
            fin = lambda t: "ok" if bool(torch.isfinite(t).all()) else f"{int((~torch.isfinite(t)).sum())} non-finite"
            self.log(f"[warn] it {it + 1}: {self._last_nbad} of {n_sub} particles had non-finite likelihoods; "
                     f"classes and poses kept from the previous iteration (classes {fin(self.classes)}, "
                     f"refs {fin(ctx['refs'])}, weights {fin(w8)}, prior {fin(ctx['logprior'])}, "
                     f"offsets {fin(old_off)}, noise min {float(self.noise2d.min()):.3g})")
            badk = ~torch.isfinite(self.classes).flatten(1).all(1)
            if badk.any():  # a non-finite class would poison every later pass too: replace it by the mean class
                self.classes[badk] = self.classes[~badk].mean(0) if (~badk).any() else 0
            h = dict(self.history[-1]) if self.history else dict(
                res=s["res"], res_rec=s["res_rec"], box=b, n_rot=R, n_trans=T1 * T1, est_res=self.est_res, maxprob=0.0,
                min_pop=0, max_pop=0, reseeded=0, enc=float("nan"))
            h.update(it=it + 1, n=n_sub, time=time.time() - t_start, changed=1.0, skipped=True)
            self.history.append(h)
            return

        # ---- noise model (exact expected residual power per frequency)
        res2 = (sumx2 + halves[0][0] + halves[1][0]) / n_sub
        sig_new = self._shell_mean(res2, shell_a, fmask_a)
        upd = torch.zeros(self.nsh, dtype=torch.bool, device=self.dev)
        upd[torch.unique(shell_a[fmask_a > 0])] = True
        lam_s = p.sigma_decay if n_sub < self.Na else 0.0   # online: running estimate; full pass: exact
        blended = lam_s * self.sigma2_emp + (1 - lam_s) * sig_new
        self.sigma2_emp = torch.where(upd & (sig_new > 0), blended, self.sigma2_emp)
        old = self.noise2d[rows_a][:, :bh_a]
        ok = (fmask_a > 0) & (res2 > 0)
        self.noise2d[rows_a, :bh_a] = torch.where(ok, lam_s * old + (1 - lam_s) * res2, old)
        del refs_h, ctx

        # ---- reconstruction from (running) sufficient statistics in the class frame
        (num0, den0), (num1, den1) = halves[0][1], halves[1][1]
        num_k, den_k = num0 + num1, den0 + den1
        del halves
        lam = 0.0 if (n_sub == self.Na or self.S_num is None) else p.online_decay
        bm = self.bmax
        self.S_num = ft_resize(num_k, bm) + (lam * self.S_num if lam else 0)
        self.S_den = ft_resize(den_k, bm) + (lam * self.S_den if lam else 0)
        self.S_pop = pop + (lam * self.S_pop if lam else 0)
        if self.classes_h is not None:
            nh, dh = torch.stack([ft_resize(num0, bm), ft_resize(num1, bm)]), torch.stack([ft_resize(den0, bm), ft_resize(den1, bm)])
            self.S_numh = nh + (lam * self.S_numh if lam and self.S_numh is not None else 0)
            self.S_denh = dh + (lam * self.S_denh if lam and self.S_denh is not None else 0)
        num_k, den_k, pop = ft_resize(self.S_num, b), ft_resize(self.S_den, b), torch.nan_to_num(self.S_pop.clone())
        sig = self.sigma2_emp[shell]  # reconstruction uses the empirical (un-annealed) noise
        sh_b = shell.flatten()
        nsh = self.nsh
        tau2 = torch.zeros((K, nsh), device=self.dev)
        a_ = (num_k.abs() ** 2 - sig * den_k).reshape(K, -1)
        bden = (den_k ** 2).reshape(K, -1)
        tau2.index_add_(1, sh_b, a_)
        tden = torch.zeros((K, nsh), device=self.dev).index_add_(1, sh_b, bden)
        tau2 = (tau2 / tden.clamp_min(1e-30)).clamp_min(0)
        # a class's SSNR must not rise again once it has fallen: running minimum outward from shell 2.
        # Otherwise noise classes aligned to noise "find" signal in the CTF-zero band (Thon-ring power of
        # empty ice) and the Wiener filter turns it into a lattice.
        dmean0 = torch.zeros((K, self.nsh), device=self.dev).index_add_(1, shell.flatten(), den_k.reshape(K, -1))
        cnt0 = torch.zeros(self.nsh, device=self.dev).index_add_(
            0, shell.flatten(), torch.ones(shell.numel(), device=self.dev))
        dm = (dmean0 / cnt0.clamp_min(1)).clamp_min(1e-30)
        if p.frc_reg:
            # signal = what the odd and even half-sets agree on (not the class's own power, which
            # includes noise that particles were aligned to): SSNR_full = 2 FRC / (1 - FRC)
            def _half(nm, dn):
                return nm / (dn + 0.05 * dn.mean((1, 2), keepdim=True).clamp_min(1e-30))
            A0, A1 = _half(num0, den0), _half(num1, den1)
            cr = torch.zeros((K, nsh), device=self.dev).index_add_(1, sh_b, (A0 * A1.conj()).real.reshape(K, -1))
            p0 = torch.zeros((K, nsh), device=self.dev).index_add_(1, sh_b, (A0.abs() ** 2).reshape(K, -1))
            p1 = torch.zeros((K, nsh), device=self.dev).index_add_(1, sh_b, (A1.abs() ** 2).reshape(K, -1))
            frc = (cr / torch.sqrt(p0 * p1).clamp_min(1e-30)).clamp(0, 0.995)
            ssnr0 = 2 * frc / (1 - frc)
            self._dups = self._duplicates(A0, A1, b, s["klim"], pop) if p.log_duplicates else []
            del A0, A1
        else:
            ssnr0 = tau2 * dm / self.sigma2_emp[None]
        ssnr0[:, 2:] = torch.cummin(ssnr0[:, 2:], dim=1).values
        tau2 = ssnr0 * self.sigma2_emp[None] / dm
        tau2f = ((1.0 if p.frc_reg else p.tau2_fudge) * tau2)[:, shell]                                       # (K, b, bh)
        # raised-cosine roll-off over the last 3 shells instead of a hard cut (no ringing / lattice artefacts)
        taper = 0.5 + 0.5 * torch.cos(math.pi * ((kr - (klim_rec - 3)) / 3).clamp(0, 1))
        ft = num_k * tau2f / (den_k * tau2f + sig) * fmask * taper
        self._wien = dict(tau2f=tau2f, sig=sig, support=fmask * taper, b=b)   # this iteration's filter (_split_class)
        # a class holding less than one particle's weight (late iterations, no more reseeding) has no average: its
        # shell-mean data weight underflows (dm -> 1e-30), tau2 ~ 1e20 multiplies stray numerator values to |class|
        # ~1e15, |ref|^2 overflows in the next E-step and every particle's likelihood is NaN (J1798 loose mask,
        # NS J75 seed 2). It keeps its previous image instead: such classes often win particles back a few
        # iterations later, which a zeroed class never does (J75: F1 0.428 when zeroed).
        low = pop < 1.0
        if low.any():
            ft[low] = ft_resize(self.classes[low], b)
        # SSNR-based resolution estimate (population weighted)
        dmean = torch.zeros((K, nsh), device=self.dev).index_add_(1, sh_b, den_k.reshape(K, -1))
        cnt = torch.zeros(nsh, device=self.dev).index_add_(0, sh_b, torch.ones_like(sh_b, dtype=torch.float32))
        ssnr = tau2 * dmean / cnt.clamp_min(1) / self.sigma2_emp[None]
        wpop = pop / pop.sum()
        gssnr = (ssnr * wpop[:, None]).sum(0)
        below = torch.nonzero(gssnr[1:] < 1.0)
        kres = int(below[0]) + 1 if len(below) else nsh - 1
        self.est_res = self.box_a / max(kres, 1)
        bad = ssnr[:, 1:] < 1.0  # per-class resolution: first shell with SSNR < 1
        kk = torch.where(bad.any(1), bad.float().argmax(1) + 1, torch.full_like(bad[:, 0], nsh - 1, dtype=torch.long))
        self.class_res = (self.box_a / kk.clamp_min(1).float()).cpu()

        if not torch.isfinite(ft).all():  # never let one bad value poison every later iteration
            nb = int((~torch.isfinite(ft)).flatten(1).any(1).sum())
            self.log(f"[warn] it {it + 1}: non-finite values in {nb} class(es) zeroed")
            ft = torch.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0)
        classes, shifts = self._postprocess(ft, b, center=p.center_classes)
        self.classes = ft_resize(classes, self.bmax)
        if self.classes_h is not None:
            # half-set classes: own data, but the signal estimate (tau2) of the full set and the same centring
            hs = []
            for h in (0, 1):
                nh, dh = ft_resize(self.S_numh[h], b), ft_resize(self.S_denh[h], b)
                fth = torch.nan_to_num(nh * tau2f / (dh * tau2f + sig) * fmask * taper, nan=0.0, posinf=0.0, neginf=0.0)
                if low.any():
                    fth[low] = ft_resize(self.classes_h[h][low], b)
                hs.append(self._postprocess(fth, b, center=p.center_classes, shifts=shifts)[0])
            self.classes_h = ft_resize(torch.stack(hs), self.bmax)
        if p.center_classes:  # keep running statistics and particles consistent with re-centred classes
            fyb, fxb = freqs(bm, self.dev)
            ramp_c = torch.exp(1j * TWO_PI / self.box_a * (
                fxb[None, None, :] * shifts[:, 0, None, None] + fyb[None, :, None] * shifts[:, 1, None, None]))
            self.S_num = self.S_num * ramp_c
            if self.S_numh is not None:
                self.S_numh = self.S_numh * ramp_c[None]
            dxy = shifts.cpu()[self.cls[idx_all]]
            a = torch.deg2rad(self.psi[idx_all])
            ca, sa = torch.cos(a), torch.sin(a)
            rot = torch.stack([ca * dxy[:, 0] - sa * dxy[:, 1], sa * dxy[:, 0] + ca * dxy[:, 1]], 1)
            self.offsets[idx_all] = (self.offsets[idx_all] + rot).clamp_(-p.max_offset_a, p.max_offset_a)

        if p.class_prior:
            self.prior = (pop / pop.sum()).clamp_min(1e-4)
            self.prior /= self.prior.sum()
        if p.log_duplicates and getattr(self, "_dups", None):
            top = self._dups[:4]
            self.log(f"[dup] it {it + 1}: most similar pairs (small, large, cross/self, self_s, self_l): "
                     + ", ".join(f"({a + 1},{b_ + 1} {r:.3f} {s0:.2f} {s1:.2f})" for a, b_, r, s0, s1 in top)
                     + f"; {sum(1 for d in self._dups if d[2] >= 1.0)} pairs >= 1")
        reseeded = self._reseed(it, pop, idx_all, b, s["klim"])

        changed = (old_cls != self.cls[idx_all])[seen_before].float().mean().item() if seen_before.any() else 1.0
        self.seen[idx_all] = True
        dt = time.time() - t_start
        info = dict(it=it + 1, res=s["res"], res_rec=s["res_rec"], box=b, n_rot=R, n_trans=T1 * T1, n=n_sub, time=dt,
                    est_res=self.est_res, maxprob=self.maxprob[idx_all].mean().item(), changed=changed,
                    min_pop=int(pop.min().item()), max_pop=int(pop.max().item()), reseeded=reseeded, enc=self._last_enc)
        self.history.append(info)
        self.pop_live = pop.detach().float().cpu()    # class sizes of this iteration (dashboard live view)
        self.log(f"[it {it + 1:3d}] align {s['res']:5.1f}A rec {s['res_rec']:5.1f}A box {b_al:3d}/{b:3d} rot {R:3d} ({360 / R:4.1f}deg) "
                 f"trans {T1}x{T1} n={n_sub:7d} | estRes {self.est_res:5.1f}A <Pmax> {info['maxprob']:.3f} ENC {self._last_enc:4.2f} "
                 f"changed {changed * 100:5.1f}% pop[{info['min_pop']},{info['max_pop']}]"
                 f"{' reseed ' + str(reseeded) if reseeded else ''} | {dt:6.2f}s")

    # ------------------------------------------------------- E/M pass (multi-GPU)
    def _run_pass(self, idx_all, ctx, bn, finish):
        """E-step + M-step accumulation over idx_all, one odd/even half-set after the other: the per-rotation
        accumulators (the largest arrays, ~ K x rotations x box^2) exist for one half only, and finish(h, numer,
        dm2) reduces them before the next half starts. Batches are pulled from a shared queue by one thread per
        GPU (dynamic load balancing across GPUs of different speed); accumulators are summed on the master.
        Returns ([finish(0, ...), finish(1, ...)], sum |x|^2, class populations)."""
        c_dev = {}
        stats = []
        self._bad_logged = False
        out = []
        par = idx_all % 2
        for h in (0, 1):
            g = idx_all[par == h]
            lock = threading.Lock()
            queue = iter([g[i:i + bn] for i in range(0, len(g), bn)])

            def worker(dev):
                with torch.cuda.device(dev):
                    if dev not in c_dev:
                        c_dev[dev] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ctx.items()}
                    c = c_dev[dev]
                    b, M, bh = c["b"], c["M"], c["b"] // 2 + 1
                    if c.get("refine"):  # class-frame sums (direct rotation) + residual power
                        acc = dict(num=torch.zeros((self.p.n_classes, b, b), device=dev),
                                   den=torch.zeros((self.p.n_classes, b, bh), device=dev),
                                   res=torch.zeros((c["b_al"], c["b_al"] // 2 + 1), device=dev))
                    else:
                        acc = dict(numer=torch.zeros((b, M, 2 * bh), device=dev),
                                   dm2=torch.zeros((M, 2 * b * (b + 1)), device=dev))
                    acc.update(sumx2=torch.zeros((c["b_al"], c["b_al"] // 2 + 1), device=dev),
                               pop=torch.zeros(self.p.n_classes, device=dev), enc=torch.zeros((), device=dev),
                               nbad=torch.zeros((), device=dev))
                    while True:
                        with lock:
                            idx = next(queue, None)
                        if idx is None:
                            break
                        self._batch(dev, c, idx, acc)
                    return acc

            if len(self.devs) == 1:
                accs = [worker(self.dev)]
            else:
                with ThreadPoolExecutor(len(self.devs)) as ex:
                    accs = list(ex.map(worker, self.devs))
            tot = {}
            for k in accs[0]:  # sum on the master, freeing each GPU's copy as it is added
                tot[k] = accs[0][k].to(self.dev)
                for a in accs[1:]:
                    tot[k] += a[k].to(self.dev)
                    a[k] = None
            del accs
            out.append(finish(h, tot))
            stats.append(tot)
        self._last_enc = float(stats[0]["enc"] + stats[1]["enc"]) / max(len(idx_all), 1)
        self._last_nbad = int(stats[0]["nbad"] + stats[1]["nbad"])
        return out, stats[0]["sumx2"] + stats[1]["sumx2"], stats[0]["pop"] + stats[1]["pop"]

    def _batch_budget(self, b, b_al, M, T1, H) -> int:
        """Particles per batch that fit the GPU memory that is actually free (the slowest-to-fit GPU decides).
        Fixed per pass: one half-set's per-rotation accumulators and the rotated references; per particle: the
        posterior tensor (T x M x T, plus a working copy) and a few spectra. Windows/WSL drivers do not fail when
        memory runs out but swap to system RAM (several times slower), so the budget leaves headroom."""
        bh, bh_a = b // 2 + 1, b_al // 2 + 1
        fixed = 4 * (b * M * 2 * bh + M * 2 * b * (b + 1))         # numer + dm2 (one half-set)
        fixed += H * M * b_al * bh_a * 12                          # rotated references (complex) + |ref|^2
        per = 3 * 4 * M * T1 * T1 + 40 * (2 * b) * (b + 1) * 8     # posterior + copies, spectra / ramps / CTFs
        free = []
        for d in self.devs:
            if d.type != "cuda":
                return int(np.clip(0.4e9 / (M * T1 * T1 * 4), 4, 1024))
            release_cache()  # cached blocks of the previous (smaller-box) iteration
            with torch.cuda.device(d):
                release_cache()
            free.append(gpu_free(d))
        avail = 0.75 * min(free) - fixed
        bn = int(np.clip(min(avail / per, 0.4e9 / (M * T1 * T1 * 4)), 4, 1024))
        need = fixed + bn * per
        if need > 0.9 * min(free) and not getattr(self, "_mem_warned", False):
            self._mem_warned = True
            self.log(f"[mem] warning: this iteration needs ~{need / 1e9:.1f} GB but only {min(free) / 1e9:.1f} GB "
                     f"of GPU memory is free; expect a slowdown (or an out-of-memory error). Fewer classes, a lower "
                     f"max resolution or more GPUs reduce it.")
        return bn

    def _precenter(self, x_a, ctf_a, blob, b_al):
        """Offsets (n, 2) (x, y) [A] that best match a centred, rotationally symmetric blob within precenter_a,
        on a 2x finer grid. irfft2 gives sum_f z exp(+2 pi i f.d / 2b), the E-step ramp exp(-2 pi i f.t / box_a),
        so t = -d * pixel."""
        b2 = 2 * b_al
        cc = torch.fft.irfft2(ft_resize(torch.conj(x_a) * ctf_a * blob, b2), s=(b2, b2)).flatten(1)
        t = -torch.fft.fftfreq(b2, d=1.0 / b2, device=x_a.device) * (self.box_a / b2)
        inside = (t[:, None] ** 2 + t[None, :] ** 2 <= self.p.precenter_a ** 2).flatten()
        a = cc.masked_fill(~inside[None], float("-inf")).argmax(1)
        return torch.stack([t[a % b2], t[a // b2]], 1)

    def _report_bad(self, idx, badn, x):
        if self._bad_logged:   # once per pass; the total is reported by iterate()
            return
        self._bad_logged = True
        bad = badn.nonzero()[:, 0]
        fin = torch.isfinite(x[bad]).flatten(1).all(1)
        self.log(f"[warn] {len(bad)} particle(s) with non-finite likelihood dropped from this pass "
                 f"(host idx {idx[bad.cpu()][:5].tolist()}; image finite: {fin[:5].tolist()})")

    def _batch(self, dev, c, idx, acc):
        p, K = self.p, self.p.n_classes
        b, b_al, R, M, T1, mc = c["b"], c["b_al"], c["R"], c["M"], c["T1"], c["mc"]
        bh, bh_a = b // 2 + 1, b_al // 2 + 1
        hset = int(idx[0]) % 2 if c["refs"].shape[0] == 2 else 0       # half-pure batches under gold standard
        refs, r2w = c["refs"][hset], c["r2w"][hset]
        w8, tg, Ex, EyT, E2 = c["w8"], c["tg"], c["Ex"], c["EyT"], c["E2"]
        fy, fx = freqs(b, dev)
        fy_a, fx_a = freqs(b_al, dev)
        fy2, fx2 = freqs(2 * b, dev)
        ph = TWO_PI / self.box_a
        n = len(idx)
        x = self._fetch(idx, dev, b)                       # reconstruction box
        x_a = ft_resize(x, b_al)                           # alignment box
        ctfp = self.ctf_p[idx].to(dev)
        ctf = ctf_2d(ctfp, self.optics, fy, fx, self.box_a)
        ctf_a = ctf_2d(ctfp, self.optics, fy_a, fx_a, self.box_a)
        t0 = self.offsets[idx].to(dev)  # (n, 2) (x, y)
        new = ~self.seen[idx]
        if "blob" in c and new.any():
            t0[new.to(dev)] = self._precenter(x_a[new.to(dev)], ctf_a[new.to(dev)], c["blob"], b_al)

        def ramp(fy_, fx_):
            return torch.exp(-1j * ph * (fy_[None, :, None] * t0[:, 1, None, None]
                                         + fx_[None, None, :] * t0[:, 0, None, None]))
        ramp0, ramp0_a = ramp(fy, fx), ramp(fy_a, fx_a)
        q = torch.conj(x_a) * ctf_a * w8 * ramp0_a                                   # (n, b_al, bh_a)
        # ---- E-step: CC[ty, m, n, tx] = Re sum_f q R_m Ex Ey  (two fp32 complex GEMMs)
        z = (q[..., None] * Ex[None, None]).permute(1, 2, 0, 3).reshape(b_al, bh_a, n * T1)
        ll = torch.empty((T1, M, n, T1), device=dev)
        for m0 in range(0, M, mc):
            m1 = min(M, m0 + mc)
            a = torch.matmul(refs[:, m0:m1], z)                                      # (b_al, mc, n*T1)
            ll[:, m0:m1] = torch.matmul(EyT, a.view(b_al, -1)).real.view(T1, m1 - m0, n, T1)
            del a
        d = ((ctf_a * ctf_a) * w8).reshape(n, -1) @ r2w.T                       # (n, M)
        if p.particle_scale:
            # each hypothesis at its own best amplitude a = CC/d (stain thickness / ice varies the contrast):
            # max_a 2a CC - a^2 d = CC^2 / d, identical to 2 CC - d when a = 1
            ll.clamp_(min=0).pow_(2).div_(d.T.clamp_min(1e-30)[None, :, :, None]).add_(c["logprior"][None, :, None, None])
        else:
            ll.mul_(2).sub_(d.T[None, :, :, None]).add_(c["logprior"][None, :, None, None])
        if c.get("lock_class"):  # polishing: refine poses inside each particle's converged class only
            cur = self.cls[idx].to(dev)
            same = (torch.arange(M, device=dev)[:, None] // R) == cur[None, :]       # (M, n)
            ll.masked_fill_(~same[None, :, :, None], float("-inf"))
        if c["temp"] != 1.0:
            ll.div_(c["temp"])
        if p.hard_pose:
            # cryoSPARC-style "force max over poses": soft over classes, each class takes its best pose only
            llv = ll.view(T1, K, R, n, T1).permute(1, 3, 2, 0, 4).reshape(K, n, R * T1 * T1)
            mx, arg = llv.max(-1)                                                      # (K, n)
            lse = torch.logsumexp(mx, dim=0)                                           # (n,)
            badn = ~torch.isfinite(lse)
            if badn.any():  # one non-finite particle would make every class NaN: park it in class 0, weight 0
                self._report_bad(idx, badn, x)
                acc["nbad"] += badn.sum()
                mx[:, badn] = float("-inf")
                mx[0, badn] = 0.0
                lse = torch.where(badn, torch.zeros_like(lse), lse)
            if c.get("refine"):
                del llv
                return self._batch_refine(dev, c, idx, acc, x, x_a, ctf, ctf_a, ctfp, t0, mx, arg, badn, hset)
            cpost = torch.exp(mx - lse[None])                                          # class posterior (K, n)
            cpost[:, badn] = 0.0
            r_, rest = arg // (T1 * T1), arg % (T1 * T1)
            ty_, tx_ = rest // T1, rest % T1
            kk = torch.arange(K, device=dev)[:, None].expand(K, n)
            nn = torch.arange(n, device=dev)[None, :].expand(K, n)
            ll.zero_()
            ll[ty_, kk * R + r_, nn, tx_] = cpost
            wts = ll
            sw, kord = torch.sort(cpost.T, dim=1, descending=True)                     # (n, K)
            kb = kord[:, 0]
            ar = torch.arange(n, device=dev)
            best = ((kb * R + r_[kb, ar]) * T1 + ty_[kb, ar]) * T1 + tx_[kb, ar]
        else:
            lse = torch.logsumexp(ll, dim=(0, 1, 3))                                  # (n,)
            badn = ~torch.isfinite(lse)
            if badn.any():                                                            # drop broken particles
                self._report_bad(idx, badn, x)
                acc["nbad"] += badn.sum()
                ll[:, :, badn, :] = float("-inf")
                ll[0, 0, badn, 0] = 0.0
                lse = torch.where(badn, torch.zeros_like(lse), lse)
            flat = ll.permute(2, 1, 0, 3).reshape(n, -1)                              # (n, M*Ty*Tx)
            top, best = torch.topk(flat, min(256, flat.shape[1]), dim=1)
            del flat
            wts = ll.sub_(lse[None, None, :, None]).exp_()                            # posterior, in place
            best = best[:, 0]
            sw = torch.exp(top - lse[:, None])

        if badn.any():  # a broken image must not reach the class sums
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # ---- per-particle bookkeeping (best hypothesis), written back to host state
        bm_, bt = best // (T1 * T1), best % (T1 * T1)
        x2 = x_a.abs() ** 2
        self.cls[idx] = (bm_ // R).cpu()
        self.psi[idx] = c["psi"][bm_ % R].cpu()
        self.offsets[idx] = (t0 + torch.stack([tg[bt % T1], tg[bt // T1]], 1)).clamp_(
            -p.max_offset_a, p.max_offset_a).cpu()
        self.maxprob[idx] = sw[:, 0].cpu()
        self.nsig[idx] = ((torch.cumsum(sw, 1) < 0.999).sum(1) + 1).cpu()
        self.loglik[idx] = (lse - (x2 * w8).reshape(n, -1).sum(1)).cpu()

        # ---- M-step accumulation (exact marginalisation, transposed GEMMs).
        # Averaging only, so fp16 tensor cores with fp32 accumulation (rel. error ~5e-4 << noise).
        y = ctf * x * torch.conj(ramp0) * self.store_scale                                  # aligned to t0, O(1) scale
        wnm = wts.sum((0, 3))                                                         # (M, n)
        cp = wnm.view(K, R, n).sum(1)                                                 # class posterior (K, n)
        self.cpost[idx] = cp.T.to(torch.float16).cpu()
        acc["enc"] += (1.0 / (cp * cp).sum(0).clamp_min(1e-12)).sum()                 # effective no. of classes
        c2all = ctf_2d(ctfp, self.optics, fy2, fx2, 2 * self.box_a) ** 2              # (n, 2b, b+1), 2x oversampled
        # batches hold one odd/even half-set only (FRC regulariser); _run_pass keeps the halves apart
        g = (y.permute(1, 0, 2)[:, :, None, :] * torch.conj(c["Exr"]).T[None, None]).reshape(b, n * T1, bh)
        g1 = torch.cat([g.real, g.imag], -1).to(self._mdt)                            # (b, nT, 2bh)
        g2 = torch.cat([-g.imag, g.real], -1).to(self._mdt)
        del g
        numer = acc["numer"]
        for m0 in range(0, M, mc):
            m1 = min(M, m0 + mc)
            h = (E2 @ wts[:, m0:m1].reshape(T1, -1).to(self._mdt)).view(2, b, m1 - m0, n * T1)  # sum over ty
            numer[:, m0:m1] += (torch.bmm(h[0], g1) + torch.bmm(h[1], g2)).float() / self.store_scale
            del h
        acc["dm2"] += (wnm.to(self._mdt) @ c2all.reshape(n, -1).to(self._mdt)).float()
        acc["sumx2"] += x2.sum(0)
        acc["pop"] += wnm.view(K, R, n).sum((1, 2))


    def _batch_refine(self, dev, c, idx, acc, x, x_a, ctf, ctf_a, ctfp, t0, mx, arg, badn, hset, chunk=96):
        """Local fine search around the coarse optimum of each particle's best classes, then the M-step by direct
        rotation into the class frame at the fine pose (hard pose: soft over classes, one pose per class)."""
        p, K = self.p, self.p.n_classes
        b, b_al, T1, C = c["b"], c["b_al"], c["T1"], c["C"]
        bh, bh_a = b // 2 + 1, b_al // 2 + 1
        tg, dpsi, sg, w8 = c["tg"], c["dpsi"], c["sg"], c["w8"]
        J, S = len(dpsi), len(sg)
        ph = TWO_PI / self.box_a
        fy, fx = freqs(b, dev)
        fy_a, fx_a = freqs(b_al, dev)
        R = c["R"]
        r_, rest = arg // (T1 * T1), arg % (T1 * T1)
        ty_, tx_ = rest // T1, rest % T1
        n = len(idx)
        up = c["up_al"][min(hset, len(c["up_al"]) - 1)]
        cp_all = torch.zeros((K, n), device=dev)
        best_cls = torch.zeros(n, dtype=torch.long, device=dev)
        best_psi = torch.zeros(n, device=dev)
        best_t = torch.zeros((n, 2), device=dev)
        lse_all = torch.zeros(n, device=dev)
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            m = i1 - i0
            mxc = mx[:, i0:i1]
            topv, topk = mxc.topk(C, dim=0)                                             # (C, m)
            kp = topk.T.reshape(-1)                                                    # pairs, particle-major
            ip = torch.arange(m, device=dev).repeat_interleave(C)                      # chunk-local particle
            g = ip + i0
            P = len(kp)
            tc = t0[g] + torch.stack([tg[tx_[kp, g]], tg[ty_[kp, g]]], 1)              # coarse best shift (x, y)
            psi_c = c["psi"][r_[kp, g]]
            # references of class k at psi_c + dpsi (built per pair, like _make_refs)
            ang = (psi_c[:, None] + dpsi[None]).reshape(-1)
            grid = rot_grid(ang, b_al, 2 * b_al).view(P, J * b_al, b_al, 2)
            img = F.grid_sample(up[kp][:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            Rf = (torch.fft.rfft2(img.view(P, J, b_al, b_al), norm="forward") * c["kmask_a"]).to(torch.complex64)
            del img, grid
            xa, ca = x_a[g], ctf_a[g]
            ramp_c = torch.exp(-1j * ph * (fy_a[None, :, None] * tc[:, 1, None, None] + fx_a[None, None, :] * tc[:, 0, None, None]))
            q = torch.conj(xa) * ca * w8 * ramp_c                                     # (P, b_al, bh_a)
            cc = (c["EyfT"] @ ((q[:, None] * Rf) @ c["Exf"])).real                      # (P, J, S[y], S[x])
            d = ((ca * ca * w8)[:, None] * Rf.abs() ** 2).sum((-2, -1))                # (P, J)
            ll = 2 * cc - d[..., None, None] + c["logprior_k"][kp][:, None, None, None]
            if c["temp"] != 1.0:
                ll = ll / c["temp"]
            val, am = ll.view(P, -1).max(1)
            jb, sy, sx = am // (S * S), (am // S) % S, am % S
            val = torch.where(torch.isfinite(val), torch.maximum(val, topv.T.reshape(-1)), topv.T.reshape(-1))
            mx2 = mxc.clone()
            mx2[topk, torch.arange(m, device=dev)[None].expand(C, m)] = val.view(m, C).T
            lse = torch.logsumexp(mx2, 0)
            cpost = torch.exp(mx2 - lse[None])
            bad = badn[i0:i1]
            cpost[:, bad] = 0.0
            cp_all[:, i0:i1] = cpost
            lse_all[i0:i1] = torch.where(bad, torch.zeros_like(lse), lse)
            wp = cpost[kp, ip]
            wp = wp / wp.view(m, C).sum(1).clamp_min(1e-12)[ip]                      # weights of the refined classes
            wp = torch.where(bad[ip], torch.zeros_like(wp), wp)
            psi_f = psi_c + dpsi[jb]
            t_f = tc + torch.stack([sg[sx], sg[sy]], 1)
            bi = wp.view(m, C).argmax(1)
            sel = torch.arange(m, device=dev) * C + bi
            best_cls[i0:i1], best_psi[i0:i1], best_t[i0:i1] = kp[sel], psi_f[sel], t_f[sel]
            # residual power of the fitted model (noise estimate), alignment box, particle frame
            model = ca * Rf[torch.arange(P, device=dev), jb] * torch.exp(
                -1j * ph * (fy_a[None, :, None] * t_f[:, 1, None, None] + fx_a[None, None, :] * t_f[:, 0, None, None]))
            acc["res"] += (wp[:, None, None] * (model.abs() ** 2 - 2 * (torch.conj(xa) * model).real)).sum(0)
            del Rf, q, cc, model
            # M-step: CTF-weighted particle shifted by -t and rotated by -psi into the class frame
            xr = torch.nan_to_num(x[g]) if bad.any() else x[g]
            y = ctf[g] * xr * torch.exp(1j * ph * (fy[None, :, None] * t_f[:, 1, None, None] + fx[None, None, :] * t_f[:, 0, None, None]))
            rim = F.grid_sample(upsample2(y, b)[:, None], rot_grid(-psi_f, b, 2 * b), mode="bilinear",
                                align_corners=False)[:, 0]                             # (P, b, b)
            acc["num"].index_add_(0, kp, rim * wp[:, None, None])
            c2 = ctf_2d(ctfp[g], self.optics, fy, fx, self.box_a, ang_offset_deg=psi_f) ** 2
            acc["den"].index_add_(0, kp, c2 * wp[:, None, None])
            acc["pop"].index_add_(0, kp, wp)
            del y, rim, c2
        # ---- per-particle bookkeeping
        x2 = x_a.abs() ** 2
        sw, _ = torch.sort(cp_all.T, dim=1, descending=True)
        self.cls[idx] = best_cls.cpu()
        self.psi[idx] = torch.remainder(best_psi, 360.0).cpu()
        self.offsets[idx] = best_t.clamp_(-p.max_offset_a, p.max_offset_a).cpu()
        self.maxprob[idx] = sw[:, 0].cpu()
        self.nsig[idx] = ((torch.cumsum(sw, 1) < 0.999).sum(1) + 1).cpu()
        self.loglik[idx] = (lse_all - (x2 * w8).reshape(n, -1).sum(1)).cpu()
        self.cpost[idx] = cp_all.T.to(torch.float16).cpu()
        acc["enc"] += (1.0 / (cp_all * cp_all).sum(0).clamp_min(1e-12))[~badn].sum()
        acc["sumx2"] += x2.sum(0)

    # --------------------------------------------------------------------
    def _rotate_back(self, numer, dm2, psi, b):
        """Sum the per-rotation accumulators in the class frame (rotate by -psi)."""
        K, R = self.p.n_classes, len(psi)
        bh = b // 2 + 1
        num_real = torch.zeros((K, b, b), device=self.dev)
        den_full = torch.zeros((K, b, b), device=self.dev)
        numer = numer.permute(1, 0, 2).reshape(K, R, b, bh)
        dm2 = dm2.view(K, R, 2 * b, b + 1)
        for r0 in range(0, R, 16):
            r1 = min(R, r0 + 16)
            g = rot_grid(-psi[r0:r1], b, 2 * b)
            up = upsample2(numer[:, r0:r1].transpose(0, 1), b)                       # (r, K, 2b, 2b)
            num_real += F.grid_sample(up, g, mode="bilinear", align_corners=False).sum(0)
            dfull = half_to_full_centered(dm2[:, r0:r1].transpose(0, 1), 2 * b)       # (r, K, 2b, 2b)
            den_full += F.grid_sample(dfull, g, mode="bilinear", align_corners=False).sum(0)
        num_k = torch.fft.rfft2(num_real, norm="forward")
        den_k = torch.fft.ifftshift(den_full, dim=(-2, -1))[..., :bh]
        return num_k, den_k

    def _shape_mask(self, ft, b, apix, circle):
        """Per-class soft envelope of the density (20 A low-pass, > 15 % of max, dilated ~12 A, soft 8 A edge).
        Keeps the solvent around compact views flat, so noise cannot build a pattern there for particles to
        align to (the 'honeycomb' around top views)."""
        fy, fx = freqs(b, self.dev)
        k = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / self.box_a
        low = torch.fft.irfft2(ft * torch.exp(-0.5 * (k * 20.0) ** 2), s=(b, b), norm="forward") * circle
        core = (low > self.p.solvent_thresh * low.amax((1, 2), keepdim=True)).float()
        # dilate + soften with a Gaussian blur of the binary core (separable, in Fourier space)
        sig = max(self.p.solvent_dilate_a / apix, 1.0)
        g = torch.exp(-0.5 * (k * self.box_a * 2 * math.pi * sig / b) ** 2)
        blur = torch.fft.irfft2(torch.fft.rfft2(core) * g, s=(b, b))
        env = (blur / 0.25).clamp(0, 1)            # anything within ~1 sigma of the core is kept
        env = 0.5 - 0.5 * torch.cos(math.pi * env)
        empty = core.sum((1, 2)) < 4                # degenerate classes: keep the circular mask only
        env[empty] = 1.0
        return env * circle

    def _postprocess(self, ft, b, center: bool, shifts=None):
        """Mask classes in real space and (optionally) shift their centre of mass to the box centre.
        Returns (spectra, applied shifts d in A as (x, y): class_new = shift_{-d}(class_old))."""
        img = torch.fft.irfft2(ft, s=(b, b), norm="forward")
        apix = self.box_a / b
        mask = soft_circle(b, self.radius_a / apix, 3.0, self.dev)
        img = img * mask
        if self.p.solvent_flatten:
            img = img * self._shape_mask(ft, b, apix, mask)
        K = img.shape[0]
        given = shifts is not None
        if not given:
            shifts = torch.zeros((K, 2), device=self.dev)
        if center and not given:
            r = (torch.arange(b, device=self.dev, dtype=torch.float32) - b // 2) * apix
            src = img
            if self.p.recenter_detrend:
                # centre of mass of the compact density only: a class of a background gradient (ice / carbon edge)
                # has its centre of mass far off-centre and walks out of the box iteration after iteration, until
                # its particles' offsets hit the cap and it dies. Remove the best-fit plane inside the mask first.
                yy, xx = torch.meshgrid(r, r, indexing="ij")
                # The plane is fitted on the outer ring of the mask (0.75-1 x its radius: background, or the
                # gradient): fitted over the whole mask it tilts towards the heavier side of an asymmetric particle
                # and moves that particle's own centre (synthetic ab initio test: orientations 0.09-0.45 vs >= 0.8)
                basis = torch.stack([torch.ones_like(xx), xx, yy], -1).reshape(-1, 3)          # (b*b, 3)
                ring = (mask * (torch.sqrt(xx * xx + yy * yy) >= 0.75 * self.radius_a)).reshape(-1, 1)
                coef = torch.linalg.lstsq(basis * ring, (img.reshape(K, -1) * ring.T).T).solution   # (3, K)
                src = (img - (basis @ coef).T.reshape(K, b, b)) * mask
            thr = self.p.recenter_thresh * src.amax((1, 2), keepdim=True)
            pos = torch.where(src > thr, src, torch.zeros_like(src))
            tot = pos.sum((1, 2)).clamp_min(1e-12)
            shifts = torch.stack([(pos.sum(1) * r).sum(1) / tot, (pos.sum(2) * r).sum(1) / tot], 1)
        ft = torch.fft.rfft2(img, norm="forward")
        if center:
            fy, fx = freqs(b, self.dev)
            ph = TWO_PI / self.box_a
            ft = ft * torch.exp(1j * ph * (fx[None, None, :] * shifts[:, 0, None, None]
                                           + fy[None, :, None] * shifts[:, 1, None, None]))
            img2 = torch.fft.irfft2(ft, s=(b, b), norm="forward") * mask
            ft = torch.fft.rfft2(img2, norm="forward")
        return ft, shifts

    def _duplicates(self, A0, A1, b, klim, pop, n_rot=120):
        """Pairs of classes the data cannot tell apart: class k's odd half agrees with class j's even half (best
        in-plane rotation) at least as well as with its own even half. Band = alignment limit, circular mask.
        Returns [(smaller, larger, ratio)] sorted by ratio; ratio = cross / min(self_k, self_j)."""
        K = A0.shape[0]
        bd = min(b, int(math.ceil(klim)) * 2 + 4)
        fy, fx = freqs(bd, self.dev)
        kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        band = ((kr <= klim) & (kr > 0)).float()
        mask = soft_circle(bd, self.radius_a / (self.box_a / bd), 2.0, self.dev)

        def img(A):
            x = torch.fft.irfft2(ft_resize(A, bd) * band, s=(bd, bd), norm="forward") * mask
            x = x - (x * mask).sum((1, 2), keepdim=True) / mask.sum()
            x = x * mask
            return x / x.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None]
        I0, I1 = img(A0), img(A1)
        psi = torch.arange(n_rot, device=self.dev) * (360.0 / n_rot)
        up1 = upsample2(torch.fft.rfft2(I1, norm="forward"), bd)                         # (K, 2bd, 2bd)
        cross = torch.full((K, K), -1.0, device=self.dev)
        for r0 in range(0, n_rot, 24):
            g = rot_grid(psi[r0:r0 + 24], bd, 2 * bd)
            rot = F.grid_sample(up1[None].expand(len(g), -1, -1, -1), g, mode="bilinear", align_corners=False)  # (r, K, bd, bd)
            cross = torch.maximum(cross, torch.einsum("kd,rjd->rkj", I0.flatten(1), rot.flatten(2)).amax(0))
        selfc = cross.diagonal().clamp_min(1e-6)
        cs = 0.5 * (cross + cross.T)
        ratio = cs / torch.minimum(selfc[:, None], selfc[None, :])
        ratio.fill_diagonal_(-1)
        out = []
        iu = torch.triu_indices(K, K, 1, device=self.dev)
        for k, j in iu.T.tolist():
            if pop[k] > 0 and pop[j] > 0 and selfc[k] > 0.2 and selfc[j] > 0.2:   # classes with signal only
                small, big = (k, j) if pop[k] < pop[j] else (j, k)
                out.append((small, big, float(ratio[k, j]), float(selfc[k]), float(selfc[j])))
        out.sort(key=lambda t: -t[2])
        return out

    def _reseed(self, it, pop, idx_all=None, b=None, klim=None):
        p = self.p
        last = min(int(p.reseed_until * p.n_iter), p.n_iter - p.n_full_iter)
        if p.reseed_full:  # also in full passes, except the last two (they must converge, not restart)
            last = max(last, p.n_iter - 2)
        if it >= last or p.reseed_frac <= 0:
            return 0
        thr = p.reseed_frac * pop.sum() / p.n_classes
        small = torch.nonzero(pop < thr).flatten().tolist()
        n = 0
        for k in small:
            big = int(torch.argmax(pop))
            split = None
            if p.reseed_mode == "split" and idx_all is not None:
                split = self._split_class(big, idx_all, b, klim)
            if split is not None:
                self.classes[k], self.classes[big] = split
                if self.classes_h is not None:
                    self.classes_h[:, k], self.classes_h[:, big] = self.classes[k], self.classes[big]
            else:
                noise = torch.randn(self.classes[big].shape, device=self.dev, dtype=torch.complex64)
                amp = self.classes[big].abs().mean()
                self.classes[k] = self.classes[big] + 0.3 * amp * noise * (self.classes[big] != 0)
                if self.classes_h is not None:  # same split in both half-sets keeps class k matched between them
                    self.classes_h[:, k] = self.classes_h[:, big] + 0.3 * amp * noise * (self.classes_h[:, big] != 0)
            if self.classes_h is not None:
                for S in (self.S_numh, self.S_denh):
                    S[:, k] = S[:, big] = 0.5 * S[:, big]
            self.prior[k] = self.prior[big] = 0.5 * self.prior[big]
            pop[k] = pop[big] = 0.5 * pop[big]
            for S in (self.S_num, self.S_den, self.S_pop):
                S[k] = S[big] = 0.5 * S[big]
            n += 1
        if n:
            self.prior /= self.prior.sum()
        return n

    def _split_class(self, big, idx_all, b, klim, n_max=1024):
        """Two classes from the members of class `big` (particles of the last pass), split along the first
        principal component of their aligned, phase-flipped, low-passed images (the direction of the largest real
        variation inside the class: a different view, a ligand, a broken particle). Each side is reconstructed with
        the Wiener filter of the parent class. A noise-perturbed copy of the parent (the alternative) explains the
        same particles as the parent; the data cannot tell the two apart and one of them dies again once reseeding
        stops. Returns (class_k, class_big) spectra at bmax, or None if the class has too few members."""
        members = idx_all[self.cls[idx_all] == big]
        if len(members) < 32 or getattr(self, "_wien", None) is None or self._wien["b"] != b:
            return None
        if len(members) > n_max:
            members = members[torch.randperm(len(members), generator=self.rng)[:n_max]]
        dev, ph = self.dev, TWO_PI / self.box_a
        fy, fx = freqs(b, dev)
        kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        lowpass = torch.exp(-0.5 * (kr / max(klim, 1.0)) ** 2)
        al, pf, den = [], [], []
        for i in range(0, len(members), 512):
            j = members[i:i + 512]
            ps, t, ctfp = self.psi[j].to(dev), self.offsets[j].to(dev), self.ctf_p[j].to(dev)
            c = ctf_2d(ctfp, self.optics, fy, fx, self.box_a)
            y = self._fetch(j, dev, b) * torch.exp(1j * ph * (fx[None, None, :] * t[:, 0, None, None]
                                                                + fy[None, :, None] * t[:, 1, None, None]))
            g = rot_grid(-ps, b, 2 * b)
            al.append(F.grid_sample(upsample2(c * y, b)[:, None], g, align_corners=False)[:, 0])
            pf.append(F.grid_sample(upsample2(torch.sign(c) * y * lowpass, b)[:, None], g, align_corners=False)[:, 0])
            den.append(ctf_2d(ctfp, self.optics, fy, fx, self.box_a, ang_offset_deg=ps) ** 2)
        al, pf, den = torch.cat(al), torch.cat(pf), torch.cat(den)
        mask = soft_circle(b, self.radius_a / (self.box_a / b), 3.0, dev)
        X = (pf * mask).flatten(1)
        X = X - X.mean(0)
        _, _, V = torch.pca_lowrank(X, q=1, center=False, niter=4)
        proj = X @ V[:, 0]
        side = proj > proj.median()
        if side.sum() < 8 or (~side).sum() < 8:
            return None
        w = self._wien
        tau2f = w["tau2f"][big]
        out = []
        for m in (~side, side):
            num = torch.fft.rfft2(al[m].sum(0), norm="forward")
            dn = den[m].sum(0)
            ft = num * tau2f / (dn * tau2f + w["sig"]) * w["support"]
            out.append(ft_resize(self._postprocess(ft[None], b, center=False)[0][0], self.bmax))
        return out[0], out[1]

    # --------------------------------------------------------------------
    def run(self, active=None, n_classes=None, seed=None, resume=False):
        """Classify the particles `active` (None = all) into n_classes (None = Params value).
        stop_requested (set from another thread) ends the run after the current iteration (self.stopped = True);
        run(resume=True) then continues the same run where it stopped, without re-initialising."""
        p = self.p
        t0 = time.time()
        if resume and self.history:
            it0 = len(self.history)
            n_iter0, n_full0 = self._run_plan
            self.log(f"[continue] from iteration {it0 + 1}")
        else:
            if n_classes is not None:
                p.n_classes = n_classes
            if seed is not None:
                self.rng.manual_seed(seed)
            self._set_active(active)
            self._elapsed = 0.0
            self.init_classes()
            it0 = 0
            n_iter0, n_full0 = p.n_iter, p.n_full_iter
            self._run_plan = (n_iter0, n_full0)
        # iterations beyond the plan (convergence full passes) are counted in the plan when resuming
        p.n_iter = max(n_iter0, it0)
        p.n_full_iter = n_full0 + max(0, it0 - n_iter0)
        self.stopped = False
        for it in range(it0, p.n_iter):
            if self.stop_requested:
                break
            self.iterate(it)
        # big data sets: one online-EM visit per particle is not enough, so keep doing full passes until the
        # assignments settle (cryoSPARC users typically run 5-10 full iterations on 0.5-1M particles)
        while (not self.stop_requested and self._auto_full and p.batch_per_class > 0
               and p.n_full_iter < p.max_full_iter and self.history and self.history[-1]["changed"] > p.full_converge):
            p.n_iter += 1
            p.n_full_iter += 1
            self.log(f"[converge] {self.history[-1]['changed'] * 100:.0f}% changed class -> full pass {p.n_full_iter}")
            self.iterate(p.n_iter - 1)
        n_done = len(self.history)
        self.stopped = bool(self.stop_requested)
        p.n_iter, p.n_full_iter = n_iter0, n_full0
        for d in self.devs:
            torch.cuda.synchronize(d)
        self._elapsed = getattr(self, "_elapsed", 0.0) + time.time() - t0
        self.total_time = self._elapsed
        if self.stopped:
            unseen = int((~self.seen[self.active]).sum())
            self.log(f"[stop] stopped after {n_done} iterations ({self.total_time:.1f}s)"
                     + (f"; {unseen} particles not classified yet" if unseen else ""))
        else:
            self.log(f"[done] {n_done} iterations in {self.total_time:.1f}s")

    def average(self, idx, labels, n_groups, psi=None, offsets=None, wiener=0.1, batch=512) -> np.ndarray:
        """CTF-Wiener averages of particles `idx` grouped by `labels` (0..n_groups-1), using the given
        in-plane angles / shifts (default: the stored ones). Any subset, no re-classification."""
        dev, bm = self.dev, self.bmax
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        lab = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        psi = self.psi[idx] if psi is None else torch.as_tensor(np.asarray(psi), dtype=torch.float32)
        off = self.offsets[idx] if offsets is None else torch.as_tensor(np.asarray(offsets), dtype=torch.float32)
        self._wait_loaded(self.N)
        fy, fx = freqs(bm, dev)
        ph = TWO_PI / self.box_a
        num = torch.zeros((n_groups, bm, bm), device=dev)
        den = torch.zeros((n_groups, bm, bm // 2 + 1), device=dev)
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]
            ps, t, g = psi[i:i + batch].to(dev), off[i:i + batch].to(dev), lab[i:i + batch].to(dev)
            ctfp = self.ctf_p[j].to(dev)
            c = ctf_2d(ctfp, self.optics, fy, fx, self.box_a)
            y = c * self._fetch(j, dev, bm) * torch.exp(1j * ph * (fx[None, None, :] * t[:, 0, None, None]
                                                                    + fy[None, :, None] * t[:, 1, None, None]))
            al = F.grid_sample(upsample2(y, bm)[:, None], rot_grid(-ps, bm, 2 * bm), align_corners=False)[:, 0]
            num.index_add_(0, g, al)
            den.index_add_(0, g, ctf_2d(ctfp, self.optics, fy, fx, self.box_a, ang_offset_deg=ps) ** 2)
        ft = torch.fft.rfft2(num, norm="forward") / (den + wiener * den.mean((1, 2), keepdim=True).clamp_min(1e-12))
        img = torch.fft.irfft2(ft, s=(bm, bm), norm="forward")
        img = img * soft_circle(bm, self.radius_a / self.apix_max, 3.0, dev)
        return img.cpu().numpy().astype(np.float32)

    def class_images(self) -> np.ndarray:
        img = torch.fft.irfft2(self.classes, s=(self.bmax, self.bmax), norm="forward")
        return img.cpu().numpy().astype(np.float32)
