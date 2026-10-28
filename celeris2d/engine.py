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
    init_res_a: float = 30.0        # starting resolution limit (frequency marching)
    march_frac: float = 0.6         # fraction of iterations used to reach max_res
    psi_step_min: float = 2.0
    psi_factor: float = 1.0         # in-plane step = psi_factor * degrees(res / diameter)
    trans_div: float = 4.0          # translation step = res / trans_div
    psi_step_max: float = 15.0
    max_shift_a: float = 16.0       # local translation search radius around previous offset
    max_trans_half: int = 4         # cap on translation grid half-width (4 -> 9 x 9 positions)
    max_offset_a: float = -1.0      # cap on accumulated offset (-1 = diameter / 4)
    tau2_fudge: float = 2.0
    batch_per_class: int = 500      # online-EM subset size per class for the early iterations (cryoSPARC-like)
    n_full_iter: int = -1           # final iterations over all particles (-1 = auto from N)
    online_decay: float = 0.0       # online EM: running-statistics decay (0 = fresh batch each iter; 0.8 hurt)
    anneal_t0: float = 1.0          # likelihood temperature at iter 1 (1 = no annealing)
    reseed_frac: float = 0.25       # reseed classes holding < frac * N/K particles (early iters)
    center_classes: bool = True
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
    gold_standard: bool = True      # each half-set is aligned only to classes built from its own half, so the
                                    # half-set FRC (Wiener filter) cannot mistake a shared noise pattern for signal
    sigma_init_factor: float = 1.0  # >1: noise inflated at the start, annealed to 1 (cryoSPARC-style); off: hurt LoG separation
    sigma_anneal_start: int = 2
    sigma_anneal_iters: int = 15
    sigma_decay: float = 0.7        # online EM: running noise estimate = decay * old + (1 - decay) * batch
    recenter_thresh: float = 0.2    # re-centre on the centre of mass of density above this fraction of max
    solvent_flatten: bool = True    # mask each class by a soft envelope of its own density (flat solvent)
    solvent_thresh: float = 0.08    # envelope core: 20 A low-passed density above this fraction of max
    solvent_dilate_a: float = 20.0  # envelope extends this far beyond the core (A)
    notch_thresh: float = 2.5       # zero Fourier pixels whose mean power is > this x their shell mean
                                    # (fixed-frame detector/extraction artefacts; 0 = off)
    noise_model: str = "coloured"   # "coloured": per-Fourier-pixel noise in the particle frame (down-weights
                                    # detector/extraction artefacts); "radial": one value per shell
    init_classes: str = ""          # optional .mrc/.mrcs of starting class images (reference-based start)
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
        self.N, B, apix = source.N, source.box, source.apix
        self.B_full, self.apix_full = B, apix
        self.box_a = B * apix
        self.rng = torch.Generator(device="cpu").manual_seed(prm.seed)
        torch.manual_seed(prm.seed)
        self._auto_full, self._auto_iter = prm.n_full_iter < 0, prm.n_iter < 0
        self._auto_hard, self._auto_prior = prm.hard_pose is None, prm.class_prior is None

        # never classify right at Nyquist: the last shells only carry interpolation/aliasing artefacts
        if prm.max_res_a < 2.2 * apix:
            log(f"[res] --max-res {prm.max_res_a:.1f} A is at Nyquist ({2 * apix:.1f} A); using {2.2 * apix:.1f} A")
            prm.max_res_a = 2.2 * apix
        # working (maximum) box: Nyquist at max_res, even, never above the input box
        bmax = int(math.ceil(2 * self.box_a / prm.max_res_a / 2.0)) * 2 + 4
        self.bmax = min(B, bmax + (bmax % 2))
        self.apix_max = self.box_a / self.bmax
        self.radius_a = prm.diameter_a / 2
        if prm.max_offset_a < 0:
            prm.max_offset_a = prm.diameter_a / 4
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
        free = torch.cuda.mem_get_info(self.dev)[0] if self.dev.type == "cuda" else 0
        if c64 < 0.35 * free:
            self.half, where = False, self.dev
        elif c64 / 2 < 0.6 * free:
            self.half, where = True, self.dev
        else:
            self.half, where = True, torch.device("cpu")
        self.store_scale = float(B)  # forward-FFT coefficients ~ 1/B for unit-variance images: keeps fp16 O(1)
        shape = (self.N, bm, bh, 2) if self.half else (self.N, bm, bh)
        self.data = torch.empty(shape, dtype=torch.float16 if self.half else torch.complex64, device=where,
                                pin_memory=where.type == "cpu" and self.dev.type == "cuda")
        self.flip = False
        self._loaded = []                 # list of host index arrays, in load order
        self._n_loaded = 0
        self._load_lock = threading.Lock()
        self._load_err = None
        self._load_t0 = time.time()
        kind = ("fp16 " if self.half else "") + ("on " + str(where) if where.type == "cuda" else "in pinned host RAM")
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
        if not self._polarity_done:
            self._wait_loaded(max(20000, K * 400))
            self._check_polarity(self.polarity_hint)
            self._polarity_done = True
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

        n_sub = s["n_sub"]
        if n_sub < self.Na:
            idx_all = self._sample(n_sub)
            n_sub = len(idx_all)
        else:
            self._wait_loaded(self.N)
            idx_all = self.active
        if p.batch_size > 0:
            bn = p.batch_size
        else:  # keep the per-batch posterior tensor (T x M x Bn x T) around ~0.4 GB
            bn = int(np.clip(0.4e9 / (M * T1 * T1 * 4), 4, 1024))
        ctx["mc"] = max(16, int(0.4e9 / (b * bn * T1 * 8)))  # reference chunk for the GEMMs
        old_cls = self.cls[idx_all].clone()
        seen_before = self.seen[idx_all].clone()

        halves, sumx2, pop = self._run_pass(idx_all, ctx, bn)

        # ---- noise model (exact expected residual power per frequency)
        res2 = sumx2.clone()
        for h in (0, 1):  # each half-set against the references it was aligned to
            refs_m = refs_h[min(h, len(refs_h) - 1)].permute(1, 0, 2)
            dm1 = ft_resize(halves[h][1].view(M, 2 * b, b + 1)[:, ::2, ::2], b_al)     # (M, b_al, bh_a)
            cross = (refs_m * torch.conj(ft_resize(halves[h][0].permute(1, 0, 2), b_al))).real.sum(0)
            res2 += (refs_m.abs() ** 2 * dm1).sum(0) - 2 * cross
        res2 /= n_sub
        sig_new = self._shell_mean(res2, shell_a, fmask_a)
        upd = torch.zeros(self.nsh, dtype=torch.bool, device=self.dev)
        upd[torch.unique(shell_a[fmask_a > 0])] = True
        lam_s = p.sigma_decay if n_sub < self.Na else 0.0   # online: running estimate; full pass: exact
        blended = lam_s * self.sigma2_emp + (1 - lam_s) * sig_new
        self.sigma2_emp = torch.where(upd & (sig_new > 0), blended, self.sigma2_emp)
        old = self.noise2d[rows_a][:, :bh_a]
        ok = (fmask_a > 0) & (res2 > 0)
        self.noise2d[rows_a, :bh_a] = torch.where(ok, lam_s * old + (1 - lam_s) * res2, old)
        del refs_h, refs_m, ctx, cross, dm1

        # ---- reconstruction from (running) sufficient statistics in the class frame
        num0, den0 = self._rotate_back(halves[0][0], halves[0][1], s["psi"], b)
        num1, den1 = self._rotate_back(halves[1][0], halves[1][1], s["psi"], b)
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
        num_k, den_k, pop = ft_resize(self.S_num, b), ft_resize(self.S_den, b), self.S_pop.clone()
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
            del A0, A1
        else:
            ssnr0 = tau2 * dm / self.sigma2_emp[None]
        ssnr0[:, 2:] = torch.cummin(ssnr0[:, 2:], dim=1).values
        tau2 = ssnr0 * self.sigma2_emp[None] / dm
        tau2f = ((1.0 if p.frc_reg else p.tau2_fudge) * tau2)[:, shell]                                       # (K, b, bh)
        # raised-cosine roll-off over the last 3 shells instead of a hard cut (no ringing / lattice artefacts)
        taper = 0.5 + 0.5 * torch.cos(math.pi * ((kr - (klim_rec - 3)) / 3).clamp(0, 1))
        ft = num_k * tau2f / (den_k * tau2f + sig) * fmask * taper
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
        reseeded = self._reseed(it, pop)

        changed = (old_cls != self.cls[idx_all])[seen_before].float().mean().item() if seen_before.any() else 1.0
        self.seen[idx_all] = True
        dt = time.time() - t_start
        info = dict(it=it + 1, res=s["res"], res_rec=s["res_rec"], box=b, n_rot=R, n_trans=T1 * T1, n=n_sub, time=dt,
                    est_res=self.est_res, maxprob=self.maxprob[idx_all].mean().item(), changed=changed,
                    min_pop=int(pop.min().item()), max_pop=int(pop.max().item()), reseeded=reseeded, enc=self._last_enc)
        self.history.append(info)
        self.log(f"[it {it + 1:3d}] align {s['res']:5.1f}A rec {s['res_rec']:5.1f}A box {b_al:3d}/{b:3d} rot {R:3d} ({360 / R:4.1f}deg) "
                 f"trans {T1}x{T1} n={n_sub:7d} | estRes {self.est_res:5.1f}A <Pmax> {info['maxprob']:.3f} ENC {self._last_enc:4.2f} "
                 f"changed {changed * 100:5.1f}% pop[{info['min_pop']},{info['max_pop']}]"
                 f"{' reseed ' + str(reseeded) if reseeded else ''} | {dt:6.2f}s")

    # ------------------------------------------------------- E/M pass (multi-GPU)
    def _run_pass(self, idx_all, ctx, bn):
        """E-step + M-step accumulation over idx_all. Batches are pulled from a shared queue by one
        thread per GPU (dynamic load balancing across GPUs of different speed); accumulators are
        summed on the master device."""
        if ctx["refs"].shape[0] == 2:  # gold standard: every batch holds one half-set only
            par = idx_all % 2
            batches = [g[i:i + bn] for g in (idx_all[par == 0], idx_all[par == 1]) for i in range(0, len(g), bn)]
        else:
            batches = [idx_all[i:i + bn] for i in range(0, len(idx_all), bn)]
        lock = threading.Lock()
        queue = iter(batches)

        def worker(dev):
            with torch.cuda.device(dev):
                c = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ctx.items()}
                b, M, bh = c["b"], c["M"], c["b"] // 2 + 1
                acc = dict(numer0=torch.zeros((b, M, 2 * bh), device=dev),
                           numer1=torch.zeros((b, M, 2 * bh), device=dev),
                           dm2_0=torch.zeros((M, 2 * b * (b + 1)), device=dev),
                           dm2_1=torch.zeros((M, 2 * b * (b + 1)), device=dev),
                           sumx2=torch.zeros((c["b_al"], c["b_al"] // 2 + 1), device=dev),
                           pop=torch.zeros(self.p.n_classes, device=dev), enc=torch.zeros((), device=dev))
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
        tot = {k: sum(a[k].to(self.dev) for a in accs) for k in accs[0]}
        self._last_enc = float(tot["enc"]) / max(len(idx_all), 1)
        bh = ctx["b"] // 2 + 1
        halves = [(torch.complex(tot[f"numer{h}"][..., :bh], tot[f"numer{h}"][..., bh:]), tot[f"dm2_{h}"])
                  for h in (0, 1)]
        return halves, tot["sumx2"], tot["pop"]

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
            cpost = torch.exp(mx - lse[None])                                          # class posterior (K, n)
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
            if not torch.isfinite(lse).all():                                         # drop broken particles
                badn = ~torch.isfinite(lse)
                ll[:, :, badn, :] = float("-inf")
                ll[0, 0, badn, 0] = 0.0
                lse = torch.where(badn, torch.zeros_like(lse), lse)
            flat = ll.permute(2, 1, 0, 3).reshape(n, -1)                              # (n, M*Ty*Tx)
            top, best = torch.topk(flat, min(256, flat.shape[1]), dim=1)
            del flat
            wts = ll.sub_(lse[None, None, :, None]).exp_()                            # posterior, in place
            best = best[:, 0]
            sw = torch.exp(top - lse[:, None])

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
        half = (idx % 2).to(dev)
        for hv in (0, 1):  # odd/even half-sets (fixed per particle) -> FRC regulariser
            sel = torch.nonzero(half == hv).flatten()
            nh = len(sel)
            if nh == 0:
                continue
            yh = y[sel]
            g = (yh.permute(1, 0, 2)[:, :, None, :] * torch.conj(c["Exr"]).T[None, None]).reshape(b, nh * T1, bh)
            g1 = torch.cat([g.real, g.imag], -1).to(self._mdt)                        # (b, nT, 2bh)
            g2 = torch.cat([-g.imag, g.real], -1).to(self._mdt)
            numer = acc[f"numer{hv}"]
            wh = wts[:, :, sel]                                                       # (T1, M, nh, T1)
            for m0 in range(0, M, mc):
                m1 = min(M, m0 + mc)
                h = (E2 @ wh[:, m0:m1].reshape(T1, -1).to(self._mdt)).view(2, b, m1 - m0, nh * T1)  # sum over ty
                numer[:, m0:m1] += (torch.bmm(h[0], g1) + torch.bmm(h[1], g2)).float() / self.store_scale
                del h
            acc[f"dm2_{hv}"] += (wnm[:, sel].to(self._mdt) @ c2all[sel].reshape(nh, -1).to(self._mdt)).float()
        acc["sumx2"] += x2.sum(0)
        acc["pop"] += wnm.view(K, R, n).sum((1, 2))


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
            thr = self.p.recenter_thresh * img.amax((1, 2), keepdim=True)
            pos = torch.where(img > thr, img, torch.zeros_like(img))
            r = (torch.arange(b, device=self.dev, dtype=torch.float32) - b // 2) * apix
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

    def _reseed(self, it, pop):
        p = self.p
        if it >= int(0.7 * p.n_iter) or p.reseed_frac <= 0:
            return 0
        thr = p.reseed_frac * pop.sum() / p.n_classes
        small = torch.nonzero(pop < thr).flatten().tolist()
        n = 0
        for k in small:
            big = int(torch.argmax(pop))
            noise = torch.randn(self.classes[big].shape, device=self.dev, dtype=torch.complex64)
            amp = self.classes[big].abs().mean()
            self.classes[k] = self.classes[big] + 0.3 * amp * noise * (self.classes[big] != 0)
            if self.classes_h is not None:  # same split in both half-sets keeps class k matched between them
                self.classes_h[:, k] = self.classes_h[:, big] + 0.3 * amp * noise * (self.classes_h[:, big] != 0)
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

    # --------------------------------------------------------------------
    def run(self, active=None, n_classes=None, seed=None):
        """Classify the particles `active` (None = all) into n_classes (None = Params value)."""
        if n_classes is not None:
            self.p.n_classes = n_classes
        if seed is not None:
            self.rng.manual_seed(seed)
        self._set_active(active)
        t0 = time.time()
        self.init_classes()
        for it in range(self.p.n_iter):
            self.iterate(it)
        for d in self.devs:
            torch.cuda.synchronize(d)
        self.total_time = time.time() - t0
        self.log(f"[done] {self.p.n_iter} iterations in {self.total_time:.1f}s")

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
