"""Modes of variation of the density at fixed poses: continuous and discrete conformational heterogeneity as a few
principal modes of the particles' 3D density (Modes3D job), and maps along them or of clusters in their coordinates
(ModeMaps job). The method is probabilistic PCA solved by EM in Fourier space, after Punjani & Fleet, J Struct Biol
213:107702 (2021).

Model (particles at their refined poses A_i, shifts t_i and CTF):

    x_i = e(t_i) CTF_i a_i P_{A_i}(mu + sum_k z_ik W_k) + noise,     z_i ~ N(0, I),  noise per shell sigma^2(k)

mu is the mean map and W_1..W_K the modes (re-solved from the particles, so they are in the particles' own units; the
consensus map is not needed); a_i a per-particle scale (ice thickness, normalisation: fitted against the mean's
projection, so a pure change of contrast is not taken for a mode). EM:

  E-step  per particle, batched GEMMs: W_i = a_i CTF_i P_{A_i} W (projections of the K modes, frequencies of the band
          only), posterior precision I + Re(W_i^H D W_i) (D = 2 / sigma^2 per shell, half-plane counting), E[z_i] and
          E[z_i z_i^T].
  M-step  the mean and the K modes together: for every 3D Fourier voxel a (K+1) x (K+1) system
          sum_i a_i^2 E[z^ z^T] CTF^2 D  V = sum_i a_i E[z^] CTF D x_i   (z^ = (1, z)), accumulated by trilinear
          back-projection (with every symmetry copy) and solved voxel by voxel, regularised by lambda x the shell's
          mean weight. Then in real space: the modes masked (MaskCreate job / the refinement's solvent mask / an
          automatic one) and low-passed to the filter resolution, the mean to the particle sphere; the modes are
          rotated to their principal axes (orthogonal, ordered by variance - the prior is isotropic, so this changes
          nothing in the model) and the noise per shell re-estimated from the residuals.

Each iteration is one pass over the particles (the spectra the 2D engine holds, cropped to the band's box), so 200k
particles on a 24 GB card are a matter of the band box (8 A on a 340 A box: 98 px, 6 GB of fp16 spectra).
Reported per mode: sd (model standard deviation of the coordinate along the unit-norm mode, in map units, and relative
to the masked mean's norm), the standard deviation of the posterior means (shrunk towards 0 by the noise) and, from
two independent solves on the gold-standard halves, the correlation of the halves' modes matched to each mode, overall
and per Fourier shell (noise does not reproduce; |cc| >= 0.5 is "reproducible"). Figures show each mode low-passed to
where its halves agree.

Synthetic checks (tests/test_modes.py, 3000 particles, box 64, CTF, SNR ~0.4): a domain moving with a uniform
parameter -> |r(z_1, truth)| 0.98; two discrete states -> a two-cluster mixture recovers them (100 %); no motion ->
modes that do not reproduce between halves (|cc| <= 0.04).

Real data (dz_2025_10, 38.5k particles at the poses of a D2 Refine3D at 5.5 A, 8 A filter, RTX PRO 4000): an
unstructured sd of ~15 % of |mean| per mode (C1) whatever the data hold, so read the half-set |cc|, not the sd.
A known motion added to those particles (positive control, a 20-30 A domain moved by +-10-15 A) is found (|r| 0.88)
only once it carries ~10 % of the particle's density; at 2-6 % it stays below this noise floor. The analysis is a few
minutes per 40k particles (5 s per iteration at a 98 px band box, plus the two half solves); 192k particles: 25 s per
iteration, 6 GB of GPU memory besides the spectra (8 GB of fp16 spectra for 200k at this box). lambda 0.01-3 changes
neither the noise floor's structure nor the injected motion's recovery; 30 zeroes the modes.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .clock import since
from .engine import TWO_PI, ctf_2d, freqs, release_cache
from .project import logged


# --------------------------------------------------------------------------------------------- projector helpers
def _project(V0, ft: torch.Tensor, A: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Central slices of a stack of spectra ft (2C, G, G, G) (real / imag channels of C volumes, abinitio.Volume's
    grid) at orientations A (n, 3, 3), frequencies keep (b, bh) -> (n, C, P) complex, in the volume frame (the
    particle-frame spectrum is this times Volume.checker)."""
    G, C = V0.G, ft.shape[0] // 2
    P = int(keep.sum())
    out = []
    chunk = max(1, 2_000_000 // max(P, 1))
    for i in range(0, len(A), chunk):
        q = V0._coords(A[i:i + chunk], keep)                                    # (m, P, 3)
        g = ((q + G // 2 + 0.5) / G * 2 - 1).reshape(1, len(q), P, 1, 3)
        s = F.grid_sample(ft[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        s = s.reshape(C, 2, len(q), P)
        out.append(torch.complex(s[:, 0], s[:, 1]).permute(1, 0, 2))
    return torch.cat(out) if out else torch.zeros((0, C, P), dtype=torch.complex64, device=ft.device)


def _splat(acc: torch.Tensor, V0, A: torch.Tensor, vals: torch.Tensor, keep: torch.Tensor, conj: torch.Tensor):
    """Trilinear back-projection of per-frequency values vals (n, P, Ch) at orientations A into acc (G^3, Ch), and of
    their Friedel mates (channels times conj: -1 on imaginary parts)."""
    G = V0.G
    q = V0._coords(A, keep).reshape(-1, 3)
    v = vals.reshape(-1, vals.shape[-1])
    for sgn in (1.0, -1.0):
        p = sgn * q + G // 2
        p0 = torch.floor(p)
        f = p - p0
        p0 = p0.long()
        vv = v if sgn > 0 else v * conj
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix, iy, iz = p0[:, 0] + dx, p0[:, 1] + dy, p0[:, 2] + dz
                    wt = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                          * (f[:, 2] if dz else 1 - f[:, 2]))
                    ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G) & (iz >= 0) & (iz < G)
                    acc.index_add_(0, ((iz * G + iy) * G + ix)[ok], vv[ok] * wt[ok, None])


def _triu(C: int):
    return [(c, d) for c in range(C) for d in range(c, C)]


def upsample(v: torch.Tensor, b_out: int) -> torch.Tensor:
    """A real map Fourier-padded (or cropped) to b_out^3 over the same box in A, values kept."""
    from .abinitio import _to_box
    b = v.shape[0]
    return v if b == b_out else _to_box(v, b_out, v.device) * (b_out / b) ** 3


# ------------------------------------------------------------------------------------------------------- solver
def solve(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, n_comp: int = 3, filter_res_a: float = 8.0,
          n_iter: int = 20, lam: float = 0.01, sym: str = "C1", mask: torch.Tensor | None = None,
          highpass_a: float = 0.0, scale: bool = True, box: int | None = None, seed: int = 0, log=print,
          tag: str = "", grid_correct: bool | None = None) -> dict:
    """Probabilistic PCA of particles idx (host indices of clf's loaded spectra) at orientations A (n, 3, 3) and model
    shifts t (n, 2) A. mask: real (b, b, b) at the working box (None: masking.auto_mask of the mean). Returns dict:
    mean (b^3), basis (K, b^3, unit norm), sd (K,) model sd along each unit component (map units), z (n, K) posterior
    means (prior sd units: frame alpha of the series is coordinate alpha), scale (n,), mask, b, apix, history."""
    from .abinitio import Volume, sphere_mask, symmetry_ops
    from .masking import _lowpass, auto_mask
    dev, box_a = clf.dev, clf.box_a
    b = int(box or clf.bmax)
    apix = box_a / b
    K, n = int(n_comp), len(idx)
    k_hi = box_a / filter_res_a
    if k_hi > b / 2 - 2:
        log(f"[modes] filter resolution {filter_res_a:.1f} A is beyond the working box's Nyquist "
            f"({2 * apix:.1f} A): using {box_a / (b / 2 - 2):.1f} A")
        k_hi = b / 2 - 2
    k_lo = max(1.0, box_a / highpass_a) if highpass_a and highpass_a > 0 else 1.0
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    # the band: the model is band-limited (cosine edge of the real-space low-pass to 1.1 k_hi)
    band = (kr >= k_lo) & (kr <= min(b / 2 - 1, 1.1 * k_hi))
    P = int(band.sum())
    shb = torch.round(kr[band]).long()
    fxb = fx[None, :].expand(b, -1)[band]
    fyb = fy[:, None].expand(-1, len(fx))[band]
    hw = torch.where(fxb == 0, 0.5, 1.0)          # the kx = 0 column holds each coefficient and its Friedel mate
    ph = TWO_PI / box_a
    ops = symmetry_ops(sym).to(dev)
    C = K + 1
    T = C * (C + 1) // 2
    Ch = 2 * C + T
    # padding: accumulators (Ch), the stack of spectra and its transients per padded voxel
    free = _free(dev)
    pad = 2 if (8 * b ** 3) * (Ch + 4 * C + 8) * 4 < 0.35 * free else 1
    # gridding correction on: without it trilinear sampling and splatting each damp the map away from the centre
    # (sinc^2 per axis), the M-step's basis came back too weak for its own E-step and the coordinates grew to make up
    # for it (synthetic domain motion: sd of the posterior means 1.9, where an EM fixed point has <= 1; with it 0.83,
    # same |r| 0.98; per-particle scales 1.12 -> 1.03)
    gc = True if grid_correct is None else (bool(grid_correct) or pad == 1)
    V0 = Volume(b, dev, pad=pad, grid_correct=gc)
    G = V0.G
    chk = V0.checker[band]
    sph = sphere_mask(b, clf.p.diameter_a / 2 / apix, 3.0, dev)
    sh3 = V0.shell_index().reshape(-1).long()
    nsh = int(sh3.max()) + 1
    cnt3 = torch.bincount(sh3, minlength=nsh).clamp_min(1).float()
    # particles per batch: projections, data, products (complex, C x P), the back-projected values (Ch x P, with
    # their trilinear transients) - a quarter of the free memory
    per = P * (C * 8 * 6 + Ch * 4 * 4 + 120)
    nb = int(max(32, min(8192, 0.25 * free / per)))
    A, t = A.float(), t.float()
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()

    def real_of(ft):
        return Volume(b, dev, pad=pad, ft=ft, pre=False, grid_correct=gc).real()

    def ft_of(v):
        vol = Volume(b, dev, pad=pad, grid_correct=gc)
        vol.set_real(v)
        return vol.ft

    def one_pass(stack, sig2b, accumulate: bool, estep: bool):
        """stack: (2Cs, G^3) spectra (mean + Ks basis) or None (accumulate the mean from the data alone); sig2b: noise
        power of each band frequency (None: unit weights)."""
        Cs = 1 if stack is None else stack.shape[0] // 2
        Ks = Cs - 1
        Cc = Cs if accumulate else 0
        Tc = Cc * (Cc + 1) // 2
        acc = torch.zeros((G ** 3, 2 * Cc + Tc), device=dev) if accumulate else None
        conj = torch.ones(2 * Cc + Tc, device=dev)
        conj[1:2 * Cc:2] = -1
        pairs = _triu(Cc)
        w = hw * (2.0 / sig2b) if sig2b is not None else hw      # log p(x|z) = -sum_half |r|^2 / sigma^2
        res_pow = torch.zeros(int(kr.max()) + 2, device=dev)
        res_cnt = torch.zeros_like(res_pow)
        Z = torch.zeros((n, Ks)) if estep and Ks else None
        S_all = torch.ones(n)
        for i in range(0, n, nb):
            j = idx[i:i + nb]
            m = len(j)
            x = clf._fetch(j, dev, b)[:, band]
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
            tt = t[i:i + nb].to(dev)
            AA = A[i:i + nb].to(dev)
            xs = x * torch.exp(1j * ph * (fxb * tt[:, 0, None] + fyb * tt[:, 1, None])) * chk   # unshifted, volume frame
            del x
            a = torch.ones(m, device=dev)
            Ez = torch.zeros((m, 0), device=dev)
            Ezz = torch.zeros((m, 0, 0), device=dev)
            if stack is not None:
                Pm = _project(V0, stack.view(-1, G, G, G), AA, band)                      # (m, Cs, P)
                CP0 = ctf * Pm[:, 0]
                if scale:
                    a = ((w * (CP0.conj() * xs).real).sum(1) / (w * CP0.abs() ** 2).sum(1).clamp_min(1e-30)).clamp(0.2, 5.0)
                r = xs - a[:, None] * CP0
                if Ks:
                    Wb = (a[:, None, None] * ctf[:, None]) * Pm[:, 1:]                   # (m, K, P)
                    Wr, Wi = Wb.real * w, Wb.imag * w
                    Mt = Wr @ Wb.real.transpose(1, 2) + Wi @ Wb.imag.transpose(1, 2)    # (m, K, K)
                    bv = Wr @ r.real[:, :, None] + Wi @ r.imag[:, :, None]                # (m, K, 1)
                    Sig = torch.linalg.inv(torch.eye(Ks, device=dev) + Mt)
                    Ez = (Sig @ bv)[:, :, 0]
                    Ezz = Sig + Ez[:, :, None] * Ez[:, None, :]
                    r = r - (Ez[:, :, None] * Wb).sum(1)
                    del Wb, Wr, Wi, Pm
                res_pow.index_add_(0, shb, (r.abs() ** 2).sum(0))
                res_cnt.index_add_(0, shb, torch.full_like(shb, m, dtype=torch.float32))
                del r, CP0
            if Z is not None:
                Z[i:i + nb] = Ez.cpu()
            S_all[i:i + nb] = a.cpu()
            if accumulate:
                zh = torch.cat([torch.ones((m, 1), device=dev), Ez], 1)                  # (m, C)
                S = torch.zeros((m, Cc, Cc), device=dev)
                S[:, 0, 0] = 1
                if Ks:
                    S[:, 0, 1:] = Ez
                    S[:, 1:, 0] = Ez
                    S[:, 1:, 1:] = Ezz
                St = torch.stack([S[:, c, d] for c, d in pairs], 1)                      # (m, T)
                base_c = ctf * w * xs
                base_r = ctf * ctf * w
                cv = torch.view_as_real((a[:, None] * zh)[:, None, :] * base_c[:, :, None]).reshape(m, P, 2 * Cc)
                rv = (a[:, None] ** 2 * St)[:, None, :] * base_r[:, :, None]
                vals = torch.cat([cv, rv], 2)
                del cv, rv, base_c, base_r
                for S_ in ops:
                    _splat(acc, V0, AA @ S_, vals, band, conj)
                del vals
            del xs, ctf
        sigma2 = torch.where(res_cnt > 0, res_pow / res_cnt.clamp_min(1), torch.zeros_like(res_pow))
        return acc, Z, S_all, sigma2

    def m_solve(acc, Cc):
        """Per-voxel (Cc x Cc) systems; returns spectra (Cc, 2, G^3)."""
        pairs = _triu(Cc)
        num = acc[:, :2 * Cc].view(-1, Cc, 2)
        H00 = acc[:, 2 * Cc]
        m_sh = torch.zeros(nsh, device=dev).index_add_(0, sh3, H00) / cnt3
        reg = m_sh[sh3]
        eps = 1e-6 * float(m_sh.max())
        out = torch.empty((Cc, 2, G ** 3), device=dev)
        step = max(1, int(2e6))
        for s0 in range(0, G ** 3, step):
            sl = slice(s0, s0 + step)
            mm = min(step, G ** 3 - s0)
            H = torch.empty((mm, Cc, Cc), device=dev)
            for k, (c, d) in enumerate(pairs):
                H[:, c, d] = acc[sl, 2 * Cc + k]
                H[:, d, c] = acc[sl, 2 * Cc + k]
            H[:, 0, 0] += 0.02 * reg[sl] + eps
            for c in range(1, Cc):
                H[:, c, c] += lam * reg[sl] + eps
            out[:, :, sl] = torch.linalg.solve(H, num[sl]).permute(1, 2, 0)
        return out

    def basis_real(sol):
        Wr = torch.stack([_lowpass(real_of(sol[c].view(2, G, G, G)) * mask, apix, box_a / k_hi) for c in range(1, C)])
        return Wr

    def orthogonalise(Wr):
        flat = Wr.reshape(K, -1)
        ev, U = torch.linalg.eigh(flat @ flat.T)
        o = torch.argsort(ev, descending=True)
        flat = U[:, o].T @ flat
        for k in range(K):                       # sign: the largest-magnitude voxel positive
            if flat[k, flat[k].abs().argmax()] < 0:
                flat[k] = -flat[k]
        return flat.reshape(Wr.shape)

    def stack_of(mean_r, Wr):
        return torch.cat([ft_of(mean_r)] + [ft_of(w_) for w_ in Wr]).reshape(2 * (1 + len(Wr)), -1)

    # ---- the mean from the data alone, then the noise against it
    acc, _, _, _ = one_pass(None, None, True, False)
    sol = m_solve(acc, 1)
    del acc
    mean_r = real_of(sol[0].view(2, G, G, G)) * sph
    del sol
    if mask is None:
        mask = auto_mask(mean_r, apix, lowpass_a=max(15.0, filter_res_a), extend_a=6.0, soft_a=8.0,
                         diameter_a=clf.p.diameter_a)
        log(f"[modes] automatic mask: {float(mask.sum()) * apix ** 3 / 1e6:.2f} M A^3")
    mask = mask.to(dev).float()
    st = torch.cat([ft_of(mean_r)]).reshape(2, -1)
    _, _, a0, sigma2 = one_pass(st, None, False, False)
    mean_r = mean_r * float(a0.mean())
    sig2b = sigma2.clamp_min(1e-30)[shb]
    # ---- random start: smooth noise in the mask, small next to the mean
    mnorm = float((mean_r * mask).norm())
    Wr = torch.stack([_lowpass(torch.randn((b, b, b), generator=g).to(dev) * mask, apix, box_a / k_hi) for _ in range(K)])
    Wr = orthogonalise(Wr)
    Wr = Wr / Wr.reshape(K, -1).norm(dim=1)[:, None, None, None] * 0.1 * mnorm
    history = []
    U_prev = None
    log(f"[modes]{tag} {n:,} particles, {K} modes, band {box_a / k_lo:.0f}-{box_a / k_hi:.1f} A ({P} "
        f"frequencies), box {b} (pad {pad}), {sym}, batches of {nb}")
    for it in range(n_iter):
        st = stack_of(mean_r, Wr)
        acc, _, a_it, sigma2 = one_pass(st, sig2b, True, True)
        del st
        sol = m_solve(acc, C)
        del acc
        # a (mu + W z) is unchanged by a -> c a, mu -> mu / c: the pass's mean scale goes into the maps, so the scales
        # stay at mean 1 (left alone, the regularised mean shrank every iteration and the scales ran off to the clamp)
        abar = float(a_it.mean())
        mean_r = real_of(sol[0].view(2, G, G, G)) * sph * abar
        Wr = orthogonalise(basis_real(sol)) * abar
        del sol
        sig2b = sigma2.clamp_min(1e-30)[shb]
        sd = Wr.reshape(K, -1).norm(dim=1)
        U = Wr.reshape(K, -1) / sd[:, None].clamp_min(1e-30)
        ov = float(((U_prev @ U.T) ** 2).sum() / K) if U_prev is not None else float("nan")
        U_prev = U
        mnorm = float((mean_r * mask).norm())
        history.append(dict(it=it + 1, sd_rel=[round(float(s) / max(mnorm, 1e-30), 5) for s in sd],
                            overlap=round(ov, 5), seconds=round(since(t0), 1)))
        log(f"[modes]{tag} it {it + 1}/{n_iter}: sd / |mean| " + " ".join(f"{float(s) / mnorm * 100:.2f}%" for s in sd)
            + (f", subspace overlap with the last {ov:.4f}" if ov == ov else "") + f" ({since(t0):.1f}s)")
    # ---- final E-step: every particle's coordinates on the final (orthogonal, ordered) basis
    st = stack_of(mean_r, Wr)
    _, Z, a, _ = one_pass(st, sig2b, False, True)
    del st
    sd = Wr.reshape(K, -1).norm(dim=1)
    basis = Wr / sd[:, None, None, None].clamp_min(1e-30)
    release_cache()
    return dict(mean=mean_r, basis=basis, sd=sd.cpu(), z=Z, scale=a, mask=mask, b=b, apix=apix, box_a=box_a,
                mean_norm=float((mean_r * mask).norm()), history=history, sigma2=sigma2.cpu(),
                band_a=(box_a / k_lo, box_a / k_hi), seconds=round(time.time() - t0, 1))


def _free(dev) -> float:
    from .abinitio import _free as f
    return f(dev)


def compare_bases(B1: torch.Tensor, B2: torch.Tensor) -> dict:
    """Two solves' unit-norm bases (K, b^3): per component of B1 the |cc| with its match in B2 (Hungarian on |cc|),
    and the overlap of the two K-dimensional subspaces (mean squared cosine of the principal angles: 1 identical, ~K /
    (number of voxels) for unrelated noise)."""
    from scipy.optimize import linear_sum_assignment
    K = len(B1)
    U1 = B1.reshape(K, -1)
    U2 = B2.reshape(K, -1).to(U1.device)
    U1 = U1 / U1.norm(dim=1, keepdim=True).clamp_min(1e-30)
    U2 = U2 / U2.norm(dim=1, keepdim=True).clamp_min(1e-30)
    cc = (U1 @ U2.T).abs().cpu().numpy()
    r, c = linear_sum_assignment(-cc)
    match = np.zeros(K)
    match[r] = cc[r, c]
    Q1, _ = torch.linalg.qr(U1.T)
    Q2, _ = torch.linalg.qr(U2.T)
    s = torch.linalg.svdvals(Q1.T @ Q2)
    return dict(cc=[round(float(v), 4) for v in match], subspace_overlap=round(float((s ** 2).mean()), 4))


# ------------------------------------------------------------------------------------------------- clustering
def gmm(X: np.ndarray, n_clusters: int, seed: int = 0, n_init: int = 5, n_iter: int = 200) -> dict:
    """Gaussian mixture (full covariances) by EM from k-means++ starts; the best of n_init by log-likelihood. Clusters
    are numbered by their mean along the first coordinate. Returns labels (n,), probs (n, C), means, covs, weights,
    loglik, bic."""
    X = np.asarray(X, np.float64)
    n, d = X.shape
    Cn = int(n_clusters)
    rng = np.random.default_rng(seed)
    best = None
    reg = 1e-6 * np.var(X, 0).mean() + 1e-12
    for _ in range(n_init):
        cen = [X[rng.integers(n)]]
        for _ in range(1, Cn):
            d2 = np.min([((X - c) ** 2).sum(1) for c in cen], 0)
            cen.append(X[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(n)])
        mu = np.array(cen)
        cov = np.array([np.cov(X.T).reshape(d, d) + reg * np.eye(d) for _ in range(Cn)])
        pi = np.full(Cn, 1.0 / Cn)
        ll_old = -np.inf
        for _ in range(n_iter):
            lp = np.empty((n, Cn))
            for c in range(Cn):
                L = np.linalg.cholesky(cov[c])
                y = np.linalg.solve(L, (X - mu[c]).T)
                lp[:, c] = np.log(pi[c] + 1e-300) - 0.5 * (y ** 2).sum(0) - np.log(np.diag(L)).sum() \
                    - 0.5 * d * math.log(2 * math.pi)
            mx = lp.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
            R = np.exp(lp - lse[:, None])
            ll = float(lse.sum())
            Nk = R.sum(0) + 1e-12
            pi = Nk / n
            mu = (R.T @ X) / Nk[:, None]
            for c in range(Cn):
                Y = X - mu[c]
                cov[c] = (R[:, c, None] * Y).T @ Y / Nk[c] + reg * np.eye(d)
            if ll - ll_old < 1e-7 * abs(ll):
                break
            ll_old = ll
        if best is None or ll > best["loglik"]:
            best = dict(probs=R, means=mu.copy(), covs=cov.copy(), weights=pi.copy(), loglik=ll)
    o = np.argsort(best["means"][:, 0])
    best = dict(probs=best["probs"][:, o], means=best["means"][o], covs=best["covs"][o], weights=best["weights"][o],
                loglik=best["loglik"])
    best["labels"] = best["probs"].argmax(1)
    n_par = Cn * (d + d * (d + 1) / 2) + Cn - 1
    best["bic"] = float(-2 * best["loglik"] + n_par * math.log(n))
    return best


def quantile_bins(v: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bins of equal numbers of particles along a coordinate: (labels 0..n_bins-1, edges)."""
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    lab = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, n_bins - 1)
    return lab, edges


# ----------------------------------------------------------------------------------------------- reconstruction
def reconstruct_groups(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, labels: np.ndarray, n_groups: int,
                       half: np.ndarray, sym: str = "C1", box: int | None = None, log=print) -> list[dict]:
    """Gold-standard reconstruction of each group of particles (labels 0..n_groups-1, -1 none) at fixed poses, one
    group at a time (two half volumes in memory): per group the FSC-weighted map, half maps, solvent mask (from the
    map), corrected masked FSC and resolution."""
    from . import abinitio as ab
    dev, box_a = clf.dev, clf.box_a
    b = int(box or clf.bmax)
    apix = box_a / b
    fy, fx = freqs(b, dev)
    ph = TWO_PI / box_a
    ops = ab.symmetry_ops(sym).to(dev)
    pad = ab.auto_pad(b, 1, dev)
    gc = True                                   # gridding-corrected (Volume: reconstruction error 7.7 -> 3.1 %)
    bp = int(max(64, min(2048, 2e8 // (b * (b // 2 + 1) * 8))))
    sph = ab.sphere_mask(b, clf.p.diameter_a / 2 / apix, 3.0, dev)
    labels = np.asarray(labels)
    half = torch.as_tensor(np.asarray(half)).long()
    out = []
    t0 = time.time()
    for k in range(n_groups):
        sel = np.flatnonzero(labels == k)
        if len(sel) < 20:
            out.append(None)
            continue
        vols = [ab.Volume(b, dev, pad=pad, grid_correct=gc) for _ in (0, 1)]
        for v in vols:
            v.clear()
        for i in range(0, len(sel), bp):
            s = torch.as_tensor(sel[i:i + bp])
            j = idx[s]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[s].to(dev).float()
            xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            hh = half[s].to(dev)
            AA = A[s].to(dev).float()
            for h in (0, 1):
                m = hh == h
                if m.any():
                    vols[h].insert((xs * ctf)[m], (ctf * ctf)[m], AA[m], kmax=b / 2 - 1, ops=ops)
        for v in vols:
            ab._regularise_solve(v, 0.02)
            v.free_acc()
        h1, h2 = vols[0].real(), vols[1].real()
        f_sph = ab.fsc_maps(h1 * sph, h2 * sph)
        avg = ab.Volume(b, dev, pad=pad, ft=0.5 * (vols[0].ft + vols[1].ft), pre=False, grid_correct=gc)
        avg.filter_shells(torch.sqrt((2 * f_sph / (1 + f_sph)).clamp(0, 1).nan_to_num(0)))
        vmap = (avg.real() * sph).nan_to_num(0)
        del avg, vols
        smask = ab.solvent_mask(vmap, apix)
        f_c = ab.corrected_masked_fsc(h1, h2, smask)
        res = ab.resolution_from_fsc(f_c, box_a)
        out.append(dict(map=vmap, h1=h1, h2=h2, mask=smask, fsc=f_c, fsc_unmasked=ab.fsc_maps(h1, h2),
                        fsc_sphere=f_sph, resolution_a=res, particles=int(len(sel))))
        log(f"[modes] group {k + 1}/{n_groups}: {len(sel):,} particles, {res:.2f} A ({since(t0):.1f}s)")
        release_cache()
    return out


# ---------------------------------------------------------------------------------------------- reproducibility
REPRODUCIBLE = 0.5      # half-set |cc| of a mode from which it counts as reproducible (noise: |cc| ~ 0-0.4 here)
# sd / |mean| of modes that did not reproduce on real data: dz_2025_10, 38k particles at 8 A (C1 14.5-16 %, D2 11-15 %);
# a reference line in the figures, not a law (synthetic noise-only particles, 3000 at SNR ~0.4 and 12 A: ~4 %)
NOISE_FLOOR = 0.15


def half_reproducibility(B: torch.Tensor, H1: torch.Tensor, H2: torch.Tensor, box_a: float,
                         thr: float = REPRODUCIBLE) -> dict:
    """Per mode of the full solve B (K, b^3): its match in each half's modes (Hungarian on |cc|, sign aligned to B),
    the |cc| of the two matched half modes, their correlation per Fourier shell and the resolution of the last shell
    where it is >= thr (None: not even the first); plus the overlap of the halves' subspaces."""
    from scipy.optimize import linear_sum_assignment
    from .abinitio import fsc_maps
    K = len(B)
    U = B.reshape(K, -1)
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)

    def matched(H):
        V = H.reshape(K, -1).to(U.device)
        V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-30)
        cc = (U @ V.T).cpu().numpy()
        r, c = linear_sum_assignment(-np.abs(cc))
        o = np.empty(K, int)
        o[r] = c
        return [H[o[k]].to(U.device) * float(np.sign(cc[k, o[k]]) or 1.0) for k in range(K)]

    m1, m2 = matched(H1), matched(H2)
    cc, curves, res = [], [], []
    for k in range(K):
        a, c = m1[k], m2[k]
        cc.append(round(abs(float((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-30))), 4))
        f = fsc_maps(a, c).cpu().numpy()
        curves.append([round(float(x), 4) for x in f])
        last = 0
        for i in range(1, len(f)):
            if f[i] < thr:
                break
            last = i
        res.append(round(box_a / last, 2) if last else None)
    return dict(cc=cc, shell_cc=curves, res_a=res, subspace_overlap=compare_bases(H1, H2)["subspace_overlap"])


def display_res(half_cc, half_res_a, filter_res_a: float) -> list:
    """Per mode, the resolution it is shown at: where its halves agree for a reproducible mode (never finer than the
    filter), else the filter resolution (a mode of noise is not smoothed into something that looks like a motion)."""
    out = []
    for k in range(len(half_res_a or [])):
        r = half_res_a[k]
        ok = half_cc is not None and half_cc[k] >= REPRODUCIBLE and r
        out.append(round(max(float(r), filter_res_a), 2) if ok else float(filter_res_a))
    return out


def lowpass_modes(basis: np.ndarray, apix: float, res_a, filter_res_a: float) -> np.ndarray:
    """The modes low-passed for display to res_a (per mode; the filter resolution: unchanged)."""
    from .masking import _lowpass
    out = np.array(basis, np.float32, copy=True)
    for k in range(len(out)):
        r = res_a[k] if res_a and k < len(res_a) else filter_res_a
        if r and r > filter_res_a * 1.01:
            out[k] = _lowpass(torch.as_tensor(out[k]), apix, float(r)).numpy()
    return out


def status(cc) -> str:
    """The plain reproducibility tag of a mode."""
    if cc is None or cc != cc:
        return "not checked"
    return "reproducible" if cc >= REPRODUCIBLE else "not reproducible (noise)"


# ------------------------------------------------------------------------------------------------------- outputs
def series(mean: np.ndarray, basis: np.ndarray, sd: np.ndarray, k: int, n_frames: int = 20, rng: float = 2.0):
    """Frames of mean + alpha sd_k u_k, alpha from -rng to +rng (prior standard deviations of the coordinate)."""
    al = np.linspace(-rng, rng, n_frames)
    return [(float(a), mean + a * float(sd[k]) * basis[k]) for a in al]


def surface_level(v: np.ndarray, mask: np.ndarray | None = None) -> float:
    """A contour level for a map: the value enclosing about half of the mask's volume (or 5 % of the box)."""
    n = int(0.5 * mask.sum()) if mask is not None and mask.sum() > 0 else int(0.05 * v.size)
    return float(np.sort(v.ravel())[::-1][max(0, min(v.size - 1, n))])


def write_cxc(path: str, frames: list[str], level: float):
    """A ChimeraX script that opens the frames (same folder) as a volume series and plays it back and forth."""
    with open(path, "w") as f:
        f.write("# ChimeraX: open this file (it plays the frames as a volume series)\n")
        f.write("open " + " ".join(os.path.basename(p) for p in frames) + " vseries true\n")
        f.write(f"volume #1 level {level:.6g} style surface\n")
        f.write("vseries play #1 loop true direction oscillate\n")


def write_series(out, mean, basis, sd, mask, apix, prefix: str, n_frames: int = 20, rng: float = 2.0,
                 modes=None, write_mrc=None) -> list[dict]:
    """Per mode (1-based numbers), the series mean + alpha sd u (float16 MRC frames, out(name) gives each path) and a
    .cxc to play it. Returns [{mode, frames: [...], cxc}]."""
    write_mrc = write_mrc or _write_mrc
    res = []
    lvl = surface_level(mean, None if mask is None else mask)
    for k in (modes or range(1, len(basis) + 1)):
        names = []
        for f, (al, v) in enumerate(series(mean, basis, sd, k - 1, n_frames, rng)):
            p = out(f"{prefix}mode{k}_f{f:02d}.mrc")
            write_mrc(p, v, apix, "float16")
            names.append(p)
        cxc = os.path.join(os.path.dirname(names[0]), os.path.basename(names[0]).rsplit("_f00.mrc", 1)[0] + ".cxc")
        write_cxc(cxc, names, lvl)
        res.append(dict(mode=k, frames=names, cxc=cxc))
    return res


def _write_mrc(path, data, apix, dtype="float32"):
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(data, dtype=np.float16 if dtype == "float16" else np.float32))
        m.voxel_size = apix


# ------------------------------------------------------------------------------------------------ figure helpers
AXIS_NAME = "zyx"        # maps are (z, y, x) arrays: a view along axis 0 looks down z


def view_axis(vols) -> int:
    """The axis along which the projections of these maps (a mode, or differences) have the most power."""
    vols = [np.asarray(v, np.float64) for v in (vols if isinstance(vols, (list, tuple)) else [vols])]
    return int(np.argmax([sum(float((v.sum(ax) ** 2).sum()) for v in vols) for ax in range(3)]))


def surface_view(v: np.ndarray, level: float, ax: int) -> np.ndarray:
    """The iso-surface of v at level seen along axis ax (viewer at index 0), shaded by its normal (light from the
    upper left) and dimmed with depth: values 0-1, NaN where a ray misses the surface. A shaded depth map, so the
    shape at a contour level reads without a 3D renderer."""
    from scipy import ndimage
    m = np.moveaxis(np.asarray(v, np.float32), ax, 0)
    above = m > level
    hit = above.any(0)
    if not hit.any():
        return np.full(m.shape[1:], np.nan)
    i = above.argmax(0)
    vi = np.take_along_axis(m, i[None], 0)[0]
    vp = np.take_along_axis(m, np.maximum(i - 1, 0)[None], 0)[0]
    depth = np.where(i > 0, i - 1 + (level - vp) / np.maximum(vi - vp, 1e-12), 0.0)
    far = float(depth[hit].max()) + 3.0
    d = ndimage.gaussian_filter(np.where(hit, depth, far), 0.7)
    gy, gx = np.gradient(d)
    nrm = np.stack([gx, gy, np.ones_like(d)])
    nrm /= np.sqrt((nrm ** 2).sum(0))
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    shade = np.clip((nrm * light[:, None, None]).sum(0), 0, 1)
    lo, hi = float(depth[hit].min()), float(depth[hit].max())
    cue = 1 - 0.35 * (depth - lo) / max(hi - lo, 1e-6)
    return np.where(hit, (0.22 + 0.78 * shade) * cue, np.nan)


def crop_views(mean: np.ndarray, *vols, mask: np.ndarray | None = None, margin: float = 0.06):
    """The maps cut to a cube about the box centre that holds the mask (> 0.5; without one the mean's top 10 % of
    voxels) plus a margin: the particle fills the panels instead of a third of them. Returns (mean, *vols, mask)."""
    b = mean.shape[0]
    sel = (mask > 0.5) if mask is not None and mask.shape == mean.shape else mean > float(np.quantile(mean, 0.9))
    idx = np.argwhere(sel)
    half = int(np.abs(idx - b // 2).max() + max(2, margin * b)) if len(idx) else b
    if half >= b // 2:
        return (mean, *vols, mask)
    sl = (slice(b // 2 - half, b // 2 + half + 1),) * 3
    cut = lambda v: None if v is None else (v[(slice(None),) + sl] if v.ndim == 4 else v[sl])  # noqa: E731
    return (mean[sl], *[cut(v) for v in vols], cut(mask))


def _silhouette(v: np.ndarray, level: float, ax: int) -> np.ndarray:
    """Outline of a map seen along ax: its projection above 15 % of its peak (what the projection panels show; the
    surface level of a generous mask gave outlines well outside the particle)."""
    p = np.asarray(v, np.float64).sum(ax)
    return (p > 0.15 * max(float(p.max()), 1e-30)).astype(np.float32)


def _outline(a, sil, color, lw=0.7, alpha=1.0):
    if sil.max() > 0 and sil.min() < 1:
        a.contour(sil, levels=[0.5], colors=[color], linewidths=lw, alpha=alpha, origin="lower")


def _surface_cmap():
    from .plotstyle import GRAY
    return GRAY.with_extremes(bad=(0, 0, 0, 0))


def _mode_label(k: int, sd_rel: float, cc, res_a=None):
    """Row label of a mode, its reproducibility tag and the tag's colour."""
    from .plotstyle import STATUS, T
    st = status(cc)
    lines = [f"mode {k}", f"sd {sd_rel * 100:.1f} % of |mean|"]
    lines.append(f"halves |cc| {cc:.2f}" if cc is not None and cc == cc else "halves not compared")
    if res_a:
        lines.append(f"shown to {res_a:.0f} Å")
    col = STATUS["good"] if st == "reproducible" else STATUS["serious"] if st.startswith("not repro") else T.ink2
    return "\n".join(lines), st, col


# ------------------------------------------------------------------------------------------------------- figures
def components_figure(path: str, mean: np.ndarray, basis: np.ndarray, sd: np.ndarray, sd_rel: list, half_cc,
                      title_: str, rng: float = 2.0, mask: np.ndarray | None = None, shown_res=None, numbers=None):
    """What moves: per mode (basis already low-passed for display) one row - projections of mean -/+ rng sd along the
    mode's strongest axis at matched contrast, their difference (diverging, symmetric) over the mean's outline, and
    where the mode lives (max projection of |mode| over the mean's silhouette)."""
    from .plotstyle import themed
    themed(_draw_components)(path, mean, basis, sd, sd_rel, half_cc, title_, rng, mask, shown_res, numbers)


def _draw_components(path, mean, basis, sd, sd_rel, half_cc, title_, rng, mask, shown_res, numbers):
    from .plotstyle import GRAY, T, image_axes, plt, title
    mean, basis, mask = crop_views(mean, basis, mask=mask)
    K = len(basis)
    numbers = numbers or list(range(1, K + 1))
    RH = 2.05
    H = RH * K + 0.95
    fig, axs = plt.subplots(K, 4, figsize=(10.5, H), squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.95 / H, bottom=0.02, wspace=0.05, hspace=0.12)
    lvl = surface_level(mean, mask)
    for k in range(K):
        u = basis[k]
        ax = view_axis(u)
        pm = mean.sum(ax)
        lo = (mean - rng * sd[k] * u).sum(ax)
        hi = (mean + rng * sd[k] * u).sum(ax)
        vmin, vmax = np.percentile(np.concatenate([lo, hi]), [1, 99.7])
        sil = _silhouette(mean, lvl, ax)
        for j, im in enumerate((lo, hi)):
            axs[k, j].imshow(im, cmap=GRAY, vmin=vmin, vmax=vmax, origin="lower")
        d = hi - lo
        L = float(np.percentile(np.abs(d), 99.8)) or 1.0
        a = axs[k, 2]
        a.imshow(d, cmap=T.diverging_cmap, vmin=-L, vmax=L, origin="lower")
        _outline(a, sil, T.ink3, 0.6, 0.9)
        a.text(0.03, 0.03, f"peak change {L / max(float(np.abs(pm).max()), 1e-30) * 100:.1f} %\nof the mean's peak",
               transform=a.transAxes, fontsize=7, color=T.ink2, va="bottom")
        w = np.abs(u).max(ax)
        a = axs[k, 3]
        a.imshow(w, cmap=T.density, vmin=0, vmax=float(np.percentile(w, 99.8)) or 1.0, origin="lower")
        _outline(a, sil, T.ink2, 0.7, 0.9)
        for j in range(4):
            image_axes(axs[k, j])
        axs[k, 0].text(0.03, 0.97, f"view along {AXIS_NAME[ax]}", transform=axs[k, 0].transAxes, fontsize=7,
                       color=T.ink3, va="top")
        txt, st, col = _mode_label(numbers[k], sd_rel[k], None if half_cc is None else half_cc[k],
                                   None if shown_res is None else shown_res[k])
        y = 1 - (0.95 + RH * k + RH / 2) / H
        fig.text(0.012, y + 0.15 / H, txt, fontsize=8.5, color=T.ink, va="center", linespacing=1.35)
        fig.text(0.012, y - 0.6 / H, st, fontsize=8.5, color=col, va="center", fontweight="bold")
    heads = [f"mean − {rng:g} sd", f"mean + {rng:g} sd", "difference (+ minus −)\nred gained · blue lost",
             "where it changes\nmax |mode| (line: the mean)"]
    for j in range(4):
        axs[0, j].set_title(heads[j], fontsize=8, color=T.ink3, fontweight="normal", loc="center")
    title(fig, "What moves", title_)
    fig.savefig(path)
    plt.close(fig)


def frames_figure(path: str, mean: np.ndarray, basis: np.ndarray, sd: np.ndarray, sd_rel: list, half_cc,
                  title_: str, rng: float = 2.0, mask: np.ndarray | None = None, numbers=None, shown_res=None):
    """Per mode two rows of five frames at -rng, -rng/2, 0, +rng/2, +rng sd, the same view: the surface at the mean's
    contour level (the mean's outline drawn on every frame) and the frame minus the mean in projection (diverging,
    limits shared along the row), so the motion reads left to right."""
    from .plotstyle import themed
    themed(_draw_frames)(path, mean, basis, sd, sd_rel, half_cc, title_, rng, mask, numbers, shown_res)


def _draw_frames(path, mean, basis, sd, sd_rel, half_cc, title_, rng, mask, numbers, shown_res):
    from .plotstyle import T, image_axes, plt, title
    mean, basis, mask = crop_views(mean, basis, mask=mask)
    K = len(basis)
    numbers = numbers or list(range(1, K + 1))
    al = np.linspace(-rng, rng, 5)
    RH = 3.9
    H = RH * K + 0.95
    fig, axs = plt.subplots(2 * K, 5, figsize=(10.5, H), squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.95 / H, bottom=0.02, wspace=0.04, hspace=0.12)
    lvl = surface_level(mean, mask)
    cm = _surface_cmap()
    for k in range(K):
        u = basis[k]
        ax = view_axis(u)
        sil = _silhouette(mean, lvl, ax)
        ssil = np.isfinite(surface_view(mean, lvl, ax)).astype(np.float32)
        frames = [mean + a * sd[k] * u for a in al]
        diffs = [(f - mean).sum(ax) for f in frames]
        L = max(float(np.percentile(np.abs(np.stack(diffs)), 99.8)), 1e-30)
        for j, (a_, f, d) in enumerate(zip(al, frames, diffs)):
            s_ = axs[2 * k, j]
            s_.imshow(surface_view(f, lvl, ax), cmap=cm, vmin=0, vmax=1, origin="lower")
            _outline(s_, ssil, T.series[1], 0.8)
            s_.set_title(("mean" if abs(a_) < 1e-9 else f"{a_:+g} sd"), fontsize=8, color=T.ink3,
                         fontweight="normal", loc="center")
            d_ = axs[2 * k + 1, j]
            d_.imshow(d, cmap=T.diverging_cmap, vmin=-L, vmax=L, origin="lower")
            _outline(d_, sil, T.ink3, 0.6, 0.9)
            image_axes(s_)
            image_axes(d_)
        axs[2 * k, 0].text(0.03, 0.97, f"surface, view along {AXIS_NAME[ax]}", transform=axs[2 * k, 0].transAxes,
                           fontsize=7, color=T.ink3, va="top")
        axs[2 * k + 1, 0].text(0.03, 0.97, "frame − mean (projection)", transform=axs[2 * k + 1, 0].transAxes,
                               fontsize=7, color=T.ink3, va="top")
        txt, st, col = _mode_label(numbers[k], sd_rel[k], None if half_cc is None else half_cc[k],
                                   None if shown_res is None else shown_res[k])
        y = 1 - (0.95 + RH * k + RH / 2) / H
        fig.text(0.012, y + 0.2 / H, txt, fontsize=8.5, color=T.ink, va="center", linespacing=1.35)
        fig.text(0.012, y - 0.55 / H, st, fontsize=8.5, color=col, va="center", fontweight="bold")
        fig.text(0.012, y - 0.85 / H, "orange line: the mean's surface", fontsize=7, color=T.ink3, va="center")
    title(fig, "Frames along each mode", title_)
    fig.savefig(path)
    plt.close(fig)


def density_share(u: np.ndarray, mean: np.ndarray, level: float, apix: float, near_a: float = 3.0) -> float:
    """Share of a mode's power (sum u^2) on the particle: inside the mean's surface at level, grown by near_a. The
    rest sits in solvent or at the mask's edge - artefacts rather than a change of the particle."""
    from scipy import ndimage
    on = ndimage.binary_dilation(np.asarray(mean) > level, iterations=max(1, int(round(near_a / apix))))
    p = np.asarray(u, np.float64) ** 2
    return float(p[on].sum() / max(p.sum(), 1e-300))


def projections_figure(path: str, mean: np.ndarray, basis: np.ndarray, sd: np.ndarray, sd_rel: list, half_cc,
                       title_: str, apix: float, rng: float = 2.0, mask: np.ndarray | None = None, numbers=None,
                       shown_res=None):
    """Where the differences are: per mode one row of the projected difference (mean + rng sd) - (mean - rng sd)
    along z, y and x, one diverging scale shared by all modes (amplitudes compare across rows), the mean's outline in
    each panel and the share of the mode's power on the particle (vs solvent / mask edge); last row the mean's
    projections for orientation."""
    from .plotstyle import themed
    themed(_draw_projections)(path, mean, basis, sd, sd_rel, half_cc, title_, apix, rng, mask, numbers, shown_res)


def _draw_projections(path, mean, basis, sd, sd_rel, half_cc, title_, apix, rng, mask, numbers, shown_res):
    from .plotstyle import GRAY, T, image_axes, plt, title
    share = [density_share(basis[k], mean, surface_level(mean, mask), apix) for k in range(len(basis))]
    mean, basis, mask = crop_views(mean, basis, mask=mask)
    K = len(basis)
    numbers = numbers or list(range(1, K + 1))
    lvl = surface_level(mean, mask)
    diffs = [[2 * rng * float(sd[k]) * basis[k].sum(ax) for ax in range(3)] for k in range(K)]
    L = max(float(np.percentile(np.abs(np.stack([d for row in diffs for d in row])), 99.8)), 1e-30)
    peak = max(float(np.abs(mean.sum(ax)).max()) for ax in range(3))
    sils = [_silhouette(mean, lvl, ax) for ax in range(3)]
    RH = 2.15
    H = RH * (K + 1) + 1.15
    fig, axs = plt.subplots(K + 1, 4, figsize=(10.5, H), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 0.55]))
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 1.15 / H, bottom=0.02, wspace=0.06, hspace=0.12)
    for k in range(K):
        for ax in range(3):
            a = axs[k, ax]
            a.imshow(diffs[k][ax], cmap=T.diverging_cmap, vmin=-L, vmax=L, origin="lower")
            _outline(a, sils[ax], T.ink3, 0.6, 0.9)
            image_axes(a)
        b_ = axs[k, 3]
        b_.barh([1, 0], [share[k] * 100, (1 - share[k]) * 100], color=[T.series[0], T.ink3], height=0.6)
        for yy, val in ((1, share[k]), (0, 1 - share[k])):
            b_.text(min(val * 100 + 3, 70), yy, f"{val * 100:.0f} %", va="center", fontsize=7.5, color=T.ink2)
        b_.set_yticks([1, 0], ["on the\nparticle", "solvent /\nmask edge"], fontsize=7.5)
        b_.set_xlim(0, 100)
        b_.set_ylim(-0.6, 1.6)
        b_.tick_params(axis="x", labelsize=7)
        b_.spines["left"].set_visible(False)
        txt, st, col = _mode_label(numbers[k], sd_rel[k], None if half_cc is None else half_cc[k],
                                   None if shown_res is None else shown_res[k])
        y = 1 - (1.15 + RH * k + RH / 2) / H
        fig.text(0.012, y + 0.15 / H, txt, fontsize=8.5, color=T.ink, va="center", linespacing=1.35)
        fig.text(0.012, y - 0.62 / H, st, fontsize=8.5, color=col, va="center", fontweight="bold")
    for ax in range(3):
        a = axs[K, ax]
        a.imshow(mean.sum(ax), cmap=GRAY, origin="lower")
        image_axes(a)
    axs[K, 3].set_visible(False)
    fig.text(0.012, 1 - (1.15 + RH * K + RH / 2) / H, "the mean\n(orientation)", fontsize=8.5, color=T.ink,
             va="center")
    for ax in range(3):
        axs[0, ax].set_title(f"along {AXIS_NAME[ax]}", fontsize=8.5, color=T.ink3, fontweight="normal", loc="center")
    axs[0, 3].set_title("power of the mode", fontsize=8.5, color=T.ink3, fontweight="normal", loc="center")
    title(fig, "Where the modes change the map", title_)
    fig.text(0.012, 1 - 0.58 / H, f"(+{rng:g} sd) − (−{rng:g} sd) in projection, one scale for all modes "
             f"(±{L / peak * 100:.1f} % of the mean's peak projection); red gained · blue lost", fontsize=9,
             color=T.ink2, va="top")
    fig.savefig(path)
    plt.close(fig)


def write_gif(path: str, mean: np.ndarray, u: np.ndarray, sd: float, caption: str, rng: float = 2.0,
              n: int = 7, size: int = 224, mask: np.ndarray | None = None) -> str:
    """An animated GIF along one mode (mean + z sd u, z from -rng to +rng and back, 2n - 2 frames): the projections
    along z, y and x side by side at contrast fixed over the frames, and below them each frame minus the mean
    (diverging, one limit for all views and frames); a caption line on top. A fixed palette (128 greys, 126 steps of
    the diverging map, the background): no quantisation banding, ~1 MB. Dark theme colours (the page's default)."""
    from PIL import Image, ImageDraw, ImageFont
    from .plotstyle import _TOKENS, _theme
    th = _theme("dark")
    mean, u, _ = crop_views(mean, u, mask=mask)
    hexrgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]  # noqa: E731
    pal = []
    for i in range(128):
        pal += [round(i * 255 / 127)] * 3
    pal += [int(round(c * 255)) for x in np.linspace(0, 1, 126) for c in th.diverging_cmap(x)[:3]]
    pal += hexrgb(_TOKENS["dark"]["plane"]) + hexrgb(_TOKENS["dark"]["surface"])      # 254 gap, 255 caption strip
    al = np.concatenate([np.linspace(-rng, rng, n), np.linspace(rng, -rng, n)[1:-1]])
    ends = [[(mean + a * sd * u).sum(ax) for ax in range(3)] for a in (-rng, rng)]
    lim = [np.percentile(np.concatenate([ends[0][ax], ends[1][ax]]), [1, 99.7]) for ax in range(3)]
    dl = max(max(float(np.abs(rng * sd * u.sum(ax)).max()) for ax in range(3)), 1e-30)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    cap_h, gap = 26, 4
    W = 3 * size + 2 * gap

    ims = []
    for a in al:
        canvas = np.full((cap_h + 2 * size + gap, W), 254, np.uint8)
        canvas[:cap_h] = 255
        for ax in range(3):
            p = (mean + a * sd * u).sum(ax)
            g = np.clip((p - lim[ax][0]) / max(lim[ax][1] - lim[ax][0], 1e-30), 0, 1)[::-1]
            gi = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC))
            canvas[cap_h:cap_h + size, ax * (size + gap):ax * (size + gap) + size] = gi // 2
            d = np.clip(a * sd * u.sum(ax) / dl * 0.5 + 0.5, 0, 1)[::-1]
            di = np.asarray(Image.fromarray((d * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC))
            canvas[cap_h + size + gap:, ax * (size + gap):ax * (size + gap) + size] = 128 + (di.astype(np.int32)
                                                                                           * 125 // 255)
        im = Image.fromarray(canvas, "P")
        im.putpalette(pal)
        ImageDraw.Draw(im).text((6, 5), f"{caption} | z {a:+.1f} sd | views along z, y, x; below: frame - mean",
                                fill=127, font=font)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=110, loop=0, optimize=True)
    return path


def variance_figure(path: str, sd_rel: list, half_cc, history: list, title_: str, warnings=(),
                    floor: float = NOISE_FLOOR):
    """Three panels: sd of each mode (% of |mean|; bars marked by reproducibility, the real-data noise floor as a
    line), the half-set |cc| of each mode (0.5 line) and convergence (subspace change per iteration, log scale)."""
    from .plotstyle import themed
    themed(_draw_variance)(path, sd_rel, half_cc, history, title_, list(warnings), floor)


def _draw_variance(path, sd_rel, half_cc, history, title_, warnings, floor):
    from .plotstyle import T, grid_y, plt, title, top, warnings_text
    K = len(sd_rel)
    nw = len(warnings)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.7 + 0.22 * nw))
    fig.subplots_adjust(left=0.07, right=0.98, top=top(fig, 0.8), bottom=(0.5 + 0.22 * nw) / fig.get_figheight(),
                        wspace=0.32)
    x = np.arange(1, K + 1)
    v = np.asarray(sd_rel) * 100
    for k in range(K):
        st = status(None if half_cc is None else half_cc[k])
        if st == "reproducible":
            a1.bar(x[k], v[k], width=0.62, color=T.series[0])
        elif st == "not checked":
            a1.bar(x[k], v[k], width=0.62, color=T.ink3)
        else:
            a1.bar(x[k], v[k], width=0.62, color="none", edgecolor=T.ink3, hatch="////", linewidth=1.0)
    a1.axhline(floor * 100, color=T.ink2, lw=1, ls="--")
    a1.text(K + 0.55, floor * 100, "noise floor, real data", ha="right", va="bottom", fontsize=7, color=T.ink2)
    a1.set_ylim(0, max(float(v.max()) if K else 0, floor * 100) * 1.3)
    a1.set_xlim(0.4, K + 0.6)
    tags = [{"reproducible": "reproducible", "not checked": ""}.get(status(None if half_cc is None else half_cc[k]),
                                                                     "noise") for k in range(K)]
    a1.set_xticks(x, [f"mode {k + 1}\n{tg}" for k, tg in enumerate(tags)], fontsize=8)
    a1.set_ylabel("sd along the mode (% of |mean|)")
    a1.set_title("Size")
    grid_y(a1)
    if half_cc is not None:
        c = np.asarray(half_cc, float)
        a2.bar(x, c, width=0.62, color=[T.series[0] if q >= REPRODUCIBLE else T.ink3 for q in c])
        a2.axhline(REPRODUCIBLE, color=T.ink2, lw=1, ls="--")
        a2.text(K + 0.55, REPRODUCIBLE, "reproducible above", ha="right", va="bottom", fontsize=7, color=T.ink2)
        a2.set_ylim(0, 1)
    else:
        a2.text(0.5, 0.5, "half sets not compared", transform=a2.transAxes, ha="center", color=T.ink2)
    a2.set_xlim(0.4, K + 0.6)
    a2.set_xticks(x)
    a2.set_xlabel("mode")
    a2.set_ylabel("|cc| between the half sets' modes")
    a2.set_title("Reproducibility")
    grid_y(a2)
    if history:
        it = [h["it"] for h in history]
        ov = np.array([1 - h["overlap"] if h["overlap"] == h["overlap"] else np.nan for h in history])
        a3.semilogy(it, np.clip(ov, 1e-8, None), color=T.series[0], lw=1.6)
        a3.set_xlabel("iteration")
        a3.set_ylabel("subspace change (1 − overlap)")
        grid_y(a3)
    a3.set_title("Convergence")
    title(fig, "Modes: size and reproducibility", title_)
    warnings_text(fig, warnings)
    fig.savefig(path)
    plt.close(fig)


def latent_figure(path: str, z: np.ndarray, title_: str, labels: np.ndarray | None = None, edges=None,
                  comp: int | None = None, centres=None, numbers=None, names=None):
    """Top: each coordinate standardised (z / sd) as a histogram with the standard normal drawn over it (one peak for
    a continuous or absent motion; two states show two); clusters drawn per cluster with their centres, bin edges as
    lines. Bottom: 2D densities of pairs (symmetric ranges), or the particles coloured by cluster."""
    from .plotstyle import themed
    themed(_draw_latent)(path, z, title_, labels, edges, comp, centres, numbers, names)


def _draw_latent(path, z, title_, labels, edges, comp, centres, numbers, names):
    from .plotstyle import T, grid_y, plt, title, top
    z = np.asarray(z, np.float64)
    n, K = z.shape
    numbers = numbers or list(range(1, K + 1))
    mu, sg = z.mean(0), np.maximum(z.std(0), 1e-12)
    zs = (z - mu) / sg
    pairs = [(0, 1), (0, 2), (1, 2)][: max(0, min(3, K * (K - 1) // 2))]
    ncol = max(min(K, 4), len(pairs), 1)
    fig, axs = plt.subplots(2 if pairs else 1, ncol, figsize=(3.3 * ncol, 6.6 if pairs else 3.3), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.8), bottom=0.08, wspace=0.3, hspace=0.38)
    for a in axs.ravel():
        a.set_visible(False)
    lim = float(np.clip(np.percentile(np.abs(zs), 99.7) * 1.1, 3.0, 8.0)) if n else 4.0
    xs = np.linspace(-lim, lim, 400)
    gauss = np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi)
    bins = np.linspace(-lim, lim, 81)
    bw = bins[1] - bins[0]
    labs = None if labels is None else np.asarray(labels)
    cen = None if centres is None else (np.asarray(centres, np.float64) - mu) / sg
    col = lambda L: T.series[(int(L) + 1) % len(T.series)]  # noqa: E731
    for k in range(min(K, ncol)):
        a = axs[0, k]
        a.set_visible(True)
        a.hist(zs[:, k], bins=bins, density=True, color=T.series[0], alpha=0.45 if labs is not None else 0.85,
               histtype="stepfilled")
        if labs is not None:
            for L in np.unique(labs):
                h, _ = np.histogram(zs[labs == L, k], bins=bins)
                a.stairs(h / (n * bw), bins, color=col(L), lw=1.2)
        a.plot(xs, gauss, color=T.ink, lw=1.2, label="standard normal")
        if cen is not None:
            for c_, row in enumerate(cen):
                a.axvline(row[k], color=col(c_), lw=1, ls=":")
        if edges is not None and comp == k:
            for e in (np.asarray(edges[1:-1]) - mu[k]) / sg[k]:
                a.axvline(e, color=T.series[1], lw=1)
        a.set_xlim(-lim, lim)
        a.set_xlabel(f"z{numbers[k]} / sd   (sd {z[:, k].std():.2f})")
        a.set_title(f"mode {numbers[k]}", fontsize=9.5)
        grid_y(a)
        if k == 0:
            a.legend(loc="upper right", fontsize=7.5)
    for c, (p, q) in enumerate(pairs):
        a = axs[1, c]
        a.set_visible(True)
        if labs is None:
            h, _, _ = np.histogram2d(zs[:, p], zs[:, q], bins=90, range=[[-lim, lim], [-lim, lim]])
            a.imshow(np.log1p(h.T), origin="lower", extent=[-lim, lim, -lim, lim], cmap=T.density, aspect="equal")
            if edges is not None and comp in (p, q):
                for e in (np.asarray(edges[1:-1]) - mu[comp]) / sg[comp]:
                    (a.axvline if comp == p else a.axhline)(e, color=T.series[1], lw=1)
        else:
            sub = np.random.default_rng(0).permutation(n)[:20000]
            for L in np.unique(labs):
                s = sub[labs[sub] == L]
                nm = names[int(L)] if names else f"cluster {int(L) + 1}"
                a.scatter(zs[s, p], zs[s, q], s=2, color=col(L), alpha=0.5, lw=0,
                          label=f"{nm} ({int((labs == L).sum()):,})")
            if cen is not None:
                for c_, row in enumerate(cen):
                    a.scatter([row[p]], [row[q]], s=70, marker="X", color=col(c_), edgecolors=T.ink, linewidths=0.8,
                              zorder=5)
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.set_aspect("equal")
            if c == 0:
                a.legend(markerscale=4, fontsize=7.5, loc="upper left")
        a.axhline(0, color=T.grid, lw=0.6, zorder=0)
        a.axvline(0, color=T.grid, lw=0.6, zorder=0)
        a.set_xlabel(f"z{numbers[p]} / sd")
        a.set_ylabel(f"z{numbers[q]} / sd")
    title(fig, "Particle coordinates along the modes", title_)
    fig.savefig(path)
    plt.close(fig)


def groups_figure(path: str, maps: list, names: list, pops: list, res: list, title_: str, mask=None):
    """ModeMaps: every group's map (bins along a mode or clusters) as a surface at one shared contour level, the
    same view for all, and below it the group's map minus the population-weighted mean (projection, diverging,
    limits shared), so what differs between the groups reads along the row."""
    from .plotstyle import themed
    themed(_draw_groups)(path, maps, names, pops, res, title_, mask)


def _draw_groups(path, maps, names, pops, res, title_, mask):
    from .plotstyle import T, image_axes, plt, title
    ok = [i for i, m in enumerate(maps) if m is not None and np.abs(m).max() > 0]
    if ok:                                       # cut to the particle (all groups alike, from their weighted mean)
        w0 = np.asarray(pops, np.float64)
        avg0 = sum(w0[i] * maps[i] for i in ok) / max(sum(w0[i] for i in ok), 1e-30)
        cut = crop_views(avg0, np.stack([maps[i] for i in ok]), mask=mask)
        maps = list(maps)
        for j, i in enumerate(ok):
            maps[i] = cut[1][j]
        mask = cut[2]
    n = len(maps)
    w = np.asarray(pops, np.float64)
    avg = sum(w[i] * maps[i] for i in ok) / max(sum(w[i] for i in ok), 1e-30) if ok else np.zeros_like(maps[0])
    diffs = {i: maps[i] - avg for i in ok}
    ax = view_axis(list(diffs.values())) if diffs else 0
    lvl = surface_level(avg, mask)
    sil = _silhouette(avg, lvl, ax)
    ssil = np.isfinite(surface_view(avg, lvl, ax)).astype(np.float32)
    pd = {i: d.sum(ax) for i, d in diffs.items()}
    L = max([float(np.percentile(np.abs(d), 99.8)) for d in pd.values()] + [1e-30])
    H = 5.4
    fig, axs = plt.subplots(2, n, figsize=(max(2.1 * n, 7.2), H), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.05 / H, bottom=0.75 / H, wspace=0.05, hspace=0.08)
    cm = _surface_cmap()
    tot = max(1, sum(pops))
    for i in range(n):
        a, b_ = axs[0, i], axs[1, i]
        if i in ok:
            a.imshow(surface_view(maps[i], lvl, ax), cmap=cm, vmin=0, vmax=1, origin="lower")
            _outline(a, ssil, T.series[1], 0.8)
            b_.imshow(pd[i], cmap=T.diverging_cmap, vmin=-L, vmax=L, origin="lower")
            _outline(b_, sil, T.ink3, 0.6, 0.9)
        image_axes(a)
        image_axes(b_)
        a.set_title(names[i], fontsize=9, color=T.ink, fontweight="bold", loc="center")
        r = res[i]
        b_.text(0.5, -0.04, f"{pops[i]:,} ({pops[i] / tot * 100:.0f} %)"
                + (f" · {r:.2f} Å" if r is not None and r == r else " · too few"),
                transform=b_.transAxes, ha="center", va="top", fontsize=8, color=T.ink2)
    axs[0, 0].text(0.03, 0.97, f"surface, view along {AXIS_NAME[ax]}", transform=axs[0, 0].transAxes, fontsize=7,
                   color=T.ink3, va="top")
    fig.text(0.012, 0.12 / H, "orange line: the mean's surface · grey line: the mean's projection", fontsize=7,
             color=T.ink3, va="bottom")
    axs[1, 0].text(0.03, 0.97, "map − mean (projection)\nred more · blue less", transform=axs[1, 0].transAxes,
                   fontsize=7, color=T.ink3, va="top")
    title(fig, "Maps of the groups", title_)
    fig.savefig(path)
    plt.close(fig)


# ========================================================================================================= jobs
@dataclass
class Modes3DSettings:
    n_modes: int = 3                # number of modes (K), solved together
    filter_res_a: float = 8.0       # finest resolution of the modes (A); the particles are loaded only to it
    highpass_a: float = 0.0         # frequencies coarser than this (A) are left out (0: none; e.g. 50 when ice or
                                    # normalisation gradients take the first mode)
    n_iter: int = 20                # EM iterations
    reg_lambda: float = 0.01        # regularisation: lambda x the shell's mean data weight added to every Fourier voxel
                                    # of the modes (larger: weaker modes where few views sample them; 0.01-3 made no
                                    # difference on real data, 30 zeroes them)
    sym: str = "C1"                 # symmetry imposed on the mean and the modes, the user's choice (C1: motions that
                                    # break the consensus symmetry are allowed)
    classes: tuple = ()             # poses from a HeteroRefine: only the particles of these 1-based classes
    per_particle_scale: bool = True  # a contrast scale per particle, fitted against the mean (else ice thickness and
                                    # normalisation differences tend to become a mode)
    half_check: bool = True         # also solve each gold-standard half on its own and report how well every mode
                                    # reproduces between them (noise does not); figures show each mode low-passed to
                                    # where its halves agree; 2x the time
    max_particles: int = 0          # a random subset of this many particles (0: all)
    n_frames: int = 20              # frames of each mode's volume series (ChimeraX: open the .cxc)
    series_range: float = 2.0       # the series runs from -range to +range standard deviations of the coordinate
    diameter_a: float = 0.0         # particle diameter (A) for the mean's sphere mask (0: from the extraction)
    seed: int = 0                   # random seed (starting modes, subset)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2" (the solve runs on the first)


def modes3d_job(project, particles, poses, mask, s: Modes3DSettings, label: str = ""):
    """particles: Extract / Polish / CtfRefine / Select job; poses: the Refine3D / HeteroRefine job whose orientations
    and shifts are used (matched by uid); mask: optional MaskCreate job (default: the refinement's solvent mask, else
    an automatic one from the mean)."""
    from .jobspec import params_of
    p, q = project.job(particles), project.job(poses)
    m = project.job(mask) if mask not in (None, "", 0) else None
    if s.n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    return project.new_job("Modes3D", dict(params_of(s), particles=p.id, poses=q.id, mask=m.id if m else None),
                           [p, q] + ([m] if m else []),
                           label or f"K{s.n_modes}_{s.filter_res_a:g}A_{s.sym}_on{p.id:03d}")


def _posed(project, job, particles, poses, classes, log, max_particles=0, seed=0):
    """jobs._posed_particles, optionally a random subset of them."""
    from .jobs import _posed_particles
    pj, qj = project.job(particles), project.job(poses)
    src, poses_, t_start, half, n_dup, dist = _posed_particles(project, job, pj, qj, tuple(classes), log)
    if max_particles and max_particles < src.N:
        sel = np.sort(np.random.default_rng(seed).choice(src.N, int(max_particles), replace=False))
        src.subset(sel)
        poses_ = {k: v[sel] for k, v in poses_.items()}
        t_start = t_start[sel]
        half = None if half is None else half[sel]
    return pj, qj, src, poses_, t_start, half, n_dup, dist


def _poses_tensors(poses, t_start):
    from .abinitio import euler_to_matrix
    A = euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                        torch.as_tensor(poses["psi"])).float()
    return A, torch.as_tensor(np.asarray(t_start)).float()


def _map_box(src_box: int, b: int, box_a: float, filter_res_a: float) -> int:
    """Box of the written maps: voxels of ~filter_res / 3 (smooth surfaces in ChimeraX), at most the particles' box."""
    return int(max(b, min(src_box, 2 * math.ceil(box_a / (filter_res_a / 3) / 2))))


def mode_outputs(out, mean, basis, sd, mask, apix, sd_rel, half_cc, shown_res, filter_res_a, tag, rng=2.0,
                 n_frames=20, modes=None, write_mrc=None, gif=True) -> dict:
    """The series, what-moves and frames figures and GIFs of modes (1-based numbers, default all) of a solve: basis
    unit-norm and unfiltered; each mode low-passed to shown_res for all of these. Frames: the reproducible modes
    (none: the first, labelled as noise); GIFs: every mode, those that do not reproduce named mode<k>_noise.gif."""
    K = len(basis)
    modes = list(modes or range(1, K + 1))
    ci = np.asarray(modes) - 1
    disp = lowpass_modes(basis[ci], apix, [shown_res[i] for i in ci] if shown_res else None, filter_res_a)
    cc = None if half_cc is None else [half_cc[i] for i in ci]
    sr = [sd_rel[i] for i in ci]
    shown = None if shown_res is None else [shown_res[i] for i in ci]
    full = np.zeros_like(basis)
    full[ci] = disp
    ser = write_series(out, mean, full, sd, mask, apix, "series_", n_frames, rng, modes, write_mrc=write_mrc)
    components_figure(out("what_moves.png"), mean, disp, sd[ci], sr, cc, tag, rng, mask, shown, modes)
    projections_figure(out("projections.png"), mean, disp, sd[ci], sr, cc, tag, apix, rng, mask, modes, shown)
    rep = [j for j in range(len(modes)) if cc is not None and cc[j] >= REPRODUCIBLE] or [0]
    frames_figure(out("frames.png"), mean, disp[rep], sd[ci][rep], [sr[j] for j in rep],
                  None if cc is None else [cc[j] for j in rep], tag, rng, mask, [modes[j] for j in rep],
                  None if shown is None else [shown[j] for j in rep])
    gifs = []
    if gif:                     # every mode, the ones that do not reproduce named and captioned as noise
        for j, k in enumerate(modes):
            noise = cc is not None and cc[j] < REPRODUCIBLE
            cap = (f"mode {k} · sd {sr[j] * 100:.1f} %"
                   + (f" · halves |cc| {cc[j]:.2f}" if cc is not None else "")
                   + (f" · noise (half |cc| < {REPRODUCIBLE})" if noise else ""))
            gifs.append(write_gif(out(f"mode{k}{'_noise' if noise else ''}.gif"), mean, disp[j], float(sd[ci][j]),
                                  cap, rng, mask=mask))
    return dict(series=ser, gifs=gifs)


def modes3d(project, job, s: Modes3DSettings, log=print) -> dict:
    """The work of a Modes3D job: files through job.out(...), the per-particle table in the job, the summary returned
    (not stored: run_modes3d does that)."""
    from .jobs import _session, _write_results, load_mask
    t0 = time.time()
    pj, qj, src, poses, t_start, half, n_dup, dist = _posed(project, job, job.params["particles"], job.params["poses"],
                                                           s.classes, log, s.max_particles, s.seed)
    max_res = max(0.9 * s.filter_res_a, 2.0 * src.apix)
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log, nyquist_margin=1.0)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    n, b, box_a = clf.N, clf.bmax, clf.box_a
    apix = box_a / b
    A, t = _poses_tensors(poses, t_start)
    mref = job.params.get("mask")
    mask, mask_src = None, "automatic (from the mean)"
    if mref:
        mask = torch.as_tensor(load_mask(project, str(mref), b, apix))
        mask_src = project.job(mref).name
    elif qj.find("mask.mrc"):
        mask = torch.as_tensor(load_mask(project, qj.find("mask.mrc"), b, apix))
        mask_src = f"{qj.name} solvent mask"
    log(f"[modes] {n:,} particles of {pj.name} at the poses of {qj.name}; mask: {mask_src}")
    kw = dict(n_comp=s.n_modes, filter_res_a=s.filter_res_a, n_iter=s.n_iter, lam=s.reg_lambda, sym=s.sym,
              highpass_a=s.highpass_a, scale=s.per_particle_scale, seed=s.seed, log=log)
    res = solve(clf, torch.arange(n), A, t, mask=mask, **kw)
    t_solve = time.time() - t0
    mask = res["mask"]
    K = s.n_modes
    rep = None
    if s.half_check:
        h = np.asarray(half) if half is not None else (np.random.default_rng(1).permutation(n) % 2)
        rh = []
        for hh in (0, 1):
            sel = torch.as_tensor(np.flatnonzero(h == hh))
            rh.append(solve(clf, sel, A[sel], t[sel], mask=mask, tag=f" half {hh + 1}", **kw)["basis"])
        rep = half_reproducibility(res["basis"], rh[0], rh[1], box_a)
        log(f"[modes] halves: per mode |cc| {rep['cc']}, agree to {rep['res_a']} A, subspace overlap "
            f"{rep['subspace_overlap']}")
        del rh
    # ---- files: maps on a finer grid than the band's box (smooth surfaces), unit-norm modes
    z, scl = res["z"].numpy(), res["scale"].numpy()
    b_out = _map_box(src.box, b, box_a, s.filter_res_a)
    a_out = box_a / b_out
    mean_o = upsample(res["mean"], b_out).cpu().numpy()
    basis_o = np.stack([upsample(res["basis"][k], b_out).cpu().numpy() for k in range(K)])
    nrm = np.sqrt((basis_o.reshape(K, -1).astype(np.float64) ** 2).sum(1))
    basis_o = (basis_o / np.maximum(nrm, 1e-30)[:, None, None, None]).astype(np.float32)
    sd_o = res["sd"].numpy() * nrm                       # mean + alpha sd u is the same map on either grid
    mask_o = upsample(mask, b_out).clamp(0, 1).cpu().numpy()
    project.write_mrc(job.out("mean.mrc"), mean_o, a_out)
    project.write_mrc(job.out("mask.mrc"), mask_o, a_out)
    for k in range(K):
        project.write_mrc(job.out(f"mode_{k + 1:02d}.mrc"), basis_o[k], a_out)
    _write_results(project, job, src, np.arange(n), dict(
        **{f"mode_{k + 1}": z[:, k].astype(np.float32) for k in range(K)}, scale=scl.astype(np.float32)))
    mnorm = float(np.sqrt(((mean_o.astype(np.float64) * mask_o) ** 2).sum()))
    sd_rel = [float(x) / max(mnorm, 1e-30) for x in sd_o]
    half_cc = rep["cc"] if rep else None
    shown = display_res(half_cc, rep["res_a"], s.filter_res_a) if rep else [float(s.filter_res_a)] * K
    warnings = []
    if half_cc:
        bad = [k + 1 for k in range(K) if half_cc[k] < REPRODUCIBLE]
        if bad:
            warnings.append(f"mode{'s' if len(bad) > 1 else ''} {', '.join(map(str, bad))} "
                            f"{'do' if len(bad) > 1 else 'does'} not reproduce between the half sets (|cc| < "
                            f"{REPRODUCIBLE}): noise rather than structure")
    last = res["history"][-1]["overlap"] if res["history"] else float("nan")
    if last == last and 1 - last > 1e-3:
        warnings.append(f"the modes still changed at the last iteration (1 - overlap {1 - last:.1e}): "
                        "more iterations")
    tag = f"{job.name} · {n:,} particles · {K} modes · {s.filter_res_a:g} Å · {s.sym}"
    outs = mode_outputs(job.out, mean_o, basis_o, sd_o, mask_o, a_out, sd_rel, half_cc, shown, s.filter_res_a, tag,
                        s.series_range, s.n_frames, write_mrc=project.write_mrc)
    variance_figure(job.out("variance.png"), sd_rel, half_cc, res["history"], tag, warnings)
    latent_figure(job.out("latent.png"), z, tag)
    modes_ = [dict(mode=k + 1, sd=round(float(sd_o[k]), 6), sd_rel_mean=round(sd_rel[k], 5),
                   z_sd=round(float(z[:, k].std()), 4), half_cc=None if not half_cc else half_cc[k],
                   status=status(None if not half_cc else half_cc[k]), display_res_a=shown[k],
                   series=project.rel(outs["series"][k]["cxc"])) for k in range(K)]
    summ = dict(particles=int(n), n_modes=K, modes=modes_, filter_res_a=s.filter_res_a,
                reproducible=sum(m["status"] == "reproducible" for m in modes_),
                band_a=[round(x, 2) for x in res["band_a"]], sym=s.sym, mask=mask_src, box=b, apix=round(apix, 4),
                map_box=b_out, map_apix=round(a_out, 4), iterations=s.n_iter,
                final_subspace_change=round(1 - last, 6) if last == last else None,
                half_check=rep, scale_sd=round(float(scl.std()), 4), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), seconds_solve=round(t_solve, 1),
                seconds=round(since(t0), 1), history=res["history"], warnings=warnings,
                gifs=[project.rel(g_) for g_ in outs["gifs"]], mean=project.rel(job.find("mean.mrc")))
    log("[modes] " + ", ".join(f"mode {m['mode']}: sd {m['sd_rel_mean'] * 100:.2f}% of |mean|"
                               + (f", halves |cc| {m['half_cc']:.2f} ({m['status']})" if m['half_cc'] is not None
                                  else "") for m in modes_) + f" ({summ['seconds']:.0f}s)")
    ses.clf = None
    release_cache()
    return summ


@logged
def run_modes3d(project, job, log=print) -> dict:
    from .jobspec import settings_of
    job = project.job(job)
    summ = modes3d(project, job, settings_of(job, Modes3DSettings), log=log)
    project.set_status(job, "finished", summ)
    return summ


# ---------------------------------------------------------------------------------------------- ModeMaps job
KINDS = ("series", "bins", "clusters")


@dataclass
class ModeMapsSettings:
    kind: str = "clusters"          # "series": a volume series along each mode (no particles read); "bins":
                                    # reconstructions of the particles in bins of equal size along one mode;
                                    # "clusters": a Gaussian mixture on the coordinates, one reconstruction per cluster
    modes: tuple = ()               # 1-based modes used (empty: all); bins: the first one listed (else 1)
    n_clusters: int = 3             # clusters: number of mixture components (clusters)
    n_bins: int = 5                 # bins: bins along the mode (equal numbers of particles)
    n_frames: int = 20              # series: frames per mode
    series_range: float = 2.0       # series: from -range to +range standard deviations of the coordinate
    max_res_a: float = 0.0          # reconstructions: finest resolution loaded (0: Nyquist for clusters, the Modes3D
                                    # job's filter resolution for bins)
    sym: str = ""                   # reconstructions: symmetry, the user's choice (empty: the Modes3D job's)
    seed: int = 0                   # random seed (mixture starts)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def mode_maps_job(project, modes3d_, s: ModeMapsSettings, label: str = ""):
    """Outputs of a Modes3D job: a series along each mode, reconstructions in bins along a mode, or of clusters in
    the particles' coordinates. Bins / clusters are classes like HeteroRefine's: Select takes them
    (jobs.select_classes) and Refine3D with poses=this job and classes=(k,) refines one."""
    import dataclasses
    from .jobspec import params_of
    v = project.job(modes3d_)
    if s.kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {s.kind!r}")
    s = dataclasses.replace(s, sym=s.sym or str(v.params.get("sym") or "C1"))
    n = s.n_clusters if s.kind == "clusters" else s.n_bins if s.kind == "bins" else s.n_frames
    return project.new_job("ModeMaps", dict(params_of(s), modes3d=v.id, particles=v.params["particles"],
                                            poses=v.params["poses"]), [v],
                           label or f"{s.kind}{n}_on{v.id:03d}")


def _read_map(path):
    import mrcfile
    with mrcfile.open(path, permissive=True) as f:
        return np.asarray(f.data, np.float32), float(f.voxel_size.x)


def mode_maps(project, job, s: ModeMapsSettings, log=print) -> dict:
    """The work of a ModeMaps job (files through job.out, summary returned)."""
    from . import abinitio as ab
    from . import diagnostics as dg
    from . import plots3d
    from .jobs import _session, _write_results
    from .jobspec import settings_of
    from .masking import _lowpass
    t0 = time.time()
    v = project.job(job.params["modes3d"])
    vs = settings_of(v, Modes3DSettings)
    vsum = project.summary_of(v)
    K = int(vsum.get("n_modes") or vs.n_modes)
    sel_modes = [int(c) for c in (s.modes or range(1, K + 1))]
    if any(c < 1 or c > K for c in sel_modes):
        raise ValueError(f"modes {sel_modes}: {v.name} has modes 1..{K}")
    vm = vsum.get("modes") or []
    half_cc = [m.get("half_cc") for m in vm] if vm and vm[0].get("half_cc") is not None else None
    if s.kind == "series":
        mean, apix = _read_map(v.find("mean.mrc"))
        basis = np.stack([_read_map(v.find(f"mode_{k:02d}.mrc"))[0] for k in range(1, K + 1)])
        sd = np.array([m["sd"] for m in vm])
        mask = _read_map(v.find("mask.mrc"))[0] if v.find("mask.mrc") else None
        outs = mode_outputs(job.out, mean, basis, sd, mask, apix, [m["sd_rel_mean"] for m in vm], half_cc,
                            [m.get("display_res_a") or vs.filter_res_a for m in vm], vs.filter_res_a,
                            f"{job.name} · {v.name}", s.series_range, s.n_frames, sel_modes,
                            write_mrc=project.write_mrc)
        return dict(kind="series", modes3d=v.name, modes=sel_modes, n_frames=s.n_frames,
                    series=[project.rel(x["cxc"]) for x in outs["series"]],
                    gifs=[project.rel(g_) for g_ in outs["gifs"]], seconds=round(time.time() - t0, 1))
    # ---- the particles at their poses, with their coordinates (by uid)
    pj, qj, src, poses, t_start, half, n_dup, dist = _posed(project, job, v.params["particles"], v.params["poses"],
                                                           vs.classes, log)
    lat = project.table(v, "particles").read()
    uid = src.parts["celUid"].to_numpy(np.int64)
    o = np.argsort(lat["uid"])
    pos = np.searchsorted(lat["uid"][o], uid).clip(0, len(o) - 1)
    ok = lat["uid"][o[pos]] == uid
    if not ok.any():
        raise ValueError(f"none of the particles have coordinates in {v.name}")
    if not ok.all():                                 # the Modes3D job ran on a subset
        sel = np.flatnonzero(ok)
        src.subset(sel)
        poses = {k: x[sel] for k, x in poses.items()}
        t_start = t_start[sel]
        half = None if half is None else half[sel]
        pos = pos[sel]
    row = o[pos]
    Z = np.stack([lat[f"mode_{k}"][row] for k in range(1, K + 1)], 1).astype(np.float64)
    n = len(Z)
    extra, edges, g = {}, None, None
    if s.kind == "clusters":
        g = gmm(Z[:, np.asarray(sel_modes) - 1], s.n_clusters, seed=s.seed)
        labels, ng = g["labels"], s.n_clusters
        extra = {f"prob_{k + 1}": g["probs"][:, k].astype(np.float32) for k in range(ng)}
        names = [f"cluster {k + 1}" for k in range(ng)]
        log(f"[modes] {ng} clusters on modes {sel_modes}: "
            + ", ".join(f"{int((labels == k).sum()):,}" for k in range(ng)))
    else:
        labels, edges = quantile_bins(Z[:, sel_modes[0] - 1], s.n_bins)
        ng = s.n_bins
        names = [f"bin {k + 1}" for k in range(ng)]
    max_res = s.max_res_a or (2.0 * src.apix if s.kind == "clusters" else vs.filter_res_a)
    ses, info = _session(project, job, src, vs.diameter_a, max(max_res, 2.0 * src.apix), s.gpus, log,
                         nyquist_margin=1.0, max_c64_gb=1.5)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    b = ab.working_box(clf, n_acc=2, n_ft=2, log=log)
    box_a = clf.box_a
    apix = box_a / b
    A, t = _poses_tensors(poses, t_start)
    hh = np.asarray(half) if half is not None else (np.random.default_rng(1).permutation(n) % 2)
    rec = reconstruct_groups(clf, torch.arange(n), A, t, labels, ng, hh, sym=s.sym, box=b, log=log)
    table, maps, res_g, fsc = [], [], [], {}
    for k, r in enumerate(rec):
        nk = int((labels == k).sum())
        row_ = dict(cls=k + 1, particles=nk, share=round(nk / n, 4), resolution_a=None, resolution_solvent_mask_a=None,
                    coordinates_mean=[round(float(x), 3) for x in Z[labels == k].mean(0)] if nk else None)
        if r is None:
            maps.append(None)
            res_g.append(float("nan"))
            table.append(row_)
            continue
        for nm, vol in (("map", r["map"] * r["mask"]), ("map_nomask", r["map"]), ("mask", r["mask"]),
                        ("half1_unfil", r["h1"]), ("half2_unfil", r["h2"])):
            ab._save_mrc(job.out(f"class{k + 1:02d}_{nm}.mrc"), vol, apix)
        dg.write_fsc(job.out(f"class{k + 1:02d}_fsc.json")[:-5],
                     dict(corrected=r["fsc"].cpu().numpy(), unmasked=r["fsc_unmasked"].cpu().numpy(),
                          sphere_masked=r["fsc_sphere"].cpu().numpy()), box_a,
                     resolution_a=round(r["resolution_a"], 3), title=f"{job.name} {names[k]}")
        fsc[k] = r["fsc"].cpu().numpy()
        # shown at the Modes3D filter resolution, all alike: the differences the modes were sought at
        maps.append(_lowpass(r["map"], apix, max(vs.filter_res_a, 2.2 * apix)).cpu().numpy())
        res_g.append(r["resolution_a"])
        row_.update(resolution_a=round(r["resolution_a"], 2), resolution_solvent_mask_a=round(r["resolution_a"], 2))
        table.append(row_)
    fit = poses["fit"] if "fit" in poses else (poses["maxprob"] if "maxprob" in poses else np.ones(n, np.float32))
    ts = np.asarray(t_start)
    _write_results(project, job, src, np.arange(n), dict(
        rot=poses["rot"].astype(np.float32), tilt=poses["tilt"].astype(np.float32), psi=poses["psi"].astype(np.float32),
        shift_x_a=ts[:, 0].astype(np.float32), shift_y_a=ts[:, 1].astype(np.float32), half=hh.astype(np.uint8),
        fit=np.asarray(fit).astype(np.float32), cls=labels.astype(np.int16),
        **{f"mode_{k}": Z[:, k - 1].astype(np.float32) for k in range(1, K + 1)}, **extra))
    tag = f"{job.name} · {v.name} · {n:,} particles · {s.kind} · {s.sym}"
    centres = np.array([Z[labels == k].mean(0) if (labels == k).any() else np.full(K, np.nan) for k in range(ng)])
    latent_figure(job.out("latent.png"), Z.astype(np.float32), tag, labels=labels if s.kind == "clusters" else None,
                  edges=edges, comp=None if edges is None else sel_modes[0] - 1,
                  centres=centres if s.kind == "clusters" else None, names=names)
    vmask = None
    if v.find("mask.mrc"):
        vmask = upsample(torch.as_tensor(_read_map(v.find("mask.mrc"))[0]), b).clamp(0, 1).numpy()
    pops = [r_["particles"] for r_ in table]
    groups_figure(job.out("map_views.png"), maps, names, pops, res_g, tag, vmask)
    curves = [(names[k], fsc[k]) for k in range(ng) if k in fsc][:3]
    if curves:
        plots3d.fsc_figure(job.out("fsc.png"), curves, box_a, apix, tag)
    if s.kind == "bins":                            # the bin maps as a series as well
        frames = [job.find(f"class{k + 1:02d}_map_nomask.mrc") for k in range(ng) if rec[k] is not None]
        ok_maps = [m_ for m_ in maps if m_ is not None]
        if frames:
            write_cxc(job.out("bins.mrc")[:-4] + ".cxc", frames, surface_level(ok_maps[0], vmask))
    summ = dict(kind=s.kind, modes3d=v.name, particles=int(n), modes=sel_modes, classes=len(table), class_table=table,
                sym=s.sym,
                box=b, apix=round(apix, 4), duplicates_removed=n_dup, seconds=round(time.time() - t0, 1))
    if g is not None:
        summ.update(bic=round(g["bic"], 1), cluster_means=[[round(float(x), 3) for x in m] for m in g["means"]],
                    confident=int((g["probs"].max(1) > 0.9).sum()))
    else:
        summ["bin_edges"] = [round(float(x), 4) for x in edges]
    log("[modes] " + ", ".join(f"{names[r_['cls'] - 1]}: {r_['particles']:,} ({r_['resolution_a']} A)" for r_ in table))
    ses.clf = None
    release_cache()
    return summ


@logged
def run_mode_maps(project, job, log=print) -> dict:
    from .jobspec import settings_of
    job = project.job(job)
    summ = mode_maps(project, job, settings_of(job, ModeMapsSettings), log=log)
    project.set_status(job, "finished", summ)
    return summ


def key(s: dict) -> str:
    """Modes3D key numbers: modes, filter, how many reproduce, sd of the first ones (% of the mean)."""
    from .jobspec import _n, warn_mark
    m = s.get("modes") or []
    if not m or not isinstance(m[0], dict):
        return ""
    sd = " / ".join(f"{(x.get('sd_rel_mean') or 0) * 100:.1f}" for x in m[:3])
    rep = f" · {s['reproducible']} of {len(m)} reproducible" if s.get("half_check") else ""
    return f"K{len(m)} · {_n(s.get('filter_res_a'), 0)} Å{rep} · sd {sd} %" + warn_mark(s)


def maps_key(s: dict) -> str:
    from .jobspec import _n
    from .project import class_table
    c = class_table(s)
    if s.get("kind") == "series":
        return "series"
    if not c:
        return ""
    best = min((x.get("resolution_a") or float("inf")) for x in c)
    return f"{s.get('kind', '')} {len(c)}" + (f" · best {_n(best, 2)} Å" if best < float("inf") else "")
