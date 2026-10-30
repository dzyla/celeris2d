"""3D variability analysis at fixed poses (cryoSPARC's 3DVA, Punjani & Fleet 2021): continuous and discrete
conformational heterogeneity as a few principal components of the particles' 3D density.

Model (probabilistic PCA in Fourier space, particles at their refined poses A_i, shifts t_i and CTF):

    x_i = e(t_i) CTF_i a_i P_{A_i}(mu + sum_k z_ik W_k) + noise,     z_i ~ N(0, I),  noise per shell sigma^2(k)

mu is the mean map and W_1..W_K the basis volumes (re-solved from the particles, so they are in the particles' own
units; the consensus map is not needed); a_i a per-particle scale (ice thickness, normalisation: fitted against the
mean's projection, so a pure change of contrast is not taken for a component). EM:

  E-step  per particle, batched GEMMs: W_i = a_i CTF_i P_{A_i} W (projections of the K basis volumes, frequencies of the
          band only), posterior precision I + Re(W_i^H D W_i) (D = 2 / sigma^2 per shell, half-plane counting),
          E[z_i] and E[z_i z_i^T].
  M-step  the mean and the K basis volumes together: for every 3D Fourier voxel a (K+1) x (K+1) system
          sum_i a_i^2 E[z^ z^T] CTF^2 D  V = sum_i a_i E[z^] CTF D x_i   (z^ = (1, z)), accumulated by trilinear
          back-projection (with every symmetry copy) and solved voxel by voxel, regularised by lambda x the shell's
          mean weight. Then in real space: the basis masked (MaskCreate job / the refinement's solvent mask / an
          automatic one) and low-passed to the filter resolution, the mean to the particle sphere; the basis is
          rotated to its principal axes (orthogonal, ordered by variance - the prior is isotropic, so this changes
          nothing in the model) and the noise per shell re-estimated from the residuals.

Each iteration is one pass over the particles (the spectra the 2D engine holds, cropped to the band's box), so 200k
particles on a 24 GB card are a matter of the band box (8 A on a 340 A box: 98 px, 6 GB of fp16 spectra).
Reported per component: sd (model standard deviation of the coordinate along the unit-norm component, in map units,
and relative to the masked mean's norm), the standard deviation of the posterior means (shrunk towards 0 by the noise)
and, from two independent solves on the gold-standard halves, the correlation of each component between halves
(a component of noise does not reproduce; |cc| > 0.5 is the warning line).

Synthetic checks (tests/test_variability.py, 3000 particles, box 64, CTF, SNR ~0.4): a domain moving with a uniform
parameter -> |r(z_1, truth)| 0.98; two discrete states -> a two-cluster mixture recovers them (100 %); no motion ->
components that do not reproduce between halves (|cc| <= 0.04).

Real data (dz_2025_10, 38.5k particles at the poses of a D2 Refine3D at 5.5 A, 8 A filter, RTX PRO 4000): an
unstructured sd of 15 % of |mean| per component (C1) whatever the data hold, so read the half-set |cc|, not the sd.
A known motion added to those particles (inject positive control, a 20-30 A domain moved by +-10-15 A) is found
(|r| 0.88) only once it carries ~10 % of the particle's density; at 2-6 % it stays below this noise floor. The
analysis is a few minutes per 40k particles (5 s per iteration at a 98 px band box, plus the two half solves);
192k particles: 25 s per iteration, 6 GB of GPU memory besides the spectra (8 GB of fp16 spectra for 200k at this box).
lambda 0.01-3 changes neither the noise floor's structure nor the injected motion's recovery; 30 zeroes the components.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

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
        log(f"[3dva] filter resolution {filter_res_a:.1f} A is beyond the working box's Nyquist "
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
        log(f"[3dva] automatic mask: {float(mask.sum()) * apix ** 3 / 1e6:.2f} M A^3")
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
    log(f"[3dva]{tag} {n:,} particles, {K} components, band {box_a / k_lo:.0f}-{box_a / k_hi:.1f} A ({P} "
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
                            overlap=round(ov, 5), seconds=round(time.time() - t0, 1)))
        log(f"[3dva]{tag} it {it + 1}/{n_iter}: sd / |mean| " + " ".join(f"{float(s) / mnorm * 100:.2f}%" for s in sd)
            + (f", subspace overlap with the last {ov:.4f}" if ov == ov else "") + f" ({time.time() - t0:.1f}s)")
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
        log(f"[3dva] group {k + 1}/{n_groups}: {len(sel):,} particles, {res:.2f} A ({time.time() - t0:.1f}s)")
        release_cache()
    return out


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
                 components=None, write_mrc=None) -> list[dict]:
    """Per component, the series mean + alpha sd u (float16 MRC frames, out(name) gives each path) and a .cxc to play
    it. Returns [{component, frames: [...], cxc}]."""
    write_mrc = write_mrc or _write_mrc
    K = len(basis)
    comps = components or range(1, K + 1)
    res = []
    lvl = surface_level(mean, None if mask is None else mask)
    for k in comps:
        names = []
        for f, (al, v) in enumerate(series(mean, basis, sd, k - 1, n_frames, rng)):
            p = out(f"{prefix}c{k}_f{f:02d}.mrc")
            write_mrc(p, v, apix, "float16")
            names.append(p)
        cxc = os.path.join(os.path.dirname(names[0]), os.path.basename(names[0]).rsplit("_f00.mrc", 1)[0] + ".cxc")
        write_cxc(cxc, names, lvl)
        res.append(dict(component=k, frames=names, cxc=cxc))
    return res


def _write_mrc(path, data, apix, dtype="float32"):
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(data, dtype=np.float16 if dtype == "float16" else np.float32))
        m.voxel_size = apix


# ------------------------------------------------------------------------------------------------------- figures
def variance_figure(path: str, sd_rel: list, zsd: list, half_cc: list | None, history: list, title_: str,
                    warnings=()):
    from .plotstyle import themed
    themed(_draw_variance)(path, sd_rel, zsd, half_cc, history, title_, list(warnings))


def _draw_variance(path, sd_rel, zsd, half_cc, history, title_, warnings):
    """Left: model sd of each component relative to the mean map's norm (bars) with the sd of the particles' posterior
    means (the share of it the data resolve) and the half-set |cc|. Right: convergence (change of the subspace per
    iteration) and the sd per iteration."""
    from .plotstyle import T, grid_y, note, plt, title, top, warnings_text
    K = len(sd_rel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.subplots_adjust(left=0.07, right=0.97, top=top(fig, 0.8), bottom=0.15 + 0.05 * len(warnings), wspace=0.28)
    x = np.arange(1, K + 1)
    a1.bar(x, np.asarray(sd_rel) * 100, color=T.series[0], width=0.6, label="model sd / |mean|")
    a1.set_xticks(x)
    a1.set_xlabel("component")
    a1.set_ylabel("sd along the component (% of |mean|)")
    grid_y(a1)
    for k in range(K):
        s = f"z sd {zsd[k]:.2f}" + (f"\nhalves |cc| {half_cc[k]:.2f}" if half_cc else "")
        a1.text(x[k], sd_rel[k] * 100, s, ha="center", va="bottom", fontsize=7.5, color=T.ink2)
    a1.set_ylim(0, max(1e-6, max(sd_rel) * 100) * 1.35)
    a1.set_title("Variance per component")
    if history:
        it = [h["it"] for h in history]
        ov = [1 - h["overlap"] if h["overlap"] == h["overlap"] else np.nan for h in history]
        a2.semilogy(it, np.clip(ov, 1e-8, None), color=T.ink2, label="1 - subspace overlap with previous")
        a2.set_xlabel("iteration")
        a2.set_ylabel("subspace change")
        grid_y(a2)
        b2 = a2.twinx()
        sds = np.array([h["sd_rel"] for h in history]) * 100
        for k in range(min(K, 3)):
            b2.plot(it, sds[:, k], color=T.series[k], lw=1.2, ls="--", label=f"component {k + 1}")
        b2.legend(loc="center right", fontsize=7.5)
        b2.set_ylabel("sd / |mean| (%), dashed", color=T.ink2)
        b2.spines["right"].set_visible(True)
        a2.set_title("Convergence")
        note(a2, f"last change {ov[-1]:.1e}" if ov[-1] == ov[-1] else "")
    title(fig, "3D variability", title_)
    warnings_text(fig, warnings)
    fig.savefig(path)
    plt.close(fig)


def latent_figure(path: str, z: np.ndarray, title_: str, labels: np.ndarray | None = None, edges=None,
                  comp: int | None = None):
    from .plotstyle import themed
    themed(_draw_latent)(path, z, title_, labels, edges, comp)


def _draw_latent(path, z, title_, labels, edges, comp):
    """Top: histogram of each coordinate (bin edges of intermediates marked). Bottom: 2D densities of the first
    pairs, or with labels (clusters) the particles coloured by cluster."""
    from .plotstyle import T, grid_y, plt, title, top
    n, K = z.shape
    pairs = [(0, 1), (0, 2), (1, 2)][: max(0, min(3, K * (K - 1) // 2))]
    ncol = max(min(K, 4), len(pairs), 1)
    fig, axs = plt.subplots(2 if pairs else 1, ncol, figsize=(3.2 * ncol, 6.0 if pairs else 3.2), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.8), bottom=0.09, wspace=0.3, hspace=0.35)
    for a in axs.ravel():
        a.set_visible(False)
    for k in range(min(K, ncol)):
        a = axs[0, k]
        a.set_visible(True)
        a.hist(z[:, k], bins=80, color=T.series[0], histtype="stepfilled", alpha=0.85)
        if edges is not None and comp == k:
            for e in edges[1:-1]:
                a.axvline(e, color=T.series[1], lw=1)
        a.set_xlabel(f"z{k + 1} (prior sd)")
        grid_y(a)
        a.set_title(f"component {k + 1}", fontsize=9.5)
    lim = np.percentile(np.abs(z), 99.5) * 1.1 if len(z) else 1
    for c, (p, q) in enumerate(pairs):
        a = axs[1, c]
        a.set_visible(True)
        if labels is None:
            h, xe, ye = np.histogram2d(z[:, p], z[:, q], bins=90, range=[[-lim, lim], [-lim, lim]])
            a.imshow(np.log1p(h.T), origin="lower", extent=[-lim, lim, -lim, lim], cmap=T.density, aspect="auto")
        else:
            sub = np.random.default_rng(0).permutation(n)[:20000]
            for L in np.unique(labels):
                s = sub[labels[sub] == L]
                a.scatter(z[s, p], z[s, q], s=2, color=T.series[int(L) % len(T.series)], alpha=0.5, lw=0,
                          label=f"{int(L) + 1}")
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            if c == 0:
                a.legend(markerscale=4, fontsize=7.5, title="cluster", title_fontsize=7.5, loc="upper right")
        a.set_xlabel(f"z{p + 1}")
        a.set_ylabel(f"z{q + 1}")
    title(fig, "Latent coordinates", title_)
    fig.savefig(path)
    plt.close(fig)


def components_figure(path: str, mean: np.ndarray, basis: np.ndarray, sd: np.ndarray, title_: str, rng: float = 2.0):
    from .plotstyle import themed
    themed(_draw_components)(path, mean, basis, sd, title_, rng)


def _draw_components(path, mean, basis, sd, title_, rng):
    """Per component: central sections of the component (diverging: blue where density decreases along +z, red where
    it grows) with the mean's contour, and projections of the series' two ends (mean -/+ rng sd u)."""
    from .plotstyle import GRAY, T, image_axes, norm_img, plt, title
    K, b = len(basis), mean.shape[0]
    H = 1.9 * K + 0.9
    fig, axs = plt.subplots(K, 5, figsize=(10.5, H), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.9 / H, bottom=0.02, wspace=0.04, hspace=0.12)
    lvl = surface_level(mean)
    names = ["section z", "section y", "section x", f"projection z, -{rng:g} sd", f"projection z, +{rng:g} sd"]
    for k in range(K):
        u = basis[k]
        lim = float(np.abs(u).max()) or 1.0
        secs = [(u[b // 2], mean[b // 2]), (u[:, b // 2], mean[:, b // 2]), (u[:, :, b // 2], mean[:, :, b // 2])]
        for j, (s, m) in enumerate(secs):
            a = axs[k, j]
            a.imshow(s, cmap=T.diverging_cmap, vmin=-lim, vmax=lim, origin="lower")
            a.contour(m, levels=[lvl], colors=[T.ink2], linewidths=0.7, origin="lower")
            image_axes(a)
        lo = (mean - rng * sd[k] * u).sum(0)
        hi = (mean + rng * sd[k] * u).sum(0)
        both = norm_img(np.concatenate([lo, hi], 1))
        axs[k, 3].imshow(both[:, :b], cmap=GRAY, vmin=0, vmax=1, origin="lower")
        axs[k, 4].imshow(both[:, b:], cmap=GRAY, vmin=0, vmax=1, origin="lower")
        for j in (3, 4):
            image_axes(axs[k, j])
        axs[k, 0].set_ylabel(f"component {k + 1}", color=T.ink2, fontsize=9)
        for j in range(5):
            if k == 0:
                axs[k, j].set_title(names[j], fontsize=8, color=T.ink3, fontweight="normal", loc="center")
    title(fig, "Components", title_)
    fig.savefig(path)
    plt.close(fig)


# ========================================================================================================= jobs
@dataclass
class Variability3DSettings:
    n_components: int = 3           # number of components (K), solved together
    filter_res_a: float = 8.0       # finest resolution of the components (A); the particles are loaded only to it
    highpass_a: float = 0.0         # frequencies coarser than this (A) are left out (0: none; e.g. 50 when ice or
                                    # normalisation gradients take the first component)
    n_iter: int = 20                # EM iterations
    reg_lambda: float = 0.01        # regularisation: lambda x the shell's mean data weight added to every Fourier voxel
                                    # of the components (larger: weaker components where few views sample them)
    sym: str = "C1"                 # symmetry imposed on the mean and the components, the user's choice (C1: motions
                                    # that break the consensus symmetry are allowed)
    classes: tuple = ()             # poses from a HeteroRefine: only the particles of these 1-based classes
    per_particle_scale: bool = True  # a contrast scale per particle, fitted against the mean (else ice thickness and
                                    # normalisation differences tend to become a component)
    half_check: bool = True         # also solve each gold-standard half on its own and report how well every component
                                    # reproduces between them (noise does not); 2x the time
    max_particles: int = 0          # a random subset of this many particles (0: all)
    n_frames: int = 20              # frames of each component's volume series (ChimeraX: open the .cxc)
    series_range: float = 2.0       # the series runs from -range to +range standard deviations of the coordinate
    diameter_a: float = 0.0         # particle diameter (A) for the mean's sphere mask (0: from the extraction)
    seed: int = 0                   # random seed (starting components, subset)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2" (the solve runs on the first)


def variability3d_job(project, particles, poses, mask, s: Variability3DSettings, label: str = ""):
    """particles: Extract / Polish / CtfRefine / Select job; poses: the Refine3D / HeteroRefine job whose orientations
    and shifts are used (matched by uid); mask: optional MaskCreate job (default: the refinement's solvent mask, else
    an automatic one from the mean)."""
    from .jobspec import params_of
    p, q = project.job(particles), project.job(poses)
    m = project.job(mask) if mask not in (None, "", 0) else None
    if s.n_components < 1:
        raise ValueError("n_components must be at least 1")
    return project.new_job("Variability3D", dict(params_of(s), particles=p.id, poses=q.id, mask=m.id if m else None),
                           [p, q] + ([m] if m else []),
                           label or f"K{s.n_components}_{s.filter_res_a:g}A_{s.sym}_on{p.id:03d}")


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


def variability3d(project, job, s: Variability3DSettings, log=print) -> dict:
    """The work of a Variability3D job: files through job.out(...), the per-particle table in the job, the summary
    returned (not stored: run_variability3d does that)."""
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
    log(f"[3dva] {n:,} particles of {pj.name} at the poses of {qj.name}; mask: {mask_src}")
    kw = dict(n_comp=s.n_components, filter_res_a=s.filter_res_a, n_iter=s.n_iter, lam=s.reg_lambda, sym=s.sym,
              highpass_a=s.highpass_a, scale=s.per_particle_scale, seed=s.seed, log=log)
    res = solve(clf, torch.arange(n), A, t, mask=mask, **kw)
    t_solve = time.time() - t0
    mask = res["mask"]
    K = s.n_components
    cmp_ = None
    if s.half_check:
        h = np.asarray(half) if half is not None else (np.random.default_rng(1).permutation(n) % 2)
        rh = []
        for hh in (0, 1):
            sel = torch.as_tensor(np.flatnonzero(h == hh))
            rh.append(solve(clf, sel, A[sel], t[sel], mask=mask, tag=f" half {hh + 1}", **kw)["basis"])
        cmp_ = dict(halves=compare_bases(rh[0], rh[1]), full_vs_half1=compare_bases(res["basis"], rh[0]),
                    full_vs_half2=compare_bases(res["basis"], rh[1]))
        log(f"[3dva] halves: per component |cc| {cmp_['halves']['cc']}, subspace overlap "
            f"{cmp_['halves']['subspace_overlap']}")
        del rh
    # ---- files: maps on a finer grid than the band's box (smooth surfaces), unit-norm components
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
        project.write_mrc(job.out(f"component_{k + 1:02d}.mrc"), basis_o[k], a_out)
    ser = write_series(job.out, mean_o, basis_o, sd_o, mask_o, a_out, "series_", s.n_frames, s.series_range,
                       write_mrc=project.write_mrc)
    _write_results(project, job, src, np.arange(n), dict(
        **{f"comp_{k + 1}": z[:, k].astype(np.float32) for k in range(K)}, scale=scl.astype(np.float32)))
    mnorm = float(np.sqrt(((mean_o.astype(np.float64) * mask_o) ** 2).sum()))
    sd_rel = [float(x) / max(mnorm, 1e-30) for x in sd_o]
    half_cc = cmp_["halves"]["cc"] if cmp_ else None
    warnings = []
    if half_cc:
        bad = [k + 1 for k in range(K) if half_cc[k] < 0.5]
        if bad:
            warnings.append(f"component{'s' if len(bad) > 1 else ''} {', '.join(map(str, bad))} "
                            f"{'do' if len(bad) > 1 else 'does'} not reproduce between the half sets (|cc| < 0.5): "
                            "noise rather than structure")
    last = res["history"][-1]["overlap"] if res["history"] else float("nan")
    if last == last and 1 - last > 1e-3:
        warnings.append(f"the components still changed at the last iteration (1 - overlap {1 - last:.1e}): "
                        "more iterations")
    tag = f"{job.name} · {n:,} particles · {K} components · {s.filter_res_a:g} Å · {s.sym}"
    variance_figure(job.out("variance.png"), sd_rel, z.std(0).tolist(), half_cc, res["history"], tag, warnings)
    latent_figure(job.out("latent.png"), z, tag)
    components_figure(job.out("components.png"), mean_o, basis_o, sd_o, tag, s.series_range)
    comps = [dict(component=k + 1, sd=round(float(sd_o[k]), 6), sd_rel_mean=round(sd_rel[k], 5),
                  z_sd=round(float(z[:, k].std()), 4), half_cc=None if not half_cc else half_cc[k],
                  series=project.rel(ser[k]["cxc"])) for k in range(K)]
    summ = dict(particles=int(n), n_components=K, components=comps, filter_res_a=s.filter_res_a,
                band_a=[round(x, 2) for x in res["band_a"]], sym=s.sym, mask=mask_src, box=b, apix=round(apix, 4),
                map_box=b_out, map_apix=round(a_out, 4), iterations=s.n_iter,
                final_subspace_change=round(1 - last, 6) if last == last else None,
                half_check=cmp_, scale_sd=round(float(scl.std()), 4), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), seconds_solve=round(t_solve, 1),
                seconds=round(time.time() - t0, 1), history=res["history"], warnings=warnings,
                mean=project.rel(job.find("mean.mrc")))
    log("[3dva] components: " + ", ".join(f"{c['component']}: sd {c['sd_rel_mean'] * 100:.2f}% of |mean|"
                                          + (f", halves |cc| {c['half_cc']:.2f}" if c['half_cc'] is not None else "")
                                          for c in comps) + f" ({summ['seconds']:.0f}s)")
    ses.clf = None
    release_cache()
    return summ


@logged
def run_variability3d(project, job, log=print) -> dict:
    from .jobspec import settings_of
    job = project.job(job)
    summ = variability3d(project, job, settings_of(job, Variability3DSettings), log=log)
    project.set_status(job, "finished", summ)
    return summ


# ------------------------------------------------------------------------------------------------- display job
MODES = ("simple", "intermediates", "cluster")


@dataclass
class VariabilityDisplaySettings:
    mode: str = "cluster"           # "simple": a volume series along each component (no particles read);
                                    # "intermediates": reconstructions of the particles in bins along one component;
                                    # "cluster": a Gaussian mixture in latent space, one reconstruction per cluster
    components: tuple = ()          # 1-based components used (empty: all); intermediates: the first one listed (else 1)
    n_clusters: int = 3             # cluster: number of mixture components (clusters)
    n_bins: int = 5                 # intermediates: bins along the component (equal numbers of particles)
    n_frames: int = 20              # simple: frames per component
    series_range: float = 2.0       # simple: from -range to +range standard deviations of the coordinate
    max_res_a: float = 0.0          # reconstructions: finest resolution loaded (0: Nyquist for clusters, the
                                    # Variability3D job's filter resolution for intermediates)
    sym: str = ""                   # reconstructions: symmetry, the user's choice (empty: the Variability3D job's)
    seed: int = 0                   # random seed (mixture starts)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def variability_display_job(project, variability, s: VariabilityDisplaySettings, label: str = ""):
    """Outputs of a Variability3D job: a series (simple), reconstructions along a component (intermediates) or of
    clusters in latent space (cluster). Clusters / bins are classes like HeteroRefine's: Select takes them
    (jobs.select_classes) and Refine3D with poses=this job and classes=(k,) refines one."""
    import dataclasses
    from .jobspec import params_of
    v = project.job(variability)
    if s.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {s.mode!r}")
    s = dataclasses.replace(s, sym=s.sym or str(v.params.get("sym") or "C1"))
    n = s.n_clusters if s.mode == "cluster" else s.n_bins if s.mode == "intermediates" else s.n_frames
    tag = dict(cluster="k", intermediates="bins", simple="frames")[s.mode]
    return project.new_job("VariabilityDisplay", dict(params_of(s), variability=v.id, particles=v.params["particles"],
                                                      poses=v.params["poses"]), [v],
                           label or f"{s.mode}_{tag}{n}_on{v.id:03d}")


def _read_map(path):
    import mrcfile
    with mrcfile.open(path, permissive=True) as f:
        return np.asarray(f.data, np.float32), float(f.voxel_size.x)


def variability_display(project, job, s: VariabilityDisplaySettings, log=print) -> dict:
    """The work of a VariabilityDisplay job (files through job.out, summary returned)."""
    from . import abinitio as ab
    from . import diagnostics as dg
    from . import plots3d
    from .jobs import _session, _write_results
    from .jobspec import settings_of
    t0 = time.time()
    v = project.job(job.params["variability"])
    vs = settings_of(v, Variability3DSettings)
    vsum = project.summary_of(v)
    K = int(vsum.get("n_components") or vs.n_components)
    comps = [int(c) for c in (s.components or range(1, K + 1))]
    if any(c < 1 or c > K for c in comps):
        raise ValueError(f"components {comps}: {v.name} has components 1..{K}")
    if s.mode == "simple":
        mean, apix = _read_map(v.find("mean.mrc"))
        basis = np.stack([_read_map(v.find(f"component_{k:02d}.mrc"))[0] for k in range(1, K + 1)])
        sd = np.array([c["sd"] for c in vsum["components"]])
        mask = _read_map(v.find("mask.mrc"))[0] if v.find("mask.mrc") else None
        ser = write_series(job.out, mean, basis, sd, mask, apix, "series_", s.n_frames, s.series_range, comps,
                           write_mrc=project.write_mrc)
        ci = np.asarray(comps) - 1
        components_figure(job.out("components.png"), mean, basis[ci], sd[ci], f"{job.name} · {v.name}", s.series_range)
        return dict(mode="simple", variability=v.name, components=comps, n_frames=s.n_frames,
                    series=[project.rel(x["cxc"]) for x in ser], seconds=round(time.time() - t0, 1))
    # ---- the particles at their poses, with their latent coordinates (by uid)
    pj, qj, src, poses, t_start, half, n_dup, dist = _posed(project, job, v.params["particles"], v.params["poses"],
                                                           vs.classes, log)
    lat = project.table(v, "particles").read()
    uid = src.parts["celUid"].to_numpy(np.int64)
    o = np.argsort(lat["uid"])
    pos = np.searchsorted(lat["uid"][o], uid).clip(0, len(o) - 1)
    ok = lat["uid"][o[pos]] == uid
    if not ok.any():
        raise ValueError(f"none of the particles have coordinates in {v.name}")
    if not ok.all():                                 # the Variability3D job ran on a subset
        sel = np.flatnonzero(ok)
        src.subset(sel)
        poses = {k: x[sel] for k, x in poses.items()}
        t_start = t_start[sel]
        half = None if half is None else half[sel]
        pos = pos[sel]
    row = o[pos]
    Z = np.stack([lat[f"comp_{k}"][row] for k in range(1, K + 1)], 1).astype(np.float64)
    n = len(Z)
    extra, edges, g = {}, None, None
    if s.mode == "cluster":
        g = gmm(Z[:, np.asarray(comps) - 1], s.n_clusters, seed=s.seed)
        labels, ng = g["labels"], s.n_clusters
        extra = {f"prob_{k + 1}": g["probs"][:, k].astype(np.float32) for k in range(ng)}
        names = [f"cluster {k + 1}" for k in range(ng)]
        log(f"[3dva] {ng} clusters on components {comps}: "
            + ", ".join(f"{int((labels == k).sum()):,}" for k in range(ng)))
    else:
        labels, edges = quantile_bins(Z[:, comps[0] - 1], s.n_bins)
        ng = s.n_bins
        names = [f"bin {k + 1}" for k in range(ng)]
    max_res = s.max_res_a or (2.0 * src.apix if s.mode == "cluster" else vs.filter_res_a)
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
                    latent_mean=[round(float(x), 3) for x in Z[labels == k].mean(0)] if nk else None)
        if r is None:
            maps.append(np.zeros((b, b, b), np.float32))
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
        maps.append(r["map"].cpu().numpy())
        res_g.append(r["resolution_a"])
        row_.update(resolution_a=round(r["resolution_a"], 2), resolution_solvent_mask_a=round(r["resolution_a"], 2))
        table.append(row_)
    fit = poses["fit"] if "fit" in poses else (poses["maxprob"] if "maxprob" in poses else np.ones(n, np.float32))
    ts = np.asarray(t_start)
    _write_results(project, job, src, np.arange(n), dict(
        rot=poses["rot"].astype(np.float32), tilt=poses["tilt"].astype(np.float32), psi=poses["psi"].astype(np.float32),
        shift_x_a=ts[:, 0].astype(np.float32), shift_y_a=ts[:, 1].astype(np.float32), half=hh.astype(np.uint8),
        fit=np.asarray(fit).astype(np.float32), cls=labels.astype(np.int16),
        **{f"comp_{k}": Z[:, k - 1].astype(np.float32) for k in range(1, K + 1)}, **extra))
    tag = f"{job.name} · {v.name} · {n:,} particles · {s.mode} · {s.sym}"
    latent_figure(job.out("latent.png"), Z.astype(np.float32), tag, labels=labels if s.mode == "cluster" else None,
                  edges=edges, comp=None if edges is None else comps[0] - 1)
    plots3d.class_summary_figure(job.out("map_views.png"), maps, names, [r_["particles"] for r_ in table], res_g, tag)
    curves = [(names[k], fsc[k]) for k in range(ng) if k in fsc][:3]
    if curves:
        plots3d.fsc_figure(job.out("fsc.png"), curves, box_a, apix, tag)
    if s.mode == "intermediates":                   # the bin maps as a series as well
        frames = [job.find(f"class{k + 1:02d}_map_nomask.mrc") for k in range(ng) if rec[k] is not None]
        write_cxc(job.out("bins.mrc")[:-4] + ".cxc", frames, surface_level(maps[0]))
    summ = dict(mode=s.mode, variability=v.name, particles=int(n), components=comps, classes=table, sym=s.sym,
                box=b, apix=round(apix, 4), duplicates_removed=n_dup, seconds=round(time.time() - t0, 1))
    if g is not None:
        summ.update(bic=round(g["bic"], 1), cluster_means=[[round(float(x), 3) for x in m] for m in g["means"]],
                    confident=int((g["probs"].max(1) > 0.9).sum()))
    else:
        summ["bin_edges"] = [round(float(x), 4) for x in edges]
    log("[3dva] " + ", ".join(f"{names[r_['cls'] - 1]}: {r_['particles']:,} ({r_['resolution_a']} A)" for r_ in table))
    ses.clf = None
    release_cache()
    return summ


@logged
def run_variability_display(project, job, log=print) -> dict:
    from .jobspec import settings_of
    job = project.job(job)
    summ = variability_display(project, job, settings_of(job, VariabilityDisplaySettings), log=log)
    project.set_status(job, "finished", summ)
    return summ


def key(s: dict) -> str:
    """Variability3D key numbers: components, filter, sd of the first ones (% of the mean)."""
    from .jobspec import _n, warn_mark
    c = s.get("components") or []
    if not c or not isinstance(c[0], dict):
        return ""
    sd = " / ".join(f"{(x.get('sd_rel_mean') or 0) * 100:.1f}" for x in c[:3])
    return f"K{len(c)} · {_n(s.get('filter_res_a'), 0)} Å · sd {sd} %" + warn_mark(s)


def display_key(s: dict) -> str:
    from .jobspec import _n
    c = s.get("classes") or []
    if s.get("mode") == "simple":
        return "series"
    if not c or not isinstance(c, list):
        return ""
    best = min((x.get("resolution_a") or float("inf")) for x in c)
    return f"{s.get('mode', '')} {len(c)}" + (f" · best {_n(best, 2)} Å" if best < float("inf") else "")
