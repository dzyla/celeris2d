"""3D classification without alignment (RELION Class3D --skip_align, cryoSPARC 3D Classification): the particles
keep the orientations and shifts of a consensus refinement, and K maps are fitted to them by expectation-
maximisation over the class assignment only.

    start    (default) labels from the main directions of variability: every particle's residual against the
             consensus, components found by EM-PCA over all particles at once (Punjani & Fleet 2021), the
             coefficients clustered into K groups. EM from random subsets (RELION) has to amplify noise into the
             first differences and at low SNR often splits by noise instead: simulated 30/70 mixture, noise 5,
             0.85 of particles right vs 0.59 (true maps: 0.90). Random subsets and consensus + noise are options.
    E-step   every particle (or symmetry copy, see below) against every class, at its fixed pose, over the
             frequencies up to res_a: ll_k = -s sum_f |x - CTF P_k e_t|^2 / sigma2(f), s the share of the box inside
             the particles' circular mask (the masked images' independent Fourier coefficients: calibrated
             posteriors); posterior over the classes with the class priors. The noise spectrum sigma2 is the residual
             power of the best class, re-estimated every iteration. From random or noisy starts the first
             assignments are kept soft by a temperature (typical particle ~60/40) that falls to 1 over the first half
             of the iterations.
    M-step   per class two half reconstructions (the particles' gold-standard halves), each particle in its best
             class (hard, default) or weighted by its posterior (soft); the reference is their average, regularised
             with the class's own half-map FSC as RELION's T does: each shell times T SSNR / (1 + T SSNR), SSNR =
             2 FSC / (1 - FSC) - T = 1 is the Wiener filter, larger T trusts the data more (more detail, more
             overfitting; RELION's default for 3D classification is 4).

Focus mask (focused classification): the references are multiplied by the mask, so only that region is compared,
and (subtract=True) every particle first loses the projection of the consensus map outside the mask (signal
subtraction on the fly, at its pose): the rest of the particle no longer pulls the assignment. Particles that are
already subtracted (a Subtract job) take subtract=False. The class maps are always reconstructed from the original
particles, whole.

Symmetry expansion (focused classification of a symmetric particle, RELION's relion_particle_symmetry_expand): every
particle is classified once per symmetry-related pose A @ R (R in the group), each copy independently; with a
focus mask on one subunit each copy looks at a different subunit. The classes are then reconstructed without
symmetry. Symmetry (imposed or expanded) is the user's choice only: never detected or suggested.

Iterations run on a small box (Nyquist just beyond res_a: the E-step uses nothing finer); the final maps and half
maps are reconstructed once at the working box, a few classes at a time (as many as fit the GPU), so the memory
is that of one or two classes whatever K is, and the particle spectra can stay in host memory (200k+ particles).

abinitio.classify3d (the 3D classes of an ab initio run) is this function with the ab initio settings.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .abinitio import (Volume, _free, _live, _real_space_constraints, _regularise_solve, _to_box, auto_pad,
                       fsc_maps, lowpass, resolution_from_fsc, sphere_mask, symmetry_ops)
from .engine import TWO_PI, ctf_2d, freqs, release_cache
from .project import logged


# --------------------------------------------------------------------------------------------- helpers
def rebox(vol: torch.Tensor, b: int, dev, values: bool = False) -> torch.Tensor:
    """A real map in the units of the particle spectra at its box (a reconstruction, or a map scaled to the data),
    Fourier-cropped / padded to box b in those units at b (values=True: the voxel values kept, for masks). A map of
    the same object at box b1 projects to spectra b1 / b times those of the map at b with the same voxel values;
    _to_box multiplies the values by (b1 / b)^3."""
    b1 = vol.shape[0]
    out = _to_box(vol.to(dev).float(), b, dev)
    return out * (b / b1) ** (3 if values else 2)


def _ssnr_filter(fsc_: torch.Tensor, reg_t: float, half_set: bool = False) -> torch.Tensor:
    """Per-shell reference filter T SSNR / (1 + T SSNR), SSNR from the half-set FSC: of the full set (2 FSC / (1 - FSC))
    or, for a half-set reference, of a half (FSC / (1 - FSC))."""
    f = fsc_.nan_to_num(0).clamp(0, 0.999)
    ssnr = (1 if half_set else 2) * f / (1 - f)
    return reg_t * ssnr / (1 + reg_t * ssnr)


def _groups(K: int, per_class: float, budget: float) -> list[list[int]]:
    """Classes in groups whose reconstructions (per_class bytes each) fit the budget: one data pass per group."""
    g = int(max(1, min(K, budget // max(per_class, 1.0))))
    return [list(range(i, min(K, i + g))) for i in range(0, K, g)]


@dataclass
class Class3DResult:
    z: torch.Tensor                 # (R,) best class per row (cpu)
    post: torch.Tensor              # (R, K) posterior at T = 1 with the class priors (cpu)
    A: torch.Tensor                 # (R, 3, 3) pose of each row in its class (the chosen symmetry copy with relax_ops)
    fit: torch.Tensor               # (R,) normalised correlation with the best class's projection (E-step band)
    maps: list                      # per class: FSC-weighted average of the halves, soft sphere (cpu), or None
    halves: list                    # per class: (half 1, half 2) unfiltered (cpu), or None
    fsc: list                       # per class: sphere-masked half-map FSC at the final box (cpu), or None
    res: list                       # per class: its resolution (FSC 0.143, sphere mask), nan if empty
    history: list                   # per iteration: populations, weights, changed, mean max posterior, T, res
    box: int                        # final box
    box_iter: int                   # box of the iterations
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------------------- classification
def classify(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, n_classes: int, res_a: float, *,
             rows: torch.Tensor | None = None, half: torch.Tensor | None = None, sym: str = "C1", n_iter: int = 25,
             z0: torch.Tensor | None = None, init: str = "variability", init_maps: list | None = None,
             init_noise: float = 0.05, reg_t: float | None = 4.0, focus: torch.Tensor | None = None,
             subtract: torch.Tensor | None = None, soft: bool = False, priors: bool = True,
             relax_ops: torch.Tensor | None = None, box: int | None = None, final_box: int | None = None,
             min_change: float = 0.005, seed: int = 0, anneal: bool | None = None, cross_half: bool = False,
             ll_scale: float | None = None, log=print) -> Class3DResult:
    """3D classification of particles idx (host indices into clf) at fixed poses.

    rows: particle (index into idx) of each row (symmetry expansion: every particle several times, with A and t
    per row; rows of one particle next to each other); default one row per particle. A (R, 3, 3), t (R, 2) A.
    half: (n,) gold-standard half of each particle (default a random split, seed). sym: symmetry imposed on the
    class maps. Initialisation: the labels z0; else init_maps (K real maps in the units of the particle spectra at
    their box, e.g. the consensus scaled to the data), each low-passed to res_a with independent noise (init_noise x
    its standard deviation in the sphere); else init="variability" (labels from the main directions of variability
    of the residuals against the consensus, variability_labels) or "random" (random particle subsets). anneal (None:
    for random subsets and init_maps, whose first classes differ by noise only): the first posteriors are kept soft
    (typical particle ~60/40), the temperature falls to 1 over the first half of the iterations. reg_t: RELION's T
    (None: every reference low-passed to res_a instead, the ab initio behaviour). focus: mask (real, any box) applied
    to the references (only that region is compared); subtract: consensus map (units of the particle spectra) whose
    part outside the focus mask is subtracted from every particle at its pose before comparing. soft:
    posterior-weighted reconstructions (else each row in its best class, once annealed); priors: class priors in the
    posterior. cross_half: each row compared with the other half-set's reference only (no self-attraction; measured
    no better). ll_scale: log-likelihood factor (None: the share of the box inside the particles' soft circular mask -
    masked images have that share of independent Fourier coefficients, so the posteriors are calibrated). relax_ops:
    (S, 3, 3) operators of a (pseudo-)symmetry: each row is also tried at A @ R and keeps its best per class (the
    first reconstruction puts every row at all of them). box: box of the iterations (default: Nyquist just beyond
    res_a); final_box: box of the final maps (default clf.bmax)."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    n, K = len(idx), int(n_classes)
    idx = torch.as_tensor(idx).long().cpu()
    rows = torch.arange(n) if rows is None else torch.as_tensor(rows).long().cpu()
    R = len(rows)
    g = torch.Generator().manual_seed(seed + 7)
    if half is None:
        half = torch.randperm(n, generator=g) % 2
    half_r = torch.as_tensor(half).long().cpu()[rows].to(dev)
    A, t = A.to(dev).float(), t.to(dev).float()
    ops = symmetry_ops(sym).to(dev)
    X = torch.eye(3, device=dev)[None] if relax_ops is None else relax_ops.to(dev).float()
    S = len(X)
    kal = box_a / res_a
    bf = final_box or clf.bmax
    be = box or min(bf, 2 * int(math.ceil(kal + 6)))
    ph = TWO_PI / box_a
    fy, fx = freqs(be, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= 1) & (kr <= kal)
    shb = torch.round(kr[band]).long()
    nsh = int(kr.max()) + 2
    fyb = fy[:, None].expand(be, be // 2 + 1)[band]
    fxb = fx[None, :].expand(be, be // 2 + 1)[band]
    Fb = int(band.sum())
    sph = sphere_mask(be, diam / 2 / (box_a / be), 3.0, dev)
    fm = None if focus is None else rebox(focus, be, dev, values=True).clamp(0, 1)
    rmask = sph if fm is None else fm * sph
    Vsub = None
    if subtract is not None:
        if fm is None:
            raise ValueError("subtraction needs a focus mask (the consensus outside it is subtracted)")
        Vsub = Volume(be, dev)
        Vsub.set_real(rebox(subtract, be, dev) * (1 - fm) * sph)
    s_best = torch.zeros((R, K), dtype=torch.long, device=dev)   # chosen relax copy per row and class
    if ll_scale is None:        # share of the box inside the particles' soft circular mask (engine._load)
        B_ = getattr(clf, "B_full", None)
        r_px = clf.radius_a / clf.apix_full if B_ else 0
        ll_scale = min(1.0, math.pi * (r_px + 0.025 * B_) ** 2 / B_ ** 2) if B_ else 1.0
    pad_e = 2
    t0 = time.time()

    def fetch(sl, b):
        """Spectra and CTFs of rows sl at box b (each particle read once, then repeated for its rows)."""
        pr = rows[sl]
        u, inv = torch.unique_consecutive(pr, return_inverse=True)
        j = idx[u]
        x = clf._fetch(j, dev, b)
        fyy, fxx = freqs(b, dev)
        ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fyy, fxx, box_a)
        inv = inv.to(dev)
        return x[inv], ctf[inv]

    def backproject(W, classes, b, kmax, pad, gc, first=False):
        """Half reconstructions (num / den accumulated, then solved) of these classes from weights W (R, K)."""
        vols = {k: [Volume(b, dev, pad=pad, grid_correct=gc), Volume(b, dev, pad=pad, grid_correct=gc)] for k in classes}
        fyy, fxx = freqs(b, dev)
        bp = int(max(32, min(2048, 2e8 // (b * (b // 2 + 1) * 8))))
        if first and S > 1:
            bp = max(8, bp // S)
        for r0 in range(0, R, bp):
            sl = slice(r0, min(R, r0 + bp))
            x, ctf = fetch(sl, b)
            tt = t[sl]
            xs = x * torch.exp(1j * ph * (fxx[None, None, :] * tt[:, 0, None, None] + fyy[None, :, None] * tt[:, 1, None, None]))
            xc, c2 = xs * ctf, ctf * ctf
            del x, xs
            hh = half_r[sl]
            for k in classes:
                w = W[sl, k]
                for h in (0, 1):
                    sel = torch.nonzero((hh == h) & (w > 1e-3)).squeeze(1)
                    if not len(sel):
                        continue
                    ww = w[sel][:, None, None]
                    if first and S > 1:     # every row at all its related poses, so no copy is favoured at the start
                        Ain = (A[sl][sel][:, None] @ X[None]).reshape(-1, 3, 3)
                        vols[k][h].insert((xc[sel] * ww / S).repeat_interleave(S, 0),
                                          (c2[sel] * ww / S).repeat_interleave(S, 0), Ain, kmax=kmax, ops=ops)
                    else:
                        Ain = A[sl][sel] if S == 1 else A[sl][sel] @ X[s_best[sl][sel, k]]
                        vols[k][h].insert(xc[sel] * ww, c2[sel] * ww, Ain, kmax=kmax, ops=ops)
        for k in classes:
            for v in vols[k]:
                _regularise_solve(v, 0.02)
                v.free_acc()
        return vols

    def make_ref(V: Volume, fk, half_set: bool = False) -> Volume:
        r = V
        if reg_t is None:
            lowpass(r, res_a, box_a)
        else:
            r.filter_shells(_ssnr_filter(fk, reg_t, half_set))
        _real_space_constraints(r, rmask, positive=False, center=False)
        return r

    def mstep(W, first=False):
        """References per class [for rows of half 0, for rows of half 1] at box be, and per-class resolution. With
        cross_half each row is compared with the other half's reconstruction: a particle is never compared with a
        reference it is part of (its own noise there pulls it into the class it is already in)."""
        refs, res = [None] * K, [float("nan")] * K
        per = 2 * 160 * (pad_e * be) ** 3 / 8 + 2 * 64 * (pad_e * be) ** 3 / 8
        for grp in _groups(K, per, 0.5 * _free(dev)):
            vols = backproject(W, grp, be, be / 2 - 1, pad_e, False, first)
            for k in grp:
                h0, h1 = vols[k]
                fk = fsc_maps(h0.real() * rmask, h1.real() * rmask)
                if float(W[:, k].sum()) >= 1:
                    res[k] = resolution_from_fsc(fk, box_a)
                if cross_half:
                    refs[k] = [make_ref(Volume(be, dev, ft=h1.ft), fk, True), make_ref(Volume(be, dev, ft=h0.ft), fk, True)]
                else:
                    r_ = make_ref(Volume(be, dev, ft=0.5 * (h0.ft + h1.ft)), fk)
                    refs[k] = [r_, r_]
                vols[k] = None
            del vols
        return refs, res

    def variability_labels(n_comp: int, n_it: int = 10) -> torch.Tensor:
        """Starting labels from the main directions of variability (EM-PCA, Punjani & Fleet 2021): every
        row's residual against the consensus (scaled per row), r_i = x_i - s_i CTF P_C, and components D found one at a
        time by alternating c_i = <r_i, CTF P_D> / |CTF P_D|^2 and D = the reconstruction of c_i r_i (weights c_i^2
        CTF^2) - power iteration on the covariance of the residuals, every particle at once. The coefficients are
        clustered (k-means, K clusters). A mixture whose classes differ by little signal is found from the
        difference all particles share, where EM from random subsets has to amplify noise first: 30/70 mixture at
        noise 5, 3 seeds: 0.54-0.85 from random subsets, see tests/test_class3d.py."""
        Wc = torch.ones((R, 1), device=dev)
        vols = backproject(Wc, [0], be, be / 2 - 1, pad_e, False)
        h0, h1 = vols[0]
        fk = fsc_maps(h0.real() * rmask, h1.real() * rmask)
        C = make_ref(Volume(be, dev, ft=0.5 * (h0.ft + h1.ft)), fk)
        del vols, h0, h1
        Cr = C.real()
        filt = _ssnr_filter(fk, 1.0)               # Wiener: a component cannot be finer than the consensus
        gD = torch.Generator().manual_seed(seed + 13)
        comps, coef = [], torch.zeros((R, n_comp), device=dev)
        isig = torch.ones(Fb, device=dev)
        bp = int(max(32, min(2048, 2e8 // (be * (be // 2 + 1) * 8))))

        def residuals(sl):
            x, ctf = fetch(sl, be)
            tt = t[sl]
            e_t = torch.exp(-1j * ph * (fxb * tt[:, 0, None] + fyb * tt[:, 1, None]))
            cb = ctf[:, band]
            ce = cb * e_t
            xs = x[:, band]
            if Vsub is not None:
                xs = xs - Vsub.project(A[sl], band) * ce
            m0 = C.project(A[sl], band) * ce
            s_ = (xs.conj() * m0).real.mul(isig).sum(1) / (m0.abs() ** 2 * isig).sum(1).clamp_min(1e-20)
            r = xs - s_[:, None] * m0
            for j, Dj in enumerate(comps):
                r = r - coef[sl, j, None] * (Dj.project(A[sl], band) * ce)
            return r, cb, e_t, ce, ctf

        # noise spectrum from the consensus residuals
        rp, rc = torch.zeros(nsh, device=dev), torch.zeros(nsh, device=dev)
        for r0 in range(0, R, bp):
            r, *_ = residuals(slice(r0, min(R, r0 + bp)))
            rp.index_add_(0, shb, (r.abs() ** 2).sum(0))
            rc.index_add_(0, shb, torch.full_like(shb, len(r), dtype=torch.float32))
        isig = 1.0 / (rp / rc.clamp_min(1)).clamp_min(1e-20)[shb]
        for c in range(n_comp):
            Dr = torch.randn((be,) * 3, generator=gD).to(dev)
            for it_ in range(n_it + 1):
                # project out the consensus and earlier components, unit norm, smooth like the consensus
                V = Volume(be, dev)
                V.set_real(Dr * rmask)
                V.filter_shells(filt)
                Dr = V.real() * rmask
                for Q in [Cr] + [q.real() for q in comps]:
                    Dr = Dr - (Dr * Q).sum() / (Q * Q).sum().clamp_min(1e-30) * Q
                Dr = Dr / Dr.norm().clamp_min(1e-30)
                D = Volume(be, dev)
                D.set_real(Dr)
                acc = Volume(be, dev)
                for r0 in range(0, R, bp):
                    sl = slice(r0, min(R, r0 + bp))
                    r, cb, e_t, ce, ctf = residuals(sl)
                    M = D.project(A[sl], band) * ce
                    ci = (r.conj() * M).real.mul(isig).sum(1) / (M.abs() ** 2 * isig).sum(1).clamp_min(1e-20)
                    coef[sl, c] = ci
                    if it_ == n_it:
                        continue
                    xf = torch.zeros((len(r), be, be // 2 + 1), dtype=torch.complex64, device=dev)
                    xf[:, band] = ci[:, None] * cb * e_t.conj() * r
                    acc.insert(xf, (ci[:, None, None] ** 2) * ctf * ctf, A[sl], kmax=min(be / 2 - 1, kal + 2), ops=ops)
                if it_ < n_it:
                    _regularise_solve(acc, 0.02)
                    acc.free_acc()
                    Dr = acc.real()
                del acc
            comps.append(D)
            sd = float(coef[:, c].std())
            log(f"[class3d] variability component {c + 1}: coefficient spread {sd:.3g} ({time.time() - t0:.1f}s)")
        # k-means (k-means++ starts, best of 8) on the coefficients
        Z = coef / coef.std(0).clamp_min(1e-12)
        best, lab = float("inf"), None
        gk = torch.Generator().manual_seed(seed + 17)
        for _ in range(8):
            cen = Z[torch.randint(R, (1,), generator=gk).to(dev)]
            for _k in range(1, K):
                d2 = torch.cdist(Z, cen).min(1).values ** 2
                cen = torch.cat([cen, Z[torch.multinomial(d2 / d2.sum(), 1, generator=None)]])
            for _i in range(50):
                l_ = torch.cdist(Z, cen).argmin(1)
                new = torch.stack([Z[l_ == k].mean(0) if (l_ == k).any() else cen[k] for k in range(K)])
                if torch.allclose(new, cen):
                    break
                cen = new
            inertia = float((torch.cdist(Z, cen).min(1).values ** 2).sum())
            if inertia < best:
                best, lab = inertia, l_
        return lab

    # ---- starting references
    if anneal is None:
        anneal = z0 is None and (init_maps is not None or init != "variability" or S > 1)
    if init_maps is None and z0 is None and init == "variability" and K > 1 and S == 1:
        z0 = variability_labels(min(K - 1, 4))
        log(f"[class3d] starting labels from the variability components: {torch.bincount(z0, minlength=K).tolist()}")
    if init_maps is not None:
        refs = []
        for k in range(K):
            m = rebox(init_maps[k], be, dev)
            if init_noise > 0:
                nz = torch.randn(m.shape, generator=g).to(dev)
                Vn = Volume(be, dev)
                Vn.set_real(nz)
                lowpass(Vn, res_a, box_a)
                nz = Vn.real()
                sd_m = float((m * sph)[sph > 0.5].std())
                m = m + init_noise * sd_m / max(float(nz[sph > 0.5].std()), 1e-12) * nz
            r = Volume(be, dev)
            r.set_real(m)
            lowpass(r, res_a, box_a)
            _real_space_constraints(r, rmask, positive=False, center=False)
            refs.append([r, r])
        z_cur = torch.zeros(R, dtype=torch.long, device=dev)
        res = [float("nan")] * K
        W = torch.full((R, K), 1.0 / K, device=dev)
    else:
        z_cur = (torch.randperm(R, generator=g) % K if z0 is None else torch.as_tensor(z0).long().cpu()).to(dev)
        W = F.one_hot(z_cur, K).float()
        refs, res = mstep(W, first=True)
    pi = torch.full((K,), 1.0 / K, device=dev)
    log(f"[class3d] {R:,} {'rows' if R != n else 'particles'}, {K} classes, E-step to {res_a:.1f} A "
        f"(box {be}, final box {bf}), {'soft' if soft else 'hard'}, T {reg_t if reg_t is not None else '-'}"
        + (", focus mask" if fm is not None else "") + (", signal subtraction" if Vsub is not None else "")
        + (f", {sym}" if sym.upper() != "C1" else "") + f" ({time.time() - t0:.1f}s)")

    def estep(refs, sigma2):
        """Log-likelihoods (R, K), best copy per class, residual power per shell of the best class, fit."""
        isig = (1.0 / sigma2[shb]) if sigma2 is not None else torch.ones(Fb, device=dev)
        ll = torch.empty((R, K), device=dev)
        fit = torch.empty(R, device=dev)
        rp = torch.zeros(nsh, device=dev)
        rc = torch.zeros(nsh, device=dev)
        B = int(max(16, min(8192, 0.25 * _free(dev) / (S * Fb * 8 * (6 + (Vsub is not None))))))
        for r0 in range(0, R, B):
            sl = slice(r0, min(R, r0 + B))
            nb = sl.stop - sl.start
            x, ctf = fetch(sl, be)
            x, ctf = x[:, band], ctf[:, band]
            tt = t[sl]
            ce = ctf * torch.exp(-1j * ph * (fxb * tt[:, 0, None] + fyb * tt[:, 1, None]))     # (nb, F)
            As = A[sl] if S == 1 else (A[sl][:, None] @ X[None]).reshape(-1, 3, 3)
            xs = x[:, None].expand(nb, S, Fb)
            if Vsub is not None:
                xs = xs - Vsub.project(As, band).reshape(nb, S, Fb) * ce[:, None]
            best = torch.full((nb,), -float("inf"), device=dev)
            best_m = torch.zeros((nb, Fb), dtype=torch.complex64, device=dev)
            best_x = torch.zeros((nb, Fb), dtype=torch.complex64, device=dev)
            ar = torch.arange(nb, device=dev)
            hb = half_r[sl]
            by_half = [torch.nonzero(hb == h).squeeze(1) for h in (0, 1)]
            for k in range(K):
                mdl = torch.empty((nb, S, Fb), dtype=torch.complex64, device=dev)
                for h in (0, 1):
                    if len(by_half[h]):
                        mdl[by_half[h]] = refs[k][h].project(As.reshape(nb, S, 3, 3)[by_half[h]].reshape(-1, 3, 3),
                                                             band).reshape(-1, S, Fb)
                mdl = mdl * ce[:, None]
                d2 = ((xs - mdl).abs() ** 2 * isig).sum(-1)                        # (nb, S)
                v, s = (-ll_scale * d2).max(1)
                ll[sl, k] = v
                s_best[sl, k] = s
                score = v + torch.log(pi[k]) if priors else v
                upd = score > best
                if sigma2 is None:                  # noise pass: the residual against the current class
                    upd = z_cur[sl] == k
                best[upd] = score[upd]
                best_m[upd] = mdl[ar[upd], s[upd]]
                best_x[upd] = xs[ar[upd], s[upd]]
                del mdl, d2
            r2 = (best_x - best_m).abs() ** 2
            rp.index_add_(0, shb, r2.sum(0))
            rc.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
            fit[sl] = ((best_x.conj() * best_m).real.sum(1)
                       / ((best_x.abs() ** 2).sum(1) * (best_m.abs() ** 2).sum(1)).sqrt().clamp_min(1e-12))
            del x, ctf, ce, xs, best_m, best_x, r2
        s2 = torch.where(rc > 0, rp / rc.clamp_min(1), torch.zeros_like(rp))
        s2[rc == 0] = s2[rc > 0][-1] if (rc > 0).any() else 1.0
        return ll, fit, s2.clamp_min(1e-20)

    # ---- noise from the starting references (no assignment yet)
    _, _, sigma2 = estep(refs, None)
    history = []
    T0, T, converged, it = None, 1.0, False, 0
    z_prev = None
    fit = torch.zeros(R, device=dev)
    ll = torch.zeros((R, K), device=dev)
    for it in range(1, n_iter + 1):
        ll, fit, sigma2 = estep(refs, sigma2)
        lp = ll + (torch.log(pi)[None] if priors else 0)
        if T0 is None:
            top2 = ll.topk(2, dim=1).values if K > 1 else torch.zeros((R, 2), device=dev)
            T0 = max(1.0, float((top2[:, 0] - top2[:, 1]).nan_to_num(0).median()) / 0.4) if anneal else 1.0
        f_ = min(1.0, (it - 1) / max(1.0, n_iter / 2))
        T = T0 ** (1 - f_)
        post = torch.softmax(ll / T + (torch.log(pi)[None] if priors else 0), dim=1)
        z = lp.argmax(1)
        W = post if soft or T > 1.0 + 1e-6 else F.one_hot(z, K).float()   # hard once annealed: hard from a random
                                                                          # start splits by the noise (0.53 vs 0.97)
        if priors:
            pi = (W.mean(0) + 1e-4 / K)
            pi = pi / pi.sum()
        moved = float((z != z_prev).float().mean()) if z_prev is not None else 1.0
        z_prev = z
        pops = torch.bincount(z, minlength=K).tolist()
        mp = float(post.max(1).values.mean())
        converged = moved < min_change and T <= 1.0 + 1e-6 and it >= 3
        last = converged or it == n_iter
        if not last:
            refs, res = mstep(W)
        # res: each class's half-map FSC at the iteration box, so never finer than its Nyquist (res_limit_a); the
        # final maps (box bf) give the real per-class resolution
        history.append(dict(it=it, pops=pops, weights=[round(float(w), 1) for w in W.sum(0)], changed=round(moved, 4),
                            max_posterior=round(mp, 4), T=round(T, 3),
                            res=[round(r, 2) if r == r else None for r in res], res_limit_a=round(2 * box_a / be, 2),
                            seconds=round(time.time() - t0, 1)))
        log(f"[class3d] it {it}/{n_iter}: per class {pops}, {moved * 100:.1f}% changed, mean max posterior {mp:.3f}, "
            f"T {T:.1f}, res {' / '.join(f'{r:.1f}' if r == r else '-' for r in res)} A (limit {2 * box_a / be:.1f} A at "
            f"box {be}) ({time.time() - t0:.1f}s)")
        if last:
            break
        _live("class3d", it, n_iter, [refs[k] for k in range(K)], res, box_a / be, counts=pops)
    refs = None
    release_cache()
    # ---- final half maps per class at the final box, as many classes per data pass as fit
    post = torch.softmax(ll + (torch.log(pi)[None] if priors else 0), dim=1)
    z = post.argmax(1)
    W = post if soft else F.one_hot(z, K).float()
    sphf = sphere_mask(bf, diam / 2 / (box_a / bf), 3.0, dev)
    maps, halves, fscs, res_f = [None] * K, [None] * K, [None] * K, [float("nan")] * K
    pad = auto_pad(bf, 2, dev)
    gc = pad == 1
    per = 2 * (pad * bf) ** 3 * 20 + 4 * bf ** 3 * 8
    live = [k for k in range(K) if float(W[:, k].sum()) >= 2]
    for grp in _groups(len(live), per, 0.6 * _free(dev)):
        cls_ = [live[i] for i in grp]
        vols = backproject(W, cls_, bf, bf / 2 - 1, pad, gc)
        for k in cls_:
            h0, h1 = vols[k][0].real().nan_to_num(0), vols[k][1].real().nan_to_num(0)
            fk = fsc_maps(h0 * sphf, h1 * sphf)
            res_f[k] = resolution_from_fsc(fk, box_a)
            out = Volume(bf, dev, pad=pad, ft=0.5 * (vols[k][0].ft + vols[k][1].ft), pre=vols[k][0].pre, grid_correct=gc)
            vols[k] = None
            out.filter_shells(torch.sqrt((2 * fk / (1 + fk)).clamp(0, 1).nan_to_num(0)))
            maps[k] = (out.real() * sphf).nan_to_num(0).cpu()
            halves[k] = (h0.cpu(), h1.cpu())
            fscs[k] = fk.cpu()
            del out, h0, h1
        del vols
        release_cache()
    Az = A if S == 1 else A @ X[s_best[torch.arange(R, device=dev), z]]
    log(f"[class3d] final maps (box {bf}): " + ", ".join(
        f"class {k + 1} {int((z == k).sum())} ({res_f[k]:.1f} A)" if res_f[k] == res_f[k] else f"class {k + 1} empty"
        for k in range(K)) + f" ({time.time() - t0:.1f}s)")
    _live("class3d", it, it, [m if m is not None else torch.zeros((bf,) * 3) for m in maps], res_f, box_a / bf,
          counts=torch.bincount(z, minlength=K).tolist())
    return Class3DResult(z=z.cpu(), post=post.cpu(), A=Az.cpu(), fit=fit.cpu(), maps=maps, halves=halves, fsc=fscs,
                         res=res_f, history=history, box=bf, box_iter=be, iterations=it, converged=converged)


# ------------------------------------------------------------------------------------------------ Class3D job
@dataclass
class Class3DSettings:
    n_classes: int = 4              # number of classes (K)
    res_a: float = 8.0              # E-step resolution limit (A): particles are compared with the classes up to this
                                    # resolution only (RELION --strict_highres_exp); the maps are reconstructed finer
    n_iter: int = 25                # iterations (at most; stops once fewer than 0.5% of the particles change class)
    reg_t: float = 4.0              # regularisation T (RELION's tau2_fudge): each class reference is its half-map
                                    # average times T SSNR / (1 + T SSNR) per shell; 1 = Wiener filter, larger trusts
                                    # the data more (more detail, more overfitting of small classes)
    mask: str = ""                  # focus mask: a MaskCreate job (number) or an .mrc path; only the masked region of
                                    # the references is compared. Empty: the whole particle (soft sphere)
    subtract: bool = True           # with a focus mask: subtract the consensus map outside the mask from every particle
                                    # at its pose before comparing (signal subtraction on the fly); off for particles
                                    # that are already subtracted (e.g. a Subtract job)
    init: str = "variability"       # starting classes: "variability" (particles clustered along the main directions of
                                    # variability of their residuals against the consensus, EM-PCA), "random"
                                    # (random particle subsets, RELION) or "consensus" (the consensus map low-passed to
                                    # res_a, independent noise per class). Simulated 30/70 mixture, noise 3 / 4 / 5:
                                    # variability 0.98 / 0.93 / 0.85 of particles right (true maps: 0.98 / 0.94 /
                                    # 0.90), random 0.97 / 0.83 / 0.59 (3 seeds)
    sym: str = "C1"                 # point-group symmetry imposed on the class maps, the user's choice (never detected)
    sym_expand: str = ""            # symmetry expansion of the poses, the user's choice (e.g. "D2"): every particle is
                                    # classified at each of its symmetry-related poses independently (focused
                                    # classification of one subunit of a symmetric particle); classes in C1, the table
                                    # has one row per copy (column copy)
    soft: bool = False              # soft: every particle in every class weighted by its posterior (RELION); hard: in
                                    # its best class only (posteriors are written either way). Hard was better on the
                                    # simulated mixture: 0.93 vs 0.91 (noise 4), 0.85 vs 0.81 (noise 5)
    classes: tuple = ()             # poses from a HeteroRefine / Class3D: only the particles of these 1-based classes
    max_res_a: float = 0.0          # finest resolution loaded and of the final maps / FSC (0: Nyquist of the particles)
    diameter_a: float = 0.0         # particle diameter for the sphere mask (0: from the extraction)
    seed: int = 0                   # random seed (starting subsets, noise, half split if the poses have none)
    sharpen: bool = True            # sharpen the final half maps with the job's own mask and compute their local
                                    # resolution (as a Sharpen job with mask "refine"; no separate job needed)
    local_res: bool = True          # ... including the local resolution
    mtf: str = ""                   # ... detector MTF (RELION .star) for the sharpening; empty: none
    mtf_apix: float = 0.0           # ... the pixel size (A) the MTF refers to (0: the map's)
    gpus: str = "auto"              # GPUs to use: "auto" or a list like "0,2"


def class3d_job(project, particles, poses, reference, s: Class3DSettings, label: str = ""):
    """3D classification without alignment: particles of an Extract / Polish / CtfRefine / Select job at the poses of
    a refinement (Refine3D, HeteroRefine, Class3D, InitialModel; matched by uid; a SymExpand job: every copy, each
    image loaded once - the same rows as sym_expand); reference: the job whose map is the
    consensus (default the poses job; used for init="consensus" and for signal subtraction)."""
    from .jobspec import params_of
    if s.n_classes < 1:
        raise ValueError("n_classes must be at least 1")
    if s.init not in ("variability", "random", "consensus"):
        raise ValueError(f"init must be 'variability', 'random' or 'consensus', not {s.init!r}")
    if s.sym_expand and s.sym.upper() != "C1":
        raise ValueError("symmetry expansion classifies every copy in C1: set sym to C1")
    for grp in (s.sym, s.sym_expand or "C1"):
        symmetry_ops(grp)
    p, q = project.job(particles), project.job(poses)
    if q.type == "SymExpand":           # already expanded: its copies are classified as they are, in C1
        if s.sym_expand:
            raise ValueError(f"{q.name} is already symmetry-expanded: leave sym_expand empty")
        if s.sym.upper() != "C1":
            raise ValueError("symmetry-expanded copies are classified in C1: set sym to C1")
    r = project.job(reference) if reference not in (None, "", 0) else q
    xs = q.params.get("sym") if q.type == "SymExpand" else s.sym_expand
    tag = f"K{s.n_classes}" + ("_focus" if s.mask else "") + (f"_x{str(xs).upper()}" if xs else "")
    ins = [p, q, r]
    return project.new_job("Class3D", dict(params_of(s), particles=p.id, poses=q.id, reference=r.id),
                           [x for i, x in enumerate(ins) if x.id not in [y.id for y in ins[:i]]],
                           label or f"{tag}_on{p.id:03d}")


def _consensus(rj, qj, poses, log):
    """The consensus map of a job before masking (a class job: the class most of these particles are in, if it is
    the poses job, else its class 1)."""
    from .jobs import _reference_map
    from .project import has_classes
    k = 0
    if has_classes(rj):
        mine = rj.id == qj.id and "cls" in poses and len(poses["cls"])
        k = int(np.bincount(poses["cls"].astype(np.int64)).argmax()) + 1 if mine else 1
    vol, apix, path = _reference_map(rj, k)
    log(f"[class3d] consensus map {path} ({vol.shape[0]} px at {apix:.3f} A)")
    return vol, apix


@logged
def run_class3d(project, job, log=print) -> dict:
    from . import abinitio as ab
    from . import diagnostics as dg
    from .jobs import _posed_particles, _scale_to_data, _session, _write_results, load_mask, map_job, map_to_box
    from .jobspec import settings_of
    job = project.job(job)
    s = settings_of(job, Class3DSettings)
    pj, qj = project.job(job.params["particles"]), project.job(job.params["poses"])
    rj = map_job(project, job.params.get("reference") or qj.id)      # (a SymExpand job: its refinement's map)
    # poses with several copies per particle (a SymExpand job): a row per copy, every image loaded once
    src, poses, t_start, half, n_dup, dist, img = _posed_particles(project, job, pj, qj, tuple(s.classes), log,
                                                                   expand=True)
    pre_expanded = "copy" in poses and "celCopy" not in src.parts
    if pre_expanded and s.sym_expand:
        raise ValueError(f"{qj.name} is already symmetry-expanded: leave sym_expand empty")
    max_res = s.max_res_a or 2.0 * src.apix
    t0 = time.time()
    ses, info = _session(project, job, src, s.diameter_a, max_res, s.gpus, log, nyquist_margin=1.0, max_c64_gb=1.5)
    clf = ses.clf
    clf._wait_loaded(clf.N)
    dev, box_a, n, K = clf.dev, clf.box_a, clf.N, int(s.n_classes)
    t_load = time.time() - t0
    # the final maps are made a few classes at a time: the box needs room for one class's two halves
    b = ab.working_box(clf, n_acc=2, n_ft=3, log=log)
    apix = box_a / b
    A = ab.euler_to_matrix(torch.as_tensor(poses["rot"]), torch.as_tensor(poses["tilt"]),
                           torch.as_tensor(poses["psi"])).float()
    t = torch.as_tensor(t_start).float()
    idx = torch.arange(n)
    if half is None:
        half = (torch.randperm(n, generator=torch.Generator().manual_seed(s.seed + 1)) % 2).numpy()
    else:                                   # one half per image (a particle's copies share their particle's)
        h_ = np.zeros(n, np.int64)
        h_[img] = half
        half = h_
    focus = None
    if s.mask:
        focus = torch.as_tensor(load_mask(project, s.mask, b, apix))
        log(f"[class3d] focus mask {s.mask}: {float((focus > 0.5).float().mean()) * 100:.1f}% of the box")
    cons = None
    if s.init == "consensus" or (focus is not None and s.subtract):
        ref, ref_apix = _consensus(rj, map_job(project, qj), poses, log)
        cons = map_to_box(ref, ref_apix, b, apix)
        cons = cons * _scale_to_data(clf, torch.as_tensor(img), A, t, cons, b, max(s.res_a, 10.0))
    rows, A_r, t_r, copy_ = (torch.as_tensor(img), A, t, torch.as_tensor(poses["copy"]).long()) if pre_expanded \
        else (None, A, t, None)
    if s.sym_expand:                        # the same expansion as a SymExpand job's (row i S + k: A_i S_k)
        from .symexpand import expand_poses
        A_r, k_, i_ = expand_poses(A, s.sym_expand)
        S = len(A_r) // n
        rows, t_r, copy_ = torch.as_tensor(img[i_]), t[torch.as_tensor(i_)], torch.as_tensor(k_)
        log(f"[class3d] symmetry expansion {s.sym_expand.upper()}: {n:,} particles x {S} = {len(rows):,} copies")
    elif pre_expanded:
        log(f"[class3d] {len(img):,} symmetry copies of {n:,} particles ({qj.name})")
    r = classify(clf, idx, A_r, t_r, K, s.res_a, rows=rows, half=torch.as_tensor(half), sym=s.sym, n_iter=s.n_iter,
                 init=s.init, init_maps=[cons] * K if s.init == "consensus" else None, reg_t=s.reg_t, focus=focus,
                 subtract=cons if (focus is not None and s.subtract) else None, soft=s.soft, final_box=b,
                 seed=s.seed, log=log)
    seconds = time.time() - t0
    rows_np = np.arange(n) if rows is None else rows.numpy()
    z, post = r.z.numpy(), r.post.numpy()
    rot, tilt, psi = ab.matrix_to_euler(r.A)
    cols = dict(rot=rot.numpy().astype(np.float32), tilt=tilt.numpy().astype(np.float32),
                psi=psi.numpy().astype(np.float32), shift_x_a=t_r[:, 0].numpy().astype(np.float32),
                shift_y_a=t_r[:, 1].numpy().astype(np.float32), cls=z.astype(np.int16),
                fit=r.fit.numpy().astype(np.float32), half=np.asarray(half)[rows_np].astype(np.uint8),
                used=np.ones(len(rows_np), bool), **{f"prob_{k + 1}": post[:, k].astype(np.float32) for k in range(K)})
    if copy_ is not None:
        cols["copy"] = copy_.numpy().astype(np.uint8)
    _write_results(project, job, src, rows_np, cols)
    # per class: solvent mask, corrected FSC, maps (a class too small for an FSC still gets its maps)
    table, fsc_all, res_cls, maps_np = [], {}, [], []
    n_rows = len(rows_np)
    min_n = max(100, int(0.005 * n_rows))
    for k in range(K):
        nk = int((z == k).sum())
        m = r.maps[k]
        r_c = float("nan")
        if m is not None:
            mk = m.to(dev)
            h1, h2 = (h.to(dev) for h in r.halves[k])
            smask = ab.solvent_mask(mk, apix)
            if nk >= min_n:
                f_c = ab.corrected_masked_fsc(h1, h2, smask)
                r_c = ab.resolution_from_fsc(f_c, box_a)
                fsc_all[f"class{k + 1}"] = f_c.cpu().numpy()
                dg.write_fsc(job.out(f"class{k + 1:02d}_fsc.json")[:-5],
                             dict(corrected=f_c.cpu().numpy(), unmasked=fsc_maps(h1, h2).cpu().numpy(),
                                  masked=fsc_maps(h1 * smask, h2 * smask).cpu().numpy(),
                                  sphere_masked=r.fsc[k].numpy()), box_a, resolution_a=round(r_c, 3),
                             title=f"{job.name} class {k + 1}")
            for nm, v in (("map", mk * smask), ("map_nomask", mk), ("mask", smask), ("half1_unfil", h1),
                          ("half2_unfil", h2)):
                ab._save_mrc(job.out(f"class{k + 1:02d}_{nm}.mrc"), v, apix)
            maps_np.append(m.numpy())
            del mk, h1, h2, smask
        else:
            maps_np.append(np.zeros((b, b, b), np.float32))
        res_cls.append(r_c)
        table.append(dict(cls=k + 1, particles=nk, share=round(nk / n_rows, 4), weight=round(float(post[:, k].sum()), 1),
                          confident=int(((z == k) & (post.max(1) > 0.9)).sum()),
                          resolution_a=round(r.res[k], 2) if r.res[k] == r.res[k] and nk >= min_n else None,
                          resolution_solvent_mask_a=round(r_c, 2) if r_c == r_c else None))
    release_cache()
    tag = f"{job.name} · {n:,} particles" + (f" ({n_rows:,} copies, {s.sym_expand.upper()})" if s.sym_expand else "") \
        + f" · {K} classes · {s.res_a:g} Å E-step" + (" · focus" if s.mask else "")
    labels = [f"class {k + 1}" for k in range(K)]
    order = sorted(range(K), key=lambda k: -table[k]["particles"])
    figs = class3d_report(job, tag, s.sym, box_a, apix, r, z, post, maps_np, labels, res_cls, fsc_all, order)
    warnings = []
    big = [k for k in range(K) if table[k]["particles"] >= min_n]
    if K > 1 and len(big) < 2:
        warnings.append(f"only {len(big)} class(es) kept more than {min_n} particles: no heterogeneity found at "
                        f"{s.res_a:g} Å with these settings")
    mmp = float(np.median(post.max(1)))
    if K > 1 and mmp < 0.6:
        warnings.append(f"assignments are uncertain (median max posterior {mmp:.2f}): the classes are hardly "
                        "distinguishable at this resolution / in this mask")
    for row in table:
        row["scf"] = figs["class_views"].get(row["cls"])
    from .jobs import _sharpen_job_maps, class_outputs, class_summary
    sharp = _sharpen_job_maps(project, job, s, classes=tuple(x["cls"] for x in table if x["particles"] >= 100), log=log)
    for row in table:
        sr = sharp.get(str(row["cls"]), {})
        if "resolution_a" in sr:
            row.update(sharpened_resolution_a=sr["resolution_a"], b_factor=sr["b_factor"])
    hist = class_outputs(job, r.history, table, tag)
    summ = dict(particles=int(n), rows=int(n_rows), duplicates_removed=n_dup,
                duplicate_distance_a=round(dist, 1), sym=s.sym, sym_expand=s.sym_expand.upper() or None,
                res_a=s.res_a, reg_t=s.reg_t, mask=s.mask or None, subtract=bool(focus is not None and s.subtract),
                init=s.init, soft=s.soft, apix=round(apix, 4), box=b, box_iter=r.box_iter, iterations=r.iterations,
                converged=r.converged, median_max_posterior=round(mmp, 3), seconds=round(seconds, 1),
                seconds_load=round(t_load, 1), history=r.history, figures=figs["figures"], warnings=warnings,
                **({"sharpened": sharp} if sharp else {}), **class_summary(project, job, table, hist))
    project.set_status(job, "finished", summ)
    log("[class3d] " + ", ".join(f"class {x['cls']}: {x['particles']} ({x['resolution_solvent_mask_a']} A)"
                                 for x in table) + f"; {seconds:.0f}s")
    ses.clf = None
    release_cache()
    return summ


def class3d_report(job, tag, sym, box_a, apix, r: Class3DResult, z, post, maps, labels, res_cls, fsc_all, order) -> dict:
    """Figures of a Class3D job (job: a Job, files through job.out, or a folder): map_views (projections / sections
    per class), progress (populations per iteration, convergence, posterior histogram), fsc (the three largest
    classes; fsc.json holds every class's corrected curve), views_classes (angular distribution per class)."""
    import os
    from . import diagnostics as dg
    from . import plots3d
    from .abinitio import matrix_to_euler
    out_ = job.out if hasattr(job, "out") else (lambda nm: os.path.join(job, nm))
    K = len(maps)
    pops = [int((z == k).sum()) for k in range(K)]
    plots3d.class_summary_figure(out_("map_views.png"), maps, labels, pops, res_cls, tag)
    plots3d.class3d_progress_figure(out_("progress.png"), r.history, post, tag)
    curves = [(labels[k], fsc_all[f"class{k + 1}"]) for k in order if f"class{k + 1}" in fsc_all]
    if curves:
        plots3d.fsc_figure(out_("fsc.png"), curves, box_a, apix, tag,
                           heading="Half-map FSC per class (solvent mask, corrected)")
        dg.write_fsc(out_("fsc.json")[:-5], fsc_all, box_a, title=tag)
    rot, tilt, _ = (x.numpy() for x in matrix_to_euler(r.A))
    big = next((m for m in (maps[k] for k in order) if m is not None), None)      # cylinders start at its surface
    radius = dg.particle_radius(None if big is None else (big > 0.2 * big.max()), apix, box_a)
    dg.write_angdist_bild(out_("angdist.bild"), rot, tilt, box_a, radius)        # ChimeraX, RELION's format
    groups, per = [], {}
    for k in order:
        sel = z == k
        if sel.sum() >= 10:
            st = dg.angular_stats(r.A[torch.as_tensor(sel)], sym)
            groups.append((labels[k], rot[sel], tilt[sel], st))
            dg.write_angdist_bild(out_(f"class{k + 1:02d}_angdist.bild"), rot[sel], tilt[sel], box_a, radius)
            per[k + 1] = st["scf"]
    if groups:
        plots3d.views_grid_figure(out_("views_classes.png"), groups, tag)
    d = os.path.dirname(out_("map_views.png"))
    figs = sorted(f for f in os.listdir(d) if f.endswith(".png") and ".light." not in f)
    return dict(figures=figs, class_views=per)
