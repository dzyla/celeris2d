"""Global optics refinement (RELION CtfRefine's "anisotropic magnification", "beam tilt / odd aberrations" and
"even aberrations"; cryoSPARC global CTF refinement) per data set, against a refinement.

Every particle of a Refine3D job is compared with the projection of its own half map (gold standard) at its pose.
The model of a particle's spectrum at image frequency k (cycles / box) is

    m(k) = checker(k) . V(A^T (M k)) . CTF(M k) . exp(i phi_odd(M k)) . exp(-2 pi i k.t / box)

with M the 2x2 magnification (identity + anisotropy), CTF's phase chi + phi_even. Aberrations as phases at the
spatial frequency s = |M k| / box_a (1/A) and its angle theta, in radians at the band's highest frequency s_max:
    odd  = c1 (s/s_max)^3 cos(theta) + c2 (s/s_max)^3 sin(theta)      (coma: beam tilt)
         + c3 (s/s_max)^3 cos(3 theta) + c4 (s/s_max)^3 sin(3 theta)  (trefoil)
    even = e1 (s/s_max)^4 + e2 (s/s_max)^4 cos(2 theta) + e3 (s/s_max)^4 sin(2 theta)
         + e4 (s/s_max)^4 cos(4 theta) + e5 (s/s_max)^4 sin(4 theta)  (spherical error, 4th-order astigmatism,
                                                                        quadrafoil)
All parameters are fitted together by gradient (L-BFGS through grid_sample) to the sum over particles of the
squared, noise-weighted normalised correlation of particle and model (scale-free). Beam tilt in mrad follows from
the coma terms as RELION defines it: phase = 2 pi Cs lambda^2 |k|^2 (k . tau).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .engine import TWO_PI, Optics


def _terms(s: torch.Tensor, th: torch.Tensor, s_max: float):
    """Odd (4, F) and even (5, F) basis phases at frequencies s (1/A), angles th."""
    u = s / s_max
    u3, u4 = u ** 3, u ** 4
    odd = torch.stack([u3 * torch.cos(th), u3 * torch.sin(th), u3 * torch.cos(3 * th), u3 * torch.sin(3 * th)])
    even = torch.stack([u4, u4 * torch.cos(2 * th), u4 * torch.sin(2 * th), u4 * torch.cos(4 * th),
                        u4 * torch.sin(4 * th)])
    return odd, even


def ctf_points(p: torch.Tensor, optics: Optics, kx: torch.Tensor, ky: torch.Tensor, box_a: float,
               odd_c: torch.Tensor | None = None, even_c: torch.Tensor | None = None, s_max: float = 1.0):
    """CTF (n, F) at frequencies (kx, ky) in cycles / box (any points, e.g. magnified): RELION convention as
    engine.ctf_2d, plus even aberration phases (added to chi) and odd ones (a complex factor). p: (n, 6) = defocus
    U, V (A), angle (deg), phase shift (deg), B-factor, scale. Real if there are no odd terms."""
    lam, cs = optics.wavelength, optics.cs_mm * 1e7
    k3 = math.atan(optics.q0 / math.sqrt(1 - optics.q0 ** 2))
    sx, sy = kx / box_a, ky / box_a
    s2 = sx * sx + sy * sy
    ang = torch.atan2(sy, sx)
    du, dv, da, ph, bf, sc = (p[:, i, None] for i in range(6))
    df = 0.5 * (du + dv) + 0.5 * (du - dv) * torch.cos(2 * (ang - torch.deg2rad(da)))
    chi = math.pi * lam * df * s2 - 0.5 * math.pi * cs * lam ** 3 * s2 * s2 + k3 + torch.deg2rad(ph)
    odd = None
    if odd_c is not None or even_c is not None:
        o, e = _terms(torch.sqrt(s2), ang, s_max)
        if even_c is not None:
            chi = chi + torch.tensordot(even_c, e, dims=1)
        if odd_c is not None:
            odd = torch.tensordot(odd_c, o, dims=1)
    out = sc * torch.sin(chi) * torch.exp(-bf * s2 / 4)
    if odd is not None:
        out = out * torch.exp(1j * odd)
    return out


@dataclass
class OpticsFit:
    mag: list            # 2x2
    odd: list            # 4 (rad at s_max)
    even: list           # 5 (rad at s_max)
    s_max: float         # 1/A
    beam_tilt_mrad: list
    score_start: float
    score_end: float

    def tensors(self, dev):
        return (torch.tensor(self.mag, device=dev, dtype=torch.float32),
                torch.tensor(self.odd, device=dev, dtype=torch.float32),
                torch.tensor(self.even, device=dev, dtype=torch.float32))


def fit_optics(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, half: torch.Tensor, vols, res_hi_a: float,
               res_lo_a: float = 10.0, n_max: int = 4000, fit_mag: bool = True, fit_odd: bool = True,
               fit_even: bool = True, iters: int = 40, log=print) -> OpticsFit:
    """Fit magnification and aberrations on up to n_max particles (indices idx into clf, poses A (n, 3, 3), model
    shifts t (n, 2) A, halves 0/1) against Volumes vols[0], vols[1] (half maps; their box b sets the band and the data
    are cropped to it), frequencies res_lo_a .. res_hi_a."""
    dev = clf.dev
    b = vols[0].b
    box_a = clf.box_a
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(idx), generator=g)[:n_max]
    idx, A, t, half = idx[sel], A[sel].to(dev).float(), t[sel].to(dev).float(), half[sel].to(dev)
    fy, fx = vols[0].fy, vols[0].fx
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= box_a / res_lo_a) & (kr <= box_a / res_hi_a)
    kxb = fx[None, :].expand(b, -1)[band]
    kyb = fy[:, None].expand(-1, len(fx))[band]
    s_max = float(box_a / res_hi_a) / box_a
    shell = torch.round(kr[band]).long()
    ph = TWO_PI / box_a
    X = clf._fetch(idx, dev, b)[:, band]                                          # (n, F)
    P0 = clf.ctf_p[idx].to(dev).float()
    shift = torch.exp(-1j * ph * (kxb * t[:, 0, None] + kyb * t[:, 1, None]))
    nsh = int(shell.max()) + 1
    pw = torch.zeros(nsh, device=dev).index_add_(0, shell, (X.abs() ** 2).mean(0))
    cnt = torch.bincount(shell, minlength=nsh).float().clamp_min(1)
    w = 1.0 / (pw / cnt).clamp_min(1e-20)[shell]                                   # noise ~ data power per shell
    X2 = (w * X.abs() ** 2).sum(1)
    params = torch.zeros(4 + 4 + 5, device=dev, requires_grad=True)               # mag delta (4), odd (4), even (5)
    mask_p = torch.tensor([fit_mag] * 4 + [fit_odd] * 4 + [fit_even] * 5, device=dev, dtype=torch.float32)

    def score(prm, chunk=512):
        M = torch.eye(2, device=dev) + prm[:4].reshape(2, 2) * mask_p[:4].reshape(2, 2)
        odd_c, even_c = prm[4:8] * mask_p[4:8], prm[8:] * mask_p[8:]
        kxm, kym = M[0, 0] * kxb + M[0, 1] * kyb, M[1, 0] * kxb + M[1, 1] * kyb
        total = 0.0
        for v in vols:
            v.mag = M
        for i in range(0, len(idx), chunk):
            hh = half[i:i + chunk]
            proj = torch.empty((len(hh), int(band.sum())), dtype=torch.complex64, device=dev)
            parts = []
            for h in (0, 1):
                k = torch.nonzero(hh == h).squeeze(1)
                if len(k):
                    parts.append((k, vols[h].project(A[i:i + chunk][k], band)))
            proj = torch.zeros_like(proj)
            for k, pr in parts:
                proj = proj.index_put((k,), pr)
            c = ctf_points(P0[i:i + chunk], clf.optics, kxm[None], kym[None], box_a, odd_c, even_c, s_max)
            m = proj * c * shift[i:i + chunk]
            num = (w * (X[i:i + chunk].conj() * m).real).sum(1)
            den = (w * m.abs() ** 2).sum(1)
            total = total + (num * num / (den * X2[i:i + chunk]).clamp_min(1e-30)).sum()
        for v in vols:
            v.mag = None
        return total / len(idx)

    fit_optics.score = score                      # for scans / diagnostics: score(params (13,)) -> mean cc^2
    t0 = time.time()
    with torch.no_grad():
        s0 = float(score(params))
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -score(params) * 1e3
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s1 = float(score(params))
    prm = (params.detach() * mask_p).cpu().numpy()
    M = (np.eye(2) + prm[:4].reshape(2, 2)).tolist()
    lam, cs = clf.optics.wavelength, clf.optics.cs_mm * 1e7
    # coma c (rad at s_max) = 2 pi Cs lambda^2 s_max^3 tau  ->  tau (rad) -> mrad
    tau = [float(prm[4 + i]) / (TWO_PI * cs * lam ** 2 * s_max ** 3) * 1e3 for i in (0, 1)]
    fit = OpticsFit(mag=M, odd=prm[4:8].tolist(), even=prm[8:].tolist(), s_max=s_max, beam_tilt_mrad=tau,
                    score_start=s0, score_end=s1)
    ev = np.linalg.eigvalsh(0.5 * (np.array(M) + np.array(M).T))
    log(f"[optics] {len(idx)} particles, {res_lo_a:g}-{res_hi_a:.2f} A: score {s0:.5f} -> {s1:.5f} "
        f"({(s1 / s0 - 1) * 100:+.2f}%); magnification {ev[0]:.4f} / {ev[1]:.4f} (anisotropy "
        f"{(ev[1] - ev[0]) / ev.mean() * 100:.2f}%), beam tilt {tau[0]:+.3f}, {tau[1]:+.3f} mrad, "
        f"trefoil {prm[6]:+.2f} {prm[7]:+.2f}, even {', '.join(f'{x:+.2f}' for x in prm[8:])} rad ({time.time() - t0:.0f} s)")
    return fit
