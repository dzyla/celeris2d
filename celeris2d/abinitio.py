"""De novo 3D initial model from a Celeris2D classification (PyTorch).

Stage 1 (classes): the 2D class averages are few and clean, so every class is matched against projections of
the current volume over the whole sphere and all in-plane angles (one GEMM), the volume is rebuilt from the
classes at their best orientations, and this is repeated from low to higher resolution. Several random starts,
the best-scoring one wins.

Stage 2 (particles): each particle starts at its class's orientation composed with its own 2D in-plane angle
and shift, and is then refined by a local search (out-of-plane tilt, in-plane angle, shift) against CTF-modulated
projections; gold-standard half maps, FSC-weighted filtering, resolution from the half-map FSC.

Geometry (RELION convention): a 3D orientation is a matrix A (Euler ZYZ rot, tilt, psi); image coordinates are
A @ volume coordinates, so the 2D frequency k = (kx, ky) samples the volume spectrum at A^T (kx, ky, 0).
Volumes are (z, y, x) arrays; spectra are rfft2/fftn with norm="forward", as in the 2D engine.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .engine import TWO_PI, ctf_2d, freqs, rot_grid, upsample2


# --------------------------------------------------------------------------- geometry
def euler_to_matrix(rot, tilt, psi) -> torch.Tensor:
    """RELION Euler angles (deg, tensors of shape (n,)) -> (n, 3, 3)."""
    a, b, g = (torch.deg2rad(torch.as_tensor(v, dtype=torch.float32)) for v in (rot, tilt, psi))
    ca, sa, cb, sb, cg, sg = a.cos(), a.sin(), b.cos(), b.sin(), g.cos(), g.sin()
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return torch.stack([
        torch.stack([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], -1),
        torch.stack([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], -1),
        torch.stack([sc, ss, cb], -1)], -2)


def matrix_to_euler(A: torch.Tensor):
    """(n, 3, 3) -> RELION (rot, tilt, psi) in degrees."""
    A = A.double()
    tilt = torch.arccos(A[:, 2, 2].clamp(-1, 1))
    rot = torch.atan2(A[:, 2, 1], A[:, 2, 0])
    psi = torch.atan2(A[:, 1, 2], -A[:, 0, 2])
    degen = torch.abs(torch.sin(tilt)) < 1e-6   # tilt 0 or 180: only rot+-psi is defined
    if degen.any():
        rot = torch.where(degen, torch.zeros_like(rot), rot)
        psi = torch.where(degen, torch.atan2(-A[:, 1, 0], A[:, 0, 0]), psi)
    return tuple(torch.rad2deg(v).float() for v in (rot, tilt, psi))


def rotz(a_deg: torch.Tensor) -> torch.Tensor:
    """In-plane rotations (n, 3, 3) about the viewing axis."""
    a = torch.deg2rad(torch.as_tensor(a_deg, dtype=torch.float32))
    c, s, z, o = a.cos(), a.sin(), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def sphere_directions(step_deg: float) -> torch.Tensor:
    """Near-uniform viewing directions (Fibonacci sphere), about step_deg apart -> RELION (rot, tilt) in deg."""
    n = max(12, int(round(4 * math.pi / math.radians(step_deg) ** 2)))
    i = torch.arange(n, dtype=torch.float64) + 0.5
    tilt = torch.arccos(1 - 2 * i / n)
    rot = (math.pi * (1 + 5 ** 0.5) * i) % TWO_PI
    return torch.rad2deg(rot).float(), torch.rad2deg(tilt).float()


def local_rotations(step_deg: float, n: int = 1) -> torch.Tensor:
    """Small rotations about the x and y axes (out-of-plane tilts) on a (2n+1)^2 grid of step step_deg, (J, 3, 3)."""
    a = torch.arange(-n, n + 1, dtype=torch.float32) * math.radians(step_deg)
    ax, ay = torch.meshgrid(a, a, indexing="ij")
    ax, ay = ax.reshape(-1), ay.reshape(-1)
    o, z = torch.ones_like(ax), torch.zeros_like(ax)
    Rx = torch.stack([torch.stack([o, z, z], -1), torch.stack([z, ax.cos(), -ax.sin()], -1),
                      torch.stack([z, ax.sin(), ax.cos()], -1)], -2)
    Ry = torch.stack([torch.stack([ay.cos(), z, ay.sin()], -1), torch.stack([z, o, z], -1),
                      torch.stack([-ay.sin(), z, ay.cos()], -1)], -2)
    return Rx @ Ry


def symmetry_ops(sym: str) -> torch.Tensor:
    """Rotation matrices (S, 3, 3) of point group Cn / Dn (RELION axis convention: Cn about z, D 2-fold about x)."""
    sym = sym.upper()
    kind, n = sym[0], int(sym[1:] or 1)
    if kind not in "CD" or n < 1:
        raise ValueError(f"symmetry {sym!r} not supported (Cn or Dn)")
    ops = [rotz(torch.tensor([360.0 * i / n]))[0] for i in range(n)]
    if kind == "D":
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # 2-fold about x
        ops += [o @ flip for o in ops]
    return torch.stack(ops)


# --------------------------------------------------------------------------- Fourier volume
class Volume:
    """A 3D spectrum on a 2x zero-padded, centred grid, in the units of the 2D particle spectra (box b): the
    central slice at orientation A, times the centring phase, is the projection's rfft2 (norm="forward")."""

    def __init__(self, b: int, dev, pad: int = 2):
        self.b, self.pad, self.G, self.dev = b, pad, pad * b, dev
        G = self.G
        self.ft = torch.zeros((2, G, G, G), device=dev)   # real, imag channels; index = frequency + G/2
        self.num = torch.zeros((G * G * G, 2), device=dev)
        self.den = torch.zeros(G * G * G, device=dev)
        fy, fx = freqs(b, dev)
        self.fy, self.fx = fy, fx
        self.checker = (1 - 2 * ((fy[:, None].long() + fx[None, :].long()) % 2)).float()  # centre at b/2
        self.kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # ---- real space <-> spectrum
    def set_real(self, vol: torch.Tensor):
        """vol: (b, b, b) real map, centre at b/2."""
        b, G = self.b, self.G
        v = torch.zeros((G, G, G), device=self.dev)
        o = (G - b) // 2
        v[o:o + b, o:o + b, o:o + b] = vol
        ft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(v), norm="forward")) * (b * self.pad ** 3)
        self.ft = torch.stack([ft.real, ft.imag])

    def real(self) -> torch.Tensor:
        b, G = self.b, self.G
        ft = torch.complex(self.ft[0], self.ft[1]) / (b * self.pad ** 3)
        v = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ft), norm="forward")).real
        o = (G - b) // 2
        return v[o:o + b, o:o + b, o:o + b]

    # ---- slices
    def _coords(self, A: torch.Tensor, kmax: float | None = None):
        """3D grid coordinates (n, b, bh, 3) (x, y, z order) of the 2D half-plane frequencies at orientations A."""
        fy, fx = self.fy, self.fx
        kx = fx[None, None, :, None]
        ky = fy[None, :, None, None]
        At = A.to(self.dev).transpose(1, 2)            # columns: volume axes in image coordinates
        q = kx * At[:, None, None, :, 0] + ky * At[:, None, None, :, 1]   # (n, b, bh, 3) = A^T (kx, ky, 0)
        return q * self.pad

    def project(self, A: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Projection spectra (n, b, b//2+1) at orientations A (n, 3, 3)."""
        out = []
        G = self.G
        for i in range(0, len(A), chunk):
            q = self._coords(A[i:i + chunk])
            g = (q + G // 2 + 0.5) / G * 2 - 1
            s = F.grid_sample(self.ft[None], g[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            out.append(torch.complex(s[0], s[1]) * self.checker)
        return torch.cat(out) if out else torch.zeros((0, self.b, self.b // 2 + 1), dtype=torch.complex64, device=self.dev)

    def insert(self, x: torch.Tensor, w: torch.Tensor, A: torch.Tensor, kmax: float, ops: torch.Tensor | None = None):
        """Accumulate spectra x (n, b, bh) (already CTF-multiplied: x*ctf) with weights w (n, b, bh) (e.g. ctf^2)
        at orientations A, trilinear, plus their Friedel mates; frequencies beyond kmax (cycles/box) are skipped.
        ops: symmetry operators S (each image is also inserted at A @ S)."""
        G = self.G
        if ops is not None and len(ops) > 1:
            S = len(ops)
            A = (A[:, None] @ ops[None]).reshape(-1, 3, 3)
            x, w = x.repeat_interleave(S, 0), w.repeat_interleave(S, 0)
        keep = self.kr <= kmax
        q = self._coords(A)[:, keep]                        # (n, P, 3)
        v = (x * self.checker)[:, keep]
        ww = w[:, keep].float()
        for sgn in (1.0, -1.0):
            p = sgn * q + G // 2
            p0 = torch.floor(p)
            f = p - p0
            p0 = p0.long()
            vr = torch.view_as_real(v if sgn > 0 else v.conj().resolve_conj())
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        ix, iy, iz = p0[..., 0] + dx, p0[..., 1] + dy, p0[..., 2] + dz
                        wt = ((f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
                              * (f[..., 2] if dz else 1 - f[..., 2]))
                        ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G) & (iz >= 0) & (iz < G)
                        idx = ((iz * G + iy) * G + ix)[ok]
                        wt = wt[ok]
                        self.num.index_add_(0, idx, vr[ok] * wt[:, None])
                        self.den.index_add_(0, idx, ww[ok] * wt)

    def clear(self):
        self.num.zero_()
        self.den.zero_()

    def shell_index(self):
        G = self.G
        r = torch.arange(G, device=self.dev, dtype=torch.float32) - G // 2
        rr = torch.sqrt(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2) / self.pad
        return torch.round(rr).long()        # in cycles per box b

    def solve(self, reg: torch.Tensor | float):
        """ft = num / (den + reg); reg: scalar or per-shell (in cycles/box b) tensor."""
        G = self.G
        den = self.den.view(G, G, G)
        if torch.is_tensor(reg):
            sh = self.shell_index().clamp_(max=len(reg) - 1)
            reg = reg[sh]
        num = self.num.view(G, G, G, 2)
        self.ft = (num / (den + reg)[..., None]).permute(3, 0, 1, 2).contiguous()


def fsc(a: Volume, b: Volume) -> torch.Tensor:
    """Fourier shell correlation of two volumes on the same grid, per shell (cycles/box b)."""
    sh = a.shell_index().reshape(-1)
    n = a.b // 2 + 1
    keep = sh < n
    sh = sh[keep]
    fa, fb = a.ft.reshape(2, -1)[:, keep], b.ft.reshape(2, -1)[:, keep]
    cross = torch.zeros(n, device=a.dev).index_add_(0, sh, (fa * fb).sum(0))
    pa = torch.zeros(n, device=a.dev).index_add_(0, sh, (fa * fa).sum(0))
    pb = torch.zeros(n, device=a.dev).index_add_(0, sh, (fb * fb).sum(0))
    return cross / torch.sqrt(pa * pb).clamp_min(1e-30)


def resolution_from_fsc(c: torch.Tensor, box_a: float, thr: float = 0.143) -> float:
    c = c.cpu().numpy()
    for i in range(1, len(c)):
        if c[i] < thr:
            return box_a / max(i, 1)
    return box_a / (len(c) - 1)


def sphere_mask(b: int, radius_px: float, edge_px: float, dev) -> torch.Tensor:
    r = torch.arange(b, device=dev, dtype=torch.float32) - b // 2
    rr = torch.sqrt(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2)
    m = torch.clamp((radius_px + edge_px - rr) / max(edge_px, 1e-3), 0, 1)
    return 0.5 - 0.5 * torch.cos(math.pi * m)


def lowpass(vol: Volume, res_a: float, box_a: float):
    sh = vol.shell_index().float()
    k = box_a / res_a
    vol.ft = vol.ft * torch.exp(-0.5 * (sh / max(k, 1.0)) ** 2 * math.log(2) * 2)[None]  # half amplitude at res


def center_of_mass(v: torch.Tensor) -> torch.Tensor:
    b = v.shape[0]
    r = torch.arange(b, device=v.device, dtype=torch.float32) - b // 2
    p = v.clamp_min(0)
    tot = p.sum().clamp_min(1e-12)
    return torch.stack([(p.sum((0, 1)) * r).sum(), (p.sum((0, 2)) * r).sum(), (p.sum((1, 2)) * r).sum()]) / tot  # x, y, z


def rotate_images(ft: torch.Tensor, b: int, angles: torch.Tensor) -> torch.Tensor:
    """Half spectra (K, b, bh) -> (K, R, b, bh): each image rotated counter-clockwise by every angle
    (rot_a(projection at A) = projection at rotz(a) @ A)."""
    up = upsample2(ft, b)                          # (K, 2b, 2b) real
    grid = rot_grid(angles.to(ft.device), b, 2 * b)  # (R, b, b, 2)
    out = []
    for k in range(len(up)):
        img = F.grid_sample(up[k][None, None].expand(len(angles), 1, 2 * b, 2 * b), grid, align_corners=False)[:, 0]
        out.append(torch.fft.rfft2(img, norm="forward"))
    return torch.stack(out)


def _band(b: int, dev, box_a: float, lo_a: float, hi_a: float) -> torch.Tensor:
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (kr >= box_a / lo_a) & (kr <= box_a / hi_a)


def _regularise_solve(vol: Volume, frac: float = 0.05):
    """num / (den + frac * mean den of the shell): damps sparsely sampled Fourier voxels."""
    G = vol.G
    sh = vol.shell_index().reshape(-1)
    n = int(sh.max()) + 1
    den = vol.den
    m = torch.zeros(n, device=vol.dev).index_add_(0, sh, den) / torch.bincount(sh, minlength=n).clamp_min(1)
    vol.solve(frac * m + 1e-6 * m.max())


def _real_space_constraints(vol: Volume, mask: torch.Tensor, positive: bool, center: bool):
    v = vol.real() * mask
    if positive:
        v = v.clamp_min(0)
    if center:
        com = torch.round(center_of_mass(v)).long()
        v = torch.roll(v, shifts=(-int(com[2]), -int(com[1]), -int(com[0])), dims=(0, 1, 2)) * mask
    vol.set_real(v)


# --------------------------------------------------------------------------- stage 1: class averages
def classes_to_volume(cls_ft: torch.Tensor, pops: torch.Tensor, box_a: float, diameter_a: float, sym: str = "C1",
                      n_starts: int = 8, res_start: float = 40.0, res_end: float = 15.0, n_iter: int = 30,
                      seed: int = 0, log=print):
    """Initial model from 2D class averages (spectra (K, b, bh) at the classification box; populations (K,)).
    Returns (Volume at box b1, class orientations (K, 3, 3) in each class's own frame, per-class correlation)."""
    dev = cls_ft.device
    K, b0 = cls_ft.shape[0], cls_ft.shape[1]
    b1 = min(b0, int(math.ceil(2 * box_a / res_end / 2)) * 2 + 2)
    from .engine import ft_resize
    X = ft_resize(cls_ft, b1)
    ops = symmetry_ops(sym).to(dev)
    mask = sphere_mask(b1, diameter_a / 2 / (box_a / b1), 2.0, dev)
    w_cls = (pops / pops.max()).float().to(dev)
    g = torch.Generator().manual_seed(seed)
    best = None
    t0 = time.time()
    # rotated copies of the classes for the finest step used (coarser steps take every n-th)
    step_min = max(2.0, math.degrees(res_end / diameter_a))
    R = int(round(360 / step_min))
    angles = torch.arange(R, dtype=torch.float32) * 360 / R
    Xr = rotate_images(X, b1, angles)                         # (K, R, b1, bh)
    for start in range(n_starts):
        vol = Volume(b1, dev)
        A0 = euler_to_matrix(torch.rand(K, generator=g) * 360, torch.rand(K, generator=g) * 180,
                             torch.rand(K, generator=g) * 360).to(dev)
        vol.insert(X * w_cls[:, None, None], w_cls[:, None, None].expand(K, b1, b1 // 2 + 1).contiguous(),
                   A0, kmax=box_a / res_start, ops=ops)
        _regularise_solve(vol)
        lowpass(vol, res_start, box_a)
        _real_space_constraints(vol, mask, positive=True, center=True)
        for it in range(n_iter):
            f = it / max(n_iter - 1, 1)
            res = res_start * (res_end / res_start) ** min(1.0, f * 1.3)
            step = max(step_min, math.degrees(res / diameter_a))
            rot, tilt = sphere_directions(step)
            Ad = euler_to_matrix(rot, tilt, torch.zeros_like(rot)).to(dev)
            P = vol.project(Ad)                               # (D, b1, bh)
            band = _band(b1, dev, box_a, 2 * res_start, res)
            every = max(1, int(round(step / (360 / R))))
            ri = torch.arange(0, R, every, device=dev)
            Pb = P[:, band]
            Pb = Pb / Pb.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
            Xb = Xr[:, ri][:, :, band]                        # (K, R', F)
            Xb = Xb / Xb.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
            cc = (Xb.reshape(-1, Xb.shape[-1]).conj() @ Pb.T).real.reshape(K, len(ri), len(Ad))  # (K, R', D)
            ccf = cc.reshape(K, -1)
            # annealed soft assignment over the top hypotheses (sharpening to hard in the last third)
            top, ti = ccf.topk(16, dim=1)
            spread = (top[:, :1] - ccf.median(1, keepdim=True).values).clamp_min(1e-4)
            beta = 3.0 + 60.0 * f ** 2
            w = torch.softmax((top - top[:, :1]) / spread * beta, dim=1)
            if f > 0.66:
                w = torch.zeros_like(w); w[:, 0] = 1
            vol.clear()
            kk = torch.arange(K, device=dev)[:, None].expand_as(ti)
            r_i, d_i = ri[ti // len(Ad)], ti % len(Ad)
            sel = w > 1e-3
            imgs = Xr[kk[sel], r_i[sel]]
            ww = (w[sel] * w_cls[kk[sel]])[:, None, None]
            vol.insert(imgs * ww, ww.expand(-1, b1, b1 // 2 + 1).contiguous(), Ad[d_i[sel]], kmax=box_a / res, ops=ops)
            _regularise_solve(vol)
            lowpass(vol, res, box_a)
            _real_space_constraints(vol, mask, positive=it < n_iter // 2, center=True)
        score = float((top[:, 0] * w_cls).sum() / w_cls.sum())
        log(f"[3d] start {start + 1}/{n_starts}: class agreement {score:.3f} ({time.time() - t0:.1f}s)")
        if best is None or score > best[0]:
            Ak = rotz(-angles[r_i[:, 0].cpu()]).to(dev) @ Ad[d_i[:, 0]]   # class = rot_{-a}(P(Ad)) = P(rotz(-a) Ad)
            best = (score, vol, Ak, top[:, 0].clone())
    log(f"[3d] best start: class agreement {best[0]:.3f} (per class {' '.join(f'{x:.2f}' for x in best[3].tolist())})")
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- stage 2: particles
def refine_particles(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, init: Volume, sym: str = "C1",
                     n_iter: int = 8, res_start: float = 15.0, step_start: float | None = None, batch: int = 256,
                     n_local: int = 1, step_scale: float | None = None, log=print):
    """Local refinement of particle orientations A (n, 3, 3) and shifts t (n, 2) (A, model shift) against
    CTF-modulated projections, gold-standard halves. Returns (volume real map, A, t, fsc, resolution)."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    b = clf.bmax
    n = len(idx)
    ops = symmetry_ops(sym).to(dev)
    g = torch.Generator().manual_seed(1)
    half = torch.randperm(n, generator=g) % 2
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ph = TWO_PI / box_a
    mask = sphere_mask(b, diam / 2 / (box_a / b), 3.0, dev)
    A, t = A.clone().to(dev), t.clone().to(dev).float()
    vols = [Volume(b, dev), Volume(b, dev)]
    from .engine import ft_resize
    # starting model at the particle box
    v0 = init.real()
    v0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(v0)))
    vb = torch.zeros((b, b, b), dtype=torch.complex64, device=dev)
    b1 = init.b
    o, h1 = (b - b1) // 2, min(b, b1)
    if b >= b1:
        vb[o:o + b1, o:o + b1, o:o + b1] = v0
    else:
        o = (b1 - b) // 2
        vb = v0[o:o + b, o:o + b, o:o + b]
    v_init = torch.fft.ifftn(torch.fft.ifftshift(vb)).real * (b / b1) ** 0
    res = res_start
    fsc_c = None
    sigma2 = None
    step = step_start or max(1.0, math.degrees(res_start / diam))
    t0 = time.time()

    def backproject(it_res):
        for v in vols:
            v.clear()
        kmax = min(b / 2 - 1, box_a / it_res * 1.0 + 4)
        for i in range(0, n, 2048):
            j = idx[i:i + 2048]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[i:i + 2048]
            xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            hh = half[i:i + 2048].to(dev)
            for h in (0, 1):
                m = hh == h
                if m.any():
                    vols[h].insert((xs * ctf)[m], (ctf * ctf)[m], A[i:i + 2048][m], kmax=kmax, ops=ops)

    for it in range(n_iter + 1):
        # ---- reconstruct halves, FSC, filter
        if it == 0:
            for v in vols:
                v.set_real(v_init)
        else:
            backproject(res)
            for v in vols:
                _regularise_solve(v, 0.02)
            fsc_c = fsc(vols[0], vols[1])
            res = max(resolution_from_fsc(fsc_c, box_a), 2.2 * box_a / b)
        if it == n_iter:
            break
        al_res = res if it == 0 else res
        refs = []
        for v in vols:
            r = Volume(b, dev)
            r.ft = v.ft.clone()
            if fsc_c is not None:
                sh = r.shell_index().clamp_(max=len(fsc_c) - 1)
                fw = torch.sqrt((2 * fsc_c / (1 + fsc_c)).clamp(0, 1))
                r.ft = r.ft * fw[sh][None]
            lowpass(r, al_res, box_a)
            _real_space_constraints(r, mask, positive=False, center=False)
            refs.append(r)
        # ---- local search
        kal = box_a / al_res
        band = (kr >= 1) & (kr <= kal)
        if step_scale is not None:
            step = max(0.5, step_scale * math.degrees(al_res / diam))
        Rl = local_rotations(step, n_local).to(dev)               # ((2n+1)^2, 3, 3) out-of-plane
        dpsi = torch.arange(-n_local, n_local + 1, dtype=torch.float32) * step
        Rz = rotz(dpsi).to(dev)
        Rc = (Rz[:, None] @ Rl[None]).reshape(-1, 3, 3)           # (J, 3, 3) applied on the left: A' = Rc @ A
        J = len(Rc)
        ts = torch.tensor([-1.0, 0.0, 1.0], device=dev) * (box_a / b) * max(0.5, step / 3)
        sx, sy = torch.meshgrid(ts, ts, indexing="xy")
        sg = torch.stack([sx.reshape(-1), sy.reshape(-1)], 1)    # (Sh, 2)
        fyb = fy[:, None].expand(b, b // 2 + 1)[band]
        fxb = fx[None, :].expand(b, b // 2 + 1)[band]
        if sigma2 is None:
            sigma2 = torch.ones(int(kr.max()) + 2, device=dev)
        shb = torch.round(kr[band]).long()
        isig = 1.0 / sigma2[shb]
        res_pow = torch.zeros_like(sigma2)
        res_cnt = torch.zeros_like(sigma2)
        changed = 0.0
        for i in range(0, n, batch):
            j = idx[i:i + batch]
            nb = len(j)
            hh = half[i:i + batch].to(dev)
            x = clf._fetch(j, dev, b)[:, band]                    # (nb, F)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
            Ac = Rc[None] @ A[i:i + batch][:, None]               # (nb, J, 3, 3)
            P = torch.empty((nb, J, x.shape[1]), dtype=torch.complex64, device=dev)
            for h in (0, 1):
                m = torch.nonzero(hh == h).squeeze(1)
                if len(m):
                    P[m] = refs[h].project(Ac[m].reshape(-1, 3, 3))[:, band].reshape(len(m), J, -1)
            CP = P * ctf[:, None]
            tt = t[i:i + batch]
            # shift hypotheses t + sg: model = exp(-i ph f.(t+s)) CP
            e_t = torch.exp(-1j * ph * (fxb[None] * tt[:, 0, None] + fyb[None] * tt[:, 1, None]))     # (nb, F)
            E = torch.exp(-1j * ph * (sg[:, 0, None] * fxb[None] + sg[:, 1, None] * fyb[None]))        # (Sh, F)
            Z = (x.conj() * e_t)[:, None] * CP * isig                                                   # (nb, J, F)
            cross = (Z @ E.T).real                                                                      # (nb, J, Sh)
            pw = (CP.abs() ** 2 * isig).sum(-1)                                                         # (nb, J)
            ll = cross - 0.5 * pw[..., None]
            best = ll.reshape(nb, -1).argmax(1)
            bj, bs = best // len(sg), best % len(sg)
            ar = torch.arange(nb, device=dev)
            A[i:i + batch] = Ac[ar, bj]
            t[i:i + batch] = tt + sg[bs]
            changed += float((bj != J // 2).float().sum() + (bs != len(sg) // 2).float().sum()) / 2
            # residual power for the noise model
            mdl = CP[ar, bj] * e_t * E[bs]
            r2 = (x - mdl).abs() ** 2
            res_pow.index_add_(0, shb, r2.sum(0))
            res_cnt.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
        sigma2 = torch.where(res_cnt > 0, res_pow / res_cnt.clamp_min(1), sigma2)
        sigma2[res_cnt == 0] = sigma2[res_cnt > 0][-1] if (res_cnt > 0).any() else 1.0
        log(f"[3d] particles it {it + 1}/{n_iter}: align {al_res:.1f} A, step {step:.2f} deg, "
            f"{changed / n * 100:.0f}% moved ({time.time() - t0:.1f}s)")
        if step_scale is None and changed / n < 0.3:
            step = max(0.5, step / 2)
    # final map: average of halves, FSC-weighted
    out = Volume(b, dev)
    out.ft = 0.5 * (vols[0].ft + vols[1].ft)
    sh = out.shell_index().clamp_(max=len(fsc_c) - 1)
    out.ft = out.ft * torch.sqrt((2 * fsc_c / (1 + fsc_c)).clamp(0, 1))[sh][None]
    vmap = out.real() * mask
    log(f"[3d] final resolution (FSC 0.143) {res:.1f} A, {time.time() - t0:.1f}s")
    return vmap, A, t, fsc_c, res, vols


def particle_poses(cls_A: torch.Tensor, labels: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Particle orientation from its class orientation and 2D in-plane angle: particle = rot_psi(class)."""
    return rotz(psi.float()).to(cls_A.device) @ cls_A[labels.to(cls_A.device)]


# --------------------------------------------------------------------------- driver + report
def _save_mrc(path, v, apix):
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(v.detach().cpu().numpy().astype(np.float32))
        m.voxel_size = apix


@dataclass
class AbinitioParams:
    sym: str = "C1"                 # point group imposed from the start (Cn, Dn)
    n_starts: int = 8               # random starts of the class stage; the best-agreeing one is kept
    class_res_start: float = 40.0   # class stage: resolution schedule (A) ...
    class_res_end: float = 15.0
    class_iter: int = 30            # ... over this many iterations
    n_iter: int = 12                # particle stage: local-refinement iterations
    res_start: float = 15.0         # particle stage: starting alignment resolution (A); then from the half-map FSC
    n_local: int = 1                # local search half-width (grid of (2n+1)^2 tilts x (2n+1) in-plane angles)
    seed: int = 0


def run(clf, result, outdir: str, prm: AbinitioParams | None = None, classes=None, log=print) -> dict:
    """Ab initio from a Session result: its class averages (all populated classes, or the boolean mask `classes`)
    and the particles in those classes. Writes maps, figures, poses and summary.json to outdir."""
    import json
    import os
    prm = prm or AbinitioParams()
    os.makedirs(outdir, exist_ok=True)
    dev = clf.dev
    b, box_a, diam = clf.bmax, clf.box_a, clf.p.diameter_a
    pops = torch.as_tensor(np.asarray(result.pops), device=dev, dtype=torch.float32)
    keep = pops > 0
    if classes is not None:
        keep &= torch.as_tensor(np.asarray(classes), dtype=torch.bool, device=dev)
    if int(keep.sum()) < 3:
        raise ValueError("ab initio needs at least 3 populated classes")
    cls_ft = torch.fft.rfft2(torch.as_tensor(np.asarray(result.images), device=dev, dtype=torch.float32), norm="forward")
    log(f"[3d] ab initio from {int(keep.sum())} classes ({prm.sym}), working box {b} ({box_a / b:.2f} A/px)")
    t0 = time.time()
    vol, Ak, cc = classes_to_volume(cls_ft[keep], pops[keep], box_a, diam, sym=prm.sym, n_starts=prm.n_starts,
                                    res_start=prm.class_res_start, res_end=prm.class_res_end, n_iter=prm.class_iter,
                                    seed=prm.seed, log=log)
    torch.cuda.synchronize()
    t1 = time.time() - t0
    kmap = torch.full((len(pops),), -1, dtype=torch.long)
    kmap[keep.cpu()] = torch.arange(int(keep.sum()))
    lab = torch.as_tensor(np.asarray(result.labels)).long()
    ok = (lab >= 0) & (kmap[lab.clamp_min(0)] >= 0)
    idx = torch.as_tensor(np.asarray(result.idx))[ok]
    if clf._n_loaded < clf.N:   # the class stage needs no particles; it ran while they were still streaming in
        log(f"[3d] waiting for particles to finish loading ({clf._n_loaded}/{clf.N})")
    clf._wait_loaded(clf.N)
    log(f"[3d] class stage {t1:.1f}s; refining {len(idx)} particles")
    A0 = particle_poses(Ak, kmap[lab[ok]], torch.as_tensor(np.asarray(result.psi))[ok])
    t0 = time.time()
    vmap, A, t, fs, res, vols = refine_particles(clf, idx, A0, torch.as_tensor(np.asarray(result.offsets))[ok], vol,
                                                 sym=prm.sym, n_iter=prm.n_iter, res_start=prm.res_start,
                                                 n_local=prm.n_local, log=log)
    torch.cuda.synchronize()
    t2 = time.time() - t0
    apix = box_a / b
    _save_mrc(os.path.join(outdir, "map.mrc"), vmap, apix)
    for h in (0, 1):
        _save_mrc(os.path.join(outdir, f"half{h + 1}_unfil.mrc"), vols[h].real(), apix)
    _save_mrc(os.path.join(outdir, "from_classes.mrc"), vol.real(), box_a / vol.b)
    rot, tilt, psi = matrix_to_euler(A.cpu())
    np.savez(os.path.join(outdir, "poses.npz"), idx=idx.numpy(), rot=rot.numpy(), tilt=tilt.numpy(), psi=psi.numpy(),
             shift_a=t.cpu().numpy())
    _figures(outdir, clf, result, keep, Ak, vmap, fs, res, A, box_a, b)
    from dataclasses import asdict
    summary = dict(params=asdict(prm), particles=int(len(idx)), classes=int(keep.sum()), seconds_classes=round(t1, 1),
                   seconds_particles=round(t2, 1), resolution_a=round(res, 2), apix=round(apix, 3), box=b,
                   fsc=[round(float(x), 4) for x in fs.cpu()], box_a=round(box_a, 2),
                   class_agreement=[round(float(x), 3) for x in cc])
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"[3d] wrote {outdir}: {summary['resolution_a']} A, {t1:.1f}s classes + {t2:.1f}s particles")
    return summary


def _figures(outdir, clf, result, keep, Ak, vmap, fs, res, A, box_a, b):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = clf.dev
    # class averages next to reprojections of the final map at the class orientations
    V = Volume(b, dev)
    V.set_real(vmap)
    P = torch.fft.irfft2(V.project(Ak), s=(b, b), norm="forward").cpu().numpy()
    imgs = np.asarray(result.images)[keep.cpu().numpy()]
    pops = np.asarray(result.pops)[keep.cpu().numpy()]
    order = np.argsort(-pops)
    n = len(order)
    cols = min(10, n)
    rows = int(math.ceil(n / cols))
    fig, ax = plt.subplots(2 * rows, cols, figsize=(1.3 * cols, 2.7 * rows), squeeze=False)
    for a in ax.flat:
        a.axis("off")
    for j, k in enumerate(order):
        r, c = divmod(j, cols)
        ax[2 * r, c].imshow(imgs[k], cmap="gray")
        ax[2 * r, c].set_title(f"{int(pops[k])}", fontsize=7)
        ax[2 * r + 1, c].imshow(P[k], cmap="gray")
    fig.suptitle("class averages (rows 1, 3, ..) vs reprojections of the 3D map at their orientations", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reprojections.png"), dpi=90)
    plt.close(fig)
    # FSC
    f = fs.cpu().numpy()
    x = np.arange(len(f)) / box_a
    fig, a = plt.subplots(figsize=(5, 3.2))
    a.plot(x[1:], f[1:], lw=2)
    a.axhline(0.143, color="0.5", lw=1, ls="--")
    a.set_xlabel("spatial frequency (1/A)")
    a.set_ylabel("half-map FSC")
    a.set_title(f"FSC 0.143 at {res:.1f} A")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fsc.png"), dpi=90)
    plt.close(fig)
    # viewing directions
    rot, tilt, _ = matrix_to_euler(A.cpu())
    fig, a = plt.subplots(figsize=(6, 3.2))
    h = a.hist2d(rot.numpy(), tilt.numpy(), bins=[72, 36], range=[[-180, 180], [0, 180]], cmap="viridis")
    plt.colorbar(h[3], ax=a, label="particles")
    a.set_xlabel("rot (deg)")
    a.set_ylabel("tilt (deg)")
    a.set_title("viewing directions")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "views.png"), dpi=90)
    plt.close(fig)
    # central projections of the map
    v = vmap.cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for j in range(3):
        ax[j].imshow(v.sum(j), cmap="gray")
        ax[j].axis("off")
    ax[0].set_title("map: projections along z, y, x", fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "map_views.png"), dpi=90)
    plt.close(fig)
