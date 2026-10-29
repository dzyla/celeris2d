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
                      seed: int = 0, n_models: int = 1, log=print):
    """Initial model(s) from 2D class averages (spectra (K, b, bh) at the classification box; populations (K,)).
    With n_models > 1 every class is also assigned to the model that explains it best (soft, then hard).
    Returns (list of Volumes at box b1, class orientations (M, K, 3, 3) in every model's frame,
    per-class correlation (K,), model of each class (K,))."""
    dev = cls_ft.device
    K, b0 = cls_ft.shape[0], cls_ft.shape[1]
    M = max(1, int(n_models))
    b1 = min(b0, int(math.ceil(2 * box_a / res_end / 2)) * 2 + 2)
    from .engine import ft_resize
    X = ft_resize(cls_ft, b1)
    bh1 = b1 // 2 + 1
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
    kk_all = torch.arange(K, device=dev)
    for start in range(n_starts):
        zk = (torch.randperm(K, generator=g) % M).to(dev)     # balanced random split of the classes
        A0 = euler_to_matrix(torch.rand(K, generator=g) * 360, torch.rand(K, generator=g) * 180,
                             torch.rand(K, generator=g) * 360).to(dev)
        vols = []
        for m in range(M):
            vol = Volume(b1, dev)
            sel = zk == m
            vol.insert(X[sel] * w_cls[sel, None, None], w_cls[sel, None, None].expand(-1, b1, bh1).contiguous(),
                       A0[sel], kmax=box_a / res_start, ops=ops)
            _regularise_solve(vol)
            lowpass(vol, res_start, box_a)
            _real_space_constraints(vol, mask, positive=True, center=True)
            vols.append(vol)
        wz = torch.full((M, K), 1.0 / M, device=dev)
        for it in range(n_iter):
            f = it / max(n_iter - 1, 1)
            res = res_start * (res_end / res_start) ** min(1.0, f * 1.3)
            step = max(step_min, math.degrees(res / diameter_a))
            rot, tilt = sphere_directions(step)
            Ad = euler_to_matrix(rot, tilt, torch.zeros_like(rot)).to(dev)
            band = _band(b1, dev, box_a, 2 * res_start, res)
            every = max(1, int(round(step / (360 / R))))
            ri = torch.arange(0, R, every, device=dev)
            Xb = Xr[:, ri][:, :, band]                        # (K, R', F)
            Xb = (Xb / Xb.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)).reshape(-1, Xb.shape[-1]).conj()
            tops, tis, meds = [], [], []
            for m in range(M):
                Pb = vols[m].project(Ad)[:, band]
                Pb = Pb / Pb.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
                ccf = (Xb @ Pb.T).real.reshape(K, -1)         # (K, R' * D)
                top, ti = ccf.topk(16, dim=1)
                tops.append(top); tis.append(ti); meds.append(ccf.median(1).values)
            top_m = torch.stack(tops)                         # (M, K, 16)
            beta = 3.0 + 60.0 * f ** 2
            hard = f > 0.66
            if M > 1:                                         # which model explains each class
                bm = top_m[:, :, 0]                           # (M, K)
                spread_z = (bm.max(0).values - torch.stack(meds).mean(0)).clamp_min(1e-4)
                wz = torch.softmax((bm - bm.max(0).values) / spread_z * beta * 2, dim=0)
                if hard:
                    wz = F.one_hot(bm.argmax(0), M).T.float()
            for m in range(M):
                top, ti = tops[m], tis[m]
                spread = (top[:, :1] - meds[m][:, None]).clamp_min(1e-4)
                w = torch.softmax((top - top[:, :1]) / spread * beta, dim=1)
                if hard:
                    w = torch.zeros_like(w); w[:, 0] = 1
                w = w * wz[m][:, None]
                vols[m].clear()
                kk = kk_all[:, None].expand_as(ti)
                r_i, d_i = ri[ti // len(Ad)], ti % len(Ad)
                sel = w > 1e-3
                if int(sel.sum()) == 0:
                    continue
                imgs = Xr[kk[sel], r_i[sel]]
                ww = (w[sel] * w_cls[kk[sel]])[:, None, None]
                vols[m].insert(imgs * ww, ww.expand(-1, b1, bh1).contiguous(), Ad[d_i[sel]], kmax=box_a / res, ops=ops)
                _regularise_solve(vols[m])
                lowpass(vols[m], res, box_a)
                _real_space_constraints(vols[m], mask, positive=it < n_iter // 2, center=True)
                tis[m] = (ti, r_i, d_i)
        zk = top_m[:, :, 0].argmax(0)                         # (K,)
        cc_k = top_m[zk, kk_all, 0]
        score = float((cc_k * w_cls).sum() / w_cls.sum())
        sizes = torch.bincount(zk, weights=w_cls * float(pops.max()), minlength=M).round().long().tolist()
        log(f"[3d] start {start + 1}/{n_starts}: class agreement {score:.3f}"
            + (f", particles per model {sizes}" if M > 1 else "") + f" ({time.time() - t0:.1f}s)")
        if M > 1 and min(sizes) == 0:
            score -= 1.0                                      # a start that left a model empty is not a split
        if best is None or score > best[0]:
            Ak = torch.empty((M, K, 3, 3), device=dev)       # each class's best orientation in every model
            for m in range(M):
                _, r_i, d_i = tis[m]
                Ak[m] = rotz(-angles[r_i[:, 0].cpu()]).to(dev) @ Ad[d_i[:, 0]]   # class = P(rotz(-a) Ad)
            best = (score, vols, Ak, cc_k.clone(), zk.clone())
    log(f"[3d] best start: class agreement {best[0]:.3f} (per class {' '.join(f'{x:.2f}' for x in best[3].tolist())})"
        + (f"; classes per model {torch.bincount(best[4], minlength=M).tolist()}" if M > 1 else ""))
    return best[1], best[2], best[3], best[4]


# --------------------------------------------------------------------------- stage 2: particles
def _to_box(init: Volume, b: int, dev) -> torch.Tensor:
    """Real map of a Volume, Fourier-cropped/padded to box b."""
    v0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(init.real())))
    b1 = init.b
    if b >= b1:
        vb = torch.zeros((b, b, b), dtype=torch.complex64, device=dev)
        o = (b - b1) // 2
        vb[o:o + b1, o:o + b1, o:o + b1] = v0
    else:
        o = (b1 - b) // 2
        vb = v0[o:o + b, o:o + b, o:o + b]
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(vb)).real)


def refine_particles(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, inits, z: torch.Tensor | None = None,
                     sym: str = "C1", n_iter: int = 8, res_start: float = 15.0, step_start: float | None = None,
                     batch: int = 256, n_local: int = 1, step_scale: float | None = None, log=print):
    """Local refinement of particle orientations A and shifts t (A, model shift) against CTF-modulated projections,
    gold-standard halves. inits: starting Volume or list of M Volumes. A: (n, 3, 3) or, with several models, the
    pose in every model's own frame (n, M, 3, 3); t: (n, 2) or (n, M, 2); z: (n,) starting model. Each iteration
    refines the pose in every model and puts the particle in the model that fits best.
    Returns (maps (M, b, b, b), A (n, 3, 3), t (n, 2) in the chosen model, z, fsc (M, shells), resolutions (M,),
    half Volumes [[h0, h1] per model])."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    b = clf.bmax
    n = len(idx)
    inits = inits if isinstance(inits, (list, tuple)) else [inits]
    M = len(inits)
    z = torch.zeros(n, dtype=torch.long, device=dev) if z is None else z.clone().to(dev).long()
    ops = symmetry_ops(sym).to(dev)
    g = torch.Generator().manual_seed(1)
    half = torch.randperm(n, generator=g) % 2
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ph = TWO_PI / box_a
    mask = sphere_mask(b, diam / 2 / (box_a / b), 3.0, dev)
    A, t = A.clone().to(dev).float(), t.clone().to(dev).float()
    if A.dim() == 3:
        A = A[:, None].expand(n, M, 3, 3).contiguous()
    if t.dim() == 2:
        t = t[:, None].expand(n, M, 2).contiguous()
    ar_n = torch.arange(n, device=dev)
    vols = [[Volume(b, dev), Volume(b, dev)] for _ in range(M)]
    v_init = [_to_box(v, b, dev) for v in inits]
    res = [res_start] * M
    fscs = [None] * M
    alive = [True] * M                    # a model that loses (almost) all its particles is retired
    sigma2 = None
    step = step_start or max(1.0, math.degrees(res_start / diam))
    t0 = time.time()

    def backproject():
        for vm in vols:
            for v in vm:
                v.clear()
        for i in range(0, n, 2048):
            j = idx[i:i + 2048]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            zz = z[i:i + 2048]
            sl = ar_n[i:i + 2048]
            tt, AA = t[sl, zz], A[sl, zz]
            xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            hh = half[i:i + 2048].to(dev)
            for m in range(M):
                kmax = min(b / 2 - 1, box_a / res[m] + 4)
                for h in (0, 1):
                    sel = (hh == h) & (zz == m)
                    if sel.any():
                        vols[m][h].insert((xs * ctf)[sel], (ctf * ctf)[sel], AA[sel], kmax=kmax, ops=ops)

    for it in range(n_iter + 1):
        # ---- reconstruct halves, FSC, filter
        if it == 0:
            for m in range(M):
                for v in vols[m]:
                    v.set_real(v_init[m])
        else:
            backproject()
            for m in range(M):
                for v in vols[m]:
                    _regularise_solve(v, 0.02)
                fscs[m] = fsc(vols[m][0], vols[m][1])
                res[m] = max(resolution_from_fsc(fscs[m], box_a), 2.2 * box_a / b)
        if it == n_iter:
            break
        live = [m for m in range(M) if alive[m]]
        # all models are compared at the same resolution (the worst live one), so a model with more particles
        # (a cleaner reconstruction) does not win particles just for being less noisy
        al_res = max(res[m] for m in live) if M > 1 else res[0]
        refs = []
        for m in range(M):
            rm = []
            for v in vols[m]:
                r = Volume(b, dev)
                r.ft = v.ft.clone()
                if fscs[m] is not None and M == 1:
                    sh = r.shell_index().clamp_(max=len(fscs[m]) - 1)
                    r.ft = r.ft * torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0))[sh][None]
                lowpass(r, al_res if M > 1 else res[m], box_a)
                _real_space_constraints(r, mask, positive=False, center=False)
                rm.append(r)
            refs.append(rm)
        # ---- local search
        kal = box_a / al_res
        band = (kr >= 1) & (kr <= kal)
        if step_scale is not None:
            step = max(0.5, step_scale * math.degrees(al_res / diam))
        Rl = local_rotations(step, n_local).to(dev)               # ((2n+1)^2, 3, 3) out-of-plane
        dpsi = torch.arange(-n_local, n_local + 1, dtype=torch.float32) * step
        Rc = (rotz(dpsi).to(dev)[:, None] @ Rl[None]).reshape(-1, 3, 3)   # (J, 3, 3) applied on the left: A' = Rc @ A
        J = len(Rc)
        ts = torch.tensor([-1.0, 0.0, 1.0], device=dev) * (box_a / b) * max(0.5, step / 3)
        sx, sy = torch.meshgrid(ts, ts, indexing="xy")
        sg = torch.stack([sx.reshape(-1), sy.reshape(-1)], 1)    # (Sh, 2)
        Sh = len(sg)
        fyb = fy[:, None].expand(b, b // 2 + 1)[band]
        fxb = fx[None, :].expand(b, b // 2 + 1)[band]
        E = torch.exp(-1j * ph * (sg[:, 0, None] * fxb[None] + sg[:, 1, None] * fyb[None]))        # (Sh, F)
        if sigma2 is None:
            sigma2 = torch.ones(int(kr.max()) + 2, device=dev)
        shb = torch.round(kr[band]).long()
        isig = 1.0 / sigma2[shb]
        res_pow = torch.zeros_like(sigma2)
        res_cnt = torch.zeros_like(sigma2)
        changed, switched = 0.0, 0
        bt = max(32, batch // M)
        llm_all = torch.zeros((n, M), device=dev)
        for i in range(0, n, bt):
            j = idx[i:i + bt]
            nb = len(j)
            hh = half[i:i + bt].to(dev)
            x = clf._fetch(j, dev, b)[:, band]                    # (nb, F)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
            Ac = Rc[None, None] @ A[i:i + bt][:, :, None]         # (nb, M, J, 3, 3): around the pose in each model
            P = torch.empty((nb, M, J, x.shape[1]), dtype=torch.complex64, device=dev)
            for m in range(M):
                for h in (0, 1):
                    sel = torch.nonzero(hh == h).squeeze(1)
                    if len(sel):
                        P[sel, m] = refs[m][h].project(Ac[sel, m].reshape(-1, 3, 3))[:, band].reshape(len(sel), J, -1)
            CP = P * ctf[:, None, None]
            tt = t[i:i + bt]                                                                            # (nb, M, 2)
            # shift hypotheses t + sg: model = exp(-i ph f.(t+s)) CP
            e_t = torch.exp(-1j * ph * (fxb * tt[..., 0, None] + fyb * tt[..., 1, None]))             # (nb, M, F)
            Z = (x.conj()[:, None] * e_t)[:, :, None] * CP * isig                                       # (nb, M, J, F)
            cross = (Z @ E.T).real                                                                      # (nb, M, J, Sh)
            pw = (CP.abs() ** 2 * isig).sum(-1)                                                         # (nb, M, J)
            ll = (cross - 0.5 * pw[..., None]).reshape(nb, M, -1)
            llm, bjs = ll.max(-1)                                                                       # best pose per model
            if not all(alive):
                llm[:, [m for m in range(M) if not alive[m]]] = -float("inf")
            bj, bs = bjs // Sh, bjs % Sh                                                                # (nb, M)
            bm = llm.argmax(1)
            llm_all[i:i + bt] = llm
            ar = torch.arange(nb, device=dev)
            arm = torch.arange(M, device=dev)
            A[i:i + bt] = Ac[ar[:, None], arm[None], bj]
            t[i:i + bt] = tt + sg[bs]
            switched += int((bm != z[i:i + bt]).sum())
            z[i:i + bt] = bm
            bj, bs = bj[ar, bm], bs[ar, bm]
            changed += float((bj != J // 2).float().sum() + (bs != Sh // 2).float().sum()) / 2
            # residual power for the noise model
            mdl = CP[ar, bm, bj] * e_t[ar, bm] * E[bs]
            r2 = (x - mdl).abs() ** 2
            res_pow.index_add_(0, shb, r2.sum(0))
            res_cnt.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
        sigma2 = torch.where(res_cnt > 0, res_pow / res_cnt.clamp_min(1), sigma2)
        sigma2[res_cnt == 0] = sigma2[res_cnt > 0][-1] if (res_cnt > 0).any() else 1.0
        pops = torch.bincount(z, minlength=M).tolist()
        for m in range(M):
            if alive[m] and M > 1 and pops[m] < max(50, 0.01 * n):
                alive[m] = False
                log(f"[3d] 3D class {m + 1} kept only {pops[m]} particles: merged into the others")
                if pops[m]:                       # its particles go to the live model that fits them best
                    others = torch.tensor([q for q in range(M) if alive[q]], device=dev)
                    sel = z == m
                    z[sel] = others[llm_all[sel][:, others].argmax(1)]
        pops = torch.bincount(z, minlength=M).tolist()
        log(f"[3d] particles it {it + 1}/{n_iter}: align {al_res:.1f} A, step {step:.2f} deg, "
            f"{changed / n * 100:.0f}% moved" + (f", {switched / n * 100:.1f}% switched model, per model {pops}" if M > 1 else "")
            + f" ({time.time() - t0:.1f}s)")
        if step_scale is None and changed / n < 0.3:
            step = max(0.5, step / 2)
    # final maps: average of halves, FSC-weighted
    maps = []
    for m in range(M):
        if fscs[m] is None or not alive[m]:
            res[m] = float("nan")
        out = Volume(b, dev)
        out.ft = 0.5 * (vols[m][0].ft + vols[m][1].ft)
        if fscs[m] is not None:
            sh = out.shell_index().clamp_(max=len(fscs[m]) - 1)
            out.ft = out.ft * torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0))[sh][None]
        maps.append((out.real() * mask).nan_to_num(0))
    fscs = [f if f is not None else torch.zeros(b // 2 + 1, device=dev) for f in fscs]
    log(f"[3d] final resolution (FSC 0.143) {' / '.join(f'{r:.1f}' if r == r else 'merged' for r in res)} A, "
        f"{time.time() - t0:.1f}s")
    return torch.stack(maps), A[ar_n, z], t[ar_n, z], z, torch.stack(fscs), res, vols


def classify3d(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, n_models: int, res_a: float,
               sym: str = "C1", n_iter: int = 20, z0: torch.Tensor | None = None, seed: int = 0,
               expand_ops: torch.Tensor | None = None, log=print):
    """3D classification at fixed poses (from a consensus refinement): M models seeded from random particle subsets
    (or the labels z0), soft assignment (posterior over models), every model filtered to the same resolution res_a.
    expand_ops: operators R of a (pseudo-)symmetry of the consensus map (S, 3, 3): each particle is also tried at
    A @ R (equally good for the consensus, not for a model that breaks the symmetry) and keeps its best per model.
    Returns (maps (M, b, b, b), z (n,), posteriors (n, M), per-model half-map resolution, half Volumes, poses (n, 3, 3))."""
    dev = clf.dev
    box_a, diam, b = clf.box_a, clf.p.diameter_a, clf.bmax
    n, M = len(idx), int(n_models)
    ops = symmetry_ops(sym).to(dev)
    A, t = A.to(dev).float(), t.to(dev).float()
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ph = TWO_PI / box_a
    mask = sphere_mask(b, diam / 2 / (box_a / b), 3.0, dev)
    g = torch.Generator().manual_seed(seed + 7)
    half = (torch.randperm(n, generator=g) % 2).to(dev)
    if z0 is None:
        z0 = torch.randperm(n, generator=g) % M
    post = F.one_hot(z0.to(dev).long(), M).float()
    kal = box_a / res_a
    band = (kr >= 1) & (kr <= kal)
    shb = torch.round(kr[band]).long()
    kmax = min(b / 2 - 1, kal + 4)
    vols = [[Volume(b, dev), Volume(b, dev)] for _ in range(M)]
    sigma2 = None
    t0 = time.time()
    B = 1024
    X = torch.eye(3, device=dev)[None] if expand_ops is None else expand_ops.to(dev).float()
    S = len(X)
    s_best = torch.zeros((n, M), dtype=torch.long, device=dev)   # which symmetry-related pose, per particle and model

    def batches():
        for i in range(0, n, B):
            j = idx[i:i + B]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[i:i + B]
            yield i, x, ctf, tt

    z_prev, T0, T = None, None, 1.0
    for it in range(n_iter + 1):
        # ---- M-step: weighted reconstructions per model and half
        for vm in vols:
            for v in vm:
                v.clear()
        for i, x, ctf, tt in batches():
            xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            hh = half[i:i + B]
            for m in range(M):
                w = post[i:i + B, m]
                # first reconstruction with symmetry expansion: every particle at all its related poses, so the
                # starting models are symmetric and no pose is favoured (RELION's relax_sym); later only its best one
                first = it == 0 and S > 1
                Am = A[i:i + B] if first else A[i:i + B] @ X[s_best[i:i + B, m]]
                for h in (0, 1):
                    sel = (hh == h) & (w > 1e-3)
                    if sel.any():
                        ww = w[sel][:, None, None] / (S if first else 1)
                        Ain = Am[sel]
                        if first:
                            Ain = (Ain[:, None] @ X[None]).reshape(-1, 3, 3)
                            vols[m][h].insert((xs[sel] * ctf[sel] * ww).repeat_interleave(S, 0),
                                              (ctf[sel] ** 2 * ww).repeat_interleave(S, 0), Ain, kmax=kmax, ops=ops)
                        else:
                            vols[m][h].insert(xs[sel] * ctf[sel] * ww, ctf[sel] ** 2 * ww, Ain, kmax=kmax, ops=ops)
        refs = []
        for m in range(M):
            for v in vols[m]:
                _regularise_solve(v, 0.02)
            r = Volume(b, dev)
            r.ft = 0.5 * (vols[m][0].ft + vols[m][1].ft)
            lowpass(r, res_a, box_a)
            _real_space_constraints(r, mask, positive=False, center=False)
            refs.append(r)
        if it == n_iter:
            break
        # ---- E-step at the consensus poses
        lls = torch.zeros((n, M), device=dev)
        res_pow = torch.zeros(int(kr.max()) + 2, device=dev)
        res_cnt = torch.zeros_like(res_pow)
        isig = None if sigma2 is None else 1.0 / sigma2[shb]
        for i, x, ctf, tt in batches():
            xb = x[:, band]
            e_t = torch.exp(-1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))[:, band]
            nb = len(xb)
            for m in range(M):
                if isig is None:                 # noise estimate at the consensus pose
                    mdl = refs[m].project(A[i:i + B])[:, band] * ctf[:, band] * e_t
                    r2 = (xb - mdl).abs() ** 2
                    res_pow.index_add_(0, shb, r2.sum(0))
                    res_cnt.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
                    continue
                As = (A[i:i + B][:, None] @ X[None]).reshape(-1, 3, 3)
                mdl = refs[m].project(As)[:, band].reshape(nb, S, -1) * (ctf[:, band] * e_t)[:, None]
                ll = -0.5 * (((xb[:, None] - mdl).abs() ** 2) * isig).sum(-1)          # (nb, S)
                lls[i:i + B, m], s_best[i:i + B, m] = ll.max(1)
        if sigma2 is None:                       # first pass only measures the noise; assignments start next pass
            sigma2 = (res_pow / res_cnt.clamp_min(1)).clamp_min(1e-12)
            continue
        # annealing: the first assignments are kept soft (the typical particle starts near 60/40), then the
        # temperature falls to 1 (the true posterior) over the first half of the iterations
        if T0 is None:
            top2 = lls.topk(2, dim=1).values
            T0 = max(1.0, float((top2[:, 0] - top2[:, 1]).median()) / 0.4)
        f = min(1.0, (it - 1) / max(1.0, n_iter / 2))
        T = T0 ** (1 - f)
        post = torch.softmax(lls / T, dim=1)
        z = post.argmax(1)
        pops = torch.bincount(z, minlength=M).tolist()
        moved = float((z != z_prev).float().mean()) if z_prev is not None else 1.0
        z_prev = z
        log(f"[3d] 3D classification it {it}/{n_iter}: per class {pops}, {moved * 100:.1f}% changed, "
            f"mean max posterior {float(post.max(1).values.mean()):.3f}, T {T:.1f} ({time.time() - t0:.1f}s)")
        if moved < 0.005 and T <= 1.0:
            break
    res, maps = [], []
    for m in range(M):
        c = fsc(vols[m][0], vols[m][1])
        res.append(resolution_from_fsc(c, box_a) if int((z == m).sum()) > 10 else float("nan"))
        out = Volume(b, dev)
        out.ft = 0.5 * (vols[m][0].ft + vols[m][1].ft)
        sh = out.shell_index().clamp_(max=len(c) - 1)
        out.ft = out.ft * torch.sqrt((2 * c / (1 + c)).clamp(0, 1).nan_to_num(0))[sh][None]
        maps.append((out.real() * mask).nan_to_num(0))
    return torch.stack(maps), z, post, res, vols, A @ X[s_best[torch.arange(n, device=dev), z]]


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
    n_models: int = 1               # 3D classes: >1 adds a 3D classification of the particles at the consensus poses
    class3d_res: float = 10.0       # ... with every class filtered to this resolution (A) (or the consensus one if worse)
    class3d_iter: int = 20          # ... for at most this many iterations
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
    M = max(1, int(prm.n_models))
    vols0, Ak, cc, zk = classes_to_volume(cls_ft[keep], pops[keep], box_a, diam, sym=prm.sym, n_starts=prm.n_starts,
                                          res_start=prm.class_res_start, res_end=prm.class_res_end,
                                          n_iter=prm.class_iter, seed=prm.seed, n_models=1, log=log)
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
    kp = kmap[lab[ok]]
    A0 = particle_poses(Ak[0], kp, torch.as_tensor(np.asarray(result.psi))[ok])
    Ak, zk = Ak[0], torch.zeros_like(zk)
    t0 = time.time()
    maps, A, t, z, fs, res, vols = refine_particles(clf, idx, A0, torch.as_tensor(np.asarray(result.offsets))[ok],
                                                    vols0, sym=prm.sym, n_iter=prm.n_iter, res_start=prm.res_start,
                                                    n_local=prm.n_local, log=log)
    consensus = maps[0]
    if M > 1:   # heterogeneity: 3D classification at the consensus poses, all classes filtered alike
        xops, csym = None, prm.sym
        if prm.sym.upper() == "C1":
            _, g_sym, _ = detect_symmetry(consensus, box_a / b, diam)
            if g_sym != "C1":
                # a (pseudo-)symmetric consensus: each particle sits in one of its symmetry-related poses at random,
                # so a difference that breaks the symmetry lands in a different place per particle. Re-run the
                # consensus with the symmetry imposed (standard axes), then classify in C1 over the group's poses.
                log(f"[3d] the consensus map looks {g_sym}: consensus again with {g_sym} imposed, then classification "
                    f"relaxed over its {len(symmetry_ops(g_sym))} symmetry-related poses")
                v2, Ak2, _, _ = classes_to_volume(cls_ft[keep], pops[keep], box_a, diam, sym=g_sym,
                                                  n_starts=prm.n_starts, res_start=prm.class_res_start,
                                                  res_end=prm.class_res_end, n_iter=prm.class_iter, seed=prm.seed,
                                                  log=lambda *a: None)
                A2 = particle_poses(Ak2[0], kp, torch.as_tensor(np.asarray(result.psi))[ok])
                m2, A, t, _, _, r2, _ = refine_particles(clf, idx, A2, torch.as_tensor(np.asarray(result.offsets))[ok],
                                                         v2, sym=g_sym, n_iter=prm.n_iter, res_start=prm.res_start,
                                                         n_local=prm.n_local, log=lambda *a: None)
                log(f"[3d] {g_sym} consensus: {r2[0]:.1f} A")
                res = [r2[0]]
                xops = symmetry_ops(g_sym).to(dev)
        cres = max(res[0], prm.class3d_res)
        maps, z, post, res, vols, A = classify3d(clf, idx, A, t, M, cres, sym=csym, n_iter=prm.class3d_iter,
                                                 seed=prm.seed, expand_ops=xops, log=log)
        fs = torch.stack([fsc(vols[m][0], vols[m][1]) for m in range(M)])
        vols0 = vols0 * M
    torch.cuda.synchronize()
    t2 = time.time() - t0
    apix = box_a / b
    pops_m = torch.bincount(z, minlength=M).tolist()
    order = sorted(range(M), key=lambda m: -pops_m[m])        # model 1 = most particles
    for r, m in enumerate(order):
        sfx = "" if M == 1 else f"_{r + 1}"
        _save_mrc(os.path.join(outdir, f"map{sfx}.mrc"), maps[m], apix)
        for h in (0, 1):
            _save_mrc(os.path.join(outdir, f"half{h + 1}_unfil{sfx}.mrc"), vols[m][h].real().nan_to_num(0), apix)
        _save_mrc(os.path.join(outdir, f"from_classes{sfx}.mrc"), vols0[m].real(), box_a / vols0[m].b)
    if M > 1:   # map.mrc / from_classes.mrc = the largest model, so single-model consumers keep working
        import shutil
        shutil.copy(os.path.join(outdir, "map_1.mrc"), os.path.join(outdir, "map.mrc"))
        shutil.copy(os.path.join(outdir, "from_classes_1.mrc"), os.path.join(outdir, "from_classes.mrc"))
    rank = torch.empty(M, dtype=torch.long)
    rank[torch.tensor(order)] = torch.arange(M)
    rot, tilt, psi = matrix_to_euler(A.cpu())
    np.savez(os.path.join(outdir, "poses.npz"), idx=idx.numpy(), rot=rot.numpy(), tilt=tilt.numpy(), psi=psi.numpy(),
             shift_a=t.cpu().numpy(), model=rank[z.cpu()].numpy())
    _figures(outdir, clf, result, keep, Ak, [maps[m] for m in order], [fs[m] for m in order], [res[m] for m in order],
             A, box_a, b, rank[zk.cpu()].numpy())
    if M > 1:
        _save_mrc(os.path.join(outdir, "consensus.mrc"), consensus, apix)
    sym_rows, sym_sugg, _ = detect_symmetry(consensus, apix, diam)
    log(f"[3d] symmetry check: {' '.join(f'{g} {sc:.2f}' for g, _, sc in sym_rows[:5])} -> "
        f"{'suggest ' + sym_sugg if sym_sugg != 'C1' else 'no symmetry found'}")
    from dataclasses import asdict
    summary = dict(params=asdict(prm), particles=int(len(idx)), classes=int(keep.sum()), seconds_classes=round(t1, 1),
                   seconds_particles=round(t2, 1), resolution_a=round(res[order[0]], 2), apix=round(apix, 3), box=b,
                   fsc=[round(float(x), 4) for x in fs[order[0]].cpu()], box_a=round(box_a, 2),
                   class_agreement=[round(float(x), 3) for x in cc],
                   models=[dict(model=r + 1, particles=pops_m[m], classes=int((zk == m).sum()),
                                resolution_a=round(res[m], 2) if res[m] == res[m] else None)
                           for r, m in enumerate(order)],
                   symmetry=[dict(group=g, cc=round(x, 3), score=round(sc, 3)) for g, x, sc in sym_rows[:8]],
                   suggested_sym=sym_sugg)
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    models = " / ".join(f"{x['resolution_a'] or 'merged'} A ({x['particles']})" for x in summary["models"])
    log(f"[3d] wrote {outdir}: {models}, {t1:.1f}s classes + {t2:.1f}s particles")
    return summary


def _figures(outdir, clf, result, keep, Ak, maps, fss, ress, A, box_a, b, zk):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = clf.dev
    # class averages next to reprojections of the final map at the class orientations
    P = np.zeros((len(Ak), b, b), np.float32)
    for m, vm in enumerate(maps):
        sel = torch.as_tensor(zk == m)
        if sel.any():
            V = Volume(b, dev)
            V.set_real(vm)
            P[sel.numpy()] = torch.fft.irfft2(V.project(Ak[sel.to(dev)]), s=(b, b), norm="forward").cpu().numpy()
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
        ax[2 * r, c].set_title(f"{int(pops[k])}" + (f" · 3D {zk[k] + 1}" if len(set(zk.tolist())) > 1 else ""), fontsize=7)
        ax[2 * r + 1, c].imshow(P[k], cmap="gray")
    fig.suptitle("class averages (rows 1, 3, ..) vs reprojections of the 3D map at their orientations", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reprojections.png"), dpi=90)
    plt.close(fig)
    # FSC
    fig, a = plt.subplots(figsize=(5, 3.2))
    for m, (fs, res) in enumerate(zip(fss, ress)):
        f = fs.cpu().numpy()
        x = np.arange(len(f)) / box_a
        a.plot(x[1:], f[1:], lw=2, label=f"3D {m + 1}: {res:.1f} A" if len(maps) > 1 else None)
    a.axhline(0.143, color="0.5", lw=1, ls="--")
    a.set_xlabel("spatial frequency (1/A)")
    a.set_ylabel("half-map FSC")
    if len(maps) > 1:
        a.legend(fontsize=8)
        a.set_title("half-map FSC per 3D class")
    else:
        a.set_title(f"FSC 0.143 at {ress[0]:.1f} A")
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
    fig, ax = plt.subplots(len(maps), 3, figsize=(7.5, 2.7 * len(maps)), squeeze=False)
    for m, vm in enumerate(maps):
        v = vm.cpu().numpy()
        for j in range(3):
            ax[m, j].imshow(v.sum(j), cmap="gray")
            ax[m, j].axis("off")
        ax[m, 0].set_title(("map" if len(maps) == 1 else f"3D class {m + 1}") + ": projections along z, y, x",
                           fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "map_views.png"), dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- symmetry detection
def _rotate_volumes(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """v (b, b, b) sampled at R @ r for each R (n, 3, 3) -> (n, b, b, b) (rotation of the map by R^T)."""
    b = v.shape[0]
    c = torch.arange(b, device=v.device, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    r = torch.stack([x, y, z], -1).reshape(-1, 3)
    out = []
    for i in range(0, len(R), 64):
        p = (r[None] @ R[i:i + 64].transpose(1, 2)).reshape(-1, b, b, b, 3)
        g = (p + b // 2 + 0.5) / b * 2 - 1
        out.append(F.grid_sample(v[None, None].expand(len(p), 1, b, b, b), g, align_corners=False)[:, 0])
    return torch.cat(out)


def _axis_rotation(axes: torch.Tensor, ang_deg: float) -> torch.Tensor:
    """Rotations by ang about unit axes (n, 3) (Rodrigues)."""
    a = math.radians(ang_deg)
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return torch.stack([torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
                        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
                        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


def detect_symmetry(vmap: torch.Tensor, apix: float, diameter_a: float, res_a: float = 15.0, max_n: int = 12):
    """Score Cn (n = 2..max_n) and Dn point groups on a C1 map: correlation of the map with its copy rotated by
    each group generator, best axis found by search. Returns (rows (group, correlation, score) sorted by score,
    suggested group). score = (cc - cc of an arbitrary axis) / (1 - cc of an arbitrary axis): 1 = perfect."""
    dev = vmap.device
    b0 = vmap.shape[0]
    b = min(b0, max(24, int(round(b0 * apix / (res_a / 2) / 2)) * 2))      # Nyquist ~ res/2
    ft = torch.fft.fftshift(torch.fft.fftn(vmap))
    o = (b0 - b) // 2
    v = torch.fft.ifftn(torch.fft.ifftshift(ft[o:o + b, o:o + b, o:o + b])).real
    ap = apix * b0 / b
    v = v * sphere_mask(b, diameter_a / 2 / ap, 2.0, dev)
    v = v - v.mean()

    def ncc(a, ref):
        a = a - a.mean((1, 2, 3), keepdim=True)
        return (a * ref).sum((1, 2, 3)) / (a.pow(2).sum((1, 2, 3)).sqrt() * ref.norm()).clamp_min(1e-12)

    rot, tilt = sphere_directions(3.0)
    keep = tilt <= 90.01                                      # an axis and its opposite are the same
    th, ph = torch.deg2rad(tilt[keep]), torch.deg2rad(rot[keep])
    axes = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1).to(dev)
    rows, best_axis, base, perp_axis = [], {}, {}, {}
    for n in range(2, max_n + 1):
        cc = ncc(_rotate_volumes(v, _axis_rotation(axes, 360.0 / n)), v)
        i = int(cc.argmax())
        base[f"C{n}"] = float(cc.median())                       # what an arbitrary axis gives for this angle
        rows.append([f"C{n}", float(cc[i])])
        best_axis[n] = axes[i]
    raw = {g: x for g, x in rows}
    for n in range(2, max_n + 1):              # Dn: the Cn axis plus a perpendicular 2-fold
        a = best_axis[n]
        u = torch.linalg.cross(a, torch.tensor([1.0, 0, 0], device=dev) if abs(float(a[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev))
        u = u / u.norm()
        w = torch.linalg.cross(a, u)
        t = torch.deg2rad(torch.arange(0, 180, 2.0, device=dev))
        perp = torch.cos(t)[:, None] * u + torch.sin(t)[:, None] * w
        c2s = ncc(_rotate_volumes(v, _axis_rotation(perp, 180.0)), v)
        perp_axis[n] = perp[int(c2s.argmax())]
        rows.append([f"D{n}", min(raw[f"C{n}"], float(c2s.max()))])
        base[f"D{n}"] = max(base[f"C{n}"], base["C2"])
    # score: the fraction of the gap between an arbitrary axis and perfect agreement that the best axis closes
    out = [(g, x, (x - base[g]) / max(1 - base[g], 1e-6)) for g, x in rows]
    out.sort(key=lambda r: -r[2])
    order = lambda g: (2 if g[0] == "D" else 1) * int(g[1:])
    good = [g for g, x, sc in out if sc >= 0.5]
    sugg = max(good, key=order) if good else "C1"
    ops = torch.eye(3, device=dev)[None]
    if sugg != "C1":
        n = int(sugg[1:])
        cn = torch.stack([_axis_rotation(best_axis[n][None], 360.0 * k / n)[0] for k in range(n)])
        ops = torch.cat([cn, _axis_rotation(perp_axis[n][None], 180.0) @ cn]) if sugg[0] == "D" else cn
    return out, sugg, ops
