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

from .engine import TWO_PI, ctf_2d, freqs, gpu_free, rot_grid, upsample2


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
_SHELLS: dict = {}


def _shells(G: int, pad: int, dev) -> torch.Tensor:
    """Cached int32 shell index (cycles per unpadded box) of a centred G^3 grid (int64 would cost 8 bytes/voxel)."""
    key = (G, pad, str(dev))
    if key not in _SHELLS:
        r = (torch.arange(G, device=dev, dtype=torch.float32) - G // 2) ** 2
        _SHELLS[key] = torch.round(torch.sqrt(r[:, None, None] + r[None, :, None] + r[None, None, :]) / pad).int()
    return _SHELLS[key]


class Volume:
    """A 3D spectrum on a 2x zero-padded, centred grid, in the units of the 2D particle spectra (box b): the
    central slice at orientation A, times the centring phase, is the projection's rfft2 (norm="forward").
    Memory: the spectrum is 64 b^3 bytes; the insert() accumulators (another 96 b^3) exist only on volumes that
    receive data."""

    def __init__(self, b: int, dev, pad: int = 2, ft: torch.Tensor | None = None, pre: bool = False,
                 grid_correct: bool = False):
        self.b, self.pad, self.G, self.dev = b, pad, pad * b, dev
        G = self.G
        # gridding correction: trilinear interpolation in the spectrum multiplies the real-space map by sinc^2(x / G)
        # per axis (0.81 at the box edge for pad 2, 0.41 for pad 1), both when projecting (sampling) and when
        # reconstructing (splatting). set_real divides it out beforehand (pre: the spectrum holds the divided map,
        # real() multiplies it back); real() of a reconstruction divides it out. Simulated 96^3 blobs: projection error
        # vs pad 4 10 % -> 3.5 % (pad 2; pad 1 20 %), reconstruction error 7.7 % -> 3.1 % (pad 2), 25 % -> 11 % (pad 1).
        # Off by default (ab initio, its low-resolution stages: the test object's wrong-basin rate went up with it);
        # the high-resolution refinements turn it on (refine_particles(grid_correct=True)).
        self.grid_correct = grid_correct
        self.pre = pre
        self.mag = None             # 2x2 magnification (optics.py): the volume is sampled at M (kx, ky) of each image
                                    # frequency (anisotropic magnification); the image-centring phase stays on k
        # real, imag channels; index = frequency + G/2
        self.ft = torch.zeros((2, G, G, G), device=dev) if ft is None else ft
        self.num = self.den = None
        fy, fx = freqs(b, dev)
        self.fy, self.fx = fy, fx
        self.checker = (1 - 2 * ((fy[:, None].long() + fx[None, :].long()) % 2)).float()  # centre at b/2
        self.kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # ---- real space <-> spectrum
    def _kernel(self) -> torch.Tensor:
        """sinc^2(pi x / G) for the b voxels of one axis (x from the centre): the trilinear interpolation's
        real-space factor per axis."""
        x = (torch.arange(self.b, device=self.dev, dtype=torch.float32) - self.b // 2) / self.G
        return torch.special.sinc(x) ** 2

    def set_real(self, vol: torch.Tensor):
        """vol: (b, b, b) real map, centre at b/2. The forward transform axis by axis into the padded grid, in slabs
        (at once it held ~4 G^3 complex arrays)."""
        b, G, dev = self.b, self.G, self.dev
        vol = vol.to(dev).float()
        if self.grid_correct:
            k = self._kernel()
            vol = vol / (k[:, None, None] * k[None, :, None] * k[None, None, :])
        pos = (torch.arange(b, device=dev) - b // 2) % G                 # the centred map, ifftshifted
        sh = (torch.arange(G, device=dev) + G // 2) % G                  # fftshift as an index
        s = max(1, int(16e6 // (G * G)))
        x1 = torch.zeros((b, b, G), dtype=torch.complex64, device=dev)
        for i in range(0, b, s):                                         # axis 2
            t = torch.zeros((min(s, b - i), b, G), dtype=torch.complex64, device=dev)
            t[:, :, pos] = vol[i:i + s].to(torch.complex64)
            x1[i:i + s] = torch.fft.fft(t, dim=2, norm="forward")[:, :, sh]
        x2 = torch.zeros((b, G, G), dtype=torch.complex64, device=dev)
        for i in range(0, b, s):                                         # axis 1
            t = torch.zeros((min(s, b - i), G, G), dtype=torch.complex64, device=dev)
            t[:, pos, :] = x1[i:i + s]
            x2[i:i + s] = torch.fft.fft(t, dim=1, norm="forward")[:, sh, :]
        del x1
        self.ft = None
        ft = torch.empty((2, G, G, G), device=dev)
        scale = b * self.pad ** 3
        for j in range(0, G, s):                                         # axis 0, slabs of axis 1
            t = torch.zeros((G, min(s, G - j), G), dtype=torch.complex64, device=dev)
            t[pos] = x2[:, j:j + s]
            t = torch.fft.fft(t, dim=0, norm="forward")[sh] * scale
            ft[0, :, j:j + s], ft[1, :, j:j + s] = t.real, t.imag
        self.ft = ft
        self.pre = self.grid_correct

    def real(self) -> torch.Tensor:
        """The real-space map (b^3): the inverse transform done axis by axis, keeping only the central b of each
        axis after its transform, in slabs (the whole padded grid at once held ~4 complex G^3 arrays: 4 GB at box 256,
        out of memory beside a refinement)."""
        b, G, dev = self.b, self.G, self.dev
        idx = (torch.arange(G, device=dev) + G // 2) % G             # ifftshift as an index
        c = (torch.arange(b, device=dev) - b // 2) % G               # the centre of the (fftshifted) result
        s = max(1, int(16e6 // (G * G)))
        out = torch.empty((b, G, G), dtype=torch.complex64, device=dev)
        for j in range(0, G, s):                                     # axis 0, slabs of axis 1
            jj = idx[j:j + s]
            x = torch.complex(self.ft[0][:, jj], self.ft[1][:, jj])[idx][:, :, idx]
            out[:, j:j + s] = torch.fft.ifft(x, dim=0, norm="forward")[c]
        out2 = torch.empty((b, b, G), dtype=torch.complex64, device=dev)
        for i in range(0, b, s):                                     # axis 1
            out2[i:i + s] = torch.fft.ifft(out[i:i + s], dim=1, norm="forward")[:, c]
        del out
        v = torch.empty((b, b, b), device=dev)
        for i in range(0, b, s):                                     # axis 2
            v[i:i + s] = torch.fft.ifft(out2[i:i + s], dim=2, norm="forward")[:, :, c].real
        v /= (b * self.pad ** 3)
        if self.grid_correct:
            k = self._kernel()
            K = k[:, None, None] * k[None, :, None] * k[None, None, :]
            if self.pre:                                             # undo set_real's pre-correction
                v *= K
            else:                                                    # a reconstruction: the splatting's factor
                v /= K
        return v

    def shell_index(self) -> torch.Tensor:
        return _shells(self.G, self.pad, self.dev)

    # ---- slices
    def _coords(self, A: torch.Tensor, keep: torch.Tensor | None = None):
        """Grid coordinates (x, y, z) of the 2D half-plane frequencies at orientations A: (n, b, bh, 3), or
        (n, P, 3) for the frequencies selected by the boolean mask keep (b, bh)."""
        fy, fx = self.fy, self.fx
        At = A.to(self.dev).transpose(1, 2)            # columns: volume axes in image coordinates
        if keep is None:
            kx, ky = fx[None, None, :, None], fy[None, :, None, None]
            if self.mag is not None:
                M = self.mag
                kx, ky = M[0, 0] * kx + M[0, 1] * ky, M[1, 0] * kx + M[1, 1] * ky
            q = kx * At[:, None, None, :, 0] + ky * At[:, None, None, :, 1]   # A^T (kx, ky, 0)
        else:
            kx = fx[None, :].expand(len(fy), -1)[keep][None, :, None]
            ky = fy[:, None].expand(-1, len(fx))[keep][None, :, None]
            if self.mag is not None:
                M = self.mag
                kx, ky = M[0, 0] * kx + M[0, 1] * ky, M[1, 0] * kx + M[1, 1] * ky
            q = kx * At[:, None, :, 0] + ky * At[:, None, :, 1]
        return q * self.pad

    def project(self, A: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Projection spectra at orientations A (n, 3, 3): (n, b, b//2+1), or (n, P) for the frequencies in keep.
        Chunked so that a few million points are sampled at a time."""
        out = []
        G = self.G
        P = int(keep.sum()) if keep is not None else self.b * (self.b // 2 + 1)
        chunk = max(1, 4_000_000 // P)
        chk = self.checker if keep is None else self.checker[keep]
        for i in range(0, len(A), chunk):
            q = self._coords(A[i:i + chunk], keep)
            g = ((q + G // 2 + 0.5) / G * 2 - 1).reshape(1, len(q), -1, 1, 3)
            s = F.grid_sample(self.ft[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            out.append(torch.complex(s[0], s[1]).reshape(q.shape[:-1]) * chk)
        if not out:
            shape = (0, self.b, self.b // 2 + 1) if keep is None else (0, P)
            return torch.zeros(shape, dtype=torch.complex64, device=self.dev)
        return torch.cat(out)

    def _acc(self):
        if self.num is None:
            G = self.G
            self.num = torch.zeros((G * G * G, 2), device=self.dev)
            self.den = torch.zeros(G * G * G, device=self.dev)

    def insert(self, x: torch.Tensor, w: torch.Tensor, A: torch.Tensor, kmax: float, ops: torch.Tensor | None = None):
        """Accumulate spectra x (n, b, bh) (already CTF-multiplied: x*ctf) with weights w (n, b, bh) (e.g. ctf^2)
        at orientations A, trilinear, plus their Friedel mates; frequencies beyond kmax (cycles/box) are skipped.
        ops: symmetry operators S (each image is also inserted at A @ S). Chunked to ~4M points per pass."""
        self._acc()
        if ops is not None and len(ops) > 1:
            S = len(ops)
            A = (A[:, None] @ ops[None]).reshape(-1, 3, 3)
            x, w = x.repeat_interleave(S, 0), w.repeat_interleave(S, 0)
        keep = self.kr <= kmax
        P = int(keep.sum())
        chunk = max(1, 4_000_000 // max(P, 1))
        for i in range(0, len(A), chunk):
            self._insert(x[i:i + chunk], w[i:i + chunk], A[i:i + chunk], keep)

    def _insert(self, x, w, A, keep):
        G = self.G
        q = self._coords(A, keep)                           # (n, P, 3)
        v = (x * self.checker)[:, keep].reshape(-1)
        ww = w[:, keep].float().reshape(-1)
        q = q.reshape(-1, 3)
        for sgn in (1.0, -1.0):
            p = sgn * q + G // 2
            p0 = torch.floor(p)
            f = p - p0
            p0 = p0.long()
            vr = torch.view_as_real(v if sgn > 0 else v.conj().resolve_conj())
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        ix, iy, iz = p0[:, 0] + dx, p0[:, 1] + dy, p0[:, 2] + dz
                        wt = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                              * (f[:, 2] if dz else 1 - f[:, 2]))
                        ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G) & (iz >= 0) & (iz < G)
                        idx = ((iz * G + iy) * G + ix)[ok]
                        wt = wt[ok]
                        self.num.index_add_(0, idx, vr[ok] * wt[:, None])
                        self.den.index_add_(0, idx, ww[ok] * wt)

    def clear(self):
        self._acc()
        self.num.zero_()
        self.den.zero_()

    def free_acc(self):
        self.num = self.den = None

    def solve(self, reg: torch.Tensor | float):
        """ft = num / (den + reg); reg: scalar or per-shell (in cycles/box b) tensor. In slabs of the first axis straight
        into the new spectrum, the old one released first: done at once it held ~5 volume-sized temporaries (out of
        memory at box 256, pad 2)."""
        G = self.G
        den = self.den.view(G, G, G)
        num = self.num.view(G, G, G, 2)
        self.ft = None
        out = torch.empty((2, G, G, G), device=self.dev)
        sh = self.shell_index() if torch.is_tensor(reg) else None
        cz = max(1, int(64e6 // (G * G)))
        for z in range(0, G, cz):
            d = den[z:z + cz] + (reg[sh[z:z + cz].clamp(max=len(reg) - 1)] if sh is not None else reg)
            out[:, z:z + cz] = (num[z:z + cz] / d[..., None]).permute(3, 0, 1, 2)
        self.ft = out
        self.pre = False

    def filter_shells(self, f: torch.Tensor):
        """Multiply the spectrum by a per-shell factor f (cycles/box b)."""
        self.ft *= f[self.shell_index().clamp(max=len(f) - 1)][None]


def fsc_maps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fourier shell correlation of two real maps (b, b, b), per shell (cycles/box)."""
    n = a.shape[0]
    fa, fb = torch.fft.rfftn(a), torch.fft.rfftn(b)
    k = torch.fft.fftfreq(n, device=a.device) * n
    kz = torch.arange(n // 2 + 1, device=a.device, dtype=torch.float32)
    sh = torch.round(torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)).long().reshape(-1)
    m = sh < n // 2 + 1
    sh = sh[m]
    fa, fb = fa.reshape(-1)[m], fb.reshape(-1)[m]
    nb = n // 2 + 1
    cross = torch.zeros(nb, device=a.device).index_add_(0, sh, (fa * fb.conj()).real)
    pa = torch.zeros(nb, device=a.device).index_add_(0, sh, fa.abs() ** 2)
    pb = torch.zeros(nb, device=a.device).index_add_(0, sh, fb.abs() ** 2)
    return cross / torch.sqrt(pa * pb).clamp_min(1e-30)


def fsc(a: Volume, b: Volume) -> torch.Tensor:
    """Unmasked FSC of two half reconstructions, per shell (cycles/box b)."""
    return fsc_maps(a.real(), b.real())


def masked_fsc(a: Volume, b: Volume, mask: torch.Tensor) -> torch.Tensor:
    """FSC of two half reconstructions after multiplying both by a real-space mask (e.g. the soft particle sphere):
    solvent noise outside the particle no longer dilutes the correlation (cryoSPARC's 'loose mask' FSC)."""
    return fsc_maps(a.real() * mask, b.real() * mask)


def corrected_masked_fsc(h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor, seed: int = 0, parts: bool = False):
    """Masked FSC of two real half maps corrected for the correlation a (tight) mask creates by itself (RELION
    PostProcess): from the shell k_r where the unmasked FSC drops below 0.8, the half maps' phases are randomised
    (whole shells, as the FSC counts them), the maps masked again, and FSC_true = (FSC_masked - FSC_random) /
    (1 - FSC_random) from shell k_r + 3 on; the shells just past k_r keep the masked FSC (as RELION): the mask spreads
    correlation from the intact shells into them, FSC_random is close to FSC_masked there and the formula dips
    (beta-gal: 0.97 -> 0.72 for one shell when it was applied from k_r, with half of shell k_r randomised).
    parts: also return (masked, randomised, unmasked FSC, k_r)."""
    n = h1.shape[0]
    f_un = fsc_maps(h1, h2)
    f_m = fsc_maps(h1 * mask, h2 * mask)
    below = torch.nonzero(f_un < 0.8)
    k_r = int(below[0]) if len(below) else len(f_un) - 1
    k = torch.fft.fftfreq(n, device=h1.device) * n
    kz = torch.arange(n // 2 + 1, device=h1.device, dtype=torch.float32)
    kk = torch.round(torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2))
    g = torch.Generator(device=h1.device).manual_seed(seed)
    out = []
    for h in (h1, h2):
        ft = torch.fft.rfftn(h)
        ph = torch.rand(ft.shape, generator=g, device=h1.device) * TWO_PI
        rnd = torch.where(kk >= k_r, ft.abs() * torch.exp(1j * ph), ft)
        out.append(torch.fft.irfftn(rnd, s=h.shape) * mask)
    f_r = fsc_maps(out[0], out[1])
    f = f_m.clone()
    hi = torch.arange(len(f), device=f.device) > k_r + 2
    f[hi] = (f_m[hi] - f_r[hi]) / (1 - f_r[hi]).clamp_min(1e-6)
    return (f, f_m, f_r, f_un, k_r) if parts else f


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
    """Gaussian low-pass, half amplitude at res_a (separable, in place)."""
    k = box_a / res_a
    r = (torch.arange(vol.G, device=vol.dev, dtype=torch.float32) - vol.G // 2) / vol.pad
    g = torch.exp(-0.5 * (r / max(k, 1.0)) ** 2 * math.log(2) * 2)
    vol.ft *= g[None, :, None, None]
    vol.ft *= g[None, None, :, None]
    vol.ft *= g[None, None, None, :]


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
    sh = vol.shell_index().reshape(-1)
    n = int(sh.max()) + 1
    m = torch.bincount(sh, weights=vol.den, minlength=n) / torch.bincount(sh, minlength=n).clamp_min(1)
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
                      seed: int = 0, n_models: int = 1, keep_top: int = 1, log=print):
    """Initial model(s) from 2D class averages (spectra (K, b, bh) at the classification box; populations (K,)).
    With n_models > 1 every class is also assigned to the model that explains it best (soft, then hard).
    Returns (list of Volumes at box b1, class orientations (M, K, 3, 3) in every model's frame,
    per-class correlation (K,), model of each class (K,), the keep_top best starts as a list of (score, Volumes,
    orientations))."""
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
    starts = []                                               # (score, vols, Ak) of every start
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
                Pb = vols[m].project(Ad, band)
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
        Ak = torch.empty((M, K, 3, 3), device=dev)           # each class's best orientation in every model
        for m in range(M):
            _, r_i, d_i = tis[m]
            Ak[m] = rotz(-angles[r_i[:, 0].cpu()]).to(dev) @ Ad[d_i[:, 0]]   # class = P(rotz(-a) Ad)
        starts.append((score, vols, Ak))
        if best is None or score > best[0]:
            best = (score, vols, Ak, cc_k.clone(), zk.clone())
    log(f"[3d] best start: class agreement {best[0]:.3f} (per class {' '.join(f'{x:.2f}' for x in best[3].tolist())})"
        + (f"; classes per model {torch.bincount(best[4], minlength=M).tolist()}" if M > 1 else ""))
    starts.sort(key=lambda x: -x[0])
    return best[1], best[2], best[3], best[4], starts[:keep_top]


# --------------------------------------------------------------------------- stage 2: particles
def auto_pad(b: int, n_vol: int, dev) -> int:
    """Padding of 3D spectra: 2 if n_vol reconstructions (~60 (2b)^3 bytes each with accumulators and references)
    fit in 60 % of the free memory, else 1 (gridding-corrected; box 512 in 16 GB)."""
    return 2 if n_vol * 60 * (2 * b) ** 3 < 0.6 * _free(dev) else 1


def _to_box(init, b: int, dev) -> torch.Tensor:
    """Real map of a Volume (or a real map itself), Fourier-cropped/padded to box b."""
    r = init.to(dev).float() if torch.is_tensor(init) else init.real()
    b1 = r.shape[0]
    if b1 == b:
        return r
    v0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(r)))
    if b >= b1:
        vb = torch.zeros((b, b, b), dtype=torch.complex64, device=dev)
        o = (b - b1) // 2
        vb[o:o + b1, o:o + b1, o:o + b1] = v0
    else:
        o = (b1 - b) // 2
        vb = v0[o:o + b, o:o + b, o:o + b]
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(vb)).real)


def _free(dev) -> float:
    """Free memory on dev (bytes) after releasing PyTorch's cached blocks."""
    if torch.device(dev).type != "cuda":
        return 8e9
    torch.cuda.empty_cache()
    return float(gpu_free(dev))                 # CELERIS2D_VRAM_GB caps it


def working_box(clf, n_acc: int, n_ft: int, log=print) -> int:
    """Largest even box <= the loaded working box whose 3D volumes fit on the GPU. Per b^3: a padded volume
    with accumulators 160 bytes (n_acc of them), a spectrum-only volume 64 bytes (n_ft), FFT transients ~320;
    plus 1.5 GB for particle batches. Data that would not fit is never pushed to (WSL: swapped) host memory."""
    free = _free(clf.dev)
    per = n_acc * 160 + n_ft * 64 + 320
    b = clf.bmax
    while b > 32 and per * b ** 3 + 1.5e9 > 0.85 * free:
        b -= 2
    if b < clf.bmax:
        log(f"[3d] GPU memory: {free / 1e9:.1f} GB free -> 3D working box {b} px instead of {clf.bmax} "
            f"(Nyquist {2 * clf.box_a / b:.1f} A instead of {2 * clf.box_a / clf.bmax:.1f} A)")
    return b


def global_search(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, refs, half: torch.Tensor, res_a: float,
                  step_deg: float, sigma2: torch.Tensor | None = None, box: int | None = None, log=print,
                  topk: int = 1):
    """Exhaustive orientation search (whole sphere x all in-plane angles, shift kept) of particles against the
    half-set references refs = [ref_half0, ref_half1], as two GEMMs per chunk of orientations:
    ll = Re<ctf conj(x) e_t, P> / sigma2 - 0.5 <ctf^2 / sigma2, |P|^2>. Returns (best A (n, 3, 3), best ll (n,),
    ll at the current pose (n,)); with topk > 1 the best topk orientations (n, topk, 3, 3) and their ll (n, topk)."""
    dev = clf.dev
    box_a, b = clf.box_a, box or clf.bmax
    n = len(idx)
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= 1) & (kr <= box_a / res_a)
    ph = TWO_PI / box_a
    isig = torch.ones(int(band.sum()), device=dev) if sigma2 is None else 1.0 / sigma2[torch.round(kr[band]).long()]
    rot, tilt = sphere_directions(step_deg)
    R = int(round(360 / step_deg))
    psi = torch.arange(R, dtype=torch.float32) * 360 / R
    Ad = euler_to_matrix(rot, tilt, torch.zeros_like(rot))
    Ah = (rotz(psi)[:, None] @ Ad[None]).reshape(-1, 3, 3).to(dev)       # (H, 3, 3)
    H = len(Ah)
    best_ll = torch.full((n,), -float("inf"), device=dev)
    best_h = torch.zeros(n, dtype=torch.long, device=dev)
    if topk > 1:
        top_ll = torch.full((n, topk), -float("inf"), device=dev)
        top_h = torch.zeros((n, topk), dtype=torch.long, device=dev)
    cur_ll = torch.zeros(n, device=dev)
    A, t = A.to(dev), t.to(dev)
    Fb = int(band.sum())
    free = _free(dev)
    npc = int(max(256, min(n, 0.1 * free / (Fb * 12))))           # particles per chunk (band spectra)
    hc = int(max(256, min(H, 0.15 * free / (npc * 20 + Fb * 24))))   # orientations per chunk (scores + projections)
    for h in (0, 1):
        sel_h = torch.nonzero(half.to(dev) == h).squeeze(1)
        for p0 in range(0, len(sel_h), npc):
            sel = sel_h[p0:p0 + npc]
            j = idx[sel.cpu()]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[sel]
            e_t = torch.exp(-1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            X = (x.conj() * e_t * ctf)[:, band] * isig
            C2 = (ctf * ctf)[:, band] * isig
            del x, e_t, ctf
            P = refs[h].project(A[sel], band)
            cur_ll[sel] = (X * P).real.sum(1) - 0.5 * (C2 * P.abs() ** 2).sum(1)
            Xr, Xi = X.real.contiguous(), X.imag.contiguous()
            del X
            for c in range(0, H, hc):
                P = refs[h].project(Ah[c:c + hc], band)                           # (Hc, F)
                ll = Xr @ P.real.T - Xi @ P.imag.T - 0.5 * (C2 @ (P.abs() ** 2).T)   # (n_chunk, Hc)
                if topk > 1:
                    v, k = torch.topk(torch.cat([top_ll[sel], ll], 1), topk, dim=1)
                    hh = torch.cat([top_h[sel], torch.arange(c, c + ll.shape[1], device=dev).expand(len(sel), -1)], 1)
                    top_ll[sel], top_h[sel] = v, hh.gather(1, k)
                    continue
                v, k = ll.max(1)
                upd = v > best_ll[sel]
                best_ll[sel[upd]] = v[upd]
                best_h[sel[upd]] = k[upd] + c
    if topk > 1:
        return Ah[top_h], top_ll, cur_ll
    return Ah[best_h], best_ll, cur_ll


def global_step(res_a: float, diameter_a: float, max_hyp: float = 2.0e5) -> float:
    """Angular step of a global search: the resolution-limited step res/diameter, but coarse enough that the whole
    sphere x all in-plane angles (8 pi^2 / step^3 orientations, step in radians) stays below max_hyp. A global search only has to
    find the right basin (local refinement follows); without the cap a 300 A particle at 12 A needs 1.2M
    orientations per particle (J341 at 4 A ran > 40 min)."""
    return max(2.0, math.degrees(res_a / diameter_a), math.degrees((8 * math.pi ** 2 / max_hyp) ** (1 / 3)))


def _view_weights(A: torch.Tensor, cap: float) -> torch.Tensor:
    """Rebalance preferred orientation: particles in over-populated viewing directions (~10 deg bins) are
    down-weighted so no bin counts more than cap x the median occupied bin."""
    rot, tilt = sphere_directions(10.0)
    th, ph = torch.deg2rad(tilt), torch.deg2rad(rot)
    bins = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1).to(A.device)
    k = (A[:, 2, :] @ bins.T).argmax(1)                                # viewing direction = third row
    cnt = torch.bincount(k, minlength=len(bins)).float()
    lim = cap * cnt[cnt > 0].median()
    return (lim / cnt[k]).clamp(max=1.0)


def view_coverage(A: torch.Tensor, sym: str = "C1") -> dict:
    """How the viewing directions (third rows of A, folded by the symmetry group) spread over the sphere, in ~10 deg
    bins: the fraction of bins with a real share of the particles (relative to the group's share of the sphere) and
    the share of
    particles in the busiest 10 % of bins. A map from a narrow range of views is smeared along the missing
    directions, whatever its FSC says."""
    ops = symmetry_ops(sym).to(A.device)
    rot, tilt = sphere_directions(10.0)
    th, ph = torch.deg2rad(tilt), torch.deg2rad(rot)
    bins = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1).to(A.device)
    d = A[:, 2, :]                                            # viewing direction in the map frame
    best = (d @ bins.T)
    for S in ops[1:]:                                         # fold symmetry-equivalent directions
        best = torch.maximum(best, (d @ S) @ bins.T)
    k = best.argmax(1)
    cnt = torch.bincount(k, minlength=len(bins)).float()
    # a direction counts if it holds at least a quarter of what a uniform spread over the group's share would give
    # (with 10^5+ particles a few strays land everywhere)
    expect = cnt.sum() * len(ops) / len(bins)
    occ = float((cnt >= 0.25 * expect).float().mean()) * len(ops)
    top = torch.sort(cnt, descending=True).values
    share = float(top[:max(1, len(bins) // (10 * len(ops)))].sum() / cnt.sum().clamp_min(1))
    return dict(occupied=round(min(1.0, occ), 3), top10_share=round(share, 3))


def solvent_mask(vmap: torch.Tensor, apix: float, lowpass_a: float = 15.0, dilate_a: float = 4.0,
                 soft_a: float = 6.0) -> torch.Tensor:
    """Automatic soft solvent mask: map low-passed to lowpass_a, thresholded at 15% of its maximum,
    dilated by dilate_a and given a cosine edge of soft_a (both in A)."""
    b = vmap.shape[0]
    dev = vmap.device
    k = torch.fft.fftfreq(b, device=dev) * b
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    lp = torch.fft.ifftn(torch.fft.fftn(vmap) * torch.exp(-0.5 * (kk / (b * apix / lowpass_a)) ** 2)).real
    m = (lp > 0.15 * lp.max()).float()[None, None]
    for _ in range(int(round(dilate_a / apix))):                       # grow by one voxel per step
        m = F.max_pool3d(m, 3, 1, 1)
    out, shell = m.clone(), m
    n_s = max(1, int(round(soft_a / apix)))
    for i in range(n_s):
        shell = F.max_pool3d(shell, 3, 1, 1)
        out = torch.maximum(out, shell * (0.5 + 0.5 * math.cos(math.pi * (i + 1) / (n_s + 1))))
    return out[0, 0]


def particle_sgd(clf, idx: torch.Tensor, t: torch.Tensor, init: Volume, sym: str = "C1", n_iter: int = 30,
                 res_start: float = 30.0, res_end: float = 12.0, batch: int = 2000, eta: float = 0.5, seed: int = 0,
                 topk: int = 1, log=print):
    """Stochastic projection matching on the particles (the role of cryoSPARC's SGD ab initio): each round, a random
    batch of particles is searched over all orientations against the current map, the map is moved part of the way
    (eta) towards the batch's reconstruction, and the resolution goes from res_start to res_end. Thousands of
    particles constrain the structure far more than a few class averages, and batch noise lets it leave a wrong
    start. Runs on a small box (Nyquist ~ res_end / 2). Returns (Volume, orientations of all particles (n, 3, 3))."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    bs = min(clf.bmax, int(math.ceil(2 * box_a / (res_end / 1.2) / 2)) * 2)
    n = len(idx)
    ops = symmetry_ops(sym).to(dev)
    mask = sphere_mask(bs, diam / 2 / (box_a / bs), 2.0, dev)
    V = Volume(bs, dev)
    V.set_real(_to_box(init, bs, dev))
    lowpass(V, res_start, box_a)
    fy, fx = freqs(bs, dev)
    ph = TWO_PI / box_a
    g = torch.Generator().manual_seed(seed + 11)
    t = t.to(dev).float()
    zeros = torch.zeros(n, dtype=torch.long)
    kr_i = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)).long().reshape(-1)
    nsh = int(kr_i.max()) + 2
    sh_i, cnt_i = kr_i, torch.bincount(kr_i, minlength=nsh).float().clamp_min(1)
    sigma2 = None
    t0 = time.time()
    for it in range(n_iter):
        f = it / max(n_iter - 1, 1)
        res = res_start * (res_end / res_start) ** f
        step = global_step(res, diam)
        sel = torch.randperm(n, generator=g)[:batch]
        j = idx[sel]
        if topk > 1:
            # soft poses: the batch's own power spectrum as the noise model (at low SNR a particle is mostly noise),
            # each particle inserted at its topk orientations weighted by their posterior probability
            if sigma2 is None:
                xs_ = clf._fetch(j[:1024], dev, bs)
                sigma2 = torch.bincount(sh_i, weights=(xs_.abs() ** 2).mean(0).reshape(-1), minlength=nsh)
                sigma2 = (sigma2 / cnt_i).clamp_min(1e-12)
                del xs_
            Ab, llb, _ = global_search(clf, j, torch.eye(3).expand(len(j), 3, 3), t[sel.to(dev)], [V, V],
                                       zeros[:len(j)], res, step, sigma2=sigma2, box=bs, topk=topk)
            wb = torch.softmax(llb, 1)                                                  # (nb, topk)
        else:
            Ab, _, _ = global_search(clf, j, torch.eye(3).expand(len(j), 3, 3), t[sel.to(dev)], [V, V],
                                     zeros[:len(j)], res, step, box=bs)
            Ab, wb = Ab[:, None], torch.ones(len(j), 1, device=dev)
        Vb = Volume(bs, dev)
        for i in range(0, len(j), 1024):
            jj = j[i:i + 1024]
            x = clf._fetch(jj, dev, bs)
            ctf = ctf_2d(clf.ctf_p[jj].to(dev), clf.optics, fy, fx, box_a)
            tt = t[sel[i:i + 1024].to(dev)]
            xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            for q in range(Ab.shape[1]):
                wq = wb[i:i + 1024, q, None, None]
                Vb.insert(xs * ctf * wq, ctf * ctf * wq, Ab[i:i + 1024, q], kmax=min(bs / 2 - 1, box_a / res + 2), ops=ops)
        _regularise_solve(Vb, 0.05)
        Vb.free_acc()
        lowpass(Vb, res, box_a)
        _real_space_constraints(Vb, mask, positive=f < 0.5, center=True)
        V.ft = (1 - eta) * V.ft + eta * Vb.ft
        lowpass(V, res, box_a) if it == 0 else None
    # every particle's orientation in the final map
    A, _, _ = global_search(clf, idx, torch.eye(3).expand(n, 3, 3), t, [V, V], zeros, res_end,
                            global_step(res_end, diam), box=bs)
    log(f"[3d] particle SGD: {n_iter} rounds of {min(batch, n)} particles, {res_start:.0f} -> {res_end:.0f} A "
        f"({time.time() - t0:.1f}s)")
    return V, A


def cross_fit(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, halves, res_a: float, box: int,
              n_max: int = 8000, seed: int = 0) -> torch.Tensor:
    """Normalised correlation (per particle, a fixed random subset of n_max) of particles with the projection of the half map they did NOT go into (the
    half split of refine_particles), at their poses A, t, over frequencies up to res_a. Comparable between maps
    only at the same res_a: a correlation over a lower band is higher for any map, and a particle compared with
    a map it helped build agrees with it whatever the map is."""
    dev, box_a, b = clf.dev, clf.box_a, box
    n = len(idx)
    half = (torch.randperm(n, generator=torch.Generator().manual_seed(1)) % 2).to(dev)
    sub = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 5))[:n_max].to(dev)
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= 1) & (kr <= box_a / res_a)
    fyb = fy[:, None].expand(b, b // 2 + 1)[band]
    fxb = fx[None, :].expand(b, b // 2 + 1)[band]
    ph = TWO_PI / box_a
    A, t = A.to(dev).float(), t.to(dev).float()
    out = torch.empty(len(sub), device=dev)
    for i in range(0, len(sub), 1024):
        s = sub[i:i + 1024]
        j = idx[s.cpu()]
        x = clf._fetch(j, dev, b)[:, band]
        ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
        e_t = torch.exp(-1j * ph * (fxb * t[s, 0, None] + fyb * t[s, 1, None]))
        mdl = torch.empty_like(x)
        for h in (0, 1):
            k = torch.nonzero(half[s] == h).squeeze(1)
            if len(k):
                mdl[k] = halves[1 - h].project(A[s[k]], band)
        mdl = mdl * ctf * e_t
        cc = (x.conj() * mdl).real.sum(1) / ((x.abs() ** 2).sum(1) * (mdl.abs() ** 2).sum(1)).sqrt().clamp_min(1e-12)
        out[i:i + 1024] = cc
    return out


def refine_particles(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, inits, z: torch.Tensor | None = None,
                     sym: str = "C1", n_iter: int = 8, res_start: float = 15.0, step_start: float | None = None,
                     batch: int = 256, n_local: int = 1, step_scale: float | None = None, global_at: tuple = (),
                     global_res: float = 12.0, drop_frac: float = 0.0, rebalance: float = 0.0,
                     box: int | None = None, relax_ops: torch.Tensor | None = None, relax_iters: int = 0,
                     step_min: float = 0.5,
                     half: torch.Tensor | None = None, soft: bool = False, solvent_mask=None,
                     solvent_from_a: float = 15.0, pad: int | None = None, grid_correct: bool = False,
                     nu: bool = False, log=print):
    """Local refinement of particle orientations A and shifts t (A, model shift) against CTF-modulated projections,
    gold-standard halves (half: (n,) 0/1 per particle, e.g. the split of the refinement the poses come from - a
    particle must stay in the half it was aligned or polished against; default a random split, seed 1). inits:
    starting Volume or list of M Volumes. A: (n, 3, 3) or, with several models, the pose in every model's own frame (n, M, 3, 3); t: (n, 2) or (n, M, 2); z: (n,) starting model. Each iteration
    refines the pose in every model and puts the particle in the model that fits best.
    relax_ops: symmetry operators of the group the start map was solved in; for the first relax_iters iterations
    the local search also covers every symmetry-related copy of each pose (RELION's relaxed symmetry), so a
    particle can move to the copy that matches an asymmetric feature; a refinement started from a symmetric map
    otherwise stays symmetric.
    global_at: iterations (1-based) before which every particle also gets an exhaustive orientation search at
    global_res (single model); a particle takes the global pose if it fits better than its current one.
    drop_frac: the worst-fitting fraction of particles (normalised fit, compared within defocus groups) is left out
    of the reconstructions; scores are recomputed every iteration, so particles can come back.
    rebalance: > 0 caps over-represented viewing directions at rebalance x the median direction in the
    reconstructions (preferred orientation).
    soft (several models): every particle goes into every model weighted by its posterior probability over the
    models (each at its own pose there), RELION-style; the temperature starts where the typical particle is ~60/40
    and falls to 1 (the true posterior) over the first half of the iterations. Off: each particle in its best model
    only. refine_particles.last_post holds the final posteriors (n, M).
    solvent_mask (one model): "auto" - a solvent mask made from the current map every iteration once it is better than
    solvent_from_a (masking.auto_mask, noise from the half maps; cryoSPARC's dynamic mask) - or a fixed mask (b^3
    tensor). It replaces the particle sphere for the FSC that sets the resolution and weights the references (then
    corrected by phase randomisation, as it is tight), for flattening the references' solvent, and for the final map:
    the references lose the solvent noise and keep more of the signal. refine_particles.last_mask holds the last one.
    pad: padding of the 3D spectra (None: 2 if the volumes, ~60 (2b)^3 bytes for one model, fit in 60 % of the free
    memory, else 1 - box 512 in 16 GB). grid_correct: gridding correction of every volume (Volume; pad 1 needs it).
    nu (one model): non-uniform regularisation (nufilter.adaptive_filter) below 15 A: each half's reference is
    low-passed region by region, the cutoff chosen by cross-validation against the other half, instead of the global
    FSC filter. refine_particles.last_nu holds the last cutoff statistics.
    Returns (maps (M, b, b, b), A (n, 3, 3), t (n, 2) in the chosen model, z, fsc (M, shells), resolutions (M,),
    half Volumes [[h0, h1] per model])."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    b = box or clf.bmax
    n = len(idx)
    inits = inits if isinstance(inits, (list, tuple)) else [inits]
    bp = int(max(64, min(2048, 2e8 // (b * (b // 2 + 1) * 8))))   # particles per back-projection batch
    M = len(inits)
    z = torch.zeros(n, dtype=torch.long, device=dev) if z is None else z.clone().to(dev).long()
    ops = symmetry_ops(sym).to(dev)
    if half is None:
        half = torch.randperm(n, generator=torch.Generator().manual_seed(1)) % 2
    half = torch.as_tensor(half).long().cpu()
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
    if pad is None:
        pad = auto_pad(b, M, dev)
        if pad == 1:
            log(f"[3d] box {b}: 3D spectra without padding (pad 1, gridding-corrected) to fit the GPU")
    gc = grid_correct or pad == 1
    vols = [[Volume(b, dev, pad=pad, grid_correct=gc), Volume(b, dev, pad=pad, grid_correct=gc)] for _ in range(M)]
    v_init = [_to_box(v, b, dev) for v in inits]
    res = [res_start] * M
    fscs = [None] * M
    alive = [True] * M                    # a model that loses (almost) all its particles is retired
    wpart = torch.ones(n, device=dev)     # per-particle reconstruction weight (dropped particles 0, rebalancing < 1)
    score = torch.zeros(n, device=dev)    # normalised fit of the best pose (for dropping particles)
    dfz = clf.ctf_p[idx, :2].mean(1)      # defocus, to compare fits within defocus groups
    dgrp = torch.bucketize(dfz, torch.quantile(dfz, torch.linspace(0, 1, 9)[1:-1])).to(dev)
    sigma2 = None
    post = F.one_hot(z, M).float()       # posterior over models (one-hot unless soft)
    T0 = None
    step = step_start or max(1.0, math.degrees(res_start / diam))
    t0 = time.time()
    history = []                          # per iteration: alignment limit, step, share moved, resolution after it
    kmax_used = [float(b)] * M            # highest shell inserted in the last reconstruction

    def backproject():
        for vm in vols:
            for v in vm:
                v.clear()
        for i in range(0, n, bp):
            j = idx[i:i + bp]
            x = clf._fetch(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            zz = z[i:i + bp]
            sl = ar_n[i:i + bp]
            hh = half[i:i + bp].to(dev)
            wp = wpart[i:i + bp]
            pp = post[i:i + bp]
            if not soft:
                tt, AA = t[sl, zz], A[sl, zz]
                xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            for m in range(M):
                kmax = min(b / 2 - 1, 1.3 * box_a / res[m] + 6)   # beyond the current resolution, so FSC can grow
                kmax_used[m] = kmax
                if soft:     # every particle at its pose in this model, weighted by its posterior
                    tt, AA = t[sl, m], A[sl, m]
                    xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
                for h in (0, 1):
                    sel = (hh == h) & (wp > 0) & ((pp[:, m] > 1e-3) if soft else (zz == m))
                    if sel.any():
                        w3 = (wp[sel] * (pp[sel, m] if soft else 1.0))[:, None, None]
                        vols[m][h].insert((xs * ctf)[sel] * w3, (ctf * ctf)[sel] * w3, AA[sel], kmax=kmax, ops=ops)

    refs = None
    cur_mask = None                 # solvent mask in use (solvent_mask), else the particle sphere
    for it in range(n_iter + 1):
        # ---- reconstruct halves, FSC, filter
        refs = r = rm = None        # the last alignment's references (also bound to the loop variables that built them)
        torch.cuda.empty_cache()    # and cached batch blocks are not needed while the halves are back-projected
        if it == 0:
            for m in range(M):
                for v in vols[m]:
                    v.set_real(v_init[m])
        else:
            backproject()
            for m in range(M):
                for v in vols[m]:
                    _regularise_solve(v, 0.02)
                    v.free_acc()        # 3/2 of the volume's size; made again by the next back-projection (box 256:
                                        # 3.2 GB for two halves that sat idle through the alignment)
                if solvent_mask is not None and M == 1 and (torch.is_tensor(solvent_mask) or res[m] < solvent_from_a):
                    from .masking import auto_mask
                    h1_, h2_ = vols[m][0].real(), vols[m][1].real()
                    if torch.is_tensor(solvent_mask):
                        cur_mask = solvent_mask.to(dev)
                    else:                   # the masked average and half difference, once each (box 512: memory)
                        cur_mask = None
                        avg_ = (h1_ + h2_).mul_(0.5).mul_(mask)
                        dif_ = (h1_ - h2_).mul_(0.5).mul_(mask)
                        cur_mask = auto_mask(avg_, box_a / b, res_a=res[m], k_sigma=5.0, halves=(dif_, None),
                                             diameter_a=diam)
                        del avg_, dif_
                    fscs[m] = corrected_masked_fsc(h1_, h2_, cur_mask)
                    del h1_, h2_
                else:
                    fscs[m] = masked_fsc(vols[m][0], vols[m][1], mask)
                # shells beyond the reconstruction limit hold only what the mask leaks from lower shells (correlated
                # between the halves): not signal
                fscs[m][int(kmax_used[m]) + 1:] = 0
                res[m] = max(resolution_from_fsc(fscs[m], box_a), 2.2 * box_a / b)
            if history:
                history[-1]["res"] = [round(r, 3) if alive[m] else None for m, r in enumerate(res)]
                history[-1]["seconds"] = round(time.time() - t0, 1)
        if it == n_iter:
            break
        live = [m for m in range(M) if alive[m]]
        # all models are compared at the same resolution (the worst live one), so a model with more particles
        # (a cleaner reconstruction) does not win particles just for being less noisy
        al_res = max(res[m] for m in live) if M > 1 else res[0]
        # align with frequencies somewhat beyond the current resolution (FSC weighting damps their noise);
        # stopping exactly at it makes the resolution self-limiting
        k_al = min(b / 2 - 1, 1.15 * box_a / al_res + 3)
        refs = []
        use_nu = nu and M == 1 and fscs[0] is not None and res[0] < solvent_from_a
        if use_nu:
            from .nufilter import adaptive_filter
            hr_ = [vols[0][0].real(), vols[0][1].real()]
            refs.append([])
            nu_info = {}
            for h in (0, 1):
                f_ = adaptive_filter(hr_[h], hr_[1 - h], box_a / b, res[0], mask=cur_mask,
                                     info=nu_info if h == 0 else None)
                r = Volume(b, dev, pad=pad, grid_correct=gc)
                r.set_real(f_ * (mask if cur_mask is None else cur_mask))
                del f_
                lowpass(r, box_a / k_al, box_a)
                refs[0].append(r)
            del hr_
            refine_particles.last_nu = {k: v for k, v in nu_info.items() if k != "cutoff_map_a"}
            if it == 1 or it == n_iter - 1:
                log(f"[3d] non-uniform filter: cutoffs in the mask {nu_info['cutoff_p05_a']:.1f} / "
                    f"{nu_info['cutoff_median_a']:.1f} / {nu_info['cutoff_p95_a']:.1f} A (5 / 50 / 95 %)")
            nu_info = None
        for m in (range(M) if not use_nu else ()):
            rm = []
            for v in vols[m]:
                r = Volume(b, dev, pad=pad, ft=v.ft.clone(), pre=v.pre, grid_correct=gc)
                if fscs[m] is not None and M == 1:
                    r.filter_shells(torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0)))
                lowpass(r, box_a / k_al, box_a)
                _real_space_constraints(r, mask if cur_mask is None else cur_mask, positive=False, center=False)
                rm.append(r)
            refs.append(rm)
        # ---- global re-search: particles that inherited a wrong pose (mixed 2D classes) can jump
        if M == 1 and (it + 1) in global_at:
            g_res = max(global_res, al_res)
            g_step = global_step(g_res, diam)
            tg = time.time()
            Ag, llg, llc = global_search(clf, idx, A[:, 0], t[:, 0], refs[0], half, g_res, g_step, sigma2, box=b,
                                         log=log)
            # adopt the global pose only if it is clearly better (a few units of log-likelihood): grid noise
            # between equivalent poses must not move particles around
            take = llg > llc + 3.0
            if take.any():
                Qs = Ag[take].transpose(1, 2)[:, None] @ (A[take, 0][:, None] @ ops[None])   # vs every equivalent pose
                d = torch.rad2deg(torch.arccos(((Qs.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))).min(1).values
                A[take, 0] = Ag[take]
                big = d > 3 * g_step
                log(f"[3d] global search at {g_res:.0f} A ({g_step:.1f} deg): {float(take.float().mean()) * 100:.1f}% of "
                    f"particles improved, {float(big.float().sum()) / n * 100:.1f}% moved to a different view "
                    f"({time.time() - tg:.1f}s)")
            step = max(step, g_step / 2)                                 # resume local search around the new poses
        # ---- local search
        band = (kr >= 1) & (kr <= k_al)
        if step_scale is not None:
            step = max(step_min, step_scale * math.degrees(al_res / diam))
        Rl = local_rotations(step, n_local).to(dev)               # ((2n+1)^2, 3, 3) out-of-plane
        dpsi = torch.arange(-n_local, n_local + 1, dtype=torch.float32) * step
        Rc = (rotz(dpsi).to(dev)[:, None] @ Rl[None]).reshape(-1, 3, 3)   # (J, 3, 3) applied on the left: A' = Rc @ A
        xo = relax_ops is not None and it < relax_iters
        J0 = len(Rc)
        J = J0 * (len(relax_ops) if xo else 1)
        jc = (J0 // 2) * (len(relax_ops) if xo else 1)                 # index of the unperturbed pose
        if xo:
            ops_x = relax_ops.to(dev).float()
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
        # particles per search batch: projections, CTF-weighted copies and products (~4 complex arrays of
        # M x J x F per particle) must fit in a third of the free memory
        Fb = int(band.sum())
        bt = int(max(8, min(batch, _free(dev) / 4 / (M * J * Fb * 8 * 4 + M * J * Sh * 8))))
        llm_all = torch.zeros((n, M), device=dev)
        for i in range(0, n, bt):
            j = idx[i:i + bt]
            nb = len(j)
            hh = half[i:i + bt].to(dev)
            x = clf._fetch(j, dev, b)[:, band]                    # (nb, F)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)[:, band]
            if xo:   # around every symmetry-related copy of the pose: (nb, M, J0 * S, 3, 3)
                Ac = (Rc[None, None, :, None] @ (A[i:i + bt][:, :, None, None] @ ops_x[None, None, None])).reshape(
                    nb, M, J, 3, 3)
            else:
                Ac = Rc[None, None] @ A[i:i + bt][:, :, None]     # (nb, M, J, 3, 3): around the pose in each model
            P = torch.empty((nb, M, J, x.shape[1]), dtype=torch.complex64, device=dev)
            for m in range(M):
                for h in (0, 1):
                    sel = torch.nonzero(hh == h).squeeze(1)
                    if len(sel):
                        P[sel, m] = refs[m][h].project(Ac[sel, m].reshape(-1, 3, 3), band).reshape(len(sel), J, -1)
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
            changed += float((bj != jc).float().sum() + (bs != Sh // 2).float().sum()) / 2
            # residual power for the noise model
            mdl = CP[ar, bm, bj] * e_t[ar, bm] * E[bs]
            score[i:i + bt] = ((x.conj() * mdl).real.sum(1)
                               / ((x.abs() ** 2).sum(1) * (mdl.abs() ** 2).sum(1)).sqrt().clamp_min(1e-12))
            r2 = (x - mdl).abs() ** 2
            res_pow.index_add_(0, shb, r2.sum(0))
            res_cnt.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
        # the last batch's arrays (sized from free memory: gigabytes) are not needed through the back-projection
        P = CP = Z = cross = pw = ll = e_t = mdl = r2 = x = ctf = Ac = None
        if M > 1:
            lm = llm_all.clone()
            lm[:, [m for m in range(M) if not alive[m]]] = -float("inf")
            if soft:
                if T0 is None:   # the typical particle starts near 60/40
                    top2 = lm.topk(2, dim=1).values
                    T0 = max(1.0, float((top2[:, 0] - top2[:, 1]).nan_to_num(0).median()) / 0.4)
                f_ = min(1.0, it / max(1.0, n_iter / 2))
                post = torch.softmax(lm / T0 ** (1 - f_), dim=1)
            else:
                post = F.one_hot(z, M).float()
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
                post[:, m] = 0
                post = post / post.sum(1, keepdim=True).clamp_min(1e-12)
        pops = torch.bincount(z, minlength=M).tolist()
        log(f"[3d] particles it {it + 1}/{n_iter}: align {al_res:.1f} A, step {step:.2f} deg, "
            f"{changed / n * 100:.0f}% moved" + (f", {switched / n * 100:.1f}% switched model, per model {pops}" if M > 1 else "")
            + (f", mean max posterior {float(post.max(1).values.mean()):.3f}" if soft and M > 1 else "")
            + f" ({time.time() - t0:.1f}s)")
        history.append(dict(it=it + 1, align_res=round(al_res, 3), step=round(step, 3), moved=round(changed / n, 4),
                            switched=round(switched / n, 4), pops=pops,
                            **({"max_posterior": round(float(post.max(1).values.mean()), 4)} if M > 1 else {})))
        if step_scale is None and changed / n < 0.3:
            step = max(step_min, step / 2)
        wpart = torch.ones(n, device=dev)
        if drop_frac > 0:
            rel = torch.empty_like(score)        # fit relative to particles of similar defocus
            for g_ in range(int(dgrp.max()) + 1):
                sel = dgrp == g_
                if sel.any():
                    rel[sel] = score[sel] / score[sel].median().clamp_min(1e-6)
            wpart[rel < torch.quantile(rel, drop_frac)] = 0
        if rebalance > 0:
            wpart = wpart * _view_weights(A[ar_n, z], rebalance)
    # final maps: average of halves, FSC-weighted
    maps = []
    for m in range(M):
        if fscs[m] is None or not alive[m]:
            res[m] = float("nan")
        out = Volume(b, dev, pad=pad, ft=0.5 * (vols[m][0].ft + vols[m][1].ft), pre=vols[m][0].pre, grid_correct=gc)
        if fscs[m] is not None:
            out.filter_shells(torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0)))
        maps.append((out.real() * (mask if cur_mask is None else cur_mask)).nan_to_num(0))
    fscs = [f if f is not None else torch.zeros(b // 2 + 1, device=dev) for f in fscs]
    unmasked = [resolution_from_fsc(fsc(vols[m][0], vols[m][1]), box_a) if alive[m] and fscs[m] is not None
                else float("nan") for m in range(M)]
    log(f"[3d] final resolution (FSC 0.143, {'soft sphere mask' if cur_mask is None else 'solvent mask, corrected'}) {' / '.join(f'{r:.1f}' if r == r else 'merged' for r in res)} A"
        f" (unmasked {' / '.join(f'{r:.1f}' if r == r else '-' for r in unmasked)} A), {time.time() - t0:.1f}s")
    refine_particles.last_weights = wpart.cpu()   # read by run(): which particles the final maps used
    refine_particles.last_score = score.cpu()
    refine_particles.last_half = half.cpu()          # gold-standard half set of every particle (0 / 1)
    refine_particles.last_history = history
    refine_particles.last_post = post.cpu()
    refine_particles.last_ll = llm_all.cpu() if n_iter > 0 else None   # per model, best pose, last pass
    refine_particles.last_mask = cur_mask
    return torch.stack(maps), A[ar_n, z], t[ar_n, z], z, torch.stack(fscs), res, vols


def classify3d(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, n_models: int, res_a: float,
               sym: str = "C1", n_iter: int = 20, z0: torch.Tensor | None = None, seed: int = 0,
               expand_ops: torch.Tensor | None = None, box: int | None = None, log=print):
    """3D classification at fixed poses (from a consensus refinement): M models seeded from random particle subsets
    (or the labels z0), soft assignment (posterior over models), every model filtered to the same resolution res_a.
    expand_ops: operators R of a (pseudo-)symmetry of the consensus map (S, 3, 3): each particle is also tried at
    A @ R (equally good for the consensus, not for a model that breaks the symmetry) and keeps its best per model.
    Returns (maps (M, b, b, b), z (n,), posteriors (n, M), per-model half-map resolution, half Volumes, poses (n, 3, 3))."""
    dev = clf.dev
    box_a, diam, b = clf.box_a, clf.p.diameter_a, box or clf.bmax
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
    X = torch.eye(3, device=dev)[None] if expand_ops is None else expand_ops.to(dev).float()
    S = len(X)
    B = int(max(32, min(1024, _free(dev) / 4 / (b * (b // 2 + 1) * 8 * 5 + S * int(band.sum()) * 8 * 3))))
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
                v.free_acc()
            r = Volume(b, dev, ft=0.5 * (vols[m][0].ft + vols[m][1].ft))
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
                    mdl = refs[m].project(A[i:i + B], band) * ctf[:, band] * e_t
                    r2 = (xb - mdl).abs() ** 2
                    res_pow.index_add_(0, shb, r2.sum(0))
                    res_cnt.index_add_(0, shb, torch.full_like(shb, nb, dtype=torch.float32))
                    continue
                As = (A[i:i + B][:, None] @ X[None]).reshape(-1, 3, 3)
                mdl = refs[m].project(As, band).reshape(nb, S, -1) * (ctf[:, band] * e_t)[:, None]
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
        c = masked_fsc(vols[m][0], vols[m][1], mask)
        res.append(resolution_from_fsc(c, box_a) if int((z == m).sum()) > 10 else float("nan"))
        out = Volume(b, dev, ft=0.5 * (vols[m][0].ft + vols[m][1].ft))
        out.filter_shells(torch.sqrt((2 * c / (1 + c)).clamp(0, 1).nan_to_num(0)))
        maps.append((out.real() * mask).nan_to_num(0))
    return torch.stack(maps), z, post, res, vols, A @ X[s_best[torch.arange(n, device=dev), z]]


def _class_poses(A: torch.Tensor, kp: torch.Tensor, psi: torch.Tensor, K: int) -> torch.Tensor:
    """Orientation of each 2D class from its particles' 3D orientations: undo each particle's in-plane angle
    (particle = rot_psi(class)) and take the member whose class pose agrees best with the others (a medoid)."""
    Ac = rotz(-psi.float()).to(A.device) @ A                  # class-frame orientation implied by each particle
    out = torch.eye(3, device=A.device).repeat(K, 1, 1)
    for k in range(K):
        m = torch.nonzero(kp.to(A.device) == k).squeeze(1)
        if not len(m):
            continue
        c = Ac[m[torch.randperm(len(m))[:200]]]
        tr = (c[:, None].transpose(-1, -2) @ c[None]).diagonal(dim1=-2, dim2=-1).sum(-1)   # cos-like agreement
        out[k] = c[int(tr.sum(1).argmax())]
    return out


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
    global_iters: int = 3           # global orientation search of every particle in the first n iterations (0 = off)
    global_res: float = 12.0        # ... at the current resolution, but not finer than this (A)
    sgd_iter: int = 30              # particle SGD rounds after the class stage (0 = off: refine the class model)
    sgd_batch: int = 2000           # ... particles per round (at most a quarter of the particles, at least 500)
    sgd_res: float = 12.0           # ... final resolution (A) of the SGD stage (it starts at 30 A)
    sgd_topk: int = 8               # ... soft poses: each particle at its best k orientations, posterior-weighted
                                    # (noise from the batch's power spectrum). Hard (1) locks in a wrong start: J341
                                    # C3 within 10 deg of cryoSPARC 76 -> 89%, map vs reference FSC 0.5 11.1 -> 9.0 A
                                    # (3 seeds); J1279 C3 equal; +5 s
    relax_from: str = ""            # C1 run of a pseudo-symmetric particle: solve with this symmetry (the user's choice,
                                    # e.g. "C3"), then refine in C1 from it. J341 (C3) C1 vs cryoSPARC poses: 89-90%
                                    # within 10 deg, map vs reference 8.9-9.3 A; direct C1 47-72%, 21-25 A
    relax_iters: int = 3            # ... the first n C1 iterations search every symmetry-related copy of each pose
    check_starts: int = 3           # without SGD: the best n class-stage starts each get a short particle refinement; the one
                                    # the particles fit best is kept (1 = trust the class agreement)
    drop_frac: float = 0.1          # worst-fitting fraction of particles left out of the reconstruction (0 = keep all)
    rebalance: float = 0.0          # > 0: cap crowded viewing directions at this x the median direction (preferred orientation)
    n_models: int = 1               # 3D classes: >1 adds a 3D classification of the particles at the consensus poses
    class3d_res: float = 10.0       # ... with every class filtered to this resolution (A) (or the consensus one if worse)
    class3d_iter: int = 20          # ... for at most this many iterations
    seed: int = 0


def write_star_poses(path: str, src, idx: np.ndarray, A: torch.Tensor, t: torch.Tensor, model=None):
    """RELION particle .star of particles idx (rows of src.parts) with their 3D poses: rlnAngleRot/Tilt/Psi from the
    orientation matrices (RELION convention, checked against Refine3D: image = A @ volume), rlnOrigin = -t (A)."""
    from .star import write_star
    rot, tilt, psi = matrix_to_euler(A.cpu())
    df = src.parts.iloc[np.asarray(idx)].reset_index(drop=True)
    df["rlnAngleRot"], df["rlnAngleTilt"], df["rlnAnglePsi"] = rot.numpy(), tilt.numpy(), psi.numpy()
    tt = t.cpu().numpy()
    df["rlnOriginXAngst"], df["rlnOriginYAngst"] = -tt[:, 0], -tt[:, 1]
    if model is not None:
        df["rlnClassNumber"] = np.asarray(model).astype(int) + 1
    blocks = {k: v for k, v in src.blocks.items() if k != "particles"}
    blocks["particles"] = df
    write_star(path, blocks)


def run(clf, result, outdir: str, prm: AbinitioParams | None = None, classes=None, log=print, src=None) -> dict:
    """Ab initio from a Session result: its class averages (all populated classes, or the boolean mask `classes`)
    and the particles in those classes. Writes maps, figures, poses and summary.json to outdir."""
    import json
    import os
    prm = prm or AbinitioParams()
    os.makedirs(outdir, exist_ok=True)
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    M = max(1, int(prm.n_models))
    b = working_box(clf, n_acc=max(2, 2 * M), n_ft=max(2, M), log=log)
    pops = torch.as_tensor(np.asarray(result.pops), device=dev, dtype=torch.float32)
    keep = pops > 0
    if classes is not None:
        keep &= torch.as_tensor(np.asarray(classes), dtype=torch.bool, device=dev)
    if int(keep.sum()) < 3:
        raise ValueError("ab initio needs at least 3 populated classes")
    cls_ft = torch.fft.rfft2(torch.as_tensor(np.asarray(result.images), device=dev, dtype=torch.float32), norm="forward")
    log(f"[3d] ab initio from {int(keep.sum())} classes ({prm.sym}), working box {b} ({box_a / b:.2f} A/px)")
    M = max(1, int(prm.n_models))
    kmap = torch.full((len(pops),), -1, dtype=torch.long)
    kmap[keep.cpu()] = torch.arange(int(keep.sum()))
    lab = torch.as_tensor(np.asarray(result.labels)).long()
    ok = (lab >= 0) & (kmap[lab.clamp_min(0)] >= 0)
    idx = torch.as_tensor(np.asarray(result.idx))[ok]
    kp = kmap[lab[ok]]
    psi_p, off_p = torch.as_tensor(np.asarray(result.psi))[ok], torch.as_tensor(np.asarray(result.offsets))[ok]
    timing = dict(classes=0.0, sgd=0.0, particles=0.0)

    def consensus(sym: str, quiet: bool = False, sel: torch.Tensor | None = None):
        """Class stage -> (candidate starts judged by the particles) -> particle SGD -> local refinement.
        sel: run on these particles only (indices into idx)."""
        lg = (lambda *a: None) if quiet else log
        idx_, kp_, psi_, off_ = (idx, kp, psi_p, off_p) if sel is None else (idx[sel], kp[sel], psi_p[sel], off_p[sel])
        t0 = time.time()
        vols0, Ak, cc, _, starts = classes_to_volume(cls_ft[keep], pops[keep], box_a, diam, sym=sym,
                                                     n_starts=prm.n_starts, res_start=prm.class_res_start,
                                                     res_end=prm.class_res_end, n_iter=prm.class_iter, seed=prm.seed,
                                                     n_models=1, log=lg,
                                                     keep_top=1 if prm.sgd_iter > 0 else max(1, prm.check_starts))
        torch.cuda.synchronize()
        timing["classes"] += time.time() - t0
        vols_cls = vols0
        if clf._n_loaded < clf.N:   # the class stage needs no particles; it ran while they were still streaming in
            log(f"[3d] waiting for particles to finish loading ({clf._n_loaded}/{clf.N})")
        clf._wait_loaded(clf.N)
        clf.ensure_polarity()       # the 2D result may have been restored from disk, with no 2D run in this session
        t0 = time.time()
        if len(starts) > 1:
            # the class averages can prefer a wrong model (few views, pseudo-symmetry): let the particles decide.
            # Short refinement (global search + 2 local iterations) of each candidate on up to 5000 particles.
            g = torch.Generator().manual_seed(prm.seed)
            sub = torch.randperm(len(idx_), generator=g)[:5000]
            outs = []
            for c_i, (c_score, c_vols, c_Ak) in enumerate(starts):
                o = refine_particles(clf, idx_[sub], particle_poses(c_Ak[0], kp_[sub], psi_[sub]), off_[sub], c_vols,
                                     sym=sym, n_iter=3, res_start=prm.res_start, n_local=prm.n_local, box=b,
                                     global_res=max(prm.global_res, 15.0), global_at=(1,), log=lambda *a: None)
                outs.append((o[1], o[2], o[6][0], o[5][0]))
            # judged like the symmetry relaxation: each particle against the half map it was not in, all
            # candidates over the same band (a candidate's own fit is higher the blurrier it is)
            r_cmp = max(12.0, max(o[3] for o in outs))
            fits = [float(cross_fit(clf, idx_[sub], o[0], o[1], o[2], r_cmp, b, seed=prm.seed).mean()) for o in outs]
            for c_i, (c_score, _, _) in enumerate(starts):
                lg(f"[3d] candidate start {c_i + 1}: class agreement {c_score:.3f}, cross-validated fit at "
                   f"{r_cmp:.1f} A {fits[c_i]:.4f} ({outs[c_i][3]:.1f} A)")
            del outs
            pick = int(np.argmax(fits))
            vols0, Ak = starts[pick][1], starts[pick][2]
            if pick:
                lg(f"[3d] particles prefer candidate {pick + 1} over the best class agreement")
        A0 = particle_poses(Ak[0], kp_, psi_)
        Ak = Ak[0]
        if prm.sgd_iter > 0:
            # the class averages constrain the structure only weakly (a wrong model can match a few dozen
            # projections); stochastic global matching of the particles themselves settles it
            Vs, A0 = particle_sgd(clf, idx_, off_, vols0[0], sym=sym, n_iter=prm.sgd_iter, res_end=prm.sgd_res,
                                  batch=min(prm.sgd_batch, max(500, len(idx_) // 4)), seed=prm.seed,
                                  topk=prm.sgd_topk, log=lg)
            vols0 = [Vs]
            Ak = _class_poses(A0, kp_, psi_, len(Ak))      # class orientations follow the particles (figures)
        torch.cuda.synchronize()
        timing["sgd"] += time.time() - t0
        t0 = time.time()
        common = dict(sym=sym, n_local=prm.n_local, drop_frac=prm.drop_frac, rebalance=prm.rebalance, log=lg)
        n_fine = min(prm.n_iter, 4)
        b_lo = min(b, int(math.ceil(2 * box_a / 5.0 / 2)) * 2)          # Nyquist ~5 A: enough below ~8 A
        if prm.n_iter > n_fine and b_lo < b - 8:
            # early iterations align at 8-15 A: run them on a smaller box (projection and reconstruction cost
            # ~ box^2 per particle), then the last iterations at the full box
            lo = refine_particles(clf, idx_, A0, off_, vols0, n_iter=prm.n_iter - n_fine, res_start=prm.res_start,
                                  box=b_lo, global_res=prm.global_res,
                                  global_at=tuple(range(1, prm.global_iters + 1)), **common)
            V_lo = Volume(b_lo, dev)
            V_lo.set_real(lo[0][0])
            for v in lo[6][0]:
                v.free_acc()
            out = refine_particles(clf, idx_, lo[1], lo[2], V_lo, n_iter=n_fine, res_start=max(lo[5][0], 2.2 * box_a / b),
                                   box=b, **common)
        else:
            out = refine_particles(clf, idx_, A0, off_, vols0, n_iter=prm.n_iter, res_start=prm.res_start, box=b,
                                   global_res=prm.global_res, global_at=tuple(range(1, prm.global_iters + 1)),
                                   **common)
        torch.cuda.synchronize()
        timing["particles"] += time.time() - t0
        return vols_cls, Ak, cc, out

    relaxed_from = prm.relax_from.upper() if prm.relax_from and prm.sym.upper() == "C1" else None
    if relaxed_from == "C1":
        relaxed_from = None
    vols_cls, Ak, cc, (maps, A, t, z, fs, res, vols) = consensus(relaxed_from or prm.sym)
    if relaxed_from:
        # pseudo-symmetric particle (the user's choice): C1 matching has several near-equivalent wrong answers.
        # Solved with the symmetry imposed (well constrained), then refined without it; the first relax_iters
        # iterations search every symmetry-related copy of each pose, so a feature that breaks the symmetry appears.
        log(f"[3d] {relaxed_from} map: {res[0]:.1f} A; refining in C1 (symmetry relaxation)")
        for v in vols[0]:
            v.free_acc()
        Vg = Volume(b, dev)
        Vg.set_real(maps[0])
        lowpass(Vg, max(res[0], 8.0), box_a)
        t0 = time.time()
        maps, A, t, z, fs, res, vols = refine_particles(
            clf, idx, A, t, Vg, sym="C1", n_iter=prm.n_iter, res_start=max(res[0], 8.0), n_local=prm.n_local, box=b,
            drop_frac=prm.drop_frac, rebalance=prm.rebalance, relax_ops=symmetry_ops(relaxed_from),
            relax_iters=prm.relax_iters, log=log)
        torch.cuda.synchronize()
        timing["particles"] += time.time() - t0
        Ak = _class_poses(A, kp, psi_p, len(Ak))
    used = refine_particles.last_weights > 0
    consensus_map = maps[0]
    cov = view_coverage(A, relaxed_from or prm.sym)
    cov["warning"] = (cov["occupied"] < 0.3 or cov["top10_share"] > 0.6)
    log(f"[3d] views: {cov['occupied'] * 100:.0f}% of directions occupied, busiest 10% hold "
        f"{cov['top10_share'] * 100:.0f}% of particles" + (" -- WARNING: narrow range of views, the map is smeared "
        "along the missing directions; include classes with other views if there are any" if cov["warning"] else ""))
    unmasked_res = resolution_from_fsc(fsc(vols[0][0], vols[0][1]), box_a)   # comparable to RELION's "without masking"
    zk = torch.zeros(len(Ak), dtype=torch.long, device=dev)
    t0 = time.time()
    if M > 1:   # heterogeneity: 3D classification at the consensus poses, all classes filtered alike
        # a (pseudo-)symmetric particle sits in one of its symmetry-related poses at random, so a difference that
        # breaks the symmetry lands in a different place per particle: classify over the group's poses (the
        # relaxed consensus is in the symmetry frame, so the group's standard operators apply)
        xops = symmetry_ops(relaxed_from).to(dev) if relaxed_from else None
        if relaxed_from:
            log(f"[3d] 3D classification relaxed over the {len(xops)} {relaxed_from}-related poses")
        cres = max(res[0], prm.class3d_res)
        for v in vols[0]:
            v.free_acc()
        maps, z, post, res, vols, A = classify3d(clf, idx, A, t, M, cres, sym=prm.sym, n_iter=prm.class3d_iter,
                                                 seed=prm.seed, expand_ops=xops, box=b, log=log)
        cmask = sphere_mask(b, diam / 2 / (box_a / b), 3.0, dev)
        fs = torch.stack([masked_fsc(vols[m][0], vols[m][1], cmask) for m in range(M)])
        vols_cls = vols_cls * M
    torch.cuda.synchronize()
    timing["particles"] += time.time() - t0
    t1, t2 = timing["classes"], timing["sgd"] + timing["particles"]
    apix = box_a / b
    pops_m = torch.bincount(z, minlength=M).tolist()
    order = sorted(range(M), key=lambda m: -pops_m[m])        # model 1 = most particles
    solvent = []
    for r, m in enumerate(order):
        sfx = "" if M == 1 else f"_{r + 1}"
        smask = solvent_mask(maps[m], apix)
        solvent.append(smask)
        _save_mrc(os.path.join(outdir, f"map{sfx}.mrc"), maps[m] * smask, apix)          # automatic solvent mask
        _save_mrc(os.path.join(outdir, f"map_nomask{sfx}.mrc"), maps[m], apix)
        _save_mrc(os.path.join(outdir, f"mask{sfx}.mrc"), smask, apix)
        for h in (0, 1):
            _save_mrc(os.path.join(outdir, f"half{h + 1}_unfil{sfx}.mrc"), vols[m][h].real().nan_to_num(0), apix)
        vc = vols_cls[min(m, len(vols_cls) - 1)]
        _save_mrc(os.path.join(outdir, f"from_classes{sfx}.mrc"), vc.real(), box_a / vc.b)
    if M > 1:   # map.mrc / from_classes.mrc = the largest model, so single-model consumers keep working
        import shutil
        shutil.copy(os.path.join(outdir, "map_1.mrc"), os.path.join(outdir, "map.mrc"))
        shutil.copy(os.path.join(outdir, "from_classes_1.mrc"), os.path.join(outdir, "from_classes.mrc"))
    rank = torch.empty(M, dtype=torch.long)
    rank[torch.tensor(order)] = torch.arange(M)
    rot, tilt, psi = matrix_to_euler(A.cpu())
    np.savez(os.path.join(outdir, "poses.npz"), idx=idx.numpy(), rot=rot.numpy(), tilt=tilt.numpy(), psi=psi.numpy(),
             shift_a=t.cpu().numpy(), model=rank[z.cpu()].numpy(), used=used.numpy(),
             fit=refine_particles.last_score.numpy())
    fsc_unmasked = fsc(vols[order[0]][0], vols[order[0]][1])
    fsc_solvent = corrected_masked_fsc(vols[order[0]][0].real(), vols[order[0]][1].real(), solvent[0])
    tag = f"{len(idx):,} particles · {prm.sym}" + (f" (relaxed from {relaxed_from})" if relaxed_from else "")
    if M > 1:
        curves = [(f"3D class {r + 1}", fs[m].cpu().numpy()) for r, m in enumerate(order[:3])]
    else:
        curves = [("soft sphere mask", fs[0].cpu().numpy()), ("no mask", fsc_unmasked.cpu().numpy()),
                  ("solvent mask", fsc_solvent.cpu().numpy())]
    if src is not None and getattr(src, "parts", None) is not None:
        u = used.numpy()
        mdl = rank[z.cpu()].numpy()
        write_star_poses(os.path.join(outdir, "particles.star"), src, idx.numpy()[u], A[torch.as_tensor(u, device=A.device)],
                         t[torch.as_tensor(u, device=t.device)], mdl[u] if M > 1 else None)
        if (~u).any():
            write_star_poses(os.path.join(outdir, "particles_dropped.star"), src, idx.numpy()[~u],
                             A[torch.as_tensor(~u, device=A.device)], t[torch.as_tensor(~u, device=t.device)])
        log(f"[3d] particles.star: {int(u.sum())} particles with 3D poses (RELION convention, this map's frame)")
    from . import diagnostics
    vstats = diagnostics.angular_stats(A, relaxed_from or prm.sym)
    diagnostics.write_views(os.path.join(outdir, "views.json"), rot.numpy(), tilt.numpy(), vstats)
    fsc_data = dict(corrected=fsc_solvent.cpu().numpy(), unmasked=fsc_unmasked.cpu().numpy(),
                    sphere_masked=fs[order[0]].cpu().numpy())
    for r_, m in enumerate(order if M > 1 else []):
        fsc_data[f"class{r_ + 1}_sphere_masked"] = fs[m].cpu().numpy()
    diagnostics.write_fsc(os.path.join(outdir, "fsc"), fsc_data, box_a,
                          resolution_a=round(resolution_from_fsc(fsc_solvent, box_a), 3), title=tag)
    reproj_cc = _figures(outdir, clf, result, keep, Ak, [maps[m] for m in order], curves, rot, tilt, box_a, b, tag,
                         vstats)
    if M > 1:
        _save_mrc(os.path.join(outdir, "consensus.mrc"), consensus_map, apix)
    from dataclasses import asdict
    summary = dict(params=asdict(prm), particles=int(len(idx)), classes=int(keep.sum()), seconds_classes=round(t1, 1),
                   seconds_particles=round(t2, 1), resolution_a=round(res[order[0]], 2), apix=round(apix, 3), box=b,
                   resolution_unmasked_a=round(unmasked_res, 2), fsc_mask="soft sphere of the particle diameter",
                   resolution_solvent_mask_a=round(resolution_from_fsc(fsc_solvent, box_a), 2),
                   particles_used=int(used.sum()), particles_dropped=int((~used).sum()),
                   relaxed_from=relaxed_from, seconds_sgd=round(timing["sgd"], 1), view_coverage=cov,
                   scf=vstats["scf"], warnings=vstats["warnings"],
                   fsc=[round(float(x), 4) for x in fs[order[0]].cpu()], box_a=round(box_a, 2),
                   fsc_unmasked=[round(float(x), 4) for x in fsc_unmasked.cpu()],
                   fsc_solvent=[round(float(x), 4) for x in fsc_solvent.cpu()],
                   reprojection_cc=[round(x, 3) for x in reproj_cc], seconds_total=round(t1 + t2, 1),
                   class_agreement=[round(float(x), 3) for x in cc],
                   models=[dict(model=r + 1, particles=pops_m[m], classes=int((zk == m).sum()),
                                resolution_a=round(res[m], 2) if res[m] == res[m] else None)
                           for r, m in enumerate(order)])
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    models = " / ".join(f"{x['resolution_a'] or 'merged'} A ({x['particles']})" for x in summary["models"])
    log(f"[3d] wrote {outdir}: {models}, {t1:.1f}s classes + {timing['sgd']:.1f}s SGD + "
        f"{timing['particles']:.1f}s particles")
    return summary


def _figures(outdir, clf, result, keep, Ak, maps, fsc_curves, rot, tilt, box_a, b, title, view_stats=None):
    """Reprojections of the (first) map at the class orientations, then the PNG figures (plots3d)."""
    from . import plots3d
    dev = clf.dev
    V = Volume(b, dev)
    V.set_real(maps[0])
    P = torch.fft.irfft2(V.project(Ak.to(dev)), s=(b, b), norm="forward").cpu().numpy()
    del V
    kk = keep.cpu().numpy()
    imgs = np.asarray(result.images)[kk]
    b2 = imgs.shape[-1]
    if b2 != b:   # class images are at the 2D working box: bring the reprojections to it
        Pt = torch.fft.rfft2(torch.as_tensor(P))
        from .engine import ft_resize
        P = torch.fft.irfft2(ft_resize(Pt, b2), s=(b2, b2)).numpy()
    labels = [f"#{k + 1}" for k in np.nonzero(kk)[0]]
    return plots3d.write_all(outdir, class_imgs=imgs, reproj=P, pops=np.asarray(result.pops)[kk], class_labels=labels,
                             maps=[m.cpu().numpy() for m in maps],
                             map_labels=["map"] if len(maps) == 1 else [f"3D class {i + 1}" for i in range(len(maps))],
                             fsc_curves=fsc_curves, box_a=box_a, apix=box_a / b, rot=rot.numpy(), tilt=tilt.numpy(),
                             title=title, view_stats=view_stats)


# --------------------------------------------------------------------------- volume rotation (map comparison)
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

