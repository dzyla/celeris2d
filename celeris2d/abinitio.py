"""De novo 3D initial model from a Celeris classification (PyTorch).

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

from . import gpu
from .clock import since
from .engine import TWO_PI, ctf_2d, freqs, ft_resize, gpu_free, rot_grid, upsample2, release_cache

# A job may set LIVE = fn(stage, it, n_iter, maps, res, apix, counts=None): called with the current maps (real
# space, one per model) after every class-stage start, SGD round, refinement iteration and 3D classification iteration -
# the dashboard's live view (live.live_maps). counts: particles per model (None: one model, or unknown). maps is None
# when LIVE.due(stage, it, n_iter) says no snapshot is wanted (history only: the maps are not made). It must be quick;
# failures are ignored.
LIVE = None


def _live(stage, it, n_iter, vols, res, apix, counts=None):
    """vols: per model a Volume, a pair of half Volumes (averaged) or a real map (tensor)."""
    if LIVE is None:
        return

    def real(v):
        if torch.is_tensor(v):
            return v
        if isinstance(v, (list, tuple)):        # halves (a 3D classification's two references may be one object)
            return v[0].real() if v[0] is v[1] else (v[0].real() + v[1].real()).mul_(0.5)
        return v.real()
    try:
        due = getattr(LIVE, "due", None)
        maps = [real(v) for v in vols] if due is None or due(stage, it, n_iter) else None
        if counts is None:
            LIVE(stage, it, n_iter, maps, res, apix)
        else:
            LIVE(stage, it, n_iter, maps, res, apix, counts=[int(c) for c in counts])
    except Exception:                                # noqa: BLE001 - a live view never stops a job
        pass


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


def _axis_rot(axis, deg: float) -> torch.Tensor:
    """Rotation by deg about a unit axis (Rodrigues)."""
    k = torch.tensor(axis, dtype=torch.float64)
    k = k / k.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float64)
    a = math.radians(deg)
    return torch.eye(3, dtype=torch.float64) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def _close_group(gens: list, order: int) -> torch.Tensor:
    """Every product of the generators (the group they generate), identity first; checked against its order."""
    ops = [torch.eye(3, dtype=torch.float64)]
    i = 0
    while i < len(ops):
        for g in gens:
            m = g @ ops[i]
            if not any(torch.allclose(m, o, atol=1e-6) for o in ops):
                ops.append(m)
        i += 1
        if len(ops) > order:
            break
    if len(ops) != order:
        raise RuntimeError(f"point group closure gave {len(ops)} operators, expected {order}")
    return torch.stack(ops).float()


# T / O / I: generators in RELION's frames (its symmetry library): T a 3-fold on z and a 2-fold in the y-z plane,
# O a 4-fold on z and a 3-fold on (1, 1, 1), I (= I2) a 2-fold on z and a 5-fold in the x-z plane
_POLY = {"T": ([((0, 0, 1), 120), ((0, 0.816496581, 0.577350269), 180)], 12),
         "O": ([((0, 0, 1), 90), ((1, 1, 1), 120)], 24),
         "I": ([((0, 0, 1), 180), ((0.525731112, 0, 0.850650808), 72)], 60)}
_POLY["I2"] = _POLY["I"]
_POLY["I1"] = ([((1, 0, 0), 180), ((0.850650808, 0, 0.525731112), 72)], 60)        # Crowther: 2-fold on x


def symmetry_ops(sym: str) -> torch.Tensor:
    """Rotation matrices (S, 3, 3) of point group Cn / Dn / T / O / I (RELION axis conventions: Cn about z, the D
    2-fold about x; T, O, I (I2) and I1 as in _POLY)."""
    sym = sym.upper().strip()
    if sym in _POLY:
        gens, order = _POLY[sym]
        if sym not in _OPS_CACHE:
            _OPS_CACHE[sym] = _close_group([_axis_rot(a, d) for a, d in gens], order)
        return _OPS_CACHE[sym].clone()
    kind, n = sym[0], int(sym[1:] or 1) if sym[1:].isdigit() or not sym[1:] else -1
    if kind not in "CD" or n < 1:
        raise ValueError(f"symmetry {sym!r} not supported (Cn, Dn, T, O, I)")
    ops = [rotz(torch.tensor([360.0 * i / n]))[0] for i in range(n)]
    if kind == "D":
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # 2-fold about x
        ops += [o @ flip for o in ops]
    return torch.stack(ops)


_OPS_CACHE: dict = {}


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

    # ---- reconstruction accumulators: the x >= 0 half of the grid only (RELION's layout). An image point at q with
    # q_x < 0 goes in as its Friedel mate (-q, conj); the x = 0 plane then gets the mates of its own points once
    # (_finish), and solve() writes the full grid with the x < 0 half as the conjugate mirror. Half the memory of
    # full-grid accumulators and half the splatting (each point once, not with its mate).
    def _acc(self):
        if self.num is None:
            G, H = self.G, self.G // 2
            self.num = torch.zeros((G * G * H, 2), device=self.dev)
            self.den = torch.zeros(G * G * H, device=self.dev)
            self._open = False                  # points inserted since the x = 0 plane was completed

    def insert(self, x: torch.Tensor, w: torch.Tensor, A: torch.Tensor, kmax: float, ops: torch.Tensor | None = None):
        """Accumulate spectra x (n, b, bh) (already CTF-multiplied: x*ctf) with weights w (n, b, bh) (e.g. ctf^2)
        at orientations A, trilinear (with their Friedel mates, implicitly); frequencies beyond kmax (cycles/box) are
        skipped. ops: symmetry operators S (each image is also inserted at A @ S). Chunked to ~4M points per pass."""
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
        self._open = True

    def _insert(self, x, w, A, keep):
        G, H = self.G, self.G // 2
        q = self._coords(A, keep).reshape(-1, 3)            # (n P, 3)
        v = (x * self.checker)[:, keep].reshape(-1)
        ww = w[:, keep].float().reshape(-1)
        neg = q[:, 0] < 0                                   # into the x >= 0 half as the Friedel mate
        q = torch.where(neg[:, None], -q, q)
        vr = torch.view_as_real(torch.where(neg, v.conj(), v).resolve_conj())
        p = q + torch.tensor([0.0, G // 2, G // 2], device=q.device)
        p0 = torch.floor(p)
        f = p - p0
        p0 = p0.long()
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix, iy, iz = p0[:, 0] + dx, p0[:, 1] + dy, p0[:, 2] + dz
                    wt = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                          * (f[:, 2] if dz else 1 - f[:, 2]))
                    ok = (ix < H) & (iy >= 0) & (iy < G) & (iz >= 0) & (iz < G)
                    idx = ((iz * G + iy) * H + ix)[ok]
                    wt = wt[ok]
                    self.num.index_add_(0, idx, vr[ok] * wt[:, None])
                    self.den.index_add_(0, idx, ww[ok] * wt)

    @staticmethod
    def _mirror_yz(P: torch.Tensor) -> torch.Tensor:
        """P[(G - z) % G, (G - y) % G] for the leading (z, y) axes: the frequency -k of the centred index."""
        return torch.roll(torch.flip(P, (0, 1)), shifts=(1, 1), dims=(0, 1))

    def _finish(self):
        """Complete the x = 0 plane: a point there also stands for its mate at (0, -y, -z) (conjugate)."""
        if self.num is None or not self._open:
            return
        G, H = self.G, self.G // 2
        num, den = self.num.view(G, G, H, 2), self.den.view(G, G, H)
        n0, d0 = num[:, :, 0].clone(), den[:, :, 0].clone()
        m0 = self._mirror_yz(n0)
        num[:, :, 0, 0] = n0[..., 0] + m0[..., 0]
        num[:, :, 0, 1] = n0[..., 1] - m0[..., 1]
        den[:, :, 0] = d0 + self._mirror_yz(d0)
        self._open = False

    def symmetrize(self, ops: torch.Tensor):
        """Symmetrise the accumulators once (RELION's BackProjector::symmetrise): num and den at grid point p become
        the sum over operators S of their trilinear samples at S p (a point with negative x read as the conjugate of
        its mate) - the same as inserting every image at A @ S (whose frequencies land at S^T A^T k), for one
        resampling of the grid instead of |S| insertions per image."""
        if self.num is None or ops is None or len(ops) < 2:
            return
        self._finish()
        G, H, dev = self.G, self.G // 2, self.dev
        acc = torch.cat([self.num.view(G, G, H, 2).permute(3, 0, 1, 2), self.den.view(1, G, G, H)])  # (3, G, G, H)
        self.num = self.den = None
        num = torch.empty((G, G, H, 2), device=dev)
        den = torch.empty((G, G, H), device=dev)
        ops = ops.to(dev).float()
        r = torch.arange(G, device=dev, dtype=torch.float32) - G // 2
        rx = torch.arange(H, device=dev, dtype=torch.float32)
        cz = max(1, int(8e6 // (G * H)))
        sc = torch.tensor([2.0 / H, 2.0 / G, 2.0 / G], device=dev)
        off = torch.tensor([0.0, G // 2, G // 2], device=dev)
        for z0 in range(0, G, cz):
            zz = r[z0:z0 + cz]
            q = torch.stack(torch.broadcast_tensors(rx[None, None, :], r[None, :, None], zz[:, None, None]), -1)
            out = acc[:, z0:z0 + cz].clone()
            for S in ops:
                if torch.allclose(S, torch.eye(3, device=dev)):
                    continue
                qs = q @ S.T
                neg = qs[..., 0] < 0
                qs = torch.where(neg[..., None], -qs, qs)
                g = ((qs + off + 0.5) * sc - 1)[None]
                smp = F.grid_sample(acc[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
                smp[1] = torch.where(neg, -smp[1], smp[1])                   # the mate: conjugate
                out += smp
            num[z0:z0 + cz] = out[:2].permute(1, 2, 3, 0)
            den[z0:z0 + cz] = out[2]
        del acc
        self.num, self.den = num.view(-1, 2), den.view(-1)
        self._open = False

    def clear(self):
        self._acc()
        self.num.zero_()
        self.den.zero_()
        self._open = False

    def add_acc(self, other: "Volume"):
        """Add another volume's accumulators (same grid, any device: another GPU's share of a back-projection)."""
        if other.num is None:
            return
        self._acc()
        self.num += other.num.to(self.dev)
        self.den += other.den.to(self.dev)
        self._open = self._open or other._open

    def on(self, dev) -> "Volume":
        """This spectrum on device dev (itself when it is there already): projections on another GPU."""
        if torch.device(dev) == torch.device(self.dev):
            return self
        v = Volume(self.b, dev, pad=self.pad, ft=self.ft.to(dev), pre=self.pre, grid_correct=self.grid_correct)
        v.mag = None if self.mag is None else self.mag.to(dev)
        return v

    def free_acc(self):
        self.num = self.den = None

    def acc_shells(self) -> torch.Tensor:
        """Shell index (cycles per unpadded box) of the accumulators' half grid (G, G, G//2), x from 0."""
        return self.shell_index()[:, :, self.G // 2:]

    def solve(self, reg: torch.Tensor | float):
        """ft = num / (den + reg) on the half grid, the x < 0 half its conjugate mirror; reg: scalar or per-shell (in
        cycles/box b) tensor. Written into the new spectrum in slabs of x, the old one released first."""
        self._finish()
        G, H = self.G, self.G // 2
        den = self.den.view(G, G, H)
        num = self.num.view(G, G, H, 2)
        self.ft = None
        out = torch.zeros((2, G, G, G), device=self.dev)
        sh = self.acc_shells() if torch.is_tensor(reg) else None
        cx = max(1, int(32e6 // (G * G)))
        for x0 in range(0, H, cx):
            x1 = min(H, x0 + cx)
            d = den[:, :, x0:x1] + (reg[sh[:, :, x0:x1].clamp(max=len(reg) - 1)] if sh is not None else reg)
            v = (num[:, :, x0:x1] / d[..., None]).permute(3, 0, 1, 2)            # (2, G, G, cx): x = x0 .. x1-1
            out[:, :, :, H + x0:H + x1] = v
            # x < 0: ft(-x, -y, -z) = conj ft(x, y, z), for x >= 1 (index H - x)
            lo = max(x0, 1)
            if lo < x1:
                m = self._mirror_yz(v[:, :, :, lo - x0:].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
                m[1] = -m[1]
                out[:, :, :, H - x1 + 1:H - lo + 1] = m.flip(3)
            del d, v
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


def corrected_masked_fsc(h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor, seed: int = 0, parts: bool = False,
                         k_r: int | None = None):
    """Masked FSC of two real half maps corrected for the correlation a (tight) mask creates by itself (RELION
    PostProcess): from the shell k_r where the unmasked FSC drops below 0.8, the half maps' phases are randomised
    (whole shells, as the FSC counts them), the maps masked again, and FSC_true = (FSC_masked - FSC_random) /
    (1 - FSC_random) from shell k_r + 3 on; the shells just past k_r keep the masked FSC (as RELION): the mask spreads
    correlation from the intact shells into them, FSC_random is close to FSC_masked there and the formula dips
    (beta-gal: 0.97 -> 0.72 for one shell when it was applied from k_r, with half of shell k_r randomised).
    parts: also return (masked, randomised, unmasked FSC, k_r). k_r: randomise from this shell instead (e.g. one shell
    for two maps compared in one mask: the unmasked FSC of a whole map says little about a small masked region)."""
    n = h1.shape[0]
    f_un = fsc_maps(h1, h2)
    f_m = fsc_maps(h1 * mask, h2 * mask)
    if k_r is None:
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


def _regularise_solve(vol: Volume, frac: float = 0.05, fsc: torch.Tensor | None = None, tau2_fudge: float = 1.0):
    """num / (den + frac * mean den of the shell): damps sparsely sampled Fourier voxels. fsc (the half maps' FSC per
    shell, cycles / box b): the Wiener / MAP regulariser instead (RELION's tau2 term), mean den x (1 - FSC) / (T FSC) per
    shell - the half map's noise-to-signal ratio from SSNR = FSC / (1 - FSC) - so a voxel weighs its data by how well
    it is sampled against how much signal the shell holds; never less than frac x mean den."""
    vol._finish()
    sh = vol.acc_shells().reshape(-1)
    n = int(sh.max()) + 1
    # the half grid: a voxel with x > 0 also stands for its mirror (counted twice, as on the full grid)
    H = vol.G // 2
    wx = torch.full((H,), 2.0, device=vol.dev)
    wx[0] = 1.0
    wv = wx.repeat(vol.G * vol.G)
    m = torch.bincount(sh, weights=vol.den * wv, minlength=n) / torch.bincount(sh, weights=wv, minlength=n).clamp_min(1)
    lam = frac * m
    if fsc is not None:
        f = torch.zeros(n, device=vol.dev)
        k = min(n, len(fsc))
        f[:k] = torch.as_tensor(fsc[:k], device=vol.dev).float().nan_to_num(0)
        f = f.clamp(1e-3, 0.999)
        lam = torch.maximum(lam, m * (1 - f) / (tau2_fudge * f))
    vol.solve(lam + 1e-6 * m.max())


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
            + (f", particles per model {sizes}" if M > 1 else "") + f" ({since(t0):.1f}s)")
        if M > 1 and min(sizes) == 0:
            score -= 1.0                                      # a start that left a model empty is not a split
        Ak = torch.empty((M, K, 3, 3), device=dev)           # each class's best orientation in every model
        for m in range(M):
            _, r_i, d_i = tis[m]
            Ak[m] = rotz(-angles[r_i[:, 0].cpu()]).to(dev) @ Ad[d_i[:, 0]]   # class = P(rotz(-a) Ad)
        starts.append((score, vols, Ak))
        if best is None or score > best[0]:
            best = (score, vols, Ak, cc_k.clone(), zk.clone())
        _live("classes", start + 1, n_starts, vols, [res_end] * M, box_a / b1, counts=sizes if M > 1 else None)
    log(f"[3d] best start: class agreement {best[0]:.3f} (per class {' '.join(f'{x:.2f}' for x in best[3].tolist())})"
        + (f"; classes per model {torch.bincount(best[4], minlength=M).tolist()}" if M > 1 else ""))
    starts.sort(key=lambda x: -x[0])
    return best[1], best[2], best[3], best[4], starts[:keep_top]


# --------------------------------------------------------------------------- stage 2: particles
def auto_pad(b: int, n_vol: float, dev) -> int:
    """Padding of 3D spectra: 2 if n_vol reconstructions (~56 (2b)^3 bytes each with half-grid accumulators,
    references and batch arrays: measured 7.4 GB peak for beta-gal at box 256) fit in 90 % of the free memory
    (gpu_free already keeps a reserve), else 1 (gridding-corrected; box 512 in 16 GB). Pad 1 costs accuracy away from
    the centre: beta-gal box 256 unmasked FSC 3.70 -> 3.93 A (masked unchanged)."""
    return 2 if n_vol * 56 * (2 * b) ** 3 < 0.9 * _free(dev) else 1


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


_PARALLEL = [0]      # shares running on several GPUs at once (on_devices)


def _free(dev) -> float:
    """Free memory on dev (bytes) after releasing PyTorch's cached blocks. While on_devices runs shares concurrently
    nothing is released (another thread's kernels may still use blocks its tensors just returned to the cache: unmapped,
    an illegal memory access); the cached blocks count as free instead."""
    if not gpu.is_gpu(dev):
        return 8e9
    if _PARALLEL[0]:
        return float(gpu_free(dev) + gpu.memory_reserved(dev) - gpu.memory_allocated(dev))
    release_cache()
    return float(gpu_free(dev))                 # CELERIS2D_VRAM_GB caps it


def working_box(clf, n_acc: int, n_ft: int, log=print, max_box: int = 0) -> int:
    """Largest even box <= the loaded working box (and <= max_box, if set) whose 3D volumes fit on the GPU. Per
    b^3: a padded volume with accumulators 160 bytes (n_acc of them), a spectrum-only volume 64 bytes (n_ft), FFT
    transients ~320; plus 1.5 GB for particle batches. Data that would not fit is never pushed to (WSL: swapped)
    host memory."""
    free = _free(clf.dev)
    per = n_acc * 160 + n_ft * 64 + 320
    b = clf.bmax
    if max_box and max_box < b:                      # the user's cap (memory, speed): Nyquist coarser accordingly
        b = max(32, int(max_box) // 2 * 2)
        log(f"[3d] maps at most {b} px (max_box): Nyquist {2 * clf.box_a / b:.1f} A instead of "
            f"{2 * clf.box_a / clf.bmax:.1f} A")
    bmax = b
    while b > 32 and per * b ** 3 + 1.5e9 > 0.85 * free:
        b -= 2
    if b < bmax:
        log(f"[3d] GPU memory: {free / 1e9:.1f} GB free -> 3D working box {b} px instead of {bmax} "
            f"(Nyquist {2 * clf.box_a / b:.1f} A instead of {2 * clf.box_a / bmax:.1f} A)")
    return b


def _devices(clf) -> list:
    return list(getattr(clf, "devs", None) or [clf.dev])


def on_devices(clf, n: int, work, min_share: int = 2000) -> list:
    """Run work(lo, hi, dev) over contiguous shares of range(n), one thread per GPU of clf (clf.devs; one share when
    there is one GPU or few items). Returns the results in share order. GPU work from several threads runs
    concurrently; each share uses its own device's memory."""
    devs = _devices(clf)
    k = max(1, min(len(devs), n // max(1, min_share)))
    if k == 1:
        return [work(0, n, devs[0])]
    cuts = [round(n * i / k) for i in range(k + 1)]
    out, errs = [None] * k, []

    def run(i):
        try:
            with gpu.use(devs[i]):
                out[i] = work(cuts[i], cuts[i + 1], devs[i])
                gpu.synchronize(devs[i])
        except BaseException as e:      # noqa: BLE001 - re-raised in the caller's thread
            errs.append(e)

    import threading
    th = [threading.Thread(target=run, args=(i,)) for i in range(k)]
    _PARALLEL[0] += 1
    try:
        for x in th:
            x.start()
        for x in th:
            x.join()
    finally:
        _PARALLEL[0] -= 1
    if errs:
        raise errs[0]
    return out


def _cat_to(parts: list, dev):
    """Concatenate per-share results (tensors or tuples of tensors, first axis = items) on dev."""
    if isinstance(parts[0], torch.Tensor):
        return torch.cat([p.to(dev) for p in parts])
    return tuple(torch.cat([p[i].to(dev) for p in parts]) for i in range(len(parts[0])))


def global_search_multi(clf, idx, A, t, refs, half, res_a, step_deg, sigma2=None, **kw):
    """global_search with the particles shared among clf's GPUs (refs copied to each); results on clf.dev."""
    def work(lo, hi, d):
        return global_search(clf, idx[lo:hi], A[lo:hi], t[lo:hi], [r.on(d) for r in refs], half[lo:hi], res_a, step_deg,
                             None if sigma2 is None else sigma2.to(d), dev=d, **kw)
    return _cat_to(on_devices(clf, len(idx), work), clf.dev)


def score_poses_multi(clf, idx, A, t, refs, half, res_a, sigma2=None, **kw):
    """score_poses with the particles shared among clf's GPUs; results on clf.dev."""
    def work(lo, hi, d):
        return score_poses(clf, idx[lo:hi], A[lo:hi], t[lo:hi], [r.on(d) for r in refs], half[lo:hi], res_a,
                           sigma2=None if sigma2 is None else sigma2.to(d), dev=d, **kw)
    return _cat_to(on_devices(clf, len(idx), work), clf.dev)


def global_search(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, refs, half: torch.Tensor, res_a: float,
                  step_deg: float, sigma2: torch.Tensor | None = None, box: int | None = None, log=print,
                  topk: int = 1, shifts: torch.Tensor | None = None, scaled: bool = False, dev=None):
    """Exhaustive orientation search (whole sphere x all in-plane angles, shift kept) of particles against the
    half-set references refs = [ref_half0, ref_half1], as two GEMMs per chunk of orientations:
    ll = Re<ctf conj(x) e_t, P> / sigma2 - 0.5 <ctf^2 / sigma2, |P|^2>. Returns (best A (n, 3, 3), best ll (n,),
    ll at the current pose (n,)); with topk > 1 the best topk orientations (n, topk, 3, 3) and their ll (n, topk).
    shifts: (S, 2) offsets (A) added to each particle's shift, searched with every orientation (projections computed
    once, one GEMM per offset): a fourth value, the best shifts (n, 2) - with topk > 1 the shift of each of the topk
    hypotheses (n, topk, 2).
    scaled: each hypothesis with its best contrast scale a >= 0 (cryoSPARC's per-particle scale): max_a a c - a^2 p / 2 =
    c^2 / 2p for c = Re<ctf conj(x) e_t, P> / sigma2 > 0, p = <ctf^2 / sigma2, |P|^2>; without it a faint particle
    scores views with weak projections higher (the -p / 2 term), whatever its true view. dev: the GPU to work on (refs
    must be there; default clf.dev)."""
    dev = dev or clf.dev
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
    S = 1 if shifts is None else len(shifts)
    best_s = torch.zeros(n, dtype=torch.long, device=dev)
    if shifts is not None:
        sh = shifts.to(dev).float()
        Es = torch.exp(-1j * ph * (sh[:, 0, None] * fx[None, :].expand_as(kr)[band][None]
                                   + sh[:, 1, None] * fy[:, None].expand_as(kr)[band][None]))        # (S, F)
    if topk > 1:
        top_ll = torch.full((n, topk), -float("inf"), device=dev)
        top_h = torch.zeros((n, topk), dtype=torch.long, device=dev)
        top_s = torch.zeros((n, topk), dtype=torch.long, device=dev)
    cur_ll = torch.zeros(n, device=dev)
    A, t = A.to(dev), t.to(dev)
    Fb = int(band.sum())
    free = _free(dev)
    # particles per chunk: the band spectra, and before them full half-plane grids (spectra, shift phases, CTF and
    # their products: ~40 B per point); sized from the band alone, J20 (198k particles, box 108) asked for 4.4 GB more
    # than a 24 GB card had
    npc = int(max(256, min(n, 0.1 * free / (Fb * 12 + b * (b // 2 + 1) * 40))))
    npc = max(64, npc // S) if S > 1 else npc       # each offset's spectra and scores
    hc = int(max(256, min(H, 0.15 * free / (npc * S * 20 + Fb * 24))))   # orientations per chunk (scores + projections)
    for h in (0, 1):
        sel_h = torch.nonzero(half.to(dev) == h).squeeze(1)
        for p0 in range(0, len(sel_h), npc):
            sel = sel_h[p0:p0 + npc]
            j = idx[sel.cpu()]
            x = clf._fetch_align(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[sel]
            e_t = torch.exp(-1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
            X = (x.conj() * e_t * ctf)[:, band] * isig
            C2 = (ctf * ctf)[:, band] * isig
            del x, e_t, ctf
            P = refs[h].project(A[sel], band)
            c_, p_ = (X * P).real.sum(1), (C2 * P.abs() ** 2).sum(1)
            cur_ll[sel] = c_.clamp_min(0) ** 2 / (2 * p_.clamp_min(1e-30)) if scaled else c_ - 0.5 * p_
            # the spectra moved by each offset, stacked: one GEMM per chunk for all offsets
            Xs = X if shifts is None else (X[None] * Es[:, None]).reshape(S * len(sel), -1)
            Xr = [Xs.real.contiguous()]
            Xi = [Xs.imag.contiguous()]
            del X, Xs
            for c in range(0, H, hc):
                P = refs[h].project(Ah[c:c + hc], band)                           # (Hc, F)
                Pr, Pi, pw = P.real.T.contiguous(), P.imag.T.contiguous(), 0.5 * (C2 @ (P.abs() ** 2).T)
                if S > 1:
                    # each orientation with its best offset, then one selection per chunk: a top-k over the
                    # (orientation, offset) pairs per offset took 2/3 of a rescue (J20, 49 offsets: 334 s of 533)
                    # and filled the list with one orientation at neighbouring offsets
                    c_ = (Xr[0] @ Pr - Xi[0] @ Pi).view(S, len(sel), -1)                 # (S, m, Hc)
                    ll = c_.clamp_min(0) ** 2 / (4 * pw[None].clamp_min(1e-30)) if scaled else c_ - pw[None]
                    del c_
                    ll, sb = ll.max(0)                                                   # (m, Hc)
                    if topk > 1:
                        v, k = torch.topk(torch.cat([top_ll[sel], ll], 1), topk, dim=1)
                        hh = torch.cat([top_h[sel], torch.arange(c, c + ll.shape[1], device=dev).expand(len(sel), -1)], 1)
                        ss = torch.cat([top_s[sel], sb], 1)
                        top_ll[sel], top_h[sel], top_s[sel] = v, hh.gather(1, k), ss.gather(1, k)
                        continue
                    v, k = ll.max(1)
                    upd = v > best_ll[sel]
                    best_ll[sel[upd]] = v[upd]
                    best_h[sel[upd]] = k[upd] + c
                    best_s[sel[upd]] = sb.gather(1, k[:, None])[:, 0][upd]
                    continue
                ll = Xr[0] @ Pr - Xi[0] @ Pi                                       # (n_chunk, Hc)
                ll = ll.clamp_min(0) ** 2 / (4 * pw.clamp_min(1e-30)) if scaled else ll - pw
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
        if shifts is not None:
            return Ah[top_h], top_ll, cur_ll, t[:, None] + sh[top_s]
        return Ah[top_h], top_ll, cur_ll
    if shifts is not None:
        return Ah[best_h], best_ll, cur_ll, t + sh[best_s]
    return Ah[best_h], best_ll, cur_ll


def score_poses(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, refs, half: torch.Tensor, res_a: float,
                sigma2: torch.Tensor | None = None, box: int | None = None, shifts: torch.Tensor | None = None,
                dev=None):
    """Log-likelihood of given candidate orientations per particle (same objective as global_search), each with the
    best of a grid of shift offsets. A: (n, J, 3, 3); t: (n, 2) shifts (A), or (n, J, 2) one per candidate; shifts: (S, 2)
    offsets (A) added to t (None: t only). Returns ll (n, J) and the best shift (n, J, 2) (A). dev: as global_search."""
    dev = dev or clf.dev
    box_a, b = clf.box_a, box or clf.bmax
    n, J = A.shape[:2]
    fy, fx = freqs(b, dev)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = (kr >= 1) & (kr <= box_a / res_a)
    ph = TWO_PI / box_a
    isig = torch.ones(int(band.sum()), device=dev) if sigma2 is None else 1.0 / sigma2[torch.round(kr[band]).long()]
    sh = torch.zeros(1, 2, device=dev) if shifts is None else shifts.to(dev).float()
    Es = torch.exp(-1j * ph * (sh[:, 0, None] * fx[None, :].expand_as(kr)[band][None]
                               + sh[:, 1, None] * fy[:, None].expand_as(kr)[band][None]))        # (S, F)
    fxb, fyb = fx[None, :].expand_as(kr)[band], fy[:, None].expand_as(kr)[band]
    Fb = int(band.sum())
    npc = int(max(1, min(n, 0.15 * _free(dev) / (J * Fb * 8 * 4 + b * (b // 2 + 1) * 40))))
    ll = torch.full((n, J), -float("inf"), device=dev)
    bs = torch.zeros((n, J), dtype=torch.long, device=dev)
    A, t = A.to(dev), t.to(dev)
    for h in (0, 1):
        sel_h = torch.nonzero(half.to(dev) == h).squeeze(1)
        for p0 in range(0, len(sel_h), npc):
            sel = sel_h[p0:p0 + npc]
            j = idx[sel.cpu()]
            x = clf._fetch_align(j, dev, b)
            ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
            tt = t[sel]
            P = refs[h].project(A[sel].reshape(-1, 3, 3), band).reshape(len(sel), J, -1)    # (m, J, F)
            if t.dim() == 3:        # a shift per candidate: on the projections (Re<conj x e_t, P> = Re<conj x, e_t P>)
                X = (x.conj() * ctf)[:, band] * isig
                P = P * torch.exp(-1j * ph * (fxb[None, None] * tt[:, :, 0, None] + fyb[None, None] * tt[:, :, 1, None]))
            else:
                e_t = torch.exp(-1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
                X = (x.conj() * e_t * ctf)[:, band] * isig                    # (m, F)
                del e_t
            C2 = (ctf * ctf)[:, band] * isig
            del x, ctf
            pw = 0.5 * torch.einsum("mf,mjf->mj", C2, P.real ** 2 + P.imag ** 2)
            for s in range(len(sh)):
                v = torch.einsum("mf,mjf->mj", X * Es[s], P).real - pw
                upd = v > ll[sel]
                ll[sel] = torch.where(upd, v, ll[sel])
                bs[sel] = torch.where(upd, s, bs[sel])
            del P, X, C2
    return ll, (t if t.dim() == 3 else t[:, None, :]) + sh[bs]


def distinct_candidates(A: torch.Tensor, t: torch.Tensor, ops: torch.Tensor, k: int, min_deg: float,
                        min_shift: float) -> torch.Tensor:
    """Greedy non-maximum suppression of per-particle candidate poses sorted best first: A (n, C, 3, 3), t (n, C, 2).
    A candidate is dropped when an already kept one is within min_deg (any symmetry-equivalent orientation) AND within
    min_shift A. Returns indices (n, k) into C of the kept ones (best first; padded with the best when fewer)."""
    n, C = A.shape[:2]
    dev = A.device
    if n > 2048:                                   # (n, C, C) closeness for every particle at once does not fit
        return torch.cat([distinct_candidates(A[i:i + 2048], t[i:i + 2048], ops, k, min_deg, min_shift)
                          for i in range(0, n, 2048)])
    # cos-like closeness: trace(A_i^T A_j S) = sum(A_i * (A_j S)) for every operator S (A S is the same pose of a
    # symmetric map), max over S
    tr = torch.full((n, C, C), -3.0, device=dev)
    for S in ops.to(dev).float():
        tr = torch.maximum(tr, torch.einsum("nixy,njxy->nij", A, A @ S))
    close = (tr >= 1 + 2 * math.cos(math.radians(min_deg))) & (torch.cdist(t, t) <= min_shift)
    keep = torch.zeros((n, C), dtype=torch.bool, device=dev)
    for c in range(C):
        clash = (close[:, c, :c] & keep[:, :c]).any(1) if c else torch.zeros(n, dtype=torch.bool, device=dev)
        keep[:, c] = ~clash & (keep.sum(1) < k)
    order = torch.argsort((~keep).float() * C + torch.arange(C, device=dev)[None].float(), dim=1)[:, :k]
    nk = keep.sum(1, keepdim=True)
    return torch.where(torch.arange(k, device=dev)[None] < nk, order, order[:, :1])


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
    """How the viewing directions spread over the sphere, in ~10 deg bins: the fraction of bins with a real share of
    the particles and the share of particles in the busiest 10 % of bins. Every particle counts at all its
    symmetry-equivalent directions, so the histogram covers the whole sphere for any group (folding each to its
    nearest copy put C8's uniform views at 0 % occupied and a top + side band at 100 %). A map from a narrow range of
    views is smeared along the missing directions, whatever its FSC says."""
    ops = symmetry_ops(sym).to(A.device).float()
    rot, tilt = sphere_directions(10.0)
    th, ph = torch.deg2rad(tilt), torch.deg2rad(rot)
    bins = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1).to(A.device)
    d = A[:, 2, :].float()                                    # viewing direction in the map frame
    cnt = torch.zeros(len(bins), device=A.device)
    for S in ops:                                             # the direction of A @ S is d @ S
        for i in range(0, len(d), 65536):
            cnt += torch.bincount(((d[i:i + 65536] @ S) @ bins.T).argmax(1), minlength=len(bins)).float()
    # a direction counts if it holds at least a quarter of what a uniform spread would give (with 10^5+ particles a
    # few strays land everywhere)
    expect = cnt.sum() / len(bins)
    occ = float((cnt >= 0.25 * expect).float().mean())
    top = torch.sort(cnt, descending=True).values
    share = float(top[:max(1, len(bins) // 10)].sum() / cnt.sum().clamp_min(1))
    return dict(occupied=round(occ, 3), top10_share=round(share, 3))


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
        _live("sgd", it + 1, n_iter, [V], [res], box_a / bs)
    # every particle's orientation in the final map
    A, _, _ = global_search(clf, idx, torch.eye(3).expand(n, 3, 3), t, [V, V], zeros, res_end,
                            global_step(res_end, diam), box=bs)
    log(f"[3d] particle SGD: {n_iter} rounds of {min(batch, n)} particles, {res_start:.0f} -> {res_end:.0f} A "
        f"({since(t0):.1f}s)")
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
        x = clf._fetch_align(j, dev, b)[:, band]
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


SYM_RESAMPLE = True      # refine_particles: symmetry by resampling the accumulators (False: insert at every operator)
FAST_STEPS = True        # refine_particles (one model, no search limits): the step halves every iteration once the poses settle, and the
                         # refinement stops at step_min when they no longer move (False: halve after < 30 % moved)


def refine_particles(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, inits, z: torch.Tensor | None = None,
                     sym: str = "C1", n_iter: int = 8, res_start: float = 15.0, step_start: float | None = None,
                     batch: int = 256, n_local: int = 1, step_scale: float | None = None, global_at: tuple = (),
                     global_res: float = 12.0, drop_frac: float = 0.0, rebalance: float = 0.0,
                     box: int | None = None, relax_ops: torch.Tensor | None = None, relax_iters: int = 0,
                     step_min: float = 0.5,
                     half: torch.Tensor | None = None, soft: bool = False, solvent_mask=None,
                     solvent_from_a: float = 15.0, pad: int | None = None, grid_correct: bool = False,
                     nu: bool = False, nu_strict: bool = True, particle_scale: bool = False, wiener: float = 0.0,
                     max_angle: float | None = None, max_shift: float | None = None,
                     limit_A: torch.Tensor | None = None, limit_t: torch.Tensor | None = None,
                     shift_per_deg: float | None = None, iter_stats: bool = False, subpixel: bool = False,
                     nu_weighted: bool = False, nu_awf: float = 0.0, nu_order: int = 0, marginalize: int = 0,
                     nu_ratio: float = 0.0, nu_shift: int = 0, noise_model: str = "residual",
                     rescue_frac: float = 0.0, rescue_at: tuple = (), rescue_shift: float = 24.0,
                     rescue_res: float = 15.0, rescue_topk: int = 1, coarse: float = 0.0, coarse_keep: int = 8,
                     log=print):
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
    solvent_mask: "auto" - a solvent mask made from each model's current map every iteration once it is better than
    solvent_from_a (masking.auto_mask, noise from the half maps; cryoSPARC's dynamic mask) - or a fixed mask (b^3
    tensor, or a list of one per model). It replaces the particle sphere for the FSC that sets the resolution and weights the references (then
    corrected by phase randomisation, as it is tight), for flattening the references' solvent, and for the final map:
    the references lose the solvent noise and keep more of the signal. refine_particles.last_masks holds the last
    one per model (None: the sphere), last_mask model 1's.
    pad: padding of the 3D spectra (None: 2 if the volumes, ~60 (2b)^3 bytes for one model, fit in 60 % of the free
    memory, else 1 - box 512 in 16 GB). grid_correct: gridding correction of every volume (Volume; pad 1 needs it).
    nu (one model): non-uniform regularisation (nufilter.adaptive_filter) below 15 A: each half's reference is
    low-passed region by region, the cutoff chosen by cross-validation against the other half, instead of the global
    FSC filter. refine_particles.last_nu holds the last cutoff statistics (nufilter.cutoff_stats; per iteration in the
    history's "nu"), refine_particles.last_nu_map the last cutoff map (A, mean of the halves). nu_strict (default): the cutoffs of each
    half chosen by cross-validation between two folds of that half's own particles (each reconstructed on a small
    pad-1 box sized to the resolution), so no information crosses between the halves (gold standard).
    marginalize (one model, hard poses): each particle is inserted at its this many best poses of the local search,
    weighted by their posterior (cryoSPARC's pose marginalisation), instead of only at the best one; 0: off.
    nu_awf / nu_order: the non-uniform filter's window per candidate and Butterworth candidates (nufilter).
    nu_weighted: the non-uniform filter's candidates are the global FSC weighting stretched per region (nufilter
    shape=), not hard low-passes.
    subpixel: shifts between the grid points: along x and y, where the best of the 3 x 3 shift hypotheses (at the best
    orientation) is the middle one, the peak of the parabola through the three log-likelihoods (at most half a step
    away). The grid step never goes below half a pixel, so without it a shift is only as precise as that step.
    iter_stats: also keep, per iteration, every model's FSC curve and every particle's angular change (deg, in its
    model), shift change (A) and fit: refine_particles.last_iters = dict(fsc=[[curve per model] per iteration],
    angle / shift / fit=[(n,) float16 per iteration]) - how far poses still move, which particles keep jumping.
    particle_scale: a contrast scale a_i per particle (cryoSPARC's per-particle scale; RELION's scale correction is per
    group): the model of a particle is a_i x its CTF-weighted projection - ice thickness and dose change the contrast -
    used in the pose search (a_i cross - a_i^2 power / 2) and the reconstruction (a_i x, a_i^2 CTF^2), re-estimated
    every iteration at the best pose (noise-weighted least squares over the alignment band; median 1, 0.2 - 5).
    refine_particles.last_scale holds it.
    wiener: > 0 - the half reconstructions regularised by the previous iteration's FSC (Wiener / MAP, RELION's tau2 with
    this fudge factor; _regularise_solve), and the references then used as they are (they hold the FSC weighting);
    0 - a fixed 2 % of the shell's mean weight, references and the final map weighted by sqrt(2 FSC / (1 + FSC)).
    max_angle (deg) / max_shift (A) (one model; local refinement, e.g. focused on a masked region): only poses within
    this angle of limit_A (n, 3, 3) and shifts within this distance of limit_t (n, 2) (default: the starting poses) are
    searched - RELION's local searches / cryoSPARC's search extents, as hard limits.
    shift_per_deg: shift step per degree of angular step (A; default: half a pixel per 1.5 deg, at least half a pixel);
    for a focused region of radius r, r pi / 180 (a turn of one step moves its edge by one shift step).
    Returns (maps (M, b, b, b), A (n, 3, 3), t (n, 2) in the chosen model, z, fsc (M, shells), resolutions (M,),
    half Volumes [[h0, h1] per model])."""
    dev = clf.dev
    box_a, diam = clf.box_a, clf.p.diameter_a
    b = box or clf.bmax
    refine_particles.last_nu = refine_particles.last_nu_map = None
    n = len(idx)
    inits = inits if isinstance(inits, (list, tuple)) else [inits]
    bp_n = int(max(64, min(2048, 2e8 // (b * (b // 2 + 1) * 8))))   # particles per back-projection batch
    M = len(inits)
    z = torch.zeros(n, dtype=torch.long, device=dev) if z is None else z.clone().to(dev).long()
    ops = symmetry_ops(sym).to(dev)
    if half is None:
        half = torch.randperm(n, generator=torch.Generator().manual_seed(1)) % 2
    half = torch.as_tensor(half).long().cpu()
    fold = torch.zeros(n, dtype=torch.long)         # strict NU: two folds inside each half
    for h_ in (0, 1):
        k_ = torch.nonzero(half == h_).squeeze(1)
        fold[k_[torch.randperm(len(k_), generator=torch.Generator().manual_seed(7 + h_))]] = torch.arange(len(k_)) % 2
    subvols = None                                   # [half][fold] Volumes of this back-projection (strict NU)
    sub_maps = None
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
    limited = max_angle is not None or max_shift is not None
    if limited:
        if M > 1:
            raise ValueError("search limits (max_angle / max_shift) are for one model")
        limit_A = (A[:, 0] if limit_A is None else torch.as_tensor(limit_A).float()).to(dev).clone()
        limit_t = (t[:, 0] if limit_t is None else torch.as_tensor(limit_t).float()).to(dev).clone()
        cos_lim = 1 + 2 * math.cos(math.radians(max_angle)) - 1e-4 if max_angle is not None else None
    if pad is None:
        pad = auto_pad(b, M * (1.15 if nu else 1.0), dev)   # NU: its filtered maps and folds (+24 % peak)
        if pad == 1:
            log(f"[3d] box {b}: 3D spectra without padding (pad 1, gridding-corrected) to fit the GPU")
    gc = grid_correct or pad == 1
    vols = [[Volume(b, dev, pad=pad, grid_correct=gc), Volume(b, dev, pad=pad, grid_correct=gc)] for _ in range(M)]
    v_init = [_to_box(v, b, dev) for v in inits]
    res = [res_start] * M
    fscs = [None] * M
    alive = [True] * M                    # a model that loses (almost) all its particles is retired
    wpart = torch.ones(n, device=dev)     # per-particle reconstruction weight (dropped particles 0, rebalancing < 1)
    pscale = torch.ones(n, device=dev)    # per-particle contrast scale (particle_scale)
    pscale_new = torch.ones(n, device=dev)
    score = torch.zeros(n, device=dev)    # normalised fit of the best pose (for dropping particles)
    # marginalisation: the K best local hypotheses as a rotation / shift relative to the best pose, and their weights
    marg = marginalize if (marginalize > 1 and M == 1 and not soft) else 0
    marg_D = marg_dt = marg_w = None
    dfz = clf.ctf_p[idx, :2].mean(1)      # defocus, to compare fits within defocus groups
    dgrp = torch.bucketize(dfz, torch.quantile(dfz, torch.linspace(0, 1, 9)[1:-1])).to(dev)
    sigma2 = None
    post = F.one_hot(z, M).float()       # posterior over models (one-hot unless soft)
    T0 = None
    step = step_start or max(1.0, math.degrees(res_start / diam))
    t0 = time.time()
    history = []                          # per iteration: alignment limit, step, share moved, resolution after it
    iters = dict(fsc=[], angle=[], shift=[], fit=[]) if iter_stats else None   # iter_stats: per iteration
    kmax_used = [float(b)] * M            # highest shell inserted in the last reconstruction

    # symmetry by one resampling of the accumulators after the back-projection instead of |S| insertions per image
    # (C8: ~8x less splatting); at pad 2 the two agree within 4 %, at pad 1 the resampling costs low-frequency accuracy
    sym_after = len(ops) > 1 and pad >= 2 and SYM_RESAMPLE

    def backproject():
        for vm in vols:
            for v in vm:
                v.clear()

        def bp(lo, hi, d):
            """Particles lo..hi inserted on GPU d: into vols / subvols on the master, else into this GPU's own
            accumulators (added to the master's afterwards)."""
            if torch.device(d) == torch.device(dev):
                vd, sd = vols, subvols
            else:
                vd = [[Volume(b, d, pad=pad, grid_correct=gc) for _ in (0, 1)] for _ in range(M)]
                sd = None if subvols is None else [[Volume(subvols[0][0].b, d, pad=1, grid_correct=True)
                                                    for _ in (0, 1)] for _ in (0, 1)]
            fy_d, fx_d = fy.to(d), fx.to(d)
            ops_d = None if sym_after else ops.to(d)
            for i in range(lo, hi, bp_n):
                i2 = min(i + bp_n, hi)
                j = idx[i:i2]
                x = clf._fetch(j, d, b)
                ctf = ctf_2d(clf.ctf_p[j].to(d), clf.optics, fy_d, fx_d, box_a)
                zz = z[i:i2].to(d)
                hh = half[i:i2].to(d)
                wp = wpart[i:i2].to(d)
                ps = pscale[i:i2].to(d)[:, None, None]
                pp = post[i:i2].to(d)
                At, tt_all = A[i:i2].to(d), t[i:i2].to(d)
                arl = torch.arange(i2 - i, device=d)
                if not soft:
                    tt, AA = tt_all[arl, zz], At[arl, zz]
                    xs = x * torch.exp(1j * ph * (fx_d[None, None, :] * tt[:, 0, None, None] + fy_d[None, :, None] * tt[:, 1, None, None]))
                for m in range(M):
                    kmax = min(b / 2 - 1, 1.3 * box_a / res[m] + 6)   # beyond the current resolution, so FSC can grow
                    kmax_used[m] = kmax
                    if soft:     # every particle at its pose in this model, weighted by its posterior
                        tt, AA = tt_all[:, m], At[:, m]
                        xs = x * torch.exp(1j * ph * (fx_d[None, None, :] * tt[:, 0, None, None] + fy_d[None, :, None] * tt[:, 1, None, None]))
                    for h in (0, 1):
                        sel = (hh == h) & (wp > 0) & ((pp[:, m] > 1e-3) if soft else (zz == m))
                        if not sel.any():
                            continue
                        if marg_w is not None:     # every particle at its K best poses, weighted by their posterior
                            mw, mD, mdt = marg_w[i:i2].to(d), marg_D[i:i2].to(d), marg_dt[i:i2].to(d)
                            for k in range(marg):
                                wk = mw[:, k]
                                sk = sel & (wk > 1e-3)
                                if not sk.any():
                                    continue
                                Ak = mD[:, k] @ AA
                                tk = tt + mdt[:, k]
                                xk = x * torch.exp(1j * ph * (fx_d[None, None, :] * tk[:, 0, None, None] + fy_d[None, :, None] * tk[:, 1, None, None]))
                                w3 = (wp[sk] * wk[sk])[:, None, None]
                                vd[m][h].insert((xk * ctf)[sk] * (w3 * ps[sk]), (ctf * ctf)[sk] * (w3 * ps[sk] ** 2), Ak[sk],
                                                kmax=kmax, ops=ops_d)
                            continue
                        w3 = (wp[sel] * (pp[sel, m] if soft else 1.0))[:, None, None]
                        vd[m][h].insert((xs * ctf)[sel] * (w3 * ps[sel]), (ctf * ctf)[sel] * (w3 * ps[sel] ** 2), AA[sel],
                                        kmax=kmax, ops=ops_d)
                if sd is not None:                  # the folds, on their small box
                    bs_ = sd[0][0].b
                    fys, fxs = sd[0][0].fy, sd[0][0].fx
                    x_s = ft_resize(x, bs_)                     # the batch already fetched, cropped (not read again)
                    c_s = ctf_2d(clf.ctf_p[j].to(d), clf.optics, fys, fxs, box_a)
                    tt, AA = tt_all[arl, zz], At[arl, zz]
                    xs_s = x_s * torch.exp(1j * ph * (fxs[None, None, :] * tt[:, 0, None, None] + fys[None, :, None] * tt[:, 1, None, None]))
                    ff = fold[i:i2].to(d)
                    for h in (0, 1):
                        for f_ in (0, 1):
                            sel = (hh == h) & (ff == f_) & (wp > 0)
                            if sel.any():
                                w3 = wp[sel][:, None, None]
                                sd[h][f_].insert((xs_s * c_s)[sel] * (w3 * ps[sel]), (c_s * c_s)[sel] * (w3 * ps[sel] ** 2),
                                                 AA[sel], kmax=bs_ / 2 - 1)
            return vd, sd

        for vd, sd in on_devices(clf, n, bp):
            if vd is not vols:                       # another GPU's share
                for m in range(M):
                    for h in (0, 1):
                        vols[m][h].add_acc(vd[m][h])
                if sd is not None:
                    for h in (0, 1):
                        for f_ in (0, 1):
                            subvols[h][f_].add_acc(sd[h][f_])
        for vm in (vols if sym_after else []) + (subvols or []):   # the folds only choose cutoffs: resampled at pad 1 too
            for v in vm:
                v.symmetrize(ops)

    refs = None
    cur_masks = [None] * M          # per model: solvent mask in use (solvent_mask), else the particle sphere
    settled = converged = False     # FAST_STEPS: the poses have settled at the starting step / at step_min
    for it in range(n_iter + 1):
        # ---- reconstruct halves, FSC, filter
        refs = r = rm = None        # the last alignment's references (also bound to the loop variables that built them)
        release_cache()    # and cached batch blocks are not needed while the halves are back-projected
        if it == 0:
            for m in range(M):
                for v in vols[m]:
                    v.set_real(v_init[m])
        else:
            want_sub = nu and nu_strict and M == 1 and fscs[0] is not None and res[0] < solvent_from_a and not soft
            subvols = None
            if want_sub:
                bs_ = min(b, int(math.ceil(2 * box_a / (0.85 * res[0]) / 2)) * 2 + 8)
                subvols = [[Volume(bs_, dev, pad=1, grid_correct=True) for _ in (0, 1)] for _ in (0, 1)]
            backproject()
            sub_maps = None
            if subvols is not None:
                sub_maps = []
                for hv in subvols:
                    row_ = []
                    for v in hv:
                        _regularise_solve(v, 0.02)
                        v.free_acc()
                        row_.append(v.real())
                    sub_maps.append(row_)
                subvols = None
            for m in range(M):
                for v in vols[m]:
                    _regularise_solve(v, 0.02, fsc=fscs[m] if wiener > 0 else None, tau2_fudge=wiener or 1.0)
                    v.free_acc()        # 3/2 of the volume's size; made again by the next back-projection (box 256:
                                        # 3.2 GB for two halves that sat idle through the alignment)
                fixed_ = (solvent_mask[m] if isinstance(solvent_mask, (list, tuple)) else solvent_mask) \
                    if not isinstance(solvent_mask, str) else None          # a fixed mask (one, or one per model)
                if solvent_mask is not None and alive[m] and (fixed_ is not None or res[m] < solvent_from_a):
                    from .masking import auto_mask
                    h1_, h2_ = vols[m][0].real(), vols[m][1].real()
                    if fixed_ is not None:
                        cur_masks[m] = torch.as_tensor(fixed_).to(dev).float()
                    else:                   # the masked average and half difference, once each (box 512: memory)
                        cur_masks[m] = None
                        avg_ = (h1_ + h2_).mul_(0.5).mul_(mask)
                        # the noise from the whole box: masked to the sphere, the low-passed difference is ~0 over
                        # most of it, its MAD ~0, and the threshold fell to the 3 % of peak floor (J341 mask 1.21 vs
                        # 1.12 M A^3 with the noise measured; cryoSPARC's 1.04)
                        dif_ = (h1_ - h2_).mul_(0.5)
                        try:
                            cur_masks[m] = auto_mask(avg_, box_a / b, res_a=res[m], k_sigma=5.0, halves=(dif_, None),
                                                     diameter_a=diam)
                        except Exception as e:      # a class too small / noisy for a mask: the sphere this time
                            log(f"[3d] model {m + 1}: no solvent mask this iteration ({e})")
                        del avg_, dif_
                    fscs[m] = (corrected_masked_fsc(h1_, h2_, cur_masks[m]) if cur_masks[m] is not None
                               else masked_fsc(vols[m][0], vols[m][1], mask))
                    del h1_, h2_
                else:
                    fscs[m] = masked_fsc(vols[m][0], vols[m][1], mask)
                # shells beyond the reconstruction limit hold only what the mask leaks from lower shells (correlated
                # between the halves): not signal
                fscs[m][int(kmax_used[m]) + 1:] = 0
                res[m] = max(resolution_from_fsc(fscs[m], box_a), 2.2 * box_a / b)
            if history:
                history[-1]["res"] = [round(r, 3) if alive[m] else None for m, r in enumerate(res)]
                history[-1]["seconds"] = round(since(t0), 1)
                if iters is not None:      # the FSC of the poses found in that iteration
                    iters["fsc"].append([None if f is None else f.cpu().numpy().astype(np.float32) for f in fscs])
            pops_ = torch.bincount(z, minlength=M).tolist() if M > 1 else None      # the particles in these maps
            _live("refine", it, n_iter, [vols[m] for m in range(M) if alive[m]], [res[m] for m in range(M) if alive[m]],
                  box_a / b, counts=None if pops_ is None else [pops_[m] for m in range(M) if alive[m]])
        if it == n_iter or converged:
            break
        A_prev, t_prev = A.clone(), t.clone()      # how far this iteration moves each pose (log, history)
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
            infos = [{}, {}]
            nu_shape = torch.sqrt((2 * fscs[0] / (1 + fscs[0])).clamp(0, 1).nan_to_num(0)) if nu_weighted else None
            for h in (0, 1):
                if sub_maps is not None:             # gold standard: cutoffs from this half's own two folds
                    from .nufilter import adaptive_filter_cv
                    f_ = adaptive_filter_cv(hr_[h], sub_maps[h][0], sub_maps[h][1], box_a / b,
                                            box_a / sub_maps[h][0].shape[0], res[0], mask=cur_masks[0], info=infos[h],
                                            shape=nu_shape, awf=nu_awf, order=nu_order,
                                            snr=(fscs[0] / (1 - fscs[0]).clamp_min(1e-3)).nan_to_num(0) if nu_ratio else None,
                                            ratio=nu_ratio, shift=nu_shift)
                else:
                    f_ = adaptive_filter(hr_[h], hr_[1 - h], box_a / b, res[0], mask=cur_masks[0], info=infos[h],
                                         shape=nu_shape, awf=nu_awf, order=nu_order)
                r = Volume(b, dev, pad=pad, grid_correct=gc)
                r.set_real(f_ * (mask if cur_masks[0] is None else cur_masks[0]))
                del f_
                lowpass(r, box_a / k_al, box_a)
                refs[0].append(r)
            del hr_
            sub_maps = None
            from .nufilter import cutoff_stats
            nu_it = cutoff_stats(infos, mask if cur_masks[0] is None else cur_masks[0], res[0])
            nu_it["strict"] = bool(infos[0].get("strict", False))
            refine_particles.last_nu = dict(nu_it, candidates_a=infos[0]["candidates_a"], window_a=infos[0]["window_a"])
            refine_particles.last_nu_map = (0.5 * (infos[0]["cutoff_map_a"] + infos[1]["cutoff_map_a"])).half().cpu()
            if it == 1 or it == n_iter - 1:
                log(f"[3d] non-uniform filter: cutoffs in the mask {nu_it['p05_a']:.1f} / {nu_it['median_a']:.1f} / "
                    f"{nu_it['p95_a']:.1f} A (5 / 50 / 95 %), {nu_it['at_global_pct']:.0f}% at the global "
                    f"resolution or finer, halves agree r = {nu_it['halves_r']:.2f}")
            infos = None
        for m in (range(M) if not use_nu else ()):
            rm = []
            for v in vols[m]:
                r = Volume(b, dev, pad=pad, ft=v.ft.clone(), pre=v.pre, grid_correct=gc)
                if fscs[m] is not None and M == 1 and not wiener:
                    r.filter_shells(torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0)))
                lowpass(r, box_a / k_al, box_a)
                _real_space_constraints(r, mask if cur_masks[m] is None else cur_masks[m], positive=False, center=False)
                rm.append(r)
            refs.append(rm)
        # ---- global re-search: particles that inherited a wrong pose (mixed 2D classes) can jump
        if M == 1 and (it + 1) in global_at:
            g_res = max(global_res, al_res)
            g_step = global_step(g_res, diam)
            tg = time.time()
            Ag, llg, llc = global_search_multi(clf, idx, A[:, 0], t[:, 0], refs[0], half, g_res, g_step, sigma2, box=b,
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
        # ---- rescue: the worst-fitting particles searched again over all orientations AND a wide shift range
        if M == 1 and rescue_frac > 0 and (it + 1) in rescue_at and it > 0:
            tr_ = time.time()
            rel_ = torch.empty_like(score)
            for g_ in range(int(dgrp.max()) + 1):
                sg_ = dgrp == g_
                if sg_.any():
                    rel_[sg_] = score[sg_] / score[sg_].median().clamp_min(1e-6)
            w_ = torch.nonzero(rel_ <= torch.quantile(rel_, rescue_frac)).squeeze(1).cpu()
            r_res = max(rescue_res, al_res)
            p_res = max(al_res, 8.0)
            Rp_ = (rotz(torch.arange(-2, 3, dtype=torch.float32) * 1.5)[:, None] @ local_rotations(1.5, 2)[None]).reshape(-1, 3, 3)
            gp_ = torch.arange(-3.0, 3.01, 1.5)
            fine_ = torch.stack(torch.meshgrid(gp_, gp_, indexing="xy"), -1).reshape(-1, 2)
            if rescue_topk > 1:      # the first polish leaves ~ +-s1/4 and +-g1/4: a 3 x 3 x 3 rotation, 3 x 3 shift grid
                Rp_ = (rotz(torch.tensor([-1.5, 0.0, 1.5]))[:, None] @ local_rotations(1.5, 1)[None]).reshape(-1, 3, 3)
                gp_ = torch.tensor([-1.5, 0.0, 1.5])
                fine_ = torch.stack(torch.meshgrid(gp_, gp_, indexing="xy"), -1).reshape(-1, 2)
                # coarse to fine: the right pose is often not the single best on a coarse grid at low resolution (J341:
                # our likelihood prefers cryoSPARC's pose for 57% of the misassigned particles at 30 A, 85% at 10 A), so
                # keep the best rescue_topk (orientation, shift) hypotheses of a cheap wide search, polish each with the
                # current pose at ~12 A, and the best 3 at p_res
                s1_ = global_step(r_res, diam)
                g1_ = max(3.0, r_res / 3)       # r_res / 5: same poses and maps, the search 2x slower (HCN1, J341)
                gg_ = torch.arange(-rescue_shift, rescue_shift + 0.01, g1_)
                wide_ = torch.stack(torch.meshgrid(gg_, gg_, indexing="xy"), -1).reshape(-1, 2)
                Ak_, _, _, tk_ = global_search_multi(clf, idx[w_], A[w_, 0], t[w_, 0], refs[0], half[w_], r_res, s1_, sigma2,
                                               box=b, log=log, shifts=wide_, topk=4 * rescue_topk)
                ts_ = [time.time() - tr_]
                # neighbours of one basin crowd the top of the list: keep rescue_topk distinct ones
                kk_ = distinct_candidates(Ak_.to(dev), tk_.to(dev), ops, rescue_topk, 2 * s1_, 2 * g1_).cpu()
                ar0_ = torch.arange(len(w_))[:, None]
                Ak_, tk_ = Ak_.cpu()[ar0_, kk_], tk_.cpu()[ar0_, kk_]
                Ac_ = torch.cat([A[w_, 0].cpu()[:, None], Ak_.cpu()], 1)               # (w, K+1): current first
                tc_ = torch.cat([t[w_, 0].cpu()[:, None], tk_.cpu()], 1)
                m_res = max(al_res, min(12.0, r_res))
                h1_ = s1_ / 2
                R1_ = (rotz(torch.tensor([-h1_, 0.0, h1_]))[:, None] @ local_rotations(h1_, 1)[None]).reshape(-1, 3, 3)
                o1_ = torch.tensor([-g1_ / 2, 0.0, g1_ / 2])
                off1_ = torch.stack(torch.meshgrid(o1_, o1_, indexing="xy"), -1).reshape(-1, 2)
                nK_, nR_ = Ac_.shape[1], len(R1_)
                nb_ = (R1_[None, None] @ Ac_[:, :, None]).reshape(len(w_), nK_ * nR_, 3, 3)
                tb_ = tc_[:, :, None].expand(-1, -1, nR_, -1).reshape(len(w_), nK_ * nR_, 2)
                ll1_, ts1_ = score_poses_multi(clf, idx[w_], nb_, tb_, refs[0], half[w_], m_res, sigma2=sigma2, box=b, shifts=off1_)
                ll1_, ts1_ = ll1_.reshape(len(w_), nK_, nR_).cpu(), ts1_.reshape(len(w_), nK_, nR_, 2).cpu()
                ts_.append(time.time() - tr_)
                v1_, k1_ = ll1_.max(2)                                                  # best neighbour per candidate
                ar_ = torch.arange(len(w_))[:, None]
                A1_ = nb_.reshape(len(w_), nK_, nR_, 3, 3)[ar_, torch.arange(nK_)[None], k1_]
                t1_ = ts1_[ar_, torch.arange(nK_)[None], k1_]
                top3 = torch.topk(v1_[:, 1:], min(3, nK_ - 1), dim=1).indices + 1      # best 3 new candidates
                Ag = A1_[ar_, top3]
                tg = t1_[ar_, top3]
                nb_ = (Rp_[None, None] @ Ag[:, :, None]).reshape(len(w_), -1, 3, 3)
                tb_ = tg[:, :, None].expand(-1, -1, len(Rp_), -1).reshape(len(w_), -1, 2)
                ll2_, ts2_ = score_poses_multi(clf, idx[w_], nb_, tb_, refs[0], half[w_], p_res, sigma2=sigma2, box=b, shifts=fine_)
                k2_ = ll2_.argmax(1).cpu()
                ar1_ = torch.arange(len(w_))
                found = (ll2_.max(1).values.cpu(), nb_[ar1_, k2_], ts2_.cpu()[ar1_, k2_])
                ts_.append(time.time() - tr_)
            else:
                gg_ = torch.arange(-rescue_shift, rescue_shift + 0.01, 3.0)
                wide_ = torch.stack(torch.meshgrid(gg_, gg_, indexing="xy"), -1).reshape(-1, 2)
                Ag, _, _, tg = global_search_multi(clf, idx[w_], A[w_, 0], t[w_, 0], refs[0], half[w_], r_res,
                                             max(6.0, global_step(r_res, diam)), sigma2, box=b, log=log, shifts=wide_)
                found = None
                ts_ = []
            # both the current pose and the found one polished at a finer resolution (1.5 deg, +-3 A), the found one
            # adopted only if clearly better there
            pol = []
            for Ac_, tc_ in ((A[w_, 0].cpu(), t[w_, 0].cpu()),) + ((((Ag.cpu(), tg.cpu())),) if found is None else ()):
                nb_ = Rp_[None] @ Ac_[:, None]
                ll_, tb_ = score_poses_multi(clf, idx[w_], nb_, tc_, refs[0], half[w_], p_res, sigma2=sigma2, box=b, shifts=fine_)
                k_ = ll_.argmax(1).cpu()
                ar_ = torch.arange(len(w_))
                pol.append((ll_.max(1).values.cpu(), nb_[ar_, k_], tb_.cpu()[ar_, k_]))
            if found is not None:
                pol.append(found)
            take = pol[1][0] > pol[0][0] + 3.0
            if take.any():
                wt = w_[take]
                A[wt, 0] = pol[1][1][take].to(A.dtype).to(A.device)
                t[wt, 0] = pol[1][2][take].to(t.dtype).to(t.device)
            log(f"[3d] rescue: the worst-fitting {rescue_frac * 100:.0f}% ({len(w_)}) searched at {r_res:.0f} A with shifts "
                f"+-{rescue_shift:.0f} A{f', best {rescue_topk} polished' if rescue_topk > 1 else ''}: {int(take.sum())} ({float(take.float().mean()) * 100:.1f}%) took a new pose "
                f"({time.time() - tr_:.1f}s{'; search / polish / fine done at ' + ' / '.join(f'{x:.0f}' for x in ts_) + 's' if ts_ else ''})")
            step = max(step, 1.5)
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
        if shift_per_deg:   # a region of radius r: a turn of `step` moves its edge by r step (in rad)
            ts = torch.tensor([-1.0, 0.0, 1.0], device=dev) * max(0.1 * box_a / b, step * shift_per_deg)
        else:
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
        Fb = int(band.sum())
        # two stages (coarse > 0): every hypothesis scored on the band's low frequencies (up to coarse x k_al), the
        # coarse_keep best orientations on all of it
        two = coarse > 0 and J > coarse_keep
        band_lo = band & (kr <= coarse * k_al) if two else None
        lo_sel = (kr[band] <= coarse * k_al) if two else None
        llm_all = torch.zeros((n, M), device=dev)
        if marg:
            marg_D = torch.eye(3, device=dev).repeat(n, marg, 1, 1)
            marg_dt = torch.zeros((n, marg, 2), device=dev)
            marg_w = torch.zeros((n, marg), device=dev)

        def estep(lo, hi, d):
            """The local search of particles lo..hi on GPU d (references copied there); poses, scores and model
            choices written into the master arrays. Returns (moved, switched, residual power, counts) of the share."""
            refs_d = [[r.on(d) for r in rm] for rm in refs]
            band_d, fy_d, fx_d = band.to(d), fy.to(d), fx.to(d)
            fyb_d, fxb_d, E_d, isig_d, shb_d, Rc_d, sg_d = (v_.to(d) for v_ in (fyb, fxb, E, isig, shb, Rc, sg))
            if two:
                blo_d, lo_d = band_lo.to(d), lo_sel.to(d)
            ops_d = ops_x.to(d) if xo else None
            rp, rc = torch.zeros(len(sigma2), device=d), torch.zeros(len(sigma2), device=d)
            ch_, sw_ = 0.0, 0
            ar_m = torch.arange(M, device=d)
            # particles per search batch: projections, CTF-weighted copies and products (~4 complex arrays of
            # M x J x F per particle) must fit in a third of the free memory
            bt = int(max(8, min(batch, _free(d) / 4 / (M * J * Fb * 8 * 4 + M * J * Sh * 8))))
            for i in range(lo, hi, bt):
                i2 = min(i + bt, hi)
                j = idx[i:i2]
                nb = len(j)
                hh = half[i:i2].to(d)
                x = clf._fetch_align(j, d, b)[:, band_d]                    # (nb, F)
                ctf = ctf_2d(clf.ctf_p[j].to(d), clf.optics, fy_d, fx_d, box_a)[:, band_d]
                Ai = A[i:i2].to(d)
                if xo:   # around every symmetry-related copy of the pose: (nb, M, J0 * S, 3, 3)
                    Ac = (Rc_d[None, None, :, None] @ (Ai[:, :, None, None] @ ops_d[None, None, None])).reshape(
                        nb, M, J, 3, 3)
                else:
                    Ac = Rc_d[None, None] @ Ai[:, :, None]     # (nb, M, J, 3, 3): around the pose in each model
                tt = t[i:i2].to(d)                                                                          # (nb, M, 2)
                ps_i = pscale[i:i2].to(d)
                aa = ps_i[:, None, None, None]                                                              # contrast scale

                def outside(Ac_):
                    """Hypotheses beyond the angular / shift limits around the starting pose (not searched)."""
                    okm = torch.ones((nb, M, Ac_.shape[2], Sh), dtype=torch.bool, device=d)
                    if max_angle is not None:      # trace(Ac limit_A^T) = 1 + 2 cos(angle between them)
                        tr = (Ac_ * limit_A[i:i2].to(d)[:, None, None]).sum((-1, -2))                      # (nb, M, J)
                        okm &= (tr >= cos_lim)[..., None]
                    if max_shift is not None:
                        dsh = tt[:, :, None, :] + sg_d[None, None] - limit_t[i:i2].to(d)[:, None, None]     # (nb, M, Sh, 2)
                        okm &= (dsh.norm(dim=-1) <= max_shift + 1e-3)[:, :, None, :]
                    return ~okm

                def project(Ac_, bmask):
                    J_ = Ac_.shape[2]
                    P_ = torch.empty((nb, M, J_, int(bmask.sum())), dtype=torch.complex64, device=d)
                    for m in range(M):
                        for h in (0, 1):
                            sel = torch.nonzero(hh == h).squeeze(1)
                            if len(sel):
                                P_[sel, m] = refs_d[m][h].project(Ac_[sel, m].reshape(-1, 3, 3), bmask).reshape(len(sel), J_, -1)
                    return P_

                jmap = None
                if two:
                    CP1 = project(Ac, blo_d) * ctf[:, lo_d][:, None, None]
                    e1 = torch.exp(-1j * ph * (fxb_d[lo_d] * tt[..., 0, None] + fyb_d[lo_d] * tt[..., 1, None]))
                    Z1 = (x[:, lo_d].conj()[:, None] * e1)[:, :, None] * CP1 * isig_d[lo_d]
                    l1 = aa * (Z1 @ E_d[:, lo_d].T).real
                    del Z1, e1
                    l1 -= 0.5 * aa * aa * (CP1.abs() ** 2 * isig_d[lo_d]).sum(-1)[..., None]
                    del CP1
                    if limited:
                        l1 = l1.masked_fill(outside(Ac), -float("inf"))
                    jmap = l1.amax(-1).topk(coarse_keep, dim=-1).indices                                    # (nb, M, T)
                    del l1
                    Ac = Ac.gather(2, jmap[..., None, None].expand(-1, -1, -1, 3, 3))
                Jx = Ac.shape[2]
                CP = project(Ac, band_d) * ctf[:, None, None]
                # shift hypotheses t + sg: model = exp(-i ph f.(t+s)) CP
                e_t = torch.exp(-1j * ph * (fxb_d * tt[..., 0, None] + fyb_d * tt[..., 1, None]))         # (nb, M, F)
                Z = (x.conj()[:, None] * e_t)[:, :, None] * CP * isig_d                                     # (nb, M, J, F)
                cross = (Z @ E_d.T).real                                                                    # (nb, M, J, Sh)
                del Z
                pw = (CP.abs() ** 2 * isig_d).sum(-1)                                                       # (nb, M, J)
                ll = (aa * cross - 0.5 * aa * aa * pw[..., None])                                           # (nb, M, J, Sh)
                if limited:     # hypotheses beyond the angular / shift limits around the starting pose are not searched
                    ll = ll.masked_fill(outside(Ac), -float("inf"))
                ll = ll.reshape(nb, M, -1)
                llm, bjs = ll.max(-1)                                                                       # best pose per model
                if not all(alive):
                    llm[:, [m for m in range(M) if not alive[m]]] = -float("inf")
                bj, bs = bjs // Sh, bjs % Sh                                                                # (nb, M)
                bm = llm.argmax(1)
                llm_all[i:i2] = llm.to(dev)
                ar = torch.arange(nb, device=d)
                if marg:                      # the K best hypotheses (the first is the best), relative to the best
                    lk, ik = ll[:, 0].topk(marg, dim=1)
                    Ak_ = Ac[ar[:, None], 0, ik // Sh]                                                     # (nb, K, 3, 3)
                    tk_ = tt[:, 0, None] + sg_d[ik % Sh]                                                   # (nb, K, 2)
                    marg_D[i:i2] = (Ak_ @ Ak_[:, :1].transpose(-1, -2)).to(dev)
                    marg_dt[i:i2] = (tk_ - tk_[:, :1]).to(dev)
                    marg_w[i:i2] = torch.softmax(lk, dim=1).to(dev)
                A[i:i2] = Ac[ar[:, None], ar_m[None], bj].to(dev)
                t_new = tt + sg_d[bs]
                if subpixel and Sh == 9:       # parabola through the 3 shift log-likelihoods along x and along y
                    lg = ll.reshape(nb, M, Jx, 3, 3)[ar[:, None], ar_m[None], bj]                            # (nb, M, y, x)
                    iy, ix = bs // 3, bs % 3
                    ds = float(ts[2] - ts[1])
                    for ax, ic in ((0, ix), (1, iy)):            # t[..., 0] is x (grid columns), t[..., 1] is y (rows)
                        line = lg[ar[:, None], ar_m[None], iy, :] if ax == 0 else lg[ar[:, None], ar_m[None], :, ix]
                        l0, l1, l2 = line[..., 0], line[..., 1], line[..., 2]
                        curv = l0 - 2 * l1 + l2
                        off = (0.5 * (l0 - l2) / curv.clamp(max=-1e-9)).clamp(-0.5, 0.5)
                        ok = (ic == 1) & (curv < 0) & torch.isfinite(off)
                        t_new[:, :, ax] += torch.where(ok, off, torch.zeros_like(off)) * ds
                t[i:i2] = t_new.to(dev)
                sw_ += int((bm != z[i:i2].to(d)).sum())
                z[i:i2] = bm.to(dev)
                bj, bs = bj[ar, bm], bs[ar, bm]
                bjo = bj if jmap is None else jmap[ar, bm, bj]                  # index on the full local grid
                ch_ += float((bjo != jc).float().sum() + (bs != Sh // 2).float().sum()) / 2
                # residual power for the noise model
                pscale_new[i:i2] = (cross[ar, bm, bj, bs] / pw[ar, bm, bj].clamp_min(1e-30)).to(dev)   # LS scale at the pose
                mdl = CP[ar, bm, bj] * e_t[ar, bm] * E_d[bs] * ps_i[:, None]
                score[i:i2] = ((x.conj() * mdl).real.sum(1)
                               / ((x.abs() ** 2).sum(1) * (mdl.abs() ** 2).sum(1)).sqrt().clamp_min(1e-12)).to(dev)
                r2 = (x - mdl).abs() ** 2
                rp.index_add_(0, shb_d, r2.sum(0))
                rc.index_add_(0, shb_d, torch.full_like(shb_d, nb, dtype=torch.float32))
                CP = cross = pw = ll = e_t = mdl = r2 = x = ctf = Ac = None
            return ch_, sw_, rp, rc

        parts_ = on_devices(clf, n, estep)
        changed = sum(p_[0] for p_ in parts_)
        switched = sum(p_[1] for p_ in parts_)
        res_pow = sum(p_[2].to(dev) for p_ in parts_)
        res_cnt = sum(p_[3].to(dev) for p_ in parts_)
        parts_ = None
        if marg:
            q_ = torch.quantile(marg_w[torch.randperm(n, device=dev)[:100000], 0], torch.tensor([0.1, 0.5, 0.9], device=dev))
            log(f"[3d] marginalisation: weight of the best of {marg} poses 10 / 50 / 90 % {float(q_[0]):.2f} / "
                f"{float(q_[1]):.2f} / {float(q_[2]):.2f}")
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
        if noise_model == "residual":       # "white": sigma2 stays 1 (diagnostic)
            sigma2 = torch.where(res_cnt > 0, res_pow / res_cnt.clamp_min(1), sigma2)
        if particle_scale:
            a_ = pscale_new.nan_to_num(1.0)
            pscale = (a_ / a_.median().clamp_min(1e-6)).clamp(0.2, 5.0)
            q_ = torch.quantile(pscale[torch.randperm(n, device=dev)[:100000]], torch.tensor([0.05, 0.5, 0.95], device=dev))
            if it == 0 or it == n_iter - 1:
                log(f"[3d] per-particle scale: 5 / 50 / 95 % {float(q_[0]):.2f} / {float(q_[1]):.2f} / {float(q_[2]):.2f}")
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
        # per particle, in its (new) model: the angle and shift this iteration moved it ("moved" only counts
        # particles whose best hypothesis is not the centre: after each halving of the step most find a neighbour
        # half a coarse step away, so it alternates ~75 % / ~25 % without saying how far)
        Q = A_prev[ar_n, z].transpose(1, 2) @ A[ar_n, z]
        d_ang = torch.rad2deg(torch.arccos(((Q.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
        d_sh = (t[ar_n, z] - t_prev[ar_n, z]).norm(dim=-1)
        del Q, A_prev, t_prev
        ch = dict(change_deg=round(float(d_ang.median()), 3), change_p90_deg=round(float(d_ang.quantile(0.9)), 3),
                  shift_change_a=round(float(d_sh.median()), 3), shift_change_p90_a=round(float(d_sh.quantile(0.9)), 3))
        log(f"[3d] particles it {it + 1}/{n_iter}: align {al_res:.1f} A, step {step:.2f} deg, "
            f"pose change median {ch['change_deg']:.2f} deg (90%: {ch['change_p90_deg']:.1f}), shift "
            f"{ch['shift_change_a']:.2f} A, {changed / n * 100:.0f}% off the grid centre" + (f", {switched / n * 100:.1f}% switched model, per model {pops}" if M > 1 else "")
            + (f", mean max posterior {float(post.max(1).values.mean()):.3f}" if soft and M > 1 else "")
            + f" ({since(t0):.1f}s)")
        history.append(dict(it=it + 1, align_res=round(al_res, 3), step=round(step, 3), moved=round(changed / n, 4),
                            switched=round(switched / n, 4), pops=pops, **ch, **({"nu": nu_it} if use_nu else {}),
                            **({"max_posterior": round(float(post.max(1).values.mean()), 4)} if M > 1 else {})))
        if iters is not None:
            iters["angle"].append(d_ang.cpu().numpy().astype(np.float16))
            iters["shift"].append(d_sh.cpu().numpy().astype(np.float16))
            iters["fit"].append(score.cpu().numpy().astype(np.float16))
        if step_scale is None and FAST_STEPS and M == 1 and not limited:
            # (not with search limits: a focused refinement's poses travel several steps at one step size - the
            # domain of tests/test_subtract.py stayed at the consensus resolution when the step halved every time)
            # once the poses settle (< 30 % off the grid centre), halve the step every iteration: after a halving most
            # particles move to a neighbour of the finer grid (where they already were, to within the old step), and a
            # second pass at the same step moved them ~0 (INX-6 C8: median 0.03 deg, 15-22 % moved, the resolution
            # unchanged; one pass of 10 was 48 s). At step_min, a pass with < 15 % moved ends the refinement, unless
            # a global / rescue / symmetry-relaxation iteration is still to come.
            at_min = step <= step_min * 1.001
            later = [a for a in tuple(global_at) + (tuple(rescue_at) if rescue_frac > 0 else ()) if a > it + 1]
            if at_min and settled and changed / n < 0.15 and not later and it + 1 >= relax_iters and it + 1 < n_iter:
                converged = True
                log(f"[3d] poses converged at the finest step ({step_min:g} deg): no more iterations needed "
                    f"({it + 1} of {n_iter})")
            settled = settled or changed / n < 0.3
            if settled:
                step = max(step_min, step / 2)
        elif step_scale is None and changed / n < 0.3:
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
        if fscs[m] is not None and not wiener:
            out.filter_shells(torch.sqrt((2 * fscs[m] / (1 + fscs[m])).clamp(0, 1).nan_to_num(0)))
        maps.append((out.real() * (mask if cur_masks[m] is None else cur_masks[m])).nan_to_num(0))
    fscs = [f if f is not None else torch.zeros(b // 2 + 1, device=dev) for f in fscs]
    unmasked = [resolution_from_fsc(fsc(vols[m][0], vols[m][1]), box_a) if alive[m] and fscs[m] is not None
                else float("nan") for m in range(M)]
    log(f"[3d] final resolution (FSC 0.143, {'soft sphere mask' if all(c is None for c in cur_masks) else 'solvent mask, corrected'}) {' / '.join(f'{r:.1f}' if r == r else 'merged' for r in res)} A"
        f" (unmasked {' / '.join(f'{r:.1f}' if r == r else '-' for r in unmasked)} A), {since(t0):.1f}s")
    refine_particles.last_weights = wpart.cpu()   # read by run(): which particles the final maps used
    refine_particles.last_score = score.cpu()
    refine_particles.last_scale = pscale.cpu()
    refine_particles.last_half = half.cpu()          # gold-standard half set of every particle (0 / 1)
    refine_particles.last_history = history
    refine_particles.last_iters = iters
    refine_particles.last_post = post.cpu()
    refine_particles.last_ll = llm_all.cpu() if n_iter > 0 else None   # per model, best pose, last pass
    refine_particles.last_mask = cur_masks[0]
    refine_particles.last_masks = cur_masks      # per model (None: the sphere)
    return torch.stack(maps), A[ar_n, z], t[ar_n, z], z, torch.stack(fscs), res, vols


def classify3d(clf, idx: torch.Tensor, A: torch.Tensor, t: torch.Tensor, n_models: int, res_a: float,
               sym: str = "C1", n_iter: int = 20, z0: torch.Tensor | None = None, seed: int = 0,
               expand_ops: torch.Tensor | None = None, box: int | None = None, log=print):
    """3D classification at fixed poses (from a consensus refinement), class3d.classify with the ab initio settings:
    M models seeded from random particle subsets (or the labels z0), soft assignment (posterior over models, no
    class priors), every model low-passed to the same resolution res_a.
    expand_ops: operators R of a (pseudo-)symmetry of the consensus map (S, 3, 3): each particle is also tried at
    A @ R (equally good for the consensus, not for a model that breaks the symmetry) and keeps its best per model.
    Returns (maps (M, b, b, b), z (n,), posteriors (n, M), per-model half-map resolution, half maps [[h1, h2] per
    model, each with .real()], poses (n, 3, 3))."""
    from .class3d import classify
    dev = clf.dev
    b = box or clf.bmax
    r = classify(clf, idx, A, t, n_models, res_a, sym=sym, n_iter=n_iter, z0=z0, seed=seed, init="random", reg_t=None,
                 soft=True, priors=False, ll_scale=0.5, relax_ops=expand_ops, final_box=b, log=log)
    classify3d.last_history = r.history          # per iteration: class sizes, half-map resolution (class3d.classify)
    zero = torch.zeros((b, b, b))
    maps = torch.stack([(m if m is not None else zero) for m in r.maps]).to(dev)
    halves = [[_RealMap(h[0] if h else zero, dev), _RealMap(h[1] if h else zero, dev)] for h in r.halves]
    return maps, r.z.to(dev), r.post.to(dev), r.res, halves, r.A.to(dev)


class _RealMap:
    """A half map where code expects a half Volume (only .real() is used)."""

    def __init__(self, v: torch.Tensor, dev):
        self.v, self.dev = v, dev

    def real(self) -> torch.Tensor:
        return self.v.to(self.dev)

    def free_acc(self):
        pass


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
    solvent_mask: str = "auto"      # particle refinement: dynamic solvent mask once better than 15 A ("auto", as Refine3D),
                                    # or the particle sphere only ("sphere")
    n_models: int = 1               # 3D classes: >1 adds a 3D classification of the particles at the consensus poses
    max_box: int = 0                # largest 3D map (px) - less GPU memory, faster; 0: the 2D run's working box
    seed_rounds: int = 2            # seeds "independent": rounds of (SGD per map on its particles, every particle
                                    # searched against every map). Simulated mixtures (box 64, 4000 particles, noise 3),
                                    # particles sorted right, seeds 0 / 1 / 2: 2 structures consensus 55 / 52 / 55 %,
                                    # independent 94 / 56 / 59 % (1 round: 81 % at seed 0); 3 structures consensus
                                    # 46 / 75 / 60 %, independent 72 / 58 / 61 % (3 rounds: 60 % at seed 0). Large
                                    # spread from the class stage's split of the classes: best-of-n attempts would help
    seeds: str = "consensus"        # ... n_models > 1: "consensus" (one map, then classified at its poses) or
                                    # "independent" (every model from its own share of the 2D classes, then the
                                    # particles refined against all of them: heterogeneous data)
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
    b = working_box(clf, n_acc=max(2, 2 * M), n_ft=max(2, M), log=log, max_box=prm.max_box)
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
        gpu.synchronize()
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
        gpu.synchronize()
        timing["sgd"] += time.time() - t0
        t0 = time.time()
        common = dict(sym=sym, n_local=prm.n_local, drop_frac=prm.drop_frac, rebalance=prm.rebalance, log=lg,
                      solvent_mask=None if prm.solvent_mask == "sphere" else prm.solvent_mask)
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
        gpu.synchronize()
        timing["particles"] += time.time() - t0
        return vols_cls, Ak, cc, out

    def independent_models(sym: str):
        """n_models > 1, seeds "independent": every model is built by the class stage from its own share of the 2D
        classes (a random balanced split, its own random orientations; the start whose models explain the classes
        best kept) - not one consensus map split afterwards, which starts every class from the same structure. Then
        the particles refine against all models, each weighted by its posterior over them (soft)."""
        t0 = time.time()
        vols0, Ak_m, cc_, zk_, _ = classes_to_volume(cls_ft[keep], pops[keep], box_a, diam, sym=sym,
                                                     n_starts=prm.n_starts, res_start=prm.class_res_start,
                                                     res_end=prm.class_res_end, n_iter=prm.class_iter, seed=prm.seed,
                                                     n_models=M, keep_top=1, log=log)
        gpu.synchronize()
        timing["classes"] += time.time() - t0
        clf._wait_loaded(clf.N)
        clf.ensure_polarity()
        t0 = time.time()
        A0 = torch.stack([particle_poses(Ak_m[m], kp, psi_p) for m in range(M)], 1)        # (n, M, 3, 3)
        z0 = zk_.to(kp.device)[kp]
        log(f"[3d] {M} independent starting maps from {int(keep.sum())} classes: particles per map "
            f"{torch.bincount(z0, minlength=M).tolist()}")
        # each map refined by particle SGD on its own particles (first those of its classes), then every particle
        # searched against every map and moved to the one it fits best; twice, so the maps and the split improve
        # together (the class stage alone left the maps at ~25 A and the refinement never got going: 2 different
        # structures 72 % sorted right, 3 of them 43 %)
        Vm = list(vols0)
        g_res = max(prm.global_res, prm.sgd_res)
        step = global_step(g_res, diam)
        zero = torch.zeros(len(idx), dtype=torch.long)
        for rnd in range(prm.seed_rounds):
            t0 = time.time()
            for m in range(M):
                sel = torch.nonzero(z0 == m).squeeze(1)
                if prm.sgd_iter > 0 and len(sel) >= 200:
                    Vm[m], _ = particle_sgd(clf, idx[sel], off_p[sel], Vm[m], sym=sym,
                                            n_iter=prm.sgd_iter if rnd == 0 else max(4, prm.sgd_iter // 2),
                                            res_end=prm.sgd_res, batch=min(prm.sgd_batch, max(500, len(sel) // 4)),
                                            seed=prm.seed + 7 * rnd + m, topk=prm.sgd_topk, log=log)
            timing["sgd"] += time.time() - t0
            t0 = time.time()
            lls = []
            for m in range(M):
                Ag, llg, _ = global_search(clf, idx, A0[:, m], off_p, [Vm[m], Vm[m]], zero, g_res, step, box=Vm[m].b,
                                           log=lambda *a: None)
                A0[:, m] = Ag.to(A0.device)
                lls.append(llg.to(A0.device))
            z_new = torch.stack(lls, 1).argmax(1).to(z0.device)
            moved = float((z_new != z0).float().mean())
            z0 = z_new
            timing["particles"] += time.time() - t0
            log(f"[3d] round {rnd + 1}: every particle against every map: per map {torch.bincount(z0, minlength=M).tolist()}, "
                f"{moved * 100:.0f}% changed map")
        t0 = time.time()
        out = refine_particles(clf, idx, A0, off_p, Vm, z=z0, sym=sym, n_iter=prm.n_iter, res_start=prm.res_start,
                               n_local=prm.n_local, box=b, soft=True, drop_frac=prm.drop_frac, rebalance=prm.rebalance,
                               solvent_mask=None if prm.solvent_mask == "sphere" else prm.solvent_mask, log=log)
        gpu.synchronize()
        timing["particles"] += time.time() - t0
        return vols0, _class_poses(out[1], kp, psi_p, int(keep.sum())), cc_, zk_, out

    relaxed_from = prm.relax_from.upper() if prm.relax_from and prm.sym.upper() == "C1" else None
    if relaxed_from == "C1":
        relaxed_from = None
    independent = M > 1 and prm.seeds == "independent" and not relaxed_from
    zk_ind = None
    if independent:
        vols_cls, Ak, cc, zk_ind, (maps, A, t, z, fs, res, vols) = independent_models(prm.sym)
    else:
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
            relax_iters=prm.relax_iters, solvent_mask=None if prm.solvent_mask == "sphere" else prm.solvent_mask,
            log=log)
        gpu.synchronize()
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
    zk = zk_ind.to(dev) if zk_ind is not None else torch.zeros(len(Ak), dtype=torch.long, device=dev)
    t0 = time.time()
    consensus = None
    if M > 1 and not independent:   # the consensus's own numbers, before its halves make way for the classes'
        dm0 = getattr(refine_particles, "last_masks", None)
        cmask0 = dm0[0] if dm0 is not None and dm0[0] is not None else solvent_mask(consensus_map, box_a / b)
        consensus = dict(resolution_a=round(res[0], 2), resolution_unmasked_a=round(unmasked_res, 2),
                         resolution_solvent_mask_a=round(resolution_from_fsc(corrected_masked_fsc(
                             vols[0][0].real(), vols[0][1].real(), cmask0), box_a), 2),
                         particles_used=int(used.sum()))
    if M > 1 and not independent:   # heterogeneity: 3D classification at the consensus poses, all classes alike
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
    gpu.synchronize()
    timing["particles"] += time.time() - t0
    t1, t2 = timing["classes"], timing["sgd"] + timing["particles"]
    apix = box_a / b
    pops_m = torch.bincount(z, minlength=M).tolist()
    order = sorted(range(M), key=lambda m: -pops_m[m])        # model 1 = most particles
    solvent, res_solv, fsc_solv = [], {}, {}
    for r, m in enumerate(order):
        sfx = "" if M == 1 else f"_{r + 1}"
        dm = getattr(refine_particles, "last_masks", None)
        smask = dm[m] if dm is not None and len(dm) == M and dm[m] is not None else solvent_mask(maps[m], apix)
        solvent.append(smask)
        if M > 1 and pops_m[m]:           # each class's corrected solvent-masked FSC (the table of the classes)
            fsc_solv[m] = corrected_masked_fsc(vols[m][0].real(), vols[m][1].real(), smask).cpu().numpy()
            res_solv[m] = resolution_from_fsc(torch.as_tensor(fsc_solv[m]), box_a)
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
    if M > 1:   # per class: solvent mask, corrected (an empty class: the sphere)
        curves = [(f"3D class {r + 1}", fsc_solv[m] if m in fsc_solv else fs[m].cpu().numpy())
                  for r, m in enumerate(order)]
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
    radius = diagnostics.particle_radius(solvent[0], apix, box_a)       # angular distribution for ChimeraX
    diagnostics.write_angdist_bild(os.path.join(outdir, "angdist.bild"), rot.numpy(), tilt.numpy(), box_a, radius)
    if M > 1:
        mdl_all = rank[z.cpu()].numpy()
        for r_ in range(M):
            sel = mdl_all == r_
            diagnostics.write_angdist_bild(os.path.join(outdir, f"angdist_{r_ + 1}.bild"), rot.numpy()[sel],
                                           tilt.numpy()[sel], box_a, radius)
    fsc_data = dict(corrected=fsc_solvent.cpu().numpy(), unmasked=fsc_unmasked.cpu().numpy(),
                    sphere_masked=fs[order[0]].cpu().numpy())
    for r_, m in enumerate(order if M > 1 else []):
        fsc_data[f"class{r_ + 1}_sphere_masked"] = fs[m].cpu().numpy()
        if m in fsc_solv:
            fsc_data[f"class{r_ + 1}"] = fsc_solv[m]
    diagnostics.write_fsc(os.path.join(outdir, "fsc"), fsc_data, box_a,
                          resolution_a=round(resolution_from_fsc(fsc_solvent, box_a), 3), title=tag)
    reproj_cc = _figures(outdir, clf, result, keep, Ak, [maps[m] for m in order], curves, rot, tilt, box_a, b, tag,
                         vstats)
    if consensus is not None:   # the consensus before the classification: unmasked (consensus.mrc), masked, its mask
        _save_mrc(os.path.join(outdir, "consensus.mrc"), consensus_map, apix)
        _save_mrc(os.path.join(outdir, "consensus_map.mrc"), consensus_map * cmask0, apix)
        _save_mrc(os.path.join(outdir, "consensus_mask.mrc"), cmask0, apix)
    from dataclasses import asdict
    summary = dict(params=asdict(prm), particles=int(len(idx)), classes=int(keep.sum()), seconds_classes=round(t1, 1),
                   seconds_particles=round(t2, 1), resolution_a=round(res[order[0]], 2), apix=round(apix, 3), box=b,
                   resolution_unmasked_a=round(unmasked_res if M == 1 else resolution_from_fsc(fsc_unmasked, box_a), 2),
                   fsc_mask="soft sphere of the particle diameter",
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
                                resolution_a=round(res[m], 2) if res[m] == res[m] else None,
                                **({"resolution_solvent_mask_a": round(res_solv[m], 2) if m in res_solv else None}
                                   if M > 1 else {}))
                           for r, m in enumerate(order)])
    if M > 1 and not independent:   # model 1 = most particles: the classification's history in that numbering
        summary["consensus"] = consensus
        summary["class_history"] = [dict(h, **{k: [h[k][m] for m in order] for k in ("pops", "weights", "res")
                                               if k in h}) for h in getattr(classify3d, "last_history", [])]
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

