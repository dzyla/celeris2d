"""Movie preprocessing on the GPU: gain reference, whole-frame motion correction, dose weighting, CTF estimation.

First pass ("easy pass"): one rigid shift per frame (no local / patch motion), per-micrograph CTF (defocus U/V,
astigmatism angle). Conventions follow the rest of Celeris: rfft2 spectra, RELION CTF convention (engine.ctf_2d:
defocus U along the astigmatism angle, measured from the x axis), Angstrom everywhere.

    mov = read_movie(path)                                        # (frames, ny, nx) uint8/uint16/float32
    gain = orient_gain(read_gain(gain_path), mov)                  # the orientation that removes the fixed pattern
    mic = motion_correct(mov, gain, apix=0.6575, dose_per_frame=1.39, bin=2)
    ctf = estimate_ctf(mic.sum, mic.apix, Optics(300, 2.7, 0.1))
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import gpu
from .engine import Optics, ctf_2d, freqs


# ------------------------------------------------------------------------------------------------- input
def read_movie(path: str) -> np.ndarray:
    """Frames of a movie (.tif/.tiff, .mrc/.mrcs) as (n, ny, nx), in their stored dtype."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(path, maxworkers=16)
    import mrcfile
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data)


def read_gain(path: str) -> np.ndarray:
    """Gain reference (multiplied into the frames), float32 (ny, nx)."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        g = tifffile.imread(path)
    else:
        import mrcfile
        with mrcfile.open(path, permissive=True) as m:
            g = np.asarray(m.data)
    return np.asarray(g, dtype=np.float32).squeeze()


GAIN_ORIENT = {"as is": lambda g: g, "flip y": lambda g: g[::-1], "flip x": lambda g: g[:, ::-1],
               "rotate 180": lambda g: g[::-1, ::-1], "transpose": lambda g: g.T, "rotate 90": lambda g: np.rot90(g, 1),
               "rotate 270": lambda g: np.rot90(g, 3), "anti-transpose": lambda g: np.rot90(g, 2).T}


def orient_gain(gain: np.ndarray, frames: np.ndarray, log=print, return_name: bool = False):
    """The orientation of the gain reference (of 8 flips / rotations with the frame's shape) that makes the frame
    sum flattest: a mis-oriented gain imprints its fixed pattern (pixel-to-pixel roughness) instead of removing it.
    EMPIAR-10025 (K2 super-resolution TIFF + norm-amibox05-0.mrc): "as is"."""
    dev = gpu.default()
    s = torch.as_tensor(np.asarray(frames.sum(0), dtype=np.float32), device=dev)
    best = None
    for name, f in GAIN_ORIENT.items():
        g = np.array(f(gain), dtype=np.float32)                       # a writable, contiguous copy
        if g.shape != s.shape:
            continue
        x = s * torch.as_tensor(g, device=dev)
        x = x[16:-16, 16:-16]
        rough = float(((x[1:, :] - x[:-1, :]).pow(2).mean() + (x[:, 1:] - x[:, :-1]).pow(2).mean()) / x.mean() ** 2)
        if best is None or rough < best[0]:
            best = (rough, name, g)
    log(f"[gain] orientation: {best[1]}")
    return (best[2], best[1]) if return_name else best[2]


# --------------------------------------------------------------------------------------- Fourier helpers
def crop_half(ft: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
    """Crop rfft2 spectra (..., NY, NX//2+1) to (..., ny, nx//2+1): exact Fourier downsampling (norm='forward')."""
    h = ny // 2
    out = torch.cat([ft[..., :h, : nx // 2 + 1], ft[..., -h:, : nx // 2 + 1]], -2)
    return out


def _grid(ny: int, nx: int, apix: float, dev):
    """Frequencies (1/A) of an rfft2 grid: ky (ny, 1), kx (1, nx//2+1)."""
    ky = torch.fft.fftfreq(ny, d=apix, device=dev)[:, None]
    kx = torch.fft.rfftfreq(nx, d=apix, device=dev)[None, :]
    return ky, kx


def _shift_phase(ky, kx, sy, sx):
    """Phase ramp that moves an image by (+sx, +sy) A: (n, ky, kx)."""
    return torch.exp(-2j * math.pi * (ky[None] * sy[:, None, None] + kx[None] * sx[:, None, None]))


# ----------------------------------------------------------------------------------------- motion
@dataclass
class Micrograph:
    sum: torch.Tensor           # aligned sum, no dose weighting (for CTF estimation), (ny, nx), A/px = apix
    sum_dw: torch.Tensor        # aligned, dose-weighted sum (for particles)
    apix: float
    shifts: np.ndarray          # (frames, 2) shift (x, y) in A applied to each frame
    halves: tuple = ()          # dose-weighted sums of the even and of the odd frames: their FRC measures the
                                # aligned signal with independent noise (an alignment quality check)
    local: dict | None = None   # local motion: fit rms, patch centres, measured and modelled patch shifts (A)
    dose_per_frame: float = 0.0
    frames_ft: torch.Tensor | None = None   # keep_frames: gain-corrected, binned frame spectra (n, ny, nx//2+1)
    model: "MotionModel | None" = None      # local motion model (frame shift at a position = shifts + model)


def dose_weights(n: int, dose_per_frame: float, k: torch.Tensor, pre_dose: float = 0.0,
                 kv: float = 300.0, frames: range | None = None) -> torch.Tensor:
    """Grant & Grigorieff (2015) exposure filter per frame, (n, *k.shape), scaled so that sum_i w_i^2 = n at
    every frequency (the dose-weighted sum keeps the noise level of a plain sum). The critical exposure was measured
    at 300 kV; at 200 kV it is 0.8 x that (as in MotionCor2 and RELION). frames: only these frames (len(frames),
    *k.shape), normalised over all n (for full-size micrographs, where all n at once is half a frame stack)."""
    ne = 0.245 * k.clamp_min(1e-4).pow(-1.665) + 2.81                 # critical exposure (e/A^2) at 300 kV
    if kv < 250:
        ne = 0.8 * ne
    dose = pre_dose + (torch.arange(n, device=k.device, dtype=torch.float32) + 0.5) * dose_per_frame
    if frames is None:
        q = torch.exp(-dose.view(-1, *[1] * k.dim()) / (2 * ne[None]))
        return q * torch.sqrt(n / q.pow(2).sum(0, keepdim=True))
    q2 = torch.zeros_like(ne)
    for d in dose:                                                   # sum over all frames of q^2, frame by frame
        q2 += torch.exp(-d / ne)
    sel = dose[list(frames)]
    return torch.exp(-sel.view(-1, *[1] * k.dim()) / (2 * ne[None])) * torch.sqrt(n / q2)[None]


def _peak(cc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-pixel peak (dy, dx) of fftshifted correlation maps (n, h, w), relative to the centre (parabola fit)."""
    n, h, w = cc.shape
    i = cc.reshape(n, -1).argmax(1)
    y, x = i // w, i % w
    y, x = y.clamp(1, h - 2), x.clamp(1, w - 2)
    ar = torch.arange(n, device=cc.device)
    c0 = cc[ar, y, x]

    def para(m, p):
        d = m - 2 * c0 + p
        return torch.where(d.abs() > 1e-12, 0.5 * (m - p) / d, torch.zeros_like(d)).clamp(-1, 1)
    oy = para(cc[ar, y - 1, x], cc[ar, y + 1, x])
    ox = para(cc[ar, y, x - 1], cc[ar, y, x + 1])
    return (y.float() + oy - h // 2), (x.float() + ox - w // 2)


def align_frames(F: torch.Tensor, apix: float, bfactor: float = 1500.0, n_iter: int = 10,
                 max_shift_a: float = 150.0, chunk: int = 8) -> torch.Tensor:
    """Rigid shift (x, y) in A per frame that aligns frame spectra F (n, ny, nx//2+1) to the sum of the other
    frames (leave-one-out, so a frame's own noise does not pull it to zero shift). Correlations are B-factor
    filtered (low-pass) and searched within max_shift_a of the current shift. F may carry a batch dimension
    (B, n, ny, nx//2+1), e.g. all patches of a micrograph at once: shifts (B, n, 2), one GPU pass for all."""
    single = F.dim() == 3
    if single:
        F = F[None]
    B, n, ny, nh = F.shape
    nx = 2 * (nh - 1)
    ky, kx = _grid(ny, nx, apix, F.device)
    filt = torch.exp(-bfactor * (ky ** 2 + kx ** 2) / 4)
    sh = torch.zeros(B, n, 2, device=F.device)                           # (x, y) A
    rad = int(min(ny, nx) // 2 - 2)
    lim = min(rad, int(max_shift_a / apix) + 2)
    for _ in range(n_iter):
        Fs = F * _shift_phase(ky, kx, sh[..., 1].reshape(-1), sh[..., 0].reshape(-1)).view(B, n, ny, nh)
        tot = Fs.sum(1, keepdim=True)
        upd = torch.zeros_like(sh)
        for i in range(0, n, chunk):
            fi = Fs[:, i:i + chunk]
            cc = torch.fft.irfft2((tot - fi) * fi.conj() * filt, s=(ny, nx))
            cc = torch.fft.fftshift(cc, dim=(-2, -1))
            c = cc[..., ny // 2 - lim: ny // 2 + lim + 1, nx // 2 - lim: nx // 2 + lim + 1]
            k = c.shape[1]
            dy, dx = _peak(c.reshape(B * k, *c.shape[-2:]))
            upd[:, i:i + k, 0], upd[:, i:i + k, 1] = (dx * apix).view(B, k), (dy * apix).view(B, k)
        sh = sh + upd
        sh = sh - sh.mean(1, keepdim=True)
        if float(upd.abs().max()) < 0.05 * apix:
            break
    return sh[0] if single else sh


class MotionModel:
    """Smooth local motion (MotionCor2-like): the shift of the frame at time t (0..1) at position (x, y) (-1..1
    over the micrograph) is sum_ij c_ij * s_i(x, y) * t^j with s = 1, x, y, x^2, xy, y^2 and j = 0..3, per axis,
    fitted by least squares (two rounds, outliers beyond 3 MAD dropped) to shifts measured on patches."""
    SPACE = 6
    TIME = 4

    def __init__(self, xy: torch.Tensor, t: torch.Tensor, r: torch.Tensor):
        """xy (P, 2) patch centres in -1..1, t (n,) frame times in 0..1, r (P, n, 2) measured shifts (A)."""
        X = self._basis(xy, t).reshape(-1, self.SPACE * self.TIME)                   # (P*n, 24)
        y = r.reshape(-1, 2)
        keep = torch.ones(len(y), dtype=torch.bool, device=y.device)
        for _ in range(2):
            self.c = torch.linalg.lstsq(X[keep], y[keep]).solution                   # (24, 2)
            res = (X @ self.c - y).norm(dim=1)
            mad = res[keep].median() * 1.4826 + 1e-6
            keep = res < 3 * mad + 0.5
        self.rms = float((X[keep] @ self.c - y[keep]).pow(2).sum(1).mean().sqrt())
        self.n_used = int(keep.sum())

    def _basis(self, xy, t):
        x, y = xy[:, 0], xy[:, 1]
        sp = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], 1)          # (P, 6)
        tm = torch.stack([t ** j for j in range(self.TIME)], 1)                       # (n, 4)
        return sp[:, None, :, None] * tm[None, :, None, :]                            # (P, n, 6, 4)

    def __call__(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Shifts (P, n, 2) at positions xy (P, 2) and times t (n,)."""
        return self._basis(xy, t).reshape(len(xy), len(t), -1) @ self.c


def patch_shifts(R, apix: float, grid: int = 5, align_apix: float = 2.6, bfactor: float = 500.0, shape=None):
    """Residual shifts of globally aligned real-space frames on a grid x grid set of patches: (centres (P, 2) in
    -1..1 as (x, y), shifts (P, n, 2) in A). All patches are aligned in one batch. R: the frames (n, ny, nx), or a
    function i -> frame i (ny, nx) with shape = (n, ny, nx): frames made one at a time, so the aligned stack never
    has to be held (only the patches' cropped spectra are kept)."""
    n, ny, nx = R.shape if shape is None else shape
    frame = (lambda i: R[i]) if shape is None else R
    py, px = (ny // grid) // 2 * 2, (nx // grid) // 2 * 2
    na = max(64, int(round(py * apix / align_apix / 2)) * 2)
    ma = max(64, int(round(px * apix / align_apix / 2)) * 2)
    cen = [((gx * px + px / 2) / nx * 2 - 1, (gy * py + py / 2) / ny * 2 - 1) for gy in range(grid) for gx in range(grid)]
    Fs = None
    for i in range(n):
        r = frame(i)
        P = torch.stack([r[gy * py:(gy + 1) * py, gx * px:(gx + 1) * px] for gy in range(grid) for gx in range(grid)])
        f = crop_half(torch.fft.rfft2(P - P.mean((1, 2), keepdim=True), norm="forward"), na, ma)    # (P, na, ma/2+1)
        if Fs is None:
            Fs = torch.empty((len(P), n) + f.shape[1:], dtype=f.dtype, device=f.device)
        Fs[:, i] = f
    out = align_frames(Fs, apix * py / na, bfactor=bfactor, max_shift_a=40.0)
    return torch.tensor(cen, device=Fs.device), out


def _frames(F):
    """The unshifted frames (n, ny, nx) from their spectra F, one frame at a time (all at once, and reflect-padded as a
    second copy, held the spectra, the frames, the padded copy and the FFT workspace together: 5.7 GB for a 1.4 GB
    frame stack)."""
    n, ny, nh = F.shape
    nx = 2 * (nh - 1)
    R = torch.empty((n, ny, nx), device=F.device)
    for i in range(n):
        R[i] = torch.fft.irfft2(F[i], s=(ny, nx), norm="forward")
    return R


def _reflect_index(n: int, before: int, after: int, dev) -> torch.Tensor:
    """Source index of every position of a reflect-padded axis (as torch's 'reflect' padding: the edge not repeated)."""
    i = torch.arange(-before, n + after, device=dev)
    i = torch.where(i < 0, -i, i)
    return torch.where(i > n - 1, 2 * (n - 1) - i, i)


def _local_sums(R, sh_g, model, w, apix, tile=512, pre_dose=0.0, kv=300.0):
    """Aligned sums with local motion: overlapping tiles (half-tile steps, sin^2 weights that add up to 1), each
    frame of a tile shifted in Fourier space by the global shift plus the model at the tile centre. R: the unshifted
    frames (n, ny, nx); tiles reach past the edges into a reflected copy, read through an index map (no padded copy
    of the stack). Returns the plain sum, the dose-weighted sum and the dose-weighted sums of the even and the odd
    frames (ny, nx)."""
    n, ny, nx = R.shape
    dev = R.device
    st = tile // 2
    pad_y, pad_x = st + (-ny) % st, st + (-nx) % st
    iy, ix = _reflect_index(ny, st, pad_y, dev), _reflect_index(nx, st, pad_x, dev)
    Ny, Nx = len(iy), len(ix)
    out = torch.zeros(4, Ny, Nx, device=dev)                                       # plain, dw, dw even, dw odd
    r = torch.arange(tile, device=dev, dtype=torch.float32)
    win1 = torch.sin(math.pi * (r + 0.5) / tile) ** 2
    win = win1[:, None] * win1[None, :]
    ky, kx = _grid(tile, tile, apix, dev)
    wt = dose_weights(n, w, torch.sqrt(ky ** 2 + kx ** 2), pre_dose, kv)            # w = dose per frame here
    t = torch.linspace(0, 1, n, device=dev)
    ys, xs = list(range(0, Ny - tile + 1, st)), list(range(0, Nx - tile + 1, st))
    for y0 in ys:
        for x0 in xs:
            cx = ((x0 - st + tile / 2) / nx) * 2 - 1
            cy = ((y0 - st + tile / 2) / ny) * 2 - 1
            shf = sh_g + model(torch.tensor([[cx, cy]], device=dev), t)[0]           # (n, 2)
            Rt = R[:, iy[y0:y0 + tile]][:, :, ix[x0:x0 + tile]]                       # (n, tile, tile)
            Ft = torch.fft.rfft2(Rt) * _shift_phase(ky, kx, shf[:, 1], shf[:, 0])
            Fw = Ft * wt
            acc = torch.stack([Ft.sum(0), Fw.sum(0), Fw[0::2].sum(0), Fw[1::2].sum(0)])
            out[:, y0:y0 + tile, x0:x0 + tile] += torch.fft.irfft2(acc, s=(tile, tile)) * win
    return out[:, st:st + ny, st:st + nx]


def motion_correct(frames: np.ndarray, gain: np.ndarray | None, apix: float, dose_per_frame: float, bin: float = 1,
                   align_apix: float = 4.0, pre_dose: float = 0.0, patches: int = 5, device=None,
                   keep_frames: bool = False, kv: float = 300.0) -> Micrograph:
    """Gain, Fourier binning by `bin` (any factor >= 1: 2 for super-resolution -> physical pixels, more to save
    space and time; the pixel size becomes apix * NY / ny, x within one pixel of the same), whole-frame alignment on a copy
    binned to ~align_apix, then the aligned sums (plain and dose-weighted) at apix * bin. patches > 0: local
    motion measured on patches x patches patches of the globally aligned frames, a smooth model fitted to it
    (MotionModel), and the sums built from overlapping tiles that each follow the model. EMPIAR-10025 (20 movies,
    J104's 10k particles at its refined poses, D7): whole-frame 3.47 A, patches=5 3.27 A, cryoSPARC's patch-motion
    micrographs 3.43 A (ours with each micrograph's particle positions registered to cryoSPARC's by a smooth field:
    the two local-motion models place particles ~1 A apart)."""
    dev = device or gpu.default()
    if frames.ndim == 2:
        frames = frames[None]
    n, NY, NX = frames.shape
    ny, nx = int(round(NY / bin / 2)) * 2, int(round(NX / bin / 2)) * 2
    g = None if gain is None else torch.as_tensor(np.ascontiguousarray(gain), device=dev)
    # hot pixels: far more counts over the movie than the Poisson spread allows -> the frame mean
    tot = torch.zeros(NY, NX, device=dev)
    for i in range(n):
        tot += torch.as_tensor(frames[i], device=dev).float()
    # robust statistics: a few absurd values (EMPIAR-10291 micrograph 42 holds 1.6e34) wreck a mean; a counting
    # movie's hot pixels are far above the Poisson spread, a float micrograph's above the median's spread
    fin = torch.isfinite(tot)
    smp = tot[fin][torch.randint(0, int(fin.sum()), (min(int(fin.sum()), 2_000_000),), device=dev)]
    mu = float(smp.median())
    sd = float(1.4826 * (smp - mu).abs().median())
    if np.issubdtype(np.asarray(frames[:1]).dtype, np.integer):          # counts: far above the Poisson spread
        hot = (~fin) | (tot > mu + 10 * math.sqrt(max(mu, 1.0)) + 10)
    else:                                                                # a float image: far from the median
        hot = (~fin) | ((tot - mu).abs() > 20 * max(sd, 1e-12))
    del tot, smp
    if n == 1:      # a micrograph (aligned and summed elsewhere, e.g. a public set): gain and Fourier binning only
        x = torch.as_tensor(np.asarray(frames[0]), device=dev).float()
        if g is not None:
            x = x * g
        x = torch.where(hot, x[~hot].mean(), x)          # outliers (corrupt values) -> the mean of the good pixels
        m = torch.fft.irfft2(crop_half(torch.fft.rfft2(x, norm="forward"), ny, nx), s=(ny, nx), norm="forward")
        return Micrograph(sum=m, sum_dw=m, apix=apix * NY / ny, shifts=np.zeros((1, 2), np.float32),
                          dose_per_frame=dose_per_frame)
    F = torch.empty((n, ny, nx // 2 + 1), dtype=torch.complex64, device=dev)
    for i in range(n):
        x = torch.as_tensor(frames[i], device=dev).float()
        if g is not None:
            x = x * g
        x = torch.where(hot, x[~hot].mean(), x)          # the mean of the good pixels
        F[i] = crop_half(torch.fft.rfft2(x, norm="forward"), ny, nx)
    p = apix * NY / ny
    na = max(64, int(round(ny * p / align_apix / 2)) * 2)
    ma = max(64, int(round(nx * p / align_apix / 2)) * 2)
    Fa = crop_half(F, na, ma)
    sh = align_frames(Fa, p * ny / na)
    ky, kx = _grid(ny, nx, p, dev)
    if patches:
        # the globally aligned frames, made one at a time for the patch measurement (holding them all next to F
        # took another stack's worth of memory)
        def aligned(i):
            return torch.fft.irfft2(F[i] * _shift_phase(ky, kx, sh[i:i + 1, 1], sh[i:i + 1, 0])[0], s=(ny, nx),
                                    norm="forward")
        cen, rs = patch_shifts(aligned, p, grid=patches, shape=(n, ny, nx))
        model = MotionModel(cen, torch.linspace(0, 1, n, device=dev), rs)
        R = _frames(F)
        if not keep_frames:
            del F, Fa                                  # the spectra are not needed again
            F = None
        out = _local_sums(R, sh, model, dose_per_frame, p, pre_dose=pre_dose, kv=kv)
        del R
        return Micrograph(sum=out[0], sum_dw=out[1], apix=p, shifts=sh.cpu().numpy(), halves=(out[2], out[3]),
                          dose_per_frame=dose_per_frame, frames_ft=F if keep_frames else None, model=model,
                          local=dict(rms_a=model.rms, used=model.n_used, measured=rs.cpu().numpy(),
                                     centres=cen.cpu().numpy(),
                                     model=model(cen, torch.linspace(0, 1, n, device=dev)).cpu().numpy()))
    kk = torch.sqrt(ky ** 2 + kx ** 2)
    S = torch.zeros((ny, nx // 2 + 1), dtype=torch.complex64, device=dev)
    Sw, He, Ho = torch.zeros_like(S), torch.zeros_like(S), torch.zeros_like(S)
    for i in range(0, n, 2):                                         # i is even: row 0 of a pair is an even frame
        Fi = F[i:i + 2] * _shift_phase(ky, kx, sh[i:i + 2, 1], sh[i:i + 2, 0])
        Fw = Fi * dose_weights(n, dose_per_frame, kk, pre_dose, kv, frames=range(i, min(i + 2, n)))
        S += Fi.sum(0)
        Sw += Fw.sum(0)
        He += Fw[0::2].sum(0)
        Ho += Fw[1::2].sum(0)
    inv = lambda X: torch.fft.irfft2(X, s=(ny, nx), norm="forward")
    return Micrograph(sum=inv(S), sum_dw=inv(Sw), apix=p, shifts=sh.cpu().numpy(), halves=(inv(He), inv(Ho)),
                      dose_per_frame=dose_per_frame, frames_ft=F if keep_frames else None)


def frc(a: torch.Tensor, b: torch.Tensor, apix: float, n_shell: int = 40):
    """Fourier ring correlation of two images: (shell centres in 1/A, FRC) as numpy arrays."""
    fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
    ky, kx = _grid(a.shape[0], a.shape[1], apix, a.device)
    sh = (torch.sqrt(ky ** 2 + kx ** 2) / (0.5 / apix) * n_shell).long().clamp(max=n_shell)
    m = sh < n_shell
    s = sh[m]
    z = lambda v: torch.zeros(n_shell, device=a.device).index_add_(0, s, v[m])
    num, da, db = z((fa * fb.conj()).real), z(fa.abs().pow(2)), z(fb.abs().pow(2))
    ks = (torch.arange(n_shell, device=a.device) + 0.5) / n_shell * 0.5 / apix
    return ks.cpu().numpy(), (num / (da * db).sqrt().clamp_min(1e-12)).cpu().numpy()


# -------------------------------------------------------------------------------------------- CTF
@dataclass
class CTFFit:
    defocus_u: float            # A (RELION: U along the angle, V perpendicular)
    defocus_v: float
    angle: float                # deg from the x axis
    fit_res_a: float            # resolution to which the fitted CTF agrees with the Thon rings (A)
    score: float
    profile: dict               # for figures (plots_preprocess): ring correlation, 2D and equal-phase 1D fit
    patch: dict | None = None   # local defocus (patch_defocus): plane, its errors, tilt, whether it is applied


def power_spectrum(mic: torch.Tensor, apix: float, tile: int = 512) -> torch.Tensor:
    """Mean |FT|^2 of half-overlapping, mean-subtracted, edge-tapered tiles: (tile, tile//2+1)."""
    ny, nx = mic.shape
    st = tile // 2
    t = mic.unfold(0, tile, st).unfold(1, tile, st).reshape(-1, tile, tile)
    t = t - t.mean((1, 2), keepdim=True)
    r = torch.arange(tile, device=mic.device, dtype=torch.float32)
    edge = torch.clamp(torch.minimum(r, tile - 1 - r) / (tile * 0.05), max=1.0)
    win = (torch.sin(0.5 * math.pi * edge) ** 2)
    t = t * win[None, :, None] * win[None, None, :]
    ps = torch.zeros(tile, tile // 2 + 1, device=mic.device)
    for i in range(0, len(t), 64):
        ps += torch.fft.rfft2(t[i:i + 64]).abs().pow(2).sum(0)
    return ps / len(t)


def _ctf2(du, dv, ang, optics, fy, fx, box_a):
    """CTF^2 on the half-plane grid for parameter vectors du, dv (A), ang (deg): (m, ny, nx)."""
    m = len(du)
    p = torch.zeros(m, 6, device=fy.device)
    p[:, 0], p[:, 1], p[:, 2], p[:, 5] = du, dv, ang, 1.0
    return ctf_2d(p, optics, fy, fx, box_a).pow(2)


def _basis(kn: torch.Tensor, deg: int) -> torch.Tensor:
    """Polynomial basis of normalised frequency (..., deg+1)."""
    return torch.stack([kn ** j for j in range(deg + 1)], -1)


def _fit_score(y, c2, B):
    """y (P,) data, c2 (m, P) candidate CTF^2, B (P, d) smooth basis. Fits y ~ B a + (B e) * c2 by least squares
    (background and envelope smooth in k) and returns the explained fraction of variance (m,)."""
    X = torch.cat([B[None].expand(len(c2), -1, -1), B[None] * c2[..., None]], -1)       # (m, P, 2d)
    sol = torch.linalg.lstsq(X, y[None, :, None].expand(len(c2), -1, -1)).solution
    res = y[None] - (X @ sol)[..., 0]
    return 1 - res.pow(2).mean(1) / y.var()


def estimate_ctf(mic: torch.Tensor, apix: float, optics: Optics, res_lo: float = 30.0, res_hi: float = 5.0,
                 df_range=(1000.0, 50000.0), tile: int = 512, deg: int = 4) -> CTFFit:
    """Per-micrograph defocus and astigmatism. The tile power spectrum (sqrt, so Thon-ring amplitude does not
    vanish under the background) is modelled as background(k) + envelope(k) * CTF^2(k, theta) with background and
    envelope low-order polynomials in |k| solved by least squares for every candidate CTF (variable projection);
    the CTF maximising the explained variance between res_lo and res_hi wins: a 1D defocus scan on circular rings,
    then astigmatism (magnitude x angle grid), then a local grid refinement."""
    dev = mic.device
    ps = power_spectrum(mic, apix, tile).sqrt()
    fy, fx = freqs(tile, dev)
    box_a = tile * apix
    k = torch.sqrt((fy[:, None] / box_a) ** 2 + (fx[None, :] / box_a) ** 2)
    band = (k >= 1 / res_lo) & (k <= 1 / res_hi)
    y = ps[band]
    y = (y - y.mean()) / y.std()
    kn = (k[band] - 1 / res_lo) / (1 / res_hi - 1 / res_lo)
    B = _basis(kn, deg)

    def score(du, dv, ang, chunk=24):
        out = []
        for i in range(0, len(du), chunk):
            c2 = _ctf2(du[i:i + chunk], dv[i:i + chunk], ang[i:i + chunk], optics, fy, fx, box_a)[:, band]
            out.append(_fit_score(y, c2, B))
        return torch.cat(out)

    # 1) mean defocus, no astigmatism (log-spaced scan)
    d = torch.exp(torch.linspace(math.log(df_range[0]), math.log(df_range[1]), 400, device=dev))
    s = score(d, d, torch.zeros_like(d))
    d0 = float(d[s.argmax()])
    # 2) astigmatism: magnitude x angle around d0
    best = (float(s.max()), d0, d0, 0.0)
    for span, step_a in ((0.08, 15.0), (0.02, 5.0)):
        mean = 0.5 * (best[1] + best[2])
        means = torch.linspace(mean * (1 - span), mean * (1 + span), 9, device=dev)
        astig = torch.linspace(0, max(4 * span * mean, 200.0), 9, device=dev)
        angs = torch.arange(0, 180, step_a, device=dev)
        M, Aa, G = torch.meshgrid(means, astig, angs, indexing="ij")
        du, dv, an = (M + Aa / 2).flatten(), (M - Aa / 2).flatten(), G.flatten()
        sc = score(du, dv, an)
        j = int(sc.argmax())
        if float(sc[j]) > best[0]:
            best = (float(sc[j]), float(du[j]), float(dv[j]), float(an[j]))
    # 3) local refinement: small steps in u, v, angle
    for step, ast in ((60.0, 3.0), (20.0, 1.0)):
        r = torch.arange(-3, 4, device=dev, dtype=torch.float32)
        U, V, G = torch.meshgrid(best[1] + r * step, best[2] + r * step, best[3] + r * ast, indexing="ij")
        sc = score(U.flatten(), V.flatten(), G.flatten())
        j = int(sc.argmax())
        if float(sc[j]) > best[0]:
            best = (float(sc[j]), float(U.flatten()[j]), float(V.flatten()[j]), float(G.flatten()[j]))
    sc, du, dv, an = best
    if du < dv:                                       # RELION: U >= V, angle of U
        du, dv, an = dv, du, an + 90.0
    an = an % 180.0
    fit_res, prof = _fit_resolution(ps, k, du, dv, an, optics, fy, fx, box_a, res_lo, deg)
    return CTFFit(du, dv, an, fit_res, sc, prof)


def patch_defocus(mic: torch.Tensor, apix: float, optics: Optics, fit: CTFFit, grid: int = 3, tile: int = 512,
                  res_lo: float = 30.0, res_hi: float = 5.0, span_a: float = 2000.0, step_a: float = 20.0,
                  deg: int = 4) -> dict:
    """Local defocus (cryoSPARC Patch CTF): the micrograph in grid x grid regions, each region's tile power spectrum
    scored like estimate_ctf for defocus offsets -span..span (astigmatism, its angle and the envelope / background
    model as the global fit's), the best offset refined by a parabola. A plane is fitted to the offsets (weighted by
    each region's fit score, residual-based standard errors): offset(x, y) = a + bx x + by y with x, y in A from the
    micrograph centre. Returns centres (P, 2) A, offsets (P,), scores (P,), plane (a, bx, by), its standard errors,
    tilt (deg) and whether the tilt is significant (> 3 standard errors)."""
    dev = mic.device
    ny, nx = mic.shape
    fy, fx = freqs(tile, dev)
    box_a = tile * apix
    k = torch.sqrt((fy[:, None] / box_a) ** 2 + (fx[None, :] / box_a) ** 2)
    band = (k >= 1 / res_lo) & (k <= 1 / res_hi)
    kn = (k[band] - 1 / res_lo) / (1 / res_hi - 1 / res_lo)
    B = _basis(kn, deg)
    offs = torch.arange(-span_a, span_a + 0.5 * step_a, step_a, device=dev)
    cen, off, scr = [], [], []
    hy, hx = ny // grid, nx // grid
    for gy in range(grid):
        for gx in range(grid):
            sub = mic[gy * hy:(gy + 1) * hy, gx * hx:(gx + 1) * hx]
            if min(sub.shape) < tile:
                continue
            ps = power_spectrum(sub, apix, tile).sqrt()
            y = ps[band]
            if not bool(torch.isfinite(y).all()) or float(y.std()) <= 0:     # empty / saturated region
                continue
            y = (y - y.mean()) / y.std()
            sc = []
            for i in range(0, len(offs), 24):
                o = offs[i:i + 24]
                c2 = _ctf2(fit.defocus_u + o, fit.defocus_v + o, torch.full_like(o, fit.angle), optics, fy, fx,
                           box_a)[:, band]
                sc.append(_fit_score(y, c2, B))
            sc = torch.cat(sc)
            if not bool(torch.isfinite(sc).all()):
                continue
            j = int(sc.argmax().clamp(1, len(offs) - 2))
            y0, y1, y2 = float(sc[j - 1]), float(sc[j]), float(sc[j + 1])
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            cen.append(((gx + 0.5) * hx * apix - 0.5 * nx * apix, (gy + 0.5) * hy * apix - 0.5 * ny * apix))
            off.append(float(offs[j]) + max(-1.0, min(1.0, frac)) * step_a)
            scr.append(y1)
    cen, off, scr = np.array(cen), np.array(off), np.array(scr)
    out = dict(centres_a=cen, offsets_a=off, scores=scr, plane=(0.0, 0.0, 0.0), plane_se=(np.inf,) * 3,
               tilt_deg=0.0, significant=False)
    if len(off) >= 4:
        w = np.clip(scr, 1e-3, None)
        X = np.c_[np.ones(len(off)), cen]
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], off * sw, rcond=None)
        r = off - X @ coef
        dof = max(1, len(off) - 3)
        s2 = float((w * r * r).sum() / w.sum()) * len(off) / dof
        cov = s2 * np.linalg.pinv((X * w[:, None]).T @ X / w.mean())
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        slope = np.hypot(coef[1], coef[2])
        slope_se = np.hypot(se[1], se[2]) / math.sqrt(2)
        out.update(plane=tuple(float(c) for c in coef), plane_se=tuple(float(x) for x in se),
                   tilt_deg=float(np.degrees(np.arctan(slope))), significant=bool(slope > 3 * slope_se),
                   residual_a=float(np.sqrt(np.mean(r * r))))
    return out


def _fit_resolution(ps, k, du, dv, an, optics, fy, fx, box_a, res_lo, deg, z=3.0):
    """How far the fitted CTF matches the Thon rings: the spectrum above res_lo minus a smooth background
    (polynomial in |k|, fitted together with the envelope x CTF^2 by least squares) is correlated with CTF^2
    ring by ring (windows of two CTF periods, by the fitted phase at every pixel, so astigmatic rings are
    followed); the fit resolution is the start of the first two consecutive windows whose correlation is not
    significant (below z / sqrt(pixels in the window): single spectrum pixels are noisy, so a real ring
    correlation can be small; EMPIAR-10025 at 3.8 A: 0.13 over ~4000 pixels)."""
    dev = k.device
    one = lambda v: torch.tensor([v], device=dev)
    c2 = _ctf2(one(du), one(dv), one(an), optics, fy, fx, box_a)[0]
    p = torch.zeros(1, 6, device=dev)
    p[0, 0], p[0, 1], p[0, 2] = du, dv, an
    lam, cs = optics.wavelength, optics.cs_mm * 1e7
    sy, sx = (fy / box_a)[:, None], (fx / box_a)[None, :]
    s2 = sx * sx + sy * sy
    ang = torch.atan2(sy.expand_as(s2), sx.expand_as(s2))
    df = 0.5 * (du + dv) + 0.5 * (du - dv) * torch.cos(2 * (ang - math.radians(an)))
    chi = math.pi * lam * df * s2 - 0.5 * math.pi * cs * lam ** 3 * s2 * s2
    band = (k >= 1 / res_lo) & (k <= float(fx[-1]) / box_a)
    y = ps[band]
    y = (y - y.mean()) / y.std()
    kn = (k[band] - k[band].min()) / (k[band].max() - k[band].min())
    B = _basis(kn, deg + 2)
    X = torch.cat([B, B * c2[band][:, None]], -1)
    sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    a = y - B @ sol[:B.shape[1]]                                     # data minus background
    ring = (chi[band] / math.pi).floor().long()
    ring = ring - ring.min()
    kb, cb = k[band], c2[band]
    rows = []
    for r in range(int(ring.max()) - 1):
        m = (ring == r) | (ring == r + 1)
        if int(m.sum()) < 50:
            continue
        x1, x2 = a[m] - a[m].mean(), cb[m] - cb[m].mean()
        rows.append((float(kb[m].mean()), float((x1 * x2).sum() / (x1.norm() * x2.norm()).clamp_min(1e-12)),
                     int(m.sum())))
    fit = 1 / rows[-1][0] if rows else res_lo
    for j in range(len(rows) - 1):
        if all(rows[i][1] * math.sqrt(rows[i][2]) < z for i in (j, j + 1)):
            fit = 1 / rows[j][0]
            break
    ks = np.array([r[0] for r in rows])
    # for figures: background-subtracted spectrum and fitted envelope x CTF^2 on the 2D grid (NaN outside the
    # band), and both averaged along equal CTF phase (astigmatism removed) against the equivalent frequency at
    # the mean defocus
    model = (B * cb[:, None]) @ sol[B.shape[1]:]
    d2, m2 = torch.full_like(ps, float("nan")), torch.full_like(ps, float("nan"))
    d2[band], m2[band] = a, model
    dfm = 0.5 * (du + dv)
    qa, qb = 0.5 * math.pi * cs * lam ** 3, math.pi * lam * dfm
    s2e = (qb - torch.sqrt((qb * qb - 4 * qa * chi[band]).clamp_min(0))) / (2 * qa)
    ke = s2e.clamp_min(0).sqrt()
    nb = 160
    lo, hi = float(kb.min()), float(kb.max())
    ib = ((ke - lo) / (hi - lo) * nb).long().clamp(0, nb - 1)
    cnt = torch.zeros(nb, device=dev).index_add_(0, ib, torch.ones_like(ke)).clamp_min(1)
    d1 = torch.zeros(nb, device=dev).index_add_(0, ib, a) / cnt
    m1 = torch.zeros(nb, device=dev).index_add_(0, ib, model) / cnt
    kc = lo + (torch.arange(nb, device=dev) + 0.5) * (hi - lo) / nb
    return fit, dict(k=ks, cc=np.array([r[1] for r in rows]), n=np.array([r[2] for r in rows]), z=z,
                     data2d=d2.cpu().numpy(), model2d=m2.cpu().numpy(), apix=box_a / ps.shape[0],
                     k1d=kc.cpu().numpy(), data1d=d1.cpu().numpy(), model1d=m1.cpu().numpy())
