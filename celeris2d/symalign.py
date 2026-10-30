"""Put a map into the frame of a symmetry the user chose (Refine3D with sym != the start's symmetry).

A C1 map lies in an arbitrary orientation; the symmetry operators (symmetry_ops, RELION convention) have the n-fold
axis on z and, for Dn, a 2-fold on x. sym_frame finds where those axes are in the map: the axis about which a
rotation by 360/n reproduces the map best (all directions, then a local refinement), then for Dn the 2-fold
perpendicular to it. It does not choose the symmetry - that stays the user's - it reports how well the chosen
operators hold (correlation of the map with its rotated copy vs the same rotation by half the angle, a control
that a cylindrical or blob-like map passes too), so a symmetry the map does not have shows as a low score.

    R, info = sym_frame(vol, apix, "D2")       # columns: the map's x (2-fold), y, z (n-fold) axes
    vol2, A2 = apply_frame(vol, A, R)          # the same images: vol2 sampled at R r, poses A @ R

Rotation only: the symmetry axes must pass through the box centre, which they do for a refined, centred map (the
centre of mass is reported; a map more than 2 px off centre gets a warning).
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as F


def _rotmat(axes: torch.Tensor, ang: float) -> torch.Tensor:
    """Rotations by ang about unit axes (n, 3) (x, y, z) -> (n, 3, 3)."""
    a = axes / axes.norm(dim=1, keepdim=True)
    K = torch.zeros(len(a), 3, 3, device=a.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
    K = K - K.transpose(1, 2)
    eye = torch.eye(3, device=a.device).expand(len(a), 3, 3)
    return eye + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _fib(n: int, dev) -> torch.Tensor:
    i = np.arange(n) + 0.5
    z = 1 - i / n                                   # upper hemisphere: an axis and its opposite are one axis
    r = np.sqrt(1 - z * z)
    ph = math.pi * (1 + 5 ** 0.5) * i
    return torch.tensor(np.stack([r * np.cos(ph), r * np.sin(ph), z], 1), dtype=torch.float32, device=dev)


def _prep(vol: torch.Tensor, apix: float, res_a: float, nb: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier-binned to nb^3, Gaussian low-pass at res_a, and a soft sphere for the correlations."""
    b = vol.shape[0]
    Fv = torch.fft.fftshift(torch.fft.fftn(vol.float()))
    if nb < b:
        c, h = b // 2, nb // 2
        Fv = Fv[c - h:c + h, c - h:c + h, c - h:c + h]
    else:
        nb = b
    ap = apix * b / nb
    k = torch.fft.fftshift(torch.fft.fftfreq(nb, d=ap)).to(vol.device)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    v = torch.fft.ifftn(torch.fft.ifftshift(Fv * torch.exp(-0.5 * (kk * res_a) ** 2))).real
    c = torch.arange(nb, device=vol.device, dtype=torch.float32) - nb // 2
    rr = torch.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    mask = (0.5 + 0.5 * torch.cos(math.pi * ((rr - 0.38 * nb) / (0.08 * nb)).clamp(0, 1)))
    return v, mask


def _scores(v: torch.Tensor, mask: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    from .abinitio import _rotate_volumes
    x = (v - (v * mask).sum() / mask.sum()) * mask
    xn = x.norm()
    out = []
    for i in range(0, len(R), 32):
        y = _rotate_volumes(v, R[i:i + 32])
        y = (y - (y * mask).sum((1, 2, 3), keepdim=True) / mask.sum()) * mask
        out.append((y * x).sum((1, 2, 3)) / (y.flatten(1).norm(dim=1) * xn).clamp_min(1e-12))
    return torch.cat(out)


def _best_axis(v, mask, ang, axes, seed=0) -> tuple[torch.Tensor, float]:
    s = _scores(v, mask, _rotmat(axes, ang))
    ax, best = axes[int(s.argmax())], float(s.max())
    g = torch.Generator(device="cpu").manual_seed(seed)
    for step in (0.04, 0.015, 0.005):                  # local refinement (radians, roughly)
        for _ in range(3):
            cand = ax + step * torch.randn(48, 3, generator=g).to(v.device)
            cand = torch.cat([ax[None], cand / cand.norm(dim=1, keepdim=True)])
            cs = _scores(v, mask, _rotmat(cand, ang))
            if float(cs.max()) > best:
                ax, best = cand[int(cs.argmax())], float(cs.max())
    return ax / ax.norm(), best


def parse(sym: str) -> tuple[str, int]:
    m = re.fullmatch(r"([CD])(\d+)", sym.strip().upper())
    if not m:
        raise ValueError(f"symmetry frame: {sym!r} not supported (Cn, Dn)")
    return m.group(1), int(m.group(2))


def sym_frame(vol, apix: float, sym: str, res_a: float = 12.0, nb: int = 72, n_axes: int = 2500,
              log=print) -> tuple[torch.Tensor, dict]:
    """R (3, 3), columns = the map's directions of the symmetry frame's x, y, z axes; info with the scores."""
    kind, n = parse(sym)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vt = torch.as_tensor(np.asarray(vol) if not torch.is_tensor(vol) else vol, dtype=torch.float32).to(dev)
    b = vt.shape[0]
    w = vt.clamp_min(0)
    c = torch.arange(b, device=dev, dtype=torch.float32) - b // 2
    com = [float((c.view(*s) * w).sum() / w.sum().clamp_min(1e-12)) for s in ((1, 1, -1), (1, -1, 1), (-1, 1, 1))]
    info = dict(sym=sym.upper(), centre_of_mass_px=[round(x, 2) for x in com], warnings=[])   # (x, y, z)
    if max(abs(x) for x in com) > 2:
        info["warnings"].append(f"map centre of mass {np.round(com, 1).tolist()} px from the box centre: symmetry "
                                f"axes through the centre may not pass through the particle")
    if n == 1 and kind == "C":
        return torch.eye(3), dict(info, scores={})
    v, mask = _prep(vt, apix, res_a, nb)
    axes = _fib(n_axes, dev)
    if n > 1:
        z, sc = _best_axis(v, mask, 2 * math.pi / n, axes)
    else:                                                # D1: the 2-fold is the only axis; put it on x below
        z, sc = torch.tensor([0.0, 0.0, 1.0], device=dev), float("nan")
    ctrl = float(_scores(v, mask, _rotmat(z[None], math.pi / n))[0]) if n > 1 else float("nan")
    info["scores"] = {f"C{n} about its axis": round(sc, 3), f"control: {180 / n:.0f} deg about it": round(ctrl, 3)}
    # x: a 2-fold perpendicular to z (Dn), else any perpendicular direction
    u = torch.linalg.svd(z[None].cpu())[2][1:].to(dev)
    th = torch.linspace(0, math.pi, 361, device=dev)[:-1]
    perp = torch.cos(th)[:, None] * u[0] + torch.sin(th)[:, None] * u[1]
    if kind == "D":
        if n == 1:
            x, s2 = _best_axis(v, mask, math.pi, axes)
            z = torch.linalg.svd(x[None].cpu())[2][1].to(dev)
        else:
            s = _scores(v, mask, _rotmat(perp, math.pi))
            x, s2 = perp[int(s.argmax())], float(s.max())
            # refine within the plane
            for step in (0.5, 0.1):
                t0 = th[int(s.argmax())] if step == 0.5 else t0
                tt = t0 + torch.deg2rad(torch.linspace(-3 * step, 3 * step, 13, device=dev))
                pp = torch.cos(tt)[:, None] * u[0] + torch.sin(tt)[:, None] * u[1]
                ss = _scores(v, mask, _rotmat(pp, math.pi))
                if float(ss.max()) >= s2:
                    x, s2, t0 = pp[int(ss.argmax())], float(ss.max()), tt[int(ss.argmax())]
        info["scores"]["2-fold perpendicular"] = round(s2, 3)
        info["scores"]["2-fold perpendicular, median over directions"] = \
            round(float(_scores(v, mask, _rotmat(perp[::8], math.pi)).median()), 3)
    else:
        x = perp[0]
    x = x - (x @ z) * z
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], 1).cpu()
    ops = _ops(sym).to(dev)
    v2 = _rotated(v, R.to(dev))
    info["scores"]["all operators, mean (in the new frame)"] = round(float(_scores(v2, mask, ops[1:]).mean()), 3) \
        if len(ops) > 1 else None
    tilt = math.degrees(math.acos(min(1.0, abs(float(z[2])))))
    info["axis_from_box_z_deg"] = round(tilt, 2)
    log(f"[sym] {sym.upper()} frame: n-fold axis {tilt:.1f} deg from the box z axis; " +
        ", ".join(f"{k} {v_}" for k, v_ in info["scores"].items()))
    return R, info


def _ops(sym):
    from .abinitio import symmetry_ops
    return symmetry_ops(sym).float()


def _rotated(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    from .abinitio import _rotate_volumes
    return _rotate_volumes(v, R[None])[0]


def apply_frame(vol, A: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The map in the symmetry frame (sampled at R r) and the poses that give the same images (A @ R)."""
    vt = torch.as_tensor(vol, dtype=torch.float32)
    dev = vt.device
    out = _rotated(vt.to("cuda" if torch.cuda.is_available() else dev), R.float().to(
        "cuda" if torch.cuda.is_available() else dev)).to(dev)
    return out, A.float() @ R.float().to(A.device)
