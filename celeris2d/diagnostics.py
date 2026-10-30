"""Numbers behind the 3D figures, written next to them so a result can be checked without the figure.

    angular_stats     how viewing directions cover the sphere and Fourier space: occupied directions, share of the
                      busiest 10%, and the sampling compensation factor SCF (Baldwin & Lyumkis 2020): each view fills
                      the central plane perpendicular to it, S(k) = how many planes pass through Fourier direction k,
                      SCF = 1 / (<S> <1/S>) over the sphere - 1 for uniform sampling, towards 0 when some directions
                      of Fourier space are barely sampled (smeared map). Symmetry copies of each view count.
    write_fsc         FSC curves as <stem>.json (all curves), <stem>.star (RELION data_fsc labels) and <stem>.xml
                      (EMDB deposition format, the curve the resolution is read from).
    orientation_change  angle between two sets of orientations (smallest over symmetry-equivalent poses).
"""
from __future__ import annotations

import json
import math
import os
from xml.sax.saxutils import escape

import numpy as np
import torch


def _fibonacci(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    ph = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def scf(directions: torch.Tensor, ops: torch.Tensor | None = None, n_k: int = 2000, width_deg: float = 2.0,
        n_max: int = 20000, seed: int = 0) -> tuple[float, float]:
    """(SCF, fraction of Fourier directions with almost no sampling). directions: (n, 3) unit viewing directions in the
    map frame; ops: symmetry operators (S, 3, 3), applied as d @ S."""
    d = torch.as_tensor(directions, dtype=torch.float32)
    if len(d) > n_max:
        d = d[torch.randperm(len(d), generator=torch.Generator().manual_seed(seed))[:n_max]]
    if ops is not None and len(ops) > 1:
        d = (d[:, None, None, :] @ torch.as_tensor(ops, dtype=torch.float32)[None]).reshape(-1, 3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = d.to(dev)
    k = torch.as_tensor(_fibonacci(n_k), dtype=torch.float32, device=dev)
    eps = math.sin(math.radians(width_deg))
    S = torch.zeros(n_k, device=dev)
    for i in range(0, len(d), 8192):
        S += ((k @ d[i:i + 8192].T).abs() < eps).float().sum(1)
    S = S / (len(d) * eps)                      # 1 everywhere for a uniform distribution
    floor = 0.02
    empty = float((S < floor).float().mean())
    Sc = S.clamp_min(floor)
    return float(1.0 / (Sc.mean() * (1.0 / Sc).mean())), empty


def angular_stats(A: torch.Tensor, sym: str = "C1") -> dict:
    """Coverage numbers of orientations A (n, 3, 3) (viewing direction = third row) with symmetry sym."""
    from .abinitio import symmetry_ops, view_coverage
    A = torch.as_tensor(A).float().cpu()
    ops = symmetry_ops(sym)
    out = view_coverage(A, sym)
    s, empty = scf(A[:, 2, :], ops)
    out.update(scf=round(s, 3), fourier_unsampled=round(empty, 4))
    w = []
    if out["occupied"] < 0.3 or out["top10_share"] > 0.6:
        w.append(f"narrow range of views: {out['occupied'] * 100:.0f}% of directions occupied, busiest 10% hold "
                 f"{out['top10_share'] * 100:.0f}% of the particles")
    if s < 0.5 or empty > 0.01:
        w.append(f"uneven Fourier sampling (SCF {s:.2f}, {empty * 100:.1f}% of directions almost empty): the map is "
                 "smeared along the missing directions whatever its FSC says")
    out["warnings"] = w
    return out


def orientation_change(A0: torch.Tensor, A1: torch.Tensor, sym: str = "C1") -> np.ndarray:
    """Angle (degrees) between orientations A0 and A1 (n, 3, 3), smallest over the symmetry-equivalent poses of A1."""
    from .abinitio import symmetry_ops
    A0, A1 = torch.as_tensor(A0).float().cpu(), torch.as_tensor(A1).float().cpu()
    ops = symmetry_ops(sym).float()
    best = torch.full((len(A0),), 180.0)
    for i in range(0, len(A0), 50000):
        a0, a1 = A0[i:i + 50000], A1[i:i + 50000]
        Q = a0.transpose(1, 2)[:, None] @ (a1[:, None] @ ops[None])
        tr = Q.diagonal(dim1=-2, dim2=-1).sum(-1)
        best[i:i + 50000] = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).min(1).values
    return best.numpy()


def angular_histogram(rot: np.ndarray, tilt: np.ndarray, nlon: int = 72, nlat: int = 36) -> dict:
    """Particles per equal-area bin (rot x cos(tilt)), as written to views.json."""
    lon = ((np.asarray(rot) + 180) % 360) - 180
    z = np.cos(np.deg2rad(np.asarray(tilt)))
    H, le, ze = np.histogram2d(lon, z, bins=[np.linspace(-180, 180, nlon + 1), np.linspace(-1, 1, nlat + 1)])
    return dict(rot_edges_deg=le.round(2).tolist(), cos_tilt_edges=ze.round(4).tolist(), counts=H.astype(int).tolist())


def write_views(path: str, rot, tilt, stats: dict):
    with open(path, "w") as f:
        json.dump(dict(stats, **angular_histogram(rot, tilt)), f)


_STAR = {"corrected": "rlnFourierShellCorrelationCorrected", "unmasked": "rlnFourierShellCorrelationUnmaskedMaps",
         "masked": "rlnFourierShellCorrelationMaskedMaps",
         "randomised": "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
         "particle": "rlnFourierShellCorrelationParticleMaskFraction"}


def write_fsc(stem: str, curves: dict, box_a: float, resolution_a: float | None = None, main: str | None = None,
              title: str = "", extra: dict | None = None):
    """curves: name -> FSC per shell (shell i = i / box_a 1/A). Names in _STAR get RELION labels in the .star file
    ('corrected', 'unmasked', 'masked', 'randomised'); every curve goes to the .json; main (default: the first) goes
    to the EMDB .xml."""
    names = list(curves)
    n = min(len(np.asarray(c)) for c in curves.values())
    shell = np.arange(n)
    k = shell / box_a
    ang = np.where(shell > 0, box_a / np.maximum(shell, 1), np.inf)
    cur = {nm: np.asarray(curves[nm], float)[:n] for nm in names}
    main = main or names[0]
    meta = dict(title=title, box_a=round(box_a, 4), shells=int(n), resolution_a=resolution_a, main_curve=main,
                **(extra or {}))
    with open(stem + ".json", "w") as f:
        json.dump(dict(meta, spatial_frequency_inv_a=k.round(6).tolist(),
                       resolution_a_per_shell=[None if not np.isfinite(a) else round(float(a), 3) for a in ang],
                       curves={nm: cur[nm].round(5).tolist() for nm in names}), f, indent=None)
    lab = [nm for nm in names if nm in _STAR]
    with open(stem + ".star", "w") as f:
        f.write("# written by celeris2d\n\ndata_general\n\n")
        if resolution_a:
            f.write(f"_rlnFinalResolution {resolution_a:.4f}\n")
        f.write("\ndata_fsc\n\nloop_\n_rlnSpectralIndex #1\n_rlnResolution #2\n_rlnAngstromResolution #3\n")
        for i, nm in enumerate(lab):
            f.write(f"_{_STAR[nm]} #{i + 4}\n")
        for i in range(n):
            a = 999.0 if not np.isfinite(ang[i]) else ang[i]
            f.write(f"{i:6d} {k[i]:.6f} {a:10.4f} " + " ".join(f"{cur[nm][i]:9.5f}" for nm in lab) + "\n")
    with open(stem + ".xml", "w") as f:
        f.write(f'<fsc title="{escape(title or main)}" xaxis="Resolution (A-1)" yaxis="Correlation Coefficient">\n')
        for i in range(n):
            f.write(f"  <coordinate>\n    <x>{k[i]:.6f}</x>\n    <y>{cur[main][i]:.5f}</y>\n  </coordinate>\n")
        f.write("</fsc>\n")
    return [os.path.basename(stem) + e for e in (".json", ".star", ".xml")]
