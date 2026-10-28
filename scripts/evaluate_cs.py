"""Classifier-independent 2D quality check on cryoSPARC-format results (cryoSPARC's own or celeris2d's).

For every class: sample up to --per-class particles, split odd/even, rebuild both half-set averages
from the file's alignments2D (pose/shift, cryoSPARC convention: psi = -pose, t = -shift*psize),
CTF-Wiener weighted, and report the FRC=0.143 resolution. Prints a per-class table and a summary.

usage: python evaluate_cs.py particles.csg|.cs [--per-class 3000] [--diameter 220]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from celeris2d.engine import TWO_PI, ctf_2d, freqs, rot_grid, soft_circle, upsample2
from celeris2d.sources import CryosparcSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cs")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--diameter", type=float, default=220)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = CryosparcSource(a.cs, a.project_dir)
    cs = src.cs
    cls = cs["alignments2D/class"].astype(int)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(np.concatenate([rng.choice(np.nonzero(cls == k)[0], min(a.per_class, (cls == k).sum()), replace=False)
                                  for k in np.unique(cls)]))
    psi_all = -np.degrees(cs["alignments2D/pose"][sel]).astype(np.float32)
    t_all = (-cs["alignments2D/shift"][sel] * src.apix).astype(np.float32)
    c_sel = cls[sel]
    src.subset(sel)
    imgs = src.read_all()
    dev = "cuda"
    B = src.box
    box = B * src.apix
    fy, fx = freqs(B, dev)
    r = torch.arange(B, device=dev, dtype=torch.float32) - B // 2
    rr = torch.sqrt(r[:, None] ** 2 + r[None, :] ** 2)
    rad = a.diameter / 2 / src.apix
    mask = soft_circle(B, rad, 0.05 * B, dev)
    bg = rr > rad
    shell = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)).long().clamp_(max=B // 2)
    rows = []
    for k in np.unique(c_sel):
        idx = np.nonzero(c_sel == k)[0]
        halves = []
        for h in (idx[0::2], idx[1::2]):
            x = torch.tensor(imgs[h], device=dev)
            xb = x[:, bg]
            x = (x - xb.mean(1)[:, None, None]) / xb.std(1).clamp_min(1e-6)[:, None, None]
            X = torch.fft.rfft2(x * mask, norm="forward")
            ctfp = torch.tensor(src.ctf[h], dtype=torch.float32, device=dev)
            ps = torch.tensor(psi_all[h], device=dev)
            t = torch.tensor(t_all[h], device=dev)
            y = ctf_2d(ctfp, src.optics, fy, fx, box) * X * torch.exp(
                1j * TWO_PI / box * (fx[None, None] * t[:, 0, None, None] + fy[None, :, None] * t[:, 1, None, None]))
            al = F.grid_sample(upsample2(y, B)[:, None], rot_grid(-ps, B, 2 * B), align_corners=False)[:, 0].sum(0)
            den = (ctf_2d(ctfp, src.optics, fy, fx, box, ang_offset_deg=ps) ** 2).sum(0)
            halves.append(torch.fft.rfft2(al, norm="forward") / (den + 0.1 * den.mean()))
        p, q = halves
        sh = shell.flatten()
        cross = torch.zeros(B // 2 + 1, device=dev).index_add_(0, sh, (p * q.conj()).real.flatten())
        pp = torch.zeros(B // 2 + 1, device=dev).index_add_(0, sh, (p.abs() ** 2).flatten())
        qq = torch.zeros(B // 2 + 1, device=dev).index_add_(0, sh, (q.abs() ** 2).flatten())
        frc = (cross / torch.sqrt(pp * qq).clamp_min(1e-30)).cpu().numpy()
        below = np.nonzero(frc[2:] < 0.143)[0]
        kk = below[0] + 2 if len(below) else B // 2
        f = np.arange(B // 2 + 1) / box
        band = (f >= 1 / 20) & (f <= 1 / (2 * src.apix))
        rows.append((k, int((cls == k).sum()), len(idx), box / kk, float(frc[band].mean())))
    rows.sort(key=lambda r: -r[1])
    res = np.array([r[3] for r in rows])
    band = np.array([r[4] for r in rows])
    n = np.array([r[1] for r in rows])
    pr = n / n.sum()
    print(f"{a.cs}: {len(rows)} classes, {n.sum()} particles, effective number of classes "
          f"{np.exp(-(pr * np.log(pr + 1e-12)).sum()):.1f}")
    print(f"  FRC0.143 (half-sets, <= {a.per_class}/class): population-weighted mean {np.average(res, weights=n):.1f} A, "
          f"median {np.median(res):.1f} A, best {res.min():.1f} A")
    print(f"  mean FRC 20 A..Nyquist (higher = better aligned/more signal): population-weighted {np.average(band, weights=n):.3f}, "
          f"median {np.median(band):.3f}")
    for thr in (12, 15, 20):
        print(f"  particles in classes better than {thr} A: {n[res <= thr].sum() / n.sum() * 100:5.1f}%  "
              f"({(res <= thr).sum()} classes)")
    print("  class  particles  FRC_A  bandFRC")
    for k, nk, ns, rk, bk in rows:
        print(f"  {k:5d} {nk:10d}  {rk:5.1f}  {bk:6.3f}")


if __name__ == "__main__":
    main()
