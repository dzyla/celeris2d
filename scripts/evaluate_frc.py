"""Alignment-quality metric independent of the classifier: for every class in a star file,
reconstruct odd/even half-set averages at the full input box from the stored
rlnAnglePsi / rlnOriginXAngst / rlnOriginYAngst (RELION conventions), CTF-Wiener
weighted, and report the FRC = 0.143 resolution. Works for RELION and celeris2d output."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from celeris2d.sources import load_particles
from celeris2d.engine import TWO_PI, ctf_2d, freqs, rot_grid, soft_circle, upsample2


def main(star, project_dir, diameter=200.0, min_n=40):
    blocks, parts, imgs, ctf, apix, opt = load_particles(star, project_dir)
    dev = "cuda"
    B = imgs.shape[1]
    box = B * apix
    cls = parts["rlnClassNumber"].to_numpy() - 1
    psi_int = -torch.tensor(parts["rlnAnglePsi"].to_numpy(), dtype=torch.float32, device=dev)
    t = -torch.tensor(parts[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(), dtype=torch.float32, device=dev)
    ctfp = torch.tensor(ctf, dtype=torch.float32, device=dev)
    fy, fx = freqs(B, dev)
    mask = soft_circle(B, diameter / 2 / apix, 0.05 * B, dev)
    shell = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)).long().clamp_(max=B // 2)
    res, pops = [], []
    for k in np.unique(cls):
        sel = np.nonzero(cls == k)[0]
        if len(sel) < min_n:
            continue
        halves = []
        for h in (sel[0::2], sel[1::2]):
            num = torch.zeros((B, B), device=dev)
            den = torch.zeros((B, B // 2 + 1), device=dev)
            for i in range(0, len(h), 128):
                j = torch.tensor(h[i:i + 128], device=dev)
                x = torch.fft.rfft2(torch.tensor(imgs[j.cpu().numpy()].astype(np.float32), device=dev) * mask, norm="forward")
                c = ctf_2d(ctfp[j], opt, fy, fx, box)
                y = c * x * torch.exp(1j * TWO_PI / box * (fx[None, None] * t[j, 0, None, None] + fy[None, :, None] * t[j, 1, None, None]))
                al = F.grid_sample(upsample2(y, B)[:, None], rot_grid(-psi_int[j], B, 2 * B), align_corners=False)[:, 0]
                num += al.sum(0)
                den += (ctf_2d(ctfp[j], opt, fy, fx, box, ang_offset_deg=psi_int[j]) ** 2).sum(0)
            ft = torch.fft.rfft2(num, norm="forward") / (den + 0.1 * den.mean())
            halves.append(ft)
        a, b = halves
        cross = torch.zeros(B // 2 + 1, device=dev).index_add_(0, shell.flatten(), (a * b.conj()).real.flatten())
        pa = torch.zeros(B // 2 + 1, device=dev).index_add_(0, shell.flatten(), (a.abs() ** 2).flatten())
        pb = torch.zeros(B // 2 + 1, device=dev).index_add_(0, shell.flatten(), (b.abs() ** 2).flatten())
        frc = (cross / torch.sqrt(pa * pb).clamp_min(1e-30)).cpu().numpy()
        below = np.nonzero(frc[2:] < 0.143)[0]
        kk = below[0] + 2 if len(below) else B // 2
        res.append(box / kk)
        pops.append(len(sel))
    res, pops = np.array(res), np.array(pops)
    print(f"{star}: {len(res)} classes (n>={min_n}); FRC0.143 resolution: "
          f"population-weighted mean {np.average(res, weights=pops):.2f} A, best {res.min():.2f} A, "
          f"median {np.median(res):.2f} A")
    return res, pops


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
