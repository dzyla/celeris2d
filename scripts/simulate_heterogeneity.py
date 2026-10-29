"""Simulate a two-state particle set from a reference map (for testing multi-class ab initio).

usage: simulate_heterogeneity.py <reference.mrc> <ctf source .star> <outdir> [n=12000] [box=84] [snr=0.08] [hole_radius_A=30]

State A = the reference, state B = the reference with a sphere of density removed (radius hole_radius_A, at the
densest spot >= hole radius from the centre). Particles: random orientations, small random shifts, CTF from the source star's particles,
white noise at the given signal-to-noise ratio (variance ratio after CTF). Writes particles.mrcs, particles.star
(RELION 3.1+ with an optics table), truth.npz (state, rot/tilt/psi, shifts) and the two state maps.
"""
import math
import os
import sys

import mrcfile
import numpy as np
import torch

from celeris2d.abinitio import Volume, euler_to_matrix
from celeris2d.engine import Optics, ctf_2d, freqs, TWO_PI
from celeris2d.star import read_star


def main():
    ref, ctfstar, out = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
    b = int(sys.argv[5]) if len(sys.argv) > 5 else 84
    snr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.08
    hole = float(sys.argv[7]) if len(sys.argv) > 7 else 30.0
    dev = "cuda"
    os.makedirs(out, exist_ok=True)
    m = mrcfile.open(ref, permissive=True)
    v, B, pa = torch.tensor(m.data.astype(np.float32), device=dev), m.data.shape[0], float(m.voxel_size.x)
    apix = B * pa / b
    ft = torch.fft.fftshift(torch.fft.fftn(v))
    o = (B - b) // 2
    va = torch.fft.ifftn(torch.fft.ifftshift(ft[o:o + b, o:o + b, o:o + b])).real
    va = va / va.std()
    # state B: remove a sphere around the densest off-centre spot
    c = torch.arange(b, device=dev, dtype=torch.float32) - b // 2
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2) * apix
    sm = torch.nn.functional.avg_pool3d(va[None, None], 5, 1, 2)[0, 0]
    sm[rr < hole] = -1e9
    iz, iy, ix = np.unravel_index(int(sm.argmax()), sm.shape)
    d = torch.sqrt((xx - (ix - b // 2)) ** 2 + (yy - (iy - b // 2)) ** 2 + (zz - (iz - b // 2)) ** 2) * apix
    cut = torch.clamp((d - hole) / 4, 0, 1)                  # soft-edged hole
    vb = va * cut
    print(f"state B removes {100 * (1 - (vb.clamp_min(0).sum() / va.clamp_min(0).sum()).item()):.1f}% of positive density")
    for name, vol in (("stateA.mrc", va), ("stateB.mrc", vb)):
        with mrcfile.new(os.path.join(out, name), overwrite=True) as f:
            f.set_data(vol.cpu().numpy()); f.voxel_size = apix
    star = read_star(ctfstar)
    parts, optics = star["particles"], star["optics"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(parts), n)
    du, dv, da = (parts[k].to_numpy()[pick].astype(np.float32) for k in ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"))
    kv, cs, q0 = (float(optics[k].iloc[0]) for k in ("rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast"))
    state = rng.integers(0, 2, n)
    rot, psi = rng.uniform(-180, 180, n), rng.uniform(-180, 180, n)
    tilt = np.degrees(np.arccos(rng.uniform(-1, 1, n)))
    sh = rng.normal(0, 2 * apix, (n, 2)).astype(np.float32)     # model shift t (A)
    vols = []
    for vol in (va, vb):
        V = Volume(b, dev); V.set_real(vol); vols.append(V)
    fy, fx = freqs(b, dev)
    ph = TWO_PI / (b * apix)
    opt = Optics(kv, cs, q0)
    imgs = np.zeros((n, b, b), np.float32)
    for i in range(0, n, 1000):
        j = np.arange(i, min(n, i + 1000))
        A = euler_to_matrix(torch.tensor(rot[j]), torch.tensor(tilt[j]), torch.tensor(psi[j])).to(dev)
        P = torch.empty((len(j), b, b // 2 + 1), dtype=torch.complex64, device=dev)
        for s_ in (0, 1):
            sel = np.nonzero(state[j] == s_)[0]
            if len(sel):
                P[sel] = vols[s_].project(A[sel])
        p6 = torch.tensor(np.stack([du[j], dv[j], da[j], np.zeros(len(j)), np.zeros(len(j)), np.ones(len(j))], 1), device=dev)
        ctf = ctf_2d(p6, opt, fy, fx, b * apix)
        t = torch.tensor(sh[j], device=dev)
        P = P * ctf * torch.exp(-1j * ph * (fx[None, None, :] * t[:, 0, None, None] + fy[None, :, None] * t[:, 1, None, None]))
        img = torch.fft.irfft2(P, s=(b, b), norm="forward")
        sig = img.std()
        img = img + torch.randn_like(img) * sig / math.sqrt(snr)
        imgs[j] = img.cpu().numpy()
    with mrcfile.new(os.path.join(out, "particles.mrcs"), overwrite=True) as f:
        f.set_data(imgs); f.voxel_size = apix
    # RELION: psi_relion etc. are the 3D angles directly; origin = -t
    with open(os.path.join(out, "particles.star"), "w") as fh:
        fh.write("\ndata_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n_rlnVoltage #3\n"
                 "_rlnSphericalAberration #4\n_rlnAmplitudeContrast #5\n_rlnImagePixelSize #6\n_rlnImageSize #7\n"
                 f"_rlnImageDimensionality #8\n1 opticsGroup1 {kv} {cs} {q0} {apix:.5f} {b} 2\n\n")
        fh.write("data_particles\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\n_rlnDefocusV #3\n_rlnDefocusAngle #4\n"
                 "_rlnOpticsGroup #5\n_rlnMicrographName #6\n")
        for i in range(n):
            fh.write(f"{i + 1:06d}@particles.mrcs {du[i]:.1f} {dv[i]:.1f} {da[i]:.2f} 1 mic{i // 200:04d}.mrc\n")
    np.savez(os.path.join(out, "truth.npz"), state=state, rot=rot, tilt=tilt, psi=psi, shift_a=sh)
    print(f"wrote {n} particles, box {b} @ {apix:.3f} A, SNR {snr}")


if __name__ == "__main__":
    main()
