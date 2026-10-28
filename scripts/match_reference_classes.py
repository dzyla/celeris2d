"""How much of a run's data sits in classes that look like a trusted reference set's classes.

For every class of the run: best rotation/mirror-invariant normalised cross-correlation against all
reference class averages (e.g. cryoSPARC's), both resampled to the same pixel size. Not biased by the
run's own alignments (unlike half-set FRC of the run's own particles).

usage: python match_reference_classes.py run_dir ref_classes.mrc ref_particles.cs [--apix-run A] [--diameter 220]
"""
import argparse
import sys

import mrcfile
import numpy as np
import pandas as pd
import torch

from celeris2d.deep2d import similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("ref_mrc"); ap.add_argument("ref_cs")
    ap.add_argument("--diameter", type=float, default=220)
    ap.add_argument("--min-sim", type=float, default=0.5)
    a = ap.parse_args()
    run = mrcfile.read(f"{a.run_dir}/classes.mrcs").astype(np.float32)
    st = pd.read_csv(f"{a.run_dir}/class_stats.csv")
    ref = mrcfile.read(a.ref_mrc).astype(np.float32)
    if a.ref_cs.endswith(".star"):  # RELION reference: populations and pixel size from the data star
        from celeris2d.star import read_star
        blk = read_star(a.ref_cs)
        rpop = np.bincount(blk["particles"]["rlnClassNumber"].to_numpy() - 1, minlength=len(ref))
        opt = blk["optics"].iloc[0]
        box_a = float(opt["rlnImagePixelSize"]) * float(opt["rlnImageSize"])
    else:
        cs = np.load(a.ref_cs, mmap_mode="r")
        rpop = np.bincount(cs["alignments2D/class"], minlength=len(ref))
        box_a = float(cs["blob/psize_A"][0]) * cs["blob/shape"][0][0]
    b = run.shape[1]
    if ref.shape[1] != b:  # Fourier-resample reference to the run's box (same field of view)
        import torch.fft as tf
        R = tf.rfft2(torch.as_tensor(ref), norm="forward")
        from celeris2d.engine import ft_resize
        ref = tf.irfft2(ft_resize(R, b), s=(b, b), norm="forward").numpy()
    apix = box_a / b

    def bandpass(im, lo=60.0, hi=18.0):  # keep shape-defining 18-60 A, drop blob envelope and fine texture
        f = np.fft.fftfreq(b)[:, None] / apix
        g = np.fft.fftfreq(b)[None, :] / apix
        k = np.hypot(f, g)
        w = (1 - np.exp(-(k * lo) ** 2 / 2)) * np.exp(-(k * hi) ** 2 / 2)
        return np.real(np.fft.ifft2(np.fft.fft2(im) * w)).astype(np.float32)
    run = np.stack([bandpass(x) for x in run])
    ref = np.stack([bandpass(x) for x in ref])
    y, x = np.mgrid[:b, :b] - b // 2
    inside = torch.as_tensor(np.hypot(x, y) * apix < a.diameter / 2, dtype=torch.float32, device="cuda")
    # only real particle views of the reference count as targets (drop junk: tiny or ice/edge classes)
    sim = similarity(run, ref, inside, "cuda")                    # (K_run, K_ref)
    best = sim.max(1)
    n = st.particles.to_numpy()
    print(f"{a.run_dir}: {len(run)} classes; fraction of particles in classes matching a reference class:")
    for t in (0.4, 0.5, 0.6):
        print(f"  sim >= {t}: {n[best >= t].sum() / n.sum() * 100:5.1f}%  ({(best >= t).sum()} classes)")
    print(f"  particle-weighted mean best similarity: {np.average(best, weights=n):.3f}")
    # coverage: how many reference views (by their own population) are reproduced by some run class
    cov = sim.max(0)
    print(f"  reference views reproduced (sim >= {a.min_sim}): {(cov >= a.min_sim).sum()} / {len(ref)} classes, "
          f"{rpop[cov >= a.min_sim].sum() / rpop.sum() * 100:.1f}% of reference particles")


if __name__ == "__main__":
    main()
