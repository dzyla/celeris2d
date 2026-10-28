"""Score particle selections against a curated reference set from the same micrographs.

A pick counts as a true particle if a reference particle lies within --radius A of it in the same
micrograph (matched by file basename). Reports precision / recall / F1 for every selection stored in a
deep2D membership.npz (single round = round00).

usage: python evaluate_selection.py picks.star reference.star deep2d_dir [--apix-mic 0.885] [--radius 50]
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from celeris2d.star import read_star


def coords(star):
    p = read_star(star)["particles"]
    mic = p["rlnMicrographName"].map(os.path.basename).to_numpy()
    return mic, p[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks")
    ap.add_argument("reference")
    ap.add_argument("deep_dir")
    ap.add_argument("--apix-mic", type=float, default=0.885)
    ap.add_argument("--radius", type=float, default=50.0)
    a = ap.parse_args()
    mic, xy = coords(a.picks)
    rmic, rxy = coords(a.reference)
    true = np.zeros(len(mic), bool)
    rad = a.radius / a.apix_mic
    for m in np.unique(mic):
        sel = np.nonzero(mic == m)[0]
        ref = rxy[rmic == m]
        if len(ref):
            d, _ = cKDTree(ref).query(xy[sel])
            true[sel] = d <= rad
    n_true = true.sum()
    print(f"{len(mic)} picks, {n_true} match a curated particle ({n_true / len(mic) * 100:.1f}% of picks are real)")
    mem = np.load(os.path.join(a.deep_dir, "membership.npz"))
    keys = sorted(k for k in mem.files if k.startswith("round")) + ["final_good"]
    rows = []
    for k in keys + ["final_good+ambiguous"]:
        idx = np.union1d(mem["final_good"], mem["ambiguous"]) if k == "final_good+ambiguous" else mem[k]
        tp = true[idx].sum()
        prec, rec = tp / max(len(idx), 1), tp / max(n_true, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append((k, len(idx), prec, rec, f1))
        print(f"  {k:22s} n={len(idx):6d}  precision {prec * 100:5.1f}%  recall {rec * 100:5.1f}%  F1 {f1:.3f}")
    for k in ("ambiguous", "unstable", "bad"):
        if k in mem.files and len(mem[k]):
            print(f"  {k:22s} n={len(mem[k]):6d}  real particles inside: {true[mem[k]].mean() * 100:5.1f}%")
    return rows


if __name__ == "__main__":
    main()
