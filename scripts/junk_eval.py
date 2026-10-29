"""Good/junk separation of a Celeris2D cryoSPARC result against a curated selection (e.g. a cryoSPARC Select 2D job).

usage: junk_eval.py <celeris particles.cs> <good particles_selected.cs> [name]

Classes are ranked by their fraction of curated particles (an oracle selection: the upper bound on what picking
whole classes can achieve) -> best F1 and average precision. If the run used --null-class (p_null.npy next to
particles.cs), particles with P(null) > 0.5 are also reported, and the class ranking is repeated with them
rejected up front.
"""
import os
import sys

import numpy as np


def oracle(cls, good, keep, G):
    K = int(cls.max()) + 1
    n = np.bincount(cls[keep], minlength=K)
    ng = np.bincount(cls[keep], weights=good[keep], minlength=K)
    pur = ng / np.maximum(n, 1)
    o = np.argsort(-pur)
    tp, sel = np.cumsum(ng[o]), np.cumsum(n[o])
    prec, rec = tp / np.maximum(sel, 1), tp / G
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    return f1[i], prec[i], rec[i], i + 1, ap


def main():
    res, goodp = sys.argv[1:3]
    name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.dirname(os.path.abspath(res)))
    c = np.load(res)
    uid, cls = np.asarray(c["uid"]), np.asarray(c["alignments2D/class"]).astype(int)
    good = np.isin(uid, np.load(goodp)["uid"]).astype(float)
    G = good.sum()
    f1, p, r, nc, ap = oracle(cls, good, np.ones(len(uid), bool), G)
    line = f"{name}: N={len(uid)} good={int(G)} ({G / len(uid) * 100:.1f}%) | classes: F1 {f1:.3f} (P {p:.2f} R {r:.2f}, {nc} cls) AP {ap:.3f}"
    pn_path = os.path.join(os.path.dirname(os.path.abspath(res)), "p_null.npy")
    if os.path.exists(pn_path):
        pn = np.load(pn_path)
        assert len(pn) == len(uid), "p_null.npy does not match particles.cs"
        null = pn > 0.5
        junk = 1 - good
        line += (f" | null: {null.mean() * 100:.1f}% of picks = {(null * junk).sum() / junk.sum() * 100:.1f}% of junk, "
                 f"{(null * good).sum() / G * 100:.1f}% of good")
        f1, p, r, nc, ap = oracle(cls, good, ~null, G)
        line += f" | null rejected + classes: F1 {f1:.3f} (P {p:.2f} R {r:.2f}, {nc} cls) AP {ap:.3f}"
    print(line)


if __name__ == "__main__":
    main()
