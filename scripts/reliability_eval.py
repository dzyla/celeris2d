"""Reliability of 2D classification runs: class occupancy, oracle good/bad separability and seed agreement.

    python scripts/reliability_eval.py RUN_DIR [RUN_DIR ...] [--good GOOD] [--name NAME]

RUN_DIR: a celeris2d output folder (particles.cs or particles.star + class_stats.csv). Several folders are
treated as repeats (seeds) of the same input and are matched to the first by Hungarian assignment on shared
membership (stability.match_to_reference). Reported per run: wall time (history.json), number of empty
classes (0 particles) and of tiny classes (< 2 % of N/K), oracle best F1 / AP vs GOOD (a cryoSPARC .cs with
uid, or a RELION .star matched on rlnImageName), i.e. the best any class-based selection could do. Across
runs: fraction of particles that land in the matched class (each run vs run 0, and the mean over all pairs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from celeris2d.stability import match_to_reference  # noqa: E402


def _star_keys(path):
    from celeris2d.star import read_star
    p = read_star(path)["particles"]
    if "celUid" in p:
        return p["celUid"].to_numpy().astype(np.int64), p
    nm = [f"{s.split('@')[0].lstrip('0')}@{os.path.basename(s.split('@')[1])}" for s in p["rlnImageName"]]
    return np.array(nm), p


def load_run(d):
    """(keys, labels, K, wall, pops) of a run folder."""
    cs = os.path.join(d, "particles.cs")
    if os.path.exists(cs):
        c = np.load(cs, mmap_mode="r")
        keys, cls = np.asarray(c["uid"]), np.asarray(c["alignments2D/class"]).astype(int)
    else:
        keys, p = _star_keys(os.path.join(d, "particles.star"))
        cls = p["rlnClassNumber"].to_numpy().astype(int) - 1
    K, wall = int(cls.max()) + 1, float("nan")
    hist = os.path.join(d, "history.json")
    if os.path.exists(hist):
        with open(hist) as fh:
            h = json.load(fh)
        K, wall = int(h["params"]["n_classes"]), float(h.get("total_time", float("nan")))
    return keys, cls, K, wall, np.bincount(cls, minlength=K)


def good_keys(path):
    if path.endswith(".cs"):
        return np.asarray(np.load(path, mmap_mode="r")["uid"])
    return _star_keys(path)[0]


def oracle(cls, g, K):
    """Best F1 and AP of a class-based selection ranked by class purity (fraction of good members)."""
    n = np.bincount(cls, minlength=K)
    ng = np.bincount(cls, weights=g, minlength=K)
    pur = ng / np.maximum(n, 1)
    o = np.argsort(-pur)
    tp, sel = np.cumsum(ng[o]), np.cumsum(n[o])
    G = max(g.sum(), 1)
    prec, rec = tp / np.maximum(sel, 1), tp / G
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(np.sum(np.diff(np.r_[0, rec]) * prec)), i + 1


def agreement(runs):
    """runs: list of (keys, labels, K). Returns (vs run 0 per run, mean over all pairs) matched-class agreement."""
    ref_keys, ref_lab, K0 = runs[0]
    order = {k: i for i, k in enumerate(ref_keys.tolist())}
    labs = [ref_lab]
    for keys, lab, K in runs[1:]:
        idx = np.fromiter((order[k] for k in keys.tolist()), dtype=np.int64, count=len(keys))
        full = np.full(len(ref_lab), -1)
        full[idx] = lab
        labs.append(full)
    M = np.stack(labs, 1)
    ok = (M >= 0).all(1)
    M = M[ok]
    vs0 = []
    for j in range(1, M.shape[1]):
        m = match_to_reference(M[:, 0], M[:, j], K0, int(M[:, j].max()) + 1)
        vs0.append(float((m == M[:, 0]).mean()))
    pairs = []
    for a in range(M.shape[1]):
        for b in range(a + 1, M.shape[1]):
            m = match_to_reference(M[:, a], M[:, b], int(M[:, a].max()) + 1, int(M[:, b].max()) + 1)
            pairs.append(float((m == M[:, a]).mean()))
    return vs0, float(np.mean(pairs)) if pairs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--good", default=None, help="curated good particles (.cs with uid or .star)")
    ap.add_argument("--name", default="")
    ap.add_argument("--tsv", default=None, help="append one summary line here")
    a = ap.parse_args(argv)
    gk = good_keys(a.good) if a.good else None
    rows, loaded = [], []
    for d in a.runs:
        keys, cls, K, wall, pops = load_run(d)
        loaded.append((keys, cls, K))
        thr = 0.02 * len(cls) / K
        row = dict(run=os.path.basename(d.rstrip("/")), wall=wall, empty=int((pops == 0).sum()),
                   tiny=int((pops < thr).sum()), max_frac=float(pops.max() / len(cls)))
        if gk is not None:
            g = np.isin(keys, gk)
            row["f1"], row["ap"], row["ncls"] = oracle(cls, g, K)
            row["good"] = int(g.sum())
        rows.append(row)
        print(f"{row['run']:28s} wall {wall:6.1f}s  empty {row['empty']:2d}  tiny {row['tiny']:2d}  "
              f"largest {row['max_frac'] * 100:4.1f}%" + (f"  F1 {row['f1']:.3f} AP {row['ap']:.3f} ({row['ncls']} cls)"
                                                         if gk is not None else ""))
    out = dict(name=a.name, n=len(rows), wall=np.mean([r["wall"] for r in rows]),
               empty=np.mean([r["empty"] for r in rows]), tiny=np.mean([r["tiny"] for r in rows]))
    if gk is not None:
        out["f1"], out["f1_sd"] = np.mean([r["f1"] for r in rows]), np.std([r["f1"] for r in rows])
        out["ap"] = np.mean([r["ap"] for r in rows])
    if len(rows) > 1:
        vs0, pairs = agreement(loaded)
        out["agree"] = pairs
        print(f"agreement vs run 0: {' '.join(f'{v:.3f}' for v in vs0)}; mean over pairs {pairs:.3f}")
    line = (f"{a.name}\tn={out['n']}\twall={out['wall']:.1f}\tempty={out['empty']:.2f}\ttiny={out['tiny']:.2f}"
            + (f"\tF1={out['f1']:.3f}+-{out['f1_sd']:.3f}\tAP={out['ap']:.3f}" if gk is not None else "")
            + (f"\tagree={out['agree']:.3f}" if "agree" in out else ""))
    print(line)
    if a.tsv:
        with open(a.tsv, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
