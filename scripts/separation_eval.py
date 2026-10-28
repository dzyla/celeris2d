"""Precision/recall of particle-selection rules against curated ground truth (same micrographs).

usage: python separation_eval.py picks.star reference.star outdir [--K 25 --diameter 250 --max-res 12 --seeds 4]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from celeris2d.engine import Classifier2D, Params
from celeris2d.sources import open_source
from celeris2d.stability import particle_scores, run_seeds
from celeris2d.star import read_star


def truth(picks, reference, apix_mic=0.885, radius=50.0):
    def coords(star):
        p = read_star(star)["particles"]
        return p["rlnMicrographName"].map(os.path.basename).to_numpy(), p[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(float)
    mic, xy = coords(picks)
    rmic, rxy = coords(reference)
    t = np.zeros(len(mic), bool)
    for m in np.unique(mic):
        sel = np.nonzero(mic == m)[0]
        ref = rxy[rmic == m]
        if len(ref):
            t[sel] = cKDTree(ref).query(xy[sel])[0] <= radius / apix_mic
    return t


def pr(sel, true):
    tp = (sel & true).sum()
    return tp / max(sel.sum(), 1), tp / max(true.sum(), 1)


def curve(score, true):
    o = np.argsort(-score)
    tp = np.cumsum(true[o])
    n = np.arange(1, len(o) + 1)
    return tp / n, tp / true.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("reference"); ap.add_argument("outdir")
    ap.add_argument("--K", type=int, default=25); ap.add_argument("--diameter", type=float, default=250)
    ap.add_argument("--max-res", type=float, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--param", action="append", default=[], help="engine parameter override, e.g. march_margin=10")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    true = truth(a.picks, a.reference)
    src = open_source(a.picks)
    over = {}
    for kv in a.param:
        k, v = kv.split("=")
        over[k] = (v.lower() == "true") if v.lower() in ("true", "false") else type(getattr(Params(), k))(v)
    clf = Classifier2D(src, Params(n_classes=a.K, diameter_a=a.diameter, max_res_a=a.max_res, **over), devices=["cuda:0"],
                       log=lambda *x: None)
    runs, std = run_seeds(clf, np.arange(src.N), a.K, list(range(a.seeds)), a.diameter)
    sc = particle_scores(runs)
    S = a.seeds
    res = {"n": int(src.N), "n_true": int(true.sum())}
    pts = {}
    pts["single run: best class good"] = pr(runs[0]["in_good"], true)
    for v in range(1, S + 1):
        pts[f"{v}/{S} seeds vote good"] = pr(sc["good_votes"] >= v, true)
    for v in range(2, S + 1):
        pts[f"stable (agree>={v}) & majority good"] = pr((sc["agreement"] >= v) & (sc["good_votes"] * 2 > S), true)
    curves = {"single run P(good)": curve(runs[0]["p_good"], true),
              f"mean P(good), {S} seeds": curve(sc["p_good"], true),
              f"mean P(good) x agreement/{S}": curve(sc["p_good"] * sc["agreement"] / S, true)}

    def f1(p, r):
        return 2 * p * r / max(p + r, 1e-12)
    print(f"{src.N} picks, {true.sum()} real ({true.mean() * 100:.1f}%)")
    for k, (p, r) in pts.items():
        print(f"  {k:40s} precision {p * 100:5.1f}%  recall {r * 100:5.1f}%  F1 {f1(p, r):.3f}")
        res[k] = dict(precision=float(p), recall=float(r), f1=float(f1(p, r)))
    for k, (p, r) in curves.items():
        f = 2 * p * r / np.maximum(p + r, 1e-12)
        i = int(np.argmax(f))
        ap_ = float(np.sum(np.diff(np.r_[0, r]) * p))
        print(f"  {k:40s} best F1 {f[i]:.3f} at precision {p[i] * 100:5.1f}% recall {r[i] * 100:5.1f}%   AP {ap_:.3f}")
        res[k] = dict(best_f1=float(f[i]), precision=float(p[i]), recall=float(r[i]), ap=ap_)
    json.dump(res, open(os.path.join(a.outdir, "separation.json"), "w"), indent=1)
    np.savez(os.path.join(a.outdir, "scores.npz"), true=true, agreement=sc["agreement"], p_good=sc["p_good"],
             good_votes=sc["good_votes"], p_good_single=runs[0]["p_good"])

    from celeris2d.plots import BLUE, ORANGE, _mpl
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cols = [BLUE, ORANGE, "#1baf7a"]
    for (k, (p, r)), c in zip(curves.items(), cols):
        ax.plot(r, p, color=c, lw=2, label=k)
    for k, (p, r) in pts.items():
        ax.plot(r, p, "o", ms=6, color="#52514e" if "single" in k else "#6250d6", mec="white")
        ax.annotate(k.replace(" seeds vote good", "").replace("stable ", ""), (r, p), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(true.mean(), color="#c3c2b7", lw=1, ls="--")
    ax.text(0.01, true.mean() + 0.01, "all picks", fontsize=7, color="#52514e")
    ax.set(xlabel="recall (real particles kept)", ylabel="precision (selection that is real)", xlim=(0, 1), ylim=(0.3, 1),
           title="Particle selection vs curated ground truth")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "precision_recall.png"), dpi=140)


if __name__ == "__main__":
    main()
