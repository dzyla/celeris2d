"""Settings optimizer: try a small grid of settings on a random subset, keep the most reproducible one.

There is no ground truth on new data, so settings are scored by REPRODUCIBILITY: each candidate classifies
the same subset from two independent random starts; classes are matched by shared members and the score is
the soft probability that both runs put a particle in the same (matched) class. Settings that fit noise give
arbitrary, non-repeatable classes; settings that resolve real views give the same classes twice. (Validated
against curated ground truth on the RELION LoG test set, see README.)
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .isac import soft_stability

# what is worth tuning (each entry: Params field -> candidate values)
DEFAULT_GRID = {
    "march_margin": [0.0, 4.0],
    "solvent_thresh": [0.05, 0.12],
}
# hard vs soft pose is NOT tuned: on 5k-20k subsets the stability score is too noisy for it (it picked hard pose,
# which lost 0.05-0.08 F1 against curated ground truth); the size rule (hard >= 100k particles) decides instead.
TRIAL_ITERS = 15


def _expand(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def optimize(clf, K, subset=20000, grid=None, seeds=(11, 23), seed=0, log=print):
    """Return (best settings, table of all trials). clf must have its dataset loaded; trials run on a random
    subset of `subset` particles and restore nothing (call apply() with the winner)."""
    grid = grid or DEFAULT_GRID
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(clf.N, min(subset, clf.N), replace=False))
    p = clf.p
    saved = {k: getattr(p, k) for k in grid}
    saved_iter, auto_iter = p.n_iter, clf._auto_iter
    p.n_iter, clf._auto_iter = TRIAL_ITERS, False
    auto = (clf._auto_hard, clf._auto_prior)
    rows = []
    for trial in _expand(grid):
        for k, v in trial.items():
            setattr(p, k, v)
        if "hard_pose" in trial:
            clf._auto_hard = False
        t0 = time.time()
        runs = []
        for s in seeds:
            clf.run(active=idx, n_classes=K, seed=s)
            runs.append(dict(labels=clf.cls.numpy()[idx].copy(), cpost=clf.cpost.numpy()[idx].astype(np.float32),
                             maxprob=clf.maxprob.numpy()[idx].copy()))
        stab = float(soft_stability(runs, K).mean())
        conf = float(np.mean([r["maxprob"].mean() for r in runs]))
        row = dict(trial, stability=round(stab, 4), confidence=round(conf, 3),
                   est_res=round(float(clf.est_res), 1), seconds=round(time.time() - t0, 1))
        rows.append(row)
        log(f"[optimize] {trial} -> stability {stab:.3f}, confidence {conf:.2f}, {row['seconds']}s")
    for k, v in saved.items():
        setattr(p, k, v)
    p.n_iter, clf._auto_iter = saved_iter, auto_iter
    clf._auto_hard, clf._auto_prior = auto
    best = max(rows, key=lambda r: r["stability"])
    return {k: best[k] for k in grid}, rows


def apply(clf, settings):
    for k, v in settings.items():
        setattr(clf.p, k, v)
    if "hard_pose" in settings:
        clf._auto_hard = False
