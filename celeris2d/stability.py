"""Multi-seed stability (the reproducibility idea behind SPHIRE's ISAC) on top of celeris2d.

The same particles are classified several times from independent random starts (no reloading: the
spectra stay on the GPU). Classes of every run are matched to a reference run by shared membership
(optimal assignment on the co-occurrence table), which gives each particle
  * agreement:  in how many runs it landed in the matched (same) group,
  * p_good:     mean posterior mass on good classes across runs (good = judged against one standard),
  * good_votes: in how many runs its best class was good.
A particle that keeps its group across independent runs is a reproducible member of that view; one that
does not is noise-limited. This turns 'jumping' into a direct, single-pass measurement.
"""
from __future__ import annotations

import numpy as np

from .deep2d import class_table, judge, learn_standard, split_two_clusters


def match_to_reference(ref, lab, k_ref, k_lab):
    """Map labels `lab` onto reference labels by maximising shared members (Hungarian assignment)."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((k_lab, k_ref), dtype=np.int64)
    np.add.at(C, (lab, ref), 1)
    r, c = linear_sum_assignment(-C)
    m = np.full(k_lab, -1)
    m[r] = c
    return m[lab]


def run_seeds(clf, idx, K, seeds, diameter_a, std=None, good_classes=None, log=print):
    """Classify `idx` once per seed. Returns per-run dicts and the standard used to judge them."""
    idx = np.asarray(idx)
    runs = []
    for i, s in enumerate(seeds):
        clf.run(active=idx, n_classes=K, seed=s)
        if i == 0:  # all runs are judged against the first run's standard, so 'good' means the same
            clf._auto_hard = clf._auto_prior = False
        t, imgs = class_table(clf, diameter_a, None if std is None else std.templates)
        if std is None:
            if good_classes:
                good = np.zeros(len(t), bool)
                good[[g - 1 for g in good_classes]] = True
            else:
                good = split_two_clusters(t)
            std = learn_standard(t, good, imgs, clf, diameter_a)
        else:
            good = judge(t, std)
        t["good"] = good
        cp = clf.cpost.numpy()[idx].astype(np.float32)
        runs.append(dict(seed=s, labels=clf.cls.numpy()[idx].copy(), cpost=cp, table=t, imgs=imgs,
                         p_good=cp[:, good].sum(1), in_good=good[clf.cls.numpy()[idx]]))
        log(f"[stability] seed {s}: {good.sum()} good classes, "
            f"{runs[-1]['in_good'].sum()} / {len(idx)} particles in good classes")
    return runs, std


def particle_scores(runs):
    """Per-particle agreement (0..S), mean p_good, good votes; plus matched labels per run."""
    ref = runs[0]["labels"]
    K = runs[0]["cpost"].shape[1]
    matched = [ref]
    for r in runs[1:]:
        matched.append(match_to_reference(ref, r["labels"], K, r["cpost"].shape[1]))
    M = np.stack(matched, 1)                                     # (n, S) labels in run-0 numbering
    # agreement with the particle's modal matched label (robust to a bad reference assignment)
    S = M.shape[1]
    agree = np.zeros(len(ref), dtype=np.int32)
    for j in range(S):
        agree = np.maximum(agree, (M == M[:, j:j + 1]).sum(1))
    return dict(agreement=agree, p_good=np.mean([r["p_good"] for r in runs], 0),
                good_votes=np.sum([r["in_good"] for r in runs], 0), matched=M)
