"""ISAC-style iterative stable alignment and clustering, built on celeris2d's fast EM engine.

SPHIRE's ISAC finds class averages that are *reproducible*: particles are grouped many times from
independent starts, and only groups whose members keep landing together are accepted. Unstable particles
go back into the pool and are tried again in the next generation with fresh classes. Here every
"independent grouping" is a full celeris2d classification (seconds on the GPU, data already loaded), and
stability is measured softly from the class posteriors instead of from hard labels:

  stability(n) = mean over runs s>0 of  sum_k  P_0(n in k) * P_s(n in match_s(k))

i.e. the probability that run 0 and run s put the particle in the same (matched) group.

Generation loop:
  1. K = pool // particles_per_class classes, `seeds` independent runs on the pool
  2. match every run's classes to run 0 by shared membership (Hungarian), compute stability
  3. a run-0 class becomes a STABLE GROUP if >= min_group of its particles have stability >= threshold;
     its stable members are accounted for (group average rebuilt from exactly those particles)
  4. pool -= accounted particles; stop when a generation accounts for < min_new of all particles
Result: stable groups (each a class average + members) and the unaccounted (never stable) particles.
Good/bad is decided afterwards, by a person or an agent, on the stable group averages.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def _run(clf, pool, K, seed):
    clf.run(active=pool, n_classes=K, seed=seed)
    return dict(labels=clf.cls.numpy()[pool].copy(), cpost=clf.cpost.numpy()[pool].astype(np.float32),
                psi=clf.psi.numpy()[pool].copy(), offsets=clf.offsets.numpy()[pool].copy(),
                maxprob=clf.maxprob.numpy()[pool].copy(), loglik=clf.loglik.numpy()[pool].copy())


def soft_stability(runs, K):
    """Probability that run 0 and run s place each particle in matched groups, averaged over s."""
    ref = runs[0]
    if len(runs) == 1:
        return ref["cpost"].max(1)
    acc = np.zeros(len(ref["labels"]), dtype=np.float32)
    for r in runs[1:]:
        # map run-s class -> run-0 class from the argmax co-occurrence, then compare posteriors
        perm = np.full(K, -1)
        C = np.zeros((K, K), dtype=np.int64)
        np.add.at(C, (r["labels"], ref["labels"]), 1)
        rr, cc = linear_sum_assignment(-C)
        perm[rr] = cc
        cp = np.zeros_like(r["cpost"])
        ok = perm >= 0
        cp[:, perm[ok]] = r["cpost"][:, ok]
        acc += (ref["cpost"] * cp).sum(1)
    return acc / (len(runs) - 1)


def run_isac(clf, idx=None, seeds=4, particles_per_class=300, k_max=100, max_generations=6, threshold=0.6,
             min_group=None, min_new=0.02, seed=0, log=print):
    t0 = time.time()
    idx = np.arange(clf.N) if idx is None else np.asarray(idx)
    N = len(idx)
    pool = idx.copy()
    groups = []
    group_of = np.full(clf.N, -1, dtype=np.int32)
    stab_of = np.zeros(clf.N, dtype=np.float32)
    gens = []
    for g in range(1, max_generations + 1):
        K = int(np.clip(len(pool) // particles_per_class, 2, k_max))
        mg = min_group or max(20, int(0.25 * len(pool) / K))
        runs = [_run(clf, pool, K, seed + 101 * g + s) for s in range(seeds)]
        if g == 1:  # keep the pose/prior mode of the first run for all later ones
            clf._auto_hard = clf._auto_prior = False
        stab = soft_stability(runs, K)
        stab_of[pool] = stab
        lab0 = runs[0]["labels"]
        new = []
        for k in range(K):
            m = (lab0 == k) & (stab >= threshold)
            if m.sum() >= mg:
                new.append((k, np.nonzero(m)[0]))
        if new:  # group averages from exactly the stable members, with run-0 alignments
            rows = np.concatenate([r for _, r in new])
            lab = np.concatenate([np.full(len(r), i) for i, (_, r) in enumerate(new)])
            imgs = clf.average(pool[rows], lab, len(new), psi=runs[0]["psi"][rows], offsets=runs[0]["offsets"][rows])
            for i, (k, r) in enumerate(new):
                gid = len(groups)
                members = pool[r]
                group_of[members] = gid
                groups.append(dict(id=gid, generation=g, members=members, image=imgs[i],
                                   stability=float(stab[r].mean()), psi=runs[0]["psi"][r], offsets=runs[0]["offsets"][r],
                                   maxprob=runs[0]["maxprob"][r], loglik=runs[0]["loglik"][r]))
        accounted = sum(len(r) for _, r in new)
        gens.append(dict(generation=g, pool=len(pool), K=K, new_groups=len(new), accounted=accounted,
                         mean_stability=float(stab.mean()), time=time.time() - t0))
        log(f"[isac] gen {g}: pool {len(pool)}, K={K}, {len(new)} stable groups, {accounted} particles accounted "
            f"(mean stability {stab.mean():.2f}, {time.time() - t0:.0f}s)")
        if accounted:
            pool = np.setdiff1d(pool, np.concatenate([pool[r] for _, r in new]))
        if accounted < min_new * N or len(pool) < 2 * mg:
            log(f"[isac] stop: generation {g} accounted for {accounted / N * 100:.1f}% (< {min_new * 100:.0f}%) "
                f"or pool exhausted")
            break
    log(f"[isac] done: {len(groups)} stable groups covering {int((group_of[idx] >= 0).sum())} / {N} particles, "
        f"{len(pool)} never stable, {time.time() - t0:.0f}s")
    return dict(groups=groups, group_of=group_of, stability=stab_of, unaccounted=pool, generations=gens, idx=idx)
