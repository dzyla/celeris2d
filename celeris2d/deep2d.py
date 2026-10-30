"""deep2D: iterative good/bad particle sorting on top of celeris2d.

Round 0 classifies everything. Its good classes (automatic split, or chosen with --good-classes) define a
STANDARD that every later run is judged against, so runs "know each other":
  * absolute cut-offs: confidence >= conf_min, resolution <= res_max, not a dot (peakiness < dot_peak);
  * a template bank of good class averages; each class gets a rotation/mirror-invariant similarity to it.
A class is good if it passes the cut-offs, or if it matches a template strongly and nearly passes them
(rare views can pass on statistics alone, so they are not lost).

Every following round
  1. re-classifies the GOOD set from scratch -> particles confidently in bad classes are purged;
  2. re-classifies BAD + purged, seeded with the good templates plus fresh random classes -> particles
     confidently in good classes are rescued (stops for good once a round rescues < 0.5% of the good set);
  3. good = kept + rescued; the template bank is refreshed from the good-set run.
Particles only move when their assignment is confident (max posterior >= p_move): uncertain ones stay
put, which keeps rounds from oscillating. Stops when < tol of the good set changes, or at max_rounds.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import mrcfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .engine import rot_grid
from .plots import gallery


@dataclass
class Standard:
    conf_min: float
    res_max: float
    sim_strong: float
    dot_peak: float = 6.5
    templates: np.ndarray = field(default=None, repr=False)

    def describe(self):
        return (f"conf >= {self.conf_min:.2f}, res <= {self.res_max:.1f} A, peak < {self.dot_peak}, "
                f"template match >= {self.sim_strong:.2f} for borderline classes, "
                f"{0 if self.templates is None else len(self.templates)} templates")


# ----------------------------------------------------------------------------- class features
def _norm_masked(x, inside):
    x = x * inside
    x = x - (x.sum((-2, -1), keepdim=True) / inside.sum()) * inside
    return x / x.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]


def similarity(imgs, templates, inside, dev, n_rot=72):
    """(K, M) max normalised cross-correlation over rotations, mirror and shifts."""
    A = _norm_masked(torch.as_tensor(imgs, device=dev), inside)
    T = torch.as_tensor(templates, device=dev)
    T = torch.cat([T, T.flip(-1)])                                         # mirror views
    b = A.shape[-1]
    ang = torch.arange(n_rot, device=dev, dtype=torch.float32) * (360.0 / n_rot)
    g = rot_grid(ang, b, b)
    Tr = F.grid_sample(T[None].expand(n_rot, -1, -1, -1), g, align_corners=False)  # (R, 2M, b, b)
    Tr = _norm_masked(Tr.reshape(-1, b, b), inside)
    FA = torch.fft.rfft2(A)
    M2 = T.shape[0]
    best = torch.full((A.shape[0], Tr.shape[0]), -1.0, device=dev)
    for j in range(0, Tr.shape[0], 256):
        FT = torch.fft.rfft2(Tr[j:j + 256])
        cc = torch.fft.irfft2(FA[:, None] * FT[None].conj(), s=(b, b))
        best[:, j:j + 256] = cc.flatten(-2).amax(-1)
    best = best.view(A.shape[0], n_rot, M2).amax(1)                        # (K, 2M)
    M = M2 // 2
    return torch.maximum(best[:, :M], best[:, M:]).cpu().numpy()


def class_table(clf, diameter_a, templates=None):
    """Per-class features of the classification that just ran on clf.active."""
    K = clf.p.n_classes
    act = clf.active.numpy()
    cls = clf.cls.numpy()[act]
    pops = np.bincount(cls, minlength=K)
    conf = np.bincount(cls, weights=clf.maxprob.numpy()[act], minlength=K) / np.maximum(pops, 1)
    imgs = clf.class_images()
    b = imgs.shape[1]
    apix = clf.box_a / b
    y, x = np.mgrid[:b, :b] - b // 2
    inside_np = np.hypot(x, y) * apix < diameter_a / 2
    peak = np.array([im[inside_np].max() / (im[inside_np].std() + 1e-12) for im in imgs])
    t = pd.DataFrame({"class": np.arange(K), "particles": pops, "conf": conf, "res_A": clf.class_res.numpy(),
                      "peak": peak})
    if templates is not None and len(templates):
        inside = torch.as_tensor(inside_np, device=clf.dev, dtype=torch.float32)
        t["sim"] = similarity(imgs, templates, inside, clf.dev).max(1)
    return t, imgs


# ----------------------------------------------------------------------------- judging
def split_two_clusters(t: pd.DataFrame, dot_peak=6.5, min_frac=0.002) -> np.ndarray:
    """Round 0 only: non-dot classes split into two clusters on (confidence, 1/resolution)."""
    n = t.particles.to_numpy()
    cand = (t.peak.to_numpy() < dot_peak) & (n >= max(5, min_frac * n.sum()))
    idx = np.nonzero(cand)[0]
    if len(idx) < 2:
        return cand
    conf, ires = t.conf.to_numpy()[idx], 1.0 / t.res_A.to_numpy()[idx]
    f = np.stack([conf / conf.max(), ires / ires.max()], 1)
    w = n[idx].astype(float)
    score = f.sum(1)
    c_hi, c_lo = f[score.argmax()].copy(), f[score.argmin()].copy()
    hi = np.ones(len(idx), bool)
    for _ in range(50):  # weighted 2-means
        hi = ((f - c_hi) ** 2).sum(1) < ((f - c_lo) ** 2).sum(1)
        if hi.all() or (~hi).all():
            break
        c_hi_new, c_lo_new = np.average(f[hi], 0, weights=w[hi]), np.average(f[~hi], 0, weights=w[~hi])
        if np.allclose(c_hi_new, c_hi) and np.allclose(c_lo_new, c_lo):
            break
        c_hi, c_lo = c_hi_new, c_lo_new
    good = np.zeros(len(t), bool)
    good[idx[hi]] = True
    return good


def learn_standard(t, good, imgs, clf, diameter_a, min_conf=None, max_res=None, dot_peak=6.5) -> Standard:
    g = t[good]
    w = g.particles.to_numpy().astype(float)
    conf_min = min_conf if min_conf is not None else 0.95 * float(np.min(g.conf))
    res_max = max_res if max_res is not None else 1.05 * float(np.max(g.res_A))
    tmpl = imgs[good]
    sim_strong = 0.8
    if good.sum() >= 3:  # leave-one-out similarity among good classes: how alike real views are
        b = imgs.shape[1]
        y, x = np.mgrid[:b, :b] - b // 2
        inside = torch.as_tensor(np.hypot(x, y) * clf.box_a / b < diameter_a / 2, device=clf.dev, dtype=torch.float32)
        s = similarity(tmpl, tmpl, inside, clf.dev)
        np.fill_diagonal(s, -1)
        sim_strong = float(np.clip(np.average(s.max(1), weights=w), 0.5, 0.95))
    return Standard(conf_min=conf_min, res_max=res_max, sim_strong=sim_strong, dot_peak=dot_peak, templates=tmpl)


def judge(t: pd.DataFrame, std: Standard, min_frac=0.002) -> np.ndarray:
    n = t.particles.to_numpy()
    ok = (t.peak.to_numpy() < std.dot_peak) & (n >= max(5, min_frac * n.sum()))
    conf, res = t.conf.to_numpy(), t.res_A.to_numpy()
    stats_pass = (conf >= std.conf_min) & (res <= std.res_max)
    near = (conf >= 0.8 * std.conf_min) & (res <= 1.25 * std.res_max)
    sim = t["sim"].to_numpy() if "sim" in t else np.zeros(len(t))
    return ok & (stats_pass | (near & (sim >= std.sim_strong)))


# ----------------------------------------------------------------------------- one classification
def _classify(clf, idx, K, seed, outdir, title, diameter_a, std=None, seed_templates=False):
    K_eff = int(min(K, max(2, len(idx) // 150)))
    if seed_templates and std is not None and std.templates is not None:
        clf.init_images = std.templates[: max(1, K_eff // 2)]  # good views to snap to; the rest stay free
    clf.run(active=idx, n_classes=K_eff, seed=seed)
    t, imgs = class_table(clf, diameter_a, None if std is None else std.templates)
    return t, imgs


def _save(outdir, title, clf, t, imgs):
    os.makedirs(outdir, exist_ok=True)
    t.to_csv(os.path.join(outdir, "class_stats.csv"), index=False)
    with mrcfile.new(os.path.join(outdir, "classes.mrcs"), overwrite=True) as m:
        m.set_data(imgs)
        m.voxel_size = clf.box_a / clf.bmax
    gallery(os.path.join(outdir, "classes.png"), imgs, t.particles.to_numpy(), clf.box_a / clf.bmax,
            t.conf.to_numpy(), title=title, marks=t.good.to_numpy())


def _particle_calls(clf, idx, good_cls, p_move):
    """Per particle: in a good class?  and is the assignment confident enough to act on?"""
    idx = np.asarray(idx)
    in_good = good_cls[clf.cls.numpy()[idx]]
    confident = clf.maxprob.numpy()[idx] >= p_move
    return in_good, confident


# ----------------------------------------------------------------------------- driver
def run_deep2d(clf, K, outdir, max_rounds=6, tol=0.02, seed=0, diameter_a=200.0, sel_kw=None, good_classes=None,
               p_move=0.5, rescue_stop=0.005, max_flips=3, log=print):
    sel_kw = sel_kw or {}
    t0 = time.time()
    N = clf.N
    everyone = np.arange(N)
    hist = []

    log(f"[deep2D] round 0: classify all {N} particles")
    t, imgs = _classify(clf, everyone, K, seed, None, None, diameter_a)
    # every later run uses round 0's pose/prior mode, so confidence means the same thing in all rounds
    clf._auto_hard = clf._auto_prior = False
    log(f"[deep2D] mode: {'hard' if clf.p.hard_pose else 'soft'} pose, "
        f"{'class' if clf.p.class_prior else 'uniform'} prior (fixed for all rounds)")
    if good_classes:
        good_c = np.zeros(len(t), bool)
        good_c[[g - 1 for g in good_classes]] = True
        log(f"[deep2D] round 0: using hand-picked good classes {sorted(good_classes)}")
    else:
        good_c = split_two_clusters(t, sel_kw.get("dot_peak", 6.5))
    std = learn_standard(t, good_c, imgs, clf, diameter_a, sel_kw.get("min_conf"), sel_kw.get("max_res"))
    log(f"[deep2D] standard: {std.describe()}")
    t["good"] = good_c
    _save(os.path.join(outdir, "round00_all"), "deep2D · round 0 · all particles", clf, t, imgs)
    good = everyone[good_c[clf.cls.numpy()]]
    hist.append(dict(round=0, good=len(good), kept=len(good), purged=0, rescued=0, changed=1.0,
                     time=time.time() - t0))
    log(f"[deep2D] round 0: {len(good)} good / {N - len(good)} bad")
    members = {"round00": good.copy()}        # good set after every round (for evaluation)
    flips = np.zeros(N, dtype=np.int32)       # good<->bad moves per particle
    unstable = np.zeros(0, dtype=int)         # frozen jumpers: excluded from further rounds

    rescuing = True
    for r in range(1, max_rounds + 1):
        # 1. good set on its own, from scratch
        t, imgs = _classify(clf, good, K, seed + 2 * r, None, None, diameter_a, std)
        t["good"] = judge(t, std)
        _save(os.path.join(outdir, f"round{r:02d}_good"), f"deep2D · round {r} · re-classify good ({len(good):,})",
              clf, t, imgs)
        in_good, conf_ok = _particle_calls(clf, good, t.good.to_numpy(), p_move)
        purged = good[~in_good & conf_ok]
        kept = good[in_good | ~conf_ok]
        if t.good.any():  # refresh the template bank from the cleanest run
            std.templates = imgs[t.good.to_numpy()]
        # 2. bad set + purged, seeded with good templates
        rescued = np.zeros(0, dtype=int)
        pool = np.setdiff1d(np.setdiff1d(everyone, kept), unstable)
        if rescuing and len(pool):
            t2, imgs2 = _classify(clf, pool, K, seed + 2 * r + 1, None, None, diameter_a, std, seed_templates=True)
            t2["good"] = judge(t2, std)
            _save(os.path.join(outdir, f"round{r:02d}_bad"),
                  f"deep2D · round {r} · re-classify bad + purged ({len(pool):,})", clf, t2, imgs2)
            in_good2, conf_ok2 = _particle_calls(clf, pool, t2.good.to_numpy(), p_move)
            rescued = pool[in_good2 & conf_ok2]
            if len(rescued) < rescue_stop * max(len(kept), 1):
                rescuing = False
                log(f"[deep2D] round {r}: bad pool exhausted (rescued {len(rescued)}), no more rescue runs")
        new_good = np.union1d(kept, rescued)
        moved = np.setxor1d(new_good, good)
        flips[moved] += 1
        jumpers = moved[flips[moved] >= max_flips]
        if len(jumpers):  # particles that keep jumping have no stable identity: freeze them out
            unstable = np.union1d(unstable, jumpers)
            new_good = np.setdiff1d(new_good, jumpers)
        changed = len(np.setxor1d(new_good, good)) / max(len(good), 1)
        hist.append(dict(round=r, good=len(new_good), kept=len(kept), purged=len(purged), rescued=len(rescued),
                         unstable=len(unstable), changed=changed, time=time.time() - t0))
        log(f"[deep2D] round {r}: kept {len(kept)}, purged {len(purged)}, rescued {len(rescued)}, "
            f"+{len(jumpers)} unstable -> good {len(new_good)} ({changed * 100:.1f}% changed, {time.time() - t0:.0f}s)")
        good = new_good
        members[f"round{r:02d}"] = good.copy()
        # plateau: the good-set size moved < tol in each of the last two rounds (a steady exchange of
        # borderline particles never stops, so set identity is not a usable criterion)
        sizes = [h["good"] for h in hist[-3:]]
        if changed < tol or (len(sizes) == 3 and r >= 2 and
                             max(abs(sizes[1] - sizes[0]), abs(sizes[2] - sizes[1])) < tol * sizes[2]):
            log(f"[deep2D] converged: good-set size stable within {tol * 100:.0f}% (round {r})")
            break

    # final pass on the good set: particles still sitting in bad classes are reported as ambiguous
    t, imgs = _classify(clf, good, K, seed + 999, None, None, diameter_a, std)
    t["good"] = judge(t, std)
    in_good, _ = _particle_calls(clf, good, t.good.to_numpy(), p_move)
    ambiguous = good[~in_good]
    st_amb = _snapshot(clf, ambiguous)
    good = good[in_good]
    st_good = _snapshot(clf, good)
    _save(os.path.join(outdir, "final_good"),
          f"deep2D · final good set ({len(good):,} good + {len(ambiguous):,} ambiguous)", clf, t, imgs)
    bad = np.setdiff1d(everyone, np.union1d(np.union1d(good, ambiguous), unstable))
    if len(bad):
        t, imgs = _classify(clf, bad, K, seed + 998, None, None, diameter_a, std)
        t["good"] = judge(t, std)
        _save(os.path.join(outdir, "final_bad"), f"deep2D · final bad set ({len(bad):,} particles)", clf, t, imgs)
    st_bad = _snapshot(clf, bad)
    st_unst = _snapshot(clf, unstable)  # alignments from the last run each was part of
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(outdir, "deep2d_rounds.csv"), index=False)
    with open(os.path.join(outdir, "deep2d_summary.json"), "w") as fh:
        json.dump(dict(n_total=N, n_good=int(len(good)), n_ambiguous=int(len(ambiguous)),
                       n_unstable=int(len(unstable)), n_bad=int(len(bad)), max_flips=max_flips,
                       standard=std.describe(),
                       rounds=hist, total_time=time.time() - t0), fh, indent=1)
    _plot_rounds(os.path.join(outdir, "deep2d_rounds.png"), hist_df, N)
    np.save(os.path.join(outdir, "flips.npy"), flips)
    np.savez(os.path.join(outdir, "membership.npz"), final_good=good, ambiguous=ambiguous, unstable=unstable,
             bad=bad, **members)
    log(f"[deep2D] done: {len(good)} good / {len(ambiguous)} ambiguous / {len(unstable)} unstable / "
        f"{len(bad)} bad particles in {time.time() - t0:.0f}s")
    return dict(good=st_good, ambiguous=st_amb, unstable=st_unst, bad=st_bad)


def _snapshot(clf, idx):
    idx = np.asarray(idx, dtype=int)
    return dict(idx=idx, cls=clf.cls.numpy()[idx].copy(), psi=clf.psi.numpy()[idx].copy(),
                offsets=clf.offsets.numpy()[idx].copy(), maxprob=clf.maxprob.numpy()[idx].copy(),
                nsig=clf.nsig.numpy()[idx].copy(), loglik=clf.loglik.numpy()[idx].copy())


def _plot_rounds(path, h, N):
    from .plotstyle import themed
    themed(_draw_rounds)(path, h, N)


def _draw_rounds(path, h, N):
    from .plotstyle import T, plt, title, top
    BLUE, ORANGE = T.series[0], T.series[1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(h["round"], h["good"], color=BLUE, lw=2, marker="o", ms=4)
    for r, g in zip(h["round"], h["good"]):
        ax[0].annotate(f"{g:,}", (r, g), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax[0].set(title=f"Good particles per round (of {N:,})", xlabel="round", ylabel="particles")
    ax[1].bar(h["round"][1:] - 0.2, h["purged"][1:], width=0.4, color=ORANGE, label="purged from good")
    ax[1].bar(h["round"][1:] + 0.2, h["rescued"][1:], width=0.4, color=BLUE, label="rescued from bad")
    ax[1].set(title="Movement between sets", xlabel="round", ylabel="particles")
    ax[1].legend()
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.yaxis.grid(True, color=T.grid, lw=0.8)
    fig.subplots_adjust(left=0.08, right=0.98, top=top(fig, 0.6), bottom=0.15, wspace=0.25)
    title(fig, "Deep 2D sorting", f"{N:,} particles")
    fig.savefig(path, dpi=120)
    plt.close(fig)
