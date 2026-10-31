"""Stateful celeris2d session: load once, then classify / isac / select / assign / score / export in seconds.

Every step that yields class images is stored as a named *result*; good/bad is set explicitly with
select() (a person or an agent looking at the gallery) - suggest() only proposes.

  s = Session(); s.load("particles.cs", diameter=220)
  s.isac("iso")                      # stable, reproducible groups
  s.show("iso")                      # -> gallery png
  s.select("iso", good=[1, 4, 7])    # human / agent decision
  s.assign("all", refs="iso", seeds=4)   # every particle vs the stable groups (+ free classes)
  s.export("all", "out/", min_p=0.5)     # good / ambiguous / bad .star / .cs
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field

import mrcfile
import numpy as np
import pandas as pd

from .deep2d import class_table, split_two_clusters
from .engine import Classifier2D, Params, release_cache
from .isac import run_isac
from .plots import gallery
from .sources import open_source
from .stability import match_to_reference


def history_rows(history):
    """Compact per-iteration statistics (for the dashboard charts and the saved result)."""
    keys = ("it", "res", "est_res", "changed", "maxprob", "enc", "time", "n")
    def num(v):
        return (round(float(v), 4) if np.isfinite(v) else None) if isinstance(v, float) else v
    return [{k: num(h[k]) for k in keys if k in h} for h in history]


@dataclass
class Result:
    name: str
    kind: str                      # classify | isac | assign
    idx: np.ndarray                # particles covered
    images: np.ndarray             # (K, b, b) class / group averages
    pops: np.ndarray               # particles per class
    conf: np.ndarray               # per-class confidence (classify) or stability (isac)
    res_a: np.ndarray              # per-class resolution estimate (nan for isac)
    labels: np.ndarray             # best class per covered particle (-1 = none)
    psi: np.ndarray
    offsets: np.ndarray
    maxprob: np.ndarray
    loglik: np.ndarray
    p_good_runs: list = field(default_factory=list)  # per seed: (n, K) posteriors, labels matched to run 0
    good: np.ndarray | None = None  # per-class selection (set by select)
    seeded_from: np.ndarray | None = None  # assign: which source group each class started from (-1 = free)
    info: dict = field(default_factory=dict)
    gallery_path: str = ""


class Session:
    def __init__(self, workdir="celeris2d_session", log=print):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.log = log
        self.src = self.clf = None
        self.results: dict[str, Result] = {}
        self.maps: dict[str, dict] = {}   # ab initio 3D maps: name -> summary (files in <workdir>/<name>/)

    # ------------------------------------------------------------------ data
    def load(self, path, diameter=None, max_res=6.0, gpus="auto", subset=0, project_dir=None, seed=0, **params):
        """diameter: mask diameter in A (None: 200, or 1.25 x the particle size of a Celeris Extract job)."""
        from .cli import pick_devices
        t0 = time.time()
        if isinstance(path, str):
            src = open_source(path, project_dir)
        else:                                        # a source object (e.g. a project job's particles, subsetted)
            src, path = path, getattr(path, "path", "<source>")
        diameter = diameter or getattr(src, "mask_diameter_a", 200.0)
        if 0 < subset < src.N:
            src.subset(np.sort(np.random.default_rng(seed).choice(src.N, subset, replace=False)))
        prm = Params(diameter_a=diameter, max_res_a=max_res, seed=seed, **params)
        self.load_args = dict(diameter=diameter, max_res=max_res, gpus=gpus, subset=subset, project_dir=project_dir,
                              seed=seed, **params)
        if self.clf is not None:  # replacing a loaded dataset: free its GPU memory first
            self.clf = None
            import torch
            release_cache()
        self.src, self.path = src, path
        self.clf = Classifier2D(src, prm, devices=pick_devices(gpus, src.N, src.box, src.apix, max_res), log=self.log)
        self.results.clear()
        self.maps.clear()
        restored = self._restore()
        return dict(restored_results=restored, particles=src.N, box=src.box, apix=src.apix, source=src.kind, ctf=not src.optics.no_ctf,
                    working_box=self.clf.bmax, gpus=[str(d) for d in self.clf.devs], seconds=round(time.time() - t0, 1))

    def _need(self):
        if self.clf is None:
            raise RuntimeError("no dataset loaded: call load() first")

    def _particles(self, on):
        """'all' | '<result>' | '<result>:good' | '<result>:bad' | '<result>:ambiguous' -> host indices."""
        if on in (None, "", "all"):
            return np.arange(self.clf.N)
        name, _, part = on.partition(":")
        r = self.results[name]
        if not part:
            return r.idx
        sets = self.sets(name)
        return sets[part]

    # ---------------------------------------------------------- classification
    def classify(self, name, K=50, seeds=1, on="all", init_from=None, free_classes=0, **params):
        """K-class classification of `on`, repeated with `seeds` independent starts (posteriors of later runs are
        matched to run 0, so P(good) and votes can be computed after select()). If clf.stop_requested is set while
        it runs, the classes so far are stored with info["stopped"] = True and resume(name) carries on."""
        self._need()
        idx = self._particles(on)
        for k, v in params.items():
            setattr(self.clf.p, k, v)
        self._pending = dict(name=name, K=K, seeds=seeds, on=on, init_from=init_from, idx=idx, runs=[], base=None,
                             table=None, s=0, seconds=0.0, resume=False)
        return self._advance()

    def resume(self, name):
        """Continue the stopped run `name` (only the most recent run can be continued: the engine holds its state)."""
        pd = getattr(self, "_pending", None)
        if not pd or pd["name"] != name:
            raise ValueError(f"'{name}' cannot be continued (only the last stopped run can)")
        pd["resume"] = True
        return self._advance()

    def _advance(self):
        pd, c = self._pending, self.clf
        name, K, idx, init_from = pd["name"], pd["K"], pd["idx"], pd["init_from"]
        c.stop_requested = False
        t0 = time.time()
        while pd["s"] < pd["seeds"]:
            s = pd["s"]
            if pd["resume"]:
                pd["resume"] = False
                c.run(resume=True)
            else:
                if init_from is not None:
                    c.init_images = init_from
                c.run(active=idx, n_classes=K, seed=c.p.seed + 1000 * s + 7)
            if c.stopped:
                break
            self._collect(pd, s)
            pd["s"] += 1
        stopped = pd["s"] < pd["seeds"]
        if stopped and pd["s"] == 0:   # first start not finished: show its classes as they are now
            self._collect(pd, 0, partial=True)
        pd["seconds"] += time.time() - t0
        t, imgs, base = pd["table"][0], pd["table"][1], pd["base"]
        seeded = None if init_from is None else np.r_[np.arange(len(init_from)), np.full(K - len(init_from), -1)]
        res = Result(name, "classify" if init_from is None else "assign", idx, imgs, t.particles.to_numpy(),
                     t.conf.to_numpy(), t.res_A.to_numpy(), base["labels"], base["psi"], base["offsets"],
                     base["maxprob"], base["loglik"], pd["runs"], seeded_from=seeded,
                     info=dict(K=K, seeds=pd["seeds"], on=pd["on"], seconds=round(pd["seconds"], 1),
                               mode=("hard" if c.p.hard_pose else "soft") + " pose",
                               iterations=len(c.history), params=self._params(), stopped=stopped,
                               history=history_rows(c.history),
                               unclassified=int((~c.seen.numpy()[idx]).sum()) if stopped else 0))
        if not stopped:
            self._pending = None
        elif pd["s"] == 0:
            pd["runs"], pd["table"], pd["base"] = [], None, None   # partial: recomputed when the run finishes
        self._store(res)
        return self.summary(name)

    def _collect(self, pd, s, partial=False):
        """Per-start bookkeeping: run 0 defines the classes; later starts are renumbered to run 0's classes."""
        c, K, idx = self.clf, pd["K"], pd["idx"]
        if s == 0:
            pd["table"] = class_table(c, c.p.diameter_a)
            pd["base"] = self._snap(idx)
        lab = c.cls.numpy()[idx]
        cp = c.cpost.numpy()[idx].astype(np.float32)
        if s > 0 and pd["init_from"] is None:  # independent starts: renumber classes to run 0 by shared members
            from scipy.optimize import linear_sum_assignment
            C = np.zeros((K, K), dtype=np.int64)
            np.add.at(C, (lab, pd["base"]["labels"]), 1)
            r_, c_ = linear_sum_assignment(-C)
            perm = np.arange(K)
            perm[r_] = c_
            cp2 = np.zeros_like(cp)
            cp2[:, perm] = cp
            cp = cp2
        pd["runs"].append(cp)

    def _snap(self, idx):
        c = self.clf
        return dict(labels=c.cls.numpy()[idx].copy(), psi=c.psi.numpy()[idx].copy(),
                    offsets=c.offsets.numpy()[idx].copy(), maxprob=c.maxprob.numpy()[idx].copy(),
                    loglik=c.loglik.numpy()[idx].copy())

    def isac(self, name, on="all", seeds=4, particles_per_class=300, threshold=0.6, max_generations=6):
        """Stable, reproducible groups (see isac.py). Particles never stable get label -1."""
        self._need()
        idx = self._particles(on)
        t0 = time.time()
        out = run_isac(self.clf, idx, seeds=seeds, particles_per_class=particles_per_class, threshold=threshold,
                       max_generations=max_generations, seed=self.clf.p.seed, log=self.log)
        G = out["groups"]
        pos = {p: i for i, p in enumerate(idx)}
        n = len(idx)
        labels = np.full(n, -1)
        psi, off = np.zeros(n, np.float32), np.zeros((n, 2), np.float32)
        mp, ll = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for g in G:
            rows = np.array([pos[m] for m in g["members"]])
            labels[rows], psi[rows], off[rows] = g["id"], g["psi"], g["offsets"]
            mp[rows], ll[rows] = g["maxprob"], g["loglik"]
        res = Result(name, "isac", idx, np.stack([g["image"] for g in G]) if G else np.zeros((0, 2, 2)),
                     np.array([len(g["members"]) for g in G]), np.array([g["stability"] for g in G]),
                     np.full(len(G), np.nan), labels, psi, off, mp, ll,
                     info=dict(seeds=seeds, threshold=threshold, generations=out["generations"],
                               unaccounted=int(len(out["unaccounted"])), seconds=round(time.time() - t0, 1)))
        self._store(res)
        return self.summary(name)

    def assign(self, name, refs, on="all", seeds=4, free_classes=None, **params):
        """Classify `on` starting from every group/class average of result `refs` (good AND bad, so junk has
        somewhere to go) plus free classes for views the references miss. Classes keep the identity of the
        reference they started from, so the selection on `refs` carries over (free classes start unselected)."""
        r = self.results[refs]
        nref = len(r.images)
        free = free_classes if free_classes is not None else max(2, nref // 4)
        out = self.classify(name, K=nref + free, seeds=seeds, on=on, init_from=r.images, **params)
        a = self.results[name]
        a.info["refs"] = refs
        if r.good is not None:
            self.select(name, good=[int(k) + 1 for k in np.nonzero(r.good)[0]])
        return out

    # ------------------------------------------------------------- decisions
    def suggest(self, name):
        """Automatic proposal (2-cluster split on confidence / resolution, dots excluded). Not applied."""
        r = self.results[name]
        if r.kind == "isac":
            t = pd.DataFrame({"particles": r.pops, "conf": r.conf, "res_A": np.full(len(r.pops), 20.0),
                              "peak": self._peak(r.images)})
            t["conf"] = r.conf
        else:
            t = pd.DataFrame({"particles": r.pops, "conf": r.conf, "res_A": r.res_a, "peak": self._peak(r.images)})
        return [int(k) + 1 for k in np.nonzero(split_two_clusters(t))[0]]

    def _peak(self, imgs):
        b = imgs.shape[1]
        y, x = np.mgrid[:b, :b] - b // 2
        inside = np.hypot(x, y) * self.clf.box_a / b < self.clf.p.diameter_a / 2
        return np.array([im[inside].max() / (im[inside].std() + 1e-12) for im in imgs])

    def abinitio(self, name, on, classes=None, **params):
        """3D initial model from result `on` ("<result>" = its populated classes, "<result>:good" = its selected
        classes) and the particles in those classes. Files go to <workdir>/<name>/; returns the summary."""
        from . import abinitio as ab
        self._need()
        rname, _, part = on.partition(":")
        if rname not in self.results:
            raise ValueError(f"no result named {rname}")
        r = self.results[rname]
        if classes is None and part == "good":
            if r.good is None or not r.good.any():
                raise ValueError(f"{rname} has no selected classes")
            classes = r.good
        elif classes is not None:          # 1-based class numbers
            m = np.zeros(len(r.pops), bool)
            m[np.asarray(classes, int) - 1] = True
            classes = m
        prm = ab.AbinitioParams(**params)
        out = os.path.join(self.workdir, name)
        summ = ab.run(self.clf, r, out, prm, classes=classes, log=self.log, src=self.src)
        summ.update(name=name, on=on, input=self.path, n=self.clf.N)
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        self.maps[name] = summ
        return summ

    def select(self, name, good):
        """Mark classes (1-based) as good; all others bad."""
        r = self.results[name]
        g = np.zeros(len(r.images), bool)
        g[[int(k) - 1 for k in good]] = True
        r.good = g
        self._gallery(r)
        self._save(r)
        return self.summary(name)

    def scores(self, name):
        """Per covered particle: p_good (mean over seeds of posterior mass on good classes), votes."""
        r = self.results[name]
        if r.good is None:
            raise RuntimeError(f"select() good classes of '{name}' first")
        if r.kind == "isac":
            p = np.where(r.labels >= 0, r.good[np.maximum(r.labels, 0)].astype(float), np.nan)
            return dict(p_good=p, votes=np.nan_to_num(p))
        pg = np.stack([cp[:, r.good].sum(1) for cp in r.p_good_runs], 0)
        return dict(p_good=pg.mean(0), votes=(pg > 0.5).sum(0))

    def sets(self, name, min_p=0.5, max_p_bad=None):
        """good: p_good >= min_p; bad: p_good < max_p_bad (default 1 - min_p); ambiguous: between / never stable."""
        r = self.results[name]
        sc = self.scores(name)["p_good"]
        lo = (1 - min_p) if max_p_bad is None else max_p_bad
        lo = min(lo, min_p)
        good = r.idx[np.nan_to_num(sc, nan=-1) >= min_p]
        bad = r.idx[np.nan_to_num(sc, nan=2) < lo]
        amb = np.setdiff1d(r.idx, np.union1d(good, bad))
        return dict(good=good, bad=bad, ambiguous=amb)

    # ---------------------------------------------------------------- output
    def _params(self):
        """Scalar settings in effect for the last run (saved with each result)."""
        return {k: v for k, v in dataclasses.asdict(self.clf.p).items() if isinstance(v, (bool, int, float, str)) or v is None}

    def summary(self, name):
        r = self.results[name]
        d = dict(name=name, kind=r.kind, particles=int(len(r.idx)), classes=int(len(r.images)),
                 gallery=r.gallery_path, **{k: v for k, v in r.info.items() if k != "generations"})
        if r.kind == "isac":
            d["generations"] = r.info["generations"]
        d["table"] = [dict(cls=k + 1, particles=int(r.pops[k]), conf=round(float(r.conf[k]), 3),
                           res_A=None if np.isnan(r.res_a[k]) else round(float(r.res_a[k]), 1),
                           good=None if r.good is None else bool(r.good[k]),
                           **({} if r.seeded_from is None else {"seeded_from": int(r.seeded_from[k]) + 1}))
                      for k in range(len(r.images))]
        if r.good is not None:
            s = self.sets(name)
            d["sets"] = {k: int(len(v)) for k, v in s.items()}
        return d

    def _save(self, r):
        """Persist a result (arrays + metadata) so a restarted session / MCP server can pick it up again."""
        arrays = {k: getattr(r, k) for k in ("idx", "images", "pops", "conf", "res_a", "labels", "psi", "offsets",
                                             "maxprob", "loglik")}
        if r.good is not None:
            arrays["good"] = r.good
        if r.seeded_from is not None:
            arrays["seeded_from"] = r.seeded_from
        for i, cp in enumerate(r.p_good_runs):
            arrays[f"cpost{i}"] = cp.astype(np.float16)
        np.savez(os.path.join(self.workdir, f"{r.name}.result.npz"), **arrays)
        with open(os.path.join(self.workdir, f"{r.name}.result.json"), "w") as fh:
            json.dump(dict(name=r.name, kind=r.kind, info=r.info, input=self.path, n=int(self.clf.N)), fh, indent=1,
                      default=str)

    def _restore(self):
        names = []
        for fn in sorted(os.listdir(self.workdir)):
            if not fn.endswith(".result.json"):
                continue
            meta = json.load(open(os.path.join(self.workdir, fn)))
            if meta.get("input") != self.path or meta.get("n") != self.clf.N:
                continue
            z = np.load(os.path.join(self.workdir, f"{meta['name']}.result.npz"))
            runs = [z[k].astype(np.float32) for k in sorted((k for k in z.files if k.startswith("cpost")),
                                                            key=lambda k: int(k[5:]))]
            r = Result(meta["name"], meta["kind"], z["idx"], z["images"], z["pops"], z["conf"], z["res_a"], z["labels"],
                       z["psi"], z["offsets"], z["maxprob"], z["loglik"], runs,
                       good=z["good"] if "good" in z.files else None,
                       seeded_from=z["seeded_from"] if "seeded_from" in z.files else None, info=meta["info"])
            self.results[r.name] = r
            self._gallery(r)
            names.append(r.name)
        for fn in sorted(os.listdir(self.workdir)):
            sp = os.path.join(self.workdir, fn, "summary.json")
            if os.path.isfile(sp):
                meta = json.load(open(sp))
                if meta.get("input") == self.path and meta.get("n") == self.clf.N:
                    self.maps[fn] = meta
                    names.append(fn)
        return names

    def _store(self, r):
        self.results[r.name] = r
        self._gallery(r)
        self._save(r)
        with mrcfile.new(os.path.join(self.workdir, f"{r.name}_classes.mrcs"), overwrite=True) as m:
            m.set_data(r.images.astype(np.float32))
            m.voxel_size = self.clf.box_a / self.clf.bmax

    def _gallery(self, r):
        r.gallery_path = os.path.join(self.workdir, f"{r.name}_classes.png")
        lab = {"classify": "classes", "assign": "classes seeded from references", "isac": "stable groups"}[r.kind]
        gallery(r.gallery_path, r.images, r.pops, self.clf.box_a / self.clf.bmax, r.conf,
                title=f"{r.name} · {lab}", marks=r.good)
        return r.gallery_path

    def export(self, name, outdir, min_p=0.5, sets=("good", "ambiguous", "bad"), fmt="auto"):
        """Write particle sets (with alignments from run 0 of `name`) as .star and/or .cs."""
        from .cli import _write_cs_particles, write_relion_particles
        r = self.results[name]
        os.makedirs(outdir, exist_ok=True)
        pos = {p: i for i, p in enumerate(r.idx)}
        written = {}
        for k, idx in self.sets(name, min_p).items():
            if k not in sets or len(idx) == 0:
                continue
            rows = np.array([pos[p] for p in idx])
            st = dict(idx=idx, cls=np.maximum(r.labels[rows], 0), psi=r.psi[rows], offsets=r.offsets[rows],
                      maxprob=r.maxprob[rows], nsig=np.ones(len(rows), int), loglik=r.loglik[rows])
            fn = f"{name}_{k}"
            if fmt in ("auto", "star") and (self.src.kind != "cryosparc" or fmt == "star"):
                write_relion_particles(outdir, self.clf, self.src, st, fn)
            if fmt in ("auto", "cs") and self.src.kind == "cryosparc":
                _write_cs_particles(outdir, self.clf, self.src, "full", st, fn)
            written[k] = int(len(idx))
        with open(os.path.join(outdir, f"{name}_export.json"), "w") as fh:
            json.dump(dict(result=name, min_p=min_p, written=written, input=self.path), fh, indent=1)
        return dict(outdir=os.path.abspath(outdir), written=written)
