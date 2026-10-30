"""Evaluate the region network: does the pickable mask keep the real particles, and how much does it remove?

    # project data exported by train_regions.py export (+ labels / train for the split file)
    python scripts/eval_regions.py project --data DATA/dz --weights celeris2d/regions.pt --diameter 140 \\
        --split DATA/regions_split.json --figures 6 --movies 100,383,430 --out EVAL/dz
    # a set with ground-truth coordinates (train_regions.py export-gt)
    python scripts/eval_regions.py gt --data DATA/10005 --weights celeris2d/regions.pt --figures 4 --out EVAL/10005
    # speed: full-size micrograph -> region map, and the candidate filter
    python scripts/eval_regions.py timing --weights celeris2d/regions.pt

project: for the held-out (and the training) micrographs, at several max_bad settings: the fraction of all
candidates / kept picks / 2D-selected / 2D-rejected picks removed, the recall of the refined particles, the area
removed. gt: recall of the ground-truth particles and the area removed. Figures: regions overlay + picks kept vs
removed per micrograph (plots_preprocess.regions_figure).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from celeris2d import regions as rg  # noqa: E402

MAX_BAD = (0.5, 0.6, 0.7, 0.8, 0.9)


def _map(model, g, dark=True) -> rg.RegionMap:
    """RegionMap from an already binned micrograph (apix_y/x in the csv rows)."""
    x = rg.net_input(torch.as_tensor(np.array(g), device="cuda"), dark)
    with torch.no_grad():
        p = torch.softmax(model(x)[0], 0)
    return p


def _rm(model, g, ay, ax, dark=True):
    x = rg.net_input(torch.as_tensor(np.array(g), device="cuda"), dark)
    with torch.no_grad():
        p = torch.softmax(model(x)[0], 0)
    return rg.RegionMap(p, ay * g.shape[0] / p.shape[1], ax * g.shape[1] / p.shape[2], (g.shape[0] * ay, g.shape[1] * ax),
                        rg.fractional_intensity(x, p.shape[-2:]))


def _area_removed(rm: rg.RegionMap, margin_a, max_bad):
    h, w = rm.prob.shape[-2:]
    yy, xx = np.meshgrid((np.arange(h) + 0.5) * rm.apix_y, (np.arange(w) + 0.5) * rm.apix_x, indexing="ij")
    ok = rm.pickable(np.stack([xx.ravel(), yy.ravel()], 1), margin_a, max_bad)
    return 1 - ok.mean()


def _figure(path, g, ay, rm, xy, ok, diameter_a, title, ref=None, ref_ok=None):
    from celeris2d.plots_preprocess import regions_figure
    c = rm.compact()
    regions_figure(path, np.asarray(g), ay, c["cls"], dict(zip(rg.CLASSES, c["area"].tolist())), xy[ok], xy[~ok],
                   diameter_a, title, ref, None if ref_ok is None else ~ref_ok, c["p_ice"] / 255.0)


def project(a):
    model = rg.load_model(a.weights)
    mics = pd.read_csv(f"{a.data}/mics.csv").set_index("movie_id")
    pk = pd.read_csv(f"{a.data}/picks.csv")
    fa = pd.read_csv(f"{a.data}/fates.csv")
    split = json.load(open(a.split)).get(os.path.abspath(a.data), {}) if a.split else {}
    sets = {"held-out": split.get("val"), "training": split.get("train")} if split else {"all": None}
    os.makedirs(a.out, exist_ok=True)
    margin = a.margin if a.margin >= 0 else 0.5 * a.diameter
    show = set(int(x) for x in a.movies.split(",") if x)
    rows = []
    for sname, ids in sets.items():
        ids = sorted(set(fa.movie_id)) if ids is None else [int(k) for k in ids]
        if not ids:
            continue
        n_fig = 0
        acc = {mb: dict(cand=[0, 0], kept=[0, 0], selected=[0, 0], rejected=[0, 0], refined=[0, 0], area=[]) for mb in MAX_BAD}
        t_map = []
        per = []
        for mid in ids:
            m = mics.loc[mid]
            g = np.load(f"{a.data}/mics/{mid:04d}.npy")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rm = _rm(model, g, m.bapix_y, m.bapix_x)
            torch.cuda.synchronize()
            t_map.append(time.perf_counter() - t0)
            p = pk[pk.movie == mid]
            f = fa[fa.movie_id == mid]
            xy = np.stack([p.x_a, p.y_a], 1)
            fxy = np.stack([f.x_a, f.y_a], 1)
            for mb in MAX_BAD:
                ok = rm.pickable(xy, margin, mb)
                fok = rm.pickable(fxy, margin, mb)
                acc[mb]["cand"][0] += int((~ok).sum()); acc[mb]["cand"][1] += len(ok)
                acc[mb]["kept"][0] += int((~ok[p.keep.to_numpy()]).sum()); acc[mb]["kept"][1] += int(p.keep.sum())
                for k in ("selected", "rejected", "refined"):
                    s = f[k].to_numpy()
                    acc[mb][k][0] += int((~fok[s]).sum()); acc[mb][k][1] += int(s.sum())
                acc[mb]["area"].append(_area_removed(rm, margin, mb))
                if mb == a.max_bad:
                    rf = f.refined.to_numpy()
                    per.append(dict(movie=mid, refined=int(rf.sum()), lost=int((~fok[rf]).sum()), cand=len(ok),
                                    removed=int((~ok).sum()), area_removed=acc[mb]["area"][-1], **rm.area()))
                if mb == a.max_bad and (n_fig < a.figures or mid in show):
                    ref = fxy[f.refined.to_numpy()]
                    _figure(f"{a.out}/{sname.replace('-', '')}_{mid:04d}.png", g, m.bapix_y, rm, xy, ok, a.diameter,
                            f"movie {mid} {m.stem} ({sname}) max_bad {mb}", ref, fok[f.refined.to_numpy()])
                    n_fig += 1
        per = pd.DataFrame(per).sort_values("lost", ascending=False)
        per.to_csv(f"{a.out}/{sname}_per_micrograph.csv", index=False, float_format="%.4f")
        print(f"\n{sname}: {len(ids)} micrographs, region map {1e3 * np.median(t_map):.2f} ms median (binned input)")
        print(f"micrographs losing most refined particles at max_bad {a.max_bad}:")
        print(per.head(a.worst).to_string(index=False))
        print(f"{'max_bad':>8} {'cand rm':>9} {'kept rm':>9} {'sel rm':>9} {'rej rm':>9} {'refined recall':>15} {'area rm':>8}")
        for mb in MAX_BAD:
            r = acc[mb]
            def frac(k):
                return r[k][0] / max(r[k][1], 1)
            print(f"{mb:8.2f} {frac('cand'):9.4f} {frac('kept'):9.4f} {frac('selected'):9.4f} {frac('rejected'):9.4f} "
                  f"{1 - frac('refined'):15.4f} {np.mean(r['area']):8.4f}   "
                  f"(refined {r['refined'][1] - r['refined'][0]}/{r['refined'][1]})")
            rows.append(dict(set=sname, max_bad=mb, **{k: frac(k) for k in ("cand", "kept", "selected", "rejected")},
                             refined_recall=1 - frac("refined"), n_refined=r["refined"][1], area_removed=float(np.mean(r["area"]))))
    pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False, float_format="%.4f")


def gt(a):
    model = rg.load_model(a.weights)
    mics = pd.read_csv(f"{a.data}/mics.csv").set_index("stem")
    g_t = pd.read_csv(f"{a.data}/gt.csv")
    os.makedirs(a.out, exist_ok=True)
    diam_a = a.diameter or float(np.median(g_t.diam_px) * mics.apix.iloc[0])
    margin = a.margin if a.margin >= 0 else 0.5 * diam_a
    stems = [s for s in mics.index if mics.loc[s].n_gt > 0]
    if a.split:
        held = json.load(open(a.split)).get(os.path.abspath(a.data), {}).get("val")
        if held is not None:
            stems = [s for s in stems if s in set(held)]
            print(f"held-out micrographs of {a.data}: {len(stems)}")
    acc = {mb: [0, 0, []] for mb in MAX_BAD}
    n_fig = 0
    t_map = []
    per = []
    for st in stems:
        m = mics.loc[st]
        g = np.load(f"{a.data}/mics/{st}.npy")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rm = _rm(model, g, m.bapix_y, m.bapix_x, dark=not a.light)
        torch.cuda.synchronize()
        t_map.append(time.perf_counter() - t0)
        q = g_t[g_t.stem == st]
        xy = np.stack([q.x_px * m.apix, q.y_px * m.apix], 1)
        for mb in MAX_BAD:
            ok = rm.pickable(xy, margin, mb)
            acc[mb][0] += int((~ok).sum()); acc[mb][1] += len(ok)
            acc[mb][2].append(_area_removed(rm, margin, mb))
            if mb == a.max_bad:
                per.append(dict(stem=st, n=len(ok), lost=int((~ok).sum()), area_removed=acc[mb][2][-1], **rm.area()))
            if mb == a.max_bad and (n_fig < a.figures or st in a.stems.split(",")):
                _figure(f"{a.out}/{st}.png", g, m.bapix_y, rm, xy, ok, diam_a, f"{os.path.basename(a.data)} {st} max_bad {mb}",
                        xy, ok)
                n_fig += 1
    print(f"\n{a.data}: {len(stems)} micrographs with ground truth, {acc[MAX_BAD[0]][1]} particles, diameter {diam_a:.0f} A, "
          f"region map {1e3 * np.median(t_map):.2f} ms median (binned input)")
    print(f"{'max_bad':>8} {'recall':>8} {'removed':>8} {'area rm':>8}")
    rows = []
    for mb in MAX_BAD:
        lost, n, ar = acc[mb]
        print(f"{mb:8.2f} {1 - lost / n:8.4f} {lost:8d} {np.mean(ar):8.4f}")
        rows.append(dict(max_bad=mb, recall=1 - lost / n, lost=lost, n=n, area_removed=float(np.mean(ar))))
    pd.DataFrame(rows).to_csv(f"{a.out}/summary.csv", index=False, float_format="%.4f")
    per = pd.DataFrame(per).sort_values("lost", ascending=False)
    per.to_csv(f"{a.out}/per_micrograph.csv", index=False, float_format="%.4f")
    print(f"worst micrographs at max_bad {a.max_bad}:")
    print(per.head(a.worst).to_string(index=False))


def timing(a):
    model = rg.load_model(a.weights)
    print("parameters", sum(p.numel() for p in model.parameters()), torch.cuda.get_device_name())
    for shape, apix in (((2046, 2880), 1.32), ((4096, 4096), 1.0), ((3710, 3838), 1.2156), ((4096, 4096), 1.77)):
        mic = torch.randn(*shape, device="cuda") + 10
        for _ in range(3):
            rm = rg.region_map(mic, apix, model=model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            rm = rg.region_map(mic, apix, model=model)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 20
        xy = np.random.rand(500, 2) * np.array([shape[1], shape[0]]) * apix
        t0 = time.perf_counter()
        for _ in range(20):
            rm.pickable(xy, 70, 0.8)
        torch.cuda.synchronize()
        tp = (time.perf_counter() - t0) / 20
        print(f"{shape[0]}x{shape[1]} at {apix} A/px -> map {tuple(rm.prob.shape[1:])} at {rm.apix:.1f} A/px: "
              f"region_map {1e3 * t:.2f} ms, pickable(500 candidates) {1e3 * tp:.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("project", "gt", "timing"):
        s = sub.add_parser(name)
        s.add_argument("--weights", default=rg.WEIGHTS)
        if name == "timing":
            continue
        s.add_argument("--data", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--figures", type=int, default=4, help="overlay figures for the first N micrographs")
        s.add_argument("--max-bad", type=float, default=0.8, help="the decision drawn in the figures")
        s.add_argument("--margin", type=float, default=-1, help="dilation of the ice (A); -1: half the diameter")
        s.add_argument("--diameter", type=float, default=0.0)
        s.add_argument("--light", action="store_true")
        s.add_argument("--split", default="", help="<weights>_split.json from train_regions.py: held-out only")
        s.add_argument("--film-contrast", type=float, default=None, help="override regions.FILM_CONTRAST (-1: no gate)")
        s.add_argument("--thick-contrast", type=float, default=None, help="override regions.THICK_CONTRAST (-1: no gate)")
        s.add_argument("--worst", type=int, default=6, help="list the N micrographs losing most particles")
        if name == "gt":
            s.add_argument("--stems", default="", help="also draw these micrographs")
        if name == "project":
            s.add_argument("--movies", default="", help="also draw these movie ids")
    a = ap.parse_args()
    for k in ("film_contrast", "thick_contrast"):
        if getattr(a, k, None) is not None:
            setattr(rg, k.upper(), None if getattr(a, k) < 0 else getattr(a, k))
    dict(project=project, gt=gt, timing=timing)[a.cmd](a)


if __name__ == "__main__":
    main()
