"""Train the region network (celeris2d/regions.py) from a project's own picking and 2D / 3D results. No hand labels.

    # 1. on the machine with the project: binned micrographs + pick fates (read-only on the project)
    python scripts/train_regions.py export --project /scratch/celeris2d_projects/dz_2025_10 --out DATA/dz \\
        --mic-job 2 --pick-job 3 --extract-job 4 --select-job 6 --refined-job 10
    #    optional check sets with ground-truth coordinates (CryoPPP layout: micrographs/*.mrc + ground_truth/
    #    particle_coordinates/<stem>.csv with X-Coordinate, Y-Coordinate, Diameter in px)
    python scripts/train_regions.py export-gt --root /scratch/dzyla_datasets/cryoppp/10017 --apix 1.77 --out DATA/10017
    # 2. weak labels from the confirmed particles + two physical features (see celeris2d/regions.py)
    python scripts/train_regions.py labels --data DATA/dz --diameter 140
    python scripts/train_regions.py labels --data DATA/10017 --diameter 190 --auto   # ground truth as confirmed
    # 3. train (held-out micrographs by seed, per folder: DIR[:val_frac]); writes the weights + <out>_split.json
    python scripts/train_regions.py train --data DATA/dz:0.3 DATA/10017:0 --out celeris2d/regions.pt --epochs 100
    # then: python scripts/eval_regions.py ... (recall of refined / ground-truth particles, area removed, figures)

Labels (uint8 map at the binned pixel size, 255 = unlabelled):
    0 ice            within 0.6 diameters of a particle kept by the Select job or present in the refined job
    1 film           local mean > +10 % above the intensity level at the confirmed particles, flat (local std < 1.3x
                     the noise floor), no confirmed particle within 2 diameters
    2 thick          local mean < -14 % below that level, no confirmed particle within 2 diameters
    3 edge           particle-scale local std > 2.5x the micrograph's noise floor
    4 contamination  dark compact blob: difference of Gaussians (3 px vs 24 px) < -6 % of the mean intensity
The +10 % / -14 % / 2.5x cut-offs are the 99.9th / 0.1st / 99th percentiles of the same features at the 27k
confirmed particles of dz_2025_10 (scripts/train_regions.py stats). Micrographs with fewer than 5 confirmed
particles get no film / thick labels (no level to compare to).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from celeris2d import regions as rg  # noqa: E402

WORK = rg.WORK_APIX


# ------------------------------------------------------------------------------------------------------ export
def _fbin(img, apix):
    t = torch.as_tensor(np.array(img, np.float32))
    g, ay, ax = rg.bin_micrograph(t, apix, WORK)
    return g.numpy().astype(np.float32), ay, ax


def export(a):
    import mrcfile
    from celeris2d.project import Project
    p = Project(a.project)
    os.makedirs(a.out + "/mics", exist_ok=True)
    mi = p.micrographs(a.mic_job)
    rows = []
    for r in mi.itertuples():
        f = f"{a.out}/mics/{int(r.movie_id):04d}.npy"
        if not os.path.exists(f):
            with mrcfile.open(p.abs(r.path), permissive=True) as m:
                g, ay, ax = _fbin(m.data, float(r.apix))
            np.save(f, g)
        else:
            g = np.load(f, mmap_mode="r")
            ay, ax = float(r.apix) * r.ny / g.shape[0], float(r.apix) * r.nx / g.shape[1]
        rows.append(dict(movie_id=int(r.movie_id), stem=r.stem, apix=float(r.apix), ny=int(r.ny), nx=int(r.nx),
                         bapix_y=ay, bapix_x=ax, by=int(g.shape[0]), bx=int(g.shape[1])))
    pd.DataFrame(rows).to_csv(f"{a.out}/mics.csv", index=False)
    z = p.table(p.job(a.pick_job), "picks").read()
    pd.DataFrame(dict(z)).to_csv(f"{a.out}/picks.csv", index=False)
    ex = p.particles(a.extract_job)[["uid", "x_a", "y_a", "movie_id"]]
    ex["selected"] = ex.uid.isin(set(p.selected_uids(a.select_job).tolist()))
    ex["rejected"] = ex.uid.isin(set(p.selected_uids(a.select_job, rejected=True).tolist()))
    ex["refined"] = ex.uid.isin(set(p.particles(a.refined_job)["uid"].tolist())) if a.refined_job else False
    ex.to_csv(f"{a.out}/fates.csv", index=False)
    print(f"{len(rows)} micrographs, {len(z['score'])} candidates, {len(ex)} picks: selected {int(ex.selected.sum())}, "
          f"rejected {int(ex.rejected.sum())}, refined {int(ex.refined.sum())}")


def export_gt(a):
    import mrcfile
    os.makedirs(a.out + "/mics", exist_ok=True)
    rows, coords = [], []
    for f in sorted(glob.glob(a.root + "/micrographs/*.mrc")):
        stem = os.path.splitext(os.path.basename(f))[0]
        o = f"{a.out}/mics/{stem}.npy"
        with mrcfile.open(f, permissive=True, header_only=True) as m:
            ny, nx = int(m.header.ny), int(m.header.nx)
        if not os.path.exists(o):
            with mrcfile.open(f, permissive=True) as m:
                g, ay, ax = _fbin(m.data, a.apix)
            np.save(o, g)
        else:
            g = np.load(o, mmap_mode="r")
            ay, ax = a.apix * ny / g.shape[0], a.apix * nx / g.shape[1]
        csv = f"{a.root}/ground_truth/particle_coordinates/{stem}.csv"
        n = 0
        if os.path.exists(csv):
            c = pd.read_csv(csv)
            n = len(c)
            coords.append(pd.DataFrame(dict(stem=stem, x_px=c["X-Coordinate"], y_px=c["Y-Coordinate"],
                                            diam_px=c["Diameter"] if "Diameter" in c else 0)))
        rows.append(dict(stem=stem, apix=a.apix, ny=ny, nx=nx, bapix_y=ay, bapix_x=ax, by=int(g.shape[0]),
                         bx=int(g.shape[1]), n_gt=n))
    pd.DataFrame(rows).to_csv(f"{a.out}/mics.csv", index=False)
    pd.concat(coords).to_csv(f"{a.out}/gt.csv", index=False)
    print(a.out, len(rows), "micrographs,", sum(r["n_gt"] > 0 for r in rows), "with ground truth,",
          sum(r["n_gt"] for r in rows), "particles")


# ------------------------------------------------------------------------------------------------------ labels
def gauss(x: torch.Tensor, s: float) -> torch.Tensor:
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    w = torch.exp(-0.5 * (t / s) ** 2)
    w = w / w.sum()
    y = F.conv2d(F.pad(x[None, None], (k // 2,) * 4, mode="reflect"), w[None, None, None, :])
    return F.conv2d(y, w[None, None, :, None])[0, 0]


def features(g: np.ndarray, dev="cuda"):
    """Fractional intensity r, its local mean fi (16 px), particle-scale local std / noise floor sr, DoG."""
    x = torch.as_tensor(np.array(g), device=dev, dtype=torch.float32)
    r = x / x.mean() - 1
    fi = gauss(r, 16)
    hp = x - gauss(x, 6)
    s = gauss(hp * hp, 10).sqrt()
    s0 = torch.quantile(s[10:-10, 10:-10].flatten()[::7], 0.05)
    return r, fi, s / s0, gauss(r, 3) - gauss(r, 24)


def weak_labels(g: np.ndarray, pos_px: np.ndarray, diam_px: float, dev="cuda", dark: bool = True,
                film=0.10, thick=-0.14, edge=2.5, cont=-0.06):
    r, fi, sr, dog = features(g, dev)
    if not dark:
        fi, dog = -fi, -dog
    h, w = fi.shape
    near = torch.zeros(h, w, dtype=torch.bool, device=dev)
    near2 = torch.zeros_like(near)
    conf = len(pos_px) >= 5
    lvl = fi.median()
    if len(pos_px):
        yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        p = torch.as_tensor(np.asarray(pos_px), device=dev, dtype=torch.float32)
        dmin = torch.full((h, w), 1e9, device=dev)
        for i in range(0, len(p), 64):                      # chunks: (64, h, w) distance maps
            q = p[i:i + 64]
            d2 = (yy[None].float() - q[:, 1, None, None]) ** 2 + (xx[None].float() - q[:, 0, None, None]) ** 2
            dmin = torch.minimum(dmin, d2.min(0).values.sqrt())
        near, near2 = dmin < 0.6 * diam_px, dmin < 2.0 * diam_px
        pi = p.round().long().clamp(min=0)
        pi[:, 0].clamp_(max=w - 1)
        pi[:, 1].clamp_(max=h - 1)
        if conf:
            lvl = fi[pi[:, 1], pi[:, 0]].median()
    dm = fi - lvl
    lab = torch.full((h, w), 255, dtype=torch.uint8, device=dev)
    if conf:
        lab[(dm > film) & (sr < 1.3)] = 1
        lab[dm < thick] = 2
    lab[sr > edge] = 3
    lab[dog < cont] = 4
    lab[near2 & (lab < 3)] = 255
    lab[near] = 0
    return lab.cpu().numpy()


def _keys(data: str):
    """Micrograph keys of a data folder: movie ids (project export) or stems (export-gt), with positions (A) of the
    confirmed particles (Select-kept / refined picks, or the ground truth) and the binned pixel sizes."""
    mics = pd.read_csv(f"{data}/mics.csv")
    out = []
    if os.path.exists(f"{data}/fates.csv"):
        fa = pd.read_csv(f"{data}/fates.csv")
        conf = fa[fa.selected | fa.refined]
        mics = mics.set_index("movie_id")
        for mid in sorted(set(fa.movie_id)):
            c = conf[conf.movie_id == mid]
            m = mics.loc[mid]
            out.append((f"{mid:04d}", np.stack([c.x_a, c.y_a], 1), m.bapix_y, m.bapix_x, m.apix))
    else:
        gt = pd.read_csv(f"{data}/gt.csv")
        mics = mics.set_index("stem")
        for st in sorted(set(gt.stem)):
            q = gt[gt.stem == st]
            m = mics.loc[st]
            out.append((st, np.stack([q.x_px * m.apix, q.y_px * m.apix], 1), m.bapix_y, m.bapix_x, m.apix))
    return out


def _cutoffs(data: str, dev="cuda") -> dict:
    """Per-dataset cut-offs: the 99.9th / 0.1st percentile of the local mean (relative to the level), the 99th of
    the local std ratio and the 0.1st of the DoG over all confirmed particles (micrographs with >= 5), floored at
    +3 % / -4 % / 2.0x / -3 % so a very uniform dataset does not label small gradients."""
    vals = []
    for key, xy_a, ay, ax, apix in _keys(data):
        if len(xy_a) < 5:
            continue
        r, fi, sr, dog = features(np.load(f"{data}/mics/{key}.npy"), dev)
        xi = np.clip(np.round(xy_a[:, 0] / ax).astype(int), 0, fi.shape[1] - 1)
        yi = np.clip(np.round(xy_a[:, 1] / ay).astype(int), 0, fi.shape[0] - 1)
        v = np.stack([fi[yi, xi].cpu().numpy(), sr[yi, xi].cpu().numpy(), dog[yi, xi].cpu().numpy()], 1)
        v[:, 0] -= np.median(v[:, 0])
        vals.append(v)
    v = np.concatenate(vals)
    return dict(film=max(float(np.percentile(v[:, 0], 99.9)), 0.03), thick=min(float(np.percentile(v[:, 0], 0.1)), -0.04),
                edge=max(float(np.percentile(v[:, 1], 99)), 2.0), cont=min(float(np.percentile(v[:, 2], 0.1)), -0.03))


def labels(a):
    os.makedirs(f"{a.data}/labels", exist_ok=True)
    cut = _cutoffs(a.data) if a.auto else dict(film=0.10, thick=-0.14, edge=2.5, cont=-0.06)
    print("cut-offs:", {k: round(v, 3) for k, v in cut.items()})
    tot = np.zeros(6)
    for key, xy_a, ay, ax, apix in _keys(a.data):
        g = np.load(f"{a.data}/mics/{key}.npy")
        pos = np.stack([xy_a[:, 0] / ax, xy_a[:, 1] / ay], 1)
        lab = weak_labels(g, pos, a.diameter / ax, dark=not a.light, **cut)
        np.save(f"{a.data}/labels/{key}.npy", lab)
        tot += np.bincount(np.where(lab == 255, 5, lab).ravel(), minlength=6)
    fr = tot / tot.sum()
    print("label fractions:", " ".join(f"{k} {v:.3f}" for k, v in zip(rg.CLASSES + ("unlabelled",), fr)))


def stats(a):
    """Percentiles of the label features at confirmed / rejected picks: where the cut-offs come from."""
    mics = pd.read_csv(f"{a.data}/mics.csv").set_index("movie_id")
    fa = pd.read_csv(f"{a.data}/fates.csv")
    rows = {"confirmed": [], "rejected": []}
    for mid in sorted(set(fa.movie_id)):
        m = mics.loc[mid]
        r, fi, sr, dog = features(np.load(f"{a.data}/mics/{mid:04d}.npy"))
        f = fa[fa.movie_id == mid]
        c = f[f.selected | f.refined]
        if len(c) < 5:
            continue
        def at(df, t):
            xi = (df.x_a / m.bapix_x).round().astype(int).clip(0, t.shape[1] - 1).to_numpy()
            yi = (df.y_a / m.bapix_y).round().astype(int).clip(0, t.shape[0] - 1).to_numpy()
            return t[yi, xi].cpu().numpy()
        lvl = np.median(at(c, fi))
        for k, df in (("confirmed", c), ("rejected", f[f.rejected])):
            rows[k].append(np.stack([at(df, fi) - lvl, at(df, sr), at(df, dog)], 1))
    for k, v in rows.items():
        v = np.concatenate(v)
        for j, name in enumerate(("local mean - level", "local std / noise", "DoG")):
            print(f"{k:9s} {name:20s}", np.round(np.percentile(v[:, j], [0.1, 1, 5, 50, 95, 99, 99.9]), 3))


# ------------------------------------------------------------------------------------------------------- train
def _augment(g: torch.Tensor, lab: torch.Tensor, rng: torch.Generator, crop: int):
    """Random scale (0.8-1.25), crop, flips / rot90, gain, extra noise, a gentle gradient."""
    dev = g.device
    s = float(torch.empty(1).uniform_(0.8, 1.25, generator=rng))
    h, w = g.shape
    nh, nw = max(crop, int(h * s)), max(crop, int(w * s))
    if abs(s - 1) > 0.02:
        g = F.interpolate(g[None, None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=s < 1)[0, 0]
        lab = F.interpolate(lab[None, None].float(), size=(nh, nw), mode="nearest")[0, 0].to(torch.uint8)
    h, w = g.shape
    y0 = int(torch.randint(0, h - crop + 1, (1,), generator=rng))
    x0 = int(torch.randint(0, w - crop + 1, (1,), generator=rng))
    g, lab = g[y0:y0 + crop, x0:x0 + crop], lab[y0:y0 + crop, x0:x0 + crop]
    k = int(torch.randint(0, 4, (1,), generator=rng))
    g, lab = torch.rot90(g, k, (0, 1)), torch.rot90(lab, k, (0, 1))
    if int(torch.randint(0, 2, (1,), generator=rng)):
        g, lab = g.flip(1), lab.flip(1)
    mean = g.mean()
    noise = float(torch.empty(1).uniform_(0, 1.0, generator=rng)) * g.std()
    g = g + torch.randn(g.shape, device=dev) * noise
    yy = torch.linspace(-1, 1, crop, device=dev)[:, None]
    xx = torch.linspace(-1, 1, crop, device=dev)[None, :]
    ga, gb = (float(torch.empty(1).uniform_(-0.03, 0.03, generator=rng)) for _ in range(2))
    g = g * (1 + ga * yy + gb * xx) * float(torch.empty(1).uniform_(0.5, 2.0, generator=rng))
    return g, lab.contiguous()


def train(a):
    """--data DIR[:val_frac] ...: each folder's micrographs are split by the seed (its own fraction held out, default
    --val-frac); the held-out keys go to <out>_split.json for eval_regions.py."""
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)
    data, train_ids, val_ids = {}, [], []
    for spec in a.data:
        d, _, vf = spec.partition(":")
        vf = float(vf) if vf else a.val_frac
        ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{d}/labels/*.npy"))
        perm = np.random.default_rng(a.seed).permutation(ids)
        n_val = int(round(len(ids) * vf))
        val_ids += [(d, k) for k in sorted(perm[:n_val].tolist())]
        train_ids += [(d, k) for k in sorted(perm[n_val:].tolist())]
        for k in ids:
            data[(d, k)] = (torch.as_tensor(np.load(f"{d}/mics/{k}.npy"), device=dev),
                            torch.as_tensor(np.load(f"{d}/labels/{k}.npy"), device=dev))
    cnt = np.zeros(len(rg.CLASSES))
    for d in sorted({d for d, _ in train_ids}):                        # class frequencies: every folder counts once
        c = np.zeros(len(rg.CLASSES))
        for m in train_ids:
            if m[0] == d:
                c += np.bincount(data[m][1].cpu().numpy().ravel(), minlength=256)[:len(rg.CLASSES)]
        cnt += c / c.sum()
    wts = torch.as_tensor(1 / np.sqrt(cnt / cnt.sum() + 1e-4), device=dev, dtype=torch.float32)
    wts = wts / wts.mean()
    print("train", len(train_ids), "val", len(val_ids), "label pixels", cnt.astype(int).tolist(),
          "weights", [round(float(x), 2) for x in wts])
    model = rg.RegionNet().to(dev)
    print("parameters", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(train_ids) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, steps), pct_start=0.1)
    t0 = time.time()
    step = 0
    best, best_sd = -1.0, None
    for ep in range(a.epochs):
        model.train()
        order = np.random.default_rng(a.seed + ep).permutation(len(train_ids))
        tl, tn = 0.0, 0
        for i in range(0, len(order) - a.batch + 1, a.batch):
            xs, ys = [], []
            for m in (train_ids[j] for j in order[i:i + a.batch]):
                g, lab = _augment(*data[m], rng, a.crop)
                xs.append(rg.net_input(g, True)[0])
                ys.append(lab)
            x, y = torch.stack(xs), torch.stack(ys).long()
            y = F.interpolate(y[:, None].float(), size=(a.crop // 2, a.crop // 2), mode="nearest")[:, 0].long()
            y[y == 255] = -1
            loss = F.cross_entropy(model(x), y, weight=wts, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tl += float(loss.detach())
            tn += 1
        if (ep + 1) % max(1, a.epochs // 20) == 0 or ep == a.epochs - 1:
            acc = evaluate(model, data, val_ids or train_ids[:50])
            score = 2 * float(acc["ice>=0.2"]) + np.mean([float(acc[k]) for k in rg.CLASSES[1:]])
            if score > best:                                              # keep the best epoch by the held-out score
                best, best_sd = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep + 1:3d}  loss {tl / max(tn, 1):.4f}  val: " + "  ".join(f"{k} {v}" for k, v in acc.items())
                  + f"  score {score:.3f}  {time.time() - t0:.0f} s", flush=True)
    sd = best_sd
    torch.save(dict(model=sd, classes=rg.CLASSES, work_apix=WORK, widths=tuple(model.widths), epochs=a.epochs,
                    seed=a.seed, data=[os.path.abspath(d.partition(":")[0]) for d in a.data]), a.out)
    split = {}
    for name, ids in (("train", train_ids), ("val", val_ids)):
        for d, k in ids:
            split.setdefault(os.path.abspath(d), {}).setdefault(name, []).append(k)
    json.dump(split, open(os.path.splitext(a.out)[0] + "_split.json", "w"), indent=0)
    print("saved", a.out)


@torch.no_grad()
def evaluate(model, data, ids) -> dict:
    """Per-class recall of the weak labels on held-out micrographs, and the fraction of ice-labelled pixels
    (confirmed particles) with P(ice) >= 0.2 (the lenient decision)."""
    model.eval()
    hit = np.zeros(len(rg.CLASSES))
    tot = np.zeros(len(rg.CLASSES))
    lenient = [0, 0]
    for m in ids:
        g, lab = data[m]
        p = torch.softmax(model(rg.net_input(g, True))[0], 0)
        p = F.interpolate(p[None], size=lab.shape, mode="bilinear", align_corners=False)[0]
        c = p.argmax(0)
        for i in range(len(rg.CLASSES)):
            sel = lab == i
            tot[i] += float(sel.sum())
            hit[i] += float((c[sel] == i).sum())
        ice = lab == 0
        lenient[0] += float((p[0][ice] >= 0.2).sum())
        lenient[1] += float(ice.sum())
    out = {k: f"{hit[i] / max(tot[i], 1):.3f}" for i, k in enumerate(rg.CLASSES)}
    out["ice>=0.2"] = f"{lenient[0] / max(lenient[1], 1):.4f}"
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--project", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--mic-job", type=int, required=True)
    e.add_argument("--pick-job", type=int, required=True)
    e.add_argument("--extract-job", type=int, required=True)
    e.add_argument("--select-job", type=int, required=True)
    e.add_argument("--refined-job", type=int, default=0)
    g = sub.add_parser("export-gt")
    g.add_argument("--root", required=True)
    g.add_argument("--apix", type=float, required=True)
    g.add_argument("--out", required=True)
    for name in ("labels", "stats"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--diameter", type=float, default=140.0, help="particle diameter (A) of the pick job")
        s.add_argument("--light", action="store_true", help="particles brighter than the background")
        s.add_argument("--auto", action="store_true", help="cut-offs from this dataset's own confirmed particles")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, nargs="+", help="DIR[:val_frac] ...")
    t.add_argument("--out", default="regions.pt")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--crop", type=int, default=192)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-frac", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dict(export=export, labels=labels, stats=stats, train=train)[a.cmd.replace("-", "_")](a) if a.cmd != "export-gt" \
        else export_gt(a)


if __name__ == "__main__":
    main()
