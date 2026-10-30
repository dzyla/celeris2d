"""celeris2d command line: RELION particles.star in, classes + aligned particles.star out."""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:i == nvidia-smi index i

import mrcfile
import numpy as np
import pandas as pd

from .engine import Classifier2D, Params
from .sources import load_particles, open_source  # noqa: F401  (load_particles re-exported for scripts)
from .plots import gallery, stats
from .star import write_star

# internal model: particle = shift_t(rot_psi(class)), psi counter-clockwise in (x=col, y=row).
# RELION: psi_relion = -psi, origin = -t (verified by rebuilding RELION Class2D averages).
# cryoSPARC alignments2D: pose = CS_POSE_SIGN * psi [rad], shift = CS_SHIFT_SIGN * t [px]
# (verified by rebuilding cryoSPARC 2D class averages, see README).
CS_POSE_SIGN = -1.0
CS_SHIFT_SIGN = -1.0


def write_outputs(outdir, clf: Classifier2D, src, in_path, particles: str = "slim"):
    """particles: 'slim' (cryoSPARC: uid + alignments only, linked to the input metadata; RELION: star),
    'full' (complete .cs copy + particles.star), 'none' (classes and statistics only)."""
    os.makedirs(outdir, exist_ok=True)
    is_cs = getattr(src, "kind", "") == "cryosparc"
    K = clf.p.n_classes
    imgs = clf.class_images()
    apix_out = clf.box_a / clf.bmax
    with mrcfile.new(os.path.join(outdir, "classes.mrcs"), overwrite=True) as m:
        m.set_data(imgs)
        m.voxel_size = apix_out

    cls = clf.cls.numpy()
    pops = np.bincount(cls, minlength=K)
    if particles == "full" or (particles == "slim" and not is_cs):
        write_relion_particles(outdir, clf, src)
    _write_rest(outdir, clf, src, in_path, imgs, pops, cls, apix_out, K, particles, is_cs)


def snapshot(clf, idx=None) -> dict:
    """Per-particle alignment state for particles idx (default: all)."""
    idx = np.arange(clf.N) if idx is None else np.asarray(idx)
    return dict(idx=idx, cls=clf.cls.numpy()[idx].copy(), psi=clf.psi.numpy()[idx].copy(),
                offsets=clf.offsets.numpy()[idx].copy(), maxprob=clf.maxprob.numpy()[idx].copy(),
                nsig=clf.nsig.numpy()[idx].copy(), loglik=clf.loglik.numpy()[idx].copy())


def write_relion_particles(outdir, clf, src, st=None, name="particles"):
    st = st or snapshot(clf)
    df = src.parts.iloc[st["idx"]].reset_index(drop=True)
    df["rlnClassNumber"] = (st["cls"] + 1).astype(int)
    # internal model: particle = shift_t(rot_psi(class)); RELION aligns by translating the image
    # by +origin and its projector convention gives psi_relion = -psi (validated, see README)
    df["rlnAnglePsi"] = (-st["psi"] + 180.0) % 360.0 - 180.0
    df["rlnOriginXAngst"] = -st["offsets"][:, 0]
    df["rlnOriginYAngst"] = -st["offsets"][:, 1]
    df["rlnMaxValueProbDistribution"] = st["maxprob"]
    df["rlnNrOfSignificantSamples"] = st["nsig"].astype(int)
    df["rlnLogLikeliContribution"] = st["loglik"]
    out_blocks = {k: v for k, v in src.blocks.items() if k != "particles"}
    out_blocks["particles"] = df
    write_star(os.path.join(outdir, f"{name}.star"), out_blocks)


def _write_rest(outdir, clf, src, in_path, imgs, pops, cls, apix_out, K, particles, is_cs):
    model = pd.DataFrame({
        "rlnReferenceImage": [f"{k + 1:06d}@{os.path.join(outdir, 'classes.mrcs')}" for k in range(K)],
        "rlnClassDistribution": pops / pops.sum(),
        "rlnClassParticles": pops.astype(int),
        "rlnAccuracyRotations": np.full(K, 360.0 / clf.history[-1]["n_rot"]),
    })
    write_star(os.path.join(outdir, "model.star"), {"model_classes": model})
    with open(os.path.join(outdir, "history.json"), "w") as fh:
        json.dump(dict(input=in_path, params=vars(clf.p), total_time=clf.total_time, history=clf.history), fh, indent=1)
    conf = np.bincount(cls, weights=clf.maxprob.numpy(), minlength=K) / np.maximum(pops, 1)
    gallery(os.path.join(outdir, "classes.png"), imgs, pops, apix_out, conf,
            title=f"Celeris2D · {os.path.basename(os.path.abspath(outdir))}")
    stats(os.path.join(outdir, "stats.png"), clf.history, pops, conf,
          title=f"{os.path.basename(os.path.abspath(outdir))}: {clf.N} particles, K={K}, "
                f"{clf.total_time:.0f} s on {len(clf.devs)} GPU(s)")
    pd.DataFrame({"class": np.arange(1, K + 1), "particles": pops, "fraction": pops / pops.sum(),
                  "mean_max_posterior": conf, "resolution_A": clf.class_res.numpy()}).to_csv(
        os.path.join(outdir, "class_stats.csv"), index=False)
    if is_cs:
        write_cryosparc(outdir, clf, src, imgs, apix_out, particles)


_A2D = [("alignments2D/split", "<u4"), ("alignments2D/shift", "<f4", (2,)), ("alignments2D/pose", "<f4"),
        ("alignments2D/psize_A", "<f4"), ("alignments2D/error", "<f4"), ("alignments2D/error_min", "<f4"),
        ("alignments2D/resid_pow", "<f4"), ("alignments2D/slice_pow", "<f4"), ("alignments2D/image_pow", "<f4"),
        ("alignments2D/cross_cor", "<f4"), ("alignments2D/alpha", "<f4"), ("alignments2D/alpha_min", "<f4"),
        ("alignments2D/weight", "<f4"), ("alignments2D/pose_ess", "<f4"), ("alignments2D/shift_ess", "<f4"),
        ("alignments2D/class_posterior", "<f4"), ("alignments2D/class", "<u4"), ("alignments2D/class_ess", "<f4")]


def _write_csg(path, name, title, gtype, links, n):
    """links: {result group: metafile path (relative to the .csg, or absolute)}"""
    import datetime
    lines = [f"created: {datetime.datetime.now(datetime.timezone.utc)}", "group:",
             f"  description: {title} (Celeris2D)", f"  name: {name}", f"  title: {title}", f"  type: {gtype}",
             "results:"]
    for g, mf in sorted(links.items()):
        lines += [f"  {g}:", f"    metafile: '>{mf}'", f"    num_items: {n}", f"    type: {gtype}.{g}"]
    lines.append("version: v5.0.6")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_cryosparc(outdir, clf, src, imgs, apix_out, particles="slim"):
    """class_averages.mrc/.cs/.csg and particles.cs/.csg with alignments2D.
    slim: particles.cs holds uid + alignments2D only; particles.csg links blob/ctf/... to the input metafiles
    (absolute paths), so nothing is duplicated. full: a complete copy of every input field."""
    cs = src.cs
    if particles == "slim" and src.subsetted:
        particles = "full"  # input metafiles hold all rows; a subset needs its own copy
    if particles != "none":
        _write_cs_particles(outdir, clf, src, particles)
    _write_cs_classes(outdir, clf, imgs, apix_out)


def _write_cs_particles(outdir, clf, src, particles, st=None, name="particles"):
    st = st or snapshot(clf)
    if len(st["idx"]) != src.N:
        particles = "full"  # a subset cannot be linked row-for-row to the input metafiles
    cs = src.cs[st["idx"]]
    keep = [d for d in cs.dtype.descr if not d[0].startswith("alignments2D/")]
    if particles == "slim":
        keep = [d for d in keep if d[0] == "uid"]
    out = np.zeros(len(cs), dtype=keep + _A2D)
    for d in keep:
        out[d[0]] = cs[d[0]]
    psize = float(src.apix)
    out["alignments2D/class"] = st["cls"].astype(np.uint32)
    out["alignments2D/pose"] = CS_POSE_SIGN * np.deg2rad(st["psi"])
    out["alignments2D/shift"] = CS_SHIFT_SIGN * st["offsets"] / psize
    out["alignments2D/psize_A"] = psize
    out["alignments2D/class_posterior"] = st["maxprob"]
    out["alignments2D/class_ess"] = st["nsig"]
    out["alignments2D/error"] = -st["loglik"]
    out["alignments2D/alpha"] = 1.0
    out["alignments2D/split"] = (np.arange(len(cs)) % 2).astype(np.uint32)
    with open(os.path.join(outdir, f"{name}.cs"), "wb") as fh:
        np.save(fh, out)
    if particles == "slim":
        links = {g: mf for g, mf in src.cs_groups.items() if g != "alignments2D"}
        links["alignments2D"] = f"{name}.cs"
    else:
        links = {g: f"{name}.cs" for g in {n.split("/")[0] for n in out.dtype.names if "/" in n}}
    _write_csg(os.path.join(outdir, f"{name}.csg"), name, name.replace("_", " ").capitalize(), "particle", links,
               len(out))


def _write_cs_classes(outdir, clf, imgs, apix_out):

    K, b = imgs.shape[0], imgs.shape[1]
    with mrcfile.new(os.path.join(outdir, "class_averages.mrc"), overwrite=True) as m:
        m.set_data(imgs)
        m.voxel_size = apix_out
    ca = np.zeros(K, dtype=[("uid", "<u8"), ("blob/path", f"|S{len(outdir) + 32}"), ("blob/idx", "<u4"),
                            ("blob/shape", "<u4", (2,)), ("blob/psize_A", "<f4"), ("blob/res_A", "<f4")])
    ca["uid"] = np.random.default_rng(0).integers(1, 2 ** 63, K, dtype=np.uint64)
    ca["blob/path"] = os.path.join(os.path.abspath(outdir), "class_averages.mrc").encode()
    ca["blob/idx"] = np.arange(K)
    ca["blob/shape"] = (b, b)
    ca["blob/psize_A"] = apix_out
    ca["blob/res_A"] = clf.est_res
    with open(os.path.join(outdir, "class_averages.cs"), "wb") as fh:
        np.save(fh, ca)
    _write_csg(os.path.join(outdir, "class_averages.csg"), "class_averages", "2D class averages", "template",
               {"blob": "class_averages.cs"}, K)


def pick_devices(spec: str, n_particles: int, box: int = 64, apix: float = 0.0, max_res: float = 0.0) -> list[str]:
    """auto: one GPU for small jobs, all GPUs once the work is worth splitting. Work ~ particles x working
    box^2 (the box after binning to max_res), so 72k particles at 320 px count like ~370k at 64 px."""
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA GPU visible")
    mem = {i: torch.cuda.get_device_properties(i).total_memory for i in range(n)}
    if spec == "auto":
        ids = sorted(range(n), key=lambda i: -mem[i])
        wbox = min(box, 2 * box * apix / max_res) if apix > 0 and max_res > 0 else min(box, 64)
        ids = ids[:1] if n_particles * (wbox / 64.0) ** 2 < 100_000 else ids
    else:
        ids = sorted((int(x) for x in spec.split(",")), key=lambda i: -mem[i])
    names = ", ".join(f"{i}:{torch.cuda.get_device_name(i)}" for i in ids)
    print(f"[gpu] using {names} (master = first)")
    return [f"cuda:{i}" for i in ids]


def flag_duplicates(a, clf, src, st=None):
    """Write duplicates.star/.cs for picks whose aligned centres coincide with a better-scoring pick."""
    from .sources import find_duplicates
    dist = a.diameter / 4 if a.dup_dist < 0 else a.dup_dist
    pos = src.pick_positions()
    if dist <= 0 or pos is None:
        return None
    st = st or snapshot(clf)
    mic, xy = pos
    idx = st["idx"]
    dup = find_duplicates(mic[idx], xy[idx], st["offsets"], st["loglik"], dist)
    print(f"[dup] {dup.sum()} of {len(idx)} particles are duplicates (aligned centres < {dist:.0f} A apart)")
    if dup.any() and a.particles != "none":
        sd = {k: v[dup] for k, v in st.items()}
        if getattr(src, "kind", "") == "cryosparc":
            _write_cs_particles(a.o, clf, src, "full", sd, "duplicates")
        else:
            write_relion_particles(a.o, clf, src, sd, "duplicates")
    return dup


def run_deep(a, clf, src, t0):
    from .deep2d import run_deep2d
    sets = run_deep2d(
        clf, a.K, a.o, max_rounds=a.deep_rounds, tol=a.deep_tol, seed=a.seed, diameter_a=a.diameter,
        sel_kw=dict(min_conf=a.good_min_conf, max_res=a.good_max_res),
        good_classes=[int(x) for x in a.good_classes.split(",") if x.strip()], max_flips=a.max_flips)
    if a.particles != "none":
        is_cs = getattr(src, "kind", "") == "cryosparc"
        for name, st in sets.items():
            if len(st["idx"]) == 0:
                continue
            if not is_cs or a.particles == "full":
                write_relion_particles(a.o, clf, src, st, f"{name}_particles")
            if is_cs:
                _write_cs_particles(a.o, clf, src, "full", st, f"{name}_particles")
    print(f"[out] deep2D results in {a.o}: " + ", ".join(f"{len(v['idx'])} {k}" for k, v in sets.items())
          + f" (total wall time {time.time() - t0:.1f}s)")


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # stream logs even when redirected to a file
    ap = argparse.ArgumentParser(description="Fast GPU maximum-likelihood 2D classification")
    ap.add_argument("--i", "-i", required=True,
                    help="particles: RELION .star, cryoSPARC .cs/.csg, or a plain .mrc/.mrcs stack")
    ap.add_argument("--o", "-o", required=True, help="output directory")
    ap.add_argument("--K", type=int, default=20, help="number of classes")
    ap.add_argument("--iter", type=int, default=-1, help="iterations (-1 = auto: 25, or 40 above 100k particles)")
    ap.add_argument("--diameter", type=float, default=None,
                    help="particle mask diameter (A); default 200, or 1.25 x the particle size of a Celeris2D Extract job")
    ap.add_argument("--max-res", type=float, default=6.0, help="final resolution limit (A)")
    ap.add_argument("--init-res", type=float, default=30.0)
    ap.add_argument("--max-shift", type=float, default=16.0, help="local translation range per iteration (A)")
    ap.add_argument("--max-offset", type=float, default=-1, help="cap on accumulated particle offset (A; -1 = diameter/4)")
    ap.add_argument("--psi-step", type=float, default=2.0, help="finest in-plane angular step (deg)")
    ap.add_argument("--tau2-fudge", type=float, default=2.0)
    ap.add_argument("--batch-per-class", type=int, default=500, help="online-EM subset per class (0 = full passes)")
    ap.add_argument("--full-iter", type=int, default=-1, help="final iterations over all particles (-1 = auto)")
    ap.add_argument("--anneal", type=float, default=1.0, help="initial likelihood temperature")
    sf = ap.add_mutually_exclusive_group()
    sf.add_argument("--solvent-flatten", dest="solvent_flatten", action="store_true",
                    help="mask each class by a soft envelope of its own density (flat solvent; can help clean, "
                         "well-separated views, harmful on small/low-SNR particles)")
    sf.add_argument("--no-solvent-flatten", dest="solvent_flatten", action="store_false",
                    help="only the circular class mask (default, as cryoSPARC)")
    ap.add_argument("--gold-standard", action="store_true",
                    help="experimental: align each half-set only to its own half-set classes (unbiased FRC; "
                         "over-smooths classes with few particles)")
    ap.add_argument("--polish", action="store_true",
                    help="final pass aligns out to the reconstruction limit with classes locked: sharper on "
                         "high-SNR data, harmful on low-SNR / heavily binned data")
    ap.add_argument("--notch", type=float, default=2.5,
                    help="zero Fourier pixels with mean power > this x their shell mean (fixed-frame artefacts; 0 = off)")
    ap.add_argument("--noise-model", choices=("coloured", "radial"), default="coloured",
                    help="coloured = per-Fourier-pixel noise (down-weights fixed-frame artefacts); radial = per shell")
    ap.add_argument("--march-margin", type=float, default=0.0,
                    help="alignment may go this many Fourier shells beyond the current SSNR=1 resolution "
                         "(>0 risks fitting noise on low-SNR data); class reconstruction always goes 10 further")
    ap.add_argument("--no-frc-reg", action="store_true", help="regularise classes from their own power (old behaviour)")
    ap.add_argument("--sigma-init-factor", type=float, default=1.0,
                    help="initial noise inflation, annealed to 1 between iterations 2 and 17 (1 = off)")
    ap.add_argument("--no-center", action="store_true")
    ap.add_argument("--speed", choices=("fast", "balanced", "accurate"), default="balanced",
                    help="sampling preset: fast = coarser angles/shifts and one full pass on big sets (~3x faster); "
                         "accurate = finest sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", choices=("slim", "full", "none"), default="slim",
                    help="per-particle output: slim = alignments only (cryoSPARC: linked to the input metadata; "
                         "RELION: star), full = complete copies (.cs + .star), none = classes/stats only")
    ap.add_argument("--dup-dist", type=float, default=-1,
                    help="flag duplicate picks whose aligned centres are closer than this (A; -1 = diameter/4, 0 = off)")
    ap.add_argument("--optimize", type=int, default=0, metavar="N",
                    help="before the full run, try a small settings grid on N random particles and keep the most "
                         "reproducible settings (e.g. 20000)")
    ap.add_argument("--deep2d", action="store_true",
                    help="iterative good/bad sorting: re-classify good and bad sets separately until stable")
    ap.add_argument("--deep-rounds", type=int, default=6, help="deep2D: maximum rounds")
    ap.add_argument("--max-flips", type=int, default=3,
                    help="deep2D: particles that move good<->bad this often are set aside as unstable")
    ap.add_argument("--deep-tol", type=float, default=0.02, help="deep2D: stop when < this fraction changes")
    ap.add_argument("--good-classes", default="",
                    help="deep2D: hand-pick round-0 good classes, e.g. '3,5,12' (default: automatic)")
    ap.add_argument("--good-min-conf", type=float, default=None, help="deep2D: fixed confidence cut (default: auto)")
    ap.add_argument("--good-max-res", type=float, default=None, help="deep2D: fixed resolution cut in A (default: auto)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hard-pose", dest="hard_pose", action="store_const", const=True,
                   help="best pose per class only (cryoSPARC 'force max'); default auto: on for >= 20k particles")
    g.add_argument("--soft-pose", dest="hard_pose", action="store_const", const=False,
                   help="marginalise over all poses (RELION-like)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--uniform-prior", dest="class_prior", action="store_const", const=False,
                   help="ignore class populations in assignment; default auto: on for >= 100k particles")
    g.add_argument("--class-prior", dest="class_prior", action="store_const", const=True,
                   help="use class populations as prior (RELION-like)")
    ap.add_argument("--init-classes", default="", help="start from these class images (.mrc/.mrcs) instead of random")
    ap.add_argument("--subset", type=int, default=0, help="classify a random subset of N particles (quick tests)")
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--no-ctf", action="store_true", help="ignore CTF (also automatic if the star has no defocus)")
    ap.add_argument("--apix", type=float, default=None, help="pixel size override (needed for headerless stacks)")
    ap.add_argument("--gpus", default="auto",
                    help="nvidia-smi GPU indices, e.g. '1' or '1,0,2,3' (auto: biggest GPU for <100k particles, "
                         "all GPUs above)")
    ap.add_argument("--invert", action="store_true", help="force contrast inversion (default: auto-detect)")
    ap.add_argument("--no-invert", action="store_true", help="never invert contrast (default: auto-detect)")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="set any engine parameter (engine.Params field), e.g. --param solvent_dilate_a=40; repeatable")
    a = ap.parse_args(argv)

    t0 = time.time()
    src = open_source(a.i, a.project_dir, no_ctf=a.no_ctf, apix=a.apix)
    if a.diameter is None:
        a.diameter = getattr(src, "mask_diameter_a", 200.0)
    print(f"[input] {src.kind}: {src.N} particles {src.box}px @ {src.apix:.4f} A/px, "
          f"{'NO CTF (CTF=1)' if src.optics.no_ctf else 'CTF from metadata'} ({time.time() - t0:.1f}s)")
    if 0 < a.subset < src.N:
        src.subset(np.sort(np.random.default_rng(a.seed).choice(src.N, a.subset, replace=False)))
        print(f"[input] random subset of {src.N} particles")
    devices = pick_devices(a.gpus, src.N, src.box, src.apix, a.max_res)
    preset = dict(fast=dict(psi_factor=1.5, trans_div=2.5), balanced=dict(psi_factor=1.0, trans_div=3.0),
                  accurate=dict(psi_factor=1.0, trans_div=4.0))[a.speed]
    if a.speed == "fast" and a.full_iter < 0 and src.N > 500_000:
        a.full_iter = 1
    prm = Params(**preset, n_classes=a.K, n_iter=a.iter, diameter_a=a.diameter, max_res_a=a.max_res, init_res_a=a.init_res,
                 max_shift_a=a.max_shift, max_offset_a=a.max_offset, psi_step_min=a.psi_step, tau2_fudge=a.tau2_fudge,
                 batch_per_class=a.batch_per_class, n_full_iter=a.full_iter, anneal_t0=a.anneal,
                 center_classes=not a.no_center, seed=a.seed, init_classes=a.init_classes, frc_reg=not a.no_frc_reg, march_margin=a.march_margin, noise_model=a.noise_model, notch_thresh=a.notch, polish=a.polish, solvent_flatten=a.solvent_flatten, gold_standard=a.gold_standard, sigma_init_factor=a.sigma_init_factor, hard_pose=a.hard_pose, class_prior=a.class_prior,
                 polarity=-1 if a.invert else (1 if a.no_invert else 0))
    for kv in a.param:  # any Params field, typed like its default: --param solvent_dilate_a=40
        k, _, v = kv.partition("=")
        if not hasattr(prm, k):
            raise SystemExit(f"--param: unknown parameter '{k}'")
        cur = getattr(prm, k)
        words = {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False,
                 "off": False, "none": None, "auto": None}
        try:
            if isinstance(cur, bool):
                v = words[v.lower()]
            elif cur is None:
                v = words[v.lower()] if v.lower() in words else float(v)
            else:
                v = type(cur)(v)
        except (KeyError, ValueError):
            raise SystemExit(f"--param {k}: cannot parse '{v}' (current value {cur!r})")
        setattr(prm, k, v)
    clf = Classifier2D(src, prm, devices=devices)
    if a.optimize > 0:
        from .optimize import apply as apply_settings
        from .optimize import optimize
        t_opt = time.time()
        best, rows = optimize(clf, a.K, subset=a.optimize)
        apply_settings(clf, best)
        print(f"[optimize] best {best} ({time.time() - t_opt:.0f}s for {len(rows)} trials)")
        os.makedirs(a.o, exist_ok=True)
        with open(os.path.join(a.o, "optimizer.json"), "w") as fh:
            json.dump(dict(best=best, trials=rows, subset=a.optimize), fh, indent=1)
    if a.deep2d:
        return run_deep(a, clf, src, t0)
    clf.run()
    t1 = time.time()
    write_outputs(a.o, clf, src, a.i, a.particles)
    flag_duplicates(a, clf, src)
    print(f"[out] outputs written in {time.time() - t1:.1f}s")
    print(f"[out] wrote {a.o} (total wall time {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
