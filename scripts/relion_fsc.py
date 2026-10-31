"""Gold-standard FSC of several half-map pairs by RELION's relion_postprocess (an independent scorer), every pair
with a solvent mask made by the same recipe from its own map, so masks differ only as the maps do: the 15 A low-pass
of the half-map average, binarised at the level that keeps the same number of voxels in every map, extended 3 px,
6 px soft edge (relion_image_handler + relion_mask_create). A pair can also name its own mask (scored a second time
with it). Reports RELION's final resolution (phase-randomisation corrected FSC = 0.143), the unmasked 0.143 point and
the corrected FSC at a few resolutions.

usage: relion_fsc.py --out DIR [--voxels N | --match MASK.mrc] [--res 8,6,5] name=half1.mrc,half2.mrc[,mask.mrc] ...
  --match: N from a reference mask (the voxels above 0.5, times --core, the share before extension; default 0.6).
Needs relion_postprocess / relion_mask_create / relion_image_handler on PATH; all maps on one grid and pixel size."""
import argparse
import os
import subprocess

import mrcfile
import numpy as np


def rd(p):
    with mrcfile.open(p, permissive=True) as f:
        return np.array(f.data, np.float32), float(f.voxel_size.x)


def wr(p, v, apix):
    with mrcfile.new(p, overwrite=True) as f:
        f.set_data(v.astype(np.float32))
        f.voxel_size = apix


def sh(*cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def star_table(path, block="fsc"):
    """Columns of one data_ block of a RELION .star file as {name: float array}."""
    cols, rows, inside = [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("data_"):
            inside = s == f"data_{block}"
            cols, rows = ([], []) if inside else (cols, rows)
            continue
        if not inside or not s or s == "loop_" or s.startswith("#"):
            continue
        if s.startswith("_rln"):
            cols.append(s.split()[0][4:])
        elif cols:
            rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(cols)}


def final_resolution(path):
    for line in open(path):
        if line.startswith("_rlnFinalResolution"):
            return float(line.split()[1])
    return float("nan")


def recipe_mask(d, h1, h2, apix, n_vox, lowpass=15.0, extend=3, soft=6):
    a, _ = rd(h1)
    b, _ = rd(h2)
    wr(f"{d}/avg.mrc", 0.5 * (a + b), apix)
    sh("relion_image_handler", "--i", f"{d}/avg.mrc", "--o", f"{d}/avg_lp.mrc", "--lowpass", lowpass, "--angpix", apix)
    lp, _ = rd(f"{d}/avg_lp.mrc")
    thr = float(np.sort(lp.ravel())[-int(n_vox)])
    sh("relion_mask_create", "--i", f"{d}/avg_lp.mrc", "--o", f"{d}/mask.mrc", "--ini_threshold", thr,
       "--extend_inimask", extend, "--width_soft_edge", soft, "--j", 8)
    m, _ = rd(f"{d}/mask.mrc")
    return f"{d}/mask.mrc", float(m.sum())


def postprocess(d, tag, h1, h2, mask, apix, res):
    o = f"{d}/{tag}"
    sh("relion_postprocess", "--i", h1, "--i2", h2, "--mask", mask, "--angpix", apix, "--o", o, "--auto_bfac",
       "--autob_lowres", 10)
    t = star_table(o + ".star")
    k_a = t["AngstromResolution"]
    order = np.argsort(-k_a)                         # increasing spatial frequency

    def at(col, r):
        return float(np.interp(1 / r, 1 / k_a[order], t[col][order]))

    def cross(col):
        f = t[col][order]
        i = np.flatnonzero(f < 0.143)
        i = i[i > 0]
        return float(k_a[order][i[0] - 1]) if len(i) else float(k_a[order][-1])

    return dict(final=final_resolution(o + ".star"), unmasked=cross("FourierShellCorrelationUnmaskedMaps"),
                **{f"{r:g}A": at("FourierShellCorrelationCorrected", r) for r in res})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxels", type=float, default=0)
    ap.add_argument("--match", default="")
    ap.add_argument("--core", type=float, default=0.6)
    ap.add_argument("--res", default="8,6,5")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    res = [float(x) for x in a.res.split(",")]
    n_vox = a.voxels or (rd(a.match)[0] > 0.5).sum() * a.core
    rows = []
    for p in a.pairs:
        name, files = p.split("=", 1)
        files = files.split(",")
        h1, h2 = files[:2]
        apix = rd(h1)[1]
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        mask, vol = recipe_mask(d, h1, h2, apix, n_vox)
        rows.append((name, "recipe", vol, postprocess(d, "pp_recipe", h1, h2, mask, apix, res)))
        if len(files) > 2:
            own = files[2]
            rows.append((name, "own", float(np.clip(rd(own)[0], 0, 1).sum()),
                         postprocess(d, "pp_own", h1, h2, own, apix, res)))
        r = rows[-1]
        print(f"{r[0]} done", flush=True)
    keys = ["final", "unmasked"] + [f"{r:g}A" for r in res]
    print(f"{'map':18s} {'mask':7s} {'mask_vox':>9s} " + " ".join(f"{k:>8s}" for k in keys))
    for name, kind, vol, r in rows:
        print(f"{name:18s} {kind:7s} {vol:9.0f} " + " ".join(
            f"{r[k]:8.2f}" if k in ("final", "unmasked") else f"{r[k]:8.3f}" for k in keys))


if __name__ == "__main__":
    main()
