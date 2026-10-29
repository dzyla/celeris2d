"""Motion correction + CTF on raw movies, compared with cryoSPARC's results for the same movies.

usage: preprocess_eval.py <movie dir> <gain.mrc> <cryoSPARC micrograph dir> <particles.cs> <outdir>
           --apix 0.6575 --dose 53 [--bin 2] [--n 0] [--kv 300 --cs 2.7 --amp 0.1]

Per movie: gain orientation, frame shifts, the aligned sums (written as <outdir>/<movie>_dw.mrc), CTF of our sum
and of cryoSPARC's micrograph with the same fitter (fit resolution = how far the Thon rings reach, a direct
measure of how well the frames were aligned), the defocus vs cryoSPARC's (median of the particles' defocus on
that micrograph in particles.cs), the FRC between our dose-weighted sum and cryoSPARC's micrograph (the same
frames and the same shot noise: it measures how close our shifts are to cryoSPARC's, not which are better), and
the FRC between the sums of the even and of the odd frames (independent noise: higher = better aligned signal).
One JSON line per movie in <outdir>/eval.jsonl; control figures <outdir>/<movie>.png (motion, local motion model,
power spectrum and CTF fit, fit significance, even/odd FRC, dose weights) and <outdir>/summary.png (all movies vs
cryoSPARC).
"""
import argparse
import glob
import json
import os
import time

import mrcfile
import numpy as np
import torch

from celeris2d import plots_preprocess as pl
from celeris2d import preprocess as pp
from celeris2d.engine import Optics


def frc(a: torch.Tensor, b: torch.Tensor, apix: float, n_shell: int = 40):
    fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
    ky, kx = pp._grid(a.shape[0], a.shape[1], apix, a.device)
    k = torch.sqrt(ky ** 2 + kx ** 2)
    sh = (k / (0.5 / apix) * n_shell).long().clamp(max=n_shell)
    m = sh < n_shell
    s = sh[m]
    num = torch.zeros(n_shell, device=a.device).index_add_(0, s, (fa * fb.conj()).real[m])
    da = torch.zeros(n_shell, device=a.device).index_add_(0, s, fa.abs().pow(2)[m])
    db = torch.zeros(n_shell, device=a.device).index_add_(0, s, fb.abs().pow(2)[m])
    ks = (torch.arange(n_shell, device=a.device) + 0.5) / n_shell * 0.5 / apix
    return ks.cpu().numpy(), (num / (da * db).sqrt().clamp_min(1e-12)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("movies")
    ap.add_argument("gain")
    ap.add_argument("cs_mics")
    ap.add_argument("particles")
    ap.add_argument("outdir")
    ap.add_argument("--apix", type=float, required=True)
    ap.add_argument("--dose", type=float, required=True, help="total exposure (e/A^2)")
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--n", type=int, default=0, help="first n movies only (0 = all)")
    ap.add_argument("--patches", type=int, default=5, help="local motion on a patches x patches grid (0 = rigid)")
    ap.add_argument("--only", default="", help="comma-separated movie name fragments to process")
    ap.add_argument("--kv", type=float, default=300)
    ap.add_argument("--cs", type=float, default=2.7)
    ap.add_argument("--amp", type=float, default=0.1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    optics = Optics(a.kv, a.cs, a.amp)
    movies = sorted(glob.glob(os.path.join(a.movies, "*.tif")) + glob.glob(os.path.join(a.movies, "*.mrc*")))
    movies = movies[:a.n] if a.n else movies
    if a.only:
        movies = [m for m in movies if any(f in m for f in a.only.split(","))]
    cs = np.load(a.particles)
    mp = np.array([os.path.basename(p.decode().lstrip(">")) for p in cs["location/micrograph_path"]])
    gain0 = pp.read_gain(a.gain)
    gain = None
    out = open(os.path.join(a.outdir, "eval.jsonl"), "a")
    for path in movies:
        stem = os.path.basename(path).split(".")[0]
        t0 = time.time()
        fr = pp.read_movie(path)
        t_read = time.time() - t0
        if gain is None:
            gain = pp.orient_gain(gain0, fr)
        torch.cuda.synchronize()
        t1 = time.time()
        mic = pp.motion_correct(fr, gain, a.apix, a.dose / len(fr), bin=a.bin, patches=a.patches)
        torch.cuda.synchronize()
        t_mc = time.time() - t1
        t2 = time.time()
        fit = pp.estimate_ctf(mic.sum, mic.apix, optics)
        torch.cuda.synchronize()
        t_ctf = time.time() - t2
        fdw = pp.estimate_ctf(mic.sum_dw, mic.apix, optics)      # like-for-like with cryoSPARC's dose-weighted sum
        with mrcfile.new(os.path.join(a.outdir, stem + "_dw.mrc"), overwrite=True) as m:
            m.set_data(mic.sum_dw.cpu().numpy().astype(np.float32))
            m.voxel_size = mic.apix
        row = dict(movie=stem, frames=len(fr), read_s=round(t_read, 1), motion_s=round(t_mc, 1), ctf_s=round(t_ctf, 2),
                   shift_total_a=round(float(np.abs(mic.shifts[-1] - mic.shifts[0]).max()), 1),
                   path_len_a=round(float(np.linalg.norm(np.diff(mic.shifts, axis=0), axis=1).sum()), 1),
                   local_rms_a=round(mic.local["rms_a"], 2) if mic.local else None,
                   frc_even_odd={f"{1 / k:.1f}A": round(float(v), 3) for k, v in
                                 zip(*[x[4::6] for x in frc(mic.halves[0], mic.halves[1], mic.apix)])},
                   ours=dict(du=round(fit.defocus_u), dv=round(fit.defocus_v), ang=round(fit.angle, 1),
                             fit_res=round(fit.fit_res_a, 2), score=round(fit.score, 3)),
                   ours_dw=dict(du=round(fdw.defocus_u), dv=round(fdw.defocus_v), fit_res=round(fdw.fit_res_a, 2),
                                score=round(fdw.score, 3)))
        sel = np.flatnonzero(np.array([stem in m for m in mp]))
        if len(sel):
            row["cryosparc_particles"] = dict(n=int(len(sel)), du=round(float(np.median(cs["ctf/df1_A"][sel]))),
                                              dv=round(float(np.median(cs["ctf/df2_A"][sel]))),
                                              ang=round(float(np.degrees(np.median(cs["ctf/df_angle_rad"][sel]))) % 180, 1))
        cm = [f for f in glob.glob(os.path.join(a.cs_mics, "*.mrc")) if stem in os.path.basename(f)]
        if cm:
            with mrcfile.open(cm[0], permissive=True) as m:
                c = torch.as_tensor(np.asarray(m.data, dtype=np.float32), device=mic.sum.device)
                capix = float(m.voxel_size.x)
            f = int(round(mic.apix / capix))
            if f > 1:
                ny, nx = mic.sum.shape
                c = torch.fft.irfft2(pp.crop_half(torch.fft.rfft2(c, norm="forward"), ny, nx), s=(ny, nx), norm="forward")
            fc = pp.estimate_ctf(c, mic.apix, optics)
            row["cryosparc_mic"] = dict(du=round(fc.defocus_u), dv=round(fc.defocus_v), ang=round(fc.angle, 1),
                                        fit_res=round(fc.fit_res_a, 2), score=round(fc.score, 3))
            ks, fr_ = frc(mic.sum_dw, c, mic.apix)
            row["frc_vs_cryosparc"] = {f"{1 / k:.1f}A": round(float(v), 3) for k, v in zip(ks[4::6], fr_[4::6])}
        eo = frc(mic.halves[0], mic.halves[1], mic.apix, n_shell=120)
        pl.movie_figure(os.path.join(a.outdir, stem + ".png"), mic, fit, title=stem, frc_eo=eo)
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()
        np.save(os.path.join(a.outdir, stem + "_shifts.npy"), mic.shifts)
        del fr, mic
    rows = [json.loads(l) for l in open(os.path.join(a.outdir, "eval.jsonl"))]
    rows = list({r["movie"]: r for r in rows}.values())
    pl.summary_figure(os.path.join(a.outdir, "summary.png"), rows)


if __name__ == "__main__":
    main()
