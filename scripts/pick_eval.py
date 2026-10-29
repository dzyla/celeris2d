"""Movies -> project (motion, CTF, blob picks, particles) and pick recall against a refined particle set.

usage: pick_eval.py <movie dir> <gain> <refined particles.cs> <project dir> --apix 0.6575 --dose 53 --bin 2
           --diameter 120 [--particle-a 160] [--box-a 0] [--threshold T] [--min-dist 0.8] [--pattern "*.tif*"]
           [--dtype float16] [--particle-apix 2.63] [--no-figures] [--export relion,cryosparc]

Makes Import, MotionCtf, Pick and Extract jobs in the project and runs them over every movie (pipeline.run), and
reports, against the refined set's particle centres (pick position minus the refined shift: cryoSPARC's pick
positions are ~20 A off for T20S), the share of refined particles found among the kept picks and among all
candidates (match radius 45 A). The refined set is a 2D-cleaned subset, so picks it lacks are not all false.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from celeris2d import picking as pk
from celeris2d.pipeline import (ExtractSettings, MotionSettings, PickSettings, extract_job, motion_job, pick_job,
                                run)
from celeris2d.project import Project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("movies")
    ap.add_argument("gain")
    ap.add_argument("particles")
    ap.add_argument("project")
    ap.add_argument("--apix", type=float, required=True)
    ap.add_argument("--dose", type=float, required=True)
    ap.add_argument("--bin", type=float, default=1)
    ap.add_argument("--diameter", type=float, required=True)
    ap.add_argument("--box-a", type=float, default=0.0, help="0: particle + CTF delocalisation")
    ap.add_argument("--particle-a", type=float, default=0.0, help="longest particle dimension (0: 1.3 x diameter)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-dist", type=float, default=0.8)
    ap.add_argument("--pattern", default="*.tif*", help="movie file pattern in the movie dir")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--particle-apix", type=float, default=0.0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--export", default="", help="relion,cryosparc: also write their files for the Extract job")
    a = ap.parse_args()
    prj = Project(a.project)
    ms = MotionSettings(apix=a.apix, dose_total=a.dose, bin=a.bin, gain=a.gain, dtype=a.dtype,
                        figures="none" if a.no_figures else "outliers")
    ps = PickSettings(diameter_a=a.diameter, threshold=a.threshold, min_dist_frac=a.min_dist, figures="none" if a.no_figures else "outliers")
    es = ExtractSettings(particle_a=a.particle_a or 1.3 * a.diameter, box_a=a.box_a, particle_apix=a.particle_apix,
                         dtype=a.dtype, export=tuple(e for e in a.export.split(",") if e))
    movies = sorted(glob.glob(os.path.join(a.movies, a.pattern)))
    t0 = time.time()
    imp = prj.import_movies(movies)
    mc = motion_job(prj, imp, ms)
    pj = pick_job(prj, mc, ps)
    ej = extract_job(prj, pj, es)
    timing = run(prj, [mc, pj, ej])
    wall = time.time() - t0
    cs = np.load(a.particles)
    mp = np.array([os.path.basename(p.decode()) for p in cs["location/micrograph_path"]])
    sh = cs["alignments3D/shift"] * cs["alignments3D/psize_A"][:, None]
    mics = prj.micrographs(mc)
    parts = prj.particles(ej)
    n_ref = n_kept = n_cand = 0
    for r in mics.itertuples():
        sel = np.flatnonzero([r.stem in m for m in mp])
        if not len(sel):
            continue
        ref = np.stack([cs["location/center_x_frac"][sel] * r.nx * r.apix,
                        cs["location/center_y_frac"][sel] * r.ny * r.apix], 1) - sh[sel]
        p = parts[parts["movie_id"] == r.movie_id][["x_a", "y_a"]].to_numpy()
        c = prj.picks(pj, r.movie_id)["xy_a"]
        n_ref += len(ref)
        n_kept += int(pk.match(p, ref, 45.0)[1].sum())
        n_cand += int(pk.match(c, ref, 45.0)[1].sum())
    meta = [json.loads(m) for m in mics["meta"]]
    sec = {k: float(np.median([m["seconds"][k] for m in meta])) for k in meta[0]["seconds"]}
    out = dict(micrographs=len(mics), picks=len(parts), refined=n_ref, recall_picks=round(n_kept / max(n_ref, 1), 3),
               recall_candidates=round(n_cand / max(n_ref, 1), 3), box=int(parts["box"].iloc[0]),
               seconds_per_movie={k: round(v, 2) for k, v in sec.items()}, wall_s=round(wall, 1), jobs=[mc.name, pj.name, ej.name],
               extract=prj.summary_of(ej), run=timing)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
