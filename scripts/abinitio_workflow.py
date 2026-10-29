"""2D classification -> class selection -> ab initio 3D, as on the dashboard.

usage: abinitio_workflow.py <particles> <workdir> <outdir> --diameter D --max-res R [--K 20]
                            [--select suggest|all|1,4,7] [--sym C1,D2] [--models 1] [--only-2d]

Writes the 2D gallery (<workdir>/k<K>_classes.png) and, unless --only-2d, one ab initio run per symmetry into
<outdir>/<sym>/ (maps, reprojections, viewing directions, FSC, summary.json). Re-running reuses the 2D result.
"""
import argparse
import json
import time

from celeris2d.session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("particles")
    ap.add_argument("workdir")
    ap.add_argument("outdir")
    ap.add_argument("--diameter", type=float, required=True)
    ap.add_argument("--max-res", type=float, required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--select", default="suggest")
    ap.add_argument("--sym", default="C1")
    ap.add_argument("--models", type=int, default=1)
    ap.add_argument("--only-2d", action="store_true")
    a = ap.parse_args()
    s = Session(a.workdir, log=lambda m: print(m) if m.startswith("[3d]") and "particles it" not in m else None)
    t0 = time.time()
    s.load(a.particles, diameter=a.diameter, max_res=a.max_res, gpus="0")
    s.clf._wait_loaded(s.clf.N)
    print(f"[load] {s.clf.N} particles, working box {s.clf.bmax} ({s.clf.box_a / s.clf.bmax:.2f} A/px), {time.time() - t0:.0f}s")
    name = f"k{a.K}"
    if name not in s.results:
        t0 = time.time()
        s.classify(name, K=a.K)
        print(f"[2D] K={a.K}: {time.time() - t0:.1f}s")
    r = s.results[name]
    print(f"[2D] gallery {r.gallery_path}")
    for k in range(len(r.pops)):
        print(f"  class {k + 1:2d}: {int(r.pops[k]):6d} particles, conf {r.conf[k]:.2f}, ~{r.res_a[k]:.1f} A")
    if a.select == "all":
        good = [k + 1 for k in range(len(r.pops)) if r.pops[k] > 0]
    elif a.select == "suggest":
        good = s.suggest(name)
    else:
        good = [int(x) for x in a.select.split(",")]
    print(f"[select] {len(good)} classes: {good}")
    if a.only_2d:
        return
    s.select(name, good)
    import torch
    torch.cuda.reset_peak_memory_stats()        # report the 3D stage's own peak, not the 2D classification's
    for sym in a.sym.split(","):
        summ = s.abinitio(f"{name}_good_{sym.lower()}" + (f"_k{a.models}" if a.models > 1 else ""), f"{name}:good",
                          sym=sym, n_models=a.models)
        import torch
        keys = ("particles", "particles_used", "classes", "seconds_classes", "seconds_particles", "box", "resolution_a",
                "resolution_unmasked_a", "resolution_solvent_mask_a", "suggested_sym")
        print(json.dumps({k: summ.get(k) for k in keys}),
              f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
        import shutil
        shutil.copytree(f"{s.workdir}/{name}_good_{sym.lower()}" + (f"_k{a.models}" if a.models > 1 else ""),
                        f"{a.outdir}/{sym}" + (f"_k{a.models}" if a.models > 1 else ""), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
