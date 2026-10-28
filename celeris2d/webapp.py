"""Celeris2D web dashboard: load a dataset, change every setting, run, select classes, rerun, export.

    celeris2d-web --port 8765            # then open http://<host>:8765  (or tunnel: ssh -L 8765:localhost:8765 host)

One dataset is held in memory (spectra on the GPU), so every rerun on a subset starts immediately. GPU jobs
run one at a time on a worker thread; the page polls /api/status for progress and the live log.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import io
import os
import threading
import time
import traceback

import numpy as np

from .engine import Params

# settings shown in the UI (Params fields); the dataset-level ones are used at load time
LOAD_KEYS = ("diameter_a", "max_res_a", "init_res_a", "psi_step_min", "psi_factor", "trans_div", "max_shift_a",
             "notch_thresh", "noise_model", "seed")
RUN_KEYS = ("n_iter", "n_full_iter", "batch_per_class", "march_margin", "recon_margin", "frc_reg", "tau2_fudge",
            "sigma_init_factor", "solvent_flatten", "solvent_thresh", "solvent_dilate_a", "center_classes",
            "recenter_thresh", "hard_pose", "class_prior", "polish", "reseed_frac", "max_offset_a")


def _defaults():
    p = Params()
    d = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
    d["n_iter"] = -1  # auto (25, or 40 above 100k particles)
    return d


class App:
    def __init__(self, workdir):
        self.workdir = workdir
        self.session = None
        self.job = None            # dict(name, kind, started, done, error)
        self.log = collections.deque(maxlen=400)
        self.lock = threading.Lock()
        self.tiles = {}
        self.optimizer = None

    def _log(self, *a):
        line = " ".join(str(x) for x in a)
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")

    def submit(self, kind, name, fn):
        with self.lock:
            if self.job and not self.job.get("done"):
                raise RuntimeError(f"busy: {self.job['kind']} '{self.job['name']}' is running")
            self.job = dict(kind=kind, name=name, started=time.time(), done=False, error=None)

        def run():
            try:
                fn()
            except Exception as e:  # report to the page instead of dying silently
                self.job["error"] = f"{type(e).__name__}: {e}"
                self._log("ERROR", self.job["error"])
                self._log(traceback.format_exc().splitlines()[-3])
            finally:
                self.job["done"] = True
                self.job["seconds"] = round(time.time() - self.job["started"], 1)
                self._log(f"[{kind}] {name} finished in {self.job['seconds']}s")
        threading.Thread(target=run, daemon=True).start()


def tile_png(img, size=192):
    from PIL import Image
    im = img.astype(np.float32)
    lo, hi = np.percentile(im, [0.5, 99.7])
    u8 = (np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).resize((size, size), Image.LANCZOS).save(buf, format="PNG")
    return buf.getvalue()


def create(workdir):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    from .optimize import apply as apply_settings
    from .optimize import optimize
    from .session import Session

    app = FastAPI(title="Celeris2D")
    st = App(workdir)

    def sess():
        if st.session is None or st.session.clf is None:
            raise HTTPException(400, "no dataset loaded")
        return st.session

    def run_params(body):
        p = {k: body[k] for k in RUN_KEYS if k in body and body[k] is not None and body[k] != ""}
        for k in ("hard_pose", "class_prior"):          # "auto" -> None
            if p.get(k) == "auto":
                p[k] = None
        return p

    @app.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(os.path.dirname(__file__), "web", "index.html")).read()

    @app.get("/api/defaults")
    def defaults():
        d = _defaults()
        return dict(load={k: d[k] for k in LOAD_KEYS}, run={k: d[k] for k in RUN_KEYS})

    @app.post("/api/load")
    def load(body: dict = Body(...)):
        path = body["path"]
        if not os.path.exists(path):
            raise HTTPException(400, f"not found: {path}")
        wd = body.get("workdir") or os.path.join(st.workdir, os.path.basename(os.path.dirname(os.path.abspath(path)))
                                                   + "_" + os.path.splitext(os.path.basename(path))[0])

        def job():
            st.tiles.clear()
            st.session = Session(wd, log=st._log)
            kw = {k: body[k] for k in LOAD_KEYS if k in body and body[k] not in (None, "")}
            info = st.session.load(path, diameter=float(kw.pop("diameter_a", 200)), max_res=float(kw.pop("max_res_a", 6)),
                                   gpus=body.get("gpus", "auto"), subset=int(body.get("subset") or 0),
                                   project_dir=body.get("project_dir") or None, **kw)
            st._log(f"[load] {info}")
        st.submit("load", os.path.basename(path), job)
        return dict(ok=True, workdir=wd)

    @app.post("/api/classify")
    def classify(body: dict = Body(...)):
        s = sess()
        name = body.get("name") or f"run{len(s.results) + 1}"
        K, seeds, on = int(body.get("K", 50)), int(body.get("seeds", 1)), body.get("on", "all")
        params = run_params(body)

        def job():
            for k in ("hard_pose", "class_prior"):
                if k in params:
                    setattr(s.clf, f"_auto_{'hard' if k == 'hard_pose' else 'prior'}", params[k] is None)
            s.classify(name, K=K, seeds=seeds, on=on, **{k: v for k, v in params.items() if v is not None})
        st.submit("classify", name, job)
        return dict(ok=True, name=name)

    @app.post("/api/optimize")
    def optimize_ep(body: dict = Body(...)):
        s = sess()
        K, subset = int(body.get("K", 50)), int(body.get("subset", 20000))

        def job():
            best, rows = optimize(s.clf, K, subset=subset, log=st._log)
            st.optimizer = dict(best=best, trials=rows, K=K, subset=subset)
            if body.get("apply", True):
                apply_settings(s.clf, best)
                st._log(f"[optimize] applied {best}")
        st.submit("optimize", f"K={K} subset={subset}", job)
        return dict(ok=True)

    @app.get("/api/status")
    def status():
        s = st.session
        loaded = s is not None and s.clf is not None
        out = dict(loaded=loaded, job=st.job, log=list(st.log)[-120:], optimizer=st.optimizer)
        if loaded:
            c = s.clf
            out.update(path=s.path, particles=c.N, loaded_particles=c._n_loaded, gpus=[str(d) for d in c.devs],
                       box=c.B_full, apix=round(c.apix_full, 4), working_box=c.bmax, workdir=s.workdir,
                       results=[dict(name=k, kind=r.kind, particles=int(len(r.idx)), classes=int(len(r.images)),
                                     selected=None if r.good is None else int(r.good.sum()),
                                     on=r.info.get("on", "all"), seconds=r.info.get("seconds"))
                                for k, r in s.results.items()],
                       settings={k: getattr(c.p, k) for k in RUN_KEYS + LOAD_KEYS})
            if st.job and not st.job["done"] and c.history:
                h = c.history[-1]
                out["progress"] = dict(it=h["it"], n_iter=c.p.n_iter, res=round(h["res"], 1), est_res=round(h["est_res"], 1))
        return out

    @app.get("/api/result/{name}")
    def result(name: str):
        s = sess()
        if name not in s.results:
            raise HTTPException(404, name)
        summ = s.summary(name)
        r = s.results[name]
        order = np.argsort(-r.pops)
        summ["order"] = [int(k) + 1 for k in order]
        summ["version"] = f"{id(r)}-{0 if r.good is None else int(r.good.sum())}"
        return summ

    @app.get("/api/result/{name}/tile/{k}.png")
    def tile(name: str, k: int):
        s = sess()
        r = s.results[name]
        key = (name, id(r), k)
        if key not in st.tiles:
            st.tiles[key] = tile_png(r.images[k - 1])
        return Response(st.tiles[key], media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/select")
    def select(body: dict = Body(...)):
        s = sess()
        return s.select(body["name"], good=[int(k) for k in body.get("good", [])])

    @app.get("/api/suggest/{name}")
    def suggest(name: str):
        return dict(suggested=sess().suggest(name))

    @app.get("/api/sets/{name}")
    def sets(name: str, min_p: float = 0.5):
        return {k: int(len(v)) for k, v in sess().sets(name, min_p).items()}

    @app.post("/api/export")
    def export(body: dict = Body(...)):
        s = sess()
        return s.export(body["name"], body["outdir"], min_p=float(body.get("min_p", 0.5)), fmt=body.get("fmt", "auto"))

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Celeris2D web dashboard")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on the network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workdir", default=os.path.abspath("celeris2d_web"))
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.makedirs(a.workdir, exist_ok=True)
    import uvicorn
    print(f"Celeris2D dashboard on http://{a.host}:{a.port}  (workdir {a.workdir})")
    uvicorn.run(create(a.workdir), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
