"""Live view of a running 3D job for the dashboard: while the job runs, abinitio.LIVE hands every class-stage start's,
SGD round's, refinement iteration's and 3D classification iteration's maps to this module, which writes them
(downsampled, at most every few seconds, on a background thread so the job does not wait) as live/live3d_mN.mrc +
live/live3d.json in the job folder. The page shows the projections, the resolution and class sizes as they change,
and opens the current maps in the 3D viewer.

    with live_maps(project, job):   # scheduler.run_job wraps every particle job in this
        spec.run(...)

live3d.json: stage ("classes" class-average starts, "sgd", "refine", "class3d"), it, n_iter, res (A per model, None
unknown), counts (particles per model, None for a single model / unknown), apix (of the written maps), models (number
of live3d_mN.mrc files, N = 1..models), history (every step, also those not written: stage, it, res, counts, t),
updated, job. The files sit in the folder live/, which Job.organize (the jobs' tidy step) leaves alone, and go when the
job ends.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import shutil
import threading
import time

import numpy as np

LIVE_DIR = "live"


def live_dir(job) -> str:
    return os.path.join(job.path, LIVE_DIR)


def _num(r):
    return None if r is None or not math.isfinite(float(r)) else round(float(r), 2)


@contextlib.contextmanager
def live_maps(project, job, every_s: float = 4.0, size: int = 128):
    from . import abinitio
    q: queue.Queue = queue.Queue(maxsize=1)
    hist: list = []
    last = [0.0, None]
    d = live_dir(job)

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            maps, head = item
            try:
                os.makedirs(d, exist_ok=True)
                for k, m in enumerate(maps):
                    p = os.path.join(d, f"live3d_m{k + 1}.mrc")
                    project.write_mrc(p + ".tmp.mrc", m, head["apix"])
                    os.replace(p + ".tmp.mrc", p)
                for f in os.listdir(d):              # models gone since (e.g. 1 consensus map -> K class maps -> fewer)
                    if f.startswith("live3d_m") and f.endswith(".mrc") and f[8:-4].isdigit() and int(f[8:-4]) > len(maps):
                        with contextlib.suppress(OSError):
                            os.remove(os.path.join(d, f))
                with open(os.path.join(d, "live3d.json.tmp"), "w") as fh:
                    json.dump(head, fh)
                os.replace(os.path.join(d, "live3d.json.tmp"), os.path.join(d, "live3d.json"))
            except Exception:                        # noqa: BLE001 - a live view never stops a job
                pass

    def due(stage, it, n_iter) -> bool:
        """A snapshot is wanted: the first of a stage, the last of a stage, else every every_s seconds."""
        return not (time.time() - last[0] < every_s and stage == last[1] and it < n_iter)

    def hook(stage, it, n_iter, maps, res, apix, counts=None):
        import torch
        now = time.time()
        res = [_num(r) for r in res]
        counts = None if counts is None else [int(c) for c in counts]
        hist.append(dict(stage=stage, it=int(it), res=res, counts=counts, t=round(now, 1)))
        if maps is None or not due(stage, it, n_iter):
            return
        last[0], last[1] = now, stage
        out = []
        b = max(maps[0].shape)
        for m in maps:
            v = m.float()
            if b > size:
                v = torch.nn.functional.adaptive_avg_pool3d(v[None, None], (size, size, size))[0, 0]
            out.append(v.cpu().numpy().astype(np.float32))
        head = dict(stage=stage, it=int(it), n_iter=int(n_iter), res=res, counts=counts,
                    apix=float(apix) * b / max(out[0].shape), models=len(out), history=hist[-300:], updated=now,
                    job=job.name)
        try:
            q.put_nowait((out, head))
        except queue.Full:                           # the writer is busy: this one replaces the snapshot waiting
            with contextlib.suppress(queue.Empty):
                q.get_nowait()
            with contextlib.suppress(queue.Full):
                q.put_nowait((out, head))

    hook.due = due
    th = threading.Thread(target=writer, daemon=True, name="live3d")
    th.start()
    abinitio.LIVE = hook
    try:
        yield
    finally:
        abinitio.LIVE = None
        with contextlib.suppress(queue.Full):
            q.put(None, timeout=5)
        th.join(timeout=10)
        shutil.rmtree(d, ignore_errors=True)
