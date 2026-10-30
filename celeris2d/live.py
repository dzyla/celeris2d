"""Live view of a running 3D job for the dashboard: while the job runs, abinitio.LIVE hands every SGD round's and
refinement iteration's maps to this module, which writes them (downsampled, at most every few seconds, on a
background thread so the job does not wait) as live3d_mN.mrc + live3d.json in the job folder. The page shows the
projections and the resolution history as they change, and opens the current map in the 3D viewer.

    with live_maps(project, job):   # scheduler.run_job wraps every particle job in this
        spec.run(...)
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time

import numpy as np


@contextlib.contextmanager
def live_maps(project, job, every_s: float = 4.0, size: int = 128):
    from . import abinitio
    q: queue.Queue = queue.Queue(maxsize=1)
    hist: list = []
    last = [0.0, None]
    written: set = set()

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            maps, head = item
            try:
                for k, m in enumerate(maps):
                    p = os.path.join(job.path, f"live3d_m{k + 1}.mrc")
                    project.write_mrc(p + ".tmp.mrc", m, head["apix"])
                    os.replace(p + ".tmp.mrc", p)
                    written.add(p)
                with open(os.path.join(job.path, "live3d.json.tmp"), "w") as fh:
                    json.dump(head, fh)
                os.replace(os.path.join(job.path, "live3d.json.tmp"), os.path.join(job.path, "live3d.json"))
                written.add(os.path.join(job.path, "live3d.json"))
            except Exception:                        # noqa: BLE001 - a live view never stops a job
                pass

    def hook(stage, it, n_iter, maps, res, apix):
        import torch
        now = time.time()
        res = [None if r is None else round(float(r), 2) for r in res]
        hist.append(dict(stage=stage, it=int(it), res=res, t=round(now, 1)))
        if now - last[0] < every_s and stage == last[1] and it < n_iter:
            return
        last[0], last[1] = now, stage
        out = []
        b = max(maps[0].shape)
        for m in maps:
            v = m.float()
            if b > size:
                v = torch.nn.functional.adaptive_avg_pool3d(v[None, None], (size, size, size))[0, 0]
            out.append(v.cpu().numpy().astype(np.float32))
        head = dict(stage=stage, it=int(it), n_iter=int(n_iter), res=res, apix=float(apix) * b / max(out[0].shape),
                    models=len(out), history=hist[-200:], updated=now, job=job.name)
        try:
            q.put_nowait((out, head))
        except queue.Full:                           # the writer is busy: this snapshot is skipped
            pass

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
        for p in written | {os.path.join(job.path, "live3d.json")}:
            with contextlib.suppress(OSError):
                os.remove(p)
