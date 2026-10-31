"""Dashboard endpoints of the ManualPick job (manualpick.py), for the picker page (web/picker.js):

    GET  /api/project/manualpick/state?job=          settings, every micrograph's state and pick counts, templates
                                                     (counts, score distributions), threshold, output, progress
    GET  /api/project/manualpick/mic?job=&stem=      one micrograph: pixel size, shape, display binning, picks,
                                                     rejected suggestions, state, suggestions (all candidates with
                                                     scores), the output's picks after Pick all
    POST /api/project/manualpick/picks               {job, stem, picks: [[x_a, y_a, src]], rejected: [[x_a, y_a]],
                                                     state}: saved as the user goes
    POST /api/project/manualpick/train               {job}: templates from the reviewed micrographs' picks
    POST /api/project/manualpick/suggest             {job, stems?, after?, n?}: the next micrographs picked with them
    POST /api/project/manualpick/threshold           {job, threshold?, power_lo?, power_hi?, zone_dark?, 
                                                     zone_mean?}: the filters
                                                     (null: no limit / zones by density off)
    POST /api/project/manualpick/off                 {job, off: [k, ...]}: templates switched off (0-based); their
                                                     picks leave the next trainings (junk)
    POST /api/project/manualpick/putback             {job}: every junk pick back into training
    POST /api/project/manualpick/settings            {job, diameter_a?, templates?}: at any time but while Pick all
                                                     runs (a new size drops the templates: train again)
    GET  /api/project/manualpick/templates.png?job=&k=  the templates side by side (white protein), or template k
    POST /api/project/manualpick/run                 {job}: Pick all, through the queue

The micrograph image itself is the dashboard's /api/project/micraw.png (micview.raw_png: binned to <= 1024 px).
Training and suggestions run in the dashboard process (seconds, on the GPU when it has room, else the CPU); one at
a time per job.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np


def _num(v, nd=None):
    """A catalog number for JSON (None for a missing one)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else (round(v) if nd is None else round(v, nd))


def router(project_of):
    """The endpoints, with project_of() giving the open project (webapp's _prj)."""
    from fastapi import APIRouter, Body, HTTPException
    from fastapi.responses import Response

    from . import manualpick as mp

    r = APIRouter(prefix="/api/project/manualpick")
    busy: dict = {}

    def job_of(name):
        prj = project_of()
        try:
            j = prj.job(name)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if j.type != "ManualPick":
            raise HTTPException(400, f"{j.name} is not a ManualPick job")
        return prj, j

    def guarded(j, what, fn):
        """One training / suggestion run per job at a time; ValueErrors are the user's to fix (400)."""
        if busy.get(j.id):
            raise HTTPException(409, f"{busy[j.id]} is running for {j.name}: wait for it")
        busy[j.id] = what
        try:
            return fn()
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        finally:
            busy.pop(j.id, None)

    @r.get("/state")
    def state(job: str):
        prj, j = job_of(job)
        from .figjson import _clean
        st = mp.load_state(j)
        mi = mp.micrographs(prj, j)
        ver = int(st.get("version", 0)) if mp.has_templates(j) else 0     # 0: none (or the size changed since)
        out_n = {}
        if j.status == "finished" or os.path.isdir(os.path.join(j.path, "picks")):
            out_n = {int(r_.movie_id): int(r_.n_kept) for r_ in prj.pick_table(j).itertuples()}
        rows = []
        for r_ in mi.itertuples():
            m = st["mics"].get(str(int(r_.movie_id)), {})
            sv = mp.suggestion_version(prj, j, r_) if m.get("state") == "suggested" else 0
            rows.append(dict(stem=r_.stem, movie_id=int(r_.movie_id), state=m.get("state", ""),
                             picks=len(m.get("picks", [])), rejected=len(m.get("rejected", [])),
                             suggested=sv, stale=bool(sv and sv != ver), output=out_n.get(int(r_.movie_id)),
                             defocus_a=_num(0.5 * (r_.defocus_u + r_.defocus_v)) if r_.defocus_u is not None else None,
                             ctf_fit_a=_num(r_.ctf_fit_res, 2)))
        summ = prj.summary_of(j)
        s = mp.settings(j)
        tr = st.get("train")
        return _clean(dict(job=j.name, status=j.status, settings=dict(diameter_a=s.diameter_a, dark=s.dark,
                           templates=s.templates, suggest=s.suggest, min_dist_frac=s.min_dist_frac),
                           mics=rows, threshold=st.get("threshold"), power_lo=st.get("power_lo"),
                           power_hi=st.get("power_hi", mp.MAX_STD), zone_dark=st.get("zone_dark", mp.ZONE_DARK),
                           zone_mean=st.get("zone_mean", mp.ZONE_MEAN),
                           off=st.get("off") or [], junk=sum(len(v) for v in (st.get("junk") or {}).values()),
                           version=ver, train=tr if ver else None, busy=busy.get(j.id),
                           summary={k: v for k, v in summ.items() if not isinstance(v, (list, dict))},
                           progress=summ.get("progress"), edited_after_run=bool(summ.get("t_run"))
                           and float(st.get("t_edit", 0)) > float(summ["t_run"]),
                           error=summ.get("error")))

    @r.get("/mic")
    def mic(job: str, stem: str):
        prj, j = job_of(job)
        from .figjson import _clean
        mi = mp.micrographs(prj, j)
        row = mi[mi["stem"] == os.path.basename(stem)]
        if not len(row):
            raise HTTPException(404, f"no micrograph {stem}")
        row = row.iloc[0]
        st = mp.load_state(j)
        m = st["mics"].get(str(int(row.movie_id)), {})
        ny, nx = int(row.ny), int(row.nx)
        out = None
        if m.get("state") != "reviewed" and os.path.isdir(os.path.join(j.path, "picks")):
            try:                                     # after Pick all: what the run picked here
                p = prj.picks(j, int(row.movie_id))
                if len(p["xy_a"]):
                    out = np.round(np.asarray(p["xy_a"])[np.asarray(p["keep"], bool)], 1).tolist()
            except Exception:                        # noqa: BLE001 - not picked (yet)
                out = None
        sg = mp.suggestions(prj, j, row) if m.get("state") == "suggested" else None
        return _clean(dict(stem=row.stem, movie_id=int(row.movie_id), apix=float(row.apix), shape=[ny, nx],
                           bin=max(1, int(math.ceil(max(ny, nx) / 1024))),       # micview.raw_png's binning
                           picks=m.get("picks", []), rejected=m.get("rejected", []), state=m.get("state", ""),
                           zones=m.get("zones", []),
                           suggest=sg, version=int(st.get("version", 0)) if mp.has_templates(j) else 0, output=out,
                           defocus_a=_num(0.5 * (row.defocus_u + row.defocus_v)) if row.defocus_u is not None
                           else None))

    @r.post("/picks")
    def picks(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        try:
            m = mp.set_picks(prj, j, os.path.basename(body["stem"]), body.get("picks"), body.get("rejected"),
                             body.get("state"), body.get("zones"))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        return dict(stem=body["stem"], picks=len(m["picks"]), state=m["state"])

    @r.post("/threshold")
    def threshold(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        num = lambda v: None if v in (None, "") else float(v)      # noqa: E731
        kw = {k: num(body[k]) for k in ("threshold", "power_lo", "power_hi", "zone_dark", "zone_mean") if k in body}
        return mp.set_threshold(prj, j, **kw)

    @r.post("/off")
    def off(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        off = mp.set_off(prj, j, body.get("off") or [])
        st = mp.load_state(prj.job(j.id))
        return dict(off=off, junk=sum(len(v) for v in (st.get("junk") or {}).values()))

    @r.post("/putback")
    def putback(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        return dict(put_back=mp.put_back(prj, j), junk=0, off=[])

    @r.post("/settings")
    def settings(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        s = guarded(j, "changing the settings", lambda: mp.set_settings(prj, j, body.get("diameter_a"),
                                                                       body.get("templates")))
        return dict(diameter_a=s.diameter_a, templates=s.templates)

    @r.post("/train")
    def train(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        return guarded(j, "training", lambda: mp.train(prj, j))

    @r.post("/suggest")
    def suggest(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        return dict(stems=guarded(j, "suggesting", lambda: mp.suggest(
            prj, j, stems=body.get("stems") or None, after=body.get("after") or "", n=body.get("n") or None)))

    @r.get("/templates.png")
    def templates_png(job: str, size: int = 96, k: int = -1):
        prj, j = job_of(job)
        tm, _ = mp.templates_of(j)
        if tm is None:
            raise HTTPException(404, "no templates yet")
        if k >= 0:
            if k >= len(tm):
                raise HTTPException(404, f"no template {k}")
            tm = tm[k:k + 1]
        from PIL import Image
        size = max(32, min(int(size), 256))
        tiles = []
        for t in tm:
            lo, hi = np.percentile(t, [1, 99.5])
            u8 = (np.clip((t - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
            tiles.append(np.asarray(Image.fromarray(u8).resize((size, size), Image.LANCZOS)))
        gap = np.zeros((size, 4), np.uint8)
        strip = np.concatenate([x for t in tiles for x in (t, gap)][:-1], 1)
        buf = io.BytesIO()
        Image.fromarray(strip).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-cache"})

    @r.post("/run")
    def run(body: dict = Body(...)):
        prj, j = job_of(body.get("job"))
        try:
            j = mp.queue_run(prj, j)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return dict(job=j.name, status=j.status)

    return r
