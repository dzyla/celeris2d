"""Job queue of a project for the dashboard: jobs are made at once with status "queued" and run one at a time (one GPU
job) as soon as every input has finished; a job whose input failed fails too, with the reason. After a
classification (Class2D, HeteroRefine, Class3D, InitialModel with several classes) run from the queue, a Select job is made in status "waiting": it
waits for the user to choose the classes in the dashboard (complete_select), and jobs queued on it run after that.
Agents (MCP) and scripts call the jobs directly and get no waiting selections.

Statuses: queued (waits for its inputs / the GPU), waiting (waits for the user), running, finished, failed,
cancelled. Each queued job runs in a process of its own (start), so a running job can be stopped (stop). The queue is the project's catalog itself, so it survives a restart of the dashboard.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time

from .project import Job, Project, has_classes

AUTO_SELECT = ("Class2D", "HeteroRefine", "Class3D")    # and an InitialModel with several classes (has_classes)
MOVIE_JOBS = ("MotionCtf", "Pick", "Extract")
DONE_OK = ("finished",)
DONE_BAD = ("failed", "cancelled", "deleted")


def queue_job(project: Project, job_type: str, inputs: dict, settings: dict | None = None, label: str = "") -> Job:
    """A new job of any registered type, queued (inputs may themselves be queued or waiting)."""
    from .jobs import new_job
    j = new_job(project, job_type, inputs, settings or {}, label)
    if j.status not in ("finished", "waiting"):  # an Import is done when made; a ManualPick waits for the user
        project.set_status(j, "queued")
    return j


def queue_rerun(project: Project, ref) -> Job:
    """Run a job again through the queue: a movie job carries on with new movies, a particle job is copied to a new
    job with the same inputs and settings (jobs.rerun)."""
    from .jobs import registry
    from .jobspec import settings_of
    old = project.job(ref)
    if old.type in MOVIE_JOBS or old.type == "ManualPick":     # ManualPick: Pick all again (with its picks now)
        project.set_status(old, "queued")
        return old
    spec = registry().get(old.type)
    if spec is None:
        raise ValueError(f"{old.type} jobs are not run again (make a new one)")
    j = spec.make(project, *(old.params.get(i.name) for i in spec.inputs), settings_of(old, spec.settings))
    project.set_status(j, "queued")
    return j


def waiting_select(project: Project, source) -> Job:
    """A Select job of `source`'s classes that waits for the user's choice."""
    src = project.job(source)
    j = project.new_job("Select", dict(source=src.id, waiting=True, note="choose the classes to keep"), [src],
                        f"of{src.id:03d}")
    project.set_status(j, "waiting", dict(waiting=f"choose the classes of {src.name} to keep"))
    return j


def complete_select(project: Project, job, classes: list[int], note: str = "") -> Job:
    """The user's choice for a waiting Select job (or a new Select job on a classification job)."""
    from .jobs import select_classes
    j = project.job(job)
    if j.type == "Select":
        if j.status != "waiting":
            raise ValueError(f"{j.name} is {j.status}, not waiting for a choice")
        return select_classes(project, j.params["source"], classes, note=note, job=j)
    return select_classes(project, j, classes, note=note)


def change_settings(project: Project, job, values: dict) -> Job:
    """New settings for a job that has not started (queued, or waiting for the user - e.g. a Pick job without a
    particle size, made before the micrographs could be seen). Settings that name another job (templates, a
    selection, ...) change the job's inputs: make a new job for those. A waiting Pick job that now has a size is
    queued; a default label follows the new settings."""
    from dataclasses import asdict
    from .jobs import registry
    from .jobspec import coerce, settings_of
    from .moviejobs import movie_registry
    from .uimeta import TYPES
    from . import pipeline as pl
    j = project.job(job)
    if j.status not in ("queued", "waiting") or j.type == "Select":
        raise ValueError(f"{j.name} is {j.status}: only the settings of a queued or waiting job can be changed")
    spec = registry().get(j.type) or movie_registry().get(j.type)
    if spec is None:
        raise ValueError(f"{j.type} jobs have no settings to change")
    old = settings_of(j, spec.settings)
    new = spec.settings(**dict(asdict(old), **coerce(spec.settings, values)))
    refs = {k for k, f in TYPES.get(j.type, {}).get("fields", {}).items() if f.get("job_types")}
    moved = [k for k in refs if getattr(new, k) != getattr(old, k)]
    if moved:
        raise ValueError(f"{', '.join(moved)} names another job: make a new job for that")
    from .manualpick import manualpick_label
    labels = {"Pick": pl.pick_label, "ManualPick": manualpick_label}
    auto = labels[j.type](old) if j.type in labels else None
    project.update_params(j, **asdict(new))
    if j.type == "Extract":                      # the box follows the particle size
        project.update_params(j, **pl.extract_box(project, project.job(j.params["picks"]), new) or dict(box=0))
    for e in project.jobs("Extract").itertuples():    # Extract jobs queued on a Pick job: their box from its new size
        ej = project.job(int(e.id))
        if j.type == "Pick" and ej.status == "queued" and int(ej.params.get("picks", -1)) == j.id:
            es = settings_of(ej, pl.ExtractSettings)
            project.update_params(ej, **pl.extract_box(project, project.job(j.id), es) or dict(box=0))
    j = project.job(j.id)
    if auto is not None and j.label == auto and labels[j.type](new) != auto:
        j = project.rename(j, labels[j.type](new))
    if j.type == "Pick" and j.status == "waiting" and new.diameter_a > 0:
        summ = {k: v for k, v in project.summary_of(j).items() if k != "waiting"}
        project.set_status(j, "queued", summ)
    elif j.type == "Pick" and j.status == "queued" and not new.diameter_a > 0:
        project.set_status(j, "waiting", dict(project.summary_of(j), waiting="set the particle size"))
    return project.job(j.id)


def cancel(project: Project, job) -> Job:
    j = project.job(job)
    if j.status not in ("queued", "waiting"):
        raise ValueError(f"{j.name} is {j.status}: only queued or waiting jobs can be cancelled")
    project.set_status(j, "cancelled", dict(project.summary_of(j), error="cancelled"))
    return j


def streaming(j: Job) -> bool:
    """A Class2D job that classifies while its particles are still being extracted (Class2D stream=True)."""
    return j.type == "Class2D" and bool(j.params.get("stream"))


# quick jobs on one map that run on the CPU beside a running GPU job (seconds: Sharpen with local resolution at box
# 240 took 9 s), so they do not wait for a refinement, nor take GPU memory it planned on
LIGHT_JOBS = ("Sharpen", "MaskCreate")


def next_job(project: Project, stream_only: bool = False, light_only: bool = False) -> Job | None:
    """The oldest queued job whose inputs have all finished - or, for a streaming Class2D, whose Extract job is
    running (it runs beside that job; stream_only: only such jobs, for when another job holds the GPU; light_only:
    only LIGHT_JOBS, for the CPU beside it); queued jobs with a failed / cancelled input fail."""
    jobs = project.jobs(include_deleted=True)
    status = dict(zip(jobs["id"].astype(int), jobs["status"]))
    for r in jobs[jobs["status"] == "queued"].itertuples():
        j = project.job(int(r.id))
        bad = [i for i in j.inputs if status.get(i, "deleted") in DONE_BAD]
        if bad:
            names = ", ".join(project.job(i).name for i in bad if i in status)
            project.set_status(j, "failed", dict(error=f"input {names or bad} did not finish"))
            status[j.id] = "failed"
            continue
        if streaming(j) and all(status.get(i) in ("running", "finished") for i in j.inputs) \
                and any(status.get(i) == "running" for i in j.inputs):
            return j
        if light_only and j.type not in LIGHT_JOBS:
            continue
        if not stream_only and all(status.get(i) in DONE_OK for i in j.inputs):
            return j
    return None


def movie_chain(project: Project, job) -> list[Job]:
    """A movie job and the queued movie jobs right after it (MotionCtf -> Pick -> Extract) whose other inputs have
    finished: pipeline.run does them together, overlapped, reading each movie once."""
    j = project.job(job)
    chain = [j]
    jobs = project.jobs()
    status = dict(zip(jobs["id"].astype(int), jobs["status"]))
    nxt = {"MotionCtf": ("Pick", "micrographs"), "Pick": ("Extract", "picks")}
    while chain[-1].type in nxt:
        typ, key = nxt[chain[-1].type]
        cand = [project.job(int(r.id)) for r in jobs[(jobs["status"] == "queued") & (jobs["type"] == typ)].itertuples()]
        cand = [c for c in cand if int(c.params.get(key, -1)) == chain[-1].id
                and all(i == chain[-1].id or status.get(i) in DONE_OK for i in c.inputs)]
        if not cand:
            break
        chain.append(cand[0])
    return chain


def run_job(project: Project, job, log=print, auto_select: bool = True):
    """Run a queued job (on the caller's thread); after a classification, a waiting Select job."""
    from . import pipeline as pl
    from .jobs import registry
    j = project.job(job)
    chain = movie_chain(project, j) if j.type in MOVIE_JOBS else [j]
    for c in chain:                               # an error from an earlier attempt no longer applies
        project.set_status(c, "running", {k: v for k, v in project.summary_of(c).items() if k != "error"})
    try:
        if j.type in MOVIE_JOBS:
            pl.run(project, chain, log=log)
        else:
            from .moviejobs import movie_registry
            spec = registry().get(j.type) or movie_registry().get(j.type)    # ManualPick: its Pick all
            if spec is None:
                raise ValueError(f"{j.type} jobs cannot run from the queue")
            from .live import live_maps
            with live_maps(project, j):           # 3D jobs: their maps as they change, for the dashboard
                spec.run(project, j, log=log)
    except BaseException as e:                    # noqa: BLE001 - recorded on the job, re-raised for the caller
        for c in chain:
            if project.job(c.id).status == "running":
                project.set_status(c, "failed", dict(project.summary_of(c), error=f"{type(e).__name__}: {e}"))
        raise
    j = project.job(j.id)
    if auto_select and (j.type in AUTO_SELECT or (j.type == "InitialModel" and has_classes(j))) and j.status == "finished":
        waiting_select(project, j)
    return j


def queue_state(project: Project) -> dict:
    """Queued, waiting and running jobs (names, oldest first) for the dashboard."""
    jobs = project.jobs()
    pick = lambda st: [n for n in jobs[jobs["status"] == st]["name"]]
    return dict(running=pick("running"), queued=pick("queued"), waiting=pick("waiting"))


def start(project: Project, job, auto_select: bool = True, device: str | None = None, **popen) -> subprocess.Popen:
    """Run a queued job in a process of its own (main() below), so that stop() can end it - GPU memory and the
    movie reader processes with it. The caller waits for the process; the job records its status itself."""
    import celeris2d
    j = project.job(job)
    env = dict(os.environ)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(celeris2d.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([pkg] + [p for p in [env.get("PYTHONPATH")] if p])
    if device:                                  # e.g. "cpu": a light job beside a running GPU job (gpu.kind)
        env["CELERIS2D_DEVICE"] = device
    cmd = [sys.executable, "-u", "-m", "celeris2d.scheduler", project.root, str(j.id)]
    if not auto_select:
        cmd.append("--no-auto-select")
    out = None
    if "stdout" not in popen:                   # a file, not a pipe: the job outlives a restart of the dashboard
        out = open(process_log(j), "ab")
        popen.update(stdout=out, stderr=subprocess.STDOUT)
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True, **popen)   # own process group: stop() ends all
    finally:
        if out is not None:
            out.close()
    project.set_status(j, "running", pid=proc.pid)
    return proc


def process_log(job: Job) -> str:
    """Everything the job's process prints (its run log, plus tracebacks)."""
    return os.path.join(job.path, f"j{job.id:03d}_process.log")


def busy(project: Project) -> Job | None:
    """A job of the project running in a live process (this dashboard's, an earlier one's, a script's): the queue
    waits for it - one GPU job at a time."""
    jobs = project.jobs()
    for r in jobs[jobs["status"] == "running"].itertuples():
        j = project.job(int(r.id))
        if project.is_running(j):
            return j
    return None


def stop(project: Project, job, grace: float = 15.0) -> Job:
    """Stop a running job: its process (and the processes it started) gets SIGTERM, SIGKILL after `grace` seconds;
    the job becomes cancelled ("stopped by the user") and jobs queued on it fail. Only processes on this host, and
    not the dashboard / an MCP server itself (a job run inside it cannot be stopped without stopping that)."""
    j = project.job(job)
    if j.status != "running":
        raise ValueError(f"{j.name} is {j.status}, not running")
    pid, host = j.params.get("run_pid"), j.params.get("run_host")
    if pid is None or host != socket.gethostname():
        raise ValueError(f"{j.name} runs on {host or 'an unknown host'}: stop it there")
    pid = int(pid)
    if pid == os.getpid():
        raise ValueError(f"{j.name} runs inside the dashboard itself: it cannot be stopped from here")
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        cmdline = ""
    if "mcp" in cmdline:
        raise ValueError(f"{j.name} runs inside the MCP server (process {pid}): stop it from there")
    group = os.getpgid(pid) == pid if cmdline else False             # a process started by start(): its group
    kill = (lambda s: os.killpg(pid, s)) if group else (lambda s: os.kill(pid, s))
    try:
        kill(signal.SIGTERM)
    except ProcessLookupError:
        pass                                                          # already gone: just record it
    project.set_status(j, "cancelled", dict(project.summary_of(j), error="stopped by the user"))
    for r in project.jobs().itertuples():       # the other jobs of the same process (a movie chain)
        if r.status == "running" and int(r.id) != j.id:
            o = project.job(int(r.id))
            if o.params.get("run_pid") == pid and o.params.get("run_host") == host:
                project.set_status(o, "cancelled", dict(project.summary_of(o), error=f"stopped with {j.name}"))

    def finish():
        t0 = time.time()
        while time.time() - t0 < grace:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)
        try:
            kill(signal.SIGKILL)
        except ProcessLookupError:
            pass
    threading.Thread(target=finish, daemon=True, name=f"stop-{j.id}").start()
    return j


def main(argv=None):
    """python -m celeris2d.scheduler <project> <job id> [--no-auto-select]: run one queued job (start())."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("project")
    ap.add_argument("job", type=int)
    ap.add_argument("--no-auto-select", action="store_true")
    a = ap.parse_args(argv)
    run_job(Project(a.project), a.job, auto_select=not a.no_auto_select)


if __name__ == "__main__":
    main()
