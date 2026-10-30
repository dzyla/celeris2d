"""Job queue of a project for the dashboard: jobs are made at once with status "queued" and run one at a time (one GPU
job) as soon as every input has finished; a job whose input failed fails too, with the reason. After a
classification (Class2D, HeteroRefine, Class3D) run from the queue, a Select job is made in status "waiting": it
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

from .project import Job, Project

AUTO_SELECT = ("Class2D", "HeteroRefine", "Class3D")
MOVIE_JOBS = ("MotionCtf", "Pick", "Extract")
DONE_OK = ("finished",)
DONE_BAD = ("failed", "cancelled", "deleted")


def queue_job(project: Project, job_type: str, inputs: dict, settings: dict | None = None, label: str = "") -> Job:
    """A new job of any registered type, queued (inputs may themselves be queued or waiting)."""
    from .jobs import new_job
    j = new_job(project, job_type, inputs, settings or {}, label)
    project.set_status(j, "queued")
    return j


def queue_rerun(project: Project, ref) -> Job:
    """Run a job again through the queue: a movie job carries on with new movies, a particle job is copied to a new
    job with the same inputs and settings (jobs.rerun)."""
    from .jobs import registry
    from .jobspec import settings_of
    old = project.job(ref)
    if old.type in MOVIE_JOBS:
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


def cancel(project: Project, job) -> Job:
    j = project.job(job)
    if j.status not in ("queued", "waiting"):
        raise ValueError(f"{j.name} is {j.status}: only queued or waiting jobs can be cancelled")
    project.set_status(j, "cancelled", dict(project.summary_of(j), error="cancelled"))
    return j


def next_job(project: Project) -> Job | None:
    """The oldest queued job whose inputs have all finished; queued jobs with a failed / cancelled input fail."""
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
        if all(status.get(i) in DONE_OK for i in j.inputs):
            return j
    return None


def run_job(project: Project, job, log=print, auto_select: bool = True):
    """Run a queued job (on the caller's thread); after a classification, a waiting Select job."""
    from . import pipeline as pl
    from .jobs import registry
    j = project.job(job)
    project.set_status(j, "running")
    try:
        if j.type in MOVIE_JOBS:
            pl.run(project, [j], log=log)
        else:
            spec = registry().get(j.type)
            if spec is None:
                raise ValueError(f"{j.type} jobs cannot run from the queue")
            spec.run(project, j, log=log)
    except BaseException as e:                    # noqa: BLE001 - recorded on the job, re-raised for the caller
        project.set_status(j, "failed", dict(project.summary_of(j), error=f"{type(e).__name__}: {e}"))
        raise
    j = project.job(j.id)
    if auto_select and j.type in AUTO_SELECT and j.status == "finished":
        waiting_select(project, j)
    return j


def queue_state(project: Project) -> dict:
    """Queued, waiting and running jobs (names, oldest first) for the dashboard."""
    jobs = project.jobs()
    pick = lambda st: [n for n in jobs[jobs["status"] == st]["name"]]
    return dict(running=pick("running"), queued=pick("queued"), waiting=pick("waiting"))


def start(project: Project, job, auto_select: bool = True, **popen) -> subprocess.Popen:
    """Run a queued job in a process of its own (main() below), so that stop() can end it - GPU memory and the
    movie reader processes with it. The caller waits for the process; the job records its status itself."""
    import celeris2d
    j = project.job(job)
    env = dict(os.environ)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(celeris2d.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([pkg] + [p for p in [env.get("PYTHONPATH")] if p])
    cmd = [sys.executable, "-u", "-m", "celeris2d.scheduler", project.root, str(j.id)]
    if not auto_select:
        cmd.append("--no-auto-select")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True, **popen)   # its own process group: stop() ends all
    project.set_status(j, "running", pid=proc.pid)
    return proc


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
