"""The job clock: every "(N s)" progress time in a job's run log counts from the job's start (project.run_log sets it;
the scheduler's run_started when the job got a process of its own), not from the start of the phase that logs it."""
from __future__ import annotations

import time

_start: float | None = None
_jobs: set = set()


def start(t0: float, jobs=()) -> bool:
    """Start the clock at t0 for these job ids; False (nothing changed) when a job is already timing (nested logs)."""
    global _start, _jobs
    if _start is not None:
        return False
    _start, _jobs = t0, set(jobs)
    return True


def stop():
    global _start, _jobs
    _start, _jobs = None, set()


def since(t0: float) -> float:
    """Seconds since the job started, or since t0 outside a job (or when t0 is earlier)."""
    return time.time() - (_start if _start is not None and _start <= t0 else t0)


def job_seconds(job_id) -> float | None:
    """Seconds since the job started, None when this process is not timing that job."""
    return time.time() - _start if _start is not None and job_id in _jobs else None
