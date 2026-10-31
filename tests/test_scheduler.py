"""The dashboard's job queue: a queued job runs only when its inputs have finished, fails with the reason when one
of them failed, waits behind a Select job that waits for the user; waiting / queued jobs can be cancelled.

    pytest -q tests/test_scheduler.py
"""
from celeris2d.project import Project
from celeris2d.scheduler import cancel, next_job, queue_state, waiting_select


def _job(p, type_, inputs=(), status="finished", label=""):
    j = p.new_job(type_, {}, list(inputs), label)
    p.set_status(j, status, {})
    return j


def test_queue_order_dependencies_and_failures(tmp_path):
    p = Project(str(tmp_path / "prj"))
    c2 = _job(p, "Class2D", label="c2")
    sel = waiting_select(p, c2)                                   # waits for the user's choice of classes
    assert p.job(sel.id).status == "waiting" and p.job(sel.id).params["source"] == c2.id
    im = _job(p, "InitialModel", [sel], "queued", "after_choice")  # queued on the waiting selection
    r3 = _job(p, "Refine3D", [c2], "queued", "ready")             # its input is finished
    assert next_job(p).id == r3.id                                # the waiting selection holds InitialModel back
    p.set_status(p.job(r3.id), "finished", {})
    assert next_job(p) is None
    st = queue_state(p)
    assert st["waiting"] == [sel.name] and st["queued"] == [im.name]
    cancel(p, sel)                                                # the user drops the selection
    assert next_job(p) is None and p.job(im.id).status == "failed"
    assert "did not finish" in p.summary_of(im)["error"]


def test_finished_input_releases_queued_job(tmp_path):
    p = Project(str(tmp_path / "prj"))
    a = _job(p, "Class2D", status="queued", label="first")
    b = _job(p, "Refine3D", [a], "queued", "second")
    assert next_job(p).id == a.id                                 # no inputs: runs first
    p.set_status(p.job(a.id), "finished", {})
    assert next_job(p).id == b.id


def test_stop_running_job_and_job_process(tmp_path):
    import os
    import subprocess
    import time
    from celeris2d.scheduler import start, stop
    p = Project(str(tmp_path / "prj"))
    j = _job(p, "Refine3D", status="queued", label="long")
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)   # stands in for the job's process
    p.set_status(j, "running", pid=proc.pid)
    assert p.is_running(p.job(j.id))
    stop(p, j, grace=2)
    assert proc.wait(timeout=10) != 0
    assert p.job(j.id).status == "cancelled" and p.summary_of(j)["error"] == "stopped by the user"
    after = _job(p, "Class3D", [j], "queued", "after")                  # queued on a stopped job: fails
    assert next_job(p) is None and p.job(after.id).status == "failed"
    try:
        stop(p, j)
        raise AssertionError("a cancelled job cannot be stopped")
    except ValueError:
        pass
    bad = _job(p, "Select", status="queued", label="noqueue")         # its own process records how it ended
    pr = start(p, bad, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert p.job(bad.id).params["run_pid"] == pr.pid and pr.wait(timeout=120) != 0
    assert p.job(bad.id).status == "failed" and "cannot run from the queue" in p.summary_of(bad)["error"]
    assert os.getpid() != pr.pid


def test_movie_jobs_from_the_dashboard(tmp_path):
    """Import from a folder is finished when made; MotionCtf -> Pick -> Extract queued on each other run as one
    chain; an empty optional field (Pick threshold) is None, an empty required one an error."""
    import numpy as np
    import pytest
    from celeris2d.jobspec import coerce
    from celeris2d.pipeline import MotionSettings, PickSettings
    from celeris2d.scheduler import movie_chain, queue_job
    mv = tmp_path / "movies"
    mv.mkdir()
    for i in range(2):
        (mv / f"m{i}.tif").write_bytes(np.zeros(16, np.uint8).tobytes())
    (mv / "gain.tif").write_bytes(b"x")
    p = Project(str(tmp_path / "prj"))
    imp = queue_job(p, "Import", {}, {"movies": str(mv)})
    assert imp.status == "finished" and len(p.movies(imp)) == 2          # the gain reference is not a movie
    mc = queue_job(p, "MotionCtf", {"movies": imp.id}, {"apix": "1.1", "dose_total": "50"})
    pk = queue_job(p, "Pick", {"micrographs": mc.id}, {"diameter_a": "130", "threshold": ""})
    ex = queue_job(p, "Extract", {"picks": pk.id}, {"particle_a": "170"})
    assert p.job(pk.id).params["threshold"] is None
    assert [j.id for j in movie_chain(p, p.job(mc.id))] == [mc.id, pk.id, ex.id]
    assert next_job(p).id == mc.id
    with pytest.raises(ValueError, match="apix is required"):
        coerce(MotionSettings, {"apix": "", "dose_total": "50"})
    assert coerce(PickSettings, {"threshold": "0.3"})["threshold"] == 0.3


def test_streaming_class2d_starts_beside_its_running_extraction(tmp_path):
    p = Project(str(tmp_path / "prj"))
    ex = _job(p, "Extract", status="running", label="ex")
    plain = p.new_job("Class2D", {}, [ex], "plain")
    p.set_status(plain, "queued", {})
    live = p.new_job("Class2D", {"stream": True}, [ex], "live")
    p.set_status(live, "queued", {})
    assert next_job(p).id == live.id                        # the streaming one may start; the plain one waits
    assert next_job(p, stream_only=True).id == live.id
    p.set_status(p.job(live.id), "running", {})
    assert next_job(p) is None
    p.set_status(p.job(ex.id), "finished", {})
    assert next_job(p).id == plain.id and next_job(p, stream_only=True) is None


def test_light_jobs_beside_a_running_gpu_job(tmp_path):
    """While a GPU job runs, only a quick map job (scheduler.LIGHT_JOBS) is offered, to run on the CPU beside it."""
    from celeris2d.scheduler import LIGHT_JOBS
    p = Project(str(tmp_path / "p"))
    ref = _job(p, "Refine3D", label="done")
    _job(p, "Refine3D", status="running", label="gpu")
    _job(p, "Class3D", [ref], "queued", "waits")
    sh = _job(p, "Sharpen", [ref], "queued", "quick")
    assert "Sharpen" in LIGHT_JOBS
    assert next_job(p, light_only=True).id == sh.id
    assert next_job(p).type == "Class3D"            # the GPU queue itself is unchanged
