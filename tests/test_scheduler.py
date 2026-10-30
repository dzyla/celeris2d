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
