"""Final marks, the publication report (flowchart, methods, settings) from the final jobs' lineage, and jobs whose
process died no longer shown running.

    pytest -q tests/test_report.py
"""
import os

from celeris2d.project import Project
from celeris2d.report import chain, job_settings, project_report


def _project(tmp_path):
    p = Project(str(tmp_path / "prj"))
    imp = p.new_job("Import", dict(root="/data", pattern="*.tif"), [], "movies")
    p.set_status(imp, "finished", dict(movies=24))
    c2 = p.new_job("Class2D", dict(n_classes=80, diameter_a=180.0), [imp], "k80")
    p.set_status(c2, "finished", dict(particles=1000, classes=80))
    r1 = p.new_job("Refine3D", dict(sym="C8", nu=True, particles=c2.id), [c2], "nu")
    p.set_status(r1, "finished", dict(particles=1000, particles_used=900, resolution_solvent_mask_a=3.6,
                                     resolution_unmasked_a=4.0, map="jobs/x/map.mrc"))
    side = p.new_job("Refine3D", dict(sym="C1"), [c2], "side")     # not in the final chain
    p.set_status(side, "finished", dict(resolution_solvent_mask_a=4.5))
    return p, imp, c2, r1, side


def test_final_report(tmp_path):
    p, imp, c2, r1, side = _project(tmp_path)
    p.mark_final(r1, note="Figure 2")
    assert list(p.final_marks()) == [r1.id] and p.final_marks()[r1.id]["note"] == "Figure 2"
    assert [j.id for j in chain(p, [r1.id])] == [imp.id, c2.id, r1.id]           # the side branch is not in it
    shown, full = job_settings(p.job(r1.id))
    assert shown["sym"] == "C8" and shown["nu"] is True and "particles" not in full and "run_pid" not in full
    r = project_report(p)
    for k in ("flowchart", "flowchart_svg", "methods", "settings"):
        assert os.path.exists(p.abs(r[k])), k
    text = open(p.abs(r["methods"])).read()
    assert "Refine3D" in text and "3.6" in text and "Figure 2" in text and "side" not in text
    p.mark_final(r1, final=False)
    assert p.final_marks() == {}


def test_dead_running_job_is_failed(tmp_path):
    p, *_ = _project(tmp_path)
    j = p.new_job("Class2D", dict(n_classes=10), [], "dies")                  # status running, this process
    assert p.reap_stale() == []                                              # its process is alive
    p.update_params(j, run_pid=2 ** 22 + 12345)                              # a process that does not exist
    assert p.reap_stale() == [j.name] and p.job(j.id).status == "failed"
