"""Every registered job type (jobs.registry()) is checked the same way, so a new type is tested by being registered:
its settings build from defaults and from form values, every field has help for the forms, stored parameters
survive unknown and missing keys and older settings versions, inputs are type-checked, the key numbers never fail,
and the dashboard / generic endpoints serve it. No GPU work: jobs are made, not run."""
import dataclasses
import inspect
import json

import pytest

from celeris2d import jobs, jobspec
from celeris2d.project import Job, Project

REG = jobs.registry()


@pytest.mark.parametrize("t", sorted(REG))
def test_spec(t):
    sp = REG[t]
    s = sp.settings()
    d = jobspec.describe(sp)
    names = [f["name"] for f in d["settings"]]
    assert names == [f.name for f in dataclasses.fields(sp.settings)]
    assert all(f["help"] for f in d["settings"]), [f["name"] for f in d["settings"] if not f["help"]]
    # form values (strings, as a page sends them) give back the defaults
    form = {f["name"]: (f["default"] if f["kind"] == "bool" else
                        ", ".join(map(str, f["default"])) if f["kind"] == "list" else str(f["default"]))
            for f in d["settings"]}
    assert sp.settings(**jobspec.coerce(sp.settings, form)) == s
    with pytest.raises(ValueError):
        jobspec.coerce(sp.settings, {"no_such_setting": 1})
    # stored parameters: version recorded, unknown keys ignored, missing ones defaulted
    p = jobspec.params_of(s)
    assert p["settings_version"] == jobspec.version_of(sp.settings)
    job = Job(1, t, "001_x", "x", dict(json.loads(json.dumps(p)), retired_setting=3), [], "finished", "/tmp")
    assert jobspec.settings_of(job, sp.settings) == s
    cur = {"settings_version": jobspec.version_of(sp.settings)}         # no version stored = version 1 (migrated)
    assert jobspec.settings_of(Job(1, t, "001_x", "x", cur, [], "finished", "/tmp"), sp.settings) == s
    # inputs are make()'s positional parameters after the project, in order
    params = list(inspect.signature(sp.make).parameters)[1:1 + len(sp.inputs)]
    assert len(params) == len(sp.inputs)
    # key numbers: never an exception, also on an empty or odd summary
    for summ in ({}, {"resolution_a": 3.1, "particles": 1000, "classes": [{"resolution_a": 4.0}]},
                 {"classes": 50, "particles": "?"}):
        assert isinstance(jobspec.key_of(t, summ, REG), str)


def test_sharpen_settings_migration():
    from celeris2d.sharpen import SharpenSettings
    old = Job(1, "Sharpen", "001_s", "s", {"mask": "", "b_factor": -50}, [], "finished", "/tmp")   # version 1
    assert jobspec.settings_of(old, SharpenSettings).mask == "refine"
    new = Job(1, "Sharpen", "001_s", "s", jobspec.params_of(SharpenSettings()), [], "finished", "/tmp")
    assert jobspec.settings_of(new, SharpenSettings).mask == "auto"


def _project(tmp_path):
    prj = Project(str(tmp_path / "prj"))
    ex = prj.new_job("Extract", {}, [], "fake")
    prj.set_status(ex, "finished", dict(particles=10, box=64))
    c2 = prj.new_job("Class2D", jobspec.params_of(jobs.Class2DSettings()) | {"particles": ex.id}, [ex], "fake")
    prj.set_status(c2, "finished", dict(particles=10, classes=50))
    return prj, ex, c2


def test_new_job_checks(tmp_path):
    prj, ex, c2 = _project(tmp_path)
    j = jobs.new_job(prj, "Class2D", {"particles": ex.id}, {"n_classes": "20", "seeds": 2})
    assert j.params["n_classes"] == 20 and j.params["seeds"] == 2 and j.params["settings_version"] == 1
    with pytest.raises(ValueError, match="must be a"):
        jobs.new_job(prj, "Class2D", {"particles": c2.id})              # a Class2D is no particle input
    with pytest.raises(ValueError, match="required"):
        jobs.new_job(prj, "Class2D", {})
    with pytest.raises(ValueError, match="unknown settings"):
        jobs.new_job(prj, "Class2D", {"particles": ex.id}, {"n_clases": 20})
    with pytest.raises(ValueError, match="unknown job type"):
        jobs.new_job(prj, "Refine2D", {})
    prj.write_index()
    assert "50" in open(tmp_path / "prj" / "JOBS.md").read()


def test_project_api(tmp_path):
    from fastapi.testclient import TestClient
    from celeris2d.webapp import create
    prj, ex, c2 = _project(tmp_path)
    c = TestClient(create(str(tmp_path / "web"), project_dir=prj.root))
    t = {x["type"]: x for x in c.get("/api/project/jobtypes").json()}
    assert set(t) == set(REG) and t["Refine3D"]["inputs"][2]["optional"]
    lst = c.get("/api/project").json()["jobs"]
    assert [x["key"] for x in lst] == ["10 ptcls · 64 px", "10 ptcls · K50"]
    r = c.post("/api/project/new", json={"type": "InitialModel", "inputs": {"class2d": ex.id}})
    assert r.status_code == 400 and "must be a" in r.json()["detail"]
    r = c.post("/api/project/new", json={"type": "Nope", "inputs": {}})
    assert r.status_code == 400
    assert c.get("/api/project/log", params={"job": ex.id}).json()["text"] == ""
