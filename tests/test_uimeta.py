"""The plain-language catalog (uimeta.py) covers every job type, setting and input of jobs.registry() and
moviejobs.movie_registry(), with valid levels, groups and choices. No GPU."""
import dataclasses

import pytest

from celeris2d import jobs, jobspec, moviejobs, uimeta

SPECS = {**moviejobs.movie_registry(), **jobs.registry()}


def test_every_type_is_in_a_group():
    grouped = [t for _, ts in uimeta.GROUPS for t in ts]
    assert len(grouped) == len(set(grouped)), "a type is in two groups"
    assert set(grouped) == set(SPECS), (sorted(set(SPECS) - set(grouped)), sorted(set(grouped) - set(SPECS)))
    assert set(uimeta.TYPES) == set(SPECS), (sorted(set(SPECS) - set(uimeta.TYPES)),
                                             sorted(set(uimeta.TYPES) - set(SPECS)))


@pytest.mark.parametrize("t", sorted(SPECS))
def test_type_entry(t):
    sp, m = SPECS[t], uimeta.TYPES[t]
    for k in ("name", "blurb", "needs"):
        assert isinstance(m.get(k), str) and m[k].strip(), f"{t}: {k}"
    names = [f.name for f in dataclasses.fields(sp.settings)]
    missing = [n for n in names if n not in m["fields"]]
    extra = [n for n in m["fields"] if n not in names]
    assert not missing and not extra, f"{t}: settings without an entry {missing}, entries without a setting {extra}"
    kinds = {f["name"]: f["kind"] for f in jobspec.schema(sp.settings)}
    defaults = {f["name"]: f["default"] for f in jobspec.schema(sp.settings)}
    for n, e in m["fields"].items():
        assert e["label"].strip() and e["help"].strip(), f"{t}.{n}"
        assert e["level"] in ("basic", "advanced"), f"{t}.{n}"
        assert e["level"] == "basic" or e["section"], f"{t}.{n}: an advanced setting needs a section"
        if "choices" in e:
            vals = [c[0] for c in e["choices"]]
            assert all(len(c) == 2 for c in e["choices"]), f"{t}.{n}"
            d = defaults[n]
            if kinds[n] != "list" and not e.get("free") and d is not None:
                assert d in vals, f"{t}.{n}: default {d!r} not among the choices {vals}"
    # a required setting (no default) must be shown to every user
    required = [f.name for f in dataclasses.fields(sp.settings)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    assert all(m["fields"][n]["level"] == "basic" for n in required), f"{t}: required settings must be basic"
    ins = [i.name for i in sp.inputs]
    assert sorted(m["inputs"]) == sorted(ins), f"{t}: inputs {ins} vs entries {sorted(m['inputs'])}"
    for n, e in m["inputs"].items():
        assert e["label"].strip() and e["help"].strip(), f"{t} input {n}"


def test_annotate():
    out = uimeta.annotate([jobspec.describe(sp) for sp in SPECS.values()])
    assert [d["order"] for d in out] == sorted(d["order"] for d in out)
    assert [d["type"] for d in out] == [t for _, ts in uimeta.GROUPS for t in ts]
    assert [d["group_order"] for d in out] == sorted(d["group_order"] for d in out)
    r = {d["type"]: d for d in out}["Refine3D"]
    assert r["name"] == "3D refinement" and r["group"] == "3D"
    s = {f["name"]: f for f in r["settings"]}
    assert s["sym"]["level"] == "basic" and s["sym"]["default"] == "C1" and s["sym"]["code_help"]
    assert r["inputs"][1]["label"] and r["inputs"][1]["optional"] and r["inputs"][1]["types"]
    # a type, setting or input missing from the catalog still works
    d = uimeta.annotate([dict(type="NewJob", title="a new job", inputs=[dict(name="particles", doc="ptcls")],
                              settings=[dict(name="box_a", kind="float", default=0.0, help="the box")])])[0]
    assert d["name"] == "NewJob" and d["group"] == uimeta.FALLBACK_GROUP and d["blurb"] == "a new job"
    assert d["settings"][0]["label"] == "Box (Å)" and d["settings"][0]["help"] == "the box"
    assert d["inputs"][0]["label"] == "Particles"
    assert uimeta.display_name("Select") == "Select 2D classes" and uimeta.display_name("MotionCtf").startswith("Motion")
