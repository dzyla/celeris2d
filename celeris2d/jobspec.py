"""Job types as data. The registry (jobs.registry()) holds one JobSpec per particle job type; every front end works
from it, so a new job type is one entry plus its settings dataclass and its make / run functions:

    rerun, the dashboard's "New job" form, MCP job_types / new_job, the job list's key numbers, JOBS.md

Settings dataclasses document each field with a trailing comment (`name: type = default   # help`, continued on
the following comment-only lines); schema() reads those comments, so the help in the forms is the code's own.

Versions: a settings class may declare VERSION (ClassVar[int], default 1) and a classmethod migrate(params,
from_version) -> params. A job stores settings_version with its parameters (params_of); settings_of upgrades an
older job's parameters before building the settings, then ignores unknown keys and gives missing ones their
defaults - adding a field needs no migration, renaming or changing a field's meaning does.
"""
from __future__ import annotations

import dataclasses
import inspect
import re
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class Input:
    name: str                       # parameter name in the job's params and make()'s position
    types: tuple                    # job types accepted
    optional: bool = False
    doc: str = ""


@dataclass(frozen=True)
class JobSpec:
    type: str
    title: str                      # one line: what the job does
    settings: type                  # dataclass of its parameters
    inputs: tuple                   # Input, in the order make() takes them
    make: typing.Callable           # make(project, *inputs, settings, label) -> Job
    run: typing.Callable            # run(project, job, log=) -> summary
    key: typing.Callable = None     # summary -> the job's key numbers (one short line)
    figures: tuple = ()             # its main figures, most telling first (reports show these)


def version_of(cls) -> int:
    return int(getattr(cls, "VERSION", 1))


def params_of(s) -> dict:
    """A settings object as the parameters a job stores (with its settings version)."""
    return dict(dataclasses.asdict(s), settings_version=version_of(type(s)))


def settings_of(job, cls):
    """A job's settings from its stored parameters: upgraded if they are from an older settings version, unknown
    keys ignored, missing ones at their defaults, lists back to the tuples the dataclass declares."""
    params = dict(job.params)
    v = int(params.get("settings_version", 1))
    if v < version_of(cls) and hasattr(cls, "migrate"):
        params = cls.migrate(params, v)
    out = {}
    for f in dataclasses.fields(cls):
        k = f.name
        if k in params:
            val = params[k]
            out[k] = tuple(val) if isinstance(val, list) and isinstance(f.default, tuple) else val
    return cls(**out)


_FIELD = re.compile(r"^\s+(\w+)\s*:\s*([^=#]+?)\s*(?:=\s*(.*?))?\s*(?:#\s?(.*))?$")
_MORE = re.compile(r"^\s+#\s?(.*)$")


def _help(cls) -> dict:
    """Field -> its trailing comment (with the comment-only lines that follow it)."""
    try:
        src = inspect.getsource(cls).splitlines()
    except (OSError, TypeError):
        return {}
    out, last = {}, None
    for line in src[1:]:
        m = _FIELD.match(line)
        if m and m.group(1) in {f.name for f in dataclasses.fields(cls)}:
            last = m.group(1)
            out[last] = (m.group(4) or "").strip()
            continue
        m2 = _MORE.match(line)
        if m2 and last is not None:
            out[last] = (out[last] + " " + m2.group(1).strip()).strip()
        else:
            last = None
    return out


def _kind(f, default) -> str:
    t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    t = re.sub(r"\s*\|\s*None$|^Optional\[(.*)\]$", r"\1", t.strip())    # float | None: a float that may be empty
    if isinstance(default, bool) or t == "bool":
        return "bool"
    if isinstance(default, int) or t == "int":
        return "int"
    if isinstance(default, float) or t == "float":
        return "float"
    if isinstance(default, (tuple, list)) or t.startswith(("tuple", "list")):
        return "list"
    return "str"


def schema(cls) -> list[dict]:
    """The settings dataclass as form fields: name, kind (int / float / bool / str / list), default, help."""
    hp = _help(cls)
    out = []
    for f in dataclasses.fields(cls):
        name = f.name
        if f.default is not dataclasses.MISSING:
            d = f.default
        elif f.default_factory is not dataclasses.MISSING:
            d = f.default_factory()
        else:
            d = None
        out.append(dict(name=name, kind=_kind(f, d), default=list(d) if isinstance(d, tuple) else d,
                        help=hp.get(name, "")))
    return out


def coerce(cls, values: dict) -> dict:
    """Form values (strings from a page, JSON from an agent) to the types of the settings fields; unknown names are
    an error (a typo must not be silently dropped)."""
    sch = {f["name"]: f for f in schema(cls)}
    kinds = {k: f["kind"] for k, f in sch.items()}
    bad = [k for k in values if k not in kinds]
    if bad:
        raise ValueError(f"unknown settings for {cls.__name__}: {bad} (known: {sorted(kinds)})")
    required = {f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    out = {}
    for k, v in values.items():
        kind = kinds[k]
        if v is None or (isinstance(v, str) and not v.strip() and kind != "str"):     # an empty form field
            if k in required:
                raise ValueError(f"{k} is required")
            if sch[k]["default"] is None:
                out[k] = None
            continue                                                                    # its default
        if kind == "bool":
            out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
        elif kind == "int":
            out[k] = int(float(v))
        elif kind == "float":
            out[k] = float(v)
        elif kind == "list":
            if isinstance(v, str):
                v = [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
            out[k] = tuple(int(x) if isinstance(x, str) and re.fullmatch(r"-?\d+", x) else x for x in v)
        else:
            out[k] = str(v)
    return out


def describe(spec: JobSpec) -> dict:
    """The job type for a form or an agent: inputs (with the job types they accept) and settings."""
    return dict(type=spec.type, title=spec.title, version=version_of(spec.settings),
                inputs=[dict(name=i.name, types=list(i.types), optional=i.optional, doc=i.doc) for i in spec.inputs],
                settings=schema(spec.settings), figures=list(spec.figures))


# key numbers of the movie jobs (not in the registry: they run through pipeline.run)
def _n(v, d=0):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def warn_mark(s: dict) -> str:
    return " · ⚠" if s.get("warnings") else ""


MOVIE_KEYS = {
    "Import": lambda s: f"{_n(s.get('movies'))} movies",
    "MotionCtf": lambda s: f"{_n(s.get('micrographs'))} mics",
    "Pick": lambda s: f"{_n(s.get('picks'))} picks",
    "Extract": lambda s: f"{_n(s.get('particles'))} ptcls · {s.get('box', '')} px",
    "Select": lambda s: f"{_n(s.get('particles'))} kept"
                        + (f" · {_n(s.get('rejected'))} rej" if s.get("rejected") is not None else ""),
}


def key_of(job_type: str, summary: dict, registry: dict | None = None) -> str:
    """The job's key numbers, as the job list and JOBS.md show them."""
    s = summary or {}
    if s.get("error"):
        return "failed: " + str(s["error"])[:80]
    if registry is None:
        from .jobs import registry as _r
        registry = _r()
    spec = registry.get(job_type)
    fn = spec.key if spec is not None and spec.key else MOVIE_KEYS.get(job_type)
    if fn is None:
        return ""
    try:
        return fn(s)
    except Exception:            # a summary from an older version: no key numbers rather than no job list
        return ""
