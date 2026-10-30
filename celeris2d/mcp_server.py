"""MCP server for celeris2d: fast GPU 2D classification with a person or an agent in the loop.

The server keeps one dataset loaded (spectra on the GPU), so every step after `load_particles` takes
seconds to a few minutes. Class decisions are never automatic: look at `show_classes`, then call
`select_good` (`suggest_good` only proposes).

Typical flow:
  load_particles -> classify (or isac) -> show_classes -> select_good -> [assign | deep2d] -> export

Projects (movies -> particles; docs/project-design.md): project_open -> import_movies -> process_movies (MotionCtf,
Pick, Extract jobs, in the background) -> job_status / job_report -> [rethreshold_picks | extract_particles] ->
load_particles(<project>/jobs/NNN_extract_...) -> classify -> select_good -> save_selection -> extract_particles(
select=...) at full size.  Every job is a folder jobs/NNN_<type>_<what it holds>; running a step again with other
settings makes a new job. Heavy data (micrographs, stacks) is in data/NNN: project_disk / purge_job / rebuild_job.

Run:  celeris2d-mcp            (stdio transport; register it in your MCP client config)
"""
from __future__ import annotations

import io
import json
import os
import threading

from mcp.server.mcpserver import Image, MCPServer

from .session import Session

server = MCPServer(
    name="celeris2d",
    instructions=(
        "Fast GPU 2D classification of cryo-EM particles (RELION .star, cryoSPARC .cs/.csg, or .mrcs). "
        "Load a dataset once, then classify / isac / show_classes / select_good / assign / export. "
        "Good/bad class decisions are made by you after looking at show_classes (suggest_good only proposes). "
        "Class numbers are 1-based and refer to the result they were shown for. "
        "From movies: project_open, import_movies, process_movies, job_status, job_report; an extract job folder "
        "is a valid load_particles path, and save_selection stores a 2D selection back in the project."),
)
_session: Session | None = None
_lock = threading.Lock()  # one GPU job at a time


def _s() -> Session:
    if _session is None:
        raise RuntimeError("no dataset loaded: call load_particles first")
    return _session


def _js(d) -> str:
    return json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


@server.tool()
def load_particles(path: str, diameter_a: float = 0.0, max_res_a: float = 6.0, gpus: str = "auto", subset: int = 0,
                   workdir: str = "", speed: str = "balanced", contrast: str = "auto") -> str:
    """Open particles (RELION .star, cryoSPARC .cs/.csg, or .mrc/.mrcs stack) and start streaming them to the GPU.
    path may be a Celeris2D Extract job folder (<project>/Extract/jobNNN).
    diameter_a: particle mask diameter in A (0: 200, or 1.25 x the particle size of an Extract job). max_res_a: finest resolution used. gpus: 'auto' or nvidia-smi
    indices like '0,2,3'. subset: random subset size for quick tests (0 = all). speed: fast|balanced|accurate.
    contrast: auto|invert|keep (protein must be white; auto decides from the particle average)."""
    global _session
    preset = dict(fast=dict(psi_factor=1.5, trans_div=2.5), balanced=dict(psi_factor=1.0, trans_div=3.0),
                  accurate=dict(psi_factor=1.0, trans_div=4.0))[speed]
    with _lock:
        if not workdir and os.path.isfile(os.path.join(path, "job.json")):    # a project job: keep 2D work in it
            workdir = os.path.join(os.path.abspath(path), "class2d")
        wd = workdir or os.path.join(os.path.dirname(os.path.abspath(path)), "celeris2d_session")
        _session = Session(wd, log=lambda *a: None)
        info = _session.load(path, diameter=diameter_a or None, max_res=max_res_a, gpus=gpus, subset=subset,
                             polarity=dict(auto=0, invert=-1, keep=1)[contrast], **preset)
    return _js(dict(info, workdir=wd))


@server.tool()
def optimize_settings(n_classes: int = 50, subset: int = 20000, apply: bool = True) -> str:
    """Try a small grid of settings (alignment margin, solvent mask threshold, hard/soft pose) on a random subset
    and keep the most reproducible one (two independent starts agree most). Applies the winner by default."""
    from .optimize import apply as apply_settings
    from .optimize import optimize
    with _lock:
        best, rows = optimize(_s().clf, n_classes, subset=subset, log=lambda *a: None)
        if apply:
            apply_settings(_s().clf, best)
    return _js(dict(best=best, applied=apply, trials=rows))


@server.tool()
def classify(name: str, n_classes: int = 50, seeds: int = 1, on: str = "all") -> str:
    """2D classification into n_classes. seeds > 1 repeats it from independent starts so that per-particle
    P(good) and votes become available after select_good (more robust selection, seeds x the time).
    on: 'all' or a particle set of an earlier result, e.g. 'run1:good', 'run1:bad', 'run1:ambiguous'."""
    with _lock:
        return _js(_s().classify(name, K=n_classes, seeds=seeds, on=on))


@server.tool()
def isac(name: str, on: str = "all", seeds: int = 4, particles_per_class: int = 300, threshold: float = 0.6) -> str:
    """ISAC-style stable groups: repeated independent classifications; only groups whose members keep landing
    together are kept (high purity, partial coverage, ~seeds x generations classifications - slower).
    Particles that are never stable are left out (label -1)."""
    with _lock:
        return _js(_s().isac(name, on=on, seeds=seeds, particles_per_class=particles_per_class, threshold=threshold))


@server.tool()
def assign(name: str, refs: str, on: str = "all", seeds: int = 2) -> str:
    """Classify particles starting from every class/group average of result `refs` (+ free classes for views the
    references miss). The good/bad selection of `refs` carries over to the classes seeded from it."""
    with _lock:
        return _js(_s().assign(name, refs=refs, on=on, seeds=seeds))


@server.tool()
def show_classes(name: str) -> list:
    """Gallery of a result (sorted by size; tile = class number, particles, share, confidence; selected-good
    classes outlined in green) plus the per-class table."""
    s = _s()
    summ = s.summary(name)
    with open(summ["gallery"], "rb") as fh:
        png = fh.read()
    return [_js(summ), Image(data=png, format="png")]


@server.tool()
def suggest_good(name: str) -> str:
    """Automatic proposal of good classes (2-cluster split on confidence/resolution, dots excluded). NOT applied:
    review show_classes and call select_good."""
    return _js(dict(result=name, suggested_good=_s().suggest(name)))


@server.tool()
def select_good(name: str, classes: list[int]) -> str:
    """Mark classes (1-based numbers from show_classes) as good; every other class becomes bad."""
    return _js(_s().select(name, good=classes))


@server.tool()
def particle_sets(name: str, min_p: float = 0.5) -> str:
    """Counts of good / ambiguous / bad particles for a result with a selection, at P(good) >= min_p."""
    sets = _s().sets(name, min_p)
    return _js({k: int(len(v)) for k, v in sets.items()})


@server.tool()
def export(name: str, outdir: str, min_p: float = 0.5, fmt: str = "auto") -> str:
    """Write good / ambiguous / bad particle sets with their 2D alignments: .star (RELION input) or .cs/.csg
    (cryoSPARC input); fmt = auto | star | cs."""
    with _lock:
        return _js(_s().export(name, outdir, min_p=min_p, fmt=fmt))


@server.tool()
def status() -> str:
    """Loaded dataset, GPUs and stored results."""
    if _session is None or _session.clf is None:
        return _js(dict(loaded=False))
    s = _session
    return _js(dict(loaded=True, path=s.path, particles=s.clf.N, loaded_so_far=s.clf._n_loaded,
                    gpus=[str(d) for d in s.clf.devs], workdir=s.workdir,
                    results={k: dict(kind=r.kind, particles=int(len(r.idx)), classes=int(len(r.images)),
                                     selected=r.good is not None) for k, r in s.results.items()}))


# ----------------------------------------------------------------------------------------------- projects
_project = None
_bg: dict = {}            # job dir -> dict(thread, progress, error)


def _p():
    if _project is None:
        raise RuntimeError("no project open: call project_open first")
    return _project


@server.tool()
def project_open(path: str) -> str:
    """Open (or create) a processing project directory. Returns its jobs with their key numbers."""
    global _project
    from .project import Project
    _project = Project(path)
    return _js(_project.summary())


@server.tool()
def import_movies(pattern: str, root: str = "", label: str = "") -> str:
    """Register movies matching a glob pattern (e.g. '/data/session1/**/*.tif') in a new Import job. root: the
    folder whose sub-directories are kept in output names (default: the movies' common parent)."""
    import glob
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise ValueError(f"no files match {pattern}")
    job = _p().import_movies(paths, root=root, label=label)
    return _js(dict(job=job.name, movies=len(paths), first=paths[0]))


def _background(jobs, movie_ids=None):
    from .pipeline import finish, run
    prj = _p()
    key = jobs[-1].name
    st = _bg[key] = dict(done=0, of=None, error=None, jobs=[j.id for j in jobs], started=__import__("time").time())

    def work():
        try:
            with _lock:
                st["result"] = run(prj, jobs, movie_ids=movie_ids, log=lambda *a: None,
                                   progress=lambda n, of: st.update(done=n, of=of))
        except Exception as e:                       # visible through job_status
            st["error"] = repr(e)
            for j in jobs:
                finish(prj, j, "failed")
    st["thread"] = threading.Thread(target=work, daemon=True)
    st["thread"].start()
    return key


@server.tool()
def process_movies(movies_job: str, apix: float, dose_total: float, diameter_a: float, particle_a: float = 0.0,
                   bin: float = 1.0, gain: str = "", kv: float = 300.0, cs_mm: float = 2.7, amp_contrast: float = 0.1,
                   dark: bool = True, box_a: float = 0.0, particle_apix: float = 0.0, dtype: str = "float16",
                   export: list[str] | None = None, figures: str = "outliers") -> str:
    """Motion correction + CTF, blob picking and extraction of an Import job's movies, as three new jobs
    (MotionCtf, Pick, Extract) run together in the background; poll job_status. apix: movie pixel size (bin 2 for
    super-resolution); dose_total e/A^2 over all frames; diameter_a: blob for picking; particle_a: longest particle
    dimension (default 1.3 x diameter) - the box is particle + CTF delocalisation unless box_a is given;
    particle_apix: bin particles to this pixel size; export: ['relion', 'cryosparc'] to also write their files;
    figures: all | outliers (+ the first micrograph) | none - any figure can be drawn later by job_report."""
    from .pipeline import (ExtractSettings, MotionSettings, PickSettings, extract_job, motion_job, pick_job)
    prj = _p()
    imp = prj.job(movies_job)
    mc = motion_job(prj, imp, MotionSettings(apix, dose_total, bin, gain, kv, cs_mm, amp_contrast, dtype=dtype,
                                             figures=figures))
    pj = pick_job(prj, mc, PickSettings(diameter_a, dark=dark, figures=figures))
    ej = extract_job(prj, pj, ExtractSettings(particle_a or 1.3 * diameter_a, box_a, particle_apix=particle_apix,
                                              dtype=dtype, dark=dark, export=tuple(export or ())))
    _background([mc, pj, ej])
    return _js(dict(jobs=[mc.name, pj.name, ej.name], box=ej.params["box"], particle_apix=ej.params["out_apix"],
                    note="running in the background: job_status"))


@server.tool()
def extract_particles(picks_job: str, particle_a: float, box_a: float = 0.0, particle_apix: float = 0.0,
                      dtype: str = "float16", dark: bool = True, select: str = "", export: list[str] | None = None,
                      label: str = "") -> str:
    """A new Extract job from a Pick job's kept picks (another box, binning or precision; micrographs are read
    from disk). select: a Select job (save_selection) - extract only its particles. label: the descriptive part of
    the folder name (default: box, pixel size, precision, selection)."""
    from .pipeline import ExtractSettings, extract_job
    prj = _p()
    ej = extract_job(prj, prj.job(picks_job), ExtractSettings(particle_a, box_a, particle_apix=particle_apix,
                                                             dtype=dtype, dark=dark, export=tuple(export or ()),
                                                             select=prj.job(select).id if select else 0),
                     label=label)
    _background([ej])
    return _js(dict(job=ej.name, box=ej.params["box"], particle_apix=ej.params["out_apix"],
                    box_a=ej.params["box_a_used"]))


def _background_job(job, fn):
    """Run fn() (one particle job: Class2D, InitialModel, Refine3D, Polish) on a background thread, one GPU job at a
    time; progress and errors through job_status."""
    prj = _p()
    st = _bg[job.name] = dict(done=0, of=None, error=None, jobs=[job.id], started=__import__("time").time())

    def work():
        try:
            with _lock:
                st["result"] = fn()
        except Exception as e:
            st["error"] = repr(e)
            prj.set_status(job, "failed", dict(error=repr(e)))
    st["thread"] = threading.Thread(target=work, daemon=True)
    st["thread"].start()
    return job.name


@server.tool()
def job_types() -> str:
    """Every particle job type this version can run: what it does, its inputs (and the job types each accepts) and
    its settings (kind, default, help). new_job makes any of them; the specific tools (classify_job, refine3d, ...)
    are shortcuts for the common settings."""
    from .jobs import registry
    from .jobspec import describe
    return _js([describe(sp) for sp in registry().values()])


@server.tool()
def new_job(job_type: str, inputs: dict, settings: dict | None = None, label: str = "") -> str:
    """Make and run (in the background; job_status / job_report) a job of any type listed by job_types, e.g.
    new_job("Sharpen", {"refine": "14"}, {"mask": "auto", "local_res": true}). Settings not given take their
    defaults; unknown inputs or settings are an error. Symmetry is the user's choice, never guessed."""
    from .jobs import new_job as _new, registry
    prj = _p()
    j = _new(prj, job_type, inputs, settings or {}, label)
    spec = registry()[job_type]
    _background_job(j, lambda: spec.run(prj, j, log=lambda *a: None))
    return _js(dict(job=j.name, params=j.params))


@server.tool()
def classify_job(particles_job: str, n_classes: int = 50, max_res_a: float = 0.0, diameter_a: float = 0.0,
                 seeds: int = 1, label: str = "") -> str:
    """A Class2D job on the particles of an Extract, Polish or Select job, in the background (job_status; then
    job_report shows the gallery). max_res_a 0: Nyquist of the particles but not finer than 6 A; diameter_a 0: mask
    1.25 x the extraction's particle size."""
    from . import jobs
    prj = _p()
    c = jobs.class2d_job(prj, particles_job, jobs.Class2DSettings(n_classes, max_res_a, diameter_a, seeds), label)
    _background_job(c, lambda: jobs.run_class2d(prj, c, log=lambda *a: None))
    return _js(dict(job=c.name))


@server.tool()
def select_classes(class2d_job: str, classes: list[int], label: str = "", note: str = "") -> str:
    """A Select job with the particles of these classes (1-based, as in the gallery) of a Class2D or HeteroRefine job; the other
    particles are kept as its rejected set, with kept / rejected class stacks and a selection figure. The user
    decides which classes are good: a selection proposed by an agent must say so in its label or note."""
    from . import jobs
    prj = _p()
    sel = jobs.select_classes(prj, class2d_job, classes, label, note)
    return _js(dict(job=sel.name, **prj.summary_of(sel)))


@server.tool()
def initial_model(class2d_job: str, sym: str = "C1", classes: list[int] | None = None, n_models: int = 1,
                  label: str = "") -> str:
    """Ab initio 3D (InitialModel job) from a Class2D job's classes (1-based; default all) and their particles, in
    the background. sym: the user's choice (C1, Cn, Dn, T, O, I) - never guessed."""
    from . import jobs
    prj = _p()
    m = jobs.initial_model_job(prj, class2d_job, jobs.InitialModelSettings(sym, tuple(classes or ()), n_models), label)
    _background_job(m, lambda: jobs.run_initial_model(prj, m, log=lambda *a: None))
    return _js(dict(job=m.name))


@server.tool()
def refine3d(particles_job: str, poses_job: str, reference_job: str = "", sym: str = "C1", n_iter: int = 10,
             global_iters: int = 0, classes: list[int] | None = None, label: str = "") -> str:
    """A Refine3D job in the background: particles of an Extract / Polish / Select job (e.g. re-centred at full box),
    starting orientations from an InitialModel, Refine3D or HeteroRefine job (matched by uid), starting map from
    reference_job (default the poses job; for a HeteroRefine, the map of the particles' class). classes: with a
    HeteroRefine poses job, only the particles of these 1-based classes (the user's choice). global_iters: first
    iterations that also search all orientations. Writes FSC (png + json/star/xml), angular distribution, progress
    per iteration and per-particle diagnostics."""
    from . import jobs
    prj = _p()
    r = jobs.refine3d_job(prj, particles_job, poses_job, reference_job or poses_job,
                          jobs.Refine3DSettings(sym=sym, n_iter=n_iter, global_iters=global_iters,
                                                classes=tuple(classes or ())), label)
    _background_job(r, lambda: jobs.run_refine3d(prj, r, log=lambda *a: None))
    return _js(dict(job=r.name))


@server.tool()
def hetero_refine(particles_job: str, poses_job: str, references: list[str], sym: str = "C1", n_iter: int = 12,
                  res_start_a: float = 20.0, global_search: bool = True, label: str = "") -> str:
    """A HeteroRefine job (heterogeneous refinement / 3D classification with alignment) in the background: the
    particles are split among the reference maps, each class refined with its own poses, half maps and FSC.
    references: one per class, job numbers ("12") or "job:class" ("25:2" = model 2 of an InitialModel, "30:3" =
    class 3 of a HeteroRefine); the same map can be given several times (a junk decoy is any poor map).
    global_search: every class first searches all orientations (references in different frames). The user chooses
    the classes to keep: select_classes on the job, or refine3d with classes=[...]."""
    from . import jobs
    prj = _p()
    h = jobs.hetero_refine_job(prj, particles_job, poses_job,
                               jobs.HeteroRefineSettings(references=tuple(str(r) for r in references), sym=sym,
                                                         n_iter=n_iter, res_start_a=res_start_a,
                                                         global_search=global_search), label)
    _background_job(h, lambda: jobs.run_hetero_refine(prj, h, log=lambda *a: None))
    return _js(dict(job=h.name))


@server.tool()
def polish(refine_job: str, label: str = "") -> str:
    """A Polish job in the background: per-particle motion tracks and per-frame weights from the movies against the
    Refine3D job's half maps; the result is a particle job (refine3d on it with that job's poses)."""
    from .polish import PolishSettings, polish_job, run_polish
    prj = _p()
    j = polish_job(prj, refine_job, PolishSettings(), label)
    _background_job(j, lambda: run_polish(prj, j, log=lambda *a: None))
    return _js(dict(job=j.name))


@server.tool()
def sharpen(refine_job: str, b_factor: float = 0.0, resolution_a: float = 0.0, mask: str = "auto", cls: int = 0,
            local_res: bool = True, label: str = "") -> list:
    """A Sharpen job (RELION PostProcess) on a Refine3D job's half maps (or class cls of a HeteroRefine): corrected
    masked FSC (phase randomisation), automatic Guinier B-factor (b_factor 0; RELION sign, negative sharpens), FSC
    weighting and low-pass at the resolution (resolution_a 0: from the corrected FSC). mask: "auto" (made from the
    half maps: threshold above their noise, round soft edge), "refine" (the refinement's) or a .mrc path. local_res: local resolution map, a map filtered to it, and its figure.
    Returns the summary (resolutions, B-factor, fit quality, mask checks, local resolution, warnings) and figures."""
    from .sharpen import SharpenSettings, run_sharpen, sharpen_job
    prj = _p()
    j = sharpen_job(prj, refine_job, SharpenSettings(b_factor=b_factor, resolution_a=resolution_a, mask=mask, cls=cls,
                                                     local_res=local_res), label)
    with _lock:
        summ = run_sharpen(prj, j, log=lambda *a: None)
    out = [_js(dict(job=j.name, **{k: v for k, v in summ.items() if k != "fsc_corrected"}))]
    for fn in ("postprocess.png", "localres.png"):
        path = j.find(fn)
        if path:
            with open(path, "rb") as fh:
                out.append(Image(data=fh.read(), format="png"))
    return out


@server.tool()
def particle_size_survey(micrographs_job: str, n_micrographs: int = 3,
                         diameters_a: list[float] | None = None) -> list:
    """Particle size of an unknown sample from the first micrographs of a MotionCtf job: one figure with the
    micrographs (phase-flipped, low-passed, 500 A bar, circles of the candidate diameters) and, per diameter, the
    best blob picks and their average with its radial profile. Choose the diameter whose gallery shows whole,
    separated particles; then pick with it."""
    from .plots_preprocess import size_survey
    prj = _p()
    mc = prj.job(micrographs_job)
    ids = prj.micrographs(mc)["movie_id"].tolist()[:n_micrographs]
    path = os.path.join(mc.path, "size_survey.png")
    with _lock:
        r = size_survey(path, prj, mc, ids, diameters_a=tuple(diameters_a or (80, 120, 160, 220, 300)))
    with open(path, "rb") as fh:
        return [_js(dict(figure=prj.rel(path), per_diameter=r)), Image(data=fh.read(), format="png")]


@server.tool()
def rethreshold_picks(picks_job: str, threshold: float | None = None, max_std: float = 2.0, max_mean: float = 3.0,
                      label: str = "") -> str:
    """A new Pick job keeping other candidates of the same picking (score >= threshold, None = automatic per
    micrograph; max_std / max_mean reject carbon edges and thick ice). Instant: nothing is picked again."""
    from .pipeline import rethreshold
    j = rethreshold(_p(), _p().job(picks_job), threshold, max_std, max_mean, label)
    return _js(dict(job=j.name, **_p().summary_of(j)))


@server.tool()
def job_status(job: str = "") -> str:
    """Progress of background jobs (or one), with each job's status and key numbers."""
    prj = _p()
    out = {}
    for key, st in _bg.items():
        if job and prj.job(job).id not in st["jobs"]:
            continue
        out[key] = dict(running=st["thread"].is_alive(), movies_done=st["done"], of=st["of"], error=st["error"],
                        jobs={prj.job(d).name: dict(status=prj.job(d).status, **prj.summary_of(d)) for d in st["jobs"]},
                        seconds=round(__import__("time").time() - st["started"], 1))
    if job and not out:
        j = prj.job(job)
        out[j.name] = dict(status=j.status, **prj.summary_of(j))
    return _js(out or prj.summary())


@server.tool()
def job_report(job: str, movies: list[str] | None = None, max_figures: int = 3) -> list:
    """A job's parameters, inputs and key numbers; for MotionCtf jobs per-micrograph statistics with outliers
    (> 4 MAD). For MotionCtf / Pick jobs also control figures: of the named movie stems (drawn now if needed,
    ~3 s each in a running server), else of the outliers and the first micrograph, at most max_figures."""
    import glob
    from .pipeline import draw_figure
    prj = _p()
    j = prj.job(job)
    rep = dict(job=j.name, status=j.status, inputs=[prj.job(i).name for i in j.inputs], params=j.params,
               summary=prj.summary_of(j))
    mc = prj.ancestor(j, "MotionCtf")
    if j.type == "MotionCtf":
        rep["micrographs"] = prj.micrograph_stats(j)
    out = [None]
    if j.type not in ("MotionCtf", "Pick"):                # the job's own figures
        figs = [j.find(f) for f in ("gallery.png", "selection.png", "map_views.png", "fsc.png", "views.png",
                                    "postprocess.png", "polish.png", "reprojections.png") if j.find(f)][:max_figures]
        for f in figs:
            with open(f, "rb") as fh:
                out.append(Image(data=fh.read(), format="png"))
        rep["figures"] = [prj.rel(f) for f in figs]
    if j.type in ("MotionCtf", "Pick"):
        if movies:
            mi = prj.micrographs(mc)
            ids = mi["movie_id"][mi["stem"].isin(movies)].tolist()[:max_figures]
            figs = [draw_figure(prj, j, i) for i in ids]
        else:
            figs = sorted(f for f in glob.glob(os.path.join(j.path, "figures", "movies", "*.png"))
                          + glob.glob(os.path.join(j.path, "figures", "*_*.png"))      # older jobs: flat
                          if ".light." not in f and "overview" not in f)[:max_figures]
        for f in figs:
            with open(f, "rb") as fh:
                out.append(Image(data=fh.read(), format="png"))
        rep["figures"] = [prj.rel(f) for f in figs]
    out[0] = _js(rep)
    return out


@server.tool()
def export_job(job: str, fmt: str = "relion") -> str:
    """Write RELION (movies.star / micrographs_ctf.star / autopick.star + coordinates / particles.star) or cryoSPARC
    (particles.cs, Extract jobs) files into the job folder. Paths are relative to the project: run RELION there."""
    from .export import export
    return _js(dict(written=_p().rel(export(_p(), _p().job(job), fmt))))


@server.tool()
def save_selection(name: str, which: str = "good", min_p: float = 0.5, label: str = "") -> str:
    """Store a particle set of a 2D result (loaded from a project extract job) in the project as a Select job: a
    list of particle uids, valid for any extraction of the same picks (extract_particles(select=...), e.g. at full
    size after classifying a binned extraction). which: good | bad | ambiguous."""
    from .project import Project
    s = _s()
    if "celUid" not in s.src.parts:
        raise ValueError("the loaded particles are not a Celeris2D extract job")
    idx = s.sets(name, min_p)[which]
    job_dir = os.path.abspath(s.path).rstrip("/")
    prj = Project(os.path.dirname(os.path.dirname(job_dir)))
    sel = prj.select(prj.job(os.path.basename(job_dir)), s.src.parts["celUid"].to_numpy()[idx],
                     label=label or f"{which}_of_{name}", note=f"result {name}: {which} at P(good) >= {min_p}")
    return _js(dict(job=sel.name, particles=int(len(idx))))


@server.tool()
def project_disk() -> str:
    """Space per job, biggest first: total, purgeable (micrographs / particle stacks that rebuild_job can make again
    from the recorded parameters), and which jobs use it."""
    d = _p().disk()
    return _js(dict(total_gb=round(float(d["gb"].sum()), 2), purgeable_gb=round(float(d["purgeable_gb"].sum()), 2),
                    jobs=d.to_dict("records")))


@server.tool()
def purge_job(job: str) -> str:
    """Delete a job's micrographs / particle stacks, keeping its tables, statistics and figures (rebuild_job remakes
    them). For cleaning up extractions or micrographs no longer worked on."""
    return _js(dict(job=_p().job(job).name, freed_gb=round(_p().purge(job), 3)))


@server.tool()
def rebuild_job(job: str) -> str:
    """Make a purged job's micrographs / particle stacks again from its recorded parameters (in the background)."""
    from .pipeline import check_rebuild
    prj = _p()
    j = check_rebuild(prj, job)
    _background([j])
    return _js(dict(job=j.name, note="running in the background: job_status"))


@server.tool()
def rename_job(job: str, label: str) -> str:
    """Change the descriptive part of a job's folder name (jobs/NNN_<type>_<label>); references are by number."""
    return _js(dict(job=_p().rename(job, label).name))


@server.tool()
def delete_job(job: str, force: bool = False) -> str:
    """Move a job and its data to Trash/ (refused while other jobs use it, unless force)."""
    return _js(dict(trash=_p().rel(_p().delete_job(job, force))))


@server.tool()
def find_particle(key_or_uid: str) -> str:
    """Everything about one particle: by key '<movie stem>:<x A>:<y A>' or by its uid (a number)."""
    k = int(key_or_uid) if key_or_uid.isdigit() else key_or_uid
    return _js(_p().find(k))


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
