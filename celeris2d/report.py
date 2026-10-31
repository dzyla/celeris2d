"""Publication report of a project: the jobs marked final (Project.mark_final) and every job they were made from, as
a flowchart (flowchart.svg / .png: one box per job with its key numbers and the settings that differ from the
defaults, arrows from inputs, final jobs highlighted), a methods text (methods.md: one paragraph per step in the
order they ran, settings and results, then a table of the final maps), and every setting of every job in the chain
(settings.csv). Written to <project>/report/.

    from celeris2d.report import project_report
    r = project_report(Project(root))          # {"flowchart": ..., "methods": ..., "settings": ..., "jobs": [...]}
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import os

import numpy as np

from .project import Job, Project, _version as _raw_version


def _version() -> str:
    """'0.4.1 (git abc1234)', or 'git abc1234' when the package version is unknown."""
    v = _raw_version().split()
    if len(v) == 2 and v[0] in ("?", ""):
        return f"(git {v[1]})"
    return f"{v[0]} (git {v[1]})" if len(v) == 2 else " ".join(v)

# job parameters that are bookkeeping, not settings
_INTERNAL = {"run_pid", "run_host", "run_started", "settings_version", "gpus", "seed", "figures"}
# parameters that name another job (shown as the arrow when it is an input)
_REFS = {"micrographs", "picks", "particles", "poses", "reference", "class2d", "refine", "map", "modes3d",
         "recenter", "mask_job"}
# settings worth showing even at their default (what a reader needs to repeat the step)
_ALWAYS = dict(MotionCtf=("apix", "bin", "dose_per_frame", "kv", "cs_mm", "patch_ctf"), Pick=("diameter_a",),
               Extract=("box_a", "particle_a", "recenter"), Class2D=("n_classes", "diameter_a"),
               InitialModel=("sym", "n_models"), Refine3D=("sym", "nu", "mask"), HeteroRefine=("sym", "references"),
               Class3D=("n_classes", "sym", "res_a"), Sharpen=("b_factor", "mask"), Select=("classes",),
               LocalRefine=("sym",), Modes3D=("n_modes", "filter_res_a"), MaskCreate=())
_SENTENCE = dict(
    Import="Movies were imported",
    MotionCtf="Beam-induced motion was corrected (whole-frame and patch motion, dose weighting) and the CTF fitted per "
              "micrograph",
    Curate="Micrographs were curated by CTF fit, defocus, astigmatism and motion",
    Pick="Particles were picked",
    Extract="Particles were extracted",
    Class2D="Particles were classified in 2D (maximum likelihood)",
    Select="2D classes were selected",
    InitialModel="An initial 3D model was computed ab initio from the selected 2D classes and their particles",
    Refine3D="The particles were refined in 3D with gold-standard half sets",
    HeteroRefine="Particles were sorted by heterogeneous refinement",
    Class3D="Particles were classified in 3D without alignment",
    Polish="Particles were polished (per-particle motion, per-frame dose weights)",
    CtfRefine="Per-particle defocus was refined against each particle's own half map",
    Sharpen="The map was sharpened and its resolution measured (corrected masked FSC 0.143)",
    MaskCreate="A solvent mask was made",
    MapScore="Particles were scored against the map and phase-randomised controls",
    Subtract="Signal outside the region of interest was subtracted",
    LocalRefine="The region of interest was refined locally",
    SymExpand="Poses were symmetry-expanded",
    Modes3D="Modes of variability were computed at fixed poses",
    ModeMaps="Maps along the modes were reconstructed")


def _settings_class(job_type: str):
    from .jobs import registry
    reg = registry()
    if job_type in reg:
        return reg[job_type].settings
    from . import pipeline as pl
    return dict(MotionCtf=pl.MotionSettings, Pick=pl.PickSettings, Extract=pl.ExtractSettings).get(job_type)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v) if len(v) <= 12 else f"{len(v)} values"
    return str(v)


def job_settings(job: Job) -> tuple[dict, dict]:
    """(settings that differ from the defaults or are always shown, all settings) of a job; inputs (job numbers) and
    bookkeeping left out."""
    cls = _settings_class(job.type)
    ins = set(job.inputs)
    params = {}
    for k, v in job.params.items():
        if k in _INTERNAL:
            continue
        if k in _REFS and isinstance(v, int) and not isinstance(v, bool):
            if v in ins or not v:
                continue
            v = f"job {v:03d}"
        params[k] = v
    if cls is None:
        return params, params
    defaults = {f.name: (f.default if f.default is not dataclasses.MISSING else
                         f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
                for f in dataclasses.fields(cls)}
    always = _ALWAYS.get(job.type, ())
    shown = {}
    for k, v in params.items():
        if k in _INTERNAL:
            continue
        d = defaults.get(k, None)
        same = (list(v) == list(d)) if isinstance(v, (list, tuple)) and isinstance(d, (list, tuple)) else v == d
        if (k in always and v not in (0, "", None, ())) or (k in defaults and not same):   # derived values: csv only
            shown[k] = v
    return shown, params


def chain(project: Project, finals: list[int]) -> list[Job]:
    """The final jobs and every job they were made from, in the order they were made."""
    ids = set()
    for f in finals:
        for name in project.lineage(f):
            ids.add(project.job(name).id)
    return [project.job(i) for i in sorted(ids)]


def _layout(jobs: list[Job]) -> dict[int, tuple[int, int]]:
    """(row, column) per job: row = longest path from a job without inputs in the chain."""
    ids = {j.id for j in jobs}
    depth = {}
    for j in jobs:                                  # made in order: inputs come first
        depth[j.id] = 1 + max((depth[i] for i in j.inputs if i in ids), default=-1)
    rows: dict[int, list[int]] = {}
    for j in jobs:
        rows.setdefault(depth[j.id], []).append(j.id)
    return {jid: (d, c) for d, members in rows.items() for c, jid in enumerate(members)}


def flowchart(project: Project, jobs: list[Job], finals: list[int], path: str, title: str = ""):
    """The chain as boxes (job, key numbers, settings) with arrows from inputs; light background (journals)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from .jobspec import key_of
    from .jobs import registry
    reg = registry()
    pos = _layout(jobs)
    n_rows = max(r for r, _ in pos.values()) + 1
    n_cols = max(c for _, c in pos.values()) + 1
    W, H = 3.3, 1.3                                  # box size (inches) and spacing
    gx, gy = 0.9, 0.5                                # room on the left for arrows that skip rows
    fig_w, fig_h = n_cols * (W + gx) + gx, n_rows * (H + gy) + gy + 0.6
    plt.rcParams["svg.fonttype"] = "none"           # text stays text in the SVG (editable in Illustrator / Corel)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    ink, muted, accent, edge, fill, ffill = "#1d1d1f", "#5a5a60", "#c2410c", "#9a9aa2", "#f6f6f8", "#fff4ec"
    box = {}
    for j in jobs:
        r, c = pos[j.id]
        members = sum(1 for (rr, _) in pos.values() if rr == r)
        x0 = gx + c * (W + gx) + (n_cols - members) * (W + gx) / 2
        y0 = 0.6 + gy + r * (H + gy)
        box[j.id] = (x0, y0)
        fin = j.id in finals
        ax.add_patch(FancyBboxPatch((x0, y0), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=ffill if fin else fill, ec=accent if fin else edge, lw=2.2 if fin else 1.0))
        ax.text(x0 + 0.12, y0 + 0.2, f"{j.type}" + ("  · FINAL" if fin else ""), fontsize=9.5, fontweight="bold",
                color=accent if fin else ink, va="center")
        ax.text(x0 + W - 0.1, y0 + 0.2, f"{j.id:03d}", fontsize=8, color=muted, va="center", ha="right")
        key = key_of(j.type, project.summary_of(j), reg)
        ax.text(x0 + 0.12, y0 + 0.45, (key or j.label or "")[:52], fontsize=8, color=ink, va="center")
        shown, _ = job_settings(j)
        lines = [f"{k} = {_fmt(v)}" for k, v in shown.items()][:4]
        if len(shown) > 4:
            lines.append(f"+ {len(shown) - 4} more (settings.csv)")
        ax.text(x0 + 0.12, y0 + 0.66, "\n".join(s[:54] for s in lines), fontsize=6.6, color=muted, va="top",
                family="monospace", linespacing=1.25)
    skip = 0
    for j in jobs:
        for i in j.inputs:
            if i in box:
                xa, ya = box[i]
                xb, yb = box[j.id]
                if pos[j.id][0] - pos[i][0] <= 1:                # next row: straight down (or across)
                    ax.add_patch(FancyArrowPatch((xa + W / 2, ya + H), (xb + W / 2, yb), arrowstyle="-|>",
                                                 mutation_scale=10, color=edge, lw=1.0,
                                                 connectionstyle="arc3,rad=0.0" if abs(xa - xb) < 0.01 else
                                                 "arc3,rad=0.08"))
                else:                                            # skips rows: out to the left, down, back in
                    skip += 1
                    arm = 22 + 9 * (skip % 4)
                    ax.add_patch(FancyArrowPatch((xa, ya + 0.7 * H), (xb, yb + 0.3 * H), arrowstyle="-|>",
                                                 mutation_scale=10, color=edge, lw=1.0,
                                                 connectionstyle=f"arc,angleA=180,angleB=180,armA={arm},armB={arm},rad=6"))
    ax.text(gx, 0.42, title or f"Processing in Celeris {_version()}", fontsize=12, fontweight="bold", color=ink)
    fig.savefig(path + ".svg", facecolor="white")
    fig.savefig(path + ".png", dpi=150, facecolor="white")
    plt.close(fig)


def _paragraph(project: Project, j: Job, n: int) -> str:
    s = project.summary_of(j)
    shown, _ = job_settings(j)
    from_ = ", ".join(f"{project.job(i).id:03d}" for i in j.inputs)
    text = f"**{n}. {j.type} (job {j.id:03d}{', from ' + from_ if from_ else ''}).** {_SENTENCE.get(j.type, j.type)}"
    res = []
    for k, label, unit in (("movies", "movies", ""), ("micrographs", "micrographs", ""),
                           ("picks", "picks", ""), ("particles", "particles", ""), ("particles_used", "particles in the reconstruction", ""),
                           ("resolution_solvent_mask_a", "resolution (corrected masked FSC 0.143)", " Å"),
                           ("resolution_unmasked_a", "unmasked", " Å"), ("b_factor", "B-factor", " Å²"),
                           ("seconds", "run time", " s")):
        v = s.get(k)
        if v is not None and not isinstance(v, (list, dict)):
            res.append(f"{label} {_fmt(v)}{unit}")
    if j.type == "Sharpen" and s.get("resolution_a"):
        res.insert(0, f"resolution {s['resolution_a']} Å (corrected masked FSC 0.143)")
    sh = s.get("sharpened")
    if isinstance(sh, dict) and sh.get("resolution_a"):
        res.append(f"sharpened in the job: {sh['resolution_a']} Å, B {sh.get('b_factor')} Å²")
    text += "; settings: " + "; ".join(f"{k} = {_fmt(v)}" for k, v in shown.items()) if shown else ""
    text += ("." if not res else ". Result: " + ", ".join(res) + ".")
    return text


def project_report(project: Project, finals: list[int] | None = None, out_dir: str | None = None) -> dict:
    """flowchart.svg / .png, methods.md and settings.csv of the final jobs' chain (all final jobs, or finals)."""
    marks = project.final_marks()
    finals = [project.job(f).id for f in (finals or sorted(marks))]
    if not finals:
        raise ValueError("no job is marked final: mark the jobs whose maps / particles the publication reports")
    out_dir = out_dir or os.path.join(project.root, "report")
    os.makedirs(out_dir, exist_ok=True)
    jobs = chain(project, finals)
    flowchart(project, jobs, finals, os.path.join(out_dir, "flowchart"))
    with open(os.path.join(out_dir, "settings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "type", "setting", "value"])
        for j in jobs:
            for k, v in job_settings(j)[1].items():
                w.writerow([j.name, j.type, k, json.dumps(v) if isinstance(v, (list, dict)) else v])
    lines = [f"# Processing report: {os.path.basename(project.root)}", "",
             f"Celeris {_version()}, {datetime.date.today().isoformat()}. {len(jobs)} jobs lead to the "
             f"{len(finals)} final job(s); the flowchart (flowchart.svg) shows them with the settings that differ from "
             f"the defaults, settings.csv lists every setting.", "", "## Methods", ""]
    lines += [_paragraph(project, j, n + 1) + "\n" for n, j in enumerate(jobs)]
    lines += ["## Final results", "", "| job | type | particles | resolution (Å) | unmasked (Å) | B (Å²) | map | note |",
              "|---|---|---|---|---|---|---|---|"]
    for f in finals:
        j = project.job(f)
        s = project.summary_of(j)
        sh = s.get("sharpened") if isinstance(s.get("sharpened"), dict) else {}
        res = s.get("resolution_solvent_mask_a") or s.get("resolution_a")
        lines.append(f"| {j.name} | {j.type} | {_fmt(s.get('particles_used') or s.get('particles') or '')} | "
                     f"{_fmt(res) if res else ''} | {_fmt(s.get('resolution_unmasked_a') or '')} | "
                     f"{_fmt(s.get('b_factor') or sh.get('b_factor') or '')} | {s.get('map') or sh.get('map') or ''} | "
                     f"{marks.get(j.id, {}).get('note', '')} |")
    with open(os.path.join(out_dir, "methods.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    rel = lambda p: project.rel(os.path.join(out_dir, p))
    return dict(flowchart=rel("flowchart.png"), flowchart_svg=rel("flowchart.svg"), methods=rel("methods.md"),
                settings=rel("settings.csv"), jobs=[j.name for j in jobs], finals=[project.job(f).name for f in finals])
