"""MCP server for celeris2d: fast GPU 2D classification with a person or an agent in the loop.

The server keeps one dataset loaded (spectra on the GPU), so every step after `load_particles` takes
seconds to a few minutes. Class decisions are never automatic: look at `show_classes`, then call
`select_good` (`suggest_good` only proposes).

Typical flow:
  load_particles -> classify (or isac) -> show_classes -> select_good -> [assign | deep2d] -> export

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
        "Class numbers are 1-based and refer to the result they were shown for."),
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
def load_particles(path: str, diameter_a: float, max_res_a: float = 6.0, gpus: str = "auto", subset: int = 0,
                   workdir: str = "", speed: str = "balanced") -> str:
    """Open particles (RELION .star, cryoSPARC .cs/.csg, or .mrc/.mrcs stack) and start streaming them to the GPU.
    diameter_a: particle mask diameter in A. max_res_a: finest resolution used. gpus: 'auto' or nvidia-smi
    indices like '0,2,3'. subset: random subset size for quick tests (0 = all). speed: fast|balanced|accurate."""
    global _session
    preset = dict(fast=dict(psi_factor=1.5, trans_div=2.5), balanced=dict(psi_factor=1.0, trans_div=3.0),
                  accurate=dict(psi_factor=1.0, trans_div=4.0))[speed]
    with _lock:
        wd = workdir or os.path.join(os.path.dirname(os.path.abspath(path)), "celeris2d_session")
        _session = Session(wd, log=lambda *a: None)
        info = _session.load(path, diameter=diameter_a, max_res=max_res_a, gpus=gpus, subset=subset, **preset)
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


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
