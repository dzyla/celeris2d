"""Celeris2D web dashboard: load a dataset, change every setting, run, select classes, rerun, export.

    celeris2d-web --port 8765            # then open http://<host>:8765  (or tunnel: ssh -L 8765:localhost:8765 host)

One dataset is held in memory (spectra on the GPU), so every rerun on a subset starts immediately. GPU jobs
run one at a time on a worker thread; the page polls /api/status for progress and the live log.
"""
from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import io
import json
import os
import re
import threading
import time
import traceback

import numpy as np

from .engine import Params, release_cache

# settings shown in the UI (Params fields); the dataset-level ones are used at load time
_LOAD_KEYS = ("diameter_a", "max_res_a", "init_res_a", "psi_step_min", "psi_factor", "trans_div", "max_shift_a",
              "notch_thresh", "noise_model", "polarity", "seed")
_RUN_KEYS = ("n_iter", "n_full_iter", "batch_per_class", "march_margin", "recon_margin", "frc_reg", "tau2_fudge",
             "sigma_init_factor", "solvent_flatten", "gold_standard", "solvent_thresh", "solvent_dilate_a",
             "center_classes", "recenter_thresh", "hard_pose", "class_prior", "polish", "reseed_frac", "max_offset_a",
             "local_refine")
_FIELDS = {f.name for f in dataclasses.fields(Params)}
POLARITY_UI = {0: "auto", -1: "invert", 1: "keep"}     # Params.polarity <-> the page's choice
LOAD_KEYS = tuple(k for k in _LOAD_KEYS if k in _FIELDS)   # tolerate Params fields being renamed / removed
RUN_KEYS = tuple(k for k in _RUN_KEYS if k in _FIELDS)

# Plain-language description of every setting the dashboard shows. Wording follows the Params comments
# (engine.py) and the CLI --help strings (cli.py); keep them in sync when a default or behaviour changes.
# group: where the page shows it; stage: when it takes effect (load = at Load, run = at every Run).
GROUPS = [
    dict(id="load_basic", title="Dataset", stage="load", basic=True,
         help="Used when you press Load. To change them for a loaded dataset, press Load again."),
    dict(id="load_more", title="Loading", stage="load", basic=False,
         help="Which GPUs, a random subset for quick tests, and the sampling preset. Applied at Load."),
    dict(id="load_adv", title="Advanced dataset settings", stage="load", basic=False,
         help="Sampling and preprocessing fixed at Load time. The defaults suit most data sets."),
    dict(id="run_basic", title="Classification", stage="run", basic=True,
         help="Used every time you press Run (no reloading needed)."),
    dict(id="run_more", title="Run options", stage="run", basic=False,
         help="Repeated independent starts (for per-particle P(good))."),
    dict(id="run_schedule", title="Iterations", stage="run", basic=False,
         help="How long the run is and how many particles each iteration sees. -1 means automatic."),
    dict(id="run_align", title="Alignment", stage="run", basic=False,
         help="How particles are matched to classes. 'auto' settings depend on the number of particles in the run."),
    dict(id="run_classes", title="Class averages", stage="run", basic=False,
         help="How class averages are filtered, masked and centred after each iteration."),
    dict(id="opt", title="Settings optimizer", stage="opt", basic=True,
         help="Optional: tries a small grid of settings on a random subset and keeps the most reproducible one."),
]

SETTINGS = {
    # ---- dataset (load) ---------------------------------------------------------------------------------------
    "diameter_a": dict(group="load_basic", label="Particle mask diameter", unit="Å", type="number", important=True,
                       help="Diameter of the circular mask applied to particles and class averages. Use a bit more than the "
                            "longest dimension of the particle. It also sets the in-plane angular step (finer for larger "
                            "particles) and the default offset cap (diameter / 4)."),
    "max_res_a": dict(group="load_basic", label="Maximum resolution", unit="Å", type="number", important=True,
                      help="Final resolution limit for alignment and class averages. Particles are Fourier-cropped to this "
                           "limit when loaded, so a coarser value (e.g. 8-12 Å) loads and runs faster. Values closer than "
                           "2.2 × the pixel size to Nyquist are raised to 2.2 × pixel size."),
    "gpus": dict(group="load_more", label="GPUs", unit="", type="text", default="auto", important=True,
                 help="'auto', or nvidia-smi GPU indices such as 0,2,3. Auto: the GPU with the most memory for small "
                      "jobs, all GPUs once the work (particles × working box²) is worth splitting (about 100k particles "
                      "at a 64 px working box)."),
    "subset": dict(group="load_more", label="Random subset", unit="particles", type="int", default=0, important=True,
                   help="Load only this many randomly chosen particles (0 = all). Useful for quick tests of settings; "
                        "the choice is reproducible through the random seed."),
    "speed": dict(group="load_more", label="Speed preset", unit="", type="choice", default="balanced", important=True,
                  choices=["fast", "balanced", "accurate"],
                  help="Angular and translational sampling. fast: in-plane step × 1.5 and translation step = "
                       "resolution / 2.5; balanced: step × 1, translation = resolution / 3; accurate: step × 1, "
                       "translation = resolution / 4 (finest, slowest)."),
    "project_dir": dict(group="load_adv", label="Project directory", unit="", type="text", default="",
                        help="Leave empty normally: Celeris2D walks up from the particle file's folder until the image "
                             "paths in the metadata resolve (cryoSPARC: the folder holding the Jxx jobs; RELION: the "
                             "project root). Set it only if loading reports that it cannot locate the images."),
    "init_res_a": dict(group="load_adv", label="Starting resolution", unit="Å", type="number",
                       help="Resolution limit of the first iteration. The limit then moves toward the maximum resolution "
                            "over the first 60 % of the iterations, but never far beyond the resolution the classes "
                            "currently support (see alignment margin)."),
    "psi_step_min": dict(group="load_adv", label="Finest in-plane angle step", unit="°", type="number",
                         help="Lower bound for the in-plane rotation step. The step used is (speed factor) × "
                              "degrees(resolution / diameter), kept between this value and 15°."),
    "max_shift_a": dict(group="load_adv", label="Translation search range", unit="Å", type="number",
                        help="Local translation search radius around each particle's previous offset, per iteration "
                             "(at most 9 × 9 positions). Larger offsets still build up over iterations."),
    "notch_thresh": dict(group="load_adv", label="Artefact notch threshold", unit="× shell mean", type="number",
                         help="Zero Fourier pixels whose mean power over all particles is more than this × the mean of "
                              "their resolution shell (fixed-frame detector or extraction artefacts, e.g. stripes). "
                              "0 = off."),
    "noise_model": dict(group="load_adv", label="Noise model", unit="", type="choice", choices=["coloured", "radial"],
                        help="coloured: one noise value per Fourier pixel in the particle frame, which down-weights "
                             "fixed-frame artefacts; radial: one value per resolution shell."),
    "polarity": dict(group="load_adv", label="Contrast", unit="", type="choice", choices=["auto", "invert", "keep"],
                     value="auto",
                     help="Protein must be white for classification. auto: decided from the unaligned average of the "
                          "particles (centre vs ring); invert: flip the contrast of the images as stored; keep: use them "
                          "as stored. If the classes or the 3D map come out with dark protein, set this the other way. "
                          "The log line [polarity] says what auto chose."),
    "seed": dict(group="load_adv", label="Random seed", unit="", type="int",
                 help="Seeds the random subset and the class initialisation. Change it to get an independent start "
                      "with otherwise identical settings."),
    # ---- classification (run) ---------------------------------------------------------------------------------
    "K": dict(group="run_basic", label="Number of classes (K)", unit="", type="int", default=50, important=True,
              help="More classes separate more views and more kinds of junk, but each class gets fewer particles and a "
                   "noisier average. Run time grows roughly with K."),
    "seeds": dict(group="run_more", label="Independent starts (seeds)", unit="", type="int", default=1, important=True,
                  help="Above 1, the same classification is repeated from different random starts and the classes are "
                       "matched by shared particles. After you select good classes every particle gets P(good) = the "
                       "fraction of starts that put it in a good class. Costs about seeds × one run."),
    "n_iter": dict(group="run_schedule", label="Iterations", unit="", type="int", value=-1, auto="-1",
                   help="Total number of iterations. -1 = auto: 25, or 40 when the run has more than 100,000 particles "
                        "(online-EM iterations on large sets are cheap)."),
    "n_full_iter": dict(group="run_schedule", label="Final full passes", unit="iterations", type="int", auto="-1",
                        help="How many of the last iterations use every particle (earlier ones use online-EM batches). "
                             "-1 = auto: 6 up to 50k particles, 3 up to 500k, 2 above. Full passes are what costs time "
                             "on large sets."),
    "batch_per_class": dict(group="run_schedule", label="Online-EM batch per class", unit="particles / class",
                            type="int",
                            help="Early iterations each use a fresh random batch of this many particles × K "
                                 "(cryoSPARC-like online EM). 0 = every iteration is a full pass over all particles."),
    "reseed_frac": dict(group="run_schedule", label="Reseed small classes below", unit="× average class size",
                        type="number",
                        help="During the first 90 % of iterations (never in full passes), a class holding fewer than this fraction of the "
                             "average class size (N / K) is replaced by a perturbed copy of the largest class. 0 = off."),
    "hard_pose": dict(group="run_align", label="Hard pose assignment", unit="", type="tri",
                      auto="on for runs with ≥ 100,000 particles, off below",
                      help="On: only the best in-plane angle and shift per class is used, while the class assignment "
                           "stays soft (cryoSPARC 'force max over poses'). Off: marginalise over all poses (RELION-like). "
                           "Hard pose resists noise attractors on large low-SNR sets."),
    "class_prior": dict(group="run_align", label="Class-size prior", unit="", type="tri",
                        auto="off (uniform prior) for runs with ≥ 100,000 particles, on below",
                        help="On: class populations act as a prior in the assignment (RELION-like; keeps per-class "
                             "confidence informative). Off: every class is equally likely a priori."),
    "march_margin": dict(group="run_align", label="Alignment resolution margin", unit="Fourier shells", type="number",
                         help="Alignment may use this many shells beyond the resolution the classes currently support "
                              "(SSNR = 1). 0 is safest; values > 0 let alignment chase noise on low-SNR data "
                              "(honeycomb attractors). 5-10 can help clean, high-SNR data."),
    "polish": dict(group="run_align", label="Polish final pass", unit="", type="bool",
                   help="The last iteration aligns out to the reconstruction limit with the classes locked. Sharper "
                        "classes on high-SNR data; scrambles poses on low-SNR or heavily binned data, hence off by default."),
    "local_refine": dict(group="run_align", label="Coarse-to-fine search", unit="", type="bool",
                         help="With hard pose (auto from 20k particles): search all classes at twice the angular step, then "
                              "refine each particle's two best classes at a quarter of that step and half-step shifts. "
                              "About 2x faster runs with finer angles; same or better separation in tests."),
    "max_offset_a": dict(group="run_align", label="Maximum particle offset", unit="Å", type="number", auto="-1",
                         help="Cap on the total shift a particle can accumulate from its picked centre. "
                              "-1 = diameter / 4."),
    "recon_margin": dict(group="run_classes", label="Reconstruction margin", unit="Fourier shells", type="number",
                         help="Class averages may be reconstructed this many shells beyond the alignment limit; the "
                              "half-set FRC Wiener filter decides how much of that is kept."),
    "frc_reg": dict(group="run_classes", label="Half-set FRC filter", unit="", type="bool",
                    help="Wiener-filter class averages using the FRC between odd and even particle halves (recommended). "
                         "Off: estimate the signal from each class's own power (older behaviour, prone to "
                         "over-fitting noise)."),
    "tau2_fudge": dict(group="run_classes", label="Regularisation T", unit="", type="number",
                       help="Multiplies the signal-power estimate (like RELION's T): higher = sharper but noisier "
                            "classes. Only used when the half-set FRC filter is off."),
    "gold_standard": dict(group="run_classes", label="Gold-standard half-sets", unit="", type="bool",
                          help="Experimental. Each half-set is aligned only to classes built from its own half, so "
                               "the half-set FRC cannot mistake a shared noise pattern for signal. Over-smooths "
                               "classes with few particles."),
    "sigma_init_factor": dict(group="run_classes", label="Initial noise inflation", unit="×", type="number",
                              help="Values > 1 inflate the noise estimate at the start and anneal it to 1 between "
                                   "iterations 2 and 17 (cryoSPARC-style). 1 = off; inflation hurt separation of "
                                   "real particles from junk in tests."),
    "center_classes": dict(group="run_classes", label="Re-centre classes", unit="", type="bool",
                           help="Shift every class average so that the centre of mass of its density sits in the box "
                                "centre, every iteration."),
    "recenter_thresh": dict(group="run_classes", label="Re-centring threshold", unit="× class maximum", type="number",
                            help="Only density above this fraction of the class maximum counts for the centre of mass."),
    "solvent_flatten": dict(group="run_classes", label="Solvent flattening", unit="", type="bool",
                            help="Mask each class by a soft envelope of its own density (flat solvent). Off by default, "
                                 "as in cryoSPARC: on small or weak particles the envelope takes in surrounding noise "
                                 "and grows a honeycomb. Can help clean, well-separated views."),
    "solvent_thresh": dict(group="run_classes", label="Solvent envelope threshold", unit="× class maximum",
                           type="number",
                           help="Envelope core = class density low-passed to 20 Å above this fraction of its maximum. "
                                "Only used with solvent flattening."),
    "solvent_dilate_a": dict(group="run_classes", label="Solvent envelope padding", unit="Å", type="number",
                             help="The envelope extends this far beyond its core. Only used with solvent flattening."),
    # ---- optimizer / export -------------------------------------------------------------------------------------
    "opt_subset": dict(group="opt", label="Optimizer subset", unit="particles", type="int", default=20000,
                       help="Random particles used for each trial. The optimizer tries the alignment margin (0 / 4 "
                            "shells) × solvent flattening (off / on), each from two independent starts, and applies "
                            "the setting whose starts agree most. It uses K from the Classification section."),
}


def _defaults():
    p = Params()
    d = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
    d["n_iter"] = -1  # auto (25, or 40 above 100k particles)
    d["polarity"] = POLARITY_UI[d["polarity"]]
    return d


def settings_meta():
    """{key: {value, label, unit, help, group, type, choices?, auto?, important?, stage}} for every setting shown."""
    d = _defaults()
    stage = {g["id"]: g["stage"] for g in GROUPS}
    out = {}
    for k, m in SETTINGS.items():
        if k in _FIELDS:
            if k not in LOAD_KEYS + RUN_KEYS:
                continue
            v = m.get("value", d[k])
        elif "default" in m:
            v = m["default"]
        else:
            continue
        e = {kk: vv for kk, vv in m.items() if kk not in ("default", "value")}
        if e["type"] == "tri":
            v = "auto" if v is None else ("on" if v else "off")
        e.update(value=v, stage=stage[m["group"]], key=k)
        out[k] = e
    for k in LOAD_KEYS + RUN_KEYS:   # a Params field wired in without a description still shows up
        if k not in out and k not in ("psi_factor", "trans_div"):
            out[k] = dict(key=k, value=d[k], label=k, unit="", type="bool" if isinstance(d[k], bool) else
                          "number" if isinstance(d[k], (int, float)) else "text", help="(no description yet)",
                          group="load_adv" if k in LOAD_KEYS else "run_classes",
                          stage="load" if k in LOAD_KEYS else "run")
    return out


# ------------------------------------------------------------------------------------- file discovery
LIST_CAP = 2000          # entries returned per directory listing
SCAN_CAP = 50_000        # entries looked at before giving up on sorting a huge directory
PARTICLE_EXT = (".star", ".cs", ".csg", ".mrcs", ".mrc")
_CS_JOB = re.compile(r"^J\d+$")
_RELION_JOB = re.compile(r"^job\d+$")
_RELION_NO_PARTICLES = {"MotionCorr", "CtfFind", "ManualPick", "AutoPick", "MaskCreate", "PostProcess", "LocalRes",
                        "ModelAngelo", "Movies", "Micrographs", "Import_movies", "DynaMight", "Tomo"}
_RELION_TYPES = {"Import", "Extract", "Select", "Class2D", "Class3D", "InitialModel", "Refine3D", "Polish", "CtfRefine",
                 "Subtract", "JoinStar", "MultiBody", "External", "MotionCorr", "CtfFind", "AutoPick", "ManualPick",
                 "MaskCreate", "PostProcess", "LocalRes"}
_cache = {}              # (kind, path, size, mtime) -> value


def _natkey(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _cached(kind, path, fn):
    try:
        s = os.stat(path)
    except OSError:
        return None
    key = (kind, path, s.st_size, s.st_mtime)
    if key not in _cache:
        if len(_cache) > 5000:
            _cache.clear()
        try:
            _cache[key] = fn(path, s.st_size)
        except Exception:
            _cache[key] = None
    return _cache[key]


def _csg_info(path, size=0):
    """cryoSPARC .csg: group type/title and particle count (num_items) without reading the .cs files."""
    import yaml
    if size > 1 << 20:
        return None
    with open(path) as fh:
        g = yaml.safe_load(fh) or {}
    grp, res = g.get("group") or {}, g.get("results") or {}
    n = None
    for key in ("blob", "ctf", "alignments2D", "location"):
        if isinstance(res.get(key), dict) and "num_items" in res[key]:
            n = int(res[key]["num_items"])
            break
    if n is None:
        ns = [int(v["num_items"]) for v in res.values() if isinstance(v, dict) and "num_items" in v]
        n = max(ns) if ns else None
    return dict(type=grp.get("type"), title=grp.get("title") or grp.get("name"), count=n,
                has_blob="blob" in res, fields=sorted(res))


def _cs_count(path, size=0):
    """Row count of a cryoSPARC .cs (numpy .npy layout) from its header only."""
    with open(path, "rb") as fh:
        ver = np.lib.format.read_magic(fh)
        hdr = np.lib.format.read_array_header_2_0 if ver[0] >= 2 else np.lib.format.read_array_header_1_0
        shape, _, dt = hdr(fh)
    return dict(count=int(shape[0]) if shape else 0, has_blob=bool(dt.names) and "blob/path" in dt.names)


def _mrc_info(path, size=0):
    """nx, ny, nz and pixel size from an MRC header."""
    with open(path, "rb") as fh:
        h = fh.read(1024)
    nx, ny, nz = np.frombuffer(h[:12], "<i4")
    mx = np.frombuffer(h[28:32], "<i4")[0]
    cella = np.frombuffer(h[40:44], "<f4")[0]
    apix = float(cella / mx) if mx > 0 else None
    return dict(count=int(nz), box=int(nx), ny=int(ny), apix=round(apix, 4) if apix else None)


def _star_count(path, size=0, exact_limit=1 << 30):
    """Particles in a RELION star file: rows of data_particles (RELION 3.1+) or of the first loop with
    _rlnImageName (3.0). Exact up to exact_limit bytes, otherwise estimated from the mean row length."""
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
        lines = head.split(b"\n")
        pos, block, labels, start, in_loop, blocks = 0, None, [], None, False, []
        for ln in lines[:-1]:
            s = ln.strip()
            if s.startswith(b"data_"):
                block, labels, in_loop = s[5:].decode(errors="replace"), [], False
                blocks.append(block)
            elif s == b"loop_":
                in_loop, labels = True, []
            elif s.startswith(b"_") and in_loop:
                labels.append(s.split()[0].decode(errors="replace"))
            elif s and in_loop and labels and not s.startswith(b"#"):
                if block == "particles" or ("_rlnImageName" in labels and "optics" not in blocks):
                    start = pos
                    break
                in_loop = False
            pos += len(ln) + 1
        if start is None:
            return dict(count=0 if blocks else None, blocks=blocks)
        fh.seek(start)
        if size - start > exact_limit:   # estimate from the first rows
            sample = fh.read(4 << 20)
            nl = max(sample.count(b"\n"), 1)
            return dict(count=int((size - start) / (len(sample) / nl)), approx=True, blocks=blocks)
        n, carry, last = 0, b"", b""
        while True:
            chunk = fh.read(16 << 20)
            if not chunk:
                break
            buf = carry + chunk
            j = 0 if buf.startswith(b"data_") else buf.find(b"\ndata_")
            stop = j >= 0
            if stop:   # another block follows: count up to it
                buf = buf[:j + 1] if j else b""
            k = buf.rfind(b"\n")
            complete, carry = buf[:k + 1], buf[k + 1:]
            n += complete.count(b"\n")
            if complete:
                last = complete[-65536:]
            if stop:
                carry = b""
                break
        for ln in reversed(last.split(b"\n")[:-1]):   # trailing blank / comment lines are not particles
            s = ln.strip()
            if s and not s.startswith(b"#"):
                break
            n -= 1
        if carry.strip():
            n += 1
        return dict(count=max(n, 0), blocks=blocks)


def _particle_kind(name):
    """Coarse guess from the file name only (cheap enough for big listings)."""
    low = name.lower()
    if low.endswith(".csg"):
        return "csg"
    if low.endswith(".cs"):
        return "cs-particles" if "particle" in low and "passthrough" not in low else "cs"
    if low.endswith(".star"):
        return ("star-particles" if low == "particles.star" or low.endswith("_data.star") or low.startswith("particles")
                or low in ("shiny.star", "join_particles.star") else "star")
    if low.endswith(".mrcs"):
        return "stack"
    if low.endswith(".mrc"):
        return "mrc"
    return None


def peek(path, exact=True):
    """What a candidate input file is: format, particle count, box/pixel size when cheap."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return dict(path=path, exists=False, error="file not found")
    if os.path.isdir(path):
        return dict(path=path, exists=True, is_dir=True, error="this is a folder - choose a particle file inside it")
    kind = _particle_kind(os.path.basename(path))
    out = dict(path=path, exists=True, kind=kind, size=os.path.getsize(path))
    try:
        if kind == "csg":
            info = _cached("csg", path, _csg_info) or {}
            out.update(format="cryoSPARC group (.csg)", count=info.get("count"), title=info.get("title"))
            if info.get("type") and info["type"] != "particle":
                out["warning"] = f"this .csg holds '{info['type']}' items, not particles"
            elif info and not info.get("has_blob"):
                out["warning"] = "no particle images in this group (picks only): use an extraction job's output"
        elif kind in ("cs", "cs-particles"):
            info = _cached("cs", path, _cs_count) or {}
            out.update(format="cryoSPARC dataset (.cs)", count=info.get("count"))
            if info and not info.get("has_blob"):
                out["warning"] = ("no particle images in this .cs (blob/path missing) - if the job has a .csg, "
                                  "choose that instead")
        elif kind in ("star", "star-particles"):
            info = _cached("star", path, lambda p, s: _star_count(p, s, (1 << 30) if exact else (64 << 20))) or {}
            out.update(format="RELION star", count=info.get("count"), approx=info.get("approx", False))
            if info.get("count") is None:
                out["warning"] = "no particle table found in this star file"
        elif kind in ("stack", "mrc"):
            info = _cached("mrc", path, _mrc_info) or {}
            out.update(format="MRC stack" if kind == "stack" else "MRC file", count=info.get("count"),
                       box=info.get("box"), apix=info.get("apix"))
            out["note"] = "plain stack: no CTF information"
        else:
            out["error"] = "not a particle file (.star, .cs, .csg, .mrcs or .mrc expected)"
    except PermissionError:
        out["error"] = "permission denied"
    except Exception as e:  # noqa: BLE001 - a peek must never fail the page
        out["error"] = f"could not read: {type(e).__name__}: {e}"
    proj = find_project(os.path.dirname(path))
    if proj:
        out["project"] = proj
    return out


def _is_cs_project(d):
    return os.path.isfile(os.path.join(d, "project.json")) or os.path.isfile(os.path.join(d, "job_manifest.json"))


def _is_relion_project(d):
    if _RELION_JOB.match(os.path.basename(d)):   # job dirs carry their own default_pipeline.star copy
        return False
    if os.path.isfile(os.path.join(d, "default_pipeline.star")):
        return True
    try:
        names = [e.name for e in os.scandir(d) if e.name in _RELION_TYPES and e.is_dir()]
    except OSError:
        return False
    return any(os.path.isdir(os.path.join(d, n, j)) for n in names for j in os.listdir(os.path.join(d, n))[:50]
               if _RELION_JOB.match(j))


def find_project(start, levels=4):
    """The cryoSPARC / RELION project that contains `start` (start itself or up to `levels` parents)."""
    d = os.path.abspath(start)
    for _ in range(levels + 1):
        try:
            if _is_cs_project(d):
                return dict(kind="cryosparc", root=d)
            if _is_relion_project(d):
                return dict(kind="relion", root=d)
        except OSError:
            pass
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def list_dir(path, show_hidden=False):
    """Directory listing for the file browser: dirs first, then particle-like files, then the rest; capped."""
    req = os.path.abspath(os.path.expanduser(path or "~"))
    note, d = None, req
    if os.path.isfile(d):
        d = os.path.dirname(d)
    while not os.path.isdir(d):   # missing: fall back to the nearest existing parent
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    if d != req and not os.path.isfile(req):
        note = f"{req} does not exist; showing {d}"
    crumbs, p = [], d
    while True:
        crumbs.append(dict(name=os.path.basename(p) or p, path=p))
        np_ = os.path.dirname(p)
        if np_ == p:
            break
        p = np_
    out = dict(path=d, parent=os.path.dirname(d) if os.path.dirname(d) != d else None, crumbs=crumbs[::-1],
               home=os.path.expanduser("~"), dirs=[], files=[], truncated=False, note=note,
               selected=req if os.path.isfile(req) else None)
    entries, scanned = [], 0
    try:
        with os.scandir(d) as it:
            for e in it:
                scanned += 1
                if scanned > SCAN_CAP:
                    out["truncated"] = True
                    break
                if not show_hidden and e.name.startswith("."):
                    continue
                try:
                    is_dir = e.is_dir()
                except OSError:
                    is_dir = False
                entries.append((e, is_dir))
    except PermissionError:
        out["error"] = f"permission denied: {d}"
        return out
    except OSError as e:
        out["error"] = f"cannot list {d}: {e.strerror or e}"
        return out
    rank = lambda t: (0 if t[1] else 1 if _particle_kind(t[0].name) else 2, _natkey(t[0].name))  # noqa: E731
    entries.sort(key=rank)
    out["total"] = len(entries)
    if len(entries) > LIST_CAP:
        out["truncated"] = True
        entries = entries[:LIST_CAP]
    n_csg = 0
    for e, is_dir in entries:
        try:
            s = e.stat()
            size, mtime = s.st_size, s.st_mtime
        except OSError:
            size, mtime = None, None
        if is_dir:
            hint = ("cryoSPARC job" if _CS_JOB.match(e.name) else "RELION job" if _RELION_JOB.match(e.name)
                    else "RELION job type" if e.name in _RELION_TYPES else None)
            out["dirs"].append(dict(name=e.name, path=e.path, mtime=mtime, hint=hint))
            continue
        kind = _particle_kind(e.name)
        f = dict(name=e.name, path=e.path, size=size, mtime=mtime, kind=kind, particles=False)
        if kind == "csg" and n_csg < 300:   # tiny YAML files: read them for type + count
            n_csg += 1
            info = _cached("csg", e.path, _csg_info)
            if info:
                f.update(count=info["count"], title=info["title"],
                         particles=info["type"] == "particle" and info["has_blob"])
        elif kind == "cs-particles" and n_csg < 300:   # row count from the .npy header only
            n_csg += 1
            info = _cached("cs", e.path, _cs_count)
            f["particles"] = bool(info and info.get("has_blob"))
            if info:
                f["count"] = info["count"]
        elif kind in ("cs-particles", "star-particles", "stack"):
            f["particles"] = True
        out["files"].append(f)
    out["project"] = find_project(d)
    return out


def _json_cached(path):
    return _cached("json", path, lambda p, s: json.load(open(p)))


def _cs_job(root, j):
    """Particle outputs of one cryoSPARC job dir (None if it has none)."""
    jd = os.path.join(root, j)
    try:
        files = os.listdir(jd)
    except OSError:
        return None
    outs = []
    for fn in files:
        if fn.endswith(".csg"):
            info = _cached("csg", os.path.join(jd, fn), _csg_info)
            if info and info.get("type") == "particle":
                outs.append(dict(name=fn, path=os.path.join(jd, fn), count=info["count"], title=info["title"],
                                 format="csg", images=info["has_blob"]))
    if not outs:   # older cryoSPARC: no .csg, take the particle .cs files
        for fn in files:
            if _particle_kind(fn) == "cs-particles":
                info = _cached("cs", os.path.join(jd, fn), _cs_count)
                if info and info.get("has_blob"):
                    outs.append(dict(name=fn, path=os.path.join(jd, fn), count=info["count"], format="cs", images=True))
    if not outs:
        return None
    outs.sort(key=lambda o: (not o["images"], "excluded" in o["name"], "rejected" in o["name"], _natkey(o["name"])))
    meta = _json_cached(os.path.join(jd, "job.json")) or {}
    return dict(job=j, type=meta.get("job_type") or meta.get("type"), status=meta.get("status"),
                title=meta.get("title") or "", outputs=outs)


def _pmap(fn, items):
    """Network file systems answer one metadata request at a time per thread: fan the job dirs out."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(fn, items))


def discover_cryosparc(root):
    try:
        names = [e.name for e in os.scandir(root) if _CS_JOB.match(e.name) and e.is_dir()]
    except OSError as e:
        return dict(kind="cryosparc", root=root, jobs=[], error=f"cannot list {root}: {e.strerror or e}")
    names.sort(key=_natkey, reverse=True)
    jobs = [j for j in _pmap(lambda j: _cs_job(root, j), names) if j]
    proj = _json_cached(os.path.join(root, "project.json")) or {}
    return dict(kind="cryosparc", root=root, title=proj.get("title"), jobs=jobs)


def _relion_outputs(jd):
    try:
        files = os.listdir(jd)
    except OSError:
        return []
    picks, iters = [], {}
    for fn in files:
        low = fn.lower()
        if not low.endswith(".star"):
            continue
        m = re.match(r"^(.*?)_it(\d+)_data\.star$", fn)
        if m:   # keep only the last iteration of each run
            if m.group(1) not in iters or int(m.group(2)) > iters[m.group(1)][0]:
                iters[m.group(1)] = (int(m.group(2)), fn)
        elif low == "particles.star" or low.endswith("_data.star") or low.startswith("particles") \
                or low in ("shiny.star", "join_particles.star"):
            picks.append(fn)
    picks += [fn for _, fn in iters.values()]
    return sorted(picks, key=_natkey)


def discover_relion(root, count_limit=64 << 20):
    jobs = []
    try:
        types = [e.name for e in os.scandir(root) if e.is_dir() and not e.name.startswith(".")
                 and e.name not in _RELION_NO_PARTICLES]
    except OSError as e:
        return dict(kind="relion", root=root, jobs=[], error=f"cannot list {root}: {e.strerror or e}")
    for t in types:
        td = os.path.join(root, t)
        try:
            ents = list(os.scandir(td))
        except OSError:
            continue
        alias = {}
        for e in ents:   # RELION aliases are symlinks next to the jobNNN dirs
            if e.is_symlink():
                alias[os.path.basename(os.path.normpath(os.readlink(e.path)))] = e.name
        def one(e, t=t, alias=alias):
            outs = []
            for fn in _relion_outputs(e.path):
                fp = os.path.join(e.path, fn)
                info = _cached("star-d", fp, lambda p, s: _star_count(p, s, count_limit)) or {}
                if info.get("count"):
                    outs.append(dict(name=fn, path=fp, count=info["count"], approx=info.get("approx", False),
                                     format="star", images=True))
            if not outs:
                return None
            status = next((s.split("_")[-1].lower() for s in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE",
                           "RELION_JOB_EXIT_ABORTED") if os.path.exists(os.path.join(e.path, s))), "running/unknown")
            return dict(job=f"{t}/{e.name}", type=t, status=dict(success="completed", failure="failed").get(status, status),
                        title=alias.get(e.name, ""), outputs=outs, _n=int(e.name[3:]))
        dirs = [e for e in ents if _RELION_JOB.match(e.name) and e.is_dir() and not e.is_symlink()]
        jobs += [j for j in _pmap(one, dirs) if j]
    jobs.sort(key=lambda j: -j.pop("_n"))
    return dict(kind="relion", root=root, jobs=jobs)


def discover(path):
    """Particle outputs of the cryoSPARC / RELION project that contains `path`."""
    path = os.path.abspath(os.path.expanduser(path))
    proj = find_project(path if os.path.isdir(path) else os.path.dirname(path))
    if not proj:
        return dict(kind=None, root=path, jobs=[],
                    error="not inside a cryoSPARC project (Jxx job folders) or a RELION project (Extract/, Class2D/, ...)")
    t0 = time.time()
    out = discover_cryosparc(proj["root"]) if proj["kind"] == "cryosparc" else discover_relion(proj["root"])
    out["seconds"] = round(time.time() - t0, 2)
    return out


class Recent:
    """Recently loaded inputs, kept in <workdir>/recent.json so they survive restarts."""

    def __init__(self, workdir, n=15):
        self.fn, self.n, self.lock = os.path.join(workdir, "recent.json"), n, threading.Lock()

    def read(self):
        try:
            with open(self.fn) as fh:
                d = json.load(fh)
            return dict(paths=list(d.get("paths", [])), last_dir=d.get("last_dir"))
        except (OSError, ValueError):
            return dict(paths=[], last_dir=None)

    def _write(self, d):
        tmp = self.fn + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, self.fn)

    def add(self, path, **info):
        with self.lock:
            d = self.read()
            path = os.path.abspath(path)
            d["paths"] = [dict(path=path, t=time.time(), **{k: v for k, v in info.items() if v not in (None, "")})] + \
                         [p for p in d["paths"] if p.get("path") != path]
            d["paths"] = d["paths"][:self.n]
            d["last_dir"] = os.path.dirname(path)
            try:
                self._write(d)
            except OSError:
                pass
            return d

    def remove(self, path):
        with self.lock:
            d = self.read()
            d["paths"] = [p for p in d["paths"] if p.get("path") != path]
            try:
                self._write(d)
            except OSError:
                pass
            return d


class App:
    def __init__(self, workdir):
        self.workdir = workdir
        self.session = None
        self.job = None            # dict(name, kind, started, done, error)
        self.log = collections.deque(maxlen=400)
        self.lock = threading.Lock()
        self.tiles = {}
        self.optimizer = None
        self.recent = Recent(workdir)

    def _log(self, *a):
        line = " ".join(str(x) for x in a)
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")

    def submit(self, kind, name, fn, prepare=None, **info):
        """Run fn on a thread; one GPU job at a time. prepare() -> (name, fn), called under the lock once the slot
        is free (e.g. to create a project job only when it can run)."""
        with self.lock:
            if self.job and not self.job.get("done"):
                raise RuntimeError(f"busy: {self.job['kind']} '{self.job['name']}' is still running")
            if prepare is not None:
                name, fn = prepare()
            self.job = dict(kind=kind, name=name, started=time.time(), done=False, error=None, info=info)

        def run():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - incl. SystemExit from the readers; report it to the page
                self.job["error"] = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self._log("ERROR", self.job["error"])
                tb = traceback.format_exc().splitlines()
                if len(tb) >= 3:
                    self._log(tb[-3])
            finally:
                self.job["done"] = True
                self.job["seconds"] = round(time.time() - self.job["started"], 1)
                self._log(f"[{kind}] {name} finished in {self.job['seconds']}s")
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return name, t


def _progress(c, job):
    """Iteration statistics of the running classification, with elapsed time and a rough time left."""
    h = c.history[-1]
    n_iter = max(int(c.p.n_iter), h["it"])
    first_full = n_iter - max(int(c.p.n_full_iter), 0)          # iterations after this one are full passes
    n_full = int(getattr(c, "Na", h["n"]))
    recent = c.history[-3:]
    per_particle = sum(x["time"] for x in recent) / max(sum(x["n"] for x in recent), 1)
    batch = h["n"] if h["it"] <= first_full else c.history[0]["n"]
    eta = sum(per_particle * (n_full if it > first_full else batch) for it in range(h["it"] + 1, n_iter + 1))
    return dict(it=h["it"], n_iter=n_iter, res=round(h["res"], 1), est_res=round(h["est_res"], 1),
                n=int(h["n"]), changed=round(100 * float(h["changed"]), 1), maxprob=round(float(h["maxprob"]), 3),
                iter_s=round(float(h["time"]), 2), elapsed=round(time.time() - job["started"], 1), eta=round(eta),
                full=h["it"] > first_full, n_rot=int(h["n_rot"]), n_trans=int(h["n_trans"]))


def tile_png(img, size=192):
    from PIL import Image
    im = img.astype(np.float32)
    lo, hi = np.percentile(im, [0.5, 99.7])
    u8 = (np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).resize((size, size), Image.LANCZOS).save(buf, format="PNG")
    return buf.getvalue()


def create(workdir, project_dir=None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    from .optimize import apply as apply_settings
    from .optimize import optimize
    from .session import Session

    app = FastAPI(title="Celeris2D")
    st = App(workdir)

    def sess():
        if st.session is None or st.session.clf is None:
            raise HTTPException(400, "no dataset loaded")
        return st.session

    def run_params(body):
        p = {k: body[k] for k in RUN_KEYS if k in body and body[k] is not None and body[k] != ""}
        for k in ("hard_pose", "class_prior"):          # "auto" -> None
            if p.get(k) in ("auto", "on", "off", "true", "false"):
                p[k] = None if p[k] == "auto" else p[k] in ("on", "true")
        return p

    set_cache = {}

    def set_counts(s, name, r):
        """good / ambiguous / bad particle counts of a result with a selection (cached per selection)."""
        if r.good is None:
            return None
        key = (name, id(r), r.good.tobytes())
        if key not in set_cache:
            set_cache.clear() if len(set_cache) > 64 else None
            set_cache[key] = {kk: int(len(v)) for kk, v in s.sets(name).items()}
        return set_cache[key]

    def load_kwargs(body):
        """Session.load() arguments from the page's load-stage settings."""
        kw = {k: body[k] for k in LOAD_KEYS if k in body and body[k] not in (None, "")}
        if "polarity" in kw:
            v = kw["polarity"]
            kw["polarity"] = {u: k for k, u in POLARITY_UI.items()}.get(v, v) if isinstance(v, str) else int(v)
        return dict(diameter=float(kw.pop("diameter_a", 200)), max_res=float(kw.pop("max_res_a", 6)),
                    gpus=str(body.get("gpus") or "auto").strip() or "auto", subset=int(body.get("subset") or 0),
                    project_dir=body.get("project_dir") or None, **kw)

    def load_changes(s, kw):
        """Load-stage settings that differ from the loaded dataset's (they need the particles loaded again)."""
        old = getattr(s, "load_args", None) or {}
        out = []
        for k, v in kw.items():
            if k == "seed" and not kw.get("subset"):
                continue           # the seed only picks the subset at load; runs take it from the settings below
            o = old[k] if k in old else getattr(s.clf.p, k, v)
            same = (float(o) == float(v)) if isinstance(v, (int, float)) and isinstance(o, (int, float)) else o == v
            if not same:
                out.append(f"{k} {o} -> {v}")
        return out

    def submit(kind, name, fn, **info):
        try:
            return st.submit(kind, name, fn, **info)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(os.path.dirname(__file__), "web", "index.html")) as fh:
            return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})

    @app.get("/web/plotly.js")
    def plotly_js():
        """Plotly for the interactive figures: the copy inside the Python plotly package when installed (offline),
        else the CDN."""
        try:
            import plotly
            path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    return Response(fh.read(), media_type="text/javascript",
                                    headers={"Cache-Control": "max-age=86400"})
        except ImportError:
            pass
        from fastapi.responses import RedirectResponse
        return RedirectResponse("https://cdn.plot.ly/plotly-2.35.2.min.js")

    @app.get("/web/{fn}")
    def web_module(fn: str):
        """Add-on page modules (e.g. abinitio3d.js), kept out of index.html so each can change on its own."""
        path = os.path.join(os.path.dirname(__file__), "web", os.path.basename(fn))
        if not fn.endswith(".js") or not os.path.isfile(path):
            raise HTTPException(404, fn)
        with open(path) as fh:
            return Response(fh.read(), media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ a Celeris2D project (jobs/, data/, project.db)
    proj_state = dict(root=os.path.abspath(project_dir) if project_dir else None, runs={})

    def _prj(root=None):
        from .project import Project
        r = root or proj_state["root"]
        if not r or not os.path.isfile(os.path.join(r, "project.db")):
            raise HTTPException(404, "no Celeris2D project open (start with --project DIR, or open one)")
        proj_state["root"] = os.path.abspath(r)
        return Project(proj_state["root"])

    def _small(summ):
        return {k: v for k, v in (summ or {}).items() if not isinstance(v, (list, dict))}

    @app.get("/project", response_class=HTMLResponse)
    def project_page():
        with open(os.path.join(os.path.dirname(__file__), "web", "project.html")) as fh:
            return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})

    @app.get("/api/project")
    def project_jobs(root: str = ""):
        """Every job: number, type, label, status, inputs, key numbers; newest last."""
        from .jobs import registry
        from .jobspec import key_of
        prj = _prj(root or None)
        live = {t["job"].name for t in proj_state["runs"].values() if t.get("job") and t["thread"].is_alive()}
        try:                                         # jobs whose process died: failed, not running for ever
            prj.reap_stale()
        except Exception:                            # noqa: BLE001 - the list must still load
            pass
        reg = registry()
        finals = prj.final_marks()
        out = []
        for r in prj.jobs().itertuples():
            summ = prj.summary_of(int(r.id))
            out.append(dict(id=int(r.id), name=r.name, type=r.type, label=r.label,
                            status="running" if r.name in live else r.status, final=int(r.id) in finals,
                            final_note=finals.get(int(r.id), {}).get("note", ""),
                            inputs=[prj.job(int(i)).name for i in json.loads(r.inputs)],
                            created=float(r.created), updated=float(r.updated),
                            key=key_of(r.type, summ, reg), summary=_small(summ)))
        running = [k for k, t in proj_state["runs"].items() if t["thread"].is_alive()]
        return dict(root=prj.root, jobs=out, running=running)

    @app.get("/api/project/job")
    def project_job(job: str):
        """One job: parameters, full summary, lineage, dependents, its files (figures shown, maps listed)."""
        prj = _prj()
        j = prj.job(job)
        files = []
        for dp, dn, fn in os.walk(j.path):
            dn[:] = sorted(d for d in dn if d not in ("particles", "picks", "rejected", "session", "staging", "movies"))
            for f in sorted(fn):
                if f.endswith((".png", ".gif", ".mrc", ".mrcs", ".star", ".cs", ".csg", ".json", ".xml", ".csv", ".md")) \
                        and f not in ("job.json", "summary.json"):
                    full = os.path.join(dp, f)
                    files.append(dict(path=prj.rel(full), size=os.path.getsize(full)))
        movies = os.path.join(j.path, "figures", "movies")
        movie_files = sorted(prj.rel(os.path.join(movies, f)) for f in os.listdir(movies)
                             if f.endswith(".png") and ".light." not in f) if os.path.isdir(movies) else []
        run = proj_state["runs"].get(j.name)
        return dict(name=j.name, type=j.type, status=j.status, params=j.params, summary=prj.summary_of(j),
                    lineage=prj.lineage(j), dependents=prj.dependents(j), files=files,
                    movie_figures=len(movie_files), movie_files=movie_files, final=j.id in prj.final_marks(),
                    log=prj.rel(j.log_path) if os.path.exists(j.log_path) else None,
                    error=run["error"] if run else None)

    @app.get("/api/project/log")
    def project_log(job: str, offset: int = 0):
        """The job's run log from byte `offset` on (the page polls it while the job runs)."""
        prj = _prj()
        j = prj.job(job)
        if not os.path.exists(j.log_path):
            return dict(text="", offset=0, status=j.status)
        with open(j.log_path, "rb") as fh:
            size = fh.seek(0, 2)
            start = offset if 0 <= offset <= size else max(0, size - 200_000)
            fh.seek(start)
            return dict(text=fh.read().decode(errors="replace"), offset=size, status=j.status)

    @app.get("/api/project/file")
    def project_file(path: str):
        """A file of the project (figures are shown in the page; others download)."""
        prj = _prj()
        full = os.path.realpath(os.path.join(prj.root, path))
        if not full.startswith(os.path.realpath(prj.root) + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, path)
        with open(full, "rb") as fh:
            data = fh.read()
        mt = {".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml", ".md": "text/markdown",
              ".csv": "text/csv", ".json": "application/json"}.get(os.path.splitext(full)[1], "application/octet-stream")
        return Response(data, media_type=mt, headers={"Cache-Control": "no-cache"})

    @app.get("/api/project/jobtypes")
    def project_jobtypes():
        """Every particle job type (jobs.registry()): title, inputs with the job types they accept, settings with
        their kinds, defaults and help (from the settings dataclasses) - the New job form is built from this."""
        from .jobs import registry
        from .jobspec import describe
        return [describe(sp) for sp in registry().values()]

    @app.post("/api/project/new")
    def project_new(body: dict = Body(...)):
        """Make and run a job of any registered type: {type, inputs: {name: job}, settings: {...}, label}."""
        from .jobs import new_job, registry
        spec = registry().get(body.get("type"))
        if spec is None:
            raise HTTPException(400, f"unknown job type {body.get('type')!r}")
        from .scheduler import queue_job
        prj = _prj()
        try:
            j = queue_job(prj, spec.type, body.get("inputs") or {}, body.get("settings") or {}, body.get("label") or "")
        except ValueError as e:
            raise HTTPException(400, str(e))
        _kick()
        return dict(job=j.name, status="queued")

    @app.post("/api/project/open")
    def project_open(body: dict = Body(...)):
        """Open another project (a folder with project.db): {root}."""
        root = os.path.expanduser(str(body.get("root", "")).strip())
        if os.name != "nt" and len(root) > 2 and root[1] == ":":           # a Windows path typed into WSL: E:\x -> /mnt/e/x
            root = "/mnt/" + root[0].lower() + root[2:].replace("\\", "/")
        prj = _prj(root)
        return dict(root=prj.root)

    @app.post("/api/project/final")
    def project_final(body: dict = Body(...)):
        """Mark a job final (or not): {job, final: bool, note}; final jobs and their lineage make the report."""
        prj = _prj()
        marks = prj.mark_final(body["job"], bool(body.get("final", True)), str(body.get("note", "")))
        return dict(finals=[prj.job(i).name for i in sorted(marks)])

    @app.post("/api/project/delete")
    def project_delete(body: dict = Body(...)):
        """Move a job to the project's Trash/ (refused while it runs, or when other jobs used it unless force)."""
        prj = _prj()
        try:
            return dict(trash=prj.delete_job(body["job"], force=bool(body.get("force", False))))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/project/report")
    def project_report_api(body: dict = Body(default={})):
        """The publication report of the final jobs (report.project_report): flowchart, methods, settings."""
        from .report import project_report
        prj = _prj()
        try:
            r = project_report(prj, finals=body.get("jobs") or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        with open(prj.abs(r["methods"])) as fh:
            r["methods_text"] = fh.read()
        return r

    @app.get("/api/project/flow")
    def project_flow(job: str):
        """Flowchart of one job's lineage (report/flow_<job>.png)."""
        from .report import chain, flowchart
        prj = _prj()
        j = prj.job(job)
        os.makedirs(os.path.join(prj.root, "report"), exist_ok=True)
        base = os.path.join(prj.root, "report", f"flow_{j.id:03d}")
        flowchart(prj, chain(prj, [j.id]), [i for i in prj.final_marks()], base, title=f"How {j.name} was made")
        return dict(png=prj.rel(base + ".png"), svg=prj.rel(base + ".svg"))

    @app.post("/api/project/rerun")
    def project_rerun(body: dict = Body(...)):
        """Run a job again (jobs.rerun): particle jobs as a new job with the same parameters, movie jobs carry on
        with movies they do not have. One run at a time."""
        from .scheduler import queue_rerun
        prj = _prj()
        try:
            j = queue_rerun(prj, body["job"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        _kick()
        return dict(job=j.name, status="queued")

    # ------------------------------------------------------------------ the queue (scheduler.py)
    proj_state["auto_select"] = True
    wake = threading.Event()

    def _kick():
        wake.set()

    def _scheduler():
        """Runs the open project's queued jobs one at a time, when the GPU is free."""
        from .project import Project
        from .scheduler import next_job, run_job
        while True:
            wake.wait(3.0)
            wake.clear()
            try:
                root = proj_state.get("root")
                if not root or (st.job and not st.job.get("done")):
                    continue
                prj = Project(root)
                prj.reap_stale()                      # crashed "running" jobs: failed, not blocking the queue
                j = next_job(prj)
                if j is None:
                    continue
                rec = dict(error=None, job=j)

                def work(prj=prj, j=j, rec=rec):
                    try:
                        run_job(prj, j, log=st._log, auto_select=proj_state["auto_select"])
                    except BaseException as e:      # noqa: BLE001 - recorded on the job by run_job
                        rec["error"] = repr(e)
                        raise
                    finally:
                        wake.set()                    # the next queued job
                name, rec["thread"] = st.submit("project", j.name, work)
                proj_state["runs"][name] = rec
            except RuntimeError:                      # the GPU became busy meanwhile
                continue
            except Exception as e:                    # noqa: BLE001 - the scheduler must keep running
                st._log(f"[queue] {type(e).__name__}: {e}")

    threading.Thread(target=_scheduler, daemon=True, name="project-queue").start()

    @app.get("/api/project/queue")
    def project_queue():
        from .scheduler import queue_state
        prj = _prj()
        try:
            prj.reap_stale()
        except Exception:                            # noqa: BLE001 - the queue must still show
            pass
        return dict(queue_state(prj), auto_select=proj_state["auto_select"])

    @app.post("/api/project/queue")
    def project_queue_set(body: dict = Body(...)):
        """{auto_select: bool}: a waiting Select job after every classification run from the queue."""
        if "auto_select" in body:
            proj_state["auto_select"] = bool(body["auto_select"])
        return dict(auto_select=proj_state["auto_select"])

    @app.post("/api/project/cancel")
    def project_cancel(body: dict = Body(...)):
        from .scheduler import cancel
        try:
            return dict(job=cancel(_prj(), body["job"]).name)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/project/select")
    def project_select(body: dict = Body(...)):
        """{job, classes: [1-based], note}: the choice for a waiting Select job, or a new Select of a class job."""
        from .scheduler import complete_select
        try:
            j = complete_select(_prj(), body["job"], [int(k) for k in body.get("classes") or []], body.get("note", ""))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        _kick()
        return dict(job=j.name)

    @app.get("/api/project/classes")
    def project_classes(job: str):
        """The classes of a Class2D / HeteroRefine / Class3D job (or of a Select job's source) for the gallery:
        image(s) per class (PNG data URIs), particles, resolution; and which are chosen if it is a Select."""
        from .gallery import class_gallery
        prj = _prj()
        try:
            return class_gallery(prj, job)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/project/volume")
    def project_volume(path: str, size: int = 160, color: str = ""):
        """A map of the project for the 3D viewer: binary (JSON header length, header, uint8 voxels[, uint8 colour
        volume]) - see gallery.volume_payload."""
        from .gallery import volume_payload
        prj = _prj()
        full = os.path.realpath(os.path.join(prj.root, path))
        if not full.startswith(os.path.realpath(prj.root) + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, path)
        cfull = None
        if color:
            cfull = os.path.realpath(os.path.join(prj.root, color))
            if not cfull.startswith(os.path.realpath(prj.root) + os.sep) or not os.path.isfile(cfull):
                raise HTTPException(404, color)
        return Response(volume_payload(full, max(48, min(int(size), 256)), cfull),
                        media_type="application/octet-stream", headers={"Cache-Control": "no-cache"})

    def _start_job(make, what):
        """make(project) -> (job, fn) under the GPU-job lock (no second job is made while one runs); fn runs on the
        job thread, a failure marks the job failed."""
        prj = _prj()
        rec = dict(error=None)

        def prepare():
            try:
                j, fn = make(prj)
            except ValueError as e:
                rec["bad"] = str(e)
                raise
            rec["job"] = j

            def work():
                try:
                    fn()
                except Exception as e:
                    rec["error"] = repr(e)
                    prj.set_status(j, "failed", dict(error=repr(e)))
                    raise
            return j.name, work
        try:
            name, rec["thread"] = submit("project", what, None, prepare=prepare)
        except ValueError as e:
            raise HTTPException(400, rec.get("bad") or str(e))
        j = rec["job"]
        proj_state["runs"][name] = rec
        return dict(job=j.name)

    @app.get("/api/defaults")
    def defaults():
        """load/run: plain default values (as before); meta: label, unit, help, group, type per setting."""
        d = _defaults()
        return dict(load={k: d[k] for k in LOAD_KEYS}, run={k: d[k] for k in RUN_KEYS},
                    meta=settings_meta(), groups=GROUPS,
                    presets=dict(fast=dict(psi_factor=1.5, trans_div=2.5), balanced=dict(psi_factor=1.0, trans_div=3.0),
                                 accurate=dict(psi_factor=1.0, trans_div=4.0)))

    # ------------------------------------------------------------------ file discovery
    @app.get("/api/browse")
    def browse(path: str = "", hidden: bool = False):
        """List a directory (default: folder of the last loaded file, else home): dirs, particle files, others."""
        if not path:
            last = st.recent.read().get("last_dir")
            path = last if last and os.path.isdir(last) else os.path.expanduser("~")
        return list_dir(path, show_hidden=hidden)

    @app.get("/api/discover")
    def discover_ep(path: str):
        """Jobs with particle outputs in the cryoSPARC / RELION project containing `path` (newest first)."""
        return discover(path)

    @app.get("/api/peek")
    def peek_ep(path: str):
        """Format and particle count of one candidate input file."""
        return peek(path)

    @app.get("/api/recent")
    def recent():
        return st.recent.read()

    @app.post("/api/recent")
    def recent_edit(body: dict = Body(...)):
        if body.get("remove"):
            return st.recent.remove(body["path"])
        return st.recent.add(body["path"], **{k: body.get(k) for k in ("project_dir", "count", "format")})

    # ------------------------------------------------------------------ jobs
    @app.post("/api/load")
    def load(body: dict = Body(...)):
        path = os.path.abspath(os.path.expanduser(str(body.get("path") or "").strip()))
        if not body.get("path") or not os.path.isfile(path):
            raise HTTPException(400, f"not found: {path}" if body.get("path") else "no particle file given")
        wd = body.get("workdir") or os.path.join(st.workdir, os.path.basename(os.path.dirname(os.path.abspath(path)))
                                                   + "_" + os.path.splitext(os.path.basename(path))[0])

        def job():
            st.tiles.clear()
            old = st.session
            if old is not None and old.clf is not None:  # free the previous dataset's GPU memory
                old.clf = None
                import torch
                release_cache()
            st.session = Session(wd, log=st._log)
            info = st.session.load(path, **load_kwargs(body))
            st._log(f"[load] {info}")
            st.recent.add(path, project_dir=body.get("project_dir"), count=info.get("particles"),
                          format=info.get("source"))
        submit("load", os.path.basename(path), job, path=path)
        return dict(ok=True, workdir=wd)

    @app.post("/api/classify")
    def classify(body: dict = Body(...)):
        s = sess()
        name = body.get("name") or f"run{len(s.results) + 1}"
        K, seeds, on = int(body.get("K", 50)), int(body.get("seeds", 1)), body.get("on", "all")
        params = run_params(body)
        lkw = load_kwargs(body) if any(k in body for k in LOAD_KEYS) else None
        changed = load_changes(s, lkw) if lkw else []

        def job():
            if changed:  # mask, resolution, sampling, GPUs ...: applied when particles are loaded
                st._log(f"[reload] load-time settings changed ({', '.join(changed)}): loading the particles again")
                st.tiles.clear()
                s.load(s.path, **lkw)
            elif lkw and "seed" in lkw:
                s.clf.p.seed = int(lkw["seed"])
            c = s.clf
            for k in ("hard_pose", "class_prior"):
                if k in params:
                    setattr(c, f"_auto_{'hard' if k == 'hard_pose' else 'prior'}", params[k] is None)
            # -1 = auto: the engine resolves these per run only if its auto flag is set (it is fixed at load time
            # from Params, where n_iter defaults to 25); without this a -1 from the page would mean zero iterations
            if "n_iter" in params:
                c._auto_iter = int(params["n_iter"]) < 0
            if "n_full_iter" in params:
                c._auto_full = int(params["n_full_iter"]) < 0
            if "max_offset_a" in params and float(params["max_offset_a"]) < 0:
                params["max_offset_a"] = c.p.diameter_a / 4   # -1 = diameter / 4 (the engine resolves it only at load)
            s.classify(name, K=K, seeds=seeds, on=on, **{k: v for k, v in params.items() if v is not None})
        submit("classify", name, job, K=K, seeds=seeds, on=on, reload=bool(changed))
        return dict(ok=True, name=name, reload=changed)

    @app.post("/api/stop")
    def stop():
        """Ask the running classification to stop after its current iteration (its classes so far are kept)."""
        j = st.job
        if not j or j.get("done") or j.get("kind") != "classify":
            raise HTTPException(409, "no classification is running")
        sess().clf.stop_requested = True
        j["stopping"] = True
        st._log(f"[stop] {j['name']}: stopping after the current iteration")
        return dict(ok=True)

    @app.post("/api/continue")
    def cont(body: dict = Body(...)):
        """Continue a stopped classification where it stopped (only the most recent run can be continued)."""
        s = sess()
        name = body.get("name")
        pd = getattr(s, "_pending", None)
        if not pd or pd["name"] != name:
            raise HTTPException(409, f"'{name}' cannot be continued (only the last stopped run can)")
        submit("classify", name, lambda: s.resume(name), K=pd["K"], seeds=pd["seeds"], on=pd["on"], resumed=True)
        return dict(ok=True, name=name)

    @app.post("/api/optimize")
    def optimize_ep(body: dict = Body(...)):
        s = sess()
        K, subset = int(body.get("K", 50)), int(body.get("subset", 20000))

        def job():
            best, rows = optimize(s.clf, K, subset=subset, log=st._log)
            st.optimizer = dict(best=best, trials=rows, K=K, subset=subset)
            if body.get("apply", True):
                apply_settings(s.clf, best)
                st._log(f"[optimize] applied {best}")
        submit("optimize", f"K={K} subset={subset}", job, K=K, subset=subset)
        return dict(ok=True)

    @app.get("/api/status")
    def status():
        s = st.session
        loaded = s is not None and s.clf is not None
        out = dict(loaded=loaded, job=st.job, log=list(st.log)[-120:], optimizer=st.optimizer, now=time.time(),
                   busy=bool(st.job and not st.job.get("done")))
        if loaded:
            c = s.clf
            out.update(path=s.path, particles=c.N, loaded_particles=c._n_loaded, gpus=[str(d) for d in c.devs],
                       box=c.B_full, apix=round(c.apix_full, 4), working_box=c.bmax, workdir=s.workdir,
                       results=[dict(name=k, kind=r.kind, particles=int(len(r.idx)), classes=int(len(r.images)),
                                     selected=None if r.good is None else int(r.good.sum()),
                                     on=r.info.get("on", "all"), seconds=r.info.get("seconds"),
                                     stopped=bool(r.info.get("stopped")), K=r.info.get("K"),
                                     iterations=r.info.get("iterations"), params=r.info.get("params"),
                                     sets=set_counts(s, k, r))
                                for k, r in s.results.items()],
                       load_args={k: v for k, v in (getattr(s, "load_args", None) or {}).items()
                                  if isinstance(v, (int, float, str, bool)) or v is None},
                       resumable=(getattr(s, "_pending", None) or {}).get("name"),
                       maps=[dict(name=k, on=m.get("on"), resolution_a=m.get("resolution_a"), particles=m.get("particles"),
                                  models=len(m.get("models") or [1]),
                                  classes=m.get("classes"), sym=(m.get("params") or {}).get("sym"),
                                  seconds=round((m.get("seconds_classes") or 0) + (m.get("seconds_particles") or 0), 1),
                                  apix=m.get("apix"), box=m.get("box"))
                             for k, m in getattr(s, "maps", {}).items()],
                       settings={k: POLARITY_UI.get(getattr(c.p, k), "auto") if k == "polarity" else getattr(c.p, k)
                                 for k in RUN_KEYS + LOAD_KEYS})
            if st.job and not st.job["done"] and c.history:
                out["progress"] = _progress(c, st.job)
        return out

    @app.get("/api/live")
    def live(since: int = -1, size: int = 128):
        """The running classification as it is now: current class averages (only when the iteration differs from
        `since`), class sizes and per-iteration statistics. Polled by the page while a job runs."""
        s = st.session
        j = st.job
        if s is None or s.clf is None or not j or j.get("kind") != "classify" or j.get("done"):
            return dict(running=False, job=j)
        c = s.clf
        if not c.history or getattr(c, "pop_live", None) is None:
            return dict(running=True, job=j, progress=None)
        prog = _progress(c, j)
        from .session import history_rows
        out = dict(running=True, job=j, progress=prog, history=history_rows(c.history))
        if prog["it"] != since:
            imgs = c.class_images()
            pops = c.pop_live.numpy()
            if len(pops) != len(imgs):     # a new run started between the two reads; next poll catches up
                return out
            res = c.class_res.numpy() if getattr(c, "class_res", None) is not None else np.full(len(imgs), np.nan)
            order = np.argsort(-pops)
            out["classes"] = [dict(cls=int(k) + 1, particles=float(pops[k]),
                                   res_A=None if not np.isfinite(res[k]) else round(float(res[k]), 1),
                                   png="data:image/png;base64," + base64.b64encode(tile_png(imgs[k], size)).decode())
                              for k in order]
        return out

    @app.get("/api/result/{name}")
    def result(name: str):
        s = sess()
        if name not in s.results:
            raise HTTPException(404, name)
        summ = s.summary(name)
        r = s.results[name]
        order = np.argsort(-r.pops)
        summ["order"] = [int(k) + 1 for k in order]
        summ["version"] = f"{id(r)}-{0 if r.good is None else int(r.good.sum())}"
        return summ

    @app.get("/api/result/{name}/tile/{k}.png")
    def tile(name: str, k: int):
        s = sess()
        r = s.results[name]
        key = (name, id(r), k)
        if key not in st.tiles:
            st.tiles[key] = tile_png(r.images[k - 1])
        return Response(st.tiles[key], media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/select")
    def select(body: dict = Body(...)):
        s = sess()
        return s.select(body["name"], good=[int(k) for k in body.get("good", [])])

    @app.get("/api/suggest/{name}")
    def suggest(name: str):
        return dict(suggested=sess().suggest(name))

    @app.get("/api/sets/{name}")
    def sets(name: str, min_p: float = 0.5):
        return {k: int(len(v)) for k, v in sess().sets(name, min_p).items()}

    # ------------------------------------------------------------------ ab initio 3D
    AB_HELP = dict(
        sym="Point group imposed from the start: C1 (none), Cn, Dn (e.g. C3, D2). RELION axis convention. "
            "Symmetry is never detected automatically.",
        relax_from="C1 runs only: solve with this symmetry first (e.g. C3), then refine without symmetry, searching "
                   "the symmetry-related poses of every particle (pseudo-symmetric particles). Empty = direct C1.",
        n_starts="Random starts of the class stage; the start whose projections agree best with the classes is kept.",
        class_res_start="Class stage starts at this resolution (A) ...",
        class_res_end="... and ends at this one (A). Class averages rarely support better than ~12-15 A here.",
        class_iter="Class-stage iterations per start.",
        n_iter="Particle-stage iterations: local search of orientation and shift around the pose inherited from 2D.",
        res_start="Particle stage: first alignment resolution (A); after that it follows the half-map FSC.",
        n_local="Local search width: (2n+1)^2 out-of-plane tilts x (2n+1) in-plane angles x 9 shifts per particle.",
        n_models="3D classes. 1 = one map. More: after the consensus map, the particles are classified into this many "
                 "3D classes at their consensus poses (after relax_from, over its symmetry-related poses).",
        class3d_res="3D classification compares the classes at this resolution (A), or the consensus one if worse.",
        class3d_iter="3D classification: at most this many iterations (it stops when < 0.5% of particles change).",
        seed="Random seed of the class-stage starts.")

    @app.get("/api/abinitio/defaults")
    def ab_defaults():
        from dataclasses import asdict
        from .abinitio import AbinitioParams
        return dict(values=asdict(AbinitioParams()), help=AB_HELP)

    @app.post("/api/abinitio")
    def abinitio_ep(body: dict = Body(...)):
        from dataclasses import fields
        from .abinitio import AbinitioParams
        s = sess()
        on = str(body.get("on") or "").strip()
        if not on:
            raise HTTPException(400, "no result given (on = '<result>' or '<result>:good')")
        name = body.get("name") or f"{on.replace(':', '_')}_3d"
        types = {f.name: f.type for f in fields(AbinitioParams)}
        prm = {}
        for k in types:
            v = body.get(k)
            if v is None or v == "":
                continue
            prm[k] = str(v).strip().upper() if types[k] in (str, "str") else (int(v) if types[k] in (int, "int") else float(v))
        submit("abinitio", name, lambda: s.abinitio(name, on, **prm), on=on, **prm)
        return dict(ok=True, name=name)

    @app.get("/api/map/{name}/{fn}")
    def map_file(name: str, fn: str):
        s = sess()
        ok = {"map.mrc", "map_nomask.mrc", "mask.mrc", "from_classes.mrc", "half1_unfil.mrc", "half2_unfil.mrc",
              "consensus.mrc", "reprojections.png", "views.png", "fsc.png", "map_views.png", "summary.json"}
        import re
        if name not in s.maps or not (fn in ok or re.fullmatch(r"(map|map_nomask|mask|half[12]_unfil|from_classes)_\d+\.mrc", fn)):
            raise HTTPException(404, f"{name}/{fn}")
        path = os.path.join(s.workdir, name, fn)
        if not os.path.isfile(path):
            raise HTTPException(404, fn)
        with open(path, "rb") as fh:
            data = fh.read()
        mt = "image/png" if fn.endswith(".png") else "application/json" if fn.endswith(".json") else "application/octet-stream"
        return Response(data, media_type=mt, headers={"Cache-Control": "no-cache"})

    @app.post("/api/export")
    def export(body: dict = Body(...)):
        s = sess()
        outdir = os.path.abspath(os.path.expanduser(str(body.get("outdir") or "").strip()))
        if not body.get("outdir"):
            raise HTTPException(400, "no export folder given")
        if body["name"] not in s.results:
            raise HTTPException(404, f"no result named {body['name']}")
        try:
            return s.export(body["name"], outdir, min_p=float(body.get("min_p", 0.5)), fmt=body.get("fmt", "auto"))
        except OSError as e:
            raise HTTPException(400, f"cannot write to {outdir}: {e.strerror or e}")

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Celeris2D web dashboard")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on the network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workdir", default=os.path.abspath("celeris2d_web"))
    ap.add_argument("--project", default="", help="a Celeris2D project directory: its jobs at /project "
                    "(default: the current directory, if it is a project)")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.makedirs(a.workdir, exist_ok=True)
    project = a.project or (os.getcwd() if os.path.isfile("project.db") else None)
    import uvicorn
    print(f"Celeris2D dashboard on http://{a.host}:{a.port}  (workdir {a.workdir})")
    if project:
        print(f"project {os.path.abspath(project)}: http://{a.host}:{a.port}/project")
    uvicorn.run(create(a.workdir, project), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
