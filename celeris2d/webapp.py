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

from .engine import Params

# settings shown in the UI (Params fields); the dataset-level ones are used at load time
_LOAD_KEYS = ("diameter_a", "max_res_a", "init_res_a", "psi_step_min", "psi_factor", "trans_div", "max_shift_a",
              "notch_thresh", "noise_model", "seed")
_RUN_KEYS = ("n_iter", "n_full_iter", "batch_per_class", "march_margin", "recon_margin", "frc_reg", "tau2_fudge",
             "sigma_init_factor", "solvent_flatten", "gold_standard", "null_class", "solvent_thresh", "solvent_dilate_a",
             "center_classes", "recenter_thresh", "hard_pose", "class_prior", "polish", "reseed_frac", "max_offset_a")
_FIELDS = {f.name for f in dataclasses.fields(Params)}
LOAD_KEYS = tuple(k for k in _LOAD_KEYS if k in _FIELDS)   # tolerate Params fields being renamed / removed
RUN_KEYS = tuple(k for k in _RUN_KEYS if k in _FIELDS)

# Plain-language description of every setting the dashboard shows. Wording follows the Params comments
# (engine.py) and the CLI --help strings (cli.py); keep them in sync when a default or behaviour changes.
# group: where the page shows it; stage: when it takes effect (load = at Load, run = at every Run).
GROUPS = [
    dict(id="load_basic", title="Dataset", stage="load", basic=True,
         help="Used when you press Load. To change them for a loaded dataset, press Load again."),
    dict(id="load_adv", title="Advanced dataset settings", stage="load", basic=False,
         help="Sampling and preprocessing fixed at Load time. The defaults suit most data sets."),
    dict(id="run_basic", title="Classification", stage="run", basic=True,
         help="Used every time you press Run (no reloading needed)."),
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
    "gpus": dict(group="load_basic", label="GPUs", unit="", type="text", default="auto", important=True,
                 help="'auto', or nvidia-smi GPU indices such as 0,2,3. Auto: the GPU with the most memory for small "
                      "jobs, all GPUs once the work (particles × working box²) is worth splitting (about 100k particles "
                      "at a 64 px working box)."),
    "subset": dict(group="load_basic", label="Random subset", unit="particles", type="int", default=0, important=True,
                   help="Load only this many randomly chosen particles (0 = all). Useful for quick tests of settings; "
                        "the choice is reproducible through the random seed."),
    "speed": dict(group="load_basic", label="Speed preset", unit="", type="choice", default="balanced", important=True,
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
    "seed": dict(group="load_adv", label="Random seed", unit="", type="int",
                 help="Seeds the random subset and the class initialisation. Change it to get an independent start "
                      "with otherwise identical settings."),
    # ---- classification (run) ---------------------------------------------------------------------------------
    "K": dict(group="run_basic", label="Number of classes (K)", unit="", type="int", default=50, important=True,
              help="More classes separate more views and more kinds of junk, but each class gets fewer particles and a "
                   "noisier average. Run time grows roughly with K."),
    "seeds": dict(group="run_basic", label="Independent starts (seeds)", unit="", type="int", default=1, important=True,
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
                        help="During the first 70 % of iterations, a class holding fewer than this fraction of the "
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
    "null_class": dict(group="run_classes", label="Null ('no particle') class", unit="", type="bool",
                       help="Adds a 'no particle' hypothesis with a learned share. A pick that fits no class better "
                            "than empty noise (likelihood averaged over all poses) goes there and is left out of "
                            "the class averages, so structureless picks stop building honeycomb classes. Lowers the "
                            "confidence of those particles; mean P(null) per class is in class_stats.csv."),
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

    def submit(self, kind, name, fn, **info):
        with self.lock:
            if self.job and not self.job.get("done"):
                raise RuntimeError(f"busy: {self.job['kind']} '{self.job['name']}' is still running")
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
        threading.Thread(target=run, daemon=True).start()


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


def create(workdir):
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

    def submit(kind, name, fn, **info):
        try:
            st.submit(kind, name, fn, **info)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(os.path.dirname(__file__), "web", "index.html")) as fh:
            return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})

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
            st.session = Session(wd, log=st._log)
            kw = {k: body[k] for k in LOAD_KEYS if k in body and body[k] not in (None, "")}
            info = st.session.load(path, diameter=float(kw.pop("diameter_a", 200)), max_res=float(kw.pop("max_res_a", 6)),
                                   gpus=str(body.get("gpus") or "auto").strip() or "auto",
                                   subset=int(body.get("subset") or 0),
                                   project_dir=body.get("project_dir") or None, **kw)
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

        def job():
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
        submit("classify", name, job, K=K, seeds=seeds, on=on)
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
                                     on=r.info.get("on", "all"), seconds=r.info.get("seconds"))
                                for k, r in s.results.items()],
                       settings={k: getattr(c.p, k) for k in RUN_KEYS + LOAD_KEYS})
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
        out = dict(running=True, job=j, progress=prog)
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
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.makedirs(a.workdir, exist_ok=True)
    import uvicorn
    print(f"Celeris2D dashboard on http://{a.host}:{a.port}  (workdir {a.workdir})")
    uvicorn.run(create(a.workdir), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
