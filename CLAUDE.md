# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Celeris2D: GPU maximum-likelihood 2D classification of cryo-EM particles in PyTorch (RELION `.star`,
cryoSPARC `.cs/.csg`, or `.mrc/.mrcs` in; RELION/cryoSPARC-compatible particles and class averages out).
README.md documents user-facing options, outputs and measured results.

## Setup and running

Needs an NVIDIA GPU and a CUDA build of PyTorch ≥ 2.4 (Blackwell / RTX 50xx: CUDA 12.8+ wheels).
```bash
conda create -n celeris2d python=3.12 -y && conda activate celeris2d
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .        # installs entry points celeris2d, celeris2d-web, celeris2d-mcp
```
- CLI: `celeris2d -i particles.star -o runs/k25 --K 25 --diameter 250 --max-res 12` (also `python -m celeris2d`).
  `--subset N` classifies a random subset — use it for quick checks of a change.
- Dashboard: `celeris2d-web --port 8765` (restart it after code changes).
- MCP server: `celeris2d-mcp` (stdio).

`pytest -q tests` runs synthetic end-to-end checks (2D: known classes and angles, stop/continue, dashboard API;
3D: ab initio orientations, symmetry detection, projector conventions; needs a CUDA GPU, ~2 min). If the
`celeris2d` env is an editable install of another checkout, prefix `PYTHONPATH=.`. There is no linter config or
CI. Algorithm changes are validated on real datasets with the scripts in
`scripts/` (run as `python scripts/<name>.py ...`, usage in each docstring): `validate.py` (ψ consistency
vs refined 3D angles), `evaluate_frc.py` / `evaluate_cs.py` (classifier-independent half-set FRC per
class from the written alignments), `separation_eval.py` / `evaluate_selection.py` (precision/recall vs a
curated particle set), `match_reference_classes.py`, `compare.py`. Commit messages report such numbers
(e.g. "ψ consistency 78.5% vs 79.0%") when an algorithm change is claimed neutral or better.

## Architecture

Layers, bottom to top:

1. **`sources.py` / `star.py`** — input readers. Every source (`RelionSource`, `CryosparcSource`,
   `StackSource`, via `open_source`) exposes the same duck-typed interface: `N, box, apix, ctf (N×6),
   optics, parts (RELION-style DataFrame for output), blocks, chunks()` (threaded per-stack-file reads,
   shuffled file order), `subset(idx)`, `pick_positions()`, `polarity_hint`, `kind`.
2. **`engine.py`** — `Params` (dataclass, every algorithm knob) and `Classifier2D`.
   - Constructing `Classifier2D` starts a background thread that streams, normalises, FFTs and
     Fourier-crops all particles once into `self.data` (fp16/complex64, on the master GPU if it fits,
     else pinned host RAM). Classification starts after the first ~20k particles; only full passes
     wait for the rest (`_wait_loaded`).
   - `run(active=idx, n_classes=K, seed=s)` re-classifies any subset of the loaded particles with no
     I/O. `_set_active` resets run state and resolves "auto" params (`n_iter`, `n_full_iter`,
     `hard_pose`, `class_prior` depend on the active particle count; ≥20k → hard pose, ≥100k also uniform prior).
     Everything above the engine (deep2D, ISAC, seeds, optimizer, Session) relies on this reuse.
   - Per-particle state (`cls, psi, offsets, maxprob, loglik, cpost`) lives on the host, indexed by
     global particle index; the master device (`devs[0]`) holds classes and does reconstruction.
   - `iterate(it)`: `_iter_settings` computes per-iteration alignment vs reconstruction resolution
     limits (separate on purpose), angular/translation grids, and online-EM subset vs full pass.
     `_run_pass` has one thread per GPU pulling batches from a shared queue; `_batch` does the E-step
     as batched complex GEMMs (separable shifts) and accumulates M-step terms per odd/even half-set.
   - Spectra are rfft2 with `norm="forward"`, so cropping a spectrum is exact Fourier downsampling and
     values are box-invariant. `bmax` is the working box (Nyquist at `max_res_a`); output class
     images are at `box_a / bmax` Å/px, not the input pixel size.
3. **Algorithms on top of a loaded `Classifier2D`**: `stability.py` (multi-seed runs, classes matched
   to run 0 by Hungarian assignment on shared membership → per-particle P(good)), `deep2d.py`
   (iterative good/bad sorting against a fixed round-0 "standard"), `isac.py` (generations of
   multi-seed runs, soft stability), `optimize.py` (small settings grid scored by reproducibility).
   `abinitio.py` (3D from a 2D result: class-average stage, particle SGD, local refinement with gold-standard
   halves, symmetry detection / relaxation, 3D classification; `Volume` = 2x-padded Fourier volume in the units
   of the 2D particle spectra; RELION Euler convention, image = A @ volume, checked against Refine3D) and
   `plots3d.py` (its figures). It only reads a finished 2D `Result` — the 2D engine does not know about it.
   `preprocess.py`: movies -> micrographs on the GPU (gain orientation check, whole-frame + patch motion
   correction, dose weighting, per-micrograph CTF fit with background/envelope by least squares), figures in
   `plots_preprocess.py`. `picking.py`: blob or template picker (NCC with a soft disk, or with 2D class averages / map projections at
   in-plane rotations - `PickSettings.templates`, on a binned, phase-flipped micrograph;
   candidates keep their scores so thresholds change without re-picking), box size from CTF delocalisation,
   batched GPU extraction. `project.py` + `pipeline.py` (+ `jobs.py`: Class2D / Select / InitialModel / Refine3D / HeteroRefine and the job registry, `polish.py`: Polish,
   `sharpen.py`: Sharpen, `ctfrefine.py`: per-particle defocus (CtfRefine, a particle job that points at the
   stacks it refined; `Project.particles` prefers per-particle defocus columns), `masking.py`: solvent masks
   (`auto_mask` also runs inside `refine_particles` every iteration: Refine3D `mask="auto"`; `make_mask` with user
   regions / segments; `masks.py` is Sharpen's wrapper; MaskCreate job), `nufilter.py`: non-uniform regularisation
   (Refine3D `nu=True`: references filtered region by region, cutoffs by cross-validation - strict by default, from two
   folds inside each half), `optics.py`: anisotropic magnification and odd / even aberrations fitted by gradient
   (diagnostic; not applied in refinement yet), `localres.py`: local resolution, `diagnostics.py`: FSC files, angular
   statistics, `mapscore.py`: MapScore, particles vs a map against phase-randomised controls with per-2D-class shares,
   `subtract.py`: Subtract (signal subtraction, re-centring on the mask, symmetry expansion: a particle job whose rows
   are (uid, copy)) and LocalRefine (focused refinement with angle / shift limits about the mask centre), `symexpand.py`:
   SymExpand (symmetry expansion of poses, no images: rows (uid, copy); LocalRefine / Refine3D / Class3D / Subtract
   take it and load every image once, `jobs._posed_particles(expand=True)` gives each row's image index),
   `modes.py`: Modes3D (modes of variation: probabilistic PCA of the density at fixed poses, half-set
   reproducibility per mode) and ModeMaps (series, bins along a mode, clusters of the coordinates as classes: one of
   `project.CLASS_JOBS`)): the
   movies -> particles -> 3D project store and its jobs
   (layout and rules in docs/project-design.md: `jobs/NNN_<type>_<label>/` small results, `data/NNN/` heavy and
   purgeable, catalog `project.db`, particle uid/key, selections as uid lists); `pipeline.run` overlaps decoding
   (worker processes), GPU stages and writes. `export.py`: RELION / cryoSPARC files on request, and `job_source`
   (an extract job folder is a particle input for the CLI / Session / MCP). `scripts/pick_eval.py` runs the chain on
   movies and scores pick recall against a refined set. `scripts/preprocess_eval.py` compares it with
   cryoSPARC on the same movies, `scripts/extract_eval.py` + `abinitio_truth.py ref` compare micrograph sets by
   reconstruction at known poses (docs/preprocess-status.md). Judge motion correction by such downstream FSC or
   the even/odd-frame FRC, not by FRC against another program's micrograph (shared shot noise).
4. **Front ends**: `cli.py` (one-shot runs, output writers, `pick_devices`), `session.py` (stateful
   `Session`), and `webapp.py` (FastAPI; its page is the project dashboard `web/project.html` at /project (markup, CSS; scripts `web/dash.js`: rail with
   List / Stages / Flow views, job pages, the New job picker and forms from `uimeta.py`'s plain names, groups and basic /
   advanced settings; `web/pages.js`: Import and Extract pages from `jobview.py`) - `/`
   redirects there; the old single-dataset 2D page is gone, its /api/* Session endpoints remain) and `mcp_server.py`,
   both thin wrappers around one `Session`, running one GPU job at a time; `mcp_server.py` also has the project tools.
   Movie jobs (Import, MotionCtf, Pick, Extract) are dashboard job types through `moviejobs.movie_registry()`.

### Cross-cutting conventions

- **GPU memory**: size batches / boxes from `engine.gpu_free` (free memory less 10 % of the card, >= 1.5 GB, and
  under `CELERIS2D_VRAM_GB`), never from `torch.cuda.mem_get_info` directly; `engine.cap_memory` hard-limits the
  process there, so an overrun is an out-of-memory error instead of WSL spilling GPU memory into system RAM (many
  times slower). Big 3D intermediates: free what the next phase does not need (accumulators after the solve,
  references before back-projection), work in slabs, build projection-only volumes on the band's box.
- **3D volumes** (`abinitio.Volume`): 2x-padded full complex grids, 64 b^3 bytes each; `refine_particles` picks
  pad 1 (gridding-corrected) when pad 2 does not fit (box 512 in 16 GB), `Volume.real` / `set_real` work in slabs.
  Symmetry in `refine_particles` (pad 2): each image inserted once, then `Volume.symmetrize` resamples the
  accumulators (RELION's way; per-operator insertion was ~90 % of a C8 iteration).
- **Refinement masks**: every refinement job has `mask` = auto (dynamic, from each model's map every iteration below
  15 A) / static (once from the starting map) / none (particle sphere) / a MaskCreate job or .mrc
  (`jobs.solvent_mask_arg`). NU refinements write `nu_cutoff.mrc`, `nu.png` and per-iteration `nu` statistics.
- **Dashboard (/project)**: `scheduler.py` is the queue (statuses queued / waiting / running / finished / failed /
  cancelled in the catalog; a queued job runs when its inputs finished, in a process of its own (`start`, output in `jNNN_process.log`, so it outlives a dashboard restart; `busy` keeps one GPU job at a time) that `stop` ends; a queued MotionCtf -> Pick -> Extract runs as one chain (`movie_chain`); a running Class2D writes `live.json` every iteration (`gallery.live_classes`) and the page shows it with the log tail; a Class2D with `stream=True` may start beside its running Extract job (rounds on the particles so far, each from the last round's classes, final round when extraction ends: `jobs._stream_class2d`); `micview.py` serves the movie jobs' panel (one MotionCtf -> Pick -> Extract `family`: per-micrograph rows; the dashboard's preview worker `warm` writes data/NNN/Preview/<stem>.png (raw binned micrograph: the page low-passes it with its slider and draws the picks), .ps.png and .json (1D CTF fit, global / local motion - MotionCtf Stats now keep the patch paths - patch defocus); never in the job's process); the movie pipeline skips a movie it cannot read and Import leaves out truncated MRC files; 3D jobs run inside `live.live_maps` (scheduler.run_job): `abinitio.LIVE` gets every SGD round's / refinement iteration's maps, written downsampled on a thread as live3d_mN.mrc + live3d.json for the page; after a Class2D / HeteroRefine / Class3D run
  from the queue a Select job waits for the user - `complete_select`; MCP / scripts get none). `gallery.py` gives
  class galleries and 8-bit volumes for the browser's WebGL2 raymarcher (`web/volume.js`; cropped to the particle first, so most maps are shown at full pixel size; default level = the map in one piece, dust hidden, rough maps smoothed by one voxel; local-resolution colour range from the values on the surface). Every `themed` figure also
  writes `x.plot.json` (`figjson.py`; `Job.out` keeps it next to x.png) that `web/plots.js` draws with Plotly in the
  same layout; panels it cannot describe (projections, meshes, colour bars) are embedded as pictures. `report.py`:
  final jobs (`Project.mark_final`) -> flowchart.svg / methods.md / settings.csv.
- **Sharpening inside refinement jobs**: Refine3D / HeteroRefine / Class3D / LocalRefine have `sharpen`, `local_res`,
  `mtf`, `mtf_apix`: the job sharpens its own half maps with its own mask (`sharpen.sharpen_in_job`, same code as
  the Sharpen job via `sharpen_maps`; files sharpened.mrc, postprocess.png, localres*, sharpen_fsc.json; classNN_
  prefix per class). MTF files are RELION's .star (not shipped here; RELION has them in relion/data/).
- **TIFF orientation**: frames are read in file row order; RELION's micrographs of the same TIFF movies are their
  y-mirror (`MotionSettings.flip_y=True` matches RELION: coordinates, astigmatism angle, hand). Motion shifts are
  relative to the frame average (RELION: frame 1), so coordinates differ from RELION's by the mean drift.

- **Pose sign conventions** (top of `cli.py`): internal model is particle = shift_t(rot_psi(class)).
  RELION output uses `psi_relion = -psi`, `origin = -t`; cryoSPARC `alignments2D` uses
  `CS_POSE_SIGN`/`CS_SHIFT_SIGN = -1`. These were verified by rebuilding real RELION/cryoSPARC class
  averages — do not change them without re-running that check (`evaluate_frc.py` / `evaluate_cs.py`).
- **cryoSPARC output**: "slim" `particles.cs` holds only uid + `alignments2D`, and `particles.csg`
  links the other result groups to the *input* metafiles by absolute path. A subsetted source forces
  "full" output.
- **Session results** are named; each is saved as `<name>.result.npz/.json` (plus gallery PNG and
  `.mrcs`) in the work dir and restored when the same input path with the same N is loaded again.
  The particle selector `on=` accepts `all`, `<result>`, or `<result>:good|bad|ambiguous`.
- Good/bad class decisions are explicit (`Session.select`, MCP `select_good`); `suggest` /
  `split_two_clusters` only propose. Class labels are 0-based in engine/result arrays; class numbers
  are 1-based from the `Session` API upward (`select`, MCP, web UI, galleries, `rlnClassNumber`).
- Parameters exposed in the web UI are listed in `webapp.LOAD_KEYS` / `RUN_KEYS`; CLI flags map to
  `Params` fields in `cli.main`. A new `Params` field must be wired into each front end that should
  expose it.
- **Figures**: every figure function is decorated with `plotstyle.themed` and takes its colours from
  `plotstyle.T` (dark default `x.png`, light `x.light.png`); no colour literals in figure code. A new particle job
  type is an entry in `jobs.registry()` (`jobspec.JobSpec`: settings dataclass with a help comment per field,
  inputs, make, run, key numbers, figures); forms, MCP `new_job`, rerun and the job list are built from it.
  Result files go through `job.out(name)` / `job.find(name)` (maps/ figures/ data/, `jNNN_` prefix); run
  functions are `@logged` (run log).
- `Params` comments record what was tried and why defaults are what they are (e.g. `march_margin=0`,
  `online_decay=0`, `sigma_init_factor=1`, `polish` off). Read them before changing a default.
