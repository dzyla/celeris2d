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
     `hard_pose`, `class_prior` depend on the active particle count; ≥100k → hard pose + uniform prior).
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
   `plots_preprocess.py`. `picking.py`: blob picker (NCC with a soft disk on a binned, phase-flipped micrograph;
   candidates keep their scores so thresholds change without re-picking), box size from CTF delocalisation,
   batched GPU extraction. `project.py` + `pipeline.py` (+ `jobs.py`: Class2D / Select / InitialModel / Refine3D / HeteroRefine and the job registry, `polish.py`: Polish,
   `sharpen.py`: Sharpen, `localres.py`: local resolution, `diagnostics.py`: FSC files, angular statistics): the
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
   `Session`), and `webapp.py` (FastAPI + `web/index.html`) and `mcp_server.py`, both thin wrappers
   around one `Session`, running one GPU job at a time; `mcp_server.py` also has the project tools. The 3D part of the page is `web/abinitio3d.js`, loaded
   by index.html, which only emits `celeris:status` / `celeris:result` events for it — keep the two independent.

### Cross-cutting conventions

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
  type is an entry in `jobs.registry()` (settings dataclass, inputs, make, run).
- `Params` comments record what was tried and why defaults are what they are (e.g. `march_margin=0`,
  `online_decay=0`, `sigma_init_factor=1`, `polish` off). Read them before changing a default.
