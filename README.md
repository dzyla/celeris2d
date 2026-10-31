# Celeris — fast GPU cryo-EM processing, from movies to 2D classes and 3D maps

Celeris (Python package and commands: `celeris2d`, or the aliases `celeris`, `celeris-web`, `celeris-mcp`) runs a
cryo-EM project on the GPU in a browser dashboard: movies, motion correction and CTF, picking, extraction, 2D
classification, ab initio and refined 3D maps. At its core is maximum-likelihood 2D classification in PyTorch, built for speed on 0.2–3M+ particles: RELION
`.star`, cryoSPARC `.cs/.csg`, or plain `.mrc/.mrcs` in; RELION/cryoSPARC-compatible particles and class
averages out. Multi-GPU, streams particles while it classifies, and keeps a person or an agent in the
loop for good/bad decisions (CLI, Python session, or MCP server).

## Install
Needs an NVIDIA GPU and a CUDA build of PyTorch ≥ 2.4 (RTX 50xx / Blackwell: CUDA 12.8+ wheels).
```bash
git clone https://github.com/dzyla/celeris2d.git
cd celeris2d
conda create -n celeris2d python=3.12 -y && conda activate celeris2d
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for your driver
pip install -e .                                                       # celeris2d, celeris2d-web, celeris2d-mcp
```
Update later with `git pull` (the editable install picks up the new code; restart a running dashboard).

## Quick start
```bash
# one classification (auto: all GPUs above 100k particles)
celeris2d -i J43/J43_particles.csg -o runs/k50 --K 50 --diameter 220 --speed fast
celeris2d -i Extract/job007/particles.star -o runs/k25 --K 25 --diameter 250 --max-res 12

# iterative good/bad sorting with a fixed standard (round 0 classes judged automatically or --good-classes)
celeris2d -i particles.star -o runs/deep --K 25 --diameter 250 --deep2d [--good-classes 3,5,12]
```

## Inputs
* RELION `particles.star` (3.0 / 3.1+), cryoSPARC `.cs` or `.csg` (each result group read from its own
  metafile), or a plain stack (`--apix` if the header lacks it; no CTF).
* CTF from metadata (RELION conventions; cryoSPARC fields converted). `--no-ctf` forces CTF = 1.
* Contrast auto-detected (protein must end up white; `--invert/--no-invert` override). Every particle is
  normalised to its background.
* CTF-premultiplied data are not supported.

## Outputs (`-o`)
| file | content |
|---|---|
| `classes.png` | gallery: tiles sorted by size, class number, particles / share, confidence bar |
| `stats.png` | resolution, confidence, convergence, time per iteration, class sizes / confidence |
| `classes.mrcs`, `class_stats.csv`, `model.star`, `history.json` | averages, per-class stats, run log |
| `particles.star` / `particles.cs+.csg` | `--particles slim` (default; cryoSPARC: uid + alignments2D, linked to the input metadata), `full`, or `none` |
| `duplicates.star/.cs` | picks whose aligned centres coincide with a better-scoring pick (`--dup-dist`) |
| deep2D | `good/ambiguous/unstable/bad_particles`, per-round galleries, `deep2d_rounds.png` |

Conventions verified against real jobs: RELION ψ/origin by rebuilding a RELION Class2D job's averages,
cryoSPARC pose/shift by rebuilding J43's class averages (correlation 0.82 vs ≤ 0.36 for other sign choices).

## How it works
* **Model** (RELION's classic EM): Fourier-space Gaussian likelihood with CTF and per-pixel noise; every
  particle is scored against all classes × in-plane angles × sub-pixel shifts.
* **Noise-safe on hard data** (what closed most of the gap to cryoSPARC on J43):
  * alignment never goes beyond the resolution the data currently support (`--march-margin 0`), and in full
    passes keeps the limit set by the constant-size online batches; class averages are reconstructed
    further (+10 shells) — alignment and reconstruction limits are separate, as in cryoSPARC;
  * class averages are Wiener-filtered with an **odd/even half-set FRC** (not their own power), SSNR forced
    to fall with resolution;
  * optional **shape-adaptive solvent flattening** (`--solvent-flatten`): each class is masked by a soft
    envelope of its own density. Off by default (as in cryoSPARC): on small or weak particles the envelope
    takes in surrounding noise and grows a honeycomb; it helps clean, well-separated views;
  * fixed-frame artefact pixels (detector stripes) are found from the mean power spectrum and notched;
  * re-centring on density > 20 % of the class maximum, after removing the best-fit background plane (a class
    of an ice or carbon edge otherwise walks out of the box and empties);
  * **no empty classes**: a class that falls below a quarter of its share is re-seeded by splitting the largest
    class along the first principal component of its members (two real, distinct averages, not a noisy copy
    that dies again), during online iterations and full passes alike (all but the last two).
* **Speed**: shifts are separable, so the pose search for a batch is two batched complex GEMMs, the
  back-projection the same GEMMs transposed (fp16 tensor cores, exact marginalisation). Particles are
  stored once as fp16 spectra (GPU if they fit, else pinned RAM) and streamed in random file order —
  classification starts after the first ~20k. GPUs pull batches from a shared queue.
* **Big data**: cryoSPARC-style online EM (500 particles / class / iteration) then full passes;
  ≥ 20k particles use hard pose (best pose per class), ≥ 100k also a uniform class prior, automatically.
* **Memory**: the whole dataset is kept as Fourier-cropped spectra (working box set by `--max-res`; fp16
  complex = 4 bytes per pixel of half the box², e.g. 64 px → 8 kB, 146 px → 43 kB per particle). It lives on the
  first GPU if it fits, otherwise in pinned host RAM (batches are copied to every GPU over PCIe, a few seconds per
  pass), otherwise in a disk-backed file in `$TMPDIR` (`--param spill_dir=/fast/ssd`) read through the page cache.
  GPU memory therefore limits the batch size, not the dataset size; 3M particles at 64 px need ~27 GB.
* Options for clean, high-SNR data: `--march-margin 5-10` (align further), `--polish` (final pass aligns to
  the reconstruction limit with classes locked; sharper on β-gal, harmful on low-SNR data).

## Separation tools (all reuse the loaded data — no re-reading)
| tool | what it does | cost |
|---|---|---|
| multi-seed P(good) | same particles, several independent starts, classes matched by membership; per-particle P(good) and votes after a selection | seeds × one run |
| deep2D | round 0 defines a fixed standard (confidence / resolution / not-a-dot cut-offs + template bank); each round re-classifies good alone (purge) and bad + purged seeded with good templates (rescue); jumpers (≥ 3 good↔bad moves) set aside | ~10 runs |
| ISAC-style | generations of multi-seed runs; groups whose members keep landing together (soft posterior agreement) are accepted, the rest goes back to the pool; stops when a generation adds < 2 % | ~16 runs |

## Ab initio 3D (from 2D classes)
After 2D classification and selecting good classes, Celeris builds a 3D map from them in about a minute on one
GPU. A 2D class is one viewing direction, so the expensive search is done on the class averages, and each particle
inherits its pose from its class (class orientation + its own in-plane angle and shift):

1. **Class stage** — every class average is matched against projections of the current map over the whole sphere
   and all in-plane angles (one GEMM), the map is rebuilt from the classes, 40 → 15 Å, from several random starts.
   The best starts are each refined briefly on the particles and the one the particles fit best is kept.
2. **Particle SGD** — rounds of a random batch of particles searched globally against the map, which moves part
   of the way towards each batch's reconstruction (30 → 12 Å). Thousands of particles constrain the structure far
   more than a few dozen averages, and batch noise lets it leave a wrong start.
3. **Refinement** — gold-standard half sets, CTF, local search of orientation and shift around each particle's
   pose, global re-searches in the first iterations (particles from mixed 2D classes can change view), the
   worst-fitting 10 % of particles left out of each reconstruction, resolution from the half-map FSC (soft sphere
   mask), alignment slightly beyond the current resolution so it can keep improving.
4. **Symmetry** — only what the user sets (no automatic detection): Cn / Dn imposed (RELION axis convention), or
   for a pseudo-symmetric particle a C1 run with `relax_from` (e.g. C3): solved with that symmetry imposed, then
   refined in C1, the first iterations searching every symmetry-related copy of each pose (symmetry relaxation;
   C1 matching of a pseudo-symmetric particle has several near-equivalent wrong answers).
5. **3D classes** (`n_models` > 1) — after the consensus, a 3D classification at the consensus poses: random-subset
   start, annealed posteriors, all classes filtered alike; after `relax_from` it runs over the
   symmetry-related poses of each particle (a difference that breaks the symmetry is found wherever it is).

Memory: the 3D working box is the largest that fits the free GPU memory (logged when it is smaller than the 2D
one); search batches are sized from free memory, so a run never spills into host RAM.

Outputs (`<workdir>/<name>/`): `map.mrc` (FSC-weighted, automatic solvent mask), `map_nomask.mrc`, `mask.mrc`,
`half1/2_unfil.mrc`, `from_classes.mrc`, `particles.star` (3D poses, RELION convention, this map's frame and
hand — ready for Refine3D), `particles_dropped.star`, `poses.npz`, `summary.json` (resolutions, timings, FSC
curves) and figures (`fsc.png`, `views.png`, `reprojections.png`, `map_views.png`). The hand of
an ab initio map is arbitrary.

Dashboard: an **InitialModel** job on a Class2D job or on the selection of its classes. Python:
`Session.abinitio(name, "run1:good", sym="C1", n_models=1)`; script: `scripts/abinitio_workflow.py`.

## Web dashboard
```bash
celeris2d-web --port 8765 --project /data/myproject   # on the GPU machine; open http://localhost:8765
ssh -L 8765:localhost:8765 <gpu-host>                  # from your laptop, if the server runs remotely
```
A processing project from movies to maps: **New project…** (a new or empty folder), then **New job** - Import
(a folder of movies), MotionCtf, Pick, Extract, Class2D, InitialModel, Refine3D, ... Jobs go into a queue and
run one at a time as soon as their inputs are ready, each in its own process (**Stop** ends it); a running job
shows its log and newest figure live. After a 2D or 3D classification a selection waits for you: click the
classes to keep (**Keep these classes**); jobs queued on it run after that. Every figure is also interactive
(zoom, hover), maps open in a 3D viewer, and **Mark final** + **Report** give the publication flowchart,
methods and settings.

## Settings optimizer (`--optimize N`, dashboard button, MCP `optimize_settings`)
Tries the alignment margin (0 / 4 shells) × solvent flattening (off / on) on N random particles,
each from two independent starts (15 iterations), and keeps the most **reproducible** setting (highest soft
agreement between the starts). ~30 s on 5k particles. Validation against curated ground truth (LoG picks,
12-setting grid): stability correlates with separation quality (Spearman 0.50 with F1, 0.62 with AP) and the
pick beats the median setting, but not always the best one — treat it as protection against bad settings.
Hard vs soft pose is left to the size rule (the score was too noisy to choose it reliably).

## Human / agent in the loop
```python
from celeris2d.session import Session
s = Session("work"); s.load("J43_particles.csg", diameter=220, gpus="0,2,3")
s.classify("r1", K=50, seeds=2)          # gallery: s.summary("r1")["gallery"] -> work/r1_classes.png
s.select("r1", good=[1, 4, 7, 9]); s.export("r1", "out/", min_p=0.5)
```
Results (and selections) are saved in the work dir and restored when the same dataset is loaded again.

**MCP server**: `celeris2d-mcp` (stdio). Tools: `load_particles`, `classify`, `isac`, `assign`,
`show_classes` (returns the gallery image + table), `suggest_good` (proposal only), `select_good`,
`particle_sets`, `export`, `status`. Register in Claude Code, e.g. for an install on a remote GPU machine:
```bash
claude mcp add celeris2d -- ssh <gpu-host> ~/miniconda3/envs/celeris2d/bin/celeris2d-mcp
```

## Measured results
**Whole 3D chain vs cryoSPARC v5.0.6, same 38k particles, one RTX PRO 4000 each** (docs/benchmark-cryosparc-dz.md):
17.3 vs 30.1 min (NU refinement 2.8x, heterogeneous 4.9x faster; 3D classification 2.6x slower); cryoSPARC's NU map is
better, 4.76 vs 5.28 A with the same mask.

**NU refinement vs cryoSPARC NU-refine, same particles and start map, all maps scored by RELION `relion_postprocess` with
one mask recipe (`scripts/relion_fsc.py`), Refine3D defaults**: J341 (24k, C3) 3.41 vs 3.17 Å; J20 (157k, C3) 3.30 vs
3.22 Å; HCN1 (CryoPPP 10081, 39k, C4) 4.06 vs 3.70 Å; dz_2025_10 (38k, D2) 5.12 vs 4.83 Å; RELION tutorial β-gal 3.19 Å
vs RELION Refine3D + PostProcess 3.15 Å. Two changes got here: the rescue (all particles searched again over all views
and ±30 Å, best 16 hypotheses polished) cut the particles > 20° from cryoSPARC's pose 29 → 13 % (HCN1) and 9.6 → 4.4 %
(J341); the 3D jobs reconstruct from unmasked particle images (a box-edge window, the particle mask only for alignment),
which on HCN1 took a reconstruction at cryoSPARC's own poses from 4.16 to 3.74 Å (= cryoSPARC's) and our refinement
from 4.38 to 4.06 Å.

**RELION tutorial β-gal (Tutorial5.0 Extract/job018, 4,452 particles, 256 px, RTX 5080)** vs RELION Class2D (K=20, same GPU):
RELION 524 s, 16/20 populated classes, 76.7 % ψ-consistent with the 3D refinement, half-set FRC 7.75 Å;
celeris2d 26–40 s, 20/20, 79–81 %, 7.2–7.4 Å.

**LoG picks (New_processing Extract/job007, 10,328 picks, 64 px @ 5.53 Å)** vs curated particles from the
same micrographs (41.5 % of picks are real): single round F1 0.56–0.62; deep2D 0.67–0.69 (precision up to
70 %, recall up to 72 %); 4-seed mean P(good) best F1 0.69 (AP 0.665) in ~⅓ of deep2D's time.

**cryoSPARC template picks (3.17M, 64 px @ 5.28 Å, K=50, 3× RTX PRO 4000)**: cryoSPARC 31 min.
Celeris 3 min 49 s on three RTX PRO 4000s (8×), 87 % of particles in classes matching a cryoSPARC view
(band-limited correlation ≥ 0.6). ~25 % of J43 picks carry no consistent structure (cryoSPARC spreads them
evenly over all its classes); Celeris collects them in their own classes — check per-class confidence.

**Trimeric glycoprotein (cryoSPARC, 198,536 particles, 128 px @ 3.2 Å, K=50, 4× RTX A4500)**:
1 min 43 s – 2 min 15 s; clean trimer top/side views with flat solvent.

**RELION VDAM Class2D jobs (same particles, K and mask; Celeris on 4× RTX A4500;
RELION wall time = start note → last model file, so approximate)**

| dataset | particles | K | RELION | Celeris | notes |
|---|---|---|---|---|---|
| small protein A | 312,500 | 75 | 122.5 min | 2 min 51 s (43×) | small particle; both show noise-attractor classes (RELION honeycomb, Celeris starburst) |
| small protein B | 473,584 | 50 | 81.9 min | 2 min 54 s (28×) | small particle, mask 450 Å > box; Celeris shows particle + neighbours, junk split off |
| elongated glycoprotein ectodomain | 284,771 | 25 | 33.2 min | 1 min 13 s (27×) | RELION: 43 % in carbon-edge classes; Celeris: many distinct stalk/head views |
| glycoprotein complex | 404,314 | 50 | stopped at it 120 | 1 min 36 s | RELION job incomplete |

**cryoSPARC Class2D jobs on in-house data (K=50, Celeris defaults: circular mask only, adaptive full passes;
4× RTX A4500; cryoSPARC times from its job record, same machine class)**

| dataset | particles | cryoSPARC | Celeris | notes |
|---|---|---|---|---|
| negative stain, 64 px @ 6.4 Å | 314,050 | 5.0 min | 39 s | singles / pairs / triples separated as in cryoSPARC |
| cryo, 64 px @ 5.28 Å | 546,979 | 36 min (50 online + 5 full it) | 2 min 12 s | ice/carbon edges split off; compact particle views |
| cryo, 64 px @ 6.39 Å | 1,234,805 | 73 min (50 online + 10 full it) | 6 min 26 s | ~1/3 of picks are structureless: Celeris gathers them in 3 honeycomb-textured classes, cryoSPARC spreads them over ~30 faint classes |
| cryo, 320 px @ 1.32 Å (pre-cleaned) | 72,275 | — | 3 min 39 s | clean trimer views, 7.5 Å |

Lessons from these runs: shape-adaptive solvent flattening (formerly the default) grew honeycomb classes on
small particles, so it is now opt-in; on 0.5M+ particles two full passes left 30-40 % of particles still
changing class, so full passes continue until < 10 % change, but at most 3: scored against the cryoSPARC selections,
6 passes doubled the time without improving separation (most of the churn is between near-identical views).

Known weakness: small, round, low-information particles still form noise-attractor classes (starburst
texture at ~20 Å) — junk-like classes that do not always look like junk; use per-class confidence / seeds.

**Ab initio 3D, RELION β-gal tutorial (Extract/job018, 4,452 particles, 256 px @ 1.245 Å; 2D K=20 →
15 suggested classes, 3,559 particles; working box 218 px @ 1.46 Å; one RTX 5080)**

| run | time | half-map FSC 0.143 (no mask / sphere / solvent mask) | vs RELION Refine3D map (FSC 0.5 / 0.143) |
|---|---|---|---|
| D2 imposed | 1 min | 4.0 / 3.7 / 3.4 Å | 4.1 / 3.3 Å |
| C1 (detects D2, solves in D2, refines in C1) | 2.4 min | 6.5 / 4.2 / 3.9 Å | 5.0 / 3.6 Å |
| RELION InitialModel (C1, 4 GPUs) | 2 min 23 s | — | — |
| RELION Refine3D D2, all 4,452 particles, 256 px | — | 3.79 Å unmasked, 3.15 Å PostProcess | (reference) |

Without the particle SGD and symmetry relaxation the C1 map was self-consistent but distorted (vs RELION 0.5 at
15.9 Å) — half-map FSC alone does not catch a wrong basin, because both halves inherit the same class poses.
Simulated two-state β-gal (12,000 particles, one state lacking a domain of 5.4 % of the density; 2D classes do not
separate the states): two 3D classes put 99.5 % of particles in the right state (oracle 100 %).

## Validation scripts
`validate.py` (ψ consistency vs 3D angles) · `evaluate_frc.py` (star) / `evaluate_cs.py` (cs) half-set
FRC per class · `evaluate_selection.py`, `separation_eval.py` (precision/recall vs a curated set) ·
`compare.py` (side-by-side galleries) · `junk_eval.py` (good/junk separation of a cryoSPARC result against a
curated Select 2D job) · `compare_maps.py` (align a map to a reference — rotation, hand, translation — and their
FSC) · `abinitio_workflow.py` (2D → class selection → ab initio) · `simulate_heterogeneity.py` (two-state test
set).
