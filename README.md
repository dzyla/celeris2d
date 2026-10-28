# Celeris2D — fast GPU 2D classification of cryo-EM particles

Maximum-likelihood 2D classification in PyTorch, built for speed on 0.2–3M+ particles: RELION
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
  * **shape-adaptive solvent flattening**: each class is masked by a soft envelope of its own density, so
    noise cannot build patterns around compact views (the former "honeycomb" classes);
  * fixed-frame artefact pixels (detector stripes) are found from the mean power spectrum and notched;
  * re-centring on density > 20 % of the class maximum.
* **Speed**: shifts are separable, so the pose search for a batch is two batched complex GEMMs, the
  back-projection the same GEMMs transposed (fp16 tensor cores, exact marginalisation). Particles are
  stored once as fp16 spectra (GPU if they fit, else pinned RAM) and streamed in random file order —
  classification starts after the first ~20k. GPUs pull batches from a shared queue.
* **Big data**: cryoSPARC-style online EM (500 particles / class / iteration) then full passes;
  ≥ 100k particles use hard pose + uniform class prior automatically.
* Options for clean, high-SNR data: `--march-margin 5-10` (align further), `--polish` (final pass aligns to
  the reconstruction limit with classes locked; sharper on β-gal, harmful on low-SNR data).

## Separation tools (all reuse the loaded data — no re-reading)
| tool | what it does | cost |
|---|---|---|
| multi-seed P(good) | same particles, several independent starts, classes matched by membership; per-particle P(good) and votes after a selection | seeds × one run |
| deep2D | round 0 defines a fixed standard (confidence / resolution / not-a-dot cut-offs + template bank); each round re-classifies good alone (purge) and bad + purged seeded with good templates (rescue); jumpers (≥ 3 good↔bad moves) set aside | ~10 runs |
| ISAC-style | generations of multi-seed runs; groups whose members keep landing together (soft posterior agreement) are accepted, the rest goes back to the pool; stops when a generation adds < 2 % | ~16 runs |

## Web dashboard
```bash
celeris2d-web --port 8765                  # on the GPU machine; open http://localhost:8765
ssh -L 8765:localhost:8765 <gpu-host>      # from your laptop, if the server runs remotely
```
Enter the particle file path (and any setting: mask, resolution, GPUs, subset, speed preset, every advanced
parameter), **Load** (particles stream to the GPU and stay there), **Run**. Click class tiles to select them
(green) — **All / None / Invert / Suggest** (Suggest is an automatic 2-cluster proposal; it always splits, so
review it). **Re-run selected / unselected** classifies just those particles (seconds, no reloading), and
**Export** writes good / ambiguous / bad particle sets (`.star` and/or `.cs`) at a P(good) threshold. Every
result is kept (chips at the top) and saved in the work directory.

## Settings optimizer (`--optimize N`, dashboard button, MCP `optimize_settings`)
Tries the alignment margin (0 / 4 shells) × solvent-mask threshold (0.05 / 0.12) on N random particles,
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
**RELION tutorial β-gal (Tutorial5.0 Extract/job018, 4,452 particles, 256 px, RTX 5080)** vs RELION Class2D (K=20, same GPU):
RELION 524 s, 16/20 populated classes, 76.7 % ψ-consistent with the 3D refinement, half-set FRC 7.75 Å;
celeris2d 26–40 s, 20/20, 79–81 %, 7.2–7.4 Å.

**LoG picks (New_processing Extract/job007, 10,328 picks, 64 px @ 5.53 Å)** vs curated particles from the
same micrographs (41.5 % of picks are real): single round F1 0.56–0.62; deep2D 0.67–0.69 (precision up to
70 %, recall up to 72 %); 4-seed mean P(good) best F1 0.69 (AP 0.665) in ~⅓ of deep2D's time.

**cryoSPARC template picks (3.17M, 64 px @ 5.28 Å, K=50, 3× RTX PRO 4000)**: cryoSPARC 31 min.
Celeris2D 3 min 49 s on three RTX PRO 4000s (8×), 87 % of particles in classes matching a cryoSPARC view
(band-limited correlation ≥ 0.6). ~25 % of J43 picks carry no consistent structure (cryoSPARC spreads them
evenly over all its classes); Celeris2D collects them in their own classes — check per-class confidence.

**Trimeric glycoprotein (cryoSPARC, 198,536 particles, 128 px @ 3.2 Å, K=50, 4× RTX A4500)**:
1 min 43 s – 2 min 15 s; clean trimer top/side views with flat solvent.

**RELION VDAM Class2D jobs (same particles, K and mask; Celeris2D on 4× RTX A4500;
RELION wall time = start note → last model file, so approximate)**

| dataset | particles | K | RELION | Celeris2D | notes |
|---|---|---|---|---|---|
| small protein A | 312,500 | 75 | 122.5 min | 2 min 51 s (43×) | small particle; both show noise-attractor classes (RELION honeycomb, Celeris2D starburst) |
| small protein B | 473,584 | 50 | 81.9 min | 2 min 54 s (28×) | small particle, mask 450 Å > box; Celeris2D shows particle + neighbours, junk split off |
| elongated glycoprotein ectodomain | 284,771 | 25 | 33.2 min | 1 min 13 s (27×) | RELION: 43 % in carbon-edge classes; Celeris2D: many distinct stalk/head views |
| glycoprotein complex | 404,314 | 50 | stopped at it 120 | 1 min 36 s | RELION job incomplete |

Known weakness: small, round, low-information particles still form noise-attractor classes (starburst
texture at ~20 Å) — junk-like classes that do not always look like junk; use per-class confidence / seeds.

## Validation scripts
`validate.py` (ψ consistency vs 3D angles) · `evaluate_frc.py` (star) / `evaluate_cs.py` (cs) half-set
FRC per class · `evaluate_selection.py`, `separation_eval.py` (precision/recall vs a curated set) ·
`compare.py` (side-by-side galleries).
