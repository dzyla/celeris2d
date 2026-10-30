# beta-galactosidase from raw movies: Celeris2D vs RELION 5 tutorial (2026-10-29)

Data: RELION 5 tutorial movies (24 K2 TIFF movies, 24 frames, 0.885 A/px, 200 kV, 1.277 e/A^2/frame, gain.mrc).
Celeris2D: commit 8a3ead3 + the changes listed under "Code changes", one RTX 5080 (16 GB), data on a local disk (WSL).
RELION: the tutorial project's own jobs (4 GPUs, 12 threads; times from job file timestamps).
Project: a local project directory (jobs/ = every job with its figures; analysis/ = this comparison).
a second project = the same first steps without `flip_y` (kept as the record of the y-mirror, below).
Driver scripts and logs are kept with the project (not in the repository).

## Resolution (FSC 0.143, first shell below; same rule for both programs)

| stage | Celeris2D | RELION |
|---|---|---|
| first refinement, corrected masked / unmasked | 3.25 / 3.93 A (job 014; 3,885 of 4,317 particles) | 3.12 / 3.93 A (job021; 4,452) |
| after polishing + per-particle CTF | 3.09 / 3.79 A (job 032) | 2.90 / 3.75 A (job030; CtfRefine + Polish) |
| same, dynamic solvent mask in the refinement | **2.95 / 3.66 A** (job 033) | (job029 refined with MaskCreate's mask) |
| map vs map, both masked with RELION's mask | FSC 0.5 at 3.9 A, 0.143 at 3.0 A (final vs job029) | - |
| poses, particles in both sets | 96 % within 2 deg, median 0.82 deg (3,107 particles) | - |

Unmasked curves coincide (analysis/fsc_*.png). The masked gap is the mask: Celeris2D's automatic solvent mask
encloses 1.55 M A^3, RELION's MaskCreate/job020 1.28 M A^3 (21 % more solvent noise in the masked FSC).

## Time (wall clock)

| step | Celeris2D (1 GPU) | RELION (4 GPUs) |
|---|---|---|
| motion + CTF + picking + extraction, 24 movies | 126 s (all control figures) | MotionCorr 371 s + CtfFind 5 s + Topaz 247 s + Extract 5 s |
| 2D classification | 57 s (K 50, 8,731 particles) | 323 s (K 100) |
| initial model | 32 s (D2) | 159 s + Class3D 810 s |
| re-centred extraction + refinement + sharpening | 10 + 75 + 3 s | 10 + 238 + 9 s (+ MaskCreate 15 s) |
| first sharpened map | **5.2 min, 3.25 A** | **37 min, 3.15 A** |
| polishing | 245 s | 339 s |
| per-particle CTF | 4 s (defocus only) | 134 s (3 CtfRefine jobs) |
| refinements after them | 80 + 87 s | 702 + 861 s |
| final map | **12.3 min, 2.95 A** (3.66 A unmasked) | **72 min, 2.92 A** (3.75 A unmasked) |

## Preprocessing vs RELION (analysis/compare_preprocessing.png, ctf.csv, motion.csv, picks.csv)

- defocus: median |difference| 5 A (max 41 A); astigmatism angle 2 deg (median); CTF fit resolution 3.34 vs 3.26 A.
- drift per movie 12.0 vs 12.1 A (median); per-micrograph offsets agree to 0.4 A once the frame of reference is
  matched (below).
- picking (blob 150 A): 90 % of RELION's refined particles among the candidates; the automatic threshold kept
  only 67 % (5,072 picks), threshold 0.15 keeps 89 % (8,731 picks; RELION's Topaz: 6,829).
- per-particle defocus vs RELION's CtfRefine changes for the same particles: r = 0.82 (2,945 particles).

## Scientific-integrity findings

1. **TIFF y-orientation (fixed by an option).** Celeris2D reads TIFF frames in file row order; RELION's micrographs
   of the same movies are their y-mirror (cc 0.32 vs -0.01; gain applied "as is" by RELION, "flip y" by Celeris2D).
   Without `MotionSettings.flip_y=True`: coordinates are mirrored relative to RELION's (pick recall 12 % as is,
   64 % flipped), astigmatism angles are mirrored (30 deg median difference -> 2 deg with flip_y), and a reference of
   known hand would not fit. De novo maps take an arbitrary hand either way. Default left unchanged (False) pending a
   decision: cryoSPARC agreement on EMPIAR-10025 suggests cryoSPARC reads TIFF as stored.
2. **Frame of reference of motion correction.** Celeris2D's shifts are relative to the frame average, RELION's to
   frame 1; particle coordinates differ by the mean drift (6.6 A median here). Not an error; matters for exporting
   to RELION polishing and for comparing coordinates.
3. **Corrected masked FSC (fixed).** Phases were randomised from the continuous radius >= k_r (half of shell k_r
   randomised) and the correction applied from k_r: a one-shell dip 0.97 -> 0.72 at 9.7 A. Now whole shells and the
   correction from k_r + 3 (as RELION); resolution unchanged (3.25 A).
4. **Particles lost at every refinement (fixed).** Each Refine3D took only the previous job's "used" particles and
   dropped another 10 %: 4,317 -> 3,885 -> 3,496 -> 3,146. Now every posed particle enters; each refinement drops its
   own 10 %.
5. **Refinement capped below Nyquist (fixed).** The 2D rule "max-res >= 2.2 x pixel" applied to Refine3D: box 238 of
   256, Nyquist 2.68 A. 3D jobs now go to Nyquist (`Params.nyquist_margin=1.0`): 3.12 -> 3.09 A.
6. Gold standard: halves kept through polishing and CTF refinement (each particle against its own half map);
   duplicates removed before 3D (583 at the initial model).
7. Not applied (RELION does): detector MTF correction (B -57 vs RELION -31 A^2 is partly that), beam tilt /
   higher-order aberrations, anisotropic magnification.

## Dynamic solvent mask (Refine3D mask="auto")

At fixed half maps a tighter mask is worth one shell (3.09 -> 3.03 A; volume 1.26 M A^3 = RELION's 1.28 M, no
density cut, randomised-phase FSC 0.01). Inside the refinement (mask from the current map every iteration, its
corrected FSC weighting the references, solvent flattened in them) the same start gives 2.95 A corrected and 3.66 A
unmasked (3.09 / 3.79 A with the particle sphere): the map itself improves, not only the number. Mask built on a
~2 A grid: 0.1-0.5 s per iteration of ~8 s.

## Code changes made during this benchmark

- `pipeline.MotionSettings.flip_y`; `figures` = N (first N micrographs drawn as processed + outliers); re-centred
  extractions write `recenter_shifts.png` / `.csv` + summary numbers.
- GPU memory: motion correction 5.7 -> 3.1 GB peak (patch path), 3.4 -> 2.4 GB (whole-frame), identical output;
  polishing half maps 18.7 -> 2.6 GB peak (band-limited volumes, projections identical to 2e-4); Refine3D
  accumulators freed after each solve, slab-wise solve, references released before back-projection; every job plans
  with 10 % (>= 1.5 GB) of the card left free and is hard-capped there (`engine.gpu_free` / `cap_memory`,
  `CELERIS2D_VRAM_GB` for a lower budget): peak 15.8 -> 14.4 GB on the card at box 256, no spill.
- `ctfrefine.py`: per-particle defocus refinement job (CtfRefine), 4 s for 4,317 particles; refined CTF carried by
  `Project.particles` and through polishing.
- Figures dark only by default (`CELERIS2D_THEMES=dark,light` for both); plotting bug (`T` shadowed) fixed; overview
  figure margins fixed.
- `scripts/compare_maps.py`: finer rotation refinement, sub-voxel translation, optional mask (7.4 -> 5.0 A unmasked,
  3.7 A masked for the same pair).
