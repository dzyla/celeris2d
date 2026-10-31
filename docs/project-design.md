# Celeris project design

Goals: everything a run produces is in one directory, organised so a person can find any output by looking (no
opaque IDs), any agent can find it through the catalog or the MCP tools, RELION and cryoSPARC can read it, and it
stays fast and tidy at 20,000 movies / 10 million particles.

## Layout

```
myproject/
  project.db                 catalog (SQLite, WAL): jobs, movies, per-job micrographs / picks / particle chunks
  project.json               name, created, Celeris version
  JOBS.md                    generated: one line per job (folder, status, inputs, key numbers)
  jobs/                      one chronological list; each name says what the job holds
    001_import_session1/                 movies.npz
    002_motionctf_1.31apx/               micrographs.npz (CTF, motion), figures/ (outliers + first)
    003_pick_blob120/                    picks.npz, figures/
    004_extract_337A_1.31apx_f16/        particles.npz              [particles.star / .cs on request]
    005_extract_252A_5.26apx_f16/        particles.npz, class2d/ (2D sessions on it)
    006_select_good_of_k40/              particles.npz = a list of uids
    007_extract_337A_1.31apx_f16_sel006/ only the selected particles, full size
  data/                      heavy and per-micrograph, by job number; purgeable, rebuildable
    002/Micrographs/<session>/<stem>.mrc     002/Stats/...json
    003/Candidates/<session>/<stem>.npz
    004/Particles/<session>/<stem>.mrcs + .npz
  Trash/                     deleted jobs (folder + data), until emptied
```

Inside a job folder (jobs made before this layout keep their flat files and are read the same way, `Job.find`):

```
  014_refine3d_C1_on012/
    job.json  summary.json         the job's record (fixed names)
    j014_run.log                   time-stamped log of the run (start, progress, end or the error with traceback);
                                   the dashboard shows it and follows it while the job runs
    maps/      j014_map.mrc  j014_half1_unfil.mrc  j014_mask.mrc ...
    figures/   j014_fsc.png  j014_fsc.light.png  j014_views.png ...   (figures/movies/: per-micrograph figures)
    data/      j014_fsc.star  j014_fsc.xml  j014_fsc.json  j014_views.json  j014_particles.star ...
    particles/                     the sharded particle table (uid, pose, half, ...)
```

Every result file carries the job number (`j014_`), so a file copied out of the project, or listed by RELION next
to another job's, is never ambiguous (RELION identifies files by name). Code writes through `job.out(name)` (folder
by extension, prefix added) and reads through `job.find(name)`; job run functions are wrapped with `@logged`
(`project.run_log`), so each job type gets its run log without code of its own.

Measured on 20 EMPIAR-10025 movies: 4.71 GB, of which 4.64 GB are micrographs and particle stacks (purgeable);
statistics, picks and tables are a few MB.

Rules:

1. **One job = one folder `jobs/NNN_<type>_<label>`**, NNN from one project-wide counter that is never reused.
   The label is generated from the parameters that tell runs apart (extract: box in A, pixel size, precision,
   selection; pick: blob size, threshold) and can be renamed: jobs refer to each other by number only. Running a
   step again with other settings makes a new job, so 256 px binned to 64 and the full box are two names side by
   side. A job is addressed by number (4, "004"), folder name or label.
2. **Every job folder has the same frame**: `job.json` (type, label, parameters, inputs, status running / finished
   / failed, Celeris version and git hash), `summary.json` (key numbers: what an agent reads first) and one
   consolidated table under a fixed name per type (`micrographs.npz`, `picks.npz`, `particles.npz`).
3. **Heavy data lives in `data/NNN/`**, apart from the small results. `purge` deletes a job's micrographs or stacks
   and keeps everything else; `rebuild` makes them again from the recorded parameters (minutes, since the
   pipeline is fast). `disk` lists space per job, purgeable share and which jobs use it.
4. **Selections are lists of uids** (Select jobs), never copies of images: classify a binned extraction, save the
   good particles, extract only those at full size (`ExtractSettings.select`).
5. **Per-micrograph files keep the movie's name and its sub-directory** relative to the import root (sessions or
   grids with the same file names cannot collide; no folder of 20,000 files).
6. **Control figures on demand**: statistics for every micrograph, figures only for the first micrograph and the
   outliers (> 4 MAD) when a job finishes; any other is drawn when asked (`draw_figure`, MCP `job_report`): ~3 s per
   figure in a running process (movie read, motion + CTF again, drawing), 7-18 s for the first one in a new process
   (CUDA start-up, gain orientation check).
   Per-micrograph figures for everything would be ~2 MB per movie (45 MB on 20 movies vs 84 KB of statistics).
7. **Writes are atomic** (temporary file + rename) and a catalog row is written only after its files exist, so a
   crash leaves either a complete output or none; a job resumes where it stopped.
8. **Deleting moves a job and its data to `Trash/`**; it is refused while other jobs use the job.

## Particle identity

`key = "<movie stem>:<x A>:<y A>"` is readable and decodes on its own; `uid` (63 bits of SHA-256 of the movie
fingerprint and the position) is the fixed-size id used in tables and in cryoSPARC's `uid` field. The same
particle has the same uid in every extraction, so a 2D result on the binned extraction maps onto the unbinned one
by uid, without any re-indexing.

## Data flow

Stages talk only through the project: each reads its inputs (job folders named in its parameters), writes into
its own job, and skips movies it already has. The live runner keeps the current micrograph and picks in GPU
memory between stages, decodes the next movies in worker processes into shared memory, and writes files and
figures on background threads (EMPIAR-10025, 20 K2 super-resolution movies, one RTX 5090: 2.45 s per movie for
motion, CTF, picking and extraction, decoding included).

Particle tables (picks, particles, selections, per-particle results) are sharded (`celeris2d/tables.py`):
`jobs/NNN/<table>/index.json` plus `shard_00000.npz ...`, each shard holding the rows of 250 micrographs, numbers
only, uncompressed, never rewritten; micrographs that arrive while a job runs are staged as small files and merged
when 250 have arrived (or when the job finishes). Names, paths and CTF live once per micrograph (catalog) and are
joined when read; a particle's key is computed from its stem and pick position. Measured, 10,000 micrographs x
500 particles: 40 shards, 120 MB; cold read 0.12 s (NVMe); one micrograph's rows 1 ms; 5 M result rows written in
0.5 s. The same layout read over NFS on 1 GbE takes 1.6 s, against 7.3 s for one file per micrograph (and ~2 min
estimated on a local HDD, one seek per file). Particle stacks stay one `.mrcs` per micrograph (MBs each: the seek
is small next to the read).

## Particle jobs (2D -> 3D -> polishing)

`celeris2d/jobs.py` and `celeris2d/polish.py`; every result is a sharded table keyed by uid, shifts in A in the
engine's model convention (the particle's centre is at +shift from the box centre; RELION origin = -shift).

- **Class2D** (on an Extract, Polish or Select job): `classes.mrcs`, `gallery.png`, table `particles` = uid, cls
  (0-based), psi, shift_x_a, shift_y_a, maxprob; the 2D session (results restored by later jobs) in `session/`.
- **Select** from classes (`jobs.select_classes`, 1-based class numbers): table `particles` (kept) and table
  `rejected` (the other particles of that Class2D job), params `classes` / `rejected_classes`,
  `selected_classes.mrcs`, `rejected_classes.mrcs` and `kept.png` / `rejected.png` (the kept and the rejected class images, each on its own). A Select
  job made by an agent says so in its label / note.
- **InitialModel** (ab initio from a Class2D job's classes and particles; symmetry is the user's choice): maps,
  half maps, figures, table `particles` = uid, rot, tilt, psi (RELION Euler), shifts, used, model. With
  `n_models` > 1 the consensus is classified into that many 3D classes at its poses, and the classes are the result,
  laid out as a Class3D's (`classNN_map.mrc` / `_map_nomask` / `_mask` / `_half1_unfil` / `_angdist.bild`, class 1 =
  most particles, table column `cls`, summary `class_table`; `project.has_classes`), so the gallery, Select,
  Refine3D `classes=`, MaskCreate / Subtract `cls=` take them; the consensus stays as `consensus_map.mrc` /
  `consensus_map_nomask.mrc` / `consensus_mask.mrc` (summary `consensus`).
- **Every multi-class 3D job** (Class3D, HeteroRefine, InitialModel with classes; ModeMaps shares the class keys):
  summary `classes` (the number), `class_table` (per class: cls, particles, share, resolution_a,
  resolution_solvent_mask_a, map, ...; older jobs kept the table under `classes`, read both with
  `project.class_table`), `best_resolution_a`, `class_history` (per iteration: it, pops, share, res per class,
  res_limit_a when the iterations' box caps the resolution), also as `data/jNNN_class_history.json`, drawn in
  `class_history.png` (class distribution, per-class resolution, the final maps' values after the last iteration).
- **Extract with `recenter=<job>`**: the aligned particles of a Class2D / InitialModel / Refine3D job cut at their
  aligned centre (centre in the extraction they were aligned on + shift), uid kept, pick position kept as dx_a /
  dy_a; particles whose new centres are closer than the duplicate distance are merged (best maxprob / fit kept).
- **Refine3D** (particles, starting poses by uid, reference map): the map is resampled to the particles' pixel
  size and box and scaled to the data; the starting shift is the aligned centre (in the extraction the poses were
  found on) minus the centre in this extraction, so any pair of extractions works. Writes map.mrc
  (solvent-masked), map_nomask, mask, half maps, fsc.png, table `particles` = uid, rot, tilt, psi, shifts, fit,
  half. `global_iters` adds exhaustive orientation searches to the first iterations.
- **HeteroRefine** (heterogeneous refinement; particles, starting poses, K reference maps given as job numbers or
  `job:class`, a map may be repeated): each class starts from its reference low-passed to `res_start_a` (the first
  scaled to the particles at their poses, the others to its low-resolution Fourier amplitudes); with
  `global_search` every particle first searches all orientations in every class (references may come from
  different ab initio runs, in different frames) and joins the class it fits best; then `refine_particles` with K
  models refines the pose in every class each iteration and moves the particle to the best one, all classes
  compared at the same (the worst) resolution. Writes per class `classNN_map.mrc` / `_map_nomask` / `_mask` /
  `_half1_unfil` / `_half2_unfil` and `classNN_fsc.{json,star,xml}`; table `particles` adds `cls`. The user picks the
  classes: `select_classes` works on it like on a Class2D job, and Refine3D takes `classes=(k, ...)` with the
  HeteroRefine as poses and reference (the map of the particles' class). Controls on T20S (homogeneous): two copies
  of the refined map split 48 / 52 % with equal resolution (4.35 / 4.26 A) and views (SCF 0.84 / 0.87); the refined
  D7 map against the C1 ab initio map (a worse map of the same particle) 54 / 46 %, and the C1 class holds a narrow
  set of views (SCF 0.11) - a poor reference can keep the particles whose views it matches, so judge classes by
  their maps, FSC and angular distribution, not by size alone.
- **Class3D** (3D classification without alignment, RELION Class3D `--skip_align`; `celeris2d/class3d.py`, method in
  its docstring; particles, poses from a Refine3D / HeteroRefine / Class3D, optional reference = the consensus, by
  default the poses job's map): the poses stay fixed, K maps are fitted by EM over the class assignment only. Start
  (`init`): "variability" (default: residuals against the consensus, EM-PCA components (Punjani & Fleet 2021),
  k-means on the coefficients), "random" (random subsets, RELION) or "consensus" (consensus + noise per class).
  Hard M-step by default (posteriors still written), RELION's T (`reg_t`) regularises each class with its own
  half-map FSC. Focus `mask` (a MaskCreate job or .mrc): only the masked region is compared, and with `subtract` the
  consensus outside it is subtracted from every particle at its pose on the fly (off for particles subtracted
  already). `sym` imposes a symmetry, `sym_expand` classifies every symmetry copy of a particle separately (table
  column `copy`); both only as the user sets them, never detected. Iterations run on a small box (Nyquist just
  beyond `res_a`); final maps per class at the working box, a few classes at a time. Writes the HeteroRefine layout
  (`classNN_map` / `_map_nomask` / `_mask` / `_half1_unfil` / `_half2_unfil`, `classNN_fsc`) and table `particles`
  = uid, pose, `cls`, `fit`, `half`, `prob_1..K` (`copy` with expansion), so `select_classes`, Refine3D
  `classes=(k,)`, MaskCreate and Sharpen take it like a HeteroRefine. Figures: class map views, sizes and
  convergence per iteration with the posterior histogram, FSC. Simulated 30/70 mixture (3000 particles, known poses):
  0.98 of particles right at noise 3 (the true maps allow 0.98), 0.85 at noise 5 (true maps 0.90; random subsets
  0.59).
- **Modes3D** (`celeris2d/modes.py`; particles, a Refine3D / HeteroRefine's poses, optional MaskCreate mask):
  probabilistic PCA of the density at fixed poses and CTF (EM in Fourier space, batched GEMMs; method after Punjani &
  Fleet 2021), the mean and K modes masked and low-passed to the filter resolution. Each gold-standard half is solved
  on its own; per mode the matched halves' |cc| (>= 0.5: "reproducible") and the resolution to which they agree, to
  which the figures low-pass it. Outputs: `mean.mrc`, `mode_NN.mrc` (unit norm; sd in the summary), series frames +
  `series_modeK.cxc` (ChimeraX), table `particles` = uid, mode_1..K, scale; figures `what_moves.png` (-/+2 sd,
  difference, where), `projections.png` (differences along z, y, x on one scale, share of power on the particle),
  `frames.png`, `variance.png`, `latent.png` and a GIF per mode (`modeK.gif`, `modeK_noise.gif` when it does not
  reproduce). Synthetic moving domain |r| 0.98; dz_2025_10 (38k particles, 8 A): sd ~15 % of |mean| per mode is noise.
- **ModeMaps** (on a Modes3D job): `kind` = series (from the saved maps), bins (equal-size bins along one mode) or
  clusters (Gaussian mixture on the coordinates); bins / clusters are reconstructed with gold-standard halves and
  written as classes (`classNN_*`, table `particles` with cls, prob_k), so Select, Refine3D `classes=`, Sharpen and
  MaskCreate `cls=` take them (`project.CLASS_JOBS`). `map_views.png`: all groups at one contour level and each minus
  the mean.
- **Polish** (on a Refine3D job): per-particle tracks and per-frame weights from the movies (method in
  `polish.py`'s docstring); a particle job like an extraction, same uids, centred at the refined centre, and the
  table keeps each particle's half (its track was fitted against that half map).
- **Subtract** (`celeris2d/subtract.py`; particles, a refinement's poses and map, a MaskCreate mask of the region
  to keep): the map outside the mask projected at each particle's pose, with its CTF, scaled and subtracted (scale
  per shell from a 2,000-image subset, times a per-particle scale shrunk towards 1 by its noise, as RELION), the
  result re-centred on the mask's centre and cut to a smaller box (`box`); `subtract=False` only re-centres / re-boxes.
  A particle job like Polish: same uids and halves, table also holds the poses (shifts relative to the new box
  centre), so a Refine3D / LocalRefine on it can take its poses. **Symmetry expansion** (`sym_expand`, the user's
  point group, never detected) writes one image per particle and operator: rows are (uid, `copy`), poses A·S_k, every
  copy keeps the particle's half; jobs match particles by (uid, copy) (`jobs.match_rows`), and a refinement that
  reads expanded copies takes its poses from the Subtract job (copies are not separate particles: duplicate
  removal keeps or drops a particle with all its copies, selections stay uid lists). Distinct uids per copy were rejected: they would break the
  movie / particle lineage, Polish and merging back.
- **SymExpand** (`celeris2d/symexpand.py`; poses of a Refine3D / HeteroRefine / Class3D (`classes`) / LocalRefine,
  optional particles; `sym` the user's point group, required, C1 refused, never detected): no images, table
  `particles` = one row per (uid, `copy`) at pose A·S_k with the particle's shift (relative to its particle job's
  boxes) and half; `symexpand.png` viewing directions before / after. LocalRefine, Refine3D, Class3D and Subtract take
  it as poses (and as particles: its particles once): `jobs._posed_particles(expand=True)` gives one row per copy and
  each row's image index, refine_particles / classify fetch the image by index, so n images are loaded, not n x
  copies. Subtract on it subtracts per copy (the chain expansion -> subtraction; its own `sym_expand` goes through the
  same `expand_poses`, same images to 1e-3), Class3D on it = Class3D `sym_expand` (synthetic C2: 99.8 % of copies in
  the same class; GPU non-determinism). RELION export: one row per copy, the particle's rlnImageName, the expanded
  angles, origin = -shift, rlnRandomSubset = half + 1, plus celCopy (relion_particle_symmetry_expand's layout;
  dz_2025_10 D2: 153,832 rows, 3 s). HeteroRefine refuses it.
- **LocalRefine** (particles - usually a Subtract job -, starting poses, focus mask, optional reference): refinement
  in a box around the mask (rotations about the mask's centre: angle limit `max_angle_deg`, shift limit `max_shift_a`
  as hard bounds on the search, the shift step tied to the angular step by the region's radius), the reference and
  the gold-standard FSC restricted to the mask, halves kept from the input. Maps and poses are written back in the
  consensus frame and box; `focus_fsc.png` compares the region's corrected masked FSC with the consensus half maps'
  in the same mask (one randomisation shell for both, shells past a map's content left out). Synthetic two-domain
  test (domain B turned up to 25 deg about a hinge): subtraction keeps B (corr 0.95 vs 0.52 before, 0.05 with A),
  pose error of B 3.2 vs 8.4 deg, half-map resolution in the mask 6.7 vs 7.5 A. dz_2025_10 (38,459 particles,
  Refine3D D2 5.54 A; mask: the z > 0 half of the map, one RTX PRO 4000): subtracting the other half and re-boxing
  256 -> 176 px 24 s; local C2 refinement of the half 5.53 A, the consensus in the same mask 5.53 A (103 s); the same
  without subtraction 6.03 A (217 s); D2 expansion (153,836 images, 84 s) + local C1 refinement 5.05 A (335 s).
  The same half, one RTX 5090 (CELERIS2D_VRAM_GB=14): SymExpand D2 (6 s) + LocalRefine C1 on the original particles,
  nothing subtracted, box 256: 5.04 A, 172 s wall, 38,458 images loaded for 153,832 rows, GPU peak 12.3 GB, host 3.8
  GB; Subtract D2 (95 s, 8.9 GB of stacks) + LocalRefine C1 on the re-boxed 176 px copies: 5.05 A, 125 s wall, 153,832
  images (9.6 GB of spectra in pinned host RAM), GPU peak 3.2 GB, host 20.6 GB. Expansion without subtraction is as
  good here: the other subunits stay in the images but are outside the focus mask. (The consensus in the same mask
  reads 5.53-5.73 A: its randomisation shell is shared with the local maps'.) Synthetic C2 (two subunits, each turned
  up to 25 deg): SymExpand + LocalRefine on subunit 0, no subtraction: 6.4 vs 6.86 A in the mask, pose error 4.4 vs
  10.2 deg.
- **Sharpen** (on a Refine3D job or a HeteroRefine class; `celeris2d/sharpen.py`, RELION PostProcess; every Refine3D,
  HeteroRefine, Class3D and LocalRefine also sharpen their own half maps with their own mask when they finish -
  `sharpen` / `local_res` / `mtf`, on by default, files in the job, numbers under `sharpened` in its summary; a Sharpen
  job is for another mask or B-factor): corrected masked FSC (phases
  randomised beyond the shell where the unmasked FSC drops below 0.8, the correction applied from 2 shells further,
  as RELION does), automatic Guinier B-factor (10 A to the resolution), low-pass at the resolution. The sharpened
  map is made from the half maps filtered region by region (`nu_filter`, default; cryoSPARC sharpens its NU map the
  same way): with one global FSC weighting, domains below the global resolution were sharpened noise (noise pieces
  in the mask, J20 9.6 vs 5.8 %, HCN1 13.7 vs 3.4 %; cryoSPARC 10.6 / 11.2 %). FSC and B-factor stay on the
  unfiltered halves; `nu_filter=False` is RELION's global FSC weighting.
  Mask (`mask="auto"`, default; `masks.py`): the half maps' average and half difference low-passed to 15 A, the
  difference (noise only, filtered the same way) sets the threshold at 5 sd (at least 5 % of the maximum), pieces
  under 5 % of the largest dropped, +3 A, cosine edge max(8 A, 6 px) by Euclidean distance (round outline).
  `mask="refine"` uses the refinement's mask. Same numbers as the refinement's mask on T20S (3.10 A, B -107 vs
  -106) and the 2025_10 data (7.86 A, B -383 vs -382), smoother outline. Shells past the half maps' content (a
  refinement reconstructs to a little beyond its resolution) get no FSC: plotting them showed the mask leakage
  only, rising towards 0.2. `sharpened.mrc`, `sharpened_masked.mrc`,
  `postprocess.png` (four FSC curves, Guinier plot, central slices with the mask outline) and warnings: the mask
  correlating randomised maps (too tight), density beyond the mask's soft edge, a Guinier fit on few shells or
  r^2 < 0.8, single-shell drops of the unmasked FSC. T20S: 3.05 A, B -102 A^2 (r^2 0.93) polished; 3.10 A,
  B -106 A^2 unpolished; no warnings, 1-2 s.

**Diagnostics every 3D job writes** (`jobs.refinement_report`, `diagnostics.py`, figures in `plots3d.py`):
`fsc.png` and the curves as `fsc.json` (all curves), `fsc.star` (RELION `data_fsc` labels) and `fsc.xml` (EMDB
deposition format); `views.png` + `views.json` (angular distribution on an equal-area map, with the share of
directions occupied, the share in the busiest 10 % and the sampling compensation factor SCF of Baldwin & Lyumkis
2020: 1 = uniform Fourier sampling, towards 0 when directions of Fourier space are missing - symmetry copies count);
`progress.png` (resolution and alignment limit, angular step and share of particles that moved, per iteration; with
classes also the class sizes); `particles.png` (fit of used vs left-out particles, aligned centre from box centre,
orientation change from the start); `map_views.png` (projections and sections with the mask outline). HeteroRefine
adds `views_classes.png` (one map per class, shared scale). The summary repeats the numbers (`scf`,
`view_coverage`, `history`, `orientation_change_*`) and lists `warnings` (narrow views, uneven Fourier sampling,
start poses that did not fit), shown at the top of the job in the dashboard. InitialModel writes the same FSC and
views files, and `angdist.bild` (RELION's angular-distribution cylinders, per class `classNN_angdist.bild`; open
with the map in ChimeraX). Sharpen adds `fsc.{json,star,xml}`, `guinier.json` and, with `local_res` (default on), local
resolution (`localres.py`): FSC 0.143 of the two half maps in a Hann-windowed cube (~7x the resolution, >= 20 px)
around grid points in the mask, interpolated to every voxel -> `localres.mrc` (A), `localres_filtered.mrc` (the
B-sharpened map low-passed at every voxel to its local resolution), `localres.png`, `localres.json`. On synthetic
half maps with two regions of known resolution (2.69 / 5.44 A) the estimates are within 2-3 % with that window
(5-7 % too fine with a 16 px window). T20S (3.10 A, Nyquist 2.85 A): median 3.22 A, 5-95 % 2.85-6.61 A; core
at Nyquist in 24 % of the mask (flagged: the map is close to Nyquist). MotionCtf / Pick / Extract jobs write
`overview.png` (every micrograph's defocus, astigmatism, CTF fit, drift, local motion, CTF score / picks /
particles in processing order) and `micrographs.csv`; per-micrograph figures for every movie with
`figures="all"`.

**Figures** share one style (`plotstyle.py`): every figure is drawn in a dark (default) and a light theme,
`x.png` and `x.light.png`; the project page switches both. Colours come from the theme (validated categorical
slots, sequential blue ramp, blue/red diverging, fixed status colours), never literals.

**Job registry** (`jobs.registry()`, types in `jobspec.py`): one `JobSpec` per particle job type - title, settings
dataclass, inputs (with the job types each accepts), make / run functions, key numbers, main figures. Every front
end reads it: rerun, the project page's "New job" / "Change settings and run" form (`/api/project/jobtypes`,
`/api/project/new`; fields, defaults and help come from the settings dataclass and its comments), MCP `job_types`
/ `new_job`, the job list's key numbers and JOBS.md. A new job type is one entry plus its settings and functions;
`tests/test_registry.py` checks every entry (form round trip, help on every field, unknown / missing stored keys,
input types, key numbers) without running it. Stored parameters carry `settings_version`; a settings class with
`VERSION` > 1 has `migrate(params, from_version)` for renamed or re-meant fields (Sharpen 2: an empty mask used to
mean the refinement's mask, now "auto"). Adding a field needs no migration: unknown keys are ignored and missing
ones take their defaults.

**Gold-standard halves are never mixed.** A particle keeps the half of the refinement its poses come from
(Refine3D reads the poses job's `half` column; RELION's rlnRandomSubset), so a polished or re-refined particle is
never compared with a half map it was aligned against. **Duplicates never reach a 3D reconstruction**: InitialModel
(on 2D-aligned centres) and Refine3D (on the starting aligned centres) keep only the best-fitting particle of any
whose centres on one micrograph are closer than the duplicate distance, half the picking diameter (two distinct
particles cannot be that close; `jobs.duplicate_distance`), before the halves are split; the summary reports
`duplicates_removed`. The picker's own minimum distance (`min_dist_frac` x diameter, between picks; 0.5 by
default: T20S recall of J104's refined set 90.8 % from 20,901 picks; 0.4: 97.2 % / 33,681; 0.6: 76.0 % / 12,973;
the old picker, which allowed half its stated distance, 84.4 % / 17,008) lets two picks land on one elongated
particle; they converge after alignment and are removed there. T20S with these rules: InitialModel D7 removed
1,177 of 11,980 particles (10 %), 11.58 A as before; Refine3D D7 on the re-centred extraction (already merged)
removed 50, 3.29 A (3.10 A solvent mask) with the inherited halves vs 3.32 / 3.13 A with fresh random ones.

**Purge / rebuild / rename / delete** refuse a running job (a job records the process running it, so one left
"running" by a process that died is recognised as not running). Only MotionCtf and Extract jobs can be purged
(Polish stacks need the movies and the refinement again). Rebuild changes nothing before it runs: a purged job's
movie counts as done once its file exists again, and the purged flag is cleared when every movie is, so a failed
rebuild can simply be repeated. Extraction pixel sizes are exact (the source box is an even number of pixels:
2.0 A/px requested from 1.3 A/px at box 128 is 198 x 1.3 / 128 = 2.0109 A/px, stored as such).

EMPIAR-10025 (20 movies, all jobs from movies, one RTX 5090): motion/CTF/picks/64 px extraction 56 s; Class2D K40
45 s; Class2D K30 on the selection 18 s; ab initio 10 s (C1) / 38 s (D7); re-centred 288 px extraction 12-16 s;
Refine3D D7 3.29 A (3.24 A solvent mask, 10,113 particles, 288 s). Polish 164 s; the same refinement from the same
start on unpolished / polished particles: 3.32 / 3.24 A (solvent mask 3.13 / 3.05 A, unmasked 3.64 / 3.47 A).
C1: refined from the D7 solution 4.26 A; from the C1 ab initio model 6.3 A - that model is distorted (correlation
with its own D7 average 0.62 vs 0.90 for the D7 map), so a reliable C1 ab initio is still open.

## Reviewing a project

`celeris2d-web --project DIR` (or `celeris2d-web` started inside a project directory) serves `/project`: every job
with its type, label, status and key number (particles, resolution...), and per job its lineage and dependents,
results, figures, files and parameters (`/project#014` opens job 14). "Run again" makes a particle job (Class2D,
InitialModel, Refine3D, HeteroRefine, Polish, Sharpen) again as a new numbered job with the same parameters (`jobs.rerun`), so both stay
for comparison; for MotionCtf / Pick / Extract it carries on with movies the job does not have yet.

## Compatibility (on request)

RELION / cryoSPARC files are not written by default. An extract job's `export` option (`relion`, `cryosparc`)
writes them when it finishes; `export.export(project, job, fmt)` (MCP `export_job`) for any job. They go into the
job's own folder; their paths point into `data/`, relative to the project. Checked: RELION 5.1 `relion_preprocess`
extracts all particles from the exported micrographs_ctf.star + autopick.star (float16 micrographs), within 1 px
of ours (RELION cuts at whole pixels; we shift sub-pixel).

- RELION: run `relion` in the project directory; every star path is relative to it. `movies.star`,
  `corrected_micrographs.star`, `micrographs_ctf.star`, `autopick.star` and `particles.star` are the inputs
  RELION's Import / CtfFind / AutoPick / Extract jobs produce, so any RELION job can take them. (Bayesian polishing
  needs RELION's per-micrograph motion metadata: not written yet.)
- cryoSPARC: `particles.cs` with blob, CTF and location fields (Import Particles takes it, or `particles.star`).
- Celeris reads both itself, so a Class2D or InitialModel job can also start from a RELION or cryoSPARC job.

## Agents (MCP)

Tools: `project_open`, `import_movies`, `process_movies` and `extract_particles` (background; `job_status`),
`rethreshold_picks`, `job_report` (parameters, statistics with outliers, figures as images, drawn on demand),
`save_selection` (2D result -> Select job), `export_job`, `find_particle` (key or uid), `project_disk`,
`purge_job`, `rebuild_job`, `rename_job`, `delete_job`. An agent can
therefore run, check and decide the next step from numbers and figures, and a person sees the same folders.
