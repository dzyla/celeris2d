# Celeris2D project design

Goals: everything a run produces is in one directory, organised so a person can find any output by looking (no
opaque IDs), any agent can find it through the catalog or the MCP tools, RELION and cryoSPARC can read it, and it
stays fast and tidy at 20,000 movies / 10 million particles.

## Layout

```
myproject/
  project.db                 catalog (SQLite, WAL): jobs, movies, per-job micrographs / picks / particle chunks
  project.json               name, created, Celeris2D version
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

Measured on 20 EMPIAR-10025 movies: 4.71 GB, of which 4.64 GB are micrographs and particle stacks (purgeable);
statistics, picks and tables are a few MB.

Rules:

1. **One job = one folder `jobs/NNN_<type>_<label>`**, NNN from one project-wide counter that is never reused.
   The label is generated from the parameters that tell runs apart (extract: box in A, pixel size, precision,
   selection; pick: blob size, threshold) and can be renamed: jobs refer to each other by number only. Running a
   step again with other settings makes a new job, so 256 px binned to 64 and the full box are two names side by
   side. A job is addressed by number (4, "004"), folder name or label.
2. **Every job folder has the same frame**: `job.json` (type, label, parameters, inputs, status running / finished
   / failed, Celeris2D version and git hash), `summary.json` (key numbers: what an agent reads first) and one
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
  `selected_classes.mrcs`, `rejected_classes.mrcs` and `selection.png` (kept framed green, rejected red). A Select
  job made by an agent says so in its label / note.
- **InitialModel** (ab initio from a Class2D job's classes and particles; symmetry is the user's choice): maps,
  half maps, figures, table `particles` = uid, rot, tilt, psi (RELION Euler), shifts, used, model.
- **Extract with `recenter=<job>`**: the aligned particles of a Class2D / InitialModel / Refine3D job cut at their
  aligned centre (centre in the extraction they were aligned on + shift), uid kept, pick position kept as dx_a /
  dy_a; particles whose new centres are closer than particle_a / 4 are merged (best maxprob / fit kept).
- **Refine3D** (particles, starting poses by uid, reference map): the map is resampled to the particles' pixel
  size and box and scaled to the data; shifts start at zero when the particles were re-centred on that job's
  shifts. Writes map.mrc (solvent-masked), map_nomask, mask, half maps, fsc.png, table `particles` = uid, rot, tilt,
  psi, shifts, fit, half. `global_iters` adds exhaustive orientation searches to the first iterations.
- **Polish** (on a Refine3D job): per-particle tracks and per-frame weights from the movies (method in
  `polish.py`'s docstring); a particle job like an extraction, same uids, centred at the refined centre.

EMPIAR-10025 (20 movies, all jobs from movies, one RTX 5090): motion/CTF/picks/64 px extraction 56 s; Class2D K40
45 s; Class2D K30 on the selection 18 s; ab initio 10 s (C1) / 38 s (D7); re-centred 288 px extraction 12-16 s;
Refine3D D7 3.29 A (3.24 A solvent mask, 10,113 particles, 288 s). Polish 164 s; the same refinement from the same
start on unpolished / polished particles: 3.32 / 3.24 A (solvent mask 3.13 / 3.05 A, unmasked 3.64 / 3.47 A).
C1: refined from the D7 solution 4.26 A; from the C1 ab initio model 6.3 A - that model is distorted (correlation
with its own D7 average 0.62 vs 0.90 for the D7 map), so a reliable C1 ab initio is still open.

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
- Celeris2D reads both itself, so a Class2D or InitialModel job can also start from a RELION or cryoSPARC job.

## Agents (MCP)

Tools: `project_open`, `import_movies`, `process_movies` and `extract_particles` (background; `job_status`),
`rethreshold_picks`, `job_report` (parameters, statistics with outliers, figures as images, drawn on demand),
`save_selection` (2D result -> Select job), `export_job`, `find_particle` (key or uid), `project_disk`,
`purge_job`, `rebuild_job`, `rename_job`, `delete_job`. An agent can
therefore run, check and decide the next step from numbers and figures, and a person sees the same folders.
