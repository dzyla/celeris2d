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

Particle tables: one `.npz` per micrograph in `data/NNN` while a job runs (append-only, cheap). The code currently
also consolidates them into one `particles.npz` per job at the end; that does not scale and has to be replaced
before 5,000-10,000-movie projects (millions of particles, live jobs that keep growing). A single file is rewritten
whole on every update and read whole for any subset. The replacement must be sharded and append-friendly
(shards of a fixed number of micrographs, or an indexed store), so reading by uid, micrograph or subset stays
cheap. It must be tested at that size with synthetic tables, not only on 20 movies. The same holds for
`micrographs.npz` and `picks.npz`.
An extract job folder is itself a particle input (`celeris2d -i jobs/004_...`, `Session.load`, MCP
`load_particles`): no star file needed; the mask defaults to 1.25 x the job's particle size.

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
