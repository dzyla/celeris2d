# Ab initio 3D — status and open problems (2026-10-29)

Code: `celeris2d/abinitio.py` (algorithm), `celeris2d/plots3d.py` (figures), `celeris2d/web/abinitio3d.js`
(dashboard module), `Session.abinitio`, `tests/test_abinitio.py`, `scripts/abinitio_workflow.py` (2D → selection →
ab initio), `scripts/compare_maps.py` (map vs reference FSC after rotation/hand/translation search),
`scripts/simulate_heterogeneity.py`. README "Ab initio 3D" explains the pipeline.

## Measured (RELION β-gal tutorial, Extract/job018; 2D K=20 → 15 suggested classes, 3,559 particles;
## max-res 3 Å → 218 px @ 1.46 Å; one RTX 5080, 16 GB)

| run | wall time | half-map FSC 0.143 no mask / sphere / solvent | vs RELION Refine3D map, FSC 0.5 / 0.143 |
|---|---|---|---|
| D2 | 4 s classes + 21 s SGD + 61 s refine = 1.4 min | 4.0 / 3.6 / 3.35 Å | 4.0 / 3.1 Å |
| C1 (relaxed from D2) | 8 s + 31 s + 100 s = 2.3 min | 6.5 / 4.2 / 3.9 Å | 5.0 / 3.6 Å |
| C1 direct (no relaxation) | ~1 min | 7.6 / 4.6 / 4.1 Å | 10.0 / 5.0 Å |
| RELION InitialModel + Refine3D (D2), 4 GPUs | 2 min 23 s + 3 min 47 s | 3.79 Å unmasked, 3.15 Å PostProcess | — |

Peak GPU memory 10.8 GB (D2) / 13.4 GB (C1 with relaxation). Simulated two-state β-gal (12k particles, one state
lacking 5.4 % of the density): 2 3D classes, 99.5 % correct.

## Open problems

1. **C1 robustness (main one).** A direct C1 run lands in a slightly wrong basin for β-gal (self-consistent: the
   half-map FSC says 4.6 Å, but vs RELION only 10 Å at FSC 0.5) — both half sets inherit the same class poses, so
   half-map FSC cannot see it. Fixed for symmetric particles by relaxation (solve in the detected group, refine
   in C1), gated by particle fit: keep the relaxed map if it fits better, or within 1 % at a better half-map
   resolution. The gate is a heuristic tuned on two cases (β-gal: relaxed fits 0.5 % worse but is right; an
   asymmetric synthetic forced into C2 fits 2.7 % worse and is wrong). Not tested on a truly asymmetric real
   particle. Leads:
   - independent half sets from the start (separate class stage / SGD per half) so FSC can expose a wrong basin;
   - compare candidate maps by log-likelihood at matched resolution instead of mean normalised cc;
   - more 2D classes (K 50–100) for the class stage — 15 averages under-determine a C1 structure;
   - symmetry detector is advisory: smooth blobby maps score 0.4–0.6 for wrong groups (see test_symmetry_detection).
2. **Speed.** For 3.5k particles we are ~4× faster than RELION on a quarter of the GPUs, not 20×. Most time is the
   refinement at the full 218 px box (61–100 s) and SGD (21–34 s). Leads: run early refinement iterations on a
   smaller box (Nyquist ~2× the current resolution; `_fetch` already crops), scale SGD rounds/batch with the
   number of particles (30 × 2000 is more than the whole set here), skip the candidate-start check when SGD is on.
3. **J341 at 4 Å did not finish in 40 min** (C1 with relaxation, ~20k particles, ~212 px box, 36 % GPU utilisation
   while running) — something is CPU/transfer bound at that size; profile before scaling up. The 8 Å run
   (16.5k particles, 108 px) took 23 s.
4. Not done: cryoSPARC `.cs` export of 3D poses (only RELION `.star`), MCP tool for ab initio, view rebalancing is
   implemented (`rebalance`) but not evaluated, J341 C3 comparison against cryoSPARC.
