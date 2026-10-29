# Ab initio 3D — status and open problems (2026-10-29)

Code: `celeris2d/abinitio.py` (algorithm), `celeris2d/plots3d.py` (figures), `celeris2d/web/abinitio3d.js`
(dashboard module), `Session.abinitio`, `tests/test_abinitio.py`, `scripts/abinitio_workflow.py` (2D → selection →
ab initio), `scripts/compare_maps.py` (map vs reference FSC after rotation/hand/translation search),
`scripts/simulate_heterogeneity.py`. README "Ab initio 3D" explains the pipeline.

## Measured (RELION β-gal tutorial, Extract/job018; 2D K=20 → 15 suggested classes, 3,559 particles;
## max-res 3 Å → 218 px @ 1.46 Å; one RTX 5080, 16 GB)

| run | wall time | half-map FSC 0.143 no mask / sphere / solvent | vs RELION Refine3D map, FSC 0.5 / 0.143 |
|---|---|---|---|
| D2 | 5 s classes + 4 s SGD + 31 s refine = 40 s | 4.0 / 3.6 / 3.3 Å | 3.9 / 3.0 Å |
| C1 (relaxed from D2) | 8 s + 7 s + 64 s = 1.3 min | — / 4.1 / — Å | 5.0 / 3.5 Å |
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
3. ~~J341 at 4 Å did not finish in 40 min~~ fixed: global searches used 1.2M orientations per particle for a
   300 Å particle; capped at 200k (step >= 4.2 deg). J341 C3 (21,899 particles, 210 px): 3 min, 6.7 A (sphere).
   J68 (816k picks, 2D K=50, 14 classes -> 250,559 particles, 62 px @ 5.45 A): 4.3 min to 12 A (Nyquist 10.9 A),
   3D peak 13.5 GB; the symmetry-relaxation trial runs on 25k particles above 30k (was 9.5 min on all).
5. **Narrow views.** J68's 14 good classes are one view rotated in plane (busiest 10 % of directions hold 82 % of
   particles): the map is smeared along z. The run now warns (`view_coverage` in summary.json, dashboard tile).
   The 2D class suggestion picked a carbon-edge class (confidence 0.99) on J68 — the 2D `suggest` needs a
   line/edge filter.
4. Not done: cryoSPARC `.cs` export of 3D poses (only RELION `.star`), MCP tool for ab initio, view rebalancing is
   implemented (`rebalance`) but not evaluated, J341 C3 comparison against cryoSPARC.
