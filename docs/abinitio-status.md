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

1. **C1 robustness (main one).** Measured against cryoSPARC NU-refine poses (`scripts/abinitio_truth.py`: share
   of particles within 10° after the best global rotation/hand/symmetry operator, and map FSC vs a map
   reconstructed at the cryoSPARC poses). J341 (C3, 21k particles, 8 Å), 3 seeds:
   - hard-pose SGD locked in wrong starts: C3 58–85 % within 10°, C1 39–58 % (map vs reference FSC 0.5 at 26–41 Å).
     More rounds made it worse (90 × 1000: 15–31 %). Soft poses (`sgd_topk=8`, posterior-weighted, noise from the
     batch power spectrum): C3 89 % (FSC 0.5 9.0 vs 11.1 Å); direct C1 (no relaxation) 47–72 %.
   - the relaxation gate compared each map's own fit at its own resolution, so a blurred wrong C1 map won (0.282 vs
     0.199 at 17.8 vs 8.7 Å). Now `cross_fit`: each particle against the half map it was not in, both maps over the
     same band. J341 C1: relaxed map chosen in 3/3 seeds, 76–90 % within 10°, FSC 0.5 vs reference 9–12 Å.
     Relaxation is also tried when the best group scores >= 0.3 (a blurred C1 map scored C3 only 0.36).
   - synthetic (tests' objects, 12 Å): true C3 100 % (relaxed kept), asymmetric 89 % (false C2 rejected by 3.9 %),
     pseudo-C3 (C3 + one extra blob) 68 %, direct C1 kept.
   Open: J1279 (C3, 77k, strong top-view preference) C1 — the blurred C1 map suggests C2, the C2-relaxed map loses,
   C3 is never tried; direct C1 keeps a smeared side view (FSC 0.5 vs reference 22–47 Å, C3 run 11 Å). Trying the
   two best-scoring groups would cover it. "Relaxed map stays symmetric after C1 refinement" is no test: a forced
   C2 start stayed C2 (0.97). Leads still open: independent half sets from the start; more 2D classes.
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
