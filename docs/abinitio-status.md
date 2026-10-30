# Ab initio 3D — status and open problems (2026-10-29)

Code: `celeris2d/abinitio.py` (algorithm), `celeris2d/plots3d.py` (figures)
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

0. **Symmetry is the user's choice only** (2026-10-29): no automatic detection. `sym` imposes a group;
   `relax_from` (C1 runs) solves with a given group and refines in C1 over its symmetry-related poses. The
   detector-driven relaxation and its gates described below are removed; the numbers stay as the record of why
   relaxation from the right group works (J341 C1 with relax_from=C3: 89-90 % within 10 deg).
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
   - relaxed refinement searched only around each pose, so a map solved with symmetry stayed symmetric (a forced
     C2 start still scored C2 0.97 after C1 refinement) and a pseudo-symmetric particle's extra feature never
     appeared. Now the first `relax_iters` (3) C1 iterations also search every symmetry-related copy of each pose
     (RELION's relaxed symmetry); a fit difference within 2 paired standard errors is a tie, decided by half-map
     resolution. J341 C1 seeds 0-2: 90 / 89 / 90 % within 10°, FSC 0.5 vs reference 8.9-9.3 Å (C3 imposed: 89 %,
     9.1 Å). Synthetic pseudo-C3: 17-68 % -> 77-89 % (direct C1 correctly kept); true C3 100 %; asymmetric 89 %.
   Open: J1279 (C3 by the user's refinement, 77k, strong top-view preference) in C1: the particles prefer the
   direct C1 map by 3 % (17x the paired noise) over the C3-relaxed one, though its side view is smeared (FSC 0.5 vs
   the C3 reference 22 Å; C3 imposed 11 Å). Either the particle is not strictly C3 or the missing side views leave
   C1 under-determined; trying the two best-scoring groups did not change it. Leads still open: independent half
   sets from the start; more 2D classes.
   Other structures (zyla-lab CS-project1, C1 NU-refines as reference; map vs a map reconstructed at cryoSPARC's
   poses, FSC 0.5 / 0.143; cryoSPARC's own ab initio poses agree with its refinement for only 8-14 % of particles,
   so maps are compared, not poses):
   - J56 (cryo, 102k, 64 px @ 5.28 Å): cryoSPARC ab initio J57 16.9 / 11.3 Å, 167 s; ours 21-22.5 / 10.6-10.9 Å,
     100-170 s. Our views collapse: busiest 10 % of directions hold 72-78 % of particles, 28 % occupied (cryoSPARC
     refinement: 59 % occupied). `rebalance` 1-2 does not change it (it reweights refinement only) and costs
     1-3 Å; on J1279 C1 mixed (18.4 / 28.2 Å vs 22-47 Å), J341 C1 10.5 vs 9.3 Å -> stays off.
     A map smeared around one axis scores all Cn alike (C12 0.38, C11 0.38, C10 0.37) and started a C12 trial
     (304 s): the low-score relaxation trigger now needs the group to stand out by 0.05 from unrelated groups
     (`_stands_out`) -> 100 s, same map. J341's blurred C3 (0.36 vs next unrelated 0.24) still triggers.
   - J85 (negative stain, 21k, 64 px @ 6.375 Å): cryoSPARC ab initio J84 27 / 19 Å. The contrast was decided only
     when a 2D run started, so ab initio from a restored 2D result used the stack as stored: dark protein, 3 of 4
     seeds without any agreement with the reference. Fixed (`Classifier2D.ensure_polarity`, called by ab initio):
     all 4 seeds now 45-58 / 27-31 Å, protein white. Still behind cryoSPARC; bounded by the weak NS 2D classes.
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
