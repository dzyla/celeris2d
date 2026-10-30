# Same particles, same GPU class: Celeris2D vs cryoSPARC v5.0.6 (2026-10-30)

Data: dz_2025_10 (in-house Krios, D2 complex), Celeris2D extraction 052: 38,464 particles, 256 px @ 1.32 A, from the
first 1,000 movies. cryoSPARC imported the same stacks and CTF from a RELION .star (`export` of job 052; import with
`blob/sign -1` - with +1 cryoSPARC refined an inverted map at 7 A, so that run was discarded and the export now writes
-1). One RTX PRO 4000 Blackwell (16 GB) per program on zyla-lab (cryoSPARC on GPU 3, Celeris2D on GPU 2), not run at the
same time; other GPU jobs of mine ran on other cards of the machine during both. Default settings in both except the
ones listed. cryoSPARC times: job record, started -> completed (excludes queueing). Celeris2D times: the job's own
`seconds` (each job reads its particles from the project's local scratch).

| step | settings | cryoSPARC | Celeris2D |
|---|---|---|---|
| 2D classification | K 50 | 139 s | 130 s |
| ab initio | 1 class, D2 | 206 s | 104 s |
| homogeneous refinement | D2 | 181 s | 139 s |
| non-uniform refinement | D2 | 449 s | 162 s |
| heterogeneous refinement | K 3, D2, from the homogeneous map | 466 s | 96 s |
| 3D classification | K 3, D2, 8 A, NU mask | **71 s** | 182 s |
| 3D variability / Modes3D | K 3, 8 A, C1, NU mask | 296 s | 224 s (includes the half-set check: 3 solves) |
| **total** | | **30.1 min** | **17.3 min** |

Resolution of the refinements, both measured the same way (`sharpen.postprocess`: phase-randomisation-corrected
masked FSC 0.143), from each program's unfiltered half maps:

| map | cryoSPARC's NU FSC mask | own automatic mask | FSC 0.5 (cryoSPARC mask) | unmasked |
|---|---|---|---|---|
| cryoSPARC homogeneous J26 | 5.04 A | 5.04 A | 6.90 A | 6.76 A |
| cryoSPARC NU J27 | **4.76 A** | 4.76 A | **5.54 A** | 5.54 A |
| Celeris2D Refine3D 076 | 5.28 A | 5.12 A | 6.63 A | 5.73 A |
| Celeris2D Refine3D NU 077 | 5.28 A | 5.04 A | 6.26 A | 5.73 A |

Shells are 0.25 A apart here (338 A box). Verdict:

- **Speed**: the chain takes 57 % of cryoSPARC's time. At 38k particles 2D and homogeneous refinement are close; ab
  initio is 2x, NU refinement 2.8x and heterogeneous refinement 4.9x faster. 3D classification is 2.6x **slower**.
  On large 2D jobs the gap is wider (README: 8-16x vs cryoSPARC at 0.3-3M particles).
- **Quality**: cryoSPARC's NU refinement is better - 4.76 vs 5.28 A with the same mask, 5.54 vs 6.26 A at FSC 0.5.
  Our plain refinement is better than cryoSPARC's at FSC 0.5 and unmasked, so the gap is in NU regularisation (and
  possibly per-particle scale / CTF terms cryoSPARC fits by default). Next to investigate.
- Not compared here: the quality of heterogeneous refinement, 3D classification and variability results (no ground
  truth on this set; Modes3D found no mode reproducible between half sets in C1).

Scripts: `csbench.py` (cryoSPARC chain through cryosparc-tools, project P2), `dz_bench.py` / `dz_bench_modes.py`
(Celeris2D), `bench_fsc.py` (the FSC table); cryoSPARC project in /scratch/cryosparc_bench on zyla-lab.

## Where the NU gap is (same day)

All numbers corrected masked FSC 0.143 with cryoSPARC's NU mask unless stated; one shell is ~0.25 A and run-to-run
noise of our NU refinement is about one shell (unmasked 5.73-6.03 A over near-identical runs).

- **Not masking**: same mask for both (table above), and cryoSPARC is also better unmasked.
- **Not reconstruction or particle weighting**: our back-projector at cryoSPARC NU's poses gives 4.83 A (all
  particles), 4.76 A on our 34,569 used particles, 4.76 A on the 30,606 where the two pose sets agree within 5 deg; at
  our poses 5.36 / 5.36 / 5.28 A. cryoSPARC's per-particle scale is not needed to get there.
- **Pose precision, not wrong poses**: median angular difference 0.93 deg; 80 % within 5 deg; the ~15 % beyond 10 deg are
  our worst fits (the 10 % we leave out differ by a median 39 deg), but the gap is the same on the agreeing particles.
- **Settings that did not help** (own mask; 077 defaults 5.04 A): drop_frac 0 5.28, nu_strict False 5.04 (5.20 with
  cryoSPARC's mask), both 5.20, global_iters 2 / 3 / 2 at 8 A 5.04 (the one global search moved 5 % to another view),
  min_step 0.25 deg 5.04, n_local 2 5.28, grid_correct 5.04.
- **Start vs procedure**: our NU refinement started from cryoSPARC NU's poses and map holds 4.83 A (FSC 0.5 5.83 -> 5.93
  A); started from cryoSPARC's ab initio (poses and map) it ends at 5.12 A, as from our own ab initio. So the refinement
  procedure converges to a worse optimum from an ab initio start - local search only after one coarse start; cryoSPARC
  searches all orientations every iteration as the resolution rises. Next: coarse-to-fine global search over the
  first iterations (narrowing to local), and a second global search for badly fitting particles before they are left
  out; then per-particle weighting.
