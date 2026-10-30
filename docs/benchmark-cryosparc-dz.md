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
