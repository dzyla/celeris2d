# INX-6 hemichannel in nanodiscs (EMPIAR-10291, CryoPPP's < 20 GB set) from micrographs (2026-10-30)

Data: 299 motion-corrected K2 micrographs (CryoPPP copy of EMPIAR-10291; single frames, so no polishing), C8,
deposited map EMD-9973 at 3.6 A (Burendei et al. 2020, RELION). One RTX PRO 4000 Blackwell (16 GB), WSL2.

**Pixel size.** The MRC headers say 0.616 A/px; the true pixel size is 1.232 A (K2 super-resolution header). At the
header value every defocus came out 4x too small and every refinement stalled near 10 A (the first CTF zero).
MotionCtf now warns on a median defocus < 0.6 um. Everything below is at 1.232 A.

## Chain and results (FSC 0.143, corrected solvent-masked / unmasked)

| step | particles | result | time |
|---|---|---|---|
| MotionCtf (bin 1, patch CTF) | 299 micrographs | defocus median 2.03 um | |
| blob picks (130 A) | 221,807 | | |
| Class2D K80, 180 A mask, 21 classes selected | 91,985 | | |
| InitialModel C8 | 82,786 (6,863 duplicates) | 8.41 A | 266 s |
| re-centred extraction, box 240 (296 A) | 82,775 | | 229 s |
| Refine3D | 74,497 used | 3.70 / 4.05 A | 2060 s |
| Refine3D NU (half-map cross-validation) | 74,497 | 3.61 / 4.05 A | 1978 s |
| Refine3D NU strict (folds inside each half) | 74,497 | 3.61 / 4.05 A | 2663 s |
| Refine3D NU strict, symmetry by accumulator resampling | 74,497 | 3.61 / 4.05 A | **610 s** |
| CtfRefine (median change 118 A) -> Refine3D NU | 74,494 | 3.65 / 4.05 A | 1973 s |
| HeteroRefine K3 (C8, local, from the NU map x 3) | 30,029 / 24,235 / 28,508 | 4.00 / 4.05 / 3.94 A | 1327 s |
| Refine3D NU, best class only | 25,657 | 3.84 / 4.41 A (pad 1) | 405 s |

Against the deposited map (scripts/compare_maps.py, EMD-9973 mask): masked map-vs-map FSC 0.5 at 3.9 A,
0.143 at 3.6 A for the NU maps (3.7 A unmasked) - the same structure at the published resolution.

## Findings

- **Non-uniform filter**: 3.70 -> 3.61 A. Cutoffs inside the mask at the end 3.7 / 4.4 / 6.1 A (5 / 50 / 95 %):
  the core at the global resolution, the periphery at ~6 A. The strict (gold-standard) version gives the same map.
- **3D classification** found no heterogeneity worth separating: three classes of similar size and resolution; the
  best one alone (a third of the particles) is worse than all of them together. Particle count is the limit here.
- **CtfRefine** did not help (3.61 -> 3.65 A): the patch-CTF defocus was already good.
- **Speed**: back-projection with C8 inserted every image 8 times (2.9 s per 1024 particles at box 240, ~90 % of
  an iteration); inserting once and resampling the accumulators with the symmetry operators (RELION's way) takes
  0.36 + 0.33 s - 220 s -> 62 s per iteration, same resolution (beta-gal D2: same masked and unmasked FSC).
- **Padding**: `auto_pad` fell back to pad 1 at box 256 on a 16 GB card (it required 60 % of the free memory; the
  measured peak is 73 %). Pad 1 kept the masked resolution but cost unmasked FSC (beta-gal 3.70 -> 3.93 A). Now 90 %
  of the free memory (the reserve is already in gpu_free); the INX-6 best-class refinement above ran at pad 1.

## What else was tried to go past 3.6 A (none moved it)

| tool | result |
|---|---|
| MapScore (other agent) on all 221,807 Class2D particles vs the 3.6 A map | proposes 31 classes / 115k particles: all 21 chosen by eye + 10 more (fit 46-81 %); InitialModel -> extraction -> NU: 91,597 used, 3.65 / 4.05 A |
| Class3D K4 at fixed poses (other agent), C8, 6 A | 30.7k / 15.5k / 15.1k / 21.5k at 3.84 / 3.94 / 4.29 / 3.94 A; NU on classes 1, 2, 4 (60,912 used): 3.65 / 4.05 A |
| HeteroRefine K3 (with alignment) | no useful classes (above) |
| optics fit (magnification, beam tilt, trefoil, 4th order) | magnification 1.0016 isotropic, no aberrations |
| Modes3D (other agent), C1, 8 A | one reproducible mode (halves cc 0.84 to 12.9 A): the nanodisc ring's size / density - outside the protein, which NU and the dynamic mask already down-weight |
| detector MTF (K2 300 kV) in Sharpen | B -221 -> -199 A^2, same resolution |

With B ~ -220 A^2 (Guinier), Rosenthal-Henderson predicts ~0.05 A from 23 % more particles - below one Fourier shell
here: the limit is the per-particle quality, and this set has no movies (no polishing).

## Next for INX-6

More particles (only 92k of 222k picks passed 2D selection; template picking from this map, a looser 2D selection
sorted in 3D), symmetry expansion and a local refinement of one protomer, per-particle scale / B.
