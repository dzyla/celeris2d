# What Celeris2D lacks against RELION 5 / cryoSPARC 4 (2026-10-30)

Single-particle SPA only (no tomography, no helical). "Have" = a job type or option in the code now; "building" =
an agent branch in progress; ordered by what a typical project needs first.

| area | RELION / cryoSPARC | Celeris2D |
|---|---|---|
| movies -> micrographs | MotionCor / Patch Motion, CTFFIND / Patch CTF | have (MotionCtf: patch motion, dose weighting, patch CTF) |
| micrograph curation | Manual/Curate Exposures | have (Curate: CTF/motion limits; region network map of ice/carbon) |
| picking | LoG, template, Topaz / blob, template, deep picker | have blob + template (2D classes or map projections); **missing: a trained picker (Topaz-like)**, manual picking UI |
| extraction, re-centring, re-boxing | yes | have |
| 2D classification + selection | yes | have (faster; selection proposals, stability, deep2D) |
| ab initio | SGD, multi-class | have (class stage + particle SGD, n_models) |
| homogeneous / NU refinement | Refine3D / Homo + NU | have (Refine3D, nu=True, dynamic mask, symmetry frame) |
| heterogeneous refinement | Class3D with alignment / Hetero Refine | have (HeteroRefine) |
| 3D classification without alignment (focused) | Class3D --skip_align / 3D Classification | have (Class3D: start from the main directions of variability, focus mask with on-the-fly subtraction, symmetry expansion) |
| **3D variability / flexibility** | multi-body / 3DVA, 3D Flex | **building** (Variability3D); 3D Flex and multi-body missing |
| signal subtraction, symmetry expansion, local refinement | Subtract, relion_particle_symmetry_expand / Particle Subtraction, Symmetry Expansion, Local Refinement | have (SymExpand: expanded poses, no images; Subtract with re-centring / re-boxing, per copy; LocalRefine: focus mask, angle / shift limits about the mask centre, on expanded or subtracted particles) |
| per-particle CTF, optics | CtfRefine (defocus, astigmatism, beam tilt, trefoil, 4th order, anisotropic mag) / Global + Local CTF | have per-particle defocus, global optics fit (validated synthetic; differs from RELION on beta-gal); per-particle astigmatism partial |
| Bayesian polishing / reference-based motion | yes | have (parametric frame weights; cryoSPARC fits free per-frequency weights) |
| masks, sharpening, local resolution | MaskCreate, PostProcess, LocalRes / Volume tools, Sharpen, Local Res + Local Filter | have; automatic mask ~20 % looser than RELION's, no detector MTF correction |
| per-particle curation by the map | (cryoSPARC: Remove duplicates, Reference-based curation) | have duplicates and MapScore (per-particle fit vs a phase-randomised control, per-2D-class shares and a class proposal) |
| volume tools | align maps, flip hand, resample, map-vs-map FSC, EMDB fetch | **missing** (improvement-ideas #19) |
| orientation diagnostics | cFAR, angular distribution | have angular statistics; no cFAR / directional FSC |
| live processing | cryoSPARC Live | partial (pipeline streams movies -> picks -> 2D; no live 3D) |
| model building / validation | (external) | out of scope |

Notes from runs: J20 (198k particles, whole stack, no 2D selection): ab initio with 4 models / 3D classification at
the consensus poses separates junk only partly (keeping the 2 best models: 80 % of clean particles kept, 56 % of
noisy ones removed - junk has no pose to agree on, so it spreads over all models); MapScore's class proposal 97 % /
81 %. HeteroRefine with "decoy" references (the first map with phases randomised beyond 30 A) on dz_2025_10's 153k
unselected particles: one class holds 76 % of the particles that fit the map (34 % fit in it) vs 51 % / 18 % with
three copies of the map - but the clean class was a decoy's in both decoy runs, not the real map's. Noise-attractor ("honeycomb") 2D classes whose texture period
follows the alignment band limit are not particles (dz_2025_10: 2.5 % fit the map vs 34 % of 2D-selected particles) -
a per-particle score against the first map separates them where class-level scores fail.
