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
| **3D classification without alignment** (focused) | Class3D --skip_align / 3D Classification | **building** (Class3D) |
| **3D variability / flexibility** | multi-body / 3DVA, 3D Flex | **building** (Variability3D); 3D Flex and multi-body missing |
| **signal subtraction, symmetry expansion, local refinement** | Subtract, relion_particle_symmetry_expand / Particle Subtraction, Symmetry Expansion, Local Refinement | **building** |
| per-particle CTF, optics | CtfRefine (defocus, astigmatism, beam tilt, trefoil, 4th order, anisotropic mag) / Global + Local CTF | have per-particle defocus, global optics fit (validated synthetic; differs from RELION on beta-gal); per-particle astigmatism partial |
| Bayesian polishing / reference-based motion | yes | have (parametric frame weights; cryoSPARC fits free per-frequency weights) |
| masks, sharpening, local resolution | MaskCreate, PostProcess, LocalRes / Volume tools, Sharpen, Local Res + Local Filter | have; automatic mask ~20 % looser than RELION's, no detector MTF correction |
| per-particle curation by the map | (cryoSPARC: Remove duplicates, Reference-based curation) | have duplicates; **missing: per-particle fit-to-map score** (dz_2025_10: honeycomb picks 2.5 % vs 2D-selected 34 % above a phase-randomised control) |
| volume tools | align maps, flip hand, resample, map-vs-map FSC, EMDB fetch | **missing** (improvement-ideas #19) |
| orientation diagnostics | cFAR, angular distribution | have angular statistics; no cFAR / directional FSC |
| live processing | cryoSPARC Live | partial (pipeline streams movies -> picks -> 2D; no live 3D) |
| model building / validation | (external) | out of scope |

Notes from runs: noise-attractor ("honeycomb") 2D classes whose texture period follows the alignment band limit are
not particles - a per-particle score against the first map separates them where class-level scores fail.
