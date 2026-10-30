# Improvement ideas from running the whole pipeline (2026-10-30)

Collected while processing the RELION 5 beta-galactosidase tutorial from raw movies (docs/benchmark-bgal-movies.md)
and EMPIAR-10291 (INX-6 hemichannels in nanodiscs, CryoPPP's < 20 GB set) end to end. Ordered by what they would
have saved or gained on those runs. "Done" marks what is already in the code.

## Scientific correctness (wrong answers that looked plausible)

1. **Input metadata plausibility.** EMPIAR-10291's MRC headers say 0.616 A/px for 1.232 A K2 images. Every fitted
   defocus came out 4x too small (0.3-0.7 um), every refinement stalled at ~10 A (the first CTF zero), and nothing
   flagged it. Done: MotionCtf warns on a median defocus < 0.6 um or CTF fits at Nyquist. Next: compare the 2D
   class size / ab initio map size with the particle diameter the user gave; check the header pixel size against
   the detector's known pixel pitch and the magnification when the metadata has them; show the pixel size in
   every figure title.
2. **CTF estimation range and ambiguity.** The defocus search started at 3000 A; lower micrographs were fitted
   ~1500 A too high and the fit-resolution check did not notice (3.80 vs 3.72 A). Done: search from 1000 A. Next:
   report the score margin between the best and the second-best defocus (an ambiguity measure), warn when the best
   is at the search edge, and fit a phase shift when the user says a phase plate was used.
3. **TIFF orientation.** RELION reads TIFF movies y-flipped relative to the file; ours are the mirror of RELION's
   micrographs (coordinates, astigmatism angles, map hand). `MotionSettings.flip_y` exists, default off. Decide the
   default (match RELION?) and record the convention in every export.
4. **Ab initio robustness.** INX-6 from 7 class averages: a wrong structure that still reprojected at cc 0.91 for
   most classes (one class at 0.55, the top-view ring as something else). 9 classes worked. Next: warn when any
   class reprojects below ~0.7; use more classes (or particle SGD longer) by default for symmetric particles;
   compare a C1 run with the imposed group.
5. **Gold standard bookkeeping.** Done this session: halves kept through polishing / CTF refinement, refinements no
   longer drop 10 % more particles each round. Done: the non-uniform filter's strict version (default)
   cross-validates inside each half (two sub-half reconstructions per half at a reduced box); same INX-6 map.

## Resolution and map quality

6. **Non-uniform refinement** (Refine3D nu=True): neutral on rigid beta-gal, INX-6 at the true pixel size 3.70 ->
   3.61 A (published 3.6 A; at the wrong pixel size 5.6 -> 4.4 A). Done: per-iteration statistics, cutoff map
   (nu_cutoff.mrc) and figure (nu.png). Consider making it the default for membrane proteins.
   **3D classification without alignment** at fixed poses (cheap last cleanup) and FSC-validated curation (item 9)
   are next: HeteroRefine with alignment found no useful classes on INX-6 (docs/benchmark-inx6.md).
7. **Dynamic solvent mask during refinement** (done, default in Refine3D, HeteroRefine (per class) and InitialModel;
   mask = auto / static / none / a MaskCreate job): beta-gal 3.09 -> 2.95 A, unmasked 3.79 -> 3.66 A.
8. **Optics refinement**: optics.py fits anisotropic magnification, beam tilt / trefoil and 4th-order aberrations
   (validated on synthetic data). On beta-gal against our own map it found 0.1 % anisotropy where RELION reports
   0.57 %; understand that difference before applying it in refinements.
9. **Particle curation validated by the FSC** (CryoSieve-like): rank particles by their high-resolution agreement
   with their own half map, remove a share only if the FSC improves; the fixed 10 % drop is a heuristic.
10. **Picking recall**: blob picks found 89-90 % of RELION's refined beta-gal set; crowded INX-6 particles were often
    picked between neighbours. Template picking (another agent's work) from the first 3D map, then re-extraction.
11. **Refinement from an external reference** (Refine3D reference_map): from EMD-9973 low-passed to 30 A with three
    global searches it converged worse (9.5 A) than from our own ab initio (5.6 A). The global search at 12-14 A
    and the start schedule need work (more global iterations, a finer global step, or SGD from the reference).
12. **Sharpening**: the automatic mask is ~20 % looser than RELION's MaskCreate (beta-gal 3.25 vs 3.15 A
    reported from the same unmasked FSC); detector MTF correction (RELION's B -31 vs our -57 A^2 on beta-gal);
    a B-factor fit only over the linear part of the Guinier plot.
13. **Polishing**: per-frame weights are parametric (scale + B per frame); cryoSPARC fits free per-frame,
    per-frequency weights and the motion-prior strengths on held-out particles. Spatially correlated tracks.

## Speed and memory

14. Done this session: box 512 refines in 16 GB (automatic padding with gridding correction, lean transforms,
    references and batch arrays released), motion correction 5.7 -> 3.1 GB, polishing half maps 18.7 -> 2.6 GB, a
    hard per-process cap so the GPU never spills into system RAM. Next: store reference volumes as half-complex
    (Hermitian) grids (half the memory), fp16 references for projection, polishing's steady 10 GB.
15. Polishing's frame-weight pass takes 107 s for 24 micrographs (beta-gal): batch it. Done: symmetry by resampling
    the accumulators once per reconstruction instead of inserting every image at every operator (INX-6 C8: 220 -> 62 s
    per iteration); auto_pad no longer falls back to pad 1 at box 256 on a 16 GB card (pad 1 cost the unmasked FSC:
    3.70 -> 3.93 A); runs of particles are copied from the pinned host store asynchronously; Polish says clearly
    that single-frame micrographs cannot be polished.
16. Refinement from the reference job at full box from the first iteration: run the early iterations on a box
    sized to the current resolution (as ab initio does).

## Usability

17. Downloads of public sets: resumable, size-verified (EMPIAR's HTTPS mirror returned 199-byte error pages and
    truncated files; curl -C - then appends to them). A small `celeris2d fetch EMPIAR-xxxxx` with checksums.
18. Corrupt input data: micrograph 42 of EMPIAR-10291 holds values up to 1.6e34. Done: robust outlier
    replacement. Next: list such micrographs in the job report (and let Curate exclude them).
19. One command / dashboard flow for a benchmark against a deposited map: fetch the EMDB map, align (rotation, hand,
    translation), masked map-vs-map FSC, figure.
20. Job report: show the pixel size, the particle diameter and the box in Å on every 2D / 3D figure, and the
    warnings above at the top.
