# Movie preprocessing — status (2026-10-29)

Control figures: `celeris2d/plots_preprocess.py` (per movie: motion, local motion model vs patch measurements,
power spectrum data | fit, equal-phase CTF profile, ring-by-ring fit significance, even/odd-frame FRC, dose
weights; summary of all movies vs cryoSPARC; FSC of reconstructions by micrograph set). `preprocess_eval.py`
writes them for every run.

Code: `celeris2d/preprocess.py`, tests `tests/test_preprocess.py`, evaluation `scripts/preprocess_eval.py`.

## First pass ("easy pass")

Gain (orientation chosen by the flattest frame sum), hot pixels, Fourier binning of super-resolution frames,
whole-frame (rigid) alignment of each frame to the sum of the others on a ~4 Å copy, dose weighting
(Grant & Grigorieff 2015), per-micrograph CTF: background + envelope x CTF^2 fitted by least squares for every
candidate (1D defocus scan, astigmatism grid, local refinement); fit resolution = how far the fitted CTF correlates
significantly with the Thon rings (ring by ring, z > 3).

## Measured: EMPIAR-10025 subset (20 K2 super-resolution movies, 38 frames, 0.6575 Å/px, 53 e/Å²; RTX 5090)

Reference: the same movies processed by cryoSPARC (patch motion, patch CTF; micrographs in `mrc/`, per-particle
defocus of NU-refine J104).

| | result |
|---|---|
| mean defocus vs cryoSPARC (median of the particles on each micrograph, 19 movies) | median 11 Å, max 43 Å, mean -10 Å |
| astigmatism vs cryoSPARC | magnitude 16 Å, angle 3° (median; astigmatism ~180 Å) |
| Thon-ring fit resolution, ours (dose-weighted) vs cryoSPARC's micrograph, same fitter | median 0.00 Å (-0.27..+0.07), 3.1-4.3 Å |
| FRC of our dose-weighted sum vs cryoSPARC's micrograph | 0.87-0.97 at 10 Å; at 3 Å median 0.80, 0.37-0.86 |
| time per movie | read 3.8 s (LZW TIFF, 16 threads), motion 0.4 s, CTF 0.9 s |

The FRC with cryoSPARC's micrograph shares the shot noise of the same frames: it measures how close our shifts
are to cryoSPARC's, not which are better (patch motion lowers it to 0.26 at 4.7 Å while improving the image).

## Local (patch) motion — default since 2026-10-29

5x5 patches of the globally aligned frames, each aligned like a whole frame (leave-one-out, ~2.6 Å, B 500), a
smooth model (quadratic in x, y times cubic in time, robust least squares; fit rms ~0.4-0.5 Å), sums from
overlapping 512-px tiles that each follow the model. +0.2 s per movie (0.6 s motion).

Downstream test (`scripts/extract_eval.py`, then `abinitio_truth.py ref`): J104's 10,000 particles cut identically
from each micrograph set, reconstructed at J104's refined poses (D7), half-map FSC 0.143:

| micrographs | as extracted | positions registered to cryoSPARC's (smooth field per micrograph) |
|---|---|---|
| cryoSPARC patch motion (its own frame of reference) | 3.43 Å | — |
| ours, whole-frame | 3.51 Å | 3.47 Å |
| ours, patch 5x5 | 3.68 Å | **3.27 Å** (curve at or slightly above cryoSPARC's from 4 to 3.3 Å) |

The two local-motion models place the same particle ~1.1 Å apart (rms, smooth over the micrograph), and J104's
shifts belong to cryoSPARC's micrographs: without registration that alone costs 0.25 Å. Per-particle registration
is too noisy (~1 Å per particle at 10 Å) and made it worse (rigid 3.59 Å). Particles picked on our own
micrographs have no such offset. The even/odd-frame FRC agrees (patch > rigid on every movie checked) but is
small; the Thon-ring fit resolution cannot tell the methods apart.

Synthetic check (tests): a smooth local field of 4 Å drops the 6-3.5 Å even/odd FRC by >0.1 with whole-frame
alignment; the patch model brings it back to the no-motion level.

## Next

Picking (blob, then templates from 2D classes, then a classifier trained on manual picks with an immediate 2D
check: EMAN-boxer-like loop in the dashboard), extraction into memory, a 2D engine that accepts particles while
running, a live runner (`celeris2d-live <movie dir>`); per-particle defocus; reading speed (TIFF decode is
~75 % of the time).
