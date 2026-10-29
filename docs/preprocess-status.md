# Movie preprocessing — status (2026-10-29)

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

The lower FRC at 3 Å on 4 movies is where cryoSPARC's local (patch) motion differs from one rigid shift per frame.

## Next

Patch (local) motion; per-particle / local defocus; a downstream check (extract the J104 particles from our
micrographs, reconstruct at the J104 poses, compare half-map FSC with cryoSPARC's micrographs); reading speed
(decode is 90 % of the time); picking and extraction; CLI / dashboard.
