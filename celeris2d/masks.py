"""Automatic soft masks from two half maps (Sharpen; usable for anything that needs a particle mask).

    auto_mask(h1, h2, apix) -> (mask (n^3, 0..1), info)

The half maps' average and half difference, both low-passed to lowpass_a: the difference holds only noise,
filtered exactly like the average, so its spread sets the threshold where the density stands clear of noise
(sigma_level x its standard deviation, never below min_frac of the maximum). Pieces much smaller than the particle
(noise, detached blobs: below keep_frac of the largest piece) are dropped; the rest is grown by dilate_a and given a
cosine edge of soft_a, both measured with a Euclidean distance (round outline, not the blocky one of repeated 3^3
dilations). A soft edge of about 6-8 px or more keeps the mask from correlating the half maps by itself.
"""
from __future__ import annotations

import numpy as np
import torch

from . import gpu


def auto_mask(h1, h2, apix: float, lowpass_a: float = 15.0, sigma_level: float = 5.0, min_frac: float = 0.05,
              keep_frac: float = 0.05, dilate_a: float = 3.0, soft_a: float = 0.0, dev=None) -> tuple[np.ndarray, dict]:
    """soft_a 0: max(8 A, 6 px). The same mask as the refinements' dynamic one (masking.auto_mask, which builds it on
    a coarse grid and also takes user regions / segments), with these defaults."""
    from .masking import auto_mask as _auto
    dev = dev or gpu.default()
    t1 = torch.as_tensor(np.asarray(h1), dtype=torch.float32, device=dev)
    t2 = torch.as_tensor(np.asarray(h2), dtype=torch.float32, device=dev)
    soft_a = soft_a or max(8.0, 6 * apix)
    info = {}
    m = _auto(0.5 * (t1 + t2), apix, lowpass_a=lowpass_a, k_sigma=sigma_level, min_frac=min_frac, extend_a=dilate_a,
              soft_a=soft_a, keep_frac=keep_frac, halves=(t1, t2), info=info)
    info.update(dilate_a=dilate_a, soft_a=round(soft_a, 2), volume_fraction=round(float(m.mean()), 4))
    return m.cpu().numpy().astype(np.float32), info
