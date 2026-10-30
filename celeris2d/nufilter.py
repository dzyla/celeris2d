"""Spatially adaptive regularisation of refinement references (the idea of cryoSPARC's non-uniform refinement,
Punjani, Zhang & Fleet, Nature Methods 2020).

A uniform refinement filters each reference by one global FSC curve: an ordered core and a disordered region (a
detergent belt, a flexible loop) get the same filter, so the reference keeps the disordered region's noise and the
alignment suffers. Here every region of a half map gets its own low-pass, chosen by cross-validation: for a set of
candidate low-pass filters, the error between the filtered map and an independent map of the same particle, averaged
over a Gaussian window around each voxel, estimates the local error against the true structure (the independent
map's noise only adds a constant); each voxel takes the candidate with the smallest local error.

adaptive_filter(h, h_ref, ...): h filtered locally, validated against h_ref (the other half map, or - gold standard
inside the half set - a reconstruction from the other particles of h's own half). Returns the map and the per-voxel
cutoff (A). Everything on the GPU: ~2 x n_cut 3D FFTs of the map's size per call.
"""
from __future__ import annotations

import math

import torch


def _kk(n: int, dev) -> torch.Tensor:
    """|k| (cycles per box) on the rfftn grid of an n^3 map."""
    k = torch.fft.fftfreq(n, device=dev) * n
    kz = torch.fft.rfftfreq(n, device=dev) * n
    return torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)


def _smooth(x: torch.Tensor, sigma_px: float, kk: torch.Tensor) -> torch.Tensor:
    """Gaussian window (sigma in voxels) by FFT."""
    n = x.shape[0]
    g = torch.exp(-2 * (math.pi * sigma_px * kk / n) ** 2)
    return torch.fft.irfftn(torch.fft.rfftn(x) * g, s=x.shape)


def adaptive_filter(h: torch.Tensor, h_ref: torch.Tensor, apix: float, res_a: float, mask: torch.Tensor | None = None,
                    n_cut: int = 12, lo_a: float = 20.0, window_a: float | None = None, edge: float = 0.1,
                    info: dict | None = None) -> torch.Tensor:
    """h filtered with a cutoff chosen per voxel: candidates log-spaced from lo_a to 1.1 x... just beyond res_a (the
    global resolution; a region can be sharper than the average, so the finest candidate is 0.85 x res_a), the
    local error |filtered h - h_ref|^2 averaged over a Gaussian of sigma window_a (default 2 x res_a, at least 8 A).
    mask: only for the summary (cutoffs inside it). Returns the filtered map; info gets the cutoff map (A) and its
    distribution."""
    n = h.shape[0]
    dev = h.device
    box_a = n * apix
    kk = _kk(n, dev)
    Fh = torch.fft.rfftn(h.float())
    hr = h_ref.float()
    k_hi = min(n / 2 - 1, box_a / (0.85 * res_a))
    k_lo = max(2.0, box_a / lo_a)
    cuts = torch.exp(torch.linspace(math.log(k_lo), math.log(max(k_hi, k_lo * 1.01)), n_cut)).tolist()
    sig = max(window_a or 2.0 * res_a, 8.0) / apix
    best_err = best = best_k = None
    for kc in cuts:
        w = torch.clamp((kk - kc * (1 - edge / 2)) / (kc * edge), 0, 1)
        f = torch.fft.irfftn(Fh * (0.5 * (1 + torch.cos(math.pi * w))), s=h.shape)
        err = _smooth((f - hr) ** 2, sig, kk)
        if best is None:
            best_err, best, best_k = err, f, torch.full_like(f, kc)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best = torch.where(take, f, best)
            best_k = torch.where(take, torch.full_like(f, kc), best_k)
        del f, err
    if info is not None:
        res_map = box_a / best_k
        sel = res_map[mask > 0.5] if mask is not None and bool((mask > 0.5).any()) else res_map.reshape(-1)
        q = torch.quantile(sel[torch.randperm(len(sel), device=dev)[:1_000_000]], torch.tensor([0.05, 0.5, 0.95], device=dev))
        info.update(cutoff_map_a=res_map, cutoff_p05_a=float(q[0]), cutoff_median_a=float(q[1]), cutoff_p95_a=float(q[2]),
                    candidates_a=[round(box_a / k, 2) for k in cuts], window_a=round(sig * apix, 1))
    return best
