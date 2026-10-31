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


def _candidate(kk: torch.Tensor, kc: float, edge: float, shape: torch.Tensor | None, k_res: float,
               order: int = 0) -> torch.Tensor:
    """A candidate filter on grid kk: a hard low-pass at kc (raised-cosine edge), or with shape (the global per-shell
    FSC weight, shell = cycles per box) that curve stretched so the global resolution k_res lands on kc. order > 0: a
    Butterworth low-pass of this order at kc instead (cryoSPARC NU's filter, order 8), times the global FSC weight
    (not stretched) when shape is given."""
    if order > 0:
        w = 1 / torch.sqrt(1 + (kk / kc) ** (2 * order))
        if shape is not None:
            i0 = kk.floor().long().clamp(max=len(shape) - 1)
            w = w * torch.where(kk < len(shape) - 1, shape[i0], torch.zeros_like(kk))
        return w
    if shape is None:
        w = torch.clamp((kk - kc * (1 - edge / 2)) / (kc * edge), 0, 1)
        return 0.5 * (1 + torch.cos(math.pi * w))
    x = kk * (k_res / kc)
    i0 = x.floor().long().clamp(max=len(shape) - 1)
    f = x - i0
    w = shape[i0] * (1 - f) + shape[(i0 + 1).clamp(max=len(shape) - 1)] * f
    return torch.where(x < len(shape) - 1, w, torch.zeros_like(w))


def _kk(n: int, dev) -> torch.Tensor:
    """|k| (cycles per box) on the rfftn grid of an n^3 map."""
    k = torch.fft.fftfreq(n, device=dev) * n
    kz = torch.fft.rfftfreq(n, device=dev) * n
    return torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)


def _sigma(window_a: float | None, awf: float, res_a: float, kc: float, box_a: float, apix: float) -> float:
    """Gaussian window (voxels) for the local error of candidate kc: awf > 0 scales it with the candidate's resolution
    (sigma = awf / 2 x its resolution, cryoSPARC NU's adaptive window factor: coarse candidates judged over larger
    regions), else one window for all, window_a or 2 x the global resolution (at least 8 A)."""
    if awf > 0:
        return max(awf * 0.5 * box_a / kc, 4.0) / apix
    return max(window_a or 2.0 * res_a, 8.0) / apix


def _snr_shift(snr: torch.Tensor, kc: float, ratio: float, k_max: float) -> float:
    """The shell (cycles per box) beyond kc where snr (per shell, made non-increasing) has fallen to snr(kc) / ratio."""
    s = torch.cummin(snr.float().clamp_min(0).nan_to_num(0), 0).values.cpu()
    k = torch.arange(len(s), dtype=torch.float32)
    at = float(torch.from_numpy(__import__("numpy").interp([kc], k.numpy(), s.numpy())))
    target = at / ratio
    beyond = torch.nonzero((k > kc) & (s <= target)).flatten()
    if at <= 0 or not len(beyond):
        return min(kc, k_max)
    j = int(beyond[0])
    s0, s1 = float(s[j - 1]), float(s[j])
    f = (s0 - target) / max(s0 - s1, 1e-12) if j - 1 >= kc else 0.0
    return float(min(max(kc, j - 1 + f), k_max))


def _smooth(x: torch.Tensor, sigma_px: float, kk: torch.Tensor) -> torch.Tensor:
    """Gaussian window (sigma in voxels) by FFT."""
    n = x.shape[0]
    g = torch.exp(-2 * (math.pi * sigma_px * kk / n) ** 2)
    return torch.fft.irfftn(torch.fft.rfftn(x) * g, s=x.shape)


def adaptive_filter(h: torch.Tensor, h_ref: torch.Tensor, apix: float, res_a: float, mask: torch.Tensor | None = None,
                    n_cut: int = 12, lo_a: float = 20.0, window_a: float | None = None, edge: float = 0.1,
                    info: dict | None = None, shape: torch.Tensor | None = None, awf: float = 0.0,
                    order: int = 0) -> torch.Tensor:
    """h filtered with a cutoff chosen per voxel: candidates log-spaced from lo_a to 1.1 x... just beyond res_a (the
    global resolution; a region can be sharper than the average, so the finest candidate is 0.85 x res_a), the
    local error |filtered h - h_ref|^2 averaged over a Gaussian of sigma window_a (default 2 x res_a, at least 8 A).
    mask: only for the summary (cutoffs inside it). Returns the filtered map; info gets the cutoff map (A) and its
    distribution. shape: candidates are this per-shell weight (the global FSC weighting) stretched per region instead
    of hard low-passes - a hard cut minimises the error where a shell's signal equals its noise (local FSC ~0.5) and
    drops the band between that and the 0.143 resolution, which the weighting keeps attenuated. awf: window per
    candidate (see _sigma); order: Butterworth candidates (see _candidate)."""
    n = h.shape[0]
    dev = h.device
    box_a = n * apix
    kk = _kk(n, dev)
    Fh = torch.fft.rfftn(h.float())
    hr = h_ref.float()
    k_hi = min(n / 2 - 1, box_a / (0.85 * res_a))
    k_lo = max(2.0, box_a / lo_a)
    cuts = torch.exp(torch.linspace(math.log(k_lo), math.log(max(k_hi, k_lo * 1.01)), n_cut)).tolist()
    best_err = best = best_k = None
    sh_ = None if shape is None else shape.to(dev).float()
    for kc in cuts:
        sig = _sigma(window_a, awf, res_a, kc, box_a, apix)
        f = torch.fft.irfftn(Fh * _candidate(kk, kc, edge, sh_, box_a / res_a, order), s=h.shape)
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
                    candidates_a=[round(box_a / k, 2) for k in cuts], window_a=round(sig * apix, 1), awf=awf, order=order)
    return best


def adaptive_filter_cv(h: torch.Tensor, fold_a: torch.Tensor, fold_b: torch.Tensor, apix: float, apix_s: float,
                       res_a: float, mask: torch.Tensor | None = None, n_cut: int = 12, lo_a: float = 20.0,
                       window_a: float | None = None, edge: float = 0.1, info: dict | None = None,
                       shape: torch.Tensor | None = None, awf: float = 0.0, order: int = 0, shift: int = 0,
                       snr: torch.Tensor | None = None, ratio: float = 1.0) -> torch.Tensor:
    """Gold-standard version of adaptive_filter: the per-voxel cutoff for half map h (n^3, apix) is chosen by
    cross-validation between two reconstructions from disjoint subsets of h's own particles (fold_a, fold_b: m^3 at
    apix_s, the same box in A, usually a smaller band-limited box) - no information from the other half - and then
    applied to h. The local error of each candidate is that of filtered fold_a against fold_b plus filtered fold_b
    against fold_a (each fold's noise independent of the other's).
    snr, ratio: the folds hold 1 / ratio of h's particles (ratio 2: two folds of a half set), so a cutoff chosen from
    them is applied to h where h's SNR (snr: per shell of the full box, from the half-map FSC) has fallen by that ratio
    from its value at the chosen cutoff - the point where h's SNR equals the folds' - not at the cutoff itself."""
    n, m = h.shape[0], fold_a.shape[0]
    dev = h.device
    box_a = n * apix
    kk_s = _kk(m, dev)
    Fa, Fb = torch.fft.rfftn(fold_a.float()), torch.fft.rfftn(fold_b.float())
    k_hi = min(m / 2 - 1, box_a / (0.85 * res_a))
    k_lo = max(2.0, box_a / lo_a)
    cuts = torch.exp(torch.linspace(math.log(k_lo), math.log(max(k_hi, k_lo * 1.01)), n_cut)).tolist()
    fa_, fb_ = fold_a.float(), fold_b.float()
    best_err = best_i = None
    sh_ = None if shape is None else shape.to(dev).float()
    for i, kc in enumerate(cuts):
        sig = _sigma(window_a, awf, res_a, kc, box_a, apix_s)
        lp = _candidate(kk_s, kc, edge, sh_, box_a / res_a, order)
        ea = torch.fft.irfftn(Fa * lp, s=fold_a.shape) - fb_
        eb = torch.fft.irfftn(Fb * lp, s=fold_b.shape) - fa_
        err = _smooth(ea * ea + eb * eb, sig, kk_s)
        if best_err is None:
            best_err, best_i = err, torch.zeros_like(err)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_i = torch.where(take, torch.full_like(err, float(i)), best_i)
        del ea, eb, err
    # the choice on the full grid (nearest), then each candidate applied to h where it was chosen (shift: that many
    # candidates finer - experimental: the folds hold a quarter of the data, the half map twice theirs)
    best_i = (best_i + shift).clamp(0, len(cuts) - 1)
    k_app = list(cuts)                        # the cutoff applied to h for each candidate
    if snr is not None and ratio > 1:
        k_app = [_snr_shift(snr, kc, ratio, n / 2 - 1) for kc in cuts]
    idx = torch.nn.functional.interpolate(best_i[None, None], size=(n, n, n), mode="nearest")[0, 0]
    kk = _kk(n, dev)
    Fh = torch.fft.rfftn(h.float())
    out = torch.zeros_like(h, dtype=torch.float32)
    for i, kc in enumerate(cuts):
        sel = idx == i
        if not bool(sel.any()):
            continue
        f = torch.fft.irfftn(Fh * _candidate(kk, k_app[i], edge, sh_, box_a / res_a, order), s=h.shape)
        out = torch.where(sel, f, out)
        del f
    if info is not None:
        res_map = torch.tensor([box_a / k for k in k_app], device=dev)[idx.long()]
        sel = res_map[mask > 0.5] if mask is not None and bool((mask > 0.5).any()) else res_map.reshape(-1)
        q = torch.quantile(sel[torch.randperm(len(sel), device=dev)[:1_000_000]], torch.tensor([0.05, 0.5, 0.95], device=dev))
        info.update(cutoff_map_a=res_map, cutoff_p05_a=float(q[0]), cutoff_median_a=float(q[1]),
                    cutoff_p95_a=float(q[2]), candidates_a=[round(box_a / k, 2) for k in cuts],
                    window_a=round(sig * apix_s, 1), strict=True, awf=awf, order=order)
    return out


def cutoff_stats(infos: list, mask: torch.Tensor, res_a: float) -> dict:
    """Summary of one non-uniform filtering step from the two halves' infos (cutoff_map_a each) inside the mask
    (> 0.5): percentiles of the mean cutoff, the share of the mask filtered at the global resolution or finer
    (cutoff <= 1.05 x res_a) and at twice it or coarser (regions the filter treats as disordered), the share per
    candidate cutoff, and how well the two halves' cutoff maps agree (Pearson r in the mask: near 1, the regions are
    real; near 0, the choice follows noise)."""
    sel = mask > 0.5
    if not bool(sel.any()):
        sel = torch.ones_like(mask, dtype=torch.bool)
    c0, c1 = infos[0]["cutoff_map_a"][sel].float(), infos[1]["cutoff_map_a"][sel].float()
    c = 0.5 * (c0 + c1)
    g = torch.Generator(device=c.device).manual_seed(0)
    sub = torch.randperm(len(c), generator=g, device=c.device)[:1_000_000]
    q = torch.quantile(c[sub], torch.tensor([0.05, 0.5, 0.95], device=c.device))
    a, b_ = c0 - c0.mean(), c1 - c1.mean()
    r = float((a * b_).sum() / ((a * a).sum() * (b_ * b_).sum()).sqrt().clamp_min(1e-12))
    cands = infos[0]["candidates_a"]
    share = [round(float(((c0 - x).abs() < 0.006).float().mean()) * 100, 2) for x in cands]   # candidates_a: 2 decimals
    return dict(p05_a=round(float(q[0]), 2), median_a=round(float(q[1]), 2), p95_a=round(float(q[2]), 2),
                res_a=round(float(res_a), 3), at_global_pct=round(float((c <= 1.05 * res_a).float().mean()) * 100, 1),
                coarse_pct=round(float((c >= 2 * res_a).float().mean()) * 100, 1), halves_r=round(r, 3),
                share_pct=share)
