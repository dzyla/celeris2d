"""Non-uniform (spatially adaptive) filtering on a map with two regions of known resolution: both the half-map
cross-validated filter and the gold-standard one (cutoffs from two folds of the same half) find the regions'
cutoffs and beat the best single global low-pass.

    pytest -q tests/test_nufilter.py
"""
import pytest
import torch

from celeris2d import nufilter as nu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
N, APIX = 64, 1.0


def _lp(x, res):
    kk = nu._kk(x.shape[0], x.device)
    return torch.fft.irfftn(torch.fft.rfftn(x) * (kk <= x.shape[0] * APIX / res).float(), s=x.shape)


def _data():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    s = torch.randn(N, N, N, generator=g, device=dev)
    x = torch.arange(N, device=dev)[None, None, :].expand(N, N, N)
    sig = torch.where(x < N // 2, _lp(s, 3.0), _lp(s, 8.0))           # left sharp (3 A), right blurred (8 A)
    noise = lambda sc: sc * torch.randn(N, N, N, generator=g, device=dev)
    return sig, noise


def test_adaptive_filter_finds_regions():
    sig, noise = _data()
    h1, h2 = sig + noise(0.6), sig + noise(0.6)
    info = {}
    f = nu.adaptive_filter(h1, h2, APIX, 4.0, info=info)
    cm = info["cutoff_map_a"]
    assert float(cm[..., 6:26].median()) < 4.5 and float(cm[..., 38:58].median()) > 6.0
    err = lambda m: float(((m - sig) ** 2).mean())
    assert err(f) < 0.95 * min(err(_lp(h1, r)) for r in (2.5, 3, 3.5, 4, 5, 6, 8))


def test_strict_filter_uses_only_its_own_half():
    sig, noise = _data()
    fa, fb = sig + noise(0.85), sig + noise(0.85)                      # two folds of one half
    h = 0.5 * (fa + fb)
    f = nu.adaptive_filter_cv(h, fa, fb, APIX, APIX, 4.0)
    err = lambda m: float(((m - sig) ** 2).mean())
    assert err(f) < 0.95 * min(err(_lp(h, r)) for r in (2.5, 3, 3.5, 4, 5, 6, 8))


def test_cutoff_stats_and_figure(tmp_path):
    """The per-iteration statistics see the two regions (the sharp half at the global resolution, the blurred one
    coarse), the halves' cutoff maps agree, and the NU figure draws from them."""
    sig, noise = _data()
    infos = [{}, {}]
    for i in range(2):
        fa, fb = sig + noise(0.85), sig + noise(0.85)
        nu.adaptive_filter_cv(0.5 * (fa + fb), fa, fb, APIX, APIX, 4.0, info=infos[i])
    mask = torch.ones(N, N, N, device="cuda")
    st = nu.cutoff_stats(infos, mask, 4.0)
    assert 20 < st["at_global_pct"] < 80 and 20 < st["coarse_pct"] < 80                 # left 3 A, right 8 A
    assert st["halves_r"] > 0.5 and abs(sum(st["share_pct"]) - 100) < 1
    from celeris2d import plots3d
    hist = [dict(it=i + 1, align_res=4.0, step=1.0, moved=0.1, nu=st) for i in range(3)]
    cm = (0.5 * (infos[0]["cutoff_map_a"] + infos[1]["cutoff_map_a"])).cpu().numpy()
    plots3d.nu_figure(str(tmp_path / "nu.png"), hist, cm, mask.cpu().numpy(), sig.cpu().numpy(),
                      infos[0]["candidates_a"], "test")
    assert (tmp_path / "nu.png").exists()
