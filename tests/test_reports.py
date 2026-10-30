"""Every 3D report figure and data file is drawn from synthetic inputs (no GPU work): a figure bug must not cost a
finished refinement."""
import os

import numpy as np
import torch

from celeris2d.abinitio import euler_to_matrix
from celeris2d.jobs import refinement_report


def _inputs(n=600, b=32, K=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = euler_to_matrix(torch.rand(n, generator=g) * 360, torch.rand(n, generator=g) * 180,
                        torch.rand(n, generator=g) * 360)
    hist = [dict(it=i + 1, align_res=10.0 - i, step=2.0 / (i + 1), moved=0.3 / (i + 1), switched=0.1 / (i + 1),
                 pops=[n // K] * K, res=[8.0 - i * 0.5] * K, seconds=10.0 * (i + 1)) for i in range(4)]
    f = np.clip(1.2 - np.linspace(0, 1.5, b // 2 + 1), 0, 1)
    maps = [np.random.default_rng(seed + k).standard_normal((b, b, b)).astype(np.float32) for k in range(K)]
    masks = [np.ones((b, b, b), np.float32) for _ in range(K)]
    return A, hist, f, maps, masks


def _files(d):
    return sorted(os.listdir(d))


def test_refine3d_report(tmp_path, monkeypatch):
    from celeris2d import plotstyle
    monkeypatch.setattr(plotstyle, "THEMES", ("dark", "light"))      # both themes drawn when asked
    A, hist, f, maps, masks = _inputs()
    n = len(A)
    out = refinement_report(str(tmp_path), "test", "D7", 100.0, 100.0 / 32, A, A, torch.zeros(n, 2),
                            np.random.rand(n), np.random.rand(n) > 0.1, hist,
                            fsc=dict(corrected=f, unmasked=f * 0.9, masked=f, sphere_masked=f), resolution_a=6.0,
                            maps=maps, masks=masks)
    fs = _files(tmp_path)
    for x in ("fsc.png", "fsc.light.png", "fsc.json", "fsc.star", "fsc.xml", "views.png", "views.json",
              "progress.png", "particles.png", "map_views.png", "map_views.light.png"):
        assert x in fs, x
    assert 0 < out["scf"] <= 1 and out["history"] == hist


def test_hetero_report(tmp_path):
    A, hist, f, maps, masks = _inputs(K=3)
    n = len(A)
    cls = np.arange(n) % 3
    out = refinement_report(str(tmp_path), "test", "C1", 100.0, 100.0 / 32, A, A, torch.zeros(n, 2),
                            np.random.rand(n), np.ones(n, bool), hist,
                            fsc={f"class{k + 1}": f for k in range(3)}, resolution_a=6.0, maps=maps, masks=masks,
                            cls=cls, class_labels=["class 1", "class 2", "class 3"], class_res=[6.0, 7.0, float("nan")])
    assert "views_classes.png" in _files(tmp_path) and len(out["class_views"]) == 3
