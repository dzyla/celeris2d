"""End-to-end checks on a synthetic stack with a known answer (needs a CUDA GPU; ~1 min).

Three asymmetric shapes, random in-plane rotations and shifts, Gaussian noise, written as an .mrcs stack and read
through the normal input path. Each test checks that the classes are recovered and that the in-plane angles are
consistent within each class (estimated - true angle is constant up to the class's own arbitrary frame).

    pytest -q tests
"""
import threading
import time

import mrcfile
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from celeris2d import gpu
from celeris2d.engine import rot_grid
from celeris2d.session import Session

pytestmark = pytest.mark.skipif(not gpu.count(), reason="needs a GPU (or CELERIS2D_DEVICE=cpu)")

BOX, APIX, N_PER, NOISE = 64, 4.0, 1000, 1.2


def _blobs(spec):
    r = torch.arange(BOX, dtype=torch.float32) - BOX // 2
    y, x = torch.meshgrid(r, r, indexing="ij")
    img = torch.zeros(BOX, BOX)
    for cx, cy, sig, amp in spec:
        img += amp * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
    return img


SHAPES = [  # (x, y, sigma, amplitude) in pixels; none has in-plane symmetry
    [(-8, 0, 3, 1.0), (0, 0, 3, 1.0), (8, 0, 3, 1.0), (8, 8, 3, 1.0)],      # L
    [(-9, -6, 3, 1.0), (0, -6, 3, 1.0), (9, -6, 3, 1.0), (0, 4, 4, 1.2)],   # T
    [(-7, 6, 5, 1.3), (7, 6, 3, 0.8), (0, -8, 2.5, 1.0)],                  # uneven triangle
]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    g = torch.Generator().manual_seed(1)
    imgs, labels, psis = [], [], []
    for k, spec in enumerate(SHAPES):
        base = _blobs(spec)
        psi = torch.rand(N_PER, generator=g) * 360
        up = torch.fft.irfft2(torch.fft.rfft2(base), s=(BOX, BOX))   # same image; rot_grid reads a 1x grid here
        rot = F.grid_sample(up[None, None].expand(N_PER, 1, BOX, BOX), rot_grid(psi, BOX, BOX), align_corners=False)[:, 0]
        shift = (torch.rand(N_PER, 2, generator=g) - 0.5) * 4                                     # +-2 px
        ft = torch.fft.rfft2(rot)
        fy = torch.fft.fftfreq(BOX)[:, None]
        fx = torch.fft.rfftfreq(BOX)[None, :]
        ft = ft * torch.exp(-2j * np.pi * (fy * shift[:, 1, None, None] + fx * shift[:, 0, None, None]))
        rot = torch.fft.irfft2(ft, s=(BOX, BOX))
        imgs.append(rot + NOISE * torch.randn(rot.shape, generator=g))
        labels.append(np.full(N_PER, k))
        psis.append(psi.numpy())
    data = torch.cat(imgs).numpy().astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(data))
    path = tmp_path_factory.mktemp("synth") / "synthetic.mrcs"
    with mrcfile.new(str(path)) as m:
        m.set_data(data[perm])
        m.voxel_size = APIX
    return str(path), np.concatenate(labels)[perm], np.concatenate(psis)[perm]


def _session(tmp_path, stack):
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(stack[0], diameter=200, max_res=12, gpus="0")
    return s


def _check(s, name, stack, min_acc=0.95, min_consistent=0.85):
    _, labels, psi_true = stack
    r = s.results[name]
    pred = r.labels
    acc = 0
    for k in np.unique(pred):
        acc += np.bincount(labels[pred == k], minlength=3).max()
    acc /= len(pred)
    assert acc >= min_acc, f"{name}: only {acc:.3f} of particles in the class of their shape"
    for k in np.unique(pred):           # angles: estimated - true is the same for every particle of a class
        m = (pred == k) & (labels == np.bincount(labels[pred == k]).argmax())
        if m.sum() < 50:
            continue
        d = (r.psi[m] - psi_true[m]) % 360
        mode = np.argmax(np.histogram(d, bins=72, range=(0, 360))[0]) * 5 + 2.5
        ok = np.mean(np.abs((d - mode + 180) % 360 - 180) < 15)   # pseudo-symmetric shapes flip a few particles
        assert ok >= min_consistent, f"{name}: class {k + 1}: only {ok:.2f} of angles consistent"
    return acc


def test_soft_pose(tmp_path, stack):
    s = _session(tmp_path, stack)
    s.classify("soft", K=3, n_iter=15)
    _check(s, "soft", stack)


def test_hard_pose_local_refine(tmp_path, stack):
    s = _session(tmp_path, stack)
    s.clf._auto_hard = False
    s.classify("hard", K=3, n_iter=15, hard_pose=True, local_refine=False)
    _check(s, "hard", stack)
    s.clf._auto_hard = False
    s.classify("refined", K=3, n_iter=15, hard_pose=True, local_refine=True)
    _check(s, "refined", stack)


def test_more_classes_than_views_none_empty(tmp_path, stack):
    """K > number of distinct views: every class must still hold particles at the end (reseeding splits the
    largest class along its real variation and goes on into the full passes), and the shapes stay separated."""
    s = _session(tmp_path, stack)
    s.classify("k8", K=8, n_iter=15, reseed_mode="split", reseed_full=True, recenter_detrend=True)
    pops = np.bincount(s.results["k8"].labels, minlength=8)
    assert (pops > 0).all(), f"empty classes: {pops.tolist()}"
    # several classes per shape: a split class may take a shape's pseudo-symmetric flips, so only the class
    # purity is checked here (the angle consistency is covered by the K=3 tests)
    _, labels, _ = stack
    pred = s.results["k8"].labels
    acc = sum(np.bincount(labels[pred == k], minlength=3).max() for k in np.unique(pred)) / len(pred)
    assert acc >= 0.9, f"only {acc:.3f} of particles in the class of their shape"


def test_stop_and_continue(tmp_path, stack):
    s = _session(tmp_path, stack)

    def stopper():
        while len(s.clf.history) < 4:
            time.sleep(0.05)
        s.clf.stop_requested = True
    threading.Thread(target=stopper, daemon=True).start()
    r = s.classify("r", K=3, n_iter=15)
    assert r["stopped"] and r["iterations"] < 15
    r = s.resume("r")
    assert not r["stopped"] and r["iterations"] >= 15
    _check(s, "r", stack)
    with pytest.raises(ValueError):
        s.resume("r")


def test_dashboard_api(tmp_path, stack):
    from fastapi.testclient import TestClient
    from celeris2d.webapp import create
    c = TestClient(create(str(tmp_path / "web")))
    assert c.get("/").status_code == 200
    d = c.get("/api/defaults").json()
    assert "K" in d["meta"] and d["meta"]["K"]["group"] == "run_basic"
    st = c.get("/api/status").json()
    assert st["loaded"] is False and st["busy"] is False
    assert c.post("/api/stop", json={}).status_code == 409          # nothing running


def test_nonfinite_particles_blanked(tmp_path):
    """A few NaN / inf particle images (cut from a broken micrograph) must not turn every class average into NaN:
    they are set to zero on load and counted (clf.nonfinite); the classes stay finite."""
    import mrcfile
    from celeris2d.engine import Classifier2D, Params
    from celeris2d.sources import open_source
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 48, 48)).astype(np.float32)
    x[:10] = np.inf
    x[10:20, 5, 5] = np.nan
    f = str(tmp_path / "p.mrcs")
    with mrcfile.new(f) as m:
        m.set_data(x)
        m.voxel_size = 4.0
    clf = Classifier2D(open_source(f), Params(diameter_a=120, max_res_a=12, n_iter=3, n_full_iter=1), log=lambda *a: None)
    clf.run(n_classes=4, seed=0)
    assert sum(len(i) for i in clf.nonfinite) == 20
    assert np.isfinite(clf.class_images()).all()
