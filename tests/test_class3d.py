"""3D classification without alignment (class3d.py) on simulated particles with a known answer (needs a CUDA GPU).

Two conformations of one object - with and without an extra domain, 30 / 70 - are projected at random known
orientations with noise and written as one .mrcs stack. Classification at the true poses must recover the
conformation of each particle, over the whole particle and with a focus mask on the variable domain (signal
subtraction of the rest on the fly), close to what the true maps themselves allow (0.98 at this noise: one E-step
against the true maps); at a lower SNR the default start (variability components) must still find the split, where
random subsets often do not; with symmetry expansion, every copy of a C2 particle whose domain sits on one of its
two subunits must find that subunit. The Class3D job's particle table must work for Select and as the poses of
Refine3D (classes=(k,)) like a HeteroRefine's. ~10 s.

    pytest -q tests/test_class3d.py
"""
import os

import mrcfile
import numpy as np
import pytest
import torch

from celeris2d import abinitio as ab
from celeris2d import class3d as c3
from celeris2d.session import Session

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

BOX, APIX, N, NOISE, SHARE = 64, 4.0, 3000, 3.0, 0.3
_r = np.random.default_rng(7)
BASE = [tuple(_r.uniform(-12, 12, 3)) + (_r.uniform(2.0, 3.5), _r.uniform(0.6, 1.2)) for _ in range(14)]
DOMAIN = [(17.0, 4.0, 2.0, 2.5, 1.0), (19.0, -1.0, -2.0, 2.2, 0.9), (15.0, 8.0, -3.0, 2.0, 0.8)]   # px from the centre
FOCUS_A = (17.0 * APIX, 4.0 * APIX, 0.0, 40.0)                                                        # x, y, z, radius (A)


def _volume(blobs, b=BOX):
    r = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    v = torch.zeros(b, b, b)
    for cx, cy, cz, s, a in blobs:
        v += a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))
    return v


def _c2(blobs):
    """The object made C2 about z (blob (x, y, z) -> (-x, -y, z))."""
    return list(blobs) + [(-x, -y, z, s, a) for x, y, z, s, a in blobs]


def _simulate(path, vols, labels, seed=4):
    """Projections of vols[labels[i]] at random orientations, one noise level for all (an absent domain is absent
    signal), written as an .mrcs stack. Returns the orientations."""
    dev = "cuda"
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    rot, psi = torch.rand(n, generator=g) * 360 - 180, torch.rand(n, generator=g) * 360 - 180
    tilt = torch.rad2deg(torch.arccos(torch.rand(n, generator=g) * 2 - 1))
    A = ab.euler_to_matrix(rot, tilt, psi)
    img = torch.empty((n, BOX, BOX), device=dev)
    for k, v in enumerate(vols):
        V = ab.Volume(BOX, dev)
        V.set_real(v.to(dev))
        sel = torch.nonzero(torch.as_tensor(labels) == k).squeeze(1)
        img[sel.to(dev)] = torch.fft.irfft2(V.project(A[sel].to(dev)), s=(BOX, BOX), norm="forward")
    img = img / img.std() + NOISE * torch.randn(img.shape, generator=g).to(dev)
    with mrcfile.new(path) as m:
        m.set_data(img.cpu().numpy().astype(np.float32))
        m.voxel_size = APIX
    return A


def _accuracy(z, truth, K):
    """Share of particles in the class matched to their conformation (best one-to-one matching of classes)."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((K, int(truth.max()) + 1))
    np.add.at(C, (np.asarray(z), np.asarray(truth)), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(z)


@pytest.fixture(scope="module")
def mixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("class3d")
    labels = (np.random.default_rng(3).random(N) < SHARE).astype(int)          # 1 = with the domain
    A = _simulate(str(d / "particles.mrcs"), [_volume(BASE), _volume(BASE + DOMAIN)], labels)
    s = Session(str(d / "work"), log=lambda *a: None)
    s.load(str(d / "particles.mrcs"), diameter=200, max_res=8, gpus="0")
    s.clf._wait_loaded(N)
    return s.clf, A, labels, d


def _focus_mask(b):
    """A soft sphere on the variable domain (box b, same 256 A field of view)."""
    apix = BOX * APIX / b
    r = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(r * apix, r * apix, r * apix, indexing="ij")
    d = torch.sqrt((x - FOCUS_A[0]) ** 2 + (y - FOCUS_A[1]) ** 2 + (z - FOCUS_A[2]) ** 2)
    return (0.5 + 0.5 * torch.cos(torch.pi * ((d - FOCUS_A[3]) / 12).clamp(0, 1)))


def test_classes_recovered(mixture):
    """Whole particle, default settings (variability start, hard): the domain splits the particles. Measured 0.978
    (true maps: 0.979; random-subset start 0.967, soft 0.973)."""
    clf, A, labels, d = mixture
    t = torch.zeros(N, 2)
    r = c3.classify(clf, torch.arange(N), A, t, 2, 12.0, n_iter=15, log=lambda *a: None)
    acc = _accuracy(r.z.numpy(), labels, 2)
    print(f"\n[class3d] whole particle, K=2: accuracy {acc:.3f}, iterations {r.iterations}")
    assert acc > 0.95, acc
    k_dom = int(np.bincount(r.z.numpy()[labels == 1], minlength=2).argmax())
    assert abs(float((r.z == k_dom).float().mean()) - SHARE) < 0.05
    # the class with the domain has density there, the other not
    b = r.box
    m = _focus_mask(b) > 0.5
    dens = [float(r.maps[k][m].mean() / r.maps[k][r.maps[k] > 0].mean()) for k in range(2)]
    assert dens[k_dom] > 2 * dens[1 - k_dom], dens
    # figures and the job's report on a plain folder
    figs = c3.class3d_report(str(d), "test", "C1", clf.box_a, clf.box_a / b, r, r.z.numpy(), r.post.numpy(),
                             [m_.numpy() for m_ in r.maps], ["class 1", "class 2"], [8.0, 9.0], {}, [0, 1])
    for f in ("map_views.png", "progress.png", "views_classes.png"):
        assert f in figs["figures"] and os.path.isfile(os.path.join(d, f))


def test_focused_with_subtraction(mixture):
    """Focus mask on the variable domain, the consensus outside it subtracted at each pose (measured 0.972); a K=3
    run from the consensus + noise must still put the domain particles in one class (measured purity 0.92, recall
    1.00: the particles without the domain fill the other two classes)."""
    clf, A, labels, d = mixture
    t = torch.zeros(N, 2)
    cons = c3.classify(clf, torch.arange(N), A, t, 1, 12.0, n_iter=1, log=lambda *a: None)
    b = cons.box
    r = c3.classify(clf, torch.arange(N), A, t, 2, 12.0, n_iter=15, focus=_focus_mask(b), subtract=cons.maps[0],
                    log=lambda *a: None)
    acc = _accuracy(r.z.numpy(), labels, 2)
    print(f"\n[class3d] focus mask + subtraction, K=2: accuracy {acc:.3f}")
    assert acc > 0.95, acc
    r3 = c3.classify(clf, torch.arange(N), A, t, 3, 12.0, n_iter=15, focus=_focus_mask(b), subtract=cons.maps[0],
                     init_maps=[cons.maps[0]] * 3, soft=False, log=lambda *a: None)
    z = r3.z.numpy()
    k_dom = int(np.bincount(z[labels == 1], minlength=3).argmax())
    purity = float((labels[z == k_dom] == 1).mean())
    recall = float((z[labels == 1] == k_dom).mean())
    print(f"[class3d] focus, K=3 hard, consensus start: domain class purity {purity:.3f}, recall {recall:.3f}, "
          f"sizes {np.bincount(z, minlength=3).tolist()}")
    assert purity > 0.9 and recall > 0.9, (purity, recall)


def test_symmetry_expansion(tmp_path):
    """A C2 particle with the extra domain on one of its two subunits (a random one per particle, 40 % of the
    particles have it): expanded over C2 and focused on one subunit's domain site, every copy is classified on its
    own - the copies that carry the domain are found, also in particles whose other copy is empty. Measured: 0.976."""
    n = 2000
    rng = np.random.default_rng(5)
    has = rng.random(n) < 0.4
    side = rng.integers(0, 2, n)
    labels = np.where(has, 1 + side, 0)                       # 0: no domain, 1: on subunit 0, 2: on the C2 copy
    base = _c2(BASE[:8])
    rotd = [(-x, -y, z, s, a) for x, y, z, s, a in DOMAIN]
    vols = [_volume(base), _volume(base + DOMAIN), _volume(base + rotd)]
    A = _simulate(str(tmp_path / "c2.mrcs"), vols, labels, seed=6)
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(str(tmp_path / "c2.mrcs"), diameter=200, max_res=8, gpus="0")
    clf = s.clf
    clf._wait_loaded(n)
    X = ab.symmetry_ops("C2")
    rows = torch.arange(n).repeat_interleave(2)
    A_r = (A[:, None] @ X[None]).reshape(-1, 3, 3)
    t_r = torch.zeros(2 * n, 2)
    cons = c3.classify(clf, torch.arange(n), A, torch.zeros(n, 2), 1, 12.0, n_iter=1, sym="C2", log=lambda *a: None)
    r = c3.classify(clf, torch.arange(n), A_r, t_r, 2, 12.0, rows=rows, n_iter=15, focus=_focus_mask(cons.box),
                    subtract=cons.maps[0], log=lambda *a: None)
    # copy c of particle i carries the domain in the masked site if the domain sits on subunit c
    truth = np.zeros(2 * n, int)
    truth[0::2] = labels == 1
    truth[1::2] = labels == 2
    acc = _accuracy(r.z.numpy(), truth, 2)
    print(f"\n[class3d] C2 expansion, focused: copy accuracy {acc:.3f} ({2 * n} copies)")
    assert acc > 0.9, acc


def test_low_snr_start(tmp_path):
    """Noise 5 (true maps: 0.90 of particles right): the variability start finds the split (measured 0.844-0.859 over
    3 seeds); random subsets mostly split by noise (0.51-0.71)."""
    labels = (np.random.default_rng(3).random(N) < SHARE).astype(int)
    global NOISE
    keep, NOISE = NOISE, 5.0
    try:
        A = _simulate(str(tmp_path / "p.mrcs"), [_volume(BASE), _volume(BASE + DOMAIN)], labels)
    finally:
        NOISE = keep
    s = Session(str(tmp_path / "work"), log=lambda *a: None)
    s.load(str(tmp_path / "p.mrcs"), diameter=200, max_res=8, gpus="0")
    s.clf._wait_loaded(N)
    r = c3.classify(s.clf, torch.arange(N), A, torch.zeros(N, 2), 2, 12.0, log=lambda *a: None)
    acc = _accuracy(r.z.numpy(), labels, 2)
    print(f"\n[class3d] noise 5, variability start: accuracy {acc:.3f}")
    assert acc > 0.8, acc


def test_class3d_table_for_select_and_refine(tmp_path):
    """A Class3D job's table (uid, pose, cls, prob_k, half, used; symmetry-expanded: one row per copy) works for
    Select (select_classes) and as the poses of Refine3D with classes=(k,) (_posed_particles), as a HeteroRefine's."""
    from celeris2d import jobs, jobspec
    from test_project import _synthetic_project
    rng = np.random.default_rng(1)
    prj, pj, ex = _synthetic_project(tmp_path, rng)
    e = prj.particles(ex)
    n = len(e)
    ref = prj.new_job("Refine3D", dict(particles=ex.id, sym="C1"), [ex], "fake")
    prj.set_status(ref, "finished", dict(resolution_a=8.0))
    c = jobs.new_job(prj, "Class3D", {"particles": ex.id, "poses": ref.id}, {"n_classes": 2, "sym_expand": "C2"})
    assert c.type == "Class3D" and c.params["poses"] == ref.id and c.params["reference"] == ref.id
    assert jobspec.settings_of(c, c3.Class3DSettings).sym_expand == "C2"
    uid = np.repeat(e["uid"].to_numpy(), 2)
    cls = np.zeros(2 * n, np.int16)
    cls[1::2] = np.arange(n) % 2                    # copy 1 of every other particle in class 2
    z = np.zeros(2 * n, np.float32)
    prj.table(c, "particles").write(dict(uid=uid, rot=z, tilt=z, psi=z, shift_x_a=z + 1, shift_y_a=z - 1, cls=cls,
                                         fit=z + 0.1, half=np.zeros(2 * n, np.uint8), used=np.ones(2 * n, bool),
                                         prob_1=1 - cls.astype(np.float32), prob_2=cls.astype(np.float32),
                                         copy=np.tile([0, 1], n).astype(np.uint8)),
                                    np.repeat(e["movie_id"].to_numpy(), 2))
    prj.set_status(c, "finished", dict(classes=[dict(cls=1, particles=int((cls == 0).sum())),
                                                dict(cls=2, particles=int((cls == 1).sum()))]))
    assert "K2" in jobspec.key_of("Class3D", prj.summary_of(c))
    sel = jobs.select_classes(prj, c, [2])
    assert len(prj.selected_uids(sel)) == (np.arange(n) % 2).sum()
    r = jobs.refine3d_job(prj, ex, c, c, jobs.Refine3DSettings(classes=(2,)))
    src, poses, t_start, half, n_dup, dist = jobs._posed_particles(prj, r, ex, c, (2,), log=lambda *a: None)
    assert src.N + n_dup == (np.arange(n) % 2).sum() and (poses["cls"] == 1).all()
    assert np.allclose(t_start, [[1.0, -1.0]], atol=1e-3)            # same extraction: the shift is the pose's
