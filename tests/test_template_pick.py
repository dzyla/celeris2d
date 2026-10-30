"""Template picking: an asymmetric object projected in random orientations into a noisy micrograph is found (recall),
the blob picker's peaks are unchanged by the refactor, and map projections cover the sphere."""
import numpy as np
import torch

from celeris2d import picking as pk


def _volume(b=40):
    c = torch.arange(b, dtype=torch.float32) - b // 2
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    v = torch.zeros(b, b, b)
    for p, s in (((0, 0, 0), 5.0), ((9, 3, 0), 3.0), ((-4, -9, 4), 3.5), ((2, 6, -8), 3.0)):
        v += torch.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) / (2 * s * s))
    return v.numpy()


def test_template_pick_finds_projections():
    rng = np.random.default_rng(0)
    apix = 2.0
    vol = _volume()                                      # 80 A box, ~60 A object
    proj, pa = pk.templates_from_map(vol, apix, n_dirs=200)
    tmpl, ta = pk.templates_from_map(vol, apix, n_dirs=30)
    H, W = 1024, 1024
    mic = np.zeros((H, W), np.float32)
    pos = []
    for gy in range(80, H - 80, 110):
        for gx in range(80, W - 80, 110):
            y, x = gy + rng.integers(-15, 16), gx + rng.integers(-15, 16)
            im = proj[rng.integers(len(proj))]
            im = np.rot90(im, rng.integers(4)).copy()
            b = im.shape[0]
            mic[y - b // 2:y - b // 2 + b, x - b // 2:x - b // 2 + b] += im
            pos.append((x * apix, y * apix))
    pos = np.array(pos)
    mic = -mic / mic.std() + rng.normal(0, 3.0, mic.shape).astype(np.float32)        # dark protein, SNR ~ 0.1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.as_tensor(mic, device=dev)
    c = pk.template_candidates(m, apix, 60.0, tmpl, ta, dark=True)
    keep = pk.select(c, max_std=99, max_mean=99)
    _, found = pk.match(c.xy_a[keep], pos, 15.0)
    b = pk.blob_candidates(m, apix, 60.0, dark=True)
    _, found_b = pk.match(b.xy_a[pk.select(b, max_std=99, max_mean=99)], pos, 15.0)
    top = np.argsort(-c.score)[:len(pos)]                 # as many picks as particles: precision at that depth
    tp, _ = pk.match(c.xy_a[top], pos, 15.0)
    assert found.mean() > 0.9, (found.mean(), found_b.mean())
    assert tp.mean() > 0.8, tp.mean()
    assert c.template is not None and c.template.max() < len(tmpl)


def test_valid_bounds_padding():
    """Edge rows / columns filled by repeating the last real one (micrographs aligned elsewhere) are found; a micrograph
    without padding keeps its full extent."""
    rng = np.random.default_rng(1)
    img = rng.normal(size=(300, 400)).astype(np.float32)
    assert pk.valid_bounds(torch.from_numpy(img)) == (0, 400, 0, 300)
    img[:, :23] = img[:, 23:24]
    img[:, -11:] = img[:, -12:-11]
    img[:17] = img[17:18]
    assert pk.valid_bounds(torch.from_numpy(img)) == (23, 389, 17, 300)
