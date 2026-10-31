"""Angular distribution BILD file (RELION's angdist.bild, for ChimeraX): one cylinder per occupied direction bin,
starting on the particle radius around the box centre, pointing along the viewing direction, the busiest bin red and
longest (CPU, instant)."""
import numpy as np

from celeris2d.diagnostics import particle_radius, write_angdist_bild


def test_angdist_bild(tmp_path):
    rng = np.random.default_rng(0)
    rot = np.concatenate([np.full(500, 0.0), rng.uniform(-180, 180, 200)])     # 500 top views (tilt 0) + a spread
    tilt = np.concatenate([np.full(500, 0.0), np.rad2deg(np.arccos(rng.uniform(-1, 1, 200)))])
    p = tmp_path / "angdist.bild"
    n = write_angdist_bild(str(p), rot, tilt, box_a=300.0, radius_a=100.0)
    lines = p.read_text().splitlines()
    assert len(lines) == 2 * n and 50 < n <= 201
    cyl = np.array([[float(v) for v in l.split()[1:]] for l in lines[1::2]])
    col = np.array([[float(v) for v in l.split()[1:]] for l in lines[0::2]])
    base = cyl[:, :3] - 150.0
    assert np.allclose(np.linalg.norm(base, axis=1), 100.0, atol=0.1)          # all start on the radius, centred
    top = int(np.argmax(np.linalg.norm(cyl[:, 3:6] - cyl[:, :3], axis=1)))
    assert base[top, 2] > 99 and col[top, 0] == 1.0 and col[top, 2] == 0.0     # the top-view bin: +z, longest, red
    m = np.zeros((32, 32, 32), np.float32)
    m[16, 16, 4:28] = 1
    assert abs(particle_radius(m, 2.0, 64.0) - 2.0 * np.sqrt(11.5 ** 2 + 0.5)) < 1e-3   # farthest voxel (centre 15.5)
    assert particle_radius(None, 2.0, 64.0) == 0.35 * 64.0


def test_view_coverage_same_for_every_group():
    """Uniform views cover the sphere for every symmetry group; a top + side band does not (C8 read 0 % for uniform
    views and 100 % for the band when each direction was folded to its nearest copy)."""
    import torch
    from celeris2d.abinitio import view_coverage
    g = torch.Generator().manual_seed(0)
    q = torch.randn(40000, 4, generator=g)
    w, x, y, z = (q / q.norm(dim=1, keepdim=True)).T
    A = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    tilt = torch.rad2deg(torch.arccos(A[:, 2, 2].clamp(-1, 1)))
    band = A[(tilt < 20) | ((tilt > 80) & (tilt < 100))]
    for sym in ("C1", "C4", "C8", "D2", "D4"):
        assert view_coverage(A, sym)["occupied"] > 0.95, sym
        assert view_coverage(band, sym)["occupied"] < 0.45, sym
