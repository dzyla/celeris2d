"""A single-frame .mrc (a micrograph aligned and summed elsewhere, e.g. a public set like CryoPPP) goes through
MotionCtf as is: no alignment, the same image binned, no even/odd halves."""
import numpy as np
import torch

from celeris2d import preprocess as pp


def test_single_frame_micrograph():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((512, 512)).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = pp.motion_correct(img, None, 1.0, 50.0, bin=1, device=dev)
    assert m.sum.shape == (512, 512) and m.halves == () and m.shifts.shape == (1, 2)
    assert np.allclose(m.sum.cpu().numpy(), img, atol=1e-4) and torch.equal(m.sum, m.sum_dw)
    m2 = pp.motion_correct(img, None, 1.0, 50.0, bin=2, device=dev)
    assert m2.sum.shape == (256, 256) and abs(m2.apix - 2.0) < 1e-6


def test_float16_fallback(tmp_path):
    """Micrographs in raw counts (CryoPPP EMPIAR-10017 Falcon: mean ~70,900) do not fit float16 (max 65504; written
    as float16 they were 98 % inf and every figure, re-pick and re-extraction read nothing): write_mrc stores them as
    float32. Normal-range data keep float16."""
    import mrcfile
    from celeris2d.project import Project, fits_float16
    rng = np.random.default_rng(1)
    raw = (70866 + 2742 * rng.standard_normal((256, 256))).astype(np.float32)
    small = rng.standard_normal((256, 256)).astype(np.float32)
    offset = (30000 + 50 * rng.standard_normal((256, 256))).astype(np.float32)   # fits, but steps of 16-32 vs std 50
    assert not fits_float16(raw) and fits_float16(small) and not fits_float16(offset)
    p = Project(str(tmp_path / "prj"))
    for name, x, want in (("raw", raw, "float32"), ("small", small, "float16")):
        f = str(tmp_path / f"{name}.mrc")
        assert p.write_mrc(f, x, 1.77, "float16") == want
        with mrcfile.open(f, permissive=True) as m:
            y = np.asarray(m.data, np.float32)
        assert np.isfinite(y).all() and np.allclose(y, x, rtol=2e-3, atol=2e-3)
