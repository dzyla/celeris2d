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
