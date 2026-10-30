"""Local resolution on synthetic half maps whose resolution changes across the box, and the angular statistics."""
import math

import numpy as np
import torch

from celeris2d import diagnostics as dg
from celeris2d.abinitio import euler_to_matrix
from celeris2d.localres import localres


def _crossing(f, box_a, thr=0.143):
    f = np.asarray(f)
    i = int(np.argmax(f[1:] < thr)) + 1
    return box_a / (i - 1 + (f[i - 1] - thr) / (f[i - 1] - f[i]))


def test_local_resolution_follows_the_truth():
    """Half maps = shared signal with amplitudes falling as exp(-B s^2 / 4) + independent noise, B = 60 A^2 in one half
    of the box and 250 A^2 in the other. The truth of each region is the FSC of whole-box maps made with its B."""
    from celeris2d.abinitio import fsc_maps
    b, apix = 128, 1.0
    rng = np.random.default_rng(0)
    k, kz = np.fft.fftfreq(b), np.fft.rfftfreq(b)
    s2 = (k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2) / apix ** 2
    sig = np.fft.rfftn(rng.standard_normal((b,) * 3))

    def halves(B, seed):
        r = np.random.default_rng(seed)
        S = np.fft.irfftn(sig * np.exp(-B * s2 / 4), s=(b,) * 3, axes=(0, 1, 2))
        return [(S + 0.3 * r.standard_normal((b,) * 3)).astype(np.float32) for _ in range(2)]
    hl, hr = halves(60, 2), halves(250, 3)
    truth = [_crossing(fsc_maps(torch.tensor(h[0]), torch.tensor(h[1])).numpy(), b * apix) for h in (hl, hr)]
    left = np.zeros((b,) * 3, np.float32)
    left[:, :, : b // 2] = 1
    h1, h2 = left * hl[0] + (1 - left) * hr[0], left * hl[1] + (1 - left) * hr[1]
    r = localres(h1, h2, None, apix, res_global=4.5, filtered=True)
    lr = r["map"].cpu().numpy()
    est = [np.median(lr[30:-30, 30:-30, 16:48]), np.median(lr[30:-30, 30:-30, 80:112])]  # away from the boundary
    for e, t in zip(est, truth):
        assert abs(e - t) / t < 0.06, (est, truth)
    assert r["filtered"].shape == h1.shape and torch.isfinite(r["filtered"]).all()


def test_scf_uniform_vs_top_views():
    g = torch.Generator().manual_seed(0)
    n = 20000
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.T
    A = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    s_uniform = dg.angular_stats(A)
    assert s_uniform["scf"] > 0.95 and not s_uniform["warnings"]
    assert dg.angular_stats(A[:2000], "D7")["scf"] > 0.95             # symmetry copies count
    assert np.median(dg.orientation_change(A[:100], A[:100], "D7")) < 0.1
    tilt = torch.rand(n, generator=g) * 8                               # top views only (tilt < 8 deg)
    rot = torch.rand(n, generator=g) * 360
    psi = torch.rand(n, generator=g) * 360
    s_top = dg.angular_stats(euler_to_matrix(rot, tilt, psi))
    assert s_top["scf"] < 0.5 and s_top["warnings"]


def test_fsc_files(tmp_path):
    f = np.linspace(1, 0, 33)
    names = dg.write_fsc(str(tmp_path / "fsc"), {"corrected": f, "unmasked": f * 0.9}, box_a=64.0, resolution_a=3.5)
    assert names == ["fsc.json", "fsc.star", "fsc.xml"]
    star = (tmp_path / "fsc.star").read_text()
    assert "_rlnFourierShellCorrelationCorrected #4" in star and "_rlnFinalResolution 3.5000" in star
    import xml.etree.ElementTree as ET
    pts = ET.parse(tmp_path / "fsc.xml").getroot().findall("coordinate")
    assert len(pts) == 33 and math.isclose(float(pts[0].find("y").text), 1.0)
