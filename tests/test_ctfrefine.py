"""CtfRefine's shrinkage of per-particle defocus offsets toward the micrograph plane (CPU, instant)."""
import numpy as np

from celeris2d.ctfrefine import _shrink


def _sim(tau, n=50000, seed=0):
    """Offsets about the plane: real ones (sd tau) plus each particle's measurement noise; the two half-band
    estimates carry twice the noise variance each, the full band their average."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, tau, n) if tau else np.zeros(n)
    sig = rng.uniform(15, 200, n)
    na, nb = rng.normal(0, np.sqrt(2) * sig), rng.normal(0, np.sqrt(2) * sig)
    curv = 7.0 / sig ** 2                             # the score peak's sharpness: any scale
    return true, true + (na + nb) / 2, na - nb, curv


def test_shrink_recovers_spread_and_beats_plane_and_raw():
    true, delta, gap, curv = _sim(50)
    out, info = _shrink(delta, np.zeros(len(true)), curv, np.zeros(len(true), bool), gap)
    assert 40 < info["tau_a"] < 60, info
    err = lambda v: float(np.sqrt(np.mean((v - true) ** 2)))
    assert err(out) < err(np.zeros(len(true))) and err(out) < err(delta)


def test_shrink_to_the_plane_when_offsets_are_noise():
    true, delta, gap, curv = _sim(0)
    out, info = _shrink(delta, np.zeros(len(true)), curv, np.zeros(len(true), bool), gap)
    assert info["tau_a"] < 25 and float(np.sqrt(np.mean(out ** 2))) < 10, info     # (a small bias at 0: 15 A)


def test_shrink_keeps_offsets_measured_well():
    """Real spread much larger than the noise (beta-gal-like heights in the ice): the offsets mostly stay."""
    true, delta, gap, curv = _sim(150)
    out, info = _shrink(delta, np.zeros(len(true)), curv, np.zeros(len(true), bool), gap)
    assert info["weight_median"] > 0.6, info
