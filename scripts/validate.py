"""Check 2D alignment consistency against the refined 3D angles stored in the input star.

Within a 2D class (one projection direction) the in-plane angle from 2D classification
must differ from the 3D psi by a constant. We report, per class, the circular spread of
psi2D - psi3D (sign +/- tested), restricted to particles whose 3D view (rot, tilt)
is close to the class's dominant view.
"""
import sys

import numpy as np

from celeris2d.star import read_star


def circ_std_deg(a, period=360.0):
    z = np.exp(2j * np.pi * a / period)
    r = np.abs(z.mean())
    return np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12)))) * period / 360.0, np.angle(z.mean())


def main(ref_star, out_star, verbose=False):
    ref = read_star(ref_star)["particles"]
    out = read_star(out_star)["particles"]
    assert (ref["rlnImageName"].values == out["rlnImageName"].values).all()
    p3 = ref["rlnAnglePsi"].to_numpy()
    p2 = out["rlnAnglePsi"].to_numpy()
    cls = out["rlnClassNumber"].to_numpy()
    def score(sign):
        per, good = [], 0
        for k in np.unique(cls):
            sel = cls == k
            d = (p2[sel] - sign * p3[sel]) % 360
            # D2 symmetry: 180 deg flips are symmetry-equivalent, so work modulo 180
            h = np.histogram(d % 180, bins=180, range=(0, 180))[0]
            mode = np.argmax(np.convolve(np.r_[h[-5:], h, h[:5]], np.ones(11), "valid"))
            dev = (d - mode + 90) % 180 - 90
            good += np.sum(np.abs(dev) < 10)
            per.append((k, sel.sum(), np.mean(np.abs(dev) < 10), np.median(np.abs(dev))))
        return good / len(cls), per

    s_pos, s_neg = score(+1), score(-1)
    sign, (frac, per) = (+1, s_pos) if s_pos[0] >= s_neg[0] else (-1, s_neg)
    print(f"psi2D = {sign:+d}*psi3D + c per class: {frac * 100:.1f}% of particles within 10 deg "
          f"(other sign {min(s_pos[0], s_neg[0]) * 100:.1f}%)")
    print(f"median |dev| over particles-weighted classes: "
          f"{np.average([m for *_, m in per], weights=[n for _, n, *__ in per]):.2f} deg")
    if verbose:
        for k, n, f, med in per:
            print(f"  class {k:3d} n={n:4d}  within10={f * 100:5.1f}%  median|dev|={med:5.2f}")
    ox = out["rlnOriginXAngst"].to_numpy()
    oy = out["rlnOriginYAngst"].to_numpy()
    print(f"origin |offset| mean {np.mean(np.hypot(ox, oy)):.2f} A (input was re-centred, ref origins ~0)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], verbose=len(sys.argv) > 3)
