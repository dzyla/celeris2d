"""Curate: fixed limits, automatic drift limit (median + k MAD) only with motion_outliers, missing values never reject,
reasons named."""
import numpy as np
import pandas as pd

from celeris2d.curate import CurateSettings, judge, limits


def test_curate_limits():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(dict(movie_id=np.arange(n), stem=[f"m{i}" for i in range(n)],
                           ctf_fit_a=rng.uniform(3, 6, n), defocus_a=rng.uniform(8000, 25000, n),
                           astigmatism_a=rng.uniform(0, 500, n), ctf_score=np.full(n, np.nan),
                           ctf_tilt_deg=np.full(n, np.nan), drift_a=rng.normal(10, 1, n), local_rms_a=rng.normal(0.6, 0.05, n)))
    df.loc[0, "ctf_fit_a"] = 12.0                 # poor CTF fit
    df.loc[1, "defocus_a"] = 1500.0               # near focus
    df.loc[2, "drift_a"] = 60.0                   # drift outlier (automatic limit)
    df.loc[3, "astigmatism_a"] = 3000.0
    assert limits(CurateSettings(), df)["drift_a"] == [None, None]          # motion limits are opt-in
    lim = limits(CurateSettings(motion_outliers=True), df)
    assert 10 < lim["drift_a"][1] < 20 and lim["ctf_score"] == [None, None]
    ok, why = judge(df, lim)
    assert not ok[:4].any() and ok[4:].all()
    assert why[0] == "ctf_fit_a" and why[1] == "defocus_a" and why[2] == "drift_a" and why[3] == "astigmatism_a"
    lim2 = limits(CurateSettings(max_ctf_fit_a=0, max_drift_a=100), df)
    assert judge(df, lim2)[0][[0, 2]].all()        # no CTF-fit limit, a fixed drift limit
