"""Region map of a synthetic micrograph (regions.py): a bright, flat support film on the left, a dark rim, ice with
dark particles and two contamination blobs on the right. With the shipped weights every particle stays pickable, the
film side and the blobs are not, and the map round-trips through its compact form. Runs in seconds on a GPU.
(Particles are placed at least 1.5 diameters from a blob: the network's edge / contamination halo around a sharp dark
blob reaches about a diameter beyond its boundary, and picks there are dropped - a known, intended property.)"""
import numpy as np
import torch

from celeris2d import regions as rg


def synthetic(apix=1.5, n=2048, seed=0, counts0=80.0):
    """Counts per pixel ~80 at 1.5 A/px: 2 % relative noise at the network's 8.5 A/px, as in a 40 e/A^2 K3 sum."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    counts = np.full((n, n), counts0, np.float32)
    x_edge = n * 0.45
    film = xx < x_edge - 40                              # support film: 12 % brighter, no particles
    counts[film] *= 1.12
    rim = np.abs(xx - x_edge) < 40                        # the hole rim: dark, rough
    counts[rim] *= 0.8 + 0.15 * rng.standard_normal((n, n)).astype(np.float32)[rim]
    pos = []
    r = 70.0 / apix                                      # 140 A particles
    blobs = ((300, n * 0.8, 90), (1500, n * 0.65, 120))  # contamination blobs (y, x, radius) in px
    for gy in np.arange(120, n - 120, 150):
        for gx in np.arange(x_edge + 150, n - 120, 150):
            y, x = gy + rng.integers(-30, 30), gx + rng.integers(-30, 30)
            if any(np.hypot(y - cy, x - cx) < cr + 3 * r for cy, cx, cr in blobs):
                continue
            d2 = (yy - y) ** 2 + (xx - x) ** 2
            counts -= 0.04 * counts0 * np.exp(-d2 / (2 * (0.5 * r) ** 2))      # 4 % contrast
            pos.append((x * apix, y * apix))
    for (cy, cx, cr) in blobs:
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        counts[d2 < cr * cr] *= 0.6
    mic = rng.poisson(np.clip(counts, 0.1, None)).astype(np.float32)
    return mic, np.array(pos), np.array([(cx * apix, cy * apix) for cy, cx, cr in blobs])


def test_region_map_synthetic():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mic, pos, blobs = synthetic()
    rm = rg.region_map(torch.as_tensor(mic, device=dev), 1.5, keep_input=True)
    assert rm.prob.shape[0] == len(rg.CLASSES) and abs(rm.apix - 2 * rg.WORK_APIX) < 1.0
    assert torch.allclose(rm.prob.sum(0), torch.ones_like(rm.prob[0]), atol=1e-4)
    ok = rm.pickable(pos, margin_a=70)
    assert ok.mean() >= 0.99, ok.mean()                   # no particle lost
    assert not rm.pickable(blobs, margin_a=70).any()      # the contamination blobs are not pickable
    # the film side is classified as film, the particle side as ice (and stays pickable)
    cls = rm.classes()
    h, w = cls.shape
    assert (cls[:, : int(w * 0.35)] == 1).mean() > 0.8, (cls[:, : int(w * 0.35)] == 1).mean()
    assert (cls[:, int(w * 0.55):] == 0).mean() > 0.7, (cls[:, int(w * 0.55):] == 0).mean()
    gy, gx = np.meshgrid((np.arange(h) + 0.5) * rm.apix_y, (np.arange(w) + 0.5) * rm.apix_x, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    keep = rm.pickable(grid, margin_a=70).reshape(h, w)
    assert keep[:, int(w * 0.55):].mean() > 0.85, keep[:, int(w * 0.55):].mean()
    area = rm.area()
    assert abs(sum(area.values()) - 1) < 1e-3 and area["film"] > 0.3, area
    c = rm.compact()
    assert c["cls"].dtype == np.uint8 and c["p_ice"].dtype == np.uint8 and c["cls"].shape == (h, w)
    assert np.array_equal(c["cls"], rm.classes())
    assert rm.pickable(np.zeros((0, 2))).shape == (0,) and rm.at(np.zeros((0, 2))).shape == (0, len(rg.CLASSES))
    n = sum(p.numel() for p in rg.load_model(device=dev).parameters())
    assert n < 100_000, n


def test_regions_figure(tmp_path):
    from celeris2d.plots_preprocess import regions_figure
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mic, pos, blobs = synthetic(n=1024)
    rm = rg.region_map(torch.as_tensor(mic, device=dev), 1.5, keep_input=True)
    ok = rm.pickable(pos, margin_a=70)
    path = str(tmp_path / "regions.png")
    regions_figure(path, rm.binned.cpu().numpy(), rm.binned_apix, rm.classes(), rm.area(), pos[ok], pos[~ok], 140.0,
                   title="synthetic", ref_xy_a=pos, ref_removed=~ok, p_ice=rm.prob[0].cpu().numpy())
    assert (tmp_path / "regions.png").stat().st_size > 10_000


def test_pick_job_region_filter(tmp_path):
    """The opt-in Pick hook: candidates on the film are dropped before thresholding, the region value is a pick
    column, the compact map is stored per micrograph, the overlay figure is drawn, and a rethreshold keeps the
    region decision."""
    import os
    from celeris2d import pipeline as pl
    from celeris2d import preprocess as pp
    from celeris2d.project import Project
    rng = np.random.default_rng(1)
    f = tmp_path / "raw" / "mov_000.tif"
    f.parent.mkdir(parents=True)
    f.write_bytes(rng.bytes(4096))
    prj = Project(str(tmp_path / "prj"))
    imp = prj.import_movies([str(f)])
    mc = pl.motion_job(prj, imp, pl.MotionSettings(apix=1.5, dose_total=40, figures="none"))
    mic, pos, blobs = synthetic(apix=1.5, n=2048, seed=1)
    mid = int(prj.movies()["id"].iloc[0])
    mv = prj.movie(mid)
    path = mc.per_movie("Micrographs", mv, ".mrc")
    prj.write_mrc(path, mic, 1.5, "float16")
    prj.add_micrograph(mc, mid, path, 1.5, mic.shape, pp.CTFFit(15000.0, 15000.0, 0.0, 6.0, 1.0, {}))
    pl.finish(prj, mc)
    pj = pl.pick_job(prj, mc, pl.PickSettings(diameter_a=140.0, figures=1, region_filter=True))
    pl.run(prj, [pj])
    z = prj.picks(pj, mid)
    assert "region" in z and z["region"].shape == z["score"].shape and (z["region"] <= 1).all()
    assert not z["keep"][z["region"] < rg.MIN_ICE].any()
    on_film = z["xy_a"][:, 0] < 0.45 * 2048 * 1.5 - 200
    assert z["region"][on_film].mean() < z["region"][~on_film].mean()
    from celeris2d import picking as pk
    got, hit = pk.match(z["xy_a"][z["keep"]], pos, 35.0)
    assert hit.mean() > 0.95, hit.mean()
    c = rg.load_compact(pj.per_movie("Regions", mv, ".npz"))
    assert c["cls"].dtype == np.uint8 and c["area"].sum() > 0.99 and list(c["classes"]) == list(rg.CLASSES)
    assert os.path.exists(pj.figure(mv)[:-4] + "_regions.png")
    rt = pl.rethreshold(prj, pj, threshold=0.0)
    z2 = prj.picks(rt, mid)
    assert not z2["keep"][z2["region"] < rg.MIN_ICE].any() and z2["keep"].sum() >= z["keep"].sum()
