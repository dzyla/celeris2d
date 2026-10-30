"""Every themed figure also writes x.plot.json for the dashboard's interactive plot: curves with their colours and
legend, images, and a picture of panels the format does not describe (map projections, colour bars).

    pytest -q tests/test_figjson.py
"""
import json
import os

import numpy as np

from celeris2d import plots3d


def test_plot_json_next_to_png(tmp_path):
    p = str(tmp_path / "fsc.png")
    plots3d.fsc_figure(p, [("soft sphere", np.linspace(1, 0, 50)), ("no mask", np.linspace(1, 0, 50) ** 2)],
                       300.0, 1.2, "test")
    j = json.load(open(str(tmp_path / "fsc.plot.json")))
    panel = next(q for q in j["panels"] if q.get("lines"))
    assert any(len(l["x"]) >= 49 for l in panel["lines"]) and panel["xticks"]["text"]     # shells 1 .. 49
    assert [i["label"].split()[0] for i in panel["legend"]["items"]] == ["soft", "no"]


def test_projection_panels_are_pictures(tmp_path):
    rng = np.random.default_rng(0)
    plots3d.views_figure(str(tmp_path / "views.png"), rng.uniform(-180, 180, 500),
                         np.degrees(np.arccos(rng.uniform(-1, 1, 500))), "test", None)
    j = json.load(open(str(tmp_path / "views.plot.json")))
    assert j["panels"] and all("raster" in q for q in j["panels"])               # the sky map and its colour bar
    assert all(q["raster"].startswith("data:image/png;base64,") for q in j["panels"])
    assert os.path.exists(str(tmp_path / "views.png"))
