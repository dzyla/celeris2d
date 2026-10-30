"""Every saved figure as data: `x.png` gets `x.plot.json`, from which the dashboard draws the same figure as an
interactive plot (zoom, hover, pan). Written by plotstyle.themed for the default theme, read from the matplotlib
figure itself after the figure function has drawn it, so no figure function needs its own export.

Format (version 1): {"version", "width", "height" (inches), "texts": figure-level texts, "panels": [...]} with per
panel (one matplotlib Axes) its position in figure fractions [x0, y0, x1, y1], title, axis labels, limits, scales,
explicit ticks (positions and labels: resolution axes label 1/A positions in A), and its marks:
    lines     x, y, color, width, dash, marker, size, label, alpha
    scatter   x, y, colors (one or per point), sizes, label
    bars      x, y (bottom), w, h, color
    fills     polygons (fill_between) x, y, color, alpha
    segments  line collections (tracks) as x / y lists with None between segments
    images    a downsampled PNG (data URI) with its extent, plus the values (z) of small images for hover
    texts     notes in axes fractions or data coordinates
Large series are thinned (lines to 4,000 points, scatters to 20,000).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os

import numpy as np

MAX_LINE, MAX_SCATTER, MAX_IMAGE, MAX_Z = 4000, 20000, 640, 160


def _hex(c) -> str | None:
    from matplotlib.colors import to_hex
    try:
        return to_hex(c, keep_alpha=False)
    except (ValueError, TypeError):
        return None


def _num(a, n_max: int):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = a[None]
    step = max(1, int(math.ceil(len(a) / n_max)))
    a = a[::step]
    return [None if not np.isfinite(v) else round(float(v), 6) for v in a]


def _thin(n: int, n_max: int) -> slice:
    return slice(None, None, max(1, int(math.ceil(n / n_max))))


def _png(rgba: np.ndarray) -> str:
    from PIL import Image
    a = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _image(im) -> dict | None:
    arr = im.get_array()
    if arr is None:
        return None
    arr = np.ma.asarray(arr)
    h, w = arr.shape[:2]
    f = max(1, int(math.ceil(max(h, w) / MAX_IMAGE)))
    sub = arr[::f, ::f]
    lower = getattr(im, "origin", "upper") == "lower"   # rows drawn bottom-up: the PNG holds them as seen
    if sub.ndim == 2:
        rgba = np.asarray(im.to_rgba(sub), float)    # its own colormap and limits
        out = dict(src=_png(rgba[::-1] if lower else rgba), shape=[h, w])
        if max(h, w) <= MAX_Z:                       # small (heatmaps): the values too, for hover
            z = np.asarray(arr.filled(np.nan), float)
            out["z"] = [[None if not np.isfinite(v) else round(float(v), 5) for v in row] for row in z]
            clim = im.get_clim()
            out["clim"] = [float(clim[0]), float(clim[1])]
    else:
        rgba = np.asarray(sub, float)
        if rgba.max() > 1.5:
            rgba = rgba / 255.0
        if rgba.shape[-1] == 3:
            rgba = np.concatenate([rgba, np.ones(rgba.shape[:2] + (1,))], -1)
        out = dict(src=_png(rgba[::-1] if lower else rgba), shape=[h, w])
    ext = im.get_extent()                              # left, right, bottom, top in data coordinates
    out.update(extent=[float(e) for e in ext], origin=getattr(im, "origin", "upper"))
    return out


def _axes(ax, fig) -> dict:
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.image import AxesImage
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    bb = ax.get_position()
    p = dict(pos=[round(bb.x0, 4), round(bb.y0, 4), round(bb.x1, 4), round(bb.y1, 4)],
             title=ax.get_title("left") or ax.get_title("center") or ax.get_title("right"),
             xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), xscale=ax.get_xscale(), yscale=ax.get_yscale(),
             xlim=[float(v) for v in ax.get_xlim()], ylim=[float(v) for v in ax.get_ylim()],
             axis_on=bool(ax.axison), lines=[], scatter=[], bars=[], fills=[], segments=[], images=[], texts=[])
    for k, get_t, get_l in (("x", ax.get_xticks, ax.get_xticklabels), ("y", ax.get_yticks, ax.get_yticklabels)):
        try:
            t = [float(v) for v in get_t()]
            lab = [x.get_text() for x in get_l()]
            if lab and any(lab) and len(lab) == len(t):
                p[k + "ticks"] = dict(v=t, text=lab)
        except Exception:                          # noqa: BLE001 - ticks are cosmetic
            pass
    p["grid_y"] = any(l.get_visible() for l in ax.yaxis.get_gridlines())
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if x.size == 0:
            continue
        lab = ln.get_label()
        p["lines"].append(dict(x=_num(x, MAX_LINE), y=_num(y, MAX_LINE), color=_hex(ln.get_color()),
                               width=float(ln.get_linewidth()), dash=str(ln.get_linestyle()),
                               marker=str(ln.get_marker()), size=float(ln.get_markersize()),
                               alpha=ln.get_alpha(), label="" if lab.startswith("_") else lab))
    for col in ax.collections:
        if not col.get_visible():
            continue
        lab = col.get_label() or ""
        lab = "" if lab.startswith("_") else lab
        if isinstance(col, PathCollection):
            off = np.asarray(col.get_offsets(), float)
            if not len(off):
                continue
            sl = _thin(len(off), MAX_SCATTER)
            fc = col.get_facecolors()
            colors = [_hex(c) for c in fc[sl]] if len(fc) > 1 else [_hex(fc[0])] if len(fc) else [None]
            arr = col.get_array()
            entry = dict(x=_num(off[sl, 0], MAX_SCATTER), y=_num(off[sl, 1], MAX_SCATTER), colors=colors,
                         sizes=[float(s) for s in np.atleast_1d(col.get_sizes())[:1]], label=lab,
                         alpha=col.get_alpha())
            if arr is not None and len(arr) == len(off):
                entry["values"] = _num(np.asarray(arr)[sl], MAX_SCATTER)
            p["scatter"].append(entry)
        elif isinstance(col, LineCollection):
            xs, ys = [], []
            segs = col.get_segments()
            for s in segs[_thin(len(segs), 3000)]:
                s = np.asarray(s, float)
                xs += [round(float(v), 4) for v in s[:, 0]] + [None]
                ys += [round(float(v), 4) for v in s[:, 1]] + [None]
            cs = col.get_colors()
            p["segments"].append(dict(x=xs, y=ys, color=_hex(cs[0]) if len(cs) else None,
                                      width=float(np.atleast_1d(col.get_linewidths())[0]), alpha=col.get_alpha()))
        elif isinstance(col, PolyCollection):
            fc = col.get_facecolors()
            for path in col.get_paths()[:20]:
                v = np.asarray(path.vertices, float)
                p["fills"].append(dict(x=_num(v[:, 0], MAX_LINE), y=_num(v[:, 1], MAX_LINE),
                                       color=_hex(fc[0]) if len(fc) else None,
                                       alpha=float(fc[0][3]) if len(fc) else col.get_alpha(), label=lab))
    rects = [r for r in ax.patches if isinstance(r, Rectangle) and r.get_visible()]
    if rects:
        p["bars"] = [dict(x=round(float(r.get_x()), 6), y=round(float(r.get_y()), 6), w=round(float(r.get_width()), 6),
                          h=round(float(r.get_height()), 6), color=_hex(r.get_facecolor()))
                     for r in rects[:5000]]
    for im in ax.get_images():
        if isinstance(im, AxesImage) and im.get_visible():
            d = _image(im)
            if d:
                p["images"].append(d)
    for t in ax.texts:
        if not isinstance(t, Text) or not t.get_visible() or not t.get_text().strip():
            continue
        x, y = t.get_position()
        frac = t.get_transform() == ax.transAxes
        p["texts"].append(dict(text=t.get_text(), x=float(x), y=float(y), ref="axes" if frac else "data",
                               color=_hex(t.get_color()), size=float(t.get_fontsize()), ha=t.get_ha(), va=t.get_va()))
    leg = ax.get_legend()
    p["legend"] = _legend(leg, fig) if leg is not None and leg.get_visible() else None
    return p


def _legend(leg, fig) -> dict | None:
    """A legend as its title, entries (label, colour, line or patch) and its box in figure fractions."""
    try:
        renderer = fig.canvas.get_renderer()
        bb = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        items = []
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for h, t in zip(handles, leg.get_texts()):
            c = None
            for get in ("get_color", "get_facecolor", "get_edgecolor"):
                if hasattr(h, get):
                    v = getattr(h, get)()
                    c = _hex(v[0] if isinstance(v, np.ndarray) and v.ndim == 2 and len(v) else v)
                    if c:
                        break
            items.append(dict(label=t.get_text(), color=c))
        title = leg.get_title().get_text() if leg.get_title() else ""
        return dict(title=title, items=items, box=[round(bb.x0, 4), round(bb.y0, 4), round(bb.x1, 4), round(bb.y1, 4)])
    except Exception:                                # noqa: BLE001 - a legend that cannot be read is left out
        return None


def figure_spec(fig) -> dict:
    _RENDER.clear()
    out = dict(version=1, width=float(fig.get_figwidth()), height=float(fig.get_figheight()),
               background=_hex(fig.get_facecolor()), texts=[], panels=[])
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            x, y = t.get_position()
            out["texts"].append(dict(text=t.get_text(), x=float(x), y=float(y), color=_hex(t.get_color()),
                                     size=float(t.get_fontsize()), weight=str(t.get_fontweight())))
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        try:
            if _needs_raster(ax):                    # map projections, meshes, colour bars: a picture of the panel
                r = _raster(ax, fig)
                if r:
                    out["panels"].append(r)
                    continue
            out["panels"].append(_axes(ax, fig))
        except Exception as e:                       # noqa: BLE001 - one odd panel must not lose the others
            out["panels"].append(dict(error=f"{type(e).__name__}: {e}"))
    return out


def _clean(o):
    """NaN / inf -> None (JSON has neither), numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _needs_raster(ax) -> bool:
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet
    if ax.name != "rectilinear":                     # aitoff / hammer / polar
        return True
    if any(isinstance(c, QuadMesh) for c in ax.collections):
        return True
    if any(isinstance(c, ContourSet) for c in getattr(ax, "containers", [])):
        return True
    return hasattr(ax, "_colorbar") or ax.get_label() == "<colorbar>"


_RENDER = {}


def _rendered(fig):
    """(RGBA buffer, renderer) of fig drawn once at its own dpi (the canvas savefig leaves may be at another dpi)."""
    key = id(fig)
    if key not in _RENDER:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        c = FigureCanvasAgg(fig)
        c.draw()
        _RENDER.clear()
        _RENDER[key] = (np.asarray(c.buffer_rgba()).copy(), c.get_renderer())
    return _RENDER[key]


def _raster(ax, fig) -> dict | None:
    """The panel (with its tick labels and title) cut out of the rendered figure, placed where it was."""
    try:
        buf, renderer = _rendered(fig)
        bb = ax.get_tightbbox(renderer)
        H, W = buf.shape[:2]
        x0, x1 = max(0, int(bb.x0)), min(W, int(np.ceil(bb.x1)))
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(np.ceil(H - bb.y0)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = buf[y0:y1, x0:x1].astype(np.float32) / 255.0
        return dict(raster=_png(crop), pos=[round(x0 / W, 4), round(1 - y1 / H, 4), round(x1 / W, 4), round(1 - y0 / H, 4)])
    except Exception:                                # noqa: BLE001
        return None


def json_path(png_path: str) -> str:
    return os.path.splitext(png_path)[0] + ".plot.json"


def write(fig, png_path: str):
    """fig's data next to its PNG (a failure is silent: the PNG is the figure; the JSON only makes it interactive)."""
    try:
        spec = _clean(figure_spec(fig))
        tmp = json_path(png_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(spec, f, separators=(",", ":"), allow_nan=False, default=lambda o: None)
        os.replace(tmp, json_path(png_path))
    except Exception:                                # noqa: BLE001
        pass
