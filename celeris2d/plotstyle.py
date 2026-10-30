"""One look for every figure the program writes, in a dark (default) and a light theme.

Every figure function takes the output path first and is decorated with @themed: it is drawn once per theme in
THEMES and saved as <name>.png (dark) and <name>.light.png (light); the dashboards pick the variant matching their
theme. Inside a figure, colours come from the current theme T (T.ink, T.series[0], T.density, ...), never literals.

Palette: the validated reference palette (categorical slots 1-8, light and dark steps; the first three validate
all-pairs, so a figure uses at most three series per axis), sequential blue ramp, blue/red diverging pair with a
gray midpoint, fixed status colours. The dark surface is the dashboard's panel colour.
"""
from __future__ import annotations

import functools
import os
import threading
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DEFAULT = "dark"
# themes written for every figure (CELERIS2D_THEMES=dark writes the dark one only)
THEMES = tuple(t for t in os.environ.get("CELERIS2D_THEMES", "dark,light").split(",") if t in ("dark", "light")) \
    or (DEFAULT,)
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")
GRAY = LinearSegmentedColormap.from_list("em", ["#000000", "#ffffff"])

_TOKENS = {
    "dark": dict(surface="#16181f", plane="#0e0f13", well="#1d2029", ink="#f0efec", ink2="#a0a3ad", ink3="#767a86",
                 line="#2a2e3a", grid="#2a2e3a", axis="#383c48",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
                 ramp=["#1d2029", "#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
                 diverging=["#3987e5", "#2a4f7f", "#383835", "#7f3434", "#e66767"]),
    "light": dict(surface="#fcfcfb", plane="#f9f9f7", well="#f0efec", ink="#0b0b0b", ink2="#52514e", ink3="#898781",
                  line="#c3c2b7", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
                  ramp=["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
                  diverging=["#2a78d6", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948"]),
}


def _theme(name: str) -> SimpleNamespace:
    t = SimpleNamespace(name=name, **_TOKENS[name])
    t.density = LinearSegmentedColormap.from_list(f"density_{name}", t.ramp)        # sequential: count, magnitude
    t.diverging_cmap = LinearSegmentedColormap.from_list(f"div_{name}", t.diverging)  # blue below, red above a centre
    return t


T = _theme(DEFAULT)            # current theme: read at draw time
_lock = threading.RLock()


def apply(name: str = DEFAULT):
    """Make `name` the current theme (T) and set matplotlib's defaults to it."""
    global T
    T.__dict__.update(_theme(name).__dict__)
    plt.rcParams.update({
        "figure.facecolor": T.surface, "axes.facecolor": T.surface, "savefig.facecolor": T.surface,
        "axes.edgecolor": T.axis, "axes.labelcolor": T.ink2, "axes.titlecolor": T.ink, "text.color": T.ink,
        "xtick.color": T.ink3, "ytick.color": T.ink3, "xtick.labelcolor": T.ink2, "ytick.labelcolor": T.ink2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": T.grid,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.prop_cycle": matplotlib.cycler(color=T.series[:3]),
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "patch.linewidth": 0,
        "image.interpolation": "lanczos", "savefig.dpi": 110,
    })


def variant(path: str, name: str) -> str:
    """File of a figure in a theme: x.png (the default theme), x.light.png / x.dark.png (the others)."""
    if name == DEFAULT:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.{name}{ext}"


def themed(fn):
    """Draw fn(path, ...) once per theme in THEMES (current theme set first); returns the default theme's result."""
    @functools.wraps(fn)
    def wrap(path, *a, **k):
        out = None
        with _lock:
            try:
                for name in sorted(THEMES, key=lambda n: n != DEFAULT):
                    apply(name)
                    r = fn(variant(path, name), *a, **k)
                    if name == DEFAULT or out is None:
                        out = r
            finally:
                apply(DEFAULT)
                plt.close("all")
        return out
    return wrap


def title(fig, text: str, sub: str = ""):
    """Figure title and subtitle at fixed distances (inches) from the top, whatever the figure height."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.1 / h, text, fontsize=12, fontweight="bold", color=T.ink, va="top")
    if sub:
        fig.text(0.012, 1 - 0.36 / h, sub, fontsize=9, color=T.ink2, va="top")


def top(fig, inches: float = 0.75) -> float:
    """Figure fraction below a title block of this height."""
    return 1 - inches / fig.get_figheight()


def warnings_text(fig, warnings: list[str], y: float = 0.006):
    """Warnings along the bottom of a figure, in the status 'serious' colour with a marker."""
    if warnings:
        fig.text(0.012, y, "\n".join("⚠ " + w for w in warnings), fontsize=8.5, color=STATUS["serious"], va="bottom")


def note(ax, text: str, loc: str = "upper right"):
    """A small text box in an axes corner (numbers that go with the plot)."""
    x = 0.98 if "right" in loc else 0.02
    y = 0.96 if "upper" in loc else 0.04
    ax.text(x, y, text, transform=ax.transAxes, ha="right" if "right" in loc else "left",
            va="top" if "upper" in loc else "bottom", fontsize=8.5, color=T.ink2,
            bbox=dict(boxstyle="round,pad=0.35", fc=T.well, ec="none"))


def grid_y(ax):
    ax.yaxis.grid(True, color=T.grid, lw=0.8)
    ax.set_axisbelow(True)


def res_axis(ax, box_a: float | None = None, nyquist: float | None = None, label: str = "resolution (Å)"):
    """x axis in spatial frequency (1/Å) labelled in Å; nyquist in 1/Å."""
    ticks = [r for r in (50, 30, 20, 15, 10, 8, 6, 5, 4, 3.5, 3, 2.5, 2.2, 2) if nyquist is None or 1 / r <= nyquist * 1.001]
    ax.set_xticks([1 / r for r in ticks], [f"{r:g}" for r in ticks])
    if nyquist:
        ax.set_xlim(0, nyquist)
    ax.set_xlabel(label)


def norm_img(im: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    a, b = np.percentile(im, [lo, hi])
    return np.clip((im - a) / max(b - a, 1e-12), 0, 1)


def image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


apply(DEFAULT)
