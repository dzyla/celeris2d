"""Run report figures: class gallery and per-iteration / per-class statistics."""
from __future__ import annotations

import numpy as np

from .plotstyle import STATUS, T, plt, themed, title as _title_block, top


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _font(size, bold=False):
    from matplotlib import font_manager
    from PIL import ImageFont
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


@themed
def gallery(path, imgs, pops, apix, conf=None, title="Celeris2D", ncol=None, tile=200, gap=8, marks=None):
    """Class mosaic (Pillow, fixed layout): rounded tiles sorted by size, class badge, count/share on a
    gradient strip, confidence bar along the bottom edge."""
    from PIL import Image, ImageDraw

    bg, ink, ink2, track = _rgb(T.plane), _rgb(T.ink), _rgb(T.ink2), _rgb(T.line)
    blue = _rgb(T.series[0])
    order = np.argsort(-pops)
    K = len(order)
    ncol = ncol or (5 if K <= 10 else 8 if K <= 32 else 10)
    nrow = int(np.ceil(K / ncol))
    head, margin = 74, 16
    W = 2 * margin + ncol * tile + (ncol - 1) * gap
    H = head + margin + nrow * tile + (nrow - 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    d.text((margin, 16), title, font=_font(22, True), fill=ink)
    sub = f"{K} classes  ·  {int(pops.sum()):,} particles  ·  {apix:.2f} Å/px  ·  sorted by size"
    if marks is not None:
        sub += f"  ·  {int(np.sum(marks))} good (green) / {int(len(marks) - np.sum(marks))} bad (dimmed)"
    if conf is not None:
        sub += "  ·  conf / blue bar = mean max posterior of the class's particles (1 = unambiguous)"
    d.text((margin, 46), sub, font=_font(13), fill=ink2)

    f_badge, f_n, f_pct = _font(13, True), _font(14, True), _font(12)
    radius = 12
    rmask = Image.new("L", (tile, tile), 0)
    ImageDraw.Draw(rmask).rounded_rectangle((0, 0, tile - 1, tile - 1), radius, fill=255)
    strip_h = 34
    grad = np.linspace(0, 200, strip_h).astype(np.uint8)[:, None].repeat(tile, 1)
    for i, k in enumerate(order):
        r, c = divmod(i, ncol)
        x0, y0 = margin + c * (tile + gap), head + r * (tile + gap)
        im = imgs[k].astype(np.float32)
        lo, hi = np.percentile(im, [0.5, 99.7])
        u8 = (np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
        t = Image.fromarray(u8).resize((tile, tile), Image.LANCZOS).convert("RGB")
        # bottom gradient strip for legible text over the image
        shade = Image.new("RGB", (tile, strip_h), bg)
        t.paste(shade, (0, tile - strip_h), Image.fromarray(grad))
        td = ImageDraw.Draw(t)
        label = str(k + 1)
        bw = td.textlength(label, font=f_badge) + 14
        td.rounded_rectangle((8, 8, 8 + bw, 30), 8, fill=bg)
        td.text((15, 11), label, font=f_badge, fill=ink)
        td.text((10, tile - 26), f"{int(pops[k]):,}", font=f_n, fill=ink)
        pct = f"{pops[k] / pops.sum() * 100:.1f}%"
        td.text((tile - 10 - td.textlength(pct, font=f_pct), tile - 24), pct, font=f_pct, fill=ink2)
        if conf is not None:
            cv = f"conf {float(conf[k]):.2f}"
            cw = td.textlength(cv, font=f_pct) + 12
            td.rounded_rectangle((tile - 8 - cw, 8, tile - 8, 30), 8, fill=bg)
            td.text((tile - 2 - cw, 12), cv, font=f_pct, fill=ink2)
            td.rectangle((0, tile - 4, tile, tile), fill=track)
            td.rectangle((0, tile - 4, int(tile * float(np.clip(conf[k], 0, 1))), tile), fill=blue)
        if marks is not None and not marks[k]:
            t = Image.blend(t, Image.new("RGB", t.size, bg), 0.55)
        canvas.paste(t, (x0, y0), rmask)
        if marks is not None and marks[k]:
            d.rounded_rectangle((x0 - 3, y0 - 3, x0 + tile + 2, y0 + tile + 2), radius + 3, outline=_rgb(STATUS["good"]),
                                width=3)
    canvas.save(path, optimize=True)


@themed
def stats(path, history, pops, conf, title=""):
    """history: list of per-iteration dicts from Classifier2D; pops/conf: per-class size and mean max-posterior."""
    BLUE, ORANGE, INK = T.series[0], T.series[1], T.ink
    it = np.array([h["it"] for h in history])
    full = np.array([h["n"] for h in history]) == max(h["n"] for h in history)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.9))
    fig.subplots_adjust(left=0.06, right=0.98, top=top(fig, 0.7), bottom=0.08, wspace=0.25, hspace=0.45)

    a = ax[0, 0]
    lim = [h["res"] for h in history]
    est = [h["est_res"] for h in history]
    a.plot(it, lim, color=BLUE, lw=2, marker="o", ms=3, label="E-step limit")
    a.plot(it, est, color=ORANGE, lw=2, marker="o", ms=3, label="SSNR=1 estimate")
    a.text(it[-1], lim[-1], f" {lim[-1]:.1f} Å", color=INK, va="center", fontsize=7)
    a.text(it[-1], est[-1], f" {est[-1]:.1f} Å", color=INK, va="center", fontsize=7)
    a.invert_yaxis()
    a.set(title="Resolution (finer ↑)", xlabel="iteration", ylabel="Å")
    a.legend(loc="lower right")

    a = ax[0, 1]
    a.plot(it, [h["maxprob"] for h in history], color=BLUE, lw=2, marker="o", ms=3)
    a.set(title="Alignment confidence: mean max posterior", xlabel="iteration", ylim=(0, 1.02))

    a = ax[0, 2]
    ch = np.maximum([h["changed"] * 100 for h in history], 1e-2)
    a.semilogy(it[1:], ch[1:], color=BLUE, lw=2, marker="o", ms=3)
    a.set(title="Convergence: particles changing class", xlabel="iteration", ylabel="% (log)")

    a = ax[1, 0]
    t = np.array([h["time"] for h in history])
    if (~full).any():
        a.bar(it[~full], t[~full], color=BLUE, width=0.8, label="online-EM subset")
    a.bar(it[full], t[full], color=ORANGE, width=0.8, label="full pass")
    a.set(title=f"Time per iteration (total {t.sum():.0f} s)", xlabel="iteration", ylabel="s")
    a.legend(loc="upper left")

    order = np.argsort(-pops)
    x = np.arange(len(order))
    a = ax[1, 1]
    a.bar(x, pops[order], color=BLUE, width=0.8)
    a.set(title="Class sizes", xlabel="class rank (by size)", ylabel="particles")
    a = ax[1, 2]
    a.bar(x, conf[order], color=BLUE, width=0.8)
    a.set(title="Per-class mean max posterior", xlabel="class rank (by size)", ylim=(0, 1.02))

    from matplotlib.ticker import MaxNLocator
    for a in ax.flat:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.yaxis.grid(True, color=T.grid, lw=0.8)
    _title_block(fig, "2D classification", title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
