"""Side-by-side gallery: celeris2d classes vs RELION classes (both sorted by population)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from celeris2d.star import read_star


def load(classes, star):
    imgs = mrcfile.read(classes).astype(np.float32)
    cls = read_star(star)["particles"]["rlnClassNumber"].to_numpy() - 1
    pops = np.bincount(cls, minlength=len(imgs))
    order = np.argsort(-pops)
    return imgs[order], pops[order]


def main(out_png, *pairs):
    rows = [(label, *load(c, s)) for label, c, s in zip(pairs[0::3], pairs[1::3], pairs[2::3])]
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 1.05, len(rows) * 1.25), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (label, imgs, pops) in enumerate(rows):
        for j, (im, n) in enumerate(zip(imgs, pops)):
            lo, hi = np.percentile(im, [0.5, 99.8])
            axes[i, j].imshow(im, cmap="gray", vmin=lo, vmax=hi)
            axes[i, j].set_title(str(n), fontsize=6, pad=1)
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.01, wspace=0.04, hspace=0.25)
    fig.savefig(out_png, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
