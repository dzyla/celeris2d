"""Region map of a micrograph: where picking is worth trying.

A tiny U-Net (RegionNet, 91k parameters, weights in regions.pt next to this file) segments a micrograph binned
to ~8.5 A/px into five regions at ~17 A/px:

    ice            pickable ice: the region where 2D-classified / refined particles were found
    film           flat region without particles, brighter than the particle ice (support film with thin ice, or
                   ice-free)
    thick          markedly darker region (thick ice, dark support, grid bar)
    edge           high-variance band: hole rims, cracks, folds, crystalline ice, the micrograph's border stripes
    contamination  compact dark blobs (ethane, ice crystals, aggregates)

Labels were never drawn by hand. They come from the user's own processing (scripts/train_regions.py): `ice` is a
disk around every particle that survived 2D classification (Select 006) or 3D refinement (Refine3D 040) of the
dz_2025_10 project; the other four are the strong tails of two physical features of the binned micrograph - the local
mean intensity relative to the level at the confirmed particles (film: > +12 % brighter, thick: < -14 % darker,
contamination: a dark compact blob) and the particle-scale local standard deviation relative to the micrograph's noise
floor (edge: > 2.3x) - and only where no confirmed particle is within two diameters. Everything else is unlabelled;
the network interpolates. The cut-offs are the 99.9th / 0.1st / 99th percentiles of the same features at that
dataset's confirmed particles, so by construction almost no real particle is labelled non-ice (scripts/eval_regions.py
measures the recall).

Trained on dz_2025_10 (350 micrographs) + half of the CryoPPP EMPIAR-10017 and -10005 micrographs (ground-truth
coordinates as the confirmed particles); held-out: 150 dz micrographs (4,639 refined particles), 42 of 10017
(25,642 particles), 14 of 10005 (2,609). At the default decision (MIN_ICE 0.1): recall of the refined particles
99.87 %, 12.7 % of the blob picks removed (16.0 % of the 2D-rejected picks vs 0.7 % of the 2D-selected ones),
12.5 % of the area; 10017 99.99 %, 10005 100 %. Within one micrograph the pickable ice is darker/textured and the
empty film brighter in these datasets; carbon can also be dark and textured (the 10017 corner), which the network
does not separate from crowded ice - the class *names* are appearance-based; `ice` vs not is what is validated.
Known failures: a dark compact blob gets an edge / contamination halo of about a particle diameter (picks next to
contamination go); "thick" is over-called on dark patches inside crowded ice (the contrast gate keeps those picks);
a micrograph that is mostly bright film with a darker particle region on the side may have that region called thick
or edge (gated for picking, not in the overlay); uniform empty ice is "ice" (nothing to compare against).

    rm = region_map(mic, apix)                      # RegionMap: prob (5, h, w) at ~17 A/px, 1.4 ms on an RTX PRO 4000
    ok = rm.pickable(cand.xy_a, margin_a=70)        # bool per candidate (rm.ice_margin: the value, pick column)
    rm.area()                                       # {"ice": 0.71, "film": 0.2, ...}
    np.savez_compressed(path, **rm.compact())       # uint8 class map + P(ice), for the dashboard overlay
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ("ice", "film", "thick", "edge", "contamination")
WORK_APIX = 8.5              # input pixel size (A); the map is at 2x this
WEIGHTS = os.path.join(os.path.dirname(__file__), "regions.pt")


# ------------------------------------------------------------------------------------------------- the network
class _Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(i, o, 3, padding=1), nn.Conv2d(o, o, 3, padding=1)
        self.n1, self.n2 = nn.BatchNorm2d(o), nn.BatchNorm2d(o)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(x)))


class RegionNet(nn.Module):
    """2-channel input (fractional intensity, noise-normalised image) at WORK_APIX; logits (5) at 2x WORK_APIX.
    Encoder widths (10, 20, 32, 40) with 3 poolings: a receptive field of ~100 input px (~850 A) at the output.
    BatchNorm (folded at inference) rather than GroupNorm: the latter costs 10x the convolutions at this size."""

    def __init__(self, widths=(10, 20, 32, 40), n_in=2, n_out=len(CLASSES)):
        super().__init__()
        self.widths = tuple(widths)
        w = widths
        self.e0, self.e1, self.e2, self.e3 = _Block(n_in, w[0]), _Block(w[0], w[1]), _Block(w[1], w[2]), _Block(w[2], w[3])
        self.d2, self.d1 = _Block(w[3] + w[2], w[2]), _Block(w[2] + w[1], w[1])
        self.head = nn.Conv2d(w[1], n_out, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(F.avg_pool2d(e0, 2))
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)


_models: dict = {}


def load_model(path: str | None = None, device="cuda") -> RegionNet:
    """The shipped network (or another weights file), cached per device, in eval mode."""
    path = path or WEIGHTS
    key = (path, str(device))
    if key not in _models:
        if not os.path.exists(path):
            raise FileNotFoundError(f"region network weights not found: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        m = RegionNet(sd.get("widths", (10, 20, 32, 40)))
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        m = m.to(device).eval()
        _models[key] = m
    return _models[key]


# ---------------------------------------------------------------------------------------------------- the input
def bin_micrograph(mic: torch.Tensor, apix: float, work_apix: float = WORK_APIX) -> tuple[torch.Tensor, float, float]:
    """Fourier-crop a micrograph to about work_apix per pixel (even sizes): (image, apix_y, apix_x)."""
    ny, nx = mic.shape
    my = max(16, int(round(ny * apix / work_apix / 2)) * 2)
    mx = max(16, int(round(nx * apix / work_apix / 2)) * 2)
    if my >= ny or mx >= nx:
        return mic.float(), apix, apix
    Fm = torch.fft.rfft2(mic.float(), norm="forward")
    hy, hx = my // 2, mx // 2 + 1
    G = torch.cat([Fm[:hy, :hx], Fm[-hy:, :hx]], 0)
    return torch.fft.irfft2(G, s=(my, mx), norm="forward"), apix * ny / my, apix * nx / mx


def net_input(binned: torch.Tensor, dark: bool = True) -> torch.Tensor:
    """(1, 2, h, w): fractional intensity (x10, mass negative) and the noise-normalised image (clipped to +-6)."""
    x = binned.float()
    mean = x.mean()
    sign = 1.0 if dark else -1.0
    if mean.abs() < 1e-6 * (x.std() + 1e-12):          # zero-mean input: no transmission information
        a = torch.zeros_like(x)
    else:
        a = (x / mean - 1) * 10 * (1.0 if float(mean) > 0 else -1.0)
    flat = x.flatten()
    k = max(1, flat.numel() // 200000)
    sub = flat[::k]
    med = sub.median()
    mad = (sub - med).abs().median() * 1.4826 + 1e-12
    b = ((x - med) / mad).clamp(-6, 6)
    return torch.stack([a * sign, b * sign])[None]


# ----------------------------------------------------------------------------------------------------- the map
# The decision (RegionMap.bad): a candidate is dropped when the probability of not-ice within half a diameter exceeds
# 1 - MIN_ICE. Two optional contrast gates make a class count only where the physics agrees: thick only where the
# local intensity is THICK_CONTRAST below the ice level, film only where it is FILM_CONTRAST above it (None: off).
# The film gate is off by default: on dz_2025_10 most of what the network removes is flat, empty ice at the same
# intensity as the particle ice (27 % of the 2D-rejected picks vs 1.5 % of the selected ones at MIN_ICE 0.2, held-out),
# and the gate would keep nearly all of it (removal 22 % -> 4 %). The price is a dataset of uniformly flat,
# low-contrast particles (EMPIAR-10005 held-out: 99.3 % recall at MIN_ICE 0.2, 99.9 % at 0.1); FILM_CONTRAST = 0.03
# makes that 100 %. MIN_ICE 0.1 is the default for that reason.
MIN_ICE = 0.1
FILM_CONTRAST: float | None = None
THICK_CONTRAST: float | None = 0.04


@dataclass
class RegionMap:
    prob: torch.Tensor          # (5, h, w) class probabilities on the device
    apix_y: float               # pixel size of the map (A), rows
    apix_x: float               # columns
    shape_a: tuple              # micrograph extent (height_a, width_a)
    fi: torch.Tensor | None = None   # (h, w) smoothed fractional intensity (mass negative; 0 = micrograph mean)
    binned: torch.Tensor | None = None   # the network's input image (keep_input=True), for figures
    binned_apix: float = 0.0

    def gated(self) -> torch.Tensor:
        """(5, h, w) probabilities after the contrast gates (module constants). Two reference levels of the local
        intensity: the P(ice)-weighted mean (the micrograph mean when there is no ice) and the micrograph's median
        (its dominant region); thick counts only where the intensity is THICK_CONTRAST below both, film only where it
        is FILM_CONTRAST above both, the rest of their probability goes to ice. Ice with particles that is darker than the rest of the micrograph (a bright film on one side) is then
        not "thick", and a uniformly flat micrograph is not "film". Everything downstream (classes, area, compact,
        pickable) uses these; `prob` is the network's raw output."""
        p = self.prob
        if self.fi is None or (FILM_CONTRAST is None and THICK_CONTRAST is None):
            return p
        w = p[0]
        ice = (self.fi * w).sum() / w.sum() if float(w.sum()) > 0.02 * w.numel() else self.fi.new_zeros(())
        med = self.fi.flatten()[::max(1, self.fi.numel() // 100000)].median()    # the micrograph's dominant level
        q = p.clone()
        if FILM_CONTRAST is not None:                    # film: brighter than both the ice and the dominant level
            keep = torch.sigmoid((self.fi - torch.maximum(ice, med) - FILM_CONTRAST) / 0.01)
            q[0], q[1] = q[0] + p[1] * (1 - keep), p[1] * keep
        if THICK_CONTRAST is not None:                   # thick: darker than both
            keep = torch.sigmoid((torch.minimum(ice, med) - self.fi - THICK_CONTRAST) / 0.01)
            q[0], q[2] = q[0] + p[2] * (1 - keep), p[2] * keep
        return q

    def bad(self) -> torch.Tensor:
        """Probability of not-ice for the decision (after the gates)."""
        return 1 - self.gated()[0]

    @property
    def apix(self) -> float:
        return 0.5 * (self.apix_y + self.apix_x)

    def at(self, xy_a: np.ndarray, prob: torch.Tensor | None = None) -> np.ndarray:
        """(n, 5) gated probabilities at positions (A), bilinear."""
        xy = np.asarray(xy_a, np.float32).reshape(-1, 2)
        prob = self.gated() if prob is None else prob
        if len(xy) == 0:
            return np.zeros((0, prob.shape[0]), np.float32)
        h, w = prob.shape[-2:]
        g = torch.as_tensor(xy, device=self.prob.device)
        gx = (g[:, 0] / self.apix_x - 0.5) / (w - 1) * 2 - 1       # pixel i is centred at (i + 0.5) * apix
        gy = (g[:, 1] / self.apix_y - 0.5) / (h - 1) * 2 - 1
        grid = torch.stack([gx, gy], 1)[None, :, None]              # (1, n, 1, 2)
        v = F.grid_sample(prob[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
        return v[0, :, :, 0].T.cpu().numpy()


    def ice_margin(self, xy_a: np.ndarray, margin_a: float = 0.0) -> np.ndarray:
        """1 - bad() at each position, taken as its maximum over a square of half-width margin_a (so a particle at a
        region border keeps its ice value): the pick column "region"."""
        xy = np.asarray(xy_a, np.float32).reshape(-1, 2)
        if len(xy) == 0:
            return np.zeros(0, np.float32)
        bad = self.bad()
        r = int(round(margin_a / self.apix))
        if r > 0:
            bad = -F.max_pool2d(-bad[None, None], 2 * r + 1, 1, r)[0, 0]
        return self.at(xy, torch.stack([1 - bad]))[:, 0]

    def pickable(self, xy_a: np.ndarray, margin_a: float = 0.0, max_bad: float = 1 - MIN_ICE) -> np.ndarray:
        """Candidates kept: the probability of not-ice within margin_a of the position is at most max_bad."""
        return self.ice_margin(xy_a, margin_a) >= 1 - max_bad

    def classes(self) -> np.ndarray:
        return self.gated().argmax(0).to(torch.uint8).cpu().numpy()

    def area(self) -> dict:
        """Fraction of the micrograph per class (argmax)."""
        c = self.gated().argmax(0)
        n = c.numel()
        return {k: round(float((c == i).sum()) / n, 4) for i, k in enumerate(CLASSES)}

    def compact(self) -> dict:
        """uint8 class map, P(ice) (0-255), pixel size, area fractions: ~20 kB per micrograph in an .npz."""
        return dict(cls=self.classes(), p_ice=(self.gated()[0] * 255).round().to(torch.uint8).cpu().numpy(),
                    apix=np.array([self.apix_y, self.apix_x], np.float32),
                    shape_a=np.array(self.shape_a, np.float32), area=np.array([self.area()[k] for k in CLASSES], np.float32),
                    classes=np.array(CLASSES))


def region_map(mic: torch.Tensor, apix: float, dark: bool = True, model: RegionNet | None = None,
               weights: str | None = None, keep_input: bool = False):
    """Region probabilities of a micrograph (any pixel size; (ny, nx) tensor, on the GPU preferably)."""
    dev = mic.device
    model = model or load_model(weights, dev)
    b, ay, ax = bin_micrograph(mic, apix)
    x = net_input(b, dark)
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits[0], 0)
    rm = RegionMap(p, ay * b.shape[0] / p.shape[1], ax * b.shape[1] / p.shape[2],
                   (mic.shape[0] * apix, mic.shape[1] * apix), fractional_intensity(x, p.shape[-2:]))
    if keep_input:
        rm.binned, rm.binned_apix = b, 0.5 * (ay + ax)
    return rm


def fractional_intensity(x: torch.Tensor, size) -> torch.Tensor:
    """Channel 0 of the network input (fractional intensity x10) smoothed over ~16 input px, at the map size."""
    a = F.avg_pool2d(x[:, :1], 2)
    a = F.avg_pool2d(F.pad(a, (4, 4, 4, 4), mode="replicate"), 9, 1)
    return F.interpolate(a, size=size, mode="bilinear", align_corners=False)[0, 0] / 10


def load_compact(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def pickable(xy_a: np.ndarray, rm: RegionMap, margin_a: float = 0.0, max_bad: float = 1 - MIN_ICE) -> np.ndarray:
    """Module-level alias of RegionMap.pickable."""
    return rm.pickable(xy_a, margin_a, max_bad)
