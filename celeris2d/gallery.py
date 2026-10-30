"""Data the dashboard shows directly (not figures): class galleries to choose classes from, and maps for the 3D
viewer.

class_gallery(project, job): the classes of a Class2D job (its class averages) or of a HeteroRefine / Class3D job
(three projections of each class map), largest first, with particles, share and resolution; for a Select job, the
classes of its source and which it keeps.

map_views(path): projections and central sections of a map, as pictures for the job page.

volume_payload(path, size, color): a map downsampled to <= size^3 as 8-bit voxels for the browser's raymarcher -
bytes: uint32 header length, JSON header (shape, apix, value range, suggested isosurface levels - "clean" the
default, _clean_level - colour range), with dust (small isolated pieces of noise at that level) hidden unless dust=False;
the voxels (z, y, x order, uint8), then optionally a colour volume (e.g. local resolution) on the same grid.
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct

import numpy as np

from .project import CLASS_JOBS, Project


def _thumb(img: np.ndarray, px: int = 128) -> str:
    from PIL import Image
    a = np.asarray(img, np.float32)
    h, w = a.shape
    yy, xx = np.ogrid[:h, :w]
    inside = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (0.45 * min(h, w)) ** 2     # the particle circle sets the scale:
    lo, hi = np.percentile(a[inside], [1, 99.5])                                 # masked-out corners stay dark
    a = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if max(im.size) > px:
        im = im.resize((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _read_mrc(path: str):
    import mrcfile
    with mrcfile.open(path, permissive=True) as f:
        return np.asarray(f.data, np.float32), float(f.voxel_size.x)


def class_gallery(project: Project, job) -> dict:
    j = project.job(job)
    chosen, waiting = None, False
    if j.type == "Select":
        chosen = [int(k) for k in (j.params.get("classes") or [])]
        waiting = j.status == "waiting"
        src = project.job(j.params["source"])
    else:
        src = j
    summ = project.summary_of(src)
    out = dict(job=j.name, source=src.name, waiting=waiting, chosen=chosen, classes=[])
    t = project.table(src, "particles").read(["cls"]) if src.type in ("Class2D", *CLASS_JOBS) else None
    if src.type == "Class2D":
        path = src.find("classes.mrcs")
        if not path:
            raise ValueError(f"{src.name} has no class averages")
        imgs, apix = _read_mrc(path)
        pops = np.bincount(np.asarray(t["cls"]).astype(int)[np.asarray(t["cls"]) >= 0], minlength=len(imgs))
        n = max(int(pops.sum()), 1)
        out.update(kind="2d", apix=apix, particles=n)
        for k in np.argsort(-pops):
            out["classes"].append(dict(cls=int(k) + 1, n=int(pops[k]), share=round(float(pops[k]) / n, 4),
                                       images=[_thumb(imgs[k])]))
    elif src.type in CLASS_JOBS:
        rows = {int(r["cls"]): r for r in summ.get("classes", [])}
        out.update(kind="3d", particles=int(summ.get("particles", 0)))
        order = sorted(rows, key=lambda c: -rows[c].get("particles", 0))
        for c in order:
            r = rows[c]
            p = src.find(f"class{c:02d}_map_nomask.mrc") or src.find(f"class{c:02d}_map.mrc")
            ims = []
            if p and r.get("particles", 0) > 0:
                v, _ = _read_mrc(p)
                ims = [_thumb(v.sum(a)) for a in (0, 1, 2)]
            out["classes"].append(dict(cls=c, n=int(r.get("particles", 0)), share=r.get("share"),
                                       res=r.get("resolution_solvent_mask_a") or r.get("resolution_a"),
                                       res_sharpened=r.get("sharpened_resolution_a"), images=ims))
    else:
        raise ValueError(f"{src.name} is not a classification (Class2D, HeteroRefine, Class3D)")
    return out


def map_views(path: str, px: int = 384) -> dict:
    """Projections along z, y, x and central sections of a map (as in plots3d.map_views_figure), large PNG data URIs
    for the job page's preview before the 3D viewer is opened."""
    import mrcfile
    with mrcfile.mmap(path, permissive=True) as f:
        v = np.asarray(f.data, np.float32)
        apix = float(f.voxel_size.x)
    b = v.shape[0]
    ims = [v.sum(0), v.sum(1), v.sum(2), v[b // 2], v[:, b // 2], v[:, :, b // 2]]
    names = ["projection z", "projection y", "projection x", "section z", "section y", "section x"]
    # the same centred square crop for every view, around the particle (the box is mostly solvent)
    r = 0
    for im in ims[:3]:
        h, w = im.shape
        yy, xx = np.ogrid[:h, :w]
        d = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        bg = np.median(im[d > 0.45 * min(h, w)]) if (d > 0.45 * min(h, w)).any() else 0.0
        a = im - bg
        on = a > 0.06 * max(float(a.max()), 1e-12)
        if on.any():
            r = max(r, float(d[on].max()))
    half = int(min(b / 2, max(b * 0.2, r * 1.12 + 2)))
    c = b // 2
    ims = [im[max(c - half, 0):c + half, max(c - half, 0):c + half] for im in ims]
    return dict(name=os.path.basename(path), box=list(v.shape), apix=apix, crop=2 * half,
                views=[dict(label=n, src=_thumb(im, px)) for n, im in zip(names, ims)])


def _downsample(v: np.ndarray, size: int) -> np.ndarray:
    import torch
    n = max(v.shape)
    if n <= size:
        return v
    t = torch.as_tensor(np.array(v, np.float32))[None, None]
    out = torch.nn.functional.adaptive_avg_pool3d(t, (round(v.shape[0] * size / n), round(v.shape[1] * size / n),
                                                      round(v.shape[2] * size / n)))
    return out[0, 0].numpy()


def _pieces(m: np.ndarray):
    from scipy import ndimage
    lab, k = ndimage.label(m)
    return lab, (np.bincount(lab.ravel())[1:] if k else np.zeros(0, int))


def _clean_level(v: np.ndarray, default: float, floor: float, peak: float):
    """The isosurface level that shows the map in one piece: the usual default (5 sigma, 20 % of the peak) when its
    surface is >= 98 % one connected body, else the level where the largest body holds the largest share of the
    surface (a sharpened map: INX-6 at 6 % of the peak, the volume of the unsharpened map at 20 %, instead of 1,600
    speckles at 20 %). Also the region to keep when dust is hidden: pieces >= 1 % of the largest, dilated by 2
    voxels (so lowering the level a little still shows their edges), less the small pieces."""
    from scipy import ndimage

    def share(t):
        lab, sz = _pieces(v >= t)
        return (float(sz.max() / sz.sum()) if sz.size else 0.0), lab, sz
    best, lab, sz = share(default)
    level = default
    if best < 0.98:
        for t in np.linspace(floor, floor + 0.5 * (peak - floor), 24):
            f, l2, s2 = share(t)
            if f > best + 1e-3:
                best, level, lab, sz = f, float(t), l2, s2
    if not sz.size:
        return level, np.ones(v.shape, bool)
    keep = np.isin(lab, np.flatnonzero(sz >= max(0.01 * sz.max(), 8)) + 1)
    body = ndimage.binary_dilation(keep, iterations=2) & ~((lab > 0) & ~keep)    # without the small pieces themselves
    return level, body


def _roughness(v: np.ndarray, level: float) -> float:
    """Surface area / volume^(2/3) of the large pieces above `level` (voxel faces): ~20 for a refined map on the
    viewer's grid, ~80 for an over-sharpened one (INX-6, B -220), whose surface is noise at any level."""
    lab, sz = _pieces(v >= level)
    if not sz.size:
        return 0.0
    m = np.isin(lab, np.flatnonzero(sz >= 0.01 * sz.max()) + 1).astype(np.int8)
    area = sum(np.count_nonzero(np.diff(m, axis=a)) for a in range(3))
    return float(area / max(m.sum(), 1) ** (2 / 3))


def volume_payload(path: str, size: int = 160, color: str | None = None, dust: bool = True,
                   smooth: bool | None = None) -> bytes:
    """smooth: Gaussian of one display voxel, for the view only (None: when the surface is rough, _roughness > 40 -
    sharpened maps; the INX-6 sharpened map goes from 1,600 pieces to one)."""
    from scipy import ndimage
    v, apix = _read_mrc(path)
    n0 = max(v.shape)
    v = _downsample(v, size)
    apix_d = apix * n0 / max(v.shape)
    if smooth is None:
        pk = float(np.percentile(v, 99.9))
        smooth = _roughness(v, 0.2 * pk) > 40 if pk > 0 else False
    if smooth:
        v = ndimage.gaussian_filter(v, 1.0)
    lo = float(min(np.percentile(v, 1), 0.0))
    hi = float(np.percentile(v, 99.995))
    q = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    data = (q * 255).astype(np.uint8)
    # suggested levels: multiples of the noise measured at the box edge (unmasked maps), or - masked maps, no noise
    # there - fractions of the strongest density inside
    zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n) for n in v.shape), indexing="ij")
    edge = v[(zz ** 2 + yy ** 2 + xx ** 2) > 0.85 ** 2]
    sd = float(edge.std()) if edge.size else 0.0
    peak = float(np.percentile(v, 99.9))
    if sd > 1e-4 * max(abs(peak), 1e-12):
        mu = float(edge.mean())
        levels = {f"{k}σ": mu + k * sd for k in (3, 5, 8)}
        default, floor = "5σ", mu + 3 * sd
    else:
        levels = {f"{int(f * 100)}%": f * peak for f in (0.1, 0.2, 0.35)}
        default, floor = "20%", 0.02 * peak
    level, body = _clean_level(v, levels[default], floor, peak)
    levels = {"clean": level, **levels}
    head = dict(shape=list(data.shape), apix=apix_d, apix_file=apix, box=n0, min=lo, max=hi, levels=levels,
                level=level, name=os.path.basename(path), dust=bool(dust), smooth_a=round(apix_d, 2) if smooth else 0)
    if dust:                                    # hide dust: only the density near the surface's large pieces
        data[~body] = 0
    on = np.argwhere(body & (v >= level))       # the particle's radius / half the box: the viewer's first zoom
    if len(on):
        c = (np.array(v.shape) - 1) / 2
        head["extent"] = round(float(np.sqrt((((on - c) / (max(v.shape) / 2)) ** 2).sum(1)).max()), 3)
    payload = [data.tobytes(order="C")]
    if color:
        c, _ = _read_mrc(color)
        c = _downsample(c, size)
        if c.shape == data.shape:
            m = np.isfinite(c) & (c > 0)
            clo, chi = (np.percentile(c[m], [2, 98]) if m.any() else (0.0, 1.0))
            cq = np.clip((c - clo) / max(chi - clo, 1e-12), 0, 1)
            payload.append((cq * 255).astype(np.uint8).tobytes(order="C"))
            head.update(color=dict(name=os.path.basename(color), min=float(clo), max=float(chi)))
    h = json.dumps(head).encode()
    return struct.pack("<I", len(h)) + h + b"".join(payload)
