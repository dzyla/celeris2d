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
import time

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


def _noise_levels(noise: np.ndarray, peak: float):
    """Suggested levels: multiples of the noise at the box corners (unmasked maps), or - masked maps, no noise there -
    fractions of the strongest density; with the default and the lowest level the clean-level search may go to."""
    sd = float(noise.std()) if noise.size else 0.0
    if sd > 1e-4 * max(abs(peak), 1e-12):
        mu = float(noise.mean())
        return {f"{k}σ": mu + k * sd for k in (3, 5, 8)}, "5σ", mu + 3 * sd
    return {f"{int(f * 100)}%": f * peak for f in (0.1, 0.2, 0.35)}, "20%", 0.02 * peak


def volume_payload(path: str, size: int = 256, color: str | None = None, dust: bool = True,
                   smooth: bool | None = None) -> bytes:
    """The map cropped to the particle (a cube about the box centre, so symmetry axes stay central) and only then
    downsampled to <= size^3: INX-6 (box 240, particle in the middle half) is shown at its full 1.23 A/px instead of
    1.85 A for the whole box at 160. smooth: Gaussian of one shown voxel, for the view only (None: when the surface
    is rough - _roughness > 40 on a 128 grid of the whole box; sharpened maps, the INX-6 one goes from 1,600 pieces to
    one)."""
    from scipy import ndimage
    v0, apix = _read_mrc(path)
    n0 = max(v0.shape)
    # 1. a coarse look at the whole box: is it rough, how far does the particle reach
    vc = _downsample(v0, 128)
    pkc = float(np.percentile(vc, 99.9))
    if smooth is None:
        smooth = _roughness(vc, 0.2 * pkc) > 40 if pkc > 0 else False
    if smooth:
        vc = ndimage.gaussian_filter(vc, 1.0)
    k = max(2, min(vc.shape) // 8)
    corners = np.concatenate([vc[z:z + k if z == 0 else None, y:y + k if y == 0 else None, x:x + k if x == 0 else None]
                              .ravel() for z in (0, -k) for y in (0, -k) for x in (0, -k)])
    lv_c, dflt_c, floor_c = _noise_levels(corners, float(np.percentile(vc, 99.9)))
    level_c, body_c = _clean_level(vc, lv_c[dflt_c], floor_c, float(np.percentile(vc, 99.9)))
    on = np.argwhere(body_c & (vc >= level_c))
    reach = 1.0
    if len(on):
        cc = (np.array(vc.shape) - 1) / 2
        reach = float(np.sqrt((((on - cc) / (max(vc.shape) / 2)) ** 2).sum(1)).max())
    # 2. the particle's cube at up to `size` voxels
    hw = int(min(n0 // 2, np.ceil(reach * 1.08 * n0 / 2) + 4))
    cut = tuple(slice(max(n // 2 - hw, 0), n // 2 + hw) for n in v0.shape)
    v = _downsample(v0[cut], size)
    shown = max(v.shape)
    apix_d = apix * max(sl.stop - sl.start for sl in cut) / shown      # (crop / shown)
    if smooth:
        v = ndimage.gaussian_filter(v, 1.0)
    crop = max(sl.stop - sl.start for sl in cut)
    d = crop / shown                            # original voxels per shown voxel: the corners' noise on the shown grid
    kk = max(4, int(min(v0.shape) // 8))
    noise = []
    for z in (0, -kk):
        for y in (0, -kk):
            for x in (0, -kk):
                cub = v0[z:z + kk if z == 0 else None, y:y + kk if y == 0 else None, x:x + kk if x == 0 else None]
                cub = _downsample(cub, max(2, round(kk / d)))
                noise.append((ndimage.gaussian_filter(cub, 1.0) if smooth else cub).ravel())
    peak = float(np.percentile(v, 99.9))
    levels, default, floor = _noise_levels(np.concatenate(noise), peak)
    vs = v if v.size <= 6e6 else _downsample(v, max(v.shape) // 2)      # the level search on <= 6 M voxels
    level, _ = _clean_level(vs, levels[default], floor, float(np.percentile(vs, 99.9)))
    _, body = _clean_level(v, level, level, peak)                        # dust at that level, on the shown grid
    lo = float(min(np.percentile(v, 1), 0.0))
    hi = float(np.percentile(v, 99.995))
    q = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    data = (q * 255).astype(np.uint8)
    levels = {"clean": level, **levels}
    head = dict(shape=list(data.shape), apix=apix_d, apix_file=apix, box=n0, crop=crop,
                min=lo, max=hi, levels=levels, level=level, name=os.path.basename(path), dust=bool(dust),
                smooth_a=round(apix_d, 2) if smooth else 0)
    if dust:                                    # hide dust: only the density near the surface's large pieces
        data[~body] = 0
    on = np.argwhere(body & (v >= level))       # the particle's radius / half the shown box: the viewer's first zoom
    if len(on):
        c = (np.array(v.shape) - 1) / 2
        head["extent"] = round(float(np.sqrt((((on - c) / (max(v.shape) / 2)) ** 2).sum(1)).max()), 3)
    payload = [data.tobytes(order="C")]
    if color:
        c, _ = _read_mrc(color)
        c = _downsample(c[cut], size)
        if c.shape == data.shape:
            # the range from the values on the surface (a shell about the shown level, inside the particle) - not
            # the whole box, where a local-resolution map holds the global value outside its mask (INX-6: 3.6 A
            # everywhere outside; on the surface 2.8 - 6.9 A)
            shell = body & (v >= 0.6 * level) & (v <= 1.6 * level) & np.isfinite(c) & (c > 0)
            vals = c[shell] if shell.sum() > 100 else c[np.isfinite(c) & (c > 0)]
            qlo, lo2, hi2, qhi = (np.percentile(vals, [0.5, 2, 98, 99.5]) if vals.size else (0.0, 0.0, 1.0, 1.0))
            cq = np.clip((np.nan_to_num(c, nan=qhi) - qlo) / max(qhi - qlo, 1e-12), 0, 1)
            payload.append((cq * 255).astype(np.uint8).tobytes(order="C"))
            head.update(color=dict(name=os.path.basename(color), lo=float(qlo), hi=float(qhi), min=float(lo2),
                                   max=float(hi2)))
    h = json.dumps(head).encode()
    return struct.pack("<I", len(h)) + h + b"".join(payload)


def live_classes(clf, job, started: float) -> dict:
    """A running 2D classification as it is now (written by the Class2D job every iteration as live.json; the job
    page draws it): class averages largest first, their sizes, the iteration statistics."""
    from .session import history_rows
    h = clf.history[-1]
    imgs = clf.class_images()
    pops = clf.pop_live.numpy() if hasattr(clf.pop_live, "numpy") else np.asarray(clf.pop_live)
    n = max(float(pops.sum()), 1.0)
    order = np.argsort(-pops)
    return dict(job=job.name, it=int(h["it"]), n_iter=int(max(int(clf.p.n_iter), h["it"])), res=round(float(h["res"]), 1),
                changed=round(100 * float(h["changed"]), 1), elapsed=round(time.time() - started, 1),
                history=history_rows(clf.history), updated=time.time(),
                classes=[dict(cls=int(k) + 1, n=round(float(pops[k])), share=round(float(pops[k]) / n, 4),
                              images=[_thumb(imgs[k])]) for k in order if k < len(imgs)])
