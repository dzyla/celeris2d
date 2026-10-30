"""Data the dashboard shows directly (not figures): class galleries to choose classes from, and maps for the 3D
viewer.

class_gallery(project, job): the classes of a Class2D job (its class averages) or of a HeteroRefine / Class3D job
(three projections of each class map), largest first, with particles, share and resolution; for a Select job, the
classes of its source and which it keeps.

volume_payload(path, size, color): a map downsampled to <= size^3 as 8-bit voxels for the browser's raymarcher -
bytes: uint32 header length, JSON header (shape, apix, value range, suggested isosurface levels, colour range),
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


def _downsample(v: np.ndarray, size: int) -> np.ndarray:
    import torch
    n = max(v.shape)
    if n <= size:
        return v
    t = torch.as_tensor(np.array(v, np.float32))[None, None]
    out = torch.nn.functional.adaptive_avg_pool3d(t, (round(v.shape[0] * size / n), round(v.shape[1] * size / n),
                                                      round(v.shape[2] * size / n)))
    return out[0, 0].numpy()


def volume_payload(path: str, size: int = 160, color: str | None = None) -> bytes:
    v, apix = _read_mrc(path)
    n0 = max(v.shape)
    v = _downsample(v, size)
    apix_d = apix * n0 / max(v.shape)
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
        default = "5σ"
    else:
        levels = {f"{int(f * 100)}%": f * peak for f in (0.1, 0.2, 0.35)}
        default = "20%"
    head = dict(shape=list(data.shape), apix=apix_d, apix_file=apix, box=n0, min=lo, max=hi, levels=levels,
                level=levels[default], name=os.path.basename(path))
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
