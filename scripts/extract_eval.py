"""Extract a refined particle set's particles from another set of micrographs (same movies), for a like-for-like
comparison of micrographs: reconstruct each extraction at the same refined poses (abinitio_truth.py ref) and
compare half-map FSC.

usage: extract_eval.py <refined particles.cs> <micrograph dir> <outdir> [--apix 1.315] [--box 224] [--register dir]

Micrographs are matched to particles by movie name (location/micrograph_path) and Fourier-cropped to --apix if
finer. Particles are cut at their exact (sub-pixel) picked position, so all micrograph sets are extracted
identically. --register <another extraction's outdir>: move each particle by its offset to the same particle in
that extraction (correlation low-passed to ~10 A; a smooth quadratic field per micrograph fitted to the
particles' offsets, since one particle's offset is mostly noise), so that the refined shifts, which belong to the micrographs the
particles were picked on, apply; different local motion moves particles by ~1 A between two motion corrections of
the same movie (EMPIAR-10025, ours vs cryoSPARC), which alone costs resolution. Writes <outdir>/stacks/*.mrcs and <outdir>/particles.cs (the input rows with new blob fields: the
poses, CTF and half-set split are kept). Then e.g.
    abinitio_truth.py ref <outdir>/particles.cs <outdir>/ref --diameter 180 --max-res 2.7 --sym D7
"""
import argparse
import glob
import os

import mrcfile
import numpy as np
import torch

from celeris2d import preprocess as pp


def stem_of(p: str) -> str:
    b = os.path.basename(p)
    i = b.find("14sep") if "14sep" in b else 0
    return b[i:].split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("particles")
    ap.add_argument("mics")
    ap.add_argument("outdir")
    ap.add_argument("--apix", type=float, default=1.315)
    ap.add_argument("--box", type=int, default=224)
    ap.add_argument("--register", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "stacks"), exist_ok=True)
    cs = np.load(a.particles)
    mp = np.array([stem_of(p.decode().lstrip(">")) for p in cs["location/micrograph_path"]])
    files = glob.glob(os.path.join(a.mics, "*.mrc"))
    dev = "cuda"
    b, h = a.box, a.box // 2
    paths = np.empty(len(cs), dtype="S256")
    idx = np.zeros(len(cs), dtype=np.uint32)
    keep = np.zeros(len(cs), dtype=bool)
    ky, kx = pp._grid(b, b, 1.0, dev)
    offs = []
    for stem in np.unique(mp):
        f = [x for x in files if stem in os.path.basename(x)]
        if not f:
            print(f"no micrograph for {stem}")
            continue
        with mrcfile.open(f[0], permissive=True) as m:
            mic = torch.as_tensor(np.asarray(m.data, dtype=np.float32), device=dev)
            apix = float(m.voxel_size.x)
        if apix < a.apix * 0.99:
            ny, nx = (np.array(mic.shape) * apix / a.apix).round().astype(int) // 2 * 2
            mic = torch.fft.irfft2(pp.crop_half(torch.fft.rfft2(mic, norm="forward"), int(ny), int(nx)),
                                   s=(int(ny), int(nx)), norm="forward")
        ny, nx = mic.shape
        mic = torch.nn.functional.pad(mic[None, None], (h + 2, h + 2, h + 2, h + 2), mode="reflect")[0, 0]
        sel = np.flatnonzero(mp == stem)
        x = cs["location/center_x_frac"][sel] * nx
        y = cs["location/center_y_frac"][sel] * ny
        x0, y0 = np.round(x).astype(int), np.round(y).astype(int)
        out = torch.empty(len(sel), b, b, device=dev)
        for k in range(len(sel)):
            out[k] = mic[y0[k] + 2: y0[k] + 2 + b, x0[k] + 2: x0[k] + 2 + b]
        # sub-pixel remainder: move the picked position (x, y) to the box centre
        dx = torch.as_tensor(x - x0, device=dev, dtype=torch.float32)
        dy = torch.as_tensor(y - y0, device=dev, dtype=torch.float32)
        if a.register:
            with mrcfile.open(os.path.join(a.register, "stacks", f"{stem}.mrcs"), permissive=True) as m:
                ref = torch.as_tensor(np.asarray(m.data, dtype=np.float32), device=dev)
            lp = torch.exp(-(ky ** 2 + kx ** 2) / a.apix ** 2 * 100)            # ky, kx in 1/px; ~10 A
            cc = torch.fft.irfft2(torch.fft.rfft2(ref) * torch.fft.rfft2(out).conj() * lp, s=(b, b))
            cc = torch.fft.fftshift(cc, dim=(-2, -1))[:, h - 8: h + 9, h - 8: h + 9]
            ry, rx = pp._peak(cc)                                  # our particle -> reference particle (px)
            # one particle's offset is mostly noise (~1 A): use a smooth field over the micrograph instead
            # (quadratic in x, y, robust least squares over all its particles)
            u, v = torch.as_tensor(x / nx * 2 - 1, device=dev), torch.as_tensor(y / ny * 2 - 1, device=dev)
            X = torch.stack([torch.ones_like(u), u, v, u * u, u * v, v * v], 1).float()
            r = torch.stack([rx, ry], 1)
            ok = torch.ones(len(r), dtype=torch.bool, device=dev)
            for _ in range(3):
                c = torch.linalg.lstsq(X[ok], r[ok]).solution
                e = (X @ c - r).norm(dim=1)
                ok = e < 3 * 1.4826 * e[ok].median() + 1e-3
            fld = X @ c
            dx, dy = dx - fld[:, 0], dy - fld[:, 1]
            offs.append(np.concatenate([fld.cpu().numpy(), (r - fld).cpu().numpy()], 1) * a.apix)
        out = torch.fft.irfft2(torch.fft.rfft2(out) * pp._shift_phase(ky, kx, -dy, -dx), s=(b, b))
        name = f"stacks/{stem}.mrcs"
        with mrcfile.new(os.path.join(a.outdir, name), overwrite=True) as m:
            m.set_data(out.cpu().numpy())
            m.voxel_size = a.apix
        paths[sel], idx[sel], keep[sel] = name.encode(), np.arange(len(sel)), True
        print(f"{stem}: {len(sel)} particles", flush=True)
    dt = [(n, "S256" if n == "blob/path" else cs.dtype[n].base, cs.dtype[n].shape) for n in cs.dtype.names]
    new = np.zeros(len(cs), dtype=dt)
    for n in cs.dtype.names:
        new[n] = cs[n]
    new["blob/path"], new["blob/idx"] = paths, idx
    new["blob/shape"] = b
    new["blob/psize_A"] = a.apix
    with open(os.path.join(a.outdir, "particles.cs"), "wb") as fh:
        np.save(fh, new[keep])
    print(f"{int(keep.sum())} of {len(cs)} particles -> {a.outdir}/particles.cs")
    if offs:
        o = np.concatenate(offs)
        print(f"registration: smooth field rms {np.sqrt((o[:, :2] ** 2).sum(1).mean()):.2f} A (max "
              f"{np.abs(o[:, :2]).max():.2f} A); per-particle scatter around it {np.sqrt((o[:, 2:] ** 2).sum(1).mean()):.2f} A")


if __name__ == "__main__":
    main()
