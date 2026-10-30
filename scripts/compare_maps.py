"""Align a map to a reference map (rotation search, both hands, then translation) and report their FSC;
writes a projection figure.

usage: compare_maps.py <map.mrc> <reference.mrc> <out.png> [label] [mask.mrc]

The reference is resampled (Fourier crop/pad) to the map's pixel size and box. The rotation is searched on
48-px copies (20 A), refined locally on 96-px copies, then applied once at full size. Ab initio maps have an
arbitrary hand, so the mirrored map is searched as well. Prints the map-vs-reference FSC 0.5 / 0.143 crossings.
"""
import sys

import matplotlib
import mrcfile
import numpy as np
import torch

from celeris2d import abinitio as ab

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def resample(v: torch.Tensor, apix: float, apix_new: float, box_new: int) -> torch.Tensor:
    """Fourier crop/pad v to pixel size apix_new, then crop/pad (real space) to box_new."""
    B = v.shape[0]
    Bn = int(round(B * apix / apix_new))
    ft = torch.fft.fftshift(torch.fft.fftn(v))
    if Bn <= B:
        o = (B - Bn) // 2
        ft = ft[o:o + Bn, o:o + Bn, o:o + Bn]
    else:
        z = torch.zeros((Bn,) * 3, dtype=ft.dtype, device=v.device)
        o = (Bn - B) // 2
        z[o:o + B, o:o + B, o:o + B] = ft
        ft = z
    v = torch.fft.ifftn(torch.fft.ifftshift(ft)).real * (Bn / B) ** 3
    if Bn >= box_new:
        o = (Bn - box_new) // 2
        return v[o:o + box_new, o:o + box_new, o:o + box_new]
    z = torch.zeros((box_new,) * 3, device=v.device)
    o = (box_new - Bn) // 2
    z[o:o + Bn, o:o + Bn, o:o + Bn] = v
    return z


def ncc(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Normalised correlation of a stack (n, b, b, b) with one map."""
    a = a - a.mean((1, 2, 3), keepdim=True)
    r = ref - ref.mean()
    return (a * r).sum((1, 2, 3)) / (a.pow(2).sum((1, 2, 3)).sqrt() * r.norm()).clamp_min(1e-12)


def main():
    mine, ref, out = sys.argv[1:4]
    label = sys.argv[4] if len(sys.argv) > 4 else "map"
    mask_path = sys.argv[5] if len(sys.argv) > 5 else ""          # on the reference's grid (e.g. RELION MaskCreate)
    dev = "cuda"
    a = mrcfile.open(mine, permissive=True)
    va, pa = torch.tensor(a.data.astype(np.float32), device=dev), float(a.voxel_size.x)
    r = mrcfile.open(ref, permissive=True)
    vr0, pr = torch.tensor(r.data.astype(np.float32), device=dev), float(r.voxel_size.x)
    b = va.shape[0]
    vr = resample(vr0, pr, pa, b)                       # reference on the map's grid
    box_a = b * pa
    nf = min(b, 144)                                    # final rotation refinement: ~2 A copies (box 300 A)
    small = {n: (resample(va, pa, box_a / n, n), resample(vr, pa, box_a / n, n)) for n in (48, 96, nf)}
    best = (-2.0, None, None)
    rot, tilt = ab.sphere_directions(6.0)
    for mirror in (False, True):
        v, ref_s = small[48]
        v = v.flip(0) if mirror else v
        for ps in range(0, 360, 6):
            A = ab.euler_to_matrix(rot, tilt, torch.full_like(rot, float(ps))).to(dev)
            c = ncc(ab._rotate_volumes(v, A), ref_s)
            i = int(c.argmax())
            if float(c[i]) > best[0]:
                best = (float(c[i]), mirror, A[i].clone())
    print(f"coarse best cc {best[0]:.3f}, {'mirrored hand' if best[1] else 'same hand'}")
    _, mirror, R = best
    v96, r96 = small[96]
    v96 = v96.flip(0) if mirror else v96
    vf, rf = small[nf]
    vf = vf.flip(0) if mirror else vf
    # coarse-to-fine local search; the last steps on the ~2 A copies (0.4 deg on 3.3 A copies left up to ~0.5 A of
    # misalignment at the particle's edge, enough to pull the map-vs-map FSC down around 4 A)
    for step, (vv, rr) in ((3.0, (v96, r96)), (1.5, (v96, r96)), (0.75, (v96, r96)), (0.4, (vf, rf)),
                           (0.2, (vf, rf)), (0.1, (vf, rf))):
        for _ in range(3):
            Rl = (ab.rotz(torch.tensor([-step, 0.0, step]))[:, None] @ ab.local_rotations(step, 1)[None]).reshape(-1, 3, 3)
            cands = Rl.to(dev) @ R
            c = torch.cat([ncc(ab._rotate_volumes(vv, cands[i:i + 9]), rr) for i in range(0, len(cands), 9)])
            R = cands[int(c.argmax())]
    al = ab._rotate_volumes(va.flip(0) if mirror else va, R[None])[0]
    # translation: an ab initio map in C1 is only centred on its centre of mass (a symmetric one sits on its axes);
    # integer peak of the cross-correlation, refined to sub-voxel by a parabola per axis, applied as a phase shift
    cc = torch.fft.ifftn(torch.fft.fftn(vr) * torch.fft.fftn(al).conj()).real
    pk = [int(i) for i in np.unravel_index(int(cc.argmax()), cc.shape)]
    sh = []
    for ax_ in range(3):
        idx = list(pk)
        vals = []
        for d in (-1, 0, 1):
            idx[ax_] = (pk[ax_] + d) % b
            vals.append(float(cc[tuple(idx)]))
        den = vals[0] - 2 * vals[1] + vals[2]
        frac = 0.5 * (vals[0] - vals[2]) / den if den < 0 else 0.0
        i = pk[ax_] + frac
        sh.append(i - b if i > b / 2 else i)
    k = torch.fft.fftfreq(b, device=dev)
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    al = torch.fft.ifftn(torch.fft.fftn(al) * torch.exp(-2j * np.pi * (kz * sh[0] + ky * sh[1] + kx * sh[2]))).real
    print(f"translation {[round(i * pa, 2) for i in sh]} A (z, y, x)")
    f = ab.fsc_maps(al, vr).cpu().numpy()
    fu = f
    if mask_path:                                       # both maps multiplied by the reference's mask
        mk = mrcfile.open(mask_path, permissive=True)
        m = resample(torch.tensor(mk.data.astype(np.float32), device=dev), float(mk.voxel_size.x), pa, b).clamp(0, 1)
        f = ab.fsc_maps(al * m, vr * m).cpu().numpy()

    def cross(t):
        for i in range(1, len(f)):
            if f[i] < t:
                return box_a / i
        return 2 * pa
    if mask_path:
        f, fm = fu, f
        print(f"map-vs-reference FSC (unmasked): 0.5 at {cross(0.5):.1f} A, 0.143 at {cross(0.143):.1f} A")
        f = fm
    print(f"map-vs-reference FSC{' (masked)' if mask_path else ''}: 0.5 at {cross(0.5):.1f} A, "
          f"0.143 at {cross(0.143):.1f} A")
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for j in range(3):
        ax[0, j].imshow(al.sum(j).cpu(), cmap="gray")
        ax[1, j].imshow(vr.sum(j).cpu(), cmap="gray")
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    ax[0, 0].set_title(f"{label} (aligned)", loc="left")
    ax[1, 0].set_title("reference", loc="left")
    gs = ax[0, 3].get_gridspec()
    ax[0, 3].remove()
    ax[1, 3].remove()
    axf = fig.add_subplot(gs[:, 3])
    x = np.arange(len(f)) / box_a
    axf.plot(x[1:], f[1:], lw=2)
    for t in (0.5, 0.143):
        axf.axhline(t, color="0.6", lw=1, ls="--")
    axf.set_xlabel("spatial frequency (1/A)")
    axf.set_ylabel("FSC map vs reference")
    axf.set_title(f"0.5 at {cross(0.5):.1f} A, 0.143 at {cross(0.143):.1f} A")
    plt.tight_layout()
    plt.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
