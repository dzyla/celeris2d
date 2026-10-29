"""Align a map to a reference map (rotation search + mirror) and report their FSC; writes a projection figure.

usage: compare_maps.py <map.mrc> <reference.mrc> <out.png> [label]
Both maps are resampled (Fourier crop/pad) to the first map's pixel size and box.
"""
import sys, math, numpy as np, torch, torch.nn.functional as F, mrcfile
from celeris2d import abinitio as ab
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
mine, ref, out = sys.argv[1:4]
dev = "cuda"
a = mrcfile.open(mine, permissive=True); va, pa = torch.tensor(a.data.astype(np.float32), device=dev), float(a.voxel_size.x)
r = mrcfile.open(ref, permissive=True); vr, pr = torch.tensor(r.data.astype(np.float32), device=dev), float(r.voxel_size.x)
b = va.shape[0]; B = vr.shape[0]
label = sys.argv[4] if len(sys.argv) > 4 else "map"
Bn = int(round(B * pr / pa))                       # reference box at our pixel size
ft = torch.fft.fftshift(torch.fft.fftn(vr))
if Bn <= B:
    o = (B - Bn) // 2; ft = ft[o:o + Bn, o:o + Bn, o:o + Bn]
else:
    z = torch.zeros((Bn,) * 3, dtype=ft.dtype, device=dev); o = (Bn - B) // 2; z[o:o + B, o:o + B, o:o + B] = ft; ft = z
vr = torch.fft.ifftn(torch.fft.ifftshift(ft)).real
if Bn >= b:
    o = (Bn - b) // 2; vr = vr[o:o + b, o:o + b, o:o + b]
else:
    z = torch.zeros((b,) * 3, device=dev); o = (b - Bn) // 2; z[o:o + Bn, o:o + Bn, o:o + Bn] = vr; vr = z
def rotvol(v, R):
    c = (torch.arange(b, device=dev, dtype=torch.float32) - b // 2)
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    p = torch.stack([x, y, z], -1) @ R.T          # sample v at R r
    g = (p + b // 2 + 0.5) / b * 2 - 1
    return F.grid_sample(v[None, None], g[None], align_corners=False)[0, 0]
def lp(v, res):
    f = torch.fft.fftn(v); k = torch.fft.fftfreq(b, device=dev) * b
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    return torch.fft.ifftn(f * torch.exp(-0.5 * (kk / (b * pa / res)) ** 2)).real
ra, rr = lp(va, 20), lp(vr, 20)
def norm(v): v = v - v.mean(); return v / v.norm()
rr_n = norm(rr)
best = (-2, None, None)
rot, tilt = ab.sphere_directions(6.0)
for mirror in (False, True):
    v = ra.flip(0) if mirror else ra
    for ps in torch.arange(0, 360, 6.0):
        A = ab.euler_to_matrix(rot, tilt, torch.full_like(rot, float(ps))).to(dev)
        for i in range(len(A)):
            c = float((norm(rotvol(v, A[i])) * rr_n).sum())
            if c > best[0]:
                best = (c, mirror, A[i].clone())
print("coarse best cc", round(best[0], 3), "mirror", best[1])
# local refine
c0, mirror, R = best
v = va.flip(0) if mirror else va
for step in (3.0, 1.5, 0.75):
    for _ in range(3):
        Rl = (ab.rotz(torch.tensor([-step, 0, step]))[:, None] @ ab.local_rotations(step, 1)[None]).reshape(-1, 3, 3).to(dev)
        cands = Rl @ R
        cs = [float((norm(rotvol(lp(v, 12), Q)) * norm(lp(vr, 12))).sum()) for Q in cands]
        R = cands[int(np.argmax(cs))]
al = rotvol(v, R)
fa, fr = torch.fft.fftn(al), torch.fft.fftn(vr)
k = torch.fft.fftfreq(b, device=dev) * b
sh = torch.round(torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)).long().reshape(-1)
n = b // 2 + 1; m = sh < n
num = torch.zeros(n, device=dev).index_add_(0, sh[m], (fa * fr.conj()).real.reshape(-1)[m])
d1 = torch.zeros(n, device=dev).index_add_(0, sh[m], (fa.abs() ** 2).reshape(-1)[m])
d2 = torch.zeros(n, device=dev).index_add_(0, sh[m], (fr.abs() ** 2).reshape(-1)[m])
fsc = (num / torch.sqrt(d1 * d2)).cpu().numpy()
res = [b * pa / i for i in range(1, n)]
def cross(t):
    for i in range(1, n):
        if fsc[i] < t: return b * pa / i
    return 2 * pa
print(f"map-vs-reference FSC: 0.5 at {cross(0.5):.1f} A, 0.143 at {cross(0.143):.1f} A")
fig, ax = plt.subplots(2, 3, figsize=(9, 6))
for j in range(3):
    ax[0, j].imshow(al.sum(j).cpu(), cmap="gray"); ax[1, j].imshow(lp(vr, 8).sum(j).cpu(), cmap="gray")
    ax[0, j].axis("off"); ax[1, j].axis("off")
ax[0, 0].set_title(f"{label} (aligned)"); ax[1, 0].set_title("reference (8 A)")
plt.tight_layout(); plt.savefig(out, dpi=80)
