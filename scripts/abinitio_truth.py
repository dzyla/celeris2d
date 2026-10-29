"""Score ab initio runs against refined 3D poses of the same particles (cryoSPARC .cs with alignments3D).

usage:
  abinitio_truth.py ref <particles.cs|.csg> <outdir> --diameter D --max-res R [--sym C3] [--poses refine.cs]
      reconstructs the reference map at the refined poses with Celeris2D's own back-projector (pose convention and
      shift sign chosen by half-map FSC on a subset) -> <outdir>/ref.mrc, ref.json
  abinitio_truth.py score <abinitio outdir | other job's .cs> <particles.cs|.csg> [--sym C3] [--ref <outdir>/ref.mrc]
      [--poses refine.cs] [--map other job's map.mrc]
      orientation accuracy: the global rotation between the two pose sets (RANSAC over single-particle candidates,
      both hands, both pose conventions, symmetry-equivalent poses), then the share of particles within 5/10/20 deg
      and the median error; with --ref also the map-vs-reference FSC (compare_maps.py, rotation and hand search).
      Prints one JSON line. A .cs in place of the run directory scores another program's 3D poses (e.g. a cryoSPARC
      ab initio job, matched by uid), with --map its map.
  --poses: the .cs holding the refined alignments3D when the particle input is a .csg (same particle order).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from celeris2d import abinitio as ab
from celeris2d.engine import TWO_PI, ctf_2d, freqs


def gt_matrices(cs) -> torch.Tensor:
    return torch.as_tensor(Rotation.from_rotvec(cs["alignments3D/pose"].astype(np.float64)).as_matrix(),
                           dtype=torch.float32)


def angle_deg(Q: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.arccos(((Q.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))


def orientation_accuracy(A: torch.Tensor, R: torch.Tensor, ops: torch.Tensor, n_cand=300, n_eval=4000, seed=0):
    """A: ab initio orientations (image = A @ volume), R: reference orientations in some convention. Finds G with
    R_i ~ A_i G S_k (S_k symmetry ops) for as many particles as possible. Returns stats of the best variant."""
    g = torch.Generator().manual_seed(seed)
    P = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    best = None
    for conv in ("R", "Rt"):
        Rv = R if conv == "R" else R.transpose(1, 2)
        for hand in ("same", "mirror"):
            Av = A if hand == "same" else P @ A @ P
            ci = torch.randperm(len(A), generator=g)[:n_cand]
            ei = torch.randperm(len(A), generator=g)[:n_eval]
            # candidates G = A_i^T R_i S_k^T
            C = (Av[ci].transpose(1, 2) @ Rv[ci])[:, None] @ ops.transpose(1, 2)[None]      # (c, S, 3, 3)
            C = C.reshape(-1, 3, 3)
            # error of every candidate on the evaluation set: min over k of angle(A_j G S_k, R_j)
            AG = Av[ei][None] @ C[:, None]                                                  # (c, e, 3, 3)
            err = angle_deg(AG[:, :, None].transpose(-1, -2) @ (Rv[ei][None, :, None] @ ops.transpose(1, 2)[None, None]))
            err = err.min(-1).values                                                        # (c, e)
            score = (err < 10).float().mean(1)
            j = int(score.argmax())
            if best is None or score[j] > best[0]:
                best = (float(score[j]), conv, hand, C[j], Av, Rv)
    _, conv, hand, G, Av, Rv = best
    # refine G on the inliers (chordal mean of the per-particle G estimates, projected to SO(3))
    for _ in range(3):
        cand = Av.transpose(1, 2)[:, None] @ Rv[:, None] @ ops.transpose(1, 2)[None]           # (n, S, 3, 3)
        d = angle_deg(G[None, None].transpose(-1, -2) @ cand)                                # (n, S)
        k = d.argmin(1)
        e = d.gather(1, k[:, None])[:, 0]
        inl = e < 10
        Mx = cand[torch.arange(len(cand)), k][inl].sum(0)
        U, _, Vt = torch.linalg.svd(Mx)
        G = U @ torch.diag(torch.tensor([1.0, 1.0, float(torch.det(U @ Vt))])) @ Vt
    cand = Av.transpose(1, 2)[:, None] @ Rv[:, None] @ ops.transpose(1, 2)[None]
    e = angle_deg(G[None, None].transpose(-1, -2) @ cand).min(1).values
    return dict(conv=conv, hand=hand, within5=float((e < 5).float().mean()), within10=float((e < 10).float().mean()),
                within20=float((e < 20).float().mean()), median_err=float(e.median())), e


def reconstruct(clf, idx, A, t, b, sym, half, kmax=None):
    dev, box_a = clf.dev, clf.box_a
    vols = [ab.Volume(b, dev), ab.Volume(b, dev)]
    fy, fx = freqs(b, dev)
    ph = TWO_PI / box_a
    ops = ab.symmetry_ops(sym).to(dev)
    kmax = kmax or b / 2 - 1
    for i in range(0, len(idx), 512):
        j = idx[i:i + 512]
        x = clf._fetch(j, dev, b)
        ctf = ctf_2d(clf.ctf_p[j].to(dev), clf.optics, fy, fx, box_a)
        tt, AA, hh = t[i:i + 512].to(dev), A[i:i + 512].to(dev), half[i:i + 512].to(dev)
        xs = x * torch.exp(1j * ph * (fx[None, None, :] * tt[:, 0, None, None] + fy[None, :, None] * tt[:, 1, None, None]))
        for h in (0, 1):
            sel = hh == h
            if sel.any():
                vols[h].insert((xs * ctf)[sel], (ctf * ctf)[sel], AA[sel], kmax=kmax, ops=ops)
    for v in vols:
        ab._regularise_solve(v, 0.02)
    mask = ab.sphere_mask(b, clf.p.diameter_a / 2 / (box_a / b), 3.0, dev)
    f = ab.masked_fsc(vols[0], vols[1], mask)
    return vols, f, ab.resolution_from_fsc(f, box_a)


def cmd_ref(a):
    from celeris2d.session import Session
    cs = np.load(a.poses or a.particles)
    s = Session(os.path.join(a.outdir, "work"), log=lambda m: None)
    s.load(a.particles, diameter=a.diameter, max_res=a.max_res, gpus="0")
    clf = s.clf
    clf._wait_loaded(clf.N)
    clf.ensure_polarity()           # same contrast as an ab initio run on these particles
    b, box_a = clf.bmax, clf.box_a
    R = gt_matrices(cs)
    sh = torch.as_tensor(cs["alignments3D/shift"] * cs["alignments3D/psize_A"][:, None], dtype=torch.float32)
    half = torch.as_tensor(cs["alignments3D/split"].astype(np.int64)) % 2
    sub = torch.randperm(clf.N, generator=torch.Generator().manual_seed(0))[:4000]
    best = None
    for conv in ("R", "Rt"):
        Rv = R if conv == "R" else R.transpose(1, 2)
        for sgn in (1, -1):
            _, f, res = reconstruct(clf, sub, Rv[sub], sgn * sh[sub], b, a.sym, half[sub])
            print(f"[ref] convention {conv}, shift sign {sgn:+d}: {res:.2f} A on 4000 particles", flush=True)
            if best is None or res < best[0]:
                best = (res, conv, sgn)
    _, conv, sgn = best
    Rv = R if conv == "R" else R.transpose(1, 2)
    idx = torch.arange(clf.N)
    vols, f, res = reconstruct(clf, idx, Rv, sgn * sh, b, a.sym, half)
    full = ab.Volume(b, clf.dev, ft=(vols[0].ft + vols[1].ft) / 2)
    full.filter_shells(torch.sqrt((2 * f / (1 + f)).clamp(0, 1).nan_to_num(0)))
    ab._save_mrc(os.path.join(a.outdir, "ref.mrc"), full.real(), box_a / b)
    info = dict(convention=conv, shift_sign=sgn, resolution_a=res, box=b, apix=box_a / b, n=clf.N, sym=a.sym)
    json.dump(info, open(os.path.join(a.outdir, "ref.json"), "w"), indent=1)
    print(json.dumps(info))


def cmd_score(a):
    cs = np.load(a.poses or a.particles)
    ops = ab.symmetry_ops(a.sym)
    if a.run.endswith(".cs"):     # another program's poses, matched by uid
        oc = np.load(a.run)
        pos = {u: i for i, u in enumerate(cs["uid"])}
        keep = np.array([u in pos for u in oc["uid"]])
        idx = np.array([pos[u] for u in oc["uid"][keep]])
        # cryoSPARC's rotation is the transpose of ours (image = A @ volume): see the convention found by `ref`
        out, e = orientation_accuracy(gt_matrices(oc[keep]).transpose(1, 2), gt_matrices(cs)[idx], ops)
        out["n"] = len(idx)
        run_map = a.map
    else:
        z = np.load(os.path.join(a.run, "poses.npz"))
        idx = z["idx"]
        A = ab.euler_to_matrix(torch.as_tensor(z["rot"]), torch.as_tensor(z["tilt"]), torch.as_tensor(z["psi"]))
        out, e = orientation_accuracy(A, gt_matrices(cs)[idx], ops)
        used = torch.as_tensor(z["used"])
        out["within10_used"] = float((e[used] < 10).float().mean())
        out["n"] = len(idx)
        run_map = os.path.join(a.run, "map_nomask.mrc")
    summ = os.path.join(a.run, "summary.json")
    if os.path.exists(summ):
        sm = json.load(open(summ))
        out.update({k: sm.get(k) for k in ("resolution_a", "resolution_unmasked_a", "seconds_classes",
                                            "seconds_particles", "sym", "suggested_sym")})
    if a.ref and run_map:
        here = os.path.dirname(os.path.abspath(__file__))
        png = os.path.splitext(a.run)[0] + "_vs_ref.png" if a.run.endswith(".cs") else os.path.join(a.run, "vs_ref.png")
        r = subprocess.run([sys.executable, os.path.join(here, "compare_maps.py"), run_map, a.ref, png],
                           capture_output=True, text=True)
        txt = r.stdout + r.stderr
        import re
        m = re.findall(r"0\.5 at ([0-9.]+) A, 0\.143 at ([0-9.]+) A", txt)
        out["vs_ref"] = [float(m[-1][0]), float(m[-1][1])] if m else txt.strip().splitlines()[-1:]
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("ref")
    r.add_argument("particles")
    r.add_argument("outdir")
    r.add_argument("--diameter", type=float, required=True)
    r.add_argument("--max-res", type=float, required=True)
    r.add_argument("--sym", default="C1")
    r.add_argument("--poses", default="")
    s = sp.add_parser("score")
    s.add_argument("run")
    s.add_argument("particles")
    s.add_argument("--sym", default="C1")
    s.add_argument("--ref", default="")
    s.add_argument("--poses", default="")
    s.add_argument("--map", default="")
    a = ap.parse_args()
    os.makedirs(getattr(a, "outdir", "."), exist_ok=True)
    cmd_ref(a) if a.cmd == "ref" else cmd_score(a)


if __name__ == "__main__":
    main()
