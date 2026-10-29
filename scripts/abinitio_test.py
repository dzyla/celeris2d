"""Ab initio 3D from a saved Celeris2D result.

usage: abinitio_test.py <input> <workdir> <result> <diameter> <max_res> <outdir> [sym]
(workdir: the dashboard/Session work dir that holds <result>.result.npz for this input)
"""
import sys
import time

from celeris2d import abinitio
from celeris2d.session import Session


def main():
    inp, wd, name, diam, mres, out = sys.argv[1:7]
    sym = sys.argv[7] if len(sys.argv) > 7 else "C1"
    s = Session(wd, log=lambda *a: None)
    t0 = time.time()
    s.load(inp, diameter=float(diam), max_res=float(mres), gpus="0")
    s.clf._wait_loaded(s.clf.N)
    print(f"[load] {time.time() - t0:.0f}s")
    if name not in s.results:
        K = int(name.lstrip("k")) if name.startswith("k") else 20
        s.classify(name, K=K)
    abinitio.run(s.clf, s.results[name], out, abinitio.AbinitioParams(sym=sym))


if __name__ == "__main__":
    main()
