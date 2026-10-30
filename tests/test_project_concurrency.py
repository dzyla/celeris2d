"""Jobs created by several processes at the same moment get distinct ids (CPU, a few seconds). Three refinements
launched together on one project once failed with "UNIQUE constraint failed: jobs.id": the counter was read before
the write lock was taken."""
import multiprocessing as mp

from celeris2d.project import Project


def _make(root, i, q, bar):
    p = Project(root)
    bar.wait()                                       # all processes start creating at the same moment
    for k in range(10):
        q.put(p.new_job("Sharpen", {}, label=f"c{i}_{k}").id)


def test_concurrent_new_job(tmp_path):
    Project(str(tmp_path))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    bar = ctx.Barrier(8)
    ps = [ctx.Process(target=_make, args=(str(tmp_path), i, q, bar)) for i in range(8)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
    assert all(p.exitcode == 0 for p in ps), [p.exitcode for p in ps]
    ids = sorted(q.get() for _ in range(80))
    assert ids == list(range(1, 81)), ids
