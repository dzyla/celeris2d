"""Importing a RELION particle set into a project: three jobs like our own chain, dark protein inverted to white,
per-particle defocus kept, Fourier binning, and the Extract job usable as a particle source."""
import os

import mrcfile
import numpy as np
import pandas as pd

from celeris2d.importparticles import import_particles
from celeris2d.jobs import particle_source
from celeris2d.project import Project
from celeris2d.star import write_star


def _relion_set(d, n_mic=3, per=20, box=64, apix=2.0):
    rng = np.random.default_rng(0)
    r = np.arange(box) - box // 2
    blob = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * 6.0 ** 2)).astype(np.float32)
    rows = []
    os.makedirs(os.path.join(d, "Extract"))
    for m in range(n_mic):
        imgs = -3 * blob[None] + rng.normal(size=(per, box, box)).astype(np.float32)   # protein dark
        stack = f"Extract/mic{m}.mrcs"
        with mrcfile.new(os.path.join(d, stack)) as f:
            f.set_data(imgs)
            f.voxel_size = apix
        for i in range(per):
            rows.append(dict(rlnImageName=f"{i + 1:06d}@{stack}", rlnMicrographName=f"MotionCorr/mic{m}.mrc",
                             rlnCoordinateX=100.0 + 70 * (i % 5), rlnCoordinateY=100.0 + 70 * (i // 5),
                             rlnDefocusU=15000.0 + 100 * i, rlnDefocusV=14500.0 + 100 * i, rlnDefocusAngle=30.0,
                             rlnOpticsGroup=1))
    optics = pd.DataFrame(dict(rlnOpticsGroupName=["opticsGroup1"], rlnOpticsGroup=[1], rlnVoltage=[300.0],
                               rlnSphericalAberration=[2.7], rlnAmplitudeContrast=[0.1], rlnImagePixelSize=[apix],
                               rlnImageSize=[box], rlnImageDimensionality=[2], rlnMicrographPixelSize=[1.0]))
    write_star(os.path.join(d, "particles.star"), dict(optics=optics, particles=pd.DataFrame(rows)))
    return os.path.join(d, "particles.star")


def test_import_relion(tmp_path):
    star = _relion_set(str(tmp_path / "relion"))
    prj = Project(str(tmp_path / "prj"))
    ex = import_particles(prj, star, particle_a=60, particle_apix=4.0, log=lambda *a: None)
    ex = prj.job(ex)
    assert ex.type == "Extract" and ex.params["imported"] and ex.status == "finished"
    assert ex.params["box"] == 32 and abs(ex.params["out_apix"] - 4.0) < 1e-6
    assert prj.summary_of(ex)["inverted"]
    p = prj.particles(ex)
    assert len(p) == 60 and p["movie_id"].nunique() == 3
    assert np.allclose(np.sort(p["defocus_u"].unique())[:2], [15000, 15100])
    assert np.allclose(p["x_a"].min(), 100.0)                      # coordinates x micrograph pixel size (1 A)
    src = particle_source(prj, ex)
    assert src.N == 60 and src.box == 32
    imgs = src.read_all()
    assert imgs[:, 14:18, 14:18].mean() > imgs[:, :4, :4].mean() + 0.5      # protein white after the import
    mc = prj.job(ex.params["micrographs"])
    assert mc.type == "MotionCtf" and mc.params["imported"] and len(prj.micrographs(mc)) == 3


def test_import_twice(tmp_path):
    """A second set on micrographs the project already has (e.g. a curated subset of the same data) imports too."""
    star = _relion_set(str(tmp_path / "relion"))
    prj = Project(str(tmp_path / "prj"))
    import_particles(prj, star, particle_a=60, log=lambda *a: None)
    ex = prj.job(import_particles(prj, star, particle_a=60, label="again", log=lambda *a: None))
    assert ex.status == "finished" and len(prj.particles(ex)) == 60


def test_import_job_from_queue(tmp_path):
    """The registered ImportParticles job (dashboard form, MCP new_job) runs from the queue and names its Extract job."""
    from celeris2d.scheduler import queue_job, run_job
    star = _relion_set(str(tmp_path / "relion"), n_mic=2, per=10)
    prj = Project(str(tmp_path / "prj"))
    j = queue_job(prj, "ImportParticles", {}, dict(path=star, particle_a=60), "set1")
    run_job(prj, j, log=lambda *a: None)
    s = prj.summary_of(j)
    assert prj.job(j).status == "finished" and s["particles"] == 20
    assert prj.job(s["particles_job"]).type == "Extract"
