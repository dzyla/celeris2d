"""Plain-language catalog of the job types for the dashboard's job menu and New job form.

The settings dataclasses carry the code's own notes (what was measured, why a default is what it is); this module
says the same things in the words of a cryo-EM user: a display name, the processing stage (group), one sentence on
what the job does, what it needs, and for every setting a short label, a plain help text, basic / advanced, a
section for the advanced ones and the fixed choices where a text setting takes them. Nothing here changes a job:
`annotate` only adds keys to jobspec.describe() output. tests/test_uimeta.py fails when a registered job type, a
setting or an input has no entry here, so a new setting gets its plain words with its code.

Per setting:
    label     short, sentence case, with the unit
    help      what it changes and when to change it
    level     "basic" (what most users set) or "advanced"
    section   group of advanced settings ("Alignment", "Masking", ...), "" for basic ones
    choices   [[value, label], ...] when the value is one of a fixed set (or a suggestion list if `free`)
    free      True: choices are suggestions, other values are allowed too (e.g. a job number or a file path)
    job_types the job types a job-number setting refers to (0 or empty: none)
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------- groups and order
GROUPS = (
    ("Movies & micrographs", ("Import", "MotionCtf", "Curate")),
    ("Particles", ("Pick", "ManualPick", "Extract", "ImportParticles")),
    ("2D", ("Class2D", "MapScore")),
    ("3D", ("InitialModel", "Refine3D", "HeteroRefine", "Class3D", "LocalRefine")),
    ("Particle corrections", ("CtfRefine", "Polish")),
    ("Variability", ("Modes3D", "ModeMaps")),
    ("Maps & validation", ("MaskCreate", "Sharpen")),
    ("Utilities", ("Subtract", "SymExpand")),
)
GROUP_NAMES = tuple(g for g, _ in GROUPS)
FALLBACK_GROUP = "Utilities"

# display names of job types that are not made from the New job form (made by the dashboard itself)
EXTRA_NAMES = {"Select": "Select 2D classes"}


def F(label, help, level="advanced", section="", choices=None, free=False, job_types=None):
    d = dict(label=label, help=help, level=level, section=section)
    if choices is not None:
        d["choices"] = [list(c) for c in choices]
    if free:
        d["free"] = True
    if job_types:
        d["job_types"] = list(job_types)
    return d


def B(label, help, **kw):
    return F(label, help, level="basic", **kw)


def I(label, help):
    return dict(label=label, help=help)


# ------------------------------------------------------------------------------------- settings used by many jobs
SYM_CHOICES = [["C1", "C1 (none)"], ["C2", "C2"], ["C3", "C3"], ["C4", "C4"], ["C5", "C5"], ["C6", "C6"],
               ["C7", "C7"], ["C8", "C8"], ["D2", "D2"], ["D3", "D3"], ["D4", "D4"], ["D7", "D7"],
               ["T", "T (tetrahedral)"], ["O", "O (octahedral)"], ["I", "I (icosahedral)"]]


def SYM(help="Point-group symmetry imposed on the map: C1 (none), Cn, Dn, T, O or I. It is never detected "
             "automatically; the job's scores show how well the chosen symmetry holds.", level="basic", section=""):
    return F("Symmetry", help, level=level, section=section, choices=SYM_CHOICES, free=True)


def GPUS(multi=True):
    if multi:
        return F("GPUs", "Which GPUs to use: \"auto\" takes all free ones, or give a list such as \"0,2\".",
                 section="Performance", choices=[["auto", "auto (all free GPUs)"]], free=True)
    return F("GPU", "Which GPU to use: \"auto\" takes the one with the most free memory, or give one number such as "
                    "\"1\".", section="Performance", choices=[["auto", "auto"]], free=True)


SEED = F("Random seed", "Seed of the random choices (starting values, subsets). Change it to repeat the job with "
                        "different random starts.", section="Other")
DTYPE_CHOICES = [["float16", "16-bit (half the disk space)"], ["float32", "32-bit"]]
MASK3D_CHOICES = [["auto", "Automatic, updated every iteration"], ["static", "Automatic, made once from the start map"],
                  ["none", "None (soft sphere only)"]]
CLASSES = B("Use only classes", "When the poses come from a HeteroRefine or Class3D job: only the particles of these "
                                "classes (1-based numbers, e.g. \"1, 3\"). Empty: all particles.")
DIAM_3D = F("Particle diameter (Å)", "Diameter of the soft spherical mask on the particle images. 0: from the "
                                    "particle size given at extraction.", section="Masking")
MAXRES = F("Highest resolution loaded (Å)", "Finest resolution of the particle images loaded into memory. 0: Nyquist "
                                           "of the particles. Set a coarser value to save memory and time on big "
                                           "boxes.", section="Resolution")
NLOCAL = F("Local search width", "Half-width of the local angular search around each particle's current pose, in "
                                 "angular steps: (2n + 1) steps per rotation axis. Larger searches further but is "
                                 "slower.", section="Alignment")
DROP = F("Discard worst-fitting fraction", "Share of the particles that fit their map worst left out of each "
                                          "reconstruction (0.1 = 10 %). 0 keeps every particle.", section="Alignment")
SAVE_ITERS = F("Save per-iteration statistics", "Keep every iteration's FSC curves and each particle's angle / shift "
                                                "change and fit per iteration, to check convergence and find particles "
                                                "that keep jumping.", section="Diagnostics")
SHARPEN = {
    "sharpen": F("Sharpen final map", "Sharpen the final half maps with this job's own mask (B-factor, masked FSC), "
                                      "as a separate Sharpen job would. Takes seconds.", section="Sharpening"),
    "local_res": F("Local resolution", "With sharpening: also compute the local resolution map.", section="Sharpening"),
    "mtf": F("Detector MTF file", "Detector MTF (RELION .star, e.g. mtf_k2_300kV.star from relion/data/) used "
                                  "when sharpening. Empty: no MTF correction.", section="Sharpening"),
    "mtf_apix": F("MTF pixel size (Å)", "Pixel size the MTF file refers to (the detector's physical pixel, before "
                                       "binning). 0: the map's pixel size.", section="Sharpening"),
}
P_PARTICLES = I("Particles", "The particle images: an extraction, a selection or a job that made new particle images "
                             "(Polish, CTF refinement, Subtract), or a symmetry expansion.")
P_POSES = I("Poses from", "The job whose particle orientations and shifts are used, matched to the particles by "
                          "particle ID.")
P_REFERENCE = I("Reference map", "The starting map. Empty: the map of the job the poses come from.")
P_REFINE = I("Refinement", "The 3D refinement whose particles, poses and half maps are used.")
P_MASK = I("Mask", "A Create mask job.")

# ------------------------------------------------------------------------------------------------- the job types
TYPES: dict[str, dict] = {}

TYPES["Import"] = dict(
    name="Import movies",
    blurb="Registers the raw movies with the project; the files stay where they are and are not copied.",
    needs="A folder or file pattern with the movies (.tif, .tiff, .mrc, .mrcs, .eer).",
    typical="The first job of a project; then Motion correction & CTF.",
    inputs={},
    fields={
        "movies": B("Movies", "A folder (every .tif / .tiff / .mrc / .mrcs / .eer file in it and its sub-folders) or a "
                              "file pattern such as /data/session1/**/*.tif. Gain and defect files are skipped; "
                              "Windows paths work."),
        "root": F("Base folder for names", "The folder whose sub-folders are kept in the output file names. Empty: the "
                                           "movies' common parent folder.", section="Output"),
    })

TYPES["MotionCtf"] = dict(
    name="Motion correction & CTF",
    blurb="Aligns the movie frames (whole-frame and local motion), applies dose weighting and fits the CTF of every "
          "micrograph.",
    needs="Movies from Import movies.",
    typical="Right after Import movies; then Curate micrographs or Pick particles.",
    inputs={"movies": I("Movies", "The Import movies job.")},
    fields={
        "apix": B("Pixel size (Å)", "Pixel size of the movies as recorded. For super-resolution movies give the "
                                    "super-resolution pixel size and set binning to 2."),
        "dose_total": B("Total dose (e⁻/Å²)", "Total electron dose over all frames of a movie; used for dose "
                                              "weighting."),
        "bin": B("Binning", "Fourier binning of the frames: 1 keeps the pixel size, 2 turns super-resolution movies "
                            "into physical pixels."),
        "gain": B("Gain reference", "Path of the gain reference file (.mrc / .tif). Its orientation is checked "
                                    "against the movies automatically. Empty: the movies are already gain-corrected."),
        "kv": B("Voltage (kV)", "Acceleration voltage of the microscope."),
        "cs_mm": B("Spherical aberration (mm)", "Spherical aberration (Cs) of the objective lens."),
        "amp_contrast": F("Amplitude contrast", "Fraction of amplitude contrast in the CTF model; 0.1 suits most cryo "
                                                "samples (often 0.07-0.1).", section="CTF"),
        "patches": F("Local motion patches", "Grid of patches per side for local (patch) motion correction. 0: "
                                             "whole-frame motion only.", section="Motion"),
        "patch_ctf": F("Local defocus (ice tilt)",
                       "Fits a defocus plane across each micrograph from a grid of regions. Automatic gives particles "
                       "their local defocus only when the ice is clearly tilted (5° or more, e.g. tilted collection); "
                       "on flat ice CTF refinement measures defocus better.", section="CTF",
                       choices=[["auto", "Automatic (apply only for a clear tilt)"], ["apply", "Always apply"],
                                ["measure", "Measure only"], ["off", "Off"]]),
        "patch_ctf_grid": F("Local defocus regions per side", "Number of regions per side for the local defocus "
                                                              "fit.", section="CTF"),
        "flip_y": F("Flip frames in y (RELION convention)",
                    "Mirror the frames in y after reading. RELION reads TIFF movies y-flipped; turn this on to get "
                    "RELION's micrographs, coordinates, astigmatism angles and map hand. The gain is oriented to the "
                    "flipped frames.", section="Input"),
        "dtype": F("Micrograph precision on disk", "16-bit takes half the disk space and is enough for "
                                                   "micrographs; values that do not fit are written in 32-bit "
                                                   "automatically.", section="Output", choices=DTYPE_CHOICES),
        "figures": F("Control figures", "Which micrographs get a control figure: a sample while the run goes (1, 2, "
                                        "5, 10, 20, 50, then every 100th, plus outliers at the end), all, outliers "
                                        "only, none, or a number N for the first N.", section="Output",
                     choices=[["sample", "Sample"], ["all", "All"], ["outliers", "Outliers"], ["none", "None"]],
                     free=True),
    })

TYPES["Curate"] = dict(
    name="Curate micrographs",
    blurb="Keeps only micrographs whose CTF fit, defocus, astigmatism and motion are within limits, so bad "
          "micrographs are not picked.",
    needs="Micrographs from Motion correction & CTF.",
    typical="Optional, between Motion correction & CTF and Pick particles (set it as the picking job's curation).",
    inputs={"micrographs": I("Micrographs", "The Motion correction & CTF job whose micrographs are judged.")},
    fields={
        "max_ctf_fit_a": B("Worst CTF fit resolution (Å)", "Reject micrographs whose Thon rings could be fitted only "
                                                         "to worse than this (thick ice, drift, carbon). 0: no "
                                                         "limit."),
        "min_defocus_a": B("Minimum defocus (Å)", "Reject micrographs closer to focus than this: too little "
                                                 "low-resolution contrast to pick."),
        "max_defocus_a": B("Maximum defocus (Å)", "Reject micrographs further from focus than this."),
        "max_astigmatism_a": B("Maximum astigmatism (Å)", "Reject a larger defocus difference between the two CTF "
                                                         "axes."),
        "min_ctf_score": F("Minimum CTF fit score", "Reject micrographs with a lower CTF fit score. 0: no limit.",
                           section="CTF"),
        "max_ctf_tilt_deg": F("Maximum ice tilt (°)", "Reject micrographs whose local-defocus plane is steeper "
                                                      "than this (needs local defocus in Motion correction & CTF). "
                                                      "0: no limit.", section="CTF"),
        "max_drift_a": F("Maximum total drift (Å)", "Reject micrographs with a longer total drift. 0: no limit, or "
                                                   "automatic with motion outliers on.", section="Motion"),
        "max_local_rms_a": F("Maximum local motion (Å)", "Reject micrographs with a larger local (patch) motion "
                                                        "residual. 0: no limit, or automatic with motion outliers "
                                                        "on.", section="Motion"),
        "motion_outliers": F("Reject motion outliers", "Set the drift and local-motion limits automatically from the "
                                                       "data (median + a number of MADs). Off by default: motion that "
                                                       "costs resolution already shows in the CTF fit.",
                             section="Motion"),
        "outlier_mad": F("Outlier limit (MADs)", "Automatic limits: this many median absolute deviations above the "
                                                 "median.", section="Motion"),
    })

TYPES["Pick"] = dict(
    name="Pick particles (automatic)",
    blurb="Finds particles on the micrographs with a blob of the particle's size, or with templates from 2D class "
          "averages or projections of a 3D map.",
    needs="Micrographs from Motion correction & CTF.",
    typical="Start with a blob pick; after a first 2D classification, template picking with the good classes picks "
            "more cleanly.",
    inputs={"micrographs": I("Micrographs", "The Motion correction & CTF job whose micrographs are picked.")},
    fields={
        "diameter_a": B("Particle diameter (Å)", "The particle's short dimension. Sets the blob size and, with "
                                                 "templates, the binning, search window and spacing."),
        "templates": B("Templates from job", "Template picking instead of the blob: a 2D classification or "
                                             "selection (its class averages) or a 3D job (projections of its map). "
                                             "0: blob picking.", job_types=("Class2D", "Select", "InitialModel",
                                                                            "Refine3D", "HeteroRefine", "Class3D",
                                                                            "Sharpen")),
        "template_classes": B("Template classes", "With 2D templates: only these classes (1-based numbers, e.g. "
                                                  "\"3, 7, 12\"; empty: all). With a HeteroRefine / Class3D job: the "
                                                  "class whose map is projected (default 1)."),
        "curate": B("Only curated micrographs", "A Curate micrographs job: only the micrographs it accepts are "
                                                "picked (and extracted). 0: all micrographs.", job_types=("Curate",)),
        "dark": F("Particles darker than background", "On for cryo-EM (dark protein); off for negative stain.",
                  section="Picking"),
        "min_dist_frac": F("Minimum distance between picks (× diameter)", "Picks closer than this fraction of the "
                                                                          "diameter are merged. Smaller finds more "
                                                                          "particles in crowded areas but also more "
                                                                          "duplicates.", section="Picking"),
        "threshold": F("Score threshold", "Picks with a lower score are dropped. Empty: chosen automatically per "
                                          "micrograph (generous; 2D classification cleans up). Can be changed later "
                                          "without picking again.", section="Picking"),
        "max_std": F("Contamination filter: local variation", "Drop picks where the local variation is more than this "
                                                              "many times the micrograph's typical value (carbon "
                                                              "edges, ice crystals).", section="Filters"),
        "max_mean": F("Contamination filter: local brightness", "Drop picks where the local mean differs from the "
                                                                "micrograph's by more than this many standard "
                                                                "deviations (thick ice, carbon).", section="Filters"),
        "region_filter": F("Filter by micrograph regions", "Also drop picks on support film, thick ice, edges or "
                                                           "contamination, from a region map of each micrograph.",
                           section="Filters"),
        "template_file": F("Template file", "Templates from outside the project: an .mrcs stack of class averages "
                                            "(white protein) or an .mrc map.", section="Templates"),
        "n_projections": F("Map projections", "With a 3D map as template: number of projection directions over the "
                                              "sphere.", section="Templates"),
        "angle_step": F("In-plane angle step (°)", "Rotation step of the templates. 0: automatic (about 6°).",
                        section="Templates"),
        "figures": F("Control figures", "Which micrographs get a figure with their picks: a sample, all, outliers "
                                        "only, none, or a number N for the first N.", section="Output",
                     choices=[["sample", "Sample"], ["all", "All"], ["outliers", "Outliers"], ["none", "None"]],
                     free=True),
    })

TYPES["ManualPick"] = dict(
    name="Pick by hand, then train",
    blurb="Click particles on a few micrographs; the boxer learns templates from them in seconds, suggests picks on "
          "the next micrographs for you to check, then picks every micrograph.",
    needs="Micrographs from Motion correction & CTF.",
    typical="When a blob pick fails (odd shapes, crowded or dirty ice), or to start template picking without a 2D "
            "classification. 30-50 picks of different views are enough.",
    inputs={"micrographs": I("Micrographs", "The Motion correction & CTF job whose micrographs you pick.")},
    fields={
        "diameter_a": B("Particle diameter (Å)", "The circle drawn around each pick, the templates' window and the "
                                                 "spacing of picks."),
        "dark": F("Particles darker than background", "On for cryo-EM (dark protein); off for negative stain.",
                  section="Picking"),
        "templates": F("Templates to learn", "How many templates (2D classes of your picks) the boxer trains; fewer "
                                             "while there are few picks.", section="Boxer"),
        "suggest": F("Micrographs per round of suggestions", "How many new micrographs the boxer picks for you to "
                                                             "check each time.", section="Boxer"),
        "min_dist_frac": F("Minimum distance between picks (× diameter)", "Picks closer than this fraction of the "
                                                                          "diameter are merged.", section="Picking"),
    },
)

TYPES["Extract"] = dict(
    name="Extract particles",
    blurb="Cuts the particle images out of the micrographs at the picked positions (box sized for the particle and "
          "its CTF-spread signal).",
    needs="Picks from Pick particles.",
    typical="After picking; then 2D classification. Re-extract later with re-centring on aligned particles, or a "
            "smaller pixel size for high resolution.",
    inputs={"picks": I("Picks", "The Pick particles job.")},
    fields={
        "particle_a": B("Particle size (Å)", "The particle's longest dimension. Sets the box (when automatic) and the "
                                             "2D mask."),
        "box_a": B("Box size (Å)", "Edge of the extraction box. 0: automatic, the particle plus its CTF-spread signal "
                                   "at 2 µm defocus."),
        "particle_apix": B("Pixel size after binning (Å)", "Fourier-bin the particles to this pixel size (smaller "
                                                         "files, faster jobs). 0: keep the micrograph pixel size."),
        "select": B("Only selected particles", "A Select job: extract only its particles. 0: all picks.",
                    job_types=("Select",)),
        "recenter": B("Re-centre on aligned particles", "A 2D classification or 3D job: extract its particles, each "
                                                        "centred where the alignment put its centre. 0: at the picked "
                                                        "positions.", job_types=("Class2D", "InitialModel", "Refine3D")),
        "dedup_a": F("Remove duplicates closer than (Å)", "With re-centring: of particles whose new centres are closer "
                                                        "than this, only the best-scoring is kept. -1: half the pick "
                                                        "diameter; 0: keep all.", section="Box"),
        "box_res_a": F("Resolution kept in the box (Å)", "Automatic box: large enough to hold the CTF-spread signal "
                                                        "up to this resolution. A finer value gives a bigger box.",
                       section="Box"),
        "dark": F("Particles darker than background", "On for cryo-EM: the contrast is inverted so the protein is "
                                                      "white. Off for negative stain.", section="Box"),
        "dtype": F("Particle precision on disk", "16-bit takes half the space and is enough for particle images.",
                   section="Output", choices=DTYPE_CHOICES),
        "export": F("Also write for", "Write particles.star (RELION) and / or particles.cs (cryoSPARC) at the end.",
                    section="Output", choices=[["relion", "RELION"], ["cryosparc", "cryoSPARC"]]),
    })

TYPES["ImportParticles"] = dict(
    name="Import particles (RELION / cryoSPARC)",
    blurb="Brings particles from RELION (.star) or cryoSPARC (.cs / .csg) into the project, with their micrographs, "
          "CTF and images, to use like an extraction.",
    needs="A RELION particles .star or a cryoSPARC particles .cs / .csg file (with micrograph names and coordinates).",
    typical="Instead of the movie jobs, to continue processing particles made elsewhere.",
    inputs={},
    fields={
        "path": B("Particles file", "RELION particles .star, or cryoSPARC particles .cs / .csg (e.g. an extraction's "
                                    "J20_particles.csg). Micrograph names and coordinates are required."),
        "particle_a": B("Particle size (Å)", "The particle's longest dimension; the 2D mask is 1.25 × this. 0: 60 % "
                                             "of the box."),
        "particle_apix": B("Pixel size after binning (Å)", "Fourier-bin the particles to this pixel size while "
                                                         "copying. 0: keep the input's."),
        "invert": B("Invert contrast", "Whether the protein is dark in the input images (and gets inverted). Auto "
                                       "decides from the data.",
                    choices=[["auto", "Auto (from the data)"], ["yes", "Yes (protein dark in input)"],
                             ["no", "No (protein already white)"]]),
        "project_dir": F("Base folder of relative paths", "Where the input's relative image paths start (the "
                                                          "RELION / cryoSPARC project folder). Empty: found from the "
                                                          "file's location.", section="Input"),
    })

TYPES["Class2D"] = dict(
    name="2D classification",
    blurb="Sorts the particles into 2D class averages, to see the views and to separate good particles from junk.",
    needs="Particles from Extract particles (or a selection, polished or subtracted particles).",
    typical="After extraction; then select the good classes (Select 2D classes) and run Ab initio 3D.",
    inputs={"particles": I("Particles", "The particles to classify: an extraction or a selection of one.")},
    fields={
        "n_classes": B("Number of classes", "How many 2D classes. More classes separate views and junk more finely "
                                            "but each class gets fewer particles; 50-200 is usual."),
        "diameter_a": B("Mask diameter (Å)", "Diameter of the circular mask on the particles. 0: 1.25 × the "
                                             "particle size given at extraction."),
        "max_res_a": B("Highest resolution used (Å)", "Finest resolution used in alignment and the averages. 0: the "
                                                     "particles' Nyquist, but not finer than 6 Å."),
        "stream": F("Run while extraction is going", "Start classifying while the Extract job still runs: rounds on "
                                                     "the particles so far, each starting from the last round's "
                                                     "classes, and a final round on all particles when extraction "
                                                     "ends.", section="Workflow"),
        "seeds": F("Repeats with different seeds", "More than 1: independent repeats; each particle then gets the "
                                                   "probability that it lands in a good class across repeats.",
                   section="Classification"),
        "gpus": GPUS(),
    })

TYPES["MapScore"] = dict(
    name="Score particles against a map",
    blurb="Tests each particle against a 3D map (and against structureless controls) and proposes which 2D classes "
          "hold real particles: an objective class selection.",
    needs="Particles or a 2D classification, and a 3D map.",
    typical="After a first 3D map: to choose 2D classes, or check a selection, by fit to the map rather than by eye.",
    inputs={"particles": I("Particles or 2D classes", "The particles to score; a 2D classification also gets the share "
                                                      "of each class that fits."),
            "map": I("Map", "The 3D map the particles are scored against.")},
    fields={
        "res_a": B("Comparison resolution (Å)", "Particles and projections are compared to this resolution. Low "
                                                "resolution carries the shape; finer adds noise and time."),
        "max_particles": B("Particles scored", "Score a random subset of this many particles (class shares need "
                                               "about 200 per class). 0: all."),
        "decoy_res_a": F("Decoy resolution (Å)", "Also score against a structureless decoy that keeps only the map's "
                                                 "envelope up to this resolution; a particle fits by how much better "
                                                 "it matches the map than the decoy, so ice and carbon edges do not "
                                                 "pass. 0: map score alone.", section="Scoring"),
        "quantile": F("Control quantile", "A particle fits when it beats this quantile of the phase-randomised "
                                          "controls (0.99 = better than 99 % of them).", section="Scoring"),
        "propose_over_chance": F("Propose class: × chance", "Propose a 2D class when (with 95 % confidence) its share "
                                                            "of fitting particles is at least this many times the "
                                                            "chance rate ...", section="Class proposal"),
        "propose_relative": F("Propose class: × best classes", "... and at least this fraction of the median share of "
                                                               "the 10 best classes. Lower keeps more classes.",
                              section="Class proposal"),
        "n_dirs": F("Projection directions", "Directions over the sphere the map is projected in.", section="Search"),
        "psi_step": F("In-plane angle step (°)", "Rotation step of the search.", section="Search"),
        "shift_a": F("Maximum shift (Å)", "Largest shift searched around the box centre.", section="Search"),
        "diameter_a": F("Comparison circle diameter (Å)", "Diameter of the circle the correlation is computed in. 0: "
                                                          "the particle size given at extraction.", section="Search"),
        "seed": SEED,
        "gpus": GPUS(multi=False),
    })

TYPES["InitialModel"] = dict(
    name="Ab initio 3D",
    blurb="Builds a first 3D map from scratch from 2D classes and their particles, without a starting model.",
    needs="A 2D classification (choose its classes) or a Select 2D classes job.",
    typical="After 2D selection; gives the starting map (and poses) for 3D refinement.",
    inputs={"class2d": I("2D classes", "A 2D classification (pick its classes below) or a selection of its "
                                       "classes.")},
    fields={
        "sym": SYM("Point-group symmetry imposed on the map (C1, Cn, Dn, T, O, I). It is never detected "
                   "automatically; C1 is the safe start."),
        "classes": B("2D classes to use", "Classes of the 2D classification (1-based numbers, e.g. \"1, 4, 9\"). "
                                          "Empty: all populated classes."),
        "n_models": B("Number of maps", "More than 1: that many maps, with the particles sorted among them (ab "
                                        "initio 3D classes)."),
        "mask": F("Solvent mask", "Automatic: a mask made from the map every iteration once below 15 Å. None: the "
                                  "soft particle sphere only.", section="Masking",
                  choices=[["auto", "Automatic"], ["none", "None (soft sphere only)"]]),
        "seed": SEED,
        "gpus": GPUS(),
    })

TYPES["Refine3D"] = dict(
    name="3D refinement",
    blurb="Refines particle orientations against a map with gold-standard half sets, giving the high-resolution map "
          "and its FSC resolution.",
    needs="Particles, and poses and / or a reference map from a 3D job (e.g. Ab initio 3D).",
    typical="After Ab initio 3D or Heterogeneous refinement; repeat after CTF refinement / polishing.",
    inputs={"particles": I("Particles", "The particles to refine (e.g. a re-centred extraction at full box)."),
            "poses": I("Starting poses from", "Starting orientations, matched by particle ID. Empty: orientations are "
                                              "found from the reference map alone (global search)."),
            "reference": P_REFERENCE},
    fields={
        "reference_class": B("Reference class", "When the reference (or poses) job has several classes or models: "
                                                "which class map to start from (1-based; the page shows the classes to "
                                                "click). 0: the class most of the particles are in."),
        "sym": SYM(),
        "mask": B("Solvent mask", "Automatic: made from the map every iteration below 15 Å. Static: made once from "
                                  "the starting map. None: soft sphere only. Or a Create mask job number / .mrc "
                                  "path.", choices=MASK3D_CHOICES, free=True, job_types=("MaskCreate",)),
        "nu": B("Non-uniform refinement", "Filter each region of the map at its own resolution, chosen by "
                                          "cross-validation (the idea of cryoSPARC's NU-refine). Helps membrane "
                                          "proteins and flexible particles."),
        "classes": CLASSES,
        "n_iter": F("Iterations", "Iterations of local orientation search and reconstruction.", section="Alignment"),
        "res_start_a": F("Starting resolution (Å)", "Resolution of the first alignment; the reference is low-passed "
                                                    "to it. 0: the reference job's resolution (at least 8 Å), or 30 Å "
                                                    "without a poses job.", section="Alignment"),
        "n_local": NLOCAL,
        "min_step_deg": F("Finest angular step (°)", "Smallest angular step; the step halves once fewer than 30 % of "
                                                     "particles still move.", section="Alignment"),
        "drop_frac": DROP,
        "global_iters": F("Global search iterations", "Number of first iterations that search all orientations, for "
                                                      "starting poses with errors. Without a poses job at least 2 "
                                                      "are run.", section="Alignment"),
        "global_res_a": F("Global search resolution (Å)", "Resolution of the global search (not finer than the "
                                                          "current one).", section="Alignment"),
        "global_shift_a": F("Global search shift range (Å)", "Without a poses job: how far from the box centre the "
                                                             "first search tries shifts. 0: a tenth of the particle "
                                                             "diameter.", section="Alignment"),
        "subpixel": F("Sub-pixel shifts", "Refine shifts between grid points, finer than half a pixel.",
                      section="Alignment"),
        "particle_scale": F("Per-particle contrast scale", "Fit a contrast scale for every particle (like "
                                                           "cryoSPARC's per-particle scale), used in alignment and "
                                                           "reconstruction.", section="Alignment"),
        "align_symmetry": F("Align map to symmetry axes", "When the symmetry differs from the starting job's: first "
                                                          "rotate map and poses so the symmetry axes lie on the map's "
                                                          "(n-fold on z, 2-fold on x).", section="Alignment"),
        "reference_map": F("Reference map file", "Start from this .mrc map instead of the reference job's (low-passed "
                                                 "to the starting resolution). If its frame differs from the poses, "
                                                 "add global search iterations.", section="Input"),
        "max_res_a": MAXRES,
        "diameter_a": DIAM_3D,
        "nu_strict": F("Strict half-set cutoffs", "Non-uniform refinement: choose each half map's filter cutoffs from "
                                                  "two folds of its own particles (strict gold standard).",
                       section="Non-uniform refinement"),
        "nu_weighted": F("FSC-weighted local filter", "Non-uniform refinement: each region's filter is the FSC "
                                                      "weighting stretched to its resolution, instead of a hard "
                                                      "cutoff.", section="Non-uniform refinement"),
        "nu_awf": F("Adaptive window factor", "Non-uniform refinement: the local error window scaled with each "
                                              "candidate resolution (cryoSPARC uses 3). 0: one window.",
                    section="Non-uniform refinement"),
        "nu_order": F("Butterworth order", "Non-uniform refinement: Butterworth filters of this order (cryoSPARC "
                                           "uses 8). 0: raised-cosine low-passes.", section="Non-uniform refinement"),
        "nu_ratio": F("Fold-to-half SNR ratio", "Non-uniform refinement, strict: apply each cutoff where the half map "
                                                "reaches the folds' SNR (2 = folds hold half the particles). 0: as "
                                                "chosen.", section="Non-uniform refinement"),
        "nu_shift": F("Finer cutoffs (steps)", "Non-uniform refinement, strict: move each region's cutoff this many "
                                               "candidates finer (the folds' choice is conservative).",
                      section="Non-uniform refinement"),
        "marginalize": F("Pose marginalisation", "Insert each particle at its this many best local poses, weighted "
                                                 "by their probability (cryoSPARC's marginalisation). 0: best pose "
                                                 "only.", section="Reconstruction"),
        "noise_model": F("Noise model", "Weights of the frequencies in the alignment.", section="Alignment",
                         choices=[["residual", "From the residuals (default)"], ["white", "White (diagnostic)"]]),
        "rescue_frac": F("Rescue worst-fitting share", "The worst-fitting share of particles searched again over all "
                                                       "orientations and a wide shift range; a new pose is kept when "
                                                       "it fits clearly better. Helps particles stuck off-centre in a "
                                                       "wrong view. 0: off.", section="Alignment"),
        "rescue_iters": F("Rescue iterations", "The iterations (1-based) that rescue.", section="Alignment"),
        "rescue_shift_a": F("Rescue shift range (A)", "Shift range (plus or minus) of the rescue search.",
                            section="Alignment"),
        "rescue_res_a": F("Rescue resolution (A)", "Resolution of the rescue's global search.", section="Alignment"),
        "rescue_topk": F("Rescue candidates", "More than 1: keep this many best poses of the rescue's search and check "
                                              "each at finer resolution (finds more misassigned particles).",
                         section="Alignment"),
        "wiener": F("FSC regularisation (T)", "More than 0: half maps regularised by their FSC, as RELION does (this "
                                              "is RELION's T; 1 = RELION's default). 0: a fixed small regulariser.",
                    section="Reconstruction"),
        "grid_correct": F("Gridding correction", "More accurate projections and maps, a little slower.",
                          section="Reconstruction"),
        "pad": F("Padding", "Oversampling of the 3D Fourier volumes: automatic uses 2 when it fits on the GPU, else "
                            "1.", section="Reconstruction",
                 choices=[[0, "Automatic"], [1, "1 (less memory)"], [2, "2 (more accurate)"]]),
        **SHARPEN,
        "save_iterations": SAVE_ITERS,
        "gpus": GPUS(),
    })

TYPES["HeteroRefine"] = dict(
    name="Heterogeneous refinement",
    blurb="Refines several maps at once and sorts the particles among them, to separate conformations or remove "
          "junk with decoy classes.",
    needs="Particles, their poses from a 3D job, and one reference map per class.",
    typical="After Ab initio 3D (one or more maps) to clean or split particles; then 3D refinement of the best "
            "class.",
    inputs={"particles": P_PARTICLES,
            "poses": I("Starting poses from", "Starting orientations, matched by particle ID.")},
    fields={
        "references": B("Reference maps (one per class)",
                        "Job numbers (\"12\") or job:class (\"25:2\" = model 2 of an Ab initio job); a map may be "
                        "repeated. \"decoy\" adds a junk class made from the first map with its structure scrambled "
                        "beyond the decoy resolution, which collects particles that fit no structure. Judge the "
                        "classes by their maps, not by the reference they started from."),
        "sym": SYM(),
        "mask": B("Solvent mask", "Per class. Automatic: made from each class's map every iteration below 15 Å. "
                                  "Static: once from each reference. None: soft sphere only. Or a Create mask job "
                                  "number / .mrc path (one mask for all classes).",
                  choices=MASK3D_CHOICES, free=True, job_types=("MaskCreate",)),
        "soft": F("Soft assignment", "On: every particle contributes to every class by its probability (RELION). "
                                     "Off: to its best class only. Probabilities are written either way.",
                  section="Classification"),
        "global_search": F("Global search first", "First iteration: every class searches all orientations (needed "
                                                  "when the references are in different frames, e.g. separate ab "
                                                  "initio runs). Off: all start from the given poses.",
                           section="Alignment"),
        "global_res_a": F("Global search resolution (Å)", "Resolution of that first search.", section="Alignment"),
        "decoy_res_a": F("Decoy resolution (Å)", "\"decoy\" references keep the first reference only to this "
                                                 "resolution.", section="Classification"),
        "n_iter": F("Iterations", "Number of iterations.", section="Alignment"),
        "res_start_a": F("Starting resolution (Å)", "Resolution of the first alignment and class comparison.",
                         section="Alignment"),
        "n_local": NLOCAL,
        "drop_frac": DROP,
        "max_res_a": MAXRES,
        "diameter_a": DIAM_3D,
        "seed": F("Random seed", "Seed of the noise added to repeated references, so copies of one map can diverge.",
                  section="Other"),
        **SHARPEN,
        "save_iterations": SAVE_ITERS,
        "gpus": GPUS(),
    })

TYPES["Class3D"] = dict(
    name="3D classification",
    blurb="Sorts particles into several 3D classes at fixed consensus poses (no alignment), optionally focused on a "
          "masked region.",
    needs="Particles and their consensus poses from a 3D refinement.",
    typical="After a consensus 3D refinement, to separate states; use a focus mask for a variable region.",
    inputs={"particles": P_PARTICLES,
            "poses": I("Consensus poses from", "The consensus orientations and shifts (kept fixed), matched by "
                                               "particle ID."),
            "reference": I("Consensus map", "Used for the consensus start and for signal subtraction. Empty: the "
                                            "poses job's map.")},
    fields={
        "reference_class": B("Reference class", "When the reference (or poses) job has several classes or models: "
                                                "which class map to start from (1-based; the page shows the classes to "
                                                "click). 0: the class most of the particles are in."),
        "n_classes": B("Number of classes", "How many 3D classes."),
        "mask": B("Focus mask", "A Create mask job number or .mrc path: only the masked region is compared. Empty: "
                                "the whole particle.", choices=[["", "None (whole particle)"]], free=True,
                  job_types=("MaskCreate",)),
        "res_a": B("Classification resolution (Å)", "Particles are compared with the classes only up to this "
                                                    "resolution (RELION's E-step limit); the maps are reconstructed "
                                                    "finer."),
        "sym": SYM("Point-group symmetry imposed on the class maps (never detected)."),
        "classes": CLASSES,
        "sym_expand": F("Symmetry expansion", "E.g. \"D2\": every particle is classified at each of its "
                                              "symmetry-related poses independently, to classify one subunit of a "
                                              "symmetric particle. Empty: none.", section="Focused classification",
                        choices=SYM_CHOICES[1:], free=True),
        "subtract": F("Subtract outside the mask", "With a focus mask: subtract the consensus map outside the mask "
                                                   "from every particle before comparing. Turn off for particles "
                                                   "that are already subtracted.", section="Focused classification"),
        "init": F("Starting classes", "How the classes start: from the main directions of variability of the "
                                      "particles (best in tests), random particle subsets (RELION), or the consensus "
                                      "map with noise.", section="Classification",
                  choices=[["variability", "Variability (recommended)"], ["random", "Random subsets"],
                           ["consensus", "Consensus map + noise"]]),
        "reg_t": F("Regularisation (T)", "RELION's T: larger trusts the data more, giving more detail but more "
                                         "overfitting of small classes.", section="Classification"),
        "soft": F("Soft assignment", "On: every particle contributes to every class by its probability (RELION). "
                                     "Off: to its best class only (better in tests).", section="Classification"),
        "n_iter": F("Maximum iterations", "Stops earlier once fewer than 0.5 % of the particles change class.",
                    section="Classification"),
        "max_res_a": F("Highest resolution loaded (Å)", "Finest resolution of the particles loaded and of the final "
                                                       "maps and FSC. 0: Nyquist.", section="Resolution"),
        "diameter_a": DIAM_3D,
        "seed": SEED,
        **SHARPEN,
        "gpus": GPUS(),
    })

TYPES["LocalRefine"] = dict(
    name="Local refinement",
    blurb="Refines a masked region of the particle with small searches around the existing poses, rotating about "
          "the mask's centre: a focused refinement.",
    needs="Particles, their poses from a 3D refinement and a focus mask.",
    typical="After a consensus 3D refinement, often on subtracted particles, to improve a flexible region.",
    inputs={"particles": I("Particles", "The particles (e.g. a Particle subtraction job's)."),
            "poses": I("Starting poses from", "Starting orientations (a Symmetry expansion: every copy)."),
            "mask": I("Focus mask", "A Create mask job around the region to refine."),
            "reference": P_REFERENCE},
    fields={
        "reference_class": B("Reference class", "When the reference (or poses) job has several classes or models: "
                                                "which class map to start from (1-based; the page shows the classes to "
                                                "click). 0: the class most of the particles are in."),
        "sym": SYM("Symmetry imposed on the focused region (C1 for symmetry-expanded copies)."),
        "max_angle_deg": B("Maximum rotation (°)", "Orientations are searched only within this angle of the input "
                                                   "pose."),
        "max_shift_a": B("Maximum shift (Å)", "The mask centre is searched only within this distance of its input "
                                              "position."),
        "n_iter": F("Iterations", "Iterations of local search and reconstruction.", section="Alignment"),
        "res_start_a": F("Starting resolution (Å)", "Resolution of the first alignment. 0: the input map's "
                                                    "resolution (at least 8 Å).", section="Alignment"),
        "step_deg": F("Starting angular step (°)", "0: automatic from the starting resolution and the region's "
                                                   "size. Halves once fewer than 30 % of particles move.",
                      section="Alignment"),
        "min_step_deg": F("Finest angular step (°)", "Smallest angular step.", section="Alignment"),
        "n_local": NLOCAL,
        "drop_frac": DROP,
        "max_res_a": MAXRES,
        "diameter_a": F("Particle mask diameter (Å)", "Soft mask on the particle images. 0: from the particles (the "
                                                      "region's size for re-centred subtracted particles).",
                        section="Masking"),
        **SHARPEN,
        "save_iterations": SAVE_ITERS,
        "gpus": GPUS(),
    })

TYPES["CtfRefine"] = dict(
    name="CTF refinement (per-particle defocus)",
    blurb="Fits each particle's own defocus against its half map (gold standard), giving new particles with "
          "refined defocus.",
    needs="A finished 3D refinement.",
    typical="After a good high-resolution 3D refinement; then refine again on its particles.",
    inputs={"refine": P_REFINE},
    fields={
        "range_a": B("Defocus search range (± Å)", "Defocus changes searched, from minus to plus this value."),
        "step_a": F("Search step (Å)", "Coarse search step; then a finer grid around the best value and a parabola "
                                       "fit.", section="Search"),
        "res_lo_a": F("Lowest resolution used (Å)", "Fit band starts at this resolution ...", section="Search"),
        "res_hi_a": F("Highest resolution used (Å)", "... and ends at this one. 0: the refinement's resolution.",
                      section="Search"),
        "plane": F("Per-micrograph plane", "Use a plane fitted to the particles' defocus on each micrograph instead "
                                           "of every particle's own value (less noisy for few particles per "
                                           "micrograph).", section="Search"),
        "batch": F("Batch size", "Particles per GPU batch (reduced automatically if memory is short).",
                   section="Performance"),
        "gpus": GPUS(),
    })

TYPES["Polish"] = dict(
    name="Particle polishing (Bayesian-style)",
    blurb="Tracks each particle's motion through the movie frames against the refined map and re-sums the frames "
          "with per-frame weights, giving sharper particles.",
    needs="A finished high-resolution 3D refinement (from extracted particles).",
    typical="Late: after 3D refinement and CTF refinement, then refine again on the polished particles.",
    inputs={"refine": I("Refinement", "The 3D refinement whose particles and map are used.")},
    fields={
        "min_resolution_a": F("Required map resolution (Å)", "The job is refused unless the refinement reaches this "
                                                            "resolution: per-particle tracks need high-resolution "
                                                            "signal. 0: no check.", section="Requirements"),
        "max_res_a": F("Highest resolution used (Å)", "Finest resolution used for tracks and weights. 0: the "
                                                     "refinement's resolution.", section="Tracks"),
        "sigma_c0_a": F("Prior: position offset (Å)", "How far a particle's mean position is expected to be from its "
                                                      "refined centre.", section="Tracks"),
        "sigma_c1_a": F("Prior: drift (Å)", "Expected straight-line drift of a particle over the whole movie.",
                        section="Tracks"),
        "sigma_c2_a": F("Prior: curvature (Å)", "Expected curvature of a particle's track.", section="Tracks"),
        "stats_movies": F("Movies for frame weights", "Number of random movies used to fit the per-frame weights.",
                          section="Frame weights"),
        "seed": F("Random seed", "Seed for choosing the movies used for the frame weights.", section="Other"),
        "dtype": F("Particle precision on disk", "Precision of the polished particle stacks.", section="Output",
                   choices=DTYPE_CHOICES),
        "batch": F("Batch size", "Particles per GPU batch.", section="Performance"),
    })

TYPES["Modes3D"] = dict(
    name="3D variability (modes)",
    blurb="Finds the main ways the density varies between particles at fixed poses (like cryoSPARC's 3D "
          "variability) and each particle's position along them.",
    needs="Particles and a consensus 3D refinement.",
    typical="After a consensus refinement, to see flexibility; then Variability maps & clusters.",
    inputs={"particles": I("Particles", "The particles."),
            "poses": I("Consensus refinement", "The consensus refinement (poses, shifts, half sets)."),
            "mask": I("Mask", "Region the modes may use. Empty: the refinement's solvent mask.")},
    fields={
        "n_modes": B("Number of modes", "How many modes of variation are solved (together)."),
        "filter_res_a": B("Resolution of the modes (Å)", "Finest resolution of the modes; the particles are loaded "
                                                         "only to this resolution."),
        "sym": SYM("Symmetry imposed on the mean and the modes. C1 allows motions that break the consensus symmetry."),
        "classes": B("Use only classes", "When the poses come from a Heterogeneous refinement: only the particles of "
                                         "these classes (1-based numbers). Empty: all."),
        "highpass_a": F("High-pass (Å)", "Leave out frequencies coarser than this, e.g. 50 when ice or normalisation "
                                         "gradients take the first mode. 0: none.", section="Modes"),
        "n_iter": F("Iterations", "EM iterations of the solve.", section="Modes"),
        "reg_lambda": F("Regularisation", "Larger makes modes weaker where few views sample them; values 0.01-3 made "
                                          "no difference on real data.", section="Modes"),
        "per_particle_scale": F("Per-particle contrast scale", "Fit a contrast scale per particle, so ice thickness "
                                                               "and normalisation differences do not become a mode.",
                                section="Modes"),
        "half_check": F("Check modes on half sets", "Also solve each half set on its own and report how well each "
                                                    "mode reproduces (noise does not). Takes twice the time.",
                        section="Validation"),
        "max_particles": F("Particles used", "A random subset of this many particles. 0: all.", section="Modes"),
        "n_frames": F("Frames per mode", "Frames of each mode's volume series (open the .cxc in ChimeraX).",
                      section="Output"),
        "series_range": F("Series range (SD)", "The series runs from minus to plus this many standard deviations "
                                               "along the mode.", section="Output"),
        "diameter_a": F("Particle diameter (Å)", "For the mean map's sphere mask. 0: from the extraction.",
                        section="Masking"),
        "seed": SEED,
        "gpus": F("GPUs", "Which GPUs to use: \"auto\" or a list such as \"0,2\" (the solve runs on the first).",
                  section="Performance", choices=[["auto", "auto"]], free=True),
    })

TYPES["ModeMaps"] = dict(
    name="Variability maps & clusters",
    blurb="Turns a 3D variability job into maps: a movie along each mode, reconstructions in bins along a mode, or "
          "clusters of particles as 3D classes.",
    needs="A 3D variability (modes) job.",
    typical="After 3D variability; clusters can be selected and refined like 3D classes.",
    inputs={"modes3d": I("3D variability job", "The 3D variability (modes) job.")},
    fields={
        "kind": B("Output", "Series: a volume series along each mode (no particles read). Bins: reconstructions of "
                            "equal-size particle bins along one mode. Clusters: groups of particles found in the mode "
                            "coordinates, one map per cluster.",
                  choices=[["clusters", "Clusters (3D classes)"], ["bins", "Bins along a mode"],
                           ["series", "Volume series"]]),
        "modes": B("Modes used", "1-based mode numbers (e.g. \"1, 2\"); empty: all. Bins use the first one listed."),
        "n_clusters": B("Number of clusters", "Clusters: number of groups."),
        "n_bins": B("Number of bins", "Bins: number of bins along the mode (equal numbers of particles)."),
        "n_frames": F("Frames per mode", "Series: frames per mode.", section="Series"),
        "series_range": F("Series range (SD)", "Series: from minus to plus this many standard deviations along the "
                                               "mode.", section="Series"),
        "max_res_a": F("Highest resolution loaded (Å)", "Reconstructions: finest resolution loaded. 0: Nyquist for "
                                                       "clusters, the modes' resolution for bins.",
                       section="Reconstruction"),
        "sym": SYM("Reconstructions: symmetry. Empty: the 3D variability job's.", level="advanced",
                   section="Reconstruction"),
        "seed": SEED,
        "gpus": GPUS(),
    })

TYPES["MaskCreate"] = dict(
    name="Create mask",
    blurb="Makes a soft mask from a map: automatically, or with your own threshold, kept segments and added / "
          "removed regions.",
    needs="A job with a 3D map (refinement, classes, sharpened map).",
    typical="For focused classification, local refinement, signal subtraction or sharpening with a custom mask.",
    inputs={"map": I("Map from", "The job whose map is masked (its half maps set the automatic threshold).")},
    fields={
        "cls": B("Class", "For a job with several maps (Heterogeneous refinement, 3D classification, variability "
                          "clusters): the class (1-based) whose map is masked."),
        "threshold": B("Threshold", "Density threshold on the low-passed map (map units). 0: automatic, from the "
                                    "noise level."),
        "extend_a": B("Extend (Å)", "The thresholded density is grown by this much ..."),
        "soft_a": B("Soft edge (Å)", "... then given a soft (cosine) edge this wide."),
        "invert": B("Invert", "The mask of everything else, e.g. to subtract a region."),
        "lowpass_a": F("Low-pass (Å)", "Low-pass filter before thresholding. 0: 1.5 × the map's resolution, within "
                                       "8-15 Å.", section="Threshold"),
        "sigma_level": F("Automatic threshold (noise SD)", "Automatic threshold in noise standard deviations (from "
                                                           "the half-map difference).", section="Threshold"),
        "keep_frac": F("Drop small pieces (fraction)", "Connected pieces smaller than this fraction of the largest "
                                                       "are dropped.", section="Threshold"),
        "n_segments": F("Number of segments", "The density is split into this many segments around its strongest "
                                              "peaks (numbered in the figure).", section="Regions"),
        "segments": F("Keep segments", "Keep only these segments (numbers from the figure, e.g. \"2, 5\"). Empty: "
                                       "all.", section="Regions"),
        "include": F("Always include regions", "Regions always in the mask: \"sphere x y z r\" or \"box x0 y0 z0 x1 "
                                               "y1 z1\" in Å from the box corner; separate several with \";\".",
                     section="Regions"),
        "exclude": F("Exclude regions", "Regions taken out, in the same form (e.g. a detergent belt, a flexible "
                                        "domain).", section="Regions"),
    })

TYPES["Sharpen"] = dict(
    name="Sharpen map (post-processing)",
    blurb="Masks the half maps, computes the corrected FSC resolution, applies a B-factor and the local resolution "
          "(like RELION's Post-processing).",
    needs="A 3D refinement (or one class of a classification).",
    typical="At the end of refinement (refinement jobs already sharpen their own maps; use this for another mask or "
            "B-factor).",
    inputs={"refine": I("Refinement", "The refinement whose half maps are sharpened (for a job with classes, set the "
                                      "class).")},
    fields={
        "b_factor": B("B-factor (Å²)", "Applied B-factor; negative sharpens (RELION's sign). 0: automatic from a "
                                      "Guinier fit."),
        "mask": B("Mask", "Automatic: made from the half maps. Refinement's: the mask the refinement used. Or a Create "
                          "mask job number / .mrc path.",
                  choices=[["auto", "Automatic"], ["refine", "Refinement's mask"]], free=True,
                  job_types=("MaskCreate",)),
        "cls": B("Class", "For a job with several maps: the class (1-based) to sharpen."),
        "mtf": B("Detector MTF file", "Detector MTF (RELION .star; RELION ships them in relion/data/, e.g. "
                                      "mtf_k2_300kV.star). The map is divided by it before sharpening. Empty: "
                                      "no correction."),
        "local_res": B("Local resolution", "Also compute the local resolution map (and a locally filtered map)."),
        "resolution_a": F("Low-pass resolution (Å)", "Low-pass of the sharpened map. 0: the corrected masked FSC at "
                                                     "0.143.", section="Filter"),
        "fit_from_a": F("B-factor fit from (Å)", "The Guinier fit runs from this resolution to the map's.",
                        section="B-factor"),
        "fit_weighted": F("Fit on FSC-weighted amplitudes", "Off matches RELION's B-factors in tests.",
                          section="B-factor"),
        "nu_filter": F("Filter region by region", "Sharpen the half maps filtered region by region (like cryoSPARC's "
                                                  "sharpened NU map): domains below the global resolution stay smooth "
                                                  "instead of becoming sharpened noise. Off: one global filter "
                                                  "(RELION).", section="Filter"),
        "fsc_weight": F("FSC weighting", "Weight each resolution shell by its FSC (RELION). Only without the "
                                         "region-by-region filter.", section="Filter"),
        "edge_shells": F("Low-pass edge (shells)", "Width of the soft edge of the low-pass, in Fourier shells.",
                         section="Filter"),
        "mtf_apix": F("MTF pixel size (Å)", "Pixel size the MTF refers to (the detector's pixel, before binning). 0: "
                                           "the half maps' pixel size.", section="Filter"),
        "local_window_a": F("Local resolution window (Å)", "Window size. 0: about 7 × the resolution.",
                            section="Local resolution"),
    })

TYPES["Subtract"] = dict(
    name="Particle subtraction",
    blurb="Removes the map outside a mask from every particle at its pose, optionally symmetry-expands, re-centres on "
          "the mask and re-boxes: particles of just one region.",
    needs="Particles, the refinement whose poses and map are used, and a mask of the region to keep.",
    typical="Before focused 3D classification or local refinement of one region.",
    inputs={"particles": I("Particles", "The particles to subtract."),
            "poses": I("Refinement", "The refinement whose poses, half sets and map are used (a Symmetry expansion: "
                                     "per copy)."),
            "mask": I("Region to keep", "A Create mask job made on that refinement's map.")},
    fields={
        "sym_expand": B("Symmetry expansion", "Each particle once per symmetry operator of this point group, each copy "
                                              "keeping the region at its own symmetry position. C1: no expansion.",
                        choices=SYM_CHOICES, free=True),
        "recentre": B("Re-centre on the mask", "Centre each new box on the mask's centre at the particle's pose."),
        "box": B("New box size (px)", "Box of the new particles (even), cut around the new centre. 0: the particles' "
                                      "box."),
        "subtract": F("Subtract", "Subtract the map outside the mask. Off: only expand, re-centre and re-box.",
                      section="Subtraction"),
        "res_a": F("Map resolution (Å)", "The map is low-passed to this before projection. 0: the refinement's "
                                         "resolution.", section="Subtraction"),
        "scale": F("Scaling of the projections", "Per particle: a per-shell scale and a per-particle scale of the "
                                                 "projection to each image. Per shell: the per-shell scale only.",
                   section="Subtraction", choices=[["particle", "Per particle"], ["shell", "Per shell only"]]),
        "res_lo_a": F("Scale fit from (Å)", "The scale fits use the band from this resolution to the map "
                                            "resolution.", section="Subtraction"),
        "dtype": F("Particle precision on disk", "Precision of the new particle stacks.", section="Output",
                   choices=DTYPE_CHOICES),
        "batch": F("Batch size", "Particles per GPU batch (reduced automatically if memory is short).",
                   section="Performance"),
        "gpus": GPUS(multi=False),
    })

TYPES["SymExpand"] = dict(
    name="Symmetry expansion",
    blurb="Copies every particle once per symmetry operator, at its symmetry-related pose (no new images), for "
          "focused work on one subunit.",
    needs="Poses from a 3D refinement or classification.",
    typical="After a symmetric refinement; then focused 3D classification or local refinement in C1.",
    inputs={"poses": I("Poses from", "The refinement whose poses are expanded."),
            "particles": I("Particles", "The particles. Empty: the refinement's.")},
    fields={
        "sym": B("Symmetry", "Point group to expand by (required; C1 is no expansion).", choices=SYM_CHOICES[1:],
                 free=True),
        "classes": B("Use only classes", "When the poses come from a Heterogeneous refinement / 3D classification: "
                                         "only the particles of these classes (1-based). Empty: all."),
        "seed": F("Random seed", "Half-set split for poses without gold-standard halves (all copies of a particle "
                                 "share it).", section="Other"),
    })


# ------------------------------------------------------------------------------------------------------- merging
def _order() -> dict:
    out, n = {}, 0
    for gi, (g, types) in enumerate(GROUPS):
        for t in types:
            out[t] = (g, gi, n)
            n += 1
    return out


def display_name(job_type: str) -> str:
    """The plain name of a job type (the type itself when unknown)."""
    m = TYPES.get(job_type)
    return m["name"] if m else EXTRA_NAMES.get(job_type, job_type)


def _label(name: str) -> str:
    unit = " (Å)" if name.endswith("_a") else ""
    s = (name[:-2] if unit else name).replace("_", " ").strip()
    return s[:1].upper() + s[1:] + unit


def annotate(described: list[dict]) -> list[dict]:
    """jobspec.describe() output of each job type + the plain-language catalog: every type gets name, group,
    group_order, order, blurb, needs, typical; every setting and input gets label, help, level, section (and
    choices / free / job_types where set). The code's own help stays as `code_help` (inputs: `doc`). A type, setting
    or input missing here still works, labelled from its name and the code's help. Sorted by `order`."""
    order = _order()
    fallback_gi = GROUP_NAMES.index(FALLBACK_GROUP)
    out = []
    for d in described:
        t = d["type"]
        m = TYPES.get(t, {})
        g, gi, n = order.get(t, (FALLBACK_GROUP, fallback_gi, 1000 + len(out)))
        d = dict(d, name=m.get("name", t), group=g, group_order=gi, order=n,
                 blurb=m.get("blurb", d.get("title", "")), needs=m.get("needs", ""), typical=m.get("typical", ""))
        fm = m.get("fields", {})
        sets = []
        for f in d.get("settings", []):
            e = fm.get(f["name"]) or dict(label=_label(f["name"]), help=f.get("help", ""), level="advanced",
                                          section="Other")
            sets.append(dict(f, code_help=f.get("help", ""), **e))
        im = m.get("inputs", {})
        ins = []
        for i in d.get("inputs", []):
            e = im.get(i["name"]) or dict(label=_label(i["name"]), help=i.get("doc", ""))
            ins.append(dict(i, **e))
        d["settings"], d["inputs"] = sets, ins
        out.append(d)
    return sorted(out, key=lambda x: x["order"])
