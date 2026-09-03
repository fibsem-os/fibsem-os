# ruff: noqa: E402  (the repo import comes before the fibsem imports on purpose)
"""Render the figures and the key table for docs/simulator.md from the code.

Every image on that page is produced here, from the scene as it is, and
the key table is read from SampleScene's defaults - so the page cannot
drift from the simulator (the FIB-772 pattern). Run from the repo root:

    python docs/developers/render_simulator_examples.py            # figures + table
    python docs/developers/render_simulator_examples.py --table-only
"""

import dataclasses
import os
import re
import sys
from pathlib import Path

# render the code this file lives with, not an installed copy elsewhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes._stage import SampleGrid
from fibsem.microscopes.sim_scene import SampleScene, fm_channel_weights
from fibsem.projection import BeamStageProjection, FMStageProjection
from fibsem.structures import BeamType, FibsemRectangleSettings, FibsemStagePosition

OUT = ROOT / "docs" / "img" / "simulator"
PAGE = ROOT / "docs" / "simulator.md"
RES = (1536, 1024)

DESCRIPTIONS = {
    "coincidence_offset": "height error at boot (m): what the coincidence alignment corrects",
    "tilt_axis_offset": "how far the surface sits above the tilt axis (m); 0 is eucentric",
    "grids_from_holder": "grids at the holder's calibrated, occupied slots (opt-in)",
    "seed": "the draw everything grows from; same seed, same scene",
    "cell_type": "`mammalian` (default), `yeast`, `bacteria`, `mixed`, `none`",
    "n_clusters": "yeast: clusters over the extent",
    "cells_per_cluster": "yeast: cells per cluster (inclusive range)",
    "cell_size": "yeast: sigma range per cell (m)",
    "cluster_spread": "yeast: how far cells scatter round a cluster (m)",
    "mammalian_density": "mammalian: cells per 150 x 150 um",
    "mammalian_radius": "mammalian: body radius range (m)",
    "nucleus_radius": "mammalian: nucleus radius range (m)",
    "bacteria_density": "bacteria: rods per 150 x 150 um",
    "bacteria_length": "bacteria: rod length range (m)",
    "extent": "content is grown over +/- extent/2 about the grid centre (m)",
    "grid_pitch": "mesh pitch (m); 125 um is a 200 mesh",
    "grid_bar_width": "bar width (m)",
    "grid_intensity": "how bright the bars read in the SEM (0-255 scale)",
    "film": "`continuous` (default) or `holey` (a Quantifoil-style lattice)",
    "hole_diameter": "holey film: hole diameter (m)",
    "hole_pitch": "holey film: hole pitch (m)",
    "broken_hole_fraction": "holey film: fraction of holes broken into larger openings",
    "rip_fraction": "fraction of squares with the film torn; neighbours rip more readily",
    "ice_density": "ice crystals per 100 x 100 um",
    "ice_size": "ice plate radius range (m)",
    "fiducial": "a cross at each grid centre, handy for navigation; absent on a real grid",
    "grid_radius": "usable film radius (m); past it the rim, then the holder",
    "grid_rim_width": "the metal rim's width (m)",
    "noise_sigma": "gaussian noise on the final image",
    "noise_fraction": "blend of full-range uniform noise (0-1)",
    "grid_rotation": "the grid's rotation (deg); random within the range when unset",
    "grid_rotation_range": "the random rotation's range (deg)",
    "contamination_density": "specks per 100 x 100 um, on film and bars alike",
    "contamination_size": "speck sigma range (m)",
    "beam_offset": "per-beam misalignment `{electron: [dx, dy], ion: [dx, dy]}` (m): a persistent lateral offset",
    "current_offset_scale": "sigma (m) of a seeded per-(beam, current) offset; 0 is off",
    "red_fraction": "fraction of cells carrying the subset (mCherry-like) dye",
    "fm_blur_px_per_um": "FM defocus blur, pixels per micron the objective is off focus",
}


def _fmt(value):
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-2:
            return f"{value:.3g}"
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{}" if not value else str(value)
    if value is None:
        return "unset"
    return str(value)


GROUPS = (
    ("Stage and coincidence", ("coincidence_offset", "tilt_axis_offset")),
    ("Beams", ("beam_offset", "current_offset_scale")),
    (
        "Grid and film",
        (
            "grid_pitch",
            "grid_bar_width",
            "grid_intensity",
            "grid_radius",
            "grid_rim_width",
            "grid_rotation",
            "grid_rotation_range",
            "film",
            "hole_diameter",
            "hole_pitch",
            "broken_hole_fraction",
            "fiducial",
        ),
    ),
    (
        "Cells",
        (
            "cell_type",
            "extent",
            "mammalian_density",
            "mammalian_radius",
            "nucleus_radius",
            "n_clusters",
            "cells_per_cluster",
            "cell_size",
            "cluster_spread",
            "bacteria_density",
            "bacteria_length",
        ),
    ),
    (
        "Rips, ice and contamination",
        (
            "rip_fraction",
            "ice_density",
            "ice_size",
            "contamination_density",
            "contamination_size",
        ),
    ),
    ("Noise", ("noise_sigma", "noise_fraction")),
    ("Fluorescence", ("red_fraction", "fm_blur_px_per_um")),
    ("Holder", ("grids_from_holder",)),
    ("Reproducibility", ("seed",)),
)


def key_table() -> str:
    """The keys by group, a table each; a key in no group lands in `Other`
    so nothing added to the scene goes undocumented."""
    defaults = {f.name: f for f in dataclasses.fields(SampleScene)}
    grouped = {k for _, keys in GROUPS for k in keys}
    groups = list(GROUPS)
    other = tuple(k for k in SampleScene.CONFIG_KEYS if k not in grouped)
    if other:
        groups.append(("Other", other))
    out = []
    for title, keys in groups:
        out.append(f"### {title}\n")
        out.append("| key | default | what it does |")
        out.append("| -- | -- | -- |")
        for key in keys:
            field = defaults[key]
            value = (
                field.default
                if field.default is not dataclasses.MISSING
                else field.default_factory()
            )
            if key in ("grid_rotation", "grid_rotation_range") and value is not None:
                value = float(np.rad2deg(value))
            out.append(f"| `{key}` | {_fmt(value)} | {DESCRIPTIONS.get(key, '')} |")
        out.append("")
    return "\n".join(out).rstrip()


def write_table() -> None:
    page = PAGE.read_text()
    page = re.sub(
        r"<!-- sample-keys:start -->.*<!-- sample-keys:end -->",
        "<!-- sample-keys:start -->\n" + key_table() + "\n<!-- sample-keys:end -->",
        page,
        flags=re.S,
    )
    PAGE.write_text(page)
    print("updated", PAGE.relative_to(ROOT))


def save(name, panels, ncols=None, height=4.2):
    ncols = ncols or len(panels)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, height * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, (title, image) in zip(axs, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=72, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope._setup_sample_scene()
    sem_orientation = microscope.get_orientation("SEM")
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(12.0)
    microscope.move_stage_absolute(pose)
    pose = microscope.get_stage_position()
    rng = np.random.default_rng(1)

    def scene(**kwargs):
        scene = SampleScene(**kwargs)
        scene.anchor(pose)
        microscope._sample_scene = scene
        return scene

    def render(beam, hfw, scene=None):
        scene = scene or microscope._sample_scene
        projection = BeamStageProjection.from_microscope(microscope, beam_type=beam)
        return scene.render(beam, pose, hfw, RES, projection, rng=rng)

    # 1. the two beams, one world
    scene()
    save(
        "beams",
        [
            ("SEM, 300 um", render(BeamType.ELECTRON, 300e-6)),
            ("FIB, 300 um", render(BeamType.ION, 300e-6)),
            ("SEM, 80 um", render(BeamType.ELECTRON, 80e-6)),
            ("FIB, 80 um", render(BeamType.ION, 80e-6)),
        ],
        ncols=2,
    )

    # 2. cell types
    panels = []
    for kind in ("mammalian", "yeast", "bacteria", "mixed", "none"):
        scene(cell_type=kind, seed=5)
        panels.append((f"cell_type: {kind}", render(BeamType.ELECTRON, 200e-6)))
    save("cell_types", panels, ncols=3)

    # 3. film, holes, rips, ice, contamination
    scene(film="holey", seed=5)
    holey = render(BeamType.ELECTRON, 60e-6)
    scene(film="continuous", seed=5)
    continuous = render(BeamType.ELECTRON, 60e-6)
    scene(rip_fraction=0.12, seed=7)
    rips = render(BeamType.ELECTRON, 1.2e-3)
    scene(ice_density=3.0, contamination_density=40.0, cell_type="none", seed=3)
    ice = render(BeamType.ELECTRON, 150e-6)
    save(
        "film",
        [
            ("film: continuous, 60 um", continuous),
            ("film: holey, 60 um", holey),
            ("rip_fraction: 0.12, 1.2 mm", rips),
            ("ice_density: 3, contamination_density: 40", ice),
        ],
        ncols=2,
    )

    # 4. milling leaves a trench in the world, seen by both beams
    # coincident views (no height error), so both beams look at the trench
    scene(cell_type="none", contamination_density=0.0, seed=2, coincidence_offset=0.0)
    projection = BeamStageProjection.from_microscope(microscope, beam_type=BeamType.ION)
    microscope._sample_scene.mill(
        [
            FibsemRectangleSettings(
                width=20e-6, height=6e-6, depth=1e-6, centre_x=0, centre_y=8e-6
            )
        ],
        BeamType.ION,
        pose,
        projection,
    )
    save(
        "milling",
        [
            ("FIB after milling a 20 x 6 um rectangle", render(BeamType.ION, 80e-6)),
            ("the same trench in the SEM", render(BeamType.ELECTRON, 80e-6)),
        ],
    )

    # 5. the fluorescence view: the dye model by excitation line (the
    # shuttle Demo has no FM; the Arctis one does)
    arctis, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    fm_scene = SampleScene(seed=5)
    fm_scene.anchor(arctis.get_stage_position())
    fm = arctis.fm
    camera = fm.camera
    fm_projection = FMStageProjection.from_microscope(arctis)
    fm_projection = FMStageProjection(
        geometry=fm_projection.geometry,
        pixel_size=camera.pixel_size[0],
        shape=(camera.resolution[1], camera.resolution[0]),
    )
    panels = []
    # reflection is an emission of None; a dye channel is the open band with
    # its excitation line. Shown in the false colour a viewer would give
    # each channel, and composited the way a CLEM overlay is
    channels = (
        ("reflection", None, 550.0, (0.85, 0.85, 0.85)),
        ("DAPI-like (365 nm excitation)", "Fluorescence", 365.0, (0.25, 0.45, 1.0)),
        ("GFP-like (470 nm excitation)", "Fluorescence", 470.0, (0.2, 1.0, 0.3)),
        ("mCherry-like (560 nm excitation)", "Fluorescence", 560.0, (1.0, 0.25, 0.2)),
    )
    panels = []
    composite = np.zeros((camera.resolution[1], camera.resolution[0], 3))
    for label, emission, excitation, colour in channels:
        frame = fm_scene.render_fm(
            arctis.get_stage_position(),
            camera.resolution,
            fm_projection,
            weights=fm_channel_weights(emission, excitation),
            rng=rng,
        ).astype(float)
        # a viewer's auto-contrast: the dim background off, the bright tail in
        normalised = np.clip(
            (frame - 400.0) / (np.percentile(frame, 99.7) - 400.0), 0, 1
        )
        panels.append((label, normalised[..., None] * np.array(colour)))
        # the dyes add into the overlay; reflection sits faintly underneath
        weight = 0.25 if emission is None else 1.0
        composite += weight * normalised[..., None] * np.array(colour)
    panels.append(("composite", np.clip(composite, 0, 1)))
    save("fluorescence", panels, ncols=3)

    # 5b. one world, three views: the beams and the FM at one stage position
    at = arctis.get_stage_position()
    fm_hfw = camera.resolution[0] * camera.pixel_size[0]
    triplet = []
    for label, beam in (("SEM", BeamType.ELECTRON), ("FIB", BeamType.ION)):
        projection = BeamStageProjection.from_microscope(arctis, beam_type=beam)
        triplet.append(
            (
                f"{label}, {fm_hfw * 1e6:.0f} um",
                fm_scene.render(beam, at, fm_hfw, RES, projection, rng=rng),
            )
        )
    triplet.append(("FM composite", np.clip(composite, 0, 1)))
    save("triplet", triplet, ncols=3)
    arctis.disconnect()

    # 6. grids at the holder's slots
    holder = microscope._stage.holder
    for slot, (x, name) in zip(
        sorted(holder.slots.values(), key=lambda s: s.index),
        ((-1.5e-3, "Grid-A"), (1.5e-3, "Grid-B")),
    ):
        slot.position = FibsemStagePosition(
            x=x, y=0.0, z=0.0, r=sem_orientation.r, t=sem_orientation.t
        )
        slot.loaded_grid = SampleGrid(name=name)
    two = scene(grids_from_holder=True, seed=5)
    two.holder_slots = microscope._scene_holder_slots()
    save(
        "holder",
        [
            (
                "grids_from_holder: two grids on a shuttle, SEM 4.5 mm",
                render(BeamType.ELECTRON, 4.5e-3),
            )
        ],
    )

    write_table()


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        write_table()
    else:
        main()
