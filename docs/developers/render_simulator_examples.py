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
from fibsem.drivers.demo.sim.scene import SampleScene, fm_channel_weights
from fibsem.geometry.projection import BeamStageProjection, FMStageProjection
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    FibsemStagePosition,
    SampleGrid,
)

OUT = ROOT / "docs" / "img" / "simulator"
PAGE = ROOT / "docs" / "simulator.md"
RES = (1536, 1024)

DESCRIPTIONS = {
    "coincidence_offset": "height error at boot (m): what the coincidence alignment corrects",
    "tilt_axis_offset": "how far the surface sits above the tilt axis (m); 0 is eucentric",
    "beam_offset": "per-beam misalignment `{electron: [dx, dy], ion: [dx, dy]}` (m): a persistent lateral offset",
    "current_offset_scale": "sigma (m) of a seeded per-(beam, current) offset; 0 is off",
    "beam_optics": "working distance, stigmation and current show in the image (see Beam settings)",
    "focus_fuzz": "how far a beam's focus is from its eucentric height, at most (m)",
    "scan_rotation_at_start": "the scan rotation each beam boots at, `{electron: deg, ion: deg}`",
    "fm_camera_mount": "the mount transform the FM camera really has; unset is as configured",
    "imaging_damage": "how fast FIB imaging wears the sample away (1 is ice, 0 is never)",
    "surface_grain": "how strongly a surface's fine texture shows (0 is smooth; see Beam settings)",
    "faults": 'things that go wrong and when, see "Things going wrong"',
    "grid_pitch": "mesh pitch (m); 125 um is a 200 mesh",
    "grid_bar_width": "bar width (m)",
    "grid_mesh": "`square`, or `hexagonal` for a honeycomb of hexagons `grid_pitch` across their flats",
    "grid_intensity": "how much reflected light from the bars leaks into a fluorescence channel",
    "grid_radius": "usable film radius (m); past it the rim, then the holder",
    "grid_rim_width": "the metal rim's width (m)",
    "grid_rotation": "the grid's rotation (deg); random within the range when unset",
    "grid_rotation_range": "the random rotation's range (deg)",
    "film": "`holey` (default, a Quantifoil-style lattice) or `continuous`",
    "hole_diameter": "holey film: hole diameter (m)",
    "hole_pitch": "holey film: hole pitch (m)",
    "broken_hole_fraction": "holey film: fraction of holes broken into larger openings",
    "centre_mark": "the orientation mark at the grid's centre: a bigger square of metal with one corner rounded",
    "fiducial": "a cross at each grid centre, a navigation landmark; absent on a real grid, so off by default",
    "preparation": "`plunge`, `waffle` for a high-pressure frozen slab on a grid, or `carrier` for a block in a carrier",
    "slab_thickness": "how thick the ice is (m): 5 to 50 um in a waffle, 50 to 300 um in a carrier",
    "hat_scratches": "waffle: how strongly the hat's polishing lines show in the top (0 = none)",
    "slab_fill": "waffle: the share of the slab's volume that is cells",
    "suspended_rpe_radius": "waffle: the radius of a rounded RPE cell (m)",
    "worm_length": "waffle: how long a `celegans` worm is (m)",
    "worm_radius": "waffle: how thick it is at its middle, as a radius (m)",
    "embryo_length": "carrier: how long a `drosophila` embryo is (m)",
    "embryo_width": "carrier: how wide it is (m)",
    "cell_type": "`rpe` (default), `mammalian`, `neuron`, `yeast`, `bacteria`, `mixed`, `none`; in a waffle `yeast`, `rpe`, `celegans`, `none`; in a carrier those and `drosophila`",
    "extent": "content is grown over +/- extent/2 about the grid centre, on the grid's disc (m); unset fills the whole grid",
    "rpe_density": "rpe: cells per 150 x 150 um",
    "rpe_radius": "rpe: cell radius range (m)",
    "mammalian_density": "mammalian: cells per 150 x 150 um",
    "mammalian_radius": "mammalian: body radius range (m)",
    "nucleus_radius": "mammalian: nucleus radius range (m)",
    "n_clusters": "yeast: clusters per 800 x 800 um",
    "cells_per_cluster": "yeast: cells per cluster (inclusive range)",
    "cell_size": "yeast: sigma range per cell (m)",
    "cluster_spread": "yeast: how far cells scatter round a cluster (m)",
    "bacteria_density": "bacteria: rods per 150 x 150 um",
    "bacteria_length": "bacteria: rod length range (m)",
    "neuron_density": "neuron: somas per 150 x 150 um",
    "soma_radius": "neuron: soma radius range (m)",
    "neurite_length": "neuron: how long its processes are (m)",
    "rip_fraction": "fraction of squares with the film torn; neighbours rip more readily",
    "grid_quality": 'what each grid is like: a name, a list in slot order or a mapping by grid name, see "Grids at the holder\'s slots"',
    "ice_density": "ice crystals per 100 x 100 um",
    "ice_size": "ice plate radius range (m)",
    "contamination_density": "specks per 100 x 100 um, on film and bars alike",
    "contamination_size": "speck sigma range (m)",
    "frost_density": "frost patches per 100 x 100 um",
    "sphere_density": "dark spheres per 100 x 100 um",
    "sphere_size": "sphere radius range (m)",
    "crack_density": "hairline cracks in the film per 100 x 100 um",
    "mill_drift": "sigma of each milling pass's drift (m)",
    "mill_overcut": "how far past its edges a pattern may cut (m)",
    "mill_kerf": "a higher current cuts wider",
    "mill_taper": "degrees the trench walls lean in going down",
    "curtaining": "strength of the stripes down a cut face (0 = none)",
    "damage_below": "the seeded thickness below which damage can start (m)",
    "damage_chance": "chance per pass below it that damage starts",
    "damage_current": "the milling current `damage_below` is given at (A); 0 and the current makes no difference",
    "charging": "SEM charging strength on an uncoated grid (0 = never)",
    "gis_thickness": "GIS coat the grid starts with (m)",
    "sputter_thickness": "sputter coat the grid starts with (m)",
    "noise_sigma": "gaussian noise on the final image",
    "noise_fraction": "blend of full-range uniform noise (0-1)",
    "red_fraction": "fraction of cells carrying the subset (mCherry-like) dye",
    "fm_blur_px_per_um": "FM defocus blur, pixels per micron the objective is off focus",
    "fm_target_fraction": "cells with a centrin pool and a centriole pair",
    "fm_no_target_chance": "chance a grid carries no pair at all",
    "fm_axial_range": "how far from its height an FM spot stays sharp (m)",
    "fm_hot_pixels": "fraction of FM camera pixels that are hot",
    "fm_green": "green label: cytoplasm, centrin or lipid_droplets",
    "fm_red": "red label: subset or cilia",
    "fm_far_red": "far-red label: none, membrane or mitochondria",
    "fm_cilium_fraction": "of the target cells, those with a primary cilium",
    "grids_from_holder": "grids at the holder's calibrated, occupied slots (opt-in)",
    "seed": "the draw everything grows from; same seed, same scene",
    "state_file": 'a file the sample\'s state is kept in between sessions, see "Keeping the sample"',
}


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"  # as it is written in the yaml
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
    (
        "Stage and coincidence",
        (
            "coincidence_offset",
            "tilt_axis_offset",
        ),
    ),
    (
        "Beams",
        (
            "beam_offset",
            "current_offset_scale",
            "beam_optics",
            "focus_fuzz",
            "scan_rotation_at_start",
            "fm_camera_mount",
            "imaging_damage",
            "surface_grain",
            "faults",
        ),
    ),
    (
        "Grid and film",
        (
            "grid_pitch",
            "grid_bar_width",
            "grid_mesh",
            "grid_intensity",
            "grid_radius",
            "grid_rim_width",
            "grid_rotation",
            "grid_rotation_range",
            "film",
            "hole_diameter",
            "hole_pitch",
            "broken_hole_fraction",
            "centre_mark",
            "fiducial",
        ),
    ),
    (
        "Waffle",
        (
            "preparation",
            "slab_thickness",
            "hat_scratches",
            "slab_fill",
            "suspended_rpe_radius",
            "worm_length",
            "worm_radius",
            "embryo_length",
            "embryo_width",
        ),
    ),
    (
        "Cells",
        (
            "cell_type",
            "extent",
            "rpe_density",
            "rpe_radius",
            "mammalian_density",
            "mammalian_radius",
            "nucleus_radius",
            "n_clusters",
            "cells_per_cluster",
            "cell_size",
            "cluster_spread",
            "bacteria_density",
            "bacteria_length",
            "neuron_density",
            "soma_radius",
            "neurite_length",
        ),
    ),
    (
        "Rips, ice and contamination",
        (
            "rip_fraction",
            "grid_quality",
            "ice_density",
            "ice_size",
            "contamination_density",
            "contamination_size",
            "frost_density",
            "sphere_density",
            "sphere_size",
            "crack_density",
            "mill_drift",
            "mill_overcut",
            "mill_kerf",
            "mill_taper",
            "curtaining",
            "damage_below",
            "damage_chance",
            "damage_current",
            "charging",
            "gis_thickness",
            "sputter_thickness",
        ),
    ),
    (
        "Noise",
        (
            "noise_sigma",
            "noise_fraction",
        ),
    ),
    (
        "Fluorescence",
        (
            "red_fraction",
            "fm_blur_px_per_um",
            "fm_target_fraction",
            "fm_no_target_chance",
            "fm_axial_range",
            "fm_hot_pixels",
            "fm_green",
            "fm_red",
            "fm_far_red",
            "fm_cilium_fraction",
        ),
    ),
    ("Holder", ("grids_from_holder",)),
    (
        "Reproducibility",
        (
            "seed",
            "state_file",
        ),
    ),
)


def key_table() -> str:
    """The keys by group, a table each; a key in no group lands in `Other`
    so nothing added to the scene goes undocumented."""
    defaults = {f.name: f for f in dataclasses.fields(SampleScene)}
    configured = SampleScene.config_defaults()
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
            # a configured scene starts with some effects on (see from_config)
            value = configured.get(key, value)
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
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
        ),
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope._connection.setup_sample_scene()
    sem_orientation = microscope.get_orientation("SEM")
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(12.0)
    microscope.move_stage_absolute(pose)
    pose = microscope.get_stage_position()
    rng = np.random.default_rng(1)

    def scene(**kwargs):
        # the figures keep the fiducial as a landmark; it is off by default
        scene = SampleScene(**{"fiducial": True, **kwargs})
        scene.anchor(pose)
        microscope._connection.sample_scene = scene
        return scene

    def render(beam, hfw, scene=None):
        scene = scene or microscope._connection.sample_scene
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
    microscope._connection.sample_scene.mill(
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
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        )
    )
    fm_scene = SampleScene(seed=5, fiducial=True)
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
    holder = microscope.sample_handling.holder
    for slot, (x, name) in zip(
        sorted(holder.slots.values(), key=lambda s: s.index),
        ((-1.5e-3, "Grid-A"), (1.5e-3, "Grid-B")),
    ):
        slot.position = FibsemStagePosition(
            x=x, y=0.0, z=0.0, r=sem_orientation.r, t=sem_orientation.t
        )
        slot.loaded_grid = SampleGrid(name=name)
    two = scene(grids_from_holder=True, seed=5)
    two.holder_slots = microscope._connection.scene_holder_slots()
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
