"""The scene's grids come from the sample holder: one grid per occupied
slot with a position, centred where that slot's stage position falls, its
content seeded by the grid's name; with no such slot, the single grid at
the anchor the scene always had."""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes._stage import SampleGrid
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

TFS_SHUTTLE_CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
SLOT_X = 3e-3  # m, slots either side of the boot position


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope._setup_sample_scene()
    scene = microscope._sample_scene
    # a quiet scene: the fiducial is the landmark, nothing else in the way
    scene.fiducial = True
    scene.cell_type = "none"
    scene.contamination_density = 0.0
    scene.ice_density = 0.0
    scene.rip_fraction = 0.0
    scene.noise_sigma = 0.0
    scene.noise_fraction = 0.0
    scene.features = []
    scene.__post_init__()
    scene.grids_from_holder = True
    yield microscope
    microscope.disconnect()


def _settings(hfw=150e-6):
    return ImageSettings(
        resolution=[768, 512], hfw=hfw, beam_type=BeamType.ELECTRON, autocontrast=False
    )


def _slot_position(microscope, x: float) -> FibsemStagePosition:
    at = microscope.get_orientation("SEM")
    return FibsemStagePosition(x=x, y=0.0, z=0.0, r=at.r, t=at.t)


def _calibrate(microscope, grids):
    """Put a grid (or None) in each holder slot, at x positions."""
    holder = microscope._stage.holder
    for slot, (x, grid) in zip(
        sorted(holder.slots.values(), key=lambda s: s.index), grids
    ):
        slot.position = _slot_position(microscope, x)
        slot.loaded_grid = None if grid is None else SampleGrid(name=grid)


def _has_fiducial(frame: np.ndarray) -> bool:
    """A bright cross at the frame centre reads far above the film."""
    h, w = frame.shape
    centre = frame[h // 2 - 40 : h // 2 + 40, w // 2 - 40 : w // 2 + 40]
    return float(np.percentile(centre, 99)) > float(np.median(frame)) + 60


def test_without_calibrated_slots_the_grid_sits_at_the_anchor(microscope):
    frame = microscope.acquire_image(_settings()).data
    scene = microscope._sample_scene
    assert [g.name for g in scene.grids] == ["default"]
    assert _has_fiducial(frame)


def test_holder_grids_are_opt_in(microscope):
    """The holder file is shared by every configuration in the directory:
    a calibrated holder must not move the grids unless the scene asks."""
    microscope._sample_scene.grids_from_holder = False
    _calibrate(microscope, [(-SLOT_X, "A"), (SLOT_X, "B")])
    frame = microscope.acquire_image(_settings()).data
    assert [g.name for g in microscope._sample_scene.grids] == ["default"]
    assert _has_fiducial(frame)


def test_grids_sit_at_the_occupied_slots(microscope):
    _calibrate(microscope, [(-SLOT_X, "A"), (SLOT_X, "B")])
    boot = microscope.acquire_image(_settings()).data  # between the grids
    scene = microscope._sample_scene
    assert sorted(g.name for g in scene.grids) == ["A", "B"]
    by_name = {g.name: g for g in scene.grids}
    assert abs(abs(by_name["A"].x) - SLOT_X) < 1e-6
    assert abs(abs(by_name["B"].x) - SLOT_X) < 1e-6
    assert np.sign(by_name["A"].x) == -np.sign(by_name["B"].x)
    # off both grids the beam sees the holder: dark, no fiducial
    assert float(np.median(boot)) < 20
    assert not _has_fiducial(boot)

    microscope.move_stage_absolute(_slot_position(microscope, -SLOT_X))
    on_grid = microscope.acquire_image(_settings()).data
    assert float(np.median(on_grid)) > 40
    assert _has_fiducial(on_grid)


def test_content_follows_the_grid_between_slots(microscope):
    _calibrate(microscope, [(-SLOT_X, "A"), (SLOT_X, "B")])
    microscope.acquire_image(_settings())
    before = {g.name: g for g in microscope._sample_scene.grids}

    _calibrate(microscope, [(-SLOT_X, "B"), (SLOT_X, "A")])  # swapped
    microscope.acquire_image(_settings())
    after = {g.name: g for g in microscope._sample_scene.grids}

    assert after["A"].seed == before["A"].seed
    assert after["A"].rotation == before["A"].rotation
    assert abs(after["A"].x - before["B"].x) < 1e-9  # A now where B was


def test_an_empty_slot_shows_the_holder(microscope):
    _calibrate(microscope, [(-SLOT_X, "A"), (SLOT_X, None)])
    microscope.move_stage_absolute(_slot_position(microscope, SLOT_X))
    frame = microscope.acquire_image(_settings()).data
    assert [g.name for g in microscope._sample_scene.grids] == ["A"]
    assert float(np.median(frame)) < 20


def test_the_rim_rings_each_grid(microscope):
    _calibrate(microscope, [(-SLOT_X, "A"), (SLOT_X, "B")])
    microscope.acquire_image(_settings())
    scene = microscope._sample_scene
    grid = scene.grids[0]
    # a line of world points out from the grid centre, across the rim
    r = np.linspace(0, grid.radius + 2 * scene.grid_rim_width, 400)
    bars, holes, rips, rim, beyond = scene.film_masks(grid.x + r, grid.y + 0 * r)
    assert not rim[r < grid.radius].any()
    assert rim[(r > grid.radius) & (r < grid.radius + scene.grid_rim_width)].all()
    assert beyond[r > grid.radius + scene.grid_rim_width].all()
    assert not beyond[r < grid.radius].any()


# ---------------------------------------------------------------------------
# The Arctis working slot (FIB-1144): the autoloader puts a grid off the origin
# ---------------------------------------------------------------------------

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
GRID_POSITION = (200e-6, 100e-6, 0.0)  # m: where the loader puts a grid


def _arctis(grid_position=GRID_POSITION, captured=None):
    """The Arctis simulator with its loader putting grids at *grid_position*
    and, if given, a captured working-slot position in its configuration."""
    from fibsem.microscopes._stage import (
        COMPUSTAGE_HOLDER_NAME,
        GridSlot,
        SampleHolder,
        SlotCalibration,
        _create_sample_stage,
    )

    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=ARCTIS_CONFIG)
    microscope.devices["sample_loader"].sim_grid_position = tuple(grid_position)
    microscope.system.sim = dict(microscope.system.sim, coincidence_projection=True)
    stage_settings = microscope.system.stage
    if captured is not None:
        slot = GridSlot(
            name="Slot-01",
            index=0,
            position=captured,
            calibration=SlotCalibration(
                orientation="SEM",
                pre_tilt=float(stage_settings.shuttle_pre_tilt),
                rotation_reference=float(stage_settings.rotation_reference),
                captured_at="2026-10-02T12:00:00",
                fibsem_version="test",
            ),
        )
        stage_settings.holders = {
            COMPUSTAGE_HOLDER_NAME: SampleHolder(
                pre_tilt=float(stage_settings.shuttle_pre_tilt),
                name=COMPUSTAGE_HOLDER_NAME,
                capacity=1,
                slots={"Slot-01": slot},
            )
        }
    microscope._stage = _create_sample_stage(microscope)
    microscope._setup_sample_scene()
    scene = microscope._sample_scene
    scene.fiducial = True
    scene.cell_type = "none"
    scene.contamination_density = 0.0
    scene.ice_density = 0.0
    scene.rip_fraction = 0.0
    scene.noise_sigma = 0.0
    scene.noise_fraction = 0.0
    scene.features = []
    scene.__post_init__()
    scene.grids_from_holder = True
    microscope._stage.get_inventory()
    microscope._stage.ensure_loaded("Grid-01")
    return microscope


def _image_at(microscope, orientation: str, beam: BeamType) -> np.ndarray:
    slot = microscope._stage.holder.slots["Slot-01"]
    target = microscope.get_target_position(slot.position, orientation)
    microscope.safe_absolute_stage_movement(target)
    settings = _settings()
    settings.beam_type = beam
    return microscope.acquire_image(settings).data


def _fiducial_at(frame: np.ndarray):
    """Where the grid's centre cross is in the frame, in pixels from the frame
    centre (x, y), or None when it is not in the frame."""
    ys, xs = np.nonzero(frame > float(np.median(frame)) + 60)
    if len(xs) == 0:
        return None
    h, w = frame.shape
    return (float(np.median(xs)) - w / 2, float(np.median(ys)) - h / 2)


def _fiducials(microscope):
    try:
        return {
            "SEM": _fiducial_at(_image_at(microscope, "SEM", BeamType.ELECTRON)),
            "FIB": _fiducial_at(_image_at(microscope, "FIB", BeamType.ION)),
        }
    finally:
        microscope.disconnect()


def _close(a, b, px=3.0):
    return (
        a is not None and b is not None and all(abs(u - v) <= px for u, v in zip(a, b))
    )


def _sem_pose():
    probe, _ = utils.setup_session(manufacturer="Demo", config_path=ARCTIS_CONFIG)
    try:
        return probe.get_orientation("SEM")
    finally:
        probe.disconnect()


def test_a_working_slot_off_the_grid_misses_it_and_a_captured_one_finds_it():
    """The reference is the grid at the origin with the built-in slot: the sim
    starts out of coincidence, so the cross is not dead centre even there. A
    grid the loader puts 200 um off is missed by the built-in slot, and found
    exactly as the reference was, in both beams, once the slot is captured."""
    reference = _fiducials(_arctis(grid_position=(0.0, 0.0, 0.0)))
    assert reference["SEM"] is not None and reference["FIB"] is not None

    missed = _arctis()
    assert missed._stage.holder.slots["Slot-01"].calibration.is_builtin
    assert not _close(_fiducials(missed)["SEM"], reference["SEM"])

    at = _sem_pose()
    x, y, z = GRID_POSITION
    captured = _arctis(
        captured=FibsemStagePosition(name="Slot-01", x=x, y=y, z=z, r=at.r, t=at.t)
    )
    slot = captured._stage.holder.slots["Slot-01"]
    assert not slot.calibration.is_builtin
    found = _fiducials(captured)
    assert _close(found["SEM"], reference["SEM"])
    assert _close(found["FIB"], reference["FIB"])
