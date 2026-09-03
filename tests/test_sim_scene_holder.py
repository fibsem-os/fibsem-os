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
