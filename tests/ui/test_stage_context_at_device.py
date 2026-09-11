"""Holder places are written in the beams' frame; the FM canvas on an offset mount is not.

A slot at x = 0 in the SEM pose, handed raw to a frame anchored 48.8 mm away in the
FIB pose, is drawn 48.8 mm off that canvas -- with the grid boundary around it. The
overlays carry the place to the frame's device first, with the same transform a
lamella's fluorescence pose is derived with, so a slot and the lamellae in it agree.

Under tests/ui because `fibsem.ui` imports Qt on import; the geometry itself needs none.
"""

import os

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import DeviceImagingState, FibsemStagePosition
from fibsem.ui.widgets.canvas.overlays.stage_context import (
    at_device,
    slot_landmark,
)

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
FM_X = 48.8e-3


@pytest.fixture(scope="module")
def iflm():
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    return microscope


@pytest.fixture(scope="module")
def arctis():
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    return microscope


def _slot_place(microscope):
    """The first holder slot's place, calibrated here if the holder file has not.

    `sample-holder.yaml` is runtime-written state: a dev box that has calibrated a
    slot ships a position, CI has none and the slot's position is None. The test is
    about where a place is carried to, not about whether anyone calibrated it.
    """
    slot = next(iter(microscope._stage.holder.slots.values()))
    if slot.position is None:
        slot.position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, name=slot.name)
    place = slot_landmark(microscope, slot)
    assert place is not None
    return place


def test_a_slot_is_carried_to_an_offset_fm(iflm):
    place = _slot_place(iflm)
    assert iflm.is_at_device("FIBSEM", place)

    carried = at_device(iflm, "FM", place)

    assert iflm.get_device_imaging_state("FM", carried) is DeviceImagingState.READY
    # Not `FM_X` exactly: the SEM-to-FIB half turn is compucentric, so the place is
    # mirrored about the instrument's rotation centre before the traverse. Within the
    # device's range is what "at the FM" means, and READY has already said so.
    assert carried.x > FM_X / 2


def test_the_carried_slot_agrees_with_a_derived_fluorescence_pose(iflm):
    """The same transform, so a lamella marked at the slot centre draws on top of the
    slot's crosshair rather than beside it."""
    from fibsem.applications.autolamella.poses import build_lamella_poses

    place = _slot_place(iflm)
    carried = at_device(iflm, "FM", place)
    poses = build_lamella_poses(iflm, place)

    fluorescence = poses.fluorescence.stage_position
    assert carried.x == pytest.approx(fluorescence.x, abs=1e-9)
    assert carried.y == pytest.approx(fluorescence.y, abs=1e-9)


def test_the_beam_canvas_leaves_the_slot_alone(iflm):
    place = _slot_place(iflm)

    assert at_device(iflm, "FIBSEM", place) is place
    assert at_device(iflm, None, place) is place


def test_it_does_not_depend_on_where_the_stage_is(iflm):
    """The FM canvas's markers do not move when the stage does; neither may the slot
    they sit in. Keyed on the device the canvas is for, not on the stage."""
    place = _slot_place(iflm)
    sem = iflm.get_orientation("SEM")
    iflm.move_stage_absolute(FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t))
    at_the_beams = at_device(iflm, "FM", place)
    iflm.move_to_device("FM")
    at_the_fm = at_device(iflm, "FM", place)

    assert at_the_fm.x == pytest.approx(at_the_beams.x, abs=1e-12)
    assert at_the_fm.y == pytest.approx(at_the_beams.y, abs=1e-12)


def test_a_compustage_draws_where_it_always_did(arctis):
    """Its FM is a flip rather than a place, so the frame is always at the beams."""
    place = _slot_place(arctis)

    assert at_device(arctis, "FM", place) is place
