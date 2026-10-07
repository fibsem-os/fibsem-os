"""Device moves on one path: a device is its pose plus its origin.

A compustage reaches its FM by turning the grid over to face the objective. When the
objective's centre is offset from the beams' coincidence point, that offset is the FM's
origin, in stage coordinates at the FM pose, and the stage travels by it after the flip
just as an offset mount travels to its FM. Which device the stage is at stays decided by
the pose (`Stage.device_at_pose`), and the FM has no window: the stage limits bound it.

A compustage whose FM shares the beams' origin takes the same path with a zero travel:
the flip, and nothing else. `tests/test_device_moves_pinned.py` pins its commands.
"""

import logging

import pytest

from fibsem.applications.autolamella.poses import _to_fluorescence, _to_milling
from fibsem.structures import DeviceImagingState, FibsemStagePosition
from tests import test_device_moves_pinned as pinned

OFFSET = FibsemStagePosition(x=50e-6, y=-30e-6)


def _offset_arctis():
    microscope = pinned._microscope("arctis")
    microscope.system.stage.devices["FM"].origin = FibsemStagePosition(
        x=OFFSET.x, y=OFFSET.y
    )
    return microscope


def _at(microscope, start: str) -> FibsemStagePosition:
    return pinned._go(microscope, start)


def _assert_xy(position, x, y):
    assert position.x == pytest.approx(x, abs=1e-12)
    assert position.y == pytest.approx(y, abs=1e-12)


# -- moving -----------------------------------------------------------------------------


def test_the_fm_is_the_flip_then_the_offset():
    microscope = _offset_arctis()
    start = _at(microscope, "SEM")
    commands = pinned._record_commands(microscope)

    microscope.move_to_device("FM")

    arrived = microscope.get_stage_position()
    _assert_xy(arrived, start.x + OFFSET.x, start.y + OFFSET.y)
    assert microscope.get_stage_orientation() == "FM"
    assert microscope.get_current_device() == "FM"
    assert microscope.fm.objective.state == "Inserted"
    # Retracted, flipped where the stage stands, then the offset, then inserted.
    assert [c[0] for c in commands] == ["retract", "absolute", "relative", "insert"]
    assert commands[2][1]["x"] == pytest.approx(OFFSET.x)
    assert commands[2][1]["y"] == pytest.approx(OFFSET.y)


def test_back_to_the_beams_is_the_offset_then_the_flip():
    microscope = _offset_arctis()
    start = _at(microscope, "SEM")
    microscope.move_to_device("FM")
    commands = pinned._record_commands(microscope)

    microscope.move_to_device("FIBSEM")

    _assert_xy(microscope.get_stage_position(), start.x, start.y)
    assert microscope.get_stage_orientation() == "SEM"
    assert [c[0] for c in commands] == ["retract", "relative", "absolute"]


def test_an_orientation_asked_for_is_reached_at_the_beams():
    microscope = _offset_arctis()
    start = _at(microscope, "SEM")
    microscope.move_to_device("FM")

    microscope.move_to_device("FIBSEM", orientation="MILLING")

    _assert_xy(microscope.get_stage_position(), start.x, start.y)
    assert microscope.get_stage_orientation() == "MILLING"


def test_the_beams_keep_the_pose_and_the_old_name_lands_at_sem():
    """`move_to_device` takes a device and an orientation, so with none it keeps the
    pose; `move_to_microscope` keeps the SEM landing its callers rely on."""
    microscope = _offset_arctis()
    _at(microscope, "FIB")

    microscope.move_to_device("FIBSEM")
    assert microscope.get_stage_orientation() == "FIB"

    microscope.move_to_microscope("FIBSEM")
    assert microscope.get_stage_orientation() == "SEM"


def test_the_old_name_takes_the_offset_too():
    microscope = _offset_arctis()
    start = _at(microscope, "SEM")

    microscope.move_to_microscope("FM")

    _assert_xy(microscope.get_stage_position(), start.x + OFFSET.x, start.y + OFFSET.y)
    assert microscope.get_stage_orientation() == "FM"


# -- at the beams' origin ----------------------------------------------------------------


def test_at_the_beams_origin_the_fm_is_the_flip_alone():
    microscope = pinned._microscope("arctis")
    start = _at(microscope, "SEM")
    commands = pinned._record_commands(microscope)

    microscope.move_to_device("FM")

    _assert_xy(microscope.get_stage_position(), start.x, start.y)
    assert microscope.get_stage_orientation() == "FM"
    assert microscope.fm.objective.state == "Inserted"
    assert [c[0] for c in commands] == ["retract", "absolute", "insert"]


def test_far_from_the_centre_the_fm_is_still_reached():
    """No window around the FM: a grid point near the edge of travel is reached by the
    flip, where an offset FM's range would have refused it."""
    microscope = pinned._microscope("arctis")
    position = pinned._start_position(microscope, "SEM")
    position.x = 1.5e-3
    microscope.move_stage_absolute(position)

    microscope.move_to_device("FM")

    assert microscope.get_stage_position().x == pytest.approx(1.5e-3, abs=1e-12)
    assert microscope.get_current_device() == "FM"


def test_back_to_the_beams_keeps_a_pose_they_image_from():
    """`move_to_device` keeps the pose with no orientation asked for; the old name
    lands at SEM, as its callers rely on (Patrick 2026-10-06)."""
    microscope = pinned._microscope("arctis")
    _at(microscope, "FIB")
    commands = pinned._record_commands(microscope)

    microscope.move_to_device("FIBSEM")

    assert microscope.get_stage_orientation() == "FIB"
    assert commands == []

    microscope.move_to_microscope("FIBSEM")
    assert microscope.get_stage_orientation() == "SEM"


# -- converting and asking --------------------------------------------------------------


def test_the_conversions_carry_the_offset_both_ways():
    microscope = _offset_arctis()
    start = pinned._start_position(microscope, "SEM")

    at_fm = microscope.get_target_position(start, "FM")
    _assert_xy(at_fm, start.x + OFFSET.x, start.y + OFFSET.y)
    _assert_xy(microscope.to_device(start, "FM"), at_fm.x, at_fm.y)
    _assert_xy(_to_fluorescence(microscope, start), at_fm.x, at_fm.y)

    back = _to_milling(microscope, at_fm)
    _assert_xy(back, start.x, start.y)
    assert microscope.get_stage_orientation(back) == "MILLING"


def test_the_pose_says_where_the_stage_is_and_no_window_does():
    microscope = _offset_arctis()
    far_out = pinned._start_position(microscope, "FM")
    far_out.x = 1.5e-3  # beyond the 1 mm a configured FM range defaults to

    assert microscope.is_at_device("FM", far_out)
    assert not microscope.is_at_device("FIBSEM", far_out)
    assert microscope.get_current_device(far_out) == "FM"

    at_sem = pinned._start_position(microscope, "SEM")
    assert not microscope.is_at_device("FM", at_sem)
    assert (
        microscope.get_device_imaging_state("FM", at_sem)
        is DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL
    )


@pytest.fixture(autouse=True)
def _quiet():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
