"""Device moves on one path: a device is its pose plus its origin.

A compustage reaches its FM by turning the grid over to face the objective. When the
objective's centre is offset from the beams' coincidence point, that offset is the FM's
origin, in stage coordinates at the FM pose, and the stage travels by it after the flip
just as an offset mount travels to its FM. Which device the stage is at stays decided by
the pose (`Stage.device_at_pose`), and the FM has no window: the stage limits bound it.

A compustage whose FM shares the beams' origin keeps its own route until the one path
has been checked on an instrument (`_keeps_the_compustage_route`). The last test runs
the one path for it anyway and compares with the pinned moves.
"""

import json
import logging

import pytest

from fibsem.applications.autolamella.poses import _to_fluorescence, _to_milling
from fibsem.microscope import FibsemMicroscope
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


def test_the_old_compustage_name_takes_the_offset_too():
    microscope = _offset_arctis()
    start = _at(microscope, "SEM")

    microscope.move_to_microscope_compustage("FM")

    _assert_xy(microscope.get_stage_position(), start.x + OFFSET.x, start.y + OFFSET.y)
    assert microscope.get_stage_orientation() == "FM"


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


# -- the one path at the beams' origin --------------------------------------------------


def _one_path_cases():
    for case_id in pinned.CASE_IDS:
        mount, start, call = case_id.split("|")
        if mount.startswith("arctis") and call != "queries":
            yield case_id


# `move_to_device("FIBSEM")` with no orientation keeps a pose the beams image from,
# where the compustage route always went to SEM (Patrick 2026-10-06).
KEEPS_THE_POSE = {
    f"{mount}|{start}|move_to_device(FIBSEM,None)"
    for mount in ("arctis", "arctis_beam_side")
    for start in ("SEM", "FIB", "MILLING", "SEM+1", "FIB+1", "MILLING+1")
}


@pytest.mark.parametrize("case_id", list(_one_path_cases()))
def test_the_one_path_arrives_where_the_compustage_route_does(monkeypatch, case_id):
    """Same position, pose, device and objective as the pinned compustage route.

    The commands differ in two ways, both checked here: the flip names x, y and z at
    their current values where the route named only r and t, and a move to the beams
    from a pose they image from commands nothing.
    """
    monkeypatch.setattr(
        FibsemMicroscope, "_keeps_the_compustage_route", lambda self, target: False
    )
    expected = pinned._load_pins()[case_id]
    actual = json.loads(json.dumps(pinned.run_case(case_id)))

    if case_id in KEEPS_THE_POSE:
        assert actual["commands"] == []
        assert actual["orientation"] == case_id.split("|")[1].split("+")[0]
        return

    for key in ("raises", "orientation", "device", "objective"):
        assert actual.get(key) == expected.get(key), key
    pinned._assert_same(actual["position"], expected["position"], "position")

    assert [c[0] for c in actual["commands"]] == [c[0] for c in expected["commands"]]
    for got, was in zip(actual["commands"], expected["commands"]):
        if got[0] != "absolute":
            assert got == was
            continue
        for axis, value in was[1].items():
            if value is not None:
                assert got[1][axis] == pytest.approx(value, abs=1e-12), axis


@pytest.fixture(autouse=True)
def _quiet():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
