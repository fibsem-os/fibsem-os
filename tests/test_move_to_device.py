"""`move_to_device(device, orientation=None)`: one call that plans the route.

The old `move_to_microscope` refused what it could have done -- "Cannot move to FM
from SEM or MILLING orientation. Please switch to FIB orientation first." was an
instruction to the user to perform, by hand, exactly the sequence the software
composes everywhere else. The replacement owns the safe order: retract the
objective, re-pose at the beams, travel out. The rotation guard (FIB-841) stays
underneath as the last-line assert -- several tests here would trip it if the legs
were ever composed in the wrong order.

See FIB-832.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


# ── the refusal is now a route ───────────────────────────────────────


def test_to_the_fm_from_sem_re_poses_then_travels():
    """The first thing a real workflow does, and the thing that used to raise."""
    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    microscope.move_to_device("FM")

    assert microscope.get_current_device() == "FM"
    assert microscope.get_stage_orientation() == "FIB"
    assert microscope.fm.objective.state == "Inserted"


def test_the_re_pose_happens_at_the_beams_not_at_the_fm():
    """The bracketing order, proven by the guard underneath.

    Coming back from the FM and asking for MILLING requires a half turn. Rotating
    where the stage stands -- at the FM -- is exactly what FIB-841 refuses, so a
    wrong leg order cannot pass this test quietly: the guard would raise.
    """
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_to_device("FM")

    microscope.move_to_device("FIBSEM", orientation="MILLING")

    assert microscope.get_current_device() == "FIBSEM"
    assert microscope.get_stage_orientation() == "MILLING"


def test_the_round_trip_is_one_call_each_way():
    """The FIB-832 complaint: the last leg used to be two moves, every time."""
    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    microscope.move_to_device("FM")
    microscope.move_to_device("FIBSEM", orientation="MILLING")
    microscope.move_to_device("FM")

    assert microscope.get_current_device() == "FM"
    assert microscope.get_stage_orientation() == "FIB"


# ── what a traverse still does not do ────────────────────────────────


def test_an_acceptable_pose_is_carried_across_untouched():
    """No orientation asked for and the pose is one the FM images from: no snap.

    Three degrees off the nominal FIB tilt -- inside the band the classifier calls
    FIB -- must survive the traverse. Routing through `move_to_orientation`
    unconditionally would snap r and t to nominal and quietly discard it.
    """
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    position = microscope.get_stage_position()
    position.t = position.t + np.radians(3)
    microscope.move_stage_absolute(position)
    tilt_before = microscope.get_stage_position().t

    microscope.move_to_device("FM")

    assert microscope.get_stage_position().t == pytest.approx(tilt_before)


def test_asking_for_the_device_it_is_at_does_not_move():
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_to_device("FM")
    before = microscope.get_stage_position()

    microscope.move_to_device("FM")

    assert microscope.get_stage_position().is_close(before, tol=1e-9)
    assert microscope.fm.objective.state == "Inserted"


def test_travelling_from_neither_device_is_still_refused():
    """Mid-traverse is a real state, and not one to guess a starting device for."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_stage_relative(FibsemStagePosition(x=24.0e-3, y=0.0, z=0.0))

    with pytest.raises(ValueError, match="not at any configured device"):
        microscope.move_to_device("FM")


# ── the compustage gets the same signature ───────────────────────────


def test_a_compustage_lands_at_the_orientation_it_asked_for():
    """The extra stage move every round trip used to pay: FIBSEM always landed at
    SEM, and MILLING was a second call."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_device("FM")

    microscope.move_to_device("FIBSEM", orientation="MILLING")

    assert microscope.get_stage_orientation() == "MILLING"


def test_a_compustage_keeps_its_default_landing_poses():
    """Unasked, FIBSEM still lands at SEM and the FM at its own orientation --
    every existing caller relies on exactly that."""
    microscope = _microscope(ARCTIS_CONFIG)

    microscope.move_to_device("FM")
    assert microscope.get_stage_orientation() == "FM"
    assert microscope.fm.objective.state == "Inserted"

    microscope.move_to_device("FIBSEM")
    assert microscope.get_stage_orientation() == "SEM"


# ── the deprecated names still work ──────────────────────────────────


def test_move_to_microscope_is_a_shim():
    """~15 production call sites and ~40 in tests keep working unchanged."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")

    microscope.move_to_microscope("FM")
    assert microscope.get_current_device() == "FM"

    microscope.move_to_microscope("FIBSEM")
    assert microscope.get_current_device() == "FIBSEM"


# ── the FM orientation no longer exists where it never was one ───────


def test_an_offset_mount_has_no_fm_orientation_to_ask_for():
    """`orientations["FM"]` off a compustage was a deepcopy of FIB -- a second name
    for a pose that already had one, never returned by the classifier, and the root
    of the whole conflation. Deleting it changes no classification; it only stops
    `get_orientation("FM")` naming a pose that does not exist.
    """
    offset = _microscope()
    compustage = _microscope(ARCTIS_CONFIG)

    with pytest.raises(Exception):
        offset.get_orientation("FM")

    assert compustage.get_orientation("FM") is not None
