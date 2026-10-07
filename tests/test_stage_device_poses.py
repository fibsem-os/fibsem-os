"""The stage device declares its poses (FIB-1101).

`tests/test_stage_poses_pinned.py` holds the values on every shipped stage type, through
the microscope. This file holds the pieces: the two declarations, which driver declares
which, and that a backend without a stage device gets the same table as one with it.
"""

import math
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.drivers.autoscript import AutoscriptCompustage, AutoscriptStage
from fibsem.devices.stage import Axes, compustage_poses, rotating_stage_poses


def _deg(poses):
    return {
        name: (round(math.degrees(pose.r), 9), round(math.degrees(pose.t), 9))
        for name, pose in poses.items()
    }


def test_a_rotating_stage_faces_the_ion_beam_half_a_turn_round():
    poses = rotating_stage_poses(0.0, 35.0, 52.0)
    assert _deg(poses) == {"SEM": (0.0, 35.0), "FIB": (180.0, 17.0)}


def test_the_half_turn_wraps():
    """Tescan's reference is 180, so FIB is at 0 and not 360."""
    poses = rotating_stage_poses(180.0, 0.0, 55.0)
    assert _deg(poses) == {"SEM": (180.0, 0.0), "FIB": (0.0, 55.0)}


def test_a_stage_without_a_rotation_axis_stays_at_the_reference():
    poses = rotating_stage_poses(0.0, 35.0, 52.0, rotates=False)
    assert _deg(poses)["FIB"] == (0.0, 17.0)


def test_a_compustage_tilts_over_and_has_an_fm_pose():
    poses = compustage_poses(0.0, 0.0, 52.0)
    assert _deg(poses) == {
        "SEM": (0.0, 0.0),
        "FIB": (0.0, -128.0),
        "FM": (0.0, -180.0),
    }


def _unconnected(cls, axes):
    # The poses read only the axes, so no AutoScript connection is needed.
    stage = object.__new__(cls)
    stage.axes = Axes({name: None for name in axes})
    return stage


@pytest.mark.parametrize(
    "cls, axes, expected",
    [
        (AutoscriptStage, "xyzrt", rotating_stage_poses(0.0, 35.0, 52.0)),
        (AutoscriptCompustage, "xyzt", compustage_poses(0.0, 35.0, 52.0)),
    ],
)
def test_each_autoscript_stage_declares_its_own(cls, axes, expected):
    stage = _unconnected(cls, axes)
    assert _deg(stage.poses(0.0, 35.0, 52.0)) == _deg(expected)


def test_without_a_stage_device_there_are_no_orientations():
    """Every backend builds a stage device; one switched off has no stage to orient."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_device = None
    microscope._update_orientations()
    assert microscope.orientations == {}
    assert microscope._stage_poses() == {}
    assert microscope._stage_turned_over(3.0) is False


def _classifier(poses_deg):
    """A Demo microscope classifying against hand-written poses, in degrees."""
    from fibsem.structures import FibsemStagePosition

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.orientations = {
        name: FibsemStagePosition(r=math.radians(r), t=math.radians(t))
        for name, (r, t) in poses_deg.items()
    }
    return microscope


def _at(r, t):
    from fibsem.structures import FibsemStagePosition

    return FibsemStagePosition(r=math.radians(r), t=math.radians(t))


def test_classification_picks_the_nearest_declared_pose():
    """Two poses whose tolerances overlap: the one the stage is nearer wins,
    whichever is listed first."""
    microscope = _classifier(
        {"SEM": (0, 0), "FIB": (0, 8), "MILLING": (0, -20), "FM": (0, 4)}
    )
    assert microscope.get_stage_orientation(_at(0, 1)) == "SEM"
    assert microscope.get_stage_orientation(_at(0, 7)) == "FIB"
    assert microscope.get_stage_orientation(_at(0, 4.5)) == "FM"


def test_a_declared_pose_inside_the_milling_range_is_not_milling():
    """A stage that reaches FIB by tilting at the SEM rotation (FIB-1101's JEOL
    case): MILLING is a range, so it never shadows a pose the stage declares."""
    microscope = _classifier({"SEM": (0, 0), "FIB": (0, 30), "MILLING": (0, 15)})
    assert microscope.get_stage_orientation(_at(0, 30)) == "FIB"
    assert microscope.get_stage_orientation(_at(0, 15)) == "MILLING"
    assert microscope.get_stage_orientation(_at(0, -50)) == "NONE"
    assert microscope.get_stage_orientation(_at(90, 30)) == "NONE"
