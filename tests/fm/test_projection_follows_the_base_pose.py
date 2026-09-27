"""`project_fm_stable_move` projects from the pose of the position it is given.

It used to read the live stage tilt instead. The two are the same when a tileset is
centred on where the stage stands, which is why nothing noticed -- but a grid planned
around a centre in one pose while the stage stands in another was laid out for the
wrong one: mirrored in y on a compustage, where the sign of y follows the tilt, and
scaled by the tilt everywhere.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.projection import FMStageProjection
from fibsem.structures import FibsemStagePosition

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
POSES = ["FM", "SEM", "MILLING"]
DX, DY = 100e-6, 60e-6


@pytest.fixture(scope="module")
def arctis():
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    return microscope


def _base(microscope, orientation):
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)


def _stand_at(microscope, orientation):
    microscope.fm.objective.retract()
    microscope.move_to_orientation(orientation)


@pytest.mark.parametrize("live", POSES)
@pytest.mark.parametrize("base", POSES)
def test_the_projection_agrees_with_the_canvas_wherever_the_stage_stands(
    arctis, live, base
):
    """`FMStageProjection.to_plane` is what the canvas and the planner read a position
    back with, and it takes its pose from the base. The forward projection has to take
    it from the same place, or a planned tile and the ground it was planned over part
    company. The plane is y-up and an image displacement is y-down, hence the sign."""
    _stand_at(arctis, live)
    projection = FMStageProjection.from_microscope(arctis)
    origin = _base(arctis, base)

    landed = arctis.project_fm_stable_move(dx=DX, dy=DY, base_position=origin)

    assert projection.to_plane(landed, origin) == pytest.approx((DX, -DY), abs=1e-9)


@pytest.mark.parametrize("base", POSES)
def test_where_the_stage_stands_does_not_change_the_answer(arctis, base):
    origin = _base(arctis, base)
    landed = []
    for live in POSES:
        _stand_at(arctis, live)
        landed.append(arctis.project_fm_stable_move(dx=DX, dy=DY, base_position=origin))

    for other in landed[1:]:
        assert other.x == pytest.approx(landed[0].x, abs=1e-12)
        assert other.y == pytest.approx(landed[0].y, abs=1e-12)
        assert other.z == pytest.approx(landed[0].z, abs=1e-12)


def test_the_sign_of_y_follows_the_base_pose_on_a_compustage(arctis):
    """The camera looks up from underneath. Flipped over it (t = -180) the sample's y
    runs one way; from the beam side (t = 0) it runs the other."""
    _stand_at(arctis, "SEM")

    flipped = arctis.project_fm_stable_move(DX, DY, _base(arctis, "FM"))
    beam_side = arctis.project_fm_stable_move(DX, DY, _base(arctis, "SEM"))

    assert np.sign(flipped.y) == -np.sign(beam_side.y)
    assert flipped.x == pytest.approx(beam_side.x)


def test_a_move_made_now_still_uses_where_the_stage_is(arctis):
    """`fm_stable_move` has no base to be given; the live pose is the right one."""
    for live in POSES:
        _stand_at(arctis, live)
        here = arctis.get_stage_position()
        expected = arctis.project_fm_stable_move(DX, DY, here)
        delta = arctis._fm_stage_delta(DX, DY)
        assert here.y + delta.y == pytest.approx(expected.y, abs=1e-12)


def test_a_base_that_does_not_say_its_pose_falls_back_to_the_stage(arctis):
    _stand_at(arctis, "SEM")
    bare = FibsemStagePosition(x=0.0, y=0.0, z=0.0)
    here = arctis.get_stage_position()

    landed = arctis.project_fm_stable_move(DX, DY, bare)

    assert landed.y == pytest.approx(
        arctis.project_fm_stable_move(DX, DY, here).y - here.y, abs=1e-12
    )


def test_an_offset_mount_splits_y_by_the_base_tilt():
    """No sign change there, but the tilt still scales y and feeds z."""
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    projection = FMStageProjection.from_microscope(microscope)
    fib = microscope.get_orientation("FIB")
    origin = FibsemStagePosition(x=48.8e-3, y=0.0, z=0.0, r=fib.r, t=fib.t)
    microscope.move_to_orientation("SEM")  # standing somewhere else entirely

    landed = microscope.project_fm_stable_move(dx=DX, dy=DY, base_position=origin)

    assert projection.to_plane(landed, origin) == pytest.approx((DX, -DY), abs=1e-9)
