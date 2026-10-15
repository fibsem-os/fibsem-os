"""The stage movement service: built where there is a stage, and what its own
commands add over the microscope methods that route to it.

What the moves command is pinned in ``test_stage_movement_pins.py``, through the old
methods; this covers the service called directly.
"""

import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.devices.stage import StageLimitError
from fibsem.services.stage_movement import StageMovement
from fibsem.structures import BeamType, FibsemStagePosition


@pytest.fixture
def microscope():
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(
        manufacturer="Demo", ip_address="localhost", setup_logging=False
    )
    return microscope


def test_the_demo_builds_one_over_its_stage_and_beams(microscope):
    movement = microscope.stage_movement
    assert isinstance(movement, StageMovement)
    assert movement.stage is microscope.stage
    assert movement.electron is microscope.beams[BeamType.ELECTRON]
    assert movement.ion is microscope.beams[BeamType.ION]
    assert movement not in microscope.devices.values()


def test_without_a_stage_there_is_none_and_the_old_moves_stay(microscope):
    from fibsem.services.stage_movement import bind_stage_movement

    microscope.stage = None
    assert bind_stage_movement(StageMovement, microscope) is None


def test_its_views_are_the_backends(microscope):
    assert microscope.stage_movement.vertical_move_views == tuple(
        microscope.vertical_move_views
    )
    microscope.vertical_move_views = (BeamType.ION,)
    assert not microscope.stage_movement.supports_vertical_move(BeamType.ELECTRON)


def test_its_plain_moves_check_the_stage_limits(microscope):
    movement = microscope.stage_movement
    start = microscope.get_stage_position()

    end = movement.move_relative(FibsemStagePosition(x=10e-6))
    assert end.x == pytest.approx(start.x + 10e-6)

    end = movement.move_absolute(FibsemStagePosition(x=0.0, y=0.0))
    assert (end.x, end.y) == (0.0, 0.0)

    with pytest.raises(StageLimitError):
        movement.move_absolute(FibsemStagePosition(x=10.0))


def test_a_move_called_on_it_is_recorded_once_under_its_name(microscope):
    moves = []
    microscope.session.record_signal.connect(
        lambda kind, payload: moves.append(payload) if kind == "stage_moved" else None
    )

    microscope.stage_movement.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ION)
    microscope.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ION)

    assert [m["move"] for m in moves] == ["stable_move", "stable_move"]
    assert moves[0]["request"] == moves[1]["request"]


@pytest.fixture
def iflm():
    from fibsem import config as cfg

    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml"),
        setup_logging=False,
    )
    return microscope


def test_an_fm_microscope_fills_the_fm_role(iflm, microscope):
    assert iflm.stage_movement._fm_device() is iflm.devices["fm"]
    assert iflm.stage_movement.fm is iflm.devices["fm"]
    assert microscope.stage_movement._fm_device() is None
    with pytest.raises(ValueError):
        microscope.stage_movement.fm_stable_move(dx=1e-6, dy=0.0)


def test_the_camera_tilt_is_derived_unless_configured(iflm):
    derived = iflm.stage_movement.camera_tilt
    assert derived == iflm.system.ion.column_tilt
    assert iflm.fm.camera_tilt == derived

    iflm.system.fm.camera_tilt = 45.0
    assert iflm.stage_movement.camera_tilt == 45.0
    assert iflm.fm.camera_tilt == 45.0


def test_the_camera_tilt_survives_a_config_round_trip():
    from fibsem.structures import FluorescenceSystemSettings

    settings = FluorescenceSystemSettings.from_dict({"camera_tilt": 30.0})
    assert settings.camera_tilt == 30.0
    assert FluorescenceSystemSettings.from_dict(settings.to_dict()).camera_tilt == 30.0
    assert FluorescenceSystemSettings.from_dict({}).camera_tilt is None


def test_the_display_transform_is_held_on_the_camera(iflm):
    from fibsem.fm.structures import CameraImageTransform

    camera = iflm.devices["fm"]._roles["camera"]
    iflm.fm.set_image_transform(CameraImageTransform.FLIP_X)
    assert iflm.fm._transform == CameraImageTransform.FLIP_X
    assert camera.parameters["display_transform"].value == "flip-x"

    base = iflm.get_stage_position()
    flipped = iflm.stage_movement.project_fm_stable_move(1e-6, 0.0, base)
    iflm.fm.set_image_transform(CameraImageTransform.NONE)
    plain = iflm.stage_movement.project_fm_stable_move(1e-6, 0.0, base)
    assert flipped.x - base.x == pytest.approx(-(plain.x - base.x))


def test_without_the_service_the_moves_say_there_is_no_stage(microscope):
    microscope.stage_movement = None
    with pytest.raises(ValueError, match="no stage"):
        microscope.stable_move(10e-6, 0.0, BeamType.ELECTRON)
    with pytest.raises(ValueError, match="no stage"):
        microscope.move_to_orientation("SEM")
    assert not microscope.supports_vertical_move(BeamType.ION)


# -- the milling angle at any rotation (FIB-1392), and the geometry read ---------------


def _formula(t_deg, pre_tilt=35.0, column_tilt=52.0, side=1):
    # The tilt-plane formula: the pre-tilt leans the other way half a turn round.
    return 90.0 - column_tilt + t_deg - side * pre_tilt


@pytest.mark.parametrize("t_deg", [-30.0, 0.0, 12.0, 35.0, 60.0])
def test_the_milling_angle_turns_with_the_shuttle(microscope, t_deg):
    movement = microscope.stage_movement
    at = lambda r_deg: movement.current_milling_angle(  # noqa: E731
        FibsemStagePosition(r=np.radians(r_deg), t=np.radians(t_deg))
    )
    assert at(0) == pytest.approx(_formula(t_deg))
    assert at(180) == pytest.approx(_formula(t_deg, side=-1))


def test_the_fib_pose_is_normal_to_the_beam_and_near_it_is_not(microscope):
    movement = microscope.stage_movement
    fib = movement.orientation("FIB")
    assert movement.current_milling_angle(fib) == pytest.approx(90.0)
    # Still classified FIB, but tilted 3 degrees off: the old answer was a flat 90.
    off = FibsemStagePosition(r=fib.r, t=fib.t + np.radians(3))
    assert movement.stage_orientation(off) == "FIB"
    assert movement.current_milling_angle(off) == pytest.approx(93.0)


@pytest.mark.parametrize("rotation", [0.0, 90.0, 180.0])
def test_the_milling_angle_move_reaches_it_at_any_rotation(microscope, rotation):
    movement = microscope.stage_movement
    assert movement.move_to_milling_angle(15.0, rotation=rotation) is True
    assert movement.current_milling_angle() == pytest.approx(15.0, abs=0.1)
    position = microscope.get_stage_position()
    assert np.degrees(position.r) % 360 == pytest.approx(rotation)


def test_the_geometry_read(microscope):
    movement = microscope.stage_movement
    geometry = movement.geometry()
    assert geometry["column_tilt_deg"] == {"electron": 0.0, "ion": 52.0}
    assert geometry["shuttle_pre_tilt_deg"] == 35.0
    assert set(geometry["orientations"]) >= {"SEM", "FIB", "MILLING"}
    assert geometry["orientation"] == movement.stage_orientation()
    assert geometry["milling_angle"]["current_deg"] == pytest.approx(
        movement.current_milling_angle()
    )
    # Raw +z at the stage's tilt, in the chamber (z up).
    t = microscope.get_stage_position().t
    assert geometry["stage_plus_z_in_chamber"] == pytest.approx(
        [0.0, -np.sin(t), np.cos(t)]
    )
    # Each view's answer is what stable_move would do: its projection, not config text.
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        view = geometry["views"][beam_type.name]
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
        moved = movement.project_stable_move(0.0, 1e-6, beam_type, base)
        assert view["stable_move_plus_dy_1um"]["y"] == pytest.approx(moved.y)
        assert view["stable_move_plus_dy_1um"]["z"] == pytest.approx(moved.z)
        assert view["stage_plus_x_in_image"] == "+x"
        assert view["scan_rotation_handled"] is True
    assert set(geometry["devices"]) == set(microscope.system.stage.devices)


def test_the_geometry_says_a_half_turn_of_scan_rotation_flips_x(microscope):
    microscope.beams[BeamType.ION].scan_rotation.set_value(np.pi)
    view = microscope.stage_movement.geometry()["views"]["ION"]
    assert view["stage_plus_x_in_image"] == "-x"
    assert view["scan_rotation_handled"] is True
    microscope.beams[BeamType.ION].scan_rotation.set_value(np.radians(30))
    view = microscope.stage_movement.geometry()["views"]["ION"]
    assert view["scan_rotation_handled"] is False
