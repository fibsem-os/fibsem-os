"""The stage movement service: built where there is a stage, and what its own
commands add over the microscope methods that route to it.

What the moves command is pinned in ``test_stage_movement_pins.py``, through the old
methods; this covers the service called directly.
"""

import os

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
    microscope.record_signal.connect(
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


def test_the_sample_stage_moves_through_the_service(microscope):
    movement = microscope.stage_movement
    calls = []
    for name in (
        "stable_move",
        "vertical_move",
        "project_stable_move",
        "move_to_orientation",
        "move_to_milling_angle",
    ):
        original = getattr(movement, name)
        setattr(
            movement,
            name,
            lambda *a, _n=name, _o=original, **k: (calls.append(_n), _o(*a, **k))[1],
        )
    stage = microscope._stage
    base = microscope.get_stage_position()

    stage.stable_move(10e-6, 0.0, BeamType.ELECTRON)
    stage.vertical_move(5e-6)
    stage.project_stable_move(1e-6, 0.0, BeamType.ION, base)
    stage.move_to_orientation("SEM")
    stage.move_to_milling_angle(0.3)

    assert calls[0] == "stable_move"
    assert set(calls) == {
        "stable_move",
        "vertical_move",
        "project_stable_move",
        "move_to_orientation",
        "move_to_milling_angle",
    }


def test_without_the_service_the_sample_stage_uses_the_microscope(microscope):
    microscope.stage_movement = None
    start = microscope.get_stage_position()
    end = microscope._stage.stable_move(10e-6, 0.0, BeamType.ELECTRON)
    assert end != start
