"""The FM decisions ask whether the stage reaches the FM by re-posing (FIB-1164).

They used to ask `stage_is_compustage`. The question each one means is whether the
stage declares an FM pose (FIB-1101): where it does, the objective is under the grid
and the beams and the FM are one place; where it does not, the FM is a place the
stage travels to. Today the two answers agree on every stage, so every answer the
FM tests pin is unchanged; these tests pull the two apart to show which one decides.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.stage import compustage_poses, rotating_stage_poses
from fibsem.structures import FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _declare(microscope, poses):
    """Make the stage declare *poses*, whatever type it reported."""
    microscope._stage_poses = lambda: poses(
        rotation_reference=microscope.system.stage.rotation_reference,
        shuttle_pre_tilt=microscope.system.stage.shuttle_pre_tilt,
        fib_column_tilt=microscope.system.ion.column_tilt,
    )


def _routes(microscope, monkeypatch):
    taken = []
    monkeypatch.setattr(
        microscope,
        "_move_to_device_compustage",
        lambda device, orientation=None: taken.append(device),
    )
    return taken


def test_the_fm_is_a_pose_on_a_compustage_and_a_place_on_an_offset_mount():
    assert _microscope(ARCTIS_CONFIG)._fm_is_a_pose()
    assert not _microscope(IFLM_CONFIG)._fm_is_a_pose()


def test_a_stage_that_declares_an_fm_pose_reaches_the_fm_by_re_posing(monkeypatch):
    """With its FM at the beams' origin. An FM origin of its own is an offset the
    stage travels by after the flip (`test_device_moves_one_path.py`)."""
    microscope = _microscope(IFLM_CONFIG)
    assert not microscope._fm_is_a_pose()
    _declare(microscope, compustage_poses)
    microscope.system.stage.devices["FM"].origin = FibsemStagePosition(x=0.0)
    taken = _routes(microscope, monkeypatch)

    microscope.move_to_device("FM")

    assert taken == ["FM"]
    translation = microscope._device_translation("FIBSEM", "FM")
    assert not any(getattr(translation, axis) for axis in ("x", "y", "z"))


def test_at_a_pose_there_is_no_parking_at_the_fm_to_refuse_a_rotation_at():
    microscope = _microscope(IFLM_CONFIG)
    microscope.get_current_device = lambda *a, **k: "FM"
    position = microscope.get_stage_position()
    half_turn = FibsemStagePosition(r=(position.r or 0.0) + 3.14159)

    # Parked at its FM, an offset mount refuses a half turn...
    with pytest.raises(ValueError, match="Cannot rotate the stage"):
        microscope._refuse_rotation_at_the_fluorescence_microscope(half_turn)

    # ...a stage whose FM is a pose has nowhere to be parked.
    _declare(microscope, compustage_poses)
    microscope._refuse_rotation_at_the_fluorescence_microscope(half_turn)


def test_a_compustage_that_declares_no_fm_pose_travels_to_it(monkeypatch):
    microscope = _microscope(ARCTIS_CONFIG)
    assert microscope._fm_is_a_pose()
    _declare(
        microscope,
        lambda **geometry: rotating_stage_poses(**geometry, rotates=False),
    )
    taken = _routes(microscope, monkeypatch)

    assert not microscope._fm_is_a_pose()
    with pytest.raises(ValueError, match="FM position on non-compustage"):
        microscope.get_target_position(
            microscope.get_stage_position(), target_orientation="FM"
        )
    # The travelling route, which then finds the FM's declared orientation is not a
    # pose this stage has.
    with pytest.raises(ValueError, match="Orientation FM not supported"):
        microscope.move_to_device("FM")
    assert taken == []
