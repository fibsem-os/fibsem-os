"""An offset FM's declared pose (FIB-1556): the FM entry's ``pose`` is the "FM"
orientation, the travel turns to it at the beams, the FM images in it, and the
camera's tilt follows from it. Without one nothing changes: the FM images in the FIB
pose. On the iFLM simulator."""

import copy
import os

import numpy as np
import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.codec.upgrades.configuration import upgrade
from fibsem.services.stage_movement import fm_camera_tilt
from fibsem.structures.system import DeviceImagingState, MicroscopeSettings

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


def _config(**fm_keys) -> dict:
    with open(IFLM_CONFIG) as f:
        data = yaml.safe_load(f)
    entry = next(d for d in data["hardware"]["devices"] if d["name"] == "fm")
    entry.pop("available_orientations")
    entry.update(fm_keys)
    return data


def _session(tmp_path, **fm_keys):
    path = tmp_path / "iflm-fm-pose.yaml"
    path.write_text(yaml.safe_dump(_config(**fm_keys)))
    microscope, _ = utils.setup_session(config_path=str(path))
    microscope.move_to_orientation("SEM")
    return microscope


def _degrees(position):
    return round(float(np.degrees(position.r)), 6), round(
        float(np.degrees(position.t)), 6
    )


def test_a_declared_pose_is_the_fm_orientation_and_the_fms_default():
    data = upgrade(_config(pose={"t": 10}))
    fm = MicroscopeSettings.from_dict(data).system.stage.devices["FM"]
    assert fm.available_orientations == ["FM"]
    assert np.degrees(fm.pose.t) == pytest.approx(10) and fm.pose.r is None
    assert fm.to_dict()["pose"] == {"t": pytest.approx(10)}


def test_a_pose_with_only_the_fib_orientation_listed_is_refused():
    data = upgrade(_config(pose={"t": 10}, available_orientations=["FIB"]))
    with pytest.raises(ValueError, match="pose is the FM orientation"):
        MicroscopeSettings.from_dict(data)


def test_a_pose_and_a_camera_tilt_together_are_refused():
    data = upgrade(_config(pose={"t": 10}, camera_tilt=45))
    with pytest.raises(ValueError, match="leave camera_tilt out"):
        MicroscopeSettings.from_dict(data)


def test_without_a_pose_nothing_changes(tmp_path):
    microscope = _session(tmp_path, available_orientations=["FIB"])
    assert "FM" not in microscope.stage_movement.orientations
    assert fm_camera_tilt(microscope) == pytest.approx(52.0)
    microscope.stage_movement.move_to_device("FM")
    assert microscope.get_stage_orientation() == "FIB"
    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.READY


def test_the_fm_is_travelled_to_in_its_pose_and_left_in_the_fib_pose(tmp_path):
    microscope = _session(tmp_path, pose={"t": 10})
    movement = microscope.stage_movement
    assert _degrees(movement.orientation("FM")) == (180.0, 10.0)
    # The camera sees the sample face on there: 10 degrees on a 35 degree shuttle.
    assert fm_camera_tilt(microscope) == pytest.approx(45.0)

    turn = [s for s in movement.plan_move_to_device("FM") if s["step"] == "turn"]
    assert [s["orientation"] for s in turn] == ["FM"]
    movement.move_to_device("FM")
    assert _degrees(microscope.get_stage_position()) == (180.0, 10.0)
    assert microscope.get_stage_orientation() == "FM"
    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.READY

    # Back at the beams by a tilt to FIB, the turn after the travel.
    steps = [s["step"] for s in movement.plan_move_to_device("FIBSEM")]
    assert steps == ["retract", "travel", "turn"]
    movement.move_to_device("FIBSEM")
    assert _degrees(microscope.get_stage_position()) == (180.0, 17.0)
    assert microscope.get_stage_orientation() == "FIB"


def test_the_fib_pose_at_the_fm_needs_a_repose_when_the_fm_has_its_own(tmp_path):
    microscope = _session(tmp_path, pose={"t": 10})
    microscope.move_to_orientation("FIB")
    position = copy.deepcopy(microscope.get_stage_position())
    position.x += 48.8e-3
    movement = microscope.stage_movement
    assert movement.stage_orientation(position) == "FIB"
    assert movement.device_imaging_state("FM", position) is (
        DeviceImagingState.NEEDS_REPOSE
    )


def test_a_pose_stated_as_the_fib_pose_is_fm_at_the_fm_and_fib_at_the_beams(
    tmp_path,
):
    microscope = _session(tmp_path, pose={"t": 17})
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")
    assert microscope.get_stage_orientation() == "FIB"
    assert microscope.get_device_imaging_state("FIBSEM") is DeviceImagingState.READY
    movement.move_to_device("FM")
    assert microscope.get_stage_orientation() == "FM"
    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.READY
    assert fm_camera_tilt(microscope) == pytest.approx(52.0)
