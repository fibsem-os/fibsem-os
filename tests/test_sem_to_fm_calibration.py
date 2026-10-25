"""SEM to FM calibration on an offset FM: the record, the gate on travel to the
FM, and the calibration's steps, on the offset simulator.

Travel to the FM is refused until the calibration holds for the fitted holder;
the way back is never refused. The calibration's own first trip goes by the FM
entry's nominal origin, and what it measures is what the stage travels by after.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.fm.sem_to_fm import CalibrationRefused, SemToFmCalibration
from fibsem.structures.settings import BeamType, FibsemStagePosition
from fibsem.structures.system import (
    DeviceCalibration,
    InsertableDeviceState,
    SystemSettings,
)

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")
NOMINAL_X = 48.8e-3


def _uncalibrated(tmp_path, **sim):
    """The offset simulator with no SEM to FM calibration, from a file of its own."""
    settings = utils.load_yaml(IFLM_CONFIG)
    del settings["calibration"]["devices"]
    settings["sim"].update(sim)
    path = tmp_path / "configuration.yaml"
    utils.save_yaml(path, settings)
    microscope, _ = utils.setup_session(config_path=str(path))
    return microscope, str(path)


def _calibrated(fm_x: float, holder=None) -> DeviceCalibration:
    return DeviceCalibration(
        fibsem=FibsemStagePosition(x=0.0, y=0.0, z=0.0),
        fm=FibsemStagePosition(x=fm_x, y=0.0, z=0.0),
        holder=holder,
    )


# ── the gate ─────────────────────────────────────────────────────────────


def test_travel_to_the_fm_waits_for_the_calibration(tmp_path):
    microscope, _ = _uncalibrated(tmp_path)
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")
    before = microscope.get_stage_position()

    assert movement.fm_calibration_needed() is True
    with pytest.raises(ValueError, match="Calibrate SEM to FM first"):
        movement.move_to_device("FM")
    assert microscope.get_stage_position().x == pytest.approx(before.x)
    # the safe move to a position at the FM goes the same way
    at_the_fm = movement.target_position(before, target_device="FM")
    with pytest.raises(ValueError, match="Calibrate SEM to FM first"):
        movement.safe_absolute_stage_movement(at_the_fm)


def test_the_way_back_is_never_refused(tmp_path):
    microscope, _ = _uncalibrated(tmp_path)
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")
    with movement.calibrating():
        movement.move_to_device("FM")

    movement.move_to_device("FIBSEM", orientation="SEM")

    assert movement.current_device() == "FIBSEM"
    assert microscope.fm.objective.state is InsertableDeviceState.RETRACTED


def test_another_holder_asks_for_a_recheck(tmp_path):
    microscope, _ = _uncalibrated(tmp_path)
    stage = microscope.system.stage
    stage.device_calibration = _calibrated(NOMINAL_X, holder="the other holder")
    microscope.move_to_orientation("FIB")

    with pytest.raises(ValueError, match="Recheck SEM to FM first"):
        microscope.stage_movement.move_to_device("FM")
    stage.device_calibration.holder = stage.active_holder
    microscope.stage_movement.move_to_device("FM")
    assert microscope.stage_movement.current_device() == "FM"


def test_a_compustage_is_not_gated():
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    assert microscope.stage_movement.fm_calibration_needed() is False
    microscope.stage_movement.move_to_device("FM")


# ── the stage travels by what was measured ───────────────────────────────


def test_the_stage_travels_by_the_calibration(tmp_path):
    microscope, _ = _uncalibrated(tmp_path)
    microscope.system.stage.device_calibration = _calibrated(50.0e-3)
    microscope.move_to_orientation("FIB")
    start = microscope.get_stage_position()

    microscope.stage_movement.move_to_device("FM")

    assert microscope.get_stage_position().x == pytest.approx(start.x + 50.0e-3)
    assert microscope.stage_movement.current_device() == "FM"


def test_the_record_is_written_under_calibration_devices():
    settings = utils.load_yaml(IFLM_CONFIG)
    settings["calibration"]["devices"] = _calibrated(50.0e-3, "Slot holder").to_dict()

    system = SystemSettings.from_dict(settings)
    again = SystemSettings.from_dict(system.to_dict())

    record = again.stage.device_calibration
    assert record.fm.x == pytest.approx(50.0e-3)
    assert record.holder == "Slot holder"
    assert record.translation().x == pytest.approx(50.0e-3)
    # the FM entry keeps its nominal origin, for the calibration's first trip
    assert again.stage.devices["FM"].origin.x == pytest.approx(NOMINAL_X)


# ── the steps ────────────────────────────────────────────────────────────


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    """The steps on an uncalibrated offset simulator whose objective sits 40 µm
    further along x than its entry says. The coincidence label is left out:
    measuring it takes seconds a site, and it is never a gate."""
    microscope, path = _uncalibrated(tmp_path, fm_origin_error={"x": 40.0e-6})
    steps = SemToFmCalibration(microscope, configuration_path=path)
    monkeypatch.setattr(steps, "measure_coincidence", lambda name: {})
    return microscope, steps, path


def _site(microscope, steps, found_x: float):
    """One site: capture at the beams, travel, centre what the FM shows (here
    *found_x* along x), capture, and come back."""
    steps.capture_beams()
    steps.travel_to_fm()
    microscope.stage_movement.move_relative(FibsemStagePosition(x=found_x))
    steps.capture_fm()
    steps.return_to_beams()
    steps.record_residual(0.0, 0.0)


def test_the_rotation_centre_is_the_midpoint_of_the_two_centrings(calibration):
    microscope, steps, _ = calibration
    microscope.move_to_orientation("SEM")
    steps.capture_rotation_reference()
    steps.turn("FIB")

    assert steps.measure_rotation_centre(0.0, 0.0) == pytest.approx((0.0, 0.0))
    assert steps.rotation_centre_correction() == pytest.approx((0.0, 0.0))
    # a mark found off centre after the turn moves the centre half as far
    here = microscope.get_stage_position()
    moved = microscope.stage_movement.project_stable_move(
        10e-6, 0.0, BeamType.ELECTRON, here
    )
    centre = steps.measure_rotation_centre(10e-6, 0.0)
    assert centre[0] == pytest.approx((moved.x - here.x) / 2, abs=1e-12)


def test_three_sites_measure_the_travel_and_lift_the_gate(calibration):
    microscope, steps, path = calibration
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")

    for dy in (0.0, 1.6e-3, 1.6e-3):
        movement.move_relative(FibsemStagePosition(y=dy))
        _site(microscope, steps, found_x=40.0e-6)
        assert movement.current_device() == "FIBSEM"

    summary = steps.summary()
    assert summary["problems"] == []
    assert summary["travel"]["x"] == pytest.approx(NOMINAL_X + 40.0e-6)
    saved = steps.save()

    assert movement.fm_calibration_needed() is False
    assert saved.translation().x == pytest.approx(NOMINAL_X + 40.0e-6)
    on_disk = utils.load_yaml(path)["calibration"]["devices"]
    assert on_disk["fm"]["origin"]["x"] == pytest.approx(NOMINAL_X + 40.0e-6)
    assert len(on_disk["sites"]) == 3


def test_sites_that_disagree_are_not_offered_for_saving(calibration):
    microscope, steps, _ = calibration
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")
    for dy, found_x in ((0.0, 40e-6), (1.6e-3, 40e-6), (1.6e-3, 100e-6)):
        movement.move_relative(FibsemStagePosition(y=dy))
        _site(microscope, steps, found_x=found_x)

    summary = steps.summary()
    assert summary["can_save"] is False
    assert "rotation centre is likely wrong" in " ".join(summary["problems"])
    with pytest.raises(CalibrationRefused):
        steps.save()


def test_sites_close_together_are_not_enough(calibration):
    microscope, steps, _ = calibration
    microscope.move_to_orientation("FIB")
    for _ in range(3):
        _site(microscope, steps, found_x=40e-6)

    assert "must be at least 3 mm off" in " ".join(steps.summary()["problems"])


def test_a_capture_away_from_the_fm_pose_is_refused(calibration):
    microscope, steps, _ = calibration
    microscope.move_to_orientation("SEM")
    with pytest.raises(CalibrationRefused, match="this step is taken at FIB"):
        steps.capture_beams()


# ── the simulator's FM sees the sample ───────────────────────────────────


def test_the_offset_objective_sees_the_sample_at_the_beams(tmp_path):
    microscope, _ = _uncalibrated(tmp_path, fm_origin_error={"x": 40.0e-6})
    microscope.move_to_orientation("FIB")
    with microscope.stage_movement.calibrating():
        microscope.stage_movement.move_to_device("FM")
    stage = microscope.get_stage_position()

    seen = microscope._connection.scene_stage_position_at_fm()

    assert seen.x == pytest.approx(stage.x - NOMINAL_X - 40.0e-6)
    assert seen.y == pytest.approx(stage.y)
