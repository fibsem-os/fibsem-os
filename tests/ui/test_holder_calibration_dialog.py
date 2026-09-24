"""The guided slot calibration: orientation gate, capture, review, save."""

import math
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes._stage import SampleHolder, _create_sample_stage
from fibsem.structures import FibsemStagePosition
from fibsem.ui.widgets.holder_calibration_dialog import (
    CALIBRATION_ORIENTATION,
    HolderCalibrationDialog,
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    return microscope


def _holder(capacity: int = 2) -> SampleHolder:
    # 35 degrees, as the Demo configuration says: saved and selected, the holder's
    # pre-tilt becomes the stage's, and a slot captured at another is not trusted.
    holder = SampleHolder(pre_tilt=35.0, name="Test shuttle", capacity=capacity)
    holder._ensure_slots()
    return holder


def _configuration(tmp_path) -> Path:
    """A copy of a shipped configuration for the wizard to save into."""
    path = tmp_path / "configuration.yaml"
    if not path.exists():
        shutil.copyfile(
            os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"), path
        )
    return path


def _dialog(qapp, microscope, holder, tmp_path):
    return HolderCalibrationDialog(
        microscope, holder, configuration_path=str(_configuration(tmp_path))
    )


def _saved_holder(tmp_path, name: str) -> SampleHolder:
    """The holder as the configuration now records it."""
    stage = utils.load_microscope_configuration(
        str(_configuration(tmp_path))
    ).system.stage
    assert stage.active_holder == name
    return stage.holders[name]


def _move_to(microscope, x, y, z=4e-3):
    sem = microscope.get_orientation(CALIBRATION_ORIENTATION)
    microscope.safe_absolute_stage_movement(
        FibsemStagePosition(x=x, y=y, z=z, r=sem.r, t=sem.t)
    )


def test_steps_follow_the_capacity(qapp, microscope, tmp_path):
    dialog = _dialog(qapp, microscope, _holder(2), tmp_path)
    assert dialog.step_titles == [
        "Holder",
        "Orientation",
        "Slot 1 of 2",
        "Slot 2 of 2",
        "Check & save",
    ]
    dialog.capacity_spin.setValue(3.0)
    assert dialog.step_titles[2:5] == ["Slot 1 of 3", "Slot 2 of 3", "Slot 3 of 3"]
    assert len(dialog._rail_rows) == 6


def test_orientation_step_refuses_next_until_at_sem(qapp, microscope, tmp_path):
    microscope.move_to_orientation("FIB")
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    dialog._on_next()  # Holder -> Orientation
    dialog._on_next()  # refused
    assert dialog.current_slot is None
    assert "refused" in dialog.orientation_status.text().lower()

    microscope.move_to_orientation(CALIBRATION_ORIENTATION)
    dialog._on_next()
    assert dialog.current_slot == "Slot-01"


def test_capture_refused_at_another_orientation(qapp, microscope, tmp_path):
    microscope.move_to_orientation(CALIBRATION_ORIENTATION)
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    dialog._show_step(2)
    microscope.move_to_orientation("FIB")
    dialog._on_capture()
    assert dialog._captured == {}
    assert "Refused" in dialog.capture_status.text()


def test_capture_reads_xyz_and_stamps_sem_rt(qapp, microscope, tmp_path):
    _move_to(microscope, x=2.5e-3, y=-1.0e-3)
    dialog = _dialog(qapp, microscope, _holder(2), tmp_path)
    dialog._show_step(2)
    dialog._on_capture()
    captured = dialog._captured["Slot-01"]
    sem = microscope.get_orientation(CALIBRATION_ORIENTATION)
    assert abs(captured.x - 2.5e-3) < 1e-9 and abs(captured.y + 1.0e-3) < 1e-9
    assert captured.r == sem.r and captured.t == sem.t
    assert captured.name == "Slot-01"
    assert dialog.button_move_to_slot.isEnabled()


def test_review_warns_when_two_slots_are_less_than_a_grid_apart(
    qapp, microscope, tmp_path
):
    dialog = _dialog(qapp, microscope, _holder(2), tmp_path)
    _move_to(microscope, x=0.0, y=0.0)
    dialog._show_step(2)
    dialog._on_capture()
    _move_to(microscope, x=0.5e-3, y=0.0)  # half a grid away
    dialog._show_step(3)
    dialog._on_capture()
    notes = dialog.review_notes()
    assert any("apart" in n for n in notes["Slot-01"])
    assert dialog.can_save()  # a warning, not a refusal


def test_review_refuses_a_slot_outside_the_stage_limits(qapp, microscope, tmp_path):
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    dialog._captured["Slot-01"] = FibsemStagePosition(
        name="Slot-01", x=5.0, y=0.0, z=4e-3, r=0.0, t=0.0
    )
    assert not dialog.can_save()
    dialog._show_step(dialog.review_step)
    assert not dialog.button_next.isEnabled()


def test_skipped_slot_keeps_its_saved_position(qapp, microscope, tmp_path):
    holder = _holder(2)
    kept = FibsemStagePosition(name="Slot-02", x=7e-3, y=1e-3, z=4e-3, r=0.0, t=0.0)
    holder.slots["Slot-02"].position = kept
    _move_to(microscope, x=1e-3, y=1e-3)
    dialog = _dialog(qapp, microscope, holder, tmp_path)
    dialog._show_step(2)
    dialog._on_capture()
    dialog._show_step(3)
    dialog._on_skip()
    assert "kept the saved position" in dialog.review_notes()["Slot-02"]
    assert dialog._position_for("Slot-02") is kept


def test_save_writes_the_config_and_emits(qapp, microscope, tmp_path):
    holder = _holder(2)
    dialog = _dialog(qapp, microscope, holder, tmp_path)
    saved = []
    dialog.holder_saved.connect(saved.append)
    dialog.name_edit.setText("Two-grid shuttle")
    dialog._on_next()  # applies the name

    _move_to(microscope, x=-3e-3, y=0.0)
    dialog._show_step(2)
    dialog._on_capture()
    _move_to(microscope, x=3e-3, y=0.0)
    dialog._show_step(3)
    dialog._on_capture()
    dialog._show_step(dialog.review_step)
    dialog._on_next()  # Save

    assert saved == [holder]
    assert holder.name == "Two-grid shuttle"
    again = _saved_holder(tmp_path, "Two-grid shuttle")
    assert again.name == "Two-grid shuttle"
    assert abs(again.slots["Slot-01"].position.x + 3e-3) < 1e-9
    assert abs(again.slots["Slot-02"].position.x - 3e-3) < 1e-9
    sem = microscope.get_orientation(CALIBRATION_ORIENTATION)
    assert again.slots["Slot-01"].position.t == sem.t


def test_cancel_leaves_the_holder_untouched(qapp, microscope, tmp_path):
    holder = _holder(1)
    before = holder.slots["Slot-01"].position
    _move_to(microscope, x=1e-3, y=2e-3)
    dialog = _dialog(qapp, microscope, holder, tmp_path)
    dialog._show_step(2)
    dialog._on_capture()
    dialog.reject()
    assert holder.slots["Slot-01"].position is before
    assert (
        "holders"
        not in (utils.load_yaml(str(_configuration(tmp_path))).get("calibration") or {})
        or not utils.load_yaml(str(_configuration(tmp_path)))["calibration"]["holders"]
    )


def test_live_position_follows_the_stage_signal(qapp, microscope, tmp_path):
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    dialog._show_step(2)
    dialog._on_stage_moved(FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.0, t=0.0))
    assert "X:1.00" in dialog.live_position.text()


def test_orientation_diagram_follows_the_stage(qapp, microscope, tmp_path):
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    microscope.move_to_orientation("FIB")
    dialog._show_step(1)
    assert dialog.stage_diagram._orientation == "FIB"
    assert dialog.stage_diagram._mirrored is True

    microscope.move_to_orientation(CALIBRATION_ORIENTATION)
    dialog._on_stage_moved(microscope.get_stage_position())
    assert dialog.stage_diagram._orientation == CALIBRATION_ORIENTATION
    assert dialog.stage_diagram._mirrored is False
    sem_tilt = microscope.get_orientation(CALIBRATION_ORIENTATION).t
    assert abs(dialog.stage_diagram._stage_tilt - math.degrees(sem_tilt)) < 1e-6


def test_save_writes_a_calibration_record_that_survives_reload(
    qapp, microscope, tmp_path
):
    holder = _holder(1)
    dialog = _dialog(qapp, microscope, holder, tmp_path)
    _move_to(microscope, x=-3e-3, y=0.0)
    dialog._show_step(2)
    dialog._on_capture()
    dialog._show_step(dialog.review_step)
    dialog._on_next()  # Save

    again = _saved_holder(tmp_path, "Test shuttle")
    slot = again.slots["Slot-01"]
    assert slot.is_calibrated
    assert slot.calibration.orientation == CALIBRATION_ORIENTATION
    assert slot.calibration.pre_tilt == microscope.system.stage.shuttle_pre_tilt
    assert slot.calibration.captured_at  # provenance
    # and the stage trusts it against the same geometry
    assert (
        again.discard_untrusted_positions(
            microscope.system.stage.shuttle_pre_tilt,
            microscope.system.stage.rotation_reference,
        )
        == []
    )


def _calibrate_one_slot(dialog, microscope, x=-3e-3):
    _move_to(microscope, x=x, y=0.0)
    dialog._show_step(2)
    dialog._on_capture()
    dialog._show_step(dialog.review_step)
    dialog._on_next()  # Save


def test_the_holder_is_saved_into_the_configuration_not_its_own_file(
    qapp, microscope, tmp_path
):
    """Where it used to go: `sample-holder.yaml`, which nothing copies into an
    experiment and which every connect re-imported."""
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)

    _calibrate_one_slot(dialog, microscope)

    assert (
        not Path(cfg.SAMPLE_HOLDER_CONFIGURATION_PATH).read_text().count("Test shuttle")
    )
    assert _saved_holder(tmp_path, "Test shuttle").slots["Slot-01"].is_calibrated


def test_a_renamed_holder_leaves_no_entry_under_its_old_name(
    qapp, microscope, tmp_path
):
    holder = _holder(1)
    microscope.system.stage.holders = {"Test shuttle": holder}
    microscope.system.stage.active_holder = "Test shuttle"
    dialog = _dialog(qapp, microscope, holder, tmp_path)
    dialog.name_edit.setText("Renamed shuttle")
    dialog._on_next()  # applies the name

    _calibrate_one_slot(dialog, microscope)

    stage = utils.load_microscope_configuration(
        str(_configuration(tmp_path))
    ).system.stage
    assert list(stage.holders) == ["Renamed shuttle"]
    assert stage.active_holder == "Renamed shuttle"
    assert microscope.system.stage.active_holder == "Renamed shuttle"


def test_the_next_connect_uses_the_calibrated_holder(qapp, microscope, tmp_path):
    dialog = _dialog(qapp, microscope, _holder(1), tmp_path)
    _calibrate_one_slot(dialog, microscope, x=-3e-3)

    again, _ = utils.setup_session(
        config_path=str(_configuration(tmp_path)), manufacturer="Demo"
    )

    holder = again._stage.holder
    assert holder.name == "Test shuttle"
    assert holder.slots["Slot-01"].is_calibrated
    assert abs(holder.slots["Slot-01"].position.x + 3e-3) < 1e-9


def test_without_a_configuration_file_there_is_nowhere_to_save(
    qapp, microscope, tmp_path
):
    microscope.configuration_path = None
    dialog = HolderCalibrationDialog(microscope, _holder(1))
    _move_to(microscope, x=-3e-3, y=0.0)
    dialog._show_step(2)
    dialog._on_capture()
    dialog._show_step(dialog.review_step)

    assert not dialog.can_save()
    assert "nowhere to save" in dialog.review_summary.text()
