"""The sample holder panel: calibration state at a glance, grids named inline."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem import utils
from fibsem.microscopes._stage import (
    SampleGrid,
    SampleHolder,
    SlotCalibration,
    _create_sample_stage,
)
from fibsem.structures import FibsemStagePosition
from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    return microscope


def _calibrate(slot, x=-5e-3):
    slot.position = FibsemStagePosition(
        name=slot.name, x=x, y=0.0, z=4e-3, r=0.0, t=0.61
    )
    slot.calibration = SlotCalibration(
        orientation="SEM",
        pre_tilt=35.0,
        rotation_reference=0.0,
        captured_at="2026-09-02T11:24:09",
        fibsem_version="0.5.2",
    )


def test_uncalibrated_holder_reads_as_such(qapp, microscope):
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(microscope._stage.holder)
    assert widget.status_chip.text() == "not calibrated"
    assert widget.hint_label.isVisibleTo(widget)
    row = widget._row_widget(0)
    assert row.calibration_chip.text() == "not calibrated"
    assert not row.btn_move.isEnabled()
    assert "2 slots" in widget.facts_label.text()
    assert "pre-tilt 35°" in widget.facts_label.text()


def test_calibrated_slot_shows_when_and_can_move(qapp, microscope):
    holder = microscope._stage.holder
    _calibrate(holder.slots["Slot-01"])
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert widget.status_chip.text() == "1 of 2 calibrated"
    row = widget._row_widget(0)
    assert row.calibration_chip.text() == "SEM · 2 Sep 11:24"
    assert row.btn_move.isEnabled()
    row.btn_move.click()
    assert abs(microscope.get_stage_position().x + 5e-3) < 1e-9


def test_fully_calibrated_hides_the_hint(qapp, microscope):
    holder = microscope._stage.holder
    _calibrate(holder.slots["Slot-01"])
    _calibrate(holder.slots["Slot-02"], x=5e-3)
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert widget.status_chip.text() == "2 of 2 calibrated"
    assert not widget.hint_label.isVisibleTo(widget)


def test_naming_a_grid_inline_updates_the_holder_and_emits(qapp, microscope):
    holder = microscope._stage.holder
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    changed = []
    widget.holder_changed.connect(changed.append)
    row = widget._row_widget(1)
    row.name_edit.setText("grid-birch")
    row.name_edit.editingFinished.emit()
    assert holder.slots["Slot-02"].loaded_grid.name == "grid-birch"
    assert changed == [holder]

    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    assert holder.slots["Slot-02"].loaded_grid is None
    assert len(changed) == 2


def test_unchanged_name_does_not_emit(qapp, microscope):
    holder = microscope._stage.holder
    holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-aspen")
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    changed = []
    widget.holder_changed.connect(changed.append)
    row = widget._row_widget(0)
    assert row.name_edit.text() == "grid-aspen"
    row.name_edit.editingFinished.emit()
    assert changed == []


def test_renaming_the_holder(qapp, microscope):
    holder = microscope._stage.holder
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    widget.name_edit.setText("Two-grid shuttle")
    widget.name_edit.editingFinished.emit()
    assert holder.name == "Two-grid shuttle"
    widget.name_edit.setText("   ")
    widget.name_edit.editingFinished.emit()
    assert widget.name_edit.text() == "Two-grid shuttle"  # blank is refused


def test_calibrate_opens_the_wizard_non_modal(qapp, microscope):
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(microscope._stage.holder)
    widget.btn_calibrate.click()
    dialog = widget._calibration_dialog
    assert dialog is not None and not dialog.isModal()
    assert dialog.step_titles[0] == "Holder"
    dialog.close()


def test_without_a_microscope_nothing_moves_or_calibrates(qapp):
    holder = SampleHolder(name="h", capacity=1)
    holder._ensure_slots()
    _calibrate(holder.slots["Slot-01"])
    widget = SampleHolderWidget(microscope=None)
    widget.set_holder(holder)
    assert not widget.btn_calibrate.isEnabled()
    assert not widget._row_widget(0).btn_move.isEnabled()
