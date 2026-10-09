"""The sample holder panel: calibration state at a glance, grids named inline."""

from datetime import datetime, timezone

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
    microscope.stage_device.compustage = False
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
    assert widget.hint_label.isVisibleTo(widget)
    row = widget._row_widget(0)
    assert row.status == "unavailable"
    assert "not calibrated" in row.toolTip()
    assert not row.btn_move.isEnabled()
    assert "2 slots, 0 calibrated" in widget.facts_label.text()
    assert "pre-tilt 35°" in widget.facts_label.text()


def test_calibrated_slot_shows_when_and_can_move(qapp, microscope):
    holder = microscope._stage.holder
    _calibrate(holder.slots["Slot-01"])
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert "2 slots, 1 calibrated" in widget.facts_label.text()
    row = widget._row_widget(0)
    assert row.status == "calibrated"  # trusted position, nothing in it yet
    assert "Calibrated 2 Sep 2026 11:24" in row.toolTip()
    assert "SEM orientation, pre-tilt 35°" in row.toolTip()
    assert row.btn_move.isEnabled()
    requested = []
    widget.move_to_requested.connect(requested.append)
    row.btn_move.click()
    # hosted: a request the Movement widget routes through its own move path
    assert len(requested) == 1 and abs(requested[0].x + 5e-3) < 1e-9
    assert abs(microscope.get_stage_position().x + 5e-3) > 1e-6  # not moved here


def test_fully_calibrated_hides_the_hint(qapp, microscope):
    holder = microscope._stage.holder
    _calibrate(holder.slots["Slot-01"])
    _calibrate(holder.slots["Slot-02"], x=5e-3)
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert "2 slots, 2 calibrated" in widget.facts_label.text()
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


def test_status_dot_is_available_only_when_calibrated_with_a_grid(qapp, microscope):
    holder = microscope._stage.holder
    holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-aspen")
    _calibrate(holder.slots["Slot-01"])
    holder.slots["Slot-02"].loaded_grid = SampleGrid(name="grid-birch")  # uncalibrated
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert widget._row_widget(0).status == "available"
    assert widget._row_widget(1).status == "unavailable"


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


def test_panel_is_titled_after_the_holder(qapp, microscope):
    holder = microscope._stage.holder
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert widget._panel._title_label.text() == holder.name  # set in the wizard
    widget.set_holder(None)
    assert widget._panel._title_label.text() == "Sample Holder"


def test_unhosted_widget_moves_the_stage_itself(qapp, microscope):
    holder = microscope._stage.holder
    _calibrate(holder.slots["Slot-01"])
    widget = SampleHolderWidget(microscope=microscope, move_directly=True)
    widget.set_holder(holder)
    widget._row_widget(0).btn_move.click()
    assert abs(microscope.get_stage_position().x + 5e-3) < 1e-9


def test_naming_persists_to_the_session_state_not_the_calibration(qapp, microscope):
    from pathlib import Path

    import fibsem.config as cfg
    from fibsem.session_state import session_state_for

    holder = microscope._stage.holder
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    row = widget._row_widget(0)
    row.name_edit.setText("grid-aspen")
    row.name_edit.editingFinished.emit()

    recorded = session_state_for(microscope).load_section("holder_occupancy")
    assert recorded["Slot-01"]["name"] == "grid-aspen"
    assert "grid-aspen" not in Path(cfg.SAMPLE_HOLDER_CONFIGURATION_PATH).read_text()


def test_names_are_read_only_with_a_loader(qapp):
    from tests.fixtures.demo_stage import demo_grid_loader

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_device.compustage = True
    microscope._stage = _create_sample_stage(microscope)
    microscope._stage.loader = demo_grid_loader(microscope, occupied=(1,))
    microscope._stage.ensure_loaded("Grid-01")
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(microscope._stage.holder)
    row = widget._row_widget(0)
    assert row.name_edit.isReadOnly()
    assert row.name_edit.text() == "Grid-01"


def test_calibrate_opens_the_wizard_non_modal(qapp, microscope):
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(microscope._stage.holder)
    widget.btn_calibrate.click()
    dialog = widget._calibration_dialog
    assert dialog is not None and not dialog.isModal()
    assert dialog.step_titles[0] == "Holder"
    dialog.close()


def test_without_a_microscope_nothing_moves_or_calibrates(qapp):
    holder = SampleHolder(pre_tilt=0.0, name="h", capacity=1)
    holder._ensure_slots()
    _calibrate(holder.slots["Slot-01"])
    widget = SampleHolderWidget(microscope=None)
    widget.set_holder(holder)
    assert not widget.btn_calibrate.isEnabled()
    assert not widget._row_widget(0).btn_move.isEnabled()


def test_a_rename_is_announced_and_a_new_or_cleared_grid_is_not(qapp, microscope):
    holder = microscope._stage.holder
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    renamed = []
    widget.grid_renamed.connect(lambda old, new: renamed.append((old, new)))
    row = widget._row_widget(1)
    row.name_edit.setText("grid-birch")  # a grid declared in an empty slot
    row.name_edit.editingFinished.emit()
    row.name_edit.setText("grid-cedar")  # that grid renamed
    row.name_edit.editingFinished.emit()
    row.name_edit.setText("")  # and taken out
    row.name_edit.editingFinished.emit()
    assert renamed == [("grid-birch", "grid-cedar")]


def test_the_host_can_refuse_a_rename_on_a_fixed_holder(qapp, microscope):
    holder = microscope._stage.holder
    holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-aspen")
    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    widget.set_rename_check(lambda old, new: f"{old} cannot be renamed.")
    changed = []
    widget.holder_changed.connect(changed.append)
    row = widget._row_widget(0)
    row.name_edit.setText("grid-birch")
    row.name_edit.editingFinished.emit()
    assert holder.slots["Slot-01"].loaded_grid.name == "grid-aspen"
    assert widget._row_widget(0).name_edit.text() == "grid-aspen"
    assert changed == []


def test_a_calibration_with_an_offset_shows_in_the_viewers_zone(qapp, microscope):
    """New records carry their UTC offset (FIB-1190); `_calibrate`'s does not."""
    holder = microscope._stage.holder
    slot = holder.slots["Slot-01"]
    _calibrate(slot)
    slot.calibration.captured_at = "2026-09-02T11:24:09+00:00"
    local = datetime(2026, 9, 2, 11, 24, 9, tzinfo=timezone.utc).astimezone()
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    expected = f"{local.day} {months[local.month - 1]} {local:%Y %H:%M}"

    widget = SampleHolderWidget(microscope=microscope)
    widget.set_holder(holder)
    assert f"Calibrated {expected}" in widget._row_widget(0).toolTip()
