"""The magazine panel on the Arctis simulator, and the Sample view that holds it."""

import os

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.widgets.sample_loader_widget import SampleLoaderWidget


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    assert microscope._stage.loader is not None
    return microscope


@pytest.fixture
def widget(qapp, arctis):
    return SampleLoaderWidget(microscope=arctis, synchronous=True)


def states(widget):
    return [r.state for r in widget._rows]


def test_every_magazine_slot_has_a_row(widget):
    assert len(widget._rows) == 12
    assert states(widget)[:4] == ["occupied", "occupied", "occupied", "empty"]
    assert "12 slots · 3 grids · not scanned this session" in widget.facts_label.text()
    first, empty = widget._row_widget(0), widget._row_widget(3)
    assert first.slot_label.text() == "01" and first.name_edit.text() == "Grid-01"
    assert (
        first.btn_action.isEnabled() and "into the beam" in first.btn_action.toolTip()
    )
    assert not first.name_edit.isReadOnly()
    assert empty.name_edit.text() == "" and empty.name_edit.isReadOnly()
    assert not empty.btn_action.isEnabled() and empty.btn_action.icon().isNull()


def test_load_brings_the_grid_into_the_beam(widget, arctis):
    changed = []
    widget.loader_changed.connect(lambda: changed.append(True))
    widget._row_widget(1).btn_action.click()
    assert arctis._stage.loaded_grids[0].name == "Grid-02"
    assert states(widget)[:3] == ["occupied", "in_beam", "occupied"]
    # the loaded row's action turns into Unload; the others still offer Load
    assert "Return Grid-02" in widget._row_widget(1).btn_action.toolTip()
    assert "into the beam" in widget._row_widget(0).btn_action.toolTip()
    assert widget.status_label.text() == "Grid-02 is in the beam."
    assert changed == [True]
    assert not widget.busy


def test_unload_returns_it(widget, arctis):
    widget._row_widget(0).btn_action.click()
    widget._row_widget(0).btn_action.click()  # now the unload action
    assert arctis._stage.loaded_grids == []
    assert states(widget)[0] == "occupied"
    assert "returned to the magazine" in widget.status_label.text()


def test_run_inventory_asks_first_then_stamps_the_scan(widget, monkeypatch):
    asked = []
    widget._synchronous = False  # so the confirmation is consulted...
    monkeypatch.setattr(
        widget, "_confirm", lambda title, text: asked.append(title) or False
    )
    widget.btn_inventory.click()
    assert asked == ["Run inventory"] and "not scanned" in widget.facts_label.text()
    monkeypatch.undo()
    widget._synchronous = True  # ...and answered yes without a dialog
    widget.btn_inventory.click()
    assert "scanned " in widget.facts_label.text()
    assert widget.status_label.text() == "Inventory complete."


def test_a_refused_exchange_is_reported_and_the_controls_come_back(widget, arctis):
    arctis._stage.loader.fail_next_exchange = True
    widget._row_widget(0).btn_action.click()
    assert arctis._stage.loaded_grids == []
    assert "Simulated autoloader exchange failure" in widget.status_label.text()
    assert widget._row_widget(0).btn_action.isEnabled()
    assert not widget.busy


def test_naming_a_grid_goes_through_the_stage(widget, arctis, monkeypatch):
    written = []
    monkeypatch.setattr(
        arctis._stage.loader,
        "_write_slot_description",
        lambda slot: written.append((slot.name, slot.loaded_grid.name)),
    )
    row = widget._row_widget(2)
    row.name_edit.setText("grid-cedar")
    row.name_edit.editingFinished.emit()
    assert written == [("Slot-03", "grid-cedar")]
    assert arctis._stage.loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
    assert widget._row_widget(2).name_edit.text() == "grid-cedar"


def test_a_grid_cannot_be_unnamed(widget, arctis):
    row = widget._row_widget(0)
    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    assert row.name_edit.text() == "Grid-01"
    assert arctis._stage.loader.slots["Slot-01"].loaded_grid.name == "Grid-01"


def test_the_host_can_lock_the_exchange_controls(widget):
    widget.set_controls_enabled(False)
    assert not widget.btn_inventory.isEnabled()
    assert not widget._row_widget(0).btn_action.isEnabled()
    assert "workflow" in widget._row_widget(0).btn_action.toolTip()
    widget.set_controls_enabled(True)
    assert widget._row_widget(0).btn_action.isEnabled()


class TestSampleView:
    def test_has_the_loader_only_when_there_is_one(self, qapp, arctis):
        view = FibsemSampleWidget(microscope=arctis)
        assert view.loader_widget is not None
        assert view.holder_widget.current_holder is arctis._stage.holder

        plain, _ = utils.setup_session(manufacturer="Demo")
        assert FibsemSampleWidget(microscope=plain).loader_widget is None

    def test_an_exchange_repaints_the_holder(self, qapp, arctis):
        view = FibsemSampleWidget(microscope=arctis)
        view.loader_widget._synchronous = True
        view.loader_widget._row_widget(0).btn_action.click()
        assert view.holder_widget._row_widget(0).name_edit.text() == "Grid-01"

    def test_move_requests_pass_through(self, qapp, arctis):
        view = FibsemSampleWidget(microscope=arctis)
        received = []
        view.move_to_requested.connect(received.append)
        view.holder_widget._row_widget(0).btn_move.click()
        assert (
            len(received) == 1
        )  # the compustage working slot is calibrated by construction
