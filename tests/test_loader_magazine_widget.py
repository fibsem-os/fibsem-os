"""Offscreen tests for LoaderMagazineWidget (Phase 4 grid UI).

Runs headlessly via the Qt 'offscreen' platform; skipped if a QApplication
cannot be created in this environment.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.microscopes._stage import SampleGrid, _create_sample_stage  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover - no Qt platform available
        pytest.skip(f"Qt unavailable: {e}")
    return app


@pytest.fixture
def magazine_widget(qapp):
    from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # gives a loader + magazine

    widget = LoaderMagazineWidget()
    widget.set_microscope(microscope)
    return widget, microscope


def _first_row(widget):
    return widget._list.itemWidget(widget._list.item(0))


def test_builds_rows_for_magazine(magazine_widget):
    widget, _ = magazine_widget
    assert widget.isEnabled()
    assert widget.capacity_label.text() == "12"
    assert widget._list.count() == 12


def test_dot_click_marks_available_and_emits(magazine_widget):
    widget, _ = magazine_widget
    seen = []
    widget.presence_toggled.connect(lambda name, available: seen.append((name, available)))

    row = _first_row(widget)
    row.status_dot.click()

    assert row.slot.loaded_grid is not None
    assert row.slot.loaded_grid.name == "Grid-01"  # auto-named from the slot number
    assert row.status() == "white"
    assert row.btn_load.isEnabled()  # auto-named → ready to load
    assert seen[-1] == (row.slot.name, True)


def test_empty_slot_name_disabled_until_added(magazine_widget):
    widget, _ = magazine_widget
    changed = []
    widget.magazine_changed.connect(lambda: changed.append(True))

    row = _first_row(widget)
    # empty slot: name/description are disabled — the dot is the only add path
    assert row.slot.loaded_grid is None
    assert not row.name_edit.isEnabled()
    assert not row.desc_edit.isEnabled()

    # click the dot to add (auto-names Grid-NN), then the name becomes editable
    row.status_dot.click()
    assert row.slot.loaded_grid is not None
    assert row.name_edit.isEnabled()
    assert row.desc_edit.isEnabled()

    # now the grid can be renamed inline
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    assert row.slot.loaded_grid.name == "grid-aspen"
    assert row.status() == "white"
    assert changed


def test_duplicate_name_is_rejected(magazine_widget):
    widget, _ = magazine_widget
    row0 = widget._list.itemWidget(widget._list.item(0))
    row1 = widget._list.itemWidget(widget._list.item(1))

    row0.name_edit.setText("grid-aspen")
    row0._on_name_edited()
    # try to reuse the same name on another slot
    row1.name_edit.setText("grid-aspen")
    row1._on_name_edited()

    # row1's grid was not given the duplicate name (reverted) and is flagged
    assert row1.slot.loaded_grid is None or row1.slot.loaded_grid.name != "grid-aspen"
    assert row1.name_edit.toolTip() == "Name already in use"
    # renaming a slot to its own current name is fine (not a self-collision)
    row0.name_edit.setText("grid-aspen")
    row0._on_name_edited()
    assert row0.slot.loaded_grid.name == "grid-aspen"


def test_row_action_toggles_to_unload_when_loaded(magazine_widget):
    widget, microscope = magazine_widget
    row = _first_row(widget)

    # available, not loaded → action button loads
    row.status_dot.click()
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    assert row.status() == "white"
    assert row.btn_load.toolTip() == "Load this grid onto the microscope"

    # put it in the working slot → the same button now unloads
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()
    assert row.status() == "green"
    assert row.btn_load.isEnabled()
    assert row.btn_load.toolTip() == "Unload this grid from the microscope"


def test_loader_name_shown(magazine_widget):
    widget, microscope = magazine_widget
    assert widget.name_label.text() == microscope._stage.loader.name


def test_inline_description_edit(magazine_widget):
    widget, _ = magazine_widget
    row = _first_row(widget)
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    row.desc_edit.setText("lamella candidate")
    row._on_desc_edited()

    assert row.slot.loaded_grid.description == "lamella candidate"


def test_run_inventory_reflects_loaded_slots(magazine_widget):
    widget, microscope = magazine_widget
    _first_row(widget).status_dot.click()

    loaded = microscope._stage.loader.run_inventory()
    assert len(loaded) == 1
    widget._on_run_inventory()  # instant path (no delay) — must not raise


def test_run_inventory_threads_with_spinner_when_delayed(qapp, magazine_widget):
    widget, microscope = magazine_widget
    microscope._stage.loader.inventory_delay_s = 0.05  # force the threaded path

    widget._on_run_inventory()
    assert widget._inv_thread is not None and widget._inv_thread.is_alive()
    assert widget._spinner._timer.isActive()  # spinner spinning during scan
    assert not widget.btn_inventory.isEnabled()

    widget._inv_thread.join(timeout=2.0)
    qapp.processEvents()  # deliver the _inventory_done signal
    assert not widget._spinner._timer.isActive()  # stopped on completion
    assert widget.btn_inventory.isEnabled()


def test_load_button_emits_load_requested(magazine_widget):
    widget, _ = magazine_widget
    seen = []
    widget.load_requested.connect(seen.append)

    row = _first_row(widget)
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    row.btn_load.click()

    assert seen == ["grid-aspen"]


def test_row_action_emits_unload_requested_when_loaded(magazine_widget):
    widget, microscope = magazine_widget
    seen = []
    widget.unload_requested.connect(lambda: seen.append(True))

    # load a grid into the working slot, then click its (now Unload) action
    row = _first_row(widget)
    row.status_dot.click()
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()

    assert row.status() == "green"
    row.btn_load.click()  # same button, now unloads

    assert seen == [True]


def test_dot_click_toggles_empty(magazine_widget):
    widget, _ = magazine_widget
    row = _first_row(widget)
    row.status_dot.click()  # empty -> available
    assert row.slot.loaded_grid is not None

    row.status_dot.click()  # available -> empty
    assert row.slot.loaded_grid is None


def test_status_dot_reflects_beam_state(magazine_widget):
    widget, microscope = magazine_widget
    row = _first_row(widget)

    # empty -> gray, load disabled
    assert row.status() == "gray"
    assert not row.btn_load.isEnabled()

    # clicking the dot marks it available and auto-names it -> white, load enabled
    row.status_dot.click()
    assert row.status() == "white"
    assert row.slot.loaded_grid.name == "Grid-01"
    assert row.btn_load.isEnabled()

    # renaming keeps it white + loadable
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    assert row.status() == "white"
    assert row.btn_load.isEnabled()

    # put it in the working slot -> green, action button toggles to Unload
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()
    assert row.status() == "green"
    assert row.btn_load.isEnabled()
    assert row.btn_load.toolTip() == "Unload this grid from the microscope"


def test_set_busy_runs_spinner_and_blocks(magazine_widget):
    widget, _ = magazine_widget

    widget.set_busy(True)
    assert widget._spinner._timer.isActive()  # spinner spinning
    assert not widget._list.isEnabled()        # rows blocked
    assert not widget.btn_inventory.isEnabled()

    widget.set_busy(False)
    assert not widget._spinner._timer.isActive()
    assert widget._list.isEnabled()
    assert widget.btn_inventory.isEnabled()


def test_disabled_without_loader(qapp):
    from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")  # non-compustage → no loader
    widget = LoaderMagazineWidget()
    widget.set_microscope(microscope)

    assert not widget.isEnabled()
    assert widget._list.count() == 0
