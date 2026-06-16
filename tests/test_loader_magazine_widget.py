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


def test_inline_naming_autoloads_without_ticking(magazine_widget):
    widget, _ = magazine_widget
    changed = []
    widget.magazine_changed.connect(lambda: changed.append(True))

    row = _first_row(widget)
    assert row.slot.loaded_grid is None  # starts empty
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()  # editingFinished

    # typing a name loaded the slot and marked it available — no dot click needed
    assert row.slot.loaded_grid is not None
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


def test_unload_button_disabled_until_loaded(magazine_widget):
    widget, microscope = magazine_widget
    assert not widget.btn_unload.isEnabled()  # nothing in the working slot

    row = _first_row(widget)
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()
    assert widget.btn_unload.isEnabled()


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
    widget._on_run_inventory()  # must not raise


def test_load_button_emits_load_requested(magazine_widget):
    widget, _ = magazine_widget
    seen = []
    widget.load_requested.connect(seen.append)

    row = _first_row(widget)
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    row.btn_load.click()

    assert seen == ["grid-aspen"]


def test_unload_button_emits_unload_requested(magazine_widget):
    widget, microscope = magazine_widget
    seen = []
    widget.unload_requested.connect(lambda: seen.append(True))

    # unload is enabled only once a grid is in the working slot
    row = _first_row(widget)
    row.name_edit.setText("grid-aspen")
    row._on_name_edited()
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()

    widget.btn_unload.click()

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

    # put it in the working slot -> green, load disabled (already loaded)
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    GridTaskManager(microscope, experiment=None).ensure_loaded(GridRecord(name="grid-aspen"))
    widget.refresh_rows()
    assert row.status() == "green"
    assert not row.btn_load.isEnabled()


def test_disabled_without_loader(qapp):
    from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")  # non-compustage → no loader
    widget = LoaderMagazineWidget()
    widget.set_microscope(microscope)

    assert not widget.isEnabled()
    assert widget._list.count() == 0
