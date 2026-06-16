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


def test_presence_toggle_creates_grid_and_emits(magazine_widget):
    widget, _ = magazine_widget
    seen = []
    widget.presence_toggled.connect(lambda name, loaded: seen.append((name, loaded)))

    row = _first_row(widget)
    row.presence_check.setChecked(True)

    assert row.slot.loaded_grid is not None
    assert row.slot.loaded_grid.name == row.slot.name  # unnamed default
    assert seen[-1] == (row.slot.name, True)


def test_naming_via_edit_panel(magazine_widget):
    widget, _ = magazine_widget
    row = _first_row(widget)
    row.presence_check.setChecked(True)

    widget._edit_panel.grid_name_edit.setText("grid-aspen")
    widget._edit_panel._handle_apply()

    assert row.slot.loaded_grid.name == "grid-aspen"


def test_run_inventory_reflects_loaded_slots(magazine_widget):
    widget, microscope = magazine_widget
    _first_row(widget).presence_check.setChecked(True)

    loaded = microscope._stage.loader.run_inventory()
    assert len(loaded) == 1
    widget._on_run_inventory()  # must not raise


def test_load_button_emits_load_requested(magazine_widget):
    widget, _ = magazine_widget
    seen = []
    widget.load_requested.connect(seen.append)

    row = _first_row(widget)
    row.presence_check.setChecked(True)
    widget._edit_panel.grid_name_edit.setText("grid-aspen")
    widget._edit_panel._handle_apply()
    row.btn_load.click()

    assert seen == ["grid-aspen"]


def test_clear_empties_slot(magazine_widget):
    widget, _ = magazine_widget
    row = _first_row(widget)
    row.presence_check.setChecked(True)
    assert row.slot.loaded_grid is not None

    row.btn_clear.click()
    assert row.slot.loaded_grid is None


def test_disabled_without_loader(qapp):
    from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")  # non-compustage → no loader
    widget = LoaderMagazineWidget()
    widget.set_microscope(microscope)

    assert not widget.isEnabled()
    assert widget._list.count() == 0
