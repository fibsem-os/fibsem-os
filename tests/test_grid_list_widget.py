"""Offscreen tests for GridListWidget (Phase 4a grid UI)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    GridRecord,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {e}")
    return app


@pytest.fixture
def widget(qapp):
    from fibsem.ui.widgets.grid_list_widget import GridListWidget

    return GridListWidget()


def _records():
    a = GridRecord(name="grid-aspen")
    a.task_history = [AutoLamellaTaskState(name="overview"), AutoLamellaTaskState(name="clean")]
    b = GridRecord(name="grid-birch")  # not started
    return [a, b]


def _row(widget, i):
    return widget._list.itemWidget(widget._list.item(i))


def test_population_and_count(widget):
    widget.set_grids(_records())
    assert widget._list.count() == 2
    assert widget._count.text() == "· 2"
    assert _row(widget, 0).name_label.text() == "grid-aspen"
    assert _row(widget, 0).status_label.text() == "2 tasks complete"
    assert _row(widget, 1).status_label.text() == "Not started"


def test_slot_label_displayed(widget):
    records = _records()
    widget.set_grids(records, slot_labels={"grid-aspen": "01"})
    assert _row(widget, 0).slot_label.text() == "01"   # mapped
    assert _row(widget, 1).slot_label.text() == "—"     # not in magazine


def test_status_text_failed(widget):
    rec = GridRecord(name="grid-cedar")
    rec.task_state.status = AutoLamellaTaskStatus.Failed
    widget.set_grids([rec])
    assert _row(widget, 0).status_label.text() == "Failed"


def test_empty_state(widget):
    widget.set_grids([])
    assert widget._list.count() == 0
    assert widget._empty.isVisible() or not widget._list.isVisible()
    assert widget._count.text() == "· 0"


def test_add_from_loader_emits(widget):
    seen = []
    widget.add_from_loader_requested.connect(lambda: seen.append(True))
    widget.btn_add.click()
    assert seen == [True]


def test_row_remove_emits(widget):
    records = _records()
    widget.set_grids(records)

    seen = []
    widget.remove_requested.connect(seen.append)
    # the trash button lives on each row; click the second row's
    _row(widget, 1).btn_remove.click()
    assert seen and seen[0].name == "grid-birch"


def test_grid_selected_emits(widget):
    seen = []
    widget.grid_selected.connect(seen.append)
    records = _records()
    widget.set_grids(records)
    widget._list.setCurrentRow(1)
    assert seen[-1].name == "grid-birch"


def test_selection_preserved_by_name(widget):
    records = _records()
    widget.set_grids(records)
    widget._list.setCurrentRow(1)  # select grid-birch
    assert widget.selected_grid.name == "grid-birch"

    # repopulate in reversed order — selection should stay on grid-birch
    widget.set_grids(list(reversed(records)))
    assert widget.selected_grid.name == "grid-birch"
