"""Offscreen tests for GridWorkflowWidget (grid run selection)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import GridRecord  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (  # noqa: E402
    GRID_TASK_REGISTRY,
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
    from fibsem.ui.widgets.grid_workflow_widget import GridWorkflowWidget

    return GridWorkflowWidget()


def _check(checklist, i):
    checklist._list.item(i).setCheckState(Qt.Checked)


def test_tasks_populate_from_registry(widget):
    assert widget._tasks._list.count() == len(GRID_TASK_REGISTRY)
    # values are the registry keys (task_type), used by GridTaskManager.run
    values = {
        widget._tasks._list.item(i).data(Qt.UserRole)
        for i in range(widget._tasks._list.count())
    }
    assert values == set(GRID_TASK_REGISTRY.keys())


def test_grids_populate_and_select(widget):
    widget.set_grids([GridRecord(name="grid-aspen"), GridRecord(name="grid-birch")])
    assert widget._grids._list.count() == 2

    _check(widget._grids, 0)
    assert [g.name for g in widget.get_selected_grids()] == ["grid-aspen"]


def test_selection_changed_signals(widget):
    widget.set_grids([GridRecord(name="grid-aspen")])
    grids_seen, tasks_seen = [], []
    widget.grid_selection_changed.connect(grids_seen.append)
    widget.task_selection_changed.connect(tasks_seen.append)

    _check(widget._grids, 0)
    _check(widget._tasks, 0)

    assert grids_seen and [g.name for g in grids_seen[-1]] == ["grid-aspen"]
    assert tasks_seen and len(tasks_seen[-1]) == 1


def test_select_all_grids(widget):
    widget.set_grids([GridRecord(name="a"), GridRecord(name="b"), GridRecord(name="c")])
    widget._grids._select_all.setChecked(True)  # user toggle
    assert len(widget.get_selected_grids()) == 3


def test_set_grids_preserves_checks_by_name(widget):
    widget.set_grids([GridRecord(name="a"), GridRecord(name="b")])
    _check(widget._grids, 1)  # check 'b'
    assert [g.name for g in widget.get_selected_grids()] == ["b"]

    # repopulate (reordered) — 'b' stays checked
    widget.set_grids([GridRecord(name="b"), GridRecord(name="a")])
    assert [g.name for g in widget.get_selected_grids()] == ["b"]
