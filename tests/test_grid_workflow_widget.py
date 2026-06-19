"""Offscreen tests for GridWorkflowWidget (grid run selection)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    GridRecord,
    GridTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (  # noqa: E402
    AcquireImageGridTaskConfig,
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


def _protocol(*names) -> GridTaskProtocol:
    proto = GridTaskProtocol()
    for n in names:
        proto.task_config[n] = AcquireImageGridTaskConfig(task_name=n)
    return proto


def test_tasks_populate_from_protocol_instances(widget):
    # the checklist lists configured instances (task_name), not registry types,
    # so multiples of the same task type each appear as a distinct entry
    widget.set_protocol(_protocol("Acquire Image", "Acquire Image (2)"))
    assert widget._tasks._list.count() == 2
    values = {
        widget._tasks._list.item(i).data(Qt.UserRole)
        for i in range(widget._tasks._list.count())
    }
    assert values == {"Acquire Image", "Acquire Image (2)"}
    # selected values are the task_names passed to the runner
    _check(widget._tasks, 0)
    assert "Acquire Image" in widget.get_selected_tasks()


def test_protocol_changes_refresh_checklist(widget):
    proto = _protocol("Acquire Image")
    widget.set_protocol(proto)
    assert widget._tasks._list.count() == 1
    # adding an instance in the (separate) editor tab refreshes the checklist live
    proto.task_config["Acquire Image (lo-mag)"] = AcquireImageGridTaskConfig(
        task_name="Acquire Image (lo-mag)"
    )
    assert widget._tasks._list.count() == 2


def test_grids_populate_and_select(widget):
    widget.set_grids([GridRecord(name="grid-aspen"), GridRecord(name="grid-birch")])
    assert widget._grids._list.count() == 2

    _check(widget._grids, 0)
    assert [g.name for g in widget.get_selected_grids()] == ["grid-aspen"]


def test_selection_changed_signals(widget):
    widget.set_grids([GridRecord(name="grid-aspen")])
    widget.set_protocol(_protocol("Acquire Image"))
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
