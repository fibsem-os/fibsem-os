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
from fibsem.applications.autolamella.workflows.tasks.grid import (  # noqa: E402
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


def _check_task(widget, i):
    widget._tasks._rows[i].checkbox.setChecked(True)


def _protocol(*names) -> GridTaskProtocol:
    proto = GridTaskProtocol()
    for n in names:
        proto.task_config[n] = AcquireImageGridTaskConfig(task_name=n)
    proto.reconcile_workflow()
    return proto


def test_tasks_populate_from_protocol_instances(widget):
    # the task list shows configured instances (task_name), not registry types,
    # so multiples of the same task type each appear as a distinct ordered row
    widget.set_protocol(_protocol("Acquire Image", "Acquire Image (2)"))
    rows = widget._tasks._rows
    assert [r.task.name for r in rows] == ["Acquire Image", "Acquire Image (2)"]
    # selected values are the task_names passed to the runner
    _check_task(widget, 0)
    assert "Acquire Image" in widget.get_selected_tasks()


def test_grid_task_supervised_lookup(qapp):
    # the status-bar chip reads grid supervision via this helper
    from types import SimpleNamespace

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )

    proto = _protocol("A", "B")
    proto.workflow_config.get("B").supervise = True
    stub = SimpleNamespace(
        autolamella_ui=SimpleNamespace(experiment=SimpleNamespace(grid_protocol=proto))
    )
    f = AutoLamellaSingleWindowUI._grid_task_supervised
    assert f(stub, "B") is True
    assert f(stub, "A") is False
    assert f(stub, "missing") is False  # absent description → automated


def test_protocol_changes_refresh_checklist(widget):
    proto = _protocol("Acquire Image")
    widget.set_protocol(proto)
    assert len(widget._tasks._rows) == 1
    # adding an instance in the (separate) editor tab reconciles + refreshes live
    proto.task_config["Acquire Image (lo-mag)"] = AcquireImageGridTaskConfig(
        task_name="Acquire Image (lo-mag)"
    )
    assert len(widget._tasks._rows) == 2


def test_task_order_and_supervise_persist(widget):
    proto = _protocol("A", "B")
    widget.set_protocol(proto)
    changed = []
    widget.workflow_changed.connect(lambda: changed.append(True))

    # drag B above A → run order B, A (simulate the list's drop → reordered)
    a, b = widget._tasks._rows[0].task, widget._tasks._rows[1].task
    widget._tasks._on_reordered([b, a])
    assert proto.workflow_config.order == ["B", "A"]

    # supervise toggle persists onto the description (row 0 is now B)
    widget._tasks._rows[0].btn_supervise.setChecked(True)
    assert proto.workflow_config.get("B").supervise is True
    assert changed  # both edits emitted workflow_changed → host persists

    # selection returns checked names in run order
    widget._tasks._rows[0].checkbox.setChecked(True)  # B
    widget._tasks._rows[1].checkbox.setChecked(True)  # A
    assert widget.get_selected_tasks() == ["B", "A"]


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
    _check_task(widget, 0)

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
