"""Headless tests for the workflow timeline reconciling against a mutable queue.

The timeline used to be built exactly once per run and could only shrink to fit
(`_workflow_timeline_initialized` in AutoLamellaMainUI, plus a `break` past the
initial row count in `update_from_status`), so an item added mid-run never
appeared and a removed one left a stale row behind.

Rows are now matched to queue items by `WorkItem.id`. That is what lets a row
keep its widget — and therefore its inner step list, error text and subtitle —
while items around it are added, removed or reordered. Note that `queue.items`
hands out *copies*, so a snapshot never shares object identity with the queue
or with the previous snapshot; the id is the only stable handle.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    StepStatus,
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue


@pytest.fixture
def queue() -> TaskQueue:
    q = TaskQueue()
    q.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return q


@pytest.fixture
def widget(qapp) -> WorkflowProgressWidget:
    return WorkflowProgressWidget()


def status_for(queue: TaskQueue, item, status, **extra) -> dict:
    """The subset of TaskManager._emit_status's payload the timeline reads."""
    payload = {
        "task_name": item.task_name,
        "item_name": item.lamella_name,
        "status": status,
        "queue_items": queue.items,
        "timestamp": 1000.0,
    }
    payload.update(extra)
    return payload


def labels(widget: WorkflowProgressWidget):
    """(lamella, task) per visible row, read back off the rendered widgets."""
    return [(r._label.text(), r._subtitle.text().split(" (")[0])
            for r in widget._outer._rows]


def layout_order(widget: WorkflowProgressWidget):
    """The order the rows actually sit in the layout.

    `_rows` is bookkeeping; this is what the user sees. They only agree if
    _relayout() really did put the widgets back in the new order.
    """
    layout = widget._outer._contents_layout
    items = (layout.itemAt(i) for i in range(layout.count()))
    return [i.widget() for i in items if i.widget() is not None]


def start_next(widget: WorkflowProgressWidget, queue: TaskQueue):
    """Consume the next queue item and push the resulting InProgress status."""
    item = queue.next()
    widget.update_from_status(status_for(queue, item, Status.InProgress))
    return item


def finish(widget: WorkflowProgressWidget, queue: TaskQueue, item,
           status=Status.Completed, **extra):
    queue.mark_done(item, status)
    widget.update_from_status(status_for(queue, item, status, **extra))


# ── First paint ───────────────────────────────────────────────────────────────

def test_first_status_builds_the_timeline(widget, queue):
    """Regression: this used to need a separate set_workflow() call first."""
    assert widget._outer._rows == []
    start_next(widget, queue)
    assert len(widget._outer._rows) == 4
    assert widget._outer_index == 0


def test_a_second_run_rebuilds_from_scratch(widget, queue):
    """The build-once flag used to be what reset this between runs."""
    item = start_next(widget, queue)
    finish(widget, queue, item)
    first_row = widget._outer._rows[0]

    queue.build_from_matrix(["Polish"], ["L9"])
    start_next(widget, queue)

    assert labels(widget) == [("L9", "Polish")]
    assert widget._outer_index == 0
    assert widget._outer._rows[0] is not first_row


# ── Add ───────────────────────────────────────────────────────────────────────

def test_item_added_mid_run_appears(widget, queue):
    start_next(widget, queue)
    queue.add("L9", "Polish")
    widget.refresh_queue(queue.items)

    assert labels(widget)[-1] == ("L9", "Polish")
    assert len(widget._outer._rows) == 5
    assert layout_order(widget) == widget._outer._rows


def test_item_added_at_the_front_appears_directly_after_the_active_row(widget, queue):
    """front=True means "run next", which is the row below the running one."""
    start_next(widget, queue)
    queue.add("L9", "Polish", front=True)
    widget.refresh_queue(queue.items)

    assert labels(widget)[1] == ("L9", "Polish")
    assert widget._outer_index == 0  # the active row did not move


# ── Remove ────────────────────────────────────────────────────────────────────

def test_removed_item_disappears_from_the_timeline(widget, queue):
    start_next(widget, queue)
    before = labels(widget)
    target = queue.pending[-1]
    queue.remove(target.id)
    widget.refresh_queue(queue.items)

    assert len(widget._outer._rows) == 3
    assert (target.lamella_name, target.task_name) not in labels(widget)
    assert labels(widget) == before[:-1]
    assert layout_order(widget) == widget._outer._rows


def test_a_removed_row_stops_painting_immediately(widget, queue):
    """deleteLater() does not run until the event loop comes back round, so a
    row that is merely dropped from the layout keeps its last geometry and
    paints over whatever moved up into its place — a visible ghost row."""
    start_next(widget, queue)
    ghost = widget._outer._rows[-1]
    queue.remove(queue.pending[-1].id)
    widget.refresh_queue(queue.items)

    assert ghost not in widget._outer._rows
    assert ghost.isHidden()
    assert ghost.parent() is None


def test_a_cancelled_task_counts_as_resolved(widget, queue):
    """Found on the microscope: the counter sat at 0/4 with a task cancelled, so it
    could never reach the total. Cancelled is terminal — it just is not success."""
    item = start_next(widget, queue)
    finish(widget, queue, item, status=Status.Cancelled)

    assert widget._header.text() == "Workflow — 1/4 (1 cancelled)"


def test_failures_and_cancellations_are_both_named(widget, queue):
    finish(widget, queue, start_next(widget, queue), status=Status.Failed)
    finish(widget, queue, start_next(widget, queue), status=Status.Cancelled)

    assert widget._header.text() == "Workflow — 1/4 (1 failed, 1 cancelled)"


def test_a_cancelled_row_does_not_look_like_a_running_one(widget, queue):
    """They were #e0a030 and #ff9800 — the same colour to the eye."""
    from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
        _DOT_ACTIVE,
        _DOT_CANCELLED,
        StepStatus,
        _status_color,
    )

    assert _DOT_CANCELLED != _DOT_ACTIVE
    assert _status_color(StepStatus.CANCELLED) == _DOT_CANCELLED


def test_removed_item_leaves_both_sides_of_the_header_fraction(widget, queue):
    """It is not work that got done, so counting it as done would be a lie —
    and leaving it in the total would mean the counter could never finish."""
    item = start_next(widget, queue)
    finish(widget, queue, item)
    assert widget._header.text() == "Workflow — 1/4"

    queue.remove(queue.pending[-1].id)
    widget.refresh_queue(queue.items)
    assert widget._header.text() == "Workflow — 1/3"


# ── Reorder ───────────────────────────────────────────────────────────────────

def test_reorder_moves_the_rows(widget, queue):
    start_next(widget, queue)
    last = queue.pending[-1]
    queue.move_to_front(last.id)
    widget.refresh_queue(queue.items)

    assert labels(widget)[1] == (last.lamella_name, last.task_name)


def test_reorder_moves_the_row_widgets_rather_than_rebuilding_them(widget, queue):
    """The load-bearing property: a row keeps its widget, so it keeps its state."""
    start_next(widget, queue)
    rows_before = {id(r) for r in widget._outer._rows}

    last = queue.pending[-1]
    moved_row = widget._outer._rows[-1]
    queue.move_to_front(last.id)
    widget.refresh_queue(queue.items)

    assert {id(r) for r in widget._outer._rows} == rows_before
    assert widget._outer._rows[1] is moved_row
    assert layout_order(widget) == widget._outer._rows


def test_reorder_keeps_the_connector_line_on_every_row_but_the_last(widget, queue):
    start_next(widget, queue)
    queue.move_to_front(queue.pending[-1].id)
    widget.refresh_queue(queue.items)

    visible = [r._line.isVisibleTo(r) for r in widget._outer._rows]
    assert visible == [True, True, True, False]


def test_reorder_carries_the_selection_with_its_row(widget, queue):
    start_next(widget, queue)
    widget._outer._on_row_clicked(3)
    selected = widget._outer._rows[3]

    queue.move_to_front(queue.pending[-1].id)
    widget.refresh_queue(queue.items)

    assert widget._outer._rows[widget._outer._selected_index] is selected
    assert selected._selected is True


# ── The active row survives all of it ─────────────────────────────────────────

@pytest.mark.parametrize("mutate", [
    pytest.param(lambda q: q.add("L9", "Polish"), id="add"),
    pytest.param(lambda q: q.add("L9", "Polish", front=True), id="add-front"),
    pytest.param(lambda q: q.remove(q.pending[-1].id), id="remove"),
    pytest.param(lambda q: q.move_to_front(q.pending[-1].id), id="reorder"),
])
def test_active_row_keeps_its_inner_steps_and_timer(widget, queue, mutate):
    """_show_inner_at() clears the inner list, so a rebuild on every edit would
    wipe the running task's step history in front of the user."""
    start_next(widget, queue)
    widget.update_step("Setup")
    widget.update_step("Mill")
    active_row = widget._outer._rows[0]
    started_at = widget._active_start_time
    assert len(active_row._inner_rows) == 2

    mutate(queue)
    widget.refresh_queue(queue.items)

    assert widget._outer._rows[0] is active_row
    assert [r._label.text() for r in active_row._inner_rows] == ["Setup", "Mill"]
    assert active_row._inner_container.isVisibleTo(active_row)
    assert widget._elapsed_timer.isActive()
    assert widget._active_start_time == started_at
    assert widget._outer_index == 0


def test_a_reorder_does_not_restart_the_running_task(widget, queue):
    """The active row is detected by id: comparing positions would read a
    shuffled queue as a new task starting."""
    start_next(widget, queue)
    widget.update_step("Setup")

    queue.move_to_front(queue.pending[-1].id)
    widget.refresh_queue(queue.items)
    # a status update after the edit must not re-trigger the "new active" branch
    widget.update_from_status(status_for(queue, queue.active, Status.InProgress))

    assert [r._label.text() for r in widget._outer._rows[0]._inner_rows] == ["Setup"]


# ── Statuses still render ─────────────────────────────────────────────────────

def test_statuses_track_the_queue_through_a_full_run(widget, queue):
    for _ in range(4):
        finish(widget, queue, start_next(widget, queue))

    assert all(s.status is StepStatus.COMPLETED for s in widget._outer._steps)
    assert widget._header.text() == "Workflow — 4/4"


def test_failure_error_text_survives_the_next_task_starting(widget, queue):
    """refresh() used to hide the error label, so the message vanished as soon
    as the next status arrived — which is one row later, every time."""
    item = start_next(widget, queue)
    finish(widget, queue, item, status=Status.Failed, error_message="beam is off")
    failed_row = widget._outer._rows[0]
    assert failed_row._error_label.text() == "beam is off"

    start_next(widget, queue)

    assert failed_row._error_label.text() == "beam is off"
    assert failed_row._error_label.isVisibleTo(failed_row)


def test_skip_reason_lands_on_the_right_row(widget, queue):
    item = queue.next()
    queue.mark_done(item, Status.Skipped)
    widget.update_from_status(
        status_for(queue, item, Status.Skipped, skip_reason="missing_prereqs")
    )

    assert widget._outer._steps[0].subtitle == "Prerequisites not met"
    assert widget._outer._steps[1].subtitle == "Trench"  # untouched


def test_what_a_finished_row_says_is_not_wiped_by_a_later_sync(widget, queue):
    """Both halves of it: the completion time in the subtitle, the duration in the
    column beside it. Neither is rebuilt from the queue snapshot, so a sync that does
    not know either of them must leave both alone."""
    item = start_next(widget, queue)
    finish(widget, queue, item, task_duration=12.0)
    subtitle = widget._outer._steps[0].subtitle
    trailing = widget._outer._steps[0].trailing
    assert trailing == "12s"

    queue.add("L9", "Polish")
    widget.refresh_queue(queue.items)

    assert widget._outer._steps[0].subtitle == subtitle
    assert widget._outer._steps[0].trailing == trailing


# ── Through the real TaskManager ──────────────────────────────────────────────

class _ParentUI(QObject):
    """The signals TaskManager emits on, declared as real pyqtSignals.

    AutoLamellaUI itself needs a live napari viewer with docked widgets, so it
    cannot be built here — but the signals are real Qt signals with real
    connection and dispatch, not a Python object with an ``emit`` method. That
    matters: a queue edit reaching the wrong signal is what crashed the app.
    """

    workflow_update_signal = pyqtSignal(dict)
    queue_changed_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.workflow_updates = []
        self.queue_changes = []
        self.workflow_update_signal.connect(self.workflow_updates.append)
        self.queue_changed_signal.connect(self.queue_changes.append)


class _NoMicroscope:
    """Enough microscope for TaskManager's constructor and its objective retract.

    Experiment.register_metadata stamps the user and experiment ref onto it, so
    it has to be something attributes can be set on. Same stand-in as
    tests/autolamella/test_task_manager_hooks.py.
    """
    fm = None


def _experiment(tmp_path, names=("L1", "L2")) -> Experiment:
    experiment = Experiment(path=tmp_path, name="test-exp")
    for i, name in enumerate(names, start=1):
        experiment.positions.append(
            Lamella(path=experiment.path, number=i, petname=name)
        )
    return experiment


@pytest.fixture
def manager(widget, tmp_path) -> TaskManager:
    """A real TaskManager, on a real Experiment, wired to the real widget.

    The slots mirror AutoLamellaMainUI._on_workflow_update and _on_queue_changed.
    Going through the real _emit_status / notify_queue_changed is what stops the
    payload contract drifting from the widget that reads it.
    """
    experiment = _experiment(tmp_path)
    parent_ui = _ParentUI()
    parent_ui.workflow_update_signal.connect(
        lambda info: widget.update_from_status(info["status"])
    )
    parent_ui.queue_changed_signal.connect(
        lambda info: widget.refresh_queue(info["queue_items"])
    )

    m = TaskManager(microscope=_NoMicroscope(), experiment=experiment,
                    parent_ui=parent_ui)
    m.queue.build_from_matrix(
        ["Trench", "Undercut"], [p.name for p in experiment.positions]
    )
    return m


def emit(manager: TaskManager, item, status, **extra) -> None:
    lamella = manager.experiment.get_lamella_by_name(item.lamella_name)
    manager._emit_status(item=item, lamella=lamella, status=status, **extra)


def test_the_managers_own_payload_builds_the_timeline(widget, manager):
    item = manager.queue.next()
    emit(manager, item, Status.InProgress)

    assert labels(widget) == [("L1", "Trench"), ("L2", "Trench"),
                              ("L1", "Undercut"), ("L2", "Undercut")]
    assert widget._outer_index == 0
    assert widget._outer._steps[0].status is StepStatus.ACTIVE


def test_notify_queue_changed_reaches_the_timeline(widget, manager):
    item = manager.queue.next()
    emit(manager, item, Status.InProgress)
    widget.update_step("Setup")

    manager.queue.add("L9", "Polish", front=True)
    manager.notify_queue_changed()

    assert labels(widget)[1] == ("L9", "Polish")
    # and the running task is undisturbed by the edit
    assert [r._label.text() for r in widget._outer._rows[0]._inner_rows] == ["Setup"]


def test_notify_queue_changed_is_a_noop_headless(tmp_path):
    m = TaskManager(microscope=_NoMicroscope(),
                    experiment=_experiment(tmp_path, names=["L1"]), parent_ui=None)
    m.queue.build_from_matrix(["Trench"], ["L1"])
    m.notify_queue_changed()  # must not raise


def test_a_queue_edit_never_reaches_the_workflow_update_slot(manager):
    """Regression: this crashed the app on every queue action.

    workflow_update_signal also drives AutoLamellaUI.handle_workflow_update,
    which reads info["msg"] with no default — so a payload without it raised
    KeyError out of a slot, and PyQt5 aborts the process on that. Beyond the
    missing key, that handler rebuilds the interaction UI and clears
    WAITING_FOR_UI_UPDATE, none of which a queue edit should touch.
    """
    manager.queue.add("L9", "Polish")
    manager.notify_queue_changed()

    assert manager.parent_ui.workflow_updates == []
    assert len(manager.parent_ui.queue_changes) == 1


def test_task_status_still_goes_out_on_the_workflow_signal(manager):
    """The other half of the split: don't quietly reroute the lifecycle stream."""
    item = manager.queue.next()
    emit(manager, item, Status.InProgress)

    assert len(manager.parent_ui.workflow_updates) == 1
    assert manager.parent_ui.queue_changes == []


def test_every_workflow_update_carries_the_key_its_other_slot_requires(manager):
    """AutoLamellaUI.handle_workflow_update reads info["msg"] with no default, and
    PyQt5 aborts the process on an exception escaping a slot. Every payload put on
    this signal has to satisfy that, whoever emits it."""
    item = manager.queue.next()
    emit(manager, item, Status.InProgress)
    manager.queue.mark_done(item, Status.Completed)
    emit(manager, item, Status.Completed, task_duration=1.0)

    assert manager.parent_ui.workflow_updates
    assert all("msg" in info for info in manager.parent_ui.workflow_updates)
