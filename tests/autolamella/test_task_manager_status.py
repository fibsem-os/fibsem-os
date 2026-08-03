"""Tests for TaskManager status emission and skip logic.

Both used to be anchored to the task x lamella lists frozen at launch, which
broke as soon as the queue could be mutated: an added task raised ValueError
out of _emit_status, and an added lamella was silently skipped as
"not_required". Progress is now measured against the live queue instead.

These exercise _emit_status/_should_skip directly with stubs — no microscope,
no Qt, no experiment on disk.
"""

from types import SimpleNamespace
from typing import Optional

import pytest

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager


class FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, payload):
        self.emitted.append(payload)


class FakeUI:
    def __init__(self):
        self.workflow_update_signal = FakeSignal()
        self._task_manager = None  # _check_for_abort reads this


def fake_lamella(name: str, is_failure: bool = False, completed=()):
    return SimpleNamespace(
        name=name,
        id=f"id-{name}",
        is_failure=is_failure,
        has_completed_task=lambda task, _done=set(completed): task in _done,
        task_config={},
        task_state=SimpleNamespace(status=Status.NotStarted, status_message="",
                                   duration=0.0, task_type="", task_id=f"task-{name}"),
    )


def fake_experiment(requirements: Optional[dict] = None):
    reqs = requirements or {}
    lamellas: dict = {}

    def get_lamella_by_name(name: str):
        return lamellas.setdefault(name, fake_lamella(name))

    return SimpleNamespace(
        positions=[],
        id="exp-1",
        name="test-exp",
        save=lambda: None,
        task_history_dataframe=lambda: "",
        get_lamella_by_name=get_lamella_by_name,
        register_metadata=lambda microscope: None,
        task_protocol=SimpleNamespace(
            workflow_config=SimpleNamespace(
                requirements=lambda name: reqs.get(name, []),
                get_scheduled_at=lambda name: None,
                # no required tasks -> _is_complete is False, so the completion
                # hooks stay out of the way of what these tests are about
                required_tasks=[],
                is_completed=lambda lamella: False,
            ),
        ),
    )


def run_queue_with(manager: TaskManager, on_task=None):
    """Drive _run_queue with task execution stubbed out.

    ``on_task(task_name, lamella)`` runs in place of the real task and may
    mutate the queue — which is the mid-run editing case under test.
    """
    executed = []

    def _run_single_task(task_name, lamella):
        executed.append((lamella.name, task_name))
        lamella.task_state.status = Status.Completed
        if on_task is not None:
            on_task(task_name, lamella)
        return None

    manager.microscope = SimpleNamespace(fm=None)
    manager.parent_ui._task_manager = manager
    manager._run_single_task = _run_single_task
    manager._run_queue()
    return executed


@pytest.fixture
def manager() -> TaskManager:
    m = TaskManager(microscope=None, experiment=fake_experiment(), parent_ui=FakeUI())
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return m


def last_status(manager: TaskManager) -> dict:
    return manager.parent_ui.workflow_update_signal.emitted[-1]["status"]


# ── _emit_status ──────────────────────────────────────────────────────────────

def test_emit_reports_position_in_the_live_queue(manager):
    item = manager.queue.items[2]
    manager._emit_status(item=item, lamella=fake_lamella(item.lamella_name),
                         status=Status.InProgress)
    status = last_status(manager)
    assert status["queue_position"] == 3
    assert status["queue_total"] == 4


def test_emit_for_a_task_outside_the_launch_plan(manager):
    """Regression: this raised ValueError out of task_names.index()."""
    added = manager.queue.add("L1", "Polishing")
    assert added.task_name not in manager.queue.task_names

    manager._emit_status(item=added, lamella=fake_lamella("L1"),
                         status=Status.InProgress)
    status = last_status(manager)
    assert status["task_name"] == "Polishing"
    assert status["queue_position"] == 5
    assert status["queue_total"] == 5


def test_emit_for_a_lamella_outside_the_launch_plan(manager):
    added = manager.queue.add("L99", "Trench")
    manager._emit_status(item=added, lamella=fake_lamella("L99"),
                         status=Status.InProgress)
    status = last_status(manager)
    assert status["item_name"] == "L99"
    # deprecated alias, dropped with the HookContext shims after v0.6 (FIB-464)
    assert status["lamella_name"] == "L99"
    assert status["queue_position"] == 5


def test_emit_position_follows_a_reorder(manager):
    item = manager.queue.items[3]
    manager.queue.move_to_front(item.id)
    manager._emit_status(item=item, lamella=fake_lamella(item.lamella_name),
                         status=Status.InProgress)
    assert last_status(manager)["queue_position"] == 1


def test_emit_carries_the_launch_plan_as_context(manager):
    item = manager.queue.items[0]
    manager._emit_status(item=item, lamella=fake_lamella("L1"), status=Status.InProgress)
    status = last_status(manager)
    assert status["task_names"] == ["Trench", "Undercut"]
    assert status["lamella_names"] == ["L1", "L2"]


def test_emit_includes_a_queue_snapshot(manager):
    item = manager.queue.items[0]
    manager._emit_status(item=item, lamella=fake_lamella("L1"), status=Status.InProgress)
    snapshot = last_status(manager)["queue_items"]
    assert len(snapshot) == 4
    snapshot[0].status = Status.Failed
    assert manager.queue.items[0].status is not Status.Failed


def test_emit_passes_through_error_and_skip_detail(manager):
    item = manager.queue.items[0]
    manager._emit_status(item=item, lamella=fake_lamella("L1"), status=Status.Failed,
                         msg="boom", error_message="it broke", task_duration=12.5)
    status = last_status(manager)
    assert status["error_message"] == "it broke"
    assert status["task_duration"] == 12.5
    assert manager.parent_ui.workflow_update_signal.emitted[-1]["msg"] == "boom"


def test_emit_is_a_noop_headless():
    m = TaskManager(microscope=None, experiment=fake_experiment(), parent_ui=None)
    m.queue.build_from_matrix(["Trench"], ["L1"])
    m._emit_status(item=m.queue.items[0], lamella=fake_lamella("L1"),
                   status=Status.InProgress)  # must not raise


# ── _should_skip ──────────────────────────────────────────────────────────────

def test_lamella_outside_the_launch_selection_is_not_skipped(manager):
    """Regression: the old allow-list check skipped anything added mid-run."""
    assert "L99" not in manager.queue.lamella_names
    assert manager._should_skip(fake_lamella("L99"), "Trench") is None


def test_failed_lamella_is_skipped(manager):
    assert manager._should_skip(fake_lamella("L1", is_failure=True), "Trench") == "failure"


def test_missing_prerequisites_are_skipped():
    m = TaskManager(microscope=None,
                    experiment=fake_experiment({"Undercut": ["Trench"]}),
                    parent_ui=FakeUI())
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1"])
    assert m._should_skip(fake_lamella("L1"), "Undercut") == "missing_prereqs"
    assert m._should_skip(fake_lamella("L1", completed=["Trench"]), "Undercut") is None


def test_no_requirements_runs(manager):
    assert manager._should_skip(fake_lamella("L1"), "Trench") is None


# ── _run_queue: mid-run queue edits ───────────────────────────────────────────

def test_baseline_run_executes_the_whole_matrix(manager):
    assert run_queue_with(manager) == [
        ("L1", "Trench"), ("L2", "Trench"),
        ("L1", "Undercut"), ("L2", "Undercut"),
    ]


def test_task_added_mid_run_executes(manager):
    """The end-to-end fix: an out-of-plan task used to raise out of the loop."""
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L1", "Polishing"))

    executed = run_queue_with(manager, on_task)
    assert ("L1", "Polishing") in executed


def test_lamella_added_mid_run_executes(manager):
    """An out-of-plan lamella used to be silently skipped as not_required."""
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L99", "Trench"))

    executed = run_queue_with(manager, on_task)
    assert ("L99", "Trench") in executed
    assert all(i.status is not Status.Skipped for i in manager.queue.items)


def test_item_moved_to_front_mid_run_runs_next(manager):
    moved = []

    def on_task(task_name, lamella):
        if not moved:
            last = manager.queue.pending[-1]
            manager.queue.move_to_front(last.id)
            moved.append((last.lamella_name, last.task_name))

    executed = run_queue_with(manager, on_task)
    assert executed[1] == moved[0]


def test_item_removed_mid_run_never_executes(manager):
    removed = []

    def on_task(task_name, lamella):
        if not removed:
            target = manager.queue.pending[-1]
            manager.queue.remove(target.id)
            removed.append((target.lamella_name, target.task_name))

    executed = run_queue_with(manager, on_task)
    assert removed[0] not in executed
    assert len(executed) == 3


def test_status_bar_text_is_derivable_for_added_items(manager):
    """What AutoLamellaMainUI builds its status string from."""
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L99", "Polishing"))

    run_queue_with(manager, on_task)
    for payload in manager.parent_ui.workflow_update_signal.emitted:
        status = payload.get("status")
        if status is None:
            continue
        assert status["queue_position"] is not None
        assert 1 <= status["queue_position"] <= status["queue_total"]
