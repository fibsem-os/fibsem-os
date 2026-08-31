"""Tests for TaskManager status emission and skip logic.

Both used to be anchored to the task x lamella lists frozen at launch, which
broke as soon as the queue could be mutated: an added task raised ValueError
out of _emit_status, and an added lamella was silently skipped as
"not_required". Progress is now measured against the live queue instead.

These drive _emit_status/_should_skip against a real Experiment, real Lamellas
and a real task protocol. The only stand-ins are the two the environment cannot
provide: a microscope and the Qt UI.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaWorkflowConfig,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager


class RecordingUI:
    """The one stand-in here: a recorder in place of AutoLamellaUI.

    AutoLamellaUI needs a live napari viewer with docked widgets, so it cannot be
    built in a test. Everything else below is the real object. Deliberately not a
    QObject with real pyqtSignals — this module has no Qt dependency, and adding
    one would take it out of CI, where PyQt5 is not installed.
    """

    def __init__(self):
        # update_status_ui still uses the dict signal; _emit_status now reports on
        # workflow_status_signal. Both record here, into per-channel lists.
        self.workflow_update_signal = _Recorder()
        self.workflow_status_signal = _Recorder()
        self._task_manager = None  # _check_for_abort reads this


class _Recorder:
    def __init__(self):
        self.emitted: List = []

    def emit(self, payload) -> None:
        self.emitted.append(payload)


class NoMicroscope:
    """Enough microscope to construct a TaskManager and retract the objective.

    Experiment.register_metadata stamps the user and experiment ref onto it, so
    it has to be something attributes can be set on — None will not do.
    """

    fm = None


def make_lamella(
    experiment: Experiment, name: str, is_failure: bool = False, completed=()
) -> Lamella:
    lamella = Lamella(
        path=Path(experiment.path) / name,
        number=len(experiment.positions) + 1,
        petname=name,
    )
    if is_failure:
        lamella.defect.state = DefectType.FAILURE
    for task_name in completed:
        lamella.task_history.append(
            AutoLamellaTaskState(name=task_name, status=Status.Completed)
        )
    experiment.positions.append(lamella)
    return lamella


def make_experiment(
    tmp_path: Path,
    requirements: Optional[Dict[str, List[str]]] = None,
    lamella_names=("L1", "L2"),
) -> Experiment:
    """A real Experiment with a real task protocol.

    ``required=False`` throughout so nothing is ever "complete" — the completion
    hooks are a different test module's subject and would only add noise here.
    """
    reqs = requirements or {}
    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=name,
                    supervise=False,
                    required=False,
                    requires=reqs.get(name, []),
                )
                for name in ("Trench", "Undercut", "Polishing")
            ]
        )
    )
    for name in lamella_names:
        make_lamella(experiment, name)
    return experiment


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

    manager.parent_ui._task_manager = manager
    manager._run_single_task = _run_single_task
    manager._run_queue()
    return executed


@pytest.fixture
def manager(tmp_path) -> TaskManager:
    m = TaskManager(
        microscope=NoMicroscope(),
        experiment=make_experiment(tmp_path),
        parent_ui=RecordingUI(),
    )
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return m


def last_status(manager: TaskManager):
    return manager.parent_ui.workflow_status_signal.emitted[-1].report


# ── _emit_status ──────────────────────────────────────────────────────────────


def test_emit_reports_position_in_the_live_queue(manager):
    item = manager.queue.items[2]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name(item.lamella_name),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.queue_position == 3
    assert status.queue_total == 4


def test_emit_for_a_task_outside_the_launch_plan(manager):
    """Regression: this raised ValueError out of task_names.index()."""
    added = manager.queue.add("L1", "Polishing")
    assert added.task_name not in manager.queue.task_names

    manager._emit_status(
        item=added,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.task_name == "Polishing"
    assert status.queue_position == 5
    assert status.queue_total == 5


def test_emit_for_a_lamella_outside_the_launch_plan(manager):
    added = manager.queue.add("L99", "Trench")
    manager._emit_status(
        item=added,
        lamella=make_lamella(manager.experiment, "L99"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.item_name == "L99"
    # deprecated alias, dropped with the HookContext shims after v0.6 (FIB-464)
    assert status.lamella_name == "L99"
    assert status.queue_position == 5


def test_emit_position_follows_a_reorder(manager):
    item = manager.queue.items[3]
    manager.queue.move_to_front(item.id)
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name(item.lamella_name),
        status=Status.InProgress,
    )
    assert last_status(manager).queue_position == 1


def test_emit_carries_the_launch_plan_as_context(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.task_names == ["Trench", "Undercut"]
    assert status.lamella_names == ["L1", "L2"]


def test_emit_includes_a_queue_snapshot(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    snapshot = last_status(manager).queue_items
    assert len(snapshot) == 4
    snapshot[0].status = Status.Failed
    assert manager.queue.items[0].status is not Status.Failed


def test_emit_passes_through_error_and_skip_detail(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.Failed,
        msg="boom",
        error_message="it broke",
        task_duration=12.5,
    )
    status = last_status(manager)
    assert status.error_message == "it broke"
    assert status.task_duration == 12.5
    assert manager.parent_ui.workflow_status_signal.emitted[-1].message == "boom"


def test_emit_is_a_noop_headless(tmp_path):
    experiment = make_experiment(tmp_path, lamella_names=["L1"])
    m = TaskManager(microscope=NoMicroscope(), experiment=experiment, parent_ui=None)
    m.queue.build_from_matrix(["Trench"], ["L1"])
    m._emit_status(
        item=m.queue.items[0],
        lamella=experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )  # must not raise


# ── _should_skip ──────────────────────────────────────────────────────────────


def test_lamella_outside_the_launch_selection_is_not_skipped(manager):
    """Regression: the old allow-list check skipped anything added mid-run."""
    lamella = make_lamella(manager.experiment, "L99")
    assert "L99" not in manager.queue.lamella_names
    assert manager._should_skip(lamella, "Trench") is None


def test_failed_lamella_is_skipped(manager):
    lamella = manager.experiment.get_lamella_by_name("L1")
    lamella.defect.state = DefectType.FAILURE
    assert manager._should_skip(lamella, "Trench") == "failure"


def test_missing_prerequisites_are_skipped(tmp_path):
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1"])
    lamella = experiment.get_lamella_by_name("L1")
    assert m._should_skip(lamella, "Undercut") == "missing_prereqs"

    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    assert m._should_skip(lamella, "Undercut") is None


def test_no_requirements_runs(manager):
    assert (
        manager._should_skip(manager.experiment.get_lamella_by_name("L1"), "Trench")
        is None
    )


# ── _run_queue: mid-run queue edits ───────────────────────────────────────────


def test_baseline_run_executes_the_whole_matrix(manager):
    assert run_queue_with(manager) == [
        ("L1", "Trench"),
        ("L2", "Trench"),
        ("L1", "Undercut"),
        ("L2", "Undercut"),
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
    """An out-of-plan lamella used to be silently skipped as not_required.

    It still has to exist in the experiment: queueing work for a name the
    experiment has never heard of is skipped with a warning, which a fake
    experiment that conjured lamellas on lookup used to hide.
    """
    make_lamella(manager.experiment, "L99")
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
    for event in manager.parent_ui.workflow_status_signal.emitted:
        status = event.report
        if status is None:
            continue
        assert status.queue_position is not None
        assert 1 <= status.queue_position <= status.queue_total
