"""The manager fills in which experiment an event belongs to, and how much is left.

Without these a consumer handling more than one run cannot tell events apart, and a
failure reads as a dead workflow when the queue is still moving. See FIB-489.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookContext, HookEvent, HookManager


@pytest.fixture
def recorder() -> "tuple[HookManager, List[HookContext]]":
    fired: List[HookContext] = []
    manager = HookManager()
    manager.register(
        FunctionHook(name="recorder", events=list(HookEvent), callback=fired.append)
    )
    return manager, fired


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


def _manager(experiment: Experiment, hook_manager: HookManager) -> TaskManager:
    class _NoMicroscope:
        fm = None

    return TaskManager(
        microscope=_NoMicroscope(),
        experiment=experiment,
        parent_ui=None,
        hook_manager=hook_manager,
    )


def test_workflow_events_say_which_experiment(experiment, recorder):
    """They used to carry nothing at all, so workflow_completed reached a webhook with
    no way to tell which experiment had finished."""
    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=[])

    assert [c.event for c in fired] == ["workflow_started", "workflow_completed"]
    for context in fired:
        assert context.experiment_id == experiment._id
        assert context.experiment_name == "test-exp"


def test_progress_counts_down_as_the_queue_drains(experiment, recorder):
    """Three lamellae, all skipped, so the queue drains without running anything."""
    hook_manager, fired = recorder
    for i in (1, 2, 3):
        lamella = Lamella(path=experiment.path, number=i, petname=f"lam-{i}")
        lamella.defect.state = DefectType.FAILURE  # skipped by _should_skip
        experiment.positions.append(lamella)
    manager = _manager(experiment, hook_manager)

    manager.run(
        task_names=["MillTrench"],
        required_lamella=[p.name for p in experiment.positions],
    )

    skipped = [c for c in fired if c.event == "task_skipped"]
    assert [c.tasks_remaining for c in skipped] == [2, 1, 0]
    assert {c.tasks_total for c in skipped} == {3}


def test_a_skipped_event_carries_the_lamella_id(experiment, recorder):
    hook_manager, fired = recorder
    lamella = Lamella(path=experiment.path, number=1, petname="lam-1")
    lamella.defect.state = DefectType.FAILURE
    experiment.positions.append(lamella)
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=[lamella.name])

    skipped = [c for c in fired if c.event == "task_skipped"]
    assert skipped[0].item_id == lamella._id


def test_a_missing_lamella_has_no_id_to_report(experiment, recorder):
    """The one path with no lamella in hand: the name is all there is."""
    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=["ghost"])

    skipped = [c for c in fired if c.event == "task_skipped"]
    assert skipped[0].item_name == "ghost"
    assert skipped[0].item_id == ""
    # still says which experiment, even with nothing else to go on
    assert skipped[0].experiment_id == experiment._id


def test_the_event_task_id_joins_to_the_task_history_entry(recorder, tmp_path):
    """The point of carrying task_id: a consumer that received "MillTrench failed" can
    find the exact history entry it referred to, instead of guessing by name and time.

    task_id identifies this *execution*, so two runs of the same task on the same
    lamella share every name field but not this.
    """
    from fibsem import utils
    from fibsem.applications.autolamella.workflows.tasks.trench import (
        MillTrenchTask,
        MillTrenchTaskConfig,
    )

    hook_manager, fired = recorder
    microscope, _ = utils.setup_session(manufacturer="Demo")
    lamella = Lamella(path=tmp_path / "lam", number=1, petname="lam-1")
    task = MillTrenchTask(
        microscope=microscope, config=MillTrenchTaskConfig(), lamella=lamella
    )
    task.task_manager = type("_M", (), {"hook_manager": hook_manager, "is_stopped": False})()

    task._run = lambda: (_ for _ in ()).throw(RuntimeError("stage timeout"))
    with pytest.raises(RuntimeError):
        task.run()

    failed = [c for c in fired if c.event == "task_failed"][0]
    assert failed.task_id == lamella.task_history[-1].task_id
    assert failed.item_id == lamella._id
    # and the snapshot is of the terminal state, not the InProgress it started in
    assert failed.task_state.status is AutoLamellaTaskStatus.Failed


def test_hook_run_context_is_a_snapshot_not_a_live_view(experiment, recorder):
    hook_manager, _ = recorder
    manager = _manager(experiment, hook_manager)
    items = manager.queue.build_from_matrix(["MillTrench"], ["lam-1", "lam-2"])

    before = manager.hook_run_context()
    manager.queue.mark_done(items[0], AutoLamellaTaskStatus.Completed)
    after = manager.hook_run_context()

    assert before["tasks_remaining"] == 2
    assert after["tasks_remaining"] == 1
    assert before["tasks_total"] == after["tasks_total"] == 2
