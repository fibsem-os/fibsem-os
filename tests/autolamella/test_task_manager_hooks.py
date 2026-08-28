"""Which lifecycle hooks TaskManager fires, and how it reports the outcome.

These exercise `_run_queue` without running a real task: the queue either drains
immediately, is stopped before it starts, or contains only items that get skipped.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookContext, HookEvent, HookManager

ALL_EVENTS = list(HookEvent)


@pytest.fixture
def recorder() -> "tuple[HookManager, List[HookContext]]":
    """A HookManager that records every event fired, in order."""
    fired: List[HookContext] = []
    manager = HookManager()
    manager.register(FunctionHook(name="recorder", events=ALL_EVENTS, callback=fired.append))
    return manager, fired


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    return Experiment(path=tmp_path, name="test-exp")


def _manager(experiment: Experiment, hook_manager: HookManager) -> TaskManager:
    # The microscope is only touched to retract the FM objective after the queue
    # drains; None would raise, so a stand-in with fm=None is enough.
    class _NoMicroscope:
        fm = None

    return TaskManager(
        microscope=_NoMicroscope(),
        experiment=experiment,
        parent_ui=None,
        hook_manager=hook_manager,
    )


def _events(fired: List[HookContext]) -> List[str]:
    return [c.event for c in fired]


# ---------------------------------------------------------------------------
# workflow_completed vs workflow_cancelled
# ---------------------------------------------------------------------------

def test_draining_the_queue_fires_workflow_completed(experiment, recorder):
    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=[])

    assert _events(fired) == ["workflow_started", "workflow_completed"]


def test_stopping_fires_workflow_cancelled_not_completed(experiment, recorder):
    """Pressing Stop used to fire workflow_completed, so a subscriber could not tell an
    aborted run from a finished one."""
    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)
    manager.stop()

    manager.run(task_names=["MillTrench"], required_lamella=[])

    assert _events(fired) == ["workflow_started", "workflow_cancelled"]
    assert "workflow_completed" not in _events(fired)


def test_stopping_midway_still_cancels(experiment, recorder):
    """Stop set while items remain: the queue does not drain, so this is a cancel."""
    hook_manager, fired = recorder
    experiment.positions.append(Lamella(path=experiment.path, number=1, petname="lam-1"))
    manager = _manager(experiment, hook_manager)
    manager.stop()

    manager.run(task_names=["MillTrench"], required_lamella=["lam-1"])

    assert _events(fired)[-1] == "workflow_cancelled"


# ---------------------------------------------------------------------------
# task_skipped
# ---------------------------------------------------------------------------

def test_missing_lamella_fires_task_skipped(experiment, recorder, caplog):
    """A queue item naming a lamella that is not in the experiment used to be marked
    Skipped and dropped silently — no status, no hook, no log."""
    import logging

    hook_manager, fired = recorder
    manager = _manager(experiment, hook_manager)

    with caplog.at_level(logging.WARNING):
        manager.run(task_names=["MillTrench"], required_lamella=["ghost-lamella"])

    skipped = [c for c in fired if c.event == "task_skipped"]
    assert len(skipped) == 1
    assert skipped[0].item_name == "ghost-lamella"
    assert skipped[0].task_name == "MillTrench"
    assert skipped[0].skip_reason == "lamella_not_found"
    assert any("ghost-lamella" in r.message for r in caplog.records)


def test_skipped_task_carries_its_reason(experiment, recorder):
    """A lamella marked as a failure is skipped by _should_skip, and the reason reaches
    the hook rather than only the UI status dict."""
    hook_manager, fired = recorder
    lamella = Lamella(path=experiment.path, number=1, petname="lam-1")
    lamella.defect.state = DefectType.FAILURE
    experiment.positions.append(lamella)
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=[lamella.name])

    skipped = [c for c in fired if c.event == "task_skipped"]
    assert len(skipped) == 1
    assert skipped[0].item_name == lamella.name
    assert skipped[0].skip_reason == "failure"
    # a skip still leaves a complete workflow: nothing was aborted
    assert _events(fired)[-1] == "workflow_completed"


def test_skip_fires_no_task_started_or_completed(experiment, recorder):
    """A skipped task never ran, so it must not look like one that did."""
    hook_manager, fired = recorder
    lamella = Lamella(path=experiment.path, number=1, petname="lam-1")
    lamella.defect.state = DefectType.FAILURE
    experiment.positions.append(lamella)
    manager = _manager(experiment, hook_manager)

    manager.run(task_names=["MillTrench"], required_lamella=[lamella.name])

    assert "task_started" not in _events(fired)
    assert "task_completed" not in _events(fired)


# ---------------------------------------------------------------------------
# closing status message
# ---------------------------------------------------------------------------

def _status_messages(monkeypatch) -> List[str]:
    """Capture the workflow_info strings the manager closes a run with."""
    captured: List[str] = []
    monkeypatch.setattr(
        "fibsem.applications.autolamella.workflows.tasks.manager.update_status_ui",
        lambda parent_ui, msg, workflow_info=None, **kw: captured.append(workflow_info),
    )
    return captured


def test_a_run_that_skipped_everything_does_not_claim_completion(
    experiment, recorder, monkeypatch, caplog
):
    """Every task skipped means no work happened. Reporting "All tasks completed." is
    the same misreport as an abort claiming completion, just quieter."""
    import logging

    messages = _status_messages(monkeypatch)
    hook_manager, _ = recorder
    lamella = Lamella(path=experiment.path, number=1, petname="lam-1")
    lamella.defect.state = DefectType.FAILURE
    experiment.positions.append(lamella)
    manager = _manager(experiment, hook_manager)

    with caplog.at_level(logging.WARNING):
        manager.run(task_names=["MillTrench"], required_lamella=[lamella.name])

    assert messages[-1] == "No tasks ran: all 1 skipped."
    # update_status_ui does not log workflow_info, so headless needs its own warning
    assert any("all 1 were skipped" in r.message for r in caplog.records)


def test_completion_message_counts_partial_skips(experiment, recorder):
    hook_manager, _ = recorder
    manager = _manager(experiment, hook_manager)
    items = manager.queue.build_from_matrix(["MillTrench"], ["lam-1", "lam-2"])
    manager.queue.mark_done(items[0], AutoLamellaTaskStatus.Skipped)
    manager.queue.mark_done(items[1], AutoLamellaTaskStatus.Completed)

    assert manager._completion_message() == "All tasks completed (1 skipped)."


def test_completion_message_unchanged_when_nothing_skipped(experiment, recorder):
    hook_manager, _ = recorder
    manager = _manager(experiment, hook_manager)
    items = manager.queue.build_from_matrix(["MillTrench"], ["lam-1"])
    manager.queue.mark_done(items[0], AutoLamellaTaskStatus.Completed)

    assert manager._completion_message() == "All tasks completed."


def test_cancelled_run_keeps_its_own_message(experiment, recorder, monkeypatch):
    """The cancelled path must not fall through to a completion message."""
    messages = _status_messages(monkeypatch)
    hook_manager, _ = recorder
    manager = _manager(experiment, hook_manager)
    manager.stop()

    manager.run(task_names=["MillTrench"], required_lamella=[])

    assert messages[-1] == "Workflow cancelled."
