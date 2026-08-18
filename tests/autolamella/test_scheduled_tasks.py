"""Tasks that wait for a clock time before they run (FIB-612).

The feature shipped in #108 behind a preference that was never covered by a test.
It is the one part of the workflow that puts a *blocking wait* on the worker thread,
so what matters is less that the wait happens than that it always ends: on the
scheduled time, on a stop, or immediately when the time has already passed.

Real Experiment, real Lamellas, real TaskQueue, real `_run_queue`. Task execution is
the one thing stubbed, following `test_stop_task.py` — a scheduled wait can then be
driven without a microscope.

Waits here are tenths of a second. A test that really waited for a scheduled time
would be measuring `time.sleep`.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.structures import (
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

TASKS = ["Trench", "Undercut"]


class NoMicroscope:
    fm = None


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[AutoLamellaTaskDescription(name=n, supervise=False, required=False)
                   for n in TASKS]
        )
    )
    experiment.positions.append(
        Lamella(path=Path(experiment.path) / "L1", number=1, petname="L1")
    )
    m = TaskManager(microscope=NoMicroscope(), experiment=experiment, parent_ui=None)
    m.queue.build_from_matrix(TASKS, ["L1"])
    return m


def schedule(manager: TaskManager, task_name: str, when) -> None:
    """Set a task's scheduled start time, as the task editor's Apply does."""
    for task in manager.experiment.task_protocol.workflow_config.tasks:
        if task.name == task_name:
            task.scheduled_at = when
            return
    raise AssertionError(f"no task named {task_name}")


def run_with(manager: TaskManager, on_task=None) -> List[tuple]:
    """Drive the real `_run_queue` with task execution stubbed."""
    executed = []

    def _run_single_task(task_name, lamella):
        executed.append((lamella.name, task_name))
        if on_task is not None:
            on_task(task_name, lamella)
        lamella.task_state.status = Status.Completed
        return None

    manager._run_single_task = _run_single_task
    manager._run_queue()
    return executed


def elapsed_running(manager: TaskManager, **kwargs) -> float:
    """Wall-clock seconds `_wait_until_scheduled` spent."""
    from time import monotonic

    lamella = manager.experiment.positions[0]
    start = monotonic()
    manager._wait_until_scheduled(task_name="Trench", lamella=lamella, **kwargs)
    return monotonic() - start


# ── the wait always ends ──────────────────────────────────────────────────────

def test_a_time_that_has_passed_does_not_wait(manager):
    """The common case after this feature went on by default: a protocol carrying a
    stale `scheduled_at` from a previous run must not strand the queue."""
    past = datetime.now() - timedelta(days=1)

    assert elapsed_running(manager, scheduled_at=past) < 0.5


def test_a_time_that_is_now_does_not_wait(manager):
    """The boundary. `remaining <= 0` breaks, so exactly-now is a past time."""
    assert elapsed_running(manager, scheduled_at=datetime.now()) < 0.5


def test_a_future_time_is_actually_waited_for(manager):
    """The other direction — otherwise every test here would pass on a no-op."""
    soon = datetime.now() + timedelta(seconds=0.3)

    assert elapsed_running(manager, scheduled_at=soon) >= 0.25


def test_a_stop_ends_the_wait(manager):
    """A user who hits Stop during a long wait cannot be made to sit through it.

    The item is already InProgress by this point, so it is the row on screen and the
    stop has to reach it — otherwise the click sits unanswered until the scheduled
    time arrives, which may be tomorrow.
    """
    far_off = datetime.now() + timedelta(hours=6)
    Thread(target=manager.stop, daemon=True).start()

    assert elapsed_running(manager, scheduled_at=far_off) < 3.0


def test_a_task_stop_ends_the_wait_too(manager):
    """The narrower stop reaches it as well: `should_abort` covers both events, so
    abandoning one scheduled task does not require ending the run."""
    far_off = datetime.now() + timedelta(hours=6)
    Thread(target=manager.stop_task, daemon=True).start()

    assert elapsed_running(manager, scheduled_at=far_off) < 3.0


def test_a_timezone_aware_time_does_not_raise(manager):
    """Times are naive-local throughout, but a hand-edited or externally-produced
    protocol may carry an offset. Subtracting that from `datetime.now()` is a
    TypeError, so it is normalised first — and a past one still returns at once."""
    past_utc = datetime.now(timezone.utc) - timedelta(days=1)

    assert elapsed_running(manager, scheduled_at=past_utc) < 0.5


# ── through the queue ─────────────────────────────────────────────────────────

def test_an_unscheduled_task_runs_straight_away(manager):
    """No schedule is the default, and must stay the cheap path."""
    from time import monotonic

    start = monotonic()
    executed = run_with(manager)

    assert executed == [("L1", "Trench"), ("L1", "Undercut")]
    assert monotonic() - start < 0.5


def test_a_scheduled_task_waits_before_it_runs(manager):
    """The feature, end to end: the queue reaches the item, waits, then runs it."""
    from time import monotonic

    schedule(manager, "Trench", datetime.now() + timedelta(seconds=0.3))

    start = monotonic()
    executed = run_with(manager)

    assert executed == [("L1", "Trench"), ("L1", "Undercut")]
    assert monotonic() - start >= 0.25


def test_a_stale_schedule_does_not_hold_up_the_queue(manager):
    """A `scheduled_at` in the past is not an error — it is a protocol that has been
    run before, or one written for this morning and started this afternoon."""
    from time import monotonic

    schedule(manager, "Trench", datetime.now() - timedelta(days=1))

    start = monotonic()
    executed = run_with(manager)

    assert executed == [("L1", "Trench"), ("L1", "Undercut")]
    assert monotonic() - start < 0.5


def test_stopping_during_a_wait_ends_the_run(manager):
    """Stop during the wait is Stop, not "skip the wait and run it anyway"."""
    schedule(manager, "Trench", datetime.now() + timedelta(hours=6))
    Thread(target=manager.stop, daemon=True).start()

    executed = run_with(manager)

    assert executed == []
    assert manager.is_stopped


# ── the schedule survives the protocol file ───────────────────────────────────

def test_a_schedule_round_trips_through_the_protocol():
    """It is stored in the task protocol yaml, so it has to serialise. `asdict` would
    otherwise hand yaml a datetime, which is why `to_dict` writes an isoformat."""
    when = datetime(2026, 8, 13, 9, 30)
    task = AutoLamellaTaskDescription(
        name="Trench", supervise=False, required=True, scheduled_at=when
    )

    restored = AutoLamellaTaskDescription.from_dict(task.to_dict())

    assert task.to_dict()["scheduled_at"] == "2026-08-13T09:30:00"
    assert restored.scheduled_at == when


def test_no_schedule_round_trips_as_none():
    """The default has to survive too — `from_dict` only parses strings, so a None
    must not become the string "None" on the way out."""
    task = AutoLamellaTaskDescription(name="Trench", supervise=False, required=True)

    assert task.to_dict()["scheduled_at"] is None
    assert AutoLamellaTaskDescription.from_dict(task.to_dict()).scheduled_at is None


def test_the_workflow_config_finds_a_task_schedule():
    """`get_scheduled_at` is what the manager asks; a miss is None, not an error."""
    when = datetime(2026, 8, 13, 9, 30)
    config = AutoLamellaWorkflowConfig(
        tasks=[
            AutoLamellaTaskDescription(name="Trench", supervise=False, required=True,
                                       scheduled_at=when),
            AutoLamellaTaskDescription(name="Undercut", supervise=False, required=True),
        ]
    )

    assert config.get_scheduled_at("Trench") == when
    assert config.get_scheduled_at("Undercut") is None
    assert config.get_scheduled_at("Polish") is None


# ── the border can tell a parked run from a running one (FIB-676) ─────────────

class _Signal:
    """The one method `update_status_ui` calls on `workflow_update_signal`."""

    def __init__(self, on_emit):
        self._on_emit = on_emit

    def emit(self, info: dict) -> None:
        self._on_emit(info)


class RecordingUI:
    """Stands in for AutoLamellaUI at the only boundary the wait touches.

    Samples WORKFLOW_PENDING at each status emit — i.e. exactly where the real UI
    reads it, since the border block runs at the end of `_on_workflow_update`. That
    makes these tests about what the border can *see*, not merely about an attribute
    being written and rewritten around the call.
    """

    def __init__(self, manager: TaskManager):
        self._task_manager = manager
        self.WORKFLOW_PENDING = False
        self.pending_at_each_emit: List[bool] = []
        self.workflow_update_signal = _Signal(
            lambda info: self.pending_at_each_emit.append(self.WORKFLOW_PENDING)
        )


@pytest.fixture
def ui(manager: TaskManager) -> RecordingUI:
    recording = RecordingUI(manager)
    manager.parent_ui = recording
    return recording


def test_the_border_sees_pending_while_the_wait_is_happening(manager, ui):
    """The whole point: a run parked until 2am must not keep showing the running
    colour. The countdown emits every 15s and immediately on entry, so the flag has
    to be true by the first of them — not merely set somewhere around the wait."""
    elapsed_running(manager, scheduled_at=datetime.now() + timedelta(seconds=0.3))

    assert ui.pending_at_each_emit == [True]
    assert ui.WORKFLOW_PENDING is False


def test_pending_clears_once_the_scheduled_time_arrives(manager, ui):
    """Cleared before the task's own InProgress status goes out, so the border never
    shows grey over a task that has started."""
    manager._wait_until_scheduled(
        scheduled_at=datetime.now() + timedelta(seconds=0.2),
        task_name="Trench",
        lamella=manager.experiment.positions[0],
    )

    assert ui.WORKFLOW_PENDING is False


def test_pending_clears_when_a_stop_ends_the_wait(manager, ui):
    """A stop leaves the queue; the flag must not outlive the wait it belongs to."""
    Thread(target=manager.stop, daemon=True).start()

    elapsed_running(manager, scheduled_at=datetime.now() + timedelta(hours=6))

    assert ui.WORKFLOW_PENDING is False


def test_pending_clears_if_the_wait_raises(manager, ui, monkeypatch):
    """Why the clear is in a `finally`. A stranded flag would leave the border
    claiming nothing is running for the remainder of the run."""
    import fibsem.applications.autolamella.workflows.tasks.manager as manager_module

    def boom(_seconds):
        raise RuntimeError("something blew up mid-wait")

    monkeypatch.setattr(manager_module.time, "sleep", boom)

    with pytest.raises(RuntimeError):
        elapsed_running(manager, scheduled_at=datetime.now() + timedelta(hours=6))

    assert ui.WORKFLOW_PENDING is False


def test_an_unscheduled_task_never_sets_pending(manager, ui):
    """The common path stays untouched: no schedule, no flag, no grey flash.

    The queue emits per-task status through the same signal, so the recorder is
    demonstrably live here — every one of those emits just sees the flag false.
    """
    run_with(manager)

    assert ui.pending_at_each_emit, "expected the queue to emit at all"
    assert not any(ui.pending_at_each_emit)
    assert ui.WORKFLOW_PENDING is False
