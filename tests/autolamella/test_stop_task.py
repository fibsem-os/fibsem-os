"""Stopping one task without ending the run (FIB-499).

Both stops unwind the task the same way and both end as Cancelled — it is a user
abort either way. What differs is scope, which lives on the manager: `is_stopped`
answers "should the run end?" and only _run_queue's loop asks it, while
`should_abort` / `abort_token` answer "should this task unwind?" and are what the
task and its long-running operations poll.

Real Experiment, real Lamellas, real TaskQueue, real _run_queue. Task execution is
the one thing stubbed, so a stop can be raised from inside a running task without
a microscope.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.cancellation import AnyStopEvent, OperationCancelledError
from fibsem.workflows.status import TaskStatus as Status
from tests.protocols import planned, task_protocol

TASKS = ["Trench", "Undercut"]


class NoMicroscope:
    fm = None
    milling = None
    spot_burn = None


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.task_protocol = task_protocol(
        *[planned(name=n, required=False) for n in TASKS]
    )
    for i, name in enumerate(("L1", "L2"), start=1):
        experiment.positions.append(
            Lamella(path=Path(experiment.path) / name, number=i, petname=name)
        )
    m = TaskManager(microscope=NoMicroscope(), experiment=experiment)
    m.queue.build_from_matrix(TASKS, ["L1", "L2"])
    return m


def run_with(manager: TaskManager, on_task=None) -> List[tuple]:
    """Drive the real _run_queue with task execution stubbed.

    ``on_task(task_name, lamella)`` stands in for the task body and may raise, as a
    cancelled task does once it polls the abort token.
    """
    executed = []

    def _run_single_task(task_name, lamella):
        executed.append((lamella.name, task_name))
        try:
            if on_task is not None:
                on_task(task_name, lamella)
        except Exception as exc:
            # Mirrors the real _run_single_task's classification.
            if manager.should_abort or isinstance(
                exc, (OperationCancelledError, InterruptedError)
            ):
                lamella.task_state.status = Status.Cancelled
            else:
                lamella.task_state.status = Status.Failed
            return exc
        lamella.task_state.status = Status.Completed
        return None

    manager._run_single_task = _run_single_task
    manager._run_queue()
    return executed


def statuses(manager: TaskManager) -> List[Status]:
    return [i.status for i in manager.queue.items]


# ── Scope: the whole point ────────────────────────────────────────────────────


def test_stopping_a_task_leaves_the_run_going(manager):
    """The item is Cancelled and the queue carries on to the next one."""

    def on_task(task_name, lamella):
        if (lamella.name, task_name) == ("L1", "Trench"):
            manager.stop_task()
            raise OperationCancelledError("cancelled")

    executed = run_with(manager, on_task)

    assert len(executed) == 4  # every item was still attempted
    assert statuses(manager)[0] is Status.Cancelled
    assert statuses(manager)[1:] == [Status.Completed] * 3


def test_stopping_the_run_ends_it(manager):
    """The contrast: everything after the current item stays untouched."""

    def on_task(task_name, lamella):
        if (lamella.name, task_name) == ("L1", "Trench"):
            manager.stop()
            raise OperationCancelledError("cancelled")

    executed = run_with(manager, on_task)

    assert len(executed) == 1
    assert statuses(manager)[0] is Status.Cancelled
    assert statuses(manager)[1:] == [Status.NotStarted] * 3


def test_stopping_a_task_never_sets_is_stopped(manager):
    """There is no path from a task cancel to ending the run — the reason these
    are two events rather than one event and a scope flag."""
    manager.stop_task()

    assert manager.is_stopped is False
    assert manager.should_abort is True


def test_a_run_stop_wins_when_both_are_set(manager):
    def on_task(task_name, lamella):
        manager.stop_task()
        manager.stop()
        raise OperationCancelledError("cancelled")

    executed = run_with(manager, on_task)

    assert len(executed) == 1
    assert manager.is_stopped is True


# ── Clearing between tasks ────────────────────────────────────────────────────


def test_the_task_stop_does_not_leak_into_the_next_task(manager):
    """Cleared at the top of each item, so one cancel cancels one task."""

    def on_task(task_name, lamella):
        if (lamella.name, task_name) == ("L1", "Trench"):
            manager.stop_task()
            raise OperationCancelledError("cancelled")
        assert manager.should_abort is False, "the cancel leaked into a later task"

    run_with(manager, on_task)

    assert statuses(manager).count(Status.Cancelled) == 1


def test_a_click_between_tasks_is_discarded(manager):
    """It was aimed at the task that has just finished, not the next one."""

    def on_task(task_name, lamella):
        if (lamella.name, task_name) == ("L1", "Trench"):
            manager.stop_task()  # set, but nothing raises: the task completes

    executed = run_with(manager, on_task)

    assert len(executed) == 4
    assert statuses(manager) == [Status.Completed] * 4


# ── The token everything inside a task polls ──────────────────────────────────


def test_the_token_reports_either_stop(manager):
    token = manager.abort_token
    assert token.is_set() is False

    manager.stop_task()
    assert token.is_set() is True

    manager._task_stop_event.clear()
    assert token.is_set() is False

    manager.stop()
    assert token.is_set() is True


def test_the_token_identity_is_stable(manager):
    """Tasks capture it once at construction, so it cannot be rebuilt per access."""
    assert manager.abort_token is manager.abort_token


def test_a_composite_cannot_be_set_directly():
    """Setting it would have to guess which underlying signal was meant."""
    assert not hasattr(AnyStopEvent(), "set")


# ── Stopping the hardware, not just the queue ─────────────────────────────────


class Milling:
    """The milling service's stop, counted."""

    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


@pytest.mark.parametrize("stop", ["stop", "stop_task"])
def test_both_stops_stop_the_mill_through_the_milling_service(manager, stop):
    """A task mills on the run's thread under the abort token, which the
    milling service polls; both stops also stop the beam now through the
    service's stop, with or without a window."""
    manager.microscope.milling = Milling()

    getattr(manager, stop)()

    assert manager.should_abort
    assert manager.microscope.milling.stops == 1


def test_a_stop_with_nothing_milling_is_not_an_error(manager):
    from fibsem.cancellation import NothingToStop

    class Idle:
        def stop(self):
            raise NothingToStop("Nothing is milling.")

    manager.microscope.milling = Idle()
    manager.stop_task()
    manager.stop()
    assert manager.is_stopped
