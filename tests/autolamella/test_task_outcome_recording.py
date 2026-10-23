"""Every terminal outcome is recorded, not just success.

``task_state`` is a single object the next task's ``pre_task`` overwrites, so an
outcome that never reaches ``task_history`` is gone from the experiment as soon as
that lamella starts its next task. Before FIB-490 only ``post_task`` appended, and
``run()`` re-raises before reaching it -- so failures and cancellations were absent
from the history entirely, leaving the logfile as the only record they happened.
"""

import threading
from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTask,
    MillTrenchTaskConfig,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.hooks import FunctionHook, HookContext, HookManager
from fibsem.microscope import FibsemMicroscope
from fibsem.workflows.status import TaskStatus
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope(scope="function")


def _make_task(microscope: FibsemMicroscope, tmp_path: Path) -> MillTrenchTask:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    return MillTrenchTask(
        microscope=microscope, config=MillTrenchTaskConfig(), lamella=lamella
    )


def _run_raising(task: MillTrenchTask, exc: Exception) -> None:
    """Drive task.run() with a _run that raises, swallowing the re-raise."""
    task._run = lambda: (_ for _ in ()).throw(exc)  # type: ignore[method-assign]
    with pytest.raises(type(exc)):
        task.run()


def test_a_failed_task_is_recorded_in_the_history(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    task = _make_task(microscope, tmp_path)
    _run_raising(task, RuntimeError("stage timeout"))

    assert len(task.lamella.task_history) == 1
    entry = task.lamella.task_history[0]
    assert entry.status is TaskStatus.Failed
    assert entry.status_message == "stage timeout"
    assert entry.end_timestamp is not None


def test_a_cancelled_task_is_recorded_as_cancelled_not_failed(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    task = _make_task(microscope, tmp_path)
    _run_raising(task, OperationCancelledError("stopped"))

    assert len(task.lamella.task_history) == 1
    assert task.lamella.task_history[0].status is TaskStatus.Cancelled


def test_a_stopped_task_records_who_stopped_it_and_why(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The stop's own words reach the record, where History shows them
    (FIB-1517): not "Cancelled by user." when the agent stopped it."""
    from fibsem.workflows.runner import stop_message

    task = _make_task(microscope, tmp_path)
    task.task_manager = type("Manager", (), {})()
    task.task_manager.stop_message = stop_message("agent", "The trench drifted.")
    _run_raising(task, OperationCancelledError("stopped"))

    entry = task.lamella.task_history[0]
    assert entry.status is TaskStatus.Cancelled
    assert entry.status_message == "Stopped by the agent: The trench drifted."
    assert stop_message("operator") == "Stopped by the operator."
    assert stop_message(None, "why") == ""


def test_a_successful_task_still_records_exactly_one_entry(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Regression: post_task's append moved into _record_outcome."""
    task = _make_task(microscope, tmp_path)
    task._run = lambda: None  # type: ignore[method-assign]
    task.run()

    assert len(task.lamella.task_history) == 1
    entry = task.lamella.task_history[0]
    assert entry.status is TaskStatus.Completed
    assert entry.end_timestamp is not None


def test_the_history_entry_survives_the_next_task_on_that_lamella(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The point of freezing a copy: pre_task resets the live task_state, so an
    entry holding a reference rather than a copy would be silently rewritten."""
    task = _make_task(microscope, tmp_path)
    _run_raising(task, RuntimeError("stage timeout"))

    # The same lamella starts its next task, resetting the live state.
    next_task = MillTrenchTask(
        microscope=microscope, config=MillTrenchTaskConfig(), lamella=task.lamella
    )
    next_task.pre_task()

    assert task.lamella.task_state.status is TaskStatus.InProgress
    assert task.lamella.task_history[0].status is TaskStatus.Failed
    assert task.lamella.task_history[0].status_message == "stage timeout"


def test_a_recording_failure_does_not_mask_the_task_error(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Recording runs inside the except block, so if it raised it would replace the
    task's own exception and hide what actually went wrong."""
    task = _make_task(microscope, tmp_path)
    task._record_outcome = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("history is broken")
    )
    task._run = lambda: (_ for _ in ()).throw(ValueError("the real failure"))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="the real failure"):
        task.run()


def test_the_failure_hook_sees_the_terminal_status(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The hook fires after the outcome is recorded, not before.

    Previously the manager set Failed only once the exception had propagated out
    of run(), so a task_failed consumer read InProgress off task_state.
    """
    seen: List[HookContext] = []
    manager = HookManager()
    manager.register(FunctionHook(events=["task_failed"], callback=seen.append))

    task = _make_task(microscope, tmp_path)
    task.task_manager = type("_M", (), {"hook_manager": manager, "is_stopped": False})()
    _run_raising(task, RuntimeError("stage timeout"))

    assert len(seen) == 1
    assert seen[0].task_state.status is TaskStatus.Failed
    assert seen[0].error == "stage timeout"


class _Stoppable:
    """A run's manager as a task polls it: its token and whether to unwind."""

    def __init__(self) -> None:
        self.abort_token = threading.Event()
        self.experiment = None

    @property
    def should_abort(self) -> bool:
        return self.abort_token.is_set()

    def deciders(self) -> tuple:
        return ()


def test_a_stop_during_the_closing_reference_images_lets_the_task_finish(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """FIB-1523: a Rough Milling that milled every stage, stopped while it took
    its final reference images, was recorded Cancelled."""
    manager = _Stoppable()
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    task = MillTrenchTask(
        microscope=microscope,
        config=MillTrenchTaskConfig(),
        lamella=lamella,
        task_manager=manager,
    )

    def _run() -> None:
        task._check_for_abort()  # the work: milled, not stopped
        manager.abort_token.set()  # Stop, as the closing images start
        task._acquire_set_of_reference_images(task.config.imaging)

    task._run = _run  # type: ignore[method-assign]
    task.run()

    entry = lamella.task_history[-1]
    assert entry.status is TaskStatus.Completed
    assert entry.outputs["final_sem"] and entry.outputs["final_fib"]


def test_a_stop_before_the_closing_images_still_cancels(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    manager = _Stoppable()
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    task = MillTrenchTask(
        microscope=microscope,
        config=MillTrenchTaskConfig(),
        lamella=lamella,
        task_manager=manager,
    )

    def _run() -> None:
        manager.abort_token.set()
        task.log_status_message("MILL", "Milling...")

    task._run = _run  # type: ignore[method-assign]
    with pytest.raises(InterruptedError):
        task.run()
    assert lamella.task_history[-1].status is TaskStatus.Cancelled
