"""Every terminal outcome is recorded, not just success.

``task_state`` is a single object the next task's ``pre_task`` overwrites, so an
outcome that never reaches ``task_history`` is gone from the experiment as soon as
that lamella starts its next task. Before FIB-490 only ``post_task`` appended, and
``run()`` re-raises before reaching it -- so failures and cancellations were absent
from the history entirely, leaving the logfile as the only record they happened.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTask,
    MillTrenchTaskConfig,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.hooks import FunctionHook, HookContext, HookManager
from fibsem.microscope import FibsemMicroscope


@pytest.fixture
def microscope() -> FibsemMicroscope:
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


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
    assert entry.status is AutoLamellaTaskStatus.Failed
    assert entry.status_message == "stage timeout"
    assert entry.end_timestamp is not None


def test_a_cancelled_task_is_recorded_as_cancelled_not_failed(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    task = _make_task(microscope, tmp_path)
    _run_raising(task, OperationCancelledError("stopped"))

    assert len(task.lamella.task_history) == 1
    assert task.lamella.task_history[0].status is AutoLamellaTaskStatus.Cancelled


def test_a_successful_task_still_records_exactly_one_entry(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Regression: post_task's append moved into _record_outcome."""
    task = _make_task(microscope, tmp_path)
    task._run = lambda: None  # type: ignore[method-assign]
    task.run()

    assert len(task.lamella.task_history) == 1
    entry = task.lamella.task_history[0]
    assert entry.status is AutoLamellaTaskStatus.Completed
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

    assert task.lamella.task_state.status is AutoLamellaTaskStatus.InProgress
    assert task.lamella.task_history[0].status is AutoLamellaTaskStatus.Failed
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
    assert seen[0].task_state.status is AutoLamellaTaskStatus.Failed
    assert seen[0].error == "stage timeout"
