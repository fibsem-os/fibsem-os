"""One predicate behind every abort check (groundwork for FIB-499).

There are two `_check_for_abort` functions at different layers: the method on
AutoLamellaTask, which has a task to ask, and the module-level one in
workflows.ui, which only has a parent_ui. They previously reached for raw state
by two different routes — `self._stop_event` and `task_manager.is_stopped` — and
so could disagree about what counts as cancelled.

Both now answer from `TaskManager`: the task side polls `abort_token` (which is
also what milling, autofocus and the inline checks in the fluorescence,
coincidence-milling and grid tasks already read), and workflows.ui asks
`should_abort` because it has no token to poll. Today both are the run-wide stop
event, so this is a no-op; the point is that adding a per-task stop has exactly
one place to change.

`is_stopped` keeps its own narrow meaning — should the *run* end — and only
_run_queue's loop asks it.
"""

from pathlib import Path
from typing import Optional

import pytest

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.ui import _check_for_abort


class NoMicroscope:
    fm = None


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.positions.append(
        Lamella(path=Path(experiment.path) / "L1", number=1, petname="L1")
    )
    return TaskManager(microscope=NoMicroscope(), experiment=experiment, parent_ui=None)


class FakeUI:
    """A parent_ui is all workflows.ui._check_for_abort ever gets."""

    def __init__(self, task_manager: Optional[TaskManager]):
        self._task_manager = task_manager


# ── The predicate ─────────────────────────────────────────────────────────────


def test_nothing_aborts_a_manager_that_has_not_been_stopped(manager):
    assert manager.should_abort is False
    assert manager.is_stopped is False
    assert manager.abort_token.is_set() is False


def test_stopping_the_run_aborts_the_task_too(manager):
    manager.stop()

    assert manager.is_stopped is True
    assert manager.should_abort is True
    assert manager.abort_token.is_set() is True


# ── Both checks answer from it ────────────────────────────────────────────────


def test_the_ui_check_follows_the_predicate(manager):
    ui = FakeUI(manager)
    assert _check_for_abort(ui) is False

    manager.stop()
    with pytest.raises(InterruptedError):
        _check_for_abort(ui)


def test_the_task_check_follows_the_predicate(manager):
    """The task polls the manager's token, so it cannot diverge from the UI check."""
    from types import SimpleNamespace

    task = SimpleNamespace(task_manager=manager, _stop_event=manager.abort_token)
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    check = AutoLamellaTask._check_for_abort
    check(task)  # must not raise

    manager.stop()
    with pytest.raises(InterruptedError):
        check(task)


def test_the_two_checks_agree(manager):
    """The property that matters: whatever the predicate says, both obey."""
    from types import SimpleNamespace

    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    ui = FakeUI(manager)
    task = SimpleNamespace(task_manager=manager, _stop_event=manager.abort_token)

    for stopped in (False, True):
        if stopped:
            manager.stop()

        ui_aborted = task_aborted = False
        try:
            _check_for_abort(ui)
        except InterruptedError:
            ui_aborted = True
        try:
            AutoLamellaTask._check_for_abort(task)
        except InterruptedError:
            task_aborted = True

        assert ui_aborted == task_aborted == stopped


# ── Fallbacks for a task with no manager ──────────────────────────────────────


def test_a_task_without_a_manager_falls_back_to_its_token():
    """A script or test driving a task directly has no manager to ask."""
    import threading
    from types import SimpleNamespace

    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    event = threading.Event()
    task = SimpleNamespace(task_manager=None, _stop_event=event)
    AutoLamellaTask._check_for_abort(task)  # must not raise

    event.set()
    with pytest.raises(InterruptedError):
        AutoLamellaTask._check_for_abort(task)


def test_a_task_with_neither_never_aborts():
    from types import SimpleNamespace

    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    task = SimpleNamespace(task_manager=None, _stop_event=None)
    AutoLamellaTask._check_for_abort(task)  # must not raise


def test_the_ui_check_is_a_noop_headless():
    assert _check_for_abort(None) is False


# ── What tasks are handed ─────────────────────────────────────────────────────


def test_both_task_hierarchies_take_the_managers_token(manager):
    """GridTask is a parallel hierarchy with its own __init__, so it has to be
    kept in step by hand — it would otherwise miss a per-task stop entirely."""
    import inspect

    from fibsem.applications.autolamella.workflows.tasks import base
    from fibsem.applications.autolamella.workflows.tasks.grid import base as grid_base

    for module in (base, grid_base):
        source = inspect.getsource(module)
        assert "task_manager.abort_token if task_manager else None" in source
        assert "task_manager._stop_event" not in source
