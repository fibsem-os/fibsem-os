"""A session whose run state a test sets by hand, for hosts and windows that
are read through ``session.runner`` without starting a real run."""

import threading
from dataclasses import replace
from typing import Any, Callable, List, Optional

from fibsem.workflows.runner import (
    APP_RUNS,
    GRIDS,
    RunRequest,
    WorkflowRunner,
    app_runs,
)
from fibsem.workflows.session import AppSession


class StubRunner:
    """The run state the App Server context, the abort check and the windows
    read; nothing runs."""

    def __init__(self, manager: Optional[Any] = None, is_running: bool = False):
        self.manager = manager
        self.is_running = is_running
        self.last_summary: Optional[Any] = None
        self.last_note = ""
        self.stops: list = []

    @property
    def should_abort(self) -> bool:
        return self.manager is not None and self.manager.should_abort

    def stop(self) -> None:
        self.stops.append("run")
        if self.manager is not None:
            self.manager.stop()

    def stop_task(self) -> None:
        self.stops.append("task")
        if self.manager is not None:
            self.manager.stop_task()


def stub_session(manager: Optional[Any] = None, is_running: bool = False) -> AppSession:
    session = AppSession()
    session.runner = StubRunner(manager, is_running)
    return session


class Stopped:
    """A run that has been told to stop, for the abort check to find."""

    should_abort = True
    is_stopped = True


class HeldManager:
    """A run's manager that records what it was asked and does nothing until
    it is let go: a run that starts for real, through the runner, and holds
    its slot without touching the microscope."""

    should_abort = False

    def __init__(self, request: Any, on_execute: Optional[Callable[[], None]] = None):
        self.request = request
        self.release = threading.Event()
        self.is_stopped = False
        self._on_execute = on_execute

    def execute(self, task_names, names) -> None:
        if self._on_execute is not None:
            self._on_execute()
        self.release.wait(30)

    def stop(self) -> None:
        self.is_stopped = True
        self.release.set()

    def stop_task(self) -> None:
        pass

    def build_run_summary_dataframe(self) -> None:
        return None

    def closing_note(self) -> str:
        return ""


class HeldRuns:
    """Every run started while held: the requests, as checked."""

    def __init__(self, on_execute: Optional[Callable[[], None]] = None) -> None:
        self.requests: List[RunRequest] = []
        self.managers: List[HeldManager] = []
        self._on_execute = on_execute

    def make_manager(self, microscope, experiment, request, **_) -> HeldManager:
        manager = HeldManager(request, self._on_execute)
        self.requests.append(request)
        self.managers.append(manager)
        return manager

    def end(self, runner: WorkflowRunner) -> None:
        """Let the run go, and wait for its slot to free."""
        for manager in self.managers:
            manager.release.set()
        assert runner.join(10), "the held run did not end"


def hold_runs(
    monkeypatch, kind: str, on_execute: Optional[Callable[[], None]] = None
) -> HeldRuns:
    """Runs of ``kind`` start through the runner as always, on held managers;
    ``on_execute`` runs on each run's thread as it begins."""
    held = HeldRuns(on_execute)
    if kind == GRIDS:
        from fibsem.workflows.grid import runs as grid_runs

        monkeypatch.setattr(
            grid_runs,
            "GRID_RUNS",
            replace(grid_runs.GRID_RUNS, make_manager=held.make_manager),
        )
    else:
        runs = app_runs("autolamella")
        monkeypatch.setitem(
            APP_RUNS,
            "autolamella",
            replace(runs, items=replace(runs.items, make_manager=held.make_manager)),
        )
    return held


def started_with(held: HeldRuns) -> list:
    """The (tasks, names) each held run was started with."""
    return [
        (list(r.task_names), list(r.names) if r.names is not None else None)
        for r in held.requests
    ]
