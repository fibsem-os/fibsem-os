"""A session whose run state a test sets by hand, for hosts and windows that
are read through ``session.runner`` without starting a real run."""

import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

from psygnal import Signal

from fibsem.workflows.queue import TaskQueue
from fibsem.workflows.runner import (
    APP_RUNS,
    GRIDS,
    ITEMS,
    RunRequest,
    WorkflowRunner,
    app_runs,
)
from fibsem.workflows.session import AppSession


class StubRunner(WorkflowRunner):
    """A runner whose run state a test sets by hand: the manager and whether a
    run is going, read by the App Server context, the abort check and the
    windows. Nothing runs; the running queue's edits are the runner's own."""

    is_running = False

    def __init__(
        self, session: Any, manager: Optional[Any] = None, is_running: bool = False
    ):
        super().__init__(session)
        self.manager = manager
        self.is_running = is_running
        self.stops: list = []

    def stop(self, **who) -> None:
        self.stops.append("run")
        super().stop(**who)

    def stop_task(self, **who) -> None:
        self.stops.append("task")
        super().stop_task(**who)


def stub_session(manager: Optional[Any] = None, is_running: bool = False) -> AppSession:
    session = AppSession()
    session.runner = StubRunner(session, manager, is_running)
    return session


@contextmanager
def run_working_on(
    runner: StubRunner, item_name: str, queued: tuple = (), kind: str = ITEMS
):
    """A *kind* run on *item_name* now, with *queued* waiting: what the edit
    refusals (run_edits) read."""
    queue = TaskQueue()
    queue.build_from_pairs([(item_name, "Task")] + [(n, "Task") for n in queued])
    queue.next()
    before = runner.manager, runner.is_running
    manager = SimpleNamespace(queue=queue, RUN_KIND=kind)
    runner.manager, runner.is_running = manager, True
    try:
        yield
    finally:
        runner.manager, runner.is_running = before


class Stopped:
    """A run that has been told to stop, for the abort check to find."""

    should_abort = True
    is_stopped = True


class HeldManager:
    """A run's manager that records what it was asked and does nothing until
    it is let go: a run that starts for real, through the runner, and holds
    its slot without touching the microscope."""

    should_abort = False
    queue_changed = Signal(dict)
    status = Signal(object)
    held = Signal(object)
    pending = Signal(bool)
    step = Signal(str)

    def __init__(self, request: Any, on_execute: Optional[Callable[[], None]] = None):
        self.request = request
        self.release = threading.Event()
        self.is_stopped = False
        self._on_execute = on_execute

    def execute(self, task_names, names) -> None:
        if self._on_execute is not None:
            self._on_execute()
        self.release.wait(30)

    def stop(self, message: str = "") -> None:
        self.is_stopped = True
        self.release.set()

    def stop_task(self, message: str = "") -> None:
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


def run_through_session(
    microscope: Any,
    experiment: Any,
    request: RunRequest,
    hook_manager: Optional[Any] = None,
    timeout: float = 300,
) -> Any:
    """Run ``request`` the one way a run starts, through a session's runner,
    and wait for it to end. Answers the run's manager, for its queue.
    ``hook_manager`` is the run's hooks in place of the app's."""
    session = AppSession()
    session.microscope = microscope
    session.experiment = experiment
    if hook_manager is not None:
        session.runner.run_hooks = lambda app: hook_manager
    managers: list = []
    session.runner.starting.connect(
        lambda _request: managers.append(session.runner.manager)
    )
    started = session.runner.start(experiment, request)
    assert started.started, started.reason
    assert session.runner.join(timeout), "the run did not end"
    return managers[0]


def run_lamellae(
    microscope: Any,
    experiment: Any,
    task_names: List[str],
    names: Optional[List[str]] = None,
    hook_manager: Optional[Any] = None,
) -> Any:
    """A lamella run of ``task_names`` over ``names`` (every lamella if None)."""
    request = RunRequest(
        kind=ITEMS,
        task_names=tuple(task_names),
        names=tuple(names) if names is not None else None,
    )
    return run_through_session(microscope, experiment, request, hook_manager)


def run_grids(
    microscope: Any,
    experiment: Any,
    task_names: Optional[List[str]] = None,
    grid_names: Optional[List[str]] = None,
    **options: bool,
) -> Any:
    """A grid run of ``task_names`` (the protocol's, in its order, if None)
    over ``grid_names`` (every grid if None), with the grid run's options."""
    if task_names is None:
        task_names = experiment.grid_protocol.ordered_task_names
    request = RunRequest(
        kind=GRIDS,
        task_names=tuple(task_names),
        names=tuple(grid_names) if grid_names is not None else None,
        options=options,
    )
    return run_through_session(microscope, experiment, request)
