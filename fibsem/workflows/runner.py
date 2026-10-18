"""The session's run controller: one place every run starts, whoever asks.

The Run button, the Grids view's Run, the agent and the App Server all start a
run with ``session.runner.start(experiment, request)``. What a run is -- which
tasks and items are valid, the gate an app puts in front of them, the task
manager that runs them, the hooks it fires -- comes from the app the
experiment belongs to (``experiment.app``), never from whichever window is
open. Grid runs are the same for every app that has grids.

The manager does the run itself (``BaseTaskManager.execute``). The runner holds
what outlives one run or comes before it: the one-run-at-a-time slot, the
thread, the last run's summary and closing note, and the signals a window
draws its chrome from. Nothing here needs Qt.
"""

from __future__ import annotations

import contextvars
import importlib
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

from psygnal import Signal

from fibsem.hooks import HookManager

# What a run is over: an app's own items (AutoLamella's lamellae), or grids.
ITEMS = "items"
GRIDS = "grids"


@dataclass(frozen=True)
class RunRequest:
    """What to run: ``task_names`` in order, over ``names`` (every item or
    grid when None). ``options`` are the kind's own: a grid run's
    ``inventory_first``, ``unload_at_end`` and ``report_at_end``."""

    kind: str
    task_names: Tuple[str, ...]
    names: Optional[Tuple[str, ...]] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def option(self, name: str) -> bool:
        return bool(self.options.get(name, False))


@dataclass(frozen=True)
class RunStart:
    """Whether a run started, and if not why, with what the caller needs to
    ask again (the valid names). ``request`` is the run as it was checked:
    the names it resolved to."""

    started: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    request: Optional[RunRequest] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"started": self.started}
        if self.reason:
            out["reason"] = self.reason
        out.update(self.details)
        return out


def refused(reason: str, **details: Any) -> RunStart:
    return RunStart(started=False, reason=reason, details=details)


@dataclass(frozen=True)
class RunRules:
    """One kind of run for one app.

    ``check(experiment, microscope, request)`` refuses the request or accepts
    it with its names resolved; ``make_manager(microscope, experiment,
    request, parent_ui, hook_manager, responder, answerers)`` makes the run's
    task manager.
    """

    check: Callable[[Any, Any, RunRequest], RunStart]
    make_manager: Callable[..., Any]


@dataclass(frozen=True)
class AppRuns:
    """What an app tells the runner: its item runs, and the hooks its runs
    fire (rebuilt per run, so a saved hook configuration takes effect on the
    next one)."""

    items: RunRules
    make_hooks: Callable[[], HookManager]


APP_RUNS: Dict[str, AppRuns] = {}

# Read on first use, so the runner never imports an app it does not run.
_BUILTIN_RUNS: Dict[str, str] = {
    "autolamella": "fibsem.applications.autolamella.workflows.runs:RUNS",
}


def register_app_runs(app: str, runs: AppRuns) -> None:
    APP_RUNS[app] = runs


def app_runs(app: str) -> AppRuns:
    """The run rules for ``app``; KeyError when nothing here runs it."""
    if app not in APP_RUNS and app in _BUILTIN_RUNS:
        module, name = _BUILTIN_RUNS[app].split(":")
        APP_RUNS[app] = getattr(importlib.import_module(module), name)
    if app not in APP_RUNS:
        raise KeyError(f"No application here runs {app} experiments.")
    return APP_RUNS[app]


class WorkflowRunner:
    """Starts, stops and remembers the session's runs; one at a time.

    ``starting(request)`` is emitted where ``start`` was called, before the
    run's thread starts: a window lands its pending edits and draws its run
    chrome there, whoever started the run. ``finished(cancelled)`` is
    emitted on the run's thread when it has ended. ``notice(message, type)``
    is what a hook asks to be shown, from the run's thread; with nobody
    listening, notification hooks log instead.
    """

    starting = Signal(object)
    finished = Signal(bool)
    notice = Signal(str, str)

    def __init__(self, session: Any) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # The run's task manager while one runs.
        self.manager: Optional[Any] = None
        # Who hears the run's hold, queue and status (``WorkflowWindow``):
        # the window, when one is attached.
        self.window: Optional[Any] = None
        # The last run's summary and why it ended short of done, kept after
        # its manager is gone: the App Server's run summary reads them.
        self.last_summary: Optional[Any] = None
        self.last_note: str = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def should_abort(self) -> bool:
        """Whether the task now running should unwind (see
        ``BaseTaskManager.should_abort``); False with no run."""
        manager = self.manager
        return manager is not None and manager.should_abort

    def stop(self) -> None:
        """End the run after the task now running unwinds."""
        manager = self.manager
        if manager is not None:
            manager.stop()

    def stop_task(self) -> None:
        """Abandon the task now running and go on with the rest."""
        manager = self.manager
        if manager is not None:
            manager.stop_task()

    def rules(self, experiment: Any, kind: str) -> RunRules:
        if kind == GRIDS:
            from fibsem.workflows.grid.runs import GRID_RUNS

            return GRID_RUNS
        if kind == ITEMS:
            return app_runs(experiment.app).items
        raise KeyError(f"No run of kind {kind!r}.")

    def run_hooks(self, app: str) -> HookManager:
        """The hooks for one run of ``app``: the app's, plus the App Server's
        lifecycle feed when it has one of its own, with notices to
        ``notice`` when someone listens."""
        manager = app_runs(app).make_hooks()
        # The event stream's lifecycle feed is registered by the task manager
        # for its run (FIB-1044). The App Server shares the recorder's hook;
        # one of its own (a server started without a recorder) is added here.
        recorder = self._session.recorder
        host = self._session.app_server
        if (
            host is not None
            and host.running
            and host.lifecycle_hook is not None
            and (recorder is None or host.lifecycle_hook is not recorder.lifecycle_hook)
        ):
            manager.register(host.lifecycle_hook)
        if len(self.notice):
            manager.set_notifier(self.notice.emit)
        return manager

    def start(self, experiment: Any, request: RunRequest) -> RunStart:
        """Check ``request`` against ``experiment``'s app and start it.

        Call it where the experiment is edited (``session.run_edit`` from
        another thread): ``starting`` is emitted here, and a window's chrome
        and save flush run on it.
        """
        with self._lock:
            if self.is_running:
                return refused("a workflow is already running")
            microscope = self._session.microscope
            if microscope is None:
                return refused("no microscope is connected")
            if experiment is None or experiment.task_protocol is None:
                return refused("no experiment is loaded")
            current = microscope.control.current
            if current is not None:
                return refused(f"Can't start the workflow: {current.describe()}.")
            try:
                rules = self.rules(experiment, request.kind)
            except KeyError as e:
                return refused(str(e.args[0]))
            checked = rules.check(experiment, microscope, request)
            if not checked.started:
                return checked
            request = checked.request or request
            self.manager = rules.make_manager(
                microscope,
                experiment,
                request,
                parent_ui=self.window,
                hook_manager=self.run_hooks(experiment.app),
                responder=self._session.responder,
                answerers=self._session.answerers,
            )
            # The run is whoever started it: the agent's run records its
            # events as the agent's (FIB-1276), so it runs in a copy of the
            # starter's context.
            context = contextvars.copy_context()
            thread = threading.Thread(
                target=context.run,
                args=(self._run, self.manager, request),
                name="workflow-run",
                daemon=True,
            )
            self._thread = thread
        try:
            self.starting.emit(request)
        except Exception:
            logging.exception("a window's run chrome failed; the run goes on")
        thread.start()
        return replace(checked, request=request)

    def _run(self, manager: Any, request: RunRequest) -> None:
        try:
            names = list(request.names) if request.names is not None else None
            manager.execute(list(request.task_names), names)
        finally:
            cancelled = manager.is_stopped
            try:
                self.last_summary = manager.build_run_summary_dataframe()
            except Exception as e:
                logging.warning(f"Failed to build the run summary: {e}")
                self.last_summary = None
            self.last_note = manager.closing_note()
            with self._lock:
                self.manager = None
                self._thread = None
            self.finished.emit(cancelled)

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for the run to end; whether it has."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.is_running
