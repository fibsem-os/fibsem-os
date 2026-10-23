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

The queue lives only while a run does, so its edits are the runner's too:
``add`` work to it, ``requeue`` a pair, ``edit`` a waiting one. The timeline,
the Add to queue button and the App Server all edit it here, checked by the
same rules a start is, and each edit is recorded with who made it.
"""

from __future__ import annotations

import contextvars
import importlib
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from psygnal import Signal

from fibsem.hooks import HookManager
from fibsem.workflows.actions import ACTION, action_record
from fibsem.workflows.queue import QUEUE_ACTIONS, QueueOp
from fibsem.workflows.status import TaskStatus

# What a run is over: an app's own items (AutoLamella's lamellae), or grids.
ITEMS = "items"
GRIDS = "grids"


def stop_message(actor: Any = None, reason: str = "") -> str:
    """What a stopped task's record says: "Stopped by the agent: <reason>.",
    or empty when nobody is named."""
    if actor is None:
        return ""
    reason = (reason or "").strip().rstrip(".")
    return f"Stopped by the {actor}" + (f": {reason}." if reason else ".")


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
class QueueEdit:
    """What an edit to the running queue did, or why it did nothing.

    ``reason`` is said as it is, by a toast or to the agent. ``error_type``
    names the refusal: ``not_running``, ``refused`` (the run's rules),
    ``invalid`` (no such action), ``not_found``, ``not_pending``, ``fixed``
    (a step the plan places), ``already_queued``. ``items`` are the items
    added, or the one edited. ``changed`` is False for an edit that moved
    nothing, the item being first or last already.
    """

    applied: bool
    reason: str = ""
    error_type: str = ""
    items: Tuple[Any, ...] = ()
    changed: bool = False
    version: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueuePlan:
    """The pairs adding ``request`` would queue, in order, and whether they go
    to run next."""

    request: RunRequest
    pairs: Tuple[Tuple[str, str], ...]
    run_next: bool = False


def not_edited(reason: str, error_type: str, **details: Any) -> QueueEdit:
    return QueueEdit(
        applied=False, reason=reason, error_type=error_type, details=details
    )


def item_task_pairs(request: RunRequest) -> List[Tuple[str, str]]:
    """Task-outer, item-inner: how a run's queue is laid out
    (``TaskQueue.build_from_matrix``), so added work interleaves the same way."""
    return [(name, task) for task in request.task_names for name in request.names]


@dataclass(frozen=True)
class RunRules:
    """One kind of run for one app.

    ``check(experiment, microscope, request)`` refuses the request or accepts
    it with its names resolved; ``make_manager(microscope, experiment,
    request, hook_manager=, responder=, deciders=, images=)`` makes the run's
    task manager. The same check is asked before work joins a running queue.
    ``plan_addition(queue, request, run_next)`` is the pairs that adding the
    request queues, or a refusal; each task over each item by default.
    ``prepare(experiment, request)`` readies the experiment for the request's
    pairs before they are queued; AutoLamella gives a lamella the config of a
    task it has none for.
    """

    check: Callable[[Any, Any, RunRequest], RunStart]
    make_manager: Callable[..., Any]
    plan_addition: Optional[
        Callable[[Any, RunRequest, bool], Union[Sequence[Tuple[str, str]], QueueEdit]]
    ] = None
    prepare: Optional[Callable[[Any, RunRequest], None]] = None


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
    "volume": "fibsem.applications.volume.runs:RUNS",
}


@dataclass(frozen=True)
class RunFailure:
    """A task of the run that failed: its item and task, why, and when (epoch s)."""

    what: str
    message: str
    at: float


class RunLine:
    """What the run last said, for a reader that asks rather than listens (the
    App Server's status): its last lifecycle report, the step it is on, who
    holds it and since when, and its failures. A new run starts it over; the
    failures stay after the run ends, as the window's failure line does."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.report: Optional[Any] = None
        # The task's step (`log_status_message`), and the run's own line
        # ("Loading grid G2...", "Waiting: ..."), which wins while it says something.
        self.task_step = ""
        self.run_step = ""
        self.hold: Optional[Any] = None
        self.held_since: Optional[float] = None
        self.failures: List[RunFailure] = []

    @property
    def step(self) -> str:
        return self.run_step or self.task_step

    def take_status(self, event: Any) -> None:
        """A ``WorkflowStatusEvent``."""
        if event.status_bar is not None:
            self.run_step = event.status_bar
        report = event.report
        if report is None:
            return
        if report.status is TaskStatus.InProgress:
            self.task_step = ""  # a new task, with no step said yet
        self.report = report
        if report.status is TaskStatus.Failed:
            self.failures.append(
                RunFailure(
                    f"{report.item_name} › {report.task_name}",
                    report.error_message or "failed",
                    time.time(),
                )
            )

    def take_hold(self, hold: Any) -> None:
        """A ``Hold``, or None when released. The same hold again keeps its clock."""
        if hold is None:
            self.held_since = None
        elif self.hold is None:
            self.held_since = time.time()
        self.hold = hold

    def take_step(self, step: str) -> None:
        self.task_step = step or ""

    def ended(self) -> None:
        self.hold = self.held_since = None
        self.run_step = self.task_step = ""


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
    chrome there, whoever started the run. ``stopping`` is emitted where a
    stop of the run was asked, whoever asked. ``finished(cancelled)`` is
    emitted on the run's thread when it has ended. ``notice(message, type)``
    is what a hook asks to be shown, from the run's thread; with nobody
    listening, notification hooks log instead. ``queue_changed(info)`` is the
    running queue's ``BaseTaskManager.queue_changed``, from whichever thread
    edited it: a window or a headless listener hears it the same way.
    ``status``, ``held``, ``pending`` and ``step`` are the run's own
    (``BaseTaskManager``), from the run's thread; a status line nobody
    hears goes to the log.
    """

    starting = Signal(object)
    stopping = Signal()
    finished = Signal(bool)
    notice = Signal(str, str)
    queue_changed = Signal(dict)
    status = Signal(object)
    held = Signal(object)
    pending = Signal(bool)
    step = Signal(str)

    def __init__(self, session: Any) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # The run's task manager while one runs.
        self.manager: Optional[Any] = None
        # The last run's summary and why it ended short of done, kept after
        # its manager is gone: the App Server's run summary reads them.
        self.last_summary: Optional[Any] = None
        self.last_note: str = ""
        self.line = RunLine()

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def working_on(self) -> Optional[str]:
        """The item the run's task is on now, by name; None with no run."""
        manager = self.manager
        active = manager.queue.active if manager is not None else None
        return active.item_name if active is not None else None

    def busy_refusal(self, names: Sequence[str], what: str) -> str:
        """Why *what* must wait, or "": the run is working on one of *names*.
        Everything else may be edited while it runs, by the window and the
        agent alike."""
        name = self.working_on
        if name is None or name not in names:
            return ""
        return f"The run is working on {name}; {what} when it has moved on."

    def run_refusal(self, what: str) -> str:
        """Why *what* must wait, or "": a workflow is running. For the edits
        no run may see change under it, by the window and the agent alike."""
        if not self.is_running:
            return ""
        return f"A workflow is running; {what} when it has stopped."

    @property
    def should_abort(self) -> bool:
        """Whether the task now running should unwind (see
        ``BaseTaskManager.should_abort``); False with no run."""
        manager = self.manager
        return manager is not None and manager.should_abort

    def stop(
        self, actor: Any = None, via: str = "", message: str = "", **details: Any
    ) -> None:
        """End the run after the task now running unwinds. Named, ``actor``'s
        stop goes on the record first, saying from where (``via``). *message*
        is what the stopped task's record says (`stop_message`)."""
        if not self.is_running:
            return
        self._record_stop("workflow", actor, via, details)
        manager = self.manager
        if manager is not None:
            manager.stop(message or stop_message(actor))
        self.stopping.emit()

    def stop_task(
        self, actor: Any = None, via: str = "", message: str = "", **details: Any
    ) -> None:
        """Abandon the task now running and go on with the rest; recorded as
        :meth:`stop` records."""
        self._record_stop("task", actor, via, details)
        manager = self.manager
        if manager is not None:
            manager.stop_task(message or stop_message(actor))

    def _record_stop(
        self, stopped: str, actor: Any, via: str, details: Dict[str, Any]
    ) -> None:
        """Put a stop on the session's record (events.jsonl) and in the log,
        before it runs: what was stopped, by whom and from where. Without it a
        stop leaves only the run's ``task_cancelled``, which does not say who
        stopped it. The agent's stop records itself (``_stop_reason`` in the server),
        so a stop with no actor named records nothing here."""
        if actor is None or not self.is_running:
            return
        logging.info("%s stopped by the %s (%s)", stopped.capitalize(), actor, via)
        buffer = self._session.event_buffer
        if buffer is not None:
            buffer.append(
                ACTION,
                action_record(
                    f"Stop the {'run' if stopped == 'workflow' else 'task'}",
                    stops=stopped,
                    via=via,
                    **details,
                ),
                actor=actor,
            )

    def rules(self, experiment: Any, kind: str) -> RunRules:
        if kind == GRIDS:
            from fibsem.workflows.grid.runs import GRID_RUNS

            return GRID_RUNS
        if kind == ITEMS:
            return app_runs(experiment.app).items
        raise KeyError(f"No run of kind {kind!r}.")

    def run_hooks(self, app: str) -> HookManager:
        """The hooks for one run of ``app``: the app's, with notices to
        ``notice`` when someone listens. The event stream's lifecycle feed is
        registered by the task manager for its run (FIB-1044)."""
        manager = app_runs(app).make_hooks()
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
                hook_manager=self.run_hooks(experiment.app),
                responder=self._session.responder,
                deciders=self._session.deciders,
                images=self._session.images,
            )
            self.line.reset()
            self.relay(self.manager)
            # The run is whoever started it: the agent's run records its
            # events as the agent's, so it runs in a copy of the
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

    def relay(self, manager: Any) -> None:
        """Pass on what ``manager``'s run says (``BaseTaskManager``'s signals)
        as this runner's own, for a window or anyone else listening."""
        manager.queue_changed.connect(self.queue_changed)
        manager.status.connect(self._on_status)
        manager.held.connect(self.line.take_hold)
        manager.held.connect(self.held)
        manager.pending.connect(self.pending)
        manager.step.connect(self.line.take_step)
        manager.step.connect(self.step)

    def _on_status(self, event: Any) -> None:
        self.line.take_status(event)
        if len(self.status):
            self.status.emit(event)
        elif event.report is None:
            logging.info(event.message or event.status_bar or "")

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
            self.line.ended()
            with self._lock:
                self.manager = None
                self._thread = None
            self.finished.emit(cancelled)

    def check(self, experiment: Any, request: RunRequest) -> RunStart:
        """Whether ``request`` could run, by its rules, without starting it:
        what a window asks before its confirmation."""
        try:
            rules = self.rules(experiment, request.kind)
        except KeyError as e:
            return refused(str(e.args[0]))
        return rules.check(experiment, self._session.microscope, request)

    # --- The running queue ---

    def joins(self, kind: str) -> str:
        """Why work of ``kind`` cannot join the running queue now, or "".
        A run is of one kind: grid tasks join a grid run only."""
        manager = self.manager
        if manager is None:
            return "no workflow is running; the queue exists only during a run"
        if manager.RUN_KIND == kind:
            return ""
        going = f"{manager.RUN_DESCRIPTION.capitalize()} is going"
        if kind == GRIDS:
            return f"{going}; grid tasks cannot join it"
        return f"{going}; only grid tasks can join it"

    def _joining(self, kind: str) -> Tuple[Optional[Any], Optional[QueueEdit]]:
        """The running manager, or why ``kind`` cannot join it."""
        why = self.joins(kind)
        if why:
            error_type = "not_running" if self.manager is None else "refused"
            return None, not_edited(why, error_type)
        return self.manager, None

    def plan_addition(
        self, request: RunRequest, run_next: bool = False
    ) -> Union[QueuePlan, QueueEdit]:
        """What adding ``request`` to the running queue would queue, or why
        not; nothing is queued. What a window prices and confirms before it
        calls ``add``."""
        manager, why = self._joining(request.kind)
        if why is not None:
            return why
        experiment = manager.experiment
        rules = self.rules(experiment, request.kind)
        checked = rules.check(experiment, manager.microscope, request)
        if not checked.started:
            return not_edited(checked.reason, "refused", **checked.details)
        request = checked.request or request
        if rules.plan_addition is None:
            return QueuePlan(request, tuple(item_task_pairs(request)), run_next)
        pairs = rules.plan_addition(manager.queue, request, run_next)
        if isinstance(pairs, QueueEdit):
            return pairs
        # Where it goes is the plan's, not the caller's.
        return QueuePlan(request, tuple(pairs), run_next=False)

    def add(
        self, request: RunRequest, run_next: bool = False, actor: Any = None
    ) -> QueueEdit:
        """Add ``request``'s work to the running queue: at the end, or to run
        next, in order. Checked as a start is; a pair already waiting is added
        again, since the operator asked for it. Call it where the experiment
        is edited (``session.run_edit``): readying a pair can edit it."""
        plan = self.plan_addition(request, run_next)
        if isinstance(plan, QueueEdit):
            return plan
        manager = self.manager
        rules = self.rules(manager.experiment, request.kind)
        if rules.prepare is not None:
            rules.prepare(manager.experiment, plan.request)
        run_next = plan.run_next
        added, anchor = [], None
        for name, task_name in plan.pairs:
            # For "run next" each item goes after the one before it: every one
            # at the front would put each ahead of the last, reversing them.
            if not run_next:
                item = manager.queue.add(name, task_name)
            elif anchor is None:
                item = manager.queue.add(name, task_name, front=True)
            else:
                item = manager.queue.add(name, task_name, after=anchor)
            if item is not None:
                anchor = item.id
                added.append(item)
        self._queue_changed(manager, "add_next" if run_next else "add", added, actor)
        return QueueEdit(
            applied=bool(added),
            items=tuple(added),
            changed=bool(added),
            version=manager.queue.version,
        )

    def requeue(
        self, name: str, task_name: str, front: bool = False, actor: Any = None
    ) -> QueueEdit:
        """Queue ``task_name`` for ``name`` (again): at the end, or the front.
        Checked as a start is. A pair already waiting is refused: it runs once.
        Call it where the experiment is edited, as for ``add``."""
        manager = self.manager
        kind = manager.RUN_KIND if manager is not None else ITEMS
        manager, why = self._joining(kind)
        if why is not None:
            return why
        if task_name in manager.queue.fixed_tasks:
            return not_edited(f"{task_name} stays with {name}'s tasks.", "fixed")
        experiment = manager.experiment
        rules = self.rules(experiment, kind)
        request = RunRequest(kind=kind, task_names=(task_name,), names=(name,))
        checked = rules.check(experiment, manager.microscope, request)
        if not checked.started:
            return not_edited(checked.reason, "refused", **checked.details)
        label = f"{task_name} for {name}"
        if manager.queue.has_pending_pair(name, task_name):
            return not_edited(
                f"{label} is already waiting in the queue; it runs once.",
                "already_queued",
            )
        if rules.prepare is not None:
            rules.prepare(experiment, request)
        result = manager.queue.requeue(name, task_name, front=front)
        if not result.ok:
            # Queued by another edit between the check and here.
            return not_edited(
                f"{label} is already waiting in the queue; it runs once.",
                "already_queued",
            )
        action = "requeue_front" if front else "requeue"
        self._queue_changed(manager, action, [result.item], actor)
        return QueueEdit(
            applied=True,
            items=(result.item,),
            changed=True,
            version=result.version,
        )

    def edit(
        self, name: str, task_name: str, action: str, actor: Any = None
    ) -> QueueEdit:
        """Remove or move the waiting ``task_name`` for ``name``: one of
        :data:`QUEUE_ACTIONS`. Only a pair still waiting can be edited."""
        if action not in QUEUE_ACTIONS:
            return not_edited(
                f"Unknown action {action!r}.",
                "invalid",
                action_names=list(QUEUE_ACTIONS),
            )
        manager = self.manager
        if manager is None:
            return not_edited(self.joins(ITEMS), "not_running")
        label = f"{task_name} for {name}"
        item = manager.queue.find_pair(name, task_name)
        if item is None:
            return not_edited(f"{label} is no longer in the queue.", "not_found")
        result = manager.queue.apply(item.id, action)
        if result.op is QueueOp.FIXED:
            return not_edited(f"{task_name} stays with {name}'s tasks.", "fixed")
        if result.op is QueueOp.NOT_PENDING:
            if item.status is TaskStatus.InProgress:
                state = "is running now; stop it instead"
            elif item.status is TaskStatus.Removed:
                state = "was already removed"
            else:
                state = f"already ran ({item.status.name})"
            return not_edited(f"{label} {state}.", "not_pending")
        if result.op is QueueOp.NOT_FOUND:
            return not_edited(f"{label} is no longer in the queue.", "not_found")
        if result.ok:
            self._queue_changed(manager, action, [item], actor)
        return QueueEdit(
            applied=True,
            items=(item,),
            changed=result.ok,
            version=result.version,
        )

    def _queue_changed(
        self, manager: Any, action: str, items: Sequence[Any], actor: Any
    ) -> None:
        """Tell the window, and put one ``queue_changed`` per item on the
        record. Unnamed, the actor is whoever marked the thread (the agent's
        requests are marked); a window names the operator."""
        if not items:
            return
        manager.notify_queue_changed()
        buffer = self._session.event_buffer
        if buffer is None:
            return
        for item in items:
            buffer.append(
                "queue_changed",
                {
                    "action": action,
                    "item_name": item.item_name,
                    "task_name": item.task_name,
                    "queue_item_id": item.id,
                },
                actor=actor,
            )

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for the run to end; whether it has."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.is_running
