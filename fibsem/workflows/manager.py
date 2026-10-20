"""What running a queue of (item, task) work needs, whatever the item is.

An application's task manager subclasses ``BaseTaskManager`` and writes the run
loop: what an item is, how it is looked up, when it is skipped. AutoLamella's
lamella and grid managers are two.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from psygnal import Signal

from fibsem.acting import AGENT, OPERATOR, WORKFLOW, ActorKind
from fibsem.cancellation import (
    AnyStopEvent,
    NothingToStop,
    OperationCancelledError,
    StopEvent,
)
from fibsem.control import MicroscopeBusy
from fibsem.hooks import HookEvent, HookManager, fire_event
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType
from fibsem.util.durations import format_duration_precise
from fibsem.workflows.event_recording import run_recorder
from fibsem.workflows.interaction import Responder
from fibsem.workflows.queue import TaskQueue, WorkItem
from fibsem.workflows.runner import ITEMS
from fibsem.workflows.status import (
    Hold,
    HoldKind,
    TaskStatus,
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
    post_status,
    take_back_finish,
)


class WorkflowExperiment(Protocol):
    """What the manager needs of an application's experiment."""

    id: str
    name: str
    task_protocol: Any

    @property
    def unsaved(self) -> bool: ...

    def save(self) -> None: ...

    def register_metadata(self, microscope: FibsemMicroscope) -> None: ...

    @property
    def turns_beams_off(self) -> bool: ...


# Exactly one of these ends every run (FIB-1327).
_RUN_ENDS = frozenset(
    {
        HookEvent.WORKFLOW_COMPLETED,
        HookEvent.WORKFLOW_CANCELLED,
        HookEvent.WORKFLOW_STALLED,
        HookEvent.WORKFLOW_FAILED,
    }
)


class BaseTaskManager:
    """What running a queue of (item, task) work needs, whatever the item is.

    The lamella and grid workflows share everything here: the queue, the two stop
    intents and the token tasks poll, the run's signals, the hook run context and
    the run loop. Each subclass says what an item's record is (``_record_for``),
    when it is skipped and what it costs to reach (``_ready``), how a task runs on
    it (``_run_task``) and how a run ends (``_complete``, ``_end_run``).

    ``queue_changed({queue_items, version})`` is emitted on whichever thread
    edited the queue. ``status(WorkflowStatusEvent)`` is a status line or a
    task's report, ``held(Hold or None)`` who holds the run, ``pending(bool)``
    whether it waits between tasks, and ``step(str)`` the step a task is on;
    these are emitted on the run's thread. The session's runner relays them all,
    to a window or anyone else listening.
    """

    queue_changed = Signal(dict)
    status = Signal(object)
    held = Signal(object)
    pending = Signal(bool)
    step = Signal(str)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        experiment: WorkflowExperiment,
        hook_manager: Optional[HookManager] = None,
        responder: Optional[Responder] = None,
        deciders: Optional[Callable[[], Tuple[ActorKind, ...]]] = None,
        images: Optional[Any] = None,
    ):
        self.microscope = microscope
        self.experiment = experiment
        self.hook_manager = hook_manager
        # Who answers the run's questions: the session's responder. None
        # asks nobody: a supervised step runs as if unsupervised.
        self.responder = responder
        # Who can decide them now (``AppSession.deciders``); nobody without one.
        self._deciders = deciders
        # The latest image of each beam (``AppSession.images``), where a task
        # puts what it acquires; None keeps nothing.
        self.images = images
        # Who holds the run now (None: nobody), and whether it waits between
        # tasks: what ``held`` and ``pending`` last said.
        self.hold: Optional[Hold] = None
        self.waiting = False
        self._stop_event = StopEvent()
        # Stopping one task is a different intent from stopping the run, so it is a
        # different event. A flag on a single event would need care not to be
        # downgraded when a run-wide Stop arrives mid-unwind; with two, there is no
        # path by which cancelling a task can set is_stopped.
        self._task_stop_event = StopEvent()
        # Built once: the token's identity is captured by every task at construction.
        self._abort_token = AnyStopEvent(self._stop_event, self._task_stop_event)
        self.queue = TaskQueue(fixed_tasks=self.FIXED_TASKS)
        # Whether this run's end event has fired; no run, nothing to report.
        self._run_ended = True
        # Whether workflow_started has fired: a run that dies before it is
        # reported by end_unstarted_run (FIB-1353).
        self._run_began = False
        # The third way a run ends: not completed, not cancelled -- drained with
        # work that can never run. Read after the run by whoever launched it.
        self.stalled = False
        self.stall_reason = ""
        # (item, last iteration) pairs the app said have no next iteration: asked once,
        # not again on every pass of the loop.
        self._rounds_ended: Set[Tuple[str, Optional[int]]] = set()

    def _prepare(self) -> None:
        """The run's own set-up, done inside the run, after workflow_started:
        what fails here ends the run with workflow_failed like anything else
        that escapes it. In the constructor, a failed save ended a run the
        agent had been told was started with no event at all (FIB-1353).

        Stamps the experiment onto the images this run acquires. Done here
        rather than only in the UI so a run with no window
        records it too; the app also calls it when it adopts an experiment,
        and repeating it just re-sets the same two attributes. See FIB-449.

        No task starts while the experiment cannot be saved (FIB-1352):
        registering saves, and an experiment still unsaved after it is saved
        here, so a file that cannot be written fails the run before any work.
        """
        self.experiment.register_metadata(self.microscope)
        if self.experiment.unsaved:
            self.experiment.save()

    # --- Public API ---

    @contextmanager
    def _recording(self):
        """The event stream for this run (FIB-1044), the same with or without the
        GUI: the microscope's own recorder when something keeps one -- the app
        does, from connecting -- else one made for the run, recording to the
        experiment, and closed after it however the run ends. Its lifecycle hook
        is on this run's hook manager for the run, once. Recording never costs
        the run: a recorder that cannot be made is a run without one."""
        recorder, made = run_recorder(self.microscope, self.experiment)
        hook = recorder.lifecycle_hook if recorder is not None else None
        if hook is not None:
            if self.hook_manager is None:
                self.hook_manager = HookManager()
            self.hook_manager.register(hook)
        try:
            yield
        finally:
            if hook is not None:
                self.hook_manager.unregister(hook)
            if made is not None:
                made.close()

    def _stall(self) -> None:
        """Nothing in the queue can run, and nothing will change that: a task
        whose requirement is queued behind it. A bug in the plan rather than
        work left to do, so it is said as one and the run ends."""
        deferred = self.deferred_items()
        self.stalled = True
        self.stall_reason = f"{len(deferred)} task(s) cannot run: " + ", ".join(
            f"{i.item_name}/{i.task_name}" for i, _ in deferred
        )
        logging.error(self.stall_reason)

    def closing_note(self) -> str:
        """Why the run ended short of done, and what to do: shown on the
        workflow label, the status bar and the run summary's headline, so a
        run that gave up waiting never reads as a finish. Empty otherwise."""
        if not self.stalled:
            return ""
        return self.stall_reason

    def _set_hold(
        self,
        hold: Optional[Hold],
        note: Optional[str] = None,
        message: Optional[str] = "",
    ) -> None:
        """Say who holds the run (None: nobody), then post the status line,
        so a window's chrome -- border, attention button, status bar --
        redraws from it, the way a pending question does. ``note`` is the
        workflow line. ``message`` is what is said about the prompt bar: ""
        takes a stale prompt down; None says nothing about it, which a hold
        on a question shown *on* that bar needs."""
        self.hold = hold
        self.held.emit(hold)
        if hold is not None:
            post_status(
                self,
                message,
                workflow_info=note,
                status_bar=f"Waiting: {hold.releases}.",
            )
        else:
            post_status(self, message, status_bar="")

    @contextmanager
    def holding_a_proposal(self, item_name: str, task_name: str):
        """The run is held on a proposal a task asked mid-run (``ask``): the
        task is stopped on its next line until the decision lands. The
        workflow line says which task is stopped and who can decide it.

        A supervised proposal waits whoever is there, nobody included: it is
        never taken unreviewed. Yields ``refresh``, which the waiting task
        calls as it waits: when who can decide changes (the window opens or
        closes, the agent goes quiet or comes back), the hold and the record
        say so (``proposal_waiting``).

        Said on the record too (``proposal_held`` / ``proposal_released``), so
        a watcher hears that the run has stopped to be told something
        (FIB-1050)."""
        names = {"item_name": item_name, "task_name": task_name}
        shown: List[Tuple[ActorKind, ...]] = []

        def refresh() -> None:
            who = self.deciders()
            if shown and shown[-1] == who:
                return
            first = not shown
            shown.append(who)
            releases, note = _proposal_hold_text(item_name, task_name, who)
            self._set_hold(
                Hold(
                    kind=HoldKind.proposal,
                    releases=releases,
                    items=(f"{item_name}/{task_name}",),
                    agent_deciding=AGENT in who,
                ),
                note=note,
                message=None,
            )
            if not first:
                self._record_event(
                    "proposal_waiting",
                    {**names, "deciders": [w.value for w in who]},
                )

        refresh()
        self._record_event("proposal_held", dict(names))
        try:
            yield refresh
        finally:
            self._set_hold(None, message=None)
            self._record_event("proposal_released", dict(names))

    def _record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """A fact for the experiment's record, through the microscope's
        ``record_event`` (which never raises); nothing without one."""
        record = getattr(self.microscope, "record_event", None)
        if callable(record):
            record(kind, payload)

    # --- Iterations: items that repeat their tasks ---

    def _next_iteration(
        self, item_name: str, finished: Optional[int]
    ) -> Optional[Tuple[int, List[str]]]:
        """The next iteration of ``item_name``'s tasks, as (iteration, task names), now
        that the iteration ``finished`` has drained (None: its queued work had no
        iteration), or None when it has no more.

        An app whose items repeat their tasks -- a volume position, once per
        slice -- says here what comes next. By default an item runs its tasks
        once, as queued."""
        return None

    def _refill(self) -> bool:
        """Queue the next iteration for every item with nothing left waiting or
        running. Only one iteration per item is queued at a time, so a hold on one
        item's iteration parks that item while the others go on, and the queue
        never holds every iteration up front. True when anything was queued."""
        busy = set()
        iterations: Dict[str, List[int]] = {}
        for work in self.queue.items:
            if work.is_pending or work.status is TaskStatus.InProgress:
                busy.add(work.item_name)
            iterations.setdefault(work.item_name, [])
            if work.iteration is not None:
                iterations[work.item_name].append(work.iteration)
        added = False
        for item_name, queued in iterations.items():
            last = max(queued) if queued else None
            if item_name in busy or (item_name, last) in self._rounds_ended:
                continue
            following = self._next_iteration(item_name, last)
            if following is None:
                self._rounds_ended.add((item_name, last))
                continue
            iteration_, task_names = following
            for task_name in task_names:
                self.queue.add(item_name, task_name, iteration=iteration_)
            added = added or bool(task_names)
        if added:
            self.notify_queue_changed()
        return added

    def _next_item(self) -> Optional[WorkItem]:
        """The next runnable work item, or None: the queue, refilled first with
        any item's next iteration."""
        self._refill()
        return self.queue.next(skip=self._is_deferred)

    # --- Deferral: what cannot run yet, and why ---

    # How many of the items a hold names read, past three: "4 lamellae".
    ITEM_NOUN = "lamellae"
    # Steps the queue keeps where the plan put them (TaskQueue.fixed_tasks).
    FIXED_TASKS: Tuple[str, ...] = ()

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        """Why this pending item cannot run *yet*, or None. Each manager says
        what defers its items; the base waits on it the same way for both."""
        return None

    def _is_deferred(self, item: WorkItem) -> bool:
        """The queue's skip predicate: pass over, do not retire."""
        return self._item_defer_reason(item) is not None

    def deferred_items(self) -> List[Tuple[WorkItem, str]]:
        """Every pending item that cannot run now, with why. What a stalled run
        reports, and what the UI can label; derived here, never stored."""
        deferred = []
        for item in self.queue.pending:
            reason = self._item_defer_reason(item)
            if reason is not None:
                deferred.append((item, reason))
        return deferred

    def _report_stall(self) -> None:
        """The run drained with work that cannot run: the WORKFLOW_STALLED
        hook, the log, and the closing note on the label and the status bar."""
        self._fire_workflow_hook(HookEvent.WORKFLOW_STALLED)
        logging.warning(f"Workflow stalled: {self.stall_reason}")
        post_status(
            self,
            "",
            workflow_info=f"Workflow stalled: {self.closing_note()}",
            status_bar=f"Workflow stalled: {self.closing_note()}",
        )

    # What the run is called to whoever else wants the microscope while it
    # holds control: "Can't move: a lamella run is in control".
    RUN_DESCRIPTION = "a run"
    # The kind of run this manager runs (``fibsem.workflows.runner``): what
    # may join its queue while it runs.
    RUN_KIND = ITEMS

    def beams_needed(self) -> Tuple[BeamType, ...]:
        """The beams turned on before the run, if they are off. Both, unless
        the app's work needs fewer: a workflow that never uses the ion beam
        should not switch it on."""
        return (BeamType.ELECTRON, BeamType.ION)

    def _names_for_run(self, names: Optional[List[str]]) -> Optional[List[str]]:
        """The items to run over, resolved in control at the start of the run:
        ``names`` as given, unless the manager finds them itself."""
        return names

    def execute(self, task_names: List[str], names: Optional[List[str]] = None) -> None:
        """The whole run, as whoever starts one calls it; never raises.

        In control of the microscope for all of it (FIB-1270), the beams it
        needs turned on, then ``run``, then both beams off however it ended
        when the experiment says so. A failure before the run began -- someone
        else in control, a beam that will not come on -- still ends the run
        with an event, so whoever was told it started sees it end (FIB-1353).
        """
        try:
            with self.microscope.control.take(WORKFLOW, self.RUN_DESCRIPTION):
                try:
                    for beam in self.beams_needed():
                        if not self.microscope.is_on(beam):
                            self.microscope.turn_on(beam)
                    names = self._names_for_run(names)
                    logging.info(
                        f"Starting tasks: {task_names}, for {self.ITEM_NOUN}: {names}"
                    )
                    self.run(task_names, names)
                finally:
                    if self.experiment.turns_beams_off:
                        self.microscope.turn_off(BeamType.ELECTRON)
                        self.microscope.turn_off(BeamType.ION)
        except MicroscopeBusy as e:
            logging.warning(f"Workflow not started: {e}")
            self._end_unstarted_quietly(e)
        except (InterruptedError, OperationCancelledError) as e:
            # A Stop, not a failure: anyone scanning the log for problems must
            # not see one per Stop press. The task layer has already recorded
            # the outcome as Cancelled.
            logging.info(f"Workflow cancelled: {e}")
            self._end_unstarted_quietly(e)
        except Exception as e:
            logging.error(f"Error during the workflow: {e}", exc_info=True)
            self._end_unstarted_quietly(e)

    def run(self, task_names: List[str], names: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    # --- The run loop: the same for every app; the steps below are each app's ---

    # The line the workflow label closes a stopped run with.
    CANCELLED_LINE = "Workflow cancelled by user."
    # What the status reports say as a task starts, completes and fails, with
    # ``{task}`` and ``{item}``.
    TASK_LINES = (
        "Starting {task} on {item}.",
        "Completed {task} on {item}.",
        "Error in {task} on {item}.",
    )

    def _run_queue(self) -> None:
        """Process queue items until empty or stopped."""
        self._before_run()
        with self._recording():
            self._fire_workflow_hook(HookEvent.WORKFLOW_STARTED)
            try:
                self._prepare()
                self._run_items()
            except Exception as e:
                self._end_escaped_run(e)
                raise

    def _run_items(self) -> None:
        while not self.is_stopped:
            item = self._next_item()
            if item is None:
                if not self.queue.is_empty:
                    self._stall()
                break

            # A stop_task click that landed between two tasks was aimed at the one
            # that has just finished, not at this one.
            self._task_stop_event.clear()

            record = self._record_for(item)
            if record is None or not self._ready(item, record):
                continue

            start, done, failed = self.TASK_LINES
            lines = {"task": item.task_name, "item": record.name}
            self._emit_report(
                item=item,
                item_name=record.name,
                status=TaskStatus.InProgress,
                msg=start.format(**lines),
            )
            err = self._run_single_task(item.task_name, record)
            final_status = record.task_state.status
            self.queue.mark_done(item, final_status)
            self._emit_report(
                item=item,
                item_name=record.name,
                status=final_status,
                error_message=record.task_state.status_message,
                task_duration=record.task_state.duration,
                msg=(done if err is None else failed).format(**lines),
            )
            self._after_task(item, record)

        self._end_run()

    def _before_run(self) -> None:
        """Before the run's first task."""

    def _record_for(self, item: WorkItem) -> Optional[Any]:
        """The record ``item`` runs on; None once the app has retired an item
        whose record is gone."""
        raise NotImplementedError

    def _ready(self, item: WorkItem, record: Any) -> bool:
        """The app's steps before a task starts. False when it retired the item
        itself (skipped it, with its report and hook) or a stop came first."""
        return True

    def _run_task(self, task_name: str, record: Any) -> None:
        """Run one task on one record; raises what the task raised."""
        raise NotImplementedError

    def _after_task(self, item: WorkItem, record: Any) -> None:
        """After a task's end is reported."""

    def _record_unstarted(self, record: Any, task_name: str) -> None:
        """The history entry for a task that failed before it existed; by
        default none."""

    def _run_single_task(self, task_name: str, record: Any) -> Optional[Exception]:
        """Execute one task on one record. Returns the exception, or None."""
        recorded = len(record.task_history)
        try:
            self._run_task(task_name, record)
        except Exception as e:
            # The task records its own outcome before re-raising (FIB-490); this
            # is the fallback for what fails before a task exists -- an unknown
            # task name, a construction failure. The cancellation predicate is
            # the task's, so the live state and the history cannot disagree.
            if self.should_abort or isinstance(
                e, (OperationCancelledError, InterruptedError)
            ):
                logging.info(f"Task {task_name} on {record.name} cancelled by user.")
                record.task_state.status = TaskStatus.Cancelled
                record.task_state.status_message = "Cancelled by user."
            else:
                logging.warning(
                    f"Error running task {task_name} on {record.name}: {e}",
                    exc_info=True,
                )
                record.task_state.status = TaskStatus.Failed
                record.task_state.status_message = str(e)
            # A task that failed before it existed left no history entry.
            if len(record.task_history) == recorded:
                self._record_unstarted(record, task_name)
            self.experiment.save()
            return e
        try:
            self.experiment.save()
        except Exception as e:
            self._result_not_saved(record, task_name, e)
            raise
        return None

    def _end_run(self) -> None:
        """The run's closing: cancelled, stalled or completed. The loop exits
        when the queue drains *or* on Stop; only one of those is a finish."""
        if self.is_stopped:
            self._fire_workflow_hook(HookEvent.WORKFLOW_CANCELLED)
            self._say(workflow_info=self.CANCELLED_LINE)
        elif self.stalled:
            # Drained with work that cannot run: not completed, not cancelled.
            # The next Run picks up from there.
            self._report_stall()
        else:
            self._complete()

    def _complete(self) -> None:
        """A run that drained with nothing left to do."""
        self._fire_workflow_hook(HookEvent.WORKFLOW_COMPLETED)
        self._say(workflow_info=self._completion_message())

    def _completion_message(self) -> str:
        raise NotImplementedError

    def _say(
        self, workflow_info: Optional[str] = None, status_bar: Optional[str] = None
    ) -> None:
        """A line for the workflow-information label or the status bar, that
        gets out after a Stop too: the closing "cancelled" line is the point."""
        post_status(self, "", workflow_info=workflow_info, status_bar=status_bar)

    def _end_unstarted_quietly(self, error: BaseException) -> None:
        try:
            self.end_unstarted_run(error)
        except Exception:
            logging.exception("reporting the end of a run that never began failed")

    def stop(self) -> None:
        """End the run once the task now running unwinds, and stop its mill."""
        self._stop_event.set()
        self._stop_milling()

    def stop_task(self) -> None:
        """Abandon the task now running and carry on with the rest of the queue.

        The task unwinds through the same path a run-wide Stop uses, so it ends
        Cancelled and whatever it was doing to the hardware is undone the same
        way. Only the loop's reaction differs: is_stopped is untouched, so
        _run_queue moves to the next item instead of ending the run.

        Cleared at the top of each task, so a click that lands between two tasks
        is discarded rather than killing the next one.
        """
        self._task_stop_event.set()
        self._stop_milling()

    def _stop_milling(self) -> None:
        """Stop the beam of a mill now, through the milling service, rather
        than at its next look at the task's stop."""
        milling = self.microscope.milling
        if milling is None:
            return
        try:
            milling.stop()
        except NothingToStop:
            pass
        except Exception:
            logging.exception("Stopping the mill failed")

    def deciders(self) -> Tuple[ActorKind, ...]:
        """Who can decide this run's proposals now: the session's surfaces
        (``AppSession.deciders``). Empty: nobody, and a supervised proposal
        waits for someone to come back."""
        return self._deciders() if self._deciders is not None else ()

    @property
    def is_stopped(self) -> bool:
        """Whether the *run* should end. Only _run_queue's loop asks this.

        Deliberately narrow. "Should this task unwind?" is a different question
        with a different answer -- see `should_abort` -- and conflating the two
        is what makes any cancel end the whole run.
        """
        return self._stop_event.is_set()

    @property
    def should_abort(self) -> bool:
        """Whether the task now running should unwind.

        For the callers with no token to poll: the helpers in
        autolamella/workflows/ui.py, which get the manager. Everything task-side
        reads `abort_token` instead, which answers the same question.
        """
        return self.abort_token.is_set()

    @property
    def abort_token(self) -> AnyStopEvent:
        """What anything inside a task polls to know it has been cancelled.

        Captured by AutoLamellaTask and GridTask, handed on to milling and
        autofocus via `raise_if_cancelled`, and read directly by the fluorescence,
        coincidence-milling and grid tasks. `is_set()` is what they call; a
        wait on it (`wait`) wakes the moment either stop is set.
        """
        return self._abort_token

    def notify_queue_changed(self) -> None:
        """Say the queue's contents changed, outside the task lifecycle.

        Status updates only fire when a task starts, finishes or is skipped, so
        an edit made between two tasks would otherwise stay invisible until the
        next one began.

        Deliberately its own signal rather than the status channel: a queue
        edit is not a step in the task lifecycle, and the status handler
        repaints run chrome an edit must not touch.
        """
        self.queue_changed.emit(
            {
                "queue_items": self.queue.items,  # thread-safe snapshot
                "version": self.queue.version,
            }
        )

    def hook_run_context(self) -> dict:
        """The fields every hook event from this run carries: which experiment, and
        how much of the queue is left.

        Public because AutoLamellaTask fires task events itself and needs the same
        fields; a task reaches this through its task_manager.

        tasks_remaining excludes the task the event is about: queue.next() marks an
        item InProgress before the task runs, so the one in flight is already out of
        the pending set on every path, skips included.
        """
        pending, total = self.queue.counts
        return {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "tasks_remaining": pending,
            "tasks_total": total,
            "iteration": self.active_iteration,
        }

    @property
    def active_iteration(self) -> Optional[int]:
        """The iteration of the queue entry running now; None outside one, or for
        an item that does not run in iterations."""
        active = self.queue.active
        return active.iteration if active is not None else None

    def _fire_workflow_hook(self, event: HookEvent, **fields) -> None:
        # Workflow events used to carry nothing at all, so workflow_completed reached a
        # webhook with no way to tell which experiment had finished.
        if event == HookEvent.WORKFLOW_STARTED:
            self._run_began = True
            self._run_ended = False
        elif event in _RUN_ENDS:
            self._run_ended = True
        fire_event(self.hook_manager, event, **fields, **self.hook_run_context())

    def _end_escaped_run(self, error: BaseException) -> None:
        """Report the end of a run an exception escaped, unless it was reported.

        Every run ends with exactly one of the run-end events (FIB-1327): one
        that dies outside a task's own handling -- a save that fails, the
        objective retract -- used to end with none, and anything waiting for
        the end waited forever. A Stop that unwinds this far is a cancellation.
        """
        if self._run_ended:
            return
        self._fire_end_for(error)

    def end_unstarted_run(self, error: BaseException) -> None:
        """Report a run that died in the launcher's set-up, before it began.

        ``execute`` does work of its own before calling ``run``: taking
        control, turning the beams on, the grid inventory. A failure there used to end a run
        that start_workflow had already answered ``started`` for with no event
        (FIB-1353). It ends with workflow_failed now, or workflow_cancelled
        for a Stop, through this run's recording. A run that began reports
        its own end, so this does nothing then.
        """
        if self._run_began:
            return
        self._run_began = True
        with self._recording():
            self._fire_end_for(error)

    def _fire_end_for(self, error: BaseException) -> None:
        if self.should_abort or isinstance(
            error, (OperationCancelledError, InterruptedError)
        ):
            self._fire_workflow_hook(HookEvent.WORKFLOW_CANCELLED)
        else:
            self._fire_workflow_hook(HookEvent.WORKFLOW_FAILED, error=str(error))

    def _fire_skipped_hook(
        self,
        task_name: str,
        item_name: str,
        skip_reason: str,
        task_type: str = "",
        item_id: str = "",
        missing_prereqs: Optional[List[str]] = None,
    ) -> None:
        """Fire TASK_SKIPPED. Unlike the other task events this cannot come from
        AutoLamellaTask, because the skip is decided before any task object exists —
        so there is no task_state to attach, and no task_id: nothing ran to have one.
        item_id is empty on the lamella-not-found path, where there is no lamella."""
        fire_event(
            self.hook_manager,
            HookEvent.TASK_SKIPPED,
            task_name=task_name,
            task_type=task_type,
            item_name=item_name,
            item_id=item_id,
            skip_reason=skip_reason,
            missing_prereqs=missing_prereqs,
            **self.hook_run_context(),
        )

    def _result_not_saved(self, item: Any, task_name: str, error: Exception) -> None:
        """A task that finished, whose result the save after it could not
        write: it did not complete (FIB-1374). Recorded Failed on the item and
        its history entry, and announced with task_failed, so the record and
        the event stream agree that it failed; the caller re-raises, which
        ends the run."""
        state = item.task_state
        take_back_finish(
            item,
            state.task_id,
            TaskStatus.Failed,
            f"Finished, but its result could not be saved: {error}",
        )
        fire_event(
            self.hook_manager,
            HookEvent.TASK_FAILED,
            task_name=task_name,
            task_type=state.task_type,
            item_name=item.name,
            item_id=item.id,
            task_id=state.task_id,
            task_state=state,
            error=str(error),
            **self.hook_run_context(),
        )

    # --- Internal helpers ---

    def _set_workflow_pending(self, pending: bool) -> None:
        """Say the run is active but nothing is executing.

        Read by a window's workflow border, which would otherwise show the
        running colour throughout a scheduled wait that can last hours.
        """
        self.waiting = pending
        self.pending.emit(pending)

    def _emit_report(
        self,
        item: WorkItem,
        item_name: str,
        status: "TaskStatus",
        msg: str = "",
        error_message: Optional[str] = None,
        task_duration: Optional[float] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        """Emit one lifecycle report on ``status``, for any kind of item.

        Fire-and-forget on the status channel: a report that a task started is
        not a step in any request's handshake, so it must not be able to touch
        one — questions and instructions live on the responder's futures.

        This stream and the hook stream describe the same lifecycle from two different
        brackets — the manager brackets the task *call*, the task brackets its own
        *execution* — and they are deliberately not merged. Hooks run user-configurable
        code synchronously on the worker thread and HookManager.fire swallows what it
        raises; the UI needs neither of those properties. See FIB-464.

        What the two must agree on is vocabulary: the item is item_name here as it is in
        HookContext. The queue position fields below have no hook counterpart and stay —
        they are timeline positioning, not lifecycle facts.
        """
        # Progress is measured against the live queue rather than the launch
        # plan. A position in the original task x lamella matrix has no answer
        # for an item the user added mid-run, and stops being meaningful for
        # everything else as soon as the queue is reordered.
        items = self.queue.items
        position = next((i for i, it in enumerate(items) if it.id == item.id), None)

        update = WorkflowStatusUpdate(
            task_name=item.task_name,
            item_name=item_name,
            status=status,
            timestamp=time.time(),
            error_message=error_message,
            task_duration=task_duration,
            skip_reason=skip_reason,
            # Already 1-based on the wire. Consumers render it as-is.
            queue_position=position + 1 if position is not None else None,
            queue_total=len(items),
            queue_items=items,  # thread-safe snapshot: Queue.items returns copies
            work_item_id=item.id,
            iteration=item.iteration,
        )

        self.status.emit(WorkflowStatusEvent(message=msg, report=update))


def _named(names: List[str], noun: str = "lamellae") -> str:
    """'01-a', '01-a and 02-b', '01-a, 02-b and 03-c', '4 lamellae'."""
    if len(names) > 3:
        return f"{len(names)} {noun}"
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _proposal_hold_text(
    item_name: str, task_name: str, who: Tuple[ActorKind, ...]
) -> Tuple[str, str]:
    """What releases a hold on a proposal, and the workflow line, given who
    can decide it now."""
    asked = f"{task_name} on {item_name}"
    names = [
        name for kind, name in ((OPERATOR, "you"), (AGENT, "the agent")) if kind in who
    ]
    if not names:
        return (
            f"nobody can answer {asked}: open the app, or connect the agent at Drive",
            f"{task_name} is waiting for an answer on {item_name}. Nobody can "
            "answer it: open the app, or connect the agent at Drive.",
        )
    by = " or ".join(names)
    return (
        f"answer {asked}",
        f"{task_name} is waiting for {by} to answer on {item_name}.",
    )
