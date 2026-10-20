"""One run of a task on one item, whatever the item is.

``pre_task -> _run -> post_task``. A failed or cancelled ``_run`` is recorded on
the item's history before the exception goes on, and a run that finished but
whose result could not be recorded is taken back. What a task needs decided
while it runs, it asks (``ask``); a finished task is its history and outputs.

An application's task base subclasses ``BaseTask``: AutoLamella's lamella and
grid tasks are two. The item is anything with a ``name``, an ``id``, a
``task_state``, a ``task_history`` and ``proposals``.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union

from fibsem.acting import TASK, acting
from fibsem.cancellation import AnyStopEvent, OperationCancelledError, StopEvent
from fibsem.control import lending
from fibsem.milling.tasks import (
    FibsemMillingTaskConfig,
    record_milling_skipped,
    run_milling_task,
)
from fibsem.structures import FibsemImage, FibsemRectangle
from fibsem.util.timestamps import now, now_iso
from fibsem.workflows.interaction import (
    INSTRUCTION_TIMEOUT_S,
    ClearMillingConfig,
    Responder,
    SetMillingConfig,
    _abort_requested,
    ask,
)
from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    ACTION_SKIP,
    ALIGNMENT_AREA,
    MILLING,
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.workflows.status import TaskStatus, take_back_finish

# Characters a folder name cannot hold on Windows (and ``/`` anywhere).
_UNSAFE_IN_FOLDER_NAMES = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def iteration_folder(iteration: int) -> str:
    """The folder name of one iteration of an item's tasks: ``0003``."""
    return f"{iteration:04d}"


def run_folder(
    item_dir: Any, task_name: str, started: datetime, iteration: Optional[int] = None
) -> str:
    """Where one run of ``task_name`` on an item saves what it makes:
    ``<item>/<task name>/<YYYYMMDD-HHMMSS>/`` (FIB-1275), so a re-run
    overwrites nothing; ``<item>/<task name>/<iteration:04d>/`` for an item that
    runs in iterations, one folder per iteration. The item's shared state
    (``ref_alignment``, the coincidence setup reference) is not a run's, and
    stays in the item's folder (FIB-1258).

    The task name is made safe for a folder name; a second run of the same
    second, or the same iteration, as one already saved gets ``-2``. Not created
    here: a run that saves nothing leaves no folder. Readers find a run's files
    from its record (``outputs``), not from this name.
    """
    task_folder = _UNSAFE_IN_FOLDER_NAMES.sub("-", task_name).strip().rstrip(".")
    leaf = (
        iteration_folder(iteration)
        if iteration is not None
        else f"{started:%Y%m%d-%H%M%S}"
    )
    base = os.path.join(str(item_dir), task_folder or "task", leaf)
    folder, n = base, 1
    while os.path.exists(folder):
        n += 1
        folder = f"{base}-{n}"
    return folder


# How often a question's hold re-reads who can answer while it waits.
_HOLD_REFRESH_S = 1.0

# Steps recorded as the task's own lifecycle events (task_started,
# task_completed), not as task_step.
LIFECYCLE_STEPS = {"STARTED", "FINISHED"}


class BaseTask(ABC):
    """The lifecycle every task shares. Subclasses implement ``_run`` and say
    whether the run is supervised; ``pre_task`` and ``post_task`` start and
    finish the item's task state, and an app may extend them."""

    config: Any
    # What the item is called in the log and in task_step events.
    item_type: ClassVar[str] = "item"
    # What this task type asks *while it runs*, as proposal kinds: the values
    # it needs answered before its next line (a detection, then the stage
    # move that follows it). Declared on the type, so what needs a person
    # present is known before the run. ``ask`` refuses a kind that is not here.
    asks: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def asks_for(cls, config: Any) -> Tuple[str, ...]:
        """The kinds this task asks under ``config``: a setting may turn one
        off (Setup's ``confirm_alignment_area``). By default, everything declared."""
        return cls.asks

    def __init__(
        self,
        microscope: Any,
        config: Any,
        item: Any,
        experiment: Optional[Any] = None,
        task_manager: Optional[Any] = None,
    ) -> None:
        self.microscope = microscope
        self.config = config
        self.item = item
        self.experiment = (
            experiment
            if experiment is not None
            else getattr(task_manager, "experiment", None)
        )
        self.task_manager = task_manager
        self.task_id = str(uuid.uuid4())
        # The iteration of the item's tasks this run is, when the item runs in
        # iterations: the queue entry the manager is running.
        self.iteration: Optional[int] = getattr(task_manager, "active_iteration", None)
        # The manager's token, not its raw event: what counts as "cancelled" is
        # the manager's to decide, and a task should not have to know.
        self._stop_event = task_manager.abort_token if task_manager else None
        # What the history entry says on success, when there is more to say
        # than "Finished".
        self.completion_note = ""
        # The last FIB image this run acquired: where a proposal sits by default.
        self._last_fib_image: Optional[FibsemImage] = None

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @property
    def display_name(self) -> str:
        return self.config.display_name

    @property
    @abstractmethod
    def supervised(self) -> bool:
        """Whether this task's attention is supervised: then its proposals are
        asked of whoever can decide while it runs."""

    # -- proposals -------------------------------------------------------------

    def ask(
        self,
        kind: str,
        values: Dict[str, Any],
        *,
        image: str = "",
        provenance: Optional[Dict[str, Any]] = None,
        message: str = "",
        decided: Optional[Callable[[], Dict[str, Any]]] = None,
        enabled: bool = True,
    ) -> Decision:
        """Ask a proposal of ``values`` of ``kind`` and wait for the decision,
        on the record.

        The one way a task asks for a value it needs before it can carry on.
        The proposal goes on the item (``values`` are what the task would use
        as they stand; ``image`` the saved file they sit on, relative to the
        item's folder), and the decision on it comes through
        ``Experiment.decide`` like every other: from the widget the kind is
        decided on (the prompt bar, the Detection tab, the milling editor), or
        from a connected agent. Returned as the decision, so the
        task reads ``.outcome`` and ``.values``; a task applies the decided
        values itself, so a kind asked here must not write them through.

        Supervised: the run holds here -- the manager raises the hold --
        until the decision lands, or Stop, which withdraws the proposal and
        raises ``InterruptedError``. It holds whoever can decide, nobody
        included: it is never taken unreviewed, so with no window and no agent
        listening at Drive it waits for one to come back. Stop and the
        decision wake it at once, and the hold says who can decide as that
        changes.
        Automated, nobody is asked: the proposal goes on the record with an
        ``Unreviewed`` decision carrying the proposed values, and returns at
        once. A declared kind that
        ``asks_for`` leaves out under this config is the same: recorded,
        not asked.

        ``decided`` is for a confirmation: a decision that says "as it stands"
        and carries no values of its own (a state, confirmed from the prompt
        bar). The task, which can read the instrument, is asked for the
        values as they stand once the confirmation lands, and they go on the
        decision, so a change the operator made before confirming is the
        delta. Not called for a decision that already carries values.

        ``enabled`` is the ask's own switch in the task's settings
        (``confirm_position``, say), for a task that asks the same kind more
        than once: off, this one is recorded and not asked, as a kind
        ``asks_for`` leaves out is.
        """
        declared = type(self).asks
        if kind not in declared:
            raise ValueError(
                f"{type(self).__name__} does not declare {kind!r} in asks "
                f"(it declares {list(declared)}); a task says what it asks."
            )
        carried = PROPOSAL_KINDS[kind].values
        unknown = [n for n in values if n not in carried]
        if unknown:
            raise ValueError(f"{kind} does not carry {unknown}; it carries {carried}")
        manager = self.task_manager
        experiment = self.experiment
        if manager is None or experiment is None:
            raise RuntimeError("ask needs a task manager with an experiment")

        item = self.item
        proposal = Proposal(
            kind=kind,
            values=dict(values),
            provenance={
                "proposer": self.task_name,
                "task_id": item.task_state.task_id or self.task_id,
                "reference_image": image,
                "message": message,
                **(provenance or {}),
            },
        )
        reason = self.unasked_reason(kind, enabled=enabled)
        if reason:
            experiment.record_unasked(item.id, self.task_name, proposal, reason)
            return proposal.current
        proposal.filling_in = decided is not None

        answered = StopEvent()
        # Woken by the answer or by either stop, whichever comes first.
        wake = AnyStopEvent(answered, *([self._stop_event] if self._stop_event else []))

        def _on_decided(item_id: str, task_name: str) -> None:
            # This question, and decided: ``decided`` also fires for the
            # previous question's fill-in, which can land after this one was
            # recorded when the task asks twice in a row (FIB-1053).
            if (item_id, task_name) == (item.id, self.task_name):
                current = item.proposal(task_name)
                if (
                    current is not None
                    and current.id == proposal.id
                    and current.current is not None
                ):
                    answered.set()

        experiment.decided.connect(_on_decided)
        try:
            # While the question stands, the run lends its control of the
            # microscope: whoever answers may act first (FIB-1270). Lent before
            # the question is put, so nobody who sees it is refused.
            with lending():
                if not experiment.ask_proposal(item.id, self.task_name, proposal):
                    raise RuntimeError(f"could not record the {kind} question")
                with manager.holding_a_proposal(item.name, self.task_name) as refresh:
                    while not answered.is_set():
                        if self._stop_event is not None and self._stop_event.is_set():
                            experiment.withdraw_proposal(
                                item.id,
                                self.task_name,
                                "the run stopped before it was answered",
                            )
                            raise InterruptedError(f"{kind} question cancelled")
                        # Who can answer changes without an event (the agent
                        # going quiet), so the hold is re-read now and then.
                        wake.wait(_HOLD_REFRESH_S)
                        refresh()
        finally:
            experiment.decided.disconnect(_on_decided)
        decision = proposal.current
        if decision is None or decision.outcome is DecisionOutcome.Withdrawn:
            raise InterruptedError(f"{kind} question withdrawn before it was answered")
        if (
            decided is not None
            and decision.outcome is DecisionOutcome.Confirmed
            and not decision.values
        ):
            values: Dict[str, Any] = {}
            try:
                values = dict(decided())
            finally:
                # Filled in or not, the decision goes on the record once.
                experiment.fill_in_decision(
                    item.id, self.task_name, proposal.id, values
                )
        return decision

    def unasked_reason(self, kind: str, *, enabled: bool = True) -> str:
        """Why a question of ``kind`` would be recorded rather than asked now;
        empty when it would be asked. Asked when this config asks it and the
        task is supervised, whether or not anyone can answer yet: a supervised
        question waits for an answer (``ask``)."""
        if not enabled or kind not in type(self).asks_for(self.config):
            return "turned off in the task's settings"
        if not self.supervised:
            return "nobody was asked: the task is not supervised"
        return ""

    # -- the milling loop and the alignment area ------------------------------
    # Shared by every app's tasks; each writes where its own ``output_dir`` says.

    @property
    def responder(self) -> Optional[Responder]:
        """Who decides this task's proposals: the run's (the session's
        responder), or None outside a run."""
        return getattr(self.task_manager, "responder", None)

    def update_milling_config_ui(
        self,
        milling_config: FibsemMillingTaskConfig,
        msg: str = "Mill the task.",
        milling_enabled: bool = True,
    ) -> FibsemMillingTaskConfig:
        """Mill ``milling_config``, asking first when supervised; return the config
        as last used.

        The mill and its loop are the task's, on this thread (FIB-1236), headless
        and with a window alike. Each pass is a ``milling`` proposal through
        ``ask``, on the record: the config this pass would mill, decided as run
        (mill the config as decided, then ask again with the pass counted and
        the error, if the mill failed), edit (ask again with the config as
        decided), skip or accept (end the loop). Whoever decides, the operator
        at the editor or an agent, changes what is milled by deciding with it.
        Asked of nobody, a pass is recorded as the kind's silence: the first
        mills once, the next accepts, and a failed mill then fails the task
        (FIB-1112).

        Each run records how it ended (``FibsemMillingTask``). A task that mills
        nothing records that too, as a skipped ``milling`` event: with milling
        turned off, or skipped at the proposal (FIB-1257).
        """
        if not milling_enabled:
            record_milling_skipped(
                self.microscope, milling_config, "milling is turned off for this task"
            )
        responder = self.responder
        # The task's own config until a decision replaces it: an unasked mill
        # runs it, so what the run records on it (a strategy's end reason)
        # stays with the task even when the run is stopped. The proposal only
        # ever carries, and is decided with, copies.
        config = milling_config
        passes, error, watched = 0, None, None
        # What the next pass is asked on: after a pass, the frame of its result.
        pass_image = ""
        try:
            while True:
                if not milling_enabled:
                    actions = (ACTION_EDIT, ACTION_ACCEPT)
                elif passes == 0:
                    actions = (ACTION_RUN, ACTION_EDIT, ACTION_SKIP)
                else:
                    actions = (ACTION_RUN, ACTION_EDIT, ACTION_ACCEPT)
                if responder is not None:
                    # The editor shows what this pass would mill.
                    self._set_milling_config_ui(config)
                asked = deepcopy(config)
                decision = self.ask(
                    MILLING,
                    {"milling_config": asked},
                    image=pass_image or self._last_fib_image_file(),
                    provenance={
                        "actions": list(actions),
                        "passes": passes,
                        "last_error": None if error is None else str(error),
                        "enabled": milling_enabled,
                    },
                    message=msg,
                    decided=lambda: {"milling_config": asked},
                )
                if decision.outcome is DecisionOutcome.Rejected:
                    raise RuntimeError(
                        f"{self.task_name} was rejected: "
                        f"{decision.reason or 'no reason given'}"
                    )
                if decision.outcome is not DecisionOutcome.Unreviewed:
                    watched = decision
                if decision.action == ACTION_ACCEPT and passes:
                    if error is not None and watched is None:
                        # Nobody looked: the task fails, rather than recording
                        # Completed with nothing milled (FIB-1112).
                        raise error
                    # The config as milled, not an edit made after it: the
                    # protocol must not claim what never ran (FIB-580).
                    break
                if watched is decision:
                    # Unreviewed, it is decided as proposed: the task's own
                    # config is milled, not the record's copy.
                    config = decision.values.get("milling_config", config)
                if decision.action == ACTION_EDIT:
                    continue
                if decision.action != ACTION_RUN:
                    break
                error = self._mill(config)
                passes += 1
                pass_image = self._image_after_pass(
                    config, passes, take=not self.unasked_reason(MILLING)
                )
        finally:
            if responder is not None:
                self.clear_milling_config_ui()
        if milling_enabled and not passes:
            record_milling_skipped(
                self.microscope, config, "skipped at the milling question"
            )
        return config

    def _mill(self, config: FibsemMillingTaskConfig) -> Optional[Exception]:
        """Run the mill once, under the task's stop; its error if it failed. A
        Stop of the workflow during it raises once it has ended."""
        self.log_status_message("MILLING", f"Milling {config.name}...")
        task = run_milling_task(
            self.microscope,
            config,
            stop_event=self._stop_event,
            output_dir=self.output_dir,
        )
        self._check_for_abort()
        if task.error is not None:
            self.log_status_message(
                "MILLING", f"Milling {config.name} Failed: {task.error}"
            )
        else:
            self.log_status_message(
                "MILLING",
                f"Milling {config.name} Complete: "
                f"{len(config.enabled_stages)} stages completed.",
            )
        return task.error

    def _set_milling_config_ui(self, milling_config: FibsemMillingTaskConfig):
        """Set the milling config in the milling widget."""
        if self.responder is None:
            return

        self._check_for_abort()

        # deepcopy kept from the signal days: the editor's copy must be the
        # operator's to edit without the task's own config moving under it.
        ask(
            self.responder,
            SetMillingConfig(config=deepcopy(milling_config)),
            abort=lambda: _abort_requested(self.task_manager),
            timeout=INSTRUCTION_TIMEOUT_S,
        )

    def clear_milling_config_ui(self):
        """Clear the milling config from the milling widget."""
        if self.responder is None:
            return

        ask(
            self.responder,
            ClearMillingConfig(),
            abort=lambda: _abort_requested(self.task_manager),
            timeout=INSTRUCTION_TIMEOUT_S,
        )

    def _last_fib_image_file(self) -> str:
        """The last FIB reference image, relative to the item's folder, for
        a proposal to sit on; empty when none has been saved yet."""
        image = self._last_fib_image
        if image is None or image.filepath is None:
            return ""
        return os.path.relpath(image.filepath, self.path)

    def _image_after_pass(
        self, config: FibsemMillingTaskConfig, passes: int, take: bool = True
    ) -> str:
        """The frame the proposal after a pass sits on, relative to the item's
        folder: the one the mill saved when it finished; empty for none. *take*
        says the proposal will be asked, so an app that can take a frame here
        (`AutoLamellaTask`) takes one rather than show the pre-mill image."""
        return self._milling_result_image_file(config)

    def _milling_result_image_file(self, config: FibsemMillingTaskConfig) -> str:
        """The FIB image a milling session left behind (the ``finished``
        acquisition, when its config acquires one), relative to the item's
        folder; empty when it saved none.

        Looked for where the milling saved it -- its config's imaging path, the
        run's ``Milling/<task>`` folder -- then in the item's folder. Only the
        item's folder was searched, where the milling task never saves, so a
        Fiducial that acquired after milling still took an extra frame to ask
        its alignment area on."""
        imaging = getattr(getattr(config, "acquisition", None), "imaging", None)
        filename = str(getattr(imaging, "filename", "") or "")
        if filename:
            for folder in (getattr(imaging, "path", None), self.path):
                if not folder:
                    continue
                for candidate in sorted(
                    glob.glob(os.path.join(str(folder), f"{filename}*_ib.tif"))
                ):
                    return os.path.relpath(candidate, self.path)
        return ""

    def _validate_alignment_area(
        self, *, image: str = "", enabled: bool = True
    ) -> None:
        """Check the alignment area with the operator before the alignment
        reference is taken in it.

        An ``alignment_area`` proposal on ``image`` -- the last FIB reference
        image unless the caller names the frame -- decided by dragging the
        rectangle on the Microscope tab's FIB view, then Continue; the decided
        area is what the task acquires with. A rejection (an agent's) fails
        the task. ``enabled`` is the task's own switch.
        """
        self.log_status_message(
            "VALIDATE_ALIGNMENT_AREA", "Validating Alignment Image..."
        )
        decision = self.ask(
            ALIGNMENT_AREA,
            {"alignment_area": deepcopy(self.item.alignment_area)},
            image=image or self._last_fib_image_file(),
            message="Check the alignment area: the reference for later "
            "alignments is taken in it. Drag it to correct it.",
            enabled=enabled,
        )
        if decision.outcome is DecisionOutcome.Rejected:
            raise RuntimeError(
                f"{self.task_name} was rejected: {decision.reason or 'no reason given'}"
            )
        area = decision.values.get("alignment_area")
        if isinstance(area, FibsemRectangle):
            self.item.alignment_area = area

    # -- lifecycle -------------------------------------------------------------

    # Everything the task does, on its own thread, is the task's (FIB-1062).
    @acting(TASK)
    def run(self) -> None:
        self.pre_task()
        self._fire_hook("task_started")
        try:
            self._run()
        except Exception as e:
            # A user Stop is a cancellation, not a failure — fire the distinct hook so it
            # notifies as "cancelled" (warning) rather than "FAILED" (error).
            cancelled = self._is_cancellation(e)
            # Record before announcing. The hook payload carries task_state, which
            # still reads InProgress from pre_task until this runs.
            #
            # Guarded: anything raising in here would replace the task's own
            # exception with this one, losing what actually went wrong.
            try:
                self.item.task_state.status = (
                    TaskStatus.Cancelled if cancelled else TaskStatus.Failed
                )
                self.item.task_state.status_message = (
                    "Cancelled by user." if cancelled else str(e)
                )
                self._record_outcome()
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook(
                "task_cancelled" if cancelled else "task_failed", error=str(e)
            )
            raise
        finally:
            # In a finally, not in post_task, which never runs when _run() raises. Left
            # set, the next thing acquired -- an operator looking at what went wrong,
            # a supervision step -- would be stamped with the task that just failed.
            # That is silently wrong, and wrong exactly when the record matters most.
            self._clear_workflow_metadata()
        try:
            self.post_task()
        except Exception as e:
            # The work finished, but what follows it failed. It did not
            # complete (FIB-1374).
            cancelled = self._is_cancellation(e)
            try:
                take_back_finish(
                    self.item,
                    self.task_id,
                    TaskStatus.Cancelled if cancelled else TaskStatus.Failed,
                    "Cancelled by user."
                    if cancelled
                    else f"Finished, but its result could not be recorded: {e}",
                )
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook(
                "task_cancelled" if cancelled else "task_failed", error=str(e)
            )
            raise
        self._fire_hook("task_completed")

    @abstractmethod
    def _run(self) -> None: ...

    def pre_task(self) -> None:
        logging.info(
            f"Running {self.task_name}, {self.task_type} ({self.task_id}) "
            f"for {self.item_type} {self.item.name} ({self.item.id})"
        )
        self._start_task_state()
        self._set_workflow_metadata()
        self.log_status_message("STARTED", "Started")

    def post_task(self) -> None:
        self.item.task_state.status = TaskStatus.Completed
        self.item.task_state.status_message = ""
        self.log_status_message("FINISHED", self.completion_note or "Finished")
        self._record_outcome()

    # -- where the item's files go ---------------------------------------------

    @property
    def path(self) -> Path:
        """The item's directory: what its recorded outputs are relative to."""
        return Path(self.experiment.item_path(self.item))

    def record_output(self, role: str, output: Union[str, Path, Any, None]) -> None:
        """Record a file this run wrote, under `role`, relative to the item's directory.

        Accepts a path or anything with a `filepath` (a saved image). Nothing that
        was never written is recorded; the same path twice is still one file.
        """
        if output is None:
            return
        filepath = getattr(output, "filepath", output)
        if filepath is None:
            return
        # Forward slashes whatever the platform: the record is a description of
        # files that travels with the experiment, not a path for this OS to open,
        # and `os.path.join` reads it back correctly on either.
        relative = Path(os.path.relpath(str(filepath), str(self.path))).as_posix()
        paths = self.item.task_state.outputs.setdefault(role, [])
        if relative not in paths:
            paths.append(relative)

    def _start_task_state(self) -> None:
        """Mark the item's task_state as this run, InProgress.

        task_state is a single object reused across every run, so each per-run
        field is cleared here or the next run inherits the previous one's value.
        Never replace the object: the item list and card widgets connect psygnal
        handlers to this instance and use them as their only refresh trigger;
        replacing it orphans those connections and they silently stop updating
        mid-workflow (FIB-325)."""
        state = self.item.task_state
        state.name = self.task_name
        state.start_timestamp = now()
        state.end_timestamp = None
        state.task_id = self.task_id
        state.task_type = self.task_type
        state.iteration = self.iteration
        state.status = TaskStatus.InProgress
        state.status_message = ""
        state.outputs = {}

    def _record_outcome(self) -> None:
        """Freeze the finished task_state into task_history.

        Runs on every terminal path, not just success. task_state is a single object
        that the next task's pre_task overwrites, so an outcome that isn't frozen here
        is gone from the experiment as soon as that item starts its next task --
        leaving the logfile as the only record that the task ever failed. See FIB-490.
        """
        if self.item.task_state is None:
            raise ValueError("Task state is not set. Did you run pre_task()?")
        self.item.task_state.end_timestamp = now()
        self.item.task_history.append(deepcopy(self.item.task_state))

    def _is_cancellation(self, exc: Exception) -> bool:
        """Whether ``exc`` is a user Stop rather than a genuine failure: it surfaces as
        OperationCancelledError (milling / autofocus) or InterruptedError (_check_for_abort),
        or the task manager says this task should be unwinding.

        should_abort, not is_stopped: abandoning one task is as much a cancellation
        as ending the run, and an exception thrown on the way out of one should not
        be recorded as a failure."""
        if isinstance(exc, (OperationCancelledError, InterruptedError)):
            return True
        return bool(getattr(self.task_manager, "should_abort", False))

    def _check_for_abort(self) -> None:
        """Raise InterruptedError if this task should stop. Call between steps.

        Polls the manager's token, which already answers for every kind of stop —
        so this needs no second way to ask, and neither do the inline token checks
        in the tasks that poll it themselves.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            raise InterruptedError("Workflow aborted by user.")

    # -- status, metadata, hooks -----------------------------------------------

    def log_status_message(
        self, message: str, display_message: Optional[str] = None
    ) -> None:
        """Record a step of this run: in the log, on the item's task_state, as a
        task_step event and on the window's step line."""
        logging.debug(
            {
                "msg": "status",
                "timestamp": now_iso(),
                self.item_type: self.item.name,
                f"{self.item_type}_id": self.item.id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "task_step": message,
            }
        )
        if self.item.task_state is not None:
            self.item.task_state.step = message
            self.item.task_state.status_message = display_message or ""
        if message in LIFECYCLE_STEPS:
            return
        record = getattr(self.microscope, "record_event", None)
        if record is not None:
            record(
                "task_step",
                {
                    "step": message,
                    "display_message": display_message,
                    "item_type": self.item_type,
                    "task_type": self.task_type,
                },
            )
        signal = getattr(self.task_manager, "step", None)
        if signal is not None:
            signal.emit(display_message or message)

    def _set_workflow_metadata(self) -> None:
        """Stamp everything acquired from here until the run ends with the item
        and task that produced it, rather than relying on which folder it
        happens to sit in (FIB-466)."""
        ref = getattr(self.microscope, "experiment", None)
        setter = getattr(ref, "set_workflow_metadata", None)
        if setter is not None:
            setter(
                item_id=self.item.id,
                item_name=self.item.name,
                task_id=self.task_id,
                task_name=self.task_name,
                iteration=self.iteration,
            )

    def _clear_workflow_metadata(self) -> None:
        ref = getattr(self.microscope, "experiment", None)
        clear = getattr(ref, "clear_workflow_metadata", None)
        if clear is not None:
            clear()

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        from fibsem.hooks import fire_event

        # Which experiment, and how much of the queue is left. Only the manager knows
        # either, and a task can run without one -- a script driving a task directly, or
        # a stand-in -- so those fields are simply absent then.
        get_run_context = getattr(self.task_manager, "hook_run_context", None)
        run_context = get_run_context() if callable(get_run_context) else {}
        fire_event(
            getattr(self.task_manager, "hook_manager", None),
            event,
            task_name=self.task_name,
            task_type=self.task_type,
            item_name=self.item.name,
            item_id=self.item.id,
            task_id=self.task_id,
            task_state=self.item.task_state,
            error=error,
            **run_context,
        )
