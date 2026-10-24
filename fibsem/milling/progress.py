"""What a milling task says about itself while it runs.

`FibsemMicroscope.milling_progress_signal` carries a `MillingProgress`: one flat record
whose `MillingProgressStatus` is the whole vocabulary. The status has two scales,
`TASK_*` for the run and `STAGE_*` for one stage within it.

`message` is the one free field, because the producer set is open: milling strategies
are plugin-loadable (entry points in `fibsem/milling/strategy/__init__.py`), drive their
own loops, and may word their own progress. Two rules, kept here rather than at each
consumer:

* **Sticky.** A consumer keeps the last non-empty message beside the countdown
  (`MillingMessageTracker`), so a *delegating* strategy -- one that hands off to
  `microscope.run_milling()` -- sets its text once and it survives the backend's
  messageless ticks.
* **Falsy, not `is None`.** A plugin returning `""` gets the default label, not a
  blank one.

`display_message` is a convenience: a consumer that wants its own wording writes
`report.message or my_label(report.status)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from fibsem.progress import ProgressStatus, ends
from fibsem.structures.settings import MillingState

__all__ = [
    "MillingProgressStatus",
    "MillingProgress",
    "MillingMessageTracker",
]


class MillingProgressStatus(ProgressStatus):
    """Where a milling task has got to.

    Seven members across **two scales**: `TASK_*` is the run, `STAGE_*` is one stage
    within it.

    A `str` mixin so a member compares equal to its own value: these are persisted
    nowhere, but they reach log lines, and `"stage-update"` reads better than
    `MillingProgressStatus.STAGE_UPDATE`.
    """

    # The task is beginning, before any stage has started. The reset boundary for a
    # sticky `message`, so a stale label does not bleed from one task into the next.
    TASK_STARTED = "task-started"
    # One stage is beginning. Emitted *before* the stage runs, so it carries no elapsed
    # progress -- a bar driven off it reads zero, which is correct.
    STAGE_STARTED = "stage-started"
    # A tick from whoever is driving the beam: a backend's poll loop, or a self-driving
    # strategy's own. The only status that carries a countdown.
    STAGE_UPDATE = "stage-update"
    # One stage is done. Makes per-stage duration measurable.
    STAGE_FINISHED = "stage-finished"
    # The task ran to completion. Terminal.
    TASK_FINISHED = "task-finished"
    # Someone stopped the task. Terminal, and deliberately not `TASK_FAILED`: a cancel
    # is a person getting what they asked for, so a consumer paints it neutrally.
    TASK_CANCELLED = "task-cancelled"
    # The task raised. Terminal, and the only status that carries `error`.
    TASK_FAILED = "task-failed"


ends(
    MillingProgressStatus.TASK_FINISHED,
    cancelled=MillingProgressStatus.TASK_CANCELLED,
    failed=MillingProgressStatus.TASK_FAILED,
)


_DEFAULT_MESSAGES: Dict[MillingProgressStatus, str] = {
    MillingProgressStatus.TASK_STARTED: "Preparing milling task...",
    MillingProgressStatus.STAGE_STARTED: "Preparing...",
    MillingProgressStatus.STAGE_UPDATE: "Milling...",
    MillingProgressStatus.STAGE_FINISHED: "Finished stage",
    MillingProgressStatus.TASK_FINISHED: "Finished milling task",
    MillingProgressStatus.TASK_CANCELLED: "Milling cancelled",
    MillingProgressStatus.TASK_FAILED: "Milling failed",
}

# Which name each default is qualified by, when the report carries one -- a stage report
# says which stage, a task report says which task. The three outcomes are in neither
# set: "Milling cancelled: Rough Mill" invites the reading that one *stage* was
# cancelled, which is the two-scale confusion the status vocabulary exists to remove.
_STAGE_QUALIFIED = frozenset(
    {
        MillingProgressStatus.STAGE_STARTED,
        MillingProgressStatus.STAGE_UPDATE,
        MillingProgressStatus.STAGE_FINISHED,
    }
)
_TASK_QUALIFIED = frozenset(
    {MillingProgressStatus.TASK_STARTED, MillingProgressStatus.TASK_FINISHED}
)


@dataclass(frozen=True)
class MillingProgress:
    """One report from a milling task.

    Most fields are absent on most reports, and that is the point: an outcome carries no
    countdown, a stage tick carries no error. `status` is the only thing every report
    has and the only required field.

    `current_stage` is 0-based on the wire, matching every other index in the codebase.
    `display_stage` is the only thing that adds one.

    Equality is left generated, unlike `TiledProgress`. Nothing here carries a numpy
    array, so no field's `__eq__` returns an array and blows up the comparison -- which
    makes these usable in `assert emitted == [...]` and in a set.
    """

    status: MillingProgressStatus
    # The producer's own wording, and the one free field on this record. `None` means
    # "no opinion, use the default"; see the module docstring for why it is sticky and
    # why it is tested falsy rather than `is not None`.
    message: Optional[str] = None
    # Which milling run and which milling task config: the milling task's own, not the
    # AutoLamella task's, which an event's envelope carries as `task.id`.
    milling_run_id: Optional[str] = None
    milling_task_name: Optional[str] = None
    # Which stage. Carried by the `STAGE_*` statuses only.
    stage_name: Optional[str] = None
    current_stage: Optional[int] = None
    total_stages: Optional[int] = None
    start_time: Optional[float] = None
    estimated_time: Optional[float] = None
    remaining_time: Optional[float] = None
    # What the instrument says it is doing, carried on the report so a consumer does not
    # have to ask. Asking is not free: on ThermoFisher `get_milling_state()` is a getter
    # that *sets the active view* as a side effect, so a UI polling it during a
    # coincidence mill competes for the view with the fluorescence acquisition the
    # strategy is running. `MillingState.UNKNOWN` is a producer declining to look for
    # exactly that reason -- see `fibsem.structures.settings.MillingState`.
    milling_state: Optional[MillingState] = None
    # Why the task ended, on `TASK_FAILED` only. The *reason* -- an exception's text --
    # never the label: a consumer words `TASK_FAILED` itself and puts this behind it.
    error: Optional[str] = None

    @property
    def display_stage(self) -> Optional[int]:
        """Which stage this report is about, 1-based, for people.

        The one place the offset is applied: stage 0 is "1" to a reader and 0 to a
        `self.stages[...]` lookup.
        """
        if self.current_stage is None:
            return None
        return self.current_stage + 1

    @property
    def display_message(self) -> str:
        """The producer's words if it supplied any, otherwise composed from `status`.

        Falsy rather than `is not None`, so a plugin that returns `""` gets the default
        instead of a blank label.
        """
        if self.message:
            return self.message
        default = _DEFAULT_MESSAGES[self.status]
        name = None
        if self.status in _STAGE_QUALIFIED:
            name = self.stage_name
        elif self.status in _TASK_QUALIFIED:
            name = self.milling_task_name
        if not name:
            return default
        # The trailing ellipsis reads as "and then the name" rather than as a pause once
        # something follows it, so it is dropped when the label is qualified.
        return f"{default.rstrip('.')}: {name}"

    @classmethod
    def from_payload(cls, payload: object) -> "MillingProgress":
        """Decode whatever actually arrived as milling progress.

        Every in-tree producer emits a `MillingProgress`, so this is a no-op for them.
        A plugin strategy built against the older contract may still emit the nested
        `{"msg", "progress": {...}}` dict, and `psygnal` does not validate emissions, so
        the dict reaches the slot unchanged. A raise there lands inside a queued Qt
        slot, where on PyQt5 it aborts the process with nothing in the logfile; a
        payload that decodes to a bare `STAGE_UPDATE` is only a bar that does not move.

        **Total by construction.** Every branch returns, nothing here indexes, and the
        coercions below cannot raise on any input.
        """
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls(MillingProgressStatus.STAGE_UPDATE)

        inner = payload.get("progress")
        if not isinstance(inner, dict):
            inner = {}
        message = payload.get("msg")

        state = inner.get("state")
        if state == "start":
            status = MillingProgressStatus.STAGE_STARTED
        elif state == "finished":
            status = MillingProgressStatus.TASK_FINISHED
        else:
            # `"update"`, and also the shape with no `state` at all -- a delegating
            # strategy's own report, its `msg` plus a countdown.
            status = MillingProgressStatus.STAGE_UPDATE

        return cls(
            status=status,
            message=message if isinstance(message, str) else None,
            # `task_id`/`task_name` are the older spellings.
            milling_run_id=inner.get("milling_run_id") or inner.get("task_id"),
            milling_task_name=inner.get("milling_task_name") or inner.get("task_name"),
            # `name` is the older spelling of `stage_name`.
            stage_name=inner.get("stage_name") or inner.get("name"),
            current_stage=_as_int(inner.get("current_stage")),
            total_stages=_as_int(inner.get("total_stages")),
            start_time=_as_float(inner.get("start_time")),
            estimated_time=_as_float(inner.get("estimated_time")),
            remaining_time=_as_float(inner.get("remaining_time")),
            milling_state=_as_milling_state(inner.get("milling_state")),
        )


def _as_int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_milling_state(value: object) -> Optional[MillingState]:
    """Coerce the field, which older producers send as an enum and one as a `str`.

    The string spelling comes from a strategy that declines to call
    `get_milling_state()` -- on ThermoFisher that getter *sets the active view*, and
    such a strategy may be holding the view for a fluorescence acquisition.
    """
    if isinstance(value, MillingState):
        return value
    if isinstance(value, str):
        return MillingState.__members__.get(value.upper())
    return None


class MillingMessageTracker:
    """Keeps the last words a producer supplied, so a messageless tick still has a label.

    It exists for the *delegating* strategy: one that calls `microscope.run_milling()`
    and hands the loop to a backend. Such a strategy emits its label once, and every tick
    after that comes from a backend that has no idea what the strategy calls itself.

    Only `STAGE_UPDATE` inherits. Every other status is its own moment and gets its own
    label: a task that has finished should not still be captioned "Running Rough Mill".
    """

    def __init__(self) -> None:
        self._message = ""

    def label(self, report: MillingProgress) -> str:
        """What to show for *report*, given everything before it."""
        if report.message:
            self._message = report.message
        elif report.status is not MillingProgressStatus.STAGE_UPDATE:
            self._message = ""
        return self._message or report.display_message


class AgentMillOwner:
    """Whether a milling run is the agent's own mill, decided at its first report.

    A run outside any task (the agent's ``prepare`` and ``run``) starts with no
    task start; one first heard while the agent holds the microscope is the
    agent's to its end, after the agent has let go, whatever task the run is on.
    A task's run starts with ``task-started``. One rule for the status bar's line
    and the agent card's.
    """

    def __init__(self) -> None:
        self._run_id: Optional[str] = None
        self._name: Optional[str] = None

    def agent_mill(
        self, report: MillingProgress, agent_in_control: bool
    ) -> Optional[str]:
        """The name of the agent's own mill *report* is from, or None."""
        run_id = report.milling_run_id
        if run_id is None:
            return None
        if run_id != self._run_id:
            self._run_id = run_id
            agents = (
                agent_in_control
                and report.status is not MillingProgressStatus.TASK_STARTED
            )
            self._name = (report.milling_task_name or "milling") if agents else None
        return self._name
