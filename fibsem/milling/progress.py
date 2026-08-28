"""What a milling task says about itself while it runs.

`FibsemMicroscope.milling_progress_signal` used to carry a **nested** dict --
`{"msg": ..., "progress": {...}}` -- with no declared shape, so every consumer opened
with a `.get("progress")`, a `None` guard, and then a second level of `.get` before it
could read anything. Four payload shapes were in flight, one of which matched no branch
in any consumer and therefore rendered nowhere.

`MillingProgress` is the whole contract, and `MillingStatus` is the whole vocabulary.

# One flat record, not a union -- but it is the closest call of the three

Unlike `TiledProgress` and `FluorescenceAcquisitionProgress`, this signal *is* a genuine
tagged union today: all three consumers open with the same `state ==` ladder over a
closed vocabulary, so a union of dataclasses would fit the dispatch. Three things tip it
to a flat record anyway.

The variants are not cleanly disjoint. `task_id`/`task_name` ride on every report;
`start_time` on both a stage start and a stage update. Every shared field becomes a base
class or a repeat.

The real win is orthogonal to the choice. The wart is the *nesting*, not the flatness --
removing a level of `.get` and its `None` guard is the same job either way.

And a third convention costs more than it buys. The other two typed progress signals are
flat records with a status enum; a union here means three signals with three shapes for
one concept.

# Two levels of status, because "start" and "finished" were not a pair

The original three-value vocabulary mixed two scales, and that is a live trap rather
than a cosmetic one. `state: "start"` was emitted per *stage*, N times; `state:
"finished"` from `run()`'s `finally`, once per *task*. They read like a matched pair and
are not one, so a consumer bracketing them is wrong N-1 times. The tiled work hit the
identical collision between `TILES_ACQUIRED` and `FINISHED`.

# `message` is the one open field, and it is a feature

The tiled signal dropped its message because its producers are a closed set of two, both
internal, so each consumer could own its wording. That argument does not hold here: the
producer set is **open**. Milling strategies are plugin-loadable (`register_strategy`
plus entry points in `fibsem/milling/strategy/__init__.py`), a strategy implements its
own execution loop, and users have specifically asked to customise the text a strategy
shows while it runs.

So `message` stays -- as the single free field on an otherwise closed record. `status`
does **not** open up. A free-form, caller-supplied discriminator is what broke the union
on FIB-401, where it turned out to be written by producers and read by nobody.

Two rules make it work, both carried here rather than at each consumer:

* **Sticky.** A consumer keeps the last non-empty message and renders it beside the
  countdown. That is what lets a *delegating* strategy -- one that calls
  `microscope.run_milling()` and hands off -- set the text once and have it survive the
  backend's messageless ticks. The backend has no idea what the strategy calls itself.
* **Falsy, not `is None`.** A plugin returning `""` is a bug, not a request for a blank
  label. FIB-401 hit this exactly: an empty channel rendered `"Acquiring  (1/1)..."`.

`display_message` is a convenience, not a mandate -- a consumer that wants its own
wording still writes `report.message or my_label(report.status)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from fibsem.structures import MillingState

__all__ = [
    "MillingStatus",
    "MillingProgress",
    "MillingMessageTracker",
]


class MillingStatus(str, Enum):
    """Where a milling task has got to.

    Seven members across **two scales**, and the prefixes are the whole point: `TASK_*`
    is the run, `STAGE_*` is one stage within it. The vocabulary this replaces had
    `start` (per stage) and `finished` (per task) sitting side by side looking like a
    pair.

    A `str` mixin so a member compares equal to its own value: these are persisted
    nowhere, but they reach log lines, and `"stage-update"` reads better than
    `MillingStatus.STAGE_UPDATE`.
    """

    # The task is beginning, before any stage has started. No producer emitted an
    # equivalent before, and it still earns its place: a sticky `message` needs a reset
    # boundary or a stale label bleeds from one task into the next. Without it a
    # consumer has to infer "new task" from `current_stage == 0`, which is the sort of
    # inference a typed contract exists to delete.
    TASK_STARTED = "task-started"
    # One stage is beginning. Emitted *before* the stage runs, so it carries no elapsed
    # progress -- a bar driven off it reads zero, which is correct.
    STAGE_STARTED = "stage-started"
    # A tick from whoever is driving the beam: a backend's poll loop, or a self-driving
    # strategy's own. The only status that carries a countdown.
    STAGE_UPDATE = "stage-update"
    # One stage is done. The weakest of the seven -- every one but the last is followed
    # immediately by a `STAGE_STARTED` -- but it makes per-stage duration measurable,
    # and it is the natural emit point in `_mill_stage`.
    STAGE_FINISHED = "stage-finished"
    # The task ran to completion. Terminal.
    TASK_FINISHED = "task-finished"
    # Someone stopped the task. Terminal, and deliberately not `TASK_FAILED`: a cancel
    # is a person getting what they asked for, so a consumer paints it neutrally.
    TASK_CANCELLED = "task-cancelled"
    # The task raised. Terminal, and the only status that carries `error`.
    TASK_FAILED = "task-failed"

    @property
    def is_terminal(self) -> bool:
        """Whether this ends the *task*.

        Answered here rather than restated at each consumer: three of them need it, and
        a membership tuple copied into three files is a tuple that drifts -- the next
        status added is one somebody forgets, and the symptom is a progress bar that
        never clears.

        `STAGE_FINISHED` is **not** terminal. Getting that backwards hides the bar
        after the first stage of an N-stage task.
        """
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {
        MillingStatus.TASK_FINISHED,
        MillingStatus.TASK_CANCELLED,
        MillingStatus.TASK_FAILED,
    }
)


# The label shown when no producer supplied a `message`. One table rather than a default
# argument at each consumer's `.get`: today the three of them default three different
# ways -- "Preparing Milling Conditions...", "Preparing...", "Milling..." -- for the
# same report, which is the drift a single home removes.
#
# `{stage}` and `{task}` are filled by `display_message` when the report carries a name,
# and the whole clause is dropped when it does not. A label reading "Preparing: None" is
# worse than one reading "Preparing...".
_DEFAULT_MESSAGES: Dict[MillingStatus, str] = {
    MillingStatus.TASK_STARTED: "Preparing milling task...",
    MillingStatus.STAGE_STARTED: "Preparing...",
    MillingStatus.STAGE_UPDATE: "Milling...",
    MillingStatus.STAGE_FINISHED: "Finished stage",
    MillingStatus.TASK_FINISHED: "Finished milling task",
    MillingStatus.TASK_CANCELLED: "Milling cancelled",
    MillingStatus.TASK_FAILED: "Milling failed",
}

# Which name each default is qualified by, when the report carries one -- a stage report
# says which stage, a task report says which task. The three outcomes are in neither
# set: "Milling cancelled: Rough Mill" invites the reading that one *stage* was
# cancelled, which is the two-scale confusion the status vocabulary exists to remove.
_STAGE_QUALIFIED = frozenset(
    {
        MillingStatus.STAGE_STARTED,
        MillingStatus.STAGE_UPDATE,
        MillingStatus.STAGE_FINISHED,
    }
)
_TASK_QUALIFIED = frozenset({MillingStatus.TASK_STARTED, MillingStatus.TASK_FINISHED})


@dataclass(frozen=True)
class MillingProgress:
    """One report from a milling task.

    Most fields are absent on most reports, and that is the point: an outcome carries no
    countdown, a stage tick carries no error. `status` is the only thing every report
    has and the only required field -- which is also what keeps this constructible on
    Python 3.8, where `kw_only` does not exist and required fields must come first.

    `current_stage` is 0-based on the wire, matching every other index in the codebase.
    `display_stage` is the only thing that adds one; that offset was applied seven times
    across three files before.

    Equality is left generated, unlike `TiledProgress`. Nothing here carries a numpy
    array, so no field's `__eq__` returns an array and blows up the comparison -- which
    makes these usable in `assert emitted == [...]` and in a set.
    """

    status: MillingStatus
    # The producer's own wording, and the one free field on this record. `None` means
    # "no opinion, use the default"; see the module docstring for why it is sticky and
    # why it is tested falsy rather than `is not None`.
    message: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None
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
    # exactly that reason -- see `fibsem.structures.MillingState`.
    milling_state: Optional[MillingState] = None
    # Why the task ended, on `TASK_FAILED` only. The *reason* -- an exception's text --
    # never the label: a consumer words `TASK_FAILED` itself and puts this behind it.
    error: Optional[str] = None

    @property
    def display_stage(self) -> Optional[int]:
        """Which stage this report is about, 1-based, for people.

        The one place the offset is applied. Stage 0 is "1" to a reader and 0 to a
        `self.stages[...]` lookup, and both spellings were written out by hand at seven
        call sites across three widgets.
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
            name = self.task_name
        if not name:
            return default
        # The trailing ellipsis reads as "and then the name" rather than as a pause once
        # something follows it, so it is dropped when the label is qualified.
        return f"{default.rstrip('.')}: {name}"

    @classmethod
    def from_payload(cls, payload: object) -> "MillingProgress":
        """Decode whatever actually arrived on `milling_progress_signal`.

        Every in-tree producer emits a `MillingProgress`, so this is a no-op for all of
        them, and on that basis it could be deleted. It is kept because the producer set
        is **open**. Milling strategies are plugin-loadable (`register_strategy` plus
        entry points in `fibsem/milling/strategy/__init__.py`) and a strategy drives its
        own execution loop, so one built against the older contract still emits the
        nested `{"msg", "progress": {...}}` dict. `psygnal` does not validate emissions:
        a `Signal(MillingProgress)` hands a dict to the slot unchanged, and the first
        attribute access on it raises.

        That raise lands inside a queued Qt slot, where on PyQt5 it is `qFatal` -- the
        process aborts with nothing written to the logfile (FIB-329). It is not a
        hypothetical: the sibling workflow-update signal killed the app exactly that way
        on every queue action, because one payload lacked a key a consumer indexed.

        The asymmetry is the whole argument. A payload that decodes to a bare
        `STAGE_UPDATE` is a progress bar that does not move; a raise is a dead
        application that takes the milling run with it.

        **Total by construction.** Every branch returns, nothing here indexes, and the
        coercions below cannot raise on any input.
        """
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls(MillingStatus.STAGE_UPDATE)

        inner = payload.get("progress")
        if not isinstance(inner, dict):
            inner = {}
        message = payload.get("msg")

        state = inner.get("state")
        if state == "start":
            status = MillingStatus.STAGE_STARTED
        elif state == "finished":
            status = MillingStatus.TASK_FINISHED
        else:
            # `"update"`, and also the shape that carries no `state` at all -- a
            # delegating strategy's own report, whose content is its `msg` plus a
            # countdown. That shape matched no branch in any consumer under the old
            # vocabulary, so its words rendered nowhere; decoding it as an update is
            # what finally puts them on the screen.
            status = MillingStatus.STAGE_UPDATE

        return cls(
            status=status,
            message=message if isinstance(message, str) else None,
            task_id=inner.get("task_id"),
            task_name=inner.get("task_name"),
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

    The stickiness rule, in one testable place rather than reimplemented in each of the
    three widgets that need it.

    It exists for the *delegating* strategy: one that calls `microscope.run_milling()`
    and hands the loop to a backend. Such a strategy emits its label once, and every tick
    after that comes from a backend that has no idea what the strategy calls itself. The
    considered alternative -- threading the message through `run_milling()` so the
    backend stamps every tick -- changes three backend signatures and still leaves the
    backend not owning the words.

    Only `STAGE_UPDATE` inherits. Every other status is its own moment and gets its own
    label: a task that has finished should not still be captioned "Running Rough Mill".
    """

    def __init__(self) -> None:
        self._message = ""

    def label(self, report: MillingProgress) -> str:
        """What to show for *report*, given everything before it."""
        if report.message:
            self._message = report.message
        elif report.status is not MillingStatus.STAGE_UPDATE:
            self._message = ""
        return self._message or report.display_message
