"""The History tab's rows: one per task run, with what its operations did.

Each row is one run of a task on a lamella, in the order they ran, joined from
two records that already exist: the lamella's ``task_history`` (one entry per run,
failed and cancelled ones included) and the operation events in ``events.jsonl``
(an ``alignment``, an ``autofocus``...), which carry the run's ``task.id``. The run
still going, when there is one, is a row too.

Operations outside any task -- an agent's autofocus or mill between runs, or the
operator's from the Microscope tab -- have no ``task.id`` (FIB-1375). Each session's
are one more row, "Outside a task", placed among the runs by when they started.
They belong to no lamella, so every lamella's History shows them.

Deliberately free of UI imports, like ``task_outputs``: what a row says is policy,
and keeping it here lets it be tested without Qt and read by other consumers. The
widget only draws rows and keeps the filter.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fibsem.acting import AGENT, OPERATOR
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.task_outputs import (
    final_reference_images,
    fluorescence_images,
)
from fibsem.cancellation import CANCELLED, COMPLETED, FAILED, SKIPPED
from fibsem.fm.preview import is_fluorescence_image
from fibsem.util.timestamps import to_datetime
from fibsem.workflows.status import TaskState, TaskStatus

# The event kinds that are a run's operations. A record written before
# operations said how they ended has no status: it was only ever written for a
# run that completed.
OPERATION_KINDS = ("alignment", "autofocus", "fm_autofocus", "milling")

# The event kinds that name a saved image. In a task the run's outputs list its
# images; outside one, these are how a session's row finds them (FIB-1521).
IMAGE_KINDS = ("image_acquired", "fm_image_acquired")

_BEAM = {"ELECTRON": "SEM", "ION": "FIB"}

# The final images a row shows: the last SEM and FIB.
_SHOWN_PAIR = 2

# The row holding a session's operations that ran outside any task.
OUTSIDE_A_TASK = "Outside a task"

# Who an operation outside a task was done by, as its line names them.
_ACTORS = {"agent": "agent", "operator": "you"}

# The figure each kind of run saves into its folder: AlignmentResult.save and
# AutoFocusResult.save, which the FM autofocus saves with too.
_PLOT_FILES = {
    "alignment": "figure.png",
    "autofocus": "plot.png",
    "fm_autofocus": "plot.png",
}


@dataclass
class HistoryOperation:
    """One operation a run performed, as the row shows it."""

    kind: str
    label: str  # "Align", "Autofocus", "Mill"
    status: str  # completed | skipped | failed | cancelled
    detail: str  # the one line: what it measured, or why it did not run
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    path: Optional[str]  # where its run was saved, when it was
    payload: Dict[str, Any] = field(repr=False, default_factory=dict)
    # A mill's drift corrections, by the index of the stage they ran in
    # (FIB-1351): folded under its line rather than lines of their own.
    drift: Dict[int, List["HistoryOperation"]] = field(repr=False, default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at).total_seconds())

    @property
    def plot(self) -> Optional[str]:
        """The figure its run saved, when it is on disk."""
        name = _PLOT_FILES.get(self.kind)
        if not self.path or name is None:
            return None
        path = os.path.join(self.path, name)
        return path if os.path.isfile(path) else None


@dataclass
class MillingStageLine:
    """One stage of a mill, as its opened line lists it."""

    name: str
    text: str  # how it ended, how long it took, its current
    drift: List[HistoryOperation]  # the drift corrections it ran
    # The strategy that milled it and what it said about the run (FIB-1257):
    # why it stopped, then its facts. A strategy with only facts says them on
    # the stage's own line (``text``), and these are empty; so are they for a
    # record from before strategies were recorded.
    strategy: str = ""
    said: str = ""
    figure: Optional[str] = None  # a figure the strategy saved, if still there
    # What each of its monitors said (FIB-1481), a line each under the strategy.
    monitors: List["MonitorLine"] = field(default_factory=list)


@dataclass
class MonitorLine:
    """What one monitor said about one mill step of a stage."""

    name: str  # "Intensity Drop", or "Intensity Drop · step 2" with several steps
    said: str
    figure: Optional[str] = None


@dataclass
class HistoryRun:
    """One run of a task on a lamella."""

    task_id: str
    task_name: str
    task_type: str
    status: TaskStatus
    status_message: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    images: List[str]  # final SEM/FIB images, then fluorescence stacks
    operations: List[HistoryOperation]
    # The later run that wrote over images this one recorded, when one did:
    # reference images are rewritten under the same name (FIB-1275).
    images_replaced_by: Optional[str] = None
    # A session's operations outside any task, not a run: it has no status, and
    # its images are the ones it saved in the lamella's folder.
    outside_task: bool = False
    # Who decided what the run proposed: "agent", "operator", both or neither.
    decided_by: Tuple[str, ...] = ()

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


@dataclass
class HistoryFilter:
    """What the tab shows. Remembered per user, so it round-trips as a dict.

    ``show``: ``all``, ``images`` (runs that left images, without their
    operations) or ``operations`` (runs that recorded operations, without
    images). ``status``: ``all``, ``failed`` or ``cancelled``. ``task``: one
    task name, or None for every task. ``decided_by``: ``anyone``, ``agent`` or
    ``operator``, the runs whose proposals they decided. ``order``: ``oldest``
    or ``newest`` first.
    """

    show: str = "all"
    status: str = "all"
    task: Optional[str] = None
    decided_by: str = "anyone"
    order: str = "oldest"

    @property
    def active(self) -> bool:
        """Whether it hides any run; the order hides none."""
        return replace(self, order="oldest") != HistoryFilter()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "show": self.show,
            "status": self.status,
            "task": self.task,
            "decided_by": self.decided_by,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HistoryFilter":
        """Anything unrecognised falls back to showing everything: a filter saved
        by another version must not hide runs without the operator knowing."""
        data = data or {}
        show = data.get("show")
        status = data.get("status")
        task = data.get("task")
        decided_by = data.get("decided_by")
        order = data.get("order")
        return cls(
            show=show if show in ("all", "images", "operations") else "all",
            status=status if status in ("all", "failed", "cancelled") else "all",
            task=task if isinstance(task, str) and task else None,
            decided_by=decided_by if decided_by in ("agent", "operator") else "anyone",
            order=order if order == "newest" else "oldest",
        )


_STATUS_OF_FILTER = {
    "failed": TaskStatus.Failed,
    "cancelled": TaskStatus.Cancelled,
}


def filter_runs(runs: List[HistoryRun], flt: HistoryFilter) -> List[HistoryRun]:
    """The runs ``flt`` lets through, in its order. A task filter names a task the
    lamella has not run lets nothing through, rather than everything."""
    wanted = _STATUS_OF_FILTER.get(flt.status)
    kept = []
    for run in runs:
        if wanted is not None and run.status is not wanted:
            continue
        if flt.task is not None and run.task_name != flt.task:
            continue
        if flt.show == "images" and not run.images:
            continue
        if flt.show == "operations" and not run.operations:
            continue
        if flt.decided_by != "anyone" and flt.decided_by not in run.decided_by:
            continue
        kept.append(run)
    return kept[::-1] if flt.order == "newest" else kept


def history_rows(
    lamella: Lamella, events: Iterable[Dict[str, Any]]
) -> List[HistoryRun]:
    """The lamella's runs, oldest first, each with its operations and images.

    ``events`` are recorded events, from ``events.jsonl`` or the live buffer; any
    order, any lamella: only this lamella's operation events are used, matched to
    a run by the task id the event was stamped with. An experiment without an
    events file still has its runs and images, with no operations.
    """
    runs: List[TaskState] = list(lamella.task_history)
    current = lamella.task_state
    known = {run.task_id for run in runs}
    if (
        current is not None
        and current.task_id not in known
        and current.status is TaskStatus.InProgress
    ):
        runs.append(current)

    operations: Dict[str, List[HistoryOperation]] = {}
    outside: Dict[str, List[HistoryOperation]] = {}
    outside_images: Dict[str, List[Tuple[Optional[datetime], str]]] = {}
    for record in events:
        kind = record.get("kind")
        task = record.get("task") or {}
        if kind in IMAGE_KINDS and not task.get("id"):
            path = _saved_in(lamella, (record.get("payload") or {}).get("path"))
            if path is not None:
                outside_images.setdefault(str(record.get("session") or ""), []).append(
                    (to_datetime(record.get("t") or record.get("timestamp")), path)
                )
            continue
        if kind not in OPERATION_KINDS:
            continue
        item = record.get("item") or {}
        if not task.get("id"):
            # another lamella's, though no task ran it, stays with that lamella
            if item.get("id") in (None, lamella.id):
                operation = _operation(record, lamella.name)
                actor = _ACTORS.get(record.get("actor"))
                if actor:
                    operation.label = f"{operation.label} ({actor})"
                outside.setdefault(str(record.get("session") or ""), []).append(
                    operation
                )
            continue
        if item.get("id") != lamella.id:
            continue
        operations.setdefault(task["id"], []).append(_operation(record, lamella.name))
    # In the order they started: a mill ends after the drift corrections it ran,
    # and is recorded after them, but it is what started them. A record with no
    # start (a skip) goes where it ended.
    for task_id, ops in operations.items():
        ops.sort(key=_started)
        operations[task_id] = _fold_drift(ops)

    images, replaced_by = _images_by_run(lamella, runs)
    deciders = _deciders_by_run(lamella)
    rows = [
        HistoryRun(
            task_id=run.task_id,
            task_name=run.name,
            task_type=run.task_type,
            status=run.status,
            status_message=run.status_message,
            started_at=to_datetime(run.start_timestamp),
            ended_at=to_datetime(run.end_timestamp),
            images=images.get(run.task_id, []),
            operations=operations.get(run.task_id, []),
            images_replaced_by=replaced_by.get(run.task_id),
            decided_by=deciders.get(run.task_id, ()),
        )
        for run in runs
    ]
    for session in {**outside, **outside_images}:
        ops = _fold_drift(sorted(outside.get(session, []), key=_started))
        saved = outside_images.get(session, [])
        times = [op.started_at or op.ended_at for op in ops] + [t for t, _ in saved]
        ends = [op.ended_at for op in ops] + [t for t, _ in saved]
        row = HistoryRun(
            task_id=f"session:{session}",
            task_name=OUTSIDE_A_TASK,
            task_type="",
            status=TaskStatus.Completed,
            status_message="",
            started_at=min(
                (t for t in times if t is not None), key=_when, default=None
            ),
            ended_at=max((t for t in ends if t is not None), key=_when, default=None),
            images=list(dict.fromkeys(path for _, path in saved)),
            operations=ops,
            outside_task=True,
        )
        _insert_by_start(rows, row)
    return rows


def _saved_in(lamella: Lamella, path: Optional[str]) -> Optional[str]:
    """``path`` when it is a file in the lamella's folder; else None."""
    if not path or not lamella.path:
        return None
    folder = os.path.abspath(str(lamella.path))
    path = os.path.abspath(str(path))
    try:
        inside = os.path.commonpath([folder, path]) == folder
    except ValueError:  # another drive
        return None
    return path if inside else None


def _deciders_by_run(lamella: Lamella) -> Dict[str, Tuple[str, ...]]:
    """By run, who decided its proposals: the agent, the operator or both. A
    withdrawn proposal was decided by no one."""
    deciders: Dict[str, set] = {}
    for proposal in lamella.proposals:
        if not proposal.task_id or proposal.withdrawn:
            continue
        for decision in proposal.decisions:
            if decision.actor.kind in (AGENT, OPERATOR):
                deciders.setdefault(proposal.task_id, set()).add(
                    str(decision.actor.kind)
                )
    return {task_id: tuple(sorted(kinds)) for task_id, kinds in deciders.items()}


def _insert_by_start(rows: List[HistoryRun], row: HistoryRun) -> None:
    """Put *row* before the first row that started after it: the runs keep the
    order they ran in, and one with no start is never stepped over."""
    when = _when(row.started_at)
    if when is None:
        rows.append(row)
        return
    at = len(rows)
    for index in range(len(rows) - 1, -1, -1):
        started = _when(rows[index].started_at)
        if started is None or started <= when:
            break
        at = index
    rows.insert(at, row)


def _when(moment: Optional[datetime]) -> Optional[float]:
    return moment.timestamp() if moment is not None else None


def shown_images(run: HistoryRun) -> List[str]:
    """The images a run's row shows: the last SEM/FIB pair (the highest
    magnification of a multi-FOV set), then its fluorescence stacks. The pair is
    picked before the stacks are added, so a stack never displaces it."""
    finals = [p for p in run.images if not is_fluorescence_image(p)]
    stacks = [p for p in run.images if is_fluorescence_image(p)]
    return finals[-_SHOWN_PAIR:] + stacks


def operations_need_a_look(run: HistoryRun) -> bool:
    """Whether a run's operations should be open without a click (FIB-1361):
    one did not complete (skipped, failed, stopped), or an alignment -- its
    own, or one a mill's stage ran -- failed or had a step too weak to apply.
    What routinely differs between runs ("methods disagree" on the Demo, a
    coincidence mill's time limit) does not count."""
    for operation in run.operations:
        alignments = [r for runs in operation.drift.values() for r in runs]
        if operation.kind == "alignment":
            alignments.append(operation)
        if operation.status != COMPLETED or any(
            a.status != COMPLETED
            or any(s.get("success") is False for s in _steps(a.payload))
            for a in alignments
        ):
            return True
    return False


def operations_summary(run: HistoryRun) -> str:
    """``3 operations · all completed``, or what did not, worst first: ``3
    operations · 1 failed · 1 skipped``. The folded line, so it names any
    problem."""
    count = len(run.operations)
    text = f"{count} operation{'' if count == 1 else 's'}"
    statuses = [op.status for op in run.operations]
    ended = [
        f"{statuses.count(status)} {status}"
        for status in (FAILED, CANCELLED, SKIPPED)
        if status in statuses
    ]
    if not ended:
        return f"{text} · all completed" if count > 1 else f"{text} · completed"
    return " · ".join([text, *ended])


def _images_by_run(
    lamella: Lamella, runs: List[TaskState]
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Each run's images. A file two runs recorded belongs to the later one:
    reference images are rewritten to the same name on every run, so what is on
    disk is the later run's picture, and showing it under the earlier run would
    put the wrong image there. Uniquely named files (fluorescence stacks, stamped
    reference sets) stay with their own run. Also returns, for a run that lost
    files that way, the later run that has them."""
    # The filename fallback is for experiments from before runs recorded their
    # outputs. Once any run has, a run that recorded no final images made none.
    fallback = not any(run.outputs for run in runs)
    recorded = {
        run.task_id: final_reference_images(lamella, run, fallback=fallback)
        + fluorescence_images(lamella, run)
        for run in runs
    }
    owner = {path: run.task_id for run in runs for path in recorded[run.task_id]}
    images = {
        task_id: [path for path in paths if owner[path] == task_id]
        for task_id, paths in recorded.items()
    }
    replaced_by = {}
    for task_id, paths in recorded.items():
        later = [owner[path] for path in paths if owner[path] != task_id]
        if later:
            replaced_by[task_id] = later[-1]
    return images, replaced_by


def _operation(record: Dict[str, Any], lamella_name: str) -> HistoryOperation:
    kind = record["kind"]
    payload = record.get("payload") or {}
    status = payload.get("status") or COMPLETED
    label = _LABELS.get(kind, kind)
    if kind == "alignment":
        label = _alignment_label(payload, lamella_name)
    return HistoryOperation(
        kind=kind,
        label=label,
        status=status,
        detail=_detail(kind, status, payload),
        started_at=to_datetime(payload.get("started_at")),
        ended_at=to_datetime(record.get("t") or record.get("timestamp")),
        path=payload.get("path"),
        payload=payload,
    )


_LABELS = {
    "alignment": "Align",
    "autofocus": "Autofocus",
    "fm_autofocus": "Autofocus (FM)",
    "milling": "Mill",
}


def _started(operation: HistoryOperation) -> float:
    """When it started, as a number: an aware and a naive time do not compare.
    One that recorded neither end goes last."""
    when = operation.started_at or operation.ended_at
    return when.timestamp() if when is not None else math.inf


def _fold_drift(operations: List[HistoryOperation]) -> List[HistoryOperation]:
    """Move each alignment that started inside a mill's stage under that mill:
    the stage's drift correction (FIB-1351). Stages run one after another on one
    clock, so an alignment's start places it. On a coarse clock (Windows ticks
    every 15.6 ms) one stage's end, the next one's start and that stage's drift
    correction can read the same instant: the later stage takes it, since a
    correction runs after its stage starts. ``operations`` are one run's, in the
    order they started; what no stage claims stays as it is."""
    claimed = set()
    for mill in operations:
        if mill.kind != "milling":
            continue
        windows = []
        for position, stage in enumerate(_stages(mill.payload)):
            start = to_datetime(stage.get("started_at"))
            end = to_datetime(stage.get("ended_at"))
            if start is not None and end is not None:
                windows.append((position, start.timestamp(), end.timestamp()))
        for index, op in enumerate(operations):
            when = op.started_at or op.ended_at
            if op.kind != "alignment" or index in claimed or when is None:
                continue
            inside = [
                p for p, start, end in windows if start <= when.timestamp() <= end
            ]
            if inside:
                mill.drift.setdefault(inside[-1], []).append(op)
                claimed.add(index)
        if mill.drift:
            mill.detail += _drift_summary(mill, completed=mill.status == COMPLETED)
    return [op for index, op in enumerate(operations) if index not in claimed]


def _drift_summary(mill: HistoryOperation, completed: bool) -> str:
    """`` · alignment ≤ 0.07 µm``, or the first milling current alignment that
    went wrong:
    folded, the mill's line is the only place a problem shows. A mill that did
    not complete already says why, so it adds only a problem."""
    names = [stage.get("name") for stage in _stages(mill.payload)]
    largest = 0.0
    for position, runs in sorted(mill.drift.items()):
        name = names[position] if position < len(names) else "a stage"
        for run in runs:
            if run.status == FAILED:
                return f" · alignment failed at {name}"
            if any(step.get("success") is False for step in _steps(run.payload)):
                return f" · alignment too weak at {name}"
            largest = max(largest, _total_shift(run.payload))
    return f" · alignment ≤ {largest * 1e6:.2f} µm" if completed else ""


def _steps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in payload.get("results") or [] if isinstance(s, dict)]


def _total_shift(payload: Dict[str, Any]) -> float:
    """How far an alignment moved, in metres: the sum of its steps."""
    steps = _steps(payload)
    dx = sum(_number((s.get("shift") or {}).get("x")) for s in steps)
    dy = sum(_number((s.get("shift") or {}).get("y")) for s in steps)
    return math.hypot(dx, dy)


def _alignment_label(payload: Dict[str, Any], lamella_name: str) -> str:
    """``Align`` for the task's own alignment to its reference; an alignment run
    inside something else -- each milling stage's drift correction, stamped with
    the same task -- is named for what ran it: ``Align (Rough Mill 01)``.

    Read off the run's name, which the caller sets (``<lamella> - <task>`` for
    the task's own, the stage name in milling) and a completed run suffixes
    with ``-HH-MM-SS``."""
    run = re.sub(r"-\d{2}-\d{2}-\d{2}$", "", str(payload.get("name") or ""))
    if not run or run.startswith(f"{lamella_name} - "):
        return "Align"
    return f"Align ({run})"


def operation_facts(operation: HistoryOperation) -> List[Tuple[str, str]]:
    """What the operation's expanded line lists, as (name, value): what it ran
    with, what each step measured, why it did not run to an end."""
    payload = operation.payload
    facts: List[Tuple[str, str]] = []
    if operation.status == SKIPPED:
        facts.append(("Reason", str(payload.get("reason") or "none recorded")))
    if operation.status == FAILED:
        facts.append(("Error", str(payload.get("error") or "none recorded")))
    beam = payload.get("beam_type")
    if beam:
        facts.append(("Beam", _BEAM.get(beam, str(beam))))
    if payload.get("method"):
        facts.append(("Method", str(payload["method"])))
    if operation.kind == "alignment":
        facts.extend(_alignment_facts(payload))
    elif operation.kind == "autofocus":
        facts.extend(_autofocus_facts(payload))
    elif operation.kind == "milling":
        pass  # its stages are milling_stages(), each with its drift corrections
    else:
        facts.extend(_fm_autofocus_facts(payload))
    return facts


def _alignment_facts(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    facts = []
    if payload.get("subsystem"):
        facts.append(("Corrected by", str(payload["subsystem"])))
    steps = [s for s in payload.get("results") or [] if isinstance(s, dict)]
    for index, step in enumerate(steps, start=1):
        shift = step.get("shift") or {}
        text = (
            f"({_number(shift.get('x')) * 1e9:+.0f}, {_number(shift.get('y')) * 1e9:+.0f}) nm"
            f" · score {_number(step.get('score')):.2f}"
        )
        if step.get("success") is False:
            text += " · too weak, not applied"
        facts.append((f"Step {index}", text))
    planned = payload.get("steps")
    if steps and isinstance(planned, int) and len(steps) < planned:
        facts.append(("Steps", f"{len(steps)} of {planned}"))
    validation = payload.get("validation")
    if isinstance(validation, dict) and "agreement" in validation:
        spread = _number(validation.get("max_disagreement_px"))
        agree = "agree" if validation.get("agreement") else "disagree"
        facts.append(("Methods", f"{agree} · {spread:.1f} px apart at most"))
    return facts


def _autofocus_facts(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    facts = []
    if payload.get("hfw") is not None:
        facts.append(("Field width", f"{_number(payload['hfw']) * 1e6:.0f} µm"))
    if payload.get("passes") is not None:
        facts.append(("Passes", str(payload["passes"])))
    if payload.get("steps") is not None:
        facts.append(("Probes", str(payload["steps"])))
    if payload.get("working_distance") is not None:
        facts.append(
            (
                "Working distance",
                f"{_mm(payload.get('initial_working_distance'))} → "
                f"{_mm(payload['working_distance'])} mm",
            )
        )
    if payload.get("focus_score") is not None:
        facts.append(("Focus score", f"{_number(payload['focus_score']):.2f}"))
    return facts


def milling_stages(operation: HistoryOperation) -> List[MillingStageLine]:
    """A mill's stages in the protocol's order: how each ended, how long it
    took, at what current, and the drift corrections it ran. The run's error is
    a fact above them, so a failed stage only says it failed."""
    lines = []
    for position, stage in enumerate(_stages(operation.payload)):
        status = str(stage.get("status") or COMPLETED)
        parts = [status]
        if status == SKIPPED and stage.get("reason"):
            parts = [f"skipped: {stage['reason']}"]
        took = _seconds_between(stage.get("started_at"), stage.get("ended_at"))
        if took is not None:
            parts.append(_short_duration(took))
        if status != SKIPPED and stage.get("milling_current") is not None:
            parts.append(_current(stage["milling_current"]))
        strategy = (
            stage.get("strategy") if isinstance(stage.get("strategy"), dict) else {}
        )
        name = _strategy_name(strategy.get("name"))
        said = _strategy_said(strategy)
        figure = strategy.get("figure")
        figure = figure if figure and os.path.isfile(str(figure)) else None
        own_line = bool(strategy.get("end_reason") or figure)
        if name and not own_line:
            # only facts: short enough for the stage's line
            parts.extend(p for p in (name, said) if p)
        lines.append(
            MillingStageLine(
                name=str(stage.get("name") or "Stage"),
                text=" · ".join(parts),
                drift=operation.drift.get(position, []),
                strategy=name if own_line else "",
                said=said if own_line else "",
                figure=figure,
                monitors=_monitor_lines(stage),
            )
        )
    return lines


def _monitor_lines(stage: Dict[str, Any]) -> List[MonitorLine]:
    """What a stage's monitors said, each step's in turn; the step named only
    when the stage milled in several."""
    steps = [s for s in stage.get("steps") or [] if isinstance(s, dict)]
    lines = []
    for number, step in enumerate(steps, start=1):
        for record in step.get("monitors") or []:
            if not isinstance(record, dict):
                continue
            name = _strategy_name(record.get("name")) or "Monitor"
            if len(steps) > 1:
                name = f"{name} · step {number}"
            said = _strategy_said(record)
            if record.get("error"):
                said = " · ".join(p for p in (f"failed: {record['error']}", said) if p)
            figure = record.get("figure")
            figure = figure if figure and os.path.isfile(str(figure)) else None
            lines.append(
                MonitorLine(name=name, said=said or "nothing to say", figure=figure)
            )
    return lines


def _strategy_name(name: Any) -> str:
    """``CoincidenceMilling`` as ``Coincidence Milling``: the strategy's
    name, which is what the record has, read as words."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name or ""))


def _strategy_said(strategy: Dict[str, Any]) -> str:
    """``intensity drop · Mill time (s) 250 · Intensity drop seen yes``: why
    the strategy stopped, then each fact as it labelled it."""
    parts = [str(strategy["end_reason"])] if strategy.get("end_reason") else []
    facts = strategy.get("facts")
    for label, value in facts.items() if isinstance(facts, dict) else []:
        parts.append(f"{label} {_fact(value)}")
    return " · ".join(parts)


def _fact(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fm_autofocus_facts(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    facts = []
    if payload.get("passes") is not None:
        done = payload.get("completed_passes", payload["passes"])
        facts.append(("Passes", f"{done} of {payload['passes']}"))
    if payload.get("position") is not None:
        facts.append(
            (
                "Objective",
                f"{_number(payload.get('initial_position')) * 1e6:.2f} → "
                f"{_number(payload['position']) * 1e6:.2f} µm",
            )
        )
    if payload.get("focus_score") is not None:
        facts.append(("Focus score", f"{_number(payload['focus_score']):.2f}"))
    return facts


def _detail(kind: str, status: str, payload: Dict[str, Any]) -> str:
    """The operation's one line."""
    if status == SKIPPED:
        return f"skipped: {payload.get('reason') or 'no reason recorded'}"
    if kind == "milling":
        return _milling_detail(status, payload)
    if status == FAILED:
        return f"failed: {payload.get('error') or 'no error recorded'}"
    if status == CANCELLED and not payload.get("results"):
        return "cancelled"
    if kind == "alignment":
        return _alignment_detail(payload, cancelled=status == CANCELLED)
    if kind == "autofocus":
        return _autofocus_detail(payload)
    return _fm_autofocus_detail(payload)


def _alignment_detail(payload: Dict[str, Any], cancelled: bool) -> str:
    steps = _steps(payload)
    text = f"{_total_shift(payload) * 1e6:.2f} µm · {len(steps)} step{'' if len(steps) == 1 else 's'}"
    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("agreement") is False:
        text += " · methods disagree"
    if cancelled:
        text += " · stopped"
    return text


def _milling_detail(status: str, payload: Dict[str, Any]) -> str:
    """``3 stages``; ``failed at Rough Mill 02: <error>``; ``stopped at Polishing
    · 1 of 2 stages milled``. Stages turned off in the protocol are not counted:
    they were never going to mill."""
    stages = [s for s in _stages(payload) if s.get("reason") != _TURNED_OFF]
    milled = sum(1 for s in stages if s.get("status") == COMPLETED)
    if status == COMPLETED:
        # why its strategies stopped, so a folded line still says it
        reasons: List[str] = []
        for stage in stages:
            said = stage.get("strategy")
            reason = said.get("end_reason") if isinstance(said, dict) else None
            if reason and str(reason) not in reasons:
                reasons.append(str(reason))
        return " · ".join([f"{milled} stage{'' if milled == 1 else 's'}", *reasons])
    ended = next((s for s in stages if s.get("status") in (FAILED, CANCELLED)), None)
    if status == FAILED:
        at = f" at {ended.get('name')}" if ended is not None else ""
        return f"failed{at}: {payload.get('error') or 'no error recorded'}"
    at = f" at {ended.get('name')}" if ended is not None else ""
    return f"stopped{at} · {milled} of {len(stages)} stages milled"


def _stages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in payload.get("stages") or [] if isinstance(s, dict)]


# As fibsem.milling.tasks records a stage turned off in the protocol.
_TURNED_OFF = "turned off in the protocol"


def _seconds_between(start: Any, end: Any) -> Optional[float]:
    started, ended = to_datetime(start), to_datetime(end)
    if started is None or ended is None:
        return None
    return max(0.0, (ended - started).total_seconds())


def _short_duration(seconds: float) -> str:
    """As the tab writes an operation's duration."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} m {seconds:02d} s"


def _current(amps: Any) -> str:
    value = _number(amps)
    if value >= 1e-9:
        return f"{value * 1e9:g} nA"
    return f"{value * 1e12:g} pA"


def _autofocus_detail(payload: Dict[str, Any]) -> str:
    before = payload.get("initial_working_distance")
    after = payload.get("working_distance")
    beam = _BEAM.get(payload.get("beam_type"), payload.get("beam_type") or "")
    text = f"{beam} WD {_mm(before)} → {_mm(after)} mm"
    if payload.get("steps") is not None:
        text += f" · {payload['steps']} probes"
    return text.strip()


def _fm_autofocus_detail(payload: Dict[str, Any]) -> str:
    before = payload.get("initial_position")
    after = payload.get("position")
    if before is None or after is None:
        return "objective moved"
    return f"objective {(_number(after) - _number(before)) * 1e6:+.2f} µm"


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mm(value: Any) -> str:
    try:
        return f"{float(value) * 1e3:.3f}"
    except (TypeError, ValueError):
        return "?"


def run_record(run: HistoryRun, root: str) -> Dict[str, Any]:
    """A run as plain data, for clients that are not the Qt tab: what its row,
    folded line and opened operations say. Files are relative to ``root`` (the
    experiment folder), so the experiment's files route serves them."""

    def where(path: Optional[str]) -> Optional[str]:
        return os.path.relpath(path, root) if path else None

    def operation(op: HistoryOperation) -> Dict[str, Any]:
        record = {
            "kind": op.kind,
            "label": op.label,
            "status": op.status,
            "detail": op.detail,
            "started_at": _when(op.started_at),
            "ended_at": _when(op.ended_at),
            "facts": [list(fact) for fact in operation_facts(op)],
            "figure": where(op.plot),
            "stages": [],
        }
        if op.kind == "milling":
            record["stages"] = [
                {
                    "name": stage.name,
                    "text": stage.text,
                    "strategy": stage.strategy,
                    "said": stage.said,
                    "figure": where(stage.figure),
                    "monitors": [
                        {"name": m.name, "said": m.said, "figure": where(m.figure)}
                        for m in stage.monitors
                    ],
                    "drift": [operation(d) for d in stage.drift],
                }
                for stage in milling_stages(op)
            ]
        return record

    return {
        "task_id": run.task_id,
        "task_name": run.task_name,
        "task_type": run.task_type,
        "status": run.status.name,
        "status_message": run.status_message,
        "started_at": _when(run.started_at),
        "ended_at": _when(run.ended_at),
        "duration": run.duration,
        "images": [where(path) for path in shown_images(run)],
        "images_replaced_by": run.images_replaced_by,
        "outside_task": run.outside_task,
        "decided_by": list(run.decided_by),
        "operations_summary": operations_summary(run) if run.operations else "",
        "operations_open": operations_need_a_look(run),
        "operations": [operation(op) for op in run.operations],
    }
