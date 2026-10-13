"""The History tab's rows: one per task run, with what its operations did.

Each row is one run of a task on a lamella, in the order they ran, joined from
two records that already exist: the lamella's ``task_history`` (one entry per run,
failed and cancelled ones included) and the operation events in ``events.jsonl``
(an ``alignment``, an ``autofocus``...), which carry the run's ``task.id``. The run
still going, when there is one, is a row too.

Deliberately free of UI imports, like ``task_outputs``: what a row says is policy,
and keeping it here lets it be tested without Qt and read by other consumers. The
widget only draws rows and keeps the filter.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Lamella,
)
from fibsem.applications.autolamella.task_outputs import (
    final_reference_images,
    fluorescence_images,
)
from fibsem.cancellation import CANCELLED, COMPLETED, FAILED, SKIPPED
from fibsem.util.timestamps import to_datetime

# The event kinds that are a run's operations. A record written before
# operations said how they ended has no status: it was only ever written for a
# run that completed.
OPERATION_KINDS = ("alignment", "autofocus", "fm_autofocus", "milling")

_BEAM = {"ELECTRON": "SEM", "ION": "FIB"}

# The figure each kind of run saves into its folder: AlignmentResult.save and
# AutoFocusResult.save.
_PLOT_FILES = {"alignment": "figure.png", "autofocus": "plot.png"}


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


@dataclass
class HistoryRun:
    """One run of a task on a lamella."""

    task_id: str
    task_name: str
    task_type: str
    status: AutoLamellaTaskStatus
    status_message: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    images: List[str]  # final SEM/FIB images, then fluorescence stacks
    operations: List[HistoryOperation]
    # The later run that wrote over images this one recorded, when one did:
    # reference images are rewritten under the same name (FIB-1275).
    images_replaced_by: Optional[str] = None

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
    task name, or None for every task.
    """

    show: str = "all"
    status: str = "all"
    task: Optional[str] = None

    @property
    def active(self) -> bool:
        return self != HistoryFilter()

    def to_dict(self) -> Dict[str, Any]:
        return {"show": self.show, "status": self.status, "task": self.task}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HistoryFilter":
        """Anything unrecognised falls back to showing everything: a filter saved
        by another version must not hide runs without the operator knowing."""
        data = data or {}
        show = data.get("show")
        status = data.get("status")
        task = data.get("task")
        return cls(
            show=show if show in ("all", "images", "operations") else "all",
            status=status if status in ("all", "failed", "cancelled") else "all",
            task=task if isinstance(task, str) and task else None,
        )


_STATUS_OF_FILTER = {
    "failed": AutoLamellaTaskStatus.Failed,
    "cancelled": AutoLamellaTaskStatus.Cancelled,
}


def filter_runs(runs: List[HistoryRun], flt: HistoryFilter) -> List[HistoryRun]:
    """The runs ``flt`` lets through, in order. A task filter names a task the
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
        kept.append(run)
    return kept


def history_rows(
    lamella: Lamella, events: Iterable[Dict[str, Any]]
) -> List[HistoryRun]:
    """The lamella's runs, oldest first, each with its operations and images.

    ``events`` are recorded events, from ``events.jsonl`` or the live buffer; any
    order, any lamella: only this lamella's operation events are used, matched to
    a run by the task id the event was stamped with. An experiment without an
    events file still has its runs and images, with no operations.
    """
    runs: List[AutoLamellaTaskState] = list(lamella.task_history)
    current = lamella.task_state
    known = {run.task_id for run in runs}
    if (
        current is not None
        and current.task_id not in known
        and current.status
        in (AutoLamellaTaskStatus.InProgress, AutoLamellaTaskStatus.AwaitingDecision)
    ):
        runs.append(current)

    operations: Dict[str, List[HistoryOperation]] = {}
    for record in events:
        if record.get("kind") not in OPERATION_KINDS:
            continue
        item = record.get("item") or {}
        task = record.get("task") or {}
        if item.get("id") != lamella.id or not task.get("id"):
            continue
        operations.setdefault(task["id"], []).append(_operation(record, lamella.name))
    # In the order they started: a mill ends after the drift corrections it ran,
    # and is recorded after them, but it is what started them. A record with no
    # start (a skip) goes where it ended.
    for task_id, ops in operations.items():
        ops.sort(key=_started)
        operations[task_id] = _fold_drift(ops)

    images, replaced_by = _images_by_run(lamella, runs)
    return [
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
        )
        for run in runs
    ]


def _images_by_run(
    lamella: Lamella, runs: List[AutoLamellaTaskState]
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
    clock, so an alignment's start places it. ``operations`` are one run's, in
    the order they started; what no stage claims stays as it is."""
    claimed = set()
    for mill in operations:
        if mill.kind != "milling":
            continue
        for position, stage in enumerate(_stages(mill.payload)):
            start = to_datetime(stage.get("started_at"))
            end = to_datetime(stage.get("ended_at"))
            if start is None or end is None:
                continue
            for index, op in enumerate(operations):
                when = op.started_at or op.ended_at
                if (
                    op.kind == "alignment"
                    and index not in claimed
                    and when is not None
                    and start.timestamp() <= when.timestamp() <= end.timestamp()
                ):
                    mill.drift.setdefault(position, []).append(op)
                    claimed.add(index)
        if mill.drift:
            mill.detail += _drift_summary(mill, completed=mill.status == COMPLETED)
    return [op for index, op in enumerate(operations) if index not in claimed]


def _drift_summary(mill: HistoryOperation, completed: bool) -> str:
    """`` · drift ≤ 0.07 µm``, or the first drift correction that went wrong:
    folded, the mill's line is the only place a problem shows. A mill that did
    not complete already says why, so it adds only a problem."""
    names = [stage.get("name") for stage in _stages(mill.payload)]
    largest = 0.0
    for position, runs in sorted(mill.drift.items()):
        name = names[position] if position < len(names) else "a stage"
        for run in runs:
            if run.status == FAILED:
                return f" · drift correction failed at {name}"
            if any(step.get("success") is False for step in _steps(run.payload)):
                return f" · drift correction too weak at {name}"
            largest = max(largest, _total_shift(run.payload))
    return f" · drift ≤ {largest * 1e6:.2f} µm" if completed else ""


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
        lines.append(
            MillingStageLine(
                name=str(stage.get("name") or "Stage"),
                text=" · ".join(parts),
                drift=operation.drift.get(position, []),
            )
        )
    return lines


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
        return f"{milled} stage{'' if milled == 1 else 's'}"
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
