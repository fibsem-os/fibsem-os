"""An experiment's record as tables, read from its ``events.jsonl`` (FIB-1036).

The second version of the report is built from these. Only the event stream is
read: not the log, and not ``Experiment.load()``. Each table is a DataFrame with
fixed columns, so an experiment that recorded none of a kind still has the
table, empty.

* ``runs``: one row per task run on an item -- when it started and ended, how it
  ended, and how much of it was spent waiting for an answer.
* ``steps``: one row per task step. A step ends where the run's next step
  starts, or where the run ends.
* ``milling``: one row per milling stage, from its start to its finish.
* ``decisions``: one row per decision on a question -- who made it, how long
  the question waited for it, and how far the decider moved what was proposed.

Waiting is the time a run's questions stood unanswered: a prompt from when it
was raised to its answer or withdrawal, and a question on the record from when
it was asked to its first decision. Overlapping waits count once, so a question
recorded both ways is not counted twice. Machine time is the rest.

Times are on the instrument's wall clock, as the replay reads them: the offset
in ``t`` is dropped. Durations are in seconds.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    read_events,
)
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    POINT_OF_INTEREST,
    STATE,
)

RUN_COLUMNS = [
    "item",
    "item_id",
    "task",
    "task_type",
    "task_id",
    "start",
    "end",
    "outcome",
    "reason",
    "duration",
    "waiting",
    "machine",
]
STEP_COLUMNS = [
    "item",
    "task",
    "task_type",
    "task_id",
    "step",
    "start",
    "end",
    "duration",
    "waiting",
    "machine",
]
MILLING_COLUMNS = [
    "item",
    "task",
    "task_id",
    "milling_task",
    "stage",
    "stage_index",
    "start",
    "end",
    "duration",
    "milling_current",
    "depth",
    "finished",
]
DECISION_COLUMNS = [
    "time",
    "item",
    "task",
    "kind",
    "proposal_id",
    "decision",
    "outcome",
    "author",
    "actor",
    "via",
    "reason",
    "asked",
    "waited",
    "changed",
    "moved",
    "unit",
    "checkpoint",
]

# How a run ended. A run with no end recorded is "unfinished": still running
# when the file was read, or cut off.
_RUN_ENDS = {
    "task_completed": "completed",
    "task_failed": "failed",
    "task_cancelled": "cancelled",
}
UNFINISHED = "unfinished"
SKIPPED = "skipped"

Interval = Tuple[datetime, datetime]


@dataclass
class EventTables:
    runs: pd.DataFrame
    steps: pd.DataFrame
    milling: pd.DataFrame
    decisions: pd.DataFrame


def read_event_tables(path: Union[str, Path]) -> EventTables:
    """The tables of the experiment at ``path``: its folder, or its events file."""
    path = Path(path)
    if path.is_dir():
        path = path / EVENTS_FILENAME
    return event_tables(read_events(path))


def event_tables(records: Iterable[Dict[str, Any]]) -> EventTables:
    """The tables of these records, in the order they were written."""
    runs: Dict[Any, Dict[str, Any]] = {}  # run id -> row
    steps: List[Dict[str, Any]] = []
    current_step: Dict[Any, Dict[str, Any]] = {}  # run id -> its step now
    milling: List[Dict[str, Any]] = []
    # (milling task id, stage name) -> the stage's row, until it finishes
    stages: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    decisions: List[Dict[str, Any]] = []
    # (proposal id, decision index) -> its row, for a fill-in to complete
    decided: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    raised: Dict[Tuple[Any, Any], Tuple[datetime, Any]] = {}  # prompt -> (t, run)
    asked: Dict[Any, Tuple[datetime, Any]] = {}  # proposal id -> (t, run)
    waits: Dict[Any, List[Interval]] = {}  # run id -> its questions' waits

    for record in records:
        time = _time(record)
        if time is None:
            continue
        kind = record.get("kind")
        payload = record.get("payload") or {}
        item = record.get("item") or {}
        task = record.get("task") or {}
        run_id = task.get("id")

        if kind == "task_started":
            runs[run_id] = _run(item, task, payload, start=time)
        elif kind in _RUN_ENDS or kind == "task_skipped":
            run = runs.get(run_id)
            if run is None:  # skipped, or started before the file did
                run = runs[run_id] = _run(item, task, payload, start=None)
            if kind == "task_skipped":
                run.update(start=time, outcome=SKIPPED)
                reason = payload.get("skip_reason")
            else:
                run["outcome"] = _RUN_ENDS[kind]
                reason = payload.get("error")
            run.update(end=time, reason=reason or None)
            _end(current_step.pop(run_id, None), time)
        elif kind == "task_step":
            _end(current_step.pop(run_id, None), time)
            step = {
                "item": item.get("name"),
                "task": task.get("name"),
                "task_type": payload.get("task_type"),
                "task_id": run_id,
                "step": payload.get("step"),
                "start": time,
                "end": None,
            }
            steps.append(step)
            current_step[run_id] = step
        elif kind == "milling_stage_started":
            stage = payload.get("stage") or {}
            row = {
                "item": item.get("name"),
                "task": task.get("name"),
                "task_id": run_id,
                "milling_task": payload.get("task_name"),
                "stage": stage.get("name"),
                "stage_index": payload.get("stage_index"),
                "start": time,
                "end": None,
                "milling_current": _get(stage, "milling", "milling_current"),
                "depth": _get(stage, "pattern", "depth"),
                "finished": False,
            }
            milling.append(row)
            stages[(payload.get("task_id"), stage.get("name"))] = row
        elif kind == "milling_progress" and payload.get("status") == "stage-finished":
            row = stages.pop((payload.get("task_id"), payload.get("stage_name")), None)
            if row is not None:
                row.update(end=time, finished=True)
        elif kind == "prompt_raised":
            raised[(record.get("session"), payload.get("nonce"))] = (time, run_id)
        elif kind in ("prompt_answered", "prompt_cancelled"):
            prompt = raised.pop((record.get("session"), payload.get("nonce")), None)
            if prompt is not None:
                waits.setdefault(prompt[1], []).append((prompt[0], time))
        elif kind == "proposal_asked":
            asked[payload.get("proposal_id")] = (time, run_id)
        elif kind == "proposal_decided":
            key = (payload.get("proposal_id"), payload.get("decision"))
            if payload.get("filled_in") and key in decided:
                # The values a confirmation "as it stands" was made at.
                _set_change(decided[key], payload)
                continue
            question = asked.pop(payload.get("proposal_id"), None)
            row = _decision(time, record, payload, question)
            if question is not None:
                waits.setdefault(question[1], []).append((question[0], time))
            decisions.append(row)
            decided[key] = row

    for run in runs.values():
        _time_spent(run, waits.get(run["task_id"], []))
    for step in steps:
        _time_spent(step, waits.get(step["task_id"], []))
    for row in milling:
        row["duration"] = _seconds(row["start"], row["end"])

    return EventTables(
        runs=_table(runs.values(), RUN_COLUMNS),
        steps=_table(steps, STEP_COLUMNS),
        milling=_table(milling, MILLING_COLUMNS),
        decisions=_table(decisions, DECISION_COLUMNS),
    )


def _time(record: Dict[str, Any]) -> Optional[datetime]:
    """When an event happened, on the instrument's wall clock: ``t`` with its
    UTC offset dropped, as the replay and the log read theirs."""
    try:
        return datetime.fromisoformat(record["t"]).replace(tzinfo=None)
    except (KeyError, TypeError, ValueError):
        return None


def _run(
    item: Dict[str, Any],
    task: Dict[str, Any],
    payload: Dict[str, Any],
    start: Optional[datetime],
) -> Dict[str, Any]:
    return {
        "item": item.get("name") or payload.get("item_name"),
        "item_id": item.get("id") or payload.get("item_id"),
        "task": task.get("name") or payload.get("task_name"),
        "task_type": payload.get("task_type"),
        "task_id": task.get("id") or payload.get("task_id"),
        "start": start,
        "end": None,
        "outcome": UNFINISHED,
        "reason": None,
    }


def _end(step: Optional[Dict[str, Any]], time: datetime) -> None:
    if step is not None:
        step["end"] = time


def _decision(
    time: datetime,
    record: Dict[str, Any],
    payload: Dict[str, Any],
    question: Optional[Tuple[datetime, Any]],
) -> Dict[str, Any]:
    row = {
        "time": time,
        # the item and task it was about, which the payload names
        "item": (payload.get("item") or {}).get("name"),
        "task": payload.get("task"),
        "kind": payload.get("kind"),
        "proposal_id": payload.get("proposal_id"),
        "decision": payload.get("decision"),
        "outcome": payload.get("outcome"),
        "author": payload.get("author"),
        "actor": record.get("actor"),
        "via": payload.get("via"),
        "reason": payload.get("reason"),
        "asked": question[0] if question is not None else None,
        "waited": _seconds(question[0], time) if question is not None else None,
        "checkpoint": payload.get("checkpoint"),
    }
    _set_change(row, payload)
    return row


def _set_change(row: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Whether the decider changed what was proposed, and for a point, a
    position or detected features, by how much."""
    proposed = payload.get("proposed") or {}
    decided = payload.get("decided") or {}
    moved, unit = _moved(row["kind"], proposed, decided)
    if not decided:  # confirmed as it stood, or decided nothing
        changed = False
    elif moved is not None:
        changed = moved > 0
    else:
        changed = not _same(proposed, decided)
    row.update(changed=changed, moved=moved, unit=unit)


def _moved(kind: Any, proposed: Dict[str, Any], decided: Dict[str, Any]):
    """How far a decision moved a point of interest (m), a stage position
    (x, y and z, m) or detected features (the furthest one, px). (None, None)
    for any other kind, or when there is nothing to compare."""
    if kind == POINT_OF_INTEREST:
        distance = _distance(proposed.get("poi"), decided.get("poi"), ("x", "y"))
        return (distance, "m") if distance is not None else (None, None)
    if kind == STATE:
        distance = _distance(
            proposed.get("stage_position"),
            decided.get("stage_position"),
            ("x", "y", "z"),
        )
        return (distance, "m") if distance is not None else (None, None)
    if kind == DETECTION:
        before = _features(proposed.get("features"))
        after = _features(decided.get("features"))
        distances = [
            _distance(before[name], after[name], ("x", "y"))
            for name in before
            if name in after
        ]
        distances = [d for d in distances if d is not None]
        return (max(distances), "px") if distances else (None, None)
    return None, None


def _features(values: Any) -> Dict[Any, Any]:
    if not isinstance(values, list):
        return {}
    return {f.get("name"): f.get("px") for f in values if isinstance(f, dict)}


def _distance(a: Any, b: Any, keys: Tuple[str, ...]) -> Optional[float]:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return None
    try:
        deltas = [float(b[k]) - float(a[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    distance = math.sqrt(sum(d * d for d in deltas))
    # a value read back through a widget is a hair off what it was given
    scale = max((abs(float(a[k])) for k in keys), default=0.0)
    return 0.0 if distance <= 1e-9 * scale else distance


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def _time_spent(row: Dict[str, Any], waits: List[Interval]) -> None:
    """Duration, and how much of it was waiting for an answer and how much the
    machine's, for a run or a step with both ends; None where it has not."""
    start, end = row["start"], row["end"]
    duration = _seconds(start, end)
    if duration is None:
        row.update(duration=None, waiting=None, machine=None)
        return
    waiting = _overlap(waits, start, end)
    row.update(duration=duration, waiting=waiting, machine=duration - waiting)


def _overlap(intervals: List[Interval], start: datetime, end: datetime) -> float:
    """Seconds of ``start``..``end`` that any of ``intervals`` covers, each
    moment counted once."""
    clipped = sorted(
        (max(a, start), min(b, end)) for a, b in intervals if a < end and b > start
    )
    total = 0.0
    reach: Optional[datetime] = None
    for a, b in clipped:
        if reach is not None and a < reach:
            a = reach
        if b > a:
            total += (b - a).total_seconds()
        reach = b if reach is None else max(reach, b)
    return total


def _seconds(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _get(values: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(values, dict):
            return None
        values = values.get(key)
    return values


def _table(rows: Iterable[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[row.get(c) for c in columns] for row in rows], columns=columns
    )
