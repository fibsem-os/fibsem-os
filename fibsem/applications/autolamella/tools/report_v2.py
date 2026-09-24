"""Report v2: an experiment's record as one self-contained HTML page (FIB-1036).

Built from ``events.jsonl`` alone, through the tables in ``event_tables``, beside
the log-based report rather than replacing it. The page answers what a session
got, where its time went, and what needs a look:

* headline numbers: lamellae finished, throughput, and the time budget --
  machine, waiting for an answer, and idle (nothing running)
* the outcome of each lamella's run of each task
* a timeline of every run, with its waits and the idle gaps between runs

One file with nothing to fetch: styles are inline and the charts are SVG, so it
opens offline on the support PC and can be passed on as it is. It prints to A4
through its print stylesheet.

The experiment is only asked for its name, its lamellae and its workflow's
order; everything that happened comes from the stream.
"""

import html
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME
from fibsem.applications.autolamella.tools.event_tables import (
    EventTables,
    read_event_tables,
)

REPORT_DIRNAME = "reporting"
REPORT_FILENAME = "report.html"

# A gap between runs shorter than this is the queue moving on, not idle time
# worth pointing out.
_IDLE_GAP_MIN_S = 60.0
_IDLE_GAP_MIN_FRACTION = 0.02

# The tasks' colours, in the workflow's order, and what each ending looks like.
_TASK_COLOURS = (
    "#378ADD",
    "#1D9E75",
    "#7F77DD",
    "#D85A30",
    "#D4537E",
    "#639922",
    "#888780",
    "#BA7517",
)
_WAIT_COLOUR = "#EF9F27"
_FAILED_COLOUR = "#E24B4A"
_CANCELLED_COLOUR = "#5F5E5A"
_NO_ITEM = "(no lamella)"

Interval = Tuple[datetime, datetime]


@dataclass
class Summary:
    """The session's headline numbers."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    span: float = 0.0  # seconds, first run's start to last run's end
    machine: float = 0.0
    waiting: float = 0.0
    idle: float = 0.0
    idle_gaps: List[Interval] = field(default_factory=list)
    finished: List[str] = field(default_factory=list)  # every task completed

    @property
    def throughput(self) -> Optional[float]:
        """Finished lamellae per hour of the session."""
        return len(self.finished) / (self.span / 3600) if self.span > 0 else None


def write_report(experiment: Any, path: Union[str, Path, None] = None) -> Path:
    """Write the report of ``experiment`` -- to ``reporting/report.html`` in its
    folder unless ``path`` says where -- and return where it went.

    Raises FileNotFoundError for an experiment with no ``events.jsonl``: one
    recorded before the event stream, whose report is the log-based one.
    """
    folder = Path(experiment.path)
    events = folder / EVENTS_FILENAME
    if not events.is_file():
        raise FileNotFoundError(
            f"{experiment.name} has no {EVENTS_FILENAME}: it was recorded before "
            "the event stream, so its report is the log-based one."
        )
    page = render_report(
        read_event_tables(events),
        name=experiment.name,
        items=[p.name for p in experiment.positions],
        tasks=workflow_tasks(experiment),
    )
    path = Path(path) if path is not None else folder / REPORT_DIRNAME / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def workflow_tasks(experiment: Any) -> List[str]:
    """The workflow's tasks, in its order."""
    protocol = getattr(experiment, "task_protocol", None)
    config = getattr(protocol, "workflow_config", None)
    return [t.name for t in getattr(config, "tasks", None) or []]


def render_report(
    tables: EventTables,
    name: str,
    items: Sequence[str],
    tasks: Sequence[str],
    generated: Optional[datetime] = None,
) -> str:
    """The page, from the record's tables. ``items`` and ``tasks`` give the
    rows' and columns' order: the experiment's lamellae and its workflow. A
    lamella or task that ran without being in either is added after them."""
    runs = _runs(tables)
    items = _ordered(items, [r["item"] for r in runs])
    tasks = _ordered(tasks, [r["task"] for r in runs])
    summary = summarise(tables, items, tasks)
    generated = generated or datetime.now()
    body = [_header(name, summary, items, tasks, generated), _headline(summary, items)]
    if summary.start is not None:
        body += [
            _section("Outcome", _outcome(runs, items, tasks)),
            _section("Timeline", _timeline(runs, tables, summary, items, tasks)),
        ]
    else:
        body.append('<p class="muted">No task runs were recorded.</p>')
    return _PAGE.format(title=_e(name), style=_STYLE, body="\n".join(body))


def summarise(tables: EventTables, items: Sequence[str], tasks: Sequence[str]):
    """The session's headline numbers, from its runs."""
    runs = _runs(tables)
    # a run still going, or cut off, counts up to when it was last heard from
    spans = [(r["start"], _run_end(r)) for r in runs if r["start"] is not None]
    summary = Summary()
    if not spans:
        return summary
    summary.start = min(a for a, _ in spans)
    summary.end = max(b for _, b in spans)
    summary.span = (summary.end - summary.start).total_seconds()
    busy = _union(spans)
    busy_s = sum((b - a).total_seconds() for a, b in busy)
    summary.waiting = sum(r["waiting"] or 0.0 for r in runs)
    summary.machine = max(busy_s - summary.waiting, 0.0)
    summary.idle = max(summary.span - busy_s, 0.0)
    shortest = max(_IDLE_GAP_MIN_S, _IDLE_GAP_MIN_FRACTION * summary.span)
    summary.idle_gaps = [
        (a, b)
        for (_, a), (b, _) in zip(busy, busy[1:])
        if (b - a).total_seconds() >= shortest
    ]
    done = {(r["item"], r["task"]) for r in runs if r["outcome"] == "completed"}
    summary.finished = [
        item for item in items if tasks and all((item, t) in done for t in tasks)
    ]
    return summary


# ── the page's parts ─────────────────────────────────────────────────────────


def _header(name, summary, items, tasks, generated) -> str:
    when = "no runs recorded"
    if summary.start is not None:
        when = (
            f"{_date_time(summary.start)} – {_time_of_day(summary.end, summary.start)}"
            f" ({_duration(summary.span)})"
        )
    facts = [when, f"{len(items)} lamellae"]
    if tasks:
        facts.append(" → ".join(tasks))
    return (
        '<header><div><h1>{name}</h1><div class="muted">{facts}</div>'
        '<div class="muted small">From events.jsonl · generated {generated}</div>'
        "</div>"
        '<button class="noprint" onclick="window.print()">Print or save as PDF</button>'
        "</header>"
    ).format(
        name=_e(name),
        facts=" · ".join(_e(f) for f in facts),
        generated=_e(generated.strftime("%d %b %Y, %H:%M")),
    )


def _headline(summary: Summary, items: Sequence[str]) -> str:
    def share(seconds: float) -> str:
        if summary.span <= 0:
            return "—"
        return f"{100 * seconds / summary.span:.0f}%"

    throughput = summary.throughput
    tiles = [
        (
            "Finished",
            f"{len(summary.finished)} of {len(items)}",
            "every task completed",
        ),
        (
            "Throughput",
            f"{throughput:.1f} / h" if throughput is not None else "—",
            "finished lamellae per hour",
        ),
        ("Machine", share(summary.machine), _duration(summary.machine)),
        ("Waiting for an answer", share(summary.waiting), _duration(summary.waiting)),
        ("Idle", share(summary.idle), f"{_duration(summary.idle)}, nothing running"),
    ]
    return '<div class="tiles">{}</div>'.format(
        "".join(
            f'<div class="tile"><span>{_e(label)}</span><b>{_e(value)}</b>'
            f"{_e(note)}</div>"
            for label, value, note in tiles
        )
    )


def _outcome(runs: List[Dict[str, Any]], items, tasks) -> str:
    """Lamella by task: each cell the last run's ending, and how many runs."""
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for run in runs:
        cells.setdefault((run["item"], run["task"]), []).append(run)
    head = "".join(f"<th>{_e(t)}</th>" for t in tasks)
    rows = []
    for item in items:
        row = [f"<th>{_e(item)}</th>"]
        for task in tasks:
            tried = cells.get((item, task), [])
            if not tried:
                row.append('<td class="cell none">not run</td>')
                continue
            last = tried[-1]
            text = _cell_text(last)
            earlier = {run["outcome"] for run in tried[:-1]}
            if len(earlier) == 1:  # a retry: say how the earlier runs went
                text += f" · after {len(tried) - 1} {earlier.pop()}"
            elif earlier:
                text += f" · after {len(tried) - 1} earlier runs"
            tip = f"{task}: {last['outcome']}"
            if last["reason"]:
                tip += f" — {last['reason']}"
            row.append(
                f'<td class="cell {_e(last["outcome"])}" title="{_e(tip)}">'
                f"{_e(text)}</td>"
            )
        rows.append(f"<tr>{''.join(row)}</tr>")
    return f'<table class="matrix"><tr><th></th>{head}</tr>{"".join(rows)}</table>'


def _cell_text(run: Dict[str, Any]) -> str:
    outcome, duration = run["outcome"], run["duration"]
    if outcome == "completed":
        return _clock(duration)
    if outcome in ("failed", "cancelled"):
        return f"{outcome} at {_clock(duration)}" if duration is not None else outcome
    return outcome  # skipped, unfinished


def _timeline(runs, tables: EventTables, summary: Summary, items, tasks) -> str:
    """Every run on its lamella's row, coloured by task, with its waits over
    it and the idle gaps between runs shaded."""
    width, left, right, top, row_h = 960, 170, 16, 26, 22
    rows = {item: i for i, item in enumerate(items)}
    height = top + row_h * len(items) + 30
    start, span = summary.start, max(summary.span, 1.0)

    def x(t: datetime) -> float:
        return left + (t - start).total_seconds() / span * (width - left - right)

    colours = {t: _TASK_COLOURS[i % len(_TASK_COLOURS)] for i, t in enumerate(tasks)}
    parts: List[str] = []
    bottom = top + row_h * len(items)
    for a, b in summary.idle_gaps:
        parts.append(
            f'<rect x="{x(a):.1f}" y="{top - 4}" width="{x(b) - x(a):.1f}" '
            f'height="{bottom - top + 4}" class="idle"/>'
        )
        if x(b) - x(a) >= 60:
            parts.append(
                f'<text x="{(x(a) + x(b)) / 2:.1f}" y="{top - 8}" '
                f'text-anchor="middle">idle {_e(_duration((b - a).total_seconds()))}'
                "</text>"
            )
    for tick, label in _ticks(summary.start, summary.end):
        parts.append(
            f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 4}" '
            f'y2="{bottom}" class="grid"/><text x="{x(tick):.1f}" y="{bottom + 16}" '
            f'text-anchor="middle">{_e(label)}</text>'
        )
    for item, i in rows.items():
        parts.append(
            f'<text x="{left - 8}" y="{top + i * row_h + 11}" text-anchor="end">'
            f"{_e(item)}</text>"
        )
    for run in runs:
        if run["start"] is None:
            continue
        end = _run_end(run)
        y = top + rows[run["item"]] * row_h
        outcome = run["outcome"]
        fill = {"failed": _FAILED_COLOUR, "cancelled": _CANCELLED_COLOUR}.get(
            outcome, colours.get(run["task"], _TASK_COLOURS[-2])
        )
        style = ' class="run unfinished"' if outcome == "unfinished" else ' class="run"'
        tip = f"{run['item']} · {run['task']} · {outcome}"
        if run["duration"] is not None:
            tip += f" · {_clock(run['duration'])}"
        if run["reason"]:
            tip += f" — {run['reason']}"
        parts.append(
            f'<rect x="{x(run["start"]):.1f}" y="{y}" '
            f'width="{max(x(end) - x(run["start"]), 1.5):.1f}" height="14" rx="2" '
            f'fill="{fill}"{style}><title>{_e(tip)}</title></rect>'
        )
    for wait in tables.waits.to_dict("records"):
        item = _item(wait["item"])
        if item not in rows or pd.isna(wait["start"]) or pd.isna(wait["end"]):
            continue
        a, b = _dt(wait["start"]), _dt(wait["end"])
        parts.append(
            f'<rect x="{x(a):.1f}" y="{top + rows[item] * row_h}" '
            f'width="{max(x(b) - x(a), 1.5):.1f}" height="14" rx="2" '
            f'fill="{_WAIT_COLOUR}" class="wait"><title>{_e(_text(wait["task"]))} '
            f"waited {_e(_clock((b - a).total_seconds()))} for an answer</title></rect>"
        )
    # what this session had: a key for failed runs only if one failed
    outcomes = {run["outcome"] for run in runs}
    legend = [(t, colours[t]) for t in tasks]
    if not tables.waits.empty:
        legend.append(("waiting for an answer", _WAIT_COLOUR))
    legend += [
        (outcome, colour)
        for outcome, colour in (
            ("failed", _FAILED_COLOUR),
            ("cancelled", _CANCELLED_COLOUR),
        )
        if outcome in outcomes
    ]
    keys = "".join(
        f'<span class="key"><i style="background:{c}"></i>{_e(label)}</span>'
        for label, c in legend
    )
    return (
        f'<div class="legend">{keys}</div>'
        f'<svg class="timeline" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img"><title>Every run across the session</title>{"".join(parts)}'
        "</svg>"
    )


def _section(title: str, content: str) -> str:
    return f"<section><h2>{_e(title)}</h2>{content}</section>"


# ── reading the tables ───────────────────────────────────────────────────────


def _runs(tables: EventTables) -> List[Dict[str, Any]]:
    """The runs as plain rows: times as datetimes (or None), a lamella name
    for every run, and when a run with no end was last heard from."""
    last_seen: Dict[Any, datetime] = {}
    for step in tables.steps.to_dict("records"):
        for key in ("start", "end"):
            t = _dt(step[key])
            if t is not None and (
                step["task_id"] not in last_seen or t > last_seen[step["task_id"]]
            ):
                last_seen[step["task_id"]] = t
    rows = []
    for run in tables.runs.to_dict("records"):
        rows.append(
            {
                "item": _item(run["item"]),
                "task": run["task"] if not pd.isna(run["task"]) else "(no task)",
                "task_id": run["task_id"],
                "start": _dt(run["start"]),
                "end": _dt(run["end"]),
                "outcome": run["outcome"],
                "reason": None if pd.isna(run["reason"]) else run["reason"],
                "duration": None if pd.isna(run["duration"]) else run["duration"],
                "waiting": None if pd.isna(run["waiting"]) else run["waiting"],
                "last_seen": last_seen.get(run["task_id"]),
            }
        )
    return rows


def _run_end(run: Dict[str, Any]) -> datetime:
    return run["end"] or run["last_seen"] or run["start"]


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _item(value: Any) -> str:
    return _NO_ITEM if value is None or pd.isna(value) else str(value)


def _dt(value: Any) -> Optional[datetime]:
    if value is None or pd.isna(value):
        return None
    return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value


def _ordered(first: Sequence[str], then: Sequence[str]) -> List[str]:
    seen = list(dict.fromkeys(first))
    return seen + [x for x in dict.fromkeys(then) if x not in seen]


def _union(spans: List[Interval]) -> List[Interval]:
    merged: List[Interval] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


# ── words and numbers ────────────────────────────────────────────────────────


def _clock(seconds: Optional[float]) -> str:
    """A run's length as a clock reads: 4:05, or 1:02:05 past an hour."""
    if seconds is None:
        return "—"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _duration(seconds: float) -> str:
    """A span in words: 3 h 40 min, 12 min, 45 s."""
    total = int(round(seconds))
    if total < 60:
        return f"{total} s"
    hours, minutes = divmod(total // 60, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def _date_time(t: datetime) -> str:
    return t.strftime("%d %b %Y, %H:%M")


def _time_of_day(t: datetime, since: datetime) -> str:
    return t.strftime("%H:%M") if t.date() == since.date() else _date_time(t)


def _ticks(start: datetime, end: datetime) -> List[Tuple[datetime, str]]:
    """Up to about eight round times across the session, labelled: seconds
    for a short one, dates for one that crosses midnight."""
    span = (end - start).total_seconds()
    steps = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)
    steps += (14400, 28800, 43200, 86400)
    step = next((s for s in steps if span / s <= 8), steps[-1])
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    offset = (start - midnight).total_seconds()
    tick = midnight + timedelta(seconds=math.ceil(offset / step) * step)
    label = "%H:%M:%S" if step < 60 else "%H:%M"
    if start.date() != end.date():
        label = "%d %b " + label
    ticks = []
    while tick <= end:
        ticks.append((tick, tick.strftime(label)))
        tick += timedelta(seconds=step)
    return ticks


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AutoLamella report</title>
<style>{style}</style></head>
<body>
{body}
</body></html>
"""

_STYLE = """
:root { --ink: #1f1f1d; --muted: #6b6a66; --line: #d9d8d3; --soft: #f4f3ef; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--ink); background: #fff; margin: 0 auto; padding: 24px 32px;
  max-width: 1040px; }
h1 { font-size: 22px; font-weight: 600; margin: 0; }
h2 { font-size: 16px; font-weight: 600; margin: 28px 0 8px; }
header { display: flex; justify-content: space-between; align-items: flex-start;
  gap: 16px; }
button { font: inherit; font-size: 13px; padding: 6px 12px; border-radius: 6px;
  border: 1px solid var(--line); background: #fff; cursor: pointer; }
.muted { color: var(--muted); } .small { font-size: 12px; }
.tiles { display: flex; gap: 8px; margin-top: 16px; }
.tile { flex: 1; border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; color: var(--muted); }
.tile span { display: block; } .tile b { display: block; font-size: 20px;
  font-weight: 600; color: var(--ink); }
.matrix { border-collapse: separate; border-spacing: 3px; font-size: 12px; }
.matrix th { font-weight: 500; color: var(--muted); text-align: left;
  padding: 2px 8px; white-space: nowrap; }
.cell { padding: 4px 10px; border-radius: 4px; min-width: 96px; white-space: nowrap; }
.completed { background: #e3f1e8; color: #1e5b35; }
.failed { background: #fbe5e4; color: #8f2320; }
.cancelled { background: #ecebe7; color: #444441; }
.skipped, .unfinished { background: #fdf0dc; color: #7a4b0c; }
.none { background: var(--soft); color: var(--muted); }
.legend { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.key { display: inline-flex; align-items: center; gap: 4px; margin-right: 14px; }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.timeline text { font-size: 11px; fill: var(--muted); }
.timeline .grid { stroke: var(--line); stroke-width: 0.5; }
.timeline .idle { fill: var(--soft); stroke: var(--line); stroke-dasharray: 3 3; }
.timeline .unfinished { fill-opacity: 0.45; stroke: var(--muted);
  stroke-dasharray: 3 2; }
section { break-inside: avoid; }
@page { size: A4; margin: 12mm; }
@media print {
  body { padding: 0; max-width: none; font-size: 12px; }
  .noprint { display: none; }
  .tiles, .cell, .key i, .timeline rect { print-color-adjust: exact;
    -webkit-print-color-adjust: exact; }
}
"""
