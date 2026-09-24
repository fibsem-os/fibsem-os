"""Report v2: an experiment's record as one self-contained HTML page (FIB-1036).

Built from ``events.jsonl`` alone, through the tables in ``event_tables``, beside
the log-based report rather than replacing it. The page answers what a session
got, where its time went, and what needs a look:

* headline numbers: lamellae finished, throughput, and the time budget --
  machine, waiting for an answer, and idle (nothing running)
* what is worth a look: failed, cancelled and unfinished runs, milling stages
  that did not finish, long waits, idle gaps, rejected proposals, and plan
  edits made while tasks were running
* the outcome of each lamella's run of each task
* a timeline of every run, with its waits and the idle gaps between runs, and
  every fluorescence acquisition
* where the operator stepped in: each kind of question, how often what was
  proposed was changed or rejected and by how much, and the plan's edits

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
from fibsem.applications.autolamella.proposals import kind_label
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
# A wait for an answer this long, or a gap with nothing running this long, is
# worth a look. Shorter gaps are still shaded on the timeline.
_LONG_WAIT_S = 300.0
_LONG_IDLE_S = 600.0

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
_FM_COLOUR = "#2C2C2A"
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
    fm = _fm(tables)
    items = _ordered(items, [r["item"] for r in runs] + [a["item"] for a in fm])
    tasks = _ordered(tasks, [r["task"] for r in runs])
    # what was on no lamella -- a grid overview -- has a row on the timeline,
    # but is not a lamella to count or to give an outcome
    lamellae = [item for item in items if item != _NO_ITEM]
    summary = summarise(tables, lamellae, tasks)
    generated = generated or datetime.now()
    body = [
        _header(name, summary, lamellae, tasks, generated),
        _headline(summary, lamellae, fm),
    ]
    if summary.start is not None:
        body += [
            _section("Worth a look", _worth_a_look(runs, tables, summary)),
            _section("Outcome", _outcome(runs, lamellae, tasks)),
            _section("Timeline", _timeline(runs, fm, tables, summary, items, tasks)),
            _section("Where the operator stepped in", _stepped_in(tables)),
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


def _headline(summary: Summary, items: Sequence[str], fm: List[Dict[str, Any]]) -> str:
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
    if fm:
        imaged = {a["item"] for a in fm if a["item"] != _NO_ITEM}
        tiles.append(
            (
                "Fluorescence",
                f"{len(fm)} acquisitions",
                f"on {len(imaged)} lamella{'e' if len(imaged) != 1 else ''}",
            )
        )
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


def _timeline(runs, fm, tables: EventTables, summary: Summary, items, tasks) -> str:
    """Every run on its lamella's row, coloured by task, with its waits over
    it and the idle gaps between runs shaded; each FM acquisition marked under
    its lamella's row. The span covers acquisitions made outside any run."""
    width, left, right, top, row_h = 960, 170, 16, 26, 22
    rows = {item: i for i, item in enumerate(items)}
    height = top + row_h * len(items) + 30
    start = min([summary.start] + [a["start"] for a in fm])
    end = max([summary.end] + [a["end"] for a in fm])
    span = max((end - start).total_seconds(), 1.0)

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
    for tick, label in _ticks(start, end):
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
    for acquisition in fm:
        middle = (
            x(acquisition["start"])
            + (x(acquisition["end"]) - x(acquisition["start"])) / 2
        )
        y = top + rows[acquisition["item"]] * row_h + 15
        parts.append(
            f'<path d="M{middle - 4:.1f},{y + 6} L{middle + 4:.1f},{y + 6} '
            f'L{middle:.1f},{y} Z" fill="{_FM_COLOUR}" class="fm">'
            f"<title>{_e(_fm_title(acquisition))}</title></path>"
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
    if fm:
        legend.append(("FM acquisition", _FM_COLOUR))
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


def _worth_a_look(runs, tables: EventTables, summary: Summary) -> str:
    """What a reader should see first: most serious first, then in order."""
    notes: List[Tuple[int, datetime, str]] = []  # (how serious, when, what)
    completed = {(r["item"], r["task"]) for r in runs if r["outcome"] == "completed"}
    for run in runs:
        item, task, when = run["item"], run["task"], run["start"] or run["end"]
        reason = f": {run['reason'].rstrip('.')}" if run["reason"] else ""
        if run["outcome"] == "failed":
            then = (
                " It was run again and completed." if (item, task) in completed else ""
            )
            notes.append(
                (
                    0,
                    when,
                    f"{item}: {task} failed after {_clock(run['duration'])}"
                    f"{reason}.{then}",
                )
            )
        elif run["outcome"] == "cancelled":
            notes.append(
                (
                    1,
                    when,
                    f"{item}: {task} was cancelled at "
                    f"{_clock(run['duration'])}{reason}.",
                )
            )
        elif run["outcome"] == "unfinished":
            notes.append(
                (
                    1,
                    when,
                    f"{item}: {task} has no end recorded; last heard from at "
                    f"{_run_end(run):%H:%M}.",
                )
            )
    for stage in tables.milling.to_dict("records"):
        if not stage["finished"] and _dt(stage["start"]) is not None:
            notes.append(
                (
                    1,
                    _dt(stage["start"]),
                    f"{_item(stage['item'])}: "
                    f"{_text(stage['milling_task'])} stage {_text(stage['stage'])} "
                    "did not finish.",
                )
            )
    for decision in tables.decisions.to_dict("records"):
        if decision["outcome"] == "Rejected":
            reason = _text(decision["reason"]).rstrip(".")
            notes.append(
                (
                    1,
                    _dt(decision["time"]),
                    f"{_item(decision['item'])}: "
                    f"{kind_label(_text(decision['kind']))} rejected by the "
                    f"{_text(decision['actor']) or 'operator'}"
                    f"{': ' + reason if reason else ''}.",
                )
            )
    for wait in tables.waits.to_dict("records"):
        if not pd.isna(wait["duration"]) and wait["duration"] >= _LONG_WAIT_S:
            start = _dt(wait["start"])
            notes.append(
                (
                    2,
                    start,
                    f"{_item(wait['item'])}: {_text(wait['task'])} waited "
                    f"{_duration(wait['duration'])} for an answer, from {start:%H:%M}.",
                )
            )
    for a, b in summary.idle_gaps:
        if (b - a).total_seconds() < _LONG_IDLE_S:
            continue
        notes.append(
            (
                2,
                a,
                f"Nothing ran from {a:%H:%M} to {b:%H:%M} "
                f"({_duration((b - a).total_seconds())}).",
            )
        )
    notes += _edits_mid_run(runs, tables)
    if not notes:
        return '<p class="muted">Nothing stood out.</p>'
    notes.sort(key=lambda note: (note[0], note[1]))
    marks = ("failed", "warning", "notice", "info")
    return '<ul class="notes">{}</ul>'.format(
        "".join(
            f'<li class="note {marks[level]}">{_e(text)}</li>'
            for level, _, text in notes
        )
    )


def _edits_mid_run(runs, tables: EventTables) -> List[Tuple[int, datetime, str]]:
    """Plan edits made while a task was running, one note for each change: an
    edit applied to many lamellae at once is one note."""
    changes: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for edit in tables.edits.to_dict("records"):
        t = _dt(edit["time"])
        running = next(
            (r for r in runs if r["start"] and r["start"] <= t <= _run_end(r)), None
        )
        if running is None:
            continue
        key = (
            _text(edit["via"]),
            _text(edit["target"]),
            _text(edit["actor"]),
            t.replace(second=0, microsecond=0),
        )
        changes.setdefault(key, []).append(dict(edit, running=running, time=t))
    notes = []
    for (via, target, actor, _), edits in changes.items():
        lamellae = sorted({_item(e["item"]) for e in edits})
        where = lamellae[0] if len(lamellae) == 1 else f"{len(lamellae)} lamellae"
        running = edits[0]["running"]
        who = f" by the {actor}" if actor else ""
        notes.append(
            (
                3,
                edits[0]["time"],
                f"{_text(edits[0]['task']) or 'The protocol'}'s "
                f"{target} changed on {where} at {edits[0]['time']:%H:%M} "
                f"({via}{who}), while {running['task']} was running on "
                f"{running['item']}.",
            )
        )
    return notes


def _stepped_in(tables: EventTables) -> str:
    """Each kind of question: how its decisions went, how long it waited, and
    how far what was proposed was moved. Then the plan's edits."""
    decisions = tables.decisions.to_dict("records")
    edits = tables.edits.to_dict("records")
    if not decisions and not edits:
        return '<p class="muted">Nobody was asked anything, and the plan was not edited.</p>'
    parts = []
    if decisions:
        kinds: Dict[str, List[Dict[str, Any]]] = {}
        for d in decisions:
            kinds.setdefault(_text(d["kind"]), []).append(d)
        rows = []
        for kind, rows_of_kind in kinds.items():
            confirmed = [d for d in rows_of_kind if d["outcome"] == "Confirmed"]
            changed = [d for d in confirmed if d["changed"]]
            waits = [d["waited"] for d in rows_of_kind if not pd.isna(d["waited"])]
            cells = [
                kind_label(kind),
                len(rows_of_kind),
                len(waits),
                len(confirmed) - len(changed),
                len(changed),
                sum(d["outcome"] == "Rejected" for d in rows_of_kind),
                sum(d["outcome"] == "Unreviewed" for d in rows_of_kind),
                _clock(sum(waits) / len(waits)) if waits else "—",
                _mean_move(changed),
            ]
            rows.append(
                "<tr>"
                + "".join(
                    f"<td{'' if i == 0 else ' class=n'}>{_e(c)}</td>"
                    for i, c in enumerate(cells)
                )
                + "</tr>"
            )
        head = (
            "Question",
            "Decisions",
            "Asked",
            "As proposed",
            "Changed",
            "Rejected",
            "Unreviewed",
            "Mean wait",
            "Mean move",
        )
        parts.append(_table(head, rows))
        by = {}
        for d in decisions:
            by[_text(d["actor"]) or "not recorded"] = (
                by.get(_text(d["actor"]) or "not recorded", 0) + 1
            )
        note = (
            " Those recorded as the task's were made by nobody: unreviewed, "
            "withdrawn, or confirmed by the task itself."
            if "task" in by
            else ""
        )
        parts.append(
            '<p class="muted small">Decided by '
            + " · ".join(f"{_e(who)} {n}" for who, n in by.items())
            + f".{_e(note)}</p>"
        )
    if edits:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edit in edits:
            groups.setdefault((_text(edit["via"]), _text(edit["actor"])), []).append(
                edit
            )
        rows = []
        for (via, actor), group in groups.items():
            targets = list(dict.fromkeys(_text(e["target"]) for e in group))
            shown = ", ".join(targets[:3]) + (
                f" and {len(targets) - 3} more" if len(targets) > 3 else ""
            )
            cells = [
                via or "—",
                actor or "—",
                len(group),
                len({_item(e["item"]) for e in group}),
                shown,
            ]
            rows.append(
                "<tr>"
                + "".join(
                    f"<td{' class=n' if i in (2, 3) else ''}>{_e(c)}</td>"
                    for i, c in enumerate(cells)
                )
                + "</tr>"
            )
        parts.append("<h3>Plan edits</h3>")
        parts.append(_table(("From", "Who", "Edits", "Lamellae", "What"), rows))
    return "".join(parts)


def _mean_move(changed: List[Dict[str, Any]]) -> str:
    """How far the changed decisions moved what was proposed, on average."""
    moves = [(d["moved"], d["unit"]) for d in changed if not pd.isna(d["moved"])]
    if not moves:
        return "—"
    unit = moves[0][1]
    mean = sum(m for m, u in moves if u == unit) / sum(1 for _, u in moves if u == unit)
    if unit == "m":
        return f"{mean * 1e6:.1f} µm" if mean >= 1e-7 else f"{mean * 1e9:.0f} nm"
    return f"{mean:.1f} {unit}" if mean < 10 else f"{mean:.0f} {unit}"


def _table(head: Sequence[str], rows: List[str]) -> str:
    return '<table class="list"><tr>{}</tr>{}</table>'.format(
        "".join(f"<th>{_e(h)}</th>" for h in head), "".join(rows)
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


def _fm(tables: EventTables) -> List[Dict[str, Any]]:
    """The FM acquisitions as plain rows, each on a lamella's row."""
    rows = []
    for acquisition in tables.fm.to_dict("records"):
        start, end = _dt(acquisition["start"]), _dt(acquisition["end"])
        if start is None or end is None:
            continue
        rows.append(
            {
                "item": _item(acquisition["item"]),
                "task": _text(acquisition["task"]),
                "start": start,
                "end": end,
                "channels": list(acquisition["channels"] or []),
                "planes": int(acquisition["planes"]),
                "overview": _text(acquisition["overview"]),
            }
        )
    return rows


def _fm_title(acquisition: Dict[str, Any]) -> str:
    """An FM acquisition as its mark's tooltip says it: what, and how long."""
    if acquisition["overview"]:
        what = f"FM overview, {acquisition['overview']} tiles"
    elif acquisition["planes"] > 1:
        what = f"FM z-stack, {acquisition['planes']} planes"
    else:
        what = "FM image"
    if acquisition["channels"]:
        what += f", {', '.join(acquisition['channels'])}"
    if acquisition["task"]:
        what += f" ({acquisition['task']})"
    seconds = (acquisition["end"] - acquisition["start"]).total_seconds()
    return f"{what} · {_clock(seconds)}"


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
h3 { font-size: 14px; font-weight: 600; margin: 16px 0 6px; }
.list { border-collapse: collapse; width: 100%; font-size: 12px; }
.list th { text-align: left; font-weight: 500; color: var(--muted);
  border-bottom: 1px solid var(--line); padding: 4px 8px; }
.list td { padding: 4px 8px; border-bottom: 1px solid var(--soft); }
.list .n { text-align: right; font-variant-numeric: tabular-nums; }
.notes { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.note { padding: 5px 10px 5px 12px; margin-bottom: 4px; border-left: 3px solid;
  background: var(--soft); }
.note.failed { border-color: #E24B4A; } .note.warning { border-color: #BA7517; }
.note.notice { border-color: #EF9F27; } .note.info { border-color: #888780; }
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
  .tiles, .cell, .key i, .timeline rect, .note { print-color-adjust: exact;
    -webkit-print-color-adjust: exact; }
}
"""
