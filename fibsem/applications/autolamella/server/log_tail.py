"""The session's ``logfile.log`` over the App Server (FIB-905).

What an agent asks of the log is "what was logged while this run was going",
so the read is a time window: ``since``/``until``, or a run's own start and end
from the experiment's ``events.jsonl``. With neither, it is the tail.

Records are served raw. Each is one header line with the lines after it that
have no header of their own (a traceback, a multi-line message); only the time
and level are read out of the header, because they are what the window and the
level filter need. The messages themselves are often Python dict reprs, not
JSON, and are not parsed.

The log grows for the whole session, so the read never takes the whole file:
a window starts from a bisection on the header times (the log is written in
time order), and the tail starts a fixed number of bytes from the end. What it
returns is capped in records and in bytes, and says when it was cut.

Times are the instrument's wall clock, as the log and the event tables write
them: an offset on ``since``/``until`` is dropped, not converted.
"""

import re
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    read_events,
)
from fibsem.util.timestamps import wall_time_of

__all__ = [
    "LOG_FILENAME",
    "LEVELS",
    "MAX_RECORDS",
    "read_log",
    "run_window",
]

LOG_FILENAME = "logfile.log"
DEFAULT_RECORDS = 200
MAX_RECORDS = 2000
# What one answer may carry, and one record within it: the step stream logs
# whole settings dicts on a line.
MAX_TEXT_BYTES = 512 * 1024
MAX_RECORD_CHARS = 16 * 1024
# How far back from the end a read with no window looks.
TAIL_BYTES = 4 * 1024 * 1024
# How far either side of a run to read: the failure is often logged just
# after the run's recorded end.
RUN_MARGIN = timedelta(seconds=5)

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# `2026-09-13 20:02:40,521 — root — DEBUG — get:748 — {...}`
_HEADER = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) — .*? — ([A-Z]+) — "
)
_RUN_ENDS = ("task_completed", "task_failed", "task_cancelled", "task_skipped")
# Below this many bytes apart, the bisection stops and the scan reads forward.
_BISECT_FLOOR = 64 * 1024


def _decode(raw: bytes) -> str:
    # The field separator is an em dash: a line from an old Windows install
    # written in cp1252 parses only when read as cp1252.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _header(line: str) -> Optional[Tuple[datetime, str]]:
    m = _HEADER.match(line)
    if m is None:
        return None
    try:
        time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return time.replace(microsecond=int(m.group(2)) * 1000), m.group(3)


def _first_time_after(f: BinaryIO, offset: int) -> Optional[datetime]:
    """The time of the first header at or after the line following *offset*."""
    f.seek(offset)
    if offset:
        f.readline()  # the rest of the line the offset fell inside
    for raw in f:
        parsed = _header(_decode(raw))
        if parsed is not None:
            return parsed[0]
    return None


def _offset_before(f: BinaryIO, size: int, since: datetime) -> int:
    """An offset at or before the first record at *since*."""
    lo, hi = 0, size
    while hi - lo > _BISECT_FLOOR:
        mid = (lo + hi) // 2
        time = _first_time_after(f, mid)
        if time is None or time >= since:
            hi = mid
        else:
            lo = mid
    return lo


def read_log(
    path: Path,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    level: Optional[str] = None,
    lines: int = DEFAULT_RECORDS,
) -> Dict[str, Any]:
    """The records of the log at *path* in the window, at or above *level*.

    The latest *lines* records are kept when more match. ``truncated`` says
    records were dropped, by that cap or the byte cap; ``read_from`` is the
    time of the first record the read looked at, so a tail that found nothing
    says how far back it looked.
    """
    minimum = LEVELS[level] if level else 0
    kept: deque = deque(maxlen=max(1, lines))
    matched = 0
    read_from: Optional[datetime] = None
    current: Optional[Dict[str, Any]] = None

    def finish(record):
        nonlocal matched
        if record is None or LEVELS.get(record["level"], 0) < minimum:
            return
        if since is not None and record["time"] < since:
            return
        text = "\n".join(record["lines"])
        if len(text) > MAX_RECORD_CHARS:
            text = text[:MAX_RECORD_CHARS] + " …[cut]"
        matched += 1
        kept.append(
            {"time": record["time"].isoformat(), "level": record["level"], "text": text}
        )

    with open(path, "rb") as f:
        size = f.seek(0, 2)
        if since is not None:
            start = _offset_before(f, size, since)
        else:
            start = max(0, size - TAIL_BYTES)
        f.seek(start)
        if start:
            f.readline()
        for raw in f:
            line = _decode(raw).rstrip("\r\n")
            parsed = _header(line)
            if parsed is None:
                # a continuation; one before the first header belongs to a
                # record that started before the read did, and is dropped
                if current is not None:
                    current["lines"].append(line)
                continue
            time, record_level = parsed
            if until is not None and time > until:
                break
            if read_from is None:
                read_from = time
            finish(current)
            current = {"time": time, "level": record_level, "lines": [line]}
        finish(current)

    records: List[Dict[str, Any]] = list(kept)
    truncated = matched > len(records)
    total = sum(len(r["text"]) for r in records)
    while records and total > MAX_TEXT_BYTES:
        total -= len(records.pop(0)["text"])
        truncated = True
    return {
        "records": records,
        "truncated": truncated,
        "read_from": read_from.isoformat() if read_from is not None else None,
    }


def run_window(
    experiment_path: Path, task_id: str
) -> Optional[Tuple[datetime, Optional[datetime]]]:
    """When the run *task_id* went, with a margin either side, from the
    experiment's event record. The end is None while the run is going.

    None when the record has no such run.
    """
    events = Path(experiment_path) / EVENTS_FILENAME
    if not events.is_file():
        return None
    start = end = None
    for record in read_events(events):
        if (record.get("task") or {}).get("id") != task_id:
            continue
        kind = record.get("kind")
        if kind == "task_started":
            start, end = wall_time_of(record.get("t")), None
        elif kind in _RUN_ENDS:
            end = wall_time_of(record.get("t"))
    if start is None and end is None:
        return None
    start = start if start is not None else end
    return start - RUN_MARGIN, end + RUN_MARGIN if end is not None else None
