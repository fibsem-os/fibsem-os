"""Durations as fibsemOS shows them: how long something took or will take (FIB-1191).

Instants -- when something happened, and on whose clock -- are
:mod:`fibsem.util.timestamps`. A duration has no zone and no legacy forms to parse,
so it is a different job, done here.

Five shapes, each for one purpose. Pick by what the figure is for, not by how it
looks at one magnitude:

- :func:`format_duration_precise`, ``2m 14.00s``: a figure quoted to the second,
  such as a milling estimate or a log line.
- :func:`format_duration_rounded`, ``4m 12s`` or padded ``1h 01m``: an estimate or
  a countdown already understood to be approximate; padded for a column.
- :func:`format_duration_as_clock`, ``4:05`` or ``1:02:05``: a run's length in a
  table.
- :func:`format_duration_in_words`, ``3 h 40 min``: prose, such as a report or the
  review tab.
- :func:`format_time_ago`, ``3 min ago``: how long ago something happened.

Every one rounds rather than truncates, except :func:`format_time_ago`, which floors:
"3 min ago" for something three minutes and fifty seconds old is still true. Every
one returns ``""`` for a value that is missing (None or NaN), so a caller can leave
the field out or substitute its own placeholder.

Qt-free, so the reports and the headless paths can share it.
"""

from __future__ import annotations

import math
from typing import Optional


def _seconds(value) -> Optional[float]:
    """*value* as seconds, or None when it is missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(seconds) or math.isinf(seconds) else seconds


def format_duration_precise(seconds) -> str:
    """Hours and minutes, then seconds to two decimals: ``1h 3m 5.00s``,
    ``2m 14.00s``, ``45.00s``. For a figure quoted to the second."""
    value = _seconds(seconds)
    if value is None:
        return ""
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    return f"{secs:.2f}s"


def format_duration_rounded(seconds, pad: bool = False) -> str:
    """The two largest whole units, rounded: ``1h 1m``, ``4m 12s``, ``5s``.

    Rounded before the units are chosen, so a value a hair under a boundary reads as
    the boundary: 59.6 s is ``1m 0s``, not ``59s``.

    *pad* zero-fills the second unit (``1h 01m``, ``4m 02s``), which a column of
    durations wants: a figure that changes width as it counts down drags everything
    after it sideways.
    """
    value = _seconds(seconds)
    if value is None:
        return ""
    total = int(round(value))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    width = "02d" if pad else "d"
    if hours:
        return f"{hours}h {minutes:{width}}m"
    if minutes:
        return f"{minutes}m {secs:{width}}s"
    return f"{secs}s"


def format_duration_as_clock(seconds) -> str:
    """As a clock reads, rounded to the second: ``4:05``, or ``1:02:05`` past an
    hour. For a run's length in a table."""
    value = _seconds(seconds)
    if value is None:
        return ""
    total = int(round(value))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_duration_in_words(seconds) -> str:
    """In words, rounded: ``45 s``, ``12 min``, ``3 h 40 min``, ``2 h``. For
    prose, where a figure sits in a sentence."""
    value = _seconds(seconds)
    if value is None:
        return ""
    total = int(round(max(0.0, value)))
    if total < 60:
        return f"{total} s"
    hours, minutes = divmod(total // 60, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def format_time_ago(seconds) -> str:
    """How long ago, coarsely: ``just now``, ``3 min ago``, ``2 h ago``,
    ``1 d ago``. Floored: a time is at least that old."""
    value = _seconds(seconds)
    if value is None:
        return ""
    value = max(0.0, value)
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if value >= size:
            return f"{int(value // size)} {unit} ago"
    return "just now"
