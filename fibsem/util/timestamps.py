"""Times as fibsem-os records and reads them: ISO 8601 with a UTC offset.

``2026-09-13T21:19:40.974286-06:00`` pins the instant and keeps the clock time at the
site that wrote it. It is what OME's ``AcquisitionDate`` uses, and
``datetime.fromisoformat`` reads it on every supported Python.

Files written before this carry one of three other forms, and stay readable:

- a POSIX float, which is what most drivers wrote into ``microscope_state.timestamp``;
- AutoScript's ``acquisition_datetime`` string, ``07/16/2026 11:07:27``, which the
  ThermoFisher driver wrote into the same field;
- a naive ISO string, the acquiring machine's local clock with no offset, which is
  what fluorescence images wrote into ``acquisition_date``.

:func:`to_datetime` is the one parser for all of them. An aware value comes back in the
viewer's zone, so a time read anywhere names the same instant. A naive value comes back
unchanged: its zone is unknown, and guessing the viewer's would move it by hours.
:func:`zone_known` says which kind a value is, so a display can mark a naive one as the
acquisition's local time.

Qt-free, so the export renderer, the reports and the headless paths can share it.
"""

from __future__ import annotations

import numbers
from datetime import datetime, timezone
from typing import Optional

# AutoScript's acquisition_datetime as the ThermoFisher driver saved it: the instrument
# PC's local time, to the second.
THERMOFISHER_FORMAT = "%m/%d/%Y %H:%M:%S"

DISPLAY_FORMAT = "%Y-%m-%d %H:%M"


def now_iso() -> str:
    """Now, in this machine's zone, with its offset: the one way to write a time."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def iso_from_posix(ts: float) -> str:
    """A POSIX time (a device's clock, a file's mtime) written as :func:`now_iso` would."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat()


def to_datetime(value) -> Optional[datetime]:
    """Read a recorded time, or None if it cannot be read.

    Accepts a POSIX number or numeric string, an ISO string with or without an offset,
    the ThermoFisher form, or a datetime. An aware result is in the viewer's zone; a
    naive one is returned as recorded.
    """
    if isinstance(value, datetime):
        return value.astimezone() if value.tzinfo is not None else value
    if isinstance(value, str):
        text = value.strip()
        try:
            value = float(text)
        except ValueError:
            pass
        else:
            return to_datetime(value)
        try:
            return to_datetime(datetime.fromisoformat(text))
        except ValueError:
            pass
        try:
            return datetime.strptime(text, THERMOFISHER_FORMAT)
        except ValueError:
            return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def zone_known(value) -> bool:
    """Whether *value* names an instant, rather than a clock time in an unknown zone."""
    parsed = to_datetime(value)
    return parsed is not None and parsed.tzinfo is not None


def format_time(value, fmt: str = DISPLAY_FORMAT) -> Optional[str]:
    """*value* formatted for display, in the viewer's zone when it is known.

    None when it cannot be read, so a caller can leave the field out.
    """
    parsed = to_datetime(value)
    return parsed.strftime(fmt) if parsed is not None else None
