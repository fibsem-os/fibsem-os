"""Times as fibsem-os records and reads them: aware datetimes, written as ISO 8601
with their UTC offset.

``2026-09-13T21:19:40.974286-06:00`` pins the instant and keeps the clock time at the
site that wrote it. It is what OME's ``AcquisitionDate`` uses, and
``datetime.fromisoformat`` reads it on every supported Python. The rules for using
these are in CONTRIBUTING.md ("Times").

In memory a time is an aware ``datetime`` that keeps the offset it was written with:
:func:`to_datetime` parses, it does not convert. Comparing, subtracting and sorting
aware datetimes goes by the instant whatever their offsets, so nothing needs them in
one zone; only a display does, and :func:`format_time` converts to the viewer's zone
there. Keeping the offset is what lets a time saved again keep the site's clock time,
and lets a timeline read it on the instrument's clock (FIB-1196).

Files written before this carry one of three other forms, and stay readable:

- a POSIX float, which is what most drivers wrote into ``microscope_state.timestamp``;
- AutoScript's ``acquisition_datetime`` string, ``07/16/2026 11:07:27``, which the
  ThermoFisher driver wrote into the same field;
- a naive ISO string, the acquiring machine's local clock with no offset, which is
  what fluorescence images wrote into ``acquisition_date``.

A naive value comes back naive: its zone is unknown, and guessing the viewer's would
move it by hours. :func:`zone_known` says which kind a value is.

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


def now() -> datetime:
    """Now, aware, in this machine's zone: the one way to take a time."""
    return datetime.now(timezone.utc).astimezone()


def now_iso() -> str:
    """:func:`now` written as ISO 8601 with its offset."""
    return now().isoformat()


def from_posix(ts: float) -> datetime:
    """A POSIX time (a device's clock, a file's mtime) as an aware datetime in this
    machine's zone: the instant is exact, the zone is the only one there is to give."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()


def iso_from_posix(ts: float) -> str:
    """:func:`from_posix` written as ISO 8601 with its offset."""
    return from_posix(ts).isoformat()


def to_datetime(value) -> Optional[datetime]:
    """Read a recorded time, or None if it cannot be read.

    Accepts a POSIX number or numeric string, an ISO string with or without an offset,
    the ThermoFisher form, or a datetime. An aware value keeps the offset it was
    written with; a POSIX one is given this machine's; a naive one stays naive.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            return to_datetime(number)
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        try:
            return datetime.strptime(text, THERMOFISHER_FORMAT)
        except ValueError:
            return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            return from_posix(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    return None


def to_aware(value) -> Optional[datetime]:
    """:func:`to_datetime`, for a field that only holds instants: a value with no
    zone is not one, and reads as None."""
    parsed = to_datetime(value)
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def acquisition_datetime_of(metadata) -> Optional[datetime]:
    """When an image was acquired, from a beam or fluorescence image's metadata.

    ``acquisition_datetime`` when the image records it. Older images fall back to what
    they carried instead: a beam image's ``microscope_state.timestamp`` (a POSIX float,
    or the ThermoFisher string), a fluorescence image's ``acquisition_date``. Nothing is
    rewritten, so an image read this way is never "upgraded" to an instant it did not
    record. None when nothing readable is there.
    """
    recorded = to_datetime(getattr(metadata, "acquisition_datetime", None))
    if recorded is not None:
        return recorded
    # A beam image. Checked by attribute rather than read through, because
    # FibsemImageMetadata.acquisition_date is a property that calls this function.
    if hasattr(metadata, "microscope_state"):
        state = metadata.microscope_state
        return to_datetime(getattr(state, "timestamp", None))
    return to_datetime(getattr(metadata, "acquisition_date", None))


def zone_known(value) -> bool:
    """Whether *value* names an instant, rather than a clock time in an unknown zone."""
    return to_aware(value) is not None


def format_time(value, fmt: str = DISPLAY_FORMAT) -> Optional[str]:
    """*value* formatted for display: in the viewer's zone when it names an instant,
    as recorded when it does not.

    None when it cannot be read, so a caller can leave the field out.
    """
    parsed = to_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime(fmt)
