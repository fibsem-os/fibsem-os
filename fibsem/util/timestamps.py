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
from datetime import datetime, timedelta, timezone, tzinfo
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


def _posix(value) -> Optional[float]:
    """*value* as a POSIX number, or None when it is not one."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_datetime(value) -> Optional[datetime]:
    """Read a recorded time, or None if it cannot be read.

    Accepts a POSIX number or numeric string, an ISO string with or without an offset,
    the ThermoFisher form, or a datetime. An aware value keeps the offset it was
    written with; a POSIX one is given this machine's; a naive one stays naive.
    """
    if isinstance(value, datetime):
        return value
    posix = _posix(value)
    if posix is not None:
        try:
            return from_posix(posix)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        try:
            return datetime.strptime(text, THERMOFISHER_FORMAT)
        except ValueError:
            return None
    return None


def to_iso(value) -> Optional[str]:
    """:func:`to_datetime` written as ISO 8601, with its offset when it has one: how a
    stored time is written (FIB-1197). None when it cannot be read."""
    parsed = to_datetime(value)
    return parsed.isoformat() if parsed is not None else None


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
    return to_datetime(_acquisition_value(metadata))


def acquisition_wall_time_of(
    metadata, zone: Optional[tzinfo] = None
) -> Optional[datetime]:
    """When an image was acquired, on the instrument's clock: :func:`wall_time_of`
    the value :func:`acquisition_datetime_of` reads, as recorded, so that an older
    image's POSIX time is read in *zone* rather than in this machine's."""
    return wall_time_of(_acquisition_value(metadata), zone)


def _acquisition_value(metadata):
    """The value an image's acquisition time is read from, as recorded."""
    recorded = getattr(metadata, "acquisition_datetime", None)
    if to_datetime(recorded) is not None:
        return recorded
    # A beam image. Checked by attribute rather than read through, because
    # FibsemImageMetadata.acquisition_date is a property that calls this function.
    if hasattr(metadata, "microscope_state"):
        return getattr(metadata.microscope_state, "timestamp", None)
    return getattr(metadata, "acquisition_date", None)


def wall_time_of(value, zone: Optional[tzinfo] = None) -> Optional[datetime]:
    """*value* as the clock on the wall where it was written read it: naive, for a
    timeline that keeps one run on the instrument's clock wherever it is read.

    An aware value keeps its own wall time, with its offset dropped rather than
    converted. A naive value is already a wall time, and is returned as it is. A
    POSIX value has no wall time of its own: it is read in *zone*, the instrument's
    as the experiment recorded it (``SessionInfo.zone``), or in this machine's when
    there is none. None when it cannot be read.
    """
    if zone is not None:
        posix = _posix(value)
        if posix is not None:
            try:
                return datetime.fromtimestamp(posix, tz=zone).replace(tzinfo=None)
            except (OverflowError, OSError, ValueError):
                return None
    parsed = to_datetime(value)
    return parsed.replace(tzinfo=None) if parsed is not None else None


def utc_offset_of(value: datetime) -> str:
    """An aware datetime's UTC offset, as ISO 8601 writes it: ``+10:00``."""
    return value.isoformat()[-6:]


def zone_from_offset(
    offset: Optional[str], name: Optional[str] = None
) -> Optional[tzinfo]:
    """The fixed zone a ``+10:00`` offset names, or None when it cannot be read."""
    if not offset:
        return None
    try:
        parsed = datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").utcoffset()
    except ValueError:
        return None
    if parsed is None or abs(parsed) >= timedelta(days=1):
        return None
    return timezone(parsed, name) if name else timezone(parsed)


def zone_label(zone: tzinfo) -> str:
    """A zone as a report names its clock: ``AEST (UTC+10:00)``, or ``UTC+10:00``
    for a zone with no name."""
    offset = timezone(zone.utcoffset(None)).tzname(None)
    name = zone.tzname(None)
    return offset if not name or name == offset else f"{name} ({offset})"


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
