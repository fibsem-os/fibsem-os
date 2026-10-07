"""fibsem.util.timestamps: one way to write a time, one parser for every recorded form.

The recorded values are copied from real files: the FIB and FM overviews of a
2026-09-13 grid run, and AutoScript's string from a ThermoFisher image.
"""

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fibsem.util.timestamps import (
    format_time,
    iso_from_posix,
    now_iso,
    to_datetime,
    zone_known,
)

# FIB overview, microscope_state.timestamp: 2026-09-14 03:12:11.874894 UTC.
FIB_POSIX = 1789355531.874894
# FM overview of the same run, acquisition_date: the acquiring machine's clock, no zone.
FM_NAIVE = "2026-09-13T21:19:40.974286"
# ThermoFisher image, microscope_state.timestamp as the driver wrote it.
THERMOFISHER = "07/16/2026 11:07:27"


@pytest.fixture
def viewer_zone(monkeypatch):
    """Read times as a viewer in *zone* would."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only")

    def set_zone(zone: str) -> None:
        monkeypatch.setenv("TZ", zone)
        time.tzset()

    yield set_zone
    monkeypatch.undo()
    time.tzset()


def _in_local_zone(value: datetime) -> bool:
    return value.utcoffset() == value.astimezone().utcoffset()


def test_now_iso_carries_this_machines_offset():
    written = now_iso()
    parsed = datetime.fromisoformat(written)
    assert parsed.tzinfo is not None
    assert abs(parsed - datetime.now(timezone.utc)) < timedelta(seconds=5)


def test_iso_from_posix_names_the_same_instant():
    written = iso_from_posix(FIB_POSIX)
    assert datetime.fromisoformat(written).timestamp() == pytest.approx(FIB_POSIX)
    assert to_datetime(written) == to_datetime(FIB_POSIX)


@pytest.mark.parametrize(
    "value",
    [FIB_POSIX, int(FIB_POSIX), str(FIB_POSIX), np.float64(FIB_POSIX)],
)
def test_a_posix_time_is_an_instant_in_the_viewers_zone(value):
    parsed = to_datetime(value)
    assert parsed.timestamp() == pytest.approx(float(value))
    assert _in_local_zone(parsed)
    assert zone_known(value)


def test_an_iso_time_with_an_offset_is_read_in_the_viewers_zone():
    parsed = to_datetime("2026-09-13T21:19:40.974-06:00")
    assert parsed == datetime(2026, 9, 14, 3, 19, 40, 974000, tzinfo=timezone.utc)
    assert _in_local_zone(parsed)
    assert zone_known("2026-09-13T21:19:40.974-06:00")


@pytest.mark.parametrize(
    "value, expected",
    [
        (FM_NAIVE, datetime(2026, 9, 13, 21, 19, 40, 974286)),
        ("2026-08-19 15:44:41", datetime(2026, 8, 19, 15, 44, 41)),
        (THERMOFISHER, datetime(2026, 7, 16, 11, 7, 27)),
    ],
)
def test_a_time_with_no_zone_is_returned_as_recorded(value, expected):
    parsed = to_datetime(value)
    assert parsed == expected
    assert parsed.tzinfo is None
    assert not zone_known(value)


def test_a_datetime_passes_through():
    naive = datetime(2026, 9, 13, 21, 19)
    assert to_datetime(naive) is naive
    aware = datetime(2026, 9, 14, 3, 19, tzinfo=timezone.utc)
    assert to_datetime(aware) == aware
    assert _in_local_zone(to_datetime(aware))


@pytest.mark.parametrize(
    "value", [None, "", "not a date", True, float("nan"), float("inf"), "nan", [1.0]]
)
def test_an_unreadable_value_is_none(value):
    assert to_datetime(value) is None
    assert format_time(value) is None
    assert not zone_known(value)


def test_a_posix_time_moves_to_the_viewers_zone_and_a_naive_one_does_not(viewer_zone):
    viewer_zone("Asia/Tokyo")
    assert format_time(FIB_POSIX) == "2026-09-14 12:12"
    # The FM overview's naive string cannot be moved into the viewer's zone.
    assert format_time(FM_NAIVE) == "2026-09-13 21:19"


def test_the_same_instant_formats_the_same_from_every_aware_form(viewer_zone):
    viewer_zone("UTC")
    forms = [
        FIB_POSIX,
        iso_from_posix(FIB_POSIX),
        "2026-09-13T21:12:11.874894-06:00",
        "2026-09-14T13:12:11.874894+10:00",
    ]
    assert {format_time(form, "%Y-%m-%d %H:%M:%S") for form in forms} == {
        "2026-09-14 03:12:11"
    }


def test_format_time_takes_a_format():
    assert format_time(THERMOFISHER, "%H:%M:%S") == "11:07:27"
