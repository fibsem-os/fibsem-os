"""A task's start and end are aware datetimes, written as ISO 8601 with their offset
(FIB-1197), and an older experiment's POSIX floats still load and show the same times.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
)
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT

# A completed Rough Milling run, as an older experiment.yaml holds it.
OLDER = {
    "name": "Rough Milling",
    "step": "FINISHED",
    "task_id": "8f1c2c1e-6c43-4b8e-9a7e-0d6a5f3b1c2d",
    "task_type": "MILL_ROUGH",
    "lamella_id": "a1b2c3",
    "start_timestamp": 1789355531.874894,
    "end_timestamp": 1789356212.25,
    "status": "Completed",
    "status_message": "",
    "outputs": {},
}


@pytest.fixture(params=["UTC", "Australia/Sydney", "America/Denver"])
def viewer_zone(request, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only")
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield request.param
    monkeypatch.undo()
    time.tzset()


def _shown(stamp: float) -> str:
    """What the task card showed before FIB-1197."""
    return datetime.fromtimestamp(stamp).strftime(TIME_DISPLAY_AMPM_SHORT)


def test_an_older_state_shows_the_times_it_always_did(viewer_zone):
    state = AutoLamellaTaskState.from_dict(OLDER)
    assert state.started_at == _shown(OLDER["start_timestamp"])
    assert state.completed_at == _shown(OLDER["end_timestamp"])
    assert state.duration == pytest.approx(
        OLDER["end_timestamp"] - OLDER["start_timestamp"]
    )
    assert state.start_timestamp.timestamp() == OLDER["start_timestamp"]


def test_an_older_state_saved_again_keeps_its_times(viewer_zone):
    state = AutoLamellaTaskState.from_dict(OLDER)
    saved = state.to_dict()
    # written as ISO with an offset, under the same key
    assert datetime.fromisoformat(saved["start_timestamp"]).tzinfo is not None
    again = AutoLamellaTaskState.from_dict(yaml.safe_load(yaml.safe_dump(saved)))
    assert again.start_timestamp == state.start_timestamp
    assert again.end_timestamp == state.end_timestamp
    assert (again.started_at, again.completed_at) == (
        state.started_at,
        state.completed_at,
    )


def test_a_new_state_records_aware_times():
    state = AutoLamellaTaskState(name="Setup")
    assert state.start_timestamp.tzinfo is not None
    assert state.end_timestamp is None
    assert state.completed_at == "in progress"
    assert state.duration == 0


def test_a_time_written_elsewhere_keeps_its_offset():
    written = dict(
        OLDER,
        start_timestamp="2026-09-13T21:12:11.874894-06:00",
        end_timestamp="2026-09-13T21:23:32.250000-06:00",
    )
    state = AutoLamellaTaskState.from_dict(written)
    assert state.start_timestamp.utcoffset() == timedelta(hours=-6)
    assert state.duration == pytest.approx(680.375106)
    assert state.to_dict()["start_timestamp"] == written["start_timestamp"]


def test_a_caller_passing_a_posix_float_gets_an_aware_time():
    state = AutoLamellaTaskState(
        name="Coincidence",
        end_timestamp=1789356212.25,
        status=AutoLamellaTaskStatus.Completed,
    )
    assert state.end_timestamp == datetime.fromtimestamp(1789356212.25, tz=timezone.utc)


def test_an_unreadable_time_reads_as_unknown():
    state = AutoLamellaTaskState.from_dict(dict(OLDER, end_timestamp="not a time"))
    assert state.end_timestamp is None
    assert state.completed_at == "in progress"
