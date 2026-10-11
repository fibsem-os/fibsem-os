"""The experiment's, grid's, overlay's, proposal's and decision's times, the session's
and the experiment reference's are aware datetimes, written as ISO 8601 with their
offset (FIB-1197); what older files hold, a POSIX float, loads and shows the same.
"""

import time
from datetime import datetime

import pytest
import yaml

from fibsem.applications.autolamella.proposals import Decision, Proposal
from fibsem.applications.autolamella.structures import (
    Experiment,
    GridRecord,
    OverlayRecord,
    QualityRecord,
    Verdict,
)
from fibsem.applications.autolamella.tools.experiments import (
    created_order,
    filter_experiments,
)
from fibsem.config import ExperimentSummary, peek_experiment
from fibsem.correlation.structures import CorrelationResult
from fibsem.structures import FibsemExperimentRef, SessionInfo, SlotCalibration
from fibsem.util.timestamps import format_time

# 2026-09-14 03:12:11 UTC, as an older file holds it
STAMP = 1789355531.0


def clock(value) -> str:
    """The review tab's clock (Qt-free here, so this runs on CI)."""
    return (format_time(value, "%H:%M") or "") if value else ""


@pytest.fixture(params=["UTC", "Australia/Sydney", "America/Denver"])
def viewer_zone(request, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only")
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield request.param
    monkeypatch.undo()
    time.tzset()


def _shown(stamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """What the float showed before FIB-1197."""
    return datetime.fromtimestamp(stamp).strftime(fmt)


def _aware_at_stamp(value) -> bool:
    return value.tzinfo is not None and value.timestamp() == STAMP


def _saved(record) -> dict:
    """What actually reaches the file."""
    return yaml.safe_load(yaml.safe_dump(record.to_dict()))


def test_an_older_experiment_shows_when_it_was_created(viewer_zone, tmp_path):
    experiment = Experiment(path=tmp_path, name="older")
    stored = dict(experiment.to_dict(), created_at=STAMP)

    loaded = Experiment.from_dict(stored)
    assert _aware_at_stamp(loaded.created_at)
    assert loaded.created_at.strftime("%d %b %Y, %H:%M") == _shown(
        STAMP, "%d %b %Y, %H:%M"
    )
    again = Experiment.from_dict(_saved(loaded))
    assert again.created_at == loaded.created_at


def test_an_experiment_that_never_said_stays_unknown(tmp_path):
    experiment = Experiment(path=tmp_path, name="older")
    stored = experiment.to_dict()
    del stored["created_at"]
    assert Experiment.from_dict(stored).created_at is None


def test_a_new_experiment_writes_iso_with_its_offset(tmp_path):
    written = Experiment(path=tmp_path, name="new").to_dict()["created_at"]
    assert datetime.fromisoformat(written).tzinfo is not None


@pytest.mark.parametrize("record", [GridRecord, OverlayRecord])
def test_an_older_grid_or_overlay_keeps_its_time(viewer_zone, record):
    stored = dict(_saved(record.from_dict({"name": "g"})), created_at=STAMP)
    loaded = record.from_dict(stored)
    assert _aware_at_stamp(loaded.created_at)
    assert record.from_dict(_saved(loaded)).created_at == loaded.created_at


def test_an_older_proposal_and_decision_show_the_same_clock(viewer_zone):
    stored = {
        "kind": "task_result",
        "values": {},
        "provenance": {"task_id": "run-1"},
        "decisions": [
            {"outcome": "Confirmed", "author": "operator", "timestamp": STAMP + 60}
        ],
        "created_at": STAMP,
    }
    proposal = Proposal.from_dict(stored)
    assert clock(proposal.created_at) == _shown(STAMP, "%H:%M")
    assert clock(proposal.decisions[0].timestamp) == _shown(STAMP + 60, "%H:%M")

    # A proposal saved before ids is named from its stored time, and keeps the
    # name once saved again, now with the time as ISO.
    saved = _saved(proposal)
    assert isinstance(saved["created_at"], str)
    assert Proposal.from_dict(saved).id == proposal.id
    assert Proposal.from_dict(stored).id == proposal.id


def test_a_decision_with_no_time_has_none():
    decision = Decision.from_dict({"outcome": "Confirmed", "author": "operator"})
    assert decision.timestamp is None
    assert clock(decision.timestamp) == ""


def test_the_session_and_the_experiment_reference_read_older_floats(viewer_zone):
    session = SessionInfo.from_dict({"recorded_at": STAMP, "plugins": {}})
    assert _aware_at_stamp(session.recorded_at)
    assert SessionInfo.from_dict(_saved(session)).recorded_at == session.recorded_at

    ref = FibsemExperimentRef.from_dict({"id": "x", "date": STAMP})
    assert _aware_at_stamp(ref.date)
    assert FibsemExperimentRef.from_dict(ref.to_dict()).date == ref.date
    # an image that did not say: "Unknown" was what a reader got; now None
    assert FibsemExperimentRef.from_dict({"id": "x"}).date is None


def test_the_listing_reads_older_floats_and_sorts_the_unknown_last(tmp_path):
    older = tmp_path / "older"
    older.mkdir()
    (older / "experiment.yaml").write_text(
        yaml.safe_dump({"name": "older", "created_at": STAMP})
    )
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "experiment.yaml").write_text(yaml.safe_dump({"name": "unknown"}))

    summary = peek_experiment(str(older / "experiment.yaml"))
    assert _aware_at_stamp(summary.created_at)
    nothing = peek_experiment(str(unknown / "experiment.yaml"))
    assert nothing.created_at is None

    ordered = sorted([nothing, summary], key=created_order, reverse=True)
    assert ordered == [summary, nothing]
    # a POSIX bound, as the cli passes, still filters
    assert filter_experiments([summary, nothing], since=STAMP - 1) == [summary]
    assert filter_experiments([summary, nothing], until=STAMP - 1) == [nothing]


def test_a_summary_built_without_a_time_is_unknown():
    assert ExperimentSummary(path="p", name="n").created_at is None


def test_an_older_verdict_keeps_when_it_was_set(viewer_zone):
    stored = {"verdict": "FAILED", "reason": "cracked", "updated_at": STAMP}
    verdict = QualityRecord.from_dict(stored)
    assert _aware_at_stamp(verdict.updated_at)
    again = QualityRecord.from_dict(_saved(verdict))
    assert again.updated_at == verdict.updated_at
    # the pre-DefectState shape too
    old = QualityRecord.from_dict({"has_defect": True, "updated_at": STAMP})
    assert _aware_at_stamp(old.updated_at)


def test_setting_a_verdict_records_an_aware_time():
    verdict = QualityRecord()
    verdict.set_defect("cracked", Verdict.FAILED)
    assert verdict.updated_at.tzinfo is not None
    assert QualityRecord().updated_at is None


def test_an_older_correlation_result_keeps_its_time(viewer_zone):
    result = CorrelationResult.from_dict(
        dict(CorrelationResult().to_dict(), updated_at=STAMP)
    )
    assert _aware_at_stamp(result.updated_at)
    assert CorrelationResult(updated_at=STAMP).updated_at == result.updated_at
    again = CorrelationResult.from_dict(_saved(result))
    assert again.updated_at == result.updated_at


@pytest.mark.parametrize(
    "written",
    ["2026-09-02T11:20:00", "2026-09-02T11:20:00+10:00"],  # before FIB-1190, since
)
def test_a_slot_calibration_reads_its_time_as_written(written):
    record = SlotCalibration.from_dict(
        {
            "orientation": "SEM",
            "pre_tilt": 35.0,
            "rotation_reference": 0.0,
            "captured_at": written,
        }
    )
    assert record.captured_at == datetime.fromisoformat(written)
    assert record.to_dict()["captured_at"] == written
    assert not record.is_builtin


def test_a_builtin_slot_calibration_has_no_time():
    record = SlotCalibration.builtin(35.0, 0.0)
    assert record.captured_at is None and record.is_builtin
    assert record.to_dict()["captured_at"] == ""
    assert SlotCalibration.from_dict(record.to_dict()).is_builtin
