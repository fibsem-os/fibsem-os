"""Tests for the on-disk correlation history view (FIB-299).

``LamellaCorrelation.discover`` reconstructs a lamella's runs from the timestamped
``Correlation/<ts>/correlation.json`` folders the app already writes; ``latest``
returns the newest. No Qt.
"""
import os

from fibsem.correlation.history import LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)


def _write_run(root, name, x):
    folder = os.path.join(root, name)
    os.makedirs(folder)
    inp = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=x), point_type=PointType.FIB)]
    )
    CorrelationState(input_data=inp).save(os.path.join(folder, "correlation.json"))
    return folder


def test_discover_orders_runs_and_latest_is_newest(tmp_path):
    _write_run(str(tmp_path), "2026-07-20-09-00-00", x=1.0)
    _write_run(str(tmp_path), "2026-07-24-14-00-00", x=9.0)

    hist = LamellaCorrelation.discover(str(tmp_path))
    assert [r.name for r in hist.runs] == [
        "2026-07-20-09-00-00",
        "2026-07-24-14-00-00",
    ]
    latest = hist.latest()
    assert latest.name == "2026-07-24-14-00-00"
    assert latest.state.input_data.fib_coordinates[0].point.x == 9.0


def test_discover_missing_or_empty_dir_has_no_runs(tmp_path):
    assert LamellaCorrelation.discover(str(tmp_path / "nope")).latest() is None
    assert LamellaCorrelation.discover(str(tmp_path)).runs == []


def test_discover_skips_unreadable_and_non_run_folders(tmp_path):
    _write_run(str(tmp_path), "2026-07-24-14-00-00", x=5.0)
    os.makedirs(str(tmp_path / "empty-folder"))              # no correlation file
    bad = tmp_path / "2026-07-25-08-00-00"
    bad.mkdir()
    (bad / "correlation.json").write_text("{ not json")      # unreadable

    hist = LamellaCorrelation.discover(str(tmp_path))
    assert [r.name for r in hist.runs] == ["2026-07-24-14-00-00"]  # the good one only


def test_discover_reads_a_legacy_run_folder(tmp_path):
    """A run folder from before FIB-264 holds correlation_data.json, not
    correlation.json — still discoverable."""
    folder = tmp_path / "2026-07-19-10-00-00"
    folder.mkdir()
    CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=3.0), point_type=PointType.FIB)]
    ).save(str(folder / "correlation_data.json"))

    latest = LamellaCorrelation.discover(str(tmp_path)).latest()
    assert latest is not None
    assert latest.state.input_data.fib_coordinates[0].point.x == 3.0
