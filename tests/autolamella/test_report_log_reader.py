"""The report's log reader: the tables the PDF's steps, milling and detections come from.

The records are the dicts the task, milling and detection code log at DEBUG, as
Python reprs. They used to be read by string substitution, which corrupted some
values and failed on others; every case here is one it got wrong. The first
test runs the real producers on the Demo microscope, so a producer changing the
shape of a record the report reads fails it.
"""

import logging
from datetime import datetime

import pandas as pd
import pytest

from fibsem.applications.autolamella.tools.data import (
    format_pretty_dataframes,
    read_log_tables,
)

from ._replay_experiment import (
    LAMELLA,
    MILLING_STAGES,
    MILLING_TASK,
    record_demo_experiment,
)

T0 = datetime(2026, 9, 21, 14, 0, 0)
STATUS = (
    "{'msg': 'status', 'timestamp': '2026-09-21T14:00:00', 'lamella': %r, "
    "'lamella_id': 'L1', 'task_id': 'T1', 'task_type': 'MILL_ROUGH', "
    "'task_name': %r, 'task_step': %r}"
)


def _line(message: str, second: int = 0, function: str = "f") -> str:
    stamp = T0.replace(second=second).strftime("%Y-%m-%d %H:%M:%S,000")
    return f"{stamp} — root — DEBUG — {function}:1 — {message}\n"


def _log(tmp_path, *lines, encoding="utf-8"):
    path = tmp_path / "logfile.log"
    path.write_bytes("".join(lines).encode(encoding))
    return str(path)


# ── what the real producers write ────────────────────────────────────────────


def test_the_real_producers_records_are_read(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    try:
        root = record_demo_experiment(tmp_path / "exp", monkeypatch)
    finally:
        monkeypatch.undo()
    steps, milling, detections = read_log_tables(str(root / "logfile.log"))

    assert steps and {s["lamella"] for s in steps} == {"test"}
    assert "ACQUIRE_REFERENCE_IMAGES" in [s["task_step"] for s in steps]
    assert [m["stage"]["name"] for m in milling] == list(MILLING_STAGES)
    assert {m["milling_task_name"] for m in milling} == {MILLING_TASK}
    # each milling row carries the step it was milled in
    assert all(m["lamella"] == "test" for m in milling)
    assert detections == []
    assert (root / LAMELLA).is_dir()


# ── the values string substitution got wrong ─────────────────────────────────


def test_a_name_with_parentheses_is_not_rewritten(tmp_path):
    """`(rough)` used to become `[rough]`: parsed, and silently wrong."""
    log = _log(tmp_path, _line(STATUS % ("01-a", "Mill Trench (rough)", "MILL")))
    (step,) = read_log_tables(log)[0]
    assert step["task_name"] == "Mill Trench (rough)"


@pytest.mark.parametrize(
    "task_name, task_step",
    [
        ("Mill", "align_reference_None_pass"),  # None inside a word
        ("the user's choice", "MILL"),  # an apostrophe
        ("Mill — rough", "MILL"),  # the log's own delimiter, in a value
    ],
)
def test_values_that_used_to_fail_are_read(tmp_path, task_name, task_step):
    log = _log(tmp_path, _line(STATUS % ("01-a", task_name, task_step)))
    (step,) = read_log_tables(log)[0]
    assert (step["task_name"], step["task_step"]) == (task_name, task_step)


def test_a_milling_record_with_numpy_values_is_read(tmp_path):
    """numpy 2 writes `np.float64(2e-09)`; the value is the argument."""
    milling = (
        "{'msg': 'milling_task', 'milling_task_id': 'M1', "
        "'milling_task_name': 'Rough Milling', 'idx': 0, "
        "'stage': {'name': 'Rough Mill (01)', 'milling': "
        "{'milling_current': np.float64(2e-09)}, 'pattern': {'depth': np.float64(1e-06)}}, "
        "'start_time': 1789964600.0, 'end_time': 1789964660.0, "
        "'timestamp': '2026-09-21T14:01:00'}"
    )
    log = _log(
        tmp_path,
        _line(STATUS % ("01-a", "Mill Rough", "MILL")),
        _line(milling, second=5),
    )
    (row,) = read_log_tables(log)[1]
    assert row["stage"]["name"] == "Rough Mill (01)"
    assert row["stage"]["milling"]["milling_current"] == pytest.approx(2e-9)
    assert row["lamella"] == "01-a" and row["task_name"] == "Mill Rough"


def test_a_log_written_as_cp1252_is_read(tmp_path):
    """Older Windows installs wrote the em-dash delimiter as cp1252."""
    log = _log(tmp_path, _line(STATUS % ("01-a", "Mill", "MILL")), encoding="cp1252")
    assert len(read_log_tables(log)[0]) == 1


# ── what is read and what is refused ─────────────────────────────────────────


def test_a_grid_task_s_step_is_not_a_lamella_step(tmp_path):
    grid = STATUS.replace(
        "'lamella': %r, 'lamella_id': 'L1'", "'grid': %r, 'grid_id': 'G1'"
    )
    log = _log(tmp_path, _line(grid % ("grid-01", "Overview", "ACQUIRE")))
    assert read_log_tables(log)[0] == []


def test_a_reported_record_that_cannot_be_read_is_counted(tmp_path, caplog):
    """Not data -- an enum repr -- is refused whole, and said so; records the
    report does not read are not its business."""
    log = _log(
        tmp_path,
        _line("{'msg': 'status', 'beam': <BeamType.ION: 2>}"),
        _line("{'msg': 'get_stage_position', 'pos': <not data>}"),
        _line(STATUS % ("01-a", "Mill", "MILL"), second=1),
    )
    with caplog.at_level(logging.WARNING):
        steps, _, _ = read_log_tables(log)
    assert len(steps) == 1
    (warning,) = [r for r in caplog.records if "could not be read" in r.message]
    assert warning.message.startswith("1 records")


# ── the detection summary ────────────────────────────────────────────────────


def _detection(feature: str, dx: float, second: int) -> str:
    # the shape `detection.utils.save_ml_feature_data` logs
    correct = dx == 0
    return _line(
        "{'msg': 'feature_detection', 'fname': 'ml-01', 'feature': %r, "
        "'px': {'x': 10.0, 'y': 20.0}, 'dpx': {'x': %r, 'y': 0.0}, "
        "'dm': {'x': %r, 'y': 0.0}, 'is_correct': %r, 'beam_type': 'ION', "
        "'pixelsize': 1e-08, 'checkpoint': 'model.pt'}"
        % (feature, dx, dx * 1e-8, correct),
        second=second,
    )


def test_the_detection_summary_counts_correct_detections(tmp_path):
    """`is_correct` is a real bool now. The summary compared it with the
    string "True", which would have emptied it."""
    log = _log(
        tmp_path,
        _line(STATUS % ("01-a", "Setup", "DETECT")),
        _detection("NeedleTip", 0.0, 1),
        _detection("NeedleTip", 5.0, 2),
        _detection("LamellaCentre", 0.0, 3),
    )
    steps, milling, detections = read_log_tables(log)
    assert [d["is_correct"] for d in detections] == [True, False, True]

    dfs = {
        "experiment": pd.DataFrame(
            columns=[
                "lamella_name",
                "last_completed",
                "milling_angle",
                "is_completed",
                "is_failure",
            ]
        ).astype({"milling_angle": float}),  # numeric, as the experiment's is
        "workflow": pd.DataFrame(
            columns=["order", "task_name", "required", "attention"]
        ),
        "task_history": pd.DataFrame(
            columns=["lamella_name", "task_name", "completed_at", "duration"]
        ).astype({"duration": float}),
        "tasks": pd.DataFrame(steps).assign(duration=0.0),
        "milling": pd.DataFrame(
            columns=[
                "lamella",
                "task_name",
                "milling_task_name",
                "stage.name",
                "start_time",
                "end_time",
                "stage.milling.milling_current",
                "stage.pattern.depth",
            ]
        ).astype(float),
        "detection": pd.json_normalize(detections),
    }
    summary = format_pretty_dataframes(dfs)["detection_summary"]
    by_feature = dict(zip(summary["Feature"], summary["Percentage"]))
    assert by_feature == {"LamellaCentre": "100.0%", "NeedleTip": "50.0%"}
