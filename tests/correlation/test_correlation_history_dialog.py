"""Tests for CorrelationHistoryDialog — the previous-run picker (FIB-257).

Headless PyQt5 with the offscreen platform (no pytest-qt), matching
tests/correlation/test_correlation_tab_widget.py. The dialog is a pure chooser,
so exec_() is never entered; the button handlers (_accept_selected /
_start_fresh) are driven directly and set the result via accept().
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QDialog

from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.correlation_history_dialog import (
    CorrelationHistoryDialog,
    _format_timestamp,
    _points_summary,
    _result_cell,
)


def _run(name, *, n_fib=0, n_fm=0, n_poi=0, result=None):
    inp = CorrelationInputData(
        fib_coordinates=[
            Coordinate(point=PointXYZ(x=float(i)), point_type=PointType.FIB)
            for i in range(n_fib)
        ],
        fm_coordinates=[
            Coordinate(point=PointXYZ(x=float(i)), point_type=PointType.FM)
            for i in range(n_fm)
        ],
        poi_coordinates=[
            Coordinate(point=PointXYZ(x=float(i)), point_type=PointType.POI)
            for i in range(n_poi)
        ],
    )
    state = CorrelationState(input_data=inp, result=result)
    return CorrelationRun(path=f"/tmp/{name}", name=name, state=state)


def _history(*runs) -> LamellaCorrelation:
    return LamellaCorrelation(runs=list(runs))


# oldest-first, as LamellaCorrelation.discover returns
OLD = "2026-07-20_09-00-00"
NEW = "2026-07-24_14-00-00"


def test_rows_are_newest_first_and_latest_preselected(qapp):
    dlg = CorrelationHistoryDialog(_history(_run(OLD), _run(NEW)))
    # display order is newest-first: row 0 is NEW, row 1 is OLD
    assert dlg.table.item(0, 0).text() == _format_timestamp(NEW)
    assert dlg.table.item(1, 0).text() == _format_timestamp(OLD)
    # newest row pre-selected
    assert dlg.table.currentRow() == 0


def test_seed_from_selected_returns_the_selected_run(qapp):
    dlg = CorrelationHistoryDialog(_history(_run(OLD), _run(NEW)))
    dlg.table.selectRow(1)  # the older run
    dlg._accept_selected()
    assert dlg.result() == QDialog.Accepted
    assert dlg.selected_run is not None
    assert dlg.selected_run.name == OLD


def test_default_selection_seeds_the_newest(qapp):
    dlg = CorrelationHistoryDialog(_history(_run(OLD), _run(NEW)))
    dlg._accept_selected()  # no explicit selection change -> newest (row 0)
    assert dlg.selected_run.name == NEW


def test_start_fresh_accepts_with_no_run(qapp):
    dlg = CorrelationHistoryDialog(_history(_run(OLD), _run(NEW)))
    dlg._start_fresh()
    assert dlg.result() == QDialog.Accepted
    assert dlg.selected_run is None


def test_points_summary_counts_each_type(qapp):
    dlg = CorrelationHistoryDialog(_history(_run(OLD, n_fib=4, n_fm=3, n_poi=1)))
    assert dlg.table.item(0, 1).text() == "4 FIB · 3 FM · 1 POI"


def test_points_summary_empty_is_dash():
    assert _points_summary(CorrelationState(input_data=CorrelationInputData())) == "—"


def test_result_cell_no_result_vs_rms():
    no_result = CorrelationState(input_data=CorrelationInputData())
    text, _ = _result_cell(no_result)
    assert text == "No result"

    inp = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=1.0), point_type=PointType.FIB)]
    )
    # a result whose snapshot matches the state's inputs -> current, not stale
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest()],
        rms_error=1.234,
        input_data=CorrelationInputData(
            fib_coordinates=[
                Coordinate(point=PointXYZ(x=1.0), point_type=PointType.FIB)
            ]
        ),
    )
    text, _ = _result_cell(CorrelationState(input_data=inp, result=result))
    assert text == "1.23 px RMS"


def test_result_cell_flags_stale_when_inputs_diverge():
    # state's current inputs differ from the result's fitted snapshot -> stale
    inp = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=9.0), point_type=PointType.FIB)]
    )
    result = CorrelationResult(
        poi=[CorrelationPointOfInterest()],
        rms_error=2.0,
        input_data=CorrelationInputData(
            fib_coordinates=[
                Coordinate(point=PointXYZ(x=1.0), point_type=PointType.FIB)
            ]
        ),
    )
    text, _ = _result_cell(CorrelationState(input_data=inp, result=result))
    assert "stale" in text


def test_format_timestamp_reformats_and_falls_back():
    assert _format_timestamp("2026-07-24_14-30-05") == "2026-07-24 14:30:05"
    # a non-conforming folder name is shown verbatim, not dropped
    assert _format_timestamp("not-a-timestamp") == "not-a-timestamp"
