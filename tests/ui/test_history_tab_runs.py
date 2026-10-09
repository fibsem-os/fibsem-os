"""The History tab: one row per run, its operations, and a remembered filter (FIB-1256).

A lamella in a real experiment folder, its history written by hand and its
operation events in a real ``events.jsonl`` beside it, read by the tab as the app
reads them. Preferences go to a throwaway file.
"""

import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EVENTS_FILENAME,
)
from fibsem.applications.autolamella.history_rows import HistoryFilter  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (  # noqa: E402
    LamellaTaskImageWidget,
    _ClickableRow,
)
from fibsem.structures import FibsemImage, MicroscopeState  # noqa: E402

T0 = datetime(2026, 7, 29, 18, 0, tzinfo=timezone(timedelta(hours=10)))
ROUGH, POLISH = "Rough Milling", "Polishing"


@pytest.fixture(autouse=True)
def preferences(tmp_path, monkeypatch):
    path = tmp_path / "prefs.yaml"
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(path))
    return path


def _image(lamella, name):
    image = FibsemImage.generate_blank_image(resolution=(64, 48))
    image.metadata.microscope_state = MicroscopeState()
    return os.path.relpath(image.save(str(lamella.path / name)), lamella.path)


def _run(
    name, task_id, minute, status=AutoLamellaTaskStatus.Completed, message="", **outputs
):
    return AutoLamellaTaskState(
        name=name,
        task_id=task_id,
        status=status,
        status_message=message,
        start_timestamp=T0 + timedelta(minutes=minute),
        end_timestamp=T0 + timedelta(minutes=minute, seconds=90),
        outputs=dict(outputs),
    )


def _write_events(lamella, *records):
    path = lamella.path.parent / EVENTS_FILENAME
    with open(path, "w") as f:
        for task_id, kind, payload in records:
            record = {
                "kind": kind,
                "t": (T0 + timedelta(minutes=1)).isoformat(),
                "item": {"id": lamella.id, "name": lamella.name},
                "task": {"id": task_id, "name": "?"},
                "payload": payload,
            }
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def lamella(tmp_path):
    lamella = Lamella(path=tmp_path / "exp" / "01-lam", number=1, petname="lam")
    lamella.path.mkdir(parents=True)
    rough_ref = _image(lamella, f"ref_{ROUGH}_final_res_01_ib.tif")
    lamella.task_history = [
        _run(ROUGH, "r1", 0, final_fib=[rough_ref]),
        _run(
            POLISH,
            "p1",
            10,
            status=AutoLamellaTaskStatus.Failed,
            message="the stage stopped",
        ),
        _run(ROUGH, "r2", 20, final_fib=[rough_ref]),
    ]
    shift = {"shift": {"x": 3e-7, "y": 4e-7}}
    _write_events(
        lamella,
        (
            "r2",
            "alignment",
            {
                "name": "lam - Rough Milling-18-20-30",
                "results": [shift, shift],
                "status": "completed",
            },
        ),
        (
            "p1",
            "alignment",
            {
                "status": "skipped",
                "reason": "reference image ref_alignment_ib.tif does not exist",
            },
        ),
    )
    return lamella


def _shown(widget):
    # rows added to a panel already on screen are shown on the next pass of the
    # event loop, and the rows they replaced are deleted on it
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return [
        label.text()
        for label in widget._content.findChildren(QLabel)
        if label.isVisibleTo(widget)
    ]


@pytest.fixture
def widget(qapp, lamella):
    widget = LamellaTaskImageWidget()
    widget.resize(420, 1600)
    widget.set_lamella(lamella)
    widget.show()
    yield widget
    widget._cancel_worker()
    widget.deleteLater()


def test_one_row_per_run_in_the_order_they_ran(widget):
    texts = _shown(widget)
    rows = [t for t in texts if t in (ROUGH, POLISH)]
    assert rows == [ROUGH, POLISH, ROUGH]
    assert "Failed" in texts and "the stage stopped" in texts
    assert "3 runs · 4 m 30 s" in texts


def test_each_run_shows_what_its_operations_did(widget):
    texts = _shown(widget)
    assert "1.00 µm · 2 steps" in texts
    assert "skipped: reference image ref_alignment_ib.tif does not exist" in texts
    assert texts.count("No operations recorded") == 1, "the first Rough Milling run"


def test_an_earlier_run_says_a_later_one_replaced_its_images(widget, lamella):
    texts = _shown(widget)
    assert any(t.startswith("Images replaced by the ") for t in texts)
    assert list(widget._placeholder_labels) == [
        str(lamella.path / f"ref_{ROUGH}_final_res_01_ib.tif")
    ], "shown once, on the run that wrote it last"


def test_an_experiment_that_records_no_events_claims_no_operations(qapp, lamella):
    os.remove(lamella.path.parent / EVENTS_FILENAME)
    widget = LamellaTaskImageWidget()
    try:
        widget.set_lamella(lamella)
        assert "No operations recorded" not in _shown(widget)
    finally:
        widget._cancel_worker()
        widget.deleteLater()


def test_the_filter_narrows_the_rows_and_says_so(widget):
    widget._filter_button._pick(HistoryFilter(status="failed"))

    texts = _shown(widget)
    assert [t for t in texts if t in (ROUGH, POLISH)] == [POLISH]
    assert "Filtered · 1 of 3 runs" in texts


def test_operations_only_drops_the_images(widget):
    widget._filter_button._pick(HistoryFilter(show="operations"))

    assert widget._placeholder_labels == {}
    assert [t for t in _shown(widget) if t in (ROUGH, POLISH)] == [POLISH, ROUGH]


def test_nothing_to_show_offers_to_show_everything(widget):
    widget._filter_button._pick(HistoryFilter(status="cancelled"))
    assert "No cancelled runs to show on this lamella." in _shown(widget)

    (reset,) = [
        b
        for b in widget._content.findChildren(QPushButton)
        if b.text() == "Show all runs"
    ]
    reset.click()

    assert widget._filter_button.filter == HistoryFilter()
    assert [t for t in _shown(widget) if t in (ROUGH, POLISH)] == [ROUGH, POLISH, ROUGH]


def test_the_filter_is_remembered_for_the_next_time(widget, qapp, lamella, preferences):
    widget._filter_button._pick(HistoryFilter(status="failed", task=POLISH))

    assert cfg.load_user_preferences().display.history_filter == {
        "show": "all",
        "status": "failed",
        "task": POLISH,
    }
    again = LamellaTaskImageWidget()
    try:
        again.set_lamella(lamella)
        assert again._filter_button.filter == HistoryFilter(
            status="failed", task=POLISH
        )
        assert [t for t in _shown(again) if t in (ROUGH, POLISH)] == [POLISH]
    finally:
        again._cancel_worker()
        again.deleteLater()


def test_the_menu_lists_the_lamellas_tasks(widget):
    texts = [a.text() for a in widget._filter_button.menu().actions()]
    assert texts == [
        "SHOW",
        "Everything",
        "Images only",
        "Operations only",
        "",
        "RUNS",
        "All runs",
        "Failed",
        "Cancelled",
        "",
        "TASK",
        "All tasks",
        ROUGH,
        POLISH,
        "",
        "Reset filters",
    ]


def _operation_lines(widget):
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return [
        line
        for line in widget._content.findChildren(_ClickableRow)
        if line.isVisibleTo(widget)
    ]


def test_clicking_an_operation_shows_its_steps_and_stays_open(widget):
    assert "Step 1" not in _shown(widget)
    alignment = _operation_lines(widget)[1]  # the second Rough Milling's

    alignment.clicked.emit()
    texts = _shown(widget)
    assert "Step 1" in texts and "(+300, +400) nm · score 0.00" in texts

    widget.refresh()  # as after a task finishes
    assert "Step 1" in _shown(widget), "a refresh keeps it open"

    _operation_lines(widget)[1].clicked.emit()
    assert "Step 1" not in _shown(widget)


def test_a_runs_time_sits_on_its_name_line(widget):
    QApplication.processEvents()
    (name,) = [
        label
        for label in widget._content.findChildren(QLabel)
        if label.text() == POLISH and label.isVisibleTo(widget)
    ]
    (when,) = [
        label
        for label in widget._content.findChildren(QLabel)
        if label.text() == "18:10 · 1 m 30 s" and label.isVisibleTo(widget)
    ]
    assert when.parent() is name.parent()
    assert abs(when.geometry().center().y() - name.geometry().center().y()) <= 2
