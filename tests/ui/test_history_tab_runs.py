"""The History tab: one row per run, its operations, and a remembered filter (FIB-1256).

A lamella in a real experiment folder, its history written by hand and its
operation events in a real ``events.jsonl`` beside it, read by the tab as the app
reads them. Preferences go to a throwaway file.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

import fibsem.applications.autolamella.ui.lamella_task_image_widget as history_tab  # noqa: E402
import fibsem.config as cfg  # noqa: E402
from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EVENTS_FILENAME,
)
from fibsem.applications.autolamella.history_rows import HistoryFilter  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskState,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (  # noqa: E402
    LamellaTaskImageWidget,
    _ClickableRow,
)
from fibsem.structures import FibsemImage, MicroscopeState  # noqa: E402
from fibsem.util.timestamps import format_time  # noqa: E402
from fibsem.workflows.status import TaskStatus  # noqa: E402

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


def _run(name, task_id, minute, status=TaskStatus.Completed, message="", **outputs):
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
            status=TaskStatus.Failed,
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


def _settled(widget):
    """Let the re-fit a show starts run out. It rebuilds the rows 150 ms later,
    and rows rebuilt during a test's last pass of the event loop are not yet
    shown, so a test that read them then found its opened lines missing."""
    deadline = time.monotonic() + 5
    while widget._refit.isActive() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return widget


@pytest.fixture
def widget(qapp, lamella):
    widget = LamellaTaskImageWidget()
    widget.resize(420, 1600)
    widget.set_lamella(lamella)
    widget.show()
    _settled(widget)
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


def _loaded(widget):
    """Wait for the background loader to fill the thumbnails."""
    for _ in range(200):
        QApplication.processEvents()
        if widget._worker is None or widget._worker.isFinished():
            QApplication.processEvents()
            return
        widget._worker.wait(25)


@pytest.mark.parametrize("change", ["none", "filter"])
def test_a_thumbnail_opens_in_the_viewer_after_the_rows_are_rebuilt(
    widget, lamella, monkeypatch, change
):
    """Changing the filter rebuilds the rows from images already loaded. The
    click used to be connected only when an image first loaded, so after a
    filter change no thumbnail opened."""
    opened = []
    monkeypatch.setattr(
        history_tab,
        "open_image_viewer",
        lambda parent, items, index: opened.append(items[index].path),
    )
    _loaded(widget)
    if change == "filter":
        widget._filter_button._pick(HistoryFilter(show="images"))
        _shown(widget)

    (path, thumbnail), *_ = widget._placeholder_labels.items()
    QTest.mouseClick(thumbnail, Qt.LeftButton)

    assert opened == [path]


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
    # in the viewer's zone, as the tab shows it
    at = format_time(T0 + timedelta(minutes=10), "%H:%M")
    QApplication.processEvents()
    (name,) = [
        label
        for label in widget._content.findChildren(QLabel)
        if label.text() == POLISH and label.isVisibleTo(widget)
    ]
    (when,) = [
        label
        for label in widget._content.findChildren(QLabel)
        if label.text() == f"{at} · 1 m 30 s" and label.isVisibleTo(widget)
    ]
    assert when.parent() is name.parent()
    assert abs(when.geometry().center().y() - name.geometry().center().y()) <= 2


def test_switching_to_a_lamella_with_other_tasks_keeps_the_filter_menu(
    widget, tmp_path
):
    """The filter button sits in the header row, which every rebuild deletes.
    Loading another experiment clears the selection first, and that rebuild makes
    no new header for the button to move to, so it was deleted with the old one;
    the next lamella whose tasks differed rebuilt its menu on a deleted QMenu,
    PyQt raised, and the app aborted."""
    other = Lamella(path=tmp_path / "exp2" / "02-other", number=2, petname="other")
    other.path.mkdir(parents=True)
    other.task_history = [_run("Setup Lamella Position", "s1", 0)]

    widget.set_lamella(None)  # as loading another experiment does
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)  # the old header goes
    widget.set_lamella(other)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    texts = [a.text() for a in widget._filter_button.menu().actions()]
    assert "Setup Lamella Position" in texts and ROUGH not in texts
    assert "Setup Lamella Position" in _shown(widget)


def test_the_filter_menu_has_the_apps_menu_background(widget, qapp):
    """The header row sets an unscoped `background: transparent`, which reached
    the menu through the filter button, its parent: the menu drew see-through
    over the runs. Parentless, it gets the app's menu style like every other."""
    from fibsem.ui import stylesheets
    from fibsem.ui.tokens import SURFACE_COLOR

    before = qapp.styleSheet()
    qapp.setStyleSheet(stylesheets.NAPARI_STYLE)
    try:
        menu = widget._filter_button.menu()
        menu.ensurePolished()
        image = menu.grab().toImage()
        pixel = image.pixelColor(image.width() - 3, image.height() // 2)
        assert pixel.name() == SURFACE_COLOR.lower()
    finally:
        qapp.setStyleSheet(before)


def test_an_opened_mill_lists_each_stage_with_its_drift_correction(qapp, tmp_path):
    """The drift corrections a mill ran fold under it, each with its figure,
    rather than being lines of their own (FIB-1351)."""
    lamella = Lamella(path=tmp_path / "exp" / "01-lam", number=1, petname="lam")
    lamella.path.mkdir(parents=True)
    lamella.task_history = [_run(ROUGH, "r1", 0)]
    figure = tmp_path / "Alignment" / "Rough Mill 01-18-01-05"
    figure.mkdir(parents=True)
    (figure / "figure.png").write_bytes(b"")
    at = lambda s: (T0 + timedelta(seconds=s)).isoformat()  # noqa: E731
    stage = {"index": 0, "status": "completed", "milling_current": 7.4e-10}
    _write_events(
        lamella,
        (
            "r1",
            "alignment",
            {
                "name": "Rough Mill 01-18-01-05",
                "started_at": at(65),
                "results": [{"shift": {"x": 7e-8, "y": 0.0}, "success": True}],
                "status": "completed",
                "path": str(figure),
            },
        ),
        (
            "r1",
            "milling",
            {
                "status": "completed",
                "started_at": at(60),
                "stages": [
                    {
                        **stage,
                        "name": "Rough Mill 01",
                        "started_at": at(60),
                        "ended_at": at(80),
                    }
                ],
            },
        ),
    )
    widget = LamellaTaskImageWidget()
    widget.resize(420, 1600)
    widget.set_lamella(lamella)
    widget.show()
    _settled(widget)
    try:
        (mill,) = _operation_lines(widget)  # the alignment is not a line
        assert "1 stage · alignment ≤ 0.07 µm" in _shown(widget)

        mill.clicked.emit()

        texts = _shown(widget)
        assert texts[texts.index("Rough Mill 01") + 1] == "completed · 20 s · 740 pA"
        assert texts[texts.index("alignment") + 1] == "0.07 µm · 1 step"
        (link,) = [
            b
            for b in widget._content.findChildren(QPushButton)
            if b.text() == "Figure" and b.isVisibleTo(widget)
        ]
        assert str(figure / "figure.png") in link.toolTip()
    finally:
        widget._cancel_worker()
        widget.deleteLater()


def test_a_run_s_operations_fold_unless_one_needs_a_look(qapp, tmp_path):
    """A run that went to plan is one line; one with a skip shows its lines;
    a click opens a fold, and a refresh keeps it open (FIB-1361)."""
    lamella = Lamella(path=tmp_path / "exp" / "01-lam", number=1, petname="lam")
    lamella.path.mkdir(parents=True)
    lamella.task_history = [_run(ROUGH, "r1", 0), _run(POLISH, "p1", 10)]
    done = {"results": [{"shift": {"x": 1e-8, "y": 0.0}}], "status": "completed"}
    _write_events(
        lamella,
        ("r1", "alignment", {**done, "name": "lam - Rough Milling-18-00-30"}),
        ("r1", "autofocus", {"status": "completed", "beam_type": "ION"}),
        ("p1", "alignment", {"status": "skipped", "reason": "no reference image"}),
        ("p1", "autofocus", {"status": "completed", "beam_type": "ION"}),
    )
    widget = LamellaTaskImageWidget()
    widget.resize(420, 1600)
    widget.set_lamella(lamella)
    widget.show()
    _settled(widget)
    try:
        texts = _shown(widget)
        assert "2 operations · all completed" in texts
        assert "2 operations · 1 skipped" in texts
        # only the skipped run's lines show: its summary and two operations
        assert "skipped: no reference image" in texts
        assert len([t for t in texts if t == "Align"]) == 1

        (folded,) = [
            line
            for line in _operation_lines(widget)
            if line.toolTip() == "Click to show or hide the operations"
            and "2 operations · all completed"
            in [label.text() for label in line.findChildren(QLabel)]
        ]
        folded.clicked.emit()
        assert len([t for t in _shown(widget) if t == "Align"]) == 2

        widget.refresh()  # as after a task finishes
        assert len([t for t in _shown(widget) if t == "Align"]) == 2
    finally:
        widget._cancel_worker()
        widget.deleteLater()


def _append_outside_a_task(lamella, minute, kind, payload, actor="agent", item=None):
    """An operation no task ran (FIB-1375): the agent's autofocus between runs."""
    path = lamella.path.parent / EVENTS_FILENAME
    with open(path, "a") as f:
        record = {
            "kind": kind,
            "t": (T0 + timedelta(minutes=minute)).isoformat(),
            "session": "s1",
            "actor": actor,
            "item": item,
            "task": None,
            "payload": payload,
        }
        f.write(json.dumps(record) + "\n")


def test_operations_outside_a_task_are_a_row_among_the_runs(qapp, lamella):
    _append_outside_a_task(
        lamella,
        15,
        "autofocus",
        {
            "beam_type": "ION",
            "initial_working_distance": 0.0165,
            "working_distance": 0.0166,
            "status": "completed",
        },
    )
    # another lamella's, though outside a task, is that lamella's
    _append_outside_a_task(
        lamella, 16, "autofocus", {"status": "completed"}, item={"id": "other"}
    )
    widget = LamellaTaskImageWidget()
    widget.resize(420, 1600)
    widget.set_lamella(lamella)
    widget.show()
    try:
        texts = _shown(_settled(widget))
        rows = [t for t in texts if t in (ROUGH, POLISH, "Outside a task")]
        assert rows == [ROUGH, POLISH, "Outside a task", ROUGH]
        assert "Autofocus (agent)" in texts
        assert texts.count("Autofocus (agent)") == 1
        assert "3 runs · 4 m 30 s" in texts, "it is not a run"
        assert texts.count("Completed") == 2, "and has no status of its own"
    finally:
        widget._cancel_worker()
        widget.deleteLater()


# --- the workflow GIF (FIB-1311) -----------------------------------------------


@pytest.fixture
def opened(monkeypatch):
    """The export dialogs opened, without showing them."""
    from fibsem.ui.widgets.animation_export_dialog import AnimationExportDialog

    dialogs = []
    monkeypatch.setattr(
        AnimationExportDialog, "exec_", lambda self: dialogs.append(self)
    )
    return dialogs


def test_the_actions_menu_sits_after_the_filter_and_outlives_a_rebuild(widget, lamella):
    actions, flt = widget._actions_button, widget._filter_button
    assert actions.isVisibleTo(widget) and actions.toolTip() == "Actions"
    assert [a.text() for a in actions.menu().actions()] == [
        "Open Lamella Folder",
        "Export GIF...",
    ]
    assert actions.parent() is flt.parent(), "in the header row"
    assert actions.geometry().left() > flt.geometry().right(), "at the far right"

    widget.set_lamella(None)
    _settled(widget)
    assert not actions.isVisibleTo(widget)

    widget.set_lamella(lamella)  # a new header, after one was deleted
    _settled(widget)
    assert actions.isVisibleTo(widget)
    assert [a.text() for a in actions.menu().actions()] == [
        "Open Lamella Folder",
        "Export GIF...",
    ]


def test_the_gif_is_made_of_the_lamellas_task_images(widget, lamella, opened):
    widget._filter_button._pick(HistoryFilter(status="failed"))  # the filter
    _settled(widget)  # narrows the tab, not the GIF

    widget._action_export_gif.trigger()

    (dialog,) = opened
    assert [frame.title for frame in dialog.frames] == [ROUGH], (
        "Polishing saved no images"
    )
    assert dialog.title == lamella.name
    assert dialog.save_directory == os.fspath(lamella.path)


def test_a_lamella_with_no_task_images_says_so(qapp, tmp_path, opened, monkeypatch):
    from fibsem.ui import notification_service

    toasts = []
    monkeypatch.setattr(
        notification_service, "show_toast", lambda msg, kind="info": toasts.append(msg)
    )
    lamella = Lamella(path=tmp_path / "exp" / "01-lam", number=1, petname="lam")
    lamella.path.mkdir(parents=True)
    lamella.task_history = [_run(ROUGH, "r1", 0)]
    widget = LamellaTaskImageWidget()
    try:
        widget.set_lamella(lamella)
        widget.show()
        _settled(widget)

        widget._action_export_gif.trigger()

        assert opened == []
        assert toasts == [f"{lamella.name} has no task images to animate yet"]
    finally:
        widget._cancel_worker()
        widget.deleteLater()


def test_open_lamella_folder_opens_it_in_the_file_browser(widget, lamella, monkeypatch):
    opened = []
    monkeypatch.setattr(history_tab, "_open_path", opened.append)

    widget._action_open_folder.trigger()

    assert opened == [os.fspath(lamella.path)]
