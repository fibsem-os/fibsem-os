"""The Lamella tab's History panel follows a running workflow (FIB-1111).

Every status report re-selects the selected lamella, but the panel skips the
lamella it already shows, so a finished task's images appeared only after
selecting another lamella and back.
"""

import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module  # noqa: E402
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (  # noqa: E402
    LamellaTaskImageWidget,
)
from fibsem.applications.autolamella.workflows.tasks.manager import (  # noqa: E402
    WorkflowStatusUpdate,
)
from fibsem.structures import (  # noqa: E402
    FibsemImage,
    FibsemStagePosition,
    MicroscopeState,
)

TASK = "Trench"


@pytest.fixture(autouse=True)
def _preferences(tmp_path, monkeypatch):
    """A throwaway preferences file: the panel opens with the user's saved
    filter, and a developer's own must not hide the rows these tests look for."""
    import fibsem.config as cfg

    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))


@pytest.fixture
def main_ui(qapp):
    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    yield window
    ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(main_ui, tmp_path):
    ui = main_ui.autolamella_ui
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui._on_experiment_update()
    for i, name in enumerate(("one", "two")):
        ui.add_new_lamella(
            stage_position=FibsemStagePosition(x=i * 1e-4, y=0, z=0, r=0, t=0),
            name=name,
        )
    return exp


def _finish(lamella, task_name=TASK, value=0):
    """What a finished task leaves behind: its final reference image on disk,
    and a history entry."""
    image = FibsemImage.generate_blank_image(resolution=(64, 48))
    # Without a state the metadata is not written, and the panel needs its pixel size.
    image.metadata.microscope_state = MicroscopeState()
    image.data[:] = value
    image.save(os.path.join(lamella.path, f"ref_{task_name}_final_high_res.tif"))
    lamella.task_history.append(
        AutoLamellaTaskState(
            name=task_name,
            status=AutoLamellaTaskStatus.Completed,
            end_timestamp=datetime.now().timestamp(),
        )
    )


def _report(name, status):
    return WorkflowStatusUpdate(
        task_name=TASK,
        item_name=name,
        status=status,
        queue_position=1,
        queue_total=1,
    )


def _labels(widget):
    # A rebuild deleteLater()s the old rows; until that runs they are still children.
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return widget._content.findChildren(QLabel)


def _texts(widget):
    return [label.text() for label in _labels(widget)]


def test_a_finished_task_shows_in_the_history_of_the_selected_lamella(
    main_ui, experiment
):
    one = experiment.positions[0]
    main_ui._on_lamella_card_selected(one)
    history = main_ui.lamella_task_image_widget
    assert "No task runs yet." in _texts(history)

    _finish(one)
    main_ui._apply_status_report(_report("one", AutoLamellaTaskStatus.Completed))

    assert TASK in _texts(history)
    assert "No task runs yet." not in _texts(history)
    assert list(history._placeholder_labels) == [
        os.path.join(one.path, f"ref_{TASK}_final_high_res.tif")
    ]


def test_another_lamella_s_task_leaves_the_history_alone(main_ui, experiment):
    one, two = experiment.positions
    main_ui._on_lamella_card_selected(one)
    history = main_ui.lamella_task_image_widget
    shown = _labels(history)

    _finish(two)
    main_ui._apply_status_report(_report("two", AutoLamellaTaskStatus.Completed))

    assert _labels(history) == shown


def test_a_task_starting_leaves_the_history_alone(main_ui, experiment):
    one = experiment.positions[0]
    main_ui._on_lamella_card_selected(one)
    history = main_ui.lamella_task_image_widget
    shown = _labels(history)

    main_ui._apply_status_report(_report("one", AutoLamellaTaskStatus.InProgress))

    assert _labels(history) == shown


def _loaded(widget, path, timeout_ms=5000):
    """The pixmap the panel shows for *path*, once its background load lands."""
    waited = 0
    while path not in widget._pixmap_cache and waited < timeout_ms:
        QTest.qWait(20)
        waited += 20
    assert path in widget._pixmap_cache, "the image never loaded"
    return widget._pixmap_cache[path].toImage()


def test_a_re_run_shows_its_own_image_not_the_last_run_s(qapp, tmp_path):
    """Reference images are written to the same filename every run, so a pixmap
    cached by path is the previous run's picture."""
    from fibsem.applications.autolamella.structures import Lamella

    lamella = Lamella(path=tmp_path / "lam", number=1, petname="lam")
    os.makedirs(lamella.path, exist_ok=True)
    path = os.path.join(lamella.path, f"ref_{TASK}_final_high_res.tif")
    widget = LamellaTaskImageWidget()
    try:
        _finish(lamella, value=0)
        widget.set_lamella(lamella)
        first = _loaded(widget, path)

        _finish(lamella, value=255)  # the re-run, over the same file
        widget.refresh()
        second = _loaded(widget, path)

        centre = (first.width() // 2, first.height() // 2)
        assert first.pixelColor(*centre) != second.pixelColor(*centre)
    finally:
        widget._cancel_worker()
        widget.deleteLater()


def test_refresh_with_nothing_shown_does_nothing(qapp):
    widget = LamellaTaskImageWidget()
    try:
        widget.refresh()
        assert _texts(widget) == ["Select a lamella card to view task images."]
    finally:
        widget.deleteLater()
