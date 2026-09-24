"""Coincidence milling over the Responder seam. Supervised: one question,
RunCoincidenceMilling, that the viewer's run mode answers -- the viewer runs
the mill. Automated: the task runs it and tells the viewer to watch and let
go (FIB-912).

A real AutoLamellaUI on the simulated Arctis, the milling question asked from
a worker thread as the task asks it. The mill itself is stubbed at the milling
widget's import site (as test_run_milling_question does), so the run goes
through the widget's real thread, buttons and finished signal without beam time.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest
import yaml

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from fibsem.applications.autolamella.workflows.interaction import (
    ReleaseCoincidenceMilling,
    RunCoincidenceMilling,
    WatchCoincidenceMilling,
    ask,
)

MSG = "Coincidence mill: check the boxes, then run."


@pytest.fixture
def ui(qapp, tmp_path, monkeypatch):
    import fibsem.config as fconfig
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.ui.widgets import milling_widget as mw

    with open(os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    monkeypatch.setattr(
        fconfig, "COINCIDENCE_MILLING_CONFIG_PATH", str(tmp_path / "cmc.yaml")
    )

    # The stubbed mill holds until the test lets it finish. A fixed sleep raced
    # the viewer's construction on CI: opening the window took longer than the
    # sleep, so the finished signal released monitor mode before the test's
    # first look at it.
    mill_gate = threading.Event()

    def fake_run_milling_task(microscope, config, parent_ui=None, **kwargs):
        mill_gate.wait(timeout=20.0)

    monkeypatch.setattr(mw, "run_milling_task", fake_run_milling_task)

    widget = AutoLamellaUI(parent_ui=None)
    widget._mill_gate = mill_gate
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: str(path),
    )
    widget.system_widget.connect_to_microscope()
    assert widget.microscope is not None and widget.microscope.fm is not None
    experiment = Experiment(path=str(tmp_path), name="monitor-question-test")
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    experiment.positions.append(lamella)
    widget.experiment = experiment
    yield widget
    viewer = getattr(widget, "_coincidence_viewer_window", None)
    if viewer is not None:
        viewer.close()
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _coincidence_config():
    from copy import deepcopy

    from fibsem.applications.autolamella.workflows._default_milling_config import (
        DEFAULT_MILLING_CONFIG,
    )
    from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
        MILL_COINCIDENT_KEY,
    )

    return deepcopy(DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY])


def _ask_on_worker_thread(ui, request):
    outcome = {}

    def target():
        try:
            outcome["config"] = ask(
                ui.ui_responder, request, abort=ui._workflow_stop_event.is_set
            )
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def _pump_until(qapp, predicate, timeout_s=15.0, what="condition"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _viewer(ui):
    return getattr(ui, "_coincidence_viewer_window", None)


def test_supervised_mill_is_run_from_the_viewer_and_continue_answers_it(ui, qapp):
    lamella = ui.experiment.positions[0]
    request = RunCoincidenceMilling(
        lamella=lamella, milling_config=_coincidence_config(), message=MSG
    )
    thread, outcome = _ask_on_worker_thread(ui, request)
    _pump_until(
        qapp,
        lambda: _viewer(ui) is not None and _viewer(ui).in_run_mode,
        what="the viewer in run mode",
    )
    viewer = _viewer(ui)
    # the main window's prompt is the same question, Continue only
    assert ui.label_instructions.text() == MSG
    assert "Coincidence Milling Viewer" in ui.hold.releases
    assert not ui.pushButton_no.isVisible()
    # the milling tab was not touched
    assert ui.milling_task_config_widget.milling_widget.running_config is None

    if viewer.fm_canvas._img_shape is None:
        viewer.set_fm_image(ui.microscope.fm.acquire_image())
        qapp.processEvents()
    H, W = viewer.fm_canvas._img_shape
    viewer.fm_canvas.rect_overlay.set_rect(0.1 * W, 0.2 * H, 0.3 * W, 0.4 * H)
    qapp.processEvents()

    viewer.btn_milling.click()  # Start Milling: the viewer's own widget runs it
    _pump_until(qapp, lambda: viewer._is_milling_active, what="the mill to start")
    running = viewer.milling_viewer_widget.milling_widget.running_config
    assert running is not None
    assert running.enabled_stages[0].strategy.config.bbox.left == pytest.approx(
        0.1, abs=0.01
    )

    ui._mill_gate.set()  # the stubbed mill finishes
    _pump_until(qapp, lambda: not viewer._is_milling_active, what="the mill to end")
    assert viewer.in_run_mode  # still ours: run again, or Continue

    viewer.btn_setup_skip.click()  # Continue, from the viewer
    _pump_until(qapp, lambda: not thread.is_alive(), what="the waiter to return")
    assert "error" not in outcome, outcome.get("error")
    answered = outcome["config"]
    assert answered is not None
    assert answered.enabled_stages[0].strategy.config.bbox.left == pytest.approx(
        0.1, abs=0.01
    )
    assert not viewer.in_run_mode
    assert not viewer.isVisible()  # Continue puts the window away
    assert ui.hold is None


def test_continue_without_milling_answers_none(ui, qapp):
    lamella = ui.experiment.positions[0]
    request = RunCoincidenceMilling(
        lamella=lamella, milling_config=_coincidence_config(), message=MSG
    )
    thread, outcome = _ask_on_worker_thread(ui, request)
    _pump_until(
        qapp,
        lambda: _viewer(ui) is not None and _viewer(ui).in_run_mode,
        what="the viewer in run mode",
    )
    ui.pushButton_yes.click()  # the main window's Continue answers too
    _pump_until(qapp, lambda: not thread.is_alive(), what="the waiter to return")
    assert "error" not in outcome, outcome.get("error")
    assert outcome["config"] is None
    viewer = _viewer(ui)
    assert not viewer.in_run_mode
    assert not viewer.isVisible()

    # the next site's question brings the same window back, not a new one
    thread, outcome = _ask_on_worker_thread(ui, request)
    _pump_until(qapp, lambda: _viewer(ui).in_run_mode, what="the viewer again")
    assert _viewer(ui) is viewer and viewer.isVisible()
    ui.pushButton_yes.click()
    _pump_until(qapp, lambda: not thread.is_alive(), what="the waiter to return")


def test_an_automated_mill_is_watched_then_released(ui, qapp):
    """The task runs the mill itself and tells the viewer to attach and let go;
    the viewer's Stop is the task's stop."""
    config = _coincidence_config()
    stops = []
    thread, outcome = _ask_on_worker_thread(
        ui,
        WatchCoincidenceMilling(
            milling_config=config, stop=lambda: stops.append(1), title="Coincident"
        ),
    )
    _pump_until(qapp, lambda: not thread.is_alive(), what="the watch to be taken")
    assert "error" not in outcome, outcome.get("error")
    viewer = _viewer(ui)
    assert viewer is not None and viewer.in_monitor_mode
    assert viewer.label_task_lock.text() == "Task owns this run"
    # attached to the very strategies the task is milling with
    assert viewer._active_strategies == [
        stage.strategy for stage in config.enabled_stages
    ]
    # the run's chrome arrives via the progress signal as for any mill; only
    # then does the button read Stop (before that it is Start, a manual run)
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    viewer._on_milling_progress(
        MillingProgress(
            status=MillingProgressStatus.STAGE_STARTED,
            stage_name="Coincident Milling 01",
            current_stage=0,
            total_stages=1,
        )
    )
    qapp.processEvents()
    assert viewer.btn_milling.text() == "Stop Milling"
    viewer.btn_milling.click()  # Stop: the task's stop, not the viewer's mill
    assert stops == [1]

    thread, outcome = _ask_on_worker_thread(ui, ReleaseCoincidenceMilling())
    _pump_until(qapp, lambda: not thread.is_alive(), what="the release")
    assert "error" not in outcome, outcome.get("error")
    assert not viewer.in_monitor_mode
