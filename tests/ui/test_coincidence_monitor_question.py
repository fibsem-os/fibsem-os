"""A supervised coincidence mill over the Responder seam opens the viewer in
monitor mode; an unsupervised one never does (FIB-912).

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

from fibsem.applications.autolamella.workflows.interaction import RunMillingTask, ask

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


def test_supervised_mill_opens_the_viewer_attached_then_releases_it(ui, qapp):
    request = RunMillingTask(
        config=_coincidence_config(), enabled=True, confirm=lambda: True, message=MSG
    )
    thread, outcome = _ask_on_worker_thread(ui, request)
    _pump_until(
        qapp,
        lambda: ui.label_instructions.text() == MSG and ui.pushButton_yes.isEnabled(),
        what="the Run Milling prompt",
    )

    ui.pushButton_yes.click()  # Run Milling
    _pump_until(
        qapp,
        lambda: _viewer(ui) is not None and _viewer(ui).in_monitor_mode,
        what="the viewer in monitor mode",
    )
    viewer = _viewer(ui)
    running = ui.milling_task_config_widget.milling_widget.running_config
    assert running is not None
    # attached to the strategies of the config actually being run
    assert viewer._active_strategies == [
        stage.strategy for stage in running.enabled_stages
    ]

    # the (stubbed) mill finishes: the viewer is released, the prompt is re-parked
    ui._mill_gate.set()
    _pump_until(qapp, lambda: not viewer.in_monitor_mode, what="monitor released")
    _pump_until(
        qapp,
        lambda: ui.label_instructions.text() == MSG and ui.pushButton_yes.isEnabled(),
        what="the re-parked prompt",
    )
    ui.pushButton_no.click()  # Continue
    _pump_until(qapp, lambda: not thread.is_alive(), what="the waiter to return")
    assert "error" not in outcome, outcome.get("error")
    assert outcome["config"].name == request.config.name


def test_unsupervised_mill_never_opens_the_viewer(ui, qapp):
    request = RunMillingTask(
        config=_coincidence_config(), enabled=True, confirm=lambda: False, message=MSG
    )
    ui._mill_gate.set()  # nothing to observe mid-run: let the mill finish at once
    thread, outcome = _ask_on_worker_thread(ui, request)
    _pump_until(qapp, lambda: not thread.is_alive(), what="the waiter to return")
    assert "error" not in outcome, outcome.get("error")
    viewer = _viewer(ui)
    assert viewer is None or not viewer.in_monitor_mode
