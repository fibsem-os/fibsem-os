"""An agent at Drive moves the stage and the window follows (FIB-1267).

The move comes through the server, off the GUI thread, the way a real agent's
does; the window's stage readout shows where it went.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from PyQt5.QtCore import QCoreApplication, QEvent

from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.structures import Experiment
from fibsem.server import AuthConfig, build_server

AGENT = {"Authorization": "Bearer agent"}
STEP = 20e-6  # metres: shows on the mm readout


@pytest.fixture(params=[True, False], ids=["experiment", "no-experiment"])
def window(qapp, tmp_path, request):
    """With an experiment open, and without one: the readout follows the
    microscope, not the experiment (FIB-1421)."""
    from fibsem.applications.autolamella.ui import window as module

    win = module.AutoLamellaWindow()
    win.system_widget.connect_to_microscope()
    ui = win
    if request.param:
        experiment = Experiment(path=tmp_path, name="drive-exp")
        os.makedirs(experiment.path, exist_ok=True)
        ui.experiment = experiment
        ui._setup_experiment_connections()
        win._on_experiment_update()

    yield win
    microscope = ui.microscope
    ui.disconnect_from_microscope()
    if microscope is not None:
        microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()


def _pump(qapp, predicate, timeout_s=10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_an_agent_at_drive_moves_the_stage_and_the_window_follows(window, qapp):
    ui = window
    auth = AuthConfig(token="agent")
    auth.set_level(AgentLevel.DRIVE)
    client = TestClient(
        build_server(ui.microscope, auth=auth), raise_server_exceptions=False
    )
    readout = ui.panel.movement_widget.position_widget.spinbox_x
    before = ui.microscope.get_stage_position().x
    qapp.processEvents()

    out = {}

    def _post():
        # Off the GUI thread, which keeps pumping, as a real agent's request is.
        out["response"] = client.post(
            "/microscope/services/stage_movement/commands/move_relative",
            headers=AGENT,
            json={
                "kwargs": {
                    "delta": {
                        "name": None,
                        "x": STEP,
                        "y": 0.0,
                        "z": 0.0,
                        "r": 0.0,
                        "t": 0.0,
                        "coordinate_system": "RAW",
                    }
                }
            },
        )

    threading.Thread(target=_post, daemon=True).start()
    assert _pump(qapp, lambda: "response" in out)
    assert out["response"].status_code == 200, out["response"].text

    after = ui.microscope.get_stage_position().x
    assert after == pytest.approx(before + STEP)
    assert _pump(qapp, lambda: readout.value() == pytest.approx(after * 1e3)), (
        f"readout {readout.value()} mm, stage {after * 1e3} mm"
    )
