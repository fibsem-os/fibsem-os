"""The run-milling question as an agent sees and answers it (FIB-1297).

A person at the milling question sees the patterns on the FIB image, can drag
them, and reads the buttons; after a mill the same question is asked again
with Run Milling meaning *mill again*. Over HTTP the agent now gets the same:
the patterns drawn on the frame, each stage's geometry, the button labels and
how many times it has milled. A True on the re-ask needs ``run_again`` -- the
"yes" that would otherwise mill twice -- and an answer can move patterns,
checked against the frame before anything is clicked.

The mill itself is stubbed at the milling widget's import site, as in
test_run_milling_question.py: the widget's thread, buttons and finished signal
are real.
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

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import (
    RunMillingTask,
    ask,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.server import AuthConfig, build_server
from fibsem.structures import BeamType, FibsemImage, Point

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
MSG = "Check the patterns, then run."


@pytest.fixture
def ui(qapp, monkeypatch):
    """A real AutoLamellaUI, connected (Demo), with the mill stubbed and a FIB
    frame on display for the patterns to sit on."""
    from fibsem.ui.widgets import milling_widget as mw

    runs = []

    def fake_run_milling_task(microscope, config, parent_ui=None, **kwargs):
        runs.append(config)
        time.sleep(0.05)

    monkeypatch.setattr(mw, "run_milling_task", fake_run_milling_task)
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    frame = FibsemImage.generate_blank_image(
        resolution=(1536, 1024), hfw=80e-6, random=True
    )
    frame.metadata.image_settings.beam_type = BeamType.ION
    widget.image_widget.ib_image = frame
    widget._mill_runs = runs
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def client(ui):
    app = build_server(
        ui.microscope,
        app_context=AutoLamellaAppContext(ui),
        auth=AuthConfig.generate(arm_control=True, token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _off_thread(qapp, call):
    """Run an HTTP call off the GUI thread and spin until it returns: the
    server marshals to the GUI thread, which here is the one spinning."""
    result = {}
    thread = threading.Thread(target=lambda: result.update(r=call()), daemon=True)
    thread.start()
    _spin_until(qapp, lambda: "r" in result)
    return result["r"]


def _pending(qapp, client):
    return _off_thread(
        qapp, lambda: client.get("/app/prompt", headers=AUTH).json()["pending"]
    )


def _answer(qapp, client, response, nonce, value=None):
    body = {"response": response, "nonce": nonce}
    if value is not None:
        body["value"] = value
    return _off_thread(
        qapp, lambda: client.post("/app/prompt/answer", headers=AUTH, json=body)
    )


def _ask(ui, qapp):
    config = FibsemMillingTaskConfig(
        name="rough-mill", stages=[FibsemMillingStage(name="trench")]
    )
    outcome = {}

    def target():
        try:
            outcome["config"] = ask(
                ui.ui_responder, RunMillingTask(config=config, message=MSG)
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: ui.ui_responder.pending_question() is not None)
    return thread, outcome


def _reasked(ui, qapp, nonce):
    """Wait for the mill to finish and the question to come back."""
    _spin_until(
        qapp,
        lambda: ui.ui_responder.pending_question_and_nonce()[1] not in (None, nonce),
    )


def test_the_first_ask_shows_the_patterns_and_names_the_buttons(ui, qapp, client):
    thread, outcome = _ask(ui, qapp)
    pending = _pending(qapp, client)

    assert pending["type"] == "RunMillingTask"
    assert pending["positive"] == "Run Milling"
    assert pending["negative"] == "Continue"
    assert pending["runs"] == 0
    assert [s["name"] for s in pending["stages"]] == ["trench"]
    assert pending["stages"][0]["index"] == 0
    assert "point" in pending["stages"][0]["pattern"]
    # The patterns drawn on the frame the widget draws them on, with its scale.
    image = pending["image"]
    assert image["image_b64_jpeg"]
    assert image["full_width"] == 1536
    assert image["hfw"] == pytest.approx(80e-6)

    ui.pushButton_no.click()
    _spin_until(qapp, lambda: outcome)
    thread.join(timeout=5)
    assert ui._mill_runs == []


def test_a_yes_on_the_reask_does_not_mill_again_unless_it_says_so(ui, qapp, client):
    thread, outcome = _ask(ui, qapp)
    first = _pending(qapp, client)
    assert _answer(qapp, client, True, first["nonce"]).status_code == 200
    _reasked(ui, qapp, first["nonce"])
    assert len(ui._mill_runs) == 1

    reask = _pending(qapp, client)
    assert reask["runs"] == 1
    assert reask["last_error"] is None
    assert "AGAIN" in reask["message"]

    # The trap: "yes" read as "done" would mill twice. Refused, nothing clicked.
    refused = _answer(qapp, client, True, reask["nonce"])
    assert refused.status_code == 422
    assert "run_again" in refused.json()["detail"]["message"]
    assert len(ui._mill_runs) == 1
    assert ui.ui_responder.pending_question_and_nonce()[1] == reask["nonce"]

    # Said on purpose, it mills again.
    again = _answer(qapp, client, True, reask["nonce"], {"run_again": True})
    assert again.status_code == 200
    _reasked(ui, qapp, reask["nonce"])
    assert len(ui._mill_runs) == 2
    assert _pending(qapp, client)["runs"] == 2

    # And false accepts what was milled.
    final = ui.ui_responder.pending_question_and_nonce()[1]
    assert _answer(qapp, client, False, final).status_code == 200
    _spin_until(qapp, lambda: outcome)
    thread.join(timeout=5)
    assert "error" not in outcome
    assert len(ui._mill_runs) == 2


def test_an_answer_moves_a_pattern_before_it_mills(ui, qapp, client):
    thread, outcome = _ask(ui, qapp)
    nonce = _pending(qapp, client)["nonce"]

    moved = {"stages": {"0": {"point": {"x": -10e-6, "y": 5e-6}}}}
    applied = _answer(qapp, client, True, nonce, moved)
    assert applied.status_code == 200
    assert applied.json()["adjusted"] is True
    _reasked(ui, qapp, nonce)

    milled = ui._mill_runs[0].stages[0].pattern.point
    assert (milled.x, milled.y) == pytest.approx((-10e-6, 5e-6))
    # The operator's editor shows where it went.
    assert _pending(qapp, client)["stages"][0]["pattern"]["point"][
        "x"
    ] == pytest.approx(-10e-6)

    final = ui.ui_responder.pending_question_and_nonce()[1]
    _answer(qapp, client, False, final)
    _spin_until(qapp, lambda: outcome)
    thread.join(timeout=5)


@pytest.mark.parametrize(
    "value, says",
    [
        ({"stages": {"0": {"point": {"x": 39e-6, "y": 0}}}}, "outside the FIB image"),
        ({"stages": {"3": {"point": {"x": 0, "y": 0}}}}, "does not exist"),
        ({"stages": {"0": {"x": 0, "y": 0}}}, "point"),
        ({"run_agian": True}, "unknown keys"),
    ],
)
def test_a_bad_edit_is_refused_and_mills_nothing(ui, qapp, client, value, says):
    thread, outcome = _ask(ui, qapp)
    nonce = _pending(qapp, client)["nonce"]

    refused = _answer(qapp, client, True, nonce, value)
    assert refused.status_code == 422
    assert refused.json()["detail"]["error_type"] == "invalid_value"
    assert says in refused.json()["detail"]["message"]
    assert ui._mill_runs == []
    assert ui.ui_responder.pending_question_and_nonce()[1] == nonce

    ui.pushButton_no.click()
    _spin_until(qapp, lambda: outcome)
    thread.join(timeout=5)
