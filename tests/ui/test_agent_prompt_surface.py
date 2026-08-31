"""Supervision answering end to end: a real workflow-thread question, seen and
answered over HTTP through the mounted server — the killer-app data path.

Three threads, like production: the worker asks through ``ask()``, the GUI spins,
and the 'agent' speaks HTTP from the test thread via TestClient."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import Confirm, ask
from fibsem.server import AuthConfig, build_server

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def ui(qapp, monkeypatch):
    import fibsem.config as fibsem_config

    arctis_config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _client(ui, arm_control):
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui),
        auth=AuthConfig.generate(arm_control=arm_control, token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _ask_on_worker(ui, qapp):
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(
                ui.ui_responder,
                Confirm("Continue to polishing?", positive="Continue", negative="Stop"),
            )
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: ui.ui_responder.pending_question() is not None)
    return thread, outcome


def test_agent_sees_and_answers_the_question_over_http(ui, qapp):
    with _client(ui, arm_control=True) as client:
        # Nothing pending yet: available, but no question.
        body = client.get("/app/prompt", headers=AUTH).json()
        assert body == {"available": True, "pending": None}

        thread, outcome = _ask_on_worker(ui, qapp)
        pending = client.get("/app/prompt", headers=AUTH).json()["pending"]
        assert pending["type"] == "Confirm"
        assert pending["message"] == "Continue to polishing?"
        assert pending["positive"] == "Continue"

        # POST from a separate thread, as production does: the answer applies on
        # the GUI thread, which in this test is the thread running the loop —
        # a blocking post from here would deadlock into the 10 s timeout.
        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/prompt/answer", headers=AUTH, json={"response": True}
            )

        poster = threading.Thread(target=do_post, daemon=True)
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        answered = posted["response"]
        assert answered.status_code == 200
        assert answered.json()["applied"] is True
        _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
        thread.join(timeout=5)
        assert outcome.get("answer") is True


def test_answering_requires_the_control_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        # Reading the prompt is observation: read scope suffices.
        assert client.get("/app/prompt", headers=AUTH).status_code == 200
        refused = client.post(
            "/app/prompt/answer", headers=AUTH, json={"response": True}
        )
        assert refused.status_code == 403
        assert refused.json()["detail"]["scope"] == "control"


def test_capabilities_reflect_the_armed_control_scope(ui, qapp):
    with _client(ui, arm_control=True) as client:
        scopes = client.get("/capabilities", headers=AUTH).json()["scopes"]
        assert scopes["control"] is True
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "answer_prompt" in names
    with _client(ui, arm_control=False) as client:
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "get_pending_prompt" in names
        assert "answer_prompt" not in names
