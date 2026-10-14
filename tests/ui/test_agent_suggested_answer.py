"""Suggested answers (FIB-1283): at Assist the agent suggests an answer to the
run's question, the prompt shows it as a button, and the operator's click on it
answers as theirs. Three threads, like production: the worker asks, the GUI
spins, and the agent speaks HTTP from another thread."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.applications.autolamella.workflows.interaction import (  # noqa: E402
    Confirm,
    ask,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402

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


@pytest.fixture
def auth():
    auth = AuthConfig(token=TOKEN)
    auth.set_level(AgentLevel.ASSIST)
    return auth


@pytest.fixture
def client(ui, auth):
    app = build_server(ui.microscope, app_context=AutoLamellaAppContext(ui), auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _post(qapp, client, path, body):
    posted = {}

    def do_post():
        posted["response"] = client.post(path, headers=AUTH, json=body)

    threading.Thread(target=do_post, daemon=True).start()
    _spin_until(qapp, lambda: "response" in posted)
    return posted["response"]


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


def _record(ui):
    events = []
    ui.ui_responder.add_question_observer(
        lambda kind, payload: events.append((kind, dict(payload)))
    )
    return events


def test_the_operator_takes_the_agent_s_suggestion(ui, qapp, client):
    events = _record(ui)
    thread, outcome = _ask_on_worker(ui, qapp)
    question_id = ui.ui_responder.pending_question_and_id()[1]
    assert not ui.pushButton_suggestion.isVisibleTo(ui)

    # At Assist, answering directly is asking first; suggesting is not.
    refused = _post(
        qapp,
        client,
        "/app/workflow/prompt/answer",
        {"response": True, "question_id": question_id},
    )
    assert refused.json()["detail"]["error_type"] == "ask_first"
    suggested = _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id, "reason": "trench is clear"},
    )
    assert suggested.status_code == 200, suggested.text
    assert suggested.json()["suggested"] is True
    assert "answer" not in outcome  # nothing answered yet

    assert ui.pushButton_suggestion.isVisibleTo(ui)
    assert ui.pushButton_suggestion.text() == "Agent suggests: Continue"
    assert ui.label_suggestion.text() == "Why: trench is clear"

    ui.pushButton_suggestion.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert outcome.get("answer") is True
    assert not ui.pushButton_suggestion.isVisibleTo(ui)

    kinds = [kind for kind, _ in events]
    assert kinds == ["prompt_raised", "prompt_suggested", "prompt_answered"]
    assert events[1][1]["reason"] == "trench is clear"
    answered = events[2][1]
    assert answered["answered_by"] == "operator"
    assert answered["suggested_by"] == "agent"
    assert "agent suggested" in ui.question_timeline.current_text()


def test_answering_otherwise_drops_the_suggestion(ui, qapp, client):
    events = _record(ui)
    thread, outcome = _ask_on_worker(ui, qapp)
    question_id = ui.ui_responder.pending_question_and_id()[1]
    _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id, "reason": "looks done"},
    )
    assert ui.pushButton_suggestion.isVisibleTo(ui)

    ui.pushButton_no.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert outcome.get("answer") is False
    assert not ui.pushButton_suggestion.isVisibleTo(ui)
    assert ui.ui_responder.suggestion is None
    assert "suggested_by" not in events[-1][1]


def test_a_suggestion_needs_a_live_question_a_reason_and_the_agent_acting(
    ui, qapp, client, auth
):
    thread, outcome = _ask_on_worker(ui, qapp)
    question_id = ui.ui_responder.pending_question_and_id()[1]

    stale = _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id + 1, "reason": "x"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["error_type"] == "stale_prompt"

    no_reason = _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id},
    )
    assert no_reason.status_code == 422

    auth.set_level(AgentLevel.WATCH)
    watching = _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id, "reason": "x"},
    )
    assert watching.status_code == 403
    assert not ui.pushButton_suggestion.isVisibleTo(ui)

    ui.pushButton_yes.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
