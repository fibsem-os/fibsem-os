"""Suggested answers (FIB-1283): at Assist the agent suggests an answer to the
run's question, the agent card shows it (FIB-1407; the window says so through
`suggestion_changed`), and the operator's Accept answers as theirs. Three threads, like production: the worker asks, the GUI
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

from fibsem.acting import current_actor  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.workflows.interaction import (  # noqa: E402
    Confirm,
    ask,
)

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
    """``(kind, payload, actor)`` for each question event, as the record marks it."""
    events = []
    ui.ui_responder.add_question_observer(
        lambda kind, payload: events.append((kind, dict(payload), current_actor()))
    )
    return events


def test_the_operator_takes_the_agent_s_suggestion(ui, qapp, client):
    events = _record(ui)
    thread, outcome = _ask_on_worker(ui, qapp)
    question_id = ui.ui_responder.pending_question_and_id()[1]
    shown = []
    ui.suggestion_changed.connect(shown.append)
    assert ui.suggestion_shown is None

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

    # Not on the prompt: the card shows it.
    assert ui.suggestion_shown == {
        "question": "Continue to polishing?",
        "answer": "Continue",
        "reason": "trench is clear",
    }
    assert shown == [ui.suggestion_shown]

    assert ui.take_suggestion() is True
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert outcome.get("answer") is True
    assert ui.suggestion_shown is None
    assert shown[-1] is None

    kinds = [kind for kind, _, _ in events]
    assert kinds == ["prompt_raised", "prompt_suggested", "prompt_answered"]
    assert events[1][1]["reason"] == "trench is clear"
    # The suggestion is the agent's on the record; the prompt is the task's.
    assert [actor for _, _, actor in events] == ["task", "agent", "task"]
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
    assert ui.suggestion_shown is not None

    ui.pushButton_no.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert outcome.get("answer") is False
    assert ui.suggestion_shown is None
    assert ui.ui_responder.suggestion is None
    assert "suggested_by" not in events[-1][1]


def test_the_operator_dismisses_the_suggestion_with_why(ui, qapp, client):
    # FIB-1424: the card had Accept only; declining by answering another way
    # dropped the suggestion silently, so the agent never learned why.
    from fibsem.applications.autolamella.ui.agent_recent import RecentActions

    events = _record(ui)
    thread, outcome = _ask_on_worker(ui, qapp)
    question_id = ui.ui_responder.pending_question_and_id()[1]
    _post(
        qapp,
        client,
        "/app/workflow/prompt/suggest",
        {"response": True, "question_id": question_id, "reason": "looks done"},
    )
    assert ui.dismiss_suggestion("the face still curtains") is True
    assert ui.suggestion_shown is None and ui.ui_responder.suggestion is None
    # The question stays up for the operator.
    assert "answer" not in outcome
    assert ui.ui_responder.pending_question() is not None
    kind, payload, actor = events[-1]
    assert kind == "prompt_suggestion_dismissed" and actor == "operator"
    assert payload["question_id"] == question_id
    assert payload["reason"] == "the face still curtains"
    assert ui.dismiss_suggestion("again") is False  # gone already
    # Read again, the question says the suggestion was dismissed, and why
    # (sim session 2026-10-16).
    pending = client.get("/app/workflow/prompt", headers=AUTH).json()["pending"]
    assert pending["suggestion_dismissed"] == {"reason": "the face still curtains"}

    # Recent says so, with the reason.
    recent = RecentActions()
    for seq, (kind, payload, actor) in enumerate(events, start=1):
        recent.add({"seq": seq, "kind": kind, "payload": payload, "actor": actor})
    assert recent.rows[0].outcome == "dismissed: the face still curtains"

    ui.pushButton_no.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert outcome.get("answer") is False


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
    assert ui.suggestion_shown is None

    ui.pushButton_yes.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)


def test_a_suggestion_carries_the_image_its_evidence_is_in(
    ui, qapp, auth, tmp_path, monkeypatch
):
    """FIB-1424: as a request does, a suggestion names the image the agent
    judged on, inside the experiment; the card shows it."""
    monkeypatch.setattr(
        AutoLamellaAppContext, "experiment_path", property(lambda self: str(tmp_path))
    )
    inside = tmp_path / "ref_setup_ib.tif"
    inside.write_bytes(b"tif")
    outside = tmp_path.parent / "elsewhere.tif"
    outside.write_bytes(b"tif")
    app = build_server(ui.microscope, app_context=AutoLamellaAppContext(ui), auth=auth)
    events = _record(ui)
    with TestClient(app, raise_server_exceptions=False) as client:
        thread, outcome = _ask_on_worker(ui, qapp)
        question_id = ui.ui_responder.pending_question_and_id()[1]

        def suggest(image):
            return _post(
                qapp,
                client,
                "/app/workflow/prompt/suggest",
                {
                    "response": True,
                    "question_id": question_id,
                    "reason": "centred in both beams",
                    "image": image,
                },
            )

        refused = suggest(str(outside))
        assert refused.status_code == 403
        assert refused.json()["detail"]["error_type"] == "not_allowed"
        missing = suggest(str(tmp_path / "none.tif"))
        assert missing.json()["detail"]["error_type"] == "invalid"
        assert suggest(str(inside)).status_code == 200
        assert ui.ui_responder.suggestion[4] == str(inside)
        suggested = [p for kind, p, _ in events if kind == "prompt_suggested"]
        assert suggested[-1]["image"] == str(inside)
        assert ui.take_suggestion() is True
        _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
        thread.join(timeout=5)
