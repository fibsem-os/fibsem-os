"""The agent's side of the Responder seam (FIB-851): a remote answer routed
through the same GUI-thread path as the buttons.

Same harness shape as test_qt_responder.py: a real worker thread asks through
``ask()`` while the test spins the GUI loop, and the "agent" answers from yet
another thread via ``submit_answer`` — three threads, exactly like production
(workflow, GUI, server)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import Confirm, ask


@pytest.fixture
def ui(qapp, monkeypatch):
    """A real, connected AutoLamellaUI — the prompt display path reaches real
    widgets, so a bare window aborts in handle_workflow_update. Same pinned
    config as test_qt_responder.py (the machine default differs on CI)."""
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


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _ask_on_worker(ui, qapp, request):
    """Start a workflow-thread ask; return (thread, outcome dict)."""
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(ui.ui_responder, request)
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: ui.ui_responder.pending_question() is not None)
    return thread, outcome


def test_agent_answer_wakes_the_asker_like_a_click(ui, qapp):
    request = Confirm("Continue to polishing?", positive="Continue", negative="Stop")
    thread, outcome = _ask_on_worker(ui, qapp, request)

    pending = ui.ui_responder.pending_question()
    assert isinstance(pending, Confirm)
    assert pending.message == "Continue to polishing?"

    answered = ui.ui_responder.submit_answer(True)
    _spin_until(qapp, answered.done)
    assert answered.result() is True  # the answer applied

    thread.join(timeout=5)
    assert outcome.get("answer") is True
    assert ui.ui_responder.pending_question() is None
    assert ui.WAITING_FOR_USER_INTERACTION is False  # display state torn down


def test_agent_no_answers_too(ui, qapp):
    thread, outcome = _ask_on_worker(ui, qapp, Confirm("Retry?", negative="Skip"))
    answered = ui.ui_responder.submit_answer(False)
    _spin_until(qapp, answered.done)
    assert answered.result() is True
    thread.join(timeout=5)
    assert outcome.get("answer") is False


def test_answer_with_nothing_pending_reports_false_not_an_error(ui, qapp):
    answered = ui.ui_responder.submit_answer(True)
    _spin_until(qapp, answered.done)
    assert answered.result() is False


def test_agent_and_human_race_produces_one_winner(ui, qapp):
    thread, outcome = _ask_on_worker(ui, qapp, Confirm("Continue?"))

    # Human clicks No on the GUI thread at the same moment the agent says Yes.
    clicked = ui.ui_responder.answer_confirm(False)
    agent = ui.ui_responder.submit_answer(True)
    _spin_until(qapp, agent.done)

    assert clicked is True  # the human's click applied first...
    assert agent.result() is False  # ...so the agent is told it lost, cleanly
    thread.join(timeout=5)
    assert outcome.get("answer") is False  # and the asker got exactly one answer


def test_an_answer_naming_the_question_applies(ui, qapp):
    thread, outcome = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    request, nonce = ui.ui_responder.pending_question_and_nonce()
    assert isinstance(request, Confirm)
    assert isinstance(nonce, int)

    answered = ui.ui_responder.submit_answer(True, nonce=nonce)
    _spin_until(qapp, answered.done)
    assert answered.result() is True
    thread.join(timeout=5)
    assert outcome.get("answer") is True


def test_a_wrong_nonce_is_refused_and_clicks_nothing(ui, qapp):
    from fibsem.applications.autolamella.workflows.interaction import (
        StalePromptError,
    )

    thread, outcome = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    _, nonce = ui.ui_responder.pending_question_and_nonce()

    answered = ui.ui_responder.submit_answer(True, nonce=nonce + 1)
    _spin_until(qapp, answered.done)
    with pytest.raises(StalePromptError):
        answered.result()
    # The refused answer clicked nothing: the question still stands.
    assert ui.ui_responder.pending_question() is not None
    assert outcome == {}

    ui.ui_responder.answer_confirm(False)
    thread.join(timeout=5)
    assert outcome.get("answer") is False


def test_each_posting_gets_a_fresh_nonce(ui, qapp):
    thread1, _ = _ask_on_worker(ui, qapp, Confirm("First?"))
    _, first = ui.ui_responder.pending_question_and_nonce()
    ui.ui_responder.answer_confirm(True)
    thread1.join(timeout=5)

    thread2, outcome2 = _ask_on_worker(ui, qapp, Confirm("Second?"))
    _, second = ui.ui_responder.pending_question_and_nonce()
    assert second != first
    # An answer still naming the first question is stale, not misapplied.
    answered = ui.ui_responder.submit_answer(True, nonce=first)
    _spin_until(qapp, answered.done)
    assert answered.exception() is not None
    ui.ui_responder.answer_confirm(False)
    thread2.join(timeout=5)
    assert outcome2.get("answer") is False


def test_question_lifecycle_events_carry_who_answered(ui, qapp):
    events = []
    ui.ui_responder.on_question_event = lambda kind, payload: events.append(
        (kind, payload)
    )
    try:
        # Operator answers via the button path.
        thread, _ = _ask_on_worker(ui, qapp, Confirm("First?"))
        ui.ui_responder.answer_confirm(True)
        thread.join(timeout=5)

        # Agent answers via the marshalled path.
        thread, _ = _ask_on_worker(ui, qapp, Confirm("Second?"))
        _, nonce = ui.ui_responder.pending_question_and_nonce()
        answered = ui.ui_responder.submit_answer(False, nonce=nonce)
        _spin_until(qapp, answered.done)
        thread.join(timeout=5)
    finally:
        ui.ui_responder.on_question_event = None

    kinds = [k for k, _ in events]
    assert kinds == [
        "prompt_raised",
        "prompt_answered",
        "prompt_raised",
        "prompt_answered",
    ]
    first_answer, second_answer = events[1][1], events[3][1]
    assert first_answer["answered_by"] == "operator"
    assert first_answer["response"] is True
    assert second_answer["answered_by"] == "agent"
    assert second_answer["response"] is False
    # The raise and its answer name the same posting.
    assert events[0][1]["nonce"] == events[1][1]["nonce"]
    assert events[2][1]["nonce"] == events[3][1]["nonce"]
    assert events[2][1]["message"] == "Second?"


def test_a_broken_observer_cannot_break_the_click(ui, qapp):
    def broken(kind, payload):
        raise RuntimeError("observer fell over")

    ui.ui_responder.on_question_event = broken
    try:
        thread, outcome = _ask_on_worker(ui, qapp, Confirm("Continue?"))
        assert ui.ui_responder.answer_confirm(True) is True
        thread.join(timeout=5)
        assert outcome.get("answer") is True  # the click still applied
    finally:
        ui.ui_responder.on_question_event = None


def test_an_abandoned_question_reports_cancelled(ui, qapp):
    events = []
    ui.ui_responder.on_question_event = lambda kind, payload: events.append(kind)
    try:
        thread, _ = _ask_on_worker(ui, qapp, Confirm("Continue?"))
        ui.ui_responder.abandon()
        thread.join(timeout=5)
    finally:
        ui.ui_responder.on_question_event = None
    assert events == ["prompt_raised", "prompt_cancelled"]


def test_aborted_question_reads_as_nothing_pending(ui, qapp):
    thread, outcome = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    # The asker's future is cancelled (abort path); the corpse must not look
    # like a live question to an agent.
    ui.ui_responder._pending_question[1].cancel()
    assert ui.ui_responder.pending_question() is None
    answered = ui.ui_responder.submit_answer(True)
    _spin_until(qapp, answered.done)
    assert answered.result() is True  # the stale prompt was taken down
    thread.join(timeout=5)
