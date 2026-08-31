"""The first question over the Responder seam: Confirm, answered by a click.

Instructions complete their future inside the handler; a question's handler shows
the prompt and returns, and the yes/no click completes the future later — the
deferred half of the seam. Every plain ``ask_user`` call (no detection, milling
or spot-burn variant) now takes this path; ``USER_RESPONSE`` and the polled
``WAITING_FOR_USER_INTERACTION`` handshake stay behind for the unconverted
variants only.

Each test calls ``ask_user`` from a real worker thread, spins the GUI loop until
the prompt is up, clicks, and reads the answer off the worker — the production
shape end to end.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.ui import ask_user


@pytest.fixture
def ui(qapp):
    """A real AutoLamellaUI, connected (Demo), same harness as the responder tests."""
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _ask_on_worker_thread(ui, qapp, msg="Continue?", pos="Continue", neg="Exit"):
    """Start a plain ask_user on a worker thread; spin until the prompt is up."""
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask_user(ui, msg=msg, pos=pos, neg=neg)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        qapp.processEvents()
        if ui.label_instructions.text() == msg and ui.pushButton_yes.isEnabled():
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the prompt never appeared")
    return thread, outcome


def _finish(thread, qapp, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "ask_user never returned"


def test_the_yes_click_answers_true(ui, qapp):
    thread, outcome = _ask_on_worker_thread(ui, qapp)

    # Mid-wait: the prompt is up, its buttons say what was asked, and the
    # waiting display state is on for the attention button and border.
    assert ui.pushButton_yes.text() == "Continue"
    assert ui.pushButton_no.text() == "Exit"
    assert ui.WAITING_FOR_USER_INTERACTION is True

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert outcome.get("answer") is True


def test_the_no_click_answers_false(ui, qapp):
    thread, outcome = _ask_on_worker_thread(ui, qapp)

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert outcome.get("answer") is False


def test_the_answer_does_not_travel_through_the_legacy_channel(ui, qapp):
    # USER_RESPONSE was the shared mutable the typed path retired; it no longer
    # exists, so a click writing it (or ask_user reading it) would have to
    # re-create it — which is exactly what this catches.
    thread, outcome = _ask_on_worker_thread(ui, qapp)

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert outcome.get("answer") is True
    assert not hasattr(ui, "USER_RESPONSE")


def test_the_prompt_comes_down_with_the_answer(ui, qapp):
    thread, _ = _ask_on_worker_thread(ui, qapp)

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert not ui.label_instructions.isVisibleTo(ui)
    assert not ui.pushButton_yes.isVisibleTo(ui)
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_a_stop_interrupts_a_prompt_nobody_answers(ui, qapp):
    thread, outcome = _ask_on_worker_thread(ui, qapp)

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)


def test_a_question_after_an_aborted_one_still_works(ui, qapp):
    # The aborted question's future is still parked in the responder; the next
    # question must cancel that corpse and proceed, not trip over it.
    thread, outcome = _ask_on_worker_thread(ui, qapp)
    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()
    assert isinstance(outcome.get("error"), InterruptedError)

    thread, outcome = _ask_on_worker_thread(ui, qapp, msg="Try again?")
    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert outcome.get("answer") is True
