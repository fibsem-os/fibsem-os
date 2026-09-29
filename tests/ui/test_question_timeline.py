"""The who-answered timeline: a dumb render of the responder's lifecycle feed.

Driven through the real window and the real answer paths — the rows can only
come from the single code path that applies answers, which is the widget's
whole design ("there is no update-the-timeline step to forget")."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui import (
    question_timeline_widget as timeline_module,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import Confirm, ask


@pytest.fixture(autouse=True)
def agent_feature_on(monkeypatch):
    """The strip is visible only with the agent feature; these tests exercise
    the display, so the gate is held open (and closed explicitly in the
    gating test)."""
    monkeypatch.setattr(timeline_module, "_feature_enabled", lambda: True)


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


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _ask_on_worker(ui, qapp, request):
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


def test_answers_land_as_rows_with_who_and_what(ui, qapp):
    timeline = ui.question_timeline
    assert timeline.isHidden()  # nothing to say until something happens

    # Operator answers via the click path.
    thread, _ = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    ui.ui_responder.answer_confirm(True)
    thread.join(timeout=5)

    # Agent answers via the marshalled path.
    thread, _ = _ask_on_worker(ui, qapp, Confirm("Again?"))
    _, nonce = ui.ui_responder.pending_question_and_nonce()
    answered = ui.ui_responder.submit_answer(False, nonce=nonce)
    _spin_until(qapp, answered.done)
    thread.join(timeout=5)

    # One visible line (the latest); the trail lives in memory + tooltip.
    assert timeline.current_text().startswith("agent · declined confirmation · ")
    rows = timeline.rows()
    assert len(rows) == 2
    assert rows[0].startswith("agent · declined confirmation · ")
    assert rows[1].startswith("you · answered confirmation · ")
    assert "nonce" not in rows[0]
    assert not timeline.isHidden()
    assert rows[1] in timeline._label.toolTip()


def test_a_withdrawn_question_reads_as_withdrawn(ui, qapp):
    thread, _ = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    ui.ui_responder.abandon()
    thread.join(timeout=5)
    assert ui.question_timeline.current_text().startswith("question withdrawn · ")


def test_the_remembered_trail_stays_bounded(ui, qapp):
    for _ in range(ui.question_timeline.HISTORY + 3):
        thread, _ = _ask_on_worker(ui, qapp, Confirm("Continue?"))
        ui.ui_responder.answer_confirm(True)
        thread.join(timeout=5)
    assert len(ui.question_timeline.rows()) == ui.question_timeline.HISTORY


def test_the_strip_stays_hidden_with_the_feature_off(ui, qapp, monkeypatch):
    # The no-behaviour-change invariant: a human-only supervised workflow must
    # look exactly as it did before the agent feature existed.
    monkeypatch.setattr(timeline_module, "_feature_enabled", lambda: False)
    thread, _ = _ask_on_worker(ui, qapp, Confirm("Continue?"))
    ui.ui_responder.answer_confirm(True)
    thread.join(timeout=5)
    assert ui.question_timeline.isHidden()
    assert len(ui.question_timeline.rows()) == 1  # still recording, just unseen
