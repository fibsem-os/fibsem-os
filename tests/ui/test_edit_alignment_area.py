"""The alignment-area question over the Responder seam: EditAlignmentArea.

The old flow needed two handshakes and a read-back: the workflow helper
emitted the area and polled ``WAITING_FOR_USER_INTERACTION``; after the click it
emitted ``"clear"`` and polled ``WAITING_FOR_UI_UPDATE``; then it read
``image_widget.get_alignment_area()`` across the seam. Now the answer IS the
area: the Continue click reads it and hides the overlay on the GUI thread, and
the future carries the rect back — one wait, no flags, no read-back.

No workflow task sends this request any more: the alignment area is asked on
the record (``AutoLamellaTask._validate_alignment_area``). The renderer stays
while the request does, so these ask it directly.

The real widgets answer here: the overlay round-trips through the actual quad-view
controller, which lives on the main window — so the fixture is the full offscreen
``AutoLamellaSingleWindowUI`` (minimap stubbed, as in test_mainui_workflow_status),
not a bare embedded window, whose ``get_alignment_area`` is None-for-no-controller.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.workflows.ui import _abort_requested
from fibsem.structures import FibsemRectangle
from fibsem.workflows.interaction import EditAlignmentArea, ask
from fibsem.workflows.status import Hold, HoldKind
from tests.run_state import Stopped

INITIAL = FibsemRectangle(left=0.3, top=0.3, width=0.2, height=0.25)


@pytest.fixture(scope="module")
def ui(qapp):
    """The embedded AutoLamellaUI of a real main window, connected (Demo)."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    # What _on_run_workflow_clicked does before any prompt can arrive: the main
    # window's _on_workflow_update reads _border_state unconditionally, and it is
    # first assigned on the Run click (the latent init gap test_mainui_workflow_status
    # documents; its fix rides the typed-status-event PR).
    window._set_border_state("idle")
    yield window.autolamella_ui
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # and breaks every later QEventLoop.exec_() — close with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def _edit_on_worker_thread(ui, qapp, msg="Adjust the area, then Continue."):
    """Ask EditAlignmentArea on a worker thread; spin until the prompt is up."""
    outcome = {}

    def target():
        try:
            outcome["area"] = ask(
                ui.ui_responder,
                EditAlignmentArea(initial=INITIAL, message=msg),
                abort=lambda: _abort_requested(ui.session.runner),
            )
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
        raise AssertionError("the alignment prompt never appeared")
    return thread, outcome


def _finish(thread, qapp, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "the asker never returned"


def test_the_click_answers_with_the_area_from_the_widget(ui, qapp):
    thread, outcome = _edit_on_worker_thread(ui, qapp)

    # Mid-wait: the overlay is up and editable, the prompt shows the caller's
    # message, and the waiting display state is on.
    shown = ui.image_widget.get_alignment_area()
    assert shown is not None
    assert ui.hold is not None and ui.hold.kind is HoldKind.question

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    area = outcome.get("area")
    assert isinstance(area, FibsemRectangle)
    assert area.width == pytest.approx(INITIAL.width)
    assert area.height == pytest.approx(INITIAL.height)
    assert ui.hold is None


def test_the_second_handshake_is_gone(ui, qapp):
    # The old flow set WAITING_FOR_UI_UPDATE for its embedded overlay-clear
    # wait; the clear runs inside the click's slot and the flag is gone.
    thread, _ = _edit_on_worker_thread(ui, qapp)

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert not hasattr(ui, "WAITING_FOR_UI_UPDATE")


def test_the_rect_survives_the_overlay_coming_down(ui, qapp):
    # clear_alignment_area hides but keeps: the answer is read before the hide,
    # and asking again must start from a working overlay, not a torn-down one.
    thread, outcome = _edit_on_worker_thread(ui, qapp)
    ui.pushButton_yes.click()
    _finish(thread, qapp)
    first = outcome["area"]

    thread, outcome = _edit_on_worker_thread(ui, qapp, msg="Round two.")
    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert isinstance(outcome["area"], FibsemRectangle)
    assert outcome["area"].width == pytest.approx(first.width)


def _answer_with_value_on_worker(ui, qapp, response, question_id, value):
    """Call AutoLamellaAppContext.answer_prompt from a worker thread, as the server does.

    The answer marshals to the GUI thread and blocks on the outcome — calling
    it from the test (GUI) thread would deadlock into the timeout.
    """
    from fibsem.applications.autolamella.server import AutoLamellaAppContext

    ctx = AutoLamellaAppContext(ui.session, window=ui)
    outcome = {}

    def target():
        outcome["result"] = ctx.answer_prompt(response, question_id, value=value)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "result" not in outcome and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert "result" in outcome, "answer_prompt never returned"
    return outcome["result"]


def test_a_value_answer_moves_the_area_and_answers_with_it(ui, qapp):
    thread, outcome = _edit_on_worker_thread(ui, qapp)
    _, question_id = ui.ui_responder.pending_question_and_id()

    events = []
    dispose = ui.ui_responder.add_question_observer(
        lambda kind, payload, actor: events.append((kind, {**payload, "actor": actor}))
    )
    proposed = {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.25}
    try:
        result = _answer_with_value_on_worker(ui, qapp, True, question_id, proposed)
    finally:
        dispose()
    _finish(thread, qapp)

    assert result["applied"] is True
    assert set(result["moved_from"]) == {"left", "top", "width", "height"}
    area = outcome.get("area")
    assert isinstance(area, FibsemRectangle)
    assert area.left == pytest.approx(proposed["left"])
    assert area.top == pytest.approx(proposed["top"])
    assert area.width == pytest.approx(proposed["width"])
    assert area.height == pytest.approx(proposed["height"])
    # The record shows the agent changed the geometry, not just accepted it.
    answered = [p for k, p in events if k == "prompt_answered"][-1]
    assert answered["actor"] == "agent"
    assert answered["adjusted"] is True


def test_echoing_the_area_is_not_an_adjustment(ui, qapp):
    # FIB-1330: sending back `current` accepts the area, it adjusts nothing.
    thread, outcome = _edit_on_worker_thread(ui, qapp)
    _, question_id = ui.ui_responder.pending_question_and_id()
    current = ui.image_widget.get_alignment_area().to_dict()

    events = []
    dispose = ui.ui_responder.add_question_observer(
        lambda kind, payload, actor: events.append((kind, {**payload, "actor": actor}))
    )
    try:
        result = _answer_with_value_on_worker(ui, qapp, True, question_id, current)
    finally:
        dispose()
    _finish(thread, qapp)

    assert result["applied"] is True
    assert "moved_from" not in result
    answered = [p for k, p in events if k == "prompt_answered"][-1]
    assert "adjusted" not in answered


def test_an_invalid_value_is_refused_and_the_prompt_stands(ui, qapp):
    thread, outcome = _edit_on_worker_thread(ui, qapp)
    _, question_id = ui.ui_responder.pending_question_and_id()
    before = ui.image_widget.get_alignment_area()

    result = _answer_with_value_on_worker(
        ui,
        qapp,
        True,
        question_id,
        {"left": 0.9, "top": 0.9, "width": 0.5, "height": 0.5},
    )

    assert result["applied"] is False
    assert "invalid_value" in result
    # Nothing was clicked and nothing moved: the question still stands.
    assert ui.ui_responder.pending_question_and_id()[1] == question_id
    after = ui.image_widget.get_alignment_area()
    assert after.left == pytest.approx(before.left)
    assert after.width == pytest.approx(before.width)

    ui.pushButton_yes.click()
    _finish(thread, qapp)
    assert isinstance(outcome.get("area"), FibsemRectangle)


def test_a_stop_interrupts_an_edit_nobody_confirms(ui, qapp):
    thread, outcome = _edit_on_worker_thread(ui, qapp)

    ui.session.runner.manager = Stopped()
    try:
        _finish(thread, qapp)
    finally:
        ui.session.runner.manager = None

    assert isinstance(outcome.get("error"), InterruptedError)
