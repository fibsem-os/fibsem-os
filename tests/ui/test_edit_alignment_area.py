"""The alignment-area question over the Responder seam: EditAlignmentArea.

The old flow needed two handshakes and a read-back: ``update_alignment_area_ui``
emitted the area and polled ``WAITING_FOR_USER_INTERACTION``; after the click it
emitted ``"clear"`` and polled ``WAITING_FOR_UI_UPDATE``; then it read
``image_widget.get_alignment_area()`` across the seam. Now the answer IS the
area: the Continue click reads it and hides the overlay on the GUI thread, and
the future carries the rect back — one wait, no flags, no read-back.

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

from fibsem.applications.autolamella.workflows.ui import update_alignment_area_ui
from fibsem.structures import FibsemRectangle

INITIAL = FibsemRectangle(left=0.3, top=0.3, width=0.2, height=0.25)


@pytest.fixture(scope="module")
def ui(qapp):
    """The embedded AutoLamellaUI of a real main window, connected (Demo)."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
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
    """Start update_alignment_area_ui on a worker thread; spin until the prompt is up."""
    outcome = {}

    def target():
        try:
            outcome["area"] = update_alignment_area_ui(
                alignment_area=INITIAL, parent_ui=ui, msg=msg, validate=True
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
    assert ui.WAITING_FOR_USER_INTERACTION is True

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    area = outcome.get("area")
    assert isinstance(area, FibsemRectangle)
    assert area.width == pytest.approx(INITIAL.width)
    assert area.height == pytest.approx(INITIAL.height)
    assert ui.WAITING_FOR_USER_INTERACTION is False


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


def test_the_request_carries_the_fib_display_frame(ui, qapp):
    # A remote responder must answer from the request alone: the frame the
    # area is judged against travels in it — the widget's FIB reference at
    # ask time, grabbed whole (never mutated), so no GUI-thread entry.
    thread, _ = _edit_on_worker_thread(ui, qapp)

    request = ui.ui_responder.pending_question()
    assert request.image is ui.image_widget.ib_image
    assert request.image is not None

    ui.pushButton_yes.click()
    _finish(thread, qapp)


def test_headless_and_unvalidated_return_the_input(ui):
    assert update_alignment_area_ui(INITIAL, parent_ui=None, validate=True) is INITIAL
    assert update_alignment_area_ui(INITIAL, parent_ui=ui, validate=False) is INITIAL


def test_a_stop_interrupts_an_edit_nobody_confirms(ui, qapp):
    thread, outcome = _edit_on_worker_thread(ui, qapp)

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)
