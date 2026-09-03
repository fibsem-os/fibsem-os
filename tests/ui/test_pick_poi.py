"""The POI question over the Responder seam: PickPOI.

The old flow parked on the polled flag, then ran a second handshake — emit
``"clear"``, poll ``WAITING_FOR_UI_UPDATE`` — whose GUI half computed the point
and stored it in ``SELECTED_POI`` for the workflow thread to read back. Now the
answer IS the point: the request carries the image the marker is placed on, and
the Continue click reads the marker, converts against that image, and takes the
overlay down on the GUI thread. ``SELECTED_POI`` is gone.

The overlay lives on the main window's quad-view controller, so the fixture is
the full offscreen ``AutoLamellaSingleWindowUI`` — same harness as the
alignment-area tests.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from copy import deepcopy

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.workflows.ui import select_poi_ui
from fibsem.structures import BeamType, Point


@pytest.fixture(scope="module")
def window(qapp):
    """A real main window, connected (Demo), minimap stubbed."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    # The main window's _on_workflow_update reads _border_state unconditionally,
    # and it is first assigned on the Run click (the latent init gap
    # test_mainui_workflow_status documents; its fix rides the typed-status PR).
    win._set_border_state("idle")
    yield win
    if win.autolamella_ui.microscope is not None:
        win.autolamella_ui.microscope.disconnect()
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # and breaks every later QEventLoop.exec_() — close with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def ui(window):
    return window.autolamella_ui


def _fib_image(ui):
    """The FIB image on display — what the workflow would pass with the request."""
    return deepcopy(ui.image_widget.ib_image)


def _pick_on_worker_thread(ui, qapp, image, initial=None, msg="Pick the point."):
    outcome = {}

    def target():
        try:
            outcome["poi"] = select_poi_ui(
                parent_ui=ui, image=image, msg=msg, validate=True, initial_poi=initial
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
        raise AssertionError("the POI prompt never appeared")
    return thread, outcome


def _finish(thread, qapp, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "the asker never returned"


def test_the_click_answers_with_the_marker_position(window, ui, qapp):
    image = _fib_image(ui)
    initial = Point(x=1e-6, y=2e-6)
    thread, outcome = _pick_on_worker_thread(ui, qapp, image, initial=initial)

    # Mid-wait: the marker is on the FIB canvas and the waiting state is on.
    controller = window.view_controller
    assert controller.overlay_points(BeamType.ION, "poi")
    assert ui.WAITING_FOR_USER_INTERACTION is True

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    poi = outcome.get("poi")
    assert isinstance(poi, Point)
    # Un-dragged, the marker answers with the initial point (pixel-rounded).
    assert poi.x == pytest.approx(initial.x, abs=image.metadata.pixel_size.x)
    assert poi.y == pytest.approx(initial.y, abs=image.metadata.pixel_size.x)


def test_the_overlay_comes_down_with_the_answer(window, ui, qapp):
    image = _fib_image(ui)
    thread, _ = _pick_on_worker_thread(ui, qapp, image)

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert not window.view_controller.overlay_points(BeamType.ION, "poi")
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_the_second_handshake_is_gone(ui, qapp):
    # The old flow set WAITING_FOR_UI_UPDATE for its embedded clear-and-compute
    # wait; both run inside the click's slot and the flag is gone.
    thread, _ = _pick_on_worker_thread(ui, qapp, _fib_image(ui))

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert not hasattr(ui, "WAITING_FOR_UI_UPDATE")


def test_no_image_or_no_validation_answers_none(ui):
    assert select_poi_ui(parent_ui=ui, image=None) is None
    assert select_poi_ui(parent_ui=None, image=_fib_image(ui)) is None
    assert select_poi_ui(parent_ui=ui, image=_fib_image(ui), validate=False) is None


def _pending_prompt_on_worker(ui, qapp):
    """Read the serialized prompt from a worker thread, as the server does.

    The live-answer peek marshals to the GUI thread, so a read from the test
    (GUI) thread would deadlock — same shape as the HTTP answer tests.
    """
    from fibsem.applications.autolamella.server import AgentContext

    ctx = AgentContext(ui)
    outcome = {}

    def target():
        outcome["payload"] = ctx.pending_prompt()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "payload" not in outcome and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert "payload" in outcome, "pending_prompt never returned"
    return outcome["payload"]


def test_the_prompt_mirrors_the_live_marker_not_the_frozen_request(window, ui, qapp):
    image = _fib_image(ui)
    initial = Point(x=1e-6, y=2e-6)
    thread, _ = _pick_on_worker_thread(ui, qapp, image, initial=initial)

    pending = _pending_prompt_on_worker(ui, qapp)["pending"]
    assert pending["type"] == "PickPOI"
    px = image.metadata.pixel_size.x
    assert pending["current"]["x"] == pytest.approx(initial.x, abs=px)
    assert pending["current"]["y"] == pytest.approx(initial.y, abs=px)

    # The operator drags the marker: the next read shows where it IS.
    from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

    controller = window.view_controller
    moved_row = image.data.shape[0] / 4
    moved_col = image.data.shape[1] / 4
    controller.set_overlay(
        BeamType.ION, PointsSpec(id="poi", points=[(moved_col, moved_row)])
    )
    moved = _pending_prompt_on_worker(ui, qapp)["pending"]["current"]
    assert moved != pending["current"]
    # And the peek disturbed nothing: the marker is still up, the asker waits.
    assert controller.overlay_points(BeamType.ION, "poi")

    ui.pushButton_yes.click()
    _finish(thread, qapp)


def _answer_with_value_on_worker(ui, qapp, response, nonce, value):
    """Call AgentContext.answer_prompt from a worker thread, as the server does."""
    from fibsem.applications.autolamella.server import AgentContext

    ctx = AgentContext(ui)
    outcome = {}

    def target():
        outcome["result"] = ctx.answer_prompt(response, nonce, value=value)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "result" not in outcome and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert "result" in outcome, "answer_prompt never returned"
    return outcome["result"]


def test_a_value_answer_moves_the_marker_and_answers_with_it(window, ui, qapp):
    image = _fib_image(ui)
    thread, outcome = _pick_on_worker_thread(
        ui, qapp, image, initial=Point(x=1e-6, y=2e-6)
    )
    _, nonce = ui.ui_responder.pending_question_and_nonce()

    proposed = {"x": 5e-6, "y": -3e-6}
    result = _answer_with_value_on_worker(ui, qapp, True, nonce, proposed)
    _finish(thread, qapp)

    assert result["applied"] is True
    assert result["adjusted"] is True
    poi = outcome.get("poi")
    assert isinstance(poi, Point)
    # The marker landed where proposed, and the answer read it back
    # (pixel-rounded through the overlay).
    px = image.metadata.pixel_size.x
    assert poi.x == pytest.approx(proposed["x"], abs=px)
    assert poi.y == pytest.approx(proposed["y"], abs=px)


def test_an_out_of_bounds_point_is_refused_and_the_marker_stands(window, ui, qapp):
    image = _fib_image(ui)
    initial = Point(x=1e-6, y=2e-6)
    thread, _ = _pick_on_worker_thread(ui, qapp, image, initial=initial)
    _, nonce = ui.ui_responder.pending_question_and_nonce()
    px = image.metadata.pixel_size.x
    half_width = image.data.shape[1] / 2 * px

    result = _answer_with_value_on_worker(
        ui, qapp, True, nonce, {"x": 2 * half_width, "y": 0.0}
    )

    assert result["applied"] is False
    assert "invalid_value" in result
    # Nothing was clicked and the marker did not move.
    assert ui.ui_responder.pending_question_and_nonce()[1] == nonce
    marker = window.view_controller.overlay_points(BeamType.ION, "poi")
    assert marker, "the marker should still be up"

    ui.pushButton_yes.click()
    _finish(thread, qapp)


def test_a_stop_interrupts_a_pick_nobody_confirms(ui, qapp):
    thread, outcome = _pick_on_worker_thread(ui, qapp, _fib_image(ui))

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)
