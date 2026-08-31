"""The detection question over the Responder seam: ConfirmDetection.

The old flow straddled threads three ways: ``ask_user(det=...)`` parked the
workflow on the polled flag, the workflow thread then called
``det_widget._get_detected_features()`` across the seam, and finally signalled
back (``detection_confirmed_signal``) so the GUI could save. Now the question's
answer IS the (possibly corrected) feature set: the click reads and saves on the
GUI thread that owns the widget, and the future carries the result back.

The real ``FibsemEmbeddedDetectionWidget`` is unimportable without the ``ml``
extra (``segmentation_models_pytorch``), which no test environment installs —
CI's ui job included. The stand-in below is a real ``QWidget`` on the real
window's tab bar exposing exactly the three methods the seam drives, so tab
fronting, thread affinity and the click path are all the production ones.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import (
    ConfirmDetection,
    ask,
)
from fibsem.detection.detection import DetectedFeatures, LamellaCentre

PROMPT = "Confirm Feature Detection. Press Continue to proceed."


def _make_detection() -> DetectedFeatures:
    return DetectedFeatures(
        features=[LamellaCentre()],
        image=np.zeros((8, 8), dtype=np.uint8),
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
    )


class _RecordingDetWidget(QWidget):
    """The three methods QtResponder drives, on a real widget in the tab bar."""

    def __init__(self):
        super().__init__()
        self.det = None
        self.confirmed = 0
        self.threads = []

    def set_detected_features(self, det):
        self.det = det

    def _get_detected_features(self):
        self.threads.append(threading.current_thread().name)
        return self.det

    def confirm_button_clicked(self):
        self.confirmed += 1


@pytest.fixture
def ui(qapp):
    """A real AutoLamellaUI, connected (Demo), with a stand-in detection tab."""
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    det_widget = _RecordingDetWidget()
    widget.det_widget = det_widget
    det_idx = widget.tabWidget.addTab(det_widget, "Detection")
    widget.tabWidget.setTabVisible(det_idx, False)
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _ask_on_worker_thread(ui, qapp, detection):
    """Start a ConfirmDetection ask on a worker thread; spin until the prompt is up."""
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(
                ui.ui_responder,
                ConfirmDetection(detection=detection),
                abort=ui._workflow_stop_event.is_set,
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        qapp.processEvents()
        if ui.label_instructions.text() == PROMPT and ui.pushButton_yes.isEnabled():
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the detection prompt never appeared")
    return thread, outcome


def _finish(thread, qapp, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "the asker never returned"


def test_the_click_answers_with_the_widgets_feature_set(ui, qapp):
    sent = _make_detection()
    thread, outcome = _ask_on_worker_thread(ui, qapp, sent)

    # Mid-wait: the features are in the widget, its tab is fronted, and the
    # waiting display state is on for the attention button and border.
    assert ui.det_widget.det is sent
    assert ui.tabWidget.currentWidget() is ui.det_widget
    assert ui.WAITING_FOR_USER_INTERACTION is True

    # The supervisor corrects the detection; the answer must be what the widget
    # holds at click time, not what the workflow sent.
    corrected = _make_detection()
    ui.det_widget.det = corrected

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert outcome.get("answer") is corrected
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_the_read_and_the_save_run_on_the_gui_thread(ui, qapp):
    # Both used to straddle threads: the workflow thread read the features
    # across the seam, then signalled back for the save. Now the click does both
    # on the thread that owns the widget, before the waiter wakes.
    thread, _ = _ask_on_worker_thread(ui, qapp, _make_detection())

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert ui.det_widget.threads == ["MainThread"]
    assert ui.det_widget.confirmed == 1


def test_a_missing_detection_widget_fails_the_asker_not_the_process(ui, qapp):
    # The model already ran to produce the request, so no widget is a defect —
    # and it surfaces on the workflow thread, not as an abort out of a slot.
    ui.det_widget = None
    outcome = {}

    def target():
        try:
            ask(ui.ui_responder, ConfirmDetection(detection=_make_detection()))
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), RuntimeError)
    assert "detection widget" in str(outcome["error"])


def test_a_failing_read_back_reraises_on_the_workflow_thread(ui, qapp):
    # Computing the answer happens in the click's slot; under the old mechanism
    # an exception there was a process abort (FIB-329). Now it is this
    # question's failure, delivered to the thread that asked.
    thread, outcome = _ask_on_worker_thread(ui, qapp, _make_detection())

    def broken():
        raise RuntimeError("features fell over")

    ui.det_widget._get_detected_features = broken
    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), RuntimeError)
    assert "features fell over" in str(outcome["error"])
    # The prompt still came down: the waiter wakes to a consistent UI either way.
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_a_stop_interrupts_a_detection_nobody_confirms(ui, qapp):
    thread, outcome = _ask_on_worker_thread(ui, qapp, _make_detection())

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)


def test_update_detection_ui_returns_the_confirmed_features(ui, qapp, monkeypatch):
    # The workflow entry point end to end: producer monkeypatched (it needs a
    # model checkpoint), validation through the real seam, and the return value
    # is the widget's set — no _get_detected_features read-back at the call site.
    from fibsem.applications.autolamella.workflows import ui as workflow_ui

    produced = _make_detection()
    monkeypatch.setattr(
        workflow_ui.detection,
        "take_image_and_detect_features",
        lambda **kwargs: produced,
    )

    outcome = {}

    def target():
        try:
            outcome["answer"] = workflow_ui.update_detection_ui(
                microscope=ui.microscope,
                image_settings=None,
                checkpoint="unused",
                features=[LamellaCentre()],
                parent_ui=ui,
                validate=True,
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        qapp.processEvents()
        if ui.label_instructions.text() == PROMPT:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the detection prompt never appeared")

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert outcome["answer"] is produced
    assert ui.det_widget.confirmed == 1
