"""The attention button is the way back to a question you navigated away from.

A question asked on a tab of its own -- a detection, a mill -- disappears from
view the moment the operator looks at something else. The button says
"Attention Required" and the run is stopped on that question, so pressing it
has to land on the question, not merely on the Microscope tab.

Before this, ``_on_user_attention_clicked`` sent everything that was not a
review hold to tab index 0, and the Detection tab is not index 0 -- it is a
tab of ``AutoLamellaUI``'s own tab bar, hidden until it is asked for. So the
operator was returned to a tab with no prompt on it and no way to find one.

The real ``FibsemEmbeddedDetectionWidget`` needs the ``ml`` extra, which no
test environment installs, so the stand-in below is a real ``QWidget`` on the
real tab bar exposing the methods the seam drives -- the same approach as
``test_confirm_detection``.
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


class _DetWidget(QWidget):
    """The three methods QtResponder drives, on a real widget in the tab bar."""

    def __init__(self):
        super().__init__()
        self.det = None

    def set_detected_features(self, det):
        self.det = det

    def _get_detected_features(self):
        return self.det

    def confirm_button_clicked(self):
        pass


@pytest.fixture
def ui(qapp):
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    det_widget = _DetWidget()
    widget.det_widget = det_widget
    index = widget.tabWidget.addTab(det_widget, "Detection")
    widget.tabWidget.setTabVisible(index, False)
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _detection() -> DetectedFeatures:
    return DetectedFeatures(
        features=[LamellaCentre()],
        image=np.zeros((8, 8), dtype=np.uint8),
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
    )


def _ask_on_worker_thread(ui, qapp):
    """Raise a real ConfirmDetection from a worker thread and wait for the
    prompt, as the workflow does."""
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(
                ui.ui_responder,
                ConfirmDetection(detection=_detection()),
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
            return thread, outcome
        time.sleep(0.01)
    raise AssertionError("the detection prompt never appeared")


def _answer(ui, thread, qapp):
    ui.pushButton_yes.click()
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)


def test_the_responder_says_which_tab_its_question_is_on(ui, qapp):
    thread, _ = _ask_on_worker_thread(ui, qapp)

    assert ui.ui_responder.question_host() is ui.det_widget

    _answer(ui, thread, qapp)


def test_there_is_no_host_when_nothing_is_being_asked(ui, qapp):
    assert ui.ui_responder.question_host() is None


def test_the_host_is_forgotten_once_the_question_is_answered(ui, qapp):
    """Otherwise the button would keep returning to a tab whose prompt is
    down, long after the run moved on."""
    thread, _ = _ask_on_worker_thread(ui, qapp)
    _answer(ui, thread, qapp)

    assert ui.ui_responder.question_host() is None


def test_navigating_away_and_coming_back_lands_on_the_question(ui, qapp):
    """The whole point: the question's tab is fronted when it is asked, the
    operator leaves, and ``front_question`` brings it back."""
    thread, _ = _ask_on_worker_thread(ui, qapp)
    assert ui.tabWidget.currentWidget() is ui.det_widget, "asked on its own tab"

    ui.tabWidget.setCurrentIndex(0)
    qapp.processEvents()
    assert ui.tabWidget.currentWidget() is not ui.det_widget, "looked elsewhere"

    ui.front_question()

    assert ui.tabWidget.currentWidget() is ui.det_widget
    _answer(ui, thread, qapp)


def test_coming_back_un_hides_a_tab_that_hides_itself(ui, qapp):
    """Detection's tab is hidden until it is asked for, and fronting a hidden
    tab is a silent no-op -- so the way back has to un-hide it first."""
    thread, _ = _ask_on_worker_thread(ui, qapp)
    index = ui.tabWidget.indexOf(ui.det_widget)
    ui.tabWidget.setCurrentIndex(0)
    ui.tabWidget.setTabVisible(index, False)
    qapp.processEvents()

    ui.front_question()

    assert ui.tabWidget.isTabVisible(index)
    assert ui.tabWidget.currentWidget() is ui.det_widget
    _answer(ui, thread, qapp)


def test_it_does_nothing_when_no_question_is_up(ui, qapp):
    """A hold with no question of its own -- an agent working, a prompt on the
    shared bar -- leaves the tab where the caller put it."""
    ui.tabWidget.setCurrentIndex(0)
    before = ui.tabWidget.currentWidget()

    ui.front_question()

    assert ui.tabWidget.currentWidget() is before


# ---------------------------------------------------------------------------
# The window's own tabs
# ---------------------------------------------------------------------------


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def test_a_question_on_one_of_the_windows_own_tabs_is_fronted_there(main_ui, qapp):
    """The destination comes from the question, not from the hold's kind.

    ``HoldKind.question`` covers a prompt answered on the Microscope tab and
    one answered on a tab of its own alike, and an in-run review answered in
    the Review tab (FIB-1025) will be a question hold too -- so the router has
    to take the answer from the pending request. Stood up here with the Review
    tab standing in for that future host.
    """
    review_tab = main_ui.review_tab
    main_ui.autolamella_ui.ui_responder.question_host = lambda: review_tab
    main_ui.tab_widget.setCurrentIndex(0)

    main_ui._on_user_attention_clicked()

    assert main_ui.tab_widget.currentWidget() is review_tab


def test_a_question_elsewhere_still_lands_on_the_microscope_tab(main_ui, qapp):
    """A yes/no prompt on the shared bar has no tab of its own and must not be
    dragged into whatever the last host was."""
    main_ui.autolamella_ui.ui_responder.question_host = lambda: None
    main_ui.tab_widget.setCurrentWidget(main_ui.review_tab)

    main_ui._on_user_attention_clicked()

    assert main_ui.tab_widget.currentIndex() == 0
