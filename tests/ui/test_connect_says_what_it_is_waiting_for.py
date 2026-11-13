"""Connecting runs on a worker thread, says which step it is on, and can be
cancelled (FIB-1153).

A ThermoFisher column turning on at connect takes a while. Connecting used to run
on the GUI thread, so the whole application stopped answering until it was done.
Now the window keeps painting, the connect controls are held while it runs, each
step is shown as it starts, and Cancel stops it at the next step.

`setup_session` is replaced only to hold the connect mid-way; what it returns is a
real Demo session.
"""

import threading
import time

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QApplication, QDialog  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ui import notification_service  # noqa: E402
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget  # noqa: E402
from fibsem.ui.widgets.connection_dialog import ConnectionDialog  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402
from tests.fixtures.connecting import wait_connected  # noqa: E402

real_setup_session = utils.setup_session


class Midway:
    """A `setup_session` that reports a step, then waits to be let go."""

    def __init__(self, fail: str = ""):
        self.fail = fail
        self.reported = threading.Event()
        self.go = threading.Event()

    def __call__(self, *args, progress=None, cancel=None, **kwargs):
        progress("Turning the beams on…")
        self.reported.set()
        assert self.go.wait(10)
        if self.fail:
            raise RuntimeError(self.fail)
        return real_setup_session(
            manufacturer="Demo", setup_logging=False, cancel=cancel
        )


def _until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QApplication.processEvents()
        time.sleep(0.002)


@pytest.fixture
def toasts(monkeypatch):
    said = []
    monkeypatch.setattr(
        notification_service, "show_toast", lambda msg, *a, **k: said.append(msg)
    )
    return said


@pytest.fixture
def tab(qapp, toasts, monkeypatch):
    widget = FibsemSystemSetupWidget()
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/a/site.yaml")
    yield widget
    widget.session.disconnect()
    widget.deleteLater()


def test_the_tab_shows_each_step_while_it_connects(tab, monkeypatch):
    midway = Midway()
    monkeypatch.setattr(utils, "setup_session", midway)

    tab.connect_to_microscope()
    assert midway.reported.wait(10)
    _until(lambda: tab._label_status_subtitle.text() == "Turning the beams on…")

    # The GUI thread is free, and the connect controls are held.
    assert tab.is_connecting
    assert tab._label_status_title.text() == "Connecting"
    assert not tab.pushButton_connect_to_microscope.isEnabled()
    assert not tab.comboBox_configuration.isEnabled()
    assert tab._button_cancel_connect.isVisibleTo(tab)

    midway.go.set()
    wait_connected(tab)
    assert tab.microscope is not None
    assert tab._label_status_title.text() == "Microscope Connected"
    assert not tab._button_cancel_connect.isVisibleTo(tab)


def test_a_failed_connect_gives_the_button_back(tab, monkeypatch):
    midway = Midway(fail="no route to 192.168.0.1")
    midway.go.set()
    monkeypatch.setattr(utils, "setup_session", midway)

    tab.connect_to_microscope()
    wait_connected(tab)

    assert tab.microscope is None
    assert tab._label_status_title.text() == "Connection Failed"
    assert "no route" in tab._label_status_subtitle.text()
    assert tab.pushButton_connect_to_microscope.isEnabled()
    assert tab.pushButton_connect_to_microscope.text() == "Connect to Microscope"


def test_cancel_on_the_tab_lets_the_microscope_go(tab, monkeypatch, toasts):
    midway = Midway()
    monkeypatch.setattr(utils, "setup_session", midway)

    tab.connect_to_microscope()
    assert midway.reported.wait(10)
    tab._button_cancel_connect.click()
    assert tab._label_status_title.text() == "Cancelling"
    midway.go.set()
    wait_connected(tab)

    assert tab.microscope is None
    assert tab.session.instrument is None
    assert "Connecting was cancelled." in toasts
    # Cancelled is not failed.
    assert tab._label_status_title.text() == "Not Connected"
    assert tab.pushButton_connect_to_microscope.isEnabled()


@pytest.fixture
def dialog(qapp):
    d = ConnectionDialog()
    d.show()
    yield d
    d.session.disconnect()
    d.deleteLater()


def test_the_dialog_shows_each_step_while_it_connects(dialog, monkeypatch):
    midway = Midway()
    monkeypatch.setattr(utils, "setup_session", midway)

    dialog.connect_to_microscope()
    assert midway.reported.wait(10)
    _until(lambda: dialog.message_label.text() == "Turning the beams on…")

    assert not dialog.connect_button.isEnabled()
    assert not dialog.configuration_combo.isEnabled()
    assert dialog.cancel_button.isVisible()
    assert not dialog.offline_button.isVisible()

    midway.go.set()
    wait_connected(dialog)
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_escape_mid_connect_cancels_and_keeps_the_dialog(dialog, monkeypatch):
    """Closing would leave a connect running that nobody is watching."""
    midway = Midway()
    monkeypatch.setattr(utils, "setup_session", midway)

    dialog.connect_to_microscope()
    assert midway.reported.wait(10)
    dialog.reject()
    midway.go.set()
    wait_connected(dialog)

    assert dialog.isVisible()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.session.instrument is None
    assert dialog.message_label.text() == "Connecting was cancelled."
    assert dialog.connect_button.isEnabled()
    assert dialog.offline_button.isVisible()


def test_the_steps_are_reported_in_order():
    steps = []
    microscope, _ = real_setup_session(
        manufacturer="Demo",
        setup_logging=False,
        beams_on=True,
        apply_defaults=True,
        progress=steps.append,
    )
    try:
        assert steps == [
            f"Connecting to {microscope.system.info.ip_address}…",
            "Turning the beams on…",
            "Applying the defaults…",
        ]
    finally:
        microscope.disconnect()


def test_a_progress_callback_that_fails_does_not_fail_the_connection():
    def broken(step):
        raise RuntimeError("the window went away")

    microscope, _ = real_setup_session(
        manufacturer="Demo", setup_logging=False, progress=broken
    )
    microscope.disconnect()


def test_a_cancelled_connect_stops_before_the_next_step():
    cancel = threading.Event()
    steps = []

    def progress(step):
        steps.append(step)
        cancel.set()  # pressed while the vendor connect runs

    with pytest.raises(utils.ConnectCancelled):
        real_setup_session(
            manufacturer="Demo",
            setup_logging=False,
            beams_on=True,
            progress=progress,
            cancel=cancel,
        )
    assert "Turning the beams on…" not in steps


def test_a_cancelled_session_connect_holds_nothing(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    session = AppSession()
    with pytest.raises(utils.ConnectCancelled):
        session.connect(None, cancel=cancel, manufacturer="Demo")
    assert session.instrument is None
    assert session.connecting is None
