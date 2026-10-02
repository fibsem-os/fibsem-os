"""Connecting runs off the GUI thread, so the window stays live while it does (FIB-1153).

Turning a ThermoFisher column on at connect takes a while, and run on the GUI thread
nothing repainted until it was done -- the application looked dead. The connect now
runs on a worker; these hold the GUI to that: the call returns while the instrument
is still connecting, the window says which step it is on, and the session arrives
afterwards.

`setup_session` is replaced only to make it wait at a gate the test opens, so
"still connecting" is a state the test can look at. What it returns is a real Demo
session.
"""

import threading

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QCoreApplication  # noqa: E402
from PyQt5.QtWidgets import QDialog  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ui import notification_service  # noqa: E402
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget  # noqa: E402
from fibsem.ui.widgets.connection_dialog import ConnectionDialog  # noqa: E402

real_setup_session = utils.setup_session


class Gate:
    """A `setup_session` that reports a step, then waits until the test opens it."""

    def __init__(self, fail: str = ""):
        self.open = threading.Event()
        self.waiting = threading.Event()
        self.calls = 0
        self.fail = fail

    def __call__(self, *args, progress=None, **kwargs):
        self.calls += 1
        if progress is not None:
            progress("Turning the beams on…")
        self.waiting.set()
        assert self.open.wait(10), "the test never opened the gate"
        if self.fail:
            raise RuntimeError(self.fail)
        return real_setup_session(manufacturer="Demo", setup_logging=False)


def _settle(gate: Gate) -> None:
    """Let the worker reach the gate and its progress reach the GUI."""
    assert gate.waiting.wait(10)
    for _ in range(3):
        QCoreApplication.processEvents()


@pytest.fixture
def toasts(monkeypatch):
    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, level="info", *a, **k: shown.append((level, message)),
    )
    return shown


@pytest.fixture
def tab(qapp, monkeypatch, toasts):
    widget = FibsemSystemSetupWidget()
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/a/site.yaml")
    yield widget
    widget.wait_for_connection()
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.deleteLater()


def test_the_tab_returns_while_the_instrument_is_still_connecting(tab, monkeypatch):
    gate = Gate()
    monkeypatch.setattr(utils, "setup_session", gate)

    assert tab.connect_to_microscope() is True  # returned with the gate shut
    _settle(gate)

    assert tab.is_connecting()
    assert tab.microscope is None
    assert tab._label_status_title.text() == "Connecting"
    assert tab._label_status_subtitle.text() == "Turning the beams on…"
    assert not tab.pushButton_connect_to_microscope.isEnabled()
    assert not tab.comboBox_configuration.isEnabled()

    finished = []
    tab.connection_attempt_finished.connect(finished.append)
    gate.open.set()
    tab.wait_for_connection()

    assert not tab.is_connecting()
    assert tab.microscope is not None
    assert finished == [True]
    assert tab._label_status_title.text() == "Microscope Connected"


def test_a_second_click_while_connecting_starts_nothing(tab, monkeypatch):
    gate = Gate()
    monkeypatch.setattr(utils, "setup_session", gate)
    tab.connect_to_microscope()
    _settle(gate)

    assert tab.connect_to_microscope() is False

    gate.open.set()
    tab.wait_for_connection()
    assert gate.calls == 1


def test_a_failure_off_the_thread_is_reported_on_the_tab(tab, monkeypatch, toasts):
    gate = Gate(fail="no route to 192.168.0.1")
    monkeypatch.setattr(utils, "setup_session", gate)
    finished = []
    tab.connection_attempt_finished.connect(finished.append)

    tab.connect_to_microscope()
    _settle(gate)
    gate.open.set()
    tab.wait_for_connection()

    assert tab.microscope is None
    assert finished == [False]
    assert tab._label_status_title.text() == "Connection Failed"
    assert ("error", "Could not connect: no route to 192.168.0.1") in toasts
    assert tab.pushButton_connect_to_microscope.isEnabled()


@pytest.fixture
def dialog(qapp):
    d = ConnectionDialog()
    d.show()
    yield d
    d.wait_for_connection()
    if d.microscope is not None:
        d.microscope.disconnect()
    d.deleteLater()


def test_the_dialog_says_which_step_and_cannot_be_closed_mid_connect(
    dialog, monkeypatch
):
    gate = Gate()
    monkeypatch.setattr(utils, "setup_session", gate)

    dialog.connect_to_microscope()
    _settle(gate)

    assert dialog.is_connecting()
    assert dialog.message_label.text() == "Turning the beams on…"
    assert not dialog.connect_button.isEnabled()
    dialog.reject()  # Escape, the close box
    assert dialog.isVisible()

    gate.open.set()
    dialog.wait_for_connection()

    assert dialog.result() == QDialog.Accepted
    assert dialog.microscope is not None


def test_the_steps_are_reported_in_order():
    steps = []
    microscope, _ = utils.setup_session(
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

    microscope, _ = utils.setup_session(
        manufacturer="Demo", setup_logging=False, progress=broken
    )
    microscope.disconnect()


def test_a_connector_runs_one_attempt_at_a_time(qapp, monkeypatch):
    """Whatever calls it -- the tab, the dialog, a script."""
    from fibsem.ui.connecting import SessionConnector

    gate = Gate()
    monkeypatch.setattr(utils, "setup_session", gate)
    connector = SessionConnector()
    sessions = []
    connector.connected.connect(
        lambda microscope, settings: sessions.append(microscope)
    )

    connector.start("/a/site.yaml")
    _settle(gate)
    connector.start("/a/site.yaml")
    gate.open.set()
    connector.wait(10)

    assert gate.calls == 1
    assert len(sessions) == 1
    sessions[0].disconnect()
