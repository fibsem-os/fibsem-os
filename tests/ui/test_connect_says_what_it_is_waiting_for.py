"""Connecting says which step it is on while it waits (FIB-1153).

Connecting runs on the GUI thread on purpose: the window takes no input while it
does, so nothing can be started against a half-made session. But a ThermoFisher
column turning on at connect takes a while, and with nothing repainted the
application looked dead. Each step is now put on screen as it starts -- and painted,
with user input held back so that still nothing can be pressed.

`setup_session` is replaced only to look at the window mid-connect; what it returns
is a real Demo session.
"""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEventLoop  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ui import notification_service  # noqa: E402
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget  # noqa: E402
from fibsem.ui.widgets.connection_dialog import ConnectionDialog  # noqa: E402

real_setup_session = utils.setup_session


class Midway:
    """A `setup_session` that reports a step and notes what the window shows then."""

    def __init__(self, look, fail: str = ""):
        self.look = look
        self.seen = None
        self.fail = fail

    def __call__(self, *args, progress=None, **kwargs):
        progress("Turning the beams on…")
        self.seen = self.look()
        if self.fail:
            raise RuntimeError(self.fail)
        return real_setup_session(manufacturer="Demo", setup_logging=False)


@pytest.fixture
def paints(monkeypatch):
    """Each repaint's flags. A repaint that let user input through would let a
    second connect, or a close, start mid-connect."""
    flags = []
    original = QApplication.processEvents

    def spy(*args):
        flags.append(args[0] if args else None)
        return original(*args)

    monkeypatch.setattr(QApplication, "processEvents", spy)
    return flags


@pytest.fixture
def tab(qapp, monkeypatch):
    monkeypatch.setattr(notification_service, "show_toast", lambda *a, **k: None)
    widget = FibsemSystemSetupWidget()
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/a/site.yaml")
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.deleteLater()


def _card(tab):
    return (
        tab._label_status_title.text(),
        tab._label_status_subtitle.text(),
        tab.pushButton_connect_to_microscope.isEnabled(),
    )


def test_the_tab_shows_each_step_while_it_connects(tab, monkeypatch, paints):
    midway = Midway(lambda: _card(tab))
    monkeypatch.setattr(utils, "setup_session", midway)

    tab.connect_to_microscope()

    assert midway.seen == ("Connecting", "Turning the beams on…", False)
    assert tab.microscope is not None
    assert tab._label_status_title.text() == "Microscope Connected"
    assert paints and all(f == QEventLoop.ExcludeUserInputEvents for f in paints)


def test_a_failed_connect_gives_the_button_back(tab, monkeypatch, paints):
    monkeypatch.setattr(
        utils, "setup_session", Midway(lambda: None, fail="no route to 192.168.0.1")
    )

    tab.connect_to_microscope()

    assert tab.microscope is None
    assert tab._label_status_title.text() == "Connection Failed"
    assert tab.pushButton_connect_to_microscope.isEnabled()
    assert tab.pushButton_connect_to_microscope.text() == "Connect to Microscope"


@pytest.fixture
def dialog(qapp):
    d = ConnectionDialog()
    d.show()
    yield d
    if d.microscope is not None:
        d.microscope.disconnect()
    d.deleteLater()


def test_the_dialog_shows_each_step_while_it_connects(dialog, monkeypatch, paints):
    midway = Midway(
        lambda: (dialog.message_label.text(), dialog.connect_button.isEnabled())
    )
    monkeypatch.setattr(utils, "setup_session", midway)

    dialog.connect_to_microscope()

    assert midway.seen == ("Turning the beams on…", False)
    assert dialog.result() == QDialog.Accepted
    assert paints and all(f == QEventLoop.ExcludeUserInputEvents for f in paints)


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
