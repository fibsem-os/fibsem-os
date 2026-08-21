"""The connection tab, and what it does when connecting does not work.

Failing to connect is an ordinary outcome on this tab -- the vendor API may not be
installed, the instrument may be off, the address may belong to another bay. What it
must never be is fatal: this runs as a Qt slot, and PyQt5 turns an unhandled exception
in a slot into ``qFatal``, which aborts the whole application (FIB-329).
"""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem import utils
from fibsem.ui import notification_service
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

# The real message a ThermoFisher backend raises with no AutoScript on the machine.
# Verbatim from microscope.py, because the toast shows it and a user reading "not
# installed" is being told what to do next.
AUTOSCRIPT_MISSING = (
    "Autoscript (ThermoFisher) not installed. "
    "Please see the user guide for installation instructions."
)


@pytest.fixture
def widget(qapp):
    w = FibsemSystemSetupWidget()
    yield w
    w.deleteLater()


@pytest.fixture
def demo_microscope():
    """A real Demo session, so the success path is exercised against the object the
    tab actually goes on to use."""
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def toasts(monkeypatch):
    """Collect the toasts rather than showing them."""
    captured = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, level="info", *a, **k: captured.append((level, message)),
    )
    return captured


def _fail_to_connect(monkeypatch, error: Exception) -> None:
    """Make the connection attempt raise, the way a real backend would.

    Patched rather than driven by pointing at a real ThermoFisher configuration: that
    would pass or fail depending on whether the developer's machine has AutoScript,
    which is the environment dependence FIB-779 is about. The failure being simulated
    is precisely `setup_session` raising, so patching it is the whole of the fixture.
    """

    def raise_it(*args, **kwargs):
        raise error

    monkeypatch.setattr(utils, "setup_session", raise_it)


def test_a_failed_connection_does_not_escape_the_slot(widget, monkeypatch, toasts):
    """The crash this file exists for: an exception here aborts the process."""
    _fail_to_connect(monkeypatch, Exception(AUTOSCRIPT_MISSING))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()  # must not raise

    assert widget.microscope is None
    assert widget.settings is None


def test_a_failed_connection_says_why(widget, monkeypatch, toasts):
    """Silence would leave someone pressing Connect again with nothing to act on."""
    _fail_to_connect(monkeypatch, Exception(AUTOSCRIPT_MISSING))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()

    errors = [message for level, message in toasts if level == "error"]
    assert errors, "a failed connection reported nothing"
    assert "not installed" in errors[-1]


def test_a_failed_connection_leaves_the_tab_usable(widget, monkeypatch, toasts):
    """``update_ui`` still has to run, or the tab keeps claiming it is connecting."""
    _fail_to_connect(monkeypatch, Exception(AUTOSCRIPT_MISSING))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()

    # Offered again rather than left disabled: the address or the missing API can be
    # fixed and the button is how you retry.
    assert widget.pushButton_connect_to_microscope.isEnabled()
    assert "Connect" in widget.pushButton_connect_to_microscope.text()


def test_an_unexpected_failure_is_caught_too(widget, monkeypatch, toasts):
    """The except is broad on purpose.

    Backends raise whatever their own SDK raises. A narrower catch would leave the
    abort in place for the exception nobody predicted, which is the one that happens.
    """
    _fail_to_connect(monkeypatch, RuntimeError("the SDK returned something odd"))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()

    assert widget.microscope is None


def test_a_failed_connection_is_visible_without_toasts(widget, monkeypatch, toasts):
    """Toasts are off by default (`display.toasts_enabled`), and `show_toast`
    deliberately does not reach notification history -- so a toast alone would turn
    the crash this guards against into silence. The status card is always on screen.
    """
    _fail_to_connect(monkeypatch, Exception(AUTOSCRIPT_MISSING))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()

    # "Not connected" and "tried and failed" are different states, and the difference
    # is the whole of what someone needs here.
    assert widget._label_status_title.text() == "Connection Failed"
    assert "not installed" in widget._label_status_subtitle.text()


def test_a_retry_does_not_show_the_previous_reason(widget, monkeypatch, toasts):
    """Stale text beside a connection being made reads as a fresh failure."""
    _fail_to_connect(monkeypatch, Exception(AUTOSCRIPT_MISSING))
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")
    widget.connect_to_microscope()
    assert widget._label_status_title.text() == "Connection Failed"

    # A configuration that is never selected returns before connecting at all.
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: None)
    widget.connect_to_microscope()

    assert widget._last_connection_error is None


def test_a_never_attempted_connection_says_so_plainly(widget):
    """Nothing has failed yet, so the card must not imply something has."""
    assert widget._label_status_title.text() == "Not Connected"
    assert widget._label_status_subtitle.text() == "No microscope connected"


def test_a_failed_disconnect_does_not_escape_the_slot(widget, toasts):
    """Disconnecting can fail for ordinary reasons -- the instrument went away, the
    client is already dead -- and none are worth losing the application over."""

    class WillNotClose:
        def disconnect(self):
            raise RuntimeError("the client is already gone")

    widget.microscope = WillNotClose()

    widget.connect_to_microscope()  # must not raise

    # One outcome, not two: the tab does drop to disconnected, so the message says
    # what happened rather than announcing a failure beside a disconnected tab.
    errors = [message for level, message in toasts if level == "error"]
    assert any("did not close cleanly" in message for message in errors)


def test_a_failed_disconnect_still_lets_go_of_the_microscope(widget, toasts):
    """Holding a client that would not close leaves the tab offering to disconnect
    something it can no longer reach, with no way back to a working connection."""

    class WillNotClose:
        def disconnect(self):
            raise RuntimeError("the client is already gone")

    widget.microscope = WillNotClose()

    widget.connect_to_microscope()

    assert widget.microscope is None
    assert widget.settings is None


def test_a_successful_disconnect_lets_go_too(widget, toasts, demo_microscope):
    """The ordinary path, against a real session rather than a stand-in."""
    widget.microscope = demo_microscope

    widget.connect_to_microscope()

    assert widget.microscope is None
    assert not [message for level, message in toasts if level == "error"]


def test_a_successful_connection_still_reports_and_holds_the_microscope(
    widget, monkeypatch, toasts, demo_microscope
):
    """The else branch. A real Demo session, so the success path is exercised against
    the object the tab actually goes on to use rather than a stand-in."""
    monkeypatch.setattr(
        utils, "setup_session", lambda *a, **k: (demo_microscope, object())
    )
    monkeypatch.setattr(widget, "load_configuration", lambda *a, **k: "/some/path.yaml")

    widget.connect_to_microscope()

    assert widget.microscope is demo_microscope
    infos = [message for level, message in toasts if level == "info"]
    assert any("Connected to microscope at" in message for message in infos)
