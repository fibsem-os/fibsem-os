"""Connecting from a dialog, and what happens when it does not work.

The Connection tab was the only way to connect and the only place that action
existed — no menu offered it — which is what made the tab impossible to remove
(FIB-775). This dialog is the other door, and the states that matter are the ones
the tab handled badly: choosing without knowing what a configuration points at,
and a failure that leaves the application running but useless.

The Demo microscope is real here. Only the failure is simulated, because there is
no way to ask a working connection to fail.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QDialog  # noqa: E402

from fibsem import config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.ui.widgets import connection_dialog  # noqa: E402
from fibsem.ui.widgets.connection_dialog import ConnectionDialog  # noqa: E402


@pytest.fixture
def dialog(qapp):
    d = ConnectionDialog()
    # Shown, not exec_'d: a child of a hidden window reports `isVisible() == False`
    # whatever its own flag says, so "the message is shown" would pass without the
    # code doing anything. exec_ would block the test on the event loop.
    d.show()
    yield d
    if d.microscope is not None:
        d.microscope.disconnect()
    d.deleteLater()


def test_it_offers_the_configurations_and_defaults_to_the_default(dialog):
    listed = [
        dialog.configuration_combo.itemText(i)
        for i in range(dialog.configuration_combo.count())
    ]

    assert listed == list(cfg.USER_CONFIGURATIONS)
    assert dialog.configuration_combo.currentText() == cfg.DEFAULT_CONFIGURATION_NAME


def test_it_says_what_the_configuration_points_at_before_connecting(dialog):
    """Read off the file, not the instrument — nothing is connected yet.

    The tab offered a name and a Connect button, so which instrument a name meant
    was something you found out by connecting to it.
    """
    settings = utils.load_microscope_configuration(dialog.configuration_path())
    info = settings.system.info

    assert dialog.microscope is None
    assert dialog.details_title.text() == "No microscope connected"
    assert info.manufacturer in dialog.details_subtitle.text()
    assert info.ip_address in dialog.details_subtitle.text()


def test_dismissing_leaves_the_session_empty(dialog):
    """Dismissable on purpose: a modal that cannot be closed makes the application
    impossible to open without an instrument, and preferences and the logs are
    reasons to do that. It says "Not now" rather than "Work offline" because there
    is no offline mode -- creating and loading an experiment are both disabled
    without a microscope."""
    dialog.offline_button.click()

    assert dialog.result() == QDialog.Rejected
    assert dialog.microscope is None
    assert dialog.settings is None


def test_connecting_holds_the_session_and_accepts(dialog):
    dialog.connect_to_microscope()

    assert dialog.result() == QDialog.Accepted
    assert dialog.microscope is not None
    assert dialog.settings is not None


def test_a_configuration_that_resolves_nowhere_is_reported(dialog):
    """Selecting a name with no registered path used to take the application down."""
    dialog.configuration_combo.addItem("a-configuration-that-was-deleted")
    dialog.configuration_combo.setCurrentText("a-configuration-that-was-deleted")

    assert dialog.configuration_path() is None
    assert not dialog.connect_button.isEnabled()

    dialog.connect_to_microscope()

    assert dialog.message_label.isVisible()
    assert "could not be found" in dialog.message_label.text()
    assert dialog.result() != QDialog.Accepted


def test_a_failure_stays_in_the_dialog(dialog, monkeypatch):
    """The reason, and a retry — not a closed dialog and a dead application."""

    def _refuse(*args, **kwargs):
        raise ConnectionError("connection timed out after 30s")

    monkeypatch.setattr(utils, "setup_session", _refuse)

    dialog.connect_to_microscope()

    assert dialog.isVisible() or dialog.result() != QDialog.Accepted
    assert dialog.microscope is None
    assert "Could not reach" in dialog.message_label.text()
    assert "timed out" in dialog.message_label.text()
    assert dialog.connect_button.text() == "Retry"
    # And usable again, rather than left disabled by the attempt.
    assert dialog.connect_button.isEnabled()
    assert dialog.configuration_combo.isEnabled()


# ── opened with a session already in progress ───────────────────────────────


def _answer_disconnect(monkeypatch, confirmed: bool) -> None:
    """Answer the confirmation. A modal cannot be clicked from a test."""
    monkeypatch.setattr(connection_dialog, "message_box_ui", lambda *a, **k: confirmed)


def _record_question(monkeypatch) -> dict:
    """Capture what the confirmation asked, and decline it."""
    asked: dict = {}

    def _decline(title, text, parent=None, **kwargs):
        asked["title"] = title
        asked["text"] = text
        return False

    monkeypatch.setattr(connection_dialog, "message_box_ui", _decline)
    return asked


@pytest.fixture
def connected_dialog(qapp):
    """The dialog as the header chip opens it: something is already connected."""
    microscope, settings = utils.setup_session(manufacturer="Demo")
    d = ConnectionDialog(microscope=microscope, settings=settings)
    d.show()
    yield d
    if d.microscope is not None:
        d.microscope.disconnect()
    d.deleteLater()


def test_it_shows_what_is_connected(connected_dialog):
    """Not the same dialog worded as though nothing had happened yet."""
    info = connected_dialog.microscope.system.info

    assert info.manufacturer in connected_dialog.details_subtitle.text()
    assert info.ip_address in connected_dialog.details_subtitle.text()
    # The panel names it; the title says what the dialog is for, once.
    assert connected_dialog.title_label.text() == "Microscope connection"
    assert info.manufacturer not in connected_dialog.title_label.text()
    assert connected_dialog.disconnect_button.isVisible()
    # Reconnect, because connecting now means dropping the session in progress.
    assert connected_dialog.connect_button.text() == "Reconnect"
    assert connected_dialog.offline_button.text() == "Close"


def test_disconnecting_ends_the_session_and_says_so(connected_dialog, monkeypatch):
    _answer_disconnect(monkeypatch, True)

    connected_dialog.disconnect_button.click()

    assert connected_dialog.microscope is None
    assert connected_dialog.settings is None
    # The part a caller acts on: a disconnect and a cancel both end with no
    # microscope, and only one of them should be applied.
    assert connected_dialog.changed is True


def test_closing_changes_nothing(connected_dialog):
    microscope = connected_dialog.microscope

    connected_dialog.offline_button.click()

    assert connected_dialog.changed is False
    assert connected_dialog.microscope is microscope


def test_reconnecting_replaces_the_session(connected_dialog, monkeypatch):
    """The old one is released first: leaving it open holds the instrument."""
    _answer_disconnect(monkeypatch, True)
    original = connected_dialog.microscope

    connected_dialog.connect_to_microscope()

    assert connected_dialog.changed is True
    assert connected_dialog.microscope is not None
    assert connected_dialog.microscope is not original


def test_the_helper_reports_no_change_when_nothing_happened(qapp):
    """A cancel must not read as a disconnect, or the caller drops the session."""
    from fibsem.ui.widgets.connection_dialog import ConnectionResult

    microscope, settings = utils.setup_session(manufacturer="Demo")
    try:
        result = ConnectionResult(False, microscope, settings)
        assert result.changed is False
        assert result.microscope is microscope
    finally:
        microscope.disconnect()


def test_a_disconnected_dialog_offers_no_disconnect(dialog):
    assert not dialog.disconnect_button.isVisible()
    assert dialog.connect_button.text() == "Connect"
    assert dialog.offline_button.text() == "Not now"


def test_disconnecting_asks_first(connected_dialog, monkeypatch):
    """Not undoable from here, so it is not done quietly."""
    asked = _record_question(monkeypatch)
    microscope = connected_dialog.microscope

    connected_dialog.disconnect_button.click()

    info = microscope.system.info
    assert info.ip_address in asked["text"]
    # Declined, so nothing happened at all.
    assert connected_dialog.microscope is microscope
    assert connected_dialog.changed is False


def test_declining_a_reconnect_keeps_the_session(connected_dialog, monkeypatch):
    """Reconnecting ends the session too, so it asks the same question.

    Saying no must leave the old microscope in place, not half-dropped.
    """
    _answer_disconnect(monkeypatch, False)
    original = connected_dialog.microscope

    connected_dialog.connect_to_microscope()

    assert connected_dialog.microscope is original
    assert connected_dialog.changed is False


def test_losing_the_session_mid_workflow_says_what_it_will_break(qapp, monkeypatch):
    """Still allowed -- an instrument that has already gone needs releasing exactly
    when a workflow believes it is using it -- but the warning has to say so."""
    asked = _record_question(monkeypatch)

    microscope, settings = utils.setup_session(manufacturer="Demo")
    d = ConnectionDialog(
        microscope=microscope, settings=settings, workflow_running=True
    )
    d.show()
    try:
        d.disconnect_button.click()
        assert "workflow is running" in asked["text"].lower()
    finally:
        microscope.disconnect()
        d.deleteLater()


def test_the_details_panel_reports_the_connection(connected_dialog):
    """Reported as the Connection tab's card reports it, tick included.

    Deliberate: this panel describes the outcome of a connection that was
    established, in a dialog someone opened to look at it. The header chip in #515
    stays wordless about the link for a different reason -- it is on screen for
    hours, and nothing watches the connection (FIB-777).
    """
    info = connected_dialog.microscope.system.info

    assert connected_dialog.details_title.text() == "Microscope connected"
    assert info.manufacturer in connected_dialog.details_subtitle.text()
    assert info.ip_address in connected_dialog.details_subtitle.text()
    assert info.serial_number in connected_dialog.details_subtitle.toolTip()


def test_a_configuration_that_resolves_nowhere_shows_in_the_panel(dialog):
    dialog.configuration_combo.addItem("a-configuration-that-was-deleted")
    dialog.configuration_combo.setCurrentText("a-configuration-that-was-deleted")

    assert dialog.details_title.text() == "No configuration"
    assert "could not be found" in dialog.details_subtitle.text()
