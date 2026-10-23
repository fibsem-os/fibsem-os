"""The Connection tab, and the first-run offer in the connection dialog.

The header chip that was to replace the tab is gone with its flag (FIB-1544): the
tab is how the operator connects, and File > Connect to Microscope opens the
dialog.

The **guided-setup offer** (#472) is a banner inside `FibsemSystemSetupWidget`
and in the connection dialog; a fresh install sees it in either.

Preferences are redirected to a temp file throughout: `is_first_run` reads the
real configuration registry, and these tests must neither depend on the
developer's own state nor write to it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import config as cfg  # noqa: E402
from fibsem import guided_setup  # noqa: E402
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
from fibsem.ui.widgets.connection_dialog import ConnectionDialog  # noqa: E402


@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """Isolated preferences, and a `load_user_preferences` that returns them."""
    preferences = cfg.UserPreferences()
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: preferences)
    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )
    return preferences


def _tabs(ui: AutoLamellaWindow) -> list[str]:
    return [ui.panel.tabWidget.tabText(i) for i in range(ui.panel.tabWidget.count())]


def test_the_tab_is_there(qapp, prefs):
    """Nothing about connecting has moved."""
    ui = AutoLamellaWindow()
    try:
        assert "Connection" in _tabs(ui)
        # And it is the one the panel opens on, as it always was.
        assert ui.panel.tabWidget.tabText(0) == "Connection"
        # The dialog is in the File menu for everyone.
        assert ui.menu_bar.action_connect_microscope in (
            ui.menuBar().actions()[0].menu().actions()
        )
    finally:
        ui.deleteLater()


# ── the offer that came with the tab ────────────────────────────────────────


def test_the_offer_shows_on_a_fresh_install(qapp, prefs, monkeypatch):
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)

    dialog = ConnectionDialog()
    try:
        dialog.show()
        assert dialog.first_run_frame.isVisible()
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize(
    "dismissed, first_run",
    [
        (True, True),  # already declined
        (False, False),  # something is configured already
    ],
)
def test_the_offer_stays_away_otherwise(qapp, prefs, monkeypatch, dismissed, first_run):
    """Two separate conditions, as the tab applied them.

    Folding them together is what broke this the first time: dismissal used to be
    inferred from the file whose absence means "fresh install", which made a write
    that recorded a dismissal look like the same thing as undoing it.
    """
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: first_run)
    if dismissed:
        guided_setup.dismiss_first_run()

    dialog = ConnectionDialog()
    try:
        dialog.show()
        assert not dialog.first_run_frame.isVisible()
    finally:
        dialog.deleteLater()


def test_declining_the_offer_records_it(qapp, prefs, monkeypatch):
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)
    recorded = []
    monkeypatch.setattr(
        guided_setup, "dismiss_first_run", lambda: recorded.append(True)
    )

    dialog = ConnectionDialog()
    try:
        dialog.show()
        dialog.dismiss_first_run_button.click()

        assert not dialog.first_run_frame.isVisible()
        assert recorded == [True], "declining has to persist, or it returns next start"
    finally:
        dialog.deleteLater()


def test_finishing_the_wizard_selects_what_it_wrote(qapp, prefs, monkeypatch):
    """Otherwise the dialog still points at whatever was selected before it ran."""
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)

    dialog = ConnectionDialog()
    try:
        dialog.show()
        written = "a-brand-new-configuration"
        monkeypatch.setitem(
            cfg.USER_CONFIGURATIONS, written, {"path": "/tmp/nope.yaml"}
        )
        monkeypatch.setattr(
            "fibsem.ui.widgets.guided_setup_dialog.open_guided_setup",
            lambda parent=None, microscope=None: written,
        )

        assert dialog.run_guided_setup() == written
        assert dialog.configuration_combo.currentText() == written
    finally:
        dialog.deleteLater()
