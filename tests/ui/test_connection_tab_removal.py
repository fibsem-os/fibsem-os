"""The Connection tab goes, and the first-run offer goes with it — somewhere.

The tab is a gate in a bar that otherwise means "the instrument needs you here
now": the only tab that can never be a workflow step, sitting at position 0 for
something done once a session (FIB-775). With `features.connection_chip` on, the
dialog and the header chip reach the connection instead, and the tab is not built.

The part that would have been lost silently is the **guided-setup offer** (#472).
It is a banner inside `FibsemSystemSetupWidget`, and the Connection tab was the
only place it rendered — removing the tab would have left a fresh install with
Tools > Guided Setup, which is not somewhere a first-time user looks. It now lives
in the connection dialog, which is what they reach first.

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
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
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


def _tabs(ui: AutoLamellaUI) -> list[str]:
    return [ui.tabWidget.tabText(i) for i in range(ui.tabWidget.count())]


def test_the_tab_is_there_with_the_flag_off(qapp, prefs):
    """The shipping default. Nothing about connecting has moved."""
    prefs.features.connection_chip = False

    ui = AutoLamellaUI(parent_ui=None)
    try:
        assert "Connection" in _tabs(ui)
        # And it is the one the panel opens on, as it always was.
        assert ui.tabWidget.tabText(0) == "Connection"
    finally:
        ui.deleteLater()


def test_the_tab_is_gone_with_the_flag_on(qapp, prefs):
    prefs.features.connection_chip = True

    ui = AutoLamellaUI(parent_ui=None)
    try:
        assert "Connection" not in _tabs(ui)
        # The widget stays: it owns the connection, and everything in the
        # application follows its signals. Only the tab was conditional.
        assert ui.system_widget is not None
    finally:
        ui.deleteLater()


# ── the offer that came with the tab ────────────────────────────────────────


def test_the_offer_shows_on_a_fresh_install(qapp, prefs, monkeypatch):
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)
    prefs.display.guided_setup_dismissed = False

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
    prefs.display.guided_setup_dismissed = dismissed

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
