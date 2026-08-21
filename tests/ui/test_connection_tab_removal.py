"""The Connection tab goes, the first-run offer goes with it, and nothing is blank.

The tab is a gate in a bar that otherwise means "the instrument needs you here
now": the only tab that can never be a workflow step, sitting at position 0 for
something done once a session (FIB-775). With `features.connection_chip` on, the
dialog and the header chip reach the connection instead, and the tab is not built.

Two things had to come with it. The **guided-setup offer** (#472) is a banner inside
`FibsemSystemSetupWidget`, and that tab was the only place it rendered — removing
it would have left a fresh install with Tools > Guided Setup, which is not
somewhere a first-time user looks. And the **panel would have been empty**: every
other tab here needs a microscope, so before a connection there was nothing left
to show.

Preferences are redirected throughout: `is_first_run` reads the real configuration
registry, and these tests must neither depend on the developer's own state nor
write to it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import config as cfg  # noqa: E402
from fibsem import guided_setup, utils  # noqa: E402
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


@pytest.fixture
def build(qapp, prefs):
    """Build windows through here, so every one of them is really torn down.

    `deleteLater` is not teardown on its own: without a turn of the event loop the
    widget is still alive, and a shown one stays `QApplication.activeWindow()`.
    That leaks into any later test that asks who is active — the coincidence
    viewer declines to raise itself while another window holds focus, which is
    exactly right of it and exactly wrong of a leftover from here.
    """
    built = []

    def _build(widget):
        built.append(widget)
        widget.show()
        return widget

    yield _build

    for widget in built:
        widget.close()
        widget.deleteLater()
    # `deleteLater` needs a turn of the loop before the object is really gone, and
    # the active window has to be cleared *after* that turn -- clearing it first
    # and then spinning the loop lets the not-yet-deleted window take it back.
    qapp.processEvents()
    qapp.setActiveWindow(None)


def _tabs(ui: AutoLamellaUI) -> list[str]:
    return [ui.tabWidget.tabText(i) for i in range(ui.tabWidget.count())]


# ── the tab ─────────────────────────────────────────────────────────────────


def test_the_tab_is_there_with_the_flag_off(build, prefs):
    """The shipping default. Nothing about connecting has moved."""
    prefs.features.connection_chip = False

    ui = build(AutoLamellaUI(parent_ui=None))

    assert "Connection" in _tabs(ui)
    # And it is the one the panel opens on, as it always was.
    assert ui.tabWidget.tabText(0) == "Connection"


def test_the_tab_is_gone_with_the_flag_on(build, prefs):
    prefs.features.connection_chip = True

    ui = build(AutoLamellaUI(parent_ui=None))

    assert "Connection" not in _tabs(ui)
    # The widget stays: it owns the connection, and everything in the application
    # follows its signals. Only the tab was ever conditional.
    assert ui.system_widget is not None


# ── the space the tab used to fill ──────────────────────────────────────────


def test_the_panel_is_not_left_blank(build, prefs):
    """Every tab in this panel needs a microscope.

    The Experiment tab is hidden until one is connected and the rest are only
    built at connection time, so with the Connection tab gone there is nothing
    left to show — and an empty panel reads as something having gone wrong rather
    than as something not having happened yet.
    """
    prefs.features.connection_chip = True

    ui = build(AutoLamellaUI(parent_ui=None))
    ui.update_ui()

    assert not any(ui.tabWidget.isTabVisible(i) for i in range(ui.tabWidget.count())), (
        "sanity: a tab is visible, so there would be nothing to place-hold"
    )
    assert ui.empty_state.isVisible()
    assert not ui.tabWidget.isVisible()


def test_the_placeholder_gives_way_to_the_tabs(build, prefs):
    prefs.features.connection_chip = True
    ui = build(AutoLamellaUI(parent_ui=None))
    ui.update_ui()
    assert ui.empty_state.isVisible()

    microscope, settings = utils.setup_session(manufacturer="Demo")
    try:
        ui.system_widget.microscope = microscope
        ui.system_widget.settings = settings
        ui.connect_to_microscope()

        assert not ui.empty_state.isVisible()
        assert ui.tabWidget.isVisible()
    finally:
        microscope.disconnect()


def test_the_placeholder_stays_away_with_the_flag_off(build, prefs):
    """The Connection tab is filling that space, as it always did."""
    prefs.features.connection_chip = False

    ui = build(AutoLamellaUI(parent_ui=None))
    ui.update_ui()

    assert ui.tabWidget.isVisible()
    assert not ui.empty_state.isVisible()


# ── the offer that came with the tab ────────────────────────────────────────


def test_the_offer_shows_on_a_fresh_install(build, prefs, monkeypatch):
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)
    prefs.features.guided_setup_enabled = True
    prefs.display.guided_setup_dismissed = False

    dialog = build(ConnectionDialog())

    assert dialog.first_run_frame.isVisible()


@pytest.mark.parametrize(
    "flag, dismissed, first_run",
    [
        (False, False, True),  # the feature is off
        (True, True, True),  # already declined
        (True, False, False),  # something is configured already
    ],
)
def test_the_offer_stays_away_otherwise(
    build, prefs, monkeypatch, flag, dismissed, first_run
):
    """Three separate conditions, as the tab applied them.

    Folding the first two together is what broke this the first time: the flag
    lives in the file whose absence means "fresh install", so turning the flag on
    suppressed the offer it enables.
    """
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: first_run)
    prefs.features.guided_setup_enabled = flag
    prefs.display.guided_setup_dismissed = dismissed

    dialog = build(ConnectionDialog())

    assert not dialog.first_run_frame.isVisible()


def test_declining_the_offer_records_it(build, prefs, monkeypatch):
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)
    prefs.features.guided_setup_enabled = True
    recorded = []
    monkeypatch.setattr(
        guided_setup, "dismiss_first_run", lambda: recorded.append(True)
    )

    dialog = build(ConnectionDialog())
    dialog.dismiss_first_run_button.click()

    assert not dialog.first_run_frame.isVisible()
    assert recorded == [True], "declining has to persist, or it returns next start"


def test_finishing_the_wizard_selects_what_it_wrote(build, prefs, monkeypatch):
    """Otherwise the dialog still points at whatever was selected before it ran."""
    monkeypatch.setattr(guided_setup, "is_first_run", lambda: True)
    prefs.features.guided_setup_enabled = True

    dialog = build(ConnectionDialog())
    written = "a-brand-new-configuration"
    monkeypatch.setitem(cfg.USER_CONFIGURATIONS, written, {"path": "/tmp/nope.yaml"})
    monkeypatch.setattr(
        "fibsem.ui.widgets.guided_setup_dialog.open_guided_setup",
        lambda parent=None, microscope=None: written,
    )

    assert dialog.run_guided_setup() == written
    assert dialog.configuration_combo.currentText() == written
