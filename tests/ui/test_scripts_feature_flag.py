"""The user-scripts feature flag (FIB-338).

Scripts ship off. A script gets the application's own access to the microscope and
none of its guard rails, so nobody should acquire the capability by upgrading.

The flag is a normal entry in ``features`` in user-preferences, alongside coincidence
milling and the bug reporter, rather than a bespoke switch -- so it persists, appears
in the Preferences dialog, and can be toggled without a restart.
"""

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as cfg  # noqa: E402
from fibsem.ui.widgets.preferences_dialog import PreferencesDialog  # noqa: E402


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def warnings(monkeypatch):
    """Capture QMessageBox.warning for every test in this module.

    Autouse, not opt-in: setChecked() fires the toggled signal exactly as a click
    does, so any test that flips the checkbox without this stub opens a real modal
    and hangs the suite until it times out. Found the hard way.
    """
    captured = []
    monkeypatch.setattr(
        "fibsem.ui.widgets.preferences_dialog.QMessageBox.warning",
        lambda *args, **kwargs: captured.append(args),
    )
    return captured


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    """Point the preferences file somewhere disposable, so a test never reads or
    overwrites the developer's real one."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    return config_dir


# ── the default ──────────────────────────────────────────────────────────────


def test_scripts_are_off_by_default():
    assert cfg.FeatureFlags().scripts_enabled is False


def test_scripts_are_off_with_no_preferences_file(prefs_file):
    """First launch on a new machine must not have scripts enabled."""
    assert cfg.load_user_preferences().features.scripts_enabled is False


def test_scripts_stay_off_for_a_file_that_predates_the_flag(prefs_file):
    """Upgrading must not turn it on. An older user-preferences.yaml has no
    scripts_enabled key at all, so this is the realistic upgrade path."""
    with open(cfg.USER_PREFERENCES_PATH, "w") as f:
        f.write("features:\n  coincidence_milling_enabled: true\n")

    prefs = cfg.load_user_preferences()

    assert prefs.features.coincidence_milling_enabled is True  # the file was read
    assert prefs.features.scripts_enabled is False


# ── persistence ──────────────────────────────────────────────────────────────


def test_the_flag_round_trips(prefs_file):
    prefs = cfg.load_user_preferences()
    prefs.features.scripts_enabled = True
    cfg.save_user_preferences(prefs)

    assert cfg.load_user_preferences().features.scripts_enabled is True


def test_enabling_scripts_leaves_the_other_flags_alone(prefs_file):
    prefs = cfg.load_user_preferences()
    prefs.features.grid_workflow = True
    prefs.features.scripts_enabled = True
    cfg.save_user_preferences(prefs)

    reloaded = cfg.load_user_preferences().features

    assert (reloaded.scripts_enabled, reloaded.grid_workflow) == (True, True)
    assert reloaded.coincidence_milling_enabled is False


# ── the Preferences dialog ───────────────────────────────────────────────────


def test_the_dialog_shows_the_current_value(qapp):
    prefs = cfg.UserPreferences()
    prefs.features.scripts_enabled = True

    dialog = PreferencesDialog(prefs)

    assert dialog._chk_scripts.isChecked()


def test_the_dialog_reports_the_new_value(qapp):
    dialog = PreferencesDialog(cfg.UserPreferences())
    assert dialog.get_preferences().features.scripts_enabled is False

    dialog._chk_scripts.setChecked(True)

    assert dialog.get_preferences().features.scripts_enabled is True


def test_opening_the_dialog_with_scripts_already_on_does_not_warn(qapp, warnings):
    """The warning is connected after the initial load for exactly this reason: it
    belongs to the act of switching the flag on, not to opening Preferences."""
    prefs = cfg.UserPreferences()
    prefs.features.scripts_enabled = True

    PreferencesDialog(prefs)

    assert warnings == []


def test_switching_scripts_on_warns_that_nothing_is_checked(qapp, warnings):
    dialog = PreferencesDialog(cfg.UserPreferences())

    dialog._chk_scripts.setChecked(True)

    assert len(warnings) == 1
    assert "none of its limits or interlocks" in warnings[0][2]


def test_switching_scripts_back_off_does_not_warn(qapp, warnings):
    prefs = cfg.UserPreferences()
    prefs.features.scripts_enabled = True
    dialog = PreferencesDialog(prefs)

    dialog._chk_scripts.setChecked(False)

    assert warnings == []
