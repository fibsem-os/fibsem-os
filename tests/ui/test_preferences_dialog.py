"""The preferences dialog, checked against the dataclasses it edits.

A preference has to survive four separate places to work: the field, the control,
`_load_from_preferences`, and `get_preferences`. Miss the last two and the control is
inert; miss the dialog entirely and the preference exists but nobody can reach it.
None of that fails loudly — it just quietly does nothing.

So these tests drive the dialog's own load/save round trip rather than looking for
attributes by name: what matters is that a value put in comes back out, not that some
particular widget exists.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.dialogs.preferences import PreferencesDialog
from fibsem.config import MODE_COMPACT, QtPreferences, UserPreferences

# Qt switches deliberately not offered in the dialog. Anything here is a decision,
# not an oversight — which is the point of listing them rather than skipping the check.
NOT_IN_DIALOG: set = set()

QT_SWITCHES = [
    f.name
    for f in dataclasses.fields(QtPreferences)
    if f.type in (bool, "bool") and f.name not in NOT_IN_DIALOG
]


def _round_trip(prefs: UserPreferences) -> UserPreferences:
    """Load preferences into a dialog and read them straight back out."""
    dialog = PreferencesDialog(prefs)
    try:
        return dialog.get_preferences()
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("field", QT_SWITCHES)
@pytest.mark.parametrize("value", [True, False])
def test_every_qt_switch_survives_the_dialog(qapp, field, value):
    """Both directions: a checkbox wired only into the save path reads back as
    whatever it defaulted to, which looks like it works until someone unticks it."""
    prefs = UserPreferences()
    setattr(prefs.qt, field, value)

    restored = _round_trip(prefs)

    assert getattr(restored.qt, field) is value, (
        f"qt.{field} did not survive the preferences dialog — it probably has no "
        f"checkbox, or is missing from _load_from_preferences or get_preferences."
    )


def test_the_exemption_list_only_names_switches_that_exist(qapp):
    known = {f.name for f in dataclasses.fields(QtPreferences)}

    assert NOT_IN_DIALOG <= known, f"stale exemptions: {NOT_IN_DIALOG - known}"


def test_the_app_server_is_not_in_preferences(qapp):
    """Its switches live in Tools → App Server, one place; Preferences hands the
    saved values back untouched."""
    prefs = UserPreferences()
    prefs.install.app_server.enabled = True
    prefs.install.app_server.microscope = True
    dialog = PreferencesDialog(prefs)
    pages = [dialog._sidebar.item(i).text() for i in range(dialog._sidebar.count())]
    assert "App Server" not in pages
    rebuilt = dialog.get_preferences()
    assert rebuilt.install.app_server.enabled and rebuilt.install.app_server.microscope
    dialog.deleteLater()


def test_the_app_server_fails_closed():
    # Absent section, empty section and the retired flag all read as off (FIB-1035):
    # `agent_server_enabled` was never in a release, so it is not migrated.
    app_server = lambda d: UserPreferences.from_dict(d).install.app_server  # noqa: E731
    assert app_server({"features": {}}).enabled is False
    assert app_server({"app_server": {}}).enabled is False
    assert app_server({"install": {"app_server": {}}}).microscope is False
    assert app_server({"features": {"agent_server_enabled": True}}).enabled is False


def test_the_features_page_is_gone(qapp):
    """Coincidence milling and the connection chip were its last two flags."""
    dialog = PreferencesDialog(UserPreferences())
    pages = [dialog._sidebar.item(i).text() for i in range(dialog._sidebar.count())]
    assert pages == ["Display", "Experiment", "Movement"]
    dialog.deleteLater()


def _every_field_changed() -> UserPreferences:
    """Preferences with every field moved off its default, so a field the dialog drops
    and rebuilds from scratch cannot pass by coincidence."""
    prefs = UserPreferences()

    prefs.shared.info_bar_fields = {"SEM": ["hfw", "pixel_size"]}
    prefs.shared.default_experiment_directory = "/data/experiments"
    prefs.shared.default_protocol_path = "/data/protocol.yaml"
    prefs.shared.user = "user"
    prefs.shared.project = "project"
    prefs.shared.organisation = "organisation"

    prefs.qt.sound_enabled = True
    prefs.qt.border_enabled = False
    prefs.qt.toast_problems_only = True
    prefs.qt.dev_mode = True
    prefs.qt.lamella_card_mode = MODE_COMPACT
    prefs.qt.acquire_sem_after_stage_movement = False
    prefs.qt.acquire_fib_after_stage_movement = False
    prefs.qt.contact_email = "someone@example.com"

    prefs.install.update_check_enabled = True
    prefs.install.app_server.enabled = True
    prefs.install.app_server.microscope = True
    prefs.install.app_server.allow_local_agents = True
    return prefs


def _leaves(value, prefix: str = "") -> dict:
    """Every field of the preferences, nested sections flattened as `section.field`."""
    leaves = {}
    for f in dataclasses.fields(value):
        inner = getattr(value, f.name)
        if dataclasses.is_dataclass(inner):
            leaves.update(_leaves(inner, f"{prefix}{f.name}."))
        else:
            leaves[f"{prefix}{f.name}"] = inner
    return leaves


def test_the_fixture_changes_every_field():
    """Keeps the round-trip test below honest: a field added to the preferences and not
    here would still be at its default, and a dialog that drops it would pass."""
    defaults = _leaves(UserPreferences())
    changed = _leaves(_every_field_changed())

    unchanged = sorted(name for name in defaults if changed[name] == defaults[name])

    assert not unchanged, (
        f"_every_field_changed leaves these at their defaults: {unchanged}. Set them to "
        f"something else so the dialog round trip covers them."
    )


def test_every_field_survives_the_dialog(qapp):
    """Pressing OK must hand back everything it was given, not only what is on a widget.

    The fields the dialog has no control for -- the App Server, the info bar fields,
    the contact email -- were once rebuilt from defaults or copied one by one, and
    anything not on the list was reset and then saved to disk.
    """
    prefs = _every_field_changed()

    restored = _round_trip(prefs)

    assert _leaves(restored) == _leaves(prefs)
    # A copy, not the object it was given: a cancelled dialog must not have edited it.
    assert restored is not prefs
    assert restored.shared.info_bar_fields is not prefs.shared.info_bar_fields


@pytest.mark.parametrize("enabled", [True, False])
def test_the_update_check_round_trips(qapp, monkeypatch, enabled):
    monkeypatch.setattr("fibsem.update_check.get_revision", lambda: None)
    prefs = UserPreferences()
    prefs.install.update_check_enabled = enabled
    dialog = PreferencesDialog(prefs)
    assert dialog._chk_update_check.isChecked() is enabled
    dialog._chk_update_check.setChecked(not enabled)
    rebuilt = UserPreferences.from_dict(dialog.get_preferences().to_dict())
    assert rebuilt.install.update_check_enabled is (not enabled)
    dialog.deleteLater()


def test_the_update_check_is_offered_only_to_a_release_install(qapp, monkeypatch):
    """A source install never checks, so its checkbox is shown but cannot be ticked."""
    monkeypatch.setattr("fibsem.update_check.get_revision", lambda: None)
    wheel = PreferencesDialog(UserPreferences())
    assert wheel._chk_update_check.isEnabled()
    wheel.deleteLater()

    monkeypatch.setattr(
        "fibsem.update_check.get_revision", lambda: "v0.5.3-1154-g6da166628"
    )
    source = PreferencesDialog(UserPreferences())
    assert not source._chk_update_check.isEnabled()
    source.deleteLater()


def test_the_update_check_stays_off_by_default():
    assert UserPreferences().install.update_check_enabled is False
    assert UserPreferences.from_dict({}).install.update_check_enabled is False
