"""The preferences dialog, checked against the dataclass it edits.

A preference has to survive four separate places to work: the `FeatureFlags` field, the
checkbox, `_load_from_preferences`, and `get_preferences`. Miss the last two and the
control is inert; miss the dialog entirely and the preference exists but nobody can
reach it. None of that fails loudly — it just quietly does nothing, which is how the FM
Overview flag once shipped without a checkbox.

So these tests drive the dialog's own load/save round trip rather than looking for
attributes by name: what matters is that a value put in comes back out, not that some
particular widget exists.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses

import pytest

pytest.importorskip("PyQt5")

from fibsem.config import MODE_COMPACT, FeatureFlags, UserPreferences
from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

# Flags deliberately not offered in the dialog. Anything here is a decision, not an
# oversight — which is the point of listing them rather than skipping the check.
#
# Empty, and kept that way on purpose: it held `viewer_movement_events`, which had no
# consumer beyond the config global it set and was deleted with the rest of the dead
# flag plumbing. The set stays so the next flag added without a checkbox has to be
# named here rather than quietly slipping through.
NOT_IN_DIALOG: set = set()


def _round_trip(prefs: UserPreferences) -> UserPreferences:
    """Load preferences into a dialog and read them straight back out."""
    dialog = PreferencesDialog(prefs)
    try:
        return dialog.get_preferences()
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize(
    "field",
    [f.name for f in dataclasses.fields(FeatureFlags) if f.name not in NOT_IN_DIALOG],
)
def test_every_feature_flag_survives_the_dialog(qapp, field):
    """A flag the dialog cannot carry is a flag nobody can turn on.

    Parameterised over the dataclass rather than written out one flag at a time, so a
    field added later is covered the day it is added — either it round-trips, or its
    author has to say in `NOT_IN_DIALOG` that leaving it out was deliberate.
    """
    prefs = UserPreferences()
    setattr(prefs.features, field, True)

    restored = _round_trip(prefs)

    assert getattr(restored.features, field) is True, (
        f"features.{field} did not survive the preferences dialog — it probably has no "
        f"checkbox, or is missing from _load_from_preferences or get_preferences."
    )


@pytest.mark.parametrize(
    "field",
    [f.name for f in dataclasses.fields(FeatureFlags) if f.name not in NOT_IN_DIALOG],
)
def test_every_feature_flag_can_be_turned_off_again(qapp, field):
    """The other direction. A checkbox wired only into the save path reads back as
    whatever it defaulted to, which looks like it works until someone unticks it."""
    prefs = UserPreferences()
    setattr(prefs.features, field, False)

    restored = _round_trip(prefs)

    assert getattr(restored.features, field) is False


def test_the_exemption_list_only_names_flags_that_exist(qapp):
    """A flag removed or renamed would otherwise leave a stale exemption behind,
    silently excusing nothing and hiding the next real gap."""
    known = {f.name for f in dataclasses.fields(FeatureFlags)}

    assert NOT_IN_DIALOG <= known, f"stale exemptions: {NOT_IN_DIALOG - known}"


def test_agent_section_round_trips(qapp):
    from fibsem.config import UserPreferences
    from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

    prefs = UserPreferences()
    prefs.agent.watchdog_minutes = 12
    dialog = PreferencesDialog(prefs)
    assert dialog._spin_watchdog.value() == 12
    dialog._spin_watchdog.setValue(7)
    rebuilt = dialog.get_preferences()
    assert rebuilt.agent.watchdog_minutes == 7
    # And the section survives serialization (absent key -> defaults).
    again = UserPreferences.from_dict(rebuilt.to_dict())
    assert again.agent.watchdog_minutes == 7
    assert UserPreferences.from_dict({"features": {}}).agent.watchdog_minutes == 5
    dialog.deleteLater()


def _every_field_changed() -> UserPreferences:
    """Preferences with every field moved off its default, so a field the dialog drops
    and rebuilds from scratch cannot pass by coincidence."""
    prefs = UserPreferences()

    prefs.display.sound_enabled = True
    prefs.display.border_enabled = False
    prefs.display.dev_mode = True
    prefs.display.lamella_card_mode = MODE_COMPACT
    prefs.display.guided_setup_dismissed = True
    prefs.display.info_bar_fields = {"SEM": ["hfw", "pixel_size"]}

    for flag in dataclasses.fields(FeatureFlags):
        setattr(prefs.features, flag.name, True)

    prefs.movement.acquire_sem_after_stage_movement = False
    prefs.movement.acquire_fib_after_stage_movement = False

    prefs.experiment.default_experiment_directory = "/data/experiments"
    prefs.experiment.default_protocol_path = "/data/protocol.yaml"
    prefs.experiment.last_experiment_path = "/data/experiments/last/experiment.yaml"
    prefs.experiment.recent_experiments = ["/data/experiments/last/experiment.yaml"]
    prefs.experiment.user = "user"
    prefs.experiment.project = "project"
    prefs.experiment.organisation = "organisation"

    prefs.reporting.contact_email = "someone@example.com"
    prefs.reporting.crash_reporting_enabled = True
    prefs.reporting.sentry_dsn = "https://dsn.example.com/1"
    prefs.reporting.update_check_enabled = True

    prefs.agent.watchdog_minutes = 12

    prefs.hooks = [{"type": "notification", "name": "custom"}]
    return prefs


def _leaves(prefs: UserPreferences) -> dict:
    """Every field of the preferences, nested sections flattened as `section.field`."""
    leaves = {}
    for top in dataclasses.fields(UserPreferences):
        value = getattr(prefs, top.name)
        if dataclasses.is_dataclass(value):
            for sub in dataclasses.fields(value):
                leaves[f"{top.name}.{sub.name}"] = getattr(value, sub.name)
        else:
            leaves[top.name] = value
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

    The fields the dialog has no control for -- the hooks, whether the guided setup was
    dismissed, the recent experiments, reporting -- were once rebuilt from defaults or
    copied one by one, and anything not on the list was reset and then saved to disk.
    """
    prefs = _every_field_changed()

    restored = _round_trip(prefs)

    assert _leaves(restored) == _leaves(prefs)
    # A copy, not the object it was given: a cancelled dialog must not have edited it.
    assert restored is not prefs
    assert restored.hooks is not prefs.hooks


@pytest.mark.parametrize(
    "hooks",
    [None, [], [{"type": "notification", "name": "custom"}]],
    ids=["never-configured", "all-turned-off", "customised"],
)
def test_hooks_survive_the_dialog(qapp, hooks):
    """None and [] mean different things (FIB-497): None is "use the defaults", [] is
    "the user turned every hook off". The dialog has no hooks control, so it must hand
    back exactly what it was given -- turning [] into None brings the defaults back."""
    prefs = UserPreferences()
    prefs.hooks = hooks

    restored = _round_trip(prefs)

    assert restored.hooks == hooks
    assert (restored.hooks is None) == (hooks is None)
