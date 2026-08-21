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

from PyQt5.QtWidgets import QApplication

from fibsem.config import FeatureFlags, UserPreferences
from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

# Flags deliberately not offered in the dialog. Anything here is a decision, not an
# oversight — which is the point of listing them rather than skipping the check.
#
# Empty, and kept that way on purpose: it held `viewer_movement_events`, which had no
# consumer beyond the config global it set and was deleted with the rest of the dead
# flag plumbing. The set stays so the next flag added without a checkbox has to be
# named here rather than quietly slipping through.
NOT_IN_DIALOG: set = set()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


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
