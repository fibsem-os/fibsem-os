"""The compact lamella card stays compact, and stays clickable (FIB-585).

The card was a 300x240 portrait tile with a 170px thumbnail, stacked one per row in a
340px strip, so two lamellae filled the panel. There are now three arrangements, named
for density the way Discord and Gmail name theirs -- `cozy` (that tile), `standard` (the
default: a small thumbnail beside the name) and `compact` (one line, no thumbnail) --
chosen from Preferences -> Display.

All three share one set of child widgets, re-parented between containers, so `refresh`
stays a single display path and there is no second one to keep correct.

Two properties are worth pinning, and neither is the pixel height:

* **A long status must not grow the row.** The density comes from a fixed-height row,
  and the status is the one field with unbounded content (a task name, or a completion
  stamp). It is an `ElidedLabel` for that reason; swapping in a wrapping `QLabel` would
  restore the old behaviour silently, for the one lamella whose task has a long name.
* **Clicking the card selects it, wherever you click.** Selection rides on
  `mousePressEvent` bubbling up from children that ignore the press. That is free with
  labels and would break the moment one is replaced with something that accepts clicks
  -- and the failure is "clicking the picture does nothing", which no other test covers.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, Lamella
from fibsem.applications.autolamella.ui.lamella_card_widget import (
    LamellaCardContainer,
    LamellaCardWidget,
)
from fibsem.config import MODE_COMPACT, MODE_COZY, MODE_STANDARD

_app = QApplication.instance() or QApplication(sys.argv)

_LONG_TASK_NAME = (
    "Acquire Fluorescence Reference Image at super high resolution, every channel"
)


def _lamella(petname="test", task=None):
    lamella = Lamella(path="/tmp", number=0, petname=petname)
    if task:
        lamella.task_state.name = task
        lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    return lamella


# Heights come from sizeHint(), not from a shown widget: a top-level window here has a
# minimum height of 100px, which floors both a 66px row and anything smaller into the
# same number and would make the comparison below pass whatever the layout does.
_COMPACT_MAX_H = 80


def _shown(card):
    """Lay the card out, so child geometry is real."""
    card.show()
    _app.processEvents()
    return card


def test_a_long_status_does_not_make_the_card_taller():
    short = LamellaCardWidget(_lamella("short", task="Mill Rough"))
    long = LamellaCardWidget(_lamella("long", task=_LONG_TASK_NAME))

    assert long.sizeHint().height() == short.sizeHint().height()
    # ... and both are actually short. Without this the equality above would still
    # hold if the row grew to fit two wrapped lines in every case.
    assert short.sizeHint().height() <= _COMPACT_MAX_H


def test_the_full_status_survives_as_a_tooltip():
    """Eliding is presentation; the string still has to be readable somehow."""
    card = _shown(LamellaCardWidget(_lamella(task=_LONG_TASK_NAME)))

    assert card._status_label.toolTip() == _LONG_TASK_NAME


def test_the_name_sits_beside_the_thumbnail_not_under_it():
    """The layout change itself, in the one form that does not depend on any margin:
    the old tile stacked the name below a full-width thumbnail. Mapped into the card's
    own coordinates, because the two live in different parent widgets."""
    card = _shown(LamellaCardWidget(_lamella()))

    thumb = card._thumb_label
    thumb_right = thumb.mapTo(card, thumb.rect().topRight()).x()
    name_left = card._name_label.mapTo(card, card._name_label.rect().topLeft()).x()

    assert name_left > thumb_right


def test_clicking_the_thumbnail_selects_the_card():
    """The click lands on a child, and only reaches the card because labels ignore it."""
    container = LamellaCardContainer(columns=1)
    lamella = _lamella("clickable")
    card = container.add_lamella(lamella)
    container.show()
    _app.processEvents()

    seen = []
    container.lamella_selected.connect(seen.append)
    QTest.mouseClick(card._thumb_label, Qt.LeftButton)

    assert seen == [lamella]
    assert container._selected_id == lamella.id


def test_clicking_the_defect_button_does_not_select_the_card():
    """The buttons accept the press, so it must not also reach the card underneath --
    otherwise opening the defect menu would silently change which lamella is selected."""
    container = LamellaCardContainer(columns=1)
    card = container.add_lamella(_lamella("not-selected"))
    container.show()
    _app.processEvents()

    seen = []
    container.lamella_selected.connect(seen.append)
    QTest.mousePress(card._btn_defect, Qt.LeftButton)

    assert seen == []


# ── switching density (FIB-585) ───────────────────────────────────────────────
#
# All three arrangements share one set of child widgets, re-parented between
# containers, so what needs pinning is that nothing is lost in the move: the menus,
# the psygnal connections and the container's selection all outlive a switch.


def test_toggling_to_the_cozy_tile_and_back_restores_the_standard_height():
    card = LamellaCardWidget(_lamella())
    standard_h = card.sizeHint().height()

    card.set_mode(MODE_COZY)
    cozy_h = card.sizeHint().height()
    card.set_mode(MODE_STANDARD)

    assert cozy_h > standard_h * 2, "cozy is the roomiest arrangement"
    assert card.sizeHint().height() == standard_h


def test_the_menus_and_thumbnail_survive_a_toggle():
    """The children are re-parented, not rebuilt, so everything hung off them stays."""
    card = _shown(LamellaCardWidget(_lamella()))

    card.set_mode(MODE_COZY)

    assert len(card._btn_actions.menu().actions()) == 3
    assert card._btn_defect.toolTip() == "No defect"
    assert not card._thumb_label.pixmap().isNull()
    assert card._thumb_label.height() > 100, "redrawn at the cozy size"


def test_the_lamella_events_still_refresh_after_a_toggle():
    """`refresh` is one path for both arrangements, and stays connected across a switch."""
    lamella = _lamella()
    card = _shown(LamellaCardWidget(lamella))

    card.set_mode(MODE_COZY)
    lamella.task_state.name = "Mill Undercut"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress

    assert card._status_label.text() == "Mill Undercut"


def test_the_container_keeps_its_selection_across_a_toggle():
    container = LamellaCardContainer(columns=1)
    lamella = _lamella("kept")
    container.add_lamella(lamella)
    container._on_card_clicked(lamella)

    container.set_mode(MODE_COZY)

    assert container._selected_id == lamella.id
    assert container.mode() == MODE_COZY


def test_a_card_added_later_follows_the_current_mode():
    container = LamellaCardContainer(columns=1)
    container.set_mode(MODE_COZY)

    card = container.add_lamella(_lamella("late"))

    assert card._thumb_label.height() > 100


def test_the_compact_mode_drops_the_thumbnail_and_keeps_the_status():
    """The densest arrangement. The status stays: without it this is a name and two
    buttons, which is what LamellaNameListWidget already draws in the Experiment tab."""
    lamella = _lamella(task="Mill Rough")
    card = _shown(LamellaCardWidget(lamella, mode=MODE_COMPACT))

    assert card._thumb_label.isVisible() is False
    assert card._status_label.text() == "Mill Rough"
    assert card._name_label.isVisible() is True


def test_the_compact_row_is_shorter_than_the_standard_row():
    standard = LamellaCardWidget(_lamella())
    compact = LamellaCardWidget(_lamella(), mode=MODE_COMPACT)

    assert compact.sizeHint().height() < standard.sizeHint().height()


def test_switching_into_compact_and_back_restores_the_thumbnail():
    """The thumbnail is left out of the list's layout entirely, so coming back has to
    both re-add it and redraw its pixmap."""
    card = _shown(LamellaCardWidget(_lamella()))

    card.set_mode(MODE_COMPACT)
    card.set_mode(MODE_STANDARD)

    assert card._thumb_label.isVisible() is True
    assert not card._thumb_label.pixmap().isNull()


def test_an_unknown_mode_is_ignored_rather_than_drawn_blank():
    """A hand-edited preferences file should not be able to empty the panel."""
    card = LamellaCardWidget(_lamella(), mode="enormous")

    assert card.mode() == MODE_STANDARD

    card.set_mode("enormous")

    assert card.mode() == MODE_STANDARD


def test_the_card_mode_survives_the_preferences_dialog():
    """The preference is only reachable if it round-trips the dialog -- the same four
    places any preference has to survive: the field, the widget, the load and the save.
    """
    from fibsem.config import UserPreferences
    from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

    prefs = UserPreferences()
    prefs.display.lamella_card_mode = MODE_COMPACT

    dialog = PreferencesDialog(prefs)
    try:
        assert dialog.get_preferences().display.lamella_card_mode == MODE_COMPACT
    finally:
        dialog.deleteLater()


def test_a_mode_this_build_does_not_know_falls_back_the_same_way_everywhere():
    """The dialog and the cards must agree about what an unreadable value means.

    A real case, not a hypothetical: this branch renamed the modes mid-review, so a
    preferences file written an hour earlier held a name that no longer exists. The
    cards fall back to the default; the dialog used to fall back to its first entry,
    which would have shown "Cozy" over a strip drawing Standard -- and pressing OK
    would then apply a layout nobody chose.
    """
    from fibsem.config import DisplayPreferences, UserPreferences
    from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

    prefs = UserPreferences()
    prefs.display.lamella_card_mode = "tile"  # a name from before the rename

    card = LamellaCardWidget(_lamella(), mode="tile")
    dialog = PreferencesDialog(prefs)
    try:
        fallback = DisplayPreferences().lamella_card_mode
        assert card.mode() == fallback
        assert dialog.get_preferences().display.lamella_card_mode == fallback
    finally:
        dialog.deleteLater()
