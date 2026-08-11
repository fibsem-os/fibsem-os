"""The compact lamella card stays compact, and stays clickable (FIB-585).

The card was a 300x240 portrait tile with a 170px thumbnail, stacked one per row in a
340px strip, so two lamellae filled the panel. It is now a row -- thumbnail, name and
status, buttons -- at roughly a third the height.

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
