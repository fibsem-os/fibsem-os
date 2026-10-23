"""A lamella outlives the rows and cards that show it.

From a real run: the agent patched a lamella's description, and a row the list
had already rebuilt away was still connected. Its Python wrapper was alive, its
QLabel was not, so the refresh raised, the patch raised through `setattr`, and
the App Server answered 500 for an edit it had applied.
"""

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import sip
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.lamella.cards import LamellaCardWidget
from fibsem.applications.autolamella.ui.lamella.rows import (
    LamellaListWidget,
    LamellaNameListWidget,
)
from fibsem.server.config_patch import apply_patch

_app = QApplication.instance() or QApplication(sys.argv)


def lamella(tmp_path) -> Lamella:
    return Lamella(path=str(tmp_path / "lam1"), number=1, petname="lam-1")


def _drain_deletes() -> None:
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("list_type", [LamellaNameListWidget, LamellaListWidget])
def test_a_rebuilt_away_row_stops_following(list_type, tmp_path):
    lam = lamella(tmp_path)
    widget = list_type()
    rebuild = (
        widget.set_lamella
        if list_type is LamellaNameListWidget
        else widget.set_lamellae
    )
    rebuild([lam])
    old = widget._row(0) if hasattr(widget, "_row") else next(widget._rows())

    rebuild([lam])
    _drain_deletes()

    # Held alive, as a reference cycle does until the collector runs.
    assert sip.isdeleted(old)
    assert len(lam.events.description) == 1  # the new row only
    apply_patch(lam, {"description": "Candidate C"})
    assert lam.description == "Candidate C"
    widget.deleteLater()
    _drain_deletes()


def test_a_destroyed_card_stops_following(tmp_path):
    lam = lamella(tmp_path)
    card = LamellaCardWidget(lam)

    card.deleteLater()
    _drain_deletes()

    assert sip.isdeleted(card)
    assert len(lam.events.defect) == 0
    assert len(lam.events.description) == 0


def test_a_failing_listener_does_not_fail_the_patch(tmp_path, caplog):
    lam = lamella(tmp_path)

    def broken(*_):
        raise RuntimeError("wrapped C/C++ object of type QLabel has been deleted")

    lam.events.description.connect(broken)

    with caplog.at_level(logging.ERROR):
        changes = apply_patch(lam, {"description": "Candidate D"})

    assert changes == [("description", "", "Candidate D")]
    assert lam.description == "Candidate D"
    assert "a listener failed on the change to 'description'" in caplog.text
