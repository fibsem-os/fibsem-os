"""Hovering a lamella row's name brings up its thumbnail. The hover used to be
watched on the defect icon alone, which is drawn only once there is a verdict,
so an unjudged lamella never showed one."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.lamella.rows import LamellaRow


@pytest.fixture
def row(tmp_path, destroy_widgets_after_test):
    QApplication.instance() or QApplication([])
    lamella = Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    widget = LamellaRow(lamella, thumbnail=True)
    widget.show()
    yield widget
    widget.close()


def test_hovering_the_name_starts_the_thumbnail_and_leaving_stops_it(row):
    assert not row.btn_defect.isVisibleTo(row)  # no verdict, no icon to hover

    QApplication.sendEvent(row.name_label, QEvent(QEvent.Type.Enter))
    assert row._hover_timer.isActive()

    QApplication.sendEvent(row.name_label, QEvent(QEvent.Type.Leave))
    assert not row._hover_timer.isActive()


def test_the_thumbnail_shows_after_the_hover(row):
    row._show_popup()
    assert row._popup is not None and row._popup.isVisible()
    QApplication.sendEvent(row.name_label, QEvent(QEvent.Type.Leave))
    assert not row._popup.isVisible()
