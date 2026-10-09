"""Deleting a FIB or FM fiducial asks what happens to its partner (FIB-1243).

FIB and FM fiducials pair by their place in the lists, so deleting FIB 2 alone
pairs FIB 3 with FM 2 and so on down, silently. Deleting one with a partner now
asks: remove both, remove only that point, or cancel. "Don't ask again" stores
the answer in the Method panel's Delete fiducial setting, which is part of the
experiment's correlation config.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_correlation_delete_fiducial_pair.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage
from fibsem.ui.correlation.widgets.correlation_tab_widget import CorrelationTabWidget

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, pt):
    return Coordinate(PointXYZ(float(x), 50.0, 0.0), pt)


@pytest.fixture
def tab():
    widget = CorrelationTabWidget()
    widget.resize(1200, 800)
    widget.show()
    widget.set_fib_image(FibsemImage(data=np.zeros((200, 300), np.uint8)))
    widget.set_data(
        CorrelationInputData(
            fib_coordinates=[_coord(40 + 40 * i, PointType.FIB) for i in range(4)],
            fm_coordinates=[_coord(41 + 40 * i, PointType.FM) for i in range(4)],
            poi_coordinates=[_coord(90, PointType.POI)],
        )
    )
    _app.processEvents()
    yield widget
    widget.close()


def _points(tab, pt):
    return tab._point_specs[pt].list_widget.coordinates


def _pairs(tab):
    return list(zip(_points(tab, PointType.FIB), _points(tab, PointType.FM)))


def _answer(tab, choice, asked):
    """Stand in for the dialog: record the question, give ``choice``."""

    def ask(coord, partner):
        asked.append((coord, partner))
        return choice

    tab._ask_pair_removal = ask


def _delete_from_list(tab, coord):
    tab._point_specs[coord.point_type].list_widget._on_remove(coord)


def test_remove_both_keeps_every_later_pair_together(tab):
    pairs = _pairs(tab)
    asked = []
    _answer(tab, "pair", asked)

    _delete_from_list(tab, pairs[1][0])  # FIB 2

    assert asked == [pairs[1]]
    assert _pairs(tab) == [pairs[0], pairs[2], pairs[3]]


def test_remove_only_that_point_does_what_it_says(tab):
    pairs = _pairs(tab)
    _answer(tab, "one", [])

    _delete_from_list(tab, pairs[1][1])  # FM 2

    assert _points(tab, PointType.FM) == [pairs[0][1], pairs[2][1], pairs[3][1]]
    assert len(_points(tab, PointType.FIB)) == 4


def test_cancel_removes_nothing_and_announces_nothing(tab):
    before = list(tab._point_store.coordinates)
    _answer(tab, "", [])
    removed = []
    for side in ("fib", "fm"):
        tab._adapters[side]._surface.picking.points.coordinate_removed.connect(
            removed.append
        )

    _delete_from_list(tab, _points(tab, PointType.FIB)[1])
    overlay = tab._adapters["fib"]._surface.picking.points
    overlay.remove_point(2)  # the canvas path too

    assert tab._point_store.coordinates == before
    assert removed == []


def test_the_canvas_and_the_delete_key_ask_too(tab):
    pairs = _pairs(tab)
    asked = []
    _answer(tab, "pair", asked)

    tab._adapters["fib"]._surface.picking.points.remove_point(0)  # canvas
    tab._point_store.select(pairs[3][1])
    tab._remove_selected_coordinate()  # Delete key, on FM 4

    assert [a[0] for a in asked] == [pairs[0][0], pairs[3][1]]
    assert _pairs(tab) == [pairs[1], pairs[2]]


def test_a_point_without_a_partner_is_removed_without_asking(tab):
    asked = []
    _answer(tab, "pair", asked)
    tab._point_store.add(_coord(250, PointType.FIB))  # FIB 5, no FM 5
    extra = _points(tab, PointType.FIB)[-1]

    _delete_from_list(tab, extra)
    _delete_from_list(tab, _points(tab, PointType.POI)[0])

    assert asked == []
    assert extra not in tab._point_store
    assert _points(tab, PointType.POI) == []


def test_dont_ask_again_stores_the_answer_in_the_method_panel(tab):
    """The real dialog: tick "Don't ask again", press Remove both."""
    pairs = _pairs(tab)

    def answer_dialog():
        box = QApplication.activeModalWidget()
        assert isinstance(box, QMessageBox)
        assert "FIB 1 is paired with FM 1" in box.text()
        box.checkBox().setChecked(True)
        both = next(b for b in box.buttons() if b.text() == "Remove both")
        both.click()

    QTimer.singleShot(0, answer_dialog)
    _delete_from_list(tab, pairs[0][0])

    assert _pairs(tab) == [pairs[1], pairs[2], pairs[3]]
    assert tab._coords_tab._delete_fiducial_combo.value() == "pair"
    assert tab.correlation_config.delete_fiducial == "pair"

    def must_not_ask(*_):
        raise AssertionError("asked again after Don't ask again")

    tab._ask_pair_removal = must_not_ask
    _delete_from_list(tab, pairs[1][1])
    assert _pairs(tab) == [pairs[2], pairs[3]]


def test_the_setting_travels_with_the_correlation_config(tab):
    config = CorrelationConfig.from_dict({"delete_fiducial": "one"})
    tab.set_correlation_config(config)
    assert tab._coords_tab._delete_fiducial_combo.value() == "one"
    assert tab.correlation_config.to_dict()["delete_fiducial"] == "one"

    # an older protocol has no key, and a bad value is not trusted: ask
    assert CorrelationConfig.from_dict({}).delete_fiducial == "ask"
    assert (
        CorrelationConfig.from_dict({"delete_fiducial": "x"}).delete_fiducial == "ask"
    )
