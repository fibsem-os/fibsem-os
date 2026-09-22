"""The correlation tab widget's lists and canvases draw from one store (FIB-973).

The five coordinate lists and the two canvases each held a copy of the points
and of the selection, and the tab widget relayed between them. They now share
one ``CorrelationPointStore``, so they cannot disagree. These check the tab
widget after each kind of gesture: every canvas shows exactly its lists' points,
and highlights exactly what its lists have selected.

Three places used to disagree, the list showing a selection the canvas did not:
after a load, after a point was added, and after a click on empty canvas.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_correlation_tab_shares_one_store.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from matplotlib.backend_bases import KeyEvent, MouseEvent
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage
from fibsem.ui.correlation.widgets.correlation_tab_widget import CorrelationTabWidget

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, pt, **kwargs):
    return Coordinate(PointXYZ(float(x), 50.0, 0.0), pt, **kwargs)


@pytest.fixture
def tab():
    widget = CorrelationTabWidget()
    widget.resize(1200, 800)
    widget.show()
    widget.set_fib_image(FibsemImage(data=np.zeros((200, 300), np.uint8)))
    fib = [_coord(40 + 40 * i, PointType.FIB) for i in range(4)]
    fm = [_coord(40 + 40 * i, PointType.FM) for i in range(4)]
    widget.set_data(
        CorrelationInputData(
            fib_coordinates=fib,
            fm_coordinates=fm,
            poi_coordinates=[_coord(90, PointType.POI)],
        )
    )
    _overlay(widget, "fib")._canvas.draw()
    _app.processEvents()
    yield widget
    widget.close()


def _overlay(tab, side):
    return tab._adapters[side]._surface.picking.points


def _list(tab, pt):
    return tab._point_specs[pt].list_widget


def _assert_in_agreement(tab):
    for side, adapter in tab._adapters.items():
        overlay = _overlay(tab, side)
        specs = [s for s in tab._point_specs.values() if s.adapter is adapter]
        listed = [c for s in specs for c in s.list_widget.coordinates]
        drawn = overlay.coordinates()
        assert len(drawn) == len(listed)
        assert all(a is b for a, b in zip(drawn, listed))

        selected = [
            s.list_widget.selected_coordinate
            for s in specs
            if s.list_widget.selected_coordinate is not None
        ]
        assert len(selected) <= 1
        assert overlay.selected_coordinate() is (selected[0] if selected else None)
        # and what is highlighted is what is selected
        assert overlay._selected == overlay.index_of(overlay.selected_coordinate())


def _mouse(tab, name, x, y):
    overlay = _overlay(tab, "fib")
    sx, sy = overlay._ax.transData.transform((x, y))
    return MouseEvent(name, overlay._canvas, sx, sy, button=1)


def _click(tab, x, y):
    overlay = _overlay(tab, "fib")
    overlay._on_press(_mouse(tab, "button_press_event", x, y))
    overlay._on_release(_mouse(tab, "button_release_event", x, y))


def test_the_lists_and_the_canvases_hold_the_same_store(tab):
    store = tab._point_store
    assert all(s.list_widget.store is store for s in tab._point_specs.values())
    assert all(_overlay(tab, side).store is store for side in ("fib", "fm"))


def test_a_load_highlights_on_the_canvas_what_the_list_selected(tab):
    assert _list(tab, PointType.POI).selected_coordinate is not None
    _assert_in_agreement(tab)


def test_an_added_point_is_highlighted_on_the_canvas(tab):
    tab._on_canvas_add_requested(150.0, 150.0, PointType.FIB)
    added = _list(tab, PointType.FIB).coordinates[-1]
    assert _overlay(tab, "fib").selected_coordinate() is added
    _assert_in_agreement(tab)


def test_a_click_on_empty_canvas_clears_the_lists_selection_too(tab):
    _click(tab, 80, 50)
    assert _list(tab, PointType.FIB).selected_coordinate is not None
    _click(tab, 280, 190)
    assert _list(tab, PointType.FIB).selected_coordinate is None
    _assert_in_agreement(tab)


def test_a_canvas_click_selects_the_row_and_clears_the_other_side(tab):
    fib = _list(tab, PointType.FIB).coordinates
    _click(tab, 80, 50)
    assert _list(tab, PointType.FIB).selected_coordinate is fib[1]
    assert _list(tab, PointType.POI).selected_coordinate is None
    _assert_in_agreement(tab)


def test_a_row_click_highlights_on_the_canvas(tab):
    fm = _list(tab, PointType.FM).coordinates
    _list(tab, PointType.FM)._on_row_clicked(fm[2])
    assert _overlay(tab, "fm").selected_coordinate() is fm[2]
    assert _overlay(tab, "fib").selected_coordinate() is None
    _assert_in_agreement(tab)


def test_a_drag_reaches_the_row(tab):
    fib = _list(tab, PointType.FIB).coordinates
    overlay = _overlay(tab, "fib")
    overlay._on_press(_mouse(tab, "button_press_event", 120, 50))
    overlay._on_motion(_mouse(tab, "motion_notify_event", 130, 70))
    overlay._on_release(_mouse(tab, "button_release_event", 130, 70))

    row = _list(tab, PointType.FIB)._list.itemWidget(
        _list(tab, PointType.FIB)._list.item(2)
    )
    assert round(fib[2].point.x) == 130
    assert round(row.x_spin.value()) == 130
    _assert_in_agreement(tab)


def test_a_pair_link_shows_the_pair_on_both_canvases(tab):
    fib = _list(tab, PointType.FIB).coordinates
    fm = _list(tab, PointType.FM).coordinates
    tab._on_status_link("pair:1")
    assert _overlay(tab, "fm").selected_coordinate() is fm[1]
    assert _overlay(tab, "fib").selected_coordinate() is fib[1]
    _assert_in_agreement(tab)


def test_canvas_delete_removes_one_point_everywhere(tab):
    fib = _list(tab, PointType.FIB).coordinates
    _click(tab, 80, 50)
    overlay = _overlay(tab, "fib")
    overlay._on_key(KeyEvent("key_press_event", overlay._canvas, "delete"))

    left = _list(tab, PointType.FIB).coordinates
    assert len(left) == 3 and all(c is not fib[1] for c in left)
    assert _list(tab, PointType.FIB).selected_coordinate is fib[2]  # the neighbour
    _assert_in_agreement(tab)


def test_the_trash_button_removes_one_point_everywhere(tab):
    fm = _list(tab, PointType.FM).coordinates
    _list(tab, PointType.FM)._on_row_clicked(fm[1])
    _list(tab, PointType.FM)._on_remove(fm[1])
    assert len(_overlay(tab, "fm").coordinates()) == 4  # 3 FM + 1 POI
    assert _overlay(tab, "fm").selected_coordinate() is fm[2]
    _assert_in_agreement(tab)


def test_surface_points_stay_mutually_exclusive(tab):
    tab._on_canvas_add_requested(60.0, 20.0, PointType.SURFACE)
    tab._on_canvas_add_requested(70.0, 25.0, PointType.SURFACE)
    assert len(_list(tab, PointType.SURFACE).coordinates) == 1
    _assert_in_agreement(tab)

    tab._on_canvas_add_requested(65.0, 22.0, PointType.SURFACE_FM)
    assert _list(tab, PointType.SURFACE).coordinates == []
    assert len(_list(tab, PointType.SURFACE_FM).coordinates) == 1
    _assert_in_agreement(tab)


def test_a_reorder_and_a_reject_keep_the_canvas_in_step(tab):
    fib_list = _list(tab, PointType.FIB)
    fib_list._on_reordered(list(reversed(fib_list.coordinates)))
    _assert_in_agreement(tab)

    # on the FIB side: the fixture's FM canvas has no image, so draws no artists
    fib = fib_list.coordinates
    tab._on_reject_toggled(tab._point_specs[PointType.FIB], fib[0])
    assert fib[0].status == PointStatus.REJECTED
    overlay = _overlay(tab, "fib")
    assert overlay._artists[overlay.index_of(fib[0])].get_alpha() == 0.35
    _assert_in_agreement(tab)


def test_reassigning_the_same_data_keeps_everything(tab):
    before = tab.data
    tab.set_data(tab.data)
    after = tab.data
    assert all(a is b for a, b in zip(before.fm_coordinates, after.fm_coordinates))
    _assert_in_agreement(tab)
