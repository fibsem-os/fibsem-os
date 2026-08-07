"""CorrelationPointOverlay: Coordinate identity survives the index-based base.

The point of the subclass is that callers address points by object identity while
PointOverlay addresses them by index. These tests pin the seam between the two --
above all the cases where indices shift under the caller (removal), which is where
an index<->identity translation layer would have broken.
"""
import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_point_overlay import (
    POINT_COLORS,
    CorrelationPointOverlay,
    generate_names,
)
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(x, y, pt=PointType.FIB):
    return Coordinate(PointXYZ(x, y, 0.0), pt)


def _attached(coords):
    canvas = FibsemImageCanvas()
    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    ov = CorrelationPointOverlay()
    canvas.add_overlay(ov)
    ov.set_coordinates(coords)
    return canvas, ov


# ── identity ──────────────────────────────────────────────────────────────


def test_selection_emits_the_caller_s_own_object():
    a, b = _coord(10, 10), _coord(20, 20)
    _, ov = _attached([a, b])
    seen = []
    ov.coordinate_selected.connect(seen.append)

    ov.point_selected.emit(1, 20.0, 20.0)

    assert len(seen) == 1
    assert seen[0] is b  # identity, not equality


def test_drag_writes_back_into_the_coordinate():
    c = _coord(10, 10)
    _, ov = _attached([c])
    seen = []
    ov.coordinate_moved.connect(seen.append)

    ov.point_moved.emit(0, 33.0, 44.0)

    assert (c.point.x, c.point.y) == (33.0, 44.0)
    assert seen == [c]


def test_removal_reports_identity_before_the_point_goes():
    a, b = _coord(10, 10), _coord(20, 20)
    _, ov = _attached([a, b])
    seen = []
    ov.coordinate_removed.connect(seen.append)

    ov.remove_coordinate(a)

    assert seen == [a]
    assert ov.coordinates() == [b]


# ── the case a translation layer would have broken ────────────────────────


def test_indices_shift_but_identity_does_not():
    a, b, c = _coord(10, 10), _coord(20, 20), _coord(30, 30)
    _, ov = _attached([a, b, c])

    ov.remove_coordinate(a)  # every later index shifts down by one

    assert ov.coordinates() == [b, c]
    assert ov.index_of(b) == 0 and ov.index_of(c) == 1
    seen = []
    ov.coordinate_selected.connect(seen.append)
    ov.point_selected.emit(1, 30.0, 30.0)
    assert seen == [c]  # index 1 is now c, not b


def test_selection_follows_its_point_across_a_removal():
    a, b = _coord(10, 10), _coord(20, 20)
    _, ov = _attached([a, b])
    ov.set_selected_coordinate(b)
    assert ov.selected_coordinate() is b

    ov.remove_coordinate(a)  # base shifts _selected 1 -> 0

    assert ov.selected_coordinate() is b


def test_equal_but_distinct_coordinates_are_not_confused():
    a, b = _coord(10, 10), _coord(10, 10)  # equal values, different objects
    _, ov = _attached([a, b])

    assert a == b
    assert ov.index_of(a) == 0 and ov.index_of(b) == 1


def test_geometry_and_identity_stay_aligned_after_edits():
    a, b, c = _coord(10, 10), _coord(20, 20), _coord(30, 30)
    _, ov = _attached([a, b, c])
    ov.remove_coordinate(b)
    ov.add_coordinate(_coord(40, 40, PointType.POI))

    assert ov.get_points() == [(c.point.x, c.point.y) for c in ov.coordinates()]


# ── heterogeneous types on one surface ────────────────────────────────────


def test_one_overlay_carries_several_point_types():
    fib, poi, surf = (
        _coord(10, 10, PointType.FIB),
        _coord(20, 20, PointType.POI),
        _coord(30, 30, PointType.SURFACE),
    )
    _, ov = _attached([fib, poi, surf])

    assert ov._point_color(0, False) == POINT_COLORS[PointType.FIB]
    assert ov._point_color(1, False) == POINT_COLORS[PointType.POI]
    assert ov._point_marker(0) == "o"
    assert ov._point_marker(2) == "+"  # SURFACE is a crosshair


def test_selected_point_keeps_its_type_colour():
    """Selection is carried by rim and size so the type stays readable."""
    c = _coord(10, 10, PointType.POI)
    _, ov = _attached([c])
    assert ov._point_color(0, True) == POINT_COLORS[PointType.POI]
    assert ov._marker_edge(0, POINT_COLORS[PointType.POI], True)[0] == "white"


def test_unfilled_markers_thicken_rather_than_whiten():
    c = _coord(10, 10, PointType.SURFACE)
    _, ov = _attached([c])
    colour = POINT_COLORS[PointType.SURFACE]
    assert ov._marker_edge(0, colour, False) == (colour, 2.0)
    assert ov._marker_edge(0, colour, True) == (colour, 3.0)


# ── labels ────────────────────────────────────────────────────────────────


def test_names_are_numbered_within_a_type():
    coords = [
        _coord(0, 0, PointType.FIB),
        _coord(1, 1, PointType.POI),
        _coord(2, 2, PointType.FIB),
    ]
    assert generate_names(coords) == ["FIB 1", "POI 1", "FIB 2"]


def test_labels_renumber_after_a_removal():
    a, b, c = (
        _coord(0, 0, PointType.FIB),
        _coord(1, 1, PointType.FIB),
        _coord(2, 2, PointType.FIB),
    )
    _, ov = _attached([a, b, c])
    ov.remove_coordinate(a)
    assert [ov._point_label(i) for i in range(2)] == ["FIB 1", "FIB 2"]


# ── interaction contract ──────────────────────────────────────────────────


def test_right_click_asks_rather_than_adds():
    _, ov = _attached([])
    seen = []
    ov.add_requested.connect(lambda x, y: seen.append((x, y)))

    ov._on_right_click(12.0, 34.0)

    assert seen == [(12.0, 34.0)]
    assert ov.coordinates() == []  # nothing added until the view supplies a type


def test_bare_add_point_is_refused():
    """A point with no matching Coordinate would desync the two lists silently."""
    _, ov = _attached([])
    with pytest.raises(NotImplementedError):
        ov.add_point(1.0, 2.0)


def test_clear_drops_both_lists():
    _, ov = _attached([_coord(10, 10), _coord(20, 20)])
    ov.clear_points()
    assert ov.coordinates() == [] and ov.get_points() == []
