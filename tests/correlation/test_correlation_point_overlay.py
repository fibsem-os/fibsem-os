"""CorrelationPointOverlay: Coordinate identity survives the index-based base.

The point of the subclass is that callers address points by object identity while
PointOverlay addresses them by index. These tests pin the seam between the two --
above all the cases where indices shift under the caller (removal), which is where
an index<->identity translation layer would have broken.

The tail of the file covers CorrelationResultOverlay, the static sibling that
carries a correlation result's reprojected markers.
"""
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] without the UI extra

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_point_overlay import (
    POINT_COLORS,
    CorrelationPointOverlay,
    CorrelationResultOverlay,
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


def _with_results(coords):
    """A canvas carrying both overlays, wired as CorrelationCanvasWidget wires them."""
    canvas, points = _attached(coords)
    results = CorrelationResultOverlay(legend_host=points)
    canvas.add_overlay(results)
    return canvas, points, results


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


# ── legend ────────────────────────────────────────────────────────────────


def test_legend_shows_one_entry_per_type_on_screen():
    """The base draws a single swatch; correlation's points are heterogeneous and
    the colour is the only thing telling a FIB point from a POI."""
    _, ov = _attached([
        _coord(10, 10, PointType.FIB),
        _coord(20, 20, PointType.POI),
        _coord(30, 30, PointType.FIB),  # a second FIB must not add a second entry
    ])
    handles, labels = ov._legend_entries()
    assert labels == ["FIB", "POI"]
    assert len(handles) == 2


def test_legend_order_is_the_type_order_not_arrival_order():
    """Iterating PointType keeps the legend stable as points come and go."""
    _, ov = _attached([_coord(10, 10, PointType.POI), _coord(20, 20, PointType.FIB)])
    _, labels = ov._legend_entries()
    assert labels == ["FIB", "POI"]  # declaration order, not [POI, FIB]


def _drawn_legend(ov):
    """Texts of the legend actually on the axes.

    Deliberately not ``_legend_entries()``: that recomputes from _coords and would
    pass even if the legend artist were never redrawn, which is the bug this test
    exists to catch (add_point/remove_point touch one artist and would otherwise
    leave a stale legend on screen).
    """
    return [] if ov._legend is None else [t.get_text() for t in ov._legend.get_texts()]


def test_legend_follows_the_points():
    a = _coord(10, 10, PointType.FIB)
    b = _coord(20, 20, PointType.SURFACE)
    _, ov = _attached([a, b])
    assert _drawn_legend(ov) == ["FIB", "SURFACE"]

    ov.remove_coordinate(b)
    assert _drawn_legend(ov) == ["FIB"]

    ov.add_coordinate(_coord(30, 30, PointType.POI))
    assert _drawn_legend(ov) == ["FIB", "POI"]


def test_legend_swatch_matches_how_the_point_is_drawn():
    """An unfilled '+' keeps its own colour as the edge; a white edge would swallow
    it and leave the legend claiming the wrong colour."""
    _, ov = _attached([_coord(10, 10, PointType.SURFACE)])
    handle = ov._legend_handle(PointType.SURFACE)
    colour = POINT_COLORS[PointType.SURFACE]
    assert handle.get_marker() == "+"
    assert handle.get_markeredgecolor() == colour

    filled = ov._legend_handle(PointType.FIB)
    assert filled.get_marker() == "o"
    assert filled.get_markeredgecolor() == "white"


def test_legend_can_be_hidden():
    _, ov = _attached([_coord(10, 10)])
    ov.set_legend_visible(False)
    assert ov._legend_entries() == ([], [])
    assert ov._legend is None
    ov.set_legend_visible(True)
    assert ov._legend is not None


# ── labels ────────────────────────────────────────────────────────────────


def _label_states(ov):
    return [a.get_visible() for a in ov._anns if a is not None]


def test_labels_hide_without_hiding_the_points():
    _, ov = _attached([_coord(10, 10), _coord(20, 20)])
    ov.set_labels_visible(False)
    assert _label_states(ov) == [False, False]
    assert all(a.get_visible() for a in ov._artists)  # markers stay


def test_a_point_added_while_labels_are_off_arrives_hidden():
    _, ov = _attached([_coord(10, 10)])
    ov.set_labels_visible(False)
    ov.add_coordinate(_coord(20, 20))
    assert _label_states(ov) == [False, False]


def test_showing_the_overlay_does_not_override_the_label_preference():
    """set_visible turns every annotation back on; a label the operator switched
    off should stay off."""
    _, ov = _attached([_coord(10, 10)])
    ov.set_labels_visible(False)
    ov.set_visible(False)
    ov.set_visible(True)
    assert _label_states(ov) == [False]


def test_labels_renumber_and_stay_hidden_after_a_removal():
    a, b, c = (_coord(0, 0), _coord(1, 1), _coord(2, 2))
    _, ov = _attached([a, b, c])
    ov.set_labels_visible(False)
    ov.remove_coordinate(a)
    assert [ov._point_label(i) for i in range(2)] == ["FIB 1", "FIB 2"]
    assert _label_states(ov) == [False, False]


# ── label legibility ──────────────────────────────────────────────────────


def test_labels_carry_a_dark_outline():
    """Every label colour is a bright saturated hue, so SURFACE and FM-SURFACE
    vanish against a bright image without a dark stroke behind them. Matches the
    old canvas, which applies the same effect at three sites."""
    import matplotlib.patheffects as pe

    _, ov = _attached([_coord(10, 10, PointType.SURFACE), _coord(20, 20)])
    for ann in ov._anns:
        assert ann is not None
        effects = ann.get_path_effects()
        assert effects, "label has no outline"
        assert any(isinstance(e, pe.withStroke) for e in effects)


def test_the_outline_is_dark_not_light():
    """A white stroke would merge into a bright background -- exactly where the
    outline has to work -- so the foreground must be dark."""
    _, ov = _attached([_coord(10, 10)])
    stroke = ov._anns[0].get_path_effects()[0]
    assert stroke._gc.get("foreground") == "black"


def test_a_point_added_later_gets_the_outline_too():
    _, ov = _attached([_coord(10, 10)])
    ov.add_coordinate(_coord(20, 20, PointType.POI))
    assert all(a.get_path_effects() for a in ov._anns if a is not None)


# ── the surface datum row ─────────────────────────────────────────────────


def _datum(ov):
    """The dashed datum line on the axes, or None."""
    lines = [ln for ln in ov._ax.get_lines() if ln.get_linestyle() == "--"]
    return lines[0] if lines else None


def test_a_fib_surface_gets_a_datum_row():
    """A single marker is a poor way to read a height off a large image; the FIB
    surface is a datum other features are read against."""
    _, ov = _attached([_coord(50, 60), _coord(100, 120, PointType.SURFACE)])
    line = _datum(ov)
    assert line is not None
    assert line.get_ydata()[0] == 120.0
    assert line.get_color() == POINT_COLORS[PointType.SURFACE]
    assert line.get_zorder() < 8  # under the markers


def test_an_fm_surface_gets_no_datum_row():
    """An FM surface is a z-plane, so no in-plane line applies to it."""
    _, ov = _attached([_coord(30, 40, PointType.SURFACE_FM)])
    assert _datum(ov) is None


def test_the_datum_row_arrives_and_leaves_with_its_point():
    surf = _coord(100, 120, PointType.SURFACE)
    _, ov = _attached([_coord(50, 60)])
    assert _datum(ov) is None

    ov.add_coordinate(surf)
    assert _datum(ov) is not None

    ov.remove_coordinate(surf)
    assert _datum(ov) is None


def test_the_datum_row_follows_an_external_edit():
    surf = _coord(100, 120, PointType.SURFACE)
    _, ov = _attached([surf])
    surf.point.y = 200.0
    ov.refresh_coordinate(surf)
    assert _datum(ov).get_ydata()[0] == 200.0


def test_selecting_the_surface_weights_its_row():
    """Selection reads as a heavier, brighter line -- the colour stays the type's,
    the same rule the markers follow."""
    surf = _coord(100, 120, PointType.SURFACE)
    other = _coord(50, 60)
    _, ov = _attached([other, surf])
    normal = (_datum(ov).get_linewidth(), _datum(ov).get_alpha())

    ov.set_selected_coordinate(surf)
    assert (_datum(ov).get_linewidth(), _datum(ov).get_alpha()) > normal

    ov.set_selected_coordinate(other)
    assert (_datum(ov).get_linewidth(), _datum(ov).get_alpha()) == normal


def test_the_datum_row_hides_with_the_overlay():
    _, ov = _attached([_coord(100, 120, PointType.SURFACE)])
    ov.set_visible(False)
    assert not _datum(ov).get_visible()
    ov.set_visible(True)
    assert _datum(ov).get_visible()


def test_a_new_image_redraws_the_datum_row():
    canvas, ov = _attached([_coord(100, 120, PointType.SURFACE)])
    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    assert _datum(ov) is not None
    assert len([ln for ln in ov._ax.get_lines() if ln.get_linestyle() == "--"]) == 1


def test_the_datum_row_keeps_up_with_a_drag():
    """The base blits only the dragged marker and its label. Without the row in
    that set it would stay at its old y until the mouse came up -- on exactly the
    point whose whole purpose is marking that y."""

    class _Event:
        xdata, ydata, x, y = 100.0, 120.0, 0.0, 0.0

    _, ov = _attached([_coord(100, 120, PointType.SURFACE)])
    ov._start_drag(0, _Event())

    line = _datum(ov)
    # animated, so the captured background does not contain it at the old y...
    assert line.get_animated()
    # ...and painted on every drag step, so it appears at the new one
    assert line in ov._blit_artists()

    ov._on_release(_Event())
    assert not line.get_animated()


# ── result markers ────────────────────────────────────────────────────────
#
# The three calls below mirror CorrelationTabWidget._overlay_result_on_fib: the
# reprojected fiducials, the uncorrected-POI ghost, and the corrected POI.


def _add_reprojected(results):
    results.add_points(
        [(70.0, 65.0), (125.0, 95.0)],
        color="#ff4444", label_prefix="E", size=4, legend_label="FM reprojected (E)",
    )


def _add_ghost(results):
    results.add_points(
        [(150.0, 150.0)],
        color="#ff00ff", size=7, alpha=0.7, show_labels=False, hollow=True,
        legend_label="POI uncorrected",
    )


def test_result_markers_never_become_pickable_points():
    """The whole reason results live on their own overlay: the interactive one
    hit-tests _points, so a marker that never enters it cannot be dragged or
    deleted by a stray click, without anything having to remember that."""
    _, points, results = _with_results([_coord(10, 10)])
    _add_reprojected(results)
    _add_ghost(results)

    assert points.get_points() == [(10.0, 10.0)]
    assert len(points.coordinates()) == 1
    assert len(results._artists) == 3  # 2 reprojected + 1 ghost, all outside _points


def test_result_groups_accumulate():
    _, _, results = _with_results([])
    _add_reprojected(results)
    _add_ghost(results)
    assert len(results._groups) == 2
    assert len(results._artists) == 3


def test_result_entries_are_appended_to_the_point_legend():
    """One legend box, not two: a second Legend would be drawn in the same corner
    and sit on top of the first. Point types read first, results after."""
    _, points, results = _with_results([_coord(10, 10, PointType.FIB)])
    _add_reprojected(results)
    _add_ghost(results)

    assert _drawn_legend(points) == ["FIB", "FM reprojected (E)", "POI uncorrected"]


def test_a_group_without_a_legend_label_adds_no_entry():
    _, points, results = _with_results([_coord(10, 10)])
    results.add_points([(20.0, 20.0)], color="#ff4444")
    assert _drawn_legend(points) == ["FIB"]


def test_clearing_the_result_takes_its_legend_entries_with_it():
    _, points, results = _with_results([_coord(10, 10)])
    _add_reprojected(results)
    assert _drawn_legend(points) == ["FIB", "FM reprojected (E)"]

    results.clear()

    assert _drawn_legend(points) == ["FIB"]
    assert results._artists == []


def test_a_new_image_discards_the_result():
    """The markers describe a correlation against the image beneath them; over a
    different one they are wrong, not stale. Matches ImagePointCanvas.set_image,
    which started clearing them after "POI (P)" was left in the legend of an image
    that had no POI (FIB-321)."""
    canvas, points, results = _with_results([_coord(10, 10)])
    _add_reprojected(results)

    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))

    assert results._groups == []
    assert results._artists == []
    assert _drawn_legend(points) == ["FIB"]  # the picked point survives, the result does not


def test_hiding_the_legend_hides_the_result_entries_too():
    _, points, results = _with_results([_coord(10, 10)])
    _add_reprojected(results)
    points.set_legend_visible(False)
    assert _drawn_legend(points) == []


def test_a_result_with_no_picked_points_still_gets_a_legend():
    """correlation_result_widget shows a saved result on a canvas with nothing
    picked. The base suppresses the legend when it holds no points of its own, so
    that rule has to key off the entries rather than the points."""
    _, points, results = _with_results([])
    _add_reprojected(results)
    assert _drawn_legend(points) == ["FM reprojected (E)"]


def test_a_hollow_swatch_matches_its_hollow_marker():
    """The ghost is a ring: filled swatch or white edge in the legend would claim
    a marker that is not on the image."""
    _, points, results = _with_results([])
    _add_ghost(results)
    (handle,), _ = results.legend_entries()
    marker = results._artists[0]

    assert str(marker.get_markerfacecolor()) == "none"
    assert str(handle.get_markerfacecolor()) == "none"
    assert handle.get_markeredgecolor() == marker.get_markeredgecolor() == "#ff00ff"
    assert handle.get_alpha() == marker.get_alpha() == pytest.approx(0.7)


def test_result_labels_are_numbered_within_their_group():
    _, _, results = _with_results([])
    _add_reprojected(results)
    results.add_points([(155.0, 152.0)], color="#ff00ff", label_prefix="P")
    assert [a.get_text() for a in results._label_artists] == ["E1", "E2", "P1"]


def test_show_labels_false_draws_no_numbers():
    """The ghost sits under the corrected POI; a number there would just collide
    with the one that matters."""
    _, _, results = _with_results([])
    _add_ghost(results)
    assert results._label_artists == []
    assert len(results._artists) == 1


def test_result_labels_follow_the_label_toggle():
    _, _, results = _with_results([])
    _add_reprojected(results)
    results.set_labels_visible(False)
    assert [a.get_visible() for a in results._label_artists] == [False, False]
    assert all(a.get_visible() for a in results._artists)  # markers stay


def test_a_group_added_while_labels_are_off_arrives_hidden():
    _, _, results = _with_results([])
    results.set_labels_visible(False)
    _add_reprojected(results)
    assert [a.get_visible() for a in results._label_artists] == [False, False]


def test_result_labels_carry_the_same_dark_outline_as_the_points():
    """Same reason: #ff4444 and #ff00ff both disappear over a blown-out region."""
    import matplotlib.patheffects as pe

    _, _, results = _with_results([])
    _add_reprojected(results)
    for ann in results._label_artists:
        effects = ann.get_path_effects()
        assert effects, "result label has no outline"
        assert any(isinstance(e, pe.withStroke) for e in effects)


def test_a_result_overlay_without_a_host_still_draws():
    """legend_host is optional -- a canvas with no point overlay (or a test) gets
    markers without needing somewhere to put a legend."""
    canvas = FibsemImageCanvas()
    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    results = CorrelationResultOverlay()
    canvas.add_overlay(results)
    _add_reprojected(results)
    assert len(results._artists) == 2
