"""CorrelationPicking: one picking implementation, any FibsemImageCanvas (FIB-535).

The behaviour itself is covered by the point-overlay and canvas-widget suites.
What is pinned here is the property the extraction exists for — that picking is
not welded to one host — because that is the claim the FM side depends on and
the one a future refactor could quietly break.
"""
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] without the UI extra

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.correlation.widgets.correlation_picking import CorrelationPicking
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)

# Hosts own their canvases; letting them fall out of scope deletes the C++ side
# while the test still holds the Python wrapper.
_ALIVE = []


def _bare_canvas():
    c = FibsemImageCanvas()
    c.set_array(np.zeros((64, 64), dtype=np.uint8))
    _ALIVE.append(c)
    return c


def _fm_canvas():
    """The canvas the FM side will pick on: FMCanvasWidget's, complete with its
    own z-slider, layers popover and MIP toolbar button."""
    w = FMCanvasWidget()
    _ALIVE.append(w)
    return w.canvas


def _coord(x, y, pt=PointType.FIB):
    return Coordinate(PointXYZ(x, y, 0.0), pt)


# ── the reason it was extracted ───────────────────────────────────────────


@pytest.mark.parametrize("make_canvas", [_bare_canvas, _fm_canvas],
                         ids=["FibsemImageCanvas", "FMCanvasWidget.canvas"])
def test_picking_attaches_to_any_shared_canvas(make_canvas):
    """The FIB side puts this on a bare canvas, the FM side on FMCanvasWidget's.
    If it only worked on one, the FM display would need its own copy of the
    picking plumbing — which is exactly the duplication being removed."""
    canvas = make_canvas()
    picking = CorrelationPicking(canvas, menu_parent=QWidget())

    picking.set_coordinates([_coord(10, 10), _coord(20, 20, PointType.POI)])

    assert picking.points.get_points() == [(10.0, 10.0), (20.0, 20.0)]
    assert picking.points in canvas._overlays
    assert picking.results in canvas._overlays


def test_result_markers_reach_the_result_overlay_on_any_canvas():
    picking = CorrelationPicking(_fm_canvas(), menu_parent=QWidget())
    picking.set_coordinates([_coord(10, 10)])

    picking.add_overlay_points([(20.0, 20.0), (30.0, 30.0)], color="#ff4444")

    assert len(picking.results._artists) == 2
    assert picking.points.get_points() == [(10.0, 10.0)]  # untouched


# ── ownership ─────────────────────────────────────────────────────────────


def test_dispose_unregisters_both_overlays():
    canvas = _bare_canvas()
    picking = CorrelationPicking(canvas, menu_parent=QWidget())
    assert len(canvas._overlays) == 2

    picking.dispose()

    assert canvas._overlays == []
    assert picking.points._ax is None  # detached, not merely dropped


def test_a_second_attachment_without_dispose_stacks_overlays():
    """Why dispose() exists. The canvas owns `_overlays`, not picking, so simply
    dropping a CorrelationPicking leaves its overlays drawing — two sets of
    markers, one of them answering to nothing."""
    canvas = _bare_canvas()
    CorrelationPicking(canvas, menu_parent=QWidget())

    CorrelationPicking(canvas, menu_parent=QWidget())

    assert len(canvas._overlays) == 4  # the trap
    canvas.clear_overlays()


# ── the host's side of the contract ───────────────────────────────────────


def test_the_widget_forwards_the_four_signals():
    """Forwarded rather than left on `.picking`, so `correlation_tab_widget` and
    its _CanvasAdapter connect to this exactly as they do to ImagePointCanvas."""
    w = CorrelationCanvasWidget()
    w.set_image(np.zeros((64, 64), dtype=np.uint8))
    a = _coord(10, 10)
    w.set_coordinates([a])
    seen = {"sel": [], "moved": [], "removed": []}
    w.point_selected.connect(seen["sel"].append)
    w.point_moved.connect(seen["moved"].append)
    w.point_removed.connect(seen["removed"].append)

    w.points.coordinate_selected.emit(a)
    w.points.coordinate_moved.emit(a)
    w.points.remove_coordinate(a)

    assert seen["sel"] == [a] and seen["moved"] == [a] and seen["removed"] == [a]


def test_the_correlation_canvas_hides_the_crosshair():
    """The shared canvas defaults it on; correlation draws SURFACE as an orange
    "+" that a yellow centre "+" is easy to confuse with — and it would land in a
    saved plot. The toolbar toggle still brings it back."""
    w = CorrelationCanvasWidget()
    w.set_image(np.zeros((64, 64), dtype=np.uint8))

    assert w.canvas._crosshair_artists == []
    assert w.canvas.btn_toggle_crosshair.isChecked() is False
    assert w.canvas.btn_toggle_crosshair.toolTip() == "Show crosshair"

    w.canvas.toggle_crosshair()
    assert w.canvas._crosshair_artists  # still available on demand
