"""PointOverlay multi-selection (FIB-1171): several points selected, moved and removed
together, with the canvas pan still working.

Events go through the canvas's own matplotlib callback registry, with a real
QMouseEvent attached for the modifiers, so the canvas pan handler and the overlay see
each press in the order they do in the app. That ordering is what decides whether a
Shift-drag box-selects or pans.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_point_overlay_multi_select.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from matplotlib.backend_bases import KeyEvent, MouseEvent
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

_app = QApplication.instance() or QApplication(sys.argv)

_POINTS = [(20.0, 20.0), (40.0, 20.0), (60.0, 60.0), (80.0, 80.0)]


def _overlay(multi_select: bool = True):
    canvas = FibsemImageCanvas()
    canvas.resize(300, 300)
    overlay = PointOverlay(multi_select=multi_select)
    canvas.add_overlay(overlay)
    canvas.set_array(np.zeros((100, 100), np.uint8))
    canvas.show()
    _app.processEvents()
    overlay.set_points(_POINTS)
    canvas.draw()
    _app.processEvents()
    return canvas, overlay


class _Recorder:
    """Every signal the overlay emits, in order."""

    def __init__(self, overlay: PointOverlay):
        self.events = []
        overlay.selection_changed.connect(lambda i: self.events.append(("sel", i)))
        overlay.point_moved.connect(lambda i, x, y: self.events.append(("moved", i)))
        overlay.points_moved.connect(lambda i: self.events.append(("moved_many", i)))
        overlay.point_removed.connect(lambda i: self.events.append(("removed", i)))
        overlay.points_removed.connect(
            lambda i: self.events.append(("removed_many", i))
        )

    def of(self, kind):
        return [payload for k, payload in self.events if k == kind]


_QT_MODS = {"Shift": Qt.ShiftModifier, "Control": Qt.ControlModifier}


def _send(canvas, overlay, name, x, y, mods=(), button=1):
    sx, sy = overlay._ax.transData.transform((x, y))
    flags = Qt.NoModifier
    for m in mods:
        flags |= _QT_MODS[m]
    qtype = {
        "button_press_event": QEvent.MouseButtonPress,
        "button_release_event": QEvent.MouseButtonRelease,
        "motion_notify_event": QEvent.MouseMove,
    }[name]
    gui = QMouseEvent(qtype, QPointF(0, 0), Qt.LeftButton, Qt.LeftButton, flags)
    btn = None if name == "motion_notify_event" else button
    event = MouseEvent(name, canvas, sx, sy, button=btn, guiEvent=gui)
    canvas.callbacks.process(name, event)


def _click(canvas, overlay, x, y, mods=()):
    _send(canvas, overlay, "button_press_event", x, y, mods)
    _send(canvas, overlay, "button_release_event", x, y, mods)


def _drag(canvas, overlay, start, end, mods=()):
    _send(canvas, overlay, "button_press_event", *start, mods)
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    _send(canvas, overlay, "motion_notify_event", *mid, mods)
    _send(canvas, overlay, "motion_notify_event", *end, mods)
    _send(canvas, overlay, "button_release_event", *end, mods)


def _key(canvas, key):
    canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", canvas, key)
    )


def test_control_click_toggles_and_shift_click_adds():
    canvas, overlay = _overlay()
    rec = _Recorder(overlay)

    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[2], mods=("Control",))
    _click(canvas, overlay, *_POINTS[3], mods=("Shift",))
    assert overlay.selected_indices() == [0, 2, 3]

    _click(canvas, overlay, *_POINTS[2], mods=("Control",))
    assert overlay.selected_indices() == [0, 3]
    assert rec.of("sel") == [[0], [0, 2], [0, 2, 3], [0, 3]]
    # the selected markers are drawn as selected, the rest are not
    assert [overlay._artists[i].get_markersize() > overlay._size for i in range(4)] == [
        True,
        False,
        False,
        True,
    ]


def test_modifier_click_does_not_drag_the_point():
    canvas, overlay = _overlay()
    _drag(canvas, overlay, _POINTS[1], (50.0, 50.0), mods=("Shift",))
    assert overlay.get_points()[1] == _POINTS[1]
    assert overlay.selected_indices() == [1]


def test_without_multi_select_modifiers_change_nothing():
    canvas, overlay = _overlay(multi_select=False)
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[2], mods=("Control",))
    assert overlay.selected_indices() == [2]
    xlim = overlay._ax.get_xlim()
    _drag(canvas, overlay, (10.0, 90.0), (30.0, 70.0), mods=("Shift",))
    assert overlay._ax.get_xlim() != xlim, "a Shift-drag on empty area still pans"


def test_shift_drag_on_empty_area_box_selects_and_does_not_pan():
    canvas, overlay = _overlay()
    rec = _Recorder(overlay)
    _click(canvas, overlay, *_POINTS[3])
    xlim, ylim = overlay._ax.get_xlim(), overlay._ax.get_ylim()

    _drag(canvas, overlay, (10.0, 10.0), (50.0, 30.0), mods=("Shift",))

    assert overlay.selected_indices() == [3, 0, 1], "the box adds to the selection"
    assert (overlay._ax.get_xlim(), overlay._ax.get_ylim()) == (xlim, ylim)
    assert overlay._box_artist is None, "the rubber band is gone after release"
    assert rec.of("sel")[-1] == [3, 0, 1]


def test_plain_drag_on_empty_area_pans_and_keeps_the_selection():
    canvas, overlay = _overlay()
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[1], mods=("Shift",))
    xlim = overlay._ax.get_xlim()

    _drag(canvas, overlay, (10.0, 90.0), (30.0, 70.0))
    assert overlay._ax.get_xlim() != xlim, "a plain drag pans"
    assert overlay.selected_indices() == [0, 1], "panning must not drop the selection"

    _drag(canvas, overlay, (10.0, 90.0), (30.0, 70.0), mods=("Control",))
    assert overlay.selected_indices() == [0, 1], "a Ctrl-drag pans too"

    _click(canvas, overlay, 10.0, 90.0)
    assert overlay.selected_indices() == [], "a click on empty area clears"


def test_dragging_a_selected_point_moves_the_group_rigidly():
    canvas, overlay = _overlay()
    rec = _Recorder(overlay)
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[1], mods=("Shift",))

    _drag(canvas, overlay, _POINTS[1], (45.0, 30.0))

    pts = overlay.get_points()
    assert pts[0] == pytest.approx((25.0, 30.0))
    assert pts[1] == pytest.approx((45.0, 30.0))
    assert pts[2:] == _POINTS[2:], "unselected points stay put"
    assert rec.of("moved_many") == [[0, 1]], "one signal for the group"
    assert rec.of("moved") == []
    assert overlay.selected_indices() == [0, 1], "the group stays selected"


def test_group_drag_stops_at_the_edge_without_squashing():
    canvas, overlay = _overlay()
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[1], mods=("Shift",))

    # 35 px left would put the leading point at x = -15; the group may only go 20
    _drag(canvas, overlay, _POINTS[1], (5.0, 20.0))

    (x0, _), (x1, _) = overlay.get_points()[:2]
    assert x0 == pytest.approx(0.0), "the leading point stops at the edge"
    assert x1 - x0 == pytest.approx(20.0), "the spacing is kept"


def test_single_point_drag_still_reports_point_moved():
    canvas, overlay = _overlay()
    rec = _Recorder(overlay)
    _drag(canvas, overlay, _POINTS[2], (65.0, 65.0))
    assert rec.of("moved") == [2]
    assert rec.of("moved_many") == []


def test_click_without_moving_on_a_group_member_narrows_to_it():
    canvas, overlay = _overlay()
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[1], mods=("Shift",))
    _click(canvas, overlay, *_POINTS[2], mods=("Shift",))

    _click(canvas, overlay, *_POINTS[1])
    assert overlay.selected_indices() == [1]


def test_delete_removes_every_selected_point_in_one_signal():
    canvas, overlay = _overlay()
    canvas.enter_overlay_mode(overlay, "Points")
    rec = _Recorder(overlay)
    _click(canvas, overlay, *_POINTS[0])
    _click(canvas, overlay, *_POINTS[2], mods=("Shift",))

    _key(canvas, "delete")

    assert overlay.get_points() == [_POINTS[1], _POINTS[3]]
    assert len(overlay._artists) == 2
    assert rec.of("removed_many") == [[0, 2]]
    assert rec.of("removed") == []
    assert overlay.selected_indices() == []
    assert rec.of("sel")[-1] == []


def test_removing_a_point_shifts_the_selection_after_it():
    canvas, overlay = _overlay()
    overlay.set_selection([1, 3])
    overlay.remove_point(0)
    assert overlay.selected_indices() == [0, 2]
    overlay.remove_points([0, 1])
    assert overlay.selected_indices() == [0]
    assert overlay.get_points() == [_POINTS[3]]


def test_programmatic_selection_is_silent():
    canvas, overlay = _overlay()
    rec = _Recorder(overlay)
    overlay.set_selection([2, 0, 9])  # out of range dropped, order kept
    assert overlay.selected_indices() == [2, 0]
    assert overlay._selected == 0, "the last selected is the current point"
    overlay.set_selected(None)
    assert overlay.selected_indices() == []
    assert rec.events == []
