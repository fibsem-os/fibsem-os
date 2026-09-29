"""A point overlay finishes its own bookkeeping before it announces anything.

FIB-958 was one instance of a general trap: emit a signal, and a synchronously
connected listener may rebuild the overlay (set_points) before control comes back.
Anything the emitting method still does with indices or artists afterwards then
runs against the rebuilt lists. remove_point was fixed there; these pin the two
remaining emit-before-mutate sites (FIB-972) so a listener that rebuilds on
point_selected or point_dragging cannot leave a drag pointing at a dead artist.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_point_overlay_emit_order.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

_app = QApplication.instance() or QApplication(sys.argv)


def _overlay():
    canvas = FibsemImageCanvas()
    canvas.resize(300, 300)
    overlay = PointOverlay()
    canvas.add_overlay(overlay)
    canvas.set_array(np.zeros((100, 100), np.uint8))
    canvas.show()
    _app.processEvents()
    overlay.set_points([(20.0, 20.0), (60.0, 60.0)])
    canvas.draw()
    _app.processEvents()
    return canvas, overlay


def _event(overlay, x: float, y: float, button=1):
    sx, sy = overlay._ax.transData.transform((x, y))
    return SimpleNamespace(
        button=button,
        x=sx,
        y=sy,
        xdata=x,
        ydata=y,
        inaxes=overlay._ax,
        key=None,
        dblclick=False,
    )


def _press_on(overlay, idx: int):
    x, y = overlay._points[idx]
    overlay._on_press(_event(overlay, x, y))


def test_drag_state_is_established_before_point_selected_is_emitted():
    canvas, overlay = _overlay()
    seen = []

    def on_selected(idx, x, y):
        # What a model-owning listener sees at the moment of the emit. If the drag
        # had not started yet, a rebuild here would leave it holding dead artists.
        seen.append((idx, overlay._drag_idx, overlay._blit_bg is not None))
        overlay.set_points(overlay.get_points())

    overlay.point_selected.connect(on_selected)
    _press_on(overlay, 1)

    assert seen == [(1, 1, True)]
    canvas.close()


def test_the_dragged_frame_is_blitted_before_point_dragging_is_emitted():
    canvas, overlay = _overlay()
    _press_on(overlay, 1)
    order = []
    real_blit = overlay._blit
    overlay._blit = lambda: (order.append("blit"), real_blit())[1]

    def on_dragging(idx, x, y):
        order.append("emit")
        overlay.set_points(overlay.get_points())

    overlay.point_dragging.connect(on_dragging)
    overlay._on_motion(_event(overlay, 70.0, 75.0))

    assert order == ["blit", "emit"]
    assert overlay.get_points()[1] == (70.0, 75.0)
    canvas.close()
