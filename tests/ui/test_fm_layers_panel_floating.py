"""The FM channels panel can be moved, closes with its host, and reopens where it was left.

FIB-961: the per-channel contrast/gamma popover opened over the FM image with no way to
move it. FIB-962: it is a top-level tool window, so it outlived the correlation dialog it
was opened from and stayed on top of whatever came next.

Now the header is a drag handle with a close button, the dragged-to position is kept as an
offset from the canvas's top-right corner (class-level, so a fresh FM canvas for the next
site reopens the panel in the same place), and hiding the host widget hides the panel.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_fm_layers_panel_floating.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.fm_canvas import (
    FMCanvasWidget,
    FMRealSpaceCanvasWidget,
    _clamp_to_screen,
)

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _forget_panel_position():
    FMCanvasWidget._panel_offset = None
    yield
    FMCanvasWidget._panel_offset = None


def _widget():
    w = FMRealSpaceCanvasWidget()
    w.resize(600, 500)
    w.show()
    _app.processEvents()
    stack = (np.random.default_rng(0).random((3, 64, 64)) * 1000).astype(np.uint16)
    w._stacks = {"GFP": stack}
    w._upsert_layer("GFP", stack.max(axis=0), "green")
    return w


def _open_panel(w):
    w._btn_layers.setChecked(True)
    w._toggle_layers_panel()
    _app.processEvents()
    assert w._panel.isVisible()


def _move(widget, pos: QPoint):
    # QTest.mouseMove only warps the cursor on the offscreen platform; the drag
    # handler needs the actual move event, so deliver one by hand.
    ev = QMouseEvent(
        QEvent.MouseMove,
        pos,
        widget.mapToGlobal(pos),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    _app.sendEvent(widget, ev)


def _drag(widget, start: QPoint, dx: int, dy: int):
    end = start + QPoint(dx, dy)
    QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, start)
    _move(widget, end)
    QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, end)
    _app.processEvents()


def _drag_header(w, dx: int, dy: int):
    _drag(w._panel, w._panel._header.geometry().center(), dx, dy)


def test_default_position_is_canvas_top_right():
    w = _widget()
    _open_panel(w)
    anchor = w._panel_anchor()
    assert w._panel.pos().x() == anchor.x() - w._panel.width()
    assert w._panel.pos().y() == anchor.y()
    assert FMCanvasWidget._panel_offset is None
    w.close()


def test_drag_moves_the_panel_and_records_the_offset():
    w = _widget()
    _open_panel(w)
    before = w._panel.pos()

    _drag_header(w, -40, 30)

    after = w._panel.pos()
    assert after == before + QPoint(-40, 30)
    assert FMCanvasWidget._panel_offset == after - w._panel_anchor()
    w.close()


def test_drag_below_the_header_does_not_move_the_panel():
    w = _widget()
    _open_panel(w)
    before = w._panel.pos()
    panel = w._panel
    body = QPoint(panel.width() // 2, panel.height() - 20)  # over the reset button
    _drag(panel, body, 30, 30)
    assert w._panel.pos() == before
    assert FMCanvasWidget._panel_offset is None
    w.close()


def test_a_fresh_widget_reopens_the_panel_where_it_was_left():
    first = _widget()
    _open_panel(first)
    _drag_header(first, -50, 60)
    offset = FMCanvasWidget._panel_offset
    first.close()

    second = _widget()  # the next correlation site builds a new FM canvas
    _open_panel(second)
    assert second._panel.pos() == second._panel_anchor() + offset
    second.close()


def test_close_button_hides_the_panel_and_unchecks_the_button():
    w = _widget()
    _open_panel(w)
    w._panel._btn_close.click()
    _app.processEvents()
    assert not w._panel.isVisible()
    assert not w._btn_layers.isChecked()
    w.close()


def test_hiding_the_host_hides_the_panel():
    w = _widget()
    _open_panel(w)
    w.hide()  # the correlation dialog closing for this site
    _app.processEvents()
    assert not w._panel.isVisible()
    assert not w._btn_layers.isChecked()
    w.close()


def test_clamp_keeps_a_remembered_position_on_screen():
    screen = _app.primaryScreen().availableGeometry()
    size = QSize(268, 500)
    near = screen.center()
    far = QPoint(screen.right() + 400, screen.bottom() + 400)
    kept = _clamp_to_screen(far, size, near)
    assert QRect(kept, size).intersected(screen) == QRect(kept, size)
    inside = QPoint(screen.left() + 10, screen.top() + 10)
    assert _clamp_to_screen(inside, size, near) == inside
