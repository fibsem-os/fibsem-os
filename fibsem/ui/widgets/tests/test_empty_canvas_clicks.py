"""Headless smoke: an empty (no-image) canvas must not emit semantic click signals.

Guards the fix for the placeholder-removal side effect — before an image is loaded the axes
span the default [0,1], so a double-click would otherwise emit ~(0.5, 0.5) "pixels" and drive
a stage move to the image corner. Once an image is present, clicks emit normally.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_empty_canvas_clicks.py
"""
import sys
import types

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _event(canvas, x, y, *, dblclick=False, button=1):
    return types.SimpleNamespace(
        inaxes=canvas._ax, xdata=x, ydata=y, dblclick=dblclick, button=button,
        guiEvent=None, x=10, y=10,
    )


def test_no_double_click_emit_on_empty_canvas():
    c = FibsemImageCanvas()  # no image -> _img_w is None
    seen = []
    c.canvas_double_clicked.connect(lambda x, y, m: seen.append((x, y)))
    c._on_press(_event(c, 0.5, 0.5, dblclick=True))
    assert seen == [], f"empty canvas emitted a double-click: {seen}"


def test_no_right_click_emit_on_empty_canvas():
    c = FibsemImageCanvas()
    seen = []
    c.canvas_right_clicked.connect(lambda x, y, m: seen.append((x, y)))
    c._on_press(_event(c, 0.5, 0.5, button=3))
    assert seen == [], f"empty canvas emitted a right-click: {seen}"


def test_double_click_emits_with_image():
    c = FibsemImageCanvas()
    c.set_array(np.zeros((64, 64), dtype=np.uint8))
    seen = []
    c.canvas_double_clicked.connect(lambda x, y, m: seen.append((x, y)))
    c._on_press(_event(c, 10.0, 12.0, dblclick=True))
    assert seen == [(10.0, 12.0)], f"expected one double-click, got {seen}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
