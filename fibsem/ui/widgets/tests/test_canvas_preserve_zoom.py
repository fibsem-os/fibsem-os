"""Headless smoke: FibsemImageCanvas keeps zoom/pan across same-shape image updates.

Live acquisition pushes a new frame every tick via set_array; re-framing each time is the
bug this guards against. Auto-fit still fires on the first image, on a resolution change,
and on the explicit reset_view() (fit-to-view) button.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_preserve_zoom.py
"""
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _img(h, w):
    return np.zeros((h, w), dtype=np.uint8)


def _zoom(c, x0, x1, y0, y1):
    c._ax.set_xlim(x0, x1)
    c._ax.set_ylim(y0, y1)  # y inverted (origin upper): pass (bottom, top)


def test_same_shape_preserves_view():
    c = FibsemImageCanvas()
    c.set_array(_img(64, 64))
    _zoom(c, 10, 30, 30, 10)          # user zooms in
    c.set_array(_img(64, 64))         # next live frame, same resolution
    assert c._ax.get_xlim() == (10, 30)
    assert c._ax.get_ylim() == (30, 10)


def test_first_image_fits():
    c = FibsemImageCanvas()
    c.set_array(_img(64, 64))
    x0, x1 = c._ax.get_xlim()
    assert x0 < 0 and x1 > 63          # framed to the image extent, not a stale view


def test_resolution_change_refits():
    c = FibsemImageCanvas()
    c.set_array(_img(64, 64))
    _zoom(c, 10, 30, 30, 10)
    c.set_array(_img(128, 128))        # different resolution -> refit
    x0, x1 = c._ax.get_xlim()
    assert not (x0 == 10 and x1 == 30)
    assert x1 > 100                    # framed to the larger image


def test_reset_view_always_fits():
    c = FibsemImageCanvas()
    c.set_array(_img(64, 64))
    _zoom(c, 10, 30, 30, 10)
    c.reset_view()                     # fit-to-view button -> refit despite same shape
    x0, x1 = c._ax.get_xlim()
    assert x0 < 0 and x1 > 63


def test_update_display_also_preserves():
    # the fast same-shape swap path must not re-frame either
    c = FibsemImageCanvas()
    c.set_array(_img(64, 64))
    _zoom(c, 5, 25, 25, 5)
    c.update_display(_img(64, 64))
    assert c._ax.get_xlim() == (5, 25)


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
