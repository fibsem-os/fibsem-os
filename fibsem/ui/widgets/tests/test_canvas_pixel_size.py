"""Headless smoke: FibsemImageCanvas.pixel_size — the public metres/px accessor.

The scale used to be reachable only as the private ``_pixel_size``, so callers that own
the scale independently of the pixel data poked the attribute and then called the private
``_refresh_scalebar()`` by hand. The property is the supported route for both directions;
these tests pin the parts that would break silently — that assigning it actually rebuilds
the scalebar, and that its None semantics differ from set_array's argument.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_pixel_size.py
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _canvas(pixel_size=1.0e-7):
    c = FibsemImageCanvas()
    c.set_array(np.zeros((64, 64)), pixel_size=pixel_size)
    return c


def test_reads_back_what_set_array_was_given():
    assert _canvas(2.5e-8).pixel_size == 2.5e-8


def test_assigning_rebuilds_the_scalebar():
    """The old hand-rolled sequence was `_pixel_size = x; _refresh_scalebar()`. A setter
    that skipped the refresh would leave a scalebar drawn at the previous scale."""
    c = _canvas()
    before = c._scalebar_artist
    assert before is not None, "no scalebar to begin with"

    c.pixel_size = 5.0e-7
    assert c.pixel_size == 5.0e-7
    assert c._scalebar_artist is not None
    assert c._scalebar_artist is not before, "scalebar not rebuilt at the new scale"
    assert c._scalebar_artist.dx == 5.0e-7


def test_assigning_none_drops_the_scalebar():
    """Unlike the *pixel_size* argument of set_array / update_display, where None means
    "leave unchanged", assigning None is an explicit "no longer known"."""
    c = _canvas()
    c.pixel_size = None
    assert c.pixel_size is None
    assert c._scalebar_artist is None

    c.set_array(np.zeros((64, 64)), pixel_size=None)  # argument form: leaves it alone
    assert c.pixel_size is None


def test_set_array_none_argument_keeps_the_existing_scale():
    c = _canvas(3.0e-8)
    c.set_array(np.zeros((32, 32)), pixel_size=None)
    assert c.pixel_size == 3.0e-8


def test_fm_canvas_widget_pushes_its_scale_through_the_property():
    """FMCanvasWidget composites its own RGB and owns the scale separately from the
    frame, so it sets the canvas scale directly rather than via set_array."""
    fm = FMCanvasWidget()
    fm.set_channel("DAPI", np.zeros((64, 64), dtype=np.uint16), "blue")
    fm.set_pixel_size(1.0e-7)
    assert fm.canvas.pixel_size == 1.0e-7
    assert fm.canvas._scalebar_artist is not None


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
