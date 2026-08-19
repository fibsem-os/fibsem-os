"""Headless smoke: the overlay content-rect protocol.

Overlays used to be told ``on_image_changed(width, height)`` and assumed content
started at (0, 0). They are now told a ContentRect, so the same overlay works whether
the canvas holds one image at the origin or many placed away from it.

Two things are easy to get wrong and are pinned here: the rect is *not* the canvas's
matplotlib extent (that carries imshow's half-pixel offset and would shift every clamp
and anchor by half a pixel), and an overlay written against the old protocol must keep
receiving updates.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_overlay_content_rect.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.canvas_base import EMPTY_CONTENT, ContentRect
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay
from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay

_app = QApplication.instance() or QApplication(sys.argv)


def _img(h, w):
    return np.zeros((h, w), dtype=np.uint8)


class _Recorder(CanvasOverlay):
    """Overlay on the current protocol."""

    def __init__(self):
        self.rects = []

    def on_content_changed(self, rect):
        self.rects.append(rect)


class _LegacyRecorder(CanvasOverlay):
    """Overlay written against the superseded protocol — must still work."""

    def __init__(self):
        self.sizes = []

    def on_image_changed(self, width, height):
        self.sizes.append((width, height))


class _DuckLegacy:
    """Legacy overlay that doesn't even subclass CanvasOverlay (pure duck-typing)."""

    def __init__(self):
        self.sizes = []

    def attach(self, ax, canvas):
        pass

    def detach(self):
        pass

    def on_image_changed(self, width, height):
        self.sizes.append((width, height))


# ── the rect itself ───────────────────────────────────────────────────────


def test_rect_derives_edges_and_centre():
    r = ContentRect(10.0, 20.0, 100.0, 50.0)
    assert (r.x1, r.y1) == (110.0, 70.0)
    assert (r.cx, r.cy) == (60.0, 45.0)
    assert not r.is_empty


def test_a_zero_sized_rect_is_empty():
    assert EMPTY_CONTENT.is_empty
    assert ContentRect(0.0, 0.0, 10.0, 0.0).is_empty


# ── what the image canvas reports ─────────────────────────────────────────


def test_image_content_rect_is_pixel_space_not_the_mpl_extent():
    """The distinction that keeps this refactor behaviour-preserving.

    The image's matplotlib extent is (-0.5, w-0.5, h-0.5, -0.5); handing that to
    overlays would move every clamp and anchor by half a pixel. The rect they get is
    the pixel grid: origin (0, 0), size (w, h).
    """
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    rect = c._content_rect()
    assert (rect.x0, rect.y0, rect.width, rect.height) == (0.0, 0.0, 64, 32)
    assert tuple(c._content_extent()) == (-0.5, 63.5, 31.5, -0.5)  # deliberately differs
    assert (rect.cx, rect.cy) == (32.0, 16.0)  # what an overlay centres on


def test_content_rect_is_empty_before_any_image():
    assert FibsemImageCanvas()._content_rect().is_empty


# ── notification ──────────────────────────────────────────────────────────


def test_overlays_are_told_the_rect_on_a_new_image():
    c = FibsemImageCanvas()
    rec = _Recorder()
    c.add_overlay(rec)
    c.set_array(_img(32, 64))
    assert rec.rects[-1] == ContentRect(0.0, 0.0, 64, 32)


def test_adding_an_overlay_to_a_live_canvas_tells_it_immediately():
    """Otherwise an overlay added after the image draws nothing until the next frame."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    rec = _Recorder()
    c.add_overlay(rec)
    assert rec.rects == [ContentRect(0.0, 0.0, 64, 32)]


def test_clearing_the_canvas_reports_empty_content():
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    rec = _Recorder()
    c.add_overlay(rec)
    c.clear()
    assert rec.rects[-1].is_empty


def test_a_failing_overlay_does_not_break_the_others():
    class _Boom(CanvasOverlay):
        def on_content_changed(self, rect):
            raise RuntimeError("boom")

    c = FibsemImageCanvas()
    rec = _Recorder()
    c.add_overlay(_Boom())
    c.add_overlay(rec)
    c.set_array(_img(32, 64))  # must not raise
    assert rec.rects  # the survivor still got told


# ── back-compat ───────────────────────────────────────────────────────────


def test_a_legacy_overlay_still_receives_updates():
    c = FibsemImageCanvas()
    legacy = _LegacyRecorder()
    c.add_overlay(legacy)
    c.set_array(_img(32, 64))
    assert legacy.sizes[-1] == (64, 32)


def test_a_duck_typed_legacy_overlay_still_receives_updates():
    """It subclasses nothing, so the base-class shim can't help — the canvas falls back."""
    c = FibsemImageCanvas()
    legacy = _DuckLegacy()
    c.add_overlay(legacy)
    c.set_array(_img(32, 64))
    assert legacy.sizes[-1] == (64, 32)


# ── overlays honour a non-zero origin ─────────────────────────────────────


def test_the_ruler_seeds_itself_on_the_content_centre_wherever_that_is():
    """The payoff: an overlay anchoring to the content works off-origin, which is what
    a canvas placing images in stage space will hand it."""
    ruler = RulerOverlay()
    ruler.on_content_changed(ContentRect(1000.0, 2000.0, 400.0, 200.0))
    ruler._seed_default()
    (x1, y1), (x2, y2) = ruler._p1, ruler._p2
    assert (x1 + x2) / 2 == 1200.0  # centred on the rect, not on (0, 0)
    assert y1 == y2 == 2100.0
    assert x2 - x1 == 100.0  # a quarter of the content width, as before


def test_the_ruler_clamps_to_the_content_bounds_not_to_zero():
    ruler = RulerOverlay()
    ruler.on_content_changed(ContentRect(1000.0, 2000.0, 400.0, 200.0))
    ruler._p1 = [-50.0, -50.0]  # dragged far outside, below the origin
    ruler._p2 = [9999.0, 9999.0]
    ruler._clamp()
    assert ruler._p1 == [1000.0, 2000.0]  # pinned to the rect's top-left
    assert ruler._p2 == [1400.0, 2200.0]  # and its bottom-right


# ── an overlay is told what it was added to ──────────────────────────────


def test_an_overlay_added_to_a_real_space_canvas_is_told_the_content():
    """`add_overlay` used to gate this on `_img_w`, which is a *single image* canvas's
    notion of content and is never set by one placing images in stage space.

    So an overlay added to a real-space canvas was never told what it had been added to.
    Harmless for one its host redraws by hand -- a tile grid -- and fatal for one that
    builds its artists on the first content report: a `RectOverlay` added this way drew
    nothing at all, with no error.
    """
    from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

    canvas = FibsemRealSpaceCanvas()
    canvas.add_image(_img(64, 64), centre=(0.0, 0.0), pixel_size=1e-6)
    overlay = _Recorder()

    canvas.add_overlay(overlay)

    assert overlay.rects, "the overlay was never told the canvas had content"
    assert not overlay.rects[-1].is_empty


def test_an_overlay_added_to_an_empty_canvas_is_told_it_is_empty():
    """Told, rather than not told. Every overlay guards on `rect.is_empty`, so the
    report costs nothing and the alternative is a silent difference between the two
    canvases."""
    canvas = FibsemImageCanvas()
    overlay = _Recorder()

    canvas.add_overlay(overlay)

    assert overlay.rects == [EMPTY_CONTENT]


def test_a_rectangle_overlay_on_a_real_space_canvas_actually_draws():
    """The regression in the shape it was found in."""
    from fibsem.ui.widgets.canvas.overlays.rect_overlay import RectOverlay
    from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

    canvas = FibsemRealSpaceCanvas()
    canvas.add_image(_img(64, 64), centre=(0.0, 0.0), pixel_size=1e-6)
    overlay = RectOverlay()

    canvas.add_overlay(overlay)

    assert overlay._patch is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
