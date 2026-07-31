"""FM channel controls composited onto a real-space canvas.

The split between FM and everything else is in the *controls*, not the display: the
canvas takes any picture with a position, and this widget owns only the one key it
composites its channels into. So FIB/SEM images can share the view and register with the
FM composite in stage space — which is the point of a real-space canvas.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python tests/ui/test_fm_real_space_canvas.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget, FMRealSpaceCanvasWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

_app = QApplication.instance() or QApplication(sys.argv)

PIXEL_SIZE = 1e-7


def _plane(h=128, w=128):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def _widget(pixel_size=PIXEL_SIZE):
    w = FMRealSpaceCanvasWidget()
    w.set_pixel_size(pixel_size)
    return w


# ── it is the same widget above the display ───────────────────────────────


def test_it_composites_onto_a_real_space_canvas():
    assert isinstance(_widget().canvas, FibsemRealSpaceCanvas)


def test_the_plain_fm_widget_is_untouched():
    """The seam is a subclass, so the existing FM canvas must be unaffected."""
    assert isinstance(FMCanvasWidget().canvas, FibsemImageCanvas)


def test_the_composite_is_placed_at_its_scale():
    w = _widget()
    w.set_channel("GFP", _plane(128, 256), "#00ff00")
    rect = w.canvas._content_rect()
    assert (rect.width, rect.height) == (256, 128)
    assert (rect.cx, rect.cy) == (0.0, 0.0)


def test_the_composite_is_placed_where_it_was_acquired():
    w = _widget()
    w.set_placement((50e-6, -20e-6))
    w.set_channel("GFP", _plane(64, 64), "#00ff00")
    rect = w.canvas._content_rect()
    assert (rect.cx, rect.cy) == pytest.approx((500.0, -200.0))


def test_a_same_shaped_frame_updates_in_place():
    """Re-placing would rebuild the artist, moving it to the top of the draw order —
    so a live frame would climb over anything deliberately drawn above it."""
    w = _widget()
    w.set_channel("GFP", _plane(), "#00ff00")
    artist = w.canvas._placed[w.COMPOSITE_KEY].artist
    w.set_channel("GFP", _plane(), "#00ff00")
    assert w.canvas._placed[w.COMPOSITE_KEY].artist is artist


def test_a_reshaped_frame_is_replaced():
    w = _widget()
    w.set_channel("GFP", _plane(64, 64), "#00ff00")
    w.set_channel("GFP", _plane(128, 128), "#00ff00")
    rect = w.canvas._content_rect()
    assert (rect.width, rect.height) == (128, 128)
    assert len(w.canvas.placed_keys) == 1  # replaced, not stacked


def test_nothing_is_placed_before_a_pixel_size_is_known():
    """There is nothing to scale against, and guessing would place it at the wrong size."""
    w = FMRealSpaceCanvasWidget()  # no set_pixel_size
    w.set_channel("GFP", _plane(), "#00ff00")
    assert w.canvas.placed_keys == []


# ── the canvas is not FM-only ─────────────────────────────────────────────


def test_greyscale_images_share_the_canvas_with_the_composite():
    """The reason there is no FM/FIBSEM split at the display layer."""
    w = _widget()
    w.set_channel("GFP", _plane(), "#00ff00")
    w.canvas.add_image(_plane(256, 256), centre=(40e-6, 0.0), pixel_size=2e-7, key="fib")
    w.canvas.add_image(_plane(512, 512), centre=(0.0, 0.0), pixel_size=4e-7,
                       key="sem", zorder=-5)

    assert set(w.canvas.placed_keys) == {w.COMPOSITE_KEY, "fib", "sem"}
    # each at its own scale: 256 px at 2x the reference covers 512 canvas px
    fib = w.canvas._placed["fib"].extent
    assert fib[1] - fib[0] == pytest.approx(512)
    assert w.canvas._placed["sem"].artist.get_zorder() < 0


def test_a_layer_change_leaves_other_images_alone():
    w = _widget()
    w.set_channel("GFP", _plane(), "#00ff00")
    w.canvas.add_image(_plane(256, 256), centre=(40e-6, 0.0), pixel_size=2e-7, key="fib")
    before = w.canvas._placed["fib"].artist

    w.set_channel("GFP", _plane(), "#00ffff")  # recolour the FM channel

    assert w.canvas._placed["fib"].artist is before
    assert [layer.name for layer in w._layers] == ["GFP"]  # not listed as a layer


# ── more than one overview ────────────────────────────────────────────────


def test_a_second_overview_joins_the_first():
    """The reason for placing anything in stage space: acquire somewhere else and it
    lines up with what is already shown instead of replacing it."""
    w = _widget()
    w.set_composite_key("a")
    w.set_placement((0.0, 0.0))
    w.set_channel("GFP", _plane(), "#00ff00")

    w.set_composite_key("b")
    w.set_placement((200e-6, 0.0))
    w.set_channel("GFP", _plane(), "#00ff00")

    assert set(w.canvas.placed_keys) == {"a", "b"}
    a, b = w.canvas._placed["a"].extent, w.canvas._placed["b"].extent
    assert (b[0] + b[1]) / 2 - (a[0] + a[1]) / 2 == pytest.approx(2000)  # 200 um


def test_reusing_a_key_replaces_that_overview():
    w = _widget()
    for _ in range(3):
        w.set_composite_key("same")
        w.set_channel("GFP", _plane(), "#00ff00")
    assert w.canvas.placed_keys == ["same"]


def test_a_layer_change_restyles_every_overview_not_just_the_newest():
    """Layers are display state shared by everything placed; the pixels are not. Without
    re-rendering the held planes, recolouring would leave the same data drawn two
    different ways side by side."""
    w = _widget()
    plane = _plane()
    for key, centre in (("a", (0.0, 0.0)), ("b", (200e-6, 0.0))):
        w.set_composite_key(key)
        w.set_placement(centre)
        w.set_channel("GFP", plane, "green")

    w._layers[0].color = "red"
    w._recomposite()

    def red_mean(key):
        return float(np.asarray(w.canvas._placed[key].artist.get_array())[..., 0].mean())

    assert red_mean("a") == pytest.approx(red_mean("b"))
    assert red_mean("a") > 0  # actually recoloured, not merely equal at zero


def test_overviews_can_be_cleared_without_disturbing_the_channel_model():
    w = _widget()
    for key in ("a", "b"):
        w.set_composite_key(key)
        w.set_channel("GFP", _plane(), "#00ff00")

    w.clear_overviews()

    assert w.canvas.placed_keys == []
    assert w.placed_overviews() == []
    assert [layer.name for layer in w._layers] == ["GFP"]  # channels survive


def test_a_coarse_overview_sits_behind_a_detailed_one_whenever_it_arrived():
    """A wide survey acquired after a detailed overview covers the same ground, so by
    arrival order it would hide the better data. Ordering by pixel size instead keeps
    the most detailed image on top however the run happened to be sequenced."""
    w = _widget()
    w.set_composite_key("detailed")
    w.set_channel("GFP", _plane(), "#00ff00")

    w.set_pixel_size(PIXEL_SIZE * 4)  # a later, coarser survey on the same canvas
    w.set_composite_key("survey")
    w.set_channel("GFP", _plane(), "#00ff00")

    detailed_z = w.canvas._placed["detailed"].artist.get_zorder()
    survey_z = w.canvas._placed["survey"].artist.get_zorder()
    assert survey_z < detailed_z  # coarser is behind, despite arriving later


def test_held_planes_track_what_was_placed():
    w = _widget()
    w.set_composite_key("a")
    w.set_channel("GFP", _plane(), "#00ff00")
    w.set_composite_key("b")
    w.set_channel("GFP", _plane(), "#00ff00")
    assert w.placed_overviews() == ["a", "b"]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print("all passed" if not failed else f"{failed} failed")
