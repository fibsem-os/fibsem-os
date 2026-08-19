"""The FM canvas draws the part of an overview that is on screen, at the screen's size.

Placed overviews used to be blended whole, at a fixed cap, once per placed image per
layer change -- on screen or not. Each now carries a `DetailSource`, so the canvas asks
for the region the viewport covers at the resolution it can use, and asks nothing at all
of an image that is not visible.

Two things follow that the fixed cap could not give: zooming in returns detail rather
than magnifying stored pixels, and cost stops scaling with how many overviews are held.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget
from fibsem.ui.widgets.canvas.fm_composite import auto_clim
from fibsem.ui.widgets.canvas.real_space_canvas import WHOLE_IMAGE, ImageRegion

_app = QApplication.instance() or QApplication(sys.argv)

CHANNELS = [("red", "red"), ("green", "green")]
# Wider than the canvas's display cap, so there is detail a whole-image draw cannot show.
SIDE = 4096


def plane(side: int = SIDE, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random((side, side)) * 4000).astype(np.uint16)


@pytest.fixture
def widget():
    w = FMRealSpaceCanvasWidget()
    w.set_pixel_size(1e-7)
    yield w
    w.deleteLater()


def place(widget, key="fm-composite", centre=(0.0, 0.0), seed=0):
    widget.set_composite_key(key)
    widget.set_placement(centre)
    widget.set_channels([(n, plane(seed=seed + i), c) for i, (n, c) in enumerate(CHANNELS)])
    return key


def watch_patches(widget, monkeypatch):
    """Every region the widget is asked to draw."""
    asked = []
    original = type(widget)._patch

    def recording(self, key, region, max_px):
        asked.append((key, region, max_px))
        return original(self, key, region, max_px)

    monkeypatch.setattr(type(widget), "_patch", recording)
    return asked


# ── the source is wired at all ───────────────────────────────────────────


def test_a_placed_overview_carries_a_source(widget):
    key = place(widget)

    assert widget.canvas._placed[key].detail is not None


def test_the_source_is_dropped_with_the_image(widget):
    """It closes over the key, and the planes behind that key are what
    `forget_overview` drops. A source outliving them would be asked to draw pixels that
    are gone."""
    key = place(widget)

    widget.canvas.remove_image(key)
    widget.forget_overview(key)

    assert key not in widget.canvas._placed
    assert widget._patch(key, WHOLE_IMAGE, 512) is None


# ── what a patch is ──────────────────────────────────────────────────────


def test_a_patch_says_which_ground_it_actually_covers(widget):
    """Snapped outward to whole stored pixels, and the region *covered* is returned
    rather than the one asked for. The canvas draws the patch over the rectangle this
    names, so a fractional-pixel disagreement would show as a seam on every view change.
    """
    key = place(widget)
    asked = ImageRegion(0.1234, 0.4321, 0.2345, 0.6789)

    _, got = widget._patch(key, asked, 512)

    # Every edge lands on a whole stored pixel. This is the assertion that bites: a
    # source returning the region it was *asked* for satisfies "covers what was asked"
    # trivially, and only this notices.
    for edge, extent in ((got.left, SIDE), (got.right, SIDE),
                         (got.top, SIDE), (got.bottom, SIDE)):
        assert edge * extent == pytest.approx(round(edge * extent), abs=1e-9)

    # Outward, strictly -- none of the asked edges falls on a pixel boundary.
    assert got.left < asked.left and got.right > asked.right
    assert got.top < asked.top and got.bottom > asked.bottom
    # And by less than one stored pixel, i.e. snapped rather than padded.
    assert (asked.left - got.left) < 1.0 / SIDE
    assert (got.right - asked.right) < 1.0 / SIDE


def test_zooming_in_returns_detail_rather_than_bigger_pixels(widget):
    """The whole point. At a fixed cap a region of the image is the same stored pixels
    magnified; through a source it is more of them."""
    key = place(widget)
    budget = 1024

    whole, _ = widget._patch(key, WHOLE_IMAGE, budget)
    eighth, _ = widget._patch(key, ImageRegion(0.4, 0.525, 0.4, 0.525), budget)

    stored_per_output_whole = SIDE / whole.shape[0]
    stored_per_output_eighth = (SIDE * 0.125) / eighth.shape[0]
    assert stored_per_output_eighth < stored_per_output_whole


def test_a_patch_is_blended_at_its_own_size(widget):
    """Blended at patch resolution rather than cut out of a finished composite -- which
    is what takes the cost off the size of the mosaic and onto the size of the window."""
    key = place(widget)

    patch, _ = widget._patch(key, ImageRegion(0.0, 0.25, 0.0, 0.25), 256)

    assert max(patch.shape[:2]) <= 256


def test_hiding_every_channel_makes_the_image_disappear(widget):
    """Declining would leave the last patch drawn, so the picture would not respond.

    Black composites to transparent through `to_rgba`, which over this canvas's
    background is the image going away -- and over another image, that one showing
    through.
    """
    key = place(widget)
    for layer in widget.layers:
        layer.visible = False

    patch, _ = widget._patch(key, WHOLE_IMAGE, 512)

    assert patch[..., 3].max() == 0


def test_contrast_is_the_image_s_and_not_the_region_s(widget):
    """End to end through the source, complementing the unit test on `_pinned`: a dim
    corner drawn on its own must not be stretched to itself."""
    widget.set_pixel_size(1e-7)
    bright = plane()
    bright[: SIDE // 8, : SIDE // 8] = (
        np.random.default_rng(1).random((SIDE // 8, SIDE // 8)) * 150
    ).astype(np.uint16)
    widget.set_channel("green", bright)
    key = widget._composite_key

    whole, _ = widget._patch(key, WHOLE_IMAGE, 1024)
    corner, _ = widget._patch(key, ImageRegion(0.0, 0.125, 0.0, 0.125), 1024)

    # Alpha, not RGB: `to_rgba` divides the colour back out by alpha, so the colour
    # channels saturate wherever there is little signal. Signal strength is the alpha.
    assert corner[..., 3].mean() < 0.5 * whole[..., 3].mean(), (
        "the dim corner was stretched to itself, so zooming in changed the data"
    )
    # And it would have been, left to itself: the corner's own limits are a fraction of
    # the image's, so an unpinned blend would have drawn it as bright as the sample.
    assert auto_clim(bright[: SIDE // 8, : SIDE // 8])[1] < 0.2 * auto_clim(bright)[1]


# ── and what it does not do ──────────────────────────────────────────────


def test_an_overview_off_screen_is_not_drawn_on_a_layer_edit(widget, monkeypatch):
    """Where the cost of many overviews went.

    `_restyle_others` re-blended every held image on every layer change, whether it was
    on screen or not, so five overviews cost five blends per frame of a slider drag. The
    canvas asks only the sources whose image the viewport intersects.
    """
    near = place(widget, key="near", centre=(0.0, 0.0), seed=0)
    far = place(widget, key="far", centre=(1.0, 1.0), seed=9)
    # Frame the near one only. Its footprint is about 4096 canvas px across.
    widget.canvas._ax.set_xlim(-2048, 2048)
    widget.canvas._ax.set_ylim(2048, -2048)
    asked = watch_patches(widget, monkeypatch)

    widget.layers[0].opacity = 0.5
    widget._panel.changed.emit()

    drawn = {key for key, _, _ in asked}
    assert near in drawn, "the visible overview was not redrawn under the new settings"
    assert far not in drawn, "an overview off screen was blended anyway"


def test_a_layer_edit_does_not_re_place_the_image(widget, monkeypatch):
    """Re-adding under the same key destroys and recreates the artist, which moves it to
    the top of the draw order -- so adjusting a colour would silently reorder overlapping
    overviews. The FIB/SEM side hit this first (`_reapply_contrast`)."""
    place(widget, key="a", centre=(0.0, 0.0))
    place(widget, key="b", centre=(2e-4, 0.0), seed=9)
    order = list(widget.canvas.placed_keys)

    widget.layers[0].gamma = 1.4
    widget._panel.changed.emit()

    assert list(widget.canvas.placed_keys) == order


def test_a_source_asked_about_pixels_that_are_gone_declines(widget):
    """It runs off a timer, and PyQt5 aborts the process on an exception out of a slot
    (FIB-329). Declining leaves whatever is drawn alone."""
    assert widget._patch("never-placed", WHOLE_IMAGE, 512) is None
