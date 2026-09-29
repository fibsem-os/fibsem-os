"""Auto contrast belongs to a placed image, not to the part of it being drawn.

`composite_fm_layers` takes its own 1st/99th percentile from whatever array it is handed,
and caches it on that array's *identity*. Both are right for a live frame and wrong for a
region of a stored one: a region would be stretched to itself, and every region is a
fresh array, so the cache would miss every time. Panning a mosaic would restyle it under
the cursor.

So the limits are pinned per (placed image, channel) before anything is blended. Nothing
in this file draws a region yet -- that is what the pinning is *for*, and it is tested
here through `_pinned` directly so the property is established before anything depends
on it.
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

_app = QApplication.instance() or QApplication(sys.argv)


# Wider than the canvas's display cap, so `_reduce` genuinely reduces and hands back a
# *different* array. At 900 px it returns the plane unchanged, and then the reduction and
# the plane are the same object -- which quietly satisfies a cache keyed on either one,
# and left the slider test below unable to fail.
SIDE = 2600


def plane(side: int = SIDE, seed: int = 0, scale: float = 4000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((side, side)) * scale).astype(np.uint16)


@pytest.fixture
def widget():
    w = FMRealSpaceCanvasWidget()
    w.set_pixel_size(1e-7)
    yield w
    w.deleteLater()


@pytest.fixture
def clims(widget, monkeypatch):
    """Every auto-contrast computation, counted."""
    calls = []
    import fibsem.ui.widgets.canvas.fm_canvas as module

    def counting(data, *args, **kwargs):
        calls.append(data.shape)
        return auto_clim(data, *args, **kwargs)

    monkeypatch.setattr(module, "auto_clim", counting)
    return calls


# ── the property the rest depends on ─────────────────────────────────────


def test_a_region_is_blended_at_the_whole_images_contrast(widget):
    """The point of the whole exercise.

    A corner of an overview is dimmer than the overview; stretched to itself it would be
    drawn as bright as the sample, so zooming in would change what the data looks like
    rather than showing more of it.
    """
    bright = plane(scale=4000.0)
    corner_px = SIDE // 8
    bright[:corner_px, :corner_px] = (
        np.random.default_rng(1).random((corner_px, corner_px)) * 200
    ).astype(np.uint16)  # a dim corner
    widget.set_channel("green", bright)
    layer = widget.layers[0]

    whole = widget._pinned("k", layer, bright, widget._reduce(bright))
    corner = widget._pinned(
        "k", layer, bright, widget._reduce(bright[:corner_px, :corner_px])
    )

    assert corner.clim == whole.clim
    assert corner.autocontrast is False, "left on Auto, the blend would restretch it"


def test_two_placed_images_keep_their_own_contrast(widget):
    """Overviews acquired at different exposures should each be readable; one shared
    stretch would flatten the dimmer one.

    This is what the widget already did -- `_restyle_others` recomputes from each held
    image's own planes -- so pinning must not quietly become one answer for the canvas.
    """
    dim, bright = plane(scale=200.0, seed=2), plane(scale=4000.0, seed=3)
    layer = widget._upsert_layer("green", bright, "green")

    first = widget._pinned("one", layer, dim, widget._reduce(dim))
    second = widget._pinned("two", layer, bright, widget._reduce(bright))

    assert first.clim != second.clim
    assert first.clim[1] < second.clim[1]


def test_a_channel_taken_off_auto_keeps_the_limits_that_were_set(widget):
    """Pinning is about making *automatic* contrast independent of the view. A manual
    choice is not something to improve on."""
    data = plane()
    layer = widget._upsert_layer("green", data, "green")
    chosen = (12.0, 34.0)
    layer.autocontrast, layer.clim, layer.manual = False, chosen, True

    pinned = widget._pinned("k", layer, data, widget._reduce(data))

    assert pinned.clim == chosen


def test_the_limits_come_from_the_stored_plane_not_the_reduction(widget):
    """Where this differs from what the widget used to do, and why.

    `downsample` box-averages, so a reduction has a narrower histogram than the plane it
    came from -- by however much power the image holds at the pixel scale. Limits taken
    from it are therefore limits for one particular display cap, and a patch drawn at a
    different reduction would not match them.

    White noise is the extreme case and is what this asserts against, because it is the
    only one where the two answers are far enough apart to tell them apart: the reduced
    span comes out 0.45x the stored one. On smooth structure with shot noise -- what
    fluorescence data actually looks like -- they agree to 1.00, which is why this is not
    a visible change.
    """
    data = plane()

    pinned = widget._pinned(
        "k", widget._upsert_layer("green", data, "green"), data, widget._reduce(data)
    )

    stored = auto_clim(data)
    assert pinned.clim == pytest.approx(stored, rel=0.02)
    reduced = auto_clim(widget._reduce(data))
    assert reduced != pytest.approx(stored, rel=0.02), (
        "the two answers agree on this data, so the assertion above proves nothing"
    )


# ── computed once ────────────────────────────────────────────────────────


def test_the_limits_survive_a_slider_drag(widget, clims):
    """A recomposite rebuilds the reduced planes, so anything keyed on *those* caches
    nothing. Keyed on the stored plane instead, which an opacity drag does not touch.
    """
    widget.set_channel("green", plane())
    before = len(clims)

    for opacity in (0.9, 0.8, 0.7):
        widget.layers[0].opacity = opacity
        widget._recomposite()

    assert len(clims) == before, "recomputed the percentile on every frame of the drag"


def test_new_pixels_under_the_same_key_are_measured_again(widget, clims):
    """The live preview replaces a key's planes tile by tile. Holding the first tile's
    limits for the whole run would leave the mosaic stretched to its top-left corner."""
    widget.set_channel("green", plane(seed=4))
    before = len(clims)

    widget.set_channel("green", plane(seed=5, scale=60000.0))

    assert len(clims) > before


def test_forgetting_an_overview_forgets_its_contrast(widget):
    """The cache is keyed by the same key the planes are, so it has to be dropped with
    them -- otherwise a key reused for a different overview inherits the old stretch."""
    widget.set_composite_key("first")
    widget.set_channel("green", plane())
    assert widget._held_clim.get("first")

    widget.forget_overview("first")

    assert "first" not in widget._held_clim


def test_clearing_forgets_every_overview_s_contrast(widget):
    for key in ("a", "b"):
        widget.set_composite_key(key)
        widget.set_channel("green", plane())
    assert len(widget._held_clim) == 2

    widget.clear_overviews()

    assert widget._held_clim == {}


def test_two_overviews_sharing_a_channel_do_not_evict_each_other(widget, clims):
    """Why the cache is nested by placed image rather than flat by channel name.

    Overviews are composited under one set of channels, so two of them routinely hold a
    channel of the same name. Flat, each would overwrite the other's entry and every
    recomposite would recompute both -- correct, because the entry carries the plane it
    was measured from, but paying the percentile per image per frame is exactly the cost
    this cache exists to remove.
    """
    for key, seed in (("first", 6), ("second", 7)):
        widget.set_composite_key(key)
        widget.set_channel("green", plane(seed=seed))
    before = len(clims)

    for opacity in (0.9, 0.8):
        widget.layers[0].opacity = opacity
        widget._recomposite()

    assert len(clims) == before
