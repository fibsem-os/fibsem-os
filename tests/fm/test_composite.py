"""Multi-channel fluorescence compositing.

The module was written for the quad-view canvas on PR #111 and lifted to
`fibsem/fm/` so it can be shared. It had no unit tests there — it lived under
`fibsem/ui/`, which CI skips for lack of napari/PyQt5 — so these are new.
"""

import numpy as np
import pytest

from fibsem.fm.composite import FMLayer, auto_clim, composite_fm_layers, tint_rgb


def _layer(color: str, shape=(4, 4), **kwargs) -> FMLayer:
    """A channel with a real intensity ramp.

    Constant data has no contrast for auto_clim to find, so it composites to
    black — see test_a_flat_channel_renders_black. A ramp makes the brightest
    pixel normalise to 1.0, which is what the tint assertions read.
    """
    data = np.linspace(0, 4095, num=int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    return FMLayer(name=color, data=data, color=color, **kwargs)


# tint_rgb


def test_canonical_colours_use_napari_endpoints_not_matplotlib():
    """The composite has to match what the napari main tab shows.

    'green' is the case that actually differs: matplotlib's named green is
    (0, 0.5, 0), while napari's green colormap ramps to full (0, 1, 0).
    """
    assert tint_rgb("green") == (0.0, 1.0, 0.0)
    assert tint_rgb("red") == (1.0, 0.0, 0.0)


def test_gray_ramps_to_white():
    """A 'gray' channel is full-intensity white, not mid-grey — fibsemOS-acquired
    stacks label a channel 'gray' and it must not render at half brightness."""
    assert tint_rgb("gray") == (1.0, 1.0, 1.0)


def test_hex_colours_fall_through_to_matplotlib():
    """METEOR-acquired stacks carry hex channel colours rather than names."""
    assert tint_rgb("#FF0000") == pytest.approx((1.0, 0.0, 0.0))
    assert tint_rgb("#04FF00") == pytest.approx((4 / 255, 1.0, 0.0), abs=1e-6)


def test_an_unrecognised_colour_falls_back_to_white():
    """A channel with a junk colour must still be visible, not invisible."""
    assert tint_rgb("not-a-colour") == (1.0, 1.0, 1.0)
    assert tint_rgb(None) == (1.0, 1.0, 1.0)


# auto_clim


def test_auto_clim_brackets_the_data():
    data = np.arange(10_000, dtype=np.uint16).reshape(100, 100)
    lo, hi = auto_clim(data)
    assert lo < hi
    assert 0 <= lo and hi <= 9_999


def test_auto_clim_never_returns_a_degenerate_range():
    """A flat channel would otherwise divide by zero when normalising."""
    lo, hi = auto_clim(np.full((8, 8), 42, dtype=np.uint16))
    assert hi > lo


# composite_fm_layers


def test_returns_uint8_rgb():
    out = composite_fm_layers([_layer("red")])
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)


def test_a_single_channel_is_tinted_by_its_colour():
    out = composite_fm_layers([_layer("red")])
    assert out[..., 0].max() == 255
    assert out[..., 1].max() == 0
    assert out[..., 2].max() == 0


def test_channels_blend_additively():
    """Red plus green reads yellow — the point of compositing rather than
    showing one channel at a time."""
    out = composite_fm_layers([_layer("red"), _layer("green")])
    brightest = np.unravel_index(out.sum(axis=2).argmax(), out.shape[:2])
    assert out[brightest][0] == 255
    assert out[brightest][1] == 255
    assert out[brightest][2] == 0


def test_a_flat_channel_renders_black():
    """Documented, not a bug: auto-contrast has no range to stretch in constant
    data, so a uniformly-bright channel composites to black. Worth knowing before
    someone reports a blank image on a saturated acquisition.
    """
    flat = FMLayer(
        name="flat", data=np.full((4, 4), 500, dtype=np.uint16), color="red"
    )
    assert composite_fm_layers([flat]).max() == 0


def test_an_invisible_channel_contributes_nothing():
    both = composite_fm_layers([_layer("red"), _layer("green", visible=False)])
    red_only = composite_fm_layers([_layer("red")])
    np.testing.assert_array_equal(both, red_only)


def test_a_mismatched_channel_is_ignored_rather_than_raising():
    """Channels of differing size would otherwise break the whole composite."""
    out = composite_fm_layers([_layer("red"), _layer("green", shape=(8, 8))])
    assert out.shape == (4, 4, 3)
    assert out[..., 1].max() == 0


def test_opacity_scales_a_channels_contribution():
    full = composite_fm_layers([_layer("red")])
    half = composite_fm_layers([_layer("red", opacity=0.5)])
    assert half[..., 0].max() < full[..., 0].max()


def test_nothing_visible_returns_zeros_for_a_known_shape():
    out = composite_fm_layers([_layer("red", visible=False)], shape=(4, 4))
    assert out.shape == (4, 4, 3)
    assert out.max() == 0


def test_nothing_visible_and_no_shape_returns_none():
    """There is no sensible image to show, and guessing a size would be worse."""
    assert composite_fm_layers([]) is None
