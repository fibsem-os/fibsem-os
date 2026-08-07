"""Multi-channel fluorescence compositing.

The module was written for the quad-view canvas on PR #111 and lifted to
`fibsem/fm/` so it can be shared. It had no unit tests there — it lived under
`fibsem/ui/`, which CI skips for lack of napari/PyQt5 — so these are new.
"""

import numpy as np
import pytest

from fibsem.fm.composite import (
    FMLayer,
    auto_clim,
    composite_fm_layers,
    tint_rgb,
    to_rgba,
)


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


# to_rgba


def _render(*images) -> np.ndarray:
    """Draw *images* over a black background, in order, and read back the pixels.

    The canvas this stands in for places many images at once over
    ``GRAY_CANVAS_COLOR`` (black), so that is the background here. Matplotlib rather
    than the canvas itself: what is under test is how the frames composite, and the
    canvas needs Qt.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1, 1), dpi=64)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("black")  # set_axis_off hides the axes patch
    for image in images:
        # One zorder for all of them, so arrival order decides -- which is the case
        # that bit: two overviews at the same pixel size tie in `_detail_zorder`.
        ax.imshow(image, extent=(0, 4, 4, 0), zorder=1)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
    plt.close(fig)
    return out


def test_it_is_the_same_picture_over_black():
    """The regression that matters. Every FM overview is drawn through this, and most
    of them have nothing underneath -- so the transform has to be invisible there."""
    rgb = composite_fm_layers([_layer("green")])

    difference = np.abs(_render(rgb) - _render(to_rgba(rgb))).max()

    assert difference <= 1.0, f"the picture changed by {difference}/255 over black"


def test_it_shows_the_overview_underneath_where_this_one_has_nothing():
    """The reported bug (FIB-519): a sparse overview acquired over a full one drew
    its unacquired tiles as black over real data."""
    beneath = composite_fm_layers([_layer("green")])
    # `stitch_tileset` leaves an unacquired tile as canvas zeros, so that is what the
    # top half of this one holds.
    sparse = _layer("green")
    sparse.data = sparse.data.copy()
    sparse.data[:2] = 0
    above = composite_fm_layers([sparse])

    opaque = _render(beneath, above)
    layered = _render(beneath, to_rgba(above))
    alone = _render(beneath)

    unacquired = np.s_[:24]  # the top half, in rendered pixels
    assert opaque[unacquired].max() == 0, "expected today's behaviour: black"
    np.testing.assert_allclose(
        layered[unacquired], alone[unacquired], atol=1.0,
        err_msg="the overview beneath is not showing through",
    )


def test_alpha_carries_the_signal_and_colour_survives_it():
    """Colour times alpha has to reconstruct the composite -- that is what makes the
    substitution invisible -- while alpha alone says how much is there."""
    rgb = composite_fm_layers([_layer("green")])

    rgba = to_rgba(rgb)

    reconstructed = (
        rgba[..., :3].astype(np.float32) * rgba[..., 3:4].astype(np.float32) / 255.0
    )
    np.testing.assert_allclose(reconstructed, rgb.astype(np.float32), atol=1.0)


def test_a_region_holding_nothing_is_fully_transparent():
    """Not merely dark: an unacquired tile has to disappear, not dim what is beneath."""
    blank = np.zeros((4, 4, 3), dtype=np.uint8)

    assert to_rgba(blank)[..., 3].max() == 0


def test_it_returns_rgba_the_canvas_accepts():
    rgba = to_rgba(composite_fm_layers([_layer("red")]))

    assert rgba.dtype == np.uint8
    assert rgba.shape == (4, 4, 4)
