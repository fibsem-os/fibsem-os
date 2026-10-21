"""Grid bars fitted on an overview (G9, after the 2026-10-15 Arctis session's
lattice fit), on synthetic grids: two families of straight bars, a stitched
overview with gaps and tiles a few microns off one another, and the faint
contrast of a coated grid."""

import math

import numpy as np
import pytest

from fibsem.imaging.grid_bars import fit_grid_bars, measure_sites, sites_picture
from fibsem.structures import FibsemImage

PX = 0.82e-6  # the 2026-10-15 montage's preview pixel
PITCH, BAR = 129e-6, 46e-6
ANGLE = math.radians(28)  # the bars' rotation on the image


def grid_image(contrast=40.0, noise=4.0, tiles=False, seed=0, size=(792, 537)):
    """Dark bars on grey film: one family at ANGLE, the other at a right angle.
    With ``tiles``, a 3x3 stitch with 12 px black gaps, each tile shifted a few
    pixels as stage-stepped tiles are."""
    w, h = size
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    shift_x = np.zeros((h, w))
    shift_y = np.zeros((h, w))
    if tiles:
        for r in range(3):
            for c in range(3):
                sl = (
                    slice(r * h // 3, (r + 1) * h // 3),
                    slice(c * w // 3, (c + 1) * w // 3),
                )
                shift_x[sl], shift_y[sl] = rng.uniform(-5, 5, 2)
    pitch, bar = PITCH / PX, BAR / PX
    on_bar = np.zeros((h, w), bool)
    for angle in (ANGLE, ANGLE + math.pi / 2):
        t = (xs + shift_x) * math.cos(angle) + (ys + shift_y) * math.sin(angle)
        on_bar |= np.mod(t, pitch) < bar
    data = 120 - contrast * on_bar + rng.normal(0, noise, (h, w))
    if tiles:
        for k in (1, 2):
            data[k * h // 3 - 6 : k * h // 3 + 6, :] = 0
            data[:, k * w // 3 - 6 : k * w // 3 + 6] = 0
    image = FibsemImage.generate_blank_image(resolution=(w, h), hfw=w * PX)
    image.data = np.clip(data, 0, 255).astype(np.uint8)
    return image, on_bar


def test_both_bar_families_are_found_with_their_pitch_and_width():
    image, _ = grid_image()
    fit = fit_grid_bars(image)
    answer = fit.to_dict()
    assert answer["confidence"] == "good"
    angles = sorted(f["angle_deg"] for f in answer["families"])
    assert angles == pytest.approx([28, 118], abs=0.5)
    for family in answer["families"]:
        assert family["pitch_um"] == pytest.approx(129, rel=0.01)
        assert family["bar_width_um"] == pytest.approx(46, abs=3)
        assert family["bars"] == "dark"


@pytest.mark.parametrize("contrast, noise", [(40, 4), (8, 4), (4, 4)])
def test_a_stitched_overview_is_fitted_per_tile_even_when_faint(contrast, noise):
    """Each tile's own offset, and a coated grid's faint bars."""
    image, on_bar = grid_image(contrast=contrast, noise=noise, tiles=True)
    fit = fit_grid_bars(image)
    assert len(fit.regions) == 9
    h, w = on_bar.shape
    agree = []
    for y in range(5, h, 7):
        for x in range(5, w, 7):
            region = fit.region_of(x, y)
            if region is not None:
                agree.append(fit.on_bar(x, y, region) == on_bar[y, x])
    assert np.mean(agree) > 0.95, (contrast, noise, np.mean(agree))


def test_noise_drowning_the_bars_says_unreliable():
    image, _ = grid_image(contrast=2, noise=25)
    try:
        fit = fit_grid_bars(image)
    except ValueError:
        return  # no lattice at all is as good an answer
    assert fit.to_dict()["confidence"] in ("weak", "unreliable")


def test_sites_are_tiered_by_their_clearance_from_the_bars():
    image, on_bar = grid_image()
    fit = fit_grid_bars(image)
    # A square's centre: walk from a point well inside an opening.
    region = fit.region_of(400, 270)
    ys, xs = np.nonzero(~on_bar[200:340, 330:470])
    start = (330 + xs[len(xs) // 2], 200 + ys[len(ys) // 2])
    centre = fit.square_centre(*start, region)
    opening = (PITCH - BAR) / PX
    # Along the first family's normal from the centre: clear, then within
    # 10 um of the bar, then on it.
    normal = fit.families[0].normal
    near = centre + normal * (opening / 2 - 5e-6 / PX)
    bar = centre + normal * (opening / 2 + BAR / PX / 2)
    sites = measure_sites(fit, [centre, near, bar])
    assert [s["tier"] for s in sites] == ["middle", "near_bar", "on_bar"]
    middle = sites[0]
    assert middle["from_centre_um"] == pytest.approx(0, abs=1)
    assert middle["clearance_um"] == pytest.approx((PITCH - BAR) / 2 * 1e6, abs=2)
    assert middle["sites_in_square"] == 2  # the near one shares its square
    assert sites[1]["clearance_um"] == pytest.approx(5, abs=1.5)
    # A cell's radius is taken off: 20 um across no longer clears the margin.
    (big,) = measure_sites(fit, [near + normal * -10e-6 / PX], radius=10e-6)
    assert big["tier"] == "near_bar"


def test_the_picture_is_a_png_and_a_jpeg():
    image, _ = grid_image(tiles=True)
    fit = fit_grid_bars(image)
    png, jpeg = sites_picture(fit, measure_sites(fit, [(100, 100), (5, 180)]))
    assert png[:4] == b"\x89PNG" and jpeg[:2] == b"\xff\xd8"


def test_refuses_what_it_cannot_fit():
    image, _ = grid_image()
    image.data[:] = 0
    with pytest.raises(ValueError):
        fit_grid_bars(image)
    image.metadata = None
    with pytest.raises(ValueError):
        fit_grid_bars(image)
