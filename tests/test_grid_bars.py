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


def grid_image(
    contrast=40.0,
    noise=4.0,
    tiles=False,
    seed=0,
    size=(792, 537),
    squash=1.0,
    gaps=True,
    holes=None,
):
    """Dark bars on grey film (bright for a negative *contrast*): one family at
    ANGLE, the other at a right angle, seen foreshortened up and down by
    *squash*. With ``tiles``, a 3x3 stitch, each tile shifted a few pixels as
    stage-stepped tiles are: 12 px black gaps between them, or with ``gaps``
    False edge to edge, each tile a little brighter or darker. *holes* are
    (x, y) px centres of broken squares: black discs 40 um across."""
    w, h = size
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    ys = ys / squash
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
    for hx, hy in holes or ():
        data[np.hypot(xs - hx, ys * squash - hy) < 20e-6 / PX] = 15
    if tiles and not gaps:
        for r in range(3):
            for c in range(3):
                data[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3] += (
                    12 if (r + c) % 2 else -12
                )
    elif tiles:
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


def test_tiles_stitched_edge_to_edge_are_split_at_their_seams():
    image, on_bar = grid_image(tiles=True, gaps=False)
    fit = fit_grid_bars(image)
    assert len(fit.regions) == 9
    h, w = on_bar.shape
    agree = [
        fit.on_bar(x, y, fit.region_of(x, y)) == on_bar[y, x]
        for y in range(5, h, 7)
        for x in range(5, w, 7)
    ]
    assert np.mean(agree) > 0.95


def test_a_tilted_view_is_a_square_mesh_squashed_up_and_down():
    """The FIB at its grazing angle: the squash that makes the mesh square is
    the view's, and clearances up and down are given in the sample plane."""
    image, on_bar = grid_image(contrast=-40, squash=0.6, size=(1200, 600))
    fit = fit_grid_bars(image)
    answer = fit.to_dict()
    assert answer["confidence"] == "good", answer
    assert answer["view_angle_deg"] == pytest.approx(
        math.degrees(math.acos(0.6)), abs=2
    )
    assert answer["mesh_pitch_um"] == pytest.approx(129, rel=0.02)
    for family in answer["families"]:
        assert family["bars"] == "bright"
    h, w = on_bar.shape
    agree = [
        fit.on_bar(x, y, 0) == on_bar[y, x]
        for y in range(5, h, 7)
        for x in range(5, w, 7)
    ]
    assert np.mean(agree) > 0.95
    # Up the image from a point in an opening to the bar edge, in image px,
    # against the site's up_um.
    ys, xs = np.nonzero(~on_bar[200:400, 500:700])
    x, y = 500 + xs[len(xs) // 2], 200 + ys[len(ys) // 2]
    steps = next(k for k in range(1, h) if on_bar[y - k, x])
    (site,) = measure_sites(fit, [(x, y)])
    assert site["up_um"] == pytest.approx(steps * PX / 0.6 * 1e6, abs=4)


def test_bars_that_are_no_square_mesh_are_unreliable():
    image, _ = grid_image()
    fit = fit_grid_bars(image)
    fit.families[1].pitch *= 1.3
    from fibsem.imaging.grid_bars import _mesh

    fit.squash, fit.mesh_error = _mesh(fit.families)
    assert fit.confidence == "unreliable"
    assert "not a square mesh" in fit.problems


def test_broken_squares_must_sit_in_the_openings():
    """Black holes where the film broke: in the openings the fit stands; on
    what it calls bars, the bars are misplaced."""
    image, on_bar = grid_image(contrast=-40)
    h, w = on_bar.shape
    clear = np.ones_like(on_bar)
    for angle in (ANGLE, ANGLE + math.pi / 2):
        ys, xs = np.mgrid[0:h, 0:w].astype(float)
        t = np.mod(xs * math.cos(angle) + ys * math.sin(angle), PITCH / PX)
        clear &= np.abs(t - (BAR + (PITCH - BAR) / 2) / PX) < 15e-6 / PX
    ys, xs = np.nonzero(clear[60:-60, 60:-60])
    centres = []
    for x, y in zip(xs + 60, ys + 60):
        if all(np.hypot(x - a, y - b) > PITCH / PX / 2 for a, b in centres):
            centres.append((x, y))
    assert len(centres) >= 5
    fit = fit_grid_bars(grid_image(contrast=-40, holes=centres)[0])
    assert fit.holes >= 5 and fit.holes_on_bars < 0.3
    assert fit.confidence == "good"
    # The same holes against bars shifted half a pitch: now they sit on them.
    for f in fit.families:
        f.offsets = [o + f.pitch / 2 for o in f.offsets]
    from fibsem.imaging.grid_bars import _holes

    data = np.asarray(grid_image(contrast=-40, holes=centres)[0].data, float)
    fit.holes, fit.holes_on_bars = _holes(fit, data, data > 0)
    assert "holes on the bars" in fit.problems


def test_a_view_angle_off_the_stage_tilt_is_unreliable():
    """A top-down mesh on an image that says it is the FIB at 52 degrees."""
    from fibsem.structures import (
        BeamSettings,
        BeamType,
        FibsemHardwareGeometry,
        MicroscopeState,
    )

    image, _ = grid_image()
    image.metadata.image_settings.beam_type = BeamType.ION
    image.metadata.microscope_state = MicroscopeState()
    image.metadata.microscope_state.ion_beam = BeamSettings(
        beam_type=BeamType.ION, scan_rotation=0.0
    )
    image.metadata.hardware_geometry = FibsemHardwareGeometry(fib_column_tilt=52.0)
    fit = fit_grid_bars(image)
    answer = fit.to_dict()
    assert answer["stage_view_angle_deg"] == pytest.approx(52, abs=0.5)
    assert answer["view_angle_deg"] < 10
    assert "view angle off the stage tilt" in answer["problems"]


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
