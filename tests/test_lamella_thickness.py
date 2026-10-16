"""The lamella band measurement (FIB-1434) on synthetic FIB images.

The numbers are the 2026-10-15 Arctis polish's: a 235 nm band, 193-251 nm, a
~0.6 degree rotation, at 15 um and at 30 um (39 nm per pixel) fields of view.
"""

import math

import numpy as np
import pytest

from fibsem.imaging.thickness import (
    MIN_BAND_PX,
    draw_band_edges,
    measure_lamella_band,
)
from fibsem.structures import FibsemImage, FibsemRectangle, Point

WIDTH, HEIGHT = 1536, 1024
FINE = 15e-6 / WIDTH  # 9.8 nm per pixel


def band_image(
    thickness_nm, angle_deg=0.0, pixel_size=FINE, offset_px=0.0, hole=None, seed=0
):
    """A bright band (the lamella) between dark trenches, grey bulk beyond them.

    ``thickness_nm`` is a number or a function of the column; ``offset_px`` moves
    the band up; ``hole`` (first, last) columns of no band. Each pixel is lit by
    how much of it the band covers, as an edge falls between pixels.
    """
    image = FibsemImage.generate_blank_image(
        resolution=(WIDTH, HEIGHT), hfw=WIDTH * pixel_size
    )
    ys = np.arange(HEIGHT)[:, None]  # pixel y is centred on y, as fibsem has it
    xs = np.arange(WIDTH)[None, :]
    centre = (
        HEIGHT / 2 - offset_px - math.tan(math.radians(angle_deg)) * (xs - WIDTH / 2)
    )
    nm = thickness_nm(xs) if callable(thickness_nm) else thickness_nm
    half = np.asarray(nm, dtype=float) * 1e-9 / pixel_size / 2
    cover = np.clip(
        np.minimum(ys + 0.5, centre + half) - np.maximum(ys - 0.5, centre - half), 0, 1
    )
    data = 40 + cover * 160
    data = np.where((np.abs(ys - centre) < 150) | (cover > 0), data, 120)
    if hole is not None:
        data[:, hole[0] : hole[1] + 1] = 40
    data += np.random.default_rng(seed).normal(0, 2.0, data.shape)
    image.data = np.clip(data, 0, 255).astype(np.uint8)
    return image


def test_a_rotated_band_reads_its_thickness_and_angle():
    answer = measure_lamella_band(band_image(235, angle_deg=0.6), Point(0, 0)).to_dict()
    one_px = answer["uncertainty_nm"]
    assert one_px == pytest.approx(9.8, abs=0.05)
    assert answer["thickness_nm"]["mean"] == pytest.approx(235, abs=one_px)
    assert answer["thickness_nm"]["median"] == pytest.approx(235, abs=one_px)
    assert answer["angle_deg"] == pytest.approx(0.6, abs=0.05)
    assert answer["straightness"]["centre_spread_px"] < 0.5
    assert answer["centre_offset_nm"] == pytest.approx(0, abs=one_px / 2)
    assert answer["continuity"]["dropped"] == 0
    assert answer["continuity"]["gaps_px"] == []
    assert answer["resolution"]["under_resolved"] is False
    assert answer["contrast"]["ratio"] > 3


def test_a_bowed_band_reads_its_range_and_thirds():
    # Thinner at both ends, as the final polish was (193-251 nm).
    def bowed(x):
        return 193 + 58 * np.sin(np.pi * x / WIDTH)

    answer = measure_lamella_band(band_image(bowed)).to_dict()
    thick = answer["thickness_nm"]
    assert thick["min"] == pytest.approx(193, abs=10)
    assert thick["max"] == pytest.approx(251, abs=10)
    left, middle, right = answer["thirds_nm"]
    assert middle > left + 20 and middle > right + 20
    assert left == pytest.approx(right, abs=5)


def test_a_wedge_reads_its_slope():
    def wedge(x):
        return 195 + 56 * x / WIDTH  # 56 nm across 15 um

    answer = measure_lamella_band(band_image(wedge)).to_dict()
    assert answer["wedge"]["slope_nm_per_um"] == pytest.approx(56 / 15, abs=0.3)
    assert answer["wedge"]["across_band_nm"] == pytest.approx(56, abs=5)
    assert answer["angle_deg"] == pytest.approx(0, abs=0.05)


def test_a_hole_is_dropped_and_named():
    measurement = measure_lamella_band(band_image(235, hole=(700, 719)), Point(0, 0))
    answer = measurement.to_dict()
    assert answer["continuity"]["dropped"] == 20
    assert answer["continuity"]["gaps_px"] == [[700, 719]]
    assert answer["thickness_nm"]["mean"] == pytest.approx(235, abs=10)

    preview = draw_band_edges(band_image(235, hole=(700, 719)), measurement)
    x0, y0, x1, y1 = measurement.region
    assert preview.shape == (y1 - y0, x1 - x0, 3)


def test_coarse_pixels_flag_the_band_and_suggest_a_field_of_view():
    # 30 um at 768 px: 39 nm per pixel, a 235 nm band is 6 px.
    coarse = 39e-9
    answer = measure_lamella_band(
        band_image(235, pixel_size=coarse, offset_px=3), Point(0, 0)
    ).to_dict()
    resolution = answer["resolution"]
    assert resolution["band_px"] < MIN_BAND_PX
    assert resolution["under_resolved"] is True
    # The band at 16 px: 235 nm / 16 * 1536 px.
    assert resolution["suggested_field_width_um"] == pytest.approx(22.6, abs=1.5)
    assert answer["thickness_nm"]["mean"] == pytest.approx(235, abs=39)
    # Moved up 3 px from the POI: above it, so positive.
    assert answer["centre_offset_nm"] == pytest.approx(3 * 39, abs=39 / 2)


def test_a_rectangle_limits_the_search():
    image = band_image(235)
    rect = FibsemRectangle(left=0.25, top=0.4, width=0.5, height=0.2)
    measurement = measure_lamella_band(image, rect)
    assert measurement.region == (384, 410, 1152, 614)
    assert measurement.to_dict()["continuity"]["columns"] == 768


def test_refuses_what_it_cannot_measure():
    image = band_image(235)
    with pytest.raises(ValueError):
        measure_lamella_band(image, Point(1e-3, 0))  # off the image
    image.metadata = None
    with pytest.raises(ValueError):
        measure_lamella_band(image)
