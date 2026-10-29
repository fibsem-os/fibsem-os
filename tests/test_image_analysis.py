"""Measurements on saved images (`fibsem.imaging.analysis`): intensity at points
per plane, a line's profile and FWHM, and image quality."""

import numpy as np
import pytest

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.imaging import analysis
from fibsem.imaging.analysis import Planes


def _spot(shape, x, y, sigma, amplitude):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def _stack(planes=5):
    """Two spots on a flat 100 background, brightest in plane 2; the second
    half as bright as the first."""
    data = np.full((1, planes, 64, 64), 100.0)
    for z in range(planes):
        weight = np.exp(-((z - 2) ** 2) / 2)
        data[0, z] += _spot((64, 64), 20, 20, 1.2, 1000 * weight)
        data[0, z] += _spot((64, 64), 44, 40, 1.2, 500 * weight)
    return Planes(data=data.astype(np.uint16), names=["GFP"], pixel_size=1e-7)


def test_intensity_finds_each_point_s_best_plane_and_its_ratio():
    result = analysis.measure_intensity(
        _stack(), [[20, 20], [44, 40]], radius_px=3, reference=0
    )
    first, second = result["points"]
    assert first["best_plane"] == second["best_plane"] == 2
    # The ring is flat background: signal is the spot above it.
    assert first["background"] == pytest.approx(100, abs=1)
    assert second["ratio"] == pytest.approx(0.5, abs=0.02)
    assert first["ratio"] == 1.0
    assert len(first["planes"]) == 5 and not first["saturated"]


def test_intensity_refuses_what_it_can_t_measure():
    stack = _stack()
    with pytest.raises(ValueError, match="outside"):
        analysis.measure_intensity(stack, [[70, 1]])
    with pytest.raises(ValueError, match="no layer RFP"):
        analysis.measure_intensity(stack, [[20, 20]], channel="RFP")
    with pytest.raises(ValueError, match="plane 9"):
        analysis.measure_intensity(stack, [[20, 20]], planes=[9])
    with pytest.raises(ValueError, match="reference"):
        analysis.measure_intensity(stack, [[20, 20]], reference=1)


def test_a_saturated_spot_says_so():
    stack = _stack()
    stack.data[0, 2, 18:23, 18:23] = 65535
    stack.full_value = 65535.0
    result = analysis.measure_intensity(stack, [[20, 20]])
    assert result["points"][0]["saturated"]


def test_the_intensity_figure_draws_a_row_per_point():
    stack = _stack()
    result = analysis.measure_intensity(stack, [[20, 20], [44, 40]])
    rgb = analysis.intensity_figure(stack, result)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[0] > 200


def test_a_gaussian_s_fwhm_is_2_355_sigma():
    sigma = 3.0
    data = 50 + _spot((64, 64), 32, 32, sigma, 1000)
    image = Planes(data=data[None, None], names=["SEM"], pixel_size=2e-9)
    result = analysis.line_profile(image, [8, 32], [56, 32], width_px=1)
    assert result["fwhm_px"] == pytest.approx(2.355 * sigma, rel=0.05)
    assert result["fwhm_m"] == pytest.approx(result["fwhm_px"] * 2e-9)
    assert result["extremum_px"] == pytest.approx(24, abs=0.5)
    assert len(result["values"]) == 49


def test_a_dip_is_measured_as_a_dip():
    data = 1000 - _spot((64, 64), 32, 32, 2.0, 800)
    image = Planes(data=data[None, None], names=["FIB"])
    result = analysis.line_profile(image, [32, 4], [32, 60], width_px=3, dip=True)
    assert result["kind"] == "dip"
    assert result["fwhm_px"] == pytest.approx(2.355 * 2.0, rel=0.1)
    assert "fwhm_m" not in result


def test_a_profile_that_never_falls_to_half_has_no_fwhm():
    data = np.tile(np.linspace(0, 100, 64), (64, 1))
    image = Planes(data=data[None, None], names=["SEM"])
    result = analysis.line_profile(image, [0, 10], [63, 10])
    assert result["fwhm_px"] is None
    with pytest.raises(ValueError, match="2 pixels"):
        analysis.line_profile(image, [0, 0], [1, 0])


def test_the_profile_figure_draws():
    data = 50 + _spot((64, 64), 32, 32, 3.0, 1000)
    image = Planes(data=data[None, None], names=["SEM"], pixel_size=2e-9)
    result = analysis.line_profile(image, [8, 32], [56, 32])
    assert analysis.profile_figure(image, result).shape[2] == 3


def test_quality_flags_a_saturated_image_and_finds_the_sharpest_plane():
    rng = np.random.default_rng(0)
    good = rng.normal(32000, 8000, (128, 128)).clip(0, 65535).astype(np.uint16)
    answer = analysis.image_quality(Planes(good[None, None], ["SEM"]))
    assert answer["problems"] == [] and not answer["unusable"]
    white = good.copy()
    white[:, :80] = 65535
    answer = analysis.image_quality(Planes(white[None, None], ["SEM"]))
    assert answer["unusable"] and "white" in answer["problems"][0]
    assert "plane_sharpness" not in answer

    stack = _stack()
    stack.fluorescence, stack.full_value = True, 65535.0
    answer = analysis.image_quality(stack)
    assert answer["sharpest_plane"] == 2
    assert len(answer["plane_sharpness"]) == 5
    # A fluorescence channel has no beam targets: its range and saturation.
    assert "problems" not in answer and answer["saturated"] == 0
    assert answer["percentiles"]["median"] == pytest.approx(100, abs=1)
    assert 0 < answer["range_used"] < 0.02


def test_a_stack_on_disk_reads_as_its_channels_and_planes(tmp_path):
    data = _stack().data
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="gray",
    )
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(64, 64),
        channels=[channel],
        z_positions=[i * 3e-7 for i in range(5)],
    )
    path = str(tmp_path / "stack.ome.tiff")
    FluorescenceImage(data=data, metadata=metadata).save(path)
    planes = analysis.load_planes(path)
    assert planes.names == ["GFP"] and planes.planes == 5
    assert planes.pixel_size == pytest.approx(1e-7)
    assert planes.z_step == pytest.approx(3e-7)
    assert planes.full_value == 65535
