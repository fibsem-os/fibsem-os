"""Measurements on saved images (`fibsem.analysis`): intensity at points
per plane, a line's profile and FWHM, and image quality."""

from pathlib import Path

import numpy as np
import pytest

from fibsem.analysis import measure as analysis
from fibsem.analysis.images import Planes, load_planes
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


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


def test_the_profile_s_summary_leaves_the_samples_out_unless_resampled():
    data = 50 + _spot((64, 64), 32, 32, 3.0, 1000)
    image = Planes(data=data[None, None], names=["SEM"], pixel_size=2e-9)
    result = analysis.line_profile(image, [8, 32], [56, 32])

    summary = analysis.profile_summary(result)
    assert "values" not in summary and "distance_px" not in summary
    assert summary["fwhm_px"] == result["fwhm_px"]
    assert summary["half_crossings_px"] == result["half_crossings_px"]
    assert {"min", "max", "baseline", "extremum_px", "length_m"} <= set(summary)

    resampled = analysis.profile_summary(result, points=5)
    assert resampled["distance_px"] == [0.0, 12.0, 24.0, 36.0, 48.0]
    assert resampled["values"][2] == pytest.approx(max(result["values"]))
    whole = analysis.profile_summary(result, points=500)
    assert whole["values"] == result["values"]
    with pytest.raises(ValueError, match="2 or more"):
        analysis.profile_summary(result, points=1)


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
    planes = load_planes(path)
    assert planes.names == ["GFP"] and planes.planes == 5
    assert planes.pixel_size == pytest.approx(1e-7)
    assert planes.z_step == pytest.approx(3e-7)
    assert planes.full_value == 65535


def test_a_plain_picture_reads_as_one_plane_with_no_scale(tmp_path):
    from PIL import Image

    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[5:10, 5:10] = 255
    path = str(tmp_path / "figure.png")
    Image.fromarray(rgb).save(path)
    planes = load_planes(path)
    assert planes.data.shape == (1, 1, 20, 30)
    assert planes.names == ["Image"] and planes.pixel_size is None
    assert planes.data[0, 0, 7, 7] == 255


def test_a_format_no_reader_reads_is_refused_with_the_readers_named(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="fluorescence stack, beam image, picture"):
        load_planes(str(path))


def test_the_tools_work_over_any_folder_and_refuse_paths_outside_it(tmp_path):
    from PIL import Image

    from fibsem.analysis.tools import ImageRefused, ImageTools

    root = tmp_path / "data"
    root.mkdir()
    picture = np.tile(np.arange(64, dtype=np.uint8) * 4, (32, 1))
    Image.fromarray(picture).save(root / "ramp.png")
    tools = ImageTools(lambda: str(root))

    answer = tools.line_profile("ramp.png", [0, 16], [63, 16])
    assert answer.measurement == "profile-0-16-63-16"
    assert Path(answer.path).parts[-2:] == ("drawn", "ramp-profile-0-16-63-16.png")
    assert "values" not in answer.result
    resampled = tools.line_profile("ramp.png", [0, 16], [63, 16], points=8)
    assert len(resampled.result["values"]) == 8
    assert resampled.result["values"][0] < resampled.result["values"][-1]

    with pytest.raises(ImageRefused) as outside:
        tools.image_quality("../elsewhere.png")
    assert outside.value.error_type == "not_allowed"
    with pytest.raises(ImageRefused) as closed:
        ImageTools(lambda: None).image_quality("ramp.png")
    assert closed.value.error_type == "no_experiment"


def test_a_shift_is_measured_between_two_files_with_no_microscope(tmp_path):
    from fibsem.analysis.tools import ImageRefused, ImageTools
    from fibsem.structures.image import FibsemImage
    from fibsem.structures.settings import MicroscopeState

    image = FibsemImage.generate_blank_image(resolution=(256, 192), random=True)
    image.metadata.microscope_state = MicroscopeState()
    image.save(str(tmp_path / "reference.tif"))
    image.data = np.roll(image.data, 6, axis=1)
    image.save(str(tmp_path / "moved.tif"))
    tools = ImageTools(lambda: str(tmp_path))

    answer = tools.measure_shift("reference.tif", "moved.tif", compare=True)
    assert abs(abs(answer.shift["shift_px"]["x"]) - 6) < 1
    assert set(answer.comparison) >= {"shifts_px", "agreement"}

    with pytest.raises(ImageRefused, match="method is one of"):
        tools.measure_shift("reference.tif", "moved.tif", method="guess")


def _beam(data, pixel_size=None, full_value=255.0):
    return Planes(
        data=data[None, None],
        names=["SEM"],
        pixel_size=pixel_size,
        full_value=full_value,
    )


def test_a_histogram_spans_the_file_s_range_and_says_what_is_clipped():
    data = np.zeros((40, 50), dtype=np.uint8)
    data[:, 25:] = 255  # half crushed, half saturated
    data[:10, :5] = 100
    answer = analysis.histogram(_beam(data), bins=16)

    assert answer["range"] == [0.0, 255.0]
    assert len(answer["counts"]) == 16 and sum(answer["counts"]) == data.size
    assert answer["clipped_high"] == pytest.approx(50.0)
    assert answer["clipped_low"] == pytest.approx(100 * (1000 - 50) / 2000)
    assert answer["percentiles"]["max"] == 255.0

    boxed = analysis.histogram(_beam(data), box=[0, 0, 5, 10])
    assert boxed["pixels"] == 50 and boxed["percentiles"]["min"] == 100.0
    with pytest.raises(ValueError, match="2x2"):
        analysis.histogram(_beam(data), box=[100, 100, 5, 5])
    with pytest.raises(ValueError, match="bins"):
        analysis.histogram(_beam(data), bins=1)


def test_a_power_spectrum_finds_finer_detail_in_a_sharper_image():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    scene = rng.normal(0, 200, (128, 128))
    noise = rng.normal(0, 1, (128, 128))
    sharp = 100 + gaussian_filter(scene, 1) + noise
    blurred = 100 + gaussian_filter(scene, 3) + noise

    fine = analysis.power_spectrum(_beam(sharp, pixel_size=2e-9))
    coarse = analysis.power_spectrum(_beam(blurred, pixel_size=2e-9))
    assert coarse["finest_period_px"] > fine["finest_period_px"]
    assert coarse["finest_period_m"] == pytest.approx(coarse["finest_period_px"] * 2e-9)
    assert 0.5 < fine["anisotropy"] < 2

    # Lines across x put their power along y: anisotropy far below 1.
    streaked = 100 + gaussian_filter(scene, (1, 4)) + noise
    assert analysis.power_spectrum(_beam(streaked))["anisotropy"] < 0.5
    # Noise alone holds no detail above its own floor.
    assert analysis.power_spectrum(_beam(noise))["cutoff"] is None
    with pytest.raises(ValueError, match="16x16"):
        analysis.power_spectrum(_beam(sharp), box=[0, 0, 10, 10])


def test_the_histogram_and_spectrum_tools_save_their_figures(tmp_path):
    from PIL import Image

    from fibsem.analysis.tools import ImageTools

    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(
        tmp_path / "noise.png"
    )
    tools = ImageTools(lambda: str(tmp_path))

    histogram = tools.histogram("noise.png", box=[0, 0, 32, 32])
    assert histogram.measurement == "histogram-box-0-0-32-32"
    assert Path(histogram.path).parts[-2:] == (
        "drawn",
        "noise-histogram-box-0-0-32-32.png",
    )
    spectrum = tools.power_spectrum("noise.png")
    assert Path(spectrum.path).name == "noise-spectrum.png"
    assert Path(spectrum.path).is_file() and spectrum.result["size_px"] == 64
    assert "radial_power" not in spectrum.result
    curve = tools.power_spectrum("noise.png", points=8).result
    assert len(curve["frequency"]) == len(curve["radial_power"]) == 8
    assert curve["frequency"][0] < curve["frequency"][-1] <= 0.5


def test_thickness_is_read_off_a_fib_image_in_any_folder(tmp_path):
    from fibsem.analysis.tools import ImageRefused, ImageTools
    from fibsem.structures.settings import BeamType, MicroscopeState
    from tests.test_lamella_thickness import band_image

    image = band_image(235)
    image.metadata.microscope_state = MicroscopeState()
    image.metadata.image_settings.beam_type = BeamType.ION
    image.save(str(tmp_path / "lamella_ib.tif"))
    tools = ImageTools(lambda: str(tmp_path))

    answer = tools.measure_thickness("lamella_ib.tif")
    assert answer["thickness_nm"]["mean"] == pytest.approx(235, abs=10)
    assert answer["centre_from"] == "image_centre"
    assert answer["window_from"] == "default" and answer["window_um"] == [10.0, 4.0]
    assert Path(answer["saved_to"]).parts[-2:] == ("thickness", "lamella_ib")
    assert "item_name" not in answer

    image.metadata.image_settings.beam_type = BeamType.ELECTRON
    image.save(str(tmp_path / "lamella_eb.tif"))
    with pytest.raises(ImageRefused, match="SEM image"):
        tools.measure_thickness("lamella_eb.tif")


class _EntryPoint:
    def __init__(self, target, name="lab-format"):
        self.name, self.value, self.dist = name, "lab:reader", None
        self._target = target

    def load(self):
        return self._target


@pytest.fixture
def lab_reader(monkeypatch):
    """A plugin's reader for its instrument's own format (here, raw uint16)."""
    from fibsem.analysis.images import READER_GROUP, Reader
    from fibsem.plugins import loader

    def read(path):
        data = np.fromfile(path, dtype=np.uint16).reshape(32, 64)
        return Planes(
            data=data[None, None],
            names=["Lab image"],
            pixel_size=4e-10,
            full_value=65535.0,
        )

    reader = Reader("lab image", lambda path: path.endswith(".lab"), read)
    loader.clear_cache()
    monkeypatch.setattr(
        loader,
        "_entry_points",
        lambda group: iter([_EntryPoint(reader)] if group == READER_GROUP else []),
    )
    yield reader
    loader.clear_cache()


def test_a_plugin_s_reader_reads_its_instrument_s_format(tmp_path, lab_reader):
    from fibsem.analysis.tools import ImageTools

    data = np.tile(np.arange(64, dtype=np.uint16) * 1000, (32, 1))
    data.tofile(tmp_path / "frame.lab")
    planes = load_planes(str(tmp_path / "frame.lab"))
    assert planes.names == ["Lab image"] and planes.pixel_size == 4e-10
    answer = ImageTools(lambda: str(tmp_path)).line_profile(
        "frame.lab", [0, 16], [63, 16]
    )
    assert answer.result["values"][0] < answer.result["values"][-1]


def test_a_plugin_entry_that_is_not_a_reader_is_left_out(monkeypatch):
    from fibsem.analysis.images import READERS, readers
    from fibsem.plugins import loader

    loader.clear_cache()
    monkeypatch.setattr(
        loader, "_entry_points", lambda group: iter([_EntryPoint(object())])
    )
    try:
        assert readers() == READERS
    finally:
        loader.clear_cache()


def test_the_tools_that_read_fibsem_metadata_refuse_another_instrument_s_file(tmp_path):
    from fibsem.analysis.tools import ImageRefused, ImageTools

    (tmp_path / "frame.mrc").write_bytes(b"\0" * 64)
    tools = ImageTools(lambda: str(tmp_path))
    for call in (
        lambda: tools.metadata("frame.mrc"),
        lambda: tools.pixel_to_stage("frame.mrc", px=[1, 1]),
        lambda: tools.stage_to_pixel(
            "frame.mrc", [{"x": 0, "y": 0, "z": 0, "r": 0, "t": 0}]
        ),
    ):
        with pytest.raises(ImageRefused) as refused:
            call()
        assert refused.value.error_type == "invalid"
        assert "isn't one (another instrument's image" in refused.value.message
