"""Brightness and contrast from the image: the stepper against a simulated detector,
and `Imaging.auto_brightness_contrast` on Demo, whose detector clips like one."""

import numpy as np
import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.autofunctions.brightness_contrast import (
    BrightnessContrastError,
    BrightnessContrastResult,
    BrightnessContrastSettings,
    BrightnessContrastStepper,
    measure,
    problems,
    unusable,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.drivers.demo import simulator
from fibsem.structures.settings import BeamType

SETTINGS = BrightnessContrastSettings()


def _uint8(values):
    return (np.clip(np.asarray(values), 0, 1) * 255).round().astype(np.uint8)


def test_measure_counts_pixels_at_each_limit():
    data = _uint8(np.r_[np.zeros(10), np.ones(30), np.full(60, 0.5)])
    stats = measure(data)
    assert stats.black == pytest.approx(0.1)
    assert stats.white == pytest.approx(0.3)
    assert stats.median == pytest.approx(0.5, abs=0.01)


def test_a_mostly_white_image_is_unusable_and_a_good_one_has_no_problems():
    rng = np.random.default_rng(0)
    good = measure(_uint8(rng.normal(0.5, 0.12, 10_000)))
    white = measure(_uint8(rng.normal(0.95, 0.12, 10_000)))
    assert problems(good, SETTINGS) == []
    assert not unusable(good, SETTINGS)
    assert unusable(white, SETTINGS)
    assert any("white" in p for p in problems(white, SETTINGS))


def test_thin_clipped_edges_dont_count_but_a_clipped_area_does():
    data = np.full((200, 300), 0.5)
    data[50, :] = data[:, 150] = 1.0  # one-pixel bright lines: 0.8% of pixels
    thin = measure(_uint8(data))
    assert thin.white == 0.0 and thin.white_total > 0.005
    data[100:120, 20:40] = 1.0  # and a 20 px saturated patch
    patch = measure(_uint8(data))
    assert patch.white == pytest.approx(400 / data.size, rel=0.05)
    data[:, :20] = 1.0  # past max_clipped_total, every clipped pixel counts
    assert measure(_uint8(data)).white == pytest.approx(
        measure(_uint8(data)).white_total
    )


class _Detector:
    """A detector more sensitive than the stepper's first guess, with an offset
    pivot: gain e^(8(c - 0.5)) about grey 0.3, and twice the offset per unit
    brightness. The scene: textured grey, a dark trench outlined by thin bright
    edges, and two bright 15 px spots."""

    def __init__(self, seed=0, sensitivity=2.0):
        self.sensitivity = sensitivity
        rng = np.random.default_rng(seed)
        scene = 0.45 + 0.06 * rng.standard_normal((128, 192))
        scene[40:90, 60:130] = 0.25
        scene[40:42, 60:130] = scene[88:90, 60:130] = 1.2
        scene[40:90, 60:62] = scene[40:90, 128:130] = 1.2
        scene[10:25, 10:25] = scene[100:115, 160:175] = 0.75
        self.scene = scene
        self.noise = rng.normal(0, 0.01, scene.shape)

    def image(self, brightness, contrast):
        signal = np.exp(8 * (contrast - 0.5)) * (self.scene - 0.3)
        return _uint8(signal + 0.5 + self.sensitivity * (brightness - 0.5) + self.noise)


@pytest.mark.parametrize(
    "start",
    [(0.95, 0.5), (0.05, 0.5), (0.5, 0.8), (0.5, 0.05), (0.8, 0.8), (0.2, 0.25)],
    ids=["white", "black", "high gain", "flat", "white, high gain", "dark, flat"],
)
def test_the_stepper_reaches_a_good_image_within_its_images(start):
    detector = _Detector()
    stepper = BrightnessContrastStepper(SETTINGS)
    x = np.array(start)
    for n in range(1, SETTINGS.max_images + 1):
        stats = measure(detector.image(*x))
        if not problems(stats, SETTINGS):
            break
        x = stepper.next(x, stats)
    assert problems(stats, SETTINGS) == [], (start, n, stats)


@pytest.mark.parametrize("start", [(0.95, 0.5), (0.5, 0.8), (0.2, 0.25), (0.6, 0.5)])
def test_a_detector_more_sensitive_than_assumed_does_not_swing(start):
    """2026-10-19 run: black, saturated, black... for all six images. Here the
    brightness moves the image 2.5x more than the stepper first assumes."""
    detector = _Detector(sensitivity=5.0)
    stepper = BrightnessContrastStepper(SETTINGS)
    x = np.array(start)
    for n in range(1, SETTINGS.max_images + 1):
        stats = measure(detector.image(*x))
        if not problems(stats, SETTINGS):
            break
        x = stepper.next(x, stats)
    assert problems(stats, SETTINGS) == [], (start, n, stats)


# -- on Demo -----------------------------------------------------------------------


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    yield microscope
    microscope.stop_acquisition()


def _set(beam, brightness, contrast):
    beam.detector_brightness.set_value(brightness)
    beam.detector_contrast.set_value(contrast)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_a_white_image_comes_back_and_the_scan_is_put_back(microscope, beam_type):
    beam = microscope.beams[beam_type]
    resolution, dwell_time = beam.resolution.get_value(), beam.dwell_time.get_value()
    _set(beam, 0.95, 0.5)
    assert measure(beam.acquire().data).white > 0.5

    result = microscope.imaging.auto_brightness_contrast(beam_type)

    assert result.converged and not result.started_from_last
    assert result.n_images <= SETTINGS.max_images
    assert beam.detector_brightness.get_value() == pytest.approx(result.brightness)
    assert beam.detector_contrast.get_value() == pytest.approx(result.contrast)
    assert problems(measure(beam.acquire().data), SETTINGS) == []
    assert beam.resolution.get_value() == resolution
    assert beam.dwell_time.get_value() == dwell_time


def test_the_next_run_at_the_same_current_starts_from_the_last(microscope):
    beam = microscope.beams[BeamType.ION]
    _set(beam, 0.9, 0.8)
    microscope.imaging.auto_brightness_contrast(BeamType.ION)

    _set(beam, 0.1, 0.9)
    result = microscope.imaging.auto_brightness_contrast(BeamType.ION)
    assert result.started_from_last
    assert result.n_images == 1

    _set(beam, 0.1, 0.9)
    fresh = microscope.imaging.auto_brightness_contrast(
        BeamType.ION, BrightnessContrastSettings(start_from_last=False)
    )
    assert not fresh.started_from_last
    assert fresh.n_images > 1


def test_a_detector_that_stays_white_raises_rather_than_end_on_it(
    microscope, monkeypatch
):
    events = []
    microscope.session.record_signal.connect(lambda event: events.append(event))
    monkeypatch.setattr(
        simulator,
        "detector_response",
        lambda data, brightness, contrast: np.full_like(data, 255),
    )
    with pytest.raises(BrightnessContrastError, match="unusable") as raised:
        microscope.imaging.auto_brightness_contrast(BeamType.ION)
    assert [e for e in events if "brightness_contrast" in str(e)], events
    result = raised.value.result
    assert not result.usable and not result.converged
    assert result.n_images == SETTINGS.max_images


def test_the_result_saves_its_images_and_plot_and_loads_back(microscope, tmp_path):
    beam = microscope.beams[BeamType.ION]
    _set(beam, 0.95, 0.5)
    result = microscope.imaging.auto_brightness_contrast(
        BeamType.ION, path=str(tmp_path), name="bc"
    )

    assert (result.initial_brightness, result.initial_contrast) == pytest.approx(
        (0.95, 0.5)
    )
    assert result.beam_type == "ION" and result.usable
    assert all(it.image is not None for it in result.iterations)
    saved = tmp_path / "bc"
    assert (saved / "plot.png").stat().st_size > 0
    assert len(list(saved.glob("iter_*.tif"))) == result.n_images

    loaded = BrightnessContrastResult.load(saved)
    assert loaded.to_dict() == result.to_dict()
    np.testing.assert_array_equal(loaded.image, result.image)
    assert loaded.settings == result.settings


def test_a_failed_run_is_saved_too(microscope, monkeypatch, tmp_path):
    monkeypatch.setattr(
        simulator,
        "detector_response",
        lambda data, brightness, contrast: np.full_like(data, 255),
    )
    with pytest.raises(BrightnessContrastError):
        microscope.imaging.auto_brightness_contrast(
            BeamType.ION, path=str(tmp_path), name="bc"
        )
    assert not BrightnessContrastResult.load(tmp_path / "bc").usable


def test_stop_cancels_and_puts_the_detector_back(microscope):
    beam = microscope.beams[BeamType.ION]
    _set(beam, 0.9, 0.8)
    acquire = beam.acquire

    def acquire_then_stop(*args, **kwargs):
        microscope.imaging.stop()
        return acquire(*args, **kwargs)

    beam.acquire = acquire_then_stop
    with pytest.raises(OperationCancelledError):
        microscope.imaging.auto_brightness_contrast(BeamType.ION)
    assert beam.detector_brightness.get_value() == pytest.approx(0.9)
    assert beam.detector_contrast.get_value() == pytest.approx(0.8)
