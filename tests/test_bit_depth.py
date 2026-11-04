"""16-bit images: the beam's bit_depth parameter is the default, an image's own
bit_depth sets it, and what reads images keeps the extra bits instead of wrapping
them into uint8."""

import logging

import numpy as np
import pytest

from fibsem.autofunctions.gamma import auto_gamma
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    ImageSettings,
    OverviewAcquisitionSettings,
)


def _settings(**kwargs) -> ImageSettings:
    return ImageSettings(
        beam_type=BeamType.ELECTRON,
        resolution=(64, 64),
        hfw=50e-6,
        dwell_time=1e-9,
        **kwargs,
    )


def test_an_image_s_bit_depth_sets_the_beam_and_stays(microscope):
    beam = microscope.imaging.beam(BeamType.ELECTRON)
    assert beam.bit_depth.get_value() == 8
    assert microscope.acquire_image(_settings(bit_depth=16)).data.dtype == np.uint16
    assert beam.bit_depth.get_value() == 16
    # None takes the beam's own: still 16
    assert microscope.acquire_image(_settings()).data.dtype == np.uint16
    assert microscope.acquire_image(_settings(bit_depth=8)).data.dtype == np.uint8


def test_a_beam_without_the_setting_warns(microscope, monkeypatch, caplog):
    beam = microscope.imaging.beam(BeamType.ELECTRON)
    parameters = {k: v for k, v in beam.parameters.items() if k != "bit_depth"}
    monkeypatch.setattr(type(beam), "parameters", property(lambda self: parameters))
    with caplog.at_level(logging.WARNING):
        image = microscope.acquire_image(_settings(bit_depth=16))
    assert image.data.dtype == np.uint8
    assert "no bit depth setting" in caplog.text


def test_a_16_bit_overview_stitches_without_wrapping(microscope, tmp_path):
    microscope.imaging.beam(BeamType.ELECTRON).bit_depth.set_value(16)
    answer = microscope.imaging.acquire_overview(
        OverviewAcquisitionSettings(
            image_settings=_settings(), nrows=2, ncols=2, overlap=0.0
        ),
        path=str(tmp_path),
    )
    image = answer["image"]
    assert image.data.dtype == np.uint16
    assert image.data.max() > 255  # the tiles' own values, not wrapped


def test_a_16_bit_focus_stack_stays_16_bit(microscope):
    microscope.imaging.beam(BeamType.ELECTRON).bit_depth.set_value(16)
    image = microscope.imaging.focus_stacked_image(
        _settings(), n_steps=3, auto_focus=False
    )
    assert image.data.dtype == np.uint16


def test_a_16_bit_thumbnail_is_its_top_byte(tmp_path):
    from PIL import Image

    from fibsem.imaging.thumbnail import write_thumbnail

    eight = np.arange(256, dtype=np.uint8).reshape(16, 16)
    sixteen = eight.astype(np.uint16) * 257
    a = write_thumbnail(eight, tmp_path / "a.png")
    b = write_thumbnail(sixteen, tmp_path / "b.png")
    assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b)))


def test_autogamma_corrects_a_16_bit_image_as_its_8_bit_one():
    eight = np.full((32, 32), 40, dtype=np.uint8)
    sixteen = eight.astype(np.uint16) * 257
    out8 = auto_gamma(FibsemImage(data=eight), method="autogamma").data
    out16 = auto_gamma(FibsemImage(data=sixteen), method="autogamma").data
    assert out16.dtype == np.uint16
    assert out16.astype(float) / 65535 == pytest.approx(
        out8.astype(float) / 255, abs=2e-3
    )
