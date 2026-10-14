"""Images saved by an instrument's own software are read by its driver's reader,
through the registry, not by format code in the correlation package."""

import numpy as np
import pytest
import tifffile as tff

from fibsem.correlation.io import load_image_and_metadata
from fibsem.drivers.autoscript.image_files import read_image
from fibsem.drivers.registry import (
    DRIVER_PLUGINS,
    DriverEntry,
    image_readers,
    register_driver,
)

METADATA = (
    "[Image]\r\nResolutionX=8\r\nResolutionY=10\r\n[Scan]\r\nPixelWidth=2.5e-08\r\n"
)


def _image_with_databar() -> np.ndarray:
    image = np.full((12, 8), 5, np.uint8)
    image[10:] = 0  # the databar under the image starts with a blank row
    return image


def test_an_instrument_image_reads_with_its_pixel_size_and_without_its_databar(
    tmp_path,
):
    path = str(tmp_path / "instrument.tif")
    tff.imwrite(
        path, _image_with_databar(), extratags=[(34682, "s", 0, METADATA, True)]
    )

    image, pixel_size = load_image_and_metadata(path)

    assert image.shape == (10, 8)
    assert pixel_size == 2.5e-08


def test_a_plain_tiff_reads_without_a_pixel_size(tmp_path):
    path = str(tmp_path / "plain.tif")
    tff.imwrite(path, _image_with_databar())

    image, pixel_size = load_image_and_metadata(path)

    assert image.shape == (12, 8)
    assert pixel_size is None


def test_a_driver_reader_refuses_a_file_not_in_its_format(tmp_path):
    path = str(tmp_path / "plain.tif")
    tff.imwrite(path, _image_with_databar())

    with pytest.raises(ValueError):
        read_image(path)


def test_a_plugin_driver_brings_its_own_image_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(DRIVER_PLUGINS, "registered", dict(DRIVER_PLUGINS.registered))
    register_driver(
        DriverEntry(
            manufacturer="Elsewhere",
            microscope_class="elsewhere:Microscope",
            image_reader=f"{__name__}:_read_anything",
        )
    )
    assert ("Elsewhere", _read_anything) in image_readers()

    path = str(tmp_path / "plain.tif")
    tff.imwrite(path, _image_with_databar())
    image, pixel_size = load_image_and_metadata(path)
    assert pixel_size == 1e-9


def _read_anything(filename):
    return tff.imread(filename), 1e-9
