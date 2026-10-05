"""Compressed TIFFs open: the files people bring from other microscopes are.

tifffile hands every codec but zlib to imagecodecs, and until it was declared nothing
installed it: a fresh install could not open an LZW TIFF at all. A Zeiss ZEN export and
a METEOR ImageJ stack are both LZW. These fail at the write, not only the read, on an
install without it -- which is the point: CI installs from pyproject.toml. No Qt.
"""

import numpy as np
import pytest
import tifffile

from fibsem.fm.structures import FluorescenceImage

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_imagecodecs_is_installed_with_fibsem():
    import imagecodecs  # noqa: F401 - the dependency itself


@pytest.mark.parametrize("compression", ["lzw", "zlib"])
def test_a_compressed_tiff_opens(tmp_path, compression):
    data = (np.arange(3 * 24 * 32) % 4000).astype(np.uint16).reshape(3, 24, 32)
    path = str(tmp_path / f"{compression}.tif")
    tifffile.imwrite(path, data, photometric="minisblack", compression=compression)

    image = FluorescenceImage.load(path)

    np.testing.assert_array_equal(image.data[0], data)


def test_an_lzw_rgb_export_opens(tmp_path):
    """The shape of a Zeiss ZEN export: 8-bit RGB, LZW."""
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    path = str(tmp_path / "export.tif")
    tifffile.imwrite(path, rgb, photometric="rgb", compression="lzw")

    with tifffile.TiffFile(path) as tif:
        assert tif.pages.first.compression.name == "LZW"
        np.testing.assert_array_equal(tif.asarray(), rgb)
