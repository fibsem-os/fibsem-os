"""Tests for `fibsem.util.filename`."""

from fibsem.util.filename import (
    _get_basename,
    _get_basename_and_extension,
    _get_extension,
)


def test_get_extension_treats_ome_tiff_as_one_extension():
    """`os.path.splitext` would only take `.tiff` off an OME-TIFF, leaving `.ome`."""
    assert _get_extension("image.ome.tiff") == ".ome.tiff"
    assert _get_extension("image.tif") == ".tif"
    assert _get_extension("image") == ""


def test_get_basename_strips_the_whole_extension():
    assert _get_basename("image.ome.tiff") == "image"
    assert _get_basename("image.tif") == "image"
    assert _get_basename("image") == "image"
    assert _get_basename_and_extension("image.ome.tiff") == ("image", ".ome.tiff")
