"""Tests for `fibsem.util.filename`."""

from fibsem.util.filename import (
    _get_basename,
    _get_basename_and_extension,
    _get_extension,
    get_unique_filename,
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


def test_get_unique_filename_suffixes_around_an_existing_file(tmp_path):
    """The `-1` goes before the extension, not after it, which needs `_get_basename`."""
    target = tmp_path / "image.ome.tiff"

    # Nothing there yet: the name is used as given.
    assert get_unique_filename(str(target)) == str(target)

    target.write_bytes(b"")
    assert get_unique_filename(str(target)) == str(tmp_path / "image-1.ome.tiff")

    (tmp_path / "image-1.ome.tiff").write_bytes(b"")
    assert get_unique_filename(str(target)) == str(tmp_path / "image-2.ome.tiff")
