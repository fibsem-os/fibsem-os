"""Tests for `fibsem.util.filename`.

The point of `remove_suffix` is that it works on Python 3.8, which `str.removesuffix`
does not. That is a *runtime* difference -- a module using `str.removesuffix` imports
fine on 3.8 and only raises when the line actually runs -- so nothing but a test that
exercises the call catches a regression here.
"""

import sys

import pytest

from fibsem.util.filename import (
    _get_basename,
    _get_basename_and_extension,
    _get_extension,
    get_unique_filename,
    remove_suffix,
)


@pytest.mark.parametrize(
    "text, suffix, expected",
    [
        ("image.ome.tiff", ".ome.tiff", "image"),
        ("image.tif", ".tif", "image"),
        # No match: the name comes back untouched rather than losing its tail.
        ("image.tif", ".ome.tiff", "image.tif"),
        ("image", ".tif", "image"),
        # Only the *trailing* occurrence goes; a suffix that also appears earlier in
        # the name stays put.
        ("tif.image.tif", ".tif", "tif.image"),
        # Empty suffix is the case a naive `text[:-len(suffix)]` gets wrong: it would
        # slice to `text[:0]` and return "".
        ("image.tif", "", "image.tif"),
        ("", ".tif", ""),
        # The whole string being the suffix leaves nothing behind.
        (".tif", ".tif", ""),
    ],
)
def test_remove_suffix(text, suffix, expected):
    assert remove_suffix(text, suffix) == expected


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="str.removesuffix requires Python 3.9"
)
@pytest.mark.parametrize(
    "text, suffix",
    [
        ("image.ome.tiff", ".ome.tiff"),
        ("image.tif", ".ome.tiff"),
        ("image.tif", ""),
        ("", ".tif"),
        (".tif", ".tif"),
        ("tif.image.tif", ".tif"),
    ],
)
def test_remove_suffix_matches_str_removesuffix(text, suffix):
    """On 3.9+ the helper must be indistinguishable from the builtin it stands in for.

    This is what stops the two implementations drifting: the 3.8 jobs prove the helper
    runs at all, and this proves it runs the same way the builtin would.
    """
    assert remove_suffix(text, suffix) == text.removesuffix(suffix)


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
    """The `-1` goes before the extension, not after it, which needs `_get_basename`
    -- the one caller that puts `remove_suffix` on a real code path."""
    target = tmp_path / "image.ome.tiff"

    # Nothing there yet: the name is used as given.
    assert get_unique_filename(str(target)) == str(target)

    target.write_bytes(b"")
    assert get_unique_filename(str(target)) == str(tmp_path / "image-1.ome.tiff")

    (tmp_path / "image-1.ome.tiff").write_bytes(b"")
    assert get_unique_filename(str(target)) == str(tmp_path / "image-2.ome.tiff")
