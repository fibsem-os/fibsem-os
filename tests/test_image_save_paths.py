"""Saving an image must not reconstruct another machine's directory tree.

`ImageSettings.path` is an absolute directory, and it travels inside the file. So a
loaded image carries wherever it was acquired -- on Windows, `D:\\SharedData\\<name>\\...`
-- and `FibsemImage.save()` falls back to it when no path is given. It used to
`os.makedirs` that unconditionally, which meant opening a colleague's image and
re-saving it silently created their layout locally.

A path the caller passes in is theirs to create. One that arrived in a file is not.
"""

import os

import numpy as np
import pytest

from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)


def _image(path, filename: str = "an-image") -> FibsemImage:
    """An image whose metadata names a directory, as an acquired one does."""
    return FibsemImage(
        data=np.zeros((8, 8), dtype=np.uint8),
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(
                beam_type=BeamType.ELECTRON, path=str(path), filename=filename
            ),
            pixel_size=Point(1e-9, 1e-9),
            microscope_state=MicroscopeState(),
        ),
    )


def test_saving_uses_the_recorded_directory_when_it_exists(tmp_path) -> None:
    """The ordinary case: a task sets the path, then acquires and saves into it."""
    written = _image(tmp_path).save()

    assert os.path.isfile(written)
    assert os.path.dirname(written) == str(tmp_path)


def test_saving_does_not_create_a_directory_that_came_out_of_a_file(tmp_path) -> None:
    """The footgun. The directory must not spring into existence."""
    absent = tmp_path / "SharedData" / "someone-else" / "their-experiment"
    image = _image(absent)

    with pytest.raises(ValueError, match="does not exist"):
        image.save()

    assert not absent.exists()
    # ...and no partial tree either.
    assert not (tmp_path / "SharedData").exists()


def test_an_explicit_path_is_still_created(tmp_path) -> None:
    """A caller asking for a directory gets one. Only the recorded path is untrusted."""
    target = tmp_path / "deliberately" / "nested" / "output.tif"

    written = _image(tmp_path).save(str(target))

    assert os.path.isfile(written)


def test_the_error_says_where_the_directory_came_from(tmp_path) -> None:
    """Otherwise this reads as a bug in the caller rather than a stale record."""
    absent = tmp_path / "not-here"

    with pytest.raises(ValueError) as raised:
        _image(absent).save()

    message = str(raised.value)
    assert str(absent) in message
    assert "acquired" in message
    assert "provide a path" in message.lower()
