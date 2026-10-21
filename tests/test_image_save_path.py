"""Image settings read without a folder have none, and saving with none is refused
with a message rather than a TypeError from os.path.join."""

import os

import pytest

from fibsem.services.imaging import image_save_path
from fibsem.structures import BeamType, ImageSettings


def test_settings_read_without_a_path_have_none():
    assert ImageSettings.from_dict({}).path is None


def test_saving_with_no_folder_is_refused():
    with pytest.raises(ValueError, match="image_settings.path"):
        image_save_path(ImageSettings(save=True))


def test_the_save_path_is_the_folder_filename_and_beam():
    settings = ImageSettings(path="/data", filename="ref", beam_type=BeamType.ION)
    assert image_save_path(settings) == os.path.join("/data", "ref_ib")
