import numpy as np
import pytest

from fibsem import acquire, utils
from fibsem.structures import (
    FibsemImage,
    FibsemRectangle,
)


def test_reduced_area_acquisition():
    """Test the reduced area acquisition functionality of the acquire module."""
    # setup a demo microscope session
    microscope, settings = utils.setup_session(manufacturer="Demo")

    resolution = settings.image.resolution

    # acquire a full frame image
    image = acquire.acquire_image(microscope, settings.image)
    assert isinstance(image, FibsemImage)
    assert image.data.shape == (resolution[1], resolution[0])

    # acquire a reduced area image
    settings.image.reduced_area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    image = acquire.acquire_image(microscope, settings.image)
    assert isinstance(image, FibsemImage)
    assert image.data.shape == (resolution[1] // 2, resolution[0] // 2)


def test_focus_stacking_leaves_the_callers_settings_alone(tmp_path):
    """The strips set a reduced area and turn saving off on the settings they acquire
    with. They did it on the caller's, which then kept scanning the last strip."""
    microscope, settings = utils.setup_session(manufacturer="Demo")
    settings.image.path = str(tmp_path)
    settings.image.filename = "stacked"
    settings.image.save = True
    settings.image.reduced_area = None

    image = acquire.acquire_focus_stacked_image(
        microscope, settings.image, n_steps=3, auto_focus=False
    )

    assert isinstance(image, FibsemImage)
    assert settings.image.reduced_area is None
    assert settings.image.save is True
