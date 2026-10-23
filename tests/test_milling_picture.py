"""Milling stages drawn as a picture (fibsem.milling.patterning.picture)."""

import numpy as np
from PIL import Image

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.picture import milling_picture
from fibsem.milling.patterning.plotting import COLOURS
from fibsem.structures import FibsemImage, FibsemRectangle


def _rgb(colour):
    return np.asarray(Image.new("RGB", (1, 1), colour)).reshape(3).astype(int)


def _has(pixels, colour, tolerance=60):
    return (np.abs(pixels.astype(int) - _rgb(colour)).max(axis=-1) < tolerance).any()


def _frame():
    image = FibsemImage.generate_blank_image(resolution=(600, 400), hfw=150e-6)
    image.data[:] = 128
    return image


def test_the_stages_are_drawn_in_the_milling_view_s_colours_with_the_bars():
    stage = FibsemMillingStage(name="Rough")
    picture = milling_picture(_frame(), [stage])
    assert picture.shape[1] == 600 and picture.shape[0] > 400  # the metadata bar
    assert _has(picture[100:300, 200:400], COLOURS[0])


def test_background_stages_are_black_and_switches_draw_what_they_name():
    image = _frame()
    stage = FibsemMillingStage(name="Rough")
    plain = milling_picture(image, [], [stage], metadata_bar=False)
    assert plain.shape[:2] == (400, 600)
    assert _has(plain[100:300, 200:400], "black", 30)
    assert not _has(plain[100:300, 200:400], COLOURS[0], 40)

    switched = milling_picture(
        image,
        [stage],
        scan_direction=True,
        alignment_area=FibsemRectangle(0.1, 0.1, 0.8, 0.8),
        metadata_bar=False,
    )
    unswitched = milling_picture(image, [stage], metadata_bar=False)
    assert _has(switched[50:70, 50:550], "limegreen", 40)  # the alignment area
    assert not _has(unswitched[50:70, 50:550], "limegreen", 40)
    # The scan arrow down the pattern's middle.
    centre = np.abs(
        switched[150:250, 295:305].astype(int) - unswitched[150:250, 295:305]
    )
    assert centre.max() > 60
