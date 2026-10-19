"""The file a recorded detection names: the one the task saved, named so the
record survives a moved experiment."""

import os

import numpy as np

from fibsem.detection.decided import detection_image_file
from fibsem.detection.detection import DetectedFeatures, LamellaCentre
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)


def _detection(save_to=None) -> DetectedFeatures:
    feature = LamellaCentre()
    feature.px = Point(10, 20)
    data = np.zeros((8, 8), dtype=np.uint8)
    detection = DetectedFeatures(
        features=[feature],
        image=data,
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
        fibsem_image=FibsemImage(
            data=data,
            metadata=FibsemImageMetadata(
                image_settings=ImageSettings(beam_type=BeamType.ION, hfw=8e-9),
                pixel_size=Point(1e-9, 1e-9),
                microscope_state=MicroscopeState(),
            ),
        ),
    )
    if save_to is not None:
        detection.fibsem_image.save(os.path.join(str(save_to), "ml-test_ib"))
    return detection


def test_it_names_the_file_the_task_saved(tmp_path):
    """The path actually written, beam suffix and extension included, and
    nothing new is written."""
    before = sorted(os.listdir(tmp_path))
    detection = _detection(save_to=tmp_path)
    after = sorted(os.listdir(tmp_path))

    stored = detection_image_file(detection, str(tmp_path))

    assert stored == "ml-test_ib.tif"
    assert after == sorted(before + [stored])
    assert sorted(os.listdir(tmp_path)) == after


def test_a_file_in_a_folder_under_the_item_is_named_portably(tmp_path):
    sub = tmp_path / "detections"
    sub.mkdir()

    stored = detection_image_file(_detection(save_to=sub), str(tmp_path))

    assert stored == "detections/ml-test_ib.tif"


def test_a_file_outside_the_item_is_named_in_full(tmp_path_factory):
    item = tmp_path_factory.mktemp("item")
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    stored = detection_image_file(_detection(save_to=elsewhere), str(item))

    assert os.path.isabs(stored)
    assert os.path.isfile(os.path.join(str(item), stored))


def test_an_image_never_saved_names_nothing(tmp_path):
    assert detection_image_file(_detection(), str(tmp_path)) == ""
