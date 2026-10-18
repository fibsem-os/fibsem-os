"""Detection through a registered model (`fibsem.ai.detection`): the features
found in its segmentation, named by the model that made it."""

import numpy as np
import pytest

from fibsem.ai import detection
from fibsem.ai.models import ModelManager, ModelRegistry


def _model():
    return ModelManager(ModelRegistry()).get("fake-segmentation")


def test_features_are_found_in_the_models_mask():
    model = _model()
    image = np.zeros((96, 128), dtype=np.uint8)
    det, result = detection.detect(
        model, image, ["ImageCentre", "LamellaCentre"], pixel_size=10e-9
    )
    centre, lamella = det.features
    assert (centre.px.x, centre.px.y) == (64, 48)
    assert centre.feature_m.x == 0 and centre.feature_m.y == 0
    assert lamella.name == "LamellaCentre" and lamella.px is not None
    assert det.checkpoint == "fake-segmentation@0.1.0"
    assert det.pixelsize == 10e-9
    assert np.array_equal(det.mask, result.mask)
    # The mask drawn in the card's colours, background black.
    cells = result.mask == 1
    assert cells.any()
    assert np.all(det.rgb[cells] == [255, 0, 0])  # Cell is #ff0000
    assert not det.rgb[result.mask == 0].any()


def test_an_unknown_feature_or_no_pixel_size_is_refused():
    model = _model()
    with pytest.raises(ValueError, match="no feature 'Nope'"):
        detection.detect(model, np.zeros((8, 8)), ["Nope"], pixel_size=1e-9)
    with pytest.raises(ValueError, match="pixel size"):
        detection.detect(model, np.zeros((8, 8)), ["ImageCentre"])


def test_the_workflows_detection_runs_through_the_registered_model(tmp_path):
    """take_image_and_detect_features names its model by checkpoint; a
    registered model of that name is the one it runs, loaded once."""
    from fibsem import utils
    from fibsem.ai.models import default_manager
    from fibsem.detection.detection import ImageCentre, take_image_and_detect_features
    from fibsem.structures import BeamType, ImageSettings

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    settings = ImageSettings(
        beam_type=BeamType.ELECTRON,
        resolution=[192, 128],
        hfw=150e-6,
        path=str(tmp_path),
    )
    det = take_image_and_detect_features(
        microscope, settings, [ImageCentre()], checkpoint="fake-segmentation"
    )
    assert det.checkpoint == "fake-segmentation@0.1.0"
    assert (det.features[0].px.x, det.features[0].px.y) == (96, 64)
    assert det.fibsem_image is not None and det.mask.shape == (128, 192)
    model = default_manager().get("fake-segmentation")
    take_image_and_detect_features(
        microscope, settings, [ImageCentre()], checkpoint="fake-segmentation"
    )
    assert default_manager().get("fake-segmentation") is model


def test_a_detection_names_its_registered_model_with_its_version():
    from fibsem.detection.review import detection_provenance

    det, _ = detection.detect(
        _model(), np.zeros((8, 8)), ["ImageCentre"], pixel_size=1e-9
    )
    assert detection_provenance(det)["proposer"] == "fake-segmentation@0.1.0"
    det.checkpoint = "models/autolamella-mega-20240107.pt"
    assert detection_provenance(det)["proposer"] == "autolamella-mega-20240107"
