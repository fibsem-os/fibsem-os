"""The model library (`fibsem.ai.models`): the registry, cards, the manager and
the fake model, without a microscope or a server."""

import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from fibsem.ai.models import (
    InputMismatch,
    ModelCard,
    ModelEntry,
    ModelManager,
    ModelNotFound,
    ModelRegistry,
    ModelUnavailable,
)
from fibsem.ai.models.backends.fake import FakeSegmentationModel
from fibsem.ai.models.card import find_card


def _manager(*entries):
    return ModelManager(ModelRegistry([*ModelRegistry().entries(), *entries]))


def test_the_fake_model_is_registered_and_the_segmentation_default():
    registry = ModelRegistry()
    assert registry.default_for("segmentation").name == "fake-segmentation"
    status = _manager().status("fake-segmentation")
    assert status["status"] == "ready" and status["loaded"] is False
    assert status["classes"][:3] == ["background", "Cell", "Grid Bar"]
    assert status["card_source"] == "backend"


def test_a_model_loads_once_per_settings():
    manager = _manager()
    first = manager.get("fake-segmentation")
    assert manager.get("fake-segmentation") is first
    assert manager.get("fake-segmentation", seed=1) is not first
    assert manager.status("fake-segmentation")["loaded"] is True
    manager.unload("fake-segmentation")
    assert manager.status("fake-segmentation")["loaded"] is False


def test_segment_answers_classes_and_the_model_that_made_it():
    result = (
        _manager()
        .get("fake-segmentation")
        .segment(np.zeros((256, 384)), pixel_size=1e-6, beam="ELECTRON")
    )
    assert result.mask.shape == (256, 384)
    assert result.fraction("Grid Bar") > 0.1
    assert result.count("Cell") >= 1
    assert result.area_um2("Grid Bar") == pytest.approx(
        result.fraction("Grid Bar") * 256 * 384, rel=1e-6
    )
    answer = result.to_dict()
    assert answer["model"]["model"] == "fake-segmentation"
    assert answer["model"]["version"] == "0.1.0"
    assert answer["model"]["config"] == {"seed": 0}
    assert "background" not in answer["classes"]


def test_the_same_seed_gives_the_same_mask():
    image = np.zeros((128, 128))
    one = _manager().get("fake-segmentation").segment(image).mask
    two = _manager().get("fake-segmentation").segment(image).mask
    assert np.array_equal(one, two)


def test_unknown_names_settings_and_versions_are_refused():
    manager = _manager()
    with pytest.raises(ModelNotFound):
        manager.get("no-such-model")
    with pytest.raises(ModelUnavailable):
        manager.get("fake-segmentation@9.9")
    with pytest.raises(ValueError, match="checkpoint"):
        manager.get("fake-segmentation", checkpoint="other.pth")
    assert manager.get("fake-segmentation@0.1.0").card.version == "0.1.0"


def test_an_image_outside_the_card_is_refused(tmp_path):
    card = FakeSegmentationModel.default_card()
    card.input.beams = ["ELECTRON"]
    card.input.pixel_size_range = (100e-9, 500e-9)
    folder = tmp_path / "scale-limited"
    folder.mkdir()
    (folder / "fibsem_model.json").write_text(json.dumps(card.to_dict()))
    model = _manager(ModelEntry("scale-limited", str(folder))).get("scale-limited")
    assert model.card_source == "model"
    with pytest.raises(InputMismatch, match="ION"):
        model.segment(np.zeros((64, 64)), pixel_size=200e-9, beam="ION")
    with pytest.raises(InputMismatch, match="1200 nm"):
        model.segment(np.zeros((64, 64)), pixel_size=1.2e-6, beam="ELECTRON")
    unknown = model.segment(np.zeros((64, 64)), beam="ELECTRON")
    assert unknown.warnings


def test_the_card_is_found_inside_then_beside_then_on_the_entry(tmp_path):
    card = FakeSegmentationModel.default_card().to_dict()
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"")
    assert find_card(str(checkpoint)) == (None, "")
    found, source = find_card(str(checkpoint), card)
    assert source == "registry" and found.id == "fake-segmentation"
    (tmp_path / "legacy.card.json").write_text(json.dumps(card))
    assert find_card(str(checkpoint), card)[1] == "beside"
    assert ModelCard.from_dict(card) == FakeSegmentationModel.default_card()


def test_the_registry_file_adds_and_overrides(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  grid-bars:\n"
        "    ref: D:/models/grid-bars\n"
        "    default_for: [segmentation]\n"
        "    agent: false\n"
        "  broken: {}\n"
    )
    registry = ModelRegistry.load(str(path))
    assert [entry.name for entry in registry.entries()] == [
        "fake-segmentation",
        "grid-bars",
    ]
    assert registry.default_for("segmentation").name == "grid-bars"
    status = ModelManager(registry).status("grid-bars")
    assert status["status"] == "unavailable"


def test_a_result_saves_its_mask_overlay_and_record(tmp_path):
    image = np.random.default_rng(0).random((64, 96))
    result = _manager().get("fake-segmentation").segment(image, pixel_size=1e-6)
    paths = result.save(str(tmp_path), "grid-01", image=image)

    assert np.array_equal(tifffile.imread(paths["mask_path"]), result.mask)
    with open(paths["result_path"]) as f:
        assert json.load(f)["model"]["version"] == "0.1.0"
    with Image.open(paths["overlay_path"]) as overlay:
        assert overlay.size == (96, 64)


def test_a_tiled_run_calls_the_backend_once_per_tile():
    model = _manager().get("fake-segmentation", seed=3)
    shapes = []
    predict = model.backend.predict
    model.backend.predict = lambda window: (
        shapes.append(window.shape) or predict(window)
    )
    result = model.segment(np.zeros((100, 150)), pixel_size=1e-6, tile=(50, 64))
    assert result.mask.shape == (100, 150)
    assert sorted(set(shapes)) == [(50, 22), (50, 64)]
    assert len(shapes) == 6


def test_regions_are_largest_first_with_centres():
    result = (
        _manager()
        .get("fake-segmentation")
        .segment(np.zeros((256, 256)), pixel_size=1e-6)
    )
    regions = result.regions("Cell")
    assert regions and len(regions) == min(result.count("Cell"), 20)
    areas = [region["area_um2"] for region in regions]
    assert areas == sorted(areas, reverse=True)
    x, y = regions[0]["centre_px"]
    assert 0 <= x < 256 and 0 <= y < 256


@pytest.mark.parametrize(
    "backend, packages", [("nnunet", "nnunetv2"), ("smp", "torch")]
)
def test_a_framework_model_says_what_it_is_missing_without_importing_it(
    backend, packages
):
    import importlib.util

    card = {**FakeSegmentationModel.default_card().to_dict(), "backend": backend}
    entry = ModelEntry("real", "D:/models/real", card=card)
    status = _manager(entry).status("real")
    assert status["backend"] == backend and status["card_source"] == "registry"
    if importlib.util.find_spec(packages) is None:
        assert status["status"] == "missing packages"
        assert packages in status["missing"]
        with pytest.raises(ModelUnavailable, match=packages):
            _manager(entry).get("real")
