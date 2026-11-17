"""Labels for the simulator's images, read from the objects it drew."""

import json
import os

import numpy as np
import pytest
from PIL import Image

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.sim import labels
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")
RESOLUTION = (384, 256)


def _microscope(**sample):
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    microscope.system.sim["sample"] = {
        **microscope.system.sim.get("sample", {}),
        "enabled": True,
        "seed": 7,
        "noise_sigma": 0.0,
        "noise_fraction": 0.0,
        "charging": 0.0,
        "coincidence_offset": 0.0,
        **sample,
    }
    microscope._connection.setup_sample_scene()
    return microscope


@pytest.fixture
def microscope():
    microscope = _microscope()
    yield microscope
    microscope.disconnect()


@pytest.fixture
def ripped():
    microscope = _microscope(rip_fraction=0.3)
    yield microscope
    microscope.disconnect()


def _take(microscope, beam_type=BeamType.ELECTRON, hfw=300e-6):
    settings = ImageSettings(
        resolution=RESOLUTION,
        hfw=hfw,
        dwell_time=1e-6,
        save=False,
        beam_type=beam_type,
    )
    image, found = labels.acquire(microscope, settings)
    return image.data.astype(np.float32), found


def test_every_pixel_has_a_class_and_the_image_agrees(ripped):
    image, found = _take(ripped)
    assert found.classes.shape == image.shape == found.cells.shape
    assert found.classes.dtype == np.uint8 and found.cells.dtype == np.uint16
    known = [number for number, _, _ in labels.CLASSES]
    assert set(np.unique(found.classes)) <= set(known)
    bar = found.classes == labels.ID["grid bar"]
    ice = found.classes == labels.ID["ice"]
    gone = found.classes == labels.ID["broken hole or rip"]
    assert bar.mean() > 0.1 and ice.mean() > 0.05 and gone.sum() > 50
    # the classes are where the image shows them: nothing through a rip, and
    # a bar under the ice brighter than ice over film
    assert image[gone].mean() < 0.5 * image[ice].mean()
    assert image[bar].mean() > image[ice].mean()


def test_each_cell_has_its_own_number_and_a_box(microscope):
    _, found = _take(microscope)
    numbers = np.unique(found.cells[found.cells > 0])
    assert len(numbers) >= 2
    cell = found.classes == labels.ID["cell"]
    assert cell[found.cells > 0].mean() > 0.9, "a numbered pixel is a cell's"
    boxes = [b for b in found.boxes() if b["class"] == "cell"]
    assert boxes
    for box in boxes:
        x0, y0, x1, y1 = box["box"]
        rows, cols = np.nonzero(found.cells == box["cell"])
        assert (x0, y0, x1, y1) == (cols.min(), rows.min(), cols.max(), rows.max())
        assert box["area"] == len(rows)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_a_milled_trench_is_labelled_where_the_image_is_dark(microscope, beam_type):
    microscope.move_to_milling_angle(15.0)
    _, before = _take(microscope, beam_type, hfw=50e-6)
    assert not (before.classes == labels.ID["milled trench"]).any()
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=12e-6, height=6e-6, depth=10e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    image, after = _take(microscope, beam_type, hfw=50e-6)
    trench = after.classes == labels.ID["milled trench"]
    faces = np.isin(after.classes, (labels.ID["cut face"], labels.ID["cell cut face"]))
    assert (trench | faces).sum() > 500
    assert image[trench | faces].mean() < 0.6 * image[~(trench | faces)].mean()
    if beam_type is BeamType.ION:
        # the beam that cut it looks straight through
        assert trench.sum() > 500
        assert image[trench].mean() < 0.3 * image[~trench].mean()


def test_the_labels_turn_with_the_scan(ripped):
    """A rip is black in the image, so where the labels say rip must be where
    the image is dark, however the scan is turned."""

    def agreement(image, found):
        gone = found.classes == labels.ID["broken hole or rip"]
        return float(((image < 40) == gone).mean())

    image, upright = _take(ripped)
    assert agreement(image, upright) > 0.97
    ripped.set_scan_rotation(np.pi, BeamType.ELECTRON)
    image, turned = _take(ripped)
    assert agreement(image, turned) > 0.97
    assert agreement(image, upright) < 0.9, "and the view did turn"


def test_a_dataset_is_written_with_its_annotations(tmp_path):
    coco = labels.generate(
        str(tmp_path),
        seeds=[7],
        views=["overview"],
        resolution=RESOLUTION,
        config_path=CONFIG,
    )
    with open(tmp_path / "annotations.json") as f:
        assert json.load(f) == coco
    assert len(coco["images"]) == 2
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert names[labels.ID["cell"]] == "cell" and 0 not in names
    for entry in coco["images"]:
        image = np.asarray(Image.open(tmp_path / entry["file_name"]))
        classes = np.asarray(Image.open(tmp_path / entry["labels"]))
        cells = np.asarray(Image.open(tmp_path / entry["instances"]))
        assert image.shape == classes.shape == cells.shape == RESOLUTION[::-1]
        assert (entry["width"], entry["height"]) == RESOLUTION
        assert os.path.exists(
            tmp_path
            / entry["file_name"].replace("images", "overlays").replace(".png", ".jpg")
        )
    assert coco["annotations"]
    for annotation in coco["annotations"]:
        runs = annotation["segmentation"]["counts"]
        assert sum(runs) == RESOLUTION[0] * RESOLUTION[1]
        assert sum(runs[1::2]) == annotation["area"] > 0
        x, y, w, h = annotation["bbox"]
        assert 0 <= x and x + w <= RESOLUTION[0] and 0 <= y and y + h <= RESOLUTION[1]


def test_labels_need_the_sample_scene():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    try:
        if microscope._connection.sample_scene is None:
            with pytest.raises(ValueError, match="sample scene"):
                labels.of(microscope, BeamType.ELECTRON, 150e-6, RESOLUTION)
    finally:
        microscope.disconnect()


# ---------------------------------------------------------------------------
# The fluorescence view
# ---------------------------------------------------------------------------


def _fm_scene(microscope):
    from fibsem.geometry.projection import FMStageProjection

    projection = FMStageProjection.from_microscope(microscope)
    projection = FMStageProjection(
        geometry=projection.geometry, pixel_size=2e-7, shape=(512, 512)
    )
    scene = microscope._connection.sample_scene
    return scene, microscope._connection.scene_stage_position_at_fm(), projection


def test_the_fm_labels_agree_with_the_fm_image():
    microscope = _microscope(seed=2, cell_type="rpe")
    try:
        scene, position, projection = _fm_scene(microscope)
        found = labels.fm_view(scene, position, (512, 512), projection)
        green = scene.render_fm(
            position,
            (512, 512),
            projection,
            weights=(0.0, 1.0, 0.0, 0.0),
            rng=np.random.default_rng(0),
        ).astype(float)
        reflection = scene.render_fm(
            position,
            (512, 512),
            projection,
            weights=(1.0, 0.0, 0.0, 0.0),
            rng=np.random.default_rng(0),
        ).astype(float)
    finally:
        microscope.disconnect()
    assert found.classes.shape == green.shape == (512, 512)
    cell = found.classes == labels.ID["cell"]
    ice = found.classes == labels.ID["ice"]
    bar = found.classes == labels.ID["grid bar"]
    assert cell.sum() > 500 and ice.sum() > 500 and bar.sum() > 500
    # the cells are what glows, and the bars are what reflects
    assert np.median(green[cell]) > 1.1 * np.median(green[ice])
    assert np.median(reflection[bar]) > 1.2 * np.median(reflection[ice])
    # a target in view is marked, on its cell, where the label is brightest
    target = found.classes == labels.ID["fluorescence target"]
    assert target.any()
    assert green[target].max() > np.percentile(green[cell], 99)
    assert any(item["class"] == "fluorescence target" for item in found.boxes())


def test_an_fm_image_comes_with_its_labels():
    microscope = _microscope(seed=2, cell_type="rpe")
    try:
        image, found = labels.acquire_fm(microscope)
    finally:
        microscope.disconnect()
    assert found.classes.shape == found.cells.shape == image.data.shape
    assert (found.classes == labels.ID["grid bar"]).any()


def test_the_dataset_lamella_is_cut_through_a_cell(microscope):
    microscope.move_to_milling_angle(labels.MILLING_ANGLE)
    assert labels.centre_on_cell(microscope)
    found = labels.of(microscope, BeamType.ION, 50e-6, RESOLUTION)
    height, width = found.cells.shape
    assert found.cells[height // 2, width // 2] > 0
