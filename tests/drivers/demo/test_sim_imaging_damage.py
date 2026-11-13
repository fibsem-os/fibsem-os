"""FIB imaging wears the simulated sample: nothing at an imaging current, a
rectangle after a few frames at a milling current."""

import json

import numpy as np
import pytest

from fibsem import utils
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import BeamType, ImageSettings

PLAIN = {
    "cell_type": "none",
    "contamination_density": 0.0,
    "ice_density": 0.0,
    "frost_density": 0.0,
    "sphere_density": 0.0,
    "crack_density": 0.0,
    "rip_fraction": 0.0,
    "noise_sigma": 0.0,
    "noise_fraction": 0.0,
}


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.system.sim["sample"] = {"enabled": True, **PLAIN}
    microscope._connection.setup_sample_scene()
    microscope.move_to_milling_angle(15)
    yield microscope
    microscope.disconnect()


def _image(microscope, beam_type, hfw, dwell=1e-6) -> np.ndarray:
    settings = ImageSettings(
        resolution=(768, 512),
        hfw=hfw,
        dwell_time=dwell,
        save=False,
        beam_type=beam_type,
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def test_an_imaging_current_leaves_no_mark(microscope):
    scene = microscope._connection.sample_scene
    microscope.set_beam_current(20e-12, BeamType.ION)
    for _ in range(3):
        _image(microscope, BeamType.ION, 40e-6)
    _image(microscope, BeamType.ELECTRON, 40e-6)
    assert scene.milled == []
    (seen,) = scene.exposures.values()  # the electron beam is not counted
    assert seen["frames"] == 3
    assert seen["depth"] == 0.0
    assert 0 < seen["dose"] * 10 < 1e-3  # under a nanometre of ice


def test_a_milling_current_wears_a_rectangle_that_deepens(microscope):
    scene = microscope._connection.sample_scene
    microscope.set_beam_current(20e-12, BeamType.ION)
    before = _image(microscope, BeamType.ION, 40e-6)
    microscope.set_beam_current(1e-9, BeamType.ION)
    depths = []
    for _ in range(3):
        _image(microscope, BeamType.ION, 10e-6, dwell=4e-6)
        (region,) = scene.milled
        assert region.scan
        depths.append(region.depth)
    assert depths[0] > 100e-9
    assert depths[2] == pytest.approx(3 * depths[0])
    microscope.set_beam_current(20e-12, BeamType.ION)
    after = _image(microscope, BeamType.ION, 40e-6)
    # the worn field of view is the middle quarter of the wider image
    h, w = after.shape
    inside = (slice(h * 7 // 16, h * 9 // 16), slice(w * 7 // 16, w * 9 // 16))
    outside = (slice(0, h // 8), slice(0, w // 8))
    assert abs((after - before)[inside].mean()) > 8
    assert abs((after - before)[outside].mean()) < 2
    exposures = scene.record()["exposures"]
    json.dumps(exposures)
    worn = [e for e in exposures if e["depth"] > 0]
    assert len(worn) == 1 and worn[0]["frames"] == 3


def test_a_scene_built_in_code_is_not_worn_unless_asked():
    assert SampleScene().imaging_damage == 0.0
    assert SampleScene.from_config({}).imaging_damage == 1.0
    assert SampleScene.from_config({"imaging_damage": 0}).imaging_damage == 0.0
