"""The relief renderer: the scene as a height field, seen along each beam."""

import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

TFS_SHUTTLE_CONFIG = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
)
MILLING_TILT = np.deg2rad(12.0)
PLAIN = {
    "cell_type": "none",
    "contamination_density": 0.0,
    "ice_density": 0.0,
    "rip_fraction": 0.0,
}


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["sample"] = {"enabled": True, "coincidence_offset": 0.0}
    microscope._connection.setup_sample_scene()
    pose = microscope.get_stage_position()
    pose.t = MILLING_TILT
    microscope.move_stage_absolute(pose)
    yield microscope
    microscope.disconnect()


def _use(microscope, **kwargs) -> SampleScene:
    kwargs = {"coincidence_offset": 0.0, **kwargs}
    scene = SampleScene(noise_sigma=0.0, noise_fraction=0.0, **kwargs)
    scene.anchor(microscope.get_stage_position())
    microscope._connection.sample_scene = scene
    return scene


def _image(microscope, beam_type, hfw=150e-6) -> np.ndarray:
    settings = ImageSettings(
        resolution=(512, 342), hfw=hfw, dwell_time=1e-9, save=False, beam_type=beam_type
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The translation (rows, columns) taking a onto b, by phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fa.conj() * fb / (np.abs(fa * fb) + 1e-9)).real
    peak = np.array(np.unravel_index(np.argmax(r), r.shape), dtype=float)
    size = np.array(r.shape)
    return np.where(peak > size / 2, peak - size, peak)


def test_the_renderer_comes_from_the_config():
    assert SampleScene.from_config({"renderer": "relief"}).renderer == "relief"
    assert SampleScene().renderer == "flat"


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_grid_lands_where_the_flat_renderer_puts_it(microscope, beam_type):
    """Positions come from the same projection: the bars (at the film's height)
    are on the same pixels whichever renderer draws them."""
    _use(microscope, **PLAIN)
    flat = _image(microscope, beam_type, hfw=400e-6)
    _use(microscope, renderer="relief", **PLAIN)
    relief = _image(microscope, beam_type, hfw=400e-6)
    assert np.abs(_shift(flat, relief)).max() <= 1


def test_a_trench_is_cut_along_the_milling_beam(microscope):
    """A rectangle milled in the FIB view is a square-edged dark rectangle in
    that view, where it was drawn."""
    _use(microscope, renderer="relief", **PLAIN)
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=30e-6, height=6e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    image = _image(microscope, BeamType.ION, hfw=60e-6)
    pixel = 60e-6 / 512
    labels, n = ndi.label(image < 40)
    assert n > 0
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    rows, cols = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    width = (cols.max() - cols.min() + 1) * pixel
    height = (rows.max() - rows.min() + 1) * pixel
    assert width == pytest.approx(30e-6, rel=0.08)
    assert height == pytest.approx(6e-6, rel=0.15)
    # square: the dark region fills its bounding box
    assert len(rows) / ((np.ptp(rows) + 1) * (np.ptp(cols) + 1)) > 0.9
    centre = ((cols.min() + cols.max()) / 2 - 256, (rows.min() + rows.max()) / 2 - 171)
    assert abs(centre[0]) <= 3 and abs(centre[1]) <= 3


def test_a_lamella_between_two_trenches_stays(microscope):
    """Two patterns with a gap between them leave the gap standing: in the
    FIB view a bright band between two dark ones."""
    _use(microscope, renderer="relief", **PLAIN)
    gap = 1e-6
    for y in (gap / 2 + 3e-6, -(gap / 2 + 3e-6)):
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=20e-6, height=6e-6, depth=1e-6, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    profile = image[:, 236:276].mean(axis=1)
    pixel = 40e-6 / 512
    band = profile[171 - int(0.3e-6 / pixel) : 171 + int(0.3e-6 / pixel) + 1]
    above = profile[171 - int(3.5e-6 / pixel)]
    below = profile[171 + int(3.5e-6 / pixel)]
    assert band.min() > 60, band
    assert above < 40 and below < 40, (above, below)


@pytest.mark.parametrize("cell_type", ["rpe", "neuron"])
def test_rpe_and_neuron_cells_grow_and_stand_up(cell_type):
    """RPE grows as packed islands, neurons as a soma with processes; both
    stand above the film in the relief renderer."""
    from fibsem.drivers.demo import sim_relief

    scene = SampleScene(cell_type=cell_type, renderer="relief")
    parts = {f.part for f in scene.features if f.kind == "cell"}
    assert "body" in parts
    if cell_type == "neuron":
        assert "neurite" in parts
    body = next(f for f in scene.features if f.kind == "cell" and f.part == "body")
    xa = body.x * 1e6 + np.arange(-20, 20, 0.5)
    ya = body.y * 1e6 + np.arange(-20, 20, 0.5)
    x, y = np.meshgrid(xa, ya)
    h, _, _, _ = sim_relief.surface(scene, x, y, (0.5, 0.5))
    assert h[40, 40] - np.median(h) > 0.5
