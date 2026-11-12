"""The solid renderer: the sample as objects that lie under one another."""

import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo import sim_solid
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
    "frost_density": 0.0,
    "sphere_density": 0.0,
    "crack_density": 0.0,
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


def _mill(microscope, *rectangles) -> None:
    for width, height, y in rectangles:
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=width, height=height, depth=10e-6, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()


def _stack(scene, x, y) -> dict:
    """The (bottom, top) of each piece at one world point (um)."""
    xa = x + np.arange(-2, 3) * 0.2
    ya = y + np.arange(-2, 3) * 0.2
    col = sim_solid.columns(scene, xa, ya, (0.2, 0.2))
    return {
        k: (float(np.broadcast_to(col[k][0], (5, 5))[2, 2]), float(col[k][1][2, 2]))
        for k in sim_solid.SLOTS
    }


def test_the_renderer_comes_from_the_config():
    assert SampleScene.from_config({"renderer": "solid"}).renderer == "solid"


def test_a_cell_lies_under_the_ice_with_its_nucleus_inside():
    """What the height field cannot hold: at a cell's nucleus the stack is
    film, cell, nucleus inside the cell, and ice over the top."""
    scene = SampleScene(renderer="solid", rip_fraction=0.0)
    nucleus = next(
        f for f in scene.features if f.kind == "cell" and f.part == "nucleus"
    )
    stack = _stack(scene, nucleus.x * 1e6, nucleus.y * 1e6)
    cell, ice = stack["cell"][1], stack["ice"][1]
    lo, hi = stack["nucleus"]
    film = stack["film"][1]
    assert film == sim_solid.FILM_TOP
    assert film < lo < hi < cell < ice


def test_a_sphere_rests_on_the_ice_with_room_under_its_sides():
    scene = SampleScene(
        renderer="solid", cell_type="none", sphere_density=2.0, rip_fraction=0.0
    )
    ball = max((f for f in scene.features if f.part == "sphere"), key=lambda f: f.sigma)
    r = ball.sigma * 1e6
    centre = _stack(scene, ball.x * 1e6, ball.y * 1e6)
    side = _stack(scene, ball.x * 1e6 + 0.8 * r, ball.y * 1e6)
    assert centre["particle"][1] > centre["ice"][1]
    # out toward its side the ball has lifted off the ice: a gap under it
    assert side["particle"][0] > side["ice"][1]


def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The translation (rows, columns) taking a onto b, by phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fa.conj() * fb / (np.abs(fa * fb) + 1e-9)).real
    peak = np.array(np.unravel_index(np.argmax(r), r.shape), dtype=float)
    size = np.array(r.shape)
    return np.where(peak > size / 2, peak - size, peak)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_grid_lands_where_the_relief_renderer_puts_it(
    microscope, beam_type, monkeypatch
):
    """Positions come from the same projection: with the ice left flat over
    the bars (it stands up over them otherwise, which a glancing beam sees as
    a shift), the bars are on the same pixels whichever renderer draws them."""
    monkeypatch.setattr(sim_solid, "BAR_RISE", 0.0)
    _use(microscope, renderer="relief", **PLAIN)
    relief = _image(microscope, beam_type, hfw=400e-6)
    _use(microscope, renderer="solid", **PLAIN)
    solid = _image(microscope, beam_type, hfw=400e-6)
    assert np.abs(_shift(relief, solid)).max() <= 1


def test_a_trench_is_cut_along_the_milling_beam(microscope):
    """A rectangle milled in the FIB view is a square-edged dark rectangle in
    that view, where it was drawn."""
    _use(microscope, renderer="solid", **PLAIN)
    _mill(microscope, (30e-6, 6e-6, 0.0))
    image = _image(microscope, BeamType.ION, hfw=60e-6)
    pixel = 60e-6 / 512
    labels, n = ndi.label(image < 40)
    assert n > 0
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    rows, cols = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    assert (np.ptp(cols) + 1) * pixel == pytest.approx(30e-6, rel=0.08)
    assert (np.ptp(rows) + 1) * pixel == pytest.approx(6e-6, rel=0.15)
    assert len(rows) / ((np.ptp(rows) + 1) * (np.ptp(cols) + 1)) > 0.9
    centre = ((cols.min() + cols.max()) / 2 - 256, (rows.min() + rows.max()) / 2 - 171)
    assert abs(centre[0]) <= 3 and abs(centre[1]) <= 3


def test_a_lamella_between_two_trenches_stays(microscope):
    _use(microscope, renderer="solid", **PLAIN)
    _mill(microscope, (20e-6, 6e-6, 3.5e-6), (20e-6, 6e-6, -3.5e-6))
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    profile = image[:, 236:276].mean(axis=1)
    pixel = 40e-6 / 512
    band = profile[171 - int(0.3e-6 / pixel) : 171 + int(0.3e-6 / pixel) + 1]
    assert band.min() > 60, band
    assert profile[171 - int(3.5e-6 / pixel)] < 40
    assert profile[171 + int(3.5e-6 / pixel)] < 40


def test_a_cut_face_shows_the_cell_and_its_nucleus(microscope, monkeypatch):
    """A trench through a cell leaves a face made of ice, the cell and the
    nucleus inside it, each its own tone."""
    scene = _use(microscope, renderer="solid", rip_fraction=0.0, sphere_density=0.0)
    nucleus = min(
        (f for f in scene.features if f.kind == "cell" and f.part == "nucleus"),
        key=lambda f: np.hypot(f.x, f.y),
    )
    for f in scene.features:
        f.x -= nucleus.x
        f.y -= nucleus.y
    _mill(microscope, (14e-6, 8e-6, 5e-6))
    hits = []
    trace = sim_solid.trace
    monkeypatch.setattr(
        sim_solid, "trace", lambda *a: hits.append(trace(*a)) or hits[-1]
    )
    _image(microscope, BeamType.ELECTRON, hfw=40e-6)
    face = hits[-1]["material"][hits[-1]["kind"] == 2]
    for material in (sim_solid.ICE, sim_solid.CELL, sim_solid.NUCLEUS):
        assert (face == material).sum() > 20, material
    tones = {sim_solid.INSIDE[m][0] for m in (1, 2, 3)}
    assert len(tones) == 3
