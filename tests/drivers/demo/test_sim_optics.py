"""The beam's settings show in the simulator's images: focus from the working
distance, stigmation, and the current's probe size and noise."""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo import sim_optics
from fibsem.drivers.demo.sim_optics import Optics
from fibsem.structures.settings import BeamType, ImageSettings, Point

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    microscope.system.sim["sample"] = {
        "enabled": True,
        "seed": 3,
        "noise_sigma": 0.0,
        "noise_fraction": 0.0,
        "charging": 0.0,
    }
    microscope._connection.setup_sample_scene()
    yield microscope
    microscope.disconnect()


def _image(microscope, beam_type, hfw=60e-6) -> np.ndarray:
    settings = ImageSettings(
        resolution=(512, 342), hfw=hfw, dwell_time=1e-9, save=False, beam_type=beam_type
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def _detail(image: np.ndarray, axis=None) -> float:
    """How much fine detail an image has: the size of its pixel-to-pixel
    steps, along one axis or both."""
    if axis is None:
        return _detail(image, 0) + _detail(image, 1)
    return float(np.abs(np.diff(image, axis=axis)).mean())


def _focus(microscope, beam_type, off=0.0) -> None:
    focus = microscope._connection.scene_focus(beam_type, here=True)
    microscope.set_working_distance(focus + off, beam_type)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_demo_starts_near_focus_at_the_eucentric_height(microscope, beam_type):
    scene = microscope._connection.sample_scene
    height = microscope._beam_config("eucentric_height", beam_type).eucentric_height
    assert microscope.get_working_distance(beam_type) == pytest.approx(height)
    off = microscope._connection.scene_focus(beam_type) - height
    assert 0 < abs(off) <= scene.focus_fuzz
    assert off == pytest.approx(scene.focus_offset(beam_type)), "the same every time"


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_a_working_distance_away_from_focus_blurs_the_image(microscope, beam_type):
    _focus(microscope, beam_type)
    sharp = _detail(_image(microscope, beam_type))
    # the FIB's beam is the narrower, so it takes more to blur it
    far = 1.0 if beam_type is BeamType.ELECTRON else 4.0
    _focus(microscope, beam_type, far * 150e-6)
    soft = _detail(_image(microscope, beam_type))
    _focus(microscope, beam_type, -far * 600e-6)
    softer = _detail(_image(microscope, beam_type))
    assert soft < 0.9 * sharp
    assert softer < 0.8 * soft


def test_autofocus_finds_the_focus(microscope):
    beam_type = BeamType.ELECTRON
    _focus(microscope, beam_type)
    sharp = _detail(_image(microscope, beam_type))
    _focus(microscope, beam_type, 400e-6)
    microscope.auto_focus(beam_type)
    focus = microscope._connection.scene_focus(beam_type, here=True)
    assert abs(microscope.get_working_distance(beam_type) - focus) <= 2e-6
    assert _detail(_image(microscope, beam_type)) > 0.97 * sharp


def test_stigmation_stretches_the_blur_one_way(microscope):
    beam_type = BeamType.ELECTRON
    _focus(microscope, beam_type)
    sharp = _image(microscope, beam_type)
    microscope.set_stigmation(Point(0.2, 0.0), beam_type)
    across = _image(microscope, beam_type)
    # at focus both directions blur alike; through focus one sharpens
    assert _detail(across) < 0.9 * _detail(sharp)
    _focus(microscope, beam_type, sim_optics.SPLIT * 0.2)
    one_way = _image(microscope, beam_type)
    ratio = _detail(one_way, 1) / _detail(one_way, 0)
    assert ratio < 0.8 * _detail(sharp, 1) / _detail(sharp, 0)
    _focus(microscope, beam_type, -sim_optics.SPLIT * 0.2)
    other_way = _image(microscope, beam_type)
    assert _detail(other_way, 1) / _detail(other_way, 0) > 1.2 * ratio


def test_a_high_current_is_a_bigger_probe_and_less_noise(microscope):
    beam_type = BeamType.ION
    _focus(microscope, beam_type)
    low = _detail(_image(microscope, beam_type, hfw=20e-6))
    microscope.set_beam_current(28e-9, beam_type)
    high = _detail(_image(microscope, beam_type, hfw=20e-6))
    assert high < 0.9 * low
    quiet, noisy = Optics(current=7.6e-9), Optics(current=1e-12)
    assert quiet.dose(1) == sim_optics.DOSE[1] and noisy.dose(1) == sim_optics.DOSE[0]
    assert Optics(current=20e-12).dose(1) == 1.0


def test_noise_falls_as_the_current_rises(microscope):
    beam_type = BeamType.ION
    scene = microscope._connection.sample_scene
    scene.noise_sigma = 12.0
    _focus(microscope, beam_type)
    low = _detail(_image(microscope, beam_type, hfw=150e-6))
    microscope.set_beam_current(1e-9, beam_type)
    assert _detail(_image(microscope, beam_type, hfw=150e-6)) < 0.8 * low


def test_a_beam_shift_moves_the_image(microscope):
    beam_type = BeamType.ELECTRON
    _focus(microscope, beam_type)
    before = _image(microscope, beam_type)
    pixel = 60e-6 / 512
    microscope.set_beam_shift(Point(20 * pixel, 0.0), beam_type)
    after = _image(microscope, beam_type)
    assert np.abs(after - before).mean() > 1.0
    moved = [
        np.abs(np.roll(before, s, axis=1) - after)[:, 40:-40].mean() for s in (-20, 20)
    ]
    assert min(moved) < 0.2 * np.abs(after - before)[:, 40:-40].mean()


def test_the_settings_can_be_turned_off(microscope):
    beam_type = BeamType.ELECTRON
    microscope._connection.sample_scene.beam_optics = False
    _focus(microscope, beam_type)
    sharp = _image(microscope, beam_type)
    _focus(microscope, beam_type, 500e-6)
    microscope.set_stigmation(Point(0.3, 0.1), beam_type)
    assert np.array_equal(_image(microscope, beam_type), sharp)


def test_a_sharp_beam_is_left_alone():
    image = np.random.default_rng(0).random((64, 96)).astype(np.float32)
    assert sim_optics.blur(image, Optics(), 0, 0.1) is image
    assert sim_optics.blur(image, Optics(current=100e-12), 0, 0.1) is image
    stretched = sim_optics.blur(image, Optics(stigmation=(0.2, 0.0)), 0, 0.1)
    assert stretched.shape == image.shape
    assert stretched.mean() == pytest.approx(image.mean(), rel=0.02)


def test_focus_follows_the_stage_height(microscope):
    """A sample moved off the eucentric height is out of focus by as much
    along the beam, and autofocus follows it there."""
    beam_type = BeamType.ELECTRON
    connection = microscope._connection
    _focus(microscope, beam_type)
    sharp = _detail(_image(microscope, beam_type))
    before = connection.scene_focus(beam_type, here=True)
    pose = microscope.get_stage_position()
    pose.z += 300e-6
    microscope.move_stage_absolute(pose)
    after = connection.scene_focus(beam_type, here=True)
    assert abs(after - before) >= 300e-6
    assert _detail(_image(microscope, beam_type)) < 0.8 * sharp
    microscope.auto_focus(beam_type)
    assert abs(microscope.get_working_distance(beam_type) - after) <= 2e-6
    assert _detail(_image(microscope, beam_type)) > 0.95 * sharp


def test_a_tilted_view_is_sharpest_where_it_is_in_focus():
    """Across a view the beam grazes, one edge is nearer than the other: with
    the centre in focus the edges are soft, and a working distance set for an
    edge sharpens that edge."""
    rng = np.random.default_rng(1)
    image = rng.random((300, 200)).astype(np.float32)
    level = sim_optics.blur(image, Optics(), 0, 0.1)
    assert level is image
    tilted = sim_optics.blur(image, Optics(tilt=(0.0, 20.0)), 0, 0.1)
    rows = np.abs(np.diff(tilted, axis=1)).mean(axis=1)
    assert rows[140:160].mean() > 1.5 * rows[:20].mean()
    assert rows[140:160].mean() > 1.5 * rows[-20:].mean()
    # the top edge is 15 um * 20 = 300 um nearer: focus there
    top = sim_optics.blur(image, Optics(defocus=300e-6, tilt=(0.0, 20.0)), 0, 0.1)
    rows = np.abs(np.diff(top, axis=1)).mean(axis=1)
    assert rows[:20].mean() > 1.5 * rows[140:160].mean()


def test_a_longer_dwell_is_less_noise(microscope):
    assert Optics(dwell=1e-6).dose(0) == 1.0
    assert Optics(dwell=3e-6).dose(0) == pytest.approx(3.0)
    usual, low, high = sim_optics.DWELL
    assert usual == 1e-6 and Optics(dwell=1e-9).dose(0) == low
    assert Optics(dwell=1e-3).dose(0) == high
    scene = microscope._connection.sample_scene
    scene.noise_sigma = 12.0
    beam_type = BeamType.ION
    _focus(microscope, beam_type)

    def image(dwell):
        settings = ImageSettings(
            resolution=(512, 342),
            hfw=150e-6,
            dwell_time=dwell,
            save=False,
            beam_type=beam_type,
        )
        return microscope.acquire_image(image_settings=settings).data.astype(np.float32)

    assert _detail(image(4e-6)) < 0.8 * _detail(image(0.5e-6))


def test_a_scene_built_in_code_is_drawn_sharp_unless_it_asks():
    from fibsem.drivers.demo.sim_scene import SampleScene

    assert SampleScene().beam_optics is False
    assert SampleScene.from_config({}).beam_optics is True
    assert SampleScene.from_config({"beam_optics": False}).beam_optics is False
