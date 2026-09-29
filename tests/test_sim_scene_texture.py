"""The world texture: the scene's features stamped once into cached tiles
of the sample plane and sampled per view, so a pose imaged repeatedly pays
the stamping once and every view sees the same world."""

import numpy as np
import pytest

from fibsem.microscopes import sim_scene
from fibsem.microscopes.sim_scene import (
    TEXTURE_BUDGET,
    SampleScene,
    SceneFeature,
    WorldTexture,
)
from fibsem.projection import BeamStageProjection
from fibsem.structures import BeamType, FibsemStagePosition
from fibsem.utils import setup_session

RESOLUTION = (768, 512)


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = setup_session(
        config_path="fibsem/config/sim-arctis-configuration.yaml"
    )
    return microscope


def _scene(**kwargs) -> SampleScene:
    """A quiet scene - no cells, no bars - with one blob, so a test can
    find it as the brightest thing."""
    scene = SampleScene(
        cell_type="none",
        fiducial=False,
        contamination_density=0.0,
        ice_density=0.0,
        rip_fraction=0.0,
        noise_sigma=0.0,
        noise_fraction=0.0,
        grid_intensity=0.0,
        **kwargs,
    )
    scene.features = [
        SceneFeature(x=12e-6, y=-7e-6, sigma=3e-6, intensity=120.0, sharpness=3.0)
    ]
    return scene


def _pose(microscope, tilt_deg: float) -> FibsemStagePosition:
    position = microscope.get_stage_position()
    position.t = np.deg2rad(tilt_deg)
    microscope.move_stage_absolute(position)
    return microscope.get_stage_position()


def _render(microscope, scene, beam, hfw, position):
    projection = BeamStageProjection.from_microscope(microscope, beam_type=beam)
    return scene.render(
        beam, position, hfw, RESOLUTION, projection, rng=np.random.default_rng(0)
    )


def _blob_centre(frame: np.ndarray):
    """Centroid (col, row) of the brightest structure."""
    bright = frame > frame.mean() + 0.5 * (frame.max() - frame.mean())
    rows, cols = np.nonzero(bright)
    return cols.mean(), rows.mean()


def test_a_repeated_pose_renders_no_new_tiles(microscope):
    scene = _scene()
    position = _pose(microscope, 18.0)
    _render(microscope, scene, BeamType.ELECTRON, 100e-6, position)
    texture = scene.texture()
    rendered = texture.renders
    assert rendered > 0
    _render(microscope, scene, BeamType.ELECTRON, 100e-6, position)
    _render(microscope, scene, BeamType.ION, 100e-6, position)
    assert texture.renders == rendered, "the same footprint was stamped again"


def test_the_blob_sits_at_the_same_world_point_at_every_level(microscope):
    """The pyramid levels agree: a view at 100 um and one at 800 um put the
    blob at the same world position (to a fine-view pixel)."""
    scene = _scene()
    position = _pose(microscope, 18.0)
    fine = _render(microscope, scene, BeamType.ELECTRON, 100e-6, position)
    wide = _render(microscope, scene, BeamType.ELECTRON, 800e-6, position)
    assert len({k[0] for k in scene.texture().tiles}) == 2, "two levels expected"
    cx, cy = RESOLUTION[0] / 2, RESOLUTION[1] / 2
    fine_px, wide_px = 100e-6 / RESOLUTION[0], 800e-6 / RESOLUTION[0]
    fu, fv = _blob_centre(fine)
    wu, wv = _blob_centre(wide)
    fine_world = ((fu - cx) * fine_px, (fv - cy) * fine_px)
    wide_world = ((wu - cx) * wide_px, (wv - cy) * wide_px)
    assert (
        np.hypot(fine_world[0] - wide_world[0], fine_world[1] - wide_world[1])
        < 1.5 * fine_px
    )


def test_swapping_the_feature_list_rebuilds_the_texture(microscope):
    scene = _scene()
    position = _pose(microscope, 18.0)
    before = _render(microscope, scene, BeamType.ELECTRON, 100e-6, position)
    first = scene.texture()
    scene.features = []
    after = _render(microscope, scene, BeamType.ELECTRON, 100e-6, position)
    assert scene.texture() is not first
    assert before.max() > after.max() + 50, "the blob should be gone"


def test_a_grazing_view_stays_within_the_texture_budget():
    """A foreshortened view covers far more of the plane than its frame; the
    level goes coarser so the texture pixels it touches stay bounded."""
    scene = _scene()
    hfw = 100e-6
    width, height = RESOLUTION
    pixel_size = hfw / width
    # a view stretched 20x in y, as a near-grazing beam sees the surface
    xs = (np.arange(width) - width / 2)[None, :] * pixel_size
    ys = (np.arange(height) - height / 2)[:, None] * pixel_size * 20.0
    texture = scene.texture()
    texture.sample(xs, ys, pixel_size)
    level = next(iter(texture.tiles))[0]
    assert level > WorldTexture.level_for(pixel_size)
    footprint = (xs.max() - xs.min()) * (ys.max() - ys.min())
    assert (
        footprint / WorldTexture.pixel_at(level) ** 2 <= TEXTURE_BUDGET * width * height
    )


def test_the_cache_is_byte_capped(microscope, monkeypatch):
    monkeypatch.setattr(sim_scene, "TEXTURE_CACHE_BYTES", 4e6)
    scene = _scene()
    scene.features = [
        SceneFeature(x=x, y=0.0, sigma=2e-6, intensity=100.0)
        for x in np.arange(-300e-6, 300e-6, 20e-6)
    ]
    position = _pose(microscope, 18.0)
    _render(microscope, scene, BeamType.ELECTRON, 700e-6, position)
    _render(microscope, scene, BeamType.ELECTRON, 60e-6, position)
    texture = scene.texture()
    assert texture.bytes <= 4e6 + (sim_scene.TEXTURE_TILE + 2) ** 2 * 3 * 4
    assert len(texture.tiles) >= 1
