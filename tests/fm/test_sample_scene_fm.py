"""The simulated FM images the same sample scene as the beams (FIB-874).

Runs under the fm package's sim-arctis configuration (compustage, FM
present). The scene is rendered through FMStageProjection, so where a
feature lands in the FM image is predicted by the same projection the app
draws stage positions with - that prediction is the oracle here.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from fibsem import utils
from fibsem.fm.structures import ChannelSettings
from fibsem.microscopes.sim_scene import SampleScene, fm_channel_weights
from fibsem.projection import FMStageProjection
from fibsem.structures import FibsemStagePosition

REFLECTION = ChannelSettings(
    name="reflection", excitation_wavelength=550, emission_wavelength=None, color="gray"
)
GREEN = ChannelSettings(
    name="green", excitation_wavelength=488, emission_wavelength=520, color="green"
)
RED = ChannelSettings(
    name="red", excitation_wavelength=561, emission_wavelength=610, color="red"
)
BLUE = ChannelSettings(
    name="blue", excitation_wavelength=405, emission_wavelength=450, color="blue"
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    assert microscope.fm is not None, "the fm package config must carry an FM"
    microscope.system.sim["sample"] = {
        "enabled": True,
        "coincidence_offset": 0.0,
        "fiducial": True,
    }
    microscope._setup_sample_scene()
    microscope.move_to_device("FM")
    # the FM images with the objective inserted and at focus; it boots retracted
    microscope.fm.objective.move_absolute(microscope.fm.objective.focus_position)
    yield microscope
    microscope.disconnect()


def _fiducial_only(microscope) -> SampleScene:
    scene = SampleScene(
        coincidence_offset=0.0,
        fiducial=True,
        cell_type="none",
        grid_intensity=0.0,
        noise_sigma=0.0,
        noise_fraction=0.0,
    )
    scene.anchor(microscope.get_stage_position())
    microscope._sample_scene = scene
    return scene


def _brightest(image) -> tuple:
    data = ndi.gaussian_filter(image.data.astype(np.float32), 2)
    y, x = np.unravel_index(np.argmax(data), data.shape)
    return float(x), float(y)


def _predicted_fiducial(microscope) -> tuple:
    """Where the world origin (the fiducial) projects into the FM image."""
    projection = FMStageProjection.from_microscope(microscope)
    fm = microscope.fm
    width, height = fm.camera.resolution
    projection = FMStageProjection(
        geometry=projection.geometry,
        pixel_size=fm.camera.pixel_size[0],
        shape=(height, width),
    )
    scene = microscope._sample_scene
    pose = microscope.get_stage_position()
    ax, ay = projection.to_plane(scene.reference_position, pose)
    return (
        width / 2 + ax / projection.pixel_size,
        height / 2 + ay / projection.pixel_size,
    )


def test_channels_respond_to_the_excitation_and_emission_pair():
    bars, cells, subset, fiducial = fm_channel_weights(None, 550)
    assert bars == 1.0 and fiducial > 0  # reflection: the grid, and the fiducial
    _, cells, subset, fiducial = fm_channel_weights(520, 488)
    assert cells > 0.7 and subset < 0.05 and fiducial < 0.05  # GFP channel
    _, cells, subset, fiducial = fm_channel_weights(610, 561)
    assert subset > 0.7 and cells < 0.05  # mCherry channel
    _, cells, subset, fiducial = fm_channel_weights(460, 365)
    assert fiducial > 0.7 and cells < 0.05  # DAPI channel
    # a mismatched pair: 488 excitation collected in the red band bleeds
    # weakly rather than showing nothing or everything
    _, cells, subset, _ = fm_channel_weights(610, 488)
    assert 0 < subset < 0.1 and cells < 0.05
    assert fm_channel_weights("reflection", None) == fm_channel_weights(None, None)
    # the simulated FM's own filter: emission "Fluorescence" is an open band,
    # so its excitation lines pick the dye - 365 fiducial, 450 cells, 550
    # subset, 635 nothing
    line = {ex: fm_channel_weights("Fluorescence", ex) for ex in (365, 450, 550, 635)}
    assert line[365][3] > 0.7 and line[365][1] < 0.05
    assert line[450][1] > 0.7 and line[450][2] < 0.05
    assert line[550][2] > 0.9 and line[550][1] < 0.05
    assert max(line[635][1:]) < 0.05


def test_channels_show_different_structures(microscope):
    reflection = microscope.fm.acquire_image(REFLECTION).data.astype(np.float32)
    green = microscope.fm.acquire_image(GREEN).data.astype(np.float32)
    red = microscope.fm.acquire_image(RED).data.astype(np.float32)

    # every channel has structure, and they are not the same picture
    for image in (reflection, green, red):
        assert ndi.gaussian_filter(image, 3).std() > 20
    # opaque cell bodies show in both (reflection faintly), so the two are
    # correlated but far from the same picture
    corr = np.corrcoef(reflection.ravel(), green.ravel())[0, 1]
    assert corr < 0.75, f"reflection and green correlate {corr:.2f}: bars vs cells lost"
    # the red subset is a subset: where red is bright, green is bright too
    bright_red = ndi.gaussian_filter(red, 3) > np.percentile(red, 99)
    assert ndi.gaussian_filter(green, 3)[bright_red].mean() > np.percentile(green, 80)


def test_the_fiducial_lands_where_the_fm_projection_says(microscope):
    _fiducial_only(microscope)
    image = microscope.fm.acquire_image(BLUE)
    found = _brightest(image)
    predicted = _predicted_fiducial(microscope)
    error = np.hypot(found[0] - predicted[0], found[1] - predicted[1])
    assert error < 3, f"fiducial {error:.1f} px from the projected position"

    # move the stage: the fiducial follows the projection, not the pixels
    microscope.move_stage_relative(
        FibsemStagePosition(x=20e-6, y=-15e-6, z=0, r=0, t=0)
    )
    image = microscope.fm.acquire_image(BLUE)
    found = _brightest(image)
    predicted = _predicted_fiducial(microscope)
    moved = np.hypot(
        found[0] - image.data.shape[1] / 2, found[1] - image.data.shape[0] / 2
    )
    assert moved > 20  # it visibly moved off centre
    error = np.hypot(found[0] - predicted[0], found[1] - predicted[1])
    assert error < 3, (
        f"after the move the fiducial is {error:.1f} px off the projection"
    )


def test_defocus_blurs_the_image(microscope):
    _fiducial_only(microscope)
    fm = microscope.fm
    fm.objective.move_absolute(fm.objective.focus_position)
    sharp = fm.acquire_image(BLUE).data.astype(np.float32)
    fm.objective.move_absolute(fm.objective.focus_position + 20e-6)
    blurred = fm.acquire_image(BLUE).data.astype(np.float32)

    # the fiducial's peak over the background: blur spreads it out. Pixel
    # noise dominates any whole-image edge measure, so look at the peak.
    def peak_contrast(image):
        smooth = ndi.gaussian_filter(image, 1)
        return smooth.max() - np.median(smooth)

    assert peak_contrast(blurred) < 0.5 * peak_contrast(sharp)
