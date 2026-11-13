"""An FM camera mounted another way than its configuration says: the simulated
image is flipped until the mount transform is set to match."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from fibsem import utils
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.fm.structures import ChannelSettings, EmissionFilter
from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import CameraImageTransform

BLUE = ChannelSettings(
    name="blue",
    excitation_wavelength=385,
    power=0.05,
    exposure_time=0.01,
    emission_filter=EmissionFilter.band("450 nm", 450),
    color="blue",
)


def _demo(**sample):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage.sim_error = None
    microscope.system.sim["sample"] = {"enabled": True, **sample}
    microscope._connection.setup_sample_scene()
    return microscope


def _fiducial(microscope, mount=None) -> tuple:
    """Where the fiducial shows in an FM image, from the image's centre (px),
    with the stage moved so it sits well off centre."""
    scene = SampleScene(
        coincidence_offset=0.0,
        fiducial=True,
        cell_type="none",
        grid_intensity=0.0,
        noise_sigma=0.0,
        noise_fraction=0.0,
        fm_camera_mount=mount,
    )
    scene.anchor(microscope.get_stage_position())
    microscope._connection.sample_scene = scene
    microscope.move_stage_relative(
        FibsemStagePosition(x=20e-6, y=-12e-6, z=0, r=0, t=0)
    )
    image = microscope.fm.acquire_image(BLUE)
    data = ndi.gaussian_filter(image.data.astype(np.float32), 2)
    y, x = np.unravel_index(np.argmax(data), data.shape)
    microscope.move_stage_relative(
        FibsemStagePosition(x=-20e-6, y=12e-6, z=0, r=0, t=0)
    )
    return x - data.shape[1] / 2, y - data.shape[0] / 2


def test_an_fm_camera_mounted_another_way_flips_the_image_until_it_is_configured():
    microscope = _demo(coincidence_offset=0.0)
    try:
        assert microscope.fm is not None
        microscope.move_to_device("FM")
        microscope.fm.objective.move_absolute(microscope.fm.objective.focus_position)
        right = _fiducial(microscope)
        assert abs(right[0]) > 20 and abs(right[1]) > 20
        flipped = _fiducial(microscope, mount="flip-x")
        assert flipped[0] == pytest.approx(-right[0], abs=3)
        assert flipped[1] == pytest.approx(right[1], abs=3)
        # the configuration's mount transform set to match puts it right
        microscope.fm.devices["camera"]._mount_transform = CameraImageTransform.FLIP_X
        microscope.fm.devices["camera"].parameters["mount_transform"].get_value()
        fixed = _fiducial(microscope, mount="flip-x")
        assert fixed == pytest.approx(right, abs=3)
    finally:
        microscope.disconnect()
