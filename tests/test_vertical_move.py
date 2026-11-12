"""vertical_move must be vertical (FIB-773).

The premise of the operation: a chamber-vertical move is invisible to the
SEM view, which is what lets it correct the FIB view without disturbing the
SEM centring it started from. The old decomposition used 1/cos(t) where a
vertical needs cos(t), so the move dragged the feature sideways in the SEM
at every tilted pose - and two magic constants (0.9, 1.11) partly absorbed
the error near the milling tilt.

These tests run the real ThermoMicroscope math through the simulator's
projection-true scene, at more than one tilt: the bug family being pinned
here is exact at t=0 and wrong everywhere else.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures.settings import BeamType, ImageSettings
from tests.fixtures.sim_scenes import mammalian_scene

# The geometry under test is a pre-tilted TFS shuttle; pin it rather than
# inherit whatever configuration an earlier test left as the default
TFS_SHUTTLE_CONFIG = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
)

HFW = 150e-6
RESOLUTION = (1536, 1024)
PIXEL_SIZE = HFW / RESOLUTION[0]


@pytest.fixture
def microscope():
    microscope, settings = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["coincidence_projection"] = True
    microscope.system.sim["coincidence_offset"] = 0.0
    microscope._connection.setup_sample_scene()

    microscope._connection.sample_scene = mammalian_scene(
        coincidence_offset=0.0,
        fiducial=True,
        cell_type="none",
        contamination_density=0.0,
        ice_density=0.0,
        rip_fraction=0.0,
        noise_sigma=0.0,
        noise_fraction=0.0,
    )
    yield microscope
    microscope.disconnect()


def _fiducial(microscope, beam_type):
    from scipy import ndimage as ndi

    settings = ImageSettings(
        resolution=RESOLUTION, hfw=HFW, dwell_time=1e-9, save=False, beam_type=beam_type
    )
    image = microscope.acquire_image(image_settings=settings)
    # the fiducial's beads are the brightest thing in both beams
    data = ndi.gaussian_filter(image.data.astype(np.float32), 3)
    y, x = np.unravel_index(np.argmax(data), data.shape)
    return x, y


@pytest.mark.parametrize("tilt_deg", [12.0, 35.0])
def test_vertical_move_is_invisible_to_the_sem(microscope, tilt_deg):
    """The whole premise: correcting the FIB view must not drag the SEM."""
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(tilt_deg)
    microscope.move_stage_absolute(pose)
    sx0, sy0 = _fiducial(microscope, BeamType.ELECTRON)

    microscope.vertical_move(dy=-20e-6, dx=0)

    sx1, sy1 = _fiducial(microscope, BeamType.ELECTRON)
    drag = np.hypot(sx1 - sx0, sy1 - sy0) * PIXEL_SIZE
    assert drag < 1e-6, f"SEM view dragged {drag * 1e6:.2f} um by a 'vertical' move"


@pytest.mark.parametrize("tilt_deg", [12.0, 35.0])
def test_vertical_move_delivers_the_requested_fib_shift(microscope, tilt_deg):
    """The contract: the FIB view shifts by exactly the requested dy."""
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(tilt_deg)
    microscope.move_stage_absolute(pose)
    fx0, fy0 = _fiducial(microscope, BeamType.ION)

    requested = -20e-6
    microscope.vertical_move(dy=requested, dx=0)

    fx1, fy1 = _fiducial(microscope, BeamType.ION)
    moved = (fy1 - fy0) * PIXEL_SIZE
    assert moved == pytest.approx(requested, abs=1e-6), (
        f"FIB view moved {moved * 1e6:+.2f} um for a {requested * 1e6:+.2f} um request"
    )
