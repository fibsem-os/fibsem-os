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
from fibsem.alignment.coincidence import measure_coincidence_from_images
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

# The geometry under test is a pre-tilted TFS shuttle; pin it rather than
# inherit whatever configuration an earlier test left as the default
TFS_SHUTTLE_CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")

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
    microscope._setup_sample_scene()
    from fibsem.microscopes.sim_scene import SampleScene

    microscope._sample_scene = SampleScene(
        coincidence_offset=0.0,
        n_clusters=0,
        grid_intensity=0.0,
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
    data = ndi.gaussian_filter(image.data.astype(np.float32), 3)
    if beam_type is BeamType.ION:
        y, x = np.unravel_index(np.argmin(data), data.shape)
    else:
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


def test_vertical_move_closes_the_measured_coincidence_error(microscope):
    """The mini align loop: measure the height error, correct it with one
    vertical move, re-measure - the residual must be near zero."""
    microscope._sample_scene.coincidence_offset = 8e-6
    microscope._sample_scene.reference_position = None  # re-anchor with offset
    pose = microscope.get_stage_position()
    pose.t = np.deg2rad(12.0)
    microscope.move_stage_absolute(pose)
    microscope._sample_scene.n_clusters = 35
    microscope._sample_scene.grid_intensity = 90.0
    microscope._sample_scene.features = []
    microscope._sample_scene.__post_init__()

    def measure():
        settings = ImageSettings(
            resolution=RESOLUTION,
            hfw=100e-6,
            dwell_time=1e-9,
            save=False,
            beam_type=BeamType.ELECTRON,
        )
        sem_image = microscope.acquire_image(image_settings=settings)
        settings.beam_type = BeamType.ION
        fib_image = microscope.acquire_image(image_settings=settings)
        return measure_coincidence_from_images(sem_image, fib_image)

    m0 = measure()
    assert m0.is_reliable, m0.refusal_reason
    assert abs(m0.dz) > 5e-6  # there is an error to correct

    # correct using the measured FIB-plane residual, as the align loop will
    microscope.vertical_move(dy=m0.dy, dx=0)

    m1 = measure()
    assert m1.is_reliable, m1.refusal_reason
    assert abs(m1.dz) < 1e-6, (
        f"residual {m1.dz * 1e6:+.2f} um after correcting {m0.dz * 1e6:+.2f} um"
    )
