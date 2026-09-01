"""Simulator shared-scene coincidence projection (FIB-874).

The projection mode is opt-in via the sim config and defaults off. When on,
both beams image one synthetic scene, the FIB view carries the geometric
height-error displacement, and measure_coincidence recovers it - including
after a known stage z move, which is the offline version of the bench
oracle test.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.alignment.coincidence import measure_coincidence_from_images
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

COINCIDENCE_OFFSET = 8e-6


@pytest.fixture
def microscope():
    microscope, settings = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def _enable_projection(microscope, offset: float = COINCIDENCE_OFFSET) -> None:
    microscope.system.sim["coincidence_projection"] = True
    microscope.system.sim["coincidence_offset"] = offset
    microscope._setup_coincidence_projection()
    # a realistic milling pose: stage tilt 12 deg -> milling angle 15 deg
    # (at the Demo's boot tilt of 0 the FIB view is a ~3 deg grazing view,
    # foreshortened x14 - unmeasurable on hardware too)
    microscope.move_stage_relative(
        FibsemStagePosition(x=0, y=0, z=0, r=0, t=np.deg2rad(12.0))
    )


def _acquire_pair(microscope):
    image_settings = ImageSettings(
        resolution=(1536, 1024), hfw=100e-6, dwell_time=1e-9, save=False
    )
    image_settings.beam_type = BeamType.ELECTRON
    sem_image = microscope.acquire_image(image_settings=image_settings)
    image_settings.beam_type = BeamType.ION
    fib_image = microscope.acquire_image(image_settings=image_settings)
    return sem_image, fib_image


def test_projection_defaults_off(microscope):
    assert microscope._coincidence_scene is None


def test_measures_configured_offset(microscope):
    _enable_projection(microscope)

    sem_image, fib_image = _acquire_pair(microscope)
    m = measure_coincidence_from_images(sem_image, fib_image)

    assert m.is_reliable, m.refusal_reason
    assert abs(abs(m.dz) - COINCIDENCE_OFFSET) < 1e-6


def test_known_z_move_changes_measurement_by_that_amount(microscope):
    """The offline oracle: apply a known dz, the measured delta must match."""
    _enable_projection(microscope)

    m0 = measure_coincidence_from_images(*_acquire_pair(microscope))
    assert m0.is_reliable

    dz_applied = 3e-6
    microscope.move_stage_relative(
        FibsemStagePosition(x=0, y=0, z=dz_applied, r=0, t=0)
    )

    m1 = measure_coincidence_from_images(*_acquire_pair(microscope))
    assert m1.is_reliable
    assert abs(abs(m1.dz - m0.dz) - dz_applied) < 1e-6


def test_views_share_the_scene_but_not_the_contrast():
    from fibsem.microscopes.sim_scene import CoincidenceScene

    # a pose where the foreshortening ratio is exactly 1 (milling angle =
    # column_tilt / 2) and the stage is at coincidence: the two views then
    # share geometry pixel-for-pixel and differ only in contrast convention,
    # so with noise off they must be strongly anti-correlated
    scene = CoincidenceScene(
        coincidence_offset=0.0, noise_sigma=0.0, noise_fraction=0.0
    )
    pose = FibsemStagePosition(
        x=0, y=0, z=0, r=0, t=np.deg2rad(26.0 + 35.0 - (90.0 - 52.0))
    )
    kwargs = dict(
        stage_position=pose,
        hfw=150e-6,
        resolution=(768, 512),
        pretilt=np.deg2rad(35.0),
        column_tilt=np.deg2rad(52.0),
    )
    sem = scene.render(beam_type=BeamType.ELECTRON, **kwargs).astype(np.float32)
    fib = scene.render(beam_type=BeamType.ION, **kwargs).astype(np.float32)
    corr = np.corrcoef(sem.ravel(), fib.ravel())[0, 1]
    assert corr < -0.9
