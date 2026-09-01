"""Simulator shared-scene coincidence projection (FIB-874).

The projection mode is opt-in via the sim config and defaults off. When on,
both beams image one synthetic scene through BeamStageProjection - the same
machinery the app's click handlers, overview stitcher, and reprojection use.

The two invariants at the bottom are the ones that caught real sign bugs
during development, after ad-hoc measurements (whole-frame cross-correlation
on the periodic mesh, unnormalised template matching) had repeatedly given
misleading answers:

1. Click-to-centre: simulate the app's double-click math on a feature; the
   feature must land on the crosshair. Tracked on a fiducial-only scene so
   nothing is ambiguous.
2. Stitched-vs-single: a tiled overview and one wide-field shot are the same
   world through two different paths (stage stepping vs in-image mapping);
   they must agree.
"""

import numpy as np
import pytest

from fibsem import conversions, utils
from fibsem.alignment.coincidence import measure_coincidence_from_images
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
    Point,
)

COINCIDENCE_OFFSET = 8e-6
MILLING_POSE_TILT = np.deg2rad(12.0)


@pytest.fixture
def microscope():
    microscope, settings = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def _enable_projection(microscope, offset: float = COINCIDENCE_OFFSET, **scene_kwargs):
    microscope.system.sim["coincidence_projection"] = True
    microscope.system.sim["coincidence_offset"] = offset
    microscope._setup_coincidence_projection()
    if scene_kwargs:
        from fibsem.microscopes.sim_scene import CoincidenceScene

        microscope._coincidence_scene = CoincidenceScene(
            coincidence_offset=offset, **scene_kwargs
        )
    # a realistic milling pose: stage tilt 12 deg -> milling angle 15 deg
    # (at the Demo's boot tilt of 0 the FIB view is a ~3 deg grazing view,
    # foreshortened x14 - unmeasurable on hardware too)
    microscope.move_stage_relative(
        FibsemStagePosition(x=0, y=0, z=0, r=0, t=MILLING_POSE_TILT)
    )


def _fiducial_only(microscope):
    """Scene with only the fiducial cross: one unambiguous trackable feature."""
    _enable_projection(
        microscope,
        offset=0.0,
        n_clusters=0,
        grid_intensity=0.0,
        noise_sigma=0.0,
        noise_fraction=0.0,
    )


def _image_settings(beam_type, hfw=150e-6, resolution=(1536, 1024)):
    return ImageSettings(
        resolution=resolution,
        hfw=hfw,
        dwell_time=1e-9,
        save=False,
        beam_type=beam_type,
    )


def _find_fiducial(image):
    """Locate the fiducial: brightest point in SEM, darkest in FIB."""
    from scipy import ndimage as ndi

    data = ndi.gaussian_filter(image.data.astype(np.float32), 3)
    if image.metadata.image_settings.beam_type is BeamType.ION:
        y, x = np.unravel_index(np.argmin(data), data.shape)
    else:
        y, x = np.unravel_index(np.argmax(data), data.shape)
    return x, y


def test_projection_defaults_off(microscope):
    assert microscope._coincidence_scene is None


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_click_centres_the_clicked_feature(microscope, beam_type):
    """The app's double-click math must land the clicked feature on the
    crosshair, in both beam views."""
    _fiducial_only(microscope)
    settings = _image_settings(beam_type)
    # anchor the world, then move away so the fiducial sits off-centre
    microscope.acquire_image(image_settings=settings)
    microscope.stable_move(dx=25e-6, dy=12e-6, beam_type=beam_type)
    image = microscope.acquire_image(image_settings=settings)
    fx, fy = _find_fiducial(image)
    height, width = image.data.shape
    pixel_size = image.metadata.pixel_size.x
    # the fiducial starts off-centre, or the test proves nothing
    assert np.hypot(fx - width // 2, fy - height // 2) * pixel_size > 5e-6

    # exactly what the stage-control widget does with a double-click
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=fx, y=fy), image=image.data, pixelsize=pixel_size
    )
    microscope.stable_move(dx=point.x, dy=point.y, beam_type=beam_type)

    after = microscope.acquire_image(image_settings=settings)
    gx, gy = _find_fiducial(after)
    error = np.hypot(gx - width // 2, gy - height // 2) * pixel_size
    assert error < 1e-6, f"clicked feature landed {error * 1e6:.2f} um off-centre"


def test_z_move_displaces_fib_view_far_more_than_sem(microscope):
    """A stage-z move is the coincidence signal: large in the FIB view,
    small in the SEM view (only the sin(tilt) stage-axis leak)."""
    _fiducial_only(microscope)
    dz = 5e-6
    positions = {}
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        settings = _image_settings(beam_type)
        positions[beam_type] = [
            _find_fiducial(microscope.acquire_image(image_settings=settings))
        ]
    microscope.move_stage_relative(FibsemStagePosition(x=0, y=0, z=dz, r=0, t=0))
    pixel_size = 150e-6 / 1536
    shifts = {}
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        settings = _image_settings(beam_type)
        x1, y1 = _find_fiducial(microscope.acquire_image(image_settings=settings))
        x0, y0 = positions[beam_type][0]
        shifts[beam_type] = abs(y1 - y0) * pixel_size
    assert shifts[BeamType.ION] > 2e-6  # clearly visible
    assert shifts[BeamType.ION] > 2 * shifts[BeamType.ELECTRON]


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_stitched_overview_matches_single_wide_shot(microscope, beam_type, tmp_path):
    """A tiled overview and one wide-field shot are the same world through
    two paths (stage stepping vs in-image mapping); they must agree."""
    from skimage.transform import resize

    from fibsem.imaging.tiled import tiled_image_acquisition_and_stitch

    _enable_projection(microscope)
    # anchor the scene's world at the centre position before tiling starts
    microscope.acquire_image(image_settings=_image_settings(beam_type))

    tile_settings = _image_settings(beam_type, hfw=150e-6, resolution=(512, 512))
    tile_settings.save = True
    tile_settings.path = str(tmp_path)
    tile_settings.filename = "ov"
    stitched = tiled_image_acquisition_and_stitch(
        microscope,
        OverviewAcquisitionSettings(
            image_settings=tile_settings, nrows=2, ncols=2, overlap=0.0
        ),
    )
    # the 2x2 grid of 150um tiles covers ~300um; one wide shot of the same area
    single = microscope.acquire_image(
        image_settings=_image_settings(beam_type, hfw=300e-6, resolution=(512, 512))
    )
    stitched_small = resize(
        stitched.data.astype(np.float32), single.data.shape, anti_aliasing=True
    )
    reference = single.data.astype(np.float32)
    stitched_small -= stitched_small.mean()
    reference -= reference.mean()
    corr = float(np.corrcoef(stitched_small.ravel(), reference.ravel())[0, 1])
    assert corr > 0.6, (
        f"stitched overview does not match ground truth (corr={corr:.2f})"
    )


def test_views_share_the_scene_but_not_the_contrast():
    """Rendered through identical geometry, the two beams differ only in
    contrast convention - strongly anti-correlated with noise off."""
    from fibsem.microscopes.sim_scene import CoincidenceScene
    from fibsem.projection import BeamStageProjection
    from fibsem.structures import FibsemHardwareGeometry

    scene = CoincidenceScene(
        coincidence_offset=0.0, noise_sigma=0.0, noise_fraction=0.0
    )
    geometry = FibsemHardwareGeometry(
        column_tilt=0, fib_column_tilt=52.0, shuttle_pre_tilt=35.0
    )
    # one projection for both renders: identical geometry, contrast differs
    projection = BeamStageProjection(
        geometry=geometry, beam_type=BeamType.ELECTRON, scan_rotation=0.0
    )
    pose = FibsemStagePosition(x=0, y=0, z=0, r=0, t=np.deg2rad(35.0))
    kwargs = dict(
        stage_position=pose, hfw=150e-6, resolution=(768, 512), projection=projection
    )
    sem = scene.render(beam_type=BeamType.ELECTRON, **kwargs).astype(np.float32)
    fib = scene.render(beam_type=BeamType.ION, **kwargs).astype(np.float32)
    corr = np.corrcoef(sem.ravel(), fib.ravel())[0, 1]
    assert corr < -0.9


@pytest.mark.xfail(
    reason="measure_coincidence still derives its stretch and dy->dz conversion "
    "from hand formulas that disagree with the calibrated BeamStageProjection "
    "at tilt (FIB-868: derive measurement geometry from the projection)",
    strict=False,
)
def test_measures_configured_offset(microscope):
    _enable_projection(microscope)
    settings = _image_settings(BeamType.ELECTRON, hfw=100e-6)
    sem_image = microscope.acquire_image(image_settings=settings)
    settings.beam_type = BeamType.ION
    fib_image = microscope.acquire_image(image_settings=settings)
    m = measure_coincidence_from_images(sem_image, fib_image)
    assert m.is_reliable, m.refusal_reason
    assert abs(abs(m.dz) - COINCIDENCE_OFFSET) < 1e-6
