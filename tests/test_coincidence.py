"""Tests for fibsem.alignment.coincidence (FIB-868).

Synthetic scenes: a blob/edge-rich SEM view, and a FIB view built by applying
the inverse of the geometric transform the measurement is supposed to undo
(y-compression by the foreshortening ratio, a known shift, contrast change,
independent noise). The measurement must recover the known offset, and refuse
on scenes or conditions it cannot trust.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from fibsem.alignment.coincidence import (
    REFUSAL_BAND_DISAGREEMENT,
    REFUSAL_WINDOW_EDGE,
    CoincidenceMeasurement,
    fib_view_y_stretch,
    height_error_from_fib_shift,
    measure_coincidence,
)

PIXEL_SIZE = 65e-9  # ~100 um hfw at 1536 px, typical reference imaging
MILLING_ANGLE = np.deg2rad(15.0)
COLUMN_TILT = np.deg2rad(52.0)
SHAPE = (512, 768)


def make_sem_scene(seed: int = 42) -> np.ndarray:
    """Feature-rich flat scene: sparse blobs and a fiducial-like cross."""
    rng = np.random.default_rng(seed)
    scene = np.zeros(SHAPE, dtype=np.float32)
    ys = rng.integers(60, SHAPE[0] - 60, size=40)
    xs = rng.integers(60, SHAPE[1] - 60, size=40)
    for y, x in zip(ys, xs):
        scene[y, x] = rng.uniform(50, 200)
    scene = ndi.gaussian_filter(scene, 3)
    cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
    scene[cy - 2 : cy + 2, cx - 40 : cx + 40] = 150
    scene[cy - 40 : cy + 40, cx - 2 : cx + 2] = 150
    scene += rng.normal(0, 2, SHAPE).astype(np.float32)
    return scene


def make_fib_view(
    sem_scene: np.ndarray,
    dy_px: float,
    dx_px: float,
    stretch: float,
    seed: int = 7,
) -> np.ndarray:
    """Project the SEM scene into the FIB view with a known residual.

    The measurement stretches the FIB image y by `stretch`; building the view
    with the inverse mapping (and the shift applied in stretched space) means
    the known (dy_px, dx_px) is exactly what it should recover.
    """
    rng = np.random.default_rng(seed)
    h, w = sem_scene.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = h / 2
    src_y = (yy - cy) * stretch + cy + dy_px
    src_x = xx + dx_px
    fib = ndi.map_coordinates(
        sem_scene, [src_y, src_x], order=1, mode="constant", cval=0
    )
    fib = 200 - fib  # contrast inversion: intensities differ across modalities
    fib += rng.normal(0, 2, fib.shape).astype(np.float32)
    return fib


def test_y_stretch_matches_geometry():
    stretch = fib_view_y_stretch(MILLING_ANGLE, COLUMN_TILT)
    expected = np.sin(COLUMN_TILT - MILLING_ANGLE) / np.sin(MILLING_ANGLE)
    assert stretch == pytest.approx(expected)
    assert stretch == pytest.approx(2.33, abs=0.01)


def test_y_stretch_rejects_degrees():
    with pytest.raises(ValueError):
        fib_view_y_stretch(15.0)  # degrees passed where radians expected


def test_recovers_known_offset():
    stretch = fib_view_y_stretch(MILLING_ANGLE, COLUMN_TILT)
    dy_px, dx_px = 30.0, -12.0  # in stretched (SEM-projection) space
    sem = make_sem_scene()
    fib = make_fib_view(sem, dy_px, dx_px, stretch)

    m = measure_coincidence(
        sem, fib, PIXEL_SIZE, MILLING_ANGLE, column_tilt=COLUMN_TILT
    )

    assert m.is_reliable, m.refusal_reason
    # dy is reported in the FIB image plane: stretched-space px / stretch
    assert m.dy == pytest.approx(dy_px / stretch * PIXEL_SIZE, abs=2 * PIXEL_SIZE)
    assert m.dx == pytest.approx(dx_px * PIXEL_SIZE, abs=2 * PIXEL_SIZE)
    assert m.dz == pytest.approx(m.dy / np.sin(COLUMN_TILT))


def test_zero_offset_measures_near_zero():
    stretch = fib_view_y_stretch(MILLING_ANGLE, COLUMN_TILT)
    sem = make_sem_scene()
    fib = make_fib_view(sem, 0.0, 0.0, stretch)

    m = measure_coincidence(
        sem, fib, PIXEL_SIZE, MILLING_ANGLE, column_tilt=COLUMN_TILT
    )

    assert m.is_reliable
    assert abs(m.dy) <= 2 * PIXEL_SIZE
    assert abs(m.dx) <= 2 * PIXEL_SIZE


def test_prior_narrows_search_and_still_recovers():
    stretch = fib_view_y_stretch(MILLING_ANGLE, COLUMN_TILT)
    dy_px, dx_px = 40.0, 8.0
    sem = make_sem_scene()
    fib = make_fib_view(sem, dy_px, dx_px, stretch)
    prior = (dx_px * PIXEL_SIZE, dy_px / stretch * PIXEL_SIZE)

    m = measure_coincidence(
        sem,
        fib,
        PIXEL_SIZE,
        MILLING_ANGLE,
        column_tilt=COLUMN_TILT,
        prior=prior,
        capture_range=3e-6,
    )

    assert m.is_reliable
    assert m.seeded
    assert m.dx == pytest.approx(dx_px * PIXEL_SIZE, abs=2 * PIXEL_SIZE)


def test_featureless_scene_is_refused():
    rng = np.random.default_rng(0)
    sem = rng.normal(100, 3, SHAPE).astype(np.float32)
    fib = rng.normal(100, 3, SHAPE).astype(np.float32)

    m = measure_coincidence(
        sem, fib, PIXEL_SIZE, MILLING_ANGLE, column_tilt=COLUMN_TILT
    )

    assert not m.is_reliable
    assert m.refusal_reason in (REFUSAL_BAND_DISAGREEMENT, REFUSAL_WINDOW_EDGE)


def test_offset_beyond_capture_range_is_refused_not_guessed():
    stretch = fib_view_y_stretch(MILLING_ANGLE, COLUMN_TILT)
    sem = make_sem_scene()
    # true offset well outside the search window
    fib = make_fib_view(sem, 120.0, 90.0, stretch)

    m = measure_coincidence(
        sem,
        fib,
        PIXEL_SIZE,
        MILLING_ANGLE,
        column_tilt=COLUMN_TILT,
        capture_range=2e-6,
    )

    assert not m.is_reliable


def test_height_error_conversion():
    assert height_error_from_fib_shift(1e-6, np.deg2rad(90)) == pytest.approx(1e-6)
    assert height_error_from_fib_shift(
        np.sin(COLUMN_TILT) * 1e-6, COLUMN_TILT
    ) == pytest.approx(1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        measure_coincidence(
            np.zeros((10, 10), dtype=np.float32),
            np.zeros((10, 12), dtype=np.float32),
            PIXEL_SIZE,
            MILLING_ANGLE,
        )


def test_measurement_dataclass_defaults():
    m = CoincidenceMeasurement(
        dx=0.0, dy=0.0, dz=0.0, band_disagreement=0.0, is_reliable=True
    )
    assert m.refusal_reason is None
    assert m.method == "crossbeam-xcorr"
    assert not m.seeded


def _image_with_metadata(data: np.ndarray, pixel_size: float) -> "FibsemImage":
    from fibsem.structures import (
        FibsemHardwareGeometry,
        FibsemImage,
        FibsemImageMetadata,
        FibsemStagePosition,
        ImageSettings,
        MicroscopeState,
        Point,
    )

    # stage tilt for a 15 deg milling angle with 35 deg pretilt, 52 deg column:
    # stage_tilt = milling + column + pretilt - 90 = 12 deg
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(
            hfw=pixel_size * data.shape[1], resolution=(data.shape[1], data.shape[0])
        ),
        microscope_state=MicroscopeState(
            stage_position=FibsemStagePosition(t=np.deg2rad(12.0))
        ),
        pixel_size=Point(pixel_size, pixel_size),
        hardware_geometry=FibsemHardwareGeometry(
            column_tilt=0,
            fib_column_tilt=52.0,
            shuttle_pre_tilt=35.0,
        ),
    )
    return FibsemImage(data=data.astype(np.uint8), metadata=metadata)


def test_geometry_from_metadata():
    from fibsem.alignment.coincidence import geometry_from_metadata

    image = _image_with_metadata(np.zeros(SHAPE, dtype=np.uint8), PIXEL_SIZE)
    pixel_size, milling_angle, column_tilt = geometry_from_metadata(image)
    assert pixel_size == pytest.approx(PIXEL_SIZE)
    assert milling_angle == pytest.approx(np.deg2rad(15.0))
    assert column_tilt == pytest.approx(np.deg2rad(52.0))


def test_measure_from_images_and_diagnostic_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from fibsem.alignment.coincidence import measure_coincidence_from_images
    from fibsem.alignment.plotting import plot_coincidence_measurement

    stretch = fib_view_y_stretch(np.deg2rad(15.0), np.deg2rad(52.0))
    sem = np.clip(make_sem_scene(), 0, 255)
    fib = np.clip(make_fib_view(sem, 20.0, 5.0, stretch), 0, 255)
    sem_image = _image_with_metadata(sem, PIXEL_SIZE)
    fib_image = _image_with_metadata(fib, PIXEL_SIZE)

    m = measure_coincidence_from_images(sem_image, fib_image)
    assert m.is_reliable
    assert m.dx == pytest.approx(5.0 * PIXEL_SIZE, abs=2 * PIXEL_SIZE)

    fig = plot_coincidence_measurement(
        sem_image, fib_image, m, save=True, path=str(tmp_path)
    )
    assert fig is not None
    pngs = list(tmp_path.glob("*coincidence*.png"))
    assert len(pngs) == 1
