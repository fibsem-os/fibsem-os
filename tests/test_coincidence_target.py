"""Tests for fibsem.alignment.coincidence_target: the experimental measure that
finds the SEM target in the FIB view (2026-10-16).

Synthetic pairs as in test_coincidence: the FIB view is the SEM scene squashed
by the pair's foreshortening, shifted by a known amount, contrast inverted,
with its own noise. The measure must find the target there, past the whole
frame measure's 20 um capture range, and report dx on its own.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from fibsem.alignment.coincidence import (
    DEFAULT_CAPTURE_RANGE,
    geometry_from_images,
    measure_coincidence_from_images,
)
from fibsem.alignment.coincidence_target import (
    measure_coincidence_at_target,
    render_target_coincidence,
)
from fibsem.structures.settings import BeamType
from tests.test_coincidence import _image_with_metadata, make_fib_view

PIXEL_SIZE = 0.2e-6  # 154 um field at 768 px
SHAPE = (512, 768)


def make_scene(seed: int = 3) -> np.ndarray:
    """Sparse blobs and one bright raised 'cell' at the centre: the target."""
    rng = np.random.default_rng(seed)
    scene = np.zeros(SHAPE, dtype=np.float32)
    ys = rng.integers(20, SHAPE[0] - 20, size=120)
    xs = rng.integers(20, SHAPE[1] - 20, size=120)
    for y, x in zip(ys, xs):
        scene[y, x] = rng.uniform(80, 250)
    scene = ndi.gaussian_filter(scene, 2.5) * 20
    cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
    yy, xx = np.ogrid[0 : SHAPE[0], 0 : SHAPE[1]]
    scene[((yy - cy) / 30) ** 2 + ((xx - cx) / 45) ** 2 < 1] += 120
    scene += rng.normal(0, 2, SHAPE).astype(np.float32)
    return np.clip(scene, 0, 255)


def _pair(sem, fib):
    return (
        _image_with_metadata(sem, PIXEL_SIZE, BeamType.ELECTRON),
        _image_with_metadata(fib, PIXEL_SIZE, BeamType.ION),
    )


def _fib(sem, dy_fib_m: float, dx_m: float = 0.0):
    """The FIB view with the target displaced by *dy_fib_m* in the FIB plane."""
    stretch = geometry_from_images(*_pair(sem, sem)).y_stretch
    dy_px = dy_fib_m / PIXEL_SIZE * stretch  # make_fib_view's stretched space
    return np.clip(make_fib_view(sem, dy_px, dx_m / PIXEL_SIZE, stretch), 0, 255)


def test_finds_the_target_past_the_whole_frame_capture_range():
    """26.8 um, what the agent measured on the Arctis: outside the fine pass."""
    sem = make_scene()
    true_dy = 26.8e-6
    assert true_dy > DEFAULT_CAPTURE_RANGE
    sem_image, fib_image = _pair(sem, _fib(sem, true_dy))

    whole_frame = measure_coincidence_from_images(sem_image, fib_image)
    result = measure_coincidence_at_target(sem_image, fib_image)

    assert not whole_frame.is_reliable
    assert result.dy == pytest.approx(true_dy, abs=3 * PIXEL_SIZE)
    assert abs(result.dx) < 3 * PIXEL_SIZE
    assert result.score > 0.5
    assert result.agrees, result.warnings
    geometry = geometry_from_images(sem_image, fib_image)
    assert result.dz == pytest.approx(result.dy * geometry.dz_per_dy)
    assert result.y_stretch == pytest.approx(geometry.y_stretch, rel=0.03)


def test_reports_dx_and_says_it_is_not_height():
    sem = make_scene()
    sem_image, fib_image = _pair(sem, _fib(sem, -8e-6, dx_m=7e-6))

    result = measure_coincidence_at_target(sem_image, fib_image)

    assert result.dy == pytest.approx(-8e-6, abs=3 * PIXEL_SIZE)
    assert result.dx == pytest.approx(7e-6, abs=3 * PIXEL_SIZE)
    assert any("cannot move x" in warning for warning in result.warnings)
    assert not result.agrees


def test_an_off_centre_target():
    sem = make_scene()
    sem_image, fib_image = _pair(sem, _fib(sem, 12e-6))
    # a target 20 um right of and 10 um above the centre, on the same scene
    result = measure_coincidence_at_target(sem_image, fib_image, target=(20e-6, 10e-6))

    assert result.target_px == pytest.approx((384 + 100, 256 - 50))
    assert result.dy == pytest.approx(12e-6, abs=3 * PIXEL_SIZE)


def test_a_target_outside_the_image_is_refused():
    sem = make_scene()
    with pytest.raises(ValueError):
        measure_coincidence_at_target(*_pair(sem, sem), target=(1e-3, 0.0))


def test_the_overlay_shows_both_views():
    sem = make_scene()
    sem_image, fib_image = _pair(sem, _fib(sem, 5e-6))
    result = measure_coincidence_at_target(sem_image, fib_image)

    overlay = render_target_coincidence(sem_image, fib_image, result)

    assert overlay.dtype == np.uint8
    assert overlay.shape[1] == 1024 and overlay.shape[2] == 3
    assert result.to_dict()["method"] == "target-patch"


def test_measured_from_the_fib_gives_the_same_height_error():
    """Centred in the FIB, the target is found in the SEM: dy is then in the
    SEM view (y_stretch times larger) and dz is the same height error."""
    sem = make_scene()
    sem_image, fib_image = _pair(sem, _fib(sem, 6e-6))
    from_sem = measure_coincidence_at_target(sem_image, fib_image)
    h, w = SHAPE
    # the target where the SEM measurement found it in the FIB
    target = (
        (from_sem.found_px[0] - w / 2) * PIXEL_SIZE,
        -(from_sem.found_px[1] - h / 2) * PIXEL_SIZE,
    )

    from_fib = measure_coincidence_at_target(
        sem_image, fib_image, target=target, reference=BeamType.ION
    )

    assert from_fib.reference == "ION"
    assert from_fib.dz == pytest.approx(from_sem.dz, abs=0.5e-6)
    assert from_fib.dy == pytest.approx(
        -from_sem.dy * from_fib.y_stretch, abs=5 * PIXEL_SIZE
    )
    assert abs(from_fib.dx) < 3 * PIXEL_SIZE
    overlay = render_target_coincidence(sem_image, fib_image, from_fib)
    assert overlay.shape[1] == 1024
