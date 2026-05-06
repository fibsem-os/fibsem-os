import pytest
from fibsem import utils, acquire, alignment
from fibsem.alignment import crosscorrelation_cv2, shift_from_crosscorrelation_v2
import numpy as np

@pytest.mark.parametrize("offset", [-30, -10, 0, 10, 30])
def test_align_from_crosscorrelation(offset):

    microscope, settings = utils.setup_session(debug=False)

    ref_image = acquire.acquire_image(microscope, settings.image)
    new_image = acquire.acquire_image(microscope, settings.image)

    ref_image.data[:] = 0
    new_image.data[:] = 0

    w = h = 150
    x, y = 200, 200
    ref_image.data[y:y + h, x:x + w] = 255
    new_image.data[y + offset:y + h + offset, x + offset:x + w + offset] = 255

    dx, dy, xcorr = alignment.shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    pixel_size = ref_image.metadata.pixel_size.x
    assert np.isclose(dx, offset * pixel_size, atol=pixel_size), f"dx: {dx}, offset: {offset * pixel_size}"
    assert np.isclose(dy, offset * pixel_size, atol=pixel_size), f"dy: {dy}, offset: {offset * pixel_size}"


# ---------------------------------------------------------------------------
# crosscorrelation_cv2
# ---------------------------------------------------------------------------

def test_crosscorrelation_cv2_known_shift():
    """crosscorrelation_cv2 recovers a known integer shift within 1 pixel."""
    rng = np.random.default_rng(42)
    img = rng.random((256, 256), dtype=np.float32)
    shift_col, shift_row = 15, -10
    shifted = np.roll(np.roll(img, shift_row, axis=0), shift_col, axis=1)

    sx, sy, response = crosscorrelation_cv2(img, shifted)

    assert abs(sx - shift_col) < 1.0, f"x shift {sx:.2f} != expected {shift_col}"
    assert abs(sy - shift_row) < 1.0, f"y shift {sy:.2f} != expected {shift_row}"


def test_crosscorrelation_cv2_response_good_match():
    """Response is high (> 0.5) for a well-matched pair."""
    rng = np.random.default_rng(7)
    img = rng.random((256, 256), dtype=np.float32)
    shifted = np.roll(img, 5, axis=1)

    _, _, response = crosscorrelation_cv2(img, shifted)

    assert response > 0.5, f"Expected high response, got {response:.3f}"


def test_crosscorrelation_cv2_response_low_for_noise():
    """Response is low (< 0.3) when the two images are unrelated noise."""
    rng = np.random.default_rng(99)
    img1 = rng.random((256, 256), dtype=np.float32)
    img2 = rng.random((256, 256), dtype=np.float32)

    _, _, response = crosscorrelation_cv2(img1, img2)

    assert response < 0.3, f"Expected low response for noise pair, got {response:.3f}"


def test_crosscorrelation_cv2_zero_shift():
    """Identical images should return near-zero shift and high response."""
    rng = np.random.default_rng(1)
    img = rng.random((256, 256), dtype=np.float32)

    sx, sy, response = crosscorrelation_cv2(img, img)

    assert abs(sx) < 0.5, f"x shift {sx:.3f} should be ~0 for identical images"
    assert abs(sy) < 0.5, f"y shift {sy:.3f} should be ~0 for identical images"
    assert response > 0.8, f"Response {response:.3f} should be high for identical images"


# ---------------------------------------------------------------------------
# shift_from_crosscorrelation_v2
# ---------------------------------------------------------------------------

def test_shift_from_crosscorrelation_v2_known_shift():
    """v2 recovers a known shift in metres within 1 pixel."""
    microscope, settings = utils.setup_session(debug=False)

    ref_image = acquire.acquire_image(microscope, settings.image)
    new_image = acquire.acquire_image(microscope, settings.image)

    ref_image.data[:] = 0
    new_image.data[:] = 0

    w = h = 150
    offset = 20
    x, y = 200, 200
    ref_image.data[y:y + h, x:x + w] = 255
    new_image.data[y + offset:y + h + offset, x + offset:x + w + offset] = 255

    pixel_size = ref_image.metadata.pixel_size.x

    dx_v2, dy_v2, response = shift_from_crosscorrelation_v2(ref_image, new_image)

    assert response > 0.0, "response should be positive"
    assert np.isclose(dx_v2, offset * pixel_size, atol=pixel_size), \
        f"dx_v2={dx_v2:.2e}, expected {offset * pixel_size:.2e}"
    assert np.isclose(dy_v2, offset * pixel_size, atol=pixel_size), \
        f"dy_v2={dy_v2:.2e}, expected {offset * pixel_size:.2e}"


def test_shift_from_crosscorrelation_v2_minimum_response_gate():
    """When response is below minimum_response, (0, 0, response) is returned."""
    microscope, settings = utils.setup_session(debug=False)

    ref_image = acquire.acquire_image(microscope, settings.image)
    new_image = acquire.acquire_image(microscope, settings.image)

    # pure noise pair → low response
    rng = np.random.default_rng(55)
    ref_image.data[:] = (rng.random(ref_image.data.shape) * 255).astype(np.uint8)
    new_image.data[:] = (rng.random(new_image.data.shape) * 255).astype(np.uint8)

    dx, dy, response = shift_from_crosscorrelation_v2(ref_image, new_image, minimum_response=1.0)

    assert dx == 0.0 and dy == 0.0, \
        f"Expected zero shift when response {response:.3f} < minimum_response"


