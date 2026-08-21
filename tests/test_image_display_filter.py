"""Pins FibsemImage.filtered_data to the scipy filter it replaced.

The display filter moved from scipy to opencv for speed (the ``.data`` setter runs it
on every assignment, including inside ``FibsemImage.load()``, which hung the app for
~57 s on a 385 MB overview). Matching scipy is not automatic: it needs a kernel size
derived from scipy's ``truncate=4.0`` and ``BORDER_REFLECT``. On opencv's defaults the
outputs are ~16x further apart, so the tolerance here is what catches a "simplification"
back to the defaults.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, median_filter

from fibsem.autofunctions.metrics import get_focus_measure_function
from fibsem.structures import FibsemImage

# most pixels differ by exactly 1 (a rounding convention), none by more than 2.
FILTER_TOLERANCE = 2

FOCUS_METHODS = ["laplacian", "sobel", "variance", "tenengrad"]


def scipy_filter(data):
    """The filter FibsemImage used before opencv. The reference these tests pin to."""
    return gaussian_filter(median_filter(data, size=3), sigma=1)


def beam_like_image(dtype=np.uint16, blur=0.0, seed=0, shape=(256, 320)):
    """Something with the structure of a beam image: gradients, a bright feature, noise."""
    height, width = shape
    rng = np.random.default_rng(seed)
    max_value = np.iinfo(dtype).max
    yy, xx = np.mgrid[0:height, 0:width]
    data = (0.5 + 0.35 * np.sin(xx / 30.0) * np.cos(yy / 41.0)) * max_value
    data[height // 3 : height // 3 + 20, width // 4 : width // 4 + 100] = (
        max_value * 0.95
    )
    data[50:200:37, :] = max_value * 0.2  # fine structure a focus measure keys on
    if blur:
        data = gaussian_filter(data, sigma=blur)
    data = data + rng.normal(0, max_value * 0.01, size=shape)
    return np.clip(data, 0, max_value).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("kind", ["beam", "noise"])
def test_filtered_data_matches_the_scipy_filter_it_replaced(dtype, kind):
    if kind == "beam":
        data = beam_like_image(dtype=dtype)
    else:
        # pure noise is the worst case for the median: every pixel is an edge.
        data = (
            np.random.default_rng(1).random((256, 320)) * np.iinfo(dtype).max
        ).astype(dtype)

    filtered = FibsemImage(data=data).filtered_data

    difference = np.abs(filtered.astype(np.int64) - scipy_filter(data).astype(np.int64))
    assert difference.max() <= FILTER_TOLERANCE, (
        f"display filter drifted from scipy: max difference {difference.max()} grey levels"
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_filtered_data_keeps_the_shape_and_dtype_of_data(dtype):
    data = beam_like_image(dtype=dtype, shape=(64, 80))

    image = FibsemImage(data=data)

    assert image.filtered_data.shape == data.shape
    assert image.filtered_data.dtype == data.dtype


def test_filtered_data_handles_an_unsqueezed_3d_assignment():
    """(H, W, 1) reaches the setter: only __init__ squeezes it, a direct assignment does not."""
    data = beam_like_image(shape=(64, 80))
    image = FibsemImage(data=data)

    image.data = data[:, :, np.newaxis]

    assert image.filtered_data.shape == image.data.shape
    assert np.array_equal(
        image.filtered_data.reshape(data.shape), FibsemImage(data=data).filtered_data
    )


@pytest.mark.parametrize("method", FOCUS_METHODS)
def test_autofocus_picks_the_same_step_after_the_filter_swap(method):
    """The filter is a display filter, but autofocus scores an algorithm off it.

    Mirrors the score at ``autofocus.py:300`` over a sweep through focus. What is
    asserted is the decision the sweep makes -- which step wins -- and that the winning
    score itself barely moves. Deliberately not the ranking of the whole sweep: on three
    real autofocus runs the peak was untouched (<=0.05%) but the deep-defocus tail moved
    up to 2.2% and did reorder, because there the score is small and nearly flat. The
    peak stands ~5x above that tail, so reordering down there cannot move the winner.
    """
    focus_fn = get_focus_measure_function(method)
    # a WD sweep: sharp in the middle, defocused at both ends.
    blurs = [4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 4.0]
    images = [
        beam_like_image(blur=blur, seed=index) for index, blur in enumerate(blurs)
    ]

    def score(data):
        return float(np.mean(focus_fn(data.astype(np.float32))))

    scores = np.array(
        [score(FibsemImage(data=image).filtered_data) for image in images]
    )
    reference = np.array([score(scipy_filter(image)) for image in images])

    assert int(np.argmax(scores)) == int(np.argmax(reference)), (
        "the sweep would choose a different working distance"
    )
    peak = np.argmax(reference)
    peak_shift = abs(scores[peak] - reference[peak]) / abs(reference[peak])
    assert peak_shift < 0.005, f"score at the chosen step moved by {peak_shift:.2%}"
