"""Images are reduced for display by averaging, not by sampling (FIB-589).

Every canvas caps what each placed image stores, so a 1024-px tile is drawn from a few
hundred. The reduction used to be `arr[::n, ::n]`, which does not blur what it leaves
out — it deletes it. A punctum a pixel or two across survived only when it happened to
land on a sampled row, so it vanished when zoomed out, came back when zoomed in, and
blinked when the view shifted by a pixel, with nothing on screen saying so. For
fluorescence that is the worst available failure mode: the small bright thing is the
thing being looked for.

What is pinned here is the property, not the implementation — `test_a_lone_punctum...`
fails outright against striding. The rest guard the things a faster implementation is
likely to quietly break: the output shape downstream inherited, the dtype matplotlib
reads the value range from, and the ragged final block.

Lives in `tests/` rather than `tests/ui/` because `fibsem.imaging.reduce` needs no Qt.
That is the point of it being there: everything under `fibsem.ui` is skipped in CI,
which installs `.[test]` without the UI extra.
"""

import math

import cv2
import numpy as np
import pytest

from fibsem.imaging.reduce import _CV2_DTYPES, _box_mean_numpy, downsample


def stride(arr: np.ndarray, max_px: int) -> np.ndarray:
    """The implementation this replaced, for shape parity and discrimination."""
    h, w = arr.shape[:2]
    if h <= max_px and w <= max_px:
        return arr
    n = max(1, math.ceil(max(h, w) / max_px))
    return arr[::n, ::n] if arr.ndim == 2 else arr[::n, ::n, :]


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Reference block mean, edge-padding the ragged final block. Deliberately naive."""
    h, w = arr.shape[:2]
    out_h, out_w = math.ceil(h / factor), math.ceil(w / factor)
    pad = [(0, out_h * factor - h), (0, out_w * factor - w)] + [(0, 0)] * (arr.ndim - 2)
    padded = np.pad(arr.astype(np.float64), pad, mode="edge")
    return padded.reshape(out_h, factor, out_w, factor, *arr.shape[2:]).mean(axis=(1, 3))


class TestFineStructureSurvives:
    def test_a_lone_punctum_does_not_blink_with_the_sampling_grid(self):
        """The reported failure. Striding fails this test at five offsets out of six.

        One bright pixel, moved a few pixels at a time, reduced 8x. Sampling reports 255
        at the one offset that lands on the grid and 0 at every other — so panning makes
        the feature flash. Averaging reports the same attenuated value wherever it sits.
        """
        seen = set()
        for shift in range(8):
            frame = np.zeros((512, 512), np.uint8)
            frame[100 + shift, 100 + shift] = 255
            seen.add(int(downsample(frame, 64).max()))

        assert len(seen) == 1, (
            f"the same punctum reduced to {sorted(seen)} depending only on where it sat "
            f"relative to the sampling grid — it will blink as the view is panned"
        )
        assert seen != {0}, "the punctum was dropped at every offset"

    def test_fine_structure_is_averaged_rather_than_deleted(self):
        """The checkerboard from the issue: 1-px structure, reduced 8x."""
        board = (np.indices((512, 512)).sum(axis=0) % 2 * 255).astype(np.uint8)

        assert set(np.unique(stride(board, 64))) == {0}, "premise: striding deletes it"
        assert set(np.unique(downsample(board, 64))) == {128}, "expected mid-grey"

    def test_a_reduced_image_keeps_the_brightness_of_the_original(self):
        """Sampling reports whatever it landed on; a mean cannot drift off the source."""
        rng = np.random.default_rng(0)
        img = (rng.random((1024, 1024)) * 200 + 20).astype(np.uint8)

        assert downsample(img, 128).mean() == pytest.approx(img.mean(), abs=0.5)


class TestTheContractDownstreamInherited:
    SHAPES = [(1024, 1024), (1000, 900), (1023, 777), (10, 10000), (513, 512), (5, 5),
              (4096, 4096), (777, 1023, 3), (100, 3000, 4)]

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("max_px", [512, 128, 64])
    def test_the_output_shape_is_what_striding_gave(self, shape, max_px):
        """`ceil(n / factor)` per axis. Nothing downstream should notice the change."""
        arr = np.zeros(shape, np.uint8)

        assert downsample(arr, max_px).shape == stride(arr, max_px).shape

    def test_an_image_small_enough_is_returned_untouched(self):
        """Not merely equal — the same object, as before. The common case allocates nothing."""
        small = np.zeros((100, 100), np.uint8)

        assert downsample(small, 512) is small

    def test_a_reduced_image_is_a_copy_rather_than_a_view(self):
        """The cost the caller is paying, stated so it cannot be mistaken for a saving.

        `arr[::n, ::n]` was free and shared the source's buffer, which also kept the
        source alive. That sounds like this change reduces memory, and on the canvas it
        does not: matplotlib copies at `imshow`, so the source was released either way.
        Measured with weakrefs over twenty placed tiles, neither reduction leaves a
        single source alive. What changes here is only that the copy happens earlier.
        """
        arr = np.zeros((1024, 1024), np.uint8)

        assert not np.shares_memory(downsample(arr, 512), arr)

    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64", "int32"])
    @pytest.mark.parametrize("shape", [(1024, 1024), (1024, 1024, 3), (1024, 1024, 4)])
    def test_dtype_and_channels_are_preserved(self, dtype, shape):
        """matplotlib reads an RGB array's value range from its dtype — uint8 is 0-255
        and float is 0-1, so returning a float from a uint8 image renders white."""
        arr = (np.random.default_rng(1).random(shape) * 100).astype(dtype)

        out = downsample(arr, 128)

        assert out.dtype == arr.dtype
        assert out.shape[2:] == arr.shape[2:]


class TestTheBlocksThemselves:
    @pytest.mark.parametrize("shape,max_px", [
        ((1024, 1024), 512),   # factor 2, divides exactly
        ((1024, 1024), 128),   # factor 8, divides exactly
        ((1023, 777), 512),    # factor 2, ragged on both axes
        ((1024, 4712), 512),   # factor 10, ragged — the mosaic case
        ((7, 7), 3),           # tiny, so a wrong tail is a third of the picture
    ])
    def test_each_output_pixel_is_the_mean_of_its_block(self, shape, max_px):
        rng = np.random.default_rng(2)
        arr = (rng.random(shape) * 255).astype(np.uint8)
        factor = max(1, math.ceil(max(shape[:2]) / max_px))

        out = downsample(arr, max_px).astype(float)

        # 0.5 is integer rounding; anything larger is a real disagreement.
        assert np.abs(out - block_mean(arr, factor)).max() <= 0.5

    def test_the_ragged_final_block_is_not_cropped_away(self):
        """Cropping the remainder instead of padding it would stretch the picture inside
        the extent it is drawn at, which for the real-space canvas means placing it
        somewhere it was not acquired."""
        arr = np.zeros((10, 10), np.uint8)
        arr[-1, -1] = 255  # only in the final, short block

        # factor 3 -> blocks of 3, 3, 3, and a last one of 1
        assert downsample(arr, 4)[-1, -1] > 0, "the last row and column were dropped"


class TestTheTwoImplementationsAgree:
    """cv2 does the work; numpy answers for the dtypes cv2 refuses. They must match."""

    @pytest.mark.parametrize("dtype", sorted(str(d) for d in _CV2_DTYPES))
    def test_cv2_really_does_accept_every_listed_dtype(self, dtype):
        """The list is a hand-written claim about OpenCV. Check it against OpenCV.

        Getting it wrong in this direction is the silent failure: an unsupported dtype
        would raise `cv2.error` out of a paint, which PyQt5 turns into a process abort
        (FIB-329) rather than a traceback.
        """
        arr = (np.random.default_rng(3).random((600, 600)) * 100).astype(dtype)

        assert downsample(arr, 128).dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "bool"])
    def test_a_dtype_cv2_refuses_still_reduces(self, dtype):
        """Striding worked on anything. The fallback keeps that true."""
        arr = (np.random.default_rng(4).random((600, 600)) * 100).astype(dtype)
        with pytest.raises(cv2.error):
            cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)

        out = downsample(arr, 128)

        assert out.dtype == arr.dtype
        assert out.shape == stride(arr, 128).shape

    def test_the_fallback_gives_what_cv2_gives(self):
        """Cross-checked on a dtype both take, since no input exercises both paths."""
        rng = np.random.default_rng(5)
        arr = (rng.random((1023, 777)) * 255).astype(np.uint8)
        factor = 2

        by_cv2 = downsample(arr, 512)
        by_numpy = _box_mean_numpy(arr, factor, math.ceil(1023 / factor), math.ceil(777 / factor))

        assert by_numpy.shape == by_cv2.shape
        assert np.abs(by_numpy.astype(int) - by_cv2.astype(int)).max() <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
