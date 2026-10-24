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

from fibsem.imaging.reduce import (
    _CV2_DTYPES,
    PreviewMosaic,
    _box_mean_numpy,
    downsample,
    downsample_mask,
)


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
    return padded.reshape(out_h, factor, out_w, factor, *arr.shape[2:]).mean(
        axis=(1, 3)
    )


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
    SHAPES = [
        (1024, 1024),
        (1000, 900),
        (1023, 777),
        (10, 10000),
        (513, 512),
        (5, 5),
        (4096, 4096),
        (777, 1023, 3),
        (100, 3000, 4),
    ]

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

    @pytest.mark.parametrize(
        "dtype", ["uint8", "uint16", "int16", "float32", "float64", "int32"]
    )
    @pytest.mark.parametrize("shape", [(1024, 1024), (1024, 1024, 3), (1024, 1024, 4)])
    def test_dtype_and_channels_are_preserved(self, dtype, shape):
        """matplotlib reads an RGB array's value range from its dtype — uint8 is 0-255
        and float is 0-1, so returning a float from a uint8 image renders white."""
        arr = (np.random.default_rng(1).random(shape) * 100).astype(dtype)

        out = downsample(arr, 128)

        assert out.dtype == arr.dtype
        assert out.shape[2:] == arr.shape[2:]


class TestTheBlocksThemselves:
    @pytest.mark.parametrize(
        "shape,max_px",
        [
            ((1024, 1024), 512),  # factor 2, divides exactly
            ((1024, 1024), 128),  # factor 8, divides exactly
            ((1023, 777), 512),  # factor 2, ragged on both axes
            ((1024, 4712), 512),  # factor 10, ragged — the mosaic case
            ((7, 7), 3),  # tiny, so a wrong tail is a third of the picture
        ],
    )
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
        by_numpy = _box_mean_numpy(
            arr, factor, math.ceil(1023 / factor), math.ceil(777 / factor)
        )

        assert by_numpy.shape == by_cv2.shape
        assert np.abs(by_numpy.astype(int) - by_cv2.astype(int)).max() <= 1


def float_mask(mask: np.ndarray, max_px: int) -> np.ndarray:
    """The spelling `downsample_mask` replaced, kept as the reference to measure against.

    Correct, and four bytes of temporary per *source* pixel — which is the cost that
    grows with the mosaic rather than with what comes out.
    """
    return downsample(mask.astype(np.float32), max_px) > 0.5


class TestReducingACoverageMask:
    """A block survives when *more* than half of it is set.

    Reduced alongside the pixels so an overview's colour and its "was anything acquired
    here" answer keep the same shape. The rule has to resolve a part-covered block one
    way or the other, and it resolves inward: coverage that spreads outward claims
    ground as acquired that half of it was not, and on a real-space canvas that ground
    is drawn opaque over whatever else is beneath it.
    """

    @pytest.mark.parametrize(
        "covered,expected",
        [
            (0, False),
            (10, False),
            (49, False),
            (50, False),  # exactly half — the tie, resolved inward
            (51, True),
            (90, True),
            (100, True),
        ],
    )
    def test_a_block_survives_when_most_of_it_is_set(self, covered, expected):
        block = np.zeros((10, 10), dtype=bool)
        block.flat[:covered] = True
        # 10x10 identical blocks, so every output pixel answers the same question.
        out = downsample_mask(np.tile(block, (10, 10)), 10)

        assert out.shape == (10, 10)
        assert bool(out.all()) is expected and bool(out.any()) is expected

    def test_it_is_not_the_pixels_thresholded(self):
        """Box-averaging intensities and testing for "greater than zero" marks a block
        acquired when *any* of it is — which over a mosaic's unacquired ground is most
        of it, and is how a part-finished overview comes to paint black over the one
        beneath it (FIB-630)."""
        data = np.zeros((100, 100), np.uint8)
        data[:, ::10] = 255  # one column of every 10x10 block, so each is a tenth set

        generous = downsample(data, 10) > 0
        honest = downsample_mask(data > 0, 10)

        assert generous.all(), "the reference rule did not do the generous thing"
        assert not honest.any(), "a tenth-covered block was called acquired"

    @pytest.mark.parametrize(
        "shape,max_px,cut",
        [
            ((5120, 5120), 2048, (3000, 3777)),  # the 5x5 mosaic, factor 3
            ((777, 301), 64, (500, 200)),  # ragged on both axes, factor 13
            ((1000, 1000), 97, (613, 411)),  # factor 11, ragged
        ],
    )
    def test_it_matches_the_float_spelling_on_a_real_mask(self, shape, max_px, cut):
        """Down to the last block, which is the claim that makes the cheaper spelling a
        substitution rather than a change.

        A *coverage* mask is what it is asked to answer for, and the edge of one is the
        straight edge of a tile: a block on it is covered in whole rows, so its fraction
        moves in steps of 1/factor and never lands near the half where eight bits of
        quantisation could decide it either way.
        """
        mask = np.zeros(shape, dtype=bool)
        mask[: cut[0], : cut[1]] = True

        assert np.array_equal(downsample_mask(mask, max_px), float_mask(mask, max_px))

    def test_a_sparse_tileset_reduces_the_same_way(self):
        """The other real shape: a masked run leaves whole tiles unacquired (FIB-618),
        so coverage is a chequerboard rather than one region."""
        mask = np.zeros((5120, 5120), dtype=bool)
        for row in range(5):
            for col in range(5):
                if (row + col) % 2 == 0:
                    mask[
                        row * 1024 : (row + 1) * 1024, col * 1024 : (col + 1) * 1024
                    ] = True

        assert np.array_equal(downsample_mask(mask, 2048), float_mask(mask, 2048))

    def test_uniform_noise_is_where_the_two_part_company(self):
        """Named so the agreement above is not read as a promise about anything else.

        Half-set noise puts a block's coverage right in the quantisation band, and the
        two spellings then disagree readily. Harmless for coverage, which is never this
        — but this is the input that would make a future caller's assumption wrong.
        """
        noise = np.random.default_rng(0).random((777, 301)) > 0.5

        differ = (downsample_mask(noise, 64) != float_mask(noise, 64)).sum()

        assert differ > 0, "the quantisation band closed; the docstring overstates it"

    def test_it_refuses_anything_that_is_not_a_mask(self):
        """`(data > 0)` is the caller's job, and doing it here would hide the one call
        that forgot: passing raw pixels reduces intensities and thresholds them at 128,
        which is a picture-dependent answer that looks plausible everywhere."""
        with pytest.raises(TypeError):
            downsample_mask(np.zeros((100, 100), np.uint8), 10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── the live preview mosaic ──────────────────────────────────────────────


class TestPreviewMosaic:
    """The decimated mosaic both tiled runners fill in as a run goes.

    Written twice until FIB-699, with the same stride rule, the same `downsample` call
    and the same paste — the only real difference being the channel axis. Tested here
    rather than through either runner: this is where the arithmetic lives, and neither
    runner's tests can be run by CI.
    """

    def test_a_mosaic_that_fits_is_not_reduced(self):
        mosaic = PreviewMosaic(2048, 1024, dtype=np.uint8)
        assert mosaic.stride == 1
        assert mosaic.canvas.shape == (1024, 2048)

    def test_the_stride_comes_from_the_long_edge(self):
        """Not from the area, and not per axis: the cap is on the longest side, so a
        wide mosaic and a tall one of the same span reduce by the same factor."""
        wide = PreviewMosaic(8192, 1024, dtype=np.uint8)
        tall = PreviewMosaic(1024, 8192, dtype=np.uint8)
        assert wide.stride == tall.stride == 4
        assert wide.canvas.shape == (256, 2048)
        assert tall.canvas.shape == (2048, 256)

    def test_channels_none_is_not_channels_one(self):
        """A beam mosaic is a plane; a one-channel fluorescence mosaic is a stack of
        one. The consumers index them differently, so the distinction is load-bearing."""
        assert PreviewMosaic(64, 64, dtype=np.uint8).canvas.ndim == 2
        assert PreviewMosaic(64, 64, dtype=np.uint16, channels=1).canvas.ndim == 3

    def test_a_tile_lands_where_the_stride_puts_it(self):
        mosaic = PreviewMosaic(8192, 8192, dtype=np.uint8)  # stride 4
        tile = np.full((1024, 1024), 200, dtype=np.uint8)
        mosaic.paint(tile, canvas_x=4096, canvas_y=2048)

        assert mosaic.canvas[512, 1024] == 200  # 2048//4, 4096//4
        assert mosaic.canvas[511, 1024] == 0  # just above it
        assert mosaic.canvas[512, 1023] == 0  # just left of it

    def test_each_channel_is_reduced_on_its_own_axes(self):
        """`downsample` reads shape as (y, x[, c]), so reducing a (c, y, x) stack whole
        would average across *channels* and y -- the right answer on the wrong axes."""
        mosaic = PreviewMosaic(4096, 4096, dtype=np.uint16, channels=3)  # stride 2
        tile = np.stack(
            [np.full((512, 512), v, dtype=np.uint16) for v in (100, 200, 300)]
        )
        mosaic.paint(tile, canvas_x=0, canvas_y=0)

        assert [int(mosaic.canvas[c, 0, 0]) for c in range(3)] == [100, 200, 300]

    def test_a_tile_over_the_edge_is_clipped_rather_than_raising(self):
        """Every axis is rounded up independently, so the last tile can extend past the
        canvas by a pixel or two. A run must not fail for that."""
        mosaic = PreviewMosaic(600, 600, dtype=np.uint8)
        tile = np.full((512, 512), 7, dtype=np.uint8)
        mosaic.paint(tile, canvas_x=500, canvas_y=500)

        assert mosaic.canvas[599, 599] == 7

    def test_a_tile_entirely_off_the_canvas_paints_nothing(self):
        mosaic = PreviewMosaic(600, 600, dtype=np.uint8)
        mosaic.paint(np.full((64, 64), 9, dtype=np.uint8), canvas_x=900, canvas_y=900)
        assert not mosaic.canvas.any()

    def test_a_small_bright_feature_survives_the_reduction(self):
        """The whole reason this uses `downsample` rather than `data[::n, ::n]`
        (FIB-589, FIB-629): sampling deletes what it leaves out, so a punctum a couple
        of pixels across is present or absent depending on where it lands."""
        mosaic = PreviewMosaic(4096, 4096, dtype=np.uint16, channels=1)  # stride 2
        tile = np.zeros((1, 512, 512), dtype=np.uint16)
        tile[0, 101, 101] = 4000  # an odd row and column, which striding drops

        mosaic.paint(tile, canvas_x=0, canvas_y=0)

        assert mosaic.canvas[0, 50, 50] > 0, "the feature was sampled away"
