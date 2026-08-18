"""The live preview averages what it reduces, instead of sampling it (FIB-629).

`_paint_preview` decimated each tile by striding -- `data[:, ::stride, ::stride]` --
which is the reduction FIB-589 removed from the canvas path, left in place on the
preview path. Sampling is free and wrong: it does not blur what it leaves out, it
deletes it, so a feature a pixel or two across survives only when it happens to land on
a sampled row. `fibsem/imaging/reduce.py` states the case, and states it about
fluorescence specifically -- the small bright thing is what someone is looking for.

Nothing saved was affected; the preview is swapped for the real stitch when the run
ends. But a tileset takes minutes, and the preview is what someone watches to decide
whether to let it run or stop and re-plan. Deciding "there is nothing there" from an
aliased preview is the failure that matters.
"""

import math

import numpy as np
import pytest

from fibsem.imaging.reduce import downsample
from fibsem.imaging.tiling.geometry import TilePosition


def _strided(plane, stride):
    """What the preview used to do."""
    return plane[::stride, ::stride]


def _averaged(plane, stride):
    """What it does now."""
    return downsample(plane, math.ceil(max(plane.shape[:2]) / stride))


class TestItIsADropInForTheStriding:
    """`downsample` returns ceil(n / factor) per axis, the same shape striding gives, so
    it replaces the call unchanged -- but only if the factor it derives from `max_px` is
    the stride. That is the arithmetic worth pinning, not assuming."""

    @pytest.mark.parametrize(
        "shape",
        [(1024, 1024), (1024, 768), (4712, 1024), (100, 100), (10, 10), (9, 9), (7, 7)],
    )
    @pytest.mark.parametrize("stride", [1, 2, 3, 4, 7, 8, 13, 16])
    def test_the_shape_is_unchanged(self, shape, stride):
        plane = np.zeros(shape, dtype=np.uint16)

        assert _averaged(plane, stride).shape == _strided(plane, stride).shape

    def test_a_stride_of_one_reduces_nothing(self):
        plane = np.arange(64, dtype=np.uint16).reshape(8, 8)

        np.testing.assert_array_equal(_averaged(plane, 1), plane)

    def test_the_dtype_survives(self):
        """matplotlib reads an array's value range from its dtype, so a float here
        would render the preview wrong as well as reduce it wrong."""
        plane = np.zeros((64, 64), dtype=np.uint16)

        assert _averaged(plane, 8).dtype == np.uint16


class TestASmallBrightFeatureSurvives:
    """The point of the exercise."""

    @pytest.mark.parametrize("offset", range(8))
    def test_a_punctum_shows_up_wherever_it_lands(self, offset):
        """At stride 8 sampling keeps 1 pixel in 64, so a 2x2 punctum is present or
        absent depending on where it falls. Averaging always carries some of it."""
        plane = np.zeros((64, 64), dtype=np.uint16)
        plane[16 + offset : 18 + offset, 16 + offset : 18 + offset] = 4000

        assert _averaged(plane, 8).max() > 0

    def test_sampling_is_what_loses_it(self):
        """Discriminates: unless the old reduction actually dropped puncta for some
        offsets, the test above proves nothing."""
        lost = 0
        for offset in range(8):
            plane = np.zeros((64, 64), dtype=np.uint16)
            plane[16 + offset : 18 + offset, 16 + offset : 18 + offset] = 4000
            if _strided(plane, 8).max() == 0:
                lost += 1

        assert lost > 0, "striding kept every punctum; the premise does not hold"

    def test_a_lone_bright_pixel_is_dimmed_not_deleted(self):
        """Averaging spreads it over its block rather than keeping it at full value.
        That is the trade and it is the right one -- visible and quantitatively wrong
        beats invisible."""
        plane = np.zeros((64, 64), dtype=np.uint16)
        plane[20, 20] = 4000

        out = _averaged(plane, 8)

        assert 0 < out.max() < 4000


class TestTheChannelAxisIsNotReduced:
    """`downsample` reads shape as (y, x[, c]) and the FM path's planes are
    (channels, y, x), so passing the stack would reduce across *channels* and y. A
    5-channel stack would also fall to the numpy branch -- right answer, wrong axes,
    200x slower -- rather than failing."""

    def test_each_channel_is_reduced_on_its_own(self):
        data = np.zeros((3, 64, 64), dtype=np.uint16)
        data[0, :, :] = 100
        data[1, :, :] = 200
        data[2, :, :] = 300

        out = np.stack([_averaged(plane, 8) for plane in data])

        assert out.shape == (3, 8, 8)
        assert out[0].max() == 100 and out[1].max() == 200 and out[2].max() == 300

    def test_passing_the_stack_would_have_been_wrong(self):
        """Pins why the loop is there, so nobody 'simplifies' it back."""
        data = np.zeros((3, 64, 64), dtype=np.uint16)

        wrong = downsample(data, math.ceil(max(data.shape[:2]) / 8))

        assert wrong.shape != (3, 8, 8)


class TestThePreviewPaintsWhatItIsGiven:
    """Through the runner method itself, not the helper, so the wiring is covered."""

    @staticmethod
    def _runner_with_preview(n_channels, stride, canvas_shape):
        from fibsem.fm.acquisition import FMTiledAcquisitionRunner

        from fibsem.imaging.reduce import PreviewMosaic

        runner = FMTiledAcquisitionRunner.__new__(FMTiledAcquisitionRunner)
        runner.channel_settings = [object()] * n_channels
        # The canvas and stride are stated directly rather than derived from a mosaic
        # size, because these tests pin the *paste* -- the geometry that produced the
        # stride is `PreviewMosaic`'s own business and is tested in
        # `tests/test_image_reduction.py`.
        runner._preview = PreviewMosaic.__new__(PreviewMosaic)
        runner._preview.stride = stride
        runner._preview.canvas = np.zeros((n_channels, *canvas_shape), dtype=np.uint16)
        return runner

    @staticmethod
    def _tile(canvas_x, canvas_y):
        return TilePosition(
            row=0, col=0, dx=0.0, dy=0.0, canvas_x=canvas_x, canvas_y=canvas_y
        )

    def test_a_punctum_reaches_the_preview_canvas(self):
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(1, stride=8, canvas_shape=(16, 16))
        data = np.zeros((1, 64, 64), dtype=np.uint16)
        data[0, 21:23, 21:23] = 4000  # an offset striding would step over

        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = data
        runner._paint_preview(self._tile(0, 0), image)

        assert runner._preview.canvas.max() > 0

    def test_it_lands_where_the_tile_says(self):
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(1, stride=8, canvas_shape=(32, 32))
        data = np.full((1, 64, 64), 500, dtype=np.uint16)

        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = data
        runner._paint_preview(self._tile(canvas_x=64, canvas_y=64), image)

        painted = runner._preview.canvas[0]
        assert painted[8:16, 8:16].min() > 0  # the tile, at 64 // 8
        assert painted[0:8, 0:8].max() == 0  # and nowhere else

    def test_the_channels_are_not_mixed(self):
        """The axis trap, through the runner rather than the helper: reducing the
        (channels, y, x) stack whole would fold channels into each other."""
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(3, stride=8, canvas_shape=(8, 8))
        data = np.zeros((3, 64, 64), dtype=np.uint16)
        data[0] = 100
        data[1] = 200
        data[2] = 300

        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = data
        runner._paint_preview(self._tile(0, 0), image)

        painted = runner._preview.canvas
        assert painted.shape == (3, 8, 8)
        assert [int(painted[c].max()) for c in range(3)] == [100, 200, 300]

    def test_the_preview_averages_rather_than_samples(self):
        """A lone bright pixel arrives dimmed -- spread over its block -- where
        sampling would either keep it whole or drop it entirely."""
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(1, stride=8, canvas_shape=(8, 8))
        data = np.zeros((1, 64, 64), dtype=np.uint16)
        data[0, 20, 20] = 4000

        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = data
        runner._paint_preview(self._tile(0, 0), image)

        assert 0 < runner._preview.canvas.max() < 4000

    @pytest.mark.parametrize("stride", [1, 2, 4, 7, 8])
    def test_the_painted_extent_matches_the_old_striding(self, stride):
        """The drop-in claim, where it actually matters: if the reduced tile came out a
        different shape, it would paint over a different span of the canvas."""
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(1, stride=stride, canvas_shape=(128, 128))
        data = np.full((1, 64, 64), 500, dtype=np.uint16)

        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = data
        runner._paint_preview(self._tile(0, 0), image)

        expected_h, expected_w = _strided(data[0], stride).shape
        painted = runner._preview.canvas[0]
        assert painted[:expected_h, :expected_w].min() > 0
        assert painted[expected_h:, :].max() == 0
        assert painted[:, expected_w:].max() == 0

    def test_a_failure_never_reaches_the_tile_loop(self):
        """Best effort by design: a preview that cannot be built is not a reason to
        fail an acquisition that is otherwise fine."""
        from fibsem.fm.structures import FluorescenceImage

        runner = self._runner_with_preview(1, stride=8, canvas_shape=(16, 16))
        image = FluorescenceImage.__new__(FluorescenceImage)
        image.data = "not an array"

        runner._paint_preview(self._tile(0, 0), image)  # must not raise
