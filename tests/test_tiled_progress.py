"""The typed contract carried by ``tiled_acquisition_signal`` (FIB-402).

Nothing emits or reads a ``TiledProgress`` yet -- the producers still build dicts and the
consumers still read them. What is worth pinning before either moves is the shape itself,
and in particular the three things that are cheap to get wrong and expensive to notice:
an event that cannot be compared, a status set that cannot answer "is this over", and a
tile index that is 0-based in one place and 1-based in another.
"""

import dataclasses

import numpy as np
import pytest

from fibsem.imaging.tiling.progress import (
    MODALITY_BEAM,
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)

TERMINAL = {TiledStatus.FINISHED, TiledStatus.CANCELLED, TiledStatus.FAILED}


def _fluorescence_preview(channels: int = 2, size: int = 8):
    """A stand-in for what the fluorescence runner will put on `preview`.

    A real `FluorescenceImage`, not a mock: it is a dataclass, so it brings a generated
    `__eq__` that compares its `data` array elementwise -- which is the whole reason
    `TiledProgress` sets `eq=False`, and a stub would not reproduce it.

    The channel metadata is not decoration. `FluorescenceImageMetadata.__post_init__`
    rejects an empty `channels`, so the producer building this preview has to carry its
    channels into it -- which is convenient, because the consumer wants their names and
    colours anyway.
    """
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )

    return FluorescenceImage(
        data=np.zeros((channels, size, size), dtype=np.uint16),
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-08-25T09:00:00",
            pixel_size_x=1e-7,
            pixel_size_y=1e-7,
            channels=[
                FluorescenceChannelMetadata(
                    name=f"channel-{index}",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                )
                for index in range(channels)
            ],
        ),
    )


class TestTheStatusVocabulary:
    def test_a_terminal_status_says_so_itself(self):
        """The question three consumers ask, answered once.

        Restated as a membership tuple at each of them, it drifts: the next status added
        is the one somebody forgets to add to one of the three, and the symptom is a
        progress bar that never clears.
        """
        for status in TiledStatus:
            assert status.is_terminal is (status in TERMINAL), status

    def test_finishing_the_tiles_is_not_finishing_the_run(self):
        """The collision that a single vocabulary has to survive.

        The runner's "the last tile is in and the stage is parked" and the widget's "the
        mosaic is written" were both spelled `finished` when they lived in separate
        phase/outcome vocabularies. Merging the two without renaming one would have
        made a mid-run report read as the end of the run.
        """
        assert TiledStatus.TILES_ACQUIRED is not TiledStatus.FINISHED
        assert not TiledStatus.TILES_ACQUIRED.is_terminal
        assert TiledStatus.FINISHED.is_terminal

    def test_a_member_compares_equal_to_its_own_value(self):
        """The `str` mixin, which is what keeps these readable in a log line."""
        assert TiledStatus.STITCHING == "stitching"

    def test_starting_a_tile_and_finishing_one_are_different_reports(self):
        """Two reports per tile, and only one of them is a progress report.

        Saying it twice with one name is what made a bar change scale at every tile
        boundary (FIB-739): a consumer choosing between them got the tile in flight
        half the time and the tally the other half.
        """
        assert TiledStatus.TILE_STARTED is not TiledStatus.TILE_COLLECTED
        assert not TiledStatus.TILE_STARTED.is_terminal
        assert not TiledStatus.TILE_COLLECTED.is_terminal

    def test_one_tile_landing_is_not_every_tile_landing(self):
        """`TILE_COLLECTED` happens N times; `TILES_ACQUIRED` happens once.

        Adjacent names for genuinely adjacent things, so the distinction is worth a
        test rather than a comment: confusing them means either a run that reports
        completion N times or a bar that only moves once.
        """
        assert TiledStatus.TILE_COLLECTED is not TiledStatus.TILES_ACQUIRED
        assert not TiledStatus.TILES_ACQUIRED.is_terminal


class TestTheEventShape:
    def test_status_is_the_only_thing_every_report_carries(self):
        """Everything else is absent on some report, so everything else has a default.

        Also what keeps this constructible on Python 3.8: `kw_only` does not exist
        there, so exactly one required field, and it has to be first.
        """
        event = TiledProgress(status=TiledStatus.MOVING)

        assert event.status is TiledStatus.MOVING
        assert event.completed is None and event.total is None
        assert event.preview is None
        assert event.error is None

        required = [
            f.name
            for f in dataclasses.fields(TiledProgress)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        assert required == ["status"]

    def test_an_unlabelled_report_is_a_beam_report(self):
        """What the signal carried for its whole life, kept as the default."""
        assert TiledProgress(status=TiledStatus.MOVING).modality == MODALITY_BEAM

    def test_a_report_cannot_be_written_to(self):
        event = TiledProgress(status=TiledStatus.MOVING)

        with pytest.raises(dataclasses.FrozenInstanceError):
            event.status = TiledStatus.FINISHED


class TestComparingReports:
    """`eq=False` is load-bearing, not incidental.

    A generated `__eq__` compares every field, `preview` holds an image that compares its
    numpy data elementwise, and the result is a `ValueError` from inside a Qt slot -- but
    only for two *distinct* events, because comparing one to itself short-circuits on
    identity. That asymmetry is what makes it worth a test: the obvious check passes
    while the real case raises.
    """

    def test_two_reports_carrying_previews_can_be_compared(self):
        first = TiledProgress(
            status=TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            preview=_fluorescence_preview(),
        )
        second = TiledProgress(
            status=TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            preview=_fluorescence_preview(),
        )

        assert first != second
        assert first == first
        assert (first in [second]) is False

    def test_a_report_carrying_a_preview_can_be_hashed(self):
        """A `set` of reports, or one used as a dict key, must not take the run out."""
        event = TiledProgress(
            status=TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            preview=_fluorescence_preview(),
        )

        assert {event, event} == {event}

    def test_the_preview_image_alone_still_raises_on_comparison(self):
        """The hazard is real and lives in the image type, not imagined here.

        If this ever stops raising, `eq=False` on the event has lost its reason and the
        comment explaining it is wrong -- which is worth finding out from a red test
        rather than from the docstring.
        """
        with pytest.raises(ValueError):
            bool(_fluorescence_preview() == _fluorescence_preview())


class TestTileCoordinates:
    def test_the_wire_is_zero_based(self):
        """Matching `_ordered[...]` and every other index in the codebase."""
        event = TiledProgress(
            status=TiledStatus.TILE_STARTED,
            row_index=0,
            column_index=0,
            rows=3,
            columns=4,
        )

        assert (event.row_index, event.column_index) == (0, 0)

    def test_the_display_form_is_one_based(self):
        """Shown as 1 of 10, never 0 of 9 -- the reader counts from one."""
        event = TiledProgress(
            status=TiledStatus.TILE_STARTED,
            row_index=0,
            column_index=9,
            rows=1,
            columns=10,
        )

        assert event.display_tile == (1, 10)
        assert f"{event.display_tile[1]}/{event.columns}" == "10/10"

    def test_the_last_tile_of_a_grid_displays_as_the_count(self):
        """The off-by-one that makes a run look like it stopped one short."""
        event = TiledProgress(
            status=TiledStatus.TILE_STARTED,
            row_index=2,
            column_index=3,
            rows=3,
            columns=4,
        )

        assert event.display_tile == (3, 4)

    def test_a_report_about_no_tile_has_no_display_form(self):
        """A stage move is not about a tile, so there is nothing to number."""
        assert TiledProgress(status=TiledStatus.MOVING).display_tile is None
