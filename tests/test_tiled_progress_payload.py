"""What the tiled acquisition tells its consumers, per tile.

`tiled_acquisition_signal` is shared: the FIB/SEM overview places tiles from it, the
AutoLamella window branches on it, and a progress bar reads its counters. So the shape
of the payload is a contract between modules that never see each other, and the sort of
thing that gets edited by someone looking at one consumer.

Deliberately **not** under `tests/ui`. The overview widget is the only consumer of the
`tile` key today and it is a Qt widget, so this test naturally wanted to live beside it
-- where CI, which installs `.[test]` and so has no PyQt5, would have skipped the whole
module and never checked the acquisition code at all.

Run directly:
    python -m pytest tests/test_tiled_progress_payload.py
"""

from __future__ import annotations

import pytest

from fibsem import utils
from fibsem.imaging.tiled import TiledAcquisitionRunner
from fibsem.imaging.tiling.progress import (
    MODALITY_BEAM,
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.structures import (
    BeamType,
    ImageSettings,
    OverviewAcquisitionSettings,
)


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture(scope="module")
def payloads(microscope, tmp_path_factory):
    """One small run, captured. Module-scoped: the simulator sleeps per tile, and every
    assertion below reads the same run rather than paying for its own."""
    collected = []
    microscope.tiled_acquisition_signal.connect(collected.append)
    try:
        TiledAcquisitionRunner(
            microscope,
            OverviewAcquisitionSettings(
                image_settings=ImageSettings(
                    hfw=100e-6,
                    resolution=(64, 64),
                    beam_type=BeamType.ELECTRON,
                    save=False,
                    path=str(tmp_path_factory.mktemp("tiles")),
                    filename="t",
                ),
                nrows=1,
                ncols=2,
            ),
        ).run()
    finally:
        microscope.tiled_acquisition_signal.disconnect(collected.append)
    return collected


def _per_tile(payloads):
    return [p for p in payloads if p.status is TiledStatus.TILE_COLLECTED]


def test_every_tile_update_carries_the_keys_its_consumers_index(payloads):
    """`counter`, `total` and `msg` are what every consumer draws its bar from.

    They used to be *indexed* -- `FibsemMinimapWidget.handle_tile_acquisition_progress`
    raised on a payload without them, so this test stood between that consumer and a
    `KeyError` mid-acquisition. Both consumers now read with `.get` and skip the update
    instead (FIB-402), so the stake is lower: a per-tile payload missing these shows no
    progress rather than taking the widget out. Still a contract, and still worth
    asserting -- a run whose progress silently stops moving is its own bug."""
    updates = _per_tile(payloads)
    assert len(updates) == 2, "expected one update per tile"
    for payload in updates:
        assert isinstance(payload, TiledProgress)
        assert payload.completed is not None
        assert payload.total == 2
        assert payload.preview is not None

    # Present is not enough: a counter stuck at its initial value satisfies every
    # structural check above while every consumer shows "0 / 2" for the whole run.
    assert [p.completed for p in updates] == [1, 2]


def test_a_tile_update_carries_a_placeable_preview(payloads):
    """The key the real-space overview places from.

    The mosaic, decimated, carrying the metadata a display needs to place it: position,
    pixel size, beam and geometry. The full-size live buffer used to travel beside it as
    a bare array for the napari minimap; that tab reads `preview.data` now, so there is
    one representation of the mosaic rather than two under different names.
    """
    for payload in _per_tile(payloads):
        preview = payload.preview
        assert preview is not None, "the per-tile update stopped carrying its preview"
        assert preview.metadata.stage_position is not None
        assert preview.metadata.pixel_size.x, "a mosaic with no scale cannot be placed"
        assert preview.metadata.hardware_geometry is not None
        assert preview.metadata.image_settings.beam_type is not None


def test_the_preview_fills_as_the_run_goes(payloads):
    """A preview that never changed would satisfy every structural check above while
    showing the same empty mosaic for the length of the run."""
    filled = [int((p.preview.data > 0).sum()) for p in _per_tile(payloads)]
    assert filled[-1] > filled[0], f"the preview did not fill: {filled}"


def test_the_preview_describes_the_whole_grid_from_the_first_tile(payloads):
    """It is the mosaic, not the tile: its shape is the finished grid's from the
    start, so a display places it once rather than resizing it on every update."""
    shapes = {p.preview.data.shape for p in _per_tile(payloads)}
    assert len(shapes) == 1, f"the preview changed shape mid-run: {shapes}"


def test_the_run_ends_with_a_terminal_update(payloads):
    """Consumers branch on `status.is_terminal` to stop showing progress; without one
    the bar just stops moving and nothing says whether the run succeeded."""
    terminal = [p for p in payloads if p.status.is_terminal]
    assert terminal, "no terminal update was emitted"
    assert terminal[-1].status is TiledStatus.FINISHED


def test_every_payload_says_which_modality_produced_it(payloads):
    """The discriminator this signal never had.

    Consumers used to work out what they had received from which keys were present,
    which was survivable while `imaging/tiled.py` was the only emitter. It is not once a
    second producer reports here: two consumers hand the mosaic straight to a *beam*
    canvas, and a fluorescence preview reaching one would be drawn into it (FIB-725).

    Asserted on every report, not just the per-tile ones: a terminal that could not say
    whose run had finished would leave a consumer unable to decide whether to clear its
    own progress.
    """
    assert payloads, "the run emitted nothing"
    for payload in payloads:
        assert payload.modality == MODALITY_BEAM, payload.status


def test_a_report_constructed_without_a_modality_reads_as_a_beam_run():
    """What this signal carried for its whole life, kept as the field's default."""
    assert TiledProgress(status=TiledStatus.MOVING).modality == MODALITY_BEAM


def test_a_fluorescence_report_is_not_mistaken_for_a_beam_one():
    report = TiledProgress(
        status=TiledStatus.TILE_COLLECTED,
        modality=MODALITY_FLUORESCENCE,
        completed=1,
        total=2,
    )
    assert report.modality == MODALITY_FLUORESCENCE
    assert report.modality != MODALITY_BEAM


def test_the_run_opens_by_saying_how_much_there_is_to_do(payloads):
    """A bar that appears at 0/N before the first stage move, rather than at the first
    tile -- otherwise the longest silence of a run is its beginning."""
    assert payloads[0].status is TiledStatus.STARTING
    assert (payloads[0].completed, payloads[0].total) == (0, 2)


def test_a_failed_run_carries_the_reason_not_just_the_fact(microscope, tmp_path):
    """The exception text used to be logged and thrown away, so a failed run told the
    UI "Acquisition Failed" and nothing about why -- and a stage-limits rejection names
    every offending tile."""
    collected = []
    microscope.tiled_acquisition_signal.connect(collected.append)
    runner = TiledAcquisitionRunner(
        microscope,
        OverviewAcquisitionSettings(
            image_settings=ImageSettings(
                hfw=100e-6,
                resolution=(64, 64),
                beam_type=BeamType.ELECTRON,
                save=False,
                path=str(tmp_path),
                filename="t",
            ),
            nrows=1,
            ncols=2,
        ),
    )
    runner._acquire_tile = lambda tile: (_ for _ in ()).throw(RuntimeError("no beam"))
    try:
        with pytest.raises(RuntimeError):
            runner.run()
    finally:
        microscope.tiled_acquisition_signal.disconnect(collected.append)

    terminal = [p for p in collected if p.status.is_terminal]
    assert terminal, "a failed run emitted no terminal report"
    assert terminal[-1].status is TiledStatus.FAILED
    assert terminal[-1].error and "no beam" in terminal[-1].error
