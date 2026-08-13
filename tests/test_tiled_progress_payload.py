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
                    hfw=100e-6, resolution=(64, 64), beam_type=BeamType.ELECTRON,
                    save=False, path=str(tmp_path_factory.mktemp("tiles")), filename="t",
                ),
                nrows=1, ncols=2,
            ),
        ).run()
    finally:
        microscope.tiled_acquisition_signal.disconnect(collected.append)
    return collected


def _per_tile(payloads):
    return [p for p in payloads if p.get("msg") == "Tile Collected"]


def test_every_tile_update_carries_the_keys_its_consumers_index(payloads):
    """`counter`, `total` and `msg` are read unconditionally by more than one consumer
    -- `FibsemMinimapWidget.handle_tile_acquisition_progress` indexes them directly and
    raises on a payload without them."""
    updates = _per_tile(payloads)
    assert len(updates) == 2, "expected one update per tile"
    for payload in updates:
        assert {"counter", "total", "msg", "image"} <= set(payload)
        assert payload["total"] == 2

    # Present is not enough: a counter stuck at its initial value satisfies every
    # structural check above while every consumer shows "0 / 2" for the whole run.
    assert [p["counter"] for p in updates] == [1, 2]


def test_a_tile_update_carries_the_tile_itself(payloads):
    """The key the real-space overview places from.

    The growing stitch buffer under `image` cannot answer where a tile belongs: it
    holds integer pixel offsets, so the error against the true stage position
    accumulates across the grid (FIB-399). The tile carries the position the stage
    actually reached.
    """
    for payload in _per_tile(payloads):
        tile = payload.get("tile")
        assert tile is not None, "the per-tile update stopped carrying its tile"
        assert tile.metadata.stage_position is not None
        assert tile.metadata.pixel_size.x, "a tile with no scale cannot be placed"
        assert tile.metadata.hardware_geometry is not None


def test_the_tiles_are_distinguishable_by_where_they_were_acquired(payloads):
    """Two tiles carrying the same position would place on top of each other, and the
    display would look like a run that only ever acquired one."""
    positions = [
        (p["tile"].metadata.stage_position.x, p["tile"].metadata.stage_position.y)
        for p in _per_tile(payloads)
    ]
    assert len(set(positions)) == len(positions), f"tiles share a position: {positions}"


def test_the_run_ends_with_a_terminal_update(payloads):
    """Consumers branch on `finished` to stop showing progress; without it the bar just
    stops moving and nothing says whether the run succeeded."""
    terminal = [p for p in payloads if p.get("finished")]
    assert terminal, "no terminal update was emitted"
    assert terminal[-1].get("outcome") == "finished"
