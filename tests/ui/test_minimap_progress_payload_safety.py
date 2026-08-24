"""Typed tiled events consumed by the legacy napari beam minimap."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QProgressBar  # noqa: E402

from fibsem.imaging.tiling.progress import (  # noqa: E402
    MODALITY_FLUORESCENCE,
    BeamTileCompletedEvent,
    CountedTiledPhaseEvent,
    TiledPhase,
)


def _tile(**overrides):
    values = dict(
        completed=3,
        total=9,
        row_index=0,
        column_index=2,
        rows=1,
        columns=9,
        image=object(),
        preview=object(),
        message="Tile Collected",
    )
    values.update(overrides)
    return BeamTileCompletedEvent(**values)


@pytest.fixture
def handler(qapp):
    from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget

    class _Host:
        handle_tile_acquisition_progress = (
            FibsemMinimapWidget.handle_tile_acquisition_progress
        )

        def __init__(self):
            self.progressBar_acquisition = QProgressBar()
            self._tiles_acquired = 0
            self._tile_total_count = 0
            self.shown = []

        def update_viewer(self, image, tmp=False):
            self.shown.append(image)

    host = _Host()
    yield host
    host.progressBar_acquisition.deleteLater()


def test_a_normal_tile_update_draws_the_existing_wording(handler):
    event = _tile()
    handler.handle_tile_acquisition_progress(event)

    assert handler._tiles_acquired == 3
    assert handler._tile_total_count == 9
    assert handler.progressBar_acquisition.value() == 33
    assert "Tile Collected" in handler.progressBar_acquisition.format()
    assert "3/9 tiles" in handler.progressBar_acquisition.format()
    assert handler.shown == [event.image]


def test_a_counted_phase_leaves_the_last_preview_standing(handler):
    event = _tile(completed=4)
    handler.handle_tile_acquisition_progress(event)
    handler.handle_tile_acquisition_progress(
        CountedTiledPhaseEvent(
            phase=TiledPhase.STITCHING,
            completed=9,
            total=9,
            message="Stitching Tiles",
        )
    )

    assert handler.progressBar_acquisition.value() == 100
    assert "Stitching Tiles" in handler.progressBar_acquisition.format()
    assert handler.shown == [event.image]


def test_a_fluorescence_event_is_not_drawn_into_the_beam_minimap(handler):
    event = _tile(modality=MODALITY_FLUORESCENCE)
    handler.handle_tile_acquisition_progress(event)
    assert handler.shown == []
    assert handler._tiles_acquired == 0


def test_an_unknown_modality_is_safely_ignored(handler):
    handler.handle_tile_acquisition_progress(_tile(modality="future"))
    assert handler.shown == []
    assert handler._tiles_acquired == 0


def test_default_modality_remains_beam(handler):
    event = _tile()
    handler.handle_tile_acquisition_progress(event)
    assert handler.shown == [event.image]
