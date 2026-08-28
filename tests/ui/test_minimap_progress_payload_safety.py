"""The napari minimap survives a progress report it did not expect.

The shapes that used to be a `KeyError` here are gone -- `TiledProgress` declares every
field, so a report cannot omit one. What survives the typing is arithmetic: `total` is
a divisor, and a run reporting nothing to do still takes the bar out with a
`ZeroDivisionError` unless it is guarded (FIB-402).

Also the modality filter. This tab drives a *beam* overview, and the signal has two
producers (FIB-725).

The whole widget cannot be built here — it constructs a `napari.Viewer`, which segfaults
under the offscreen platform — so the handler is borrowed onto a host carrying the four
attributes it touches. The progress bar is a real `QProgressBar`, which is the part
under test.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QProgressBar  # noqa: E402


@pytest.fixture
def handler(qapp):
    from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget

    class _Host:
        handle_tile_acquisition_progress = (
            FibsemMinimapWidget.handle_tile_acquisition_progress
        )
        _STATUS_LABELS = FibsemMinimapWidget._STATUS_LABELS

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


def _beam(status, **fields):
    from fibsem.imaging.tiling.progress import MODALITY_BEAM, TiledProgress

    fields.setdefault("modality", MODALITY_BEAM)
    return TiledProgress(status=status, **fields)


def _preview(width: int = 4, height: int = 4):
    """A beam preview: a `FibsemImage`, which is what the tiler will attach."""
    import numpy as np

    from fibsem.structures import FibsemImage

    return FibsemImage(data=np.zeros((height, width), dtype=np.uint8))


def test_a_tile_report_draws_the_bar(handler):
    from fibsem.imaging.tiling.progress import TiledStatus

    handler.handle_tile_acquisition_progress(
        _beam(TiledStatus.TILE_COLLECTED, completed=3, total=9)
    )

    assert handler._tiles_acquired == 3
    assert handler._tile_total_count == 9
    assert handler.progressBar_acquisition.value() == 33
    assert "3/9 tiles" in handler.progressBar_acquisition.format()


def test_the_wording_comes_from_this_tab_not_the_producer(handler):
    """The whole point of dropping `message` from the wire.

    The producer says `TILE_COLLECTED`; what a reader sees is this tab's choice, and a
    different consumer of the same report is free to word it differently.
    """
    from fibsem.imaging.tiling.progress import TiledStatus

    handler.handle_tile_acquisition_progress(
        _beam(TiledStatus.STITCHING, completed=9, total=9)
    )

    assert "Stitching Tiles" in handler.progressBar_acquisition.format()


def test_a_report_with_nothing_to_do_does_not_take_the_bar_out(handler):
    """`total` is a divisor. A run reporting 0/0 must not raise in a Qt slot."""
    from fibsem.imaging.tiling.progress import TiledStatus

    handler.progressBar_acquisition.setValue(42)

    handler.handle_tile_acquisition_progress(
        _beam(TiledStatus.STARTING, completed=0, total=0)
    )

    assert handler.progressBar_acquisition.value() == 42


def test_a_report_with_no_counts_leaves_the_last_progress_standing(handler):
    """A stage move is still true progress-wise: the last count has not stopped being
    the last count, so nothing is drawn rather than resetting to zero."""
    from fibsem.imaging.tiling.progress import TiledStatus

    handler.handle_tile_acquisition_progress(
        _beam(TiledStatus.TILE_COLLECTED, completed=4, total=9)
    )
    before = handler.progressBar_acquisition.value()

    handler.handle_tile_acquisition_progress(_beam(TiledStatus.MOVING))

    assert handler.progressBar_acquisition.value() == before
    assert handler._tiles_acquired == 4


def test_a_preview_is_drawn_from_the_decimated_mosaic(handler):
    """`preview.data`, not the live full-resolution canvas this used to be handed.

    A deliberate change: the array it got before was the one the acquisition thread was
    still painting into, and this tab is being retired anyway.
    """
    from fibsem.imaging.tiling.progress import TiledStatus

    preview = _preview()

    handler.handle_tile_acquisition_progress(
        _beam(TiledStatus.TILE_COLLECTED, completed=1, total=9, preview=preview)
    )

    assert len(handler.shown) == 1
    assert handler.shown[0] is preview.data


def test_a_fluorescence_report_is_not_drawn_into_the_beam_minimap(handler):
    """This tab drives a beam overview; the other modality's mosaic is not its to
    draw, whatever else the report carries (FIB-725)."""
    from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE, TiledStatus

    handler.handle_tile_acquisition_progress(
        _beam(
            TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            completed=3,
            total=9,
            preview=_preview(),
        )
    )

    assert handler.shown == []
    assert handler._tiles_acquired == 0
