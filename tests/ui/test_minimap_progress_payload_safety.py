"""The napari minimap survives a progress payload it did not expect.

`tiled_acquisition_signal` has no discriminator: a consumer works out what it received
from which keys are present. This handler indexed `counter`, `total` and `msg` directly,
so any payload shape omitting one was a `KeyError` inside a Qt slot, mid-acquisition —
and `total` is a divisor, so a payload reporting nothing to do took the bar out with a
`ZeroDivisionError` instead (FIB-402).

That was latent while `imaging/tiled.py` was the signal's only emitter. It stops being
latent the moment a second producer emits there, which is what FIB-725 proposes.

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


def test_a_normal_tile_update_still_draws_the_bar(handler):
    handler.handle_tile_acquisition_progress(
        {"counter": 3, "total": 9, "msg": "Tile Collected"}
    )
    assert handler._tiles_acquired == 3
    assert handler._tile_total_count == 9
    assert handler.progressBar_acquisition.value() == 33
    assert "Tile Collected" in handler.progressBar_acquisition.format()
    assert "3/9 tiles" in handler.progressBar_acquisition.format()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"total": 9, "msg": "Tile Collected"},  # no counter
        {"counter": 3, "msg": "Tile Collected"},  # no total
        {"counter": 3, "total": 9},  # no msg
        {"counter": 0, "total": 0, "msg": "Nothing"},  # nothing to do
        {"current": 3, "total": 9, "task": "tileset"},  # the fluorescence spelling
    ],
)
def test_an_unexpected_payload_does_not_take_the_bar_out(handler, payload):
    """Each of these was a crash: the first four a `KeyError`, the fifth a
    `ZeroDivisionError`, and the last is what a fluorescence run would send if it
    emitted here — the case that turns all of this from latent into live."""
    handler.handle_tile_acquisition_progress(payload)


def test_a_payload_with_no_counts_leaves_the_last_progress_standing(handler):
    """Not reset to zero. The run has not gone backwards, and the terminal payload from
    a cancelled run carries no counts — so zeroing here would end every cancelled run by
    claiming it never started."""
    handler.handle_tile_acquisition_progress(
        {"counter": 4, "total": 9, "msg": "Tile Collected"}
    )
    before = handler.progressBar_acquisition.value()
    handler.handle_tile_acquisition_progress({"msg": "Stitching Tiles"})
    assert handler.progressBar_acquisition.value() == before
    assert handler._tiles_acquired == 4


def test_a_fluorescence_run_is_not_drawn_into_the_beam_minimap(handler):
    """This tab drives a *beam* overview and assigns the payload's mosaic straight into
    its napari layer. The fluorescence preview is keyed `image` already — deliberately,
    to match this signal — so once a fluorescence run reports here, nothing but the
    modality stops it being painted into the beam minimap (FIB-725)."""
    from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE

    sentinel = object()
    handler.handle_tile_acquisition_progress(
        {
            "modality": MODALITY_FLUORESCENCE,
            "counter": 3,
            "total": 9,
            "msg": "Tile Collected",
            "image": sentinel,
        }
    )
    assert handler.shown == [], "a fluorescence mosaic was drawn into the beam minimap"
    assert handler._tiles_acquired == 0, "a fluorescence run moved the beam tile count"


def test_a_beam_run_is_still_drawn(handler):
    """The filter must not cost the thing it guards."""
    sentinel = object()
    handler.handle_tile_acquisition_progress(
        {
            "modality": "beam",
            "counter": 3,
            "total": 9,
            "msg": "Tile Collected",
            "image": sentinel,
        }
    )
    assert handler.shown == [sentinel]
    assert handler._tiles_acquired == 3


def test_a_preview_is_still_shown_when_the_counts_are_missing(handler):
    """The image and the counters are independent: dropping the bar update must not
    drop the picture with it."""
    sentinel = object()
    handler.handle_tile_acquisition_progress({"image": sentinel})
    assert handler.shown == [sentinel]
