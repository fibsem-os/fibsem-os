"""How a finished tiled acquisition reads in the main status bar.

Everything terminal used to take one branch and render "Done", so a run that was
cancelled — or that failed — told the status bar it had completed. The runner has
reported which it was since it learned to emit a terminal payload for every outcome;
nothing was reading it.

The window cannot be constructed here — it builds the napari minimap, and a
`napari.Viewer` segfaults under the offscreen platform — so its handler is borrowed onto
a host holding a real `FibsemProgressWidget`. That is enough: the handler's whole job is
turning a payload into a rendered bar, and the bar here is the real one.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.imaging.tiling.progress import (  # noqa: E402
    MODALITY_FLUORESCENCE,
    BeamTileCompletedEvent,
    FluorescenceTileCountEvent,
    TiledOutcome,
    TiledPhase,
    TiledPhaseEvent,
    TiledTerminalEvent,
)
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget  # noqa: E402

RED = "#99121F"


@pytest.fixture
def status(qapp):
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI as _Real,
    )

    class _StatusHost:
        _on_tile_acquisition_progress = _Real._on_tile_acquisition_progress
        # A staticmethod reached through the class is a plain function, which would
        # bind as a method here and eat `self`.
        _overview_outcome = staticmethod(_Real._overview_outcome)

        def __init__(self):
            self.progress_widget = FibsemProgressWidget()

    host = _StatusHost()
    yield host
    host.progress_widget.deleteLater()


def _text(host) -> str:
    return host.progress_widget._bar.format()


def _is_red(host) -> bool:
    return RED in host.progress_widget._bar.styleSheet()


def _terminal(outcome: TiledOutcome, message: str):
    return TiledTerminalEvent(outcome=outcome, message=message)


def test_a_completed_run_says_done(status):
    status._on_tile_acquisition_progress(
        _terminal(TiledOutcome.FINISHED, "Acquisition Complete")
    )
    assert "Done" in _text(status)
    assert not _is_red(status)


def test_a_cancelled_run_does_not_claim_to_have_finished(status):
    status._on_tile_acquisition_progress(
        _terminal(TiledOutcome.CANCELLED, "Acquisition Cancelled")
    )
    assert "Cancelled" in _text(status)
    assert "Done" not in _text(status)


def test_a_cancelled_run_is_not_painted_as_a_failure(status):
    """A cancel is someone getting what they asked for, so the bar does not go red —
    the distinction FIB-375 drew between a completed operation and a failed one."""
    status._on_tile_acquisition_progress(
        _terminal(TiledOutcome.CANCELLED, "Acquisition Cancelled")
    )
    assert not _is_red(status)


def test_a_failed_run_is_red_and_says_so(status):
    status._on_tile_acquisition_progress(
        _terminal(TiledOutcome.FAILED, "Acquisition Failed")
    )
    assert "Failed" in _text(status)
    assert _is_red(status)


def test_a_beam_count_uses_its_existing_message(status):
    status._on_tile_acquisition_progress(
        BeamTileCompletedEvent(
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
    )
    assert "Tile Collected" in _text(status)


def test_a_fluorescence_count_uses_the_existing_fallback(status):
    status._on_tile_acquisition_progress(
        FluorescenceTileCountEvent(
            modality=MODALITY_FLUORESCENCE,
            completed=3,
            total=9,
            estimated_total_seconds=9.0,
            estimated_remaining_seconds=6.0,
            elapsed_seconds=3.0,
        )
    )
    assert "Collecting tiles" in _text(status)


def test_an_unknown_modality_is_still_shown_by_the_application_status(status):
    status._on_tile_acquisition_progress(
        FluorescenceTileCountEvent(
            modality="future",
            completed=3,
            total=9,
            estimated_total_seconds=9.0,
            estimated_remaining_seconds=6.0,
            elapsed_seconds=3.0,
        )
    )
    assert "Collecting tiles" in _text(status)


def test_a_countless_phase_does_not_reset_the_bar(status):
    test_a_fluorescence_count_uses_the_existing_fallback(status)
    before = status.progress_widget._bar.value()
    status._on_tile_acquisition_progress(
        TiledPhaseEvent(
            modality=MODALITY_FLUORESCENCE, phase=TiledPhase.MOVING
        )
    )
    assert status.progress_widget._bar.value() == before
