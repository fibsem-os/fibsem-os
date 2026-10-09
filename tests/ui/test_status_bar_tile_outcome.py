"""How a tiled acquisition reads in the status bar's line (FIB-1188).

Everything terminal used to take one branch and render "Done", so a run that was
cancelled -- or that failed -- told the status bar it had completed. The runner has
reported which it was since it learned to emit a terminal payload for every outcome.

The status bar decodes the reports itself now, so these run against the real
`FibsemStatusBar`; they used to borrow the main window's handler onto a host.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.imaging.tiling.progress import (  # noqa: E402
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.ui.widgets import status_bar  # noqa: E402

RED = "#d04040"


@pytest.fixture
def bar(qapp, monkeypatch):
    monkeypatch.setattr(status_bar, "OUTCOME_MS", 20)
    bar = status_bar.FibsemStatusBar()
    bar.set_instruction("Create or load an experiment to begin.")
    yield bar
    bar.deleteLater()


def _is_red(bar) -> bool:
    return RED in bar._step.styleSheet()


def _report(status, **fields):
    return TiledProgress(status=status, **fields)


def _outlast_the_outcome():
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(80, loop.quit)
    loop.exec_()


def test_tiles_show_the_count_and_the_bar(bar):
    bar._on_tiled_progress(
        _report(
            TiledStatus.TILE_COLLECTED,
            completed=4,
            total=9,
            estimated_remaining_seconds=48.0,
        )
    )
    assert bar.text == "FIB/SEM overview collecting tiles · 4 of 9 44% · 48s left"
    assert bar.fraction == pytest.approx(4 / 9, abs=1e-3)


def test_a_completed_run_says_done_then_gives_the_line_back(bar):
    bar._on_tiled_progress(_report(TiledStatus.FINISHED, completed=9, total=9))
    assert bar.text == "FIB/SEM overview done"
    assert not _is_red(bar)
    _outlast_the_outcome()
    assert bar.text == "Create or load an experiment to begin."


def test_a_cancelled_run_does_not_claim_to_have_finished(bar):
    bar._on_tiled_progress(_report(TiledStatus.CANCELLED, completed=3, total=9))
    assert bar.text == "FIB/SEM overview cancelled"


def test_a_cancelled_run_is_not_painted_as_a_failure(bar):
    """A cancel is someone getting what they asked for, so it is not red."""
    bar._on_tiled_progress(_report(TiledStatus.CANCELLED, completed=3, total=9))
    assert not _is_red(bar)


def test_a_failed_run_is_red_and_says_why(bar):
    bar._on_tiled_progress(
        _report(TiledStatus.FAILED, completed=3, total=9, error="stage limits")
    )
    assert bar.text == "FIB/SEM overview failed: stage limits"
    assert _is_red(bar)


def test_a_terminal_report_needs_no_counts(bar):
    """The fluorescence terminal carries none, and it still has to end the run.
    Getting the order wrong leaves the line mid-run for the whole session."""
    bar._on_tiled_progress(_report(TiledStatus.CANCELLED, modality=MODALITY_FLUORESCENCE))
    assert bar.text == "Fluorescence overview cancelled"


def test_both_modalities_reach_the_status_bar(bar):
    """Deliberately unfiltered, unlike the two overview widgets: its whole job is saying
    what is happening while you are looking at another tab (FIB-725)."""
    bar._on_tiled_progress(
        _report(
            TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            completed=4,
            total=9,
        )
    )
    assert bar.text == "Fluorescence overview collecting tiles · 4 of 9 44%"


def test_a_state_carrying_no_counts_leaves_the_last_count_standing(bar):
    """A stage move must not snap the bar back to nothing; only the words change."""
    bar._on_tiled_progress(
        _report(
            TiledStatus.TILE_COLLECTED,
            completed=4,
            total=9,
            estimated_remaining_seconds=48.0,
        )
    )
    before = bar.fraction
    bar._on_tiled_progress(_report(TiledStatus.MOVING))
    assert bar.fraction == before
    assert bar.text == "FIB/SEM overview moving stage · 4 of 9 44%", (
        "the count stands; its time left would only sit there frozen"
    )


def test_the_next_run_starts_its_count_afresh(bar):
    bar._on_tiled_progress(_report(TiledStatus.TILE_COLLECTED, completed=4, total=9))
    bar._on_tiled_progress(_report(TiledStatus.FINISHED, completed=9, total=9))
    bar._on_tiled_progress(_report(TiledStatus.STARTING))
    assert bar.fraction is None, "the last run's count is not this one's"
    assert bar.text == "FIB/SEM overview collecting tiles"
