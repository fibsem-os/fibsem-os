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
        # The typed path the dict one is heading for (FIB-402), borrowed the same way.
        _on_tiled_progress = _Real._on_tiled_progress
        _tiled_outcome = staticmethod(_Real._tiled_outcome)
        _STATUS_LABELS = _Real._STATUS_LABELS

        def __init__(self):
            self.progress_widget = FibsemProgressWidget()

    host = _StatusHost()
    yield host
    host.progress_widget.deleteLater()


def _text(host) -> str:
    return host.progress_widget._bar.format()


def _is_red(host) -> bool:
    return RED in host.progress_widget._bar.styleSheet()


def _terminal(outcome: str, msg: str) -> dict:
    """What `TiledAcquisitionRunner._emit_terminal` sends for each outcome."""
    return {"msg": msg, "counter": 3, "total": 9, "finished": True, "outcome": outcome}


def test_a_completed_run_says_done(status):
    status._on_tile_acquisition_progress(_terminal("finished", "Acquisition Complete"))
    assert "Done" in _text(status)
    assert not _is_red(status)


def test_a_cancelled_run_does_not_claim_to_have_finished(status):
    status._on_tile_acquisition_progress(
        _terminal("cancelled", "Acquisition Cancelled")
    )
    assert "Cancelled" in _text(status)
    assert "Done" not in _text(status)


def test_a_cancelled_run_is_not_painted_as_a_failure(status):
    """A cancel is someone getting what they asked for, so the bar does not go red —
    the distinction FIB-375 drew between a completed operation and a failed one."""
    status._on_tile_acquisition_progress(
        _terminal("cancelled", "Acquisition Cancelled")
    )
    assert not _is_red(status)


def test_a_failed_run_is_red_and_says_so(status):
    status._on_tile_acquisition_progress(_terminal("failed", "Acquisition Failed"))
    assert "Failed" in _text(status)
    assert _is_red(status)


def test_a_producer_that_reports_no_outcome_still_says_done(status):
    """The napari minimap drives the same runner and the same signal, and older payloads
    carry no `outcome` at all. Absent, the bar reads as it always did."""
    status._on_tile_acquisition_progress(
        {"msg": "Done", "counter": 9, "total": 9, "finished": True}
    )
    assert "Done" in _text(status)
    assert not _is_red(status)


# ── the typed contract (FIB-402) ─────────────────────────────────────────────
#
# The status bar is the one consumer that watches both modalities, so it is also the one
# that has to keep working while the two producers flip one at a time. The dict tests
# above prove that path is untouched; these drive the typed path directly.


def _report(status, **fields):
    from fibsem.imaging.tiling.progress import TiledProgress

    return TiledProgress(status=status, **fields)


def test_a_typed_completed_run_says_done(status):
    from fibsem.imaging.tiling.progress import TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.FINISHED, completed=9, total=9)
    )

    assert "Done" in _text(status)
    assert not _is_red(status)


def test_a_typed_cancelled_run_does_not_claim_to_have_finished(status):
    from fibsem.imaging.tiling.progress import TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.CANCELLED, completed=3, total=9)
    )

    assert "Cancelled" in _text(status)
    assert "Done" not in _text(status)


def test_a_typed_cancelled_run_is_not_painted_as_a_failure(status):
    """A cancel is someone getting what they asked for, so the bar does not go red."""
    from fibsem.imaging.tiling.progress import TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.CANCELLED, completed=3, total=9)
    )

    assert not _is_red(status)


def test_a_typed_failed_run_is_painted_as_a_failure(status):
    from fibsem.imaging.tiling.progress import TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.FAILED, completed=3, total=9, error="stage limits")
    )

    assert _is_red(status)


def test_a_terminal_report_needs_no_counts(status):
    """The fluorescence terminal carries none, and it still has to clear the bar.

    Under the dict contract this was the `finished` key being checked before the counts;
    under the typed one it is `status.is_terminal`, and getting the order wrong means a
    cancelled fluorescence run leaves the status bar mid-run for the whole session.
    """
    from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE, TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.CANCELLED, modality=MODALITY_FLUORESCENCE)
    )

    assert "Cancelled" in _text(status)


def test_both_modalities_reach_the_status_bar(status):
    """Deliberately unfiltered, unlike the two overview widgets: its whole job is saying
    what is happening while you are looking at another tab (FIB-725)."""
    from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE, TiledStatus

    status._on_tile_acquisition_progress(
        _report(
            TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            completed=4,
            total=9,
        )
    )

    assert _text(status) == "Collecting tiles — 4 / 9"


def test_a_state_carrying_no_counts_leaves_the_last_count_standing(status):
    """A stage move must not snap the bar back to zero."""
    from fibsem.imaging.tiling.progress import TiledStatus

    status._on_tile_acquisition_progress(
        _report(TiledStatus.TILE_COLLECTED, completed=4, total=9)
    )
    before = _text(status)

    status._on_tile_acquisition_progress(_report(TiledStatus.MOVING))

    assert _text(status) == before
