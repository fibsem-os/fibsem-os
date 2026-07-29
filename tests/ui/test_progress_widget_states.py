"""Headless tests for FibsemProgressWidget's finished/failed rendering.

Covers the two states that previously looked identical: a completed operation and
a failed one both painted a full green bar, so a failed spot burn read as "Done"
in everything but the text (FIB-375).

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.ui import stylesheets
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate

GREEN = "#4caf50"
RED = "#99121F"


def _chunk_colour(widget: FibsemProgressWidget) -> str:
    """The chunk colour currently applied to the bar."""
    sheet = widget._bar.styleSheet()
    if RED in sheet:
        return RED
    if GREEN in sheet:
        return GREEN
    raise AssertionError(f"unrecognised progress bar stylesheet: {sheet!r}")


@pytest.fixture
def widget(qapp):
    return FibsemProgressWidget()


def test_done_is_not_failed():
    """done() must not set the failure flag.

    Regression guard for the field/classmethod name collision: a field named
    `failed` would be shadowed by the `failed()` classmethod, and @dataclass would
    take the method object as the default — truthy, so every done() rendered red.
    """
    assert ProgressUpdate.done().is_failed is False
    assert ProgressUpdate.failed("boom").is_failed is True
    assert ProgressUpdate.combined(1, 2, 3, 4).is_failed is False


def test_done_stays_green(widget):
    widget.update_progress(ProgressUpdate.combined(2, 5, 8, 10, "Burning spots"))
    assert _chunk_colour(widget) == GREEN

    widget.update_progress(ProgressUpdate.done())
    assert _chunk_colour(widget) == GREEN
    assert widget._bar.value() == 100
    assert widget._bar.format() == "Done"


def test_failed_renders_red_with_its_message(widget):
    widget.update_progress(ProgressUpdate.failed("Spot burn failed"))
    assert _chunk_colour(widget) == RED
    assert widget._bar.format() == "Spot burn failed"


def test_next_run_clears_the_failed_style(widget):
    """A failure must not leave the bar red for the following operation."""
    widget.update_progress(ProgressUpdate.failed("Spot burn failed"))
    widget.update_progress(ProgressUpdate.combined(1, 3, 5, 10, "Burning spots"))
    assert _chunk_colour(widget) == GREEN


def test_reset_clears_the_failed_style(widget):
    widget.update_progress(ProgressUpdate.failed("Spot burn failed"))
    widget.reset()
    assert _chunk_colour(widget) == GREEN
    assert widget.isHidden()


def test_reset_if_finished_only_resets_when_finished(widget):
    """The delayed hide must not blank a run that has already started."""
    widget.update_progress(ProgressUpdate.combined(1, 3, 5, 10, "Burning spots"))
    widget.reset_if_finished()
    assert not widget.isHidden()
    assert widget._bar.format() == "Burning spots — 1/3 · 5s remaining"

    widget.update_progress(ProgressUpdate.done())
    widget.reset_if_finished()
    assert widget.isHidden()


def test_reset_if_finished_resets_after_a_failure(widget):
    """Failed is a finished state — an explicit reset still clears it."""
    widget.update_progress(ProgressUpdate.failed("Spot burn failed"))
    widget.reset_if_finished()
    assert widget.isHidden()
    assert _chunk_colour(widget) == GREEN


def test_failed_stylesheet_is_the_bar_with_an_error_chunk():
    """The failed sheet should differ from the normal one only in chunk colour."""
    normal = stylesheets.MILLING_PROGRESS_BAR_STYLESHEET
    failed = stylesheets.FAILED_PROGRESS_BAR_STYLESHEET
    assert failed.replace(RED, GREEN) == normal
