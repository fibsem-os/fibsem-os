"""What the Create Experiment dialog says about free disk space.

Two things are checked: the wording (`describe_free_space`, a plain function, so most
of this needs no dialog at all), and that the dialog paints the right colour for each
band and drops an answer that arrives too late.

The late answer is not a hypothetical. `disk_usage` on a disconnected mapped drive can
hang for tens of seconds, which is why the probe is on a worker thread, and which means
a result can land after the directory has changed or after the dialog has closed.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_create_experiment_disk_space.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_create_experiment_widget as mod,
)
from fibsem.ui.tokens import ERROR_COLOR, TEXT_MUTED_COLOR, WARN_COLOR  # noqa: E402
from fibsem.util.system import DiskSpace, FreeSpaceLevel  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

GIGABYTE = 1_000_000_000


def _space(free_gb: float, total_gb: float = 500) -> DiskSpace:
    free = int(free_gb * GIGABYTE)
    total = int(total_gb * GIGABYTE)
    return DiskSpace(path="Z:\\shared", total=total, used=total - free, free=free)


@pytest.fixture
def dialog():
    widget = mod.AutoLamellaCreateExperimentWidget()
    yield widget
    # done() before deleteLater: it invalidates any probe still out on a thread, which
    # is the thing that would otherwise write to a deleted label.
    widget.done(0)
    widget.deleteLater()
    _app.processEvents()


# ── the wording ──────────────────────────────────────────────────────────


def test_the_line_states_what_is_free_beside_what_a_run_costs():
    """Two totals in the same units, so the comparison needs no arithmetic."""
    text = mod.describe_free_space(_space(free_gb=1200, total_gb=4000))
    assert text == "1.2 TB free of 4.0 TB · about 6.0 GB for a 20-lamella experiment"


def test_the_estimate_is_the_experiment_not_the_rate():
    """A per-lamella rate makes the reader multiply; it belongs in the tooltip."""
    text = mod.describe_free_space(_space(free_gb=14.2))
    assert "per lamella" not in text
    assert "for a 20-lamella experiment" in text


def test_the_wording_does_not_change_with_the_band():
    """Only the colour moves. A second sentence at 6 GB would say what red says."""
    estimate = "about 6.0 GB for a 20-lamella experiment"
    assert mod.describe_free_space(_space(free_gb=1200, total_gb=4000)).endswith(
        estimate
    )
    assert mod.describe_free_space(_space(free_gb=14.2)).endswith(estimate)
    assert mod.describe_free_space(_space(free_gb=6.1)).endswith(estimate)


def test_a_nearly_full_disk_is_legible_against_the_estimate():
    """The case the whole line exists for: 6.1 free, 6.0 needed, and it is in red."""
    space = _space(free_gb=6.1)
    assert mod.describe_free_space(space).startswith("6.1 GB free of 500.0 GB")
    assert space.level is FreeSpaceLevel.CRITICAL


def test_the_estimate_follows_the_measured_per_lamella_figure():
    """The sentence quotes `BYTES_PER_LAMELLA`, it does not restate a literal."""
    assert mod.TYPICAL_LAMELLA_COUNT * mod.BYTES_PER_LAMELLA == 6e9


# ── the colours ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "free_gb, colour",
    [
        (1200, TEXT_MUTED_COLOR),  # ample: a fact, not a warning
        (14.2, WARN_COLOR),  # low
        (6.1, ERROR_COLOR),  # critical
    ],
)
def test_the_band_sets_the_colour(dialog, free_gb, colour):
    dialog._on_disk_space(_space(free_gb=free_gb), dialog._disk_probe_generation)
    assert colour in dialog.label_disk_space.styleSheet()


def test_a_critical_reading_does_not_block_creating_the_experiment(dialog):
    """The line is advice. One lamella onto a nearly full disk is the user's call."""
    dialog._on_disk_space(_space(free_gb=0.5), dialog._disk_probe_generation)
    assert dialog.btn_ok.isEnabled()


def test_the_tooltip_carries_the_rate_and_the_volume_that_answered(dialog):
    """Both are second questions: 40 lamellae, and "measured where?" on a mapped drive."""
    dialog._on_disk_space(_space(free_gb=100), dialog._disk_probe_generation)
    tooltip = dialog.label_disk_space.toolTip()
    assert "300 MB per lamella" in tooltip
    assert "Z:\\shared" in tooltip


def test_an_unreadable_volume_says_so_without_colouring_it(dialog):
    """An unmapped drive is not a full one, and red would claim it was."""
    dialog._on_disk_space(None, dialog._disk_probe_generation)
    assert "not available" in dialog.label_disk_space.text()
    assert TEXT_MUTED_COLOR in dialog.label_disk_space.styleSheet()
    assert dialog.label_disk_space.toolTip() == ""


# ── answers that arrive too late ─────────────────────────────────────────


def test_a_result_for_a_previous_directory_is_dropped(dialog):
    dialog._on_disk_space(_space(free_gb=100), dialog._disk_probe_generation)
    settled = dialog.label_disk_space.text()

    stale_generation = dialog._disk_probe_generation
    dialog._start_disk_probe()  # as a keystroke in the directory field would
    dialog._on_disk_space(_space(free_gb=6.1), stale_generation)

    assert "6.1 GB" not in dialog.label_disk_space.text()
    assert dialog.label_disk_space.text() != settled  # it is showing the new probe


def test_closing_the_dialog_invalidates_a_probe_still_running(dialog):
    generation = dialog._disk_probe_generation
    dialog.done(0)
    assert generation != dialog._disk_probe_generation


# ── end to end ───────────────────────────────────────────────────────────


def test_showing_the_dialog_fills_the_line_in_from_a_real_volume(dialog):
    """No keystroke required: the directory is prefilled, so the probe is kicked off
    by showEvent. Runs the worker thread for real, on whatever volume tmp is on."""
    dialog.show()
    deadline = time.time() + 10
    while time.time() < deadline and "free of" not in dialog.label_disk_space.text():
        _app.processEvents()
        time.sleep(0.01)
    assert "free of" in dialog.label_disk_space.text()
    assert "lamella experiment" in dialog.label_disk_space.text()
    assert "per lamella" in dialog.label_disk_space.toolTip()
