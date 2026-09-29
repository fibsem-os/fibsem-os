"""What the Load Experiment dialog says an experiment costs on disk.

Everything else in that form comes out of experiment.yaml, which is already in memory
by the time the form is filled. The size has to be walked for -- one `stat` per file,
thousands of them for a real experiment, every one a round trip when the experiment is
on a share -- so it runs on a worker thread and arrives late. What is checked here is
the arriving-late part: that the field says so meanwhile, and that an answer for an
experiment the user has already clicked past is dropped rather than shown against the
wrong one.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_load_experiment_size.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_load_experiment_widget as mod,
)

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def dialog():
    widget = mod.AutoLamellaLoadExperimentWidget()
    yield widget
    # done() before deleteLater: it invalidates a walk still running, which is the
    # thing that would otherwise write to a deleted field.
    widget.done(0)
    widget.deleteLater()
    _app.processEvents()


@pytest.fixture
def experiment(tmp_path):
    """A real experiment on disk with a known number of bytes in it."""
    exp = Experiment.create(path=tmp_path, name="AutoLamella-test")
    lamella_dir = Path(exp.path) / "01-test-lamella"
    lamella_dir.mkdir()
    (lamella_dir / "ref_start_eb.tif").write_bytes(b"x" * 400_000)
    (lamella_dir / "ref_start_ib.tif").write_bytes(b"x" * 400_000)
    return exp


def _settle(dialog, deadline_s: float = 15.0) -> str:
    """Pump the event loop until the walk lands, and return what the field says."""
    deadline = time.time() + deadline_s
    while (
        time.time() < deadline
        and dialog.lineEdit_experiment_size.text() == mod.SIZE_MEASURING_TEXT
    ):
        _app.processEvents()
        time.sleep(0.01)
    return dialog.lineEdit_experiment_size.text()


def _load(dialog, experiment) -> bool:
    return dialog._load_experiment_from_path(
        os.path.join(str(experiment.path), "experiment.yaml"),
        warn_on_missing_protocol=False,
    )


# ── the field ────────────────────────────────────────────────────────────


def test_nothing_loaded_shows_no_size(dialog):
    assert dialog.lineEdit_experiment_size.text() == ""
    assert (
        dialog.lineEdit_experiment_size.placeholderText() == mod.SIZE_UNAVAILABLE_TEXT
    )


def test_loading_says_it_is_measuring_before_the_walk_finishes(dialog, experiment):
    """The form is filled from memory; the size is not there yet and says so."""
    assert _load(dialog, experiment)
    assert dialog.lineEdit_experiment_size.text() == mod.SIZE_MEASURING_TEXT


def test_the_measured_size_replaces_it(dialog, experiment):
    assert _load(dialog, experiment)
    assert _settle(dialog) == "800 kB"


def test_clearing_the_display_empties_the_field(dialog, experiment):
    assert _load(dialog, experiment)
    _settle(dialog)
    dialog._clear_display()
    assert dialog.lineEdit_experiment_size.text() == ""


# ── answers that arrive too late ─────────────────────────────────────────


def test_a_measurement_for_a_previous_selection_is_dropped(dialog, experiment):
    """Clicking down the recent list starts a walk per row; only the last may show."""
    assert _load(dialog, experiment)
    _settle(dialog)

    stale_generation = dialog._size_probe_generation
    dialog._start_size_probe()  # as settling on the next row would
    dialog._on_size_measured(999_000_000, stale_generation)

    assert "999" not in dialog.lineEdit_experiment_size.text()


def test_clearing_the_display_invalidates_a_walk_still_running(dialog, experiment):
    assert _load(dialog, experiment)
    generation = dialog._size_probe_generation
    dialog._clear_display()
    assert generation != dialog._size_probe_generation


def test_closing_the_dialog_invalidates_a_walk_still_running(dialog, experiment):
    assert _load(dialog, experiment)
    generation = dialog._size_probe_generation
    dialog.done(0)
    assert generation != dialog._size_probe_generation


# ── the directory going away ─────────────────────────────────────────────


def test_an_unreadable_directory_reads_as_unknown_rather_than_empty(
    dialog, experiment, tmp_path
):
    """`directory_size` skips what it cannot read, so an unreachable share and an
    empty experiment both come back as zero. "0 B" would be a claim; blank is not."""
    assert _load(dialog, experiment)
    # The share drops between reading experiment.yaml and measuring.
    dialog.experiment.path = str(tmp_path / "gone")
    dialog._start_size_probe()
    assert dialog.lineEdit_experiment_size.text() == ""
