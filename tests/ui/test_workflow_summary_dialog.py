"""The run summary says when the run ended short of done: a run that stalled
on work it could not run carries that as a headline, a finished run carries none.
The run-end toast counts the run's tasks by status."""

import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.widgets.workflow_summary_dialog import (
    WorkflowSummaryDialog,
    run_counts_text,
)

ROWS = pd.DataFrame(
    [
        {
            "lamella_name": "01-a",
            "task_name": "Setup Lamella Position",
            "task_status": "Skipped",
            "completed_at": None,
            "duration": 12.0,
        }
    ]
)


def test_a_run_that_stalled_says_so_in_the_headline(qapp):
    note = "1 task(s) cannot run: 01-a/Rough Milling"
    dialog = WorkflowSummaryDialog(ROWS, note=note)
    try:
        assert dialog.note_label.text() == note
    finally:
        dialog.deleteLater()


def test_a_finished_run_has_no_headline(qapp):
    dialog = WorkflowSummaryDialog(ROWS)
    try:
        assert not hasattr(dialog, "note_label")
    finally:
        dialog.deleteLater()


def test_the_run_end_toast_counts_only_the_statuses_that_occurred():
    rows = pd.DataFrame(
        [
            {"lamella_name": "01", "task_status": "Completed"},
            {"lamella_name": "02", "task_status": "Skipped"},
            {"lamella_name": "03", "task_status": "Completed"},
        ]
    )
    assert run_counts_text(rows) == "2 completed, 1 skipped"
