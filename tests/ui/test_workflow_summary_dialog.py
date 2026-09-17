"""The run summary says when the run ended short of done: a run that gave up
waiting for a review carries that as a headline, a finished run carries none."""

import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.widgets.workflow_summary_dialog import WorkflowSummaryDialog

ROWS = pd.DataFrame(
    [
        {
            "lamella_name": "01-a",
            "task_name": "Setup Lamella Position",
            "task_status": "AwaitingDecision",
            "completed_at": None,
            "duration": 12.0,
        }
    ]
)


def test_a_run_that_gave_up_waiting_says_so_in_the_headline(qapp):
    note = (
        "Timed out after 30m waiting for a review: 1 decision(s) still pending. "
        "Decide in the Review tab, then Run again."
    )
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
