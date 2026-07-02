#!/usr/bin/env python
"""Standalone test/runner for WorkflowSummaryDialog — no napari/workflow required.

Builds a synthetic per-run summary dataframe (the same shape produced by
TaskManager.build_run_summary_dataframe) and shows the modal summary dialog.

Usage:
    python scripts/test_workflow_summary_dialog.py
    python scripts/test_workflow_summary_dialog.py --empty   # test the empty case
"""

import argparse
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.workflow_summary_dialog import WorkflowSummaryDialog


def sample_dataframe() -> pd.DataFrame:
    """A representative run summary: completed, failed and skipped tasks."""
    return pd.DataFrame(
        [
            {"lamella_name": "lamella-01", "task_name": "Mill Trench", "task_status": "Completed", "completed_at": "10:31 AM", "duration": 182.4},
            {"lamella_name": "lamella-01", "task_name": "Mill Undercut", "task_status": "Completed", "completed_at": "10:45 AM", "duration": 405.1},
            {"lamella_name": "lamella-02", "task_name": "Mill Trench", "task_status": "Failed", "completed_at": "10:52 AM", "duration": 60.0},
            {"lamella_name": "lamella-02", "task_name": "Mill Undercut", "task_status": "Skipped", "completed_at": "", "duration": None},
            {"lamella_name": "lamella-03", "task_name": "Mill Trench", "task_status": "Completed", "completed_at": "11:05 AM", "duration": 175.9},
            {"lamella_name": "lamella-03", "task_name": "Mill Undercut", "task_status": "Completed", "completed_at": "11:20 AM", "duration": 398.7},
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="WorkflowSummaryDialog test runner")
    parser.add_argument("--empty", action="store_true", help="show the dialog with no tasks")
    args = parser.parse_args()

    df = pd.DataFrame() if args.empty else sample_dataframe()

    app = QApplication(sys.argv)
    dialog = WorkflowSummaryDialog(df)
    dialog.setWindowTitle("Workflow Summary (test)")
    result = dialog.exec_()
    print(f"Dialog closed with result={result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
