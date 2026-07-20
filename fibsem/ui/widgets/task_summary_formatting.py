"""
Shared formatting helpers for task summary / task history tables.

Used by both TaskHistoryTableWidget (full experiment history) and
WorkflowSummaryDialog (per-run summary) so the two stay visually consistent.
"""
from typing import Callable, List, Optional

import pandas as pd

# Default per-run summary columns (in display order)
SUMMARY_COLUMNS: List[str] = [
    "lamella_name",
    "task_name",
    "task_status",
    "completed_at",
    "duration",
]

# Map raw dataframe columns to nice display titles
COLUMN_NAME_MAPPING = {
    "lamella_name": "Lamella",
    "task_name": "Task Name",
    "task_status": "Status",
    "task_status_message": "Status Message",
    "completed_at": "Completed At",
    "duration": "Duration",
}

# Foreground colour for each status value (plain table cells)
STATUS_COLORS = {
    "Failed": "red",
    "Completed": "white",
    "Skipped": "gray",
    "InProgress": "cyan",
    "Cancelled": "orange",
}

# Semantic badge colours (matching the fibsem.ui.stylesheets palette) used by
# the workflow summary dialog: status -> (dot/strong colour, muted text colour).
STATUS_BADGE_COLORS = {
    "Completed": ("#4caf50", "#7fd083"),
    "Failed": ("#d04040", "#e08585"),
    "Skipped": ("#868e93", "#aeb4b9"),
    "InProgress": ("#50a6ff", "#9cc7f5"),
    "Cancelled": ("#e0a030", "#e8c37f"),  # amber: user-aborted, not an error
}

# Status order for count chips (these three are always shown, even at zero)
STATUS_CHIP_ORDER = ["Completed", "Failed", "Skipped"]


def format_duration_short(seconds) -> str:
    """Format a duration in seconds as MMm:SSs (blank for missing values)."""
    if seconds is None or seconds == "" or pd.isna(seconds):
        return ""
    try:
        total_seconds = int(float(seconds))
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes:02d}m:{secs:02d}s"
    except (ValueError, TypeError):
        return str(seconds)


def prepare_summary_dataframe(df: Optional[pd.DataFrame],
                              columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Select summary columns, format the duration column, and rename to display titles.

    Args:
        df: raw dataframe (e.g. from task_history_dataframe / build_run_summary_dataframe)
        columns: raw column names to keep (defaults to SUMMARY_COLUMNS)

    Returns:
        A new dataframe with display titles, or the input unchanged if empty/None.
    """
    if df is None or df.empty:
        return df

    if columns is None:
        columns = SUMMARY_COLUMNS

    available = [col for col in columns if col in df.columns]
    df = df[available].copy()

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(format_duration_short)

    return df.rename(columns=COLUMN_NAME_MAPPING)


def make_status_cell_formatter(df: pd.DataFrame) -> Callable[[int, int, object], dict]:
    """Return a DataFrameTableWidget cell_formatter that colours rows by Status.

    The dataframe passed here must already use display titles (i.e. have a
    'Status' column), as produced by prepare_summary_dataframe.
    """
    def format_cell(row_idx: int, col_idx: int, value) -> dict:
        format_dict: dict = {}
        if "Status" in df.columns:
            status_col_idx = df.columns.get_loc("Status")
            status_value = str(df.iloc[row_idx, status_col_idx])
            color = STATUS_COLORS.get(status_value)
            if color:
                format_dict["color"] = color
        return format_dict

    return format_cell
