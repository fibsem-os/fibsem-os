"""
Modal dialog showing a summary of the tasks run in a single workflow run.

Displays one row per (lamella, task) attempted in the run with status,
completion time and duration, count chips and a primary OK button.
"""
from typing import List, Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET
from fibsem.ui.widgets.task_summary_formatting import (
    STATUS_BADGE_COLORS,
    STATUS_CHIP_ORDER,
    format_duration_short,
)

# Palette (matching fibsem.ui.stylesheets napari theme)
_BG = "#262930"
_PANEL = "#1e2027"
_ROW_ALT = "#2b2f38"
_BORDER = "#3d4251"
_TEXT = "#d6d6d6"
_TEXT_STRONG = "#f0f1f2"
_TEXT_MUTED = "#868e93"

_TABLE_STYLE = f"""
QTableWidget {{
    background-color: {_BG};
    alternate-background-color: {_ROW_ALT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    gridline-color: transparent;
    outline: none;
}}
QTableWidget::item {{
    padding: 6px 10px;
    border: none;
}}
QTableWidget::item:selected {{
    background-color: #2d3f5c;
    color: {_TEXT_STRONG};
}}
QHeaderView::section {{
    background-color: {_PANEL};
    color: {_TEXT_MUTED};
    padding: 7px 10px;
    border: none;
    border-bottom: 1px solid {_BORDER};
    font-weight: 500;
}}
"""

_COLUMNS = ["Lamella", "Task", "Status", "Completed", "Duration"]


class _NumericItem(QTableWidgetItem):
    """Table item that sorts by its numeric Qt.UserRole value rather than text.

    Lets the Duration column (displayed as 'MMm:SSs') sort by the underlying
    seconds, so e.g. '100m:00s' sorts after '99m:00s'.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a = self.data(Qt.UserRole)
        b = other.data(Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


class WorkflowSummaryDialog(QDialog):
    """A modal dialog that displays a per-run task summary table with an OK button."""

    def __init__(self, dataframe: pd.DataFrame, parent: Optional[QWidget] = None):
        """
        Args:
            dataframe: raw run-summary dataframe with columns
                lamella_name, task_name, task_status, completed_at, duration
            parent: the parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Workflow Summary")
        self.resize(780, 460)
        self.setStyleSheet(f"QDialog {{ background-color: {_BG}; }}")

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)
        self.setLayout(layout)

        # header: title + meta line
        header_layout = QHBoxLayout()
        title_label = QLabel("Workflow summary")
        title_label.setStyleSheet(f"font-size: 16px; font-weight: 500; color: {_TEXT_STRONG};")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        meta_label = QLabel(self._format_meta(dataframe))
        meta_label.setStyleSheet(f"font-size: 12px; color: {_TEXT_MUTED};")
        header_layout.addWidget(meta_label)
        layout.addLayout(header_layout)

        # count chips
        layout.addLayout(self._build_chip_row(dataframe))

        # table
        self.table = self._build_table(dataframe)
        layout.addWidget(self.table)

        # footer: primary OK button
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        ok_button.setMinimumWidth(80)
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        footer_layout.addWidget(ok_button)
        layout.addLayout(footer_layout)

    # --- builders ---

    def _build_chip_row(self, df: Optional[pd.DataFrame]) -> QHBoxLayout:
        """Build the row of status count chips."""
        chip_layout = QHBoxLayout()
        chip_layout.setSpacing(8)

        counts = {}
        if df is not None and not df.empty and "task_status" in df.columns:
            counts = df["task_status"].value_counts().to_dict()

        # always show the standard three, plus any other statuses that appear
        statuses = list(STATUS_CHIP_ORDER)
        for status in counts:
            if status not in statuses:
                statuses.append(status)

        for status in statuses:
            chip_layout.addWidget(self._make_chip(status, int(counts.get(status, 0))))
        chip_layout.addStretch()
        return chip_layout

    @staticmethod
    def _make_chip(status: str, count: int) -> QLabel:
        """A pill label: coloured dot + 'N status'."""
        dot_color, text_color = STATUS_BADGE_COLORS.get(status, ("#868e93", "#aeb4b9"))
        bg = QColor(dot_color)
        bg_rgba = f"rgba({bg.red()}, {bg.green()}, {bg.blue()}, 0.15)"
        chip = QLabel(f'<span style="color:{dot_color};">&#9679;</span> {count} {status.lower()}')
        chip.setStyleSheet(
            f"background-color: {bg_rgba}; color: {text_color};"
            f"padding: 3px 10px; border-radius: 10px; font-size: 12px;"
        )
        return chip

    def _build_table(self, df: Optional[pd.DataFrame]) -> QTableWidget:
        """Build the styled summary table from the raw dataframe."""
        table = QTableWidget()
        table.setStyleSheet(_TABLE_STYLE)
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(Qt.NoFocus)

        mono_font = QFont("Menlo")
        mono_font.setStyleHint(QFont.Monospace)

        rows = [] if df is None or df.empty else df.to_dict("records")
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            status = str(row.get("task_status", ""))
            _dot_color, text_color = STATUS_BADGE_COLORS.get(status, (_TEXT, _TEXT))

            lamella_item = QTableWidgetItem(str(row.get("lamella_name", "")))
            lamella_item.setForeground(QColor(_TEXT_STRONG))

            task_item = QTableWidgetItem(str(row.get("task_name", "")))

            status_item = QTableWidgetItem(f"● {status}")
            status_item.setForeground(QColor(text_color))

            completed_item = QTableWidgetItem(str(row.get("completed_at", "") or ""))
            completed_item.setForeground(QColor(_TEXT_MUTED))

            dur_seconds = row.get("duration")
            duration_item = _NumericItem(format_duration_short(dur_seconds))
            try:
                sort_value = float(dur_seconds) if not pd.isna(dur_seconds) else -1.0
            except (TypeError, ValueError):
                sort_value = -1.0
            duration_item.setData(Qt.UserRole, sort_value)
            duration_item.setForeground(QColor(_TEXT_MUTED))
            duration_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            duration_item.setFont(mono_font)

            for col, item in enumerate(
                [lamella_item, task_item, status_item, completed_item, duration_item]
            ):
                table.setItem(i, col, item)

        # fixed widths for the narrow columns; Task stretches into the remainder
        # so long task names stay readable without a horizontal scrollbar.
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        fixed_widths = {0: 155, 2: 140, 3: 115, 4: 100}
        for col, width in fixed_widths.items():
            header.setSectionResizeMode(col, QHeaderView.Interactive)
            table.setColumnWidth(col, width)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        table.verticalHeader().setDefaultSectionSize(34)
        return table

    # --- helpers ---

    @staticmethod
    def _format_meta(df: Optional[pd.DataFrame]) -> str:
        """Build the 'N tasks · M lamellae · T total' meta line."""
        if df is None or df.empty:
            return "No tasks were run"

        n_tasks = len(df)
        parts: List[str] = [f"{n_tasks} task" + ("" if n_tasks == 1 else "s")]

        if "lamella_name" in df.columns:
            n_lamellae = int(df["lamella_name"].nunique())
            parts.append(f"{n_lamellae} lamella" + ("" if n_lamellae == 1 else "e"))

        if "duration" in df.columns:
            total = pd.to_numeric(df["duration"], errors="coerce").fillna(0).sum()
            if total > 0:
                parts.append(f"{format_duration_short(total)} total")

        return " · ".join(parts)


def main():
    """Standalone demo with sample data (no UI/workflow required)."""
    import sys

    from PyQt5.QtWidgets import QApplication

    df = pd.DataFrame(
        [
            {"lamella_name": "01-nice-mako", "task_name": "Setup Lamella Position", "task_status": "Completed", "completed_at": "01:37 PM", "duration": 24.0},
            {"lamella_name": "02-awake-stork", "task_name": "Setup Lamella Position", "task_status": "Completed", "completed_at": "01:37 PM", "duration": 23.0},
            {"lamella_name": "01-nice-mako", "task_name": "Mill Fiducial", "task_status": "Failed", "completed_at": "01:38 PM", "duration": 39.0},
            {"lamella_name": "02-awake-stork", "task_name": "Mill Fiducial", "task_status": "Skipped", "completed_at": "", "duration": None},
        ]
    )

    app = QApplication(sys.argv)
    dialog = WorkflowSummaryDialog(df)
    dialog.exec_()
    sys.exit(0)


if __name__ == "__main__":
    main()
