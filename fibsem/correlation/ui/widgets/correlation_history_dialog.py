"""CorrelationHistoryDialog — pick which previous run a re-correlation seeds from.

Each open of the correlation tool writes a run to
``<lamella>/Correlation/<timestamp>/`` (FIB-264); :class:`LamellaCorrelation`
reconstructs that history from disk (FIB-299). With one prior run the newest is
seeded silently; with several, this dialog lets the user choose which to seed
from — or start fresh — so a bad re-correlation isn't the only thing to build on
(FIB-257).

Purely a chooser: it returns the selected :class:`CorrelationRun` (or ``None``
for "start fresh") and never mutates the runs.
"""
from __future__ import annotations

import datetime
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
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

from fibsem import constants
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import CorrelationState
from fibsem.ui import stylesheets

_MUTED = "#8a8d93"
_OK = "#a5d6a7"
_STALE = "#ffcc80"

_TABLE_STYLE = (
    "QTableWidget { background: #1e2124; color: #d0d0d0; border: 1px solid #2a2d31; }"
    "QHeaderView::section { background: #262930; color: #b0b3b8; padding: 4px 10px;"
    " border: none; }"
    "QTableWidget::item { padding: 5px 10px; }"
    "QTableWidget::item:selected { background: #264f78; color: #ffffff; }"
)


def _format_timestamp(name: str) -> str:
    """Reformat a run folder name (a ``DATETIME_FILE`` stamp) for display.

    Falls back to the raw name for a folder that doesn't parse (e.g. a legacy or
    hand-named directory), so an odd name still shows rather than disappearing.
    """
    try:
        dt = datetime.datetime.strptime(name, constants.DATETIME_FILE)
    except ValueError:
        return name
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _points_summary(state: CorrelationState) -> str:
    """Compact count of the seed coordinates a run carries."""
    data = state.input_data
    parts = []
    for count, label in (
        (len(data.fib_coordinates), "FIB"),
        (len(data.fm_coordinates), "FM"),
        (len(data.poi_coordinates), "POI"),
    ):
        if count:
            parts.append(f"{count} {label}")
    return " · ".join(parts) if parts else "—"


def _result_cell(state: CorrelationState) -> tuple[str, str]:
    """(text, hex colour) describing a run's computed result, if any.

    Reuses ``matches_inputs`` — the same derived-staleness machinery FIB-295 added
    — to flag a run whose transform no longer describes its own coordinates.
    """
    result = state.result
    if result is None:
        return "No result", _MUTED
    text = f"{result.rms_error:.2f} px RMS"
    if not result.matches_inputs(state.input_data):
        return f"{text} · stale", _STALE
    return text, _OK


class CorrelationHistoryDialog(QDialog):
    """Modal chooser over a lamella's previous correlation runs.

    ``exec_()`` returns ``QDialog.Accepted`` for both "seed from selected" and
    "start fresh"; :attr:`selected_run` disambiguates (a run, or ``None`` for
    fresh). ``Rejected`` means the user cancelled opening the correlation tool.
    """

    def __init__(
        self,
        history: LamellaCorrelation,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Newest first — the run most likely to be seeded sits at the top and is
        # pre-selected. LamellaCorrelation.runs is oldest-first.
        self._display_runs: List[CorrelationRun] = list(reversed(history.runs))
        self._selected_run: Optional[CorrelationRun] = None

        self.setWindowTitle("Correlation history")
        self.setModal(True)
        self.setStyleSheet("background: #1e2124; color: #d0d0d0;")
        self.setMinimumWidth(420)  # a floor; the table's content usually drives it wider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)

        title = QLabel("Seed from a previous correlation")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #e0e0e0;")
        subtitle = QLabel(
            "The newest run is selected. Choose an older one, or start fresh."
        )
        subtitle.setStyleSheet(f"color: {_MUTED}; font-size: 12px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.table = self._build_table()
        layout.addWidget(self.table)
        self._fit_dialog_width_to_table()

        layout.addLayout(self._build_buttons())

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(len(self._display_runs), 3)
        table.setHorizontalHeaderLabels(["When", "Points", "Result"])
        table.setStyleSheet(_TABLE_STYLE)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.itemDoubleClicked.connect(lambda _item: self._accept_selected())

        # Size every column to its content (which counts the item padding) so no
        # cell text elides. _fit_dialog_width_to_table then widens the dialog to
        # fit all three and lets the last column stretch into the small remainder —
        # done in that order because a stretch column sized before the dialog is
        # wide enough would just clip the RMS/stale text.
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        for row, run in enumerate(self._display_runs):
            table.setItem(row, 0, QTableWidgetItem(_format_timestamp(run.name)))
            table.setItem(row, 1, QTableWidgetItem(_points_summary(run.state)))
            text, colour = _result_cell(run.state)
            result_item = QTableWidgetItem(text)
            result_item.setForeground(QColor(colour))
            table.setItem(row, 2, result_item)

        if self._display_runs:
            table.selectRow(0)  # newest
        return table

    def _fit_dialog_width_to_table(self) -> None:
        """Widen the dialog to the table's content so no cell text elides.

        Offscreen (and some styles) don't propagate the table's size hint up to
        the dialog, so set the width from the measured column widths directly —
        font metrics need no display. The last column then stretches to absorb the
        slack, so the guaranteed-ample width leaves no trailing gap.
        """
        self.table.resizeColumnsToContents()
        cols = sum(
            self.table.columnWidth(c) for c in range(self.table.columnCount())
        )
        frame = 2 * self.table.frameWidth()
        margins = self.layout().contentsMargins()
        needed = cols + frame + margins.left() + margins.right() + 24  # border slack
        self.setMinimumWidth(max(self.minimumWidth(), needed))
        self.table.horizontalHeader().setStretchLastSection(True)

        # Hug the rows so the dialog doesn't carry a band of empty table below the
        # last run. Capped so a long history scrolls instead of running off-screen.
        self.table.resizeRowsToContents()
        rows_h = sum(
            self.table.rowHeight(r) for r in range(self.table.rowCount())
        )
        header_h = self.table.horizontalHeader().height()
        self.table.setMaximumHeight(header_h + rows_h + frame + 2)

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        fresh_btn = QPushButton("Start fresh")
        fresh_btn.setToolTip("Open the correlation tool with no seeded coordinates.")
        fresh_btn.setAutoDefault(False)
        fresh_btn.clicked.connect(self._start_fresh)
        row.addWidget(fresh_btn)

        row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)

        self.seed_btn = QPushButton("Seed from selected")
        self.seed_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.seed_btn.setDefault(True)
        self.seed_btn.setAutoDefault(True)
        self.seed_btn.setEnabled(bool(self._display_runs))
        self.seed_btn.clicked.connect(self._accept_selected)
        row.addWidget(self.seed_btn)
        return row

    @property
    def selected_run(self) -> Optional[CorrelationRun]:
        """The run chosen to seed from, or ``None`` when the user started fresh."""
        return self._selected_run

    def _accept_selected(self) -> None:
        if not self._display_runs:
            return
        row = self.table.currentRow()
        self._selected_run = self._display_runs[row if row >= 0 else 0]
        self.accept()

    def _start_fresh(self) -> None:
        self._selected_run = None
        self.accept()
