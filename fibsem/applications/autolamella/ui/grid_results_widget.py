"""Grids tab · Results: what the selected grid's runs recorded.

Follows card selection, in the shape of the Lamella tab's Review view: the name,
a line on the latest run, then one row per history entry in the order they
happened. A task's row carries the thumbnail it recorded, and clicking it opens
the full overview; a task that recorded nothing says why instead. The load is a
row like any other. Everything is read off the history and the outputs recorded
on it, never by globbing the grid's directory.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    ClickableLabel,
    ExpandedImageDialog,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME as _LOAD_ENTRY_NAME,
)
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.ui import stylesheets
from fibsem.ui.tokens import (
    ERROR_COLOR,
    NEUTRAL_200,
    NEUTRAL_400,
    NEUTRAL_550,
    NEUTRAL_900,
    OK_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel

_TILE_W, _TILE_H = 320, 213  # 3:2, the Review tab's proportions at a card-friendly size

_STATUS_COLOUR = {
    AutoLamellaTaskStatus.Completed: OK_COLOR,
    AutoLamellaTaskStatus.Failed: ERROR_COLOR,
    AutoLamellaTaskStatus.Cancelled: stylesheets.DEFECT_ORANGE_COLOR,
    AutoLamellaTaskStatus.Skipped: NEUTRAL_550,
    AutoLamellaTaskStatus.InProgress: stylesheets.GREEN_COLOR,
}


def _when(state: AutoLamellaTaskState) -> str:
    stamp = state.end_timestamp or state.start_timestamp
    return datetime.fromtimestamp(stamp).strftime(TIME_DISPLAY_AMPM_SHORT)


def _recorded(
    experiment: Experiment,
    grid: GridRecord,
    state: AutoLamellaTaskState,
    thumbnail: bool,
) -> Optional[str]:
    """The last existing file this run recorded: the thumbnail, or the image
    itself. None when there is none on disk."""
    root = experiment.grid_path(grid)
    for role, relpaths in reversed(list(state.outputs.items())):
        if role.endswith("_thumbnail") != thumbnail:
            continue
        for relpath in reversed(relpaths):
            path = os.path.join(root, relpath)
            if os.path.isfile(path):
                return path
    return None


def thumbnail_for(
    experiment: Experiment, grid: GridRecord, state: AutoLamellaTaskState
) -> Optional[str]:
    return _recorded(experiment, grid, state, thumbnail=True)


def image_for(
    experiment: Experiment, grid: GridRecord, state: AutoLamellaTaskState
) -> Optional[str]:
    return _recorded(experiment, grid, state, thumbnail=False)


def latest_runs(grid: GridRecord) -> dict:
    """The most recent history entry per task, in first-run order; the load
    step is not a task."""
    latest = {}
    for state in grid.task_history:
        if state.name != _LOAD_ENTRY_NAME:
            latest[state.name] = state
    return latest


class _HistoryRow(QWidget):
    """One history entry: a separator, a line saying what and how it went,
    and the image it recorded, if any."""

    def __init__(
        self,
        state: AutoLamellaTaskState,
        thumbnail: Optional[str],
        image: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.thumbnail = thumbnail
        self.image = image
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        line = QHBoxLayout()
        line.setSpacing(8)
        self.name_label = QLabel(state.name)
        self.name_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {NEUTRAL_400}; "
            "background: transparent;"
        )
        line.addWidget(self.name_label)
        colour = _STATUS_COLOUR.get(state.status, NEUTRAL_550)
        self.status_label = QLabel(state.status.name)
        self.status_label.setStyleSheet(
            f"font-size: 11px; color: {colour}; background: transparent;"
        )
        line.addWidget(self.status_label)
        self.when_label = QLabel(_when(state))
        self.when_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        line.addWidget(self.when_label)
        self.detail_label = ElidedLabel(state.status_message or "")
        self.detail_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        line.addWidget(self.detail_label, 1)
        layout.addLayout(line)

        self.tile: Optional[ClickableLabel] = None
        if thumbnail is not None:
            # The tile shows the thumbnail but opens the full overview: the
            # dialog reads the image and its pixel size for a scale bar, which
            # a PNG cannot give it.
            self.tile = ClickableLabel(image or thumbnail)
            self.tile.setFixedSize(_TILE_W, _TILE_H)
            self.tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tile.setStyleSheet(f"background: {NEUTRAL_900}; border-radius: 4px;")
            pixmap = QPixmap(thumbnail)
            if not pixmap.isNull():
                self.tile.setPixmap(
                    pixmap.scaled(
                        _TILE_W, _TILE_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            self.tile.clicked.connect(self._open)
            layout.addWidget(self.tile, 0, Qt.AlignLeft)
        elif (
            state.name != _LOAD_ENTRY_NAME
            and state.status is AutoLamellaTaskStatus.Completed
        ):
            note = QLabel("No image recorded.")
            note.setStyleSheet(
                f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
            )
            layout.addWidget(note)

    def _open(self, path: str) -> None:
        dialog = ExpandedImageDialog(
            path,
            title=f"{self.name_label.text()} · {self.when_label.text()}",
            parent=self,
        )
        dialog.show()


class GridResultsWidget(QWidget):
    """The selected grid: its name, the latest run, and its history with images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._grid: Optional[GridRecord] = None
        self._rows: List[QWidget] = []
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(4)

        self.name_label = QLabel()
        self.name_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        self._layout.addWidget(self.name_label)
        self.subtitle_label = QLabel()
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        self._layout.addWidget(self.subtitle_label)

        self.rows_widget = QWidget()
        self.rows_widget.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self.rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._layout.addWidget(self.rows_widget)

        self.empty_label = QLabel("Select a grid card to see its results.")
        self.empty_label.setStyleSheet(
            f"color: {NEUTRAL_550}; font-size: 12px; background: transparent;"
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self.empty_label)
        self._layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    # -- model -----------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        self.refresh()

    def set_grid(self, grid: Optional[GridRecord]) -> None:
        self._grid = grid
        self.refresh()

    @property
    def grid(self) -> Optional[GridRecord]:
        return self._grid

    @property
    def rows(self) -> List[QWidget]:
        return list(self._rows)

    def refresh(self) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []

        grid, experiment = self._grid, self._experiment
        has_grid = grid is not None and experiment is not None
        for widget in (self.name_label, self.subtitle_label, self.rows_widget):
            widget.setVisible(has_grid)
        self.empty_label.setVisible(not has_grid)
        if not has_grid:
            return

        self.name_label.setText(grid.name)
        self.name_label.setToolTip(str(experiment.grid_path(grid)))
        parts = []
        if grid.description:
            parts.append(grid.description)
        last = next(
            (
                t
                for t in reversed(grid.task_history)
                if t.name != _LOAD_ENTRY_NAME
                and t.status is AutoLamellaTaskStatus.Completed
            ),
            None,
        )
        parts.append(
            f"{last.name}, completed at {last.completed_at}"
            if last is not None
            else "No completed tasks"
        )
        self.subtitle_label.setText(" · ".join(parts))

        if not grid.task_history:
            note = QLabel("Nothing has run on this grid yet.")
            note.setStyleSheet(
                f"font-size: 11px; color: {NEUTRAL_550}; background: transparent; "
                "padding-top: 8px;"
            )
            self._rows_layout.addWidget(note)
            self._rows.append(note)
            return
        for state in grid.task_history:
            row = _HistoryRow(
                state,
                thumbnail_for(experiment, grid, state),
                image_for(experiment, grid, state),
            )
            self._rows_layout.addWidget(row)
            self._rows.append(row)
