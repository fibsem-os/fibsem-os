"""Grids tab · Results: the selected grid's overviews and history.

Follows card selection. Everything here is read off the grid's task history and
the outputs recorded on it, never by globbing the grid's directory: a task that
recorded nothing shows a labelled placeholder saying why, which is the whole
difference between "failed" and "never ran".
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
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
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.ui import stylesheets
from fibsem.ui.tokens import (
    ERROR_COLOR,
    NEUTRAL_200,
    NEUTRAL_400,
    NEUTRAL_550,
    NEUTRAL_900,
    OK_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel

_LOAD_ENTRY_NAME = "load"
_TILE_W, _TILE_H = 220, 150
_TILES_PER_LINE = 3

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


def thumbnail_for(
    experiment: Experiment, grid: GridRecord, state: AutoLamellaTaskState
) -> Optional[str]:
    """The thumbnail a task run recorded, if the file is still there."""
    root = experiment.grid_path(grid)
    for role, relpaths in state.outputs.items():
        if not role.endswith("_thumbnail"):
            continue
        for relpath in reversed(relpaths):
            path = os.path.join(root, relpath)
            if os.path.isfile(path):
                return path
    return None


def latest_runs(grid: GridRecord) -> Dict[str, AutoLamellaTaskState]:
    """The most recent history entry per task, in first-run order. The load step
    is not a task and is reported separately."""
    latest: Dict[str, AutoLamellaTaskState] = {}
    for state in grid.task_history:
        if state.name != _LOAD_ENTRY_NAME:
            latest[state.name] = state
    return latest


class _OverviewTile(QWidget):
    """One task's latest result: its thumbnail, or a placeholder that says why not."""

    def __init__(
        self,
        title: str,
        state: Optional[AutoLamellaTaskState],
        thumbnail: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.thumbnail = thumbnail
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.image = ClickableLabel(thumbnail or "")
        self.image.setFixedSize(_TILE_W, _TILE_H)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet(f"background: {NEUTRAL_900}; border-radius: 4px;")
        if thumbnail is not None:
            pixmap = QPixmap(thumbnail)
            if not pixmap.isNull():
                self.image.setPixmap(
                    pixmap.scaled(
                        _TILE_W, _TILE_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            self.image.clicked.connect(self._open)
        else:
            self.image.setWordWrap(True)
            if state is None:
                text, colour = "not run", NEUTRAL_550
            elif state.status is AutoLamellaTaskStatus.Completed:
                text, colour = "no image recorded", NEUTRAL_550
            else:
                text = state.status.name.lower()
                if state.status_message:
                    text += f"\n{state.status_message}"
                colour = _STATUS_COLOUR.get(state.status, NEUTRAL_550)
            self.image.setText(text)
            self.image.setStyleSheet(
                f"background: {NEUTRAL_900}; border-radius: 4px; color: {colour}; "
                "font-size: 11px; padding: 6px;"
            )
        layout.addWidget(self.image)

        caption = title if state is None else f"{title} · {_when(state)}"
        self.caption = ElidedLabel(caption)
        self.caption.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_400}; background: transparent;"
        )
        layout.addWidget(self.caption)

    def _open(self, path: str) -> None:
        dialog = ExpandedImageDialog(path, title=self.caption.text(), parent=self)
        dialog.show()


class GridResultsWidget(QWidget):
    """The selected grid: its overviews by task, the load, and the history."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._grid: Optional[GridRecord] = None
        self._tiles: List[_OverviewTile] = []
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
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.name_label = QLabel()
        self.name_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        layout.addWidget(self.name_label)
        self.path_label = ElidedLabel("", mode=Qt.ElideLeft)
        self.path_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.path_label)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_400}; background: transparent;"
        )
        layout.addWidget(self.description_label)

        self.tiles_widget = QWidget()
        self.tiles_widget.setStyleSheet("background: transparent;")
        self._tiles_layout = QGridLayout(self.tiles_widget)
        self._tiles_layout.setContentsMargins(0, 4, 0, 4)
        self._tiles_layout.setSpacing(10)
        self._tiles_layout.setColumnStretch(_TILES_PER_LINE, 1)
        layout.addWidget(self.tiles_widget)

        self.load_label = QLabel()
        self.load_label.setWordWrap(True)
        self.load_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_400}; background: transparent;"
        )
        layout.addWidget(self.load_label)

        history_title = QLabel("History")
        history_title.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {NEUTRAL_400}; "
            "background: transparent;"
        )
        layout.addWidget(history_title)
        self.history = QTableWidget(0, 4)
        self.history.setHorizontalHeaderLabels(["Task", "When", "Status", "Details"])
        self.history.verticalHeader().setVisible(False)
        self.history.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history.setSelectionMode(QAbstractItemView.NoSelection)
        self.history.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.history)

        self.empty_label = QLabel("Select a grid card to see its results.")
        self.empty_label.setStyleSheet(
            f"color: {NEUTRAL_550}; font-size: 12px; background: transparent;"
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)
        layout.addStretch(1)
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

    def refresh(self) -> None:
        for tile in self._tiles:
            self._tiles_layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles = []
        self.history.setRowCount(0)

        grid, experiment = self._grid, self._experiment
        has_grid = grid is not None and experiment is not None
        for widget in (
            self.name_label,
            self.path_label,
            self.description_label,
            self.tiles_widget,
            self.load_label,
            self.history,
        ):
            widget.setVisible(has_grid)
        self.empty_label.setVisible(not has_grid)
        if not has_grid:
            return

        self.name_label.setText(grid.name)
        self.path_label.setText(str(experiment.grid_path(grid)))
        self.description_label.setText(grid.description)
        self.description_label.setVisible(bool(grid.description))

        # One tile per task: the protocol's tasks in its order, then anything the
        # history knows that the protocol no longer names.
        names: List[str] = []
        try:
            names = list(experiment.grid_protocol.ordered_task_names)
        except ValueError:  # no task protocol on this experiment
            pass
        runs = latest_runs(grid)
        for name in runs:
            if name not in names:
                names.append(name)
        for index, name in enumerate(names):
            state = runs.get(name)
            thumbnail = (
                thumbnail_for(experiment, grid, state) if state is not None else None
            )
            tile = _OverviewTile(name, state, thumbnail)
            row, column = divmod(index, _TILES_PER_LINE)
            self._tiles_layout.addWidget(tile, row, column)
            self._tiles.append(tile)
        self.tiles_widget.setVisible(bool(names))

        loads = [t for t in grid.task_history if t.name == _LOAD_ENTRY_NAME]
        if loads:
            load = loads[-1]
            if load.status is AutoLamellaTaskStatus.Completed:
                text = f"Loaded {_when(load)} · {load.status_message}"
                if load.duration:
                    text += f" · {load.duration:.0f} s"
            else:
                text = f"Load {load.status.name.lower()} {_when(load)} · {load.status_message}"
            self.load_label.setText(text)
        else:
            self.load_label.setText("Not loaded by a run yet.")

        for state in grid.task_history:
            row = self.history.rowCount()
            self.history.insertRow(row)
            status = QTableWidgetItem(state.status.name)
            status.setForeground(
                Qt.GlobalColor.white
                if state.status not in _STATUS_COLOUR
                else _brush(_STATUS_COLOUR[state.status])
            )
            for column, item in enumerate(
                [
                    QTableWidgetItem(state.name),
                    QTableWidgetItem(_when(state)),
                    status,
                    QTableWidgetItem(self._details(state)),
                ]
            ):
                self.history.setItem(row, column, item)
        self.history.resizeRowsToContents()

    @staticmethod
    def _details(state: AutoLamellaTaskState) -> str:
        if state.status_message:
            return state.status_message
        roles = [r for r in state.outputs if not r.endswith("_thumbnail")]
        return ", ".join(roles)


def _brush(colour: str):
    from PyQt5.QtGui import QBrush, QColor

    return QBrush(QColor(colour))
