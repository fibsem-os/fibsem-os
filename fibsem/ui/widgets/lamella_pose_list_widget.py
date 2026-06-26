from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Lamella
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton

_NAME_WIDTH = 110
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 2 + 8  # 2 buttons + 1 gap

# Preferred display order; poses not listed keep their insertion order after these.
_POSE_ORDER = ["MILLING", "FLUORESCENCE"]


class LamellaPoseRowWidget(QWidget):
    """A single pose row: name, pretty position, update and move-to buttons."""

    update_clicked = pyqtSignal(str)   # pose name
    move_to_clicked = pyqtSignal(str)  # pose name

    def __init__(self, pose_name: str, pretty: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.pose_name = pose_name
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_label = QLabel(pose_name.capitalize())
        self.name_label.setFixedWidth(_NAME_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.position_label = QLabel(pretty)
        self.position_label.setStyleSheet("background: transparent; color: #909090;")
        layout.addWidget(self.position_label, 1)

        self.btn_update = IconToolButton(
            icon="mdi:map-marker-check",
            tooltip="Update Position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_update)

        self.btn_move_to = IconToolButton(
            icon="mdi:crosshairs-gps",
            tooltip="Move to Position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_move_to)

        self.btn_update.clicked.connect(lambda: self.update_clicked.emit(self.pose_name))
        self.btn_move_to.clicked.connect(lambda: self.move_to_clicked.emit(self.pose_name))

    def set_pretty(self, pretty: str) -> None:
        self.position_label.setText(pretty)


class _LamellaPoseListHeader(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        name_header = QLabel("Pose")
        name_header.setFixedWidth(_NAME_WIDTH)
        name_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(name_header)

        position_header = QLabel("Position")
        position_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(position_header, 1)

        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)


class LamellaPoseListWidget(QWidget):
    """List widget displaying a Lamella's poses with name, position and actions.

    Self-contained: holds only the pose names of the lamella set via
    :meth:`set_lamella`. Emits :attr:`update_requested` / :attr:`move_to_requested`
    with the pose name when the row buttons are clicked.
    """

    update_requested = pyqtSignal(str)   # pose name
    move_to_requested = pyqtSignal(str)  # pose name

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _LamellaPoseListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Rebuild the rows from the lamella's existing poses."""
        self._list.clear()
        if lamella is None or not lamella.poses:
            return
        for pose_name in self._sorted_pose_names(lamella.poses):
            self._add_row(pose_name, self._pretty(lamella.poses[pose_name]))

    def refresh_pose(self, pose_name: str, pretty: str) -> None:
        """Update the displayed position for an existing pose row, in place.

        Avoids rebuilding the list so row selection/scroll state is preserved.
        No-op if no row matches *pose_name*.
        """
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, LamellaPoseRowWidget) and row.pose_name == pose_name:
                row.set_pretty(pretty)
                return

    def clear(self) -> None:
        self._list.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_pose_names(poses) -> list:
        """Order poses by _POSE_ORDER first; remaining keep insertion order after."""
        def key(name: str):
            try:
                return (0, _POSE_ORDER.index(name))
            except ValueError:
                return (1, 0)
        return sorted(poses.keys(), key=key)

    @staticmethod
    def _pretty(pose) -> str:
        if pose is not None and pose.stage_position is not None:
            return pose.stage_position.pretty
        return "Unknown"

    def _add_row(self, pose_name: str, pretty: str) -> LamellaPoseRowWidget:
        row = LamellaPoseRowWidget(pose_name, pretty)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        row.update_clicked.connect(self.update_requested)
        row.move_to_clicked.connect(self.move_to_requested)
        return row
