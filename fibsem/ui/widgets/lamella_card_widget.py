from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.applications.autolamella.structures import DefectState, Lamella
from fibsem.ui.widgets.lamella_list_widget import _defect_icon, _status_text

_CARD_WIDTH = 300
_THUMB_PADDING = 6        # inset from card edges so rounded corners stay visible
_THUMB_HEIGHT = 170
_BTN_SIZE = 24

_CARD_STYLE = """
QFrame#LamellaCard {
    background: #2b2d31;
    border: 1px solid #3a3d42;
    border-radius: 8px;
}
"""

_CARD_SELECTED_STYLE = """
QFrame#LamellaCard {
    background: #2b2d31;
    border: 2px solid #007ACC;
    border-radius: 8px;
}
"""

_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
"""


def _arr_to_pixmap(arr: np.ndarray, w: int, h: int) -> QPixmap:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    ih, iw, c = arr.shape
    qimg = QImage(arr.data, iw, ih, iw * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


class LamellaCardWidget(QWidget):
    """Modern card-style widget for a single Lamella."""

    clicked = pyqtSignal(object)          # Lamella
    defect_changed = pyqtSignal(object)   # Lamella

    def __init__(self, lamella: Lamella, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self.setFixedWidth(_CARD_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # ── card frame ──────────────────────────────────────────────────
        self._card = QFrame()
        self._card.setObjectName("LamellaCard")
        self._card.setStyleSheet(_CARD_STYLE)
        self._card.setFixedWidth(_CARD_WIDTH - 8)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(_THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING, 0)
        card_layout.setSpacing(0)

        # ── thumbnail ───────────────────────────────────────────────────
        _thumb_w = _CARD_WIDTH - 8 - _THUMB_PADDING * 2
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(_thumb_w, _THUMB_HEIGHT)
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setStyleSheet("background: #1a1b1e; border-radius: 4px;")
        card_layout.addWidget(self._thumb_label)

        # ── divider ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        card_layout.addWidget(sep)

        # ── info section ─────────────────────────────────────────────────
        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 10)
        info_layout.setSpacing(3)

        # name row: label + defect icon
        name_row = QHBoxLayout()
        name_row.setSpacing(6)

        self._name_label = QLabel()
        self._name_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #e0e0e0; background: transparent;"
        )
        name_row.addWidget(self._name_label, 1)

        self._btn_defect = QToolButton()
        self._btn_defect.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_defect.setStyleSheet(_BTN_STYLE)
        self._btn_defect.clicked.connect(self._on_defect_clicked)
        name_row.addWidget(self._btn_defect)

        info_layout.addLayout(name_row)

        self._status_label = QLabel()
        self._status_label.setStyleSheet(
            "font-size: 11px; color: #909090; background: transparent;"
        )
        self._status_label.setWordWrap(True)
        info_layout.addWidget(self._status_label)

        card_layout.addWidget(info)
        outer.addWidget(self._card)

        # ── evented connections ──────────────────────────────────────────
        lamella.task_state.events.name.connect(self.refresh)    # type: ignore[union-attr]
        lamella.task_state.events.status.connect(self.refresh)  # type: ignore[union-attr]
        lamella.events.defect.connect(self.refresh)             # type: ignore[union-attr]

        self.refresh()

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read all display fields from the stored Lamella."""
        self._name_label.setText(self.lamella.name)

        icon_name, icon_color, tooltip = _defect_icon(self.lamella)
        self._btn_defect.setIcon(QIconifyIcon(icon_name, color=icon_color))
        self._btn_defect.setToolTip(tooltip)

        status_text, status_style = _status_text(self.lamella)
        self._status_label.setText(status_text or "—")
        self._status_label.setStyleSheet(
            f"font-size: 11px; background: transparent; {status_style}"
        )

        arr = self.lamella.get_thumbnail()
        self._thumb_label.setPixmap(
            _arr_to_pixmap(arr, _CARD_WIDTH - 8 - _THUMB_PADDING * 2, _THUMB_HEIGHT)
        )

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.lamella)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._card.setStyleSheet(
            _CARD_SELECTED_STYLE if selected else _CARD_STYLE
        )

    def _on_defect_clicked(self) -> None:
        menu = QMenu(self)
        action_none = menu.addAction(
            QIconifyIcon("mdi:check-circle", color="#4caf50"), "No defect"
        )
        action_rework = menu.addAction(
            QIconifyIcon("mdi:refresh-circle", color="#e8a020"), "Rework required"
        )
        action_failure = menu.addAction(
            QIconifyIcon("mdi:close-circle", color="#d04040"), "Failure"
        )

        chosen = menu.exec_(self._btn_defect.mapToGlobal(
            self._btn_defect.rect().bottomLeft()
        ))

        if chosen == action_none:
            self.lamella.defect = DefectState(has_defect=False)
        elif chosen == action_rework:
            self.lamella.defect = DefectState(has_defect=True, requires_rework=True)
        elif chosen == action_failure:
            self.lamella.defect = DefectState(has_defect=True, requires_rework=False)
        else:
            return

        self.refresh()
        self.defect_changed.emit(self.lamella)


_N_COLS = 4


class LamellaCardContainer(QWidget):
    """Grid container that displays LamellaCardWidget in 4 columns."""

    lamella_selected = pyqtSignal(object)  # Lamella | None
    defect_changed = pyqtSignal(object)   # Lamella

    def __init__(self, columns: int = _N_COLS, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cards: Dict[str, LamellaCardWidget] = {}   # lamella._id → card
        self._selected_id: Optional[str] = None
        self._n_cols: int = max(1, columns)

        self._grid = QGridLayout(self)
        self._grid.setSpacing(12)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_lamella(self, lamella: Lamella) -> LamellaCardWidget:
        card = LamellaCardWidget(lamella)
        card.defect_changed.connect(self.defect_changed)
        card.clicked.connect(self._on_card_clicked)
        self._cards[lamella._id] = card
        self._rebuild_grid()
        return card

    def remove_lamella(self, lamella: Lamella) -> None:
        card = self._cards.pop(lamella._id, None)
        if card is not None:
            card.deleteLater()
            self._rebuild_grid()

    def refresh_all(self) -> None:
        for card in self._cards.values():
            card.refresh()

    def clear(self) -> None:
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def set_columns(self, n: int) -> None:
        self._n_cols = max(1, n)
        self._rebuild_grid()

    def _on_card_clicked(self, lamella: Lamella) -> None:
        prev_id = self._selected_id
        new_id = lamella._id

        if prev_id and prev_id in self._cards:
            self._cards[prev_id].set_selected(False)

        if prev_id == new_id:
            # clicking the already-selected card deselects it
            self._selected_id = None
            self.lamella_selected.emit(None)
        else:
            self._selected_id = new_id
            self._cards[new_id].set_selected(True)
            self.lamella_selected.emit(lamella)

    def _rebuild_grid(self) -> None:
        # Remove all items from the layout without deleting widgets
        while self._grid.count():
            self._grid.takeAt(0)
        for i, card in enumerate(self._cards.values()):
            row, col = divmod(i, self._n_cols)
            self._grid.addWidget(card, row, col)
