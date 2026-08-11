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
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.ui.icon import fibsem_icon

from fibsem.applications.autolamella.structures import DefectState, DefectType, Lamella
from fibsem.applications.autolamella.ui.lamella_list_widget import _defect_icon, _status_text
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.ui.tokens import (
    NEUTRAL_200,
    NEUTRAL_550,
    NEUTRAL_900,
    PRIMARY_COLOR,
    SURFACE_COLOR,
)

_CARD_WIDTH = 300
_THUMB_PADDING = 6        # inset from card edges so rounded corners stay visible
# 3:2, matching the 1536x1024 frames the microscope acquires, so the thumbnail
# scales rather than crops -- _arr_to_pixmap expands to fill. Kept small: at this
# size it is a scanning cue ("which of these looks milled"), and every pixel of
# width it takes comes off the status line, which is the part carrying detail.
_THUMB_W = 66
_THUMB_H = 44
# The tile arrangement's thumbnail: the full card width less its padding, at the
# height the card carried before it was made compact.
_FULL_THUMB_W = _CARD_WIDTH - 8 - _THUMB_PADDING * 2
_FULL_THUMB_H = 170
_BTN_SIZE = 24

_CARD_STYLE = f"""
QFrame#LamellaCard {{
    background: {SURFACE_COLOR};
    border: 1px solid #3a3d42;
    border-radius: 8px;
}}
"""

_CARD_SELECTED_STYLE = f"""
QFrame#LamellaCard {{
    background: {SURFACE_COLOR};
    border: 2px solid {PRIMARY_COLOR};
    border-radius: 8px;
}}
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

    clicked = pyqtSignal(object)                    # Lamella
    defect_changed = pyqtSignal(object)             # Lamella
    move_to_requested = pyqtSignal(object)          # Lamella
    update_position_requested = pyqtSignal(object)  # Lamella
    remove_requested = pyqtSignal(object)           # Lamella

    def __init__(
        self,
        lamella: Lamella,
        compact: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self._compact = compact
        self.setFixedWidth(_CARD_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # ── card frame ──────────────────────────────────────────────────
        self._card = QFrame()
        self._card.setObjectName("LamellaCard")
        self._card.setStyleSheet(_CARD_STYLE)
        self._card.setFixedWidth(_CARD_WIDTH - 8)

        # The frame holds one content widget, which `_apply_layout` swaps. The two
        # arrangements differ in geometry only -- same children, same signals, same
        # refresh() -- so nothing below this line knows which one is showing, and
        # there is no second display path to keep correct.
        self._frame_layout = QVBoxLayout(self._card)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(0)
        self._content: Optional[QWidget] = None

        # ── thumbnail ───────────────────────────────────────────────────
        self._thumb_label = QLabel()
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setStyleSheet(f"background: {NEUTRAL_900}; border-radius: 4px;")

        self._name_label = QLabel()
        self._name_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {NEUTRAL_200}; background: transparent;"
        )

        self._btn_actions = QToolButton()
        self._btn_actions.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_actions.setStyleSheet(_BTN_STYLE + " QToolButton::menu-indicator { image: none; }")
        self._btn_actions.setIcon(fibsem_icon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR))
        self._btn_actions.setToolTip("Actions")
        self._btn_actions.setPopupMode(QToolButton.InstantPopup)
        _actions_menu = QMenu(self)
        self._action_move = _actions_menu.addAction(
            fibsem_icon("mdi:crosshairs-gps", color=stylesheets.GRAY_ICON_COLOR), "Move to Position"
        )
        self._action_update = _actions_menu.addAction(
            fibsem_icon("mdi:map-marker-check", color=stylesheets.GRAY_ICON_COLOR), "Update Position"
        )
        self._action_remove = _actions_menu.addAction(
            fibsem_icon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR), "Remove"
        )
        self._btn_actions.setMenu(_actions_menu)
        self._action_move.triggered.connect(lambda: self.move_to_requested.emit(self.lamella))
        self._action_update.triggered.connect(lambda: self.update_position_requested.emit(self.lamella))
        self._action_remove.triggered.connect(self._on_remove_clicked)

        self._btn_defect = QToolButton()
        self._btn_defect.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._btn_defect.setStyleSheet(_BTN_STYLE)
        self._btn_defect.clicked.connect(self._on_defect_clicked)

        # Elided, not wrapped: the status is a task name or a completion stamp, and
        # wrapping one in the compact row's narrow middle column would grow the row and
        # undo the density that layout exists for. ElidedLabel keeps the full string as
        # its tooltip, and its Ignored width policy stops a long name widening the card.
        # Used in both arrangements so the fixed height holds either way.
        self._status_label = ElidedLabel()
        self._status_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )

        outer.addWidget(self._card)
        self._apply_layout()

        # ── evented connections ──────────────────────────────────────────
        lamella.task_state.events.name.connect(self.refresh)    # type: ignore[union-attr]
        lamella.task_state.events.status.connect(self.refresh)  # type: ignore[union-attr]
        lamella.events.defect.connect(self.refresh)             # type: ignore[union-attr]
        lamella.events.description.connect(self.refresh)        # type: ignore[union-attr]

        self.refresh()

    # ------------------------------------------------------------------

    def set_compact(self, compact: bool) -> None:
        """Switch between the dense row and the large tile.

        Cheap enough to call on every card at once: it rebuilds one container widget
        and re-parents the existing children into it. Nothing is recreated, so the
        menus, connections and selection state all survive the switch.
        """
        if compact == self._compact:
            return
        self._compact = compact
        self._apply_layout()
        self.refresh()  # the thumbnail is drawn at the new size

    def _apply_layout(self) -> None:
        """(Re)build the card's content in the current arrangement."""
        if self._content is not None:
            # Re-parent the shared children out first, or deleting their container
            # takes them with it.
            for widget in (self._thumb_label, self._name_label,
                           self._status_label, self._btn_defect, self._btn_actions):
                widget.setParent(None)
            self._frame_layout.removeWidget(self._content)
            self._content.deleteLater()

        self._content = (self._build_compact() if self._compact else self._build_full())
        self._content.setStyleSheet("background: transparent;")
        self._frame_layout.addWidget(self._content)

        for widget in (self._thumb_label, self._name_label,
                       self._status_label, self._btn_defect, self._btn_actions):
            widget.show()

        # Qt caches size hints and only recomputes them when the layout is activated,
        # which for a card that has never been shown never happens -- so without this
        # a toggled card keeps reporting the height of the arrangement it replaced.
        self._frame_layout.invalidate()
        self._card.updateGeometry()
        self.updateGeometry()

    def _build_compact(self) -> QWidget:
        """Row: thumbnail | name over status | defect | actions.

        The strip is one column in a 340px-wide scroll area, so height is what limits
        how many lamellae are visible at once (FIB-585).
        """
        self._thumb_label.setFixedSize(_THUMB_W, _THUMB_H)

        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(_THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING)
        row.setSpacing(6)
        row.addWidget(self._thumb_label)

        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        info_layout.addStretch(1)
        info_layout.addWidget(self._name_label)
        info_layout.addWidget(self._status_label)
        info_layout.addStretch(1)

        row.addWidget(info, 1)
        row.addWidget(self._btn_defect, 0, Qt.AlignVCenter)
        row.addWidget(self._btn_actions, 0, Qt.AlignVCenter)
        return content

    def _build_full(self) -> QWidget:
        """Tile: large thumbnail over a name row and status — the original card."""
        self._thumb_label.setFixedSize(_FULL_THUMB_W, _FULL_THUMB_H)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(_THUMB_PADDING, _THUMB_PADDING, _THUMB_PADDING, 0)
        column.setSpacing(0)
        column.addWidget(self._thumb_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        column.addWidget(sep)

        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 10)
        info_layout.setSpacing(3)

        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_row.addWidget(self._name_label, 1)
        name_row.addWidget(self._btn_actions)
        name_row.addWidget(self._btn_defect)
        info_layout.addLayout(name_row)
        info_layout.addWidget(self._status_label)

        column.addWidget(info)
        return content

    @ensure_main_thread
    def refresh(self) -> None:
        """Re-read all display fields from the stored Lamella."""
        self._name_label.setText(self.lamella.name)
        self._card.setToolTip(self.lamella.description or "")

        icon_name, icon_color, tooltip = _defect_icon(self.lamella)
        self._btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self._btn_defect.setToolTip(tooltip)

        status_text, status_style = _status_text(self.lamella)
        self._status_label.setText(status_text or "—")
        self._status_label.setStyleSheet(
            f"font-size: 11px; background: transparent; {status_style}"
        )

        # Size read off the label rather than branched on the mode, so this method
        # stays the single display path whichever arrangement is showing.
        arr = self.lamella.get_thumbnail()
        self._thumb_label.setPixmap(
            _arr_to_pixmap(arr, self._thumb_label.width(), self._thumb_label.height())
        )

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.lamella)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._card.setStyleSheet(
            _CARD_SELECTED_STYLE if selected else _CARD_STYLE
        )

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove Lamella",
            f"Remove '{self.lamella.name}'?",
            QMessageBox.Yes,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_requested.emit(self.lamella)

    def _on_defect_clicked(self) -> None:
        menu = QMenu(self)
        action_none = menu.addAction(
            fibsem_icon("mdi:check-circle", color=stylesheets.GREEN_COLOR), "No defect"
        )
        action_rework = menu.addAction(
            fibsem_icon("mdi:refresh-circle", color=stylesheets.DEFECT_ORANGE_COLOR), "Rework required"
        )
        action_failure = menu.addAction(
            fibsem_icon("mdi:close-circle", color=stylesheets.DEFECT_RED_COLOR), "Failure"
        )

        chosen = menu.exec_(self._btn_defect.mapToGlobal(
            self._btn_defect.rect().bottomLeft()
        ))

        if chosen == action_none:
            self.lamella.defect = DefectState(state=DefectType.NONE)
        elif chosen == action_rework:
            self.lamella.defect = DefectState(state=DefectType.REWORK)
        elif chosen == action_failure:
            self.lamella.defect = DefectState(state=DefectType.FAILURE)
        else:
            return

        self.refresh()
        self.defect_changed.emit(self.lamella)


_N_COLS = 4


class LamellaCardContainer(QWidget):
    """Grid container that displays LamellaCardWidget in 4 columns."""

    lamella_selected = pyqtSignal(object)           # Lamella | None
    defect_changed = pyqtSignal(object)             # Lamella
    move_to_requested = pyqtSignal(object)          # Lamella
    update_position_requested = pyqtSignal(object)  # Lamella
    remove_requested = pyqtSignal(object)           # Lamella

    def __init__(
        self,
        columns: int = _N_COLS,
        parent: Optional[QWidget] = None,
        compact: bool = True,
    ) -> None:
        super().__init__(parent)
        self._cards: Dict[str, LamellaCardWidget] = {}   # lamella.id → card
        self._selected_id: Optional[str] = None
        self._n_cols: int = max(1, columns)
        self._compact: bool = compact

        self._grid = QGridLayout(self)
        self._grid.setSpacing(12)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_compact(self, compact: bool) -> None:
        """Switch every card, and any added later, between the row and the tile."""
        self._compact = compact
        for card in self._cards.values():
            card.set_compact(compact)

    def is_compact(self) -> bool:
        return self._compact

    def add_lamella(self, lamella: Lamella) -> LamellaCardWidget:
        card = LamellaCardWidget(lamella, compact=self._compact)
        card.defect_changed.connect(self.defect_changed)
        card.clicked.connect(self._on_card_clicked)
        card.move_to_requested.connect(self.move_to_requested)
        card.update_position_requested.connect(self.update_position_requested)
        card.remove_requested.connect(self._on_card_remove_requested)
        self._cards[lamella.id] = card
        self._rebuild_grid()
        return card

    def remove_lamella(self, lamella: Lamella) -> None:
        card = self._cards.pop(lamella.id, None)
        if card is not None:
            card.deleteLater()
            if self._selected_id == lamella.id:
                self._selected_id = None
            self._rebuild_grid()

    def refresh_lamella(self, lamella: Lamella) -> None:
        card = self._cards.get(lamella.id)
        if card is not None:
            card.refresh()

    def refresh_all(self) -> None:
        for card in self._cards.values():
            card.refresh()

    def clear(self) -> None:
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()
        # Lamella ids are stable, and the host clears then re-adds the same set on
        # every rebuild -- so a surviving id would name a card that is drawn
        # unselected while the container thinks it is selected, and the next click on
        # it would toggle off rather than select (FIB-578).
        self._selected_id = None

    def select_lamella(self, name: str) -> None:
        """Programmatically select the card matching name. Does NOT emit lamella_selected."""
        if self._selected_id and self._selected_id in self._cards:
            self._cards[self._selected_id].set_selected(False)
        for lid, card in self._cards.items():
            if card.lamella.name == name:
                self._selected_id = lid
                card.set_selected(True)
                return
        self._selected_id = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def set_columns(self, n: int) -> None:
        self._n_cols = max(1, n)
        self._rebuild_grid()

    def _on_card_clicked(self, lamella: Lamella) -> None:
        prev_id = self._selected_id
        new_id = lamella.id

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

    def _on_card_remove_requested(self, lamella: Lamella) -> None:
        self.remove_lamella(lamella)
        self.remove_requested.emit(lamella)

    def _rebuild_grid(self) -> None:
        # Remove all items from the layout without deleting widgets
        while self._grid.count():
            self._grid.takeAt(0)
        for i, card in enumerate(self._cards.values()):
            row, col = divmod(i, self._n_cols)
            self._grid.addWidget(card, row, col)
