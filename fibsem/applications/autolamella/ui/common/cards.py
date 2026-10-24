"""The card the Grids and Lamella strips are both made of: a framed record that
switches between the cozy tile, the standard row and the compact line, and the
container that holds the cards and keeps one selected.

The sizes are shared so the two strips line up.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QWidget

from fibsem.config import CARD_MODES, MODE_COZY
from fibsem.theme.tokens import PRIMARY_COLOR, SURFACE_COLOR

CARD_WIDTH = 300
THUMB_PADDING = 6  # inset from card edges so rounded corners stay visible
# 3:2, matching the 1536x1024 frames the microscope acquires, so the thumbnail
# scales rather than crops -- cached_thumbnail expands to fill. Kept small: at this
# size it is a scanning cue ("which of these looks milled"), and every pixel of
# width it takes comes off the status line, which is the part carrying detail.
THUMB_W = 66
THUMB_H = 44
# The cozy arrangement's thumbnail: the full card width less its padding, at the
# height the card carried before the denser layouts existed.
COZY_THUMB_W = CARD_WIDTH - 8 - THUMB_PADDING * 2
COZY_THUMB_H = 170
BTN_SIZE = 24

CARD_BUTTON_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:pressed { background: rgba(255, 255, 255, 15); }
QToolButton::menu-indicator { image: none; }
"""


def card_style(frame_name: str, selected: bool) -> str:
    """The card frame's sheet: a thin rule, or the accent border when selected."""
    border = f"2px solid {PRIMARY_COLOR}" if selected else "1px solid #3a3d42"
    return f"""
QFrame#{frame_name} {{
    background: {SURFACE_COLOR};
    border: {border};
    border-radius: 8px;
}}
"""


class RecordCard(QWidget):
    """A card for one record, in one of the CARD_MODES.

    The frame holds one content widget, which `_apply_layout` swaps. The three
    arrangements differ in geometry only -- same children, same signals, same
    `refresh()` -- so nothing outside `_apply_layout` knows which one is showing.
    """

    clicked = pyqtSignal(object)  # the record

    def __init__(
        self,
        record: Any,
        frame_name: str,
        mode: str = MODE_COZY,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.record = record
        self._mode = mode if mode in CARD_MODES else MODE_COZY
        self.setFixedWidth(CARD_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        self._card = QFrame()
        self._card.setObjectName(frame_name)
        self._card.setStyleSheet(card_style(frame_name, False))
        self._card.setFixedWidth(CARD_WIDTH - 8)
        self._frame_layout = QVBoxLayout(self._card)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(0)
        self._content: Optional[QWidget] = None
        outer.addWidget(self._card)

    def set_mode(self, mode: str) -> None:
        """Switch arrangement: cozy tile, standard row, or compact one-liner.

        Cheap enough to call on every card at once: it rebuilds one container widget
        and re-parents the existing children into it. Nothing is recreated, so the
        menus, connections and selection state all survive the switch.
        """
        if mode not in CARD_MODES or mode == self._mode:
            return
        self._mode = mode
        self._apply_layout()
        self.refresh()  # the thumbnail is drawn at the new size

    def mode(self) -> str:
        return self._mode

    def _apply_layout(self) -> None:
        """(Re)build the card's content in the current arrangement."""
        raise NotImplementedError

    def refresh(self) -> None:
        raise NotImplementedError

    def set_selected(self, selected: bool) -> None:
        self._card.setStyleSheet(card_style(self._card.objectName(), selected))

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.record)
        super().mousePressEvent(event)


class CardContainer(QWidget):
    """Holds the cards by record id, all in one mode, at most one selected."""

    def __init__(self, parent: Optional[QWidget] = None, mode: str = MODE_COZY) -> None:
        super().__init__(parent)
        self._cards: Dict[str, RecordCard] = {}  # record id -> card
        self._selected_id: Optional[str] = None
        self._mode = mode if mode in CARD_MODES else MODE_COZY

    def set_mode(self, mode: str) -> None:
        """Switch every card, and any added later, to one of the CARD_MODES."""
        if mode not in CARD_MODES:
            return
        self._mode = mode
        for card in self._cards.values():
            card.set_mode(mode)

    def mode(self) -> str:
        return self._mode

    def refresh_all(self) -> None:
        for card in self._cards.values():
            card.refresh()

    def clear(self) -> None:
        for card in self._cards.values():
            self.layout().removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        # Ids are stable and the host clears then re-adds the same set, so a
        # surviving id would name a card drawn unselected while this thinks it is
        # selected, and the next click on it would toggle off rather than select.
        self._selected_id = None
