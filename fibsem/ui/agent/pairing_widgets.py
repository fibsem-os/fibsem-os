"""Pairing in the window: the button that shows a code, and the paired list
with Unpair, for one role (`fibsem.server.pairing`). Tools → App Server pairs
the operator's computers; agents pair from the web app's Settings."""

import time
from typing import Callable, Optional

from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from fibsem.server.pairing import OPERATOR
from fibsem.theme.tokens import TEXT_MUTED_COLOR, TEXT_STRONG_COLOR
from fibsem.ui.widgets.custom_widgets import IconToolButton

__all__ = ["PairingCode", "PairedList", "STOP_PAIRING_LABEL"]

STOP_PAIRING_LABEL = "Stop pairing"
_NOTE_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"


def _note(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(_NOTE_STYLE)
    return label


class PairingCode:
    """A button that opens pairing for *role*, and the code with its time left
    while it is open. The button goes in the caller's row; ``code_row`` below it.
    Pairing open for the other role shows nothing here, and a click replaces it."""

    def __init__(
        self,
        pairing: Callable[[], Optional[object]],
        label: str,
        tooltip: str,
        role: str = OPERATOR,
        changed: Callable[[], None] = lambda: None,
    ):
        self._pairing = pairing
        self._label = label
        self._role = role
        self._changed = changed
        self.button = QPushButton(label)
        self.button.setToolTip(tooltip)
        self.button.clicked.connect(self._on_clicked)
        self.code_row = QHBoxLayout()
        self.code = QLabel("")
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(18)
        font.setBold(True)
        font.setLetterSpacing(font.AbsoluteSpacing, 6)
        self.code.setFont(font)
        self.code.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        self.left = _note()
        self.code_row.addWidget(self.code)
        self.code_row.addWidget(self.left)
        self.code_row.addStretch(1)

    def is_open(self) -> bool:
        """Whether pairing is open for this role."""
        pairing = self._pairing()
        if pairing is None:
            return False
        state = pairing.state()
        return state["open"] and state["role"] == self._role

    def refresh(self) -> None:
        pairing = self._pairing()
        self.button.setEnabled(pairing is not None)
        state = pairing.state() if pairing is not None else {"open": False}
        mine = state["open"] and state["role"] == self._role
        self.code.setVisible(mine)
        self.left.setVisible(mine)
        if mine:
            left = int(state["expires_in_s"])
            self.code.setText(state["code"])
            self.left.setText(f"{left // 60}:{left % 60:02d} left")
            self.button.setText(STOP_PAIRING_LABEL)
        else:
            self.button.setText(self._label)

    def _on_clicked(self) -> None:
        pairing = self._pairing()
        if pairing is None:
            return
        if self.is_open():
            pairing.close()
        else:
            pairing.open(self._role)
        self._changed()


class PairedList(QWidget):
    """The computers paired as *role*: name, when, and Unpair."""

    def __init__(
        self,
        pairing: Callable[[], Optional[object]],
        role: str = OPERATOR,
        changed: Callable[[], None] = lambda: None,
        parent=None,
    ):
        super().__init__(parent)
        self._pairing = pairing
        self._role = role
        self._changed = changed
        self._rows = QVBoxLayout(self)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(2)
        # The paired computers as last drawn, so a refresh redraws only a change.
        self._shown: Optional[tuple] = None

    def refresh(self) -> None:
        pairing = self._pairing()
        computers = (
            tuple(c for c in pairing.tokens.computers() if c.role == self._role)
            if pairing is not None
            else ()
        )
        if computers != self._shown:
            self._shown = computers
            self._draw(computers)

    def _draw(self, computers) -> None:
        while self._rows.count():
            widget = self._rows.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for computer in computers:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(QLabel(computer.name), stretch=1)
            paired = time.strftime("%d %b", time.localtime(computer.paired_at))
            line.addWidget(_note(f"paired {paired}"))
            unpair = IconToolButton(
                icon="mdi:link-variant-off",
                tooltip=f"Unpair {computer.name}: its token stops working at once",
                size=22,
            )
            unpair.clicked.connect(
                lambda _=False, cid=computer.id: self._on_unpair(cid)
            )
            line.addWidget(unpair)
            self._rows.addWidget(row)

    def _on_unpair(self, computer_id: str) -> None:
        pairing = self._pairing()
        if pairing is not None:
            pairing.tokens.remove(computer_id)
        self._changed()
