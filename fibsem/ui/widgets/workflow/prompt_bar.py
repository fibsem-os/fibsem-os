"""The prompt bar: the question a running workflow is asking, and the buttons
that decide it. Hidden when nothing is asked; idle guidance lives in the
window's status bar."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from fibsem.ui import stylesheets

# Positive buttons that start the hardware, styled as automated.
_RUN_BUTTONS = ("Run Milling", "Run Spot Burn")


class PromptBar(QWidget):
    # A button was pressed: True for the positive one. Both are disabled until
    # the next question, or `reopen`.
    answered = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.yes = QPushButton()
        self.no = QPushButton()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.message, 0, 0, 1, 2)
        layout.addWidget(self.yes, 1, 0)
        layout.addWidget(self.no, 1, 1)
        self.yes.clicked.connect(lambda: self._answer(True))
        self.no.clicked.connect(lambda: self._answer(False))
        self.ask("")

    def ask(
        self, msg: str = "", pos: Optional[str] = None, neg: Optional[str] = None
    ) -> None:
        """Show *msg* with a button for each of *pos* and *neg* given. An empty
        message is no question: the label goes rather than reserving a line."""
        self.message.setText(msg)
        self.message.setVisible(bool(msg))
        for button, text in ((self.yes, pos), (self.no, neg)):
            button.setText(text or "")
            button.setEnabled(text is not None)
            button.setVisible(text is not None)
        self.yes.setStyleSheet(
            stylesheets.SUPERVISION_STATUS_AUTOMATED_STYLESHEET
            if pos in _RUN_BUTTONS
            else stylesheets.PRIMARY_BUTTON_STYLESHEET
        )
        self.no.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

    def clear(self) -> None:
        self.ask("")

    def revise(self, msg: str, *, can_continue: bool) -> None:
        """Change the message of the question up, and whether it can be
        answered yet (an alignment area outside the image cannot)."""
        self.message.setText(msg)
        self.yes.setEnabled(can_continue)

    def reopen(self) -> None:
        """The answer did not land: the buttons shown can be pressed again."""
        self.yes.setEnabled(self.yes.isVisibleTo(self))
        self.no.setEnabled(self.no.isVisibleTo(self))

    def _answer(self, yes: bool) -> None:
        self.yes.setEnabled(False)
        self.no.setEnabled(False)
        self.answered.emit(yes)
