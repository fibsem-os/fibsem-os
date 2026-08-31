"""Who answered what: the question timeline under the workflow interaction area.

A dumb render of the responder's question-lifecycle feed — one row per
answered (or withdrawn) question, newest first, capped. It subscribes to the
same single code path that applies answers, so it can never show something
that didn't happen; there is no "update the timeline" step to forget.

User-facing wording rule: who · what · when. Wire vocabulary (nonces, request
type names) stays off the screen.
"""

import time
from typing import Dict

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from fibsem.ui import tokens
from fibsem.ui.stylesheets import BORDER_STATE_COLOURS

__all__ = ["QuestionTimelineWidget"]

# Request type -> what the row calls it. Fallback is the type name itself,
# so a new question type degrades to jargon rather than to silence.
_QUESTION_LABELS = {
    "Confirm": "confirmation",
    "RunMillingTask": "Run Milling",
    "RunSpotBurn": "Run Spot Burn",
    "PickPOI": "point of interest",
    "EditAlignmentArea": "alignment area",
    "ConfirmDetection": "detection",
}

_ROW_STYLESHEET = (
    f"color: {tokens.TEXT_MUTED_COLOR}; font-size: 11px; "
    'font-family: "Menlo", "Consolas", monospace;'
)
_AGENT_ROW_STYLESHEET = (
    f"color: {BORDER_STATE_COLOURS['agent']}; font-size: 11px; "
    'font-family: "Menlo", "Consolas", monospace;'
)


class QuestionTimelineWidget(QWidget):
    """The last few supervision answers, newest first. GUI thread only."""

    MAX_ROWS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 0)
        self._layout.setSpacing(1)
        # Nothing to show until a question is answered; an empty strip is noise.
        self.setVisible(False)

    def record(self, kind: str, payload: Dict) -> None:
        """One lifecycle event in; zero or one row out. Ignores prompt_raised —
        the standing question already has the whole instruction label."""
        if kind == "prompt_answered":
            who = "agent" if payload.get("answered_by") == "agent" else "you"
            verb = "answered" if payload.get("response") else "declined"
            label = _QUESTION_LABELS.get(payload.get("type"), payload.get("type"))
            self._add_row(f"{who} · {verb} {label} · {self._now()}", who == "agent")
        elif kind == "prompt_cancelled":
            self._add_row(f"question withdrawn · {self._now()}", False)

    def _now(self) -> str:
        return time.strftime("%H:%M:%S")

    def _add_row(self, text: str, is_agent: bool) -> None:
        row = QLabel(text)
        row.setStyleSheet(_AGENT_ROW_STYLESHEET if is_agent else _ROW_STYLESHEET)
        self._layout.insertWidget(0, row)
        while self._layout.count() > self.MAX_ROWS:
            item = self._layout.takeAt(self._layout.count() - 1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.setVisible(True)

    def rows(self):
        """The visible row texts, newest first (for tests and tooling)."""
        return [
            self._layout.itemAt(i).widget().text() for i in range(self._layout.count())
        ]
