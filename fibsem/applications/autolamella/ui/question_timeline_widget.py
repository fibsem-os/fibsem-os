"""Who answered what: the latest supervision answer, under the interaction area.

A dumb render of the responder's question-lifecycle feed — the single most
recent answered (or withdrawn) question on one line, with the last few kept
as the line's tooltip. One line rather than a stack: the glance question is
"did the agent just act?", and a growing list under the buttons is noise
(operator feedback from the first live look). The durable history is the
experiment logfile and the event stream, not this label.

It subscribes to the same single code path that applies answers, so it can
never show something that didn't happen; there is no "update the timeline"
step to forget.

User-facing wording rule: who · what · when. Wire vocabulary (nonces, request
type names) stays off the screen.
"""

import time
from collections import deque
from typing import Deque, Dict, Tuple

from PyQt5.QtCore import Qt
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
_AGENT_COLOR = BORDER_STATE_COLOURS["agent"]


class QuestionTimelineWidget(QWidget):
    """The most recent supervision answer; recent ones on hover. GUI thread only."""

    HISTORY = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(0)
        self._label = QLabel("")
        self._label.setTextFormat(Qt.RichText)  # only "agent" is coloured
        self._label.setStyleSheet(_ROW_STYLESHEET)
        layout.addWidget(self._label)
        self._current = ""
        self._history: Deque[Tuple[str, bool]] = deque(maxlen=self.HISTORY)
        # Nothing to show until a question is answered; an empty strip is noise.
        self.setVisible(False)

    def record(self, kind: str, payload: Dict) -> None:
        """One lifecycle event in; the line updates. Ignores prompt_raised —
        the standing question already has the whole instruction label."""
        if kind == "prompt_answered":
            who = "agent" if payload.get("answered_by") == "agent" else "you"
            verb = "answered" if payload.get("response") else "declined"
            label = _QUESTION_LABELS.get(payload.get("type"), payload.get("type"))
            self._show(f"{who} · {verb} {label} · {self._now()}", who == "agent")
        elif kind == "prompt_cancelled":
            self._show(f"question withdrawn · {self._now()}", False)

    def _now(self) -> str:
        return time.strftime("%H:%M:%S")

    def _show(self, text: str, is_agent: bool) -> None:
        self._history.appendleft((text, is_agent))
        self._current = text
        if is_agent:
            # Just the who is purple; the rest of the line stays quiet.
            self._label.setText(
                text.replace(
                    "agent", f'<span style="color:{_AGENT_COLOR}">agent</span>', 1
                )
            )
        else:
            self._label.setText(text)
        # The recent trail lives in the tooltip, newest first.
        self._label.setToolTip("\n".join(row for row, _ in self._history))
        self.setVisible(True)

    def rows(self):
        """The remembered rows, newest first (for tests, tooling — and the
        tooltip)."""
        return [row for row, _ in self._history]

    def current_text(self) -> str:
        """The one visible line, as plain text."""
        return self._current
