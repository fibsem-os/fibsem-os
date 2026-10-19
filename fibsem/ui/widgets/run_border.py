"""The coloured frame round a window's content that says what the run is doing.

Drawn by the AutoLamella main window and the coincidence viewer. The colours and
their QSS come from ``stylesheets.border_stylesheet``, applied by the window that
holds the frame.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QWidget

from fibsem.ui.stylesheets import shown_border_state


def run_border_state(
    current: str, *, held: bool, pending: bool, running: Optional[str]
) -> str:
    """The border for the run as it is now. A stop stays red until the run has
    unwound; then a hold, a wait between tasks, the running task's state
    (supervised or automated), else idle."""
    if current == "stopping":
        return current
    if held:
        return "waiting"
    if pending:
        return "pending"
    return running or "idle"


class RunBorder(QFrame):
    """A frame whose border shows ``state``: the agent's purple while it drives,
    and nothing while the operator has the border turned off."""

    def __init__(self, object_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("borderState", "idle")
        self.state = "idle"
        # The operator's Show Workflow Border switch.
        self.on = True
        self.agent_driving = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def set_content(self, widget: QWidget) -> None:
        self.layout().addWidget(widget)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self._paint()

    def set_on(self, on: bool) -> None:
        self.on = on
        self._paint()

    def set_agent_driving(self, driving: bool) -> None:
        if driving != self.agent_driving:
            self.agent_driving = driving
            self._paint()

    @property
    def shown(self) -> str:
        """The state painted now."""
        return self.property("borderState")

    def _paint(self) -> None:
        state = shown_border_state(self.state, self.agent_driving)
        self.setProperty("borderState", state if self.on else "idle")
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()
