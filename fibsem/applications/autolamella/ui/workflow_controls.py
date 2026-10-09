"""AutoLamella's workflow buttons, as one component (FIB-1188).

Attention Required, the supervision chip, and Run or Stop Workflow: the right of the
status bar, placed there with `FibsemStatusBar.add_action`. They were four buttons
built and restyled inside the main window. Here they are one widget with a small
vocabulary -- running or not, which supervision, what (if anything) needs attention --
and four signals for what was clicked.

What a click does stays the window's: stopping asks for confirmation, the supervision
chip toggles the current task's mode, Attention Required goes to whatever holds the
run. This only shows the state it is told and says what was pressed.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QWidget

from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    DANGER_BUTTON_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SUPERVISION_STATUS_AGENT_STYLESHEET,
    SUPERVISION_STATUS_AUTOMATED_STYLESHEET,
    SUPERVISION_STATUS_SUPERVISED_STYLESHEET,
    USER_ATTENTION_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import GRAY_ICON_COLOR

# The chip's three modes: its icon, its word, its colours, and what it says over the
# current task.
SUPERVISED = "supervised"
AUTOMATED = "automated"
AGENT = "agent"
_SUPERVISION = {
    AGENT: (
        "mdi:star-four-points",
        "Agent",
        SUPERVISION_STATUS_AGENT_STYLESHEET,
        "{task} is supervised by the connected agent. You can still answer any "
        "question first. Click to toggle supervision.",
    ),
    SUPERVISED: (
        "mdi:account-hard-hat",
        "Supervised",
        SUPERVISION_STATUS_SUPERVISED_STYLESHEET,
        "{task} is running in supervised mode. Your input will be required. "
        "Click to toggle.",
    ),
    AUTOMATED: (
        "mdi:lightning-bolt",
        "Automated",
        SUPERVISION_STATUS_AUTOMATED_STYLESHEET,
        "{task} is running in automated mode. Click to toggle.",
    ),
}


class WorkflowControls(QWidget):
    """Attention Required, the supervision chip, and Run or Stop Workflow."""

    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    supervision_clicked = pyqtSignal()
    attention_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Shown while the run waits on someone; the window says on what.
        self.attention_btn = QPushButton("Attention Required")
        self.attention_btn.setStyleSheet(USER_ATTENTION_BUTTON_STYLESHEET)
        self.attention_btn.setIcon(
            fibsem_icon("mdi:alert-circle", color=GRAY_ICON_COLOR)
        )
        self.attention_btn.setToolTip(
            "User Input Required - Click to go to Microscope tab"
        )
        self.attention_btn.clicked.connect(self.attention_clicked)

        # The current task's supervision while a run is going.
        self.supervision_btn = QPushButton("Supervised")
        self.supervision_btn.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.supervision_btn.setToolTip("Click to toggle supervision")
        self.supervision_btn.clicked.connect(self.supervision_clicked)

        self.run_btn = QPushButton("Run Workflow")
        self.run_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_btn.setIcon(fibsem_icon("mdi:play-circle", color=GRAY_ICON_COLOR))
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Run the AutoLamella workflow.")
        self.run_btn.clicked.connect(self.run_clicked)

        self.stop_btn = QPushButton("Stop Workflow")
        self.stop_btn.setStyleSheet(DANGER_BUTTON_STYLESHEET)
        self.stop_btn.setIcon(fibsem_icon("mdi:stop-circle", color=GRAY_ICON_COLOR))
        self.stop_btn.setToolTip(
            "Stop the current workflow. You will be asked to confirm."
        )
        self.stop_btn.clicked.connect(self.stop_clicked)

        for button in (
            self.attention_btn,
            self.supervision_btn,
            self.run_btn,
            self.stop_btn,
        ):
            layout.addWidget(button)
        self.attention_btn.hide()
        self.supervision_btn.hide()
        self.stop_btn.hide()

    def set_running(self, running: bool) -> None:
        """Stop while a run is going, Run otherwise. The supervision chip goes with
        the run; it comes back with the next task's `set_supervision`."""
        self.run_btn.setVisible(not running)
        self.stop_btn.setVisible(running)
        if not running:
            self.supervision_btn.hide()

    def set_run_enabled(self, enabled: bool, tooltip: str) -> None:
        """Whether Run can start what is selected, and why not when it cannot."""
        self.run_btn.setEnabled(enabled)
        self.run_btn.setToolTip(tooltip)

    def set_supervision(self, mode: str, task_name: str) -> None:
        """Show the chip for *task_name* in *mode*: `SUPERVISED`, `AUTOMATED` or
        `AGENT`."""
        icon, text, style, tooltip = _SUPERVISION[mode]
        self.supervision_btn.setIcon(fibsem_icon(icon, color="white"))
        self.supervision_btn.setText(text)
        self.supervision_btn.setToolTip(tooltip.format(task=task_name))
        self.supervision_btn.setStyleSheet(style)
        self.supervision_btn.show()

    def set_attention(self, label: Optional[str], tooltip: str = "") -> None:
        """Show Attention Required -- or *label*, "Review Required (2)" -- while the
        run waits on someone; None hides it."""
        if not label:
            self.attention_btn.hide()
            return
        self.attention_btn.setText(label)
        self.attention_btn.setToolTip(tooltip)
        self.attention_btn.show()
