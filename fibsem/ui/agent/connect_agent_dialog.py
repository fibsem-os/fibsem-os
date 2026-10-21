"""Connect an agent: everything needed to point an agent at this App Server.

Opened from the agent chip's menu (FIB-1035): the agent's page, separate from
Tools → App Server, which is about the server. It holds the sidecar's absolute
path and one "Copy for your agent" text: pasted into an agent, it registers the
sidecar, checks the connection and reports its level. An agent on another
computer pairs with a code (``fibsem-mcp pair``) and gets a token of its own;
the agent token stays for scripts. The level is shown, not set: the chip is the
one place it changes.
"""

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.agent_setup import agent_setup_text, sidecar_path
from fibsem.server.discovery import dev_checkout
from fibsem.server.pairing import AGENT
from fibsem.ui import stylesheets
from fibsem.ui.agent.app_server_dialog import note, section
from fibsem.ui.agent.level_chip import LEVEL_WORDS
from fibsem.ui.agent.pairing_widgets import PairedList, PairingCode
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    GRAY_ICON_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton

__all__ = ["ConnectAgentDialog"]

_FIELD_STYLE = (
    f"QLineEdit {{ background: {PANEL_COLOR}; color: {TEXT_COLOR}; "
    f"border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 5px 8px; "
    "font-size: 11px; }"
)


def _field(text: str = "", secret: bool = False) -> QLineEdit:
    field = QLineEdit(text)
    field.setReadOnly(True)
    field.setStyleSheet(_FIELD_STYLE)
    # The system's fixed-width face, set in code: QSS font-family takes a single
    # family, which cannot be right cross-platform.
    field.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    if secret:
        field.setEchoMode(QLineEdit.Password)
    return field


class ConnectAgentDialog(QDialog):
    """The agent token, the sidecar, and how to register it."""

    def __init__(
        self,
        host_provider: Callable[[], Optional[object]],
        parent=None,
        sidecar: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Connect an agent")
        self.setMinimumWidth(480)
        self._host_provider = host_provider
        self._sidecar = sidecar if sidecar is not None else sidecar_path()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)

        # --- the level ------------------------------------------------------
        layout.addWidget(section("Agent"))
        self._level_label = QLabel("")
        layout.addWidget(self._level_label)
        layout.addWidget(
            note(
                "Change the level from the Agent chip in the status bar. "
                "“Agent” is any client holding an agent token."
            )
        )

        layout.addSpacing(6)

        # --- the sidecar ------------------------------------------------------
        layout.addWidget(section("Register the sidecar"))
        self._path_edit = _field(str(self._sidecar) if self._sidecar else "")
        layout.addWidget(self._path_edit)
        buttons = QHBoxLayout()
        self._btn_copy_setup = QPushButton("Copy for your agent")
        self._btn_copy_setup.clicked.connect(self._on_copy_setup)
        buttons.addWidget(self._btn_copy_setup)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        if self._sidecar is None:
            self._path_edit.setPlaceholderText("fibsem-mcp is not installed here")
            self._btn_copy_setup.setEnabled(False)
            layout.addWidget(
                note("Install it into this environment with: pip install fibsem[mcp]")
            )
        layout.addWidget(
            note(
                "Paste it into Claude Code: the agent registers the sidecar, checks "
                "the connection and tells you its level. On this machine the "
                "sidecar finds the App Server by itself, through "
                "~/.fibsem/app-server.json, and finds it again after a restart."
            )
        )

        layout.addSpacing(6)

        # --- another computer -----------------------------------------------
        layout.addWidget(section("Another computer"))
        pair_row = QHBoxLayout()
        self._pair = PairingCode(
            self._pairing,
            "Pair an agent",
            "Show a code for an agent on another computer: it gets a token of "
            "its own, gated by the level. Works once, for 5 minutes.",
            role=AGENT,
            changed=self.refresh,
        )
        pair_row.addWidget(self._pair.button)
        pair_row.addStretch(1)
        layout.addLayout(pair_row)
        layout.addLayout(self._pair.code_row)
        self._pair_command = note("")
        self._pair_command.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._pair_command)
        self._agents = PairedList(self._pairing, role=AGENT, changed=self.refresh)
        layout.addWidget(self._agents)

        layout.addSpacing(6)

        # --- the token --------------------------------------------------------
        layout.addWidget(section("Agent token"))
        token_row = QHBoxLayout()
        token_row.setSpacing(6)
        self._token_edit = _field(secret=True)
        # checked-state swap is the widget's own machinery — icons and colours
        # both, so a toggle can never drop to the black default.
        self._btn_reveal = IconToolButton(
            icon="mdi:eye-outline",
            checked_icon="mdi:eye-off-outline",
            tooltip="Show the token",
            checked_tooltip="Hide the token",
            checkable=True,
            size=26,
        )
        self._btn_copy_token = IconToolButton(
            icon="mdi:content-copy", tooltip="Copy the token", size=26
        )
        token_row.addWidget(self._token_edit, stretch=1)
        token_row.addWidget(self._btn_reveal)
        token_row.addWidget(self._btn_copy_token)
        layout.addLayout(token_row)
        layout.addWidget(note("For scripts. New each time AutoLamella starts."))

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)
        layout.addLayout(footer)

        self._btn_reveal.toggled.connect(self._on_reveal_toggled)
        self._btn_copy_token.clicked.connect(self._on_copy_token)
        # A pairing code's time left ticks while open.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _host(self):
        host = self._host_provider()
        if host is None or not getattr(host, "running", False):
            return None
        return host

    def _pairing(self):
        return getattr(getattr(self._host(), "auth", None), "pairing", None)

    def refresh(self) -> None:
        self._pair.refresh()
        self._agents.refresh()
        code = self._pair.code.text() if self._pair.is_open() else ""
        self._pair_command.setVisible(bool(code))
        self._pair_command.setText(f"On that computer: fibsem-mcp pair {code}")
        host = self._host()
        auth = getattr(host, "auth", None)
        if auth is None:
            self._level_label.setText("The App Server isn't running.")
            self._token_edit.setText("")
            self._token_edit.setEnabled(False)
            self._btn_reveal.setEnabled(False)
            self._btn_copy_token.setEnabled(False)
            return
        level = auth.level or AgentLevel.WATCH
        self._level_label.setText(LEVEL_WORDS[level])
        self._token_edit.setText(auth.token)
        for widget in (self._token_edit, self._btn_reveal, self._btn_copy_token):
            widget.setEnabled(True)

    # --- actions --------------------------------------------------------------

    @staticmethod
    def _copy(text: str) -> bool:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        clipboard.setText(text)
        return True

    def _on_copy_setup(self) -> None:
        if self._sidecar is None:
            return
        host = self._host()
        self._copy(
            agent_setup_text(
                self._sidecar, getattr(host, "url", None), pythonpath=dev_checkout()
            )
        )

    def _on_reveal_toggled(self, show: bool) -> None:
        self._token_edit.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password)

    def _on_copy_token(self) -> None:
        if not self._copy(self._token_edit.text()):
            return
        # A moment of confirmation, then back — the tooltip carries the words.
        self._btn_copy_token.setIcon(fibsem_icon("mdi:check", color=OK_COLOR))
        self._btn_copy_token.setToolTip("Copied")
        QTimer.singleShot(1500, self._restore_copy_button)

    def _restore_copy_button(self) -> None:
        self._btn_copy_token.setIcon(
            fibsem_icon("mdi:content-copy", color=GRAY_ICON_COLOR)
        )
        self._btn_copy_token.setToolTip("Copy the token")
