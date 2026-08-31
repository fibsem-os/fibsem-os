"""Tools → Agent Server: session status, token, and scope arming.

The durable policy (enable flag, watchdog minutes) lives in Preferences; this
dialog holds only what belongs to the running session. Arming is consent and
deliberately does not persist: every session starts read-only, and flipping a
scope here mutates the live ``AuthConfig`` (the token stays the same, so a
connected agent gains or loses the scope without being disconnected).

This dialog retires the ``FIBSEM_AGENT_ARM_CONTROL`` developer override as the
GUI path (the env var remains for headless/bench use).

Layout follows the project mockup: a status line, the token with show/copy,
and a scope ladder — each scope a row with its name, what it permits, and its
switch; hardware present but visibly not yet climbable.
"""

import logging
from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from fibsem.ui import stylesheets
from fibsem.ui.stylesheets import BORDER_STATE_COLOURS
from fibsem.ui.tokens import (
    BORDER_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

__all__ = ["AgentServerDialog"]

_AGENT_COLOR = BORDER_STATE_COLOURS["agent"]

_SECTION_STYLE = (
    f"color: {TEXT_MUTED_COLOR}; font-size: 10px; font-weight: bold; "
    "letter-spacing: 1px;"
)
_CARD_STYLE = (
    f"QFrame#scopeCard {{ background: {PANEL_COLOR}; "
    f"border: 1px solid {BORDER_COLOR}; border-radius: 5px; }}"
)
_TOKEN_STYLE = (
    f"QLineEdit {{ background: {PANEL_COLOR}; color: {TEXT_COLOR}; "
    f"border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 5px 8px; "
    "font-size: 11px; }"
)


def _section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(_SECTION_STYLE)
    return label


class _ScopeRow(QFrame):
    """One rung of the scope ladder: name, what it permits, and its switch."""

    def __init__(self, name: str, detail: str, accent: str, parent=None):
        super().__init__(parent)
        self.setObjectName("scopeCard")
        self.setStyleSheet(_CARD_STYLE)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 12px; border: none;"
        )
        self.detail_label = QLabel(detail)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; border: none;"
        )
        text_col.addWidget(self.name_label)
        text_col.addWidget(self.detail_label)
        row.addLayout(text_col, stretch=1)
        self.toggle = QCheckBox()
        self.toggle.setStyleSheet("border: none; background: transparent;")
        row.addWidget(self.toggle, alignment=Qt.AlignVCenter)


class AgentServerDialog(QDialog):
    """Live view and arming controls for the embedded agent server."""

    def __init__(self, host_provider: Callable[[], Optional[object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Server")
        self.setMinimumWidth(440)
        self._host_provider = host_provider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)

        # --- status ---------------------------------------------------------
        layout.addWidget(_section("Status"))
        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_label, stretch=1)
        layout.addLayout(status_row)

        layout.addSpacing(6)

        # --- token ----------------------------------------------------------
        layout.addWidget(_section("Session token"))
        token_row = QHBoxLayout()
        token_row.setSpacing(6)
        self._token_edit = QLineEdit()
        self._token_edit.setReadOnly(True)
        self._token_edit.setEchoMode(QLineEdit.Password)
        self._token_edit.setStyleSheet(_TOKEN_STYLE)
        # The system's fixed-width face, set in code: QSS font-family takes a
        # single family, which cannot be right cross-platform.
        from PyQt5.QtGui import QFontDatabase

        self._token_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._btn_reveal = QPushButton("Show")
        self._btn_reveal.setCheckable(True)
        self._btn_reveal.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self._btn_copy = QPushButton("Copy")
        self._btn_copy.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        token_row.addWidget(self._token_edit, stretch=1)
        token_row.addWidget(self._btn_reveal)
        token_row.addWidget(self._btn_copy)
        layout.addLayout(token_row)
        token_note = QLabel(
            "New each session. An agent on this machine finds it by itself; "
            "copy it only to connect one from elsewhere."
        )
        token_note.setWordWrap(True)
        token_note.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(token_note)

        layout.addSpacing(6)

        # --- the scope ladder ----------------------------------------------
        layout.addWidget(_section("What the agent may do"))
        self._read_row = _ScopeRow(
            "Observe",
            "Watch the workflow, read images and state. Always on.",
            OK_COLOR,
        )
        self._read_row.toggle.setChecked(True)
        self._read_row.toggle.setEnabled(False)
        layout.addWidget(self._read_row)

        self._control_row = _ScopeRow(
            "Act",
            "Answer questions, start, stop and change workflows. "
            "You can always override.",
            _AGENT_COLOR,
        )
        layout.addWidget(self._control_row)
        self._chk_control = self._control_row.toggle  # the arming switch

        self._hardware_row = _ScopeRow(
            "Command hardware",
            "Move the stage, acquire, mill. Not yet available.",
            TEXT_MUTED_COLOR,
        )
        self._hardware_row.toggle.setEnabled(False)
        self._chk_hardware = self._hardware_row.toggle
        layout.addWidget(self._hardware_row)

        # --- footer ---------------------------------------------------------
        note = QLabel(
            "Arming lasts this session only — it never persists. The enable "
            "switch and the watchdog live in Preferences → Agent."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(note)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)
        layout.addLayout(footer)

        self._btn_reveal.toggled.connect(self._on_reveal_toggled)
        self._btn_copy.clicked.connect(self._on_copy_clicked)
        self._chk_control.toggled.connect(self._on_control_toggled)

        self.refresh()

    # --- live state -----------------------------------------------------------

    def _host(self):
        host = self._host_provider()
        if host is None or not getattr(host, "running", False):
            return None
        return host

    def refresh(self) -> None:
        """Re-read the session's server into the dialog."""
        host = self._host()
        running = host is not None
        for widget in (self._token_edit, self._btn_reveal, self._btn_copy):
            widget.setEnabled(running)
        self._chk_control.setEnabled(running)
        if not running:
            self._status_dot.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._status_label.setText(
                "Not running — enable it in Preferences → Agent and connect "
                "a microscope."
            )
            self._status_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._token_edit.setText("")
            self._chk_control.blockSignals(True)
            self._chk_control.setChecked(False)
            self._chk_control.blockSignals(False)
            return
        from fibsem.server.auth import Scope

        self._status_dot.setStyleSheet(f"color: {OK_COLOR};")
        self._status_label.setText(f"Running on {host.url}")
        self._status_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        self._token_edit.setText(host.auth.token)
        # Reflect without re-arming: setChecked fires toggled, which would log
        # a no-op arm; block while mirroring.
        self._chk_control.blockSignals(True)
        self._chk_control.setChecked(host.auth.is_armed(Scope.CONTROL))
        self._chk_control.blockSignals(False)

    # --- actions --------------------------------------------------------------

    def _on_reveal_toggled(self, show: bool) -> None:
        self._token_edit.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password)
        self._btn_reveal.setText("Hide" if show else "Show")

    def _on_copy_clicked(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._token_edit.text())
            self._btn_copy.setText("Copied")
            self._btn_copy.setStyleSheet(stylesheets.CONFIRM_BUTTON_STYLESHEET)

    def _on_control_toggled(self, checked: bool) -> None:
        host = self._host()
        if host is None:
            return
        from fibsem.server.auth import Scope

        host.auth.set_armed(Scope.CONTROL, checked)
        logging.warning(
            "agent server: control scope %s via the Agent Server dialog",
            "ARMED" if checked else "disarmed",
        )
