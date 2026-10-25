"""Tools → Host: the server itself -- status, services, the operator's
token and paired computers, and a link to the event log.

The App Server is the API inside AutoLamella that the agent, the event log and
``FibsemClient`` reach. This dialog is about the server, not the agent:
what the agent may do is its level, on the chip in the status bar,
and connecting one is the web app's (Agent page, Settings).

Whether it runs, and whether it serves the microscope, are both set here and
saved: they are the operator's standing choice, so the next launch starts the
same way. The agent's level is not: it resets to Watch at every launch.
"""

from typing import Callable, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from fibsem.server.hosting import event_log_url
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    OK_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui import stylesheets
from fibsem.ui.agent.level_chip import last_heard_text
from fibsem.ui.agent.pairing_widgets import PairedList, PairingCode

__all__ = ["AppServerDialog"]

# A change to either switch starts, stops or restarts the server, so it applies
# once the switches have been still this long: a double click restarts nothing.
SETTLE_MS = 600

ENABLED_LABEL = "Serve this session"
ENABLED_TIP = (
    "Serve this session, with or without a microscope connected: a local, "
    "token-protected API over this session that an AI agent (through the fibsem-mcp sidecar) and "
    "the event log can reach. Kept for the next launch. What the agent may do "
    "is its level, set from the Agent chip in the status bar, which resets to "
    "Watch at every launch."
)
MICROSCOPE_LABEL = "Serve the microscope"
MICROSCOPE_TIP = (
    "Also serve the routes that reach the instrument: its devices, stage, beams "
    "and milling state, reads included. Off, clients see the session but cannot "
    "query the microscope. Changing this restarts the host's server; a "
    "connected agent carries on. Kept for the next launch."
)

PAIR_LABEL = "Pair a computer"
COPY_TOKEN_LABEL = "Copy token"

_LINK_STYLE = (
    f"QPushButton {{ background: transparent; color: {ACCENT_COLOR}; border: none; "
    f"padding: 0px; }} QPushButton:disabled {{ color: {TEXT_MUTED_COLOR}; }}"
)

_SECTION_STYLE = (
    f"color: {TEXT_MUTED_COLOR}; font-size: 10px; font-weight: bold; "
    "letter-spacing: 1px;"
)
_NOTE_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"


def section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(_SECTION_STYLE)
    return label


def note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(_NOTE_STYLE)
    return label


def open_in_browser(url: str) -> None:
    """Open *url* in the system browser."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl(url))


class AppServerDialog(QDialog):
    """Live view of the embedded App Server."""

    def __init__(
        self,
        host_provider: Callable[[], Optional[object]],
        parent=None,
        set_enabled: Optional[Callable[[bool], None]] = None,
        set_microscope: Optional[Callable[[bool], None]] = None,
        enabled: bool = False,
        serve_microscope: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Host")
        self.setMinimumWidth(440)
        self._host_provider = host_provider
        # Each saves the operator's choice and starts, stops or restarts the server.
        self._set_enabled = set_enabled
        self._set_microscope = set_microscope
        # What was last applied, so a switch put back where it was does nothing.
        self._applied = (bool(enabled), bool(serve_microscope))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)

        # --- server ---------------------------------------------------------
        layout.addWidget(section("Server"))
        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._btn_event_log = QPushButton("Event log")
        self._btn_event_log.setStyleSheet(_LINK_STYLE)
        self._btn_event_log.setCursor(Qt.PointingHandCursor)
        self._btn_event_log.setToolTip(
            "Open the event log in your browser: what the agent, the operator "
            "and the run did. It reads the session and can't act on it."
        )
        self._btn_event_log.clicked.connect(self._on_event_log_clicked)
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_label, stretch=1)
        status_row.addWidget(self._btn_event_log)
        layout.addLayout(status_row)
        self._services_label = note("")
        layout.addWidget(self._services_label)
        self._chk_enabled = QCheckBox(ENABLED_LABEL)
        self._chk_enabled.setToolTip(ENABLED_TIP)
        self._chk_enabled.setChecked(bool(enabled))
        self._chk_enabled.setVisible(set_enabled is not None)
        self._chk_microscope = QCheckBox(MICROSCOPE_LABEL)
        self._chk_microscope.setToolTip(MICROSCOPE_TIP)
        self._chk_microscope.setChecked(bool(serve_microscope))
        self._chk_microscope.setVisible(set_microscope is not None)
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SETTLE_MS)
        self._settle_timer.timeout.connect(self._apply_switches)
        switches = QHBoxLayout()
        for box in (self._chk_enabled, self._chk_microscope):
            box.toggled.connect(self._on_switch_toggled)
            switches.addWidget(box)
        switches.addStretch(1)
        layout.addLayout(switches)

        layout.addSpacing(6)

        # --- agent ------------------------------------------------------------
        layout.addWidget(section("Agent"))
        self._agent_label = QLabel("")
        self._agent_label.setWordWrap(True)
        self._agent_label.setToolTip(
            "Set what the agent may do, or connect one, from the Agent chip in "
            "the status bar."
        )
        layout.addWidget(self._agent_label)

        layout.addSpacing(6)

        # --- operator ---------------------------------------------------------
        layout.addWidget(section("Operator"))
        operator_row = QHBoxLayout()
        self._btn_copy_operator = QPushButton(COPY_TOKEN_LABEL)
        self._btn_copy_operator.setToolTip(
            "Your own token, for the web app and your scripts: it acts as you. "
            "Kept between launches."
        )
        self._btn_copy_operator.clicked.connect(self._on_copy_operator_clicked)
        self._btn_new_operator = QPushButton("New token")
        self._btn_new_operator.setToolTip(
            "Replace the operator token: the old one stops working at once."
        )
        self._btn_new_operator.clicked.connect(self._on_new_operator_clicked)
        operator_row.addWidget(self._btn_copy_operator)
        operator_row.addWidget(self._btn_new_operator)
        operator_row.addStretch(1)
        self._pair = PairingCode(
            self._pairing,
            PAIR_LABEL,
            "Show a code to type on another computer or in the web app: it "
            "gets a token of its own and acts as you. Works once, for 5 minutes.",
            changed=self.refresh,
        )
        self._btn_pair = self._pair.button
        operator_row.addWidget(self._btn_pair)
        layout.addLayout(operator_row)
        layout.addLayout(self._pair.code_row)
        self._computers = PairedList(self._pairing, changed=self.refresh)
        layout.addWidget(self._computers)

        layout.addSpacing(6)

        # --- footer ---------------------------------------------------------
        footer = QHBoxLayout()
        footer.addStretch(1)
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)
        layout.addLayout(footer)

        # Keep "last heard" and a pairing code's time ticking while open.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self.refresh)
        self._status_timer.start()

        self.refresh()

    # --- live state -----------------------------------------------------------

    def _host(self):
        host = self._host_provider()
        if host is None or not getattr(host, "running", False):
            return None
        return host

    def _not_running_text(self) -> str:
        if not self._chk_enabled.isChecked():
            return "Not running."
        error = getattr(self._host_provider(), "start_error", None)
        if error:
            return f"Not running: it needs {error}."
        return "Not running. It did not start; see the log."

    @staticmethod
    def _status_text(host) -> str:
        return f"Running on {host.url}"

    @staticmethod
    def services_text(host) -> str:
        served = "on" if getattr(host, "serve_microscope", False) else "off"
        return f"Services: monitoring · microscope {served}"

    @staticmethod
    def agent_text(host) -> str:
        """When the agent's token was last used, in units read at a glance."""
        try:
            age = host.agent_seconds_since_seen()
        except Exception:
            age = None
        try:
            client = host.agent_client()
        except Exception:
            client = None
        return last_heard_text(age, client)

    def refresh(self) -> None:
        """Re-read the session's server into the dialog."""
        host = self._host()
        running = host is not None
        self._btn_event_log.setEnabled(running)
        self._btn_copy_operator.setEnabled(running)
        self._btn_new_operator.setEnabled(running)
        self._pair.refresh()
        self._computers.refresh()
        # Nothing to serve while the server is off.
        self._chk_microscope.setEnabled(
            self._chk_enabled.isChecked() or self._set_enabled is None
        )
        if not running:
            self._status_dot.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._status_label.setText(self._not_running_text())
            self._status_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._services_label.setText("")
            self._agent_label.setText("No clients while the host is not serving.")
            self._agent_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            return
        self._status_dot.setStyleSheet(f"color: {OK_COLOR};")
        self._status_label.setText(self._status_text(host))
        self._status_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        self._services_label.setText(self.services_text(host))
        self._agent_label.setText(self.agent_text(host))
        self._agent_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")

    # --- actions --------------------------------------------------------------

    def _on_switch_toggled(self, _checked: bool) -> None:
        self._chk_microscope.setEnabled(
            self._chk_enabled.isChecked() or self._set_enabled is None
        )
        self._settle_timer.start()  # restarts the wait on each toggle

    def _apply_switches(self) -> None:
        """Apply the switches as they stand; one put back where it was does nothing.

        The microscope first, so turning both on starts the server once,
        already serving it.
        """
        self._settle_timer.stop()
        enabled = self._chk_enabled.isChecked()
        serve = self._chk_microscope.isChecked()
        was_enabled, was_serving = self._applied
        if self._set_microscope is not None and serve != was_serving:
            self._set_microscope(serve)
        if self._set_enabled is not None and enabled != was_enabled:
            self._set_enabled(enabled)
        self._applied = (enabled, serve)
        self.refresh()

    def hideEvent(self, event) -> None:
        # Closing the dialog applies a change that is still waiting.
        if self._settle_timer.isActive():
            self._apply_switches()
        super().hideEvent(event)

    def _on_event_log_clicked(self) -> None:
        host = self._host()
        if host is None:
            return
        open_in_browser(event_log_url(host))

    def _copy_operator_token(self, token: Optional[str], copied: str) -> None:
        clipboard = QApplication.clipboard()
        if not token or clipboard is None:
            return
        clipboard.setText(token)
        self._btn_copy_operator.setText(copied)
        QTimer.singleShot(1500, self._restore_copy_operator)

    def _on_copy_operator_clicked(self) -> None:
        host = self._host()
        if host is None:
            return
        self._copy_operator_token(host.auth.operator_token, "Copied")

    def _on_new_operator_clicked(self) -> None:
        host = self._host()
        if host is None:
            return
        self._copy_operator_token(host.new_operator_token(), "New token copied")

    def _pairing(self):
        host = self._host()
        return getattr(getattr(host, "auth", None), "pairing", None)

    def _restore_copy_operator(self) -> None:
        self._btn_copy_operator.setText(COPY_TOKEN_LABEL)
