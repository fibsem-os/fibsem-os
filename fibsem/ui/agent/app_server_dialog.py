"""Tools → App Server: the server itself -- status, services, paired computers,
the event log.

The App Server is the API inside AutoLamella that the agent, the event log and
``FibsemClient`` reach. This dialog is about the server, not the agent
(FIB-1035): what the agent may do is its level, on the chip in the status bar,
and connecting one is the chip's "Connect an agent…" (``connect_agent_dialog``).

Whether it runs, and whether it serves the microscope, are both set here and
saved: they are the operator's standing choice, so the next launch starts the
same way. The agent's level is not: it resets to Watch at every launch.
"""

import time
from typing import Callable, Optional

from PyQt5.QtCore import QTimer
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
from fibsem.ui import stylesheets
from fibsem.ui.agent.level_chip import last_heard_text
from fibsem.ui.tokens import OK_COLOR, TEXT_MUTED_COLOR, TEXT_STRONG_COLOR

__all__ = ["AppServerDialog"]

# A change to either switch starts, stops or restarts the server, so it applies
# once the switches have been still this long: a double click restarts nothing
# (FIB-1409).
SETTLE_MS = 600

ENABLED_LABEL = "Run the App Server"
ENABLED_TIP = (
    "Run the App Server while a microscope is connected: a local, token-protected "
    "API over this session that an AI agent (through the fibsem-mcp sidecar) and "
    "the event log can reach. Kept for the next launch. What the agent may do "
    "is its level, set from the Agent chip in the status bar, which resets to "
    "Watch at every launch."
)
MICROSCOPE_LABEL = "Serve the microscope"
MICROSCOPE_TIP = (
    "Also serve the routes that reach the instrument: its devices, stage, beams "
    "and milling state, reads included. Off, clients see the session but cannot "
    "query the microscope. Changing this restarts a running App Server; a "
    "connected agent carries on. Kept for the next launch."
)

PAIR_LABEL = "Pair a computer"
STOP_PAIRING_LABEL = "Stop pairing"

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
        self.setWindowTitle("App Server")
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

        # --- status ---------------------------------------------------------
        layout.addWidget(section("Status"))
        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_label, stretch=1)
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
        for box in (self._chk_enabled, self._chk_microscope):
            box.toggled.connect(self._on_switch_toggled)
            layout.addWidget(box)

        layout.addSpacing(6)

        # --- clients ----------------------------------------------------------
        layout.addWidget(section("Clients"))
        self._agent_label = QLabel("")
        self._agent_label.setWordWrap(True)
        layout.addWidget(self._agent_label)
        layout.addWidget(
            note(
                "Set what the agent may do from the Agent chip in the status bar; "
                "its menu also shows how to connect one."
            )
        )

        layout.addSpacing(6)

        # --- operator ---------------------------------------------------------
        layout.addWidget(section("Operator"))
        layout.addWidget(
            note(
                "Your own token, for the web page and your own scripts: it "
                "messages the agent, sets its level, stops it and decides its "
                "requests, as you. Kept between launches; a new one replaces it."
            )
        )
        operator_row = QHBoxLayout()
        operator_row.addStretch(1)
        self._btn_copy_operator = QPushButton("Copy operator token")
        self._btn_copy_operator.clicked.connect(self._on_copy_operator_clicked)
        self._btn_new_operator = QPushButton("New operator token")
        self._btn_new_operator.setToolTip(
            "Replace the operator token: the old one stops working at once."
        )
        self._btn_new_operator.clicked.connect(self._on_new_operator_clicked)
        operator_row.addWidget(self._btn_new_operator)
        operator_row.addWidget(self._btn_copy_operator)
        layout.addLayout(operator_row)

        layout.addSpacing(6)

        # --- computers --------------------------------------------------------
        layout.addWidget(section("Computers"))
        layout.addWidget(
            note(
                "Pair your own computer or a browser: type the code shown here on "
                "it. It acts as you, as the operator token does. Unpair one to "
                "cut it off."
            )
        )
        pair_row = QHBoxLayout()
        self._pair_code = QLabel("")
        self._pair_code.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 20px; font-weight: bold; "
            "letter-spacing: 4px;"
        )
        self._pair_left = note("")
        pair_row.addWidget(self._pair_code)
        pair_row.addWidget(self._pair_left, stretch=1)
        self._btn_pair = QPushButton(PAIR_LABEL)
        self._btn_pair.setToolTip(
            "Show a code to type on the other computer. It works once, for 5 minutes."
        )
        self._btn_pair.clicked.connect(self._on_pair_clicked)
        pair_row.addWidget(self._btn_pair)
        layout.addLayout(pair_row)
        self._computers = QVBoxLayout()
        self._computers.setSpacing(2)
        layout.addLayout(self._computers)
        # The paired computers as last drawn, so a refresh redraws only a change.
        self._shown_computers: Optional[tuple] = None

        layout.addSpacing(6)

        # --- event log --------------------------------------------------------
        layout.addWidget(section("Event log"))
        layout.addWidget(
            note(
                "A read-only page in your browser, with a token of its own: it "
                "reads the session and can't act on it. The agent's level "
                "doesn't affect it."
            )
        )
        log_row = QHBoxLayout()
        log_row.addStretch(1)
        self._btn_copy_link = QPushButton("Copy link")
        self._btn_copy_link.setToolTip("Copy the event log's address and token.")
        self._btn_copy_link.clicked.connect(self._on_copy_link_clicked)
        self._btn_event_log = QPushButton("Open Event Log")
        self._btn_event_log.setToolTip(
            "What the agent, the operator and the run did, filterable."
        )
        self._btn_event_log.clicked.connect(self._on_event_log_clicked)
        log_row.addWidget(self._btn_copy_link)
        log_row.addWidget(self._btn_event_log)
        layout.addLayout(log_row)

        layout.addSpacing(6)

        # --- footer ---------------------------------------------------------
        footer = QHBoxLayout()
        footer.addStretch(1)
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)
        layout.addLayout(footer)

        # Keep "last heard" and the pairing countdown ticking while open.
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
        self._btn_copy_link.setEnabled(running)
        self._btn_copy_operator.setEnabled(running)
        self._btn_new_operator.setEnabled(running)
        self._refresh_pairing(self._pairing(host))
        # Nothing to serve while the server is off.
        self._chk_microscope.setEnabled(
            self._chk_enabled.isChecked() or self._set_enabled is None
        )
        if not running:
            self._status_dot.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._status_label.setText(
                "Not running. It starts when a microscope connects."
                if self._chk_enabled.isChecked()
                else "Not running."
            )
            self._status_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._services_label.setText("")
            self._agent_label.setText("No clients while the App Server is off.")
            self._agent_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            return
        self._status_dot.setStyleSheet(f"color: {OK_COLOR};")
        self._status_label.setText(self._status_text(host))
        self._status_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        self._services_label.setText(self.services_text(host))
        self._agent_label.setText(self.agent_text(host))
        self._agent_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")

    @staticmethod
    def _pairing(host):
        auth = getattr(host, "auth", None) if host is not None else None
        return getattr(auth, "pairing", None)

    def _refresh_pairing(self, pairing) -> None:
        state = pairing.state() if pairing is not None else {"open": False}
        self._btn_pair.setEnabled(pairing is not None)
        if state["open"]:
            left = int(state["expires_in_s"])
            self._pair_code.setText(state["code"])
            self._pair_left.setText(f"{left // 60}:{left % 60:02d} left")
            self._btn_pair.setText(STOP_PAIRING_LABEL)
        else:
            self._pair_code.setText("")
            self._pair_left.setText("")
            self._btn_pair.setText(PAIR_LABEL)
        computers = tuple(pairing.tokens.computers()) if pairing is not None else ()
        if computers != self._shown_computers:
            self._shown_computers = computers
            self._draw_computers(pairing, computers)

    def _draw_computers(self, pairing, computers) -> None:
        while self._computers.count():
            item = self._computers.takeAt(0)
            row = item.layout()
            while row is not None and row.count():
                widget = row.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            if item.widget() is not None:
                item.widget().deleteLater()
        if not computers:
            self._computers.addWidget(note("No computers paired."))
            return
        for computer in computers:
            row = QHBoxLayout()
            paired = time.strftime("%d %b %Y", time.localtime(computer.paired_at))
            row.addWidget(QLabel(computer.name), stretch=1)
            row.addWidget(note(f"paired {paired}"))
            unpair = QPushButton("Unpair")
            unpair.setToolTip(f"{computer.name}'s token stops working at once.")
            unpair.clicked.connect(
                lambda _=False, cid=computer.id: self._on_unpair_clicked(cid)
            )
            row.addWidget(unpair)
            self._computers.addLayout(row)

    # --- actions --------------------------------------------------------------

    def _on_pair_clicked(self) -> None:
        pairing = self._pairing(self._host())
        if pairing is None:
            return
        if pairing.state()["open"]:
            pairing.close()
        else:
            pairing.open()
        self.refresh()

    def _on_unpair_clicked(self, computer_id: str) -> None:
        pairing = self._pairing(self._host())
        if pairing is not None:
            pairing.tokens.remove(computer_id)
        self.refresh()

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

    def _on_copy_link_clicked(self) -> None:
        host = self._host()
        clipboard = QApplication.clipboard()
        if host is None or clipboard is None:
            return
        clipboard.setText(event_log_url(host))
        self._btn_copy_link.setText("Copied")
        QTimer.singleShot(1500, self._restore_copy_link)

    def _restore_copy_link(self) -> None:
        self._btn_copy_link.setText("Copy link")

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

    def _restore_copy_operator(self) -> None:
        self._btn_copy_operator.setText("Copy operator token")
