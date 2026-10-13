"""Tools → App Server: the server itself -- status, services, the dashboard.

The App Server is the API inside AutoLamella that the agent, the dashboard and
``FibsemClient`` reach. This dialog is about the server, not the agent
(FIB-1035): what the agent may do is its level, on the chip in the status bar,
and connecting one is the chip's "Connect an agent…" (``connect_agent_dialog``).

Whether it runs lives in Preferences → App Server. Whether this session
serves the microscope's devices is chosen here (FIB-1359), per session like the
agent level; Preferences keeps only the start-up default.
"""

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

from fibsem.ui import stylesheets
from fibsem.ui.tokens import OK_COLOR, TEXT_MUTED_COLOR, TEXT_STRONG_COLOR

__all__ = ["AppServerDialog"]

DEVICES_LABEL = "Serve the microscope's devices"
DEVICES_TIP = (
    "Also serve the routes that reach the instrument: its devices, stage, beams "
    "and milling state, reads included. Off, clients see the session but cannot "
    "query the microscope. Moving it still needs the agent at Drive. Changing "
    "this restarts the App Server; a connected agent carries on. For this "
    "session only; Preferences → App Server sets what a new session starts with."
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


class AppServerDialog(QDialog):
    """Live view of the embedded App Server."""

    def __init__(
        self,
        host_provider: Callable[[], Optional[object]],
        parent=None,
        set_devices: Optional[Callable[[bool], None]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("App Server")
        self.setMinimumWidth(440)
        self._host_provider = host_provider
        # Serves or stops serving the devices this session; restarts the server.
        self._set_devices = set_devices

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
        self._chk_devices = QCheckBox(DEVICES_LABEL)
        self._chk_devices.setToolTip(DEVICES_TIP)
        self._chk_devices.setVisible(set_devices is not None)
        self._chk_devices.toggled.connect(self._on_devices_toggled)
        layout.addWidget(self._chk_devices)

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

        # --- dashboard --------------------------------------------------------
        layout.addWidget(section("Dashboard"))
        layout.addWidget(
            note(
                "A read-only monitor page in your browser, with a token of its "
                "own: it watches the session and can't act on it. The agent's "
                "level doesn't affect it."
            )
        )
        dashboard_row = QHBoxLayout()
        dashboard_row.addStretch(1)
        self._btn_copy_link = QPushButton("Copy link")
        self._btn_copy_link.setToolTip("Copy the dashboard's address and token.")
        self._btn_copy_link.clicked.connect(self._on_copy_link_clicked)
        self._btn_dashboard = QPushButton("Open Dashboard")
        self._btn_dashboard.setToolTip(
            "Item progress, the workflow, and the live event feed."
        )
        self._btn_dashboard.clicked.connect(self._on_dashboard_clicked)
        dashboard_row.addWidget(self._btn_copy_link)
        dashboard_row.addWidget(self._btn_dashboard)
        layout.addLayout(dashboard_row)

        layout.addSpacing(6)

        # --- footer ---------------------------------------------------------
        layout.addWidget(note("Whether it runs is in Preferences → App Server."))
        footer = QHBoxLayout()
        footer.addStretch(1)
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)
        layout.addLayout(footer)

        # Keep "last heard" ticking while the dialog is open.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
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
        devices = "on" if getattr(host, "devices", False) else "off"
        return f"Services: monitoring · devices {devices}"

    @staticmethod
    def agent_text(host) -> str:
        """When the agent's token was last used, in units read at a glance."""
        try:
            age = host.agent_seconds_since_seen()
        except Exception:
            age = None
        if age is None:
            return "Agent · not heard from yet"
        if age < 60:
            return f"Agent · last heard {int(age)} s ago"
        return f"Agent · last heard {int(age // 60)} min ago"

    def refresh(self) -> None:
        """Re-read the session's server into the dialog."""
        host = self._host()
        running = host is not None
        self._btn_dashboard.setEnabled(running)
        self._btn_copy_link.setEnabled(running)
        if not running:
            self._status_dot.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._status_label.setText(
                "Not running — enable it in Preferences → App Server and connect "
                "a microscope."
            )
            self._status_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            self._services_label.setText("")
            self._chk_devices.setEnabled(False)
            self._agent_label.setText("No clients while the App Server is off.")
            self._agent_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
            return
        self._status_dot.setStyleSheet(f"color: {OK_COLOR};")
        self._status_label.setText(self._status_text(host))
        self._status_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        self._services_label.setText(self.services_text(host))
        self._chk_devices.setEnabled(True)
        self._chk_devices.blockSignals(True)
        self._chk_devices.setChecked(bool(getattr(host, "devices", False)))
        self._chk_devices.blockSignals(False)
        self._agent_label.setText(self.agent_text(host))
        self._agent_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")

    # --- actions --------------------------------------------------------------

    def _on_devices_toggled(self, checked: bool) -> None:
        if self._set_devices is None:
            return
        self._set_devices(checked)
        self.refresh()

    @staticmethod
    def dashboard_url(host) -> str:
        """The dashboard address with the dashboard's own read-only token
        (FIB-1035), never the agent's, in the URL *fragment* — the one part of
        a URL a browser never sends anywhere, so it reaches the page's own
        script and nothing else (no server log, no referrer)."""
        return f"{host.url}/dashboard#token={host.auth.dashboard_token}"

    def _on_dashboard_clicked(self) -> None:
        host = self._host()
        if host is None:
            return
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(self.dashboard_url(host)))

    def _on_copy_link_clicked(self) -> None:
        host = self._host()
        clipboard = QApplication.clipboard()
        if host is None or clipboard is None:
            return
        clipboard.setText(self.dashboard_url(host))
        self._btn_copy_link.setText("Copied")
        QTimer.singleShot(1500, self._restore_copy_link)

    def _restore_copy_link(self) -> None:
        self._btn_copy_link.setText("Copy link")
