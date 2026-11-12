"""The gateway's window: whether the app and the agent run, and the page. The
gateway runs while the window is open; closing it asks, then stops the gateway.
Its icon's badge is the app's colour: grey off, blue with its window, purple
without one."""

from __future__ import annotations

import os
import sys
from typing import Optional

from qtpy.QtCore import QByteArray, Qt, QTimer, QUrl
from qtpy.QtGui import QDesktopServices, QIcon, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.gateway.gateway import OFF, STARTING, STOPPING, Gateway, GatewayState
from fibsem.gateway.main import quit_gateway
from fibsem.theme.tokens import (
    GRAY_SECONDARY_COLOR,
    PRIMARY_COLOR,
    PURPLE_COLOR,
    SEMANTIC_WARNING_COLOR,
)
from fibsem.tools.app_identity import set_app_identity
from fibsem.tools.apps import GATEWAY, ICONS_DIR

__all__ = ["GatewayWindow", "app_line", "mode_colour", "run_window"]

AGENT_SAYS = {
    "running": "The agent is running.",
    "waiting": "The agent is waiting for the app.",
}


def app_line(state: GatewayState) -> str:
    if state.switch == STARTING:
        return "The app is starting…"
    if state.switch == STOPPING:
        return "The app is stopping…"
    if state.mode == OFF:
        return "The app is off."
    name = state.app or "The app"
    return f"{name} is running {'with its window' if state.window else 'without a window'}."


def mode_colour(state: GatewayState) -> str:
    if state.switching:
        return SEMANTIC_WARNING_COLOR
    if state.mode == OFF:
        return GRAY_SECONDARY_COLOR
    return PRIMARY_COLOR if state.window else PURPLE_COLOR


def gateway_icon(colour: str) -> QIcon:
    """The gateway's icon with its badge in the app's colour."""
    svg = (ICONS_DIR / "gateway.svg").read_text().replace("#007ACC", colour)
    size = 128
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    return QIcon(pixmap)


def stop_question(state: GatewayState) -> str:
    """What stopping the gateway does to the app, for the question."""
    if state.mode == OFF:
        return "The gateway's page stops working until the gateway starts again."
    if state.window:
        return "The app keeps running."
    return "The app without a window stops too, unless it is running a workflow."


class GatewayWindow(QWidget):
    def __init__(self, gateway: Gateway, auth, port: int, agent=None, stopping=None):
        super().__init__()
        self._gateway = gateway
        self._auth = auth
        self._port = port
        self._agent = agent
        # Set by the quit route (`fibsem-gateway stop`, the page): close.
        self._stopping = stopping
        self._quitting = False
        self._colour: Optional[str] = None
        self.setWindowTitle(GATEWAY.name)

        self.app_label = QLabel()
        self.experiment_label = QLabel()
        self.experiment_label.setStyleSheet(f"color: {GRAY_SECONDARY_COLOR};")
        self.agent_label = QLabel()
        self.open_button = QPushButton("Open page")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.open_page)
        self.stop_button = QPushButton("Stop gateway")
        self.stop_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.app_label)
        layout.addWidget(self.experiment_label)
        layout.addWidget(self.agent_label)
        layout.addStretch()
        layout.addLayout(buttons)
        self.setMinimumWidth(340)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        if self._stopping is not None and self._stopping.is_set():
            self._quit()
            return
        state = self._gateway.mode()
        self.app_label.setText(app_line(state))
        folder = state.experiment if state.mode != OFF else None
        self.experiment_label.setText(os.path.basename(folder) if folder else "")
        self.experiment_label.setVisible(bool(folder))
        running = self._agent.running() if self._agent is not None else None
        self.agent_label.setText(AGENT_SAYS.get(running or "", "No agent."))
        colour = mode_colour(state)
        if colour != self._colour:
            self._colour = colour
            icon = gateway_icon(colour)
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)

    def open_page(self) -> None:
        QDesktopServices.openUrl(
            QUrl(f"http://127.0.0.1:{self._port}/ui/gateway#token={self._auth.token}")
        )

    def ask_to_stop(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Stop the gateway",
            f"Stop the gateway?\n\n{stop_question(self._gateway.mode())}",
        )
        return answer == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:
        if self._quitting or self.ask_to_stop():
            event.accept()
            self._quit()
        else:
            event.ignore()

    def _quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._timer.stop()
        self.hide()
        quit_gateway(self._gateway)
        QApplication.instance().quit()


def has_display() -> bool:
    """Whether a window can open: Qt aborts the process on Linux with no
    display."""
    if sys.platform in ("win32", "darwin"):
        return True
    return any(
        os.environ.get(name)
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")
    )


def run_window(
    gateway: Gateway, auth, port: int, agent=None, stopping=None, minimised=False
) -> bool:
    """The window until it closes (or *stopping* is set); False when there is
    no display to show it on."""
    if not has_display():
        return False
    set_app_identity(GATEWAY.name, GATEWAY.app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(GATEWAY.name)
    window = GatewayWindow(gateway, auth, port, agent, stopping)
    if minimised:
        window.showMinimized()
    else:
        window.show()
    app.exec()
    return True
