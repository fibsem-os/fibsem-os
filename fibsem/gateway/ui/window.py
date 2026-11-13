"""The gateway's window: the app, the microscope and the agent at a glance,
starting and stopping the app, a pairing code, and the page. The gateway runs
while the window is open; closing it asks, then stops the gateway.

Reading the mode is cheap and done every second on the Qt thread. The agent's
record and the microscope's connection (a call to the app) are read on a
thread of their own and arrive by a signal, so the window never waits on them.
A start or stop can take a minute, so it runs off the Qt thread too."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Optional

from qtpy.QtCore import QByteArray, QObject, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QIcon, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.gateway.gateway import (
    OFF,
    STARTING,
    STOPPING,
    Gateway,
    GatewayRefused,
    GatewayState,
)
from fibsem.server.pairing import CODE_TTL_S, OPERATOR
from fibsem.theme.tokens import (
    BODY_MUTED_STYLE,
    BODY_STYLE,
    BORDER_COLOR,
    CAPTION_STYLE,
    GRAY_SECONDARY_COLOR,
    NUMBER_FONT,
    OK_COLOR,
    PANEL_COLOR,
    PRIMARY_COLOR,
    PURPLE_COLOR,
    SEMANTIC_WARNING_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.tools.app_identity import bring_to_front, set_app_identity
from fibsem.tools.apps import GATEWAY, ICONS_DIR
from fibsem.ui.stylesheets import (
    MESSAGE_BOX_STYLESHEET,
    NAPARI_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel

__all__ = ["GatewayWindow", "app_line", "mode_colour", "run_window"]

VIA = "window"
READ_EVERY_S = 3.0
ASK_FORCE = "ask_force"  # a stop refused because a workflow runs: ask to force it


def app_line(state: GatewayState) -> str:
    if state.switch == STARTING:
        return "Starting…"
    if state.switch == STOPPING:
        return "Stopping…"
    if state.mode == OFF:
        return "Off"
    name = state.app or "The app"
    return f"{name}, {'with its window' if state.window else 'without a window'}"


def mode_colour(state: GatewayState) -> str:
    if state.switching:
        return SEMANTIC_WARNING_COLOR
    if state.mode == OFF:
        return GRAY_SECONDARY_COLOR
    return PRIMARY_COLOR if state.window else PURPLE_COLOR


def microscope_line(connection: Optional[dict]) -> tuple:
    """``(text, colour)`` for the microscope: *connection* is the app's
    ``/microscope/connection``, None while the app is off."""
    if connection is None:
        return "Connects once the app runs", GRAY_SECONDARY_COLOR
    if connection.get("connecting"):
        return f"Connecting to {connection['connecting']}…", WARN_COLOR
    if connection.get("connected"):
        name = (connection.get("configuration") or {}).get("name")
        return (f"Connected · {name}" if name else "Connected"), OK_COLOR
    if connection.get("error"):
        return "Not connected · it failed", WARN_COLOR
    return "Not connected", GRAY_SECONDARY_COLOR


def agent_line(running: Optional[dict], connected: Optional[dict] = None) -> tuple:
    """``(text, detail, colour)`` for the agent's row: the gateway's agent and
    its goal, else an agent connected from elsewhere and whether it is quiet."""
    if not running and connected:
        quiet = connected.get("quiet")
        text = f"{connected.get('name')} connected"
        return text, quiet or "", WARN_COLOR if quiet else OK_COLOR
    if not running:
        return "No agent", "", GRAY_SECONDARY_COLOR
    level = (running.get("level") or "").capitalize()
    if running.get("state") == "waiting":
        return "Waiting for the app", running.get("goal") or "", WARN_COLOR
    text = f"Running at {level}" if level else "Running"
    return text, running.get("goal") or "", OK_COLOR


def gateway_icon(colour: str, size: int = 128) -> QIcon:
    """The gateway's icon with its badge in the app's colour."""
    return QIcon(_gateway_pixmap(colour, size))


def _gateway_pixmap(colour: str, size: int) -> QPixmap:
    svg = (ICONS_DIR / "gateway.svg").read_text().replace("#007ACC", colour)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    return pixmap


def stop_question(state: GatewayState, restart: bool = False) -> str:
    """What stopping (or restarting) the gateway does to the app."""
    if state.mode == OFF:
        if restart:
            return "It starts again in a few seconds."
        return "The gateway's page stops working until the gateway starts again."
    if restart:
        return "The app keeps running, and the gateway takes it back as it starts."
    if state.window:
        return "The app keeps running."
    return "The app without a window stops too, unless it is running a workflow."


def _dot(colour: str) -> str:
    return f"<span style='color: {colour};'>●</span>"


class _Section(QFrame):
    """A titled panel: a caption, a dot and a line, a muted second line, and
    a button on the right."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("section")
        self.setStyleSheet(
            f"QFrame#section {{ background-color: {PANEL_COLOR};"
            f" border: 1px solid {BORDER_COLOR}; border-radius: 6px; }}"
            "QFrame#section QLabel { background: transparent; border: none; }"
        )
        caption = QLabel(title.upper())
        caption.setStyleSheet(f"{CAPTION_STYLE} letter-spacing: 1px;")
        self.line = QLabel()
        self.line.setStyleSheet(BODY_STYLE)
        self.line.setTextFormat(Qt.TextFormat.RichText)
        # One line, the whole text in the tooltip: a long agent goal wrapped
        # past the panel's height and was drawn squashed into it.
        self.detail = ElidedLabel()
        self.detail.setStyleSheet(BODY_MUTED_STYLE)
        self.button = QPushButton()
        self.button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.button.hide()
        text = QVBoxLayout()
        text.setSpacing(3)
        text.addWidget(caption)
        text.addWidget(self.line)
        text.addWidget(self.detail)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.addLayout(text, 1)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)

    def show_line(self, text: str, colour: str, detail: str = "") -> None:
        self.line.setText(f"{_dot(colour)}&nbsp;&nbsp;{text}")
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))


class _Reports(QObject):
    # The agent's running row, the microscope's connection, an agent
    # connected from elsewhere.
    read = Signal(object, object, object)
    done = Signal(str, str)  # title, refusal ("" when it worked)


class GatewayWindow(QWidget):
    def __init__(
        self,
        gateway: Gateway,
        auth,
        port: int,
        agent=None,
        stopping=None,
        microscope=None,
        clock: Callable[[], float] = time.monotonic,
        shown: Optional[tuple] = None,
        restart: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self._gateway = gateway
        self._auth = auth
        self._port = port
        self._agent = agent
        self._microscope = microscope
        self._clock = clock
        # (has_window, bring_forward): set the first once up; the desktop
        # shortcut sets the second to bring this window to the front.
        self._has_window, self._bring_forward = shown or (None, None)
        # Set by the quit route (`fibsem-gateway stop`, the page): close.
        self._stopping = stopping
        # Asks the gateway to stop and start again (it then sets *stopping*).
        self._restart = restart
        self._quitting = False
        self._closed = threading.Event()
        self._colour: Optional[str] = None
        self._state: Optional[GatewayState] = None
        self._busy = False
        self._code_until = 0.0
        self._reports = _Reports()
        self._reports.read.connect(self._show_read)
        self._reports.done.connect(self._finished)
        self.setWindowTitle(GATEWAY.name)
        self.setStyleSheet(
            NAPARI_STYLE
            + MESSAGE_BOX_STYLESHEET
            + "QPushButton, QMenu { font-size: 12px; }"
        )
        self._build()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)
        self.refresh()
        self._show_read(None, None)
        threading.Thread(
            target=self._read_loop, name="gateway-window", daemon=True
        ).start()

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        self.icon_label = QLabel()
        title = QLabel(GATEWAY.name)
        title.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 16px; font-weight: 600;"
        )
        where = QLabel(f"Running on this PC · 127.0.0.1:{self._port}")
        where.setStyleSheet(CAPTION_STYLE)
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(title)
        names.addWidget(where)
        self.open_button = QPushButton("Open page")
        self.open_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.open_page)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(self.icon_label)
        header.addLayout(names, 1)
        header.addWidget(self.open_button, 0, Qt.AlignmentFlag.AlignTop)

        self.app = _Section("App")
        self.app.button.clicked.connect(self._app_button)
        self.microscope = _Section("Microscope")
        self.agent = _Section("Agent")
        self.agent.button.setText("End")
        self.agent.button.clicked.connect(self._end_agent)

        # Pair shows a code here; the button then hides it again.
        self.code_box = _Section("Pair a computer")
        self.code_box.line.setTextFormat(Qt.TextFormat.PlainText)
        self.pair_button = self.code_box.button
        self.pair_button.show()
        self.pair_button.clicked.connect(self._pair_button)
        self._hide_code()

        self.restart_button = QPushButton("Restart")
        self.restart_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.restart_button.clicked.connect(self.restart_gateway)
        self.restart_button.setVisible(self._restart is not None)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.stop_button.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.restart_button)
        buttons.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addSpacing(4)
        layout.addWidget(self.app)
        layout.addWidget(self.microscope)
        layout.addWidget(self.agent)
        layout.addWidget(self.code_box)
        layout.addStretch()
        layout.addLayout(buttons)
        self.setMinimumWidth(420)

    # -- reading -----------------------------------------------------------

    def refresh(self) -> None:
        if self._stopping is not None and self._stopping.is_set():
            self._quit()
            return
        if self._bring_forward is not None and self._bring_forward.is_set():
            self._bring_forward.clear()
            self.bring_to_front()
        state = self._gateway.mode()
        self._state = state
        folder = state.experiment if state.mode != OFF else None
        self.app.show_line(
            app_line(state),
            mode_colour(state),
            os.path.basename(folder) if folder else "",
        )
        button = self.app.button
        if state.mode == OFF:
            button.setText("Start")
            button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        else:
            button.setText("Stop")
            button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        button.setVisible(True)
        button.setEnabled(not state.switching and not self._busy)
        colour = mode_colour(state)
        if colour != self._colour:
            self._colour = colour
            self.setWindowIcon(gateway_icon(colour))
            self.icon_label.setPixmap(_gateway_pixmap(colour, 40))
            QApplication.instance().setWindowIcon(gateway_icon(colour))
        if self._code_until and self._clock() >= self._code_until:
            self._hide_code()

    def _pair_button(self) -> None:
        if self._code_until:
            self._hide_code()
        else:
            self.show_pairing_code()

    def _hide_code(self) -> None:
        self._code_until = 0.0
        self.code_box.line.setStyleSheet(BODY_MUTED_STYLE)
        self.code_box.line.setText("A browser on another computer")
        self.code_box.detail.hide()
        self.pair_button.setText("Pair")

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        bring_to_front()

    def _read_loop(self) -> None:
        """The agent's row and the microscope's connection, every few seconds."""
        while not self._closed.is_set():
            running = connection = connected = None
            try:
                if self._agent is not None:
                    state = self._agent.state()
                    running, connected = state.running, state.connected
            except Exception:  # noqa: BLE001 - shown as no agent until next read
                running = connected = None
            try:
                if self._microscope is not None and self._gateway.mode().mode != OFF:
                    connection = self._microscope.connection()
            except (GatewayRefused, OSError, ValueError, KeyError):
                connection = None
            if self._closed.is_set():
                return
            self._reports.read.emit(running, connection, connected)
            self._closed.wait(READ_EVERY_S)

    def _show_read(
        self,
        running: Optional[dict],
        connection: Optional[dict],
        connected: Optional[dict] = None,
    ) -> None:
        text, colour = microscope_line(connection)
        self.microscope.show_line(text, colour)
        text, detail, colour = agent_line(running, connected)
        self.agent.show_line(text, colour, detail)
        self.agent.button.setVisible(bool(running))
        if not running and not connected:
            self.agent.detail.setText("Start one on the page.")
            self.agent.detail.show()

    # -- acting ------------------------------------------------------------

    def _in_background(self, title: str, act: Callable[[], object]) -> None:
        self._busy = True

        def work():
            try:
                act()
            except GatewayRefused as refusal:
                kind = ASK_FORCE if refusal.error_type == "run_in_progress" else title
                self._reports.done.emit(kind, str(refusal))
                return
            self._reports.done.emit(title, "")

        threading.Thread(target=work, name="gateway-window-act", daemon=True).start()
        self.refresh()

    def _finished(self, title: str, message: str) -> None:
        self._busy = False
        self.refresh()
        if title == ASK_FORCE:
            if self._ask(
                "Stop the app", f"{message}\n\nStop it anyway? The workflow stops too."
            ):
                self._stop_app(force=True)
        elif message:
            box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self)
            box.exec()

    def _app_button(self) -> None:
        state = self._state or self._gateway.mode()
        if state.mode == OFF:
            self._start_menu()
        else:
            self._stop_app()

    def _start_menu(self) -> None:
        apps = [a for a in self._gateway.state().apps]
        if len(apps) == 1:
            self._start(apps[0].key, apps[0].name)
            return
        menu = QMenu(self)
        for app in apps:
            menu.addAction(app.name, lambda k=app.key, n=app.name: self._start(k, n))
        menu.exec(self.app.button.mapToGlobal(self.app.button.rect().bottomLeft()))

    def _start(self, key: str, name: str) -> None:
        self._in_background(
            f"Start {name}",
            lambda: self._gateway.start_app("operator", VIA, app=key),
        )

    def _stop_app(self, force: bool = False) -> None:
        state = self._state or self._gateway.mode()
        if (
            not force
            and state.window
            and not self._ask("Stop the app", "Close the app's window? It saves first.")
        ):
            return
        self._in_background(
            "Stop the app",
            lambda: self._gateway.stop("operator", VIA, force, window=True),
        )

    def _end_agent(self) -> None:
        if self._agent is None or not self._ask(
            "End the agent", "End the agent? The app keeps running."
        ):
            return
        self._in_background("End the agent", lambda: self._agent.end("operator", VIA))

    def show_pairing_code(self) -> None:
        """A code another computer's browser types to pair as the operator."""
        code = self._auth.pairing.open(OPERATOR, ())
        self.code_box.line.setStyleSheet(
            f"font-family: {NUMBER_FONT}; font-size: 22px; letter-spacing: 6px;"
            f" color: {TEXT_STRONG_COLOR};"
        )
        self.code_box.line.setText(code)
        self.code_box.detail.setText(
            f"Type it on the other computer within {int(CODE_TTL_S // 60)} minutes."
        )
        self.code_box.detail.show()
        self.pair_button.setText("Hide")
        self._code_until = self._clock() + CODE_TTL_S

    def open_page(self) -> None:
        QDesktopServices.openUrl(
            QUrl(f"http://127.0.0.1:{self._port}/ui/gateway#token={self._auth.token}")
        )

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    def restart_gateway(self) -> None:
        if self._restart is None or not self._ask(
            "Restart the gateway",
            f"Restart the gateway?\n\n{stop_question(self._gateway.mode(), True)}",
        ):
            return
        self._restart()
        self.refresh()

    def closeEvent(self, event) -> None:
        if self._quitting or self._ask(
            "Stop the gateway",
            f"Stop the gateway?\n\n{stop_question(self._gateway.mode())}",
        ):
            event.accept()
            self._quit()
        else:
            event.ignore()

    def _quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._closed.set()
        self._timer.stop()
        self.hide()
        # The gateway stops the app as it quits, unless it is restarting.
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
    gateway: Gateway,
    auth,
    port: int,
    agent=None,
    stopping=None,
    minimised=False,
    microscope=None,
    shown: Optional[tuple] = None,
    restart: Optional[Callable[[], None]] = None,
) -> bool:
    """The window until it closes (or *stopping* is set); False when there is
    no display to show it on."""
    if not has_display():
        return False
    set_app_identity(GATEWAY.name, GATEWAY.app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(GATEWAY.name)
    window = GatewayWindow(
        gateway, auth, port, agent, stopping, microscope, shown=shown, restart=restart
    )
    if minimised:
        window.showMinimized()
    else:
        window.show()
    if shown is not None:
        shown[0].set()
    app.exec()
    return True
