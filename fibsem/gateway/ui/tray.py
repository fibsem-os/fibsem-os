"""The gateway's tray icon: the same switches as its page, for the operator at
the PC. Its colour follows the mode. A switch can take a minute (a headless
app stops its run), so it runs off the Qt thread and reports back in a
tray message."""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

from fibsem.gateway.gateway import HEADLESS, OFF, WINDOW, Gateway, GatewayRefused
from fibsem.gateway.main import quit_gateway
from fibsem.server.pairing import HUB, OPERATOR, SWITCH_MODES
from fibsem.theme.tokens import (
    GRAY_SECONDARY_COLOR,
    PRIMARY_COLOR,
    PURPLE_COLOR,
    SEMANTIC_WARNING_COLOR,
)
from fibsem.tools.apps import ICONS_DIR

__all__ = ["GatewayTray", "mode_colour", "run_tray"]

MODE_NAMES = {OFF: "Off", WINDOW: "Window", HEADLESS: "Headless app"}
VIA = "tray"
ASK_FORCE = "ask_force"  # a report that asks whether to stop past a running workflow


def mode_colour(mode: str, switching: bool) -> str:
    if switching:
        return SEMANTIC_WARNING_COLOR
    return {WINDOW: PRIMARY_COLOR, HEADLESS: PURPLE_COLOR}.get(
        mode, GRAY_SECONDARY_COLOR
    )


def tray_icon(colour: str) -> QIcon:
    """fibsemOS's icon with a dot in the mode's colour."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    QIcon(str(ICONS_DIR / "fibsemos.svg")).paint(painter, 0, 0, size, size)
    painter.setPen(QColor("#1e2124"))
    painter.setBrush(QColor(colour))
    painter.drawEllipse(size - 26, size - 26, 24, 24)
    painter.end()
    return QIcon(pixmap)


class _Report(QObject):
    done = pyqtSignal(str, str)  # title, message ("" when it worked)


class GatewayTray(QSystemTrayIcon):
    def __init__(self, gateway: Gateway, auth, port: int):
        super().__init__()
        self._gateway = gateway
        self._auth = auth
        self._port = port
        self._shown: Optional[tuple] = None
        self._report = _Report()
        self._report.done.connect(self._reported)
        self.menu = QMenu()
        self.menu.aboutToShow.connect(self._build_menu)
        self.setContextMenu(self.menu)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        state = self._gateway.state()
        shown = (state.mode, state.switching, state.app, state.experiment)
        if shown == self._shown:
            return
        self._shown = shown
        self.setIcon(tray_icon(mode_colour(state.mode, state.switching)))
        what = MODE_NAMES[state.mode]
        if state.app:
            what += f" · {state.app}"
        if state.experiment:
            what += f" · {state.experiment}"
        self.setToolTip(f"fibsemOS Gateway: {what}")

    def _build_menu(self) -> None:
        menu = self.menu
        menu.clear()
        state = self._gateway.state()
        title = menu.addAction(f"Mode: {MODE_NAMES[state.mode]}")
        title.setEnabled(False)
        if state.holder is not None:
            held = menu.addAction(f"Another App Server runs (pid {state.holder.pid})")
            held.setEnabled(False)
        menu.addSeparator()
        free = state.mode != WINDOW and not state.switching
        windows = menu.addMenu("Open window")
        headless = menu.addMenu("Run headless")
        for app in state.apps:
            self._add(windows, app.name, free, lambda k=app.key: self._window(k))
            if app.headless:
                sub = headless.addMenu(app.name)
                self._add(
                    sub,
                    "Last experiment",
                    free,
                    lambda k=app.key: self._headless(k, None),
                )
                self._add(sub, "Choose…", free, lambda k=app.key: self._choose(k))
        windows.setEnabled(free)
        headless.setEnabled(free)
        stop = self._add(
            menu, "Stop", state.mode == HEADLESS and not state.switching, self._stop
        )
        if state.mode == WINDOW:
            stop.setText("Stop (close the window)")
        menu.addSeparator()
        pair = menu.addMenu("Pair a computer")
        self._add(pair, "A browser (operator)", True, lambda: self._pair(OPERATOR, ()))
        self._add(
            pair,
            "A hub that may start and stop apps",
            True,
            lambda: self._pair(HUB, (SWITCH_MODES,)),
        )
        self._add(pair, "A hub that only reads", True, lambda: self._pair(HUB, ()))
        self._add(menu, "Open gateway page", True, self._open_page)
        menu.addSeparator()
        self._add(menu, "Quit gateway", True, self._quit)

    @staticmethod
    def _add(menu: QMenu, text: str, enabled: bool, slot: Callable) -> QAction:
        action = menu.addAction(text)
        action.setEnabled(enabled)
        action.triggered.connect(lambda _=False: slot())
        return action

    def _in_background(self, title: str, act: Callable[[], object]) -> None:
        def work():
            try:
                act()
            except GatewayRefused as refusal:
                self._report.done.emit(title, str(refusal))
                return
            self._report.done.emit(title, "")

        threading.Thread(target=work, name="gateway-tray", daemon=True).start()

    def _reported(self, title: str, message: str) -> None:
        self.refresh()
        if title == ASK_FORCE:
            answer = QMessageBox.question(
                None,
                "Stop the headless app",
                f"{message}\n\nStop it anyway? The workflow stops too.",
            )
            if answer == QMessageBox.Yes:
                self._stop(force=True)
            return
        if message:
            self.showMessage(title, message, QSystemTrayIcon.Warning)

    def _window(self, key: str) -> None:
        self._in_background(
            "Open window",
            lambda: self._gateway.switch(WINDOW, "operator", VIA, app=key),
        )

    def _headless(self, key: str, folder: Optional[str]) -> None:
        self._in_background(
            "Run headless",
            lambda: self._gateway.switch(
                HEADLESS,
                "operator",
                VIA,
                app=key,
                experiment=folder,
                last=folder is None,
            ),
        )

    def _choose(self, key: str) -> None:
        folder = QFileDialog.getExistingDirectory(
            None, "Choose the experiment's folder"
        )
        if folder:
            self._headless(key, folder)

    def _stop(self, force: bool = False) -> None:
        def act():
            try:
                self._gateway.stop("operator", VIA, force)
            except GatewayRefused as refusal:
                if refusal.error_type != "run_in_progress" or force:
                    raise
                self._report.done.emit(ASK_FORCE, str(refusal))

        self._in_background("Stop", act)

    def _pair(self, role: str, grants: tuple) -> None:
        code = self._auth.pairing.open(role, grants)
        what = {OPERATOR: "a browser"}.get(role, "a hub")
        QMessageBox.information(
            None,
            "Pair a computer",
            f"Pairing code for {what}: {code}\n\nType it on the other computer "
            "within 5 minutes.",
        )

    def _open_page(self) -> None:
        QDesktopServices.openUrl(
            QUrl(f"http://127.0.0.1:{self._port}/ui/gateway#token={self._auth.token}")
        )

    def _quit(self) -> None:
        self.hide()
        quit_gateway(self._gateway)
        QApplication.instance().quit()


def run_tray(gateway: Gateway, auth, port: int) -> bool:
    """The tray until Quit; False when this desktop has no tray."""
    app = QApplication.instance() or QApplication(sys.argv)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return False
    app.setQuitOnLastWindowClosed(False)
    tray = GatewayTray(gateway, auth, port)
    tray.show()
    app.exec_()
    return True
