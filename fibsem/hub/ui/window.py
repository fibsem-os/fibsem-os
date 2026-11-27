"""The Hub's window: the instruments and the agents on this PC at a glance,
a pairing code for a browser, and the page. The Hub runs while the window is
open; closing it asks, then stops the Hub.

It is the gateway's window (`fibsem.gateway.ui.window`) with the Hub's own
rows: the panels, the icon and the styles are the gateway's. Everything shown
is what the Hub last read, so the window never waits on an instrument."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, Optional

from qtpy.QtCore import Qt, QTimer, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.gateway.ui.window import (
    Section,
    gateway_icon,
    gateway_pixmap,
    has_display,
)
from fibsem.hub.hub import Hub, HubState
from fibsem.server.pairing import CODE_TTL_S, OPERATOR
from fibsem.theme.tokens import (
    BODY_MUTED_STYLE,
    CAPTION_STYLE,
    GRAY_SECONDARY_COLOR,
    NUMBER_FONT,
    OK_COLOR,
    PRIMARY_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.tools.app_identity import bring_to_front, set_app_identity
from fibsem.tools.apps import HUB
from fibsem.ui.stylesheets import (
    MESSAGE_BOX_STYLESHEET,
    NAPARI_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)

__all__ = ["HubWindow", "agents_line", "instruments_line", "run_window"]


def _names(names: list) -> str:
    return ", ".join(names[:4]) + (f" and {len(names) - 4} more" if names[4:] else "")


def instruments_line(state: HubState) -> tuple:
    """``(text, detail, colour)`` for the instruments' row."""
    rows = state.instruments
    if not rows:
        return "No instruments yet", "Add one on the page.", GRAY_SECONDARY_COLOR
    away = [r.name for r in rows if r.reachable is False]
    unpaired = [r.name for r in rows if r.reachable is not False and not r.paired]
    well = [r for r in rows if r.reachable and r.paired]
    count = f"{len(rows)} instrument{'' if len(rows) == 1 else 's'}"
    text = f"{count} · {len(well)} reachable"
    detail = []
    if away:
        detail.append(f"Not reachable: {_names(away)}")
    if unpaired:
        detail.append(f"Not paired: {_names(unpaired)}")
    colour = WARN_COLOR if detail else OK_COLOR
    if len(well) < len(rows) and not detail:
        colour = GRAY_SECONDARY_COLOR  # still reading
    return text, " · ".join(detail), colour


def agents_line(found: Dict[str, Dict[str, Any]]) -> tuple:
    """``(text, detail, colour)`` for the row of agents on this PC: *found*
    is `HubAgents.state`."""
    held = {n: a for n, a in found.items() if a.get("state") != "ended"}
    if not held:
        return "No agent", "Start one on the page.", GRAY_SECONDARY_COLOR
    away = [n for n, a in held.items() if a.get("state") != "running"]
    count = f"{len(held)} agent{'' if len(held) == 1 else 's'} on this PC"
    if away:
        return count, f"Waiting for its instrument: {_names(away)}", WARN_COLOR
    return count, _names(sorted(held)), OK_COLOR


class HubWindow(QWidget):
    def __init__(
        self,
        hub: Hub,
        auth,
        port: int,
        agents=None,
        stopping=None,
        clock: Callable[[], float] = time.monotonic,
        shown: Optional[tuple] = None,
    ):
        super().__init__()
        self._hub = hub
        self._auth = auth
        self._port = port
        self._agents = agents
        self._clock = clock
        # (has_window, bring_forward): as the gateway's window has them.
        self._has_window, self._bring_forward = shown or (None, None)
        # Set by the quit route (`agent-hub stop`): close.
        self._stopping = stopping
        self._quitting = False
        self._code_until = 0.0
        self.setWindowTitle(HUB.name)
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

    def _build(self) -> None:
        self.icon_label = QLabel()
        self.icon_label.setPixmap(gateway_pixmap(PRIMARY_COLOR, 40))
        self.setWindowIcon(gateway_icon(PRIMARY_COLOR))
        title = QLabel(HUB.name)
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

        self.instruments = Section("Instruments")
        self.agents = Section("Agents")
        # Pair shows a code here; the button then hides it again.
        self.code_box = Section("Pair a computer")
        self.code_box.line.setTextFormat(Qt.TextFormat.PlainText)
        self.pair_button = self.code_box.button
        self.pair_button.show()
        self.pair_button.clicked.connect(self._pair_button)
        self._hide_code()

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.stop_button.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addSpacing(4)
        layout.addWidget(self.instruments)
        layout.addWidget(self.agents)
        layout.addWidget(self.code_box)
        layout.addStretch()
        layout.addLayout(buttons)
        self.setMinimumWidth(420)

    def refresh(self) -> None:
        if self._stopping is not None and self._stopping.is_set():
            self._quit()
            return
        if self._bring_forward is not None and self._bring_forward.is_set():
            self._bring_forward.clear()
            self.bring_to_front()
        text, detail, colour = instruments_line(self._hub.state())
        self.instruments.show_line(text, colour, detail)
        found = self._agents.state() if self._agents is not None else {}
        text, detail, colour = agents_line(found)
        self.agents.show_line(text, colour, detail)
        if self._code_until and self._clock() >= self._code_until:
            self._hide_code()

    def _running(self) -> int:
        found = self._agents.state() if self._agents is not None else {}
        return sum(1 for a in found.values() if a.get("state") != "ended")

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

    def show_pairing_code(self) -> None:
        """A code another computer's browser types to pair as the operator:
        the one `agent-hub pair` shows."""
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

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        bring_to_front()

    def open_page(self) -> None:
        QDesktopServices.openUrl(
            QUrl(f"http://127.0.0.1:{self._port}/ui/hub#token={self._auth.token}")
        )

    def stop_question(self) -> str:
        """What stopping the Hub does to the agents it runs."""
        running = self._running()
        if not running:
            return "The Hub's page stops working until the Hub starts again."
        agents = "The agent" if running == 1 else f"The {running} agents"
        verb = "ends" if running == 1 else "end"
        return (
            f"{agents} on this PC {verb}, and each slot is given back. "
            "Resume one from Sessions once the Hub runs again."
        )

    def closeEvent(self, event) -> None:
        if self._quitting or (
            QMessageBox.question(
                self, "Stop the Hub", f"Stop the Hub?\n\n{self.stop_question()}"
            )
            == QMessageBox.StandardButton.Yes
        ):
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
        QApplication.instance().quit()


def run_window(
    hub: Hub,
    auth,
    port: int,
    agents=None,
    stopping=None,
    minimised: bool = False,
    shown: Optional[tuple] = None,
) -> bool:
    """The window until it closes (or *stopping* is set); False when there is
    no display to show it on."""
    if not has_display():
        return False
    set_app_identity(HUB.name, HUB.app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(HUB.name)
    app.setWindowIcon(gateway_icon(PRIMARY_COLOR))
    window = HubWindow(hub, auth, port, agents, stopping, shown=shown)
    if minimised:
        window.showMinimized()
    else:
        window.show()
    if shown is not None:
        shown[0].set()
    app.exec()
    return True
