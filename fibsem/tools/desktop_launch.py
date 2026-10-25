"""Start the desktop apps behind a splash screen.

The ``fibsem-autolamella-ui`` and ``fibsem_ui`` entry points, and the way into
any app registered in fibsem.tools.apps. Importing an app takes seconds, and
until then a launch shows nothing, so people click the shortcut again. This
module puts the app's icon and name on screen first, then imports the app in
the steps its registry entry names, saying which on the splash.

Importing blocks Qt's event loop, so the splash can only repaint between steps,
in jumps. So while the app loads it shows the status and holds still; once the
window is up and the event loop runs, AutoLamella's splash mills its icon the
way a lamella is made, on its own clock: the top trench rough-milled, then the
bottom, then both faces fine-polished at once. It plays in full however fast the
load was, then closes; two seconds in all, and a click closes it sooner. Clicks
before the window is up are ignored: there is nothing behind the splash yet.

It must stay cheap to import: the standard library, ``fibsem`` itself and
Qt-free modules only. Qt is imported inside the functions, as everywhere outside
a ui/ path (tests/test_ui_extra_isolation.py): core installs have no ui extra.
Anything under ``fibsem.ui`` would load the widgets before the splash screen
appears and defeat it; tests/ui/test_desktop_launch.py checks that.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import fibsem
from fibsem.theme.tokens import (
    BORDER_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.tools.app_identity import bring_to_front, set_app_identity
from fibsem.tools.apps import App, apps

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter, QPixmap
    from PyQt5.QtWidgets import QWidget

SPLASH_WIDTH = 440
SPLASH_HEIGHT = 240
ICON_SIZE = 96
ICON_TOP = 36
ANIMATION_MS = 1600  # rough top, rough bottom, polish: the whole milling
ROUGH_SHARE = 0.6  # of that, the two rough passes; the polish has the rest
HOLD_MS = 400  # the finished icon stays up this long before the splash closes

# AutoLamella's icon (fibsem/ui/icons/autolamella.svg), in its 32-unit grid. The
# milling ends exactly on that drawing: trenches 4-14 and 18-28, the lamella
# 15-17, and the tile showing in the 1-unit gaps either side of it.
_TILE = "#007ACC"
_TRENCH = "#0B3A63"
_LAMELLA = "#FFFFFF"
_MATERIAL = "#5AA9E0"  # the slab before milling: lighter than the tile
_ROUGH_TOP, _ROUGH_BOTTOM = 12.0, 20.0  # where rough milling stops
_FINE_TOP, _FINE_BOTTOM = 14.0, 18.0  # where polishing stops


@dataclass(frozen=True)
class Milling:
    """How far AutoLamella's icon has been milled, each pass from 0 to 1."""

    top: float = 0.0
    bottom: float = 0.0
    polish: float = 0.0
    beam: Optional[float] = None
    """Where the beam is along the cut, 0 to 1; None when it is off."""

    @classmethod
    def at(cls, progress: float, beam: Optional[float] = None) -> "Milling":
        """Rough milling at ``progress`` through the load: top, then bottom."""
        return cls(
            top=min(1.0, 2 * progress), bottom=max(0.0, 2 * progress - 1), beam=beam
        )


MILLED = Milling(top=1, bottom=1, polish=1)


def splash_pixmap(
    icon: Path,
    title: str,
    caption: str,
    device_pixel_ratio: float = 1.0,
    status: str = "",
    milling: Optional[Milling] = None,
) -> QPixmap:
    """Draw a splash screen: icon, title, caption and load status.

    With ``milling``, AutoLamella's icon is drawn milled that far instead of
    ``icon``.
    """
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
    from PyQt5.QtSvg import QSvgRenderer

    pixmap = QPixmap(
        round(SPLASH_WIDTH * device_pixel_ratio),
        round(SPLASH_HEIGHT * device_pixel_ratio),
    )
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    pixmap.fill(QColor(SURFACE_COLOR))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setPen(QPen(QColor(BORDER_COLOR), 1))
    painter.drawRect(QRectF(0.5, 0.5, SPLASH_WIDTH - 1, SPLASH_HEIGHT - 1))

    icon_rect = QRectF((SPLASH_WIDTH - ICON_SIZE) / 2, ICON_TOP, ICON_SIZE, ICON_SIZE)
    if milling is None:
        QSvgRenderer(str(icon)).render(painter, icon_rect)
    else:
        painter.save()
        painter.translate(icon_rect.topLeft())
        painter.scale(ICON_SIZE / 32, ICON_SIZE / 32)
        _draw_milling(painter, milling)
        painter.restore()

    title_font = QFont()
    title_font.setPointSize(20)
    title_font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title_font)
    painter.setPen(QColor(TEXT_STRONG_COLOR))
    painter.drawText(
        QRectF(0, 144, SPLASH_WIDTH, 32), Qt.AlignmentFlag.AlignCenter, title
    )

    caption_font = QFont()
    caption_font.setPointSize(11)
    painter.setFont(caption_font)
    painter.setPen(QColor(TEXT_MUTED_COLOR))
    painter.drawText(
        QRectF(0, 176, SPLASH_WIDTH, 20), Qt.AlignmentFlag.AlignCenter, caption
    )
    if status:
        status_font = QFont()
        status_font.setPointSize(10)
        painter.setFont(status_font)
        painter.drawText(
            QRectF(16, SPLASH_HEIGHT - 30, SPLASH_WIDTH - 32, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            status,
        )
    painter.end()
    return pixmap


def _draw_milling(painter: QPainter, milling: Milling) -> None:
    """Paint AutoLamella's icon milled as far as ``milling`` says, in grid units."""
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtGui import QColor

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def fill(x: float, y: float, w: float, h: float, colour, radius: float = 0):
        if w <= 0 or h <= 0:
            return
        painter.setBrush(QColor(colour))
        painter.drawRoundedRect(QRectF(x, y, w, h), radius, radius)

    painter.setPen(Qt.PenStyle.NoPen)
    fill(0, 0, 32, 32, _TILE, 7)

    # Each trench's milled edge: rough to 12 and 20, polishing on to 14 and 18.
    top_edge = (
        lerp(4, _ROUGH_TOP, milling.top) + (_FINE_TOP - _ROUGH_TOP) * milling.polish
    )
    bottom_edge = (
        lerp(28, _ROUGH_BOTTOM, milling.bottom)
        - (_ROUGH_BOTTOM - _FINE_BOTTOM) * milling.polish
    )
    # The slab left between them whitens and pulls back from the trench walls
    # as it is polished, ending as the lamella with the tile either side.
    gap = milling.polish
    material = QColor(_MATERIAL)
    lamella = QColor(_LAMELLA)
    slab = QColor.fromRgbF(
        lerp(material.redF(), lamella.redF(), milling.polish),
        lerp(material.greenF(), lamella.greenF(), milling.polish),
        lerp(material.blueF(), lamella.blueF(), milling.polish),
    )
    fill(4, top_edge + gap, 24, bottom_edge - top_edge - 2 * gap, slab)
    fill(4, 4, 24, top_edge - 4, _TRENCH, 1)
    fill(4, bottom_edge, 24, 28 - bottom_edge, _TRENCH, 1)

    if milling.beam is not None:
        fine = milling.polish > 0
        width, height = (2.0, 0.5) if fine else (4.0, 1.5)
        x = 4 + milling.beam * (24 - width)
        if fine or milling.top < 1:  # the top cut, rough or fine
            fill(x, top_edge - height, width, height, _LAMELLA)
        if fine or (milling.top >= 1 and milling.bottom < 1):  # the bottom cut
            fill(x, bottom_edge, width, height, _LAMELLA)


class Splash:
    """The splash screen while an app loads, and its close once the window is up.

    ``finish(window)`` is what the app calls; the rest is the launcher's.
    """

    def __init__(self, app: App):
        from PyQt5.QtCore import QEvent, QObject
        from PyQt5.QtWidgets import QApplication, QSplashScreen

        class IgnoreClicks(QObject):
            """Qt closes a splash on any click; while loading there is nothing
            behind it to get to, so a stray click would only hide the progress."""

            def eventFilter(self, watched, event) -> bool:
                return event.type() in (
                    QEvent.Type.MouseButtonPress,
                    QEvent.Type.MouseButtonDblClick,
                )

        # Before the QApplication exists: the OS reads the identity then.
        set_app_identity(app.name, windows_app_id=app.app_id)
        self._qapp = QApplication.instance() or QApplication(sys.argv)
        self._app = app
        version = fibsem.__version__
        self._caption = version if app.name == "fibsemOS" else f"fibsemOS {version}"
        self._milling = app.splash_animation == "milling"
        self._screen = QSplashScreen(
            self._frame("", Milling() if self._milling else None)
        )
        self._click_guard = IgnoreClicks()
        self._screen.installEventFilter(self._click_guard)
        self._screen.show()
        self._screen.raise_()
        bring_to_front()  # else macOS opens it behind the app it was launched from
        self._qapp.processEvents()
        self._timer = None

    def _frame(self, status: str, milling: Optional[Milling]) -> QPixmap:
        return splash_pixmap(
            self._app.icon,
            self._app.name,
            self._caption,
            self._qapp.devicePixelRatio(),
            status=status,
            milling=milling,
        )

    def step(self, status: str) -> None:
        """Say what is loading. The icon holds still: imports block the repaint."""
        self._screen.setPixmap(
            self._frame(status, Milling() if self._milling else None)
        )
        self._qapp.processEvents()

    def finish(self, window: QWidget) -> None:
        """Close once ``window`` is up; AutoLamella's first mills its icon.

        From here a click closes the splash: the window is up behind it.
        """
        self._screen.removeEventFilter(self._click_guard)
        if not self._milling:
            self._screen.finish(window)
            return
        from PyQt5.QtCore import QElapsedTimer, QTimer

        clock = QElapsedTimer()  # started on the first frame, not here: the
        self._timer = QTimer()  # window's first paint comes between the two
        self._timer.setInterval(16)

        def tick() -> None:
            if not self._screen.isVisible():  # clicked away: Qt hides it on click
                self._timer.stop()
                return
            if not clock.isValid():
                clock.start()
            elapsed = clock.elapsed()
            if elapsed >= ANIMATION_MS + HOLD_MS:  # shown the finished icon: close
                self._timer.stop()
                self._screen.finish(window)
                return
            t = min(1.0, elapsed / ANIMATION_MS)
            if t >= 1.0:
                milling = MILLED
            elif t < ROUGH_SHARE:
                rough = t / ROUGH_SHARE
                milling = Milling.at(rough, beam=(rough * 6) % 1)
            else:
                fine = (t - ROUGH_SHARE) / (1 - ROUGH_SHARE)
                polish = 1 - (1 - fine) ** 2  # ease out: fast at first, settling
                milling = Milling(1, 1, polish, beam=(fine * 3) % 1)
            self._screen.setPixmap(self._frame("Ready", milling))

        self._timer.timeout.connect(tick)
        self._timer.start()


def launch(key: str, argv: Optional[List[str]] = None) -> None:
    """Run the app registered as ``key`` (see fibsem.tools.apps) behind its splash."""
    app = apps()[key]
    argv = sys.argv[1:] if argv is None else argv
    module, function = app.target.split(":")
    if "-h" in argv or "--help" in argv:
        # Prints and exits without a window, so it gets no splash screen.
        getattr(importlib.import_module(module), function)(argv)
        return

    if not gateway_allows(key, app):
        return
    splash = Splash(app)
    for status, modules in app.load_steps:
        splash.step(f"{status}…")
        for name in modules:
            importlib.import_module(name)
    splash.step("Opening the window…")
    getattr(importlib.import_module(module), function)(argv, splash=splash)


def gateway_allows(key: str, app: App) -> bool:
    """Whether the window may open, asking the gateway on this PC when one runs
    (fibsem.gateway): with it off, the window takes the app's place; with the
    app running without its window, the operator chooses Take over (it is
    stopped) or Cancel. No gateway, or one that doesn't answer, changes
    nothing."""
    from urllib.error import HTTPError

    from fibsem.gateway.gateway import CHILD_ENV, OFF, STOP_WAIT_S
    from fibsem.gateway.main import call

    if os.environ.get(CHILD_ENV):
        return True
    try:
        state = call("GET", "/gateway")
    except (OSError, ValueError) as e:
        logging.warning("Gateway didn't answer (%s); opening as usual", e)
        return True
    if state is None or state.get("window"):
        return True
    if state["mode"] != OFF:
        if not _take_over(app, state):
            return False
        try:
            call(
                "POST",
                "/gateway/stop",
                {"force": True, "via": "desktop launch"},
                timeout=STOP_WAIT_S + 10,
            )
        except HTTPError as e:
            _say(
                app,
                f"The remote run didn't stop: {e.read().decode('utf-8', 'replace')}",
            )
            return False
        except (OSError, ValueError) as e:
            _say(app, f"The remote run didn't stop: {e}")
            return False
    try:
        call("POST", "/gateway/claim", {"pid": os.getpid(), "app": key})
    except (OSError, ValueError) as e:
        logging.warning("Gateway claim failed (%s); opening anyway", e)
    return True


def _take_over(app: App, state: dict) -> bool:
    """Ask the operator whether to stop the remote run and open the window."""
    box, take = take_over_box(app, state)
    box.exec_()
    return box.clickedButton() is take


def take_over_box(app: App, state: dict):
    """The Take over / Cancel question, and its Take over button."""
    from PyQt5.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication(sys.argv)
    experiment = state.get("experiment") or "no experiment"
    who = state.get("who") or "remotely"
    box = QMessageBox()
    box.setWindowTitle(app.name)
    box.setIcon(QMessageBox.Question)
    box.setText(f"{app.name} is running remotely ({experiment}, started by {who}).")
    box.setInformativeText(
        "Take over stops it, and any workflow it is running, then opens the window."
    )
    take = box.addButton("Take over", QMessageBox.AcceptRole)
    box.addButton(QMessageBox.Cancel)
    return box, take


def _say(app: App, text: str) -> None:
    from PyQt5.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning(None, app.name, text)


# The console commands. Their names are what existing installs' scripts call.
def autolamella(argv: Optional[List[str]] = None) -> None:
    """``fibsem-autolamella-ui``."""
    launch("autolamella", argv)


def fibsem_ui(argv: Optional[List[str]] = None) -> None:
    """``fibsem_ui``."""
    launch("fibsem-ui", argv)


def volume(argv: Optional[List[str]] = None) -> None:
    """``fibsem-volume-ui``."""
    launch("volume", argv)


if __name__ == "__main__":
    # How the gateway opens a window: python -m fibsem.tools.desktop_launch <app>
    launch(sys.argv[1], sys.argv[2:])
