"""Start the desktop apps behind a splash screen.

The ``fibsem-autolamella-ui`` and ``fibsem_ui`` entry points, and the way into
any app registered in fibsem.tools.apps. Importing an app takes seconds (most of
it ``fibsem.ui``, which builds every widget module), and until then a launch
shows nothing, so people click the shortcut again. This module puts the app's
icon and name on screen first, then imports the app.

It must stay cheap to import: the standard library, ``fibsem`` itself and
Qt-free modules only. Qt is imported inside the functions, as everywhere outside
a ui/ path (tests/test_ui_extra_isolation.py): core installs have no ui extra. Anything under ``fibsem.ui`` would load the widgets before
the splash screen appears and defeat it; tests/ui/test_launcher.py checks that.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import fibsem
from fibsem.theme.tokens import (
    BORDER_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.tools.app_identity import set_app_identity
from fibsem.tools.apps import App, apps

if TYPE_CHECKING:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QSplashScreen

SPLASH_WIDTH = 440
SPLASH_HEIGHT = 240
ICON_SIZE = 96


def splash_pixmap(
    icon: Path, title: str, caption: str, device_pixel_ratio: float = 1.0
) -> QPixmap:
    """Draw a splash screen: icon, title and caption on the dialog surface."""
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

    icon_x = (SPLASH_WIDTH - ICON_SIZE) / 2
    QSvgRenderer(str(icon)).render(painter, QRectF(icon_x, 36, ICON_SIZE, ICON_SIZE))

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
    painter.end()
    return pixmap


def show_splash(app: App) -> QSplashScreen:
    """Name the process, create the QApplication and put the splash on screen.

    The app identity goes before the QApplication exists, as the OS reads it then.
    """
    from PyQt5.QtWidgets import QApplication, QSplashScreen

    set_app_identity(app.name, windows_app_id=app.app_id)
    qapp = QApplication.instance() or QApplication(sys.argv)
    version = fibsem.__version__
    caption = version if app.name == "fibsemOS" else f"fibsemOS {version}"
    splash = QSplashScreen(
        splash_pixmap(app.icon, app.name, caption, qapp.devicePixelRatio())
    )
    splash.show()
    qapp.processEvents()  # paint it now: the app's import blocks the event loop
    return splash


def launch(key: str, argv: Optional[List[str]] = None) -> None:
    """Run the app registered as ``key`` (see fibsem.tools.apps) behind its splash."""
    app = apps()[key]
    argv = sys.argv[1:] if argv is None else argv
    # --help prints and exits without a window, so it gets no splash screen.
    splash = None if "-h" in argv or "--help" in argv else show_splash(app)
    module, function = app.target.split(":")
    run = getattr(importlib.import_module(module), function)
    if splash is None:
        run(argv)
    else:
        run(argv, splash=splash)


# The console commands. Their names are what existing installs' scripts call.
def autolamella(argv: Optional[List[str]] = None) -> None:
    """``fibsem-autolamella-ui``."""
    launch("autolamella", argv)


def fibsem_ui(argv: Optional[List[str]] = None) -> None:
    """``fibsem_ui``."""
    launch("fibsem-ui", argv)
