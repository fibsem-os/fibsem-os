"""Start the desktop apps behind a splash screen.

The ``fibsem-autolamella-ui`` and ``fibsem_ui`` entry points. Importing either
app takes seconds (most of it ``fibsem.ui``, which builds every widget module),
and until then a launch shows nothing, so people click the shortcut again. This
module puts the app's icon and name on screen first, then imports the app.

It must stay cheap to import: the standard library, ``fibsem`` itself and
Qt-free modules only. Qt is imported inside the functions, as everywhere outside
a ui/ path (tests/test_ui_extra_isolation.py): core installs have no ui extra. Anything under ``fibsem.ui`` would load the widgets before
the splash screen appears and defeat it; tests/ui/test_launcher.py checks that.
"""

from __future__ import annotations

import importlib.resources
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

if TYPE_CHECKING:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QSplashScreen

# The drawings fibsem.ui.icon renders, found without importing fibsem.ui.
ICONS_DIR = Path(str(importlib.resources.files("fibsem"))) / "ui" / "icons"

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


def show_splash(
    name: str, windows_app_id: str, icon: Path, caption: str
) -> QSplashScreen:
    """Name the process, create the QApplication and put the splash on screen.

    The app identity goes before the QApplication exists, as the OS reads it then.
    """
    from PyQt5.QtWidgets import QApplication, QSplashScreen

    set_app_identity(name, windows_app_id=windows_app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    splash = QSplashScreen(splash_pixmap(icon, name, caption, app.devicePixelRatio()))
    splash.show()
    app.processEvents()  # paint it now: the app's import blocks the event loop
    return splash


def autolamella(argv: Optional[List[str]] = None) -> None:
    """Show the AutoLamella splash screen, then import and run the UI."""
    argv = sys.argv[1:] if argv is None else argv
    if "-h" in argv or "--help" in argv:
        # Print the usage and exit; no window, so no splash screen either.
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import run_ui

        run_ui(argv)
        return

    splash = show_splash(
        "AutoLamella",
        "fibsemOS.AutoLamella",
        ICONS_DIR / "autolamella.svg",
        f"fibsemOS {fibsem.__version__}",
    )
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import run_ui

    run_ui(argv, splash=splash)


def fibsem_ui() -> None:
    """Show the fibsemOS splash screen, then import and run the fibsemOS UI."""
    splash = show_splash(
        "fibsemOS",
        "fibsemOS.FibsemUI",
        ICONS_DIR / "fibsemos.svg",
        fibsem.__version__,
    )
    from fibsem.ui.FibsemUI import main

    main(splash=splash)
