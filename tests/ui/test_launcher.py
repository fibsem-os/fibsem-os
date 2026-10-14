"""The launcher puts a splash screen up before the apps' slow imports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtGui import QColor

import fibsem
from fibsem.tools import launcher


def test_importing_the_launcher_does_not_load_fibsem_ui():
    # fibsem.ui is most of the startup time; loading it before the splash shows
    # would leave the splash nothing to cover. A fresh interpreter, because this
    # one has long since imported fibsem.ui for other tests.
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, fibsem.tools.launcher; "
            "print(sorted(m for m in sys.modules if m.startswith('fibsem.ui')))",
        ],
        capture_output=True,
        text=True,
        check=True,
        # The same fibsem this test imported, not whichever the environment finds.
        env={**os.environ, "PYTHONPATH": str(Path(fibsem.__file__).parents[1])},
    ).stdout.strip()

    assert loaded == "[]"


@pytest.mark.parametrize(
    "drawing, tile",
    [("autolamella.svg", "#007ACC"), ("fibsemos.svg", "#262930")],
)
def test_splash_draws_the_app_icon_on_the_surface(qapp, drawing, tile):
    image = launcher.splash_pixmap(
        launcher.ICONS_DIR / drawing, "Name", "caption", device_pixel_ratio=2.0
    ).toImage()

    assert (image.width(), image.height()) == (
        2 * launcher.SPLASH_WIDTH,
        2 * launcher.SPLASH_HEIGHT,
    )
    # Inside the icon's tile, clear of its drawing: the tile colour.
    icon_left = (launcher.SPLASH_WIDTH - launcher.ICON_SIZE) // 2
    assert image.pixelColor(2 * (icon_left + 3), 2 * (36 + 48)) == QColor(tile)
    # A corner: the dialog surface.
    assert image.pixelColor(20, 20) == QColor(launcher.SURFACE_COLOR)
