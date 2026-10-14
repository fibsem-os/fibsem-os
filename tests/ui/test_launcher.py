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
from fibsem.tools.apps import ICONS_DIR


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
        ICONS_DIR / drawing, "Name", "caption", device_pixel_ratio=2.0
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


# AutoLamella's milling splash, sampled in the icon's own 32-unit grid.
def _icon_pixel(image, u, v):
    left = (launcher.SPLASH_WIDTH - launcher.ICON_SIZE) / 2
    scale = launcher.ICON_SIZE / 32
    return image.pixelColor(
        round(2 * (left + scale * u)), round(2 * (launcher.ICON_TOP + scale * v))
    )


def _frame(milling):
    from fibsem.tools.apps import AUTOLAMELLA

    return launcher.splash_pixmap(
        AUTOLAMELLA.icon, "AutoLamella", "", device_pixel_ratio=2.0, milling=milling
    ).toImage()


TRENCH, LAMELLA, TILE = QColor("#0B3A63"), QColor("#FFFFFF"), QColor("#007ACC")
MATERIAL = QColor("#5AA9E0")


def test_milling_ends_on_the_icon(qapp):
    milled = _frame(launcher.MILLED)
    static = _frame(None)
    # Mid-trench, the gap above the lamella, the lamella, the gap below, mid-trench.
    for v, colour in [
        (9, TRENCH),
        (14.5, TILE),
        (16, LAMELLA),
        (17.5, TILE),
        (23, TRENCH),
    ]:
        assert _icon_pixel(milled, 16, v) == colour, v
        assert _icon_pixel(static, 16, v) == colour, v


def test_the_rough_passes_cut_top_then_bottom(qapp):
    start = _frame(launcher.Milling.at(0.0))
    half = _frame(launcher.Milling.at(0.5))
    rough = _frame(launcher.Milling.at(1.0))

    assert _icon_pixel(start, 16, 6) == MATERIAL
    assert _icon_pixel(start, 16, 26) == MATERIAL
    assert _icon_pixel(half, 16, 6) == TRENCH  # top trench cut
    assert _icon_pixel(half, 16, 26) == MATERIAL  # bottom not yet
    assert _icon_pixel(rough, 16, 26) == TRENCH
    assert _icon_pixel(rough, 16, 16) == MATERIAL  # a thick slab, not yet polished


@pytest.mark.parametrize("key", ["autolamella", "fibsem-ui"])
def test_every_load_step_names_a_real_module(key):
    import importlib.util

    from fibsem.tools.apps import BUILTIN_APPS

    for status, modules in BUILTIN_APPS[key].load_steps:
        for name in modules:
            assert importlib.util.find_spec(name) is not None, (status, name)


def test_the_splash_polishes_then_closes_once_the_window_is_up(qapp):
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QWidget

    from fibsem.tools.apps import AUTOLAMELLA

    splash = launcher.Splash(AUTOLAMELLA)
    splash.step("Loading…")
    window = QWidget()
    window.show()
    splash.finish(window)

    assert splash._screen.isVisible()  # milling
    QTest.qWait(launcher.ANIMATION_MS + launcher.HOLD_MS + 400)
    assert not splash._screen.isVisible()
    window.close()


def test_a_click_closes_the_splash_while_it_mills(qapp):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QWidget

    from fibsem.tools.apps import AUTOLAMELLA

    splash = launcher.Splash(AUTOLAMELLA)
    window = QWidget()
    window.show()
    splash.finish(window)
    QTest.qWait(100)  # mid-milling

    QTest.mouseClick(splash._screen, Qt.MouseButton.LeftButton)
    QTest.qWait(50)

    assert not splash._screen.isVisible()
    assert not splash._timer.isActive()
    window.close()


def test_clicks_are_ignored_until_the_window_is_up(qapp):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QWidget

    from fibsem.tools.apps import AUTOLAMELLA

    splash = launcher.Splash(AUTOLAMELLA)
    splash.step("Loading…")

    QTest.mouseClick(splash._screen, Qt.MouseButton.LeftButton)
    QTest.qWait(50)
    assert splash._screen.isVisible()  # still loading: the click is ignored

    window = QWidget()
    window.show()
    splash.finish(window)
    QTest.mouseClick(splash._screen, Qt.MouseButton.LeftButton)
    QTest.qWait(50)
    assert not splash._screen.isVisible()  # window up: the click closes it
    window.close()
