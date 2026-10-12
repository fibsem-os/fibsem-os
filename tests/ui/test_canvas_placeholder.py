"""What an empty canvas says, and the quad's FM panel on a system with no FM."""

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.structures import FibsemImage  # noqa: E402
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


def _texts(canvas):
    return [t.get_text() for t in canvas._ax.texts]


def test_an_empty_canvas_says_no_image():
    assert "No image" in _texts(FibsemImageCanvas())


def test_a_placeholder_shows_at_once_on_an_empty_canvas():
    canvas = FibsemImageCanvas()
    canvas.set_placeholder("No fluorescence microscope available")

    assert _texts(canvas) == ["No fluorescence microscope available"]


def test_a_placeholder_does_not_cover_an_image():
    canvas = FibsemImageCanvas()
    canvas.set_image(FibsemImage.generate_blank_image(resolution=(8, 8)))
    canvas.set_placeholder("No fluorescence microscope available")

    assert "No fluorescence microscope available" not in _texts(canvas)
    canvas.clear()
    assert "No fluorescence microscope available" in _texts(canvas)


class _UI:
    def __init__(self, microscope):
        self.microscope = microscope


class _Scope:
    def __init__(self, fm):
        self.fm = fm


@pytest.mark.parametrize(
    "microscope, expected",
    [
        (_Scope(fm=None), "No fluorescence microscope available"),
        (_Scope(fm=object()), "No image"),
        (None, "No image"),
    ],
)
def test_the_fm_panel_says_when_there_is_no_fm(microscope, expected):
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

    host = AutoLamellaSingleWindowUI.__new__(AutoLamellaSingleWindowUI)
    host.autolamella_ui = _UI(microscope)
    host.view_controller = MicroscopeViewController()
    AutoLamellaSingleWindowUI._refresh_fm_placeholder(host)

    assert _texts(host.view_controller.fm_canvas) == [expected]
