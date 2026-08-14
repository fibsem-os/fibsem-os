"""Shift+scroll on the FM canvas says where the objective is going (FIB-635),
and the gesture reaches the canvas at all on a macOS mouse (FIB-552).

The beam canvases have flashed the working distance on this gesture since the quad view
shipped. The FM -- the canvas you focus by while watching the stream -- showed nothing,
and its info-bar `OBJECTIVE` field is deliberately behind during a burst, because
FIB-625's `_wheel_is_driving` suppresses the intermediates rather than let the number
snap backwards past the user's own scrolling.

Underneath both: `FigureCanvasQT.wheelEvent` reads only the vertical delta and emits
*nothing* when it is zero, while macOS turns Shift+wheel into horizontal scrolling before
Qt sees it. Every Shift+scroll feature was silently dead there.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QWidget

import numpy as np

from fibsem.constants import METRE_TO_MICRON
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.fm.widgets.objective_control_widget import ObjectiveControlWidget
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


class _Host(QWidget):
    is_acquisition_active = False
    viewer = None
    microscope = None

    def _view_controller(self):
        return None


@pytest.fixture
def fm():
    return FluorescenceMicroscope()


@pytest.fixture
def widget(fm):
    w = ObjectiveControlWidget(fm, parent=_Host())
    yield w
    w.cleanup()
    w.close()


@pytest.fixture
def canvas():
    c = FibsemImageCanvas()
    c.resize(400, 400)
    c.set_array(np.zeros((64, 64), dtype=np.uint8), pixel_size=1e-8)
    return c


def _wheel(dx: int, dy: int, modifiers=Qt.ShiftModifier) -> QWheelEvent:
    """A wheel event with the delta on the axis of our choosing.

    macOS puts a Shift+wheel magnitude on x with y zero; Windows and Linux put it on y.
    """
    pos = QPoint(200, 200)
    return QWheelEvent(
        pos, pos, QPoint(0, 0), QPoint(dx, dy), Qt.NoButton, modifiers,
        Qt.NoScrollPhase, False,
    )


class TestTheGestureSurvivesAHorizontalDelta:
    """FIB-552. Not reproducible on this machine's platform alone -- the event is
    synthesised in the shape macOS delivers, which is the only part that differs."""

    def test_a_horizontal_shift_wheel_reaches_the_canvas(self, canvas):
        seen = []
        canvas.canvas_scrolled.connect(lambda x, y, d, m: seen.append((d, m)))

        canvas.wheelEvent(_wheel(dx=120, dy=0))

        assert seen, "matplotlib dropped it and nothing rescued it (FIB-552)"
        direction, mods = seen[0]
        assert direction == 1
        assert "Shift" in mods

    def test_the_direction_follows_the_sign(self, canvas):
        seen = []
        canvas.canvas_scrolled.connect(lambda x, y, d, m: seen.append(d))

        canvas.wheelEvent(_wheel(dx=-120, dy=0))

        assert seen == [-1]

    def test_a_normal_vertical_wheel_is_untouched(self, canvas):
        """Windows and Linux take this path, and it must keep going through
        matplotlib rather than the rescue."""
        seen = []
        canvas.canvas_scrolled.connect(lambda x, y, d, m: seen.append(d))

        canvas.wheelEvent(_wheel(dx=0, dy=120))

        assert seen == [1]

    def test_an_unmodified_horizontal_wheel_is_left_alone(self, canvas):
        """The rescue is scoped to modified scrolls. A bare horizontal wheel is a
        legitimate no-op, not something to synthesise a zoom out of."""
        seen = []
        canvas.canvas_scrolled.connect(lambda x, y, d, m: seen.append(d))

        canvas.wheelEvent(_wheel(dx=120, dy=0, modifiers=Qt.NoModifier))

        assert seen == []


class TestTheFlashSaysWhereItIsGoing:
    """FIB-635."""

    @pytest.fixture(autouse=True)
    def _wire(self, widget, canvas):
        """Connect once. Qt permits duplicate connections, so reconnecting per scroll
        would fire the handler once per previous call and the objective would run away."""
        canvas.canvas_scrolled.connect(widget._on_canvas_scroll)

    @staticmethod
    def _scroll(widget, canvas, direction=+1):
        """Drive the handler the way the canvas does, so `sender()` resolves."""
        canvas.canvas_scrolled.emit(0.0, 0.0, direction, ("Shift",))

    def test_it_flashes_the_target_and_the_step(self, fm, widget, canvas):
        widget.doubleSpinBox_objective_step_size.setValue(1.0)
        base = widget.doubleSpinBox_objective_position.value()

        self._scroll(widget, canvas)

        assert canvas._flash_text == f"OBJ {base + 1.0:.1f} um  (+1.0 um)"

    def test_the_step_is_signed_so_it_shows_direction(self, fm, widget, canvas):
        widget.doubleSpinBox_objective_step_size.setValue(2.5)

        self._scroll(widget, canvas, direction=-1)

        assert "(-2.5 um)" in canvas._flash_text

    def test_it_shows_the_step_currently_set_not_the_distance_travelled(
        self, fm, widget, canvas
    ):
        """The point of showing it: the step is what silently changes what the gesture
        means. Three notches at 1 um is still '+1.0 um', not '+3.0 um'."""
        widget.doubleSpinBox_objective_step_size.setValue(1.0)
        base = widget.doubleSpinBox_objective_position.value()

        for _ in range(3):
            self._scroll(widget, canvas)

        assert canvas._flash_text == f"OBJ {base + 3.0:.1f} um  (+1.0 um)"

    def test_a_refused_scroll_flashes_nothing(self, fm, widget, canvas):
        """Nothing is going anywhere, so there is no target to show."""
        widget._action_worker = object()  # a retract in flight

        self._scroll(widget, canvas)

        assert canvas._flash_text is None

    def test_a_plain_scroll_flashes_nothing(self, fm, widget, canvas):
        canvas.canvas_scrolled.emit(0.0, 0.0, +1, ())

        assert canvas._flash_text is None

    def test_it_does_not_disturb_the_info_bar(self, fm, widget, canvas):
        """The two surfaces answer different questions -- the flash is the command,
        the info bar is the device -- so the flash must not write the field."""
        canvas.set_info_text("OBJECTIVE: 0.0 um")

        self._scroll(widget, canvas)

        assert canvas._info_text == "OBJECTIVE: 0.0 um"

    def test_no_canvas_is_not_an_error(self, fm, widget):
        """The napari host has no `FibsemCanvas` behind the callback, and the standalone
        objective widget has no host at all."""
        widget._flash_focus_target(1.0, 1.0)  # sender() is None here; must not raise
