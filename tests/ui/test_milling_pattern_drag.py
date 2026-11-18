"""A left-drag on a milling pattern in the FIB view moves it.

The drag changes the point the pattern is placed at, nothing else: the overlay reports
how far the cursor has gone from where it pressed, and the milling viewer writes that
onto the stage. A press off every pattern is left to the canvas, which pans.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QApplication

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import LinePattern, RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures.image import FibsemImage, FibsemImageMetadata
from fibsem.structures.settings import (
    BeamType,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.milling_overlay import (
    MillingPatternOverlay,
    PatternDrag,
)
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

_app = QApplication.instance() or QApplication(sys.argv)

SIZE = 512
PIXEL = 2e-7  # a ~102 um field


@pytest.fixture()
def fib_image() -> FibsemImage:
    return FibsemImage(
        data=np.zeros((SIZE, SIZE), dtype=np.uint8),
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(
                resolution=(SIZE, SIZE), beam_type=BeamType.ION
            ),
            pixel_size=Point(PIXEL, PIXEL),
            microscope_state=MicroscopeState(),
        ),
    )


def _rectangle(name: str = "rectangle") -> FibsemMillingStage:
    return FibsemMillingStage(
        name=name, pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6)
    )


def _line() -> FibsemMillingStage:
    pattern = LinePattern(start_x=-5e-6, start_y=20e-6, end_x=5e-6, end_y=20e-6)
    pattern.point = Point(0.0, 20e-6)
    return FibsemMillingStage(name="line", pattern=pattern)


# ── the overlay: the gesture ──────────────────────────────────────────────────


@pytest.fixture()
def overlay(fib_image):
    canvas = FibsemImageCanvas()
    canvas.set_image(fib_image)
    overlay = MillingPatternOverlay()
    canvas.add_overlay(overlay)
    overlay.set_stages([_rectangle()], fib_image)
    overlay.drags = []
    overlay.on_drag = overlay.drags.append
    yield overlay
    canvas.deleteLater()


def test_a_press_on_a_pattern_starts_a_drag_that_reports_the_distance(overlay):
    assert overlay.press(1e-6, 1e-6, ("Shift",))
    overlay.move(4e-6, -1e-6)
    overlay.release()

    assert [d.phase for d in overlay.drags] == ["press", "move", "release"]
    assert all(d.index == 0 and d.modifiers == ("Shift",) for d in overlay.drags)
    assert (overlay.drags[-1].dx, overlay.drags[-1].dy) == pytest.approx((3e-6, -2e-6))


def test_a_press_off_every_pattern_is_not_a_drag(overlay):
    assert not overlay.press(20e-6, 20e-6)
    overlay.move(25e-6, 20e-6)
    overlay.release()
    assert overlay.drags == []


def test_an_overlay_that_is_not_draggable_takes_no_press(overlay, fib_image):
    overlay.set_stages([_rectangle()], fib_image, draggable=False)
    assert not overlay.press(1e-6, 1e-6)


def _mouse(canvas, name: str, x: float, y: float) -> None:
    """A left-button mouse event at image pixel (x, y), as matplotlib delivers it."""
    from matplotlib.backend_bases import MouseEvent

    sx, sy = canvas._ax.transData.transform((x, y))
    canvas.callbacks.process(name, MouseEvent(name, canvas, sx, sy, button=1))


def test_a_drag_on_a_pattern_does_not_pan_and_one_beside_it_does(overlay):
    canvas = overlay._canvas
    canvas.draw()
    centre = SIZE / 2
    view = canvas._ax.get_xlim()

    _mouse(canvas, "button_press_event", centre, centre)
    _mouse(canvas, "motion_notify_event", centre + 40, centre)
    _mouse(canvas, "button_release_event", centre + 40, centre)
    assert canvas._ax.get_xlim() == view
    assert overlay.drags[-1].dx == pytest.approx(40 * PIXEL)

    _mouse(canvas, "button_press_event", centre, centre + 100)
    _mouse(canvas, "motion_notify_event", centre + 40, centre + 100)
    _mouse(canvas, "button_release_event", centre + 40, centre + 100)
    assert canvas._ax.get_xlim() != view


# ── the viewer: the drag changes the point ────────────────────────────────────


@pytest.fixture()
def widget(microscope, fib_image):
    w = MillingTaskViewerWidget(
        microscope=microscope,
        milling_task_config=FibsemMillingTaskConfig(
            stages=[_rectangle("first"), _line()]
        ),
        milling_enabled=True,
    )
    w.set_fib_image(fib_image)
    yield w
    w.deleteLater()


def _points(widget):
    stages = widget.config_widget.milling_stages_widget.get_enabled_stages()
    return [(s.pattern.point.x, s.pattern.point.y) for s in stages]


def _drag(widget, index: int, dx: float, dy: float, modifiers=()) -> None:
    for phase, x, y in (("press", 0.0, 0.0), ("move", dx, dy), ("release", dx, dy)):
        widget._on_overlay_edited(
            BeamType.ION, "milling", PatternDrag(index, x, y, phase, modifiers)
        )
        _app.processEvents()  # the selection a press makes lands on the next turn


def test_a_drag_changes_the_point_of_the_pattern_it_grabbed(widget):
    _drag(widget, 0, 3e-6, -2e-6)

    assert _points(widget) == [pytest.approx((3e-6, -2e-6)), (0.0, 20e-6)]
    stages_widget = widget.config_widget.milling_stages_widget
    assert stages_widget._list._selected_stage.name == "first"
    shown = stages_widget._pattern_widget.get_pattern().point
    assert (shown.x, shown.y) == pytest.approx((3e-6, -2e-6))


def test_a_shift_drag_moves_every_pattern_and_a_line_keeps_its_ends(widget):
    _drag(widget, 0, 3e-6, 1e-6, ("Shift",))

    assert _points(widget) == [
        pytest.approx((3e-6, 1e-6)),
        pytest.approx((3e-6, 21e-6)),
    ]
    line = widget.config_widget.milling_stages_widget.get_enabled_stages()[1].pattern
    assert (line.start_x, line.end_x) == pytest.approx((-2e-6, 8e-6))


def test_a_drag_stops_at_the_edge_of_the_image(widget):
    _drag(widget, 0, 3e-6, 0.0)
    widget._on_overlay_edited(
        BeamType.ION, "milling", PatternDrag(0, 0.0, 0.0, "press")
    )
    widget._on_overlay_edited(
        BeamType.ION, "milling", PatternDrag(0, 200e-6, 0.0, "move")
    )

    assert _points(widget)[0] == pytest.approx((3e-6, 0.0))


def test_a_drag_announces_the_change_once_when_it_ends(widget):
    announced = []
    widget.settings_changed.connect(announced.append)

    widget._on_overlay_edited(
        BeamType.ION, "milling", PatternDrag(0, 0.0, 0.0, "press")
    )
    for step in (1e-6, 2e-6, 3e-6):
        widget._on_overlay_edited(
            BeamType.ION, "milling", PatternDrag(0, step, 0.0, "move")
        )
    _app.processEvents()
    assert announced == []

    widget._on_overlay_edited(
        BeamType.ION, "milling", PatternDrag(0, 3e-6, 0.0, "release")
    )
    _app.processEvents()
    assert len(announced) == 1


def test_a_drag_is_refused_while_milling(widget):
    class _RunningThread:
        def is_alive(self) -> bool:
            return True

    widget.milling_widget._milling_thread = _RunningThread()
    _drag(widget, 0, 3e-6, -2e-6)

    assert _points(widget)[0] == (0.0, 0.0)
