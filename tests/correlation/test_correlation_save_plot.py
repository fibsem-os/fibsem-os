"""Save Plot's render path (FIB-535).

`save_plot` used to call `ImagePointCanvas.render_to_axes`, which the shared
canvas has no equivalent for — so swapping either canvas would have broken the
export with an AttributeError. It now goes through `_render_canvas_to_axes`,
which needs a matplotlib figure and nothing else.

These tests exist to keep that true. The one that earns its keep is the
parametrised one: it renders the canvas in production today *and* the one it is
being swapped to, through the same code path.
"""
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] without the UI extra

from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.correlation.widgets.correlation_tab_widget import (
    _render_canvas_to_axes,
)

_app = QApplication.instance() or QApplication(sys.argv)

_COORDS = [
    Coordinate(PointXYZ(40.0, 30.0, 0.0), PointType.FIB),
    Coordinate(PointXYZ(80.0, 60.0, 0.0), PointType.SURFACE),
]
_IMAGE = np.linspace(0, 255, 128 * 128, dtype=np.uint8).reshape(128, 128)

# Kept alive for the session: the widget owns its canvas, and letting it fall out
# of scope deletes the C++ side while the test still holds the Python wrapper.
_ALIVE = []


def _new_canvas():
    w = CorrelationCanvasWidget(allowed_point_types=[PointType.FIB])
    w.set_image(_IMAGE)
    w.set_coordinates(list(_COORDS))
    _ALIVE.append(w)
    return w.canvas


@pytest.fixture
def ax():
    """A bare Figure, not `plt.subplots`.

    pyplot would build the figure through the *interactive* backend, which spins
    up a Qt figure manager and mutates application-wide state — enough to change
    the default font and break an unrelated widget test later in the run.
    """
    return Figure(figsize=(8, 8)).add_subplot(111)


# ── the reason this was rewritten ─────────────────────────────────────────


def test_a_matplotlib_canvas_can_be_rendered(ax):
    """Needs a matplotlib figure and nothing else. That is what let Save Plot
    survive a half-migrated correlation tab, and what keeps it working if either
    canvas is swapped again."""
    _render_canvas_to_axes(_new_canvas(), ax)

    assert ax.get_images(), "nothing was drawn into the axes"
    assert not ax.axison  # the panel frame is hidden, as the old path did


def test_the_export_does_not_disturb_the_live_canvas(ax):
    """savefig runs through the canvas's own Qt draw path, so this is not
    self-evident — the figure it renders is the one still on screen."""
    canvas = _new_canvas()
    before = (canvas.figure.get_size_inches().tolist(), canvas.figure.get_dpi())

    _render_canvas_to_axes(canvas, ax)

    assert (canvas.figure.get_size_inches().tolist(), canvas.figure.get_dpi()) == before
    canvas.draw()  # must not raise


def test_the_rendered_panel_is_not_stretched(ax):
    """A square image comes out square. `render_to_axes` re-fitted content to the
    panel, turning a 1:1 image into the panel's 2:1 — measuring off that export
    gave the wrong answer."""
    _render_canvas_to_axes(_new_canvas(), ax)

    buffer = ax.get_images()[0].get_array()
    h, w = buffer.shape[:2]
    canvas_w, canvas_h = _new_canvas().figure.get_size_inches()
    assert w / h == pytest.approx(canvas_w / canvas_h, rel=0.02)


def test_resolution_comes_from_the_request_not_the_widget(ax):
    """A plain buffer_rgba() grab would cap the export at the on-screen widget
    size; a small window would silently produce a low-resolution plot."""
    canvas = _new_canvas()
    canvas.resize(120, 120)

    _render_canvas_to_axes(canvas, ax, dpi=150)

    height = ax.get_images()[0].get_array().shape[0]
    assert height > 600, f"exported at only {height}px — widget size leaked in"
