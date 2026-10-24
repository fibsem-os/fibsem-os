"""The colour a point overlay is asked for has to reach the marker.

`PointsOverlay` drew every marker with a hardcoded white edge. Filled markers (o, s)
were fine -- a white outline is what makes them legible against a bright image -- but an
unfilled one (+, x) has no face and is drawn *entirely* in its edge colour, so it came
out white whatever colour it was given. The label beside it kept the colour, which is
what made it look intentional rather than broken.

Every `PointsOverlay` in the codebase uses `+`, so every one of them was affected: the
FM overview's stage position, its saved lamella positions, and the selected one -- three
markers meant to be told apart by colour, all drawn the same white.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_point_overlay_colour.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay

_app = QApplication.instance() or QApplication(sys.argv)

CYAN = "#26c6da"


def _canvas_with(overlay):
    canvas = FibsemImageCanvas()
    canvas.add_overlay(overlay)
    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))
    return canvas


def _markers(overlay):
    """The overlay's marker artists, excluding the annotations."""
    from matplotlib.lines import Line2D

    return [a for a in overlay._artists if isinstance(a, Line2D)]


def test_an_unfilled_marker_is_drawn_in_the_colour_it_was_given():
    """The bug. `+` has no face, so its edge colour is the whole glyph."""
    overlay = PointsOverlay(color=CYAN, marker="+", size=11)
    canvas = _canvas_with(overlay)

    overlay.set_points([(10.0, 10.0)])

    marker = _markers(overlay)[0]
    assert marker.get_markeredgecolor() == CYAN
    assert marker.get_color() == CYAN


def test_a_filled_marker_keeps_its_white_outline():
    """Unchanged, and deliberately: a filled marker has a face to carry the colour, and
    the white outline is what keeps it legible against a bright image."""
    overlay = PointsOverlay(color=CYAN, marker="o", size=8)
    canvas = _canvas_with(overlay)

    overlay.set_points([(10.0, 10.0)])

    marker = _markers(overlay)[0]
    assert marker.get_markeredgecolor() == "white"
    assert marker.get_color() == CYAN


def test_two_overlays_of_different_colours_do_not_come_out_the_same():
    """What this actually protects. The FM overview distinguishes the stage position
    from saved positions from the selected one *by colour alone* -- all three are
    crosshairs -- so a marker that ignores its colour makes them indistinguishable."""
    saved = PointsOverlay(color=CYAN, marker="+", size=11)
    selected = PointsOverlay(color="#76ff03", marker="+", size=15)
    canvas = FibsemImageCanvas()
    canvas.add_overlay(saved)
    canvas.add_overlay(selected)
    canvas.set_array(np.zeros((64, 64), dtype=np.uint8))

    saved.set_points([(10.0, 10.0)])
    selected.set_points([(20.0, 20.0)])

    assert (
        _markers(saved)[0].get_markeredgecolor()
        != _markers(selected)[0].get_markeredgecolor()
    )


def test_the_label_and_the_marker_agree():
    """They disagreed before -- the label took the colour and the marker did not, which
    is exactly what made it read as a design choice rather than a bug."""
    from matplotlib.text import Annotation

    overlay = PointsOverlay(color=CYAN, marker="+", size=11)
    canvas = _canvas_with(overlay)

    overlay.set_points([(10.0, 10.0)], labels=["Lamella-01"])

    label = next(a for a in overlay._artists if isinstance(a, Annotation))
    assert label.get_color() == _markers(overlay)[0].get_markeredgecolor()
