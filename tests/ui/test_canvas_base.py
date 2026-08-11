"""Headless smoke: the FibsemCanvasBase / FibsemImageCanvas seam.

The base owns navigation, overlays, toolbar and chrome but knows nothing about what is
drawn; subclasses answer one question, _content_extent(), and the base fits the view to
whatever rectangle that returns. These pin that contract — the only thing the extraction
changed, since _fit_view previously read the image artist's extent directly.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_base.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.canvas_base import FibsemCanvasBase
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _img(h, w):
    return np.zeros((h, w), dtype=np.uint8)


def test_image_canvas_is_a_base_canvas():
    assert issubclass(FibsemImageCanvas, FibsemCanvasBase)


def test_bare_base_has_no_content():
    """The base draws nothing of its own, so it reports an empty extent."""
    assert FibsemCanvasBase()._content_extent() is None


def test_empty_canvas_fit_is_a_no_op():
    """reset_view() on a canvas with no content must not raise or move the axes."""
    c = FibsemCanvasBase()
    before = (c._ax.get_xlim(), c._ax.get_ylim())
    c.reset_view()
    assert (c._ax.get_xlim(), c._ax.get_ylim()) == before


def test_image_canvas_reports_the_image_extent():
    c = FibsemImageCanvas()
    assert c._content_extent() is None  # nothing set yet
    c.set_array(_img(32, 64))
    # matplotlib's extent for an image drawn at the origin, as set_array supplies it
    assert tuple(c._content_extent()) == (-0.5, 63.5, 31.5, -0.5)


def test_fit_view_frames_the_content_extent():
    """Routing through the hook must frame exactly what reading the artist did."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    c.reset_view()
    assert c._ax.get_xlim() == (-0.5, 63.5)
    assert c._ax.get_ylim() == (31.5, -0.5)  # y inverted (origin upper)


def test_view_margin_expands_the_content_extent():
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    c.set_view_margin(0.5)  # half the span of empty space on each side
    x0, x1 = c._ax.get_xlim()
    assert (x0, x1) == (-32.5, 95.5)  # 64-wide extent grown by 32 per side


def test_a_subclass_extent_drives_the_fit():
    """The seam is usable by any subclass, not just the image canvas — this is what
    the real-space canvas will rely on to frame images placed in stage space."""

    class _FakeContentCanvas(FibsemCanvasBase):
        def _content_extent(self):
            return (100.0, 300.0, 250.0, 50.0)

    c = _FakeContentCanvas()
    c.reset_view()
    assert c._ax.get_xlim() == (100.0, 300.0)
    assert c._ax.get_ylim() == (250.0, 50.0)


# ── chrome toggles keep their buttons honest ──────────────────────────────


def test_setting_crosshair_visibility_syncs_its_button():
    """The setter, not only the toggle, has to sync — a widget picking its own
    default at construction would otherwise show a checked button whose tooltip
    offers to hide a crosshair that is not drawn."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32))

    c.set_crosshair_visible(False)

    assert c._crosshair_artists == []
    assert c.btn_toggle_crosshair.isChecked() is False
    assert c.btn_toggle_crosshair.toolTip() == "Show crosshair"


def test_setting_scalebar_visibility_syncs_its_button():
    """`set_scalebar_visible` is new: the shared canvas had only a toggle, which
    cannot express "off" without first knowing what it is currently set to."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)

    assert c._scalebar_artist is not None  # drawn, because pixel_size is known

    c.set_scalebar_visible(False)

    assert c._scalebar_artist is None  # actually removed, not just flagged
    assert c.btn_toggle_scalebar.isChecked() is False
    assert c.btn_toggle_scalebar.toolTip() == "Show scalebar"


@pytest.mark.parametrize("noun", ["crosshair", "scalebar"])
def test_the_toolbar_toggle_still_round_trips(noun):
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)
    toggle = getattr(c, f"toggle_{noun}")
    button = getattr(c, f"btn_toggle_{noun}")

    toggle()
    assert button.isChecked() is False and button.toolTip() == f"Show {noun}"
    toggle()
    assert button.isChecked() is True and button.toolTip() == f"Hide {noun}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")


def test_the_scalebar_is_drawn_at_the_smaller_font():
    """The size is pinned because the constructor is wrapped in a bare except.

    A kwarg matplotlib_scalebar does not accept is swallowed there and the bar simply
    stops appearing, with nothing in the log to say why -- so "it still constructs"
    is as much the point of this test as the size itself (FIB-583).
    """
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)

    assert c._scalebar_artist is not None
    assert c._scalebar_artist.font_properties.get_size() == 8.0

    c.draw()  # a bad font spec raises here rather than at construction
