"""Headless smoke: point-overlay + canvas legends render the actual marker glyph.

The lamella-editor POI is a "+" marker with a legend; the swatch must be a "+" (Line2D),
not a filled square (Patch). Also covers FibsemImageCanvas.set_legend's per-entry marker.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_point_overlay_legend.py
"""
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay

_app = QApplication.instance() or QApplication(sys.argv)


def _canvas():
    c = FibsemImageCanvas()
    c.set_array(np.zeros((32, 32), dtype=np.uint8))
    return c


def test_point_overlay_legend_swatch_is_the_marker():
    c = _canvas()
    ov = PointOverlay(
        color="magenta", marker="+", edge_width=1.2,
        legend_label="Point of Interest", add_on_right_click=False, removable=False,
    )
    c.add_overlay(ov)
    ov.set_points([(16, 16)])
    assert ov._legend is not None
    lines = ov._legend.get_lines()
    assert len(lines) == 1 and lines[0].get_marker() == "+"


def test_set_legend_marker_vs_patch_entries():
    c = _canvas()
    c.set_legend([("magenta", "POI", "+"), ("limegreen", "area")])
    leg = c._legend_artist
    assert leg is not None
    lines, patches = leg.get_lines(), leg.get_patches()
    assert len(lines) == 1 and lines[0].get_marker() == "+"
    assert len(patches) == 1  # the 2-tuple entry stays a filled swatch


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
