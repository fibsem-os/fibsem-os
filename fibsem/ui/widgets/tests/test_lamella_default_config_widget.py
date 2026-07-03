"""Headless smoke for LamellaDefaultConfigWidget (canvas + editable overlays).

Verifies the two-way sync between the spinboxes and the canvas overlays:
  * set_template positions the alignment rect + POI marker (and is silent),
  * dragging the alignment rect / POI marker (simulated via the overlay signals)
    writes back to the spinboxes and emits template_changed,
  * editing a spinbox repositions the overlay,
  * get_template round-trips and is_valid still reflects the alignment area,
  * the POI overlay is attached before the alignment overlay (press precedence).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_lamella_default_config_widget.py
"""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

import numpy as np

from fibsem.applications.autolamella.structures import LamellaDefaultConfig
from fibsem.structures import FibsemRectangle, Point
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.lamella_default_config_widget import (
    LamellaDefaultConfigWidget,
    _pixel_to_poi,
    _poi_to_pixel,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def _widget(template: LamellaDefaultConfig | None = None) -> LamellaDefaultConfigWidget:
    w = LamellaDefaultConfigWidget()
    if template is not None:
        w.set_template(template)
    return w


_TPL = LamellaDefaultConfig(
    use_petname=False,
    name_prefix="GridA",
    alignment_area=FibsemRectangle(left=0.2, top=0.3, width=0.4, height=0.5),
    poi=Point(x=5e-6, y=-3e-6),
)


# ── transforms ─────────────────────────────────────────────────────────────

def test_poi_pixel_roundtrip():
    p = Point(x=7.5e-6, y=-4.2e-6)
    px, py = _poi_to_pixel(p)
    back = _pixel_to_poi(px, py)
    assert _close(back.x, p.x, tol=1e-12) and _close(back.y, p.y, tol=1e-12)


def test_poi_centre_maps_to_image_centre():
    px, py = _poi_to_pixel(Point(0, 0))
    assert _close(px, 384 / 2) and _close(py, 288 / 2)


# ── set_template → overlays ──────────────────────────────────────────────────

def test_set_template_positions_overlays():
    w = _widget(_TPL)
    area = w._aa_overlay.get_area()
    # the rect snaps to integer pixels, so allow ~1px of normalised error
    assert _close(area.left, 0.2, 0.005) and _close(area.top, 0.3, 0.005)
    assert _close(area.width, 0.4, 0.005) and _close(area.height, 0.5, 0.005)
    pt = w._poi_overlay.get_points()[0]
    assert _close(pt[0], _poi_to_pixel(_TPL.poi)[0]) and _close(pt[1], _poi_to_pixel(_TPL.poi)[1])


def test_set_template_is_silent():
    w = _widget()
    seen = []
    w.template_changed.connect(seen.append)
    w.set_template(_TPL)
    assert seen == []  # programmatic set must not emit


def test_name_hint_follows_petname():
    w = _widget(_TPL)
    assert w.canvas._hint_text == "GridA-Lamella-01"
    w.set_template(LamellaDefaultConfig(use_petname=True, name_prefix="GridA"))
    assert w.canvas._hint_text == "GridA-01-brave-tiger"


# ── overlay drag → spinbox + emit ────────────────────────────────────────────

def test_alignment_drag_updates_spinboxes_and_emits():
    w = _widget(_TPL)
    seen = []
    w.template_changed.connect(seen.append)
    new_area = FibsemRectangle(left=0.1, top=0.15, width=0.25, height=0.35)
    w._aa_overlay.alignment_area_changed.emit(new_area)
    assert _close(w.aa_left.value(), 0.1) and _close(w.aa_top.value(), 0.15)
    assert _close(w.aa_width.value(), 0.25) and _close(w.aa_height.value(), 0.35)
    assert len(seen) == 1
    assert _close(seen[0].alignment_area.left, 0.1)


def test_poi_move_updates_spinboxes_and_emits():
    w = _widget(_TPL)
    seen = []
    w.template_changed.connect(seen.append)
    px, py = 250.0, 100.0
    w._poi_overlay.point_moved.emit(0, px, py)
    expected = _pixel_to_poi(px, py)
    assert _close(w.poi_x.value(), expected.x * 1e6, 0.01)  # spinbox rounds to 2 dp
    assert _close(w.poi_y.value(), expected.y * 1e6, 0.01)
    assert len(seen) == 1
    assert _close(seen[0].poi.x, w.poi_x.value() * 1e-6, 1e-12)  # emit matches spinbox exactly


def test_poi_dragging_updates_spinboxes_without_emit():
    w = _widget(_TPL)
    seen = []
    w.template_changed.connect(seen.append)
    px, py = 300.0, 80.0
    w._poi_overlay.point_dragging.emit(0, px, py)
    expected = _pixel_to_poi(px, py)
    assert _close(w.poi_x.value(), expected.x * 1e6, 0.01)
    assert seen == []  # live drag feedback is silent until release


# ── spinbox → overlay ────────────────────────────────────────────────────────

def test_spinbox_edit_repositions_alignment_overlay():
    w = _widget(_TPL)
    seen = []
    w.template_changed.connect(seen.append)
    w.aa_left.setValue(0.05)  # distinct from _TPL (0.2) → valueChanged fires
    assert _close(w._aa_overlay.get_area().left, 0.05, 0.005)  # pixel-snapped
    assert len(seen) == 1


def test_spinbox_edit_repositions_poi_overlay():
    w = _widget(_TPL)
    w.poi_x.setValue(12.0)  # µm, distinct from _TPL (5 µm)
    px, _ = w._poi_overlay.get_points()[0]
    assert _close(px, _poi_to_pixel(Point(x=12e-6, y=_TPL.poi.y))[0])


# ── round-trip + validation + invariants ─────────────────────────────────────

def test_get_template_roundtrips():
    w = _widget(_TPL)
    t = w.get_template()
    assert t.use_petname is False and t.name_prefix == "GridA"
    assert _close(t.alignment_area.left, 0.2) and _close(t.alignment_area.height, 0.5)
    assert _close(t.poi.x, 5e-6, tol=1e-12) and _close(t.poi.y, -3e-6, tol=1e-12)


def test_is_valid_tracks_alignment_area():
    w = _widget(_TPL)
    assert w.is_valid() is True
    w.set_template(
        LamellaDefaultConfig(alignment_area=FibsemRectangle(left=0.8, top=0.0, width=0.5, height=0.5))
    )
    assert w.is_valid() is False  # left + width = 1.3 > 1


def test_poi_attached_before_alignment_for_press_precedence():
    w = _widget()
    overlays = w.canvas._overlays
    assert overlays.index(w._poi_overlay) < overlays.index(w._aa_overlay)


# ── POI style + legend ───────────────────────────────────────────────────────

def test_poi_marker_is_thin_cross():
    w = _widget()
    assert w._poi_overlay._marker == "+"       # thin cross, not a filled marker
    assert w._poi_overlay._edge_width == 1.2    # thin lines


def test_toolbar_shows_only_view_scalebar_crosshair():
    w = _widget()
    assert w.canvas.btn_reset_view.isHidden() is False       # fit to view
    assert w.canvas.btn_toggle_scalebar.isHidden() is False   # scalebar toggle
    assert w.canvas.btn_toggle_crosshair.isHidden() is False  # crosshair toggle
    assert w.canvas.btn_contrast.isHidden() is True           # dropped
    assert w.canvas.btn_toggle_ruler.isHidden() is True       # dropped


def test_legend_has_three_patches():
    w = _widget()
    labels = [label for _, label in w.canvas._legend_entries]
    assert labels == ["Image centre", "Alignment area", "Point of interest"]
    assert w.canvas._legend_artist is not None


def test_legend_survives_image_change_and_clears():
    c = FibsemImageCanvas()
    c.set_array(np.zeros((32, 32), dtype=np.uint8))
    c.set_legend([("yellow", "A"), ("limegreen", "B")])
    assert c._legend_artist is not None
    c.set_array(np.zeros((32, 32), dtype=np.uint8))  # a new frame must not drop it
    assert c._legend_artist is not None
    c.set_legend(None)  # explicit clear
    assert c._legend_artist is None and c._legend_entries is None


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
