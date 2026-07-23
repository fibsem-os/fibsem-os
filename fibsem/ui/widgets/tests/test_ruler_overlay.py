"""Headless tests for the canvas drag-to-measure RulerOverlay (Phase 7.1).

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_ruler_overlay.py

Or via pytest. Covers SI formatting, the toolbar toggle (seed + activate +
restore), endpoint/line drag with bounds clamping, screen-space hit testing,
and that the overlay stays fully inert (no artists, no input) while hidden.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay, _format_distance

_W, _H = 1536, 1024
_PX = 10e-9  # 10 nm / px


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _ev(**kw):
    return types.SimpleNamespace(**kw)


def _canvas() -> FibsemImageCanvas:
    _app()
    c = FibsemImageCanvas()
    c.resize(800, 600)
    c.set_array(np.zeros((_H, _W), dtype=np.uint8), pixel_size=_PX)
    c.draw()  # realise transData for screen-space hit tests
    return c


def _ruler_on(c: FibsemImageCanvas) -> RulerOverlay:
    c.btn_toggle_ruler.setChecked(True)
    c.toggle_ruler()
    return c._ruler_overlay


def test_si_formatting():
    assert _format_distance(5e-9) == "5.0 nm"
    assert _format_distance(2.5e-6) == "2.50 µm"
    assert _format_distance(3e-3) == "3.000 mm"
    assert _format_distance(0) == "0 nm"


def test_toggle_on_seeds_and_activates():
    c = _canvas()
    r = _ruler_on(c)
    assert r is not None and r in c._overlays
    assert r._visible is True
    assert r._p1 is not None and r._p2 is not None
    assert r._line is not None and r._dots is not None and r._label is not None
    assert c.active_overlay is r
    seed_len = abs(r._p2[0] - r._p1[0])
    assert abs(seed_len - 0.25 * _W) < 1e-6
    assert abs(r.measurement() - seed_len * _PX) < 1e-18
    assert r._label.get_text() == _format_distance(seed_len * _PX)


def test_drag_endpoint_updates_distance():
    c = _canvas()
    r = _ruler_on(c)
    seed_len = abs(r._p2[0] - r._p1[0])
    r._drag = "p2"
    r._drag_start_data = (r._p2[0], r._p2[1])
    r._drag_start_pts = (list(r._p1), list(r._p2))
    r._blit_bg = None  # force the draw_idle path (skip blit)
    r._on_motion(_ev(xdata=r._p2[0] + 200, ydata=r._p2[1] + 100))
    expected = ((seed_len + 200) ** 2 + 100 ** 2) ** 0.5
    assert abs(r.measurement() - expected * _PX) < 1e-15
    r._on_release(_ev(button=1))
    assert c._overlay_consuming_event is False
    assert r._drag is None


def test_drag_clamps_to_bounds():
    c = _canvas()
    r = _ruler_on(c)
    r._drag = "p2"
    r._drag_start_data = (r._p2[0], r._p2[1])
    r._drag_start_pts = (list(r._p1), list(r._p2))
    r._blit_bg = None
    r._on_motion(_ev(xdata=r._p2[0] + 99999, ydata=r._p2[1] + 99999))
    assert r._p2[0] <= _W and r._p2[1] <= _H
    r._on_release(_ev(button=1))


def test_move_line_preserves_length_and_clamps():
    c = _canvas()
    r = _ruler_on(c)
    p1_0, p2_0 = list(r._p1), list(r._p2)
    length0 = ((p2_0[0] - p1_0[0]) ** 2 + (p2_0[1] - p1_0[1]) ** 2) ** 0.5
    r._drag = "line"
    r._drag_start_data = (0.0, 0.0)
    r._drag_start_pts = (list(r._p1), list(r._p2))
    r._blit_bg = None
    r._on_motion(_ev(xdata=-99999, ydata=0.0))  # shove far left
    length1 = ((r._p2[0] - r._p1[0]) ** 2 + (r._p2[1] - r._p1[1]) ** 2) ** 0.5
    assert abs(length1 - length0) < 1e-6
    assert min(r._p1[0], r._p2[0]) >= -1e-9
    r._on_release(_ev(button=1))


def test_screen_space_hit_testing():
    c = _canvas()
    r = _ruler_on(c)
    c.draw()
    sx1, sy1 = r._screen(r._p1)
    sx2, sy2 = r._screen(r._p2)
    mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    assert r._hit(_ev(x=sx1, y=sy1)) == "p1"
    assert r._hit(_ev(x=sx2, y=sy2)) == "p2"
    assert r._hit(_ev(x=mx, y=my)) == "line"
    assert r._hit(_ev(x=sx1, y=sy1 + 200)) is None


def test_ruler_off_restores_active_mode():
    """Ruler off returns input to the overlay MODE that is active, derived from the live
    mode state (not a snapshot taken when the ruler was switched on)."""
    c = _canvas()
    mode_ov = RulerOverlay()  # stands in for a workflow overlay (POI / spot burn)
    c.add_overlay(mode_ov)
    c.enter_overlay_mode(mode_ov, "Edit")
    assert c.active_overlay is mode_ov
    r = _ruler_on(c)
    assert c.active_overlay is r
    c.btn_toggle_ruler.setChecked(False)
    c.toggle_ruler()
    assert r._line is None and r._visible is False
    assert c.active_overlay is mode_ov  # mode restored, not a stale snapshot


def test_ruler_off_after_mode_exit_returns_to_move():
    """Regression: a mode exited while measuring must not leave a stale overlay owning input
    (with no toolbar affordance) when the ruler is switched off — it returns to Move."""
    c = _canvas()
    mode_ov = RulerOverlay()
    c.add_overlay(mode_ov)
    c.enter_overlay_mode(mode_ov, "Edit")
    _ruler_on(c)
    c.exit_overlay_mode(mode_ov)  # the workflow ends the step while measuring
    c.btn_toggle_ruler.setChecked(False)
    c.toggle_ruler()
    assert c.active_overlay is None  # Move, not the stale mode_ov


def test_ruler_off_after_mode_overlay_removed_returns_to_move():
    """Regression: if the mode overlay is removed while measuring (the reducer drops its
    spec), switching the ruler off must not re-activate the now-detached overlay — that
    would suppress every semantic click permanently. It returns to Move."""
    c = _canvas()
    mode_ov = RulerOverlay()
    c.add_overlay(mode_ov)
    c.enter_overlay_mode(mode_ov, "Edit")
    _ruler_on(c)
    c.remove_overlay(mode_ov)  # e.g. controller.remove_overlay / clear()
    c.btn_toggle_ruler.setChecked(False)
    c.toggle_ruler()
    assert c.active_overlay is None
    assert mode_ov not in c._overlays


def test_ruler_off_with_no_mode_returns_to_move():
    """No overlay mode active: ruler off returns to Move (None)."""
    c = _canvas()
    r = _ruler_on(c)
    assert c.active_overlay is r
    c.btn_toggle_ruler.setChecked(False)
    c.toggle_ruler()
    assert c.active_overlay is None


def test_inert_while_hidden():
    c = _canvas()
    r = RulerOverlay()
    c.add_overlay(r)  # never toggled visible
    assert r._line is None
    r._on_press(_ev(inaxes=c._ax, button=1, xdata=10, ydata=10, x=10, y=10))
    assert r._drag is None
    c.set_array(np.zeros((_H, _W), dtype=np.uint8), pixel_size=_PX)
    assert r._line is None  # image change keeps it inert


def test_pixel_size_refreshes_on_image_change():
    c = _canvas()
    r = _ruler_on(c)
    # a new image with a different pixel size updates the label units
    c.set_array(np.zeros((_H, _W), dtype=np.uint8), pixel_size=2 * _PX)
    seed_len = abs(r._p2[0] - r._p1[0])
    assert abs(r.measurement() - seed_len * 2 * _PX) < 1e-18


def _run_all():
    tests = [
        test_si_formatting,
        test_toggle_on_seeds_and_activates,
        test_drag_endpoint_updates_distance,
        test_drag_clamps_to_bounds,
        test_move_line_preserves_length_and_clamps,
        test_screen_space_hit_testing,
        test_ruler_off_restores_active_mode,
        test_ruler_off_after_mode_exit_returns_to_move,
        test_ruler_off_after_mode_overlay_removed_returns_to_move,
        test_ruler_off_with_no_mode_returns_to_move,
        test_inert_while_hidden,
        test_pixel_size_refreshes_on_image_change,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
