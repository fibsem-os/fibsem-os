"""Headless tests for the canvas-state reducer on MicroscopeViewController.

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_overlay_reducer.py

Or via pytest. Covers the milling slice (MillingSpec -> MillingPatternOverlay) and
the alignment slice (one AlignmentSpec; edit > display; the input round-trip).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, Point
from fibsem.ui.widgets.canvas.canvas_state import MaskSpec, MillingSpec, PointsSpec
from fibsem.ui.widgets.canvas.overlays.alignment_overlay import AlignmentAreaOverlay
from fibsem.ui.widgets.canvas.overlays.milling_overlay import MillingPatternOverlay
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _image() -> FibsemImage:
    return FibsemImage.generate_blank_image(hfw=80e-6, random=True)


def _stage(name: str = "s", x_um: float = 0.0, y_um: float = 0.0) -> FibsemMillingStage:
    pattern = RectanglePattern(width=10e-6, height=5e-6)
    pattern.point = Point(x=x_um * 1e-6, y=y_um * 1e-6)
    return FibsemMillingStage(name=name, pattern=pattern)


def _rect(left=0.25, top=0.25, width=0.5, height=0.5) -> FibsemRectangle:
    return FibsemRectangle(left=left, top=top, width=width, height=height)


def _flush(app: QApplication) -> None:
    for _ in range(3):
        app.processEvents()


# ── milling slice ───────────────────────────────────────────────────────────

def test_set_overlay_creates_attaches_and_renders():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas

    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage()]))
    _flush(app)

    objs = ctl._overlay_objs[fib]
    assert "milling" in objs, "milling overlay not created"
    overlay = objs["milling"]
    assert isinstance(overlay, MillingPatternOverlay)
    assert overlay in fib._overlays, "overlay not attached to the canvas"
    assert len(overlay._stages) == 1
    assert len(overlay._artists) > 0, "patterns not rendered"
    print("ok: set_overlay creates + attaches + renders")


def test_remove_overlay_detaches():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage()]))
    _flush(app)
    overlay = ctl._overlay_objs[fib]["milling"]

    ctl.remove_overlay(BeamType.ION, "milling")
    _flush(app)

    assert "milling" not in ctl._overlay_objs[fib]
    assert overlay not in fib._overlays
    assert "milling" not in ctl._scene.fib.overlays
    print("ok: remove_overlay detaches + drops the spec")


def test_image_injected_from_canvas():
    """No image -> overlay attaches but draws nothing; image arriving re-renders it."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas

    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage()]))  # before any image
    _flush(app)
    overlay = ctl._overlay_objs[fib]["milling"]
    assert overlay in fib._overlays
    assert len(overlay._artists) == 0, "should draw nothing without an image"

    ctl.set_image(BeamType.ION, _image())  # image swap marks dirty -> re-render
    _flush(app)
    assert len(overlay._artists) > 0, "did not render after image arrived"
    print("ok: reducer injects the canvas image (no image -> no draw -> renders on image)")


def test_renders_coalesce_into_one_pass():
    app = _app()
    ctl = MicroscopeViewController()
    emits = []
    ctl._render_requested.connect(lambda: emits.append(1))

    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage("a")]))
    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage("a"), _stage("b")]))
    assert len(emits) == 1, f"expected 1 coalesced render, got {len(emits)}"

    _flush(app)
    assert len(ctl._overlay_objs[ctl.fib_canvas]["milling"]._stages) == 2
    print("ok: renders coalesce into one pass")


def test_overlay_on_sem_canvas_beam_generic():
    app = _app()
    ctl = MicroscopeViewController()
    ctl.set_overlay(BeamType.ELECTRON, MillingSpec(stages=[_stage()]))
    _flush(app)
    assert "milling" in ctl._overlay_objs[ctl.sem_canvas]
    assert "milling" not in ctl._overlay_objs[ctl.fib_canvas]
    print("ok: reducer is beam-generic (SEM canvas)")


# ── alignment slice (one overlay; edit > display) ────────────────────────────

def test_alignment_edit_shows_editable_and_arms():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    r = _rect()
    ctl.set_alignment_edit(BeamType.ION, r, editing=True)
    _flush(app)

    ov = ctl._overlay_objs[fib]["alignment"]
    assert isinstance(ov, AlignmentAreaOverlay)
    assert ov in fib._overlays
    assert ov._area_visible is True and ov._interactive is True
    assert fib.active_overlay is ov, "editing should arm the overlay"
    assert ctl.alignment_area(BeamType.ION) is r
    print("ok: set_alignment_edit -> editable + armed + visible")


def test_alignment_display_shows_readonly_not_armed():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    r = _rect()
    ctl.set_alignment_display(BeamType.ION, r, show=True)
    _flush(app)

    ov = ctl._overlay_objs[fib]["alignment"]
    assert ov._area_visible is True and ov._interactive is False
    assert fib.active_overlay is None, "read-only display must not arm"
    assert ctl.alignment_area(BeamType.ION) is r
    print("ok: set_alignment_display -> read-only, not armed")


def test_edit_overrides_display_then_falls_back():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_alignment_display(BeamType.ION, _rect(), show=True)   # milling: read-only
    ctl.set_alignment_edit(BeamType.ION, _rect(), editing=True)   # image widget: edit wins
    _flush(app)
    ov = ctl._overlay_objs[fib]["alignment"]
    assert ov._interactive is True and fib.active_overlay is ov

    ctl.set_alignment_edit(BeamType.ION, None, editing=False)     # end edit; display still on
    _flush(app)
    assert ov._area_visible is True and ov._interactive is False, "should fall back to read-only"
    assert fib.active_overlay is None

    ctl.set_alignment_display(BeamType.ION, None, show=False)     # milling stops
    _flush(app)
    assert ov._area_visible is False, "nothing wants it -> hidden"
    print("ok: edit > display, falls back to read-only, then hidden")


def test_alignment_value_survives_end_of_edit():
    """The clear-then-read guarantee: ending an edit hides but keeps the value."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    r = _rect()
    ctl.set_alignment_edit(BeamType.ION, r, editing=True)
    _flush(app)
    ctl.set_alignment_edit(BeamType.ION, None, editing=False)  # no display -> hidden, value kept
    _flush(app)
    ov = ctl._overlay_objs[fib]["alignment"]
    assert ov._area_visible is False
    assert ctl.alignment_area(BeamType.ION) is r, "value must survive (workflow reads after clear)"
    print("ok: alignment value survives the end of an edit")


def test_alignment_edit_roundtrips_to_signal_and_model():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_alignment_edit(BeamType.ION, _rect(), editing=True)
    _flush(app)
    ov = ctl._overlay_objs[fib]["alignment"]

    seen = []
    ctl.overlay_edited.connect(lambda b, i, v: seen.append((b, i, v)))
    edited = _rect(0.1, 0.1, 0.3, 0.3)
    ov.alignment_area_changed.emit(edited)  # simulate a user drag/resize commit
    _flush(app)
    assert seen == [(BeamType.ION, "alignment", edited)], f"got {seen}"
    assert ctl.alignment_area(BeamType.ION) is edited, "edit must update the model"
    print("ok: alignment edit re-emits overlay_edited + updates the model")


def test_true_clear_removes_alignment():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_alignment_edit(BeamType.ION, _rect(), editing=True)
    _flush(app)
    ov = ctl._overlay_objs[fib]["alignment"]

    ctl.arm_overlay(BeamType.ION, None)
    ctl.remove_overlay(BeamType.ION, "alignment")  # true teardown
    _flush(app)
    assert "alignment" not in ctl._overlay_objs[fib]
    assert ov not in fib._overlays
    assert ctl.alignment_area(BeamType.ION) is None
    assert fib.active_overlay is None
    print("ok: remove_overlay (true clear) drops the overlay + value + disarms")


# ── points slice (POI / spot / detection share this) ─────────────────────────

def test_points_create_arm_and_set():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(
        BeamType.ION,
        PointsSpec(id="poi", points=[(10.0, 12.0)], color="magenta", marker="+", size=18),
    )
    ctl.arm_overlay(BeamType.ION, "poi", label="POI")
    _flush(app)

    from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay
    ov = ctl._overlay_objs[fib]["poi"]
    assert isinstance(ov, PointOverlay)
    assert ov in fib._overlays
    assert fib.active_overlay is ov, "POI should own input"
    assert ctl.overlay_points(BeamType.ION, "poi") == [(10.0, 12.0)]
    print("ok: points overlay creates + arms + sets points")


def test_points_move_roundtrips_to_model():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="poi", points=[(10.0, 10.0)]))
    _flush(app)
    ov = ctl._overlay_objs[fib]["poi"]

    # simulate a drag: the overlay geometry changes, point_moved fires on release
    ov.set_points([(25.0, 30.0)])
    ov.point_moved.emit(0, 25.0, 30.0)
    assert ctl.overlay_points(BeamType.ION, "poi") == [(25.0, 30.0)], "model not updated from drag"
    print("ok: point move round-trips into the model (read-back authoritative)")


def test_points_remove_disarms():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="poi", points=[(10.0, 10.0)]))
    ctl.arm_overlay(BeamType.ION, "poi")
    _flush(app)
    ov = ctl._overlay_objs[fib]["poi"]

    ctl.arm_overlay(BeamType.ION, None)
    ctl.remove_overlay(BeamType.ION, "poi")
    _flush(app)
    assert "poi" not in ctl._overlay_objs[fib]
    assert ov not in fib._overlays
    assert ctl.overlay_points(BeamType.ION, "poi") == []
    assert fib.active_overlay is None
    print("ok: remove points overlay drops it + disarms")


def test_points_visible_toggle_keeps_points():
    """Spot burn show/hide: visibility toggles without losing the points."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="spot", points=[(5.0, 5.0)], visible=True))
    _flush(app)
    ov = ctl._overlay_objs[fib]["spot"]
    assert ov._visible is True

    ctl.set_overlay_visible(BeamType.ION, "spot", False)  # hide, keep points
    _flush(app)
    assert ov._visible is False
    assert ctl.overlay_points(BeamType.ION, "spot") == [(5.0, 5.0)], "points kept on hide"

    ctl.set_overlay_visible(BeamType.ION, "spot", True)
    _flush(app)
    assert ov._visible is True
    print("ok: set_overlay_visible toggles visibility, keeps points")


def test_set_points_partial_keeps_config():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="spot", points=[(5.0, 5.0)], visible=False))
    _flush(app)
    ctl.set_points(BeamType.ION, "spot", [(1.0, 2.0), (3.0, 4.0)])  # points only
    _flush(app)
    assert ctl.overlay_points(BeamType.ION, "spot") == [(1.0, 2.0), (3.0, 4.0)]
    ov = ctl._overlay_objs[fib]["spot"]
    assert ov._visible is False, "visibility preserved by set_points"
    print("ok: set_points updates points, keeps config/visibility")


def test_canvas_selection_survives_unrelated_reconcile():
    """Regression: a reconcile runs on ANY change to a canvas (e.g. the info bar ticking on
    a stage move). set_points is destructive — it nulls the selection — so re-driving
    UNCHANGED points would silently clear a canvas-made selection, breaking Delete. The
    reducer must skip the rebuild when the point data is unchanged."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="spot", points=[(5.0, 5.0), (9.0, 9.0)]))
    _flush(app)
    ov = ctl._overlay_objs[fib]["spot"]

    ov.set_selected(1)  # user selects a point on the canvas (overlay-side; not in the spec)
    assert ov._selected == 1

    ctl._reconcile(fib)  # unrelated reconcile (same points) — must not touch the selection
    assert ov._selected == 1, "unrelated reconcile wiped the canvas selection"
    print("ok: canvas selection survives an unrelated reconcile")


def test_in_progress_drag_survives_unrelated_reconcile():
    """Regression: during a drag the overlay's live points move but the spec is not updated
    until release (point_moved). An unrelated reconcile mid-drag must not re-drive the
    unchanged spec — set_points would snap the point back to its pre-drag position."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(id="spot", points=[(5.0, 5.0)]))
    _flush(app)
    ov = ctl._overlay_objs[fib]["spot"]

    ov._points = [[25.0, 30.0]]  # mid-drag live position (spec still holds (5, 5))
    ctl._reconcile(fib)          # unrelated reconcile while dragging
    assert ov.get_points() == [(25.0, 30.0)], "reconcile snapped the in-progress drag back"
    print("ok: in-progress drag survives an unrelated reconcile")


def test_mask_overlay_displays_and_removes():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    ctl.set_overlay(BeamType.ION, MaskSpec(id="mask", mask=mask))
    _flush(app)
    from fibsem.ui.widgets.canvas.overlays.mask_overlay import MaskOverlay
    ov = ctl._overlay_objs[fib]["mask"]
    assert isinstance(ov, MaskOverlay)
    assert ov in fib._overlays
    ctl.remove_overlay(BeamType.ION, "mask")
    _flush(app)
    assert "mask" not in ctl._overlay_objs[fib]
    print("ok: mask overlay displays + removes")


def test_detection_features_colors_labels_and_move():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_overlay(BeamType.ION, PointsSpec(
        id="detection", points=[(10.0, 10.0), (20.0, 20.0)],
        colors=["red", "lime"], labels=["A", "B"],
        marker="+", removable=False, add_on_right_click=False, modal=True,
    ))
    ctl.arm_overlay(BeamType.ION, "detection", label="Detection")
    _flush(app)
    ov = ctl._overlay_objs[fib]["detection"]
    assert ov in fib._overlays
    assert fib.active_overlay is ov
    assert ctl.overlay_points(BeamType.ION, "detection") == [(10.0, 10.0), (20.0, 20.0)]

    # dragging a feature round-trips into the model (features re-read from the points)
    ov.set_points([(15.0, 15.0), (20.0, 20.0)])
    ov.point_moved.emit(0, 15.0, 15.0)
    assert ctl.overlay_points(BeamType.ION, "detection") == [(15.0, 15.0), (20.0, 20.0)]
    print("ok: detection features (colors/labels) + move round-trip")


# ── info bar ─────────────────────────────────────────────────────────────────

def test_info_bar_renders_ordered_fields_per_canvas():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_image(BeamType.ION, _image())
    ctl.set_info(BeamType.ION, "stage", "STAGE: x")
    ctl.set_info(BeamType.ION, "milling", "MILLING ANGLE: 30.0°")
    ctl.set_info(BeamType.ELECTRON, "stage", "STAGE: x")  # SEM gets stage only
    _flush(app)
    assert fib._info_text == "STAGE: x\nMILLING ANGLE: 30.0°"
    assert fib._info_artist is not None
    assert ctl.sem_canvas._info_text == "STAGE: x"
    print("ok: info bar renders ordered fields per canvas (milling FIB-only)")


def test_set_fm_info_targets_fm_canvas():
    app = _app()
    ctl = MicroscopeViewController()
    ctl.set_fm_info("objective", "OBJECTIVE: 12.3 µm")
    _flush(app)
    assert ctl.fm_canvas._info_text == "OBJECTIVE: 12.3 µm"
    print("ok: set_fm_info targets the FM canvas")


def test_info_bar_survives_image_change():
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas
    ctl.set_info(BeamType.ION, "stage", "STAGE: x")
    _flush(app)
    assert fib._info_text == "STAGE: x"
    ctl.set_image(BeamType.ION, _image())  # a new image clears the axes
    _flush(app)
    assert fib._info_text == "STAGE: x", "info must survive an image change"
    assert fib._info_artist is not None
    print("ok: info bar survives an image change (microscope state, not image)")


def _run_all():
    tests = [
        test_set_overlay_creates_attaches_and_renders,
        test_remove_overlay_detaches,
        test_image_injected_from_canvas,
        test_renders_coalesce_into_one_pass,
        test_overlay_on_sem_canvas_beam_generic,
        test_alignment_edit_shows_editable_and_arms,
        test_alignment_display_shows_readonly_not_armed,
        test_edit_overrides_display_then_falls_back,
        test_alignment_value_survives_end_of_edit,
        test_alignment_edit_roundtrips_to_signal_and_model,
        test_true_clear_removes_alignment,
        test_points_create_arm_and_set,
        test_points_move_roundtrips_to_model,
        test_points_remove_disarms,
        test_points_visible_toggle_keeps_points,
        test_set_points_partial_keeps_config,
        test_mask_overlay_displays_and_removes,
        test_detection_features_colors_labels_and_move,
        test_info_bar_renders_ordered_fields_per_canvas,
        test_set_fm_info_targets_fm_canvas,
        test_info_bar_survives_image_change,
    ]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
