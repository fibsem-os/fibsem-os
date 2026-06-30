"""Headless tests for the canvas-state reducer on MicroscopeViewController.

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_overlay_reducer.py

Or via pytest. Covers the milling slice (MillingSpec -> MillingPatternOverlay) and
the alignment slice (one AlignmentSpec; edit > display; the input round-trip).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, Point
from fibsem.ui.widgets.alignment_overlay import AlignmentAreaOverlay
from fibsem.ui.widgets.canvas_state import MillingSpec
from fibsem.ui.widgets.milling_overlay import MillingPatternOverlay
from fibsem.ui.widgets.quad_view import MicroscopeViewController


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
    ]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
