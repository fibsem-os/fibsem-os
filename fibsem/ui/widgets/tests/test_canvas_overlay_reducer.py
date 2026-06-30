"""Headless tests for the canvas-state reducer on MicroscopeViewController.

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_overlay_reducer.py

Or via pytest. Covers the milling slice: producers push a ``MillingSpec`` through
the reducer; the controller owns the ``MillingPatternOverlay``, injects the canvas
image, coalesces renders, and removes the overlay when the spec is gone.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.structures import BeamType, FibsemImage, Point
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


def _flush(app: QApplication) -> None:
    # deliver the queued _render_requested → _do_render
    for _ in range(3):
        app.processEvents()


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
    """No image → overlay attaches but draws nothing; image arriving re-renders it."""
    app = _app()
    ctl = MicroscopeViewController()
    fib = ctl.fib_canvas

    ctl.set_overlay(BeamType.ION, MillingSpec(stages=[_stage()]))  # before any image
    _flush(app)
    overlay = ctl._overlay_objs[fib]["milling"]
    assert overlay in fib._overlays
    assert len(overlay._artists) == 0, "should draw nothing without an image"

    ctl.set_image(BeamType.ION, _image())  # image swap marks dirty → re-render
    _flush(app)
    assert len(overlay._artists) > 0, "did not render after image arrived"
    print("ok: reducer injects the canvas image (no image → no draw → renders on image)")


def test_renders_coalesce_into_one_pass():
    app = _app()
    ctl = MicroscopeViewController()
    emits = []
    ctl._render_requested.connect(lambda: emits.append(1))

    # 3 mutations in one event-loop turn → a single queued render
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
    # the reducer is beam-generic: a spec routes to whichever canvas the beam maps to
    ctl.set_overlay(BeamType.ELECTRON, MillingSpec(stages=[_stage()]))
    _flush(app)
    assert "milling" in ctl._overlay_objs[ctl.sem_canvas]
    assert "milling" not in ctl._overlay_objs[ctl.fib_canvas]
    print("ok: reducer is beam-generic (SEM canvas)")


def _run_all():
    tests = [
        test_set_overlay_creates_attaches_and_renders,
        test_remove_overlay_detaches,
        test_image_injected_from_canvas,
        test_renders_coalesce_into_one_pass,
        test_overlay_on_sem_canvas_beam_generic,
    ]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
