"""Headless tests for the protocol-level spot-burn coordinate editor (FIB-380).

Covers the widget itself (settings round-trip, list <-> overlay sync, teardown) and
the config bridge the protocol editor's dialog relies on:

    SpotBurnFiducialTaskConfig --to_settings()--> SpotBurnSettings --> widget
    widget.get_settings() --> SpotBurnSettings --apply_settings()--> config

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_spot_burn_coordinates_widget.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.imaging.spot import SpotBurnSettings
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui.widgets.canvas.quad_view import LamellaEditorView, MicroscopeViewController
from fibsem.ui.widgets.spot_burn_coordinates_widget import SpotBurnCoordinatesWidget

_app = QApplication.instance() or QApplication(sys.argv)

_RESOLUTION = (1536, 1024)  # (x, y) — deliberately non-square, see the aspect test
_HFW = 150e-6


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def _host(settings=None):
    """A widget + its own controller/canvas, as the protocol-editor dialog builds it."""
    view = LamellaEditorView()
    controller = MicroscopeViewController(view=view)
    image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW, random=False)
    controller.set_image(BeamType.ION, image)
    w = SpotBurnCoordinatesWidget(controller=controller, beam=BeamType.ION, settings=settings)
    w.set_image_shape(image.data.shape)
    return w, controller, view, image


# ── the config <-> settings bridge the dialog depends on ────────────────────

def test_config_round_trips_through_settings():
    cfg = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn Fiducial",
        milling_current=80e-12,
        exposure_time=7,
        coordinates=[Point(0.16, 0.45), Point(0.15, 0.45)],
    )
    w, _, _, _ = _host(settings=cfg.to_settings())
    cfg.apply_settings(w.get_settings())
    assert [(p.x, p.y) for p in cfg.coordinates] == [(0.16, 0.45), (0.15, 0.45)]
    assert cfg.milling_current == 80e-12 and cfg.exposure_time == 7.0


def test_apply_settings_preserves_unrelated_task_fields():
    """The dialog applies onto the stored config; milling/reference/autofocus must survive."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial", autofocus=True)
    ref_before = cfg.reference_imaging
    cfg.apply_settings(SpotBurnSettings(coordinates=[Point(0.5, 0.5)]))
    assert cfg.autofocus is True
    assert cfg.reference_imaging is ref_before
    assert cfg.task_type == "SPOT_BURN_FIDUCIAL"


# ── coordinates are relative to the frame, and the frame is not square ──────

def test_normalised_coordinates_survive_a_non_square_frame():
    """x and y are normalised independently — a 3:2 frame must not skew them."""
    pts = [Point(0.25, 0.75), Point(0.9, 0.1)]
    w, _, _, image = _host(settings=SpotBurnSettings(coordinates=pts))
    h, width = image.data.shape
    assert (width, h) == _RESOLUTION  # generate_blank_image takes (x, y)
    out = w.get_settings().coordinates
    assert _close(out[0].x, 0.25) and _close(out[0].y, 0.75)
    assert _close(out[1].x, 0.9) and _close(out[1].y, 0.1)


def test_overlay_receives_pixel_positions_scaled_by_each_axis():
    w, controller, _, image = _host(settings=SpotBurnSettings(coordinates=[Point(0.25, 0.75)]))
    w.set_active(True)
    _app.processEvents()
    spec = controller._scene.fib.overlays[SpotBurnCoordinatesWidget.OVERLAY_ID]
    h, width = image.data.shape
    (px, py), = spec.points
    assert _close(px, 0.25 * width) and _close(py, 0.75 * h)


# ── list <-> canvas sync ────────────────────────────────────────────────────

def test_canvas_edit_writes_back_to_settings_and_emits():
    w, controller, _, image = _host(settings=SpotBurnSettings(coordinates=[Point(0.2, 0.2)]))
    w.set_active(True)
    seen = []
    w.settings_changed.connect(seen.append)
    h, width = image.data.shape
    # simulate a drag + an add on the canvas
    controller.overlay_edited.emit(
        BeamType.ION, SpotBurnCoordinatesWidget.OVERLAY_ID,
        [(0.4 * width, 0.6 * h), (0.8 * width, 0.1 * h)],
    )
    out = w.get_settings().coordinates
    assert len(out) == 2
    assert _close(out[0].x, 0.4) and _close(out[0].y, 0.6)
    assert len(seen) == 1


def test_rows_mirror_the_coordinates():
    w, _, _, _ = _host(settings=SpotBurnSettings(
        coordinates=[Point(0.16, 0.45), Point(0.15, 0.45), Point(0.17, 0.45)]))
    assert w._list.count() == 3
    w.set_settings(SpotBurnSettings(coordinates=[Point(0.5, 0.5)]))
    assert w._list.count() == 1


def test_rebuilding_rows_does_not_accumulate_widgets():
    """_rebuild_rows runs on every edit (each canvas drag-release), so leaked row
    widgets would grow without bound over a session."""
    from fibsem.ui.widgets.spot_burn_coordinates_widget import _SpotBurnRow

    w, _, _, _ = _host(settings=SpotBurnSettings(coordinates=[Point(0.1, 0.1)]))
    for _ in range(20):
        w.set_settings(SpotBurnSettings(coordinates=[Point(0.5, 0.5)]))
    for _ in range(5):
        _app.processEvents()
    live = [r for r in w.findChildren(_SpotBurnRow)]
    assert len(live) <= 2, f"{len(live)} row widgets alive for 1 coordinate"


# ── teardown ────────────────────────────────────────────────────────────────

def test_hide_after_controller_is_gone_does_not_raise():
    """Qt gives no teardown ordering guarantee between the widget and the controller.

    The dialog owns both, so on close the controller's C++ object can go first; the
    widget's hideEvent then disarms an overlay on a dead object. Must be survivable.
    """
    import sip

    w, controller, view, _ = _host(settings=SpotBurnSettings(coordinates=[Point(0.3, 0.3)]))
    w.set_active(True)
    sip.delete(controller)          # controller dies first
    assert sip.isdeleted(controller)
    w.set_active(False)             # what hideEvent does — must not raise


def test_deactivate_removes_the_overlay():
    w, controller, _, _ = _host(settings=SpotBurnSettings(coordinates=[Point(0.3, 0.3)]))
    w.set_active(True)
    _app.processEvents()
    assert SpotBurnCoordinatesWidget.OVERLAY_ID in controller._scene.fib.overlays
    w.set_active(False)
    _app.processEvents()
    assert SpotBurnCoordinatesWidget.OVERLAY_ID not in controller._scene.fib.overlays


# ── layout: the list absorbs spare height, and survives a cramped host ──────

def test_list_takes_the_spare_height():
    """The list used to be capped at 180px with no stretch, so a dozen-point pattern
    scrolled a six-row window while the panel below it sat empty."""
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    w, _, _, _ = _host(settings=SpotBurnSettings(
        coordinates=[Point(0.1 * i, 0.1 * i) for i in range(1, 12)]))
    host = QWidget()
    QVBoxLayout(host).addWidget(w, 1)
    host.resize(360, 800)
    host.show()
    for _ in range(8):
        _app.processEvents()
    assert w._list.height() > 400, w._list.height()
    # every row reachable without scrolling
    visible = sum(
        1 for i in range(w._list.count())
        if w._list.visualItemRect(w._list.item(i)).bottom() <= w._list.viewport().height()
    )
    assert visible == w._list.count() == 11


def test_cramped_host_keeps_the_summary_and_a_usable_list():
    """The live spot-burn tab is short — the list must not eat the footer."""
    from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

    w, _, _, _ = _host(settings=SpotBurnSettings(
        coordinates=[Point(0.1 * i, 0.1 * i) for i in range(1, 12)]))
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(w, 1)
    sibling = QPushButton("Run Spot Burn")
    lay.addWidget(sibling, 0)
    host.resize(360, 260)
    host.show()
    for _ in range(8):
        _app.processEvents()
    assert w._list.height() >= 100          # did not collapse
    assert not w.label_summary.isHidden()   # footer survived
    assert sibling.isVisible()              # and so did the host's own controls


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
