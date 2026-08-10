"""CorrelationFMCanvasWidget: the FM half of the canvas migration (FIB-535).

Picking behaviour is covered by the picking and overlay suites. What is pinned
here is what putting correlation on the *standard* FM widget has to preserve —
the controls it keeps, and the z semantics a picked point depends on.

The z tests are the ones that matter. `FMCanvasWidget` defaults to
max-projection because the quad view is a viewer; correlation is a picker, and a
projection has no z to give a point. Getting that wrong is silent: every FM point
lands at z=0 and the correlation is quietly wrong.
"""
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] without the UI extra

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.ui.correlation.widgets.correlation_fm_canvas_widget import (
    CorrelationFMCanvasWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

_NZ = 9


def _fm_image(nz=_NZ, size=32):
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-08-10T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(size, size),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP", color="#00ff00", excitation_wavelength=488,
                emission_wavelength=509, power=1.0, exposure_time=0.1,
                gain=1.0, offset=0.0,
            )
        ],
        filename="stack",
    )
    data = np.zeros((1, nz, size, size), dtype=np.uint16)
    return FluorescenceImage(data=data, metadata=metadata)


def _widget():
    w = CorrelationFMCanvasWidget(allowed_point_types=[PointType.FM])
    w.set_fm_image(_fm_image())
    return w


# ── z, and why it is not the shared default ───────────────────────────────


def test_max_projection_is_off_so_points_get_a_real_z():
    """FMCanvasWidget defaults it on; a projection has no plane, so `current_z`
    would answer 0 for every point picked on it."""
    w = _widget()
    assert w._max_projection is False
    assert w._btn_mip.isChecked() is False  # button agrees with what is drawn


@pytest.mark.parametrize("nz", [1, 2, 5, 9, 10])
def test_a_loaded_stack_starts_in_the_middle(nz):
    """FMCanvasWidget starts at plane 0, which suits a viewer that opens on the
    max projection. A picker opens on planes, and plane 0 is the out-of-focus
    edge of the volume, so every operator would scrub to the middle first.

    The widget this replaced started at ``n_z // 2``; the swap silently lost
    that, and retiring its test is what surfaced it.
    """
    w = CorrelationFMCanvasWidget()
    w.set_fm_image(_fm_image(nz=nz))
    assert w.current_z == nz // 2


@pytest.mark.parametrize("z", [0, 4, 8])
def test_current_z_follows_the_slider(z):
    """This is the number that becomes a picked Coordinate's z, so getting it
    wrong is a silently wrong correlation rather than a visible bug."""
    w = _widget()
    w._z_slider.setValue(z)
    assert w.current_z == z


def test_the_step_arrows_move_one_plane_and_stop_at_the_ends():
    """Present because FMImageDisplayWidget has them; without them the swap would
    lose a control the correlation display has today."""
    w = _widget()
    w._z_slider.setValue(0)

    w._z_next.click()
    assert w.current_z == 1
    w._z_prev.click()
    w._z_prev.click()  # already at 0
    assert w.current_z == 0

    w._z_slider.setValue(_NZ - 1)
    w._z_next.click()
    assert w.current_z == _NZ - 1


def test_shift_scroll_steps_z_and_plain_scroll_does_not():
    """Rides `canvas_scrolled` rather than the old bespoke z_scroll_requested --
    the shared canvas already broadcasts modified scroll and already declines to
    zoom under a modifier, which is how objective_control_widget and
    beam_settings_widget consume it."""
    w = _widget()
    w._z_slider.setValue(4)

    w.canvas.canvas_scrolled.emit(0.0, 0.0, 1, ("Shift",))
    assert w.current_z == 5
    w.canvas.canvas_scrolled.emit(0.0, 0.0, -1, ("Shift",))
    assert w.current_z == 4

    w.canvas.canvas_scrolled.emit(0.0, 0.0, 1, ())  # plain scroll is the canvas's zoom
    assert w.current_z == 4


def test_the_z_slider_advertises_shift_scroll():
    """The gesture has no visible affordance, so the tooltip is the only place it
    can be discovered. Set here, not on FMCanvasWidget: only this widget wires
    it, so the quad view would be advertising something that does nothing."""
    tip = _widget()._z_slider.toolTip()
    assert "Shift" in tip and "scroll" in tip.lower()


def test_stepping_does_nothing_under_max_projection():
    w = _widget()
    w.set_max_projection(True)
    w.step_z(3)
    assert w.current_z == 0


# ── the standard FM control set, kept ─────────────────────────────────────


def test_it_keeps_the_shared_fm_controls():
    """The point of replacing FMImageDisplayWidget rather than porting it: the
    correlation FM display gets the same z-slider, layers popover and
    max-projection button the quad view and FM overview use."""
    w = _widget()
    assert w._btn_layers is not None
    assert w._btn_mip is not None
    assert (w._z_prev.text(), w._z_next.text()) == ("‹", "›")
    assert w._panel is not None  # the per-channel layers popover


def test_pixel_size_is_not_shadowed():
    """FMCanvasWidget.set_pixel_size also records _pixel_size, which compositing
    reads; an override that only touched the canvas would drop it."""
    w = _widget()
    assert w._pixel_size == pytest.approx(1e-7)


# ── the picking surface ───────────────────────────────────────────────────


def test_points_and_signals_come_through():
    w = _widget()
    a = Coordinate(PointXYZ(10.0, 12.0, 3.0), PointType.FM)
    seen = []
    w.point_selected.connect(seen.append)

    w.set_coordinates([a])
    w.points.coordinate_selected.emit(a)

    assert w.points.get_points() == [(10.0, 12.0)]
    assert seen == [a]


def test_the_crosshair_is_off():
    w = _widget()
    assert w.canvas._crosshair_artists == []
    assert w.canvas.btn_toggle_crosshair.isChecked() is False


# ── the contract with the tab widget ──────────────────────────────────────


def test_the_tab_widget_can_connect_its_four_signals():
    """Both correlation surfaces are wired in one loop in the tab widget, so
    these have to be present under exactly these names."""
    for name in ("point_selected", "point_moved", "point_removed", "point_add_requested"):
        assert hasattr(CorrelationFMCanvasWidget, name), name


def test_the_adapter_surface_is_answered():
    """_CanvasAdapter is duck-typed over these three plus a z provider."""
    from fibsem.ui.correlation.widgets.correlation_tab_widget import _CanvasAdapter

    w = _widget()
    adapter = _CanvasAdapter(w, side="fm", z_provider=lambda: w.current_z)
    for name in ("set_coordinates", "set_selected", "refresh_coordinate"):
        assert callable(getattr(CorrelationFMCanvasWidget, name, None)), name
    w._z_slider.setValue(3)
    assert adapter.current_z() == 3.0
