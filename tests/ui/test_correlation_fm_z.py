"""The z an FM point carries in the correlation tab.

After a z-interpolation each FM point must still sit on the feature it was
placed on: the points used to move by ``new_nz / old_nz`` while the volume was
resampled end slice to end slice, which drifts a point off its feature by up to
a slice, more with depth (FIB-1238). A point placed on the max projection takes
z = 0, which the projection cannot do better than; the user is told so
(FIB-1241).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_correlation_fm_z.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.correlation.util import interpolate_fm_volume
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage
from fibsem.ui import notification_service
from fibsem.ui.correlation.widgets.correlation_tab_widget import CorrelationTabWidget

_app = QApplication.instance() or QApplication(sys.argv)

_NZ = 21
_Z_STEP = 500e-9
_XY = 130e-9
_PER_SLICE = 1000  # grey levels per source slice: the ramp reads back its depth


def _ramp_stack() -> FluorescenceImage:
    """One channel whose every plane holds its own index (x1000), so the value
    under a point is the source slice it sits on."""
    planes = np.arange(_NZ, dtype=np.uint16)[:, None, None] * _PER_SLICE
    channel = FluorescenceChannelMetadata(
        name="Reflection",
        excitation_wavelength=550,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="gray",
        objective_magnification=100,
        objective_numerical_aperture=0.85,
    )
    return FluorescenceImage(
        data=np.broadcast_to(planes, (_NZ, 32, 32)).copy()[None],
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-10-09T12:00:00",
            pixel_size_x=_XY,
            pixel_size_y=_XY,
            pixel_size_z=_Z_STEP,
            resolution=(32, 32),
            channels=[channel],
        ),
    )


@pytest.fixture
def tab():
    widget = CorrelationTabWidget()
    widget.resize(1200, 800)
    widget.show()
    widget.set_fib_image(FibsemImage(data=np.zeros((200, 300), np.uint8)))
    widget.set_fm_image(_ramp_stack())
    _app.processEvents()
    yield widget
    widget.close()


@pytest.fixture
def toasts(monkeypatch):
    events = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda msg, notification_type="info": events.append((msg, notification_type)),
    )
    return events


def _fm_points(tab):
    return tab._point_specs[PointType.FM].list_widget.coordinates


def _show_interpolated(tab, target_m):
    """What the worker hands back, adopted on this thread rather than a worker's."""
    src = tab._fm_image
    view = interpolate_fm_volume(src, target_m)
    tab._adopt_interpolated_volume(view, src.data.shape[1])
    return view


@pytest.mark.parametrize("target_m", [130e-9, 250e-9, 1200e-9])
def test_interpolating_the_view_leaves_every_fm_point_where_it_was(tab, target_m):
    placed = [16.0, 7.0, 20.0, 12.4]
    tab.set_data(
        CorrelationInputData(
            fm_coordinates=[
                Coordinate(PointXYZ(10.0, 10.0, z), PointType.FM) for z in placed
            ]
        )
    )
    stack = tab._fm_image

    _show_interpolated(tab, target_m)

    assert [c.point.z for c in _fm_points(tab)] == placed
    assert tab._fm_image is stack  # the fits and the file keep the stack


@pytest.mark.parametrize("target_m", [130e-9, 250e-9, 1200e-9])
def test_a_point_picked_on_the_interpolated_view_lands_on_the_same_feature(
    tab, target_m
):
    """The view's plane is converted to the stack's: the stored z reads, in
    the stack, the feature the view showed at the plane it was picked on."""
    view = _show_interpolated(tab, target_m)
    tab._fm_display.set_max_projection(False)
    tab._fm_display.step_z(3)
    plane = tab._fm_display.current_z
    shown = view.data[0, plane, 0, 0] / _PER_SLICE  # the feature on screen

    tab._on_canvas_add_requested(10.0, 12.0, PointType.FM)

    z = _fm_points(tab)[-1].point.z
    stack = tab._fm_image.data[0, :, 0, 0].astype(float) / _PER_SLICE
    assert np.interp(z, np.arange(len(stack)), stack) == pytest.approx(shown, abs=0.01)


def test_a_point_placed_on_the_max_projection_says_it_is_at_z0(tab, toasts):
    tab._fm_display.set_max_projection(True)
    tab._on_canvas_add_requested(10.0, 12.0, PointType.FM)

    assert _fm_points(tab)[-1].point.z == 0
    assert len(toasts) == 1
    msg, kind = toasts[0]
    assert "z = 0" in msg and kind == "warning"


def test_a_point_placed_on_a_plane_takes_that_plane_quietly(tab, toasts):
    tab._fm_display.set_max_projection(False)
    tab._fm_display.step_z(3)
    plane = tab._fm_display.current_z
    assert plane != 0
    tab._on_canvas_add_requested(10.0, 12.0, PointType.FM)

    assert _fm_points(tab)[-1].point.z == plane
    assert toasts == []


def test_a_fib_point_never_warns_about_the_fm_projection(tab, toasts):
    tab._fm_display.set_max_projection(True)
    tab._on_canvas_add_requested(10.0, 12.0, PointType.FIB)
    assert toasts == []


def test_the_interpolated_view_keeps_the_stacks_corrected_pixel_size(tab):
    """A stack whose data is narrower than its recorded resolution gets a
    corrected scale bar; showing its interpolated view must not drop that."""
    stack = _ramp_stack()
    stack.metadata.resolution = (64, 64)  # recorded at twice the data width
    tab.set_fm_image(stack)
    corrected = tab._fm_display._pixel_size
    assert corrected == pytest.approx(2 * _XY)

    _show_interpolated(tab, 250e-9)

    assert tab._fm_display._pixel_size == pytest.approx(corrected)


def test_a_view_finished_after_the_stack_changed_is_not_shown(tab):
    """FIB-1242: an interpolation that finishes after another stack was loaded
    must not put its planes over that stack."""
    import time

    tab._start_fm_interpolation(250e-9, "linear")
    tab.set_fm_image(_ramp_stack())  # the user opens another stack meanwhile
    deadline = time.monotonic() + 30
    while tab._interp_worker is not None and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert tab._interp_worker is None, "interpolation did not finish"

    assert tab._fm_view_scale == 1.0
    assert tab._fm_display._z_max == _NZ - 1  # still the new stack's planes


def _z_tip(tab, i=-1):
    lst = tab._point_specs[PointType.FM].list_widget._list
    row = lst.itemWidget(lst.item(lst.count() + i if i < 0 else i))
    return row.z_spin.toolTip()


def test_the_z_tooltip_gives_the_view_plane_as_the_slider_numbers_it(tab):
    """The table keeps the stack's slice; the tooltip says where that is on
    the interpolated view's slider, which counts from 1."""
    tab.set_data(
        CorrelationInputData(
            fm_coordinates=[Coordinate(PointXYZ(10.0, 10.0, 16.0), PointType.FM)]
        )
    )
    assert _z_tip(tab) == "Z (slice)"  # no view: as before

    _show_interpolated(tab, 130e-9)
    scale = tab._fm_view_scale
    planes = tab._fm_display._z_max + 1
    assert _z_tip(tab) == (
        f"Stack slice 16.00 (of 0–{_NZ - 1}) · view plane "
        f"{round(16.0 * scale) + 1}/{planes}"
    )

    # a pick on the view: its tooltip names the plane the slider showed
    tab._fm_display.set_max_projection(False)
    tab._fm_display.step_z(5)
    shown = tab._fm_display._z_label.text()  # e.g. "46/81"
    tab._on_canvas_add_requested(10.0, 12.0, PointType.FM)
    assert _z_tip(tab).endswith(f"view plane {shown}")

    tab.set_fm_image(_ramp_stack())  # a new stack is shown as it is
    assert _z_tip(tab, 0) == "Z (slice)"
