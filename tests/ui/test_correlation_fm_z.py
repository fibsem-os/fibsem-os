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


@pytest.mark.parametrize("target_m", [130e-9, 250e-9, 1200e-9])
def test_interpolation_keeps_every_fm_point_on_its_feature(tab, target_m):
    placed = [16.0, 7.0, 20.0, 12.4]
    tab.set_data(
        CorrelationInputData(
            fm_coordinates=[
                Coordinate(PointXYZ(10.0, 10.0, z), PointType.FM) for z in placed
            ]
        )
    )
    src = tab._fm_image

    # what the worker hands back, adopted on this thread rather than a worker's
    tab._adopt_interpolated_volume(
        interpolate_fm_volume(src, target_m), src.data.shape[1]
    )

    ramp = tab._fm_image.data[0, :, 0, 0].astype(float) / _PER_SLICE
    new_step = tab._fm_image.metadata.pixel_size_z
    for z_before, coord in zip(placed, _fm_points(tab)):
        on = np.interp(coord.point.z, np.arange(len(ramp)), ramp)
        assert on == pytest.approx(z_before, abs=0.01)  # the same feature
        # and the same physical depth, so the correlation sees no change
        assert coord.point.z * new_step == pytest.approx(z_before * _Z_STEP)


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
