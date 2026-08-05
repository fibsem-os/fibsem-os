"""The mapping between stage space and a real-space canvas.

Tested without a widget, and mostly without a microscope: the frame is the piece a
second modality has to be able to reuse, so anything it needs from a canvas or an
instrument is worth seeing spelled out here.

The property that matters is that the two directions are exact inverses. Everything on
the canvas is placed by one and clicked through the other, so a gap between them is the
distance between what you point at and what you get.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# `stage_frame` itself needs neither Qt nor napari -- it is geometry -- but importing it
# runs `fibsem.ui.widgets.canvas.__init__`, which eagerly imports the whole canvas
# package and so pulls both in. Both live in the `ui` extra, and CI installs only
# `.[test]`, so the guard every other module in this directory carries is what keeps
# collection from failing there rather than skipping.
pytest.importorskip("PyQt5")

from fibsem.fm.structures import CameraImageTransform, FibsemHardwareGeometry
from fibsem.structures import FibsemStagePosition
from fibsem.ui.widgets.canvas.stage_frame import FMStageProjection, StageFrame

PIXEL_SIZE = 1e-7
SHAPE = (1024, 1024)


class _FakeCanvas:
    """Only what a frame asks of a canvas: the scale, in both directions."""

    def __init__(self, reference_pixel_size=PIXEL_SIZE):
        self.reference_pixel_size = reference_pixel_size

    def metres_to_canvas(self, x, y):
        return x / self.reference_pixel_size, y / self.reference_pixel_size

    def canvas_to_metres(self, x, y):
        return x * self.reference_pixel_size, y * self.reference_pixel_size


def _geometry(**overrides):
    fields = dict(
        transform=CameraImageTransform.NONE,
        camera_tilt=180.0,
        column_tilt=0.0,
        fib_column_tilt=52.0,
        shuttle_pre_tilt=0.0,
        rotation_reference=0.0,
        rotation_180=180.0,
        is_compustage=True,
    )
    fields.update(overrides)
    return FibsemHardwareGeometry(**fields)


def _frame(canvas=None, origin=None, **geometry):
    return StageFrame(
        canvas or _FakeCanvas(),
        origin or FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180)),
        FMStageProjection(_geometry(**geometry), PIXEL_SIZE, SHAPE),
    )


def _offset(base, dx=0.0, dy=0.0):
    return FibsemStagePosition(
        x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t,
        coordinate_system=base.coordinate_system,
    )


# ── the round trip ───────────────────────────────────────────────────────


@pytest.mark.parametrize("point", [(0.0, 0.0), (2500.0, 0.0), (0.0, -1750.0), (-3200.0, 4100.0)])
def test_a_canvas_point_round_trips_through_stage_space(point):
    frame = _frame()

    assert frame.to_canvas(frame.to_stage(*point)) == pytest.approx(point, abs=1e-6)


@pytest.mark.parametrize("pretilt", [0.0, 35.0])
def test_the_round_trip_holds_on_a_pre_tilted_mount(pretilt):
    """A pre-tilt puts a real z on every sideways move, which is the case a compustage
    hides -- z stays 0 there, so an error in the y/z split cannot show up."""
    frame = _frame(shuttle_pre_tilt=pretilt, is_compustage=False)

    resolved = frame.to_stage(2500.0, 4000.0)

    assert frame.to_canvas(resolved) == pytest.approx((2500.0, 4000.0), abs=1e-6)


def test_a_sideways_click_carries_z_on_a_tilted_mount():
    """What keeps the sample in focus when you click across it. Zero here would mean
    the focal plane was being ignored, which a compustage would never reveal."""
    flat = _frame().to_stage(0.0, 2500.0)
    tilted = _frame(shuttle_pre_tilt=35.0, is_compustage=False).to_stage(0.0, 2500.0)

    assert flat.z == pytest.approx(0.0, abs=1e-15)
    assert abs(tilted.z) > 1e-9


def test_the_origin_maps_to_the_canvas_origin():
    origin = FibsemStagePosition(x=1e-3, y=-2e-3, z=0.0, r=0.0, t=np.deg2rad(-180))
    frame = _frame(origin=origin)

    assert frame.to_canvas(origin) == pytest.approx((0.0, 0.0))
    assert frame.offset(origin) == pytest.approx((0.0, 0.0))


# ── the scale ────────────────────────────────────────────────────────────


def test_a_length_is_measured_on_the_canvas_scale():
    frame = _frame(canvas=_FakeCanvas(reference_pixel_size=2e-7))

    assert frame.length(1e-3) == pytest.approx(5000.0)


def test_the_scale_cancels_out_of_the_round_trip():
    """`pixel_size` and `shape` on the projection are arguments to functions that
    measure in pixels, not inputs to the geometry -- so a canvas at a wildly different
    scale must resolve a click to the same stage position."""
    origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))
    fine = StageFrame(_FakeCanvas(1e-9), origin, FMStageProjection(_geometry(), 1e-9, SHAPE))
    coarse = StageFrame(_FakeCanvas(1e-9), origin, FMStageProjection(_geometry(), 27e-6, SHAPE))

    a, b = fine.to_stage(1500.0, -900.0), coarse.to_stage(1500.0, -900.0)

    assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), abs=1e-15)


def test_offset_is_metres_and_to_canvas_is_pixels():
    """The two answer the same question in different units, and confusing them places
    an image a factor of the pixel size away from where it belongs."""
    frame = _frame()
    position = _offset(frame.origin, dx=100e-6)

    assert frame.offset(position) == pytest.approx((100e-6, 0.0))
    assert frame.to_canvas(position) == pytest.approx((1000.0, 0.0))


# ── reading a projection off its source ──────────────────────────────────


def test_a_projection_can_be_read_off_the_live_microscope():
    from fibsem.ui.fm.overview_app import build_microscope

    projection = FMStageProjection.from_microscope(build_microscope())

    assert projection is not None
    assert projection.pixel_size > 0
    assert len(projection.shape) == 2


def test_a_microscope_without_fluorescence_yields_no_projection():
    class _NoFM:
        fm = None

    assert FMStageProjection.from_microscope(_NoFM()) is None


def test_an_image_without_a_recorded_geometry_yields_no_projection():
    """Acquired tiles currently lose their geometry (FIB-416), so this is the live
    case, not a hypothetical one -- and a caller has to be able to tell."""

    class _Metadata:
        geometry = None
        pixel_size_x = PIXEL_SIZE

    class _Image:
        metadata = _Metadata()
        data = np.zeros(SHAPE)

    assert FMStageProjection.from_image(_Image()) is None


def test_a_projection_read_off_an_image_uses_the_image_geometry():
    """The reason this is not simply always read live: an image may have been acquired
    under a configuration the instrument is no longer in, and has to be placed as it
    was taken."""

    class _Metadata:
        geometry = _geometry(transform=CameraImageTransform.FLIP_X)
        pixel_size_x = 5e-7

    class _Image:
        metadata = _Metadata()
        data = np.zeros((512, 256))

    projection = FMStageProjection.from_image(_Image())

    assert projection.geometry.transform is CameraImageTransform.FLIP_X
    assert projection.pixel_size == 5e-7
    assert tuple(projection.shape) == (512, 256)


def test_a_camera_flip_reverses_the_axis_it_names():
    origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))
    plain = _frame(origin=origin).to_canvas(_offset(origin, dx=100e-6))
    flipped = _frame(
        origin=origin, transform=CameraImageTransform.FLIP_X
    ).to_canvas(_offset(origin, dx=100e-6))

    assert flipped[0] == pytest.approx(-plain[0])
    assert flipped[1] == pytest.approx(plain[1])
