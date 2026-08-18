"""The reprojection paths, against the shared image <-> microscope conversion.

`fibsem.conversions` owns one mapping between image pixels and the microscope image
frame -- centred on the image, +y *up*. Three reprojection sites re-derived it inline
until FIB-644 routed them through it:

* `fibsem.fm.reprojection` -- both directions, fluorescence
* `fibsem.imaging.tiling.reprojection` -- both `calculate_reprojected_stage_position`
  variants, FIB/SEM
* `fibsem.projection.FMStageProjection` -- the centre only, deliberately (see below)

**Every shape here is odd and non-square, and that is the whole point.** The centre is
`shape // 2` when the conversion floors it and `shape / 2` when it does not, and those
are *identical on an even dimension*. Both existing reprojection test modules use even
square images -- 1024x1024 and 64x64 -- so neither can tell the two apart, and neither
can see an axis swap either. Measured: FIB-688's first mutation run scored 14/15 until
its shapes were parametrised over (511, 1023).
"""
from __future__ import annotations

import numpy as np
import pytest

from fibsem.conversions import get_image_pixel_centre
from fibsem.fm.reprojection import project_image_point, project_stage_position
from fibsem.fm.structures import CameraImageTransform, FibsemHardwareGeometry
from fibsem.projection import FMStageProjection
from fibsem.structures import FibsemStagePosition, Point

# odd in both axes and non-square: an odd dimension separates the floored centre from
# the true one, a non-square shape separates (height, width) from (width, height)
ODD_SHAPE = (511, 1023)
PIXEL_SIZE = 1e-7

BASE = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))


def _geometry(transform=CameraImageTransform.NONE) -> FibsemHardwareGeometry:
    return FibsemHardwareGeometry(
        transform=transform,
        camera_tilt=180.0,
        is_compustage=True,
        shuttle_pre_tilt=0.0,
    )


class TestTheFluorescenceProjection:
    def test_the_base_position_lands_on_the_true_centre(self):
        """511.5, not 511.

        This is the assertion that distinguishes the two centres at all. With the
        conversion flooring instead, the marker for the image's own stage position
        lands half a pixel off -- visible on nothing, and wrong in the same direction
        for every position drawn on that image.
        """
        point = project_stage_position(
            BASE, BASE, PIXEL_SIZE, ODD_SHAPE, _geometry()
        )

        assert (point.x, point.y) == (ODD_SHAPE[1] / 2, ODD_SHAPE[0] / 2)
        assert (point.x, point.y) != (ODD_SHAPE[1] // 2, ODD_SHAPE[0] // 2)

    @pytest.mark.parametrize("transform", list(CameraImageTransform))
    @pytest.mark.parametrize(
        "pixel", [(0.0, 0.0), (511.5, 255.5), (1022.0, 510.0), (-300.0, 900.25)]
    )
    def test_the_round_trip_closes(self, transform, pixel):
        """The forward and the inverse must pass the same `subpixel_precision`.

        Pairing a true centre one way with a floored one the other loses half a pixel
        per round trip -- the defect FIB-688 fixed in `microscope_image_to_image_
        coordinates`, and the one this pins against coming back through either side.
        """
        point = Point(x=pixel[0], y=pixel[1])
        geometry = _geometry(transform)

        position = project_image_point(
            point, BASE, PIXEL_SIZE, ODD_SHAPE, geometry
        )
        back = project_stage_position(
            position, BASE, PIXEL_SIZE, ODD_SHAPE, geometry
        )

        assert back.x == pytest.approx(point.x, abs=1e-6)
        assert back.y == pytest.approx(point.y, abs=1e-6)

    def test_the_axes_are_not_swapped(self):
        """A non-square image is what makes this assertion mean anything.

        `image_shape` is (height, width) while `Point` is (x, y) -- opposite orders.
        Routing a shape through the wrong one is the FIB-694 trap, and on a square
        image it is invisible.
        """
        point = project_stage_position(
            BASE, BASE, PIXEL_SIZE, ODD_SHAPE, _geometry()
        )

        assert point.x == pytest.approx(1023 / 2)  # width
        assert point.y == pytest.approx(511 / 2)  # height

    @pytest.mark.parametrize("shape", [(512,), (512, 512, 3), ()])
    def test_a_shape_that_is_not_two_dimensional_raises(self, shape):
        """A tightening, and a deliberate one.

        The inline centring indexed [0] and [1], so a (H, W, C) shape quietly worked
        off the first two axes and a 1-D shape raised IndexError. The shared conversion
        requires exactly two. Every caller passes `image.data.shape[-2:]`, so this is
        inert today -- pinned so it stays a decision rather than becoming a surprise.
        """
        with pytest.raises(ValueError):
            project_stage_position(BASE, BASE, PIXEL_SIZE, shape, _geometry())

        with pytest.raises(ValueError):
            project_image_point(
                Point(0.0, 0.0), BASE, PIXEL_SIZE, shape, _geometry()
            )


class TestTheBeamProjection:
    """`imaging.tiling.reprojection`, both variants, on an odd non-square image."""

    @pytest.fixture()
    def image(self):
        from fibsem import utils
        from fibsem.structures import BeamType, ImageSettings

        microscope, _ = utils.setup_session(manufacturer="Demo")
        image = microscope.acquire_image(
            ImageSettings(
                hfw=80e-6, resolution=[64, 64], beam_type=BeamType.ELECTRON, save=False
            )
        )
        # the acquisition path will not hand back an odd resolution, and the shape is
        # all these functions read of the data
        image.data = np.zeros(ODD_SHAPE, dtype=np.uint16)
        return image

    @pytest.mark.parametrize("variant", ["", "2"])
    def test_the_image_position_lands_on_the_true_centre(self, image, variant):
        from fibsem.imaging.tiling import reprojection

        calculate = getattr(
            reprojection, f"calculate_reprojected_stage_position{variant}"
        )
        point = calculate(image, image.metadata.stage_position)

        assert (point.x, point.y) == (ODD_SHAPE[1] / 2, ODD_SHAPE[0] / 2)
        assert (point.x, point.y) != (ODD_SHAPE[1] // 2, ODD_SHAPE[0] // 2)

    @pytest.mark.parametrize("variant", ["", "2"])
    def test_the_axes_are_not_swapped(self, image, variant):
        from fibsem.imaging.tiling import reprojection

        calculate = getattr(
            reprojection, f"calculate_reprojected_stage_position{variant}"
        )
        point = calculate(image, image.metadata.stage_position)

        assert point.x == pytest.approx(1023 / 2)  # width
        assert point.y == pytest.approx(511 / 2)  # height


class TestTheDisplayPlaneIsNotThisConversion:
    """`FMStageProjection` is the near-miss, and it must stay one.

    `to_plane`/`from_plane` look like the same arithmetic and are not: the plane's y
    runs **down**, image-fashion, so they do not mirror y. They share only the centre
    with `fibsem.conversions`.

    The reason to pin it rather than trust the comment: the pair is self-consistent, so
    it round-trips perfectly either way. A round-trip test -- the obvious test to write
    here -- passes whether or not someone "deduplicates" it into the mirroring
    conversion, and the overview canvas would silently draw everything upside down.
    """

    @pytest.fixture()
    def projection(self):
        return FMStageProjection(
            geometry=_geometry(), pixel_size=PIXEL_SIZE, shape=ODD_SHAPE
        )

    def test_to_plane_does_not_mirror_y(self, projection):
        above = BASE + FibsemStagePosition(x=0.0, y=10e-6, z=0.0, r=0.0, t=0.0)

        pixel = project_stage_position(
            above, BASE, PIXEL_SIZE, ODD_SHAPE, _geometry()
        )
        plane_x, plane_y = projection.to_plane(above, BASE)
        cy, cx = get_image_pixel_centre(ODD_SHAPE, subpixel_precision=True)

        # same sense as the pixel offset it is derived from, not the opposite one
        assert np.sign(plane_y) == np.sign(pixel.y - cy)
        assert np.sign(plane_x) == np.sign(pixel.x - cx)

    def test_the_plane_origin_is_the_true_centre(self, projection):
        """Half a pixel of pixel_size, and it is shared with the conversion."""
        assert projection.to_plane(BASE, BASE) == (0.0, 0.0)

        point = projection.from_plane(0.0, 0.0, BASE)
        assert (point.x, point.y, point.z) == pytest.approx(
            (BASE.x, BASE.y, BASE.z), abs=1e-15
        )

    @pytest.mark.parametrize(
        "plane", [(0.0, 0.0), (1e-6, -1e-6), (-3.5e-5, 8.25e-5)]
    )
    def test_the_plane_round_trip_closes(self, projection, plane):
        position = projection.from_plane(plane[0], plane[1], BASE)

        assert projection.to_plane(position, BASE) == pytest.approx(plane, abs=1e-15)
