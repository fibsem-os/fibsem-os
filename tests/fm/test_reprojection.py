"""Projecting stage positions onto fluorescence images.

The property that matters is agreement: an overlay drawn with the reprojection has to
land where clicking there would send the stage. Two things can break that quietly --
the microscope-free path drifting from the live one, and a saved image being projected
against the wrong pose -- so both are pinned here rather than left to inspection.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.reprojection import (
    project_image_point,
    project_stage_position,
    reproject_stage_positions_onto_fm_image,
    stage_position_from_fm_image,
)
from fibsem.fm.structures import (
    CameraImageTransform,
    ChannelSettings,
    FibsemHardwareGeometry,
)
from fibsem.structures import FibsemStagePosition, Point
from fibsem.transformations import (
    inverse_view_corrected_dy,
    view_corrected_stage_movement,
)

TRANSFORMS = list(CameraImageTransform)
SHAPE = (1024, 1024)
PIXEL_SIZE = 1e-7  # 102.4 um across


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    if scope.fm is None:
        scope.fm = FluorescenceMicroscope(parent=scope)
    return scope


def _pose(microscope, *, compustage, pretilt_deg, rotation_deg, tilt_deg):
    """Pin the microscope into a known geometry without moving the stage."""
    microscope.stage_is_compustage = compustage
    microscope.system.stage.shuttle_pre_tilt = pretilt_deg
    microscope._update_orientations()
    position = FibsemStagePosition(
        x=0.0,
        y=0.0,
        z=0.0,
        r=np.deg2rad(rotation_deg),
        t=np.deg2rad(tilt_deg),
    )
    microscope.get_stage_position = lambda: position  # type: ignore[method-assign]
    return position


# A compustage cannot rotate (see FibsemMicroscope._view_corrected_stage_movement), so
# only the reference rotation is a reachable pose for it.
POSES = [
    pytest.param(True, 0, 0, tilt, id=f"compustage-t{tilt}")
    for tilt in (0, -30, -128, -180)
] + [
    pytest.param(False, pretilt, rot, tilt, id=f"offset-p{pretilt}-r{rot}-t{tilt}")
    for pretilt in (0, 35)
    for rot in (0, 180)
    for tilt in (0, 25, -30)
]


class TestParityWithTheLiveMicroscope:
    """The microscope-free projection must reproduce the live one exactly.

    They are separate implementations of one formula: the live path decides where the
    stage goes, the pure path decides where the overlay is drawn. Any disagreement puts
    a marker somewhere the stage will not go, and nothing on screen says so.
    """

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("camera_tilt", [0.0, 90.0, 180.0])
    def test_inverse_matches(
        self, microscope, compustage, pretilt, rot, tilt, camera_tilt, monkeypatch
    ):
        position = _pose(
            microscope,
            compustage=compustage,
            pretilt_deg=pretilt,
            rotation_deg=rot,
            tilt_deg=tilt,
        )
        monkeypatch.setattr(
            type(microscope.fm), "camera_tilt", property(lambda self: camera_tilt)
        )

        live = microscope._inverse_view_corrected_stage_movement(
            dy=12e-6, dz=-3e-6, view_tilt=np.deg2rad(camera_tilt)
        )
        pure = inverse_view_corrected_dy(
            dy=12e-6,
            dz=-3e-6,
            view_tilt=np.deg2rad(microscope.fm.camera_tilt),
            geometry=microscope.fm_image_geometry(),
            stage_rotation=position.r,
            stage_tilt=position.t,
        )

        assert pure == pytest.approx(live, abs=1e-15)

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("camera_tilt", [0.0, 90.0, 180.0])
    def test_forward_matches(
        self, microscope, compustage, pretilt, rot, tilt, camera_tilt, monkeypatch
    ):
        """The forward direction is what actually moves the stage, so a click resolved
        by the pure path has to agree with it exactly -- not merely closely."""
        position = _pose(
            microscope,
            compustage=compustage,
            pretilt_deg=pretilt,
            rotation_deg=rot,
            tilt_deg=tilt,
        )
        monkeypatch.setattr(
            type(microscope.fm), "camera_tilt", property(lambda self: camera_tilt)
        )

        live = microscope._view_corrected_stage_movement(
            expected_y=17e-6, view_tilt=np.deg2rad(camera_tilt)
        )
        dy, dz = view_corrected_stage_movement(
            17e-6,
            view_tilt=np.deg2rad(microscope.fm.camera_tilt),
            geometry=microscope.fm_image_geometry(),
            stage_rotation=position.r,
            stage_tilt=position.t,
        )

        assert dy == pytest.approx(live.y, abs=1e-18)
        assert dz == pytest.approx(live.z, abs=1e-18)

    def test_an_unreachable_compustage_rotation_does_not_flip_the_sign(
        self, microscope
    ):
        """Regression: keying the FIB orientation on tilt alone disagreed with the live
        path at a rotated compustage. No physical run can reach that pose, which is
        exactly why the disagreement would have survived."""
        position = _pose(
            microscope,
            compustage=True,
            pretilt_deg=0,
            rotation_deg=180,
            tilt_deg=-128,
        )
        assert microscope.get_stage_orientation() != "FIB", "pose assumption drifted"

        live = microscope._inverse_view_corrected_stage_movement(
            dy=12e-6, dz=-3e-6, view_tilt=np.deg2rad(microscope.fm.camera_tilt)
        )
        pure = inverse_view_corrected_dy(
            dy=12e-6,
            dz=-3e-6,
            view_tilt=np.deg2rad(microscope.fm.camera_tilt),
            geometry=microscope.fm_image_geometry(),
            stage_rotation=position.r,
            stage_tilt=position.t,
        )

        assert pure == pytest.approx(live, abs=1e-15)


class TestRoundTrip:
    """Clicking a pixel and drawing that pixel must agree."""

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("transform", TRANSFORMS)
    def test_pixel_to_stage_and_back(
        self, microscope, compustage, pretilt, rot, tilt, transform
    ):
        base = _pose(
            microscope,
            compustage=compustage,
            pretilt_deg=pretilt,
            rotation_deg=rot,
            tilt_deg=tilt,
        )
        microscope.fm.set_image_transform(transform)

        for pixel in [(512, 512), (0, 0), (1023, 40), (300, 900)]:
            point = Point(x=float(pixel[0]), y=float(pixel[1]))
            position = project_image_point(
                point,
                base,
                PIXEL_SIZE,
                SHAPE,
                microscope.fm_image_geometry(),
            )
            back = project_stage_position(
                position,
                base,
                PIXEL_SIZE,
                SHAPE,
                microscope.fm_image_geometry(),
            )

            assert back.x == pytest.approx(point.x, abs=1e-6)
            assert back.y == pytest.approx(point.y, abs=1e-6)

    def test_the_base_position_lands_on_the_image_centre(self, microscope):
        base = _pose(
            microscope, compustage=True, pretilt_deg=0, rotation_deg=0, tilt_deg=-180
        )
        point = project_stage_position(
            base, base, PIXEL_SIZE, SHAPE, microscope.fm_image_geometry()
        )

        assert (point.x, point.y) == (SHAPE[1] / 2, SHAPE[0] / 2)

    def test_positions_outside_the_field_of_view_are_not_clamped(self, microscope):
        """The tile grid is drawn larger than the image it sits on, so a position off
        the edge has to return coordinates off the edge rather than being pulled to
        the border -- which would silently stack every outer tile on the boundary."""
        base = _pose(
            microscope, compustage=True, pretilt_deg=0, rotation_deg=0, tilt_deg=-180
        )
        far = project_image_point(
            Point(x=-4000.0, y=-4000.0),
            base,
            PIXEL_SIZE,
            SHAPE,
            microscope.fm_image_geometry(),
        )
        point = project_stage_position(
            far, base, PIXEL_SIZE, SHAPE, microscope.fm_image_geometry()
        )

        assert point.x == pytest.approx(-4000.0, abs=1e-6)
        assert point.y == pytest.approx(-4000.0, abs=1e-6)


class TestGeometryIsRecorded:
    def test_an_acquired_image_carries_the_geometry(self, microscope):
        _pose(microscope, compustage=True, pretilt_deg=0, rotation_deg=0, tilt_deg=-180)
        microscope.fm.set_image_transform(CameraImageTransform.FLIP_XY)

        image = microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))

        geometry = image.metadata.geometry
        assert geometry is not None
        assert geometry.transform is CameraImageTransform.FLIP_XY
        assert geometry.is_compustage is True
        assert geometry.camera_tilt == microscope.fm.camera_tilt

    @pytest.mark.parametrize("transform", TRANSFORMS)
    def test_geometry_round_trips_through_a_dict(self, transform):
        geometry = FibsemHardwareGeometry(
            transform=transform,
            camera_tilt=90.0,
            column_tilt=0.0,
            fib_column_tilt=52.0,
            shuttle_pre_tilt=35.0,
            rotation_reference=10.0,
            rotation_180=190.0,
            is_compustage=True,
        )

        assert FibsemHardwareGeometry.from_dict(geometry.to_dict()) == geometry


class TestReprojectionOntoAnImage:
    @pytest.fixture()
    def image(self, microscope):
        _pose(microscope, compustage=True, pretilt_deg=0, rotation_deg=0, tilt_deg=-180)
        microscope.fm.set_image_transform(CameraImageTransform.NONE)
        return microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))

    def test_it_uses_the_images_own_pose_not_the_live_one(self, microscope, image):
        """The point of recording the geometry: the stage has usually moved on by the
        time anyone looks at a saved overview, and the answer must not follow it."""
        base = image.metadata.stage_position
        expected = reproject_stage_positions_onto_fm_image(image, [base])

        _pose(
            microscope,
            compustage=False,
            pretilt_deg=35,
            rotation_deg=180,
            tilt_deg=25,
        )
        microscope.fm.set_image_transform(CameraImageTransform.FLIP_XY)
        after = reproject_stage_positions_onto_fm_image(image, [base])

        assert (after[0].x, after[0].y) == (expected[0].x, expected[0].y)

    def test_names_are_carried_across(self, image):
        base = image.metadata.stage_position
        named = FibsemStagePosition(
            x=base.x, y=base.y, z=base.z, r=base.r, t=base.t, name="Lamella-01"
        )

        points = reproject_stage_positions_onto_fm_image(image, [named])

        assert points[0].name == "Lamella-01"

    def test_bound_drops_positions_outside_the_image(self, microscope, image):
        base = image.metadata.stage_position
        height, width = image.data.shape[-2:]
        outside = stage_position_from_fm_image(
            image, Point(x=float(width * 4), y=float(height * 4))
        )

        assert len(reproject_stage_positions_onto_fm_image(image, [base, outside])) == 2
        assert (
            len(
                reproject_stage_positions_onto_fm_image(
                    image, [base, outside], bound=True
                )
            )
            == 1
        )

    def test_an_image_without_geometry_raises_rather_than_guessing(self, image):
        """Images written before the geometry was recorded. A marker in the wrong place
        is indistinguishable on screen from one in the right place, so falling back to
        live state or to a default pose would be worse than refusing."""
        base = image.metadata.stage_position
        image.metadata.geometry = None

        with pytest.raises(ValueError, match="geometry"):
            reproject_stage_positions_onto_fm_image(image, [base])

    def test_an_image_without_a_stage_position_raises(self, image):
        image.metadata.stage_position = None

        with pytest.raises(ValueError, match="stage position"):
            reproject_stage_positions_onto_fm_image(image, [FibsemStagePosition()])


def test_the_pure_projection_needs_no_microscope():
    """The whole point of the split: reprojection must work on a loaded file with no
    instrument attached, which is how it will usually be used."""
    geometry = FibsemHardwareGeometry(
        transform=CameraImageTransform.NONE,
        camera_tilt=180.0,
        is_compustage=True,
        shuttle_pre_tilt=0.0,
    )
    base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180))

    point = project_stage_position(base, base, PIXEL_SIZE, SHAPE, geometry)

    assert (point.x, point.y) == (SHAPE[1] / 2, SHAPE[0] / 2)


class TestBothDirectionsFromAnImageAlone:
    """The pair has to close on the image's own record, with no instrument attached --
    which is the situation whenever someone opens a saved overview."""

    @pytest.fixture()
    def image(self, microscope):
        _pose(microscope, compustage=True, pretilt_deg=0, rotation_deg=0, tilt_deg=-180)
        microscope.fm.set_image_transform(CameraImageTransform.FLIP_XY)
        return microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))

    @pytest.mark.parametrize("pixel", [(512, 512), (0, 0), (900, 120), (-300, 1500)])
    def test_pixel_to_stage_and_back(self, image, pixel):
        point = Point(x=float(pixel[0]), y=float(pixel[1]))

        position = stage_position_from_fm_image(image, point)
        back = reproject_stage_positions_onto_fm_image(image, [position])[0]

        assert back.x == pytest.approx(point.x, abs=1e-6)
        assert back.y == pytest.approx(point.y, abs=1e-6)

    def test_it_does_not_follow_the_live_microscope(self, microscope, image):
        point = Point(x=300.0, y=800.0)
        expected = stage_position_from_fm_image(image, point)

        _pose(
            microscope,
            compustage=False,
            pretilt_deg=35,
            rotation_deg=180,
            tilt_deg=25,
        )
        microscope.fm.set_image_transform(CameraImageTransform.NONE)
        after = stage_position_from_fm_image(image, point)

        assert (after.x, after.y, after.z) == (expected.x, expected.y, expected.z)

    def test_an_image_without_geometry_raises(self, image):
        image.metadata.geometry = None

        with pytest.raises(ValueError, match="geometry"):
            stage_position_from_fm_image(image, Point(x=0.0, y=0.0))
