"""Characterisation tests for the stage-position reprojection maths.

Written against `fibsem.imaging.tiled` before that code moves into the
`fibsem.imaging.tiling` package (FIB-390), so the move can be shown to be inert:
these must pass unchanged before and after.

Reprojection had no direct coverage. `TestTiledInverseMatchesMicroscope` in
test_view_corrected_movement.py pins `_inverse_y_corrected_stage_movement` against
the microscope's own inverse, but nothing pinned the functions that consume it --
which is what the minimap and overview widgets actually call to decide where to draw
a marker.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiled import (
    _to_raw_coordinate_system,
    _to_specimen_coordinate_system,
    _transform_position,
    calculate_reprojected_stage_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image,
)
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings


@pytest.fixture()
def image():
    """A real demo image, so metadata is fully populated rather than hand-built."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    return microscope.acquire_image(
        ImageSettings(
            hfw=80e-6, resolution=[64, 64], beam_type=BeamType.ELECTRON, save=False
        )
    )


def _base(image) -> FibsemStagePosition:
    return image.metadata.stage_position


class TestCalculateReprojectedStagePosition:
    def test_the_image_position_lands_at_the_centre(self, image):
        """Zero displacement must reproject to the middle of the frame."""
        point = calculate_reprojected_stage_position(image, _base(image))

        assert point.x == pytest.approx(image.data.shape[1] / 2)
        assert point.y == pytest.approx(image.data.shape[0] / 2)

    def test_x_offset_moves_along_the_image_x_axis_only(self, image):
        pos = _base(image) + FibsemStagePosition(x=10e-6, y=0, z=0)
        point = calculate_reprojected_stage_position(image, pos)
        centre = calculate_reprojected_stage_position(image, _base(image))

        assert point.x != pytest.approx(centre.x)
        assert point.y == pytest.approx(centre.y)

    def test_y_offset_moves_along_the_image_y_axis_only(self, image):
        pos = _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)
        point = calculate_reprojected_stage_position(image, pos)
        centre = calculate_reprojected_stage_position(image, _base(image))

        assert point.x == pytest.approx(centre.x)
        assert point.y != pytest.approx(centre.y)

    def test_y_sign_flips_across_the_image_position(self, image):
        """The `delta.y < 0` branch on the y displacement is worth pinning."""
        above = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)
        )
        below = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=0, y=-10e-6, z=0)
        )
        centre = calculate_reprojected_stage_position(image, _base(image))

        assert (above.y - centre.y) == pytest.approx(-(below.y - centre.y))

    def test_a_higher_stage_y_draws_above_the_image_centre(self, image):
        """The absolute sense, not just the flip.

        Every other sign test in this class is symmetric -- above vs below, plain vs
        rotated -- so inverting the whole y convention passes all of them and draws
        every marker mirrored. That is exactly how FIB-692 survived in the polygon
        converter, so the direction is pinned outright: +y on the stage is *up* the
        image, which is a smaller row index.
        """
        centre = calculate_reprojected_stage_position(image, _base(image))
        higher = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)
        )
        righter = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=10e-6, y=0, z=0)
        )

        assert higher.y < centre.y
        assert righter.x > centre.x

    def test_z_contributes_to_the_y_displacement(self, image):
        """dy is sqrt(delta.y^2 + delta.z^2), so z is folded into the in-image y."""
        flat = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)
        )
        raised = calculate_reprojected_stage_position(
            image, _base(image) + FibsemStagePosition(x=0, y=10e-6, z=10e-6)
        )

        centre = calculate_reprojected_stage_position(image, _base(image))
        assert abs(raised.y - centre.y) > abs(flat.y - centre.y)

    def test_scan_rotation_of_pi_inverts_both_axes(self, image):
        """A half-turn raster flips the reprojection, matching stable_move."""
        pos = _base(image) + FibsemStagePosition(x=10e-6, y=10e-6, z=0)

        image.metadata.microscope_state.electron_beam.scan_rotation = 0.0
        plain = calculate_reprojected_stage_position(image, pos)
        centre = calculate_reprojected_stage_position(image, _base(image))

        image.metadata.microscope_state.electron_beam.scan_rotation = np.pi
        rotated = calculate_reprojected_stage_position(image, pos)

        assert (rotated.x - centre.x) == pytest.approx(-(plain.x - centre.x))
        assert (rotated.y - centre.y) == pytest.approx(-(plain.y - centre.y))

    def test_compustage_pose_inverts_y(self, image):
        """At t = -180 the grid is imaged from behind, so in-image y flips."""
        pos = _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)

        image.metadata.stage_position.t = 0.0
        upright = calculate_reprojected_stage_position(image, pos)
        centre_upright = calculate_reprojected_stage_position(image, _base(image))

        image.metadata.stage_position.t = np.radians(-180)
        flipped = calculate_reprojected_stage_position(image, pos)
        centre_flipped = calculate_reprojected_stage_position(image, _base(image))

        assert (flipped.y - centre_flipped.y) == pytest.approx(
            -(upright.y - centre_upright.y)
        )


class TestReprojectStagePositionsOntoImage:
    def test_returns_one_point_per_position_when_unbounded(self, image):
        positions = [
            _base(image) + FibsemStagePosition(x=d, y=0, z=0)
            for d in (0.0, 5e-6, -5e-6)
        ]
        points = reproject_stage_positions_onto_image(image, positions, bound=False)

        assert len(points) == len(positions)

    def test_bound_filters_positions_outside_the_frame(self, image):
        """`bound=True` drops anything that reprojects off the image."""
        inside = _base(image)
        far_away = _base(image) + FibsemStagePosition(x=1.0, y=0, z=0)  # 1 metre

        unbounded = reproject_stage_positions_onto_image(
            image, [inside, far_away], bound=False
        )
        bounded = reproject_stage_positions_onto_image(
            image, [inside, far_away], bound=True
        )

        assert len(unbounded) == 2
        assert len(bounded) < len(unbounded)

    def test_agrees_with_the_single_position_function(self, image):
        pos = _base(image) + FibsemStagePosition(x=5e-6, y=5e-6, z=0)
        [point] = reproject_stage_positions_onto_image(image, [pos], bound=False)
        direct = calculate_reprojected_stage_position(image, pos)

        assert point.x == pytest.approx(direct.x)
        assert point.y == pytest.approx(direct.y)


class TestVariant2:
    """The `2` variant projects through the microscope-free inverse instead."""

    def test_the_image_position_lands_at_the_centre(self, image):
        point = calculate_reprojected_stage_position2(image, _base(image))

        assert point.x == pytest.approx(image.data.shape[1] / 2)
        assert point.y == pytest.approx(image.data.shape[0] / 2)

    def test_a_higher_stage_y_draws_above_the_image_centre(self, image):
        """The same absolute sense as the original variant -- see its twin above."""
        centre = calculate_reprojected_stage_position2(image, _base(image))
        higher = calculate_reprojected_stage_position2(
            image, _base(image) + FibsemStagePosition(x=0, y=10e-6, z=0)
        )
        righter = calculate_reprojected_stage_position2(
            image, _base(image) + FibsemStagePosition(x=10e-6, y=0, z=0)
        )

        assert higher.y < centre.y
        assert righter.x > centre.x

    def test_x_offset_matches_the_original_variant(self, image):
        """x carries no projection, so the two variants must agree on it exactly."""
        pos = _base(image) + FibsemStagePosition(x=10e-6, y=0, z=0)

        assert calculate_reprojected_stage_position2(image, pos).x == pytest.approx(
            calculate_reprojected_stage_position(image, pos).x
        )


class TestCoordinateSystemHelpers:
    def test_specimen_and_raw_are_inverses(self):
        pos = FibsemStagePosition(
            x=1e-3, y=2e-3, z=3e-3, r=0.5, t=0.2, coordinate_system="RAW"
        )
        roundtrip = _to_raw_coordinate_system(_to_specimen_coordinate_system(pos))

        assert roundtrip.x == pytest.approx(pos.x)
        assert roundtrip.y == pytest.approx(pos.y)
        assert roundtrip.z == pytest.approx(pos.z)

    def test_transform_position_preserves_the_name(self):
        pos = FibsemStagePosition(
            x=1e-3, y=2e-3, z=0.0, r=0, t=0, coordinate_system="RAW", name="lamella-1"
        )
        assert _transform_position(pos).name == "lamella-1"

    def test_transform_position_inverts_xy_about_the_specimen_origin(self):
        """A 180 degree compucentric rotation, plus a fixed calibration offset."""
        pos = FibsemStagePosition(
            x=1e-3, y=2e-3, z=0.0, r=0, t=0, coordinate_system="RAW"
        )
        transformed = _transform_position(pos)

        specimen = _to_specimen_coordinate_system(pos)
        expected = _to_raw_coordinate_system(
            FibsemStagePosition(
                x=-specimen.x + 50e-6,
                y=-specimen.y + 25e-6,
                z=specimen.z,
                r=specimen.r,
                t=specimen.t,
                coordinate_system="RAW",
            )
        )

        assert transformed.x == pytest.approx(expected.x)
        assert transformed.y == pytest.approx(expected.y)
