"""A driver-supplied half-turn centre, shared by stage moves and reprojection (FIB-1081).

A position recorded on one side of a rotating stage is at 2c - p on the other, for a
rotation centre c. Reprojection hardcoded one instrument's c, and every backend but
ThermoFisher assumed c at the stage origin for moves. The driver now says where its
stage turns about. These pin that a driver saying nothing behaves exactly as before,
and that a centre it does give reaches both paths.
"""

from copy import deepcopy

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.reprojection import (
    X_OFFSET,
    Y_OFFSET,
    _transform_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image2,
)
from fibsem.projection import BeamStageProjection
from fibsem.structures import (
    LEGACY_ROTATION_CENTRE,
    BeamType,
    FibsemHardwareGeometry,
    FibsemStagePosition,
    ImageSettings,
)

CENTRE = (1.5e-3, -0.75e-3)


class TestTransformPosition:
    def test_the_default_is_the_legacy_formula(self):
        """Specimen offset, negate, the (+50, +25) um residual, back to raw -- what
        _transform_position computed before it took a centre."""
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.1, t=0.2)
        transformed = _transform_position(pos)

        assert transformed.x == pytest.approx(2 * X_OFFSET - pos.x + 50e-6, abs=1e-15)
        assert transformed.y == pytest.approx(2 * Y_OFFSET - pos.y + 25e-6, abs=1e-15)

    def test_a_centre_reflects_xy_through_it(self):
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.1, t=0.2)
        transformed = _transform_position(pos, CENTRE)

        assert transformed.x == pytest.approx(2 * CENTRE[0] - pos.x)
        assert transformed.y == pytest.approx(2 * CENTRE[1] - pos.y)
        assert (transformed.z, transformed.r, transformed.t) == (pos.z, pos.r, pos.t)

    def test_it_is_its_own_inverse(self):
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)
        twice = _transform_position(_transform_position(pos, CENTRE), CENTRE)

        assert (twice.x, twice.y) == pytest.approx((pos.x, pos.y))

    def test_it_does_not_change_the_caller_s_position(self):
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0, name="lamella")
        before = deepcopy(pos)
        _transform_position(pos, CENTRE)

        assert pos == before


class TestGeometry:
    def test_an_image_saved_before_the_field_draws_as_it_did(self):
        assert (
            FibsemHardwareGeometry.from_dict({}).rotation_centre
            == LEGACY_ROTATION_CENTRE
        )

    def test_round_trips(self):
        geometry = FibsemHardwareGeometry(rotation_centre=CENTRE)

        assert (
            FibsemHardwareGeometry.from_dict(geometry.to_dict()).rotation_centre
            == CENTRE
        )

    def test_without_one_the_legacy_centre_is_recorded(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = (
            None  # the Demo names one; this is a driver that does not
        )

        assert microscope.hardware_geometry().rotation_centre == LEGACY_ROTATION_CENTRE

    def test_a_driver_s_centre_is_recorded(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE

        assert microscope.hardware_geometry().rotation_centre == CENTRE


class TestStageMoves:
    """`get_target_position` reflects a half turn through -_get_compucentric_rotation_offset."""

    def test_without_one_it_reflects_through_the_origin_as_before(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = (
            None  # the Demo names one; this is a driver that does not
        )
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)
        target = microscope._get_compucentric_rotation_position(pos)

        assert (target.x, target.y) == pytest.approx((-pos.x, -pos.y))

    def test_a_driver_s_centre_is_reflected_through(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)
        target = microscope._get_compucentric_rotation_position(pos)

        assert (target.x, target.y) == pytest.approx(
            (2 * CENTRE[0] - pos.x, 2 * CENTRE[1] - pos.y)
        )

    def test_the_stage_and_the_reprojection_agree(self):
        """The point of the field: one centre, so a marker drawn from the other side
        of the stage is where a move to that side would go."""
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)

        moved = microscope._get_compucentric_rotation_position(pos)
        drawn = _transform_position(pos, microscope.hardware_geometry().rotation_centre)

        assert (moved.x, moved.y) == pytest.approx((drawn.x, drawn.y))


class TestTheDemo:
    """The Demo names the centre its stage turns about, so its moves and its drawings
    agree with nothing set by the caller."""

    def test_it_turns_about_its_origin(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)
        target = microscope._get_compucentric_rotation_position(pos)

        assert microscope.rotation_centre == (0.0, 0.0)
        assert (target.x, target.y) == pytest.approx((-pos.x, -pos.y))

    def test_its_images_record_that_centre(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")

        assert microscope.hardware_geometry().rotation_centre == (0.0, 0.0)

    def test_a_position_from_the_other_side_is_drawn_where_a_move_goes(self):
        """The 1.6 mm the overview and the stage map were apart on the Demo: a lamella
        saved at MILLING, with the stage at FIB over it."""
        microscope, _ = utils.setup_session(manufacturer="Demo")
        lamella = deepcopy(microscope.get_orientation("MILLING"))
        lamella.x, lamella.y, lamella.z = -2.8e-3, 0.2e-3, 0.0
        at_fib = microscope.get_target_position(lamella, "FIB")

        sem = microscope.get_orientation("SEM")
        origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t)
        projection = BeamStageProjection(
            geometry=microscope.hardware_geometry(),
            beam_type=BeamType.ELECTRON,
            scan_rotation=0.0,
        )
        drawn_lamella = projection._compucentric_corrected(lamella, origin, (0.0, 0.0))
        drawn_stage = projection._compucentric_corrected(
            at_fib, origin, microscope.hardware_geometry().rotation_centre
        )

        assert (drawn_stage.x, drawn_stage.y) == pytest.approx(
            (drawn_lamella.x, drawn_lamella.y)
        )


@pytest.fixture()
def image_with_centre():
    """A real demo image, acquired with a driver-supplied centre."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.rotation_centre = CENTRE
    return microscope.acquire_image(
        ImageSettings(
            hfw=2e-3, resolution=[64, 64], beam_type=BeamType.ELECTRON, save=False
        )
    )


class TestReprojection:
    def test_a_position_from_the_other_side_is_drawn_through_the_image_s_centre(
        self, image_with_centre
    ):
        image = image_with_centre
        base = image.metadata.microscope_state.stage_position
        assert image.metadata.hardware_geometry.rotation_centre == CENTRE

        other_side = deepcopy(base)
        other_side.r = base.r + np.pi
        other_side.x += 100e-6

        (drawn,) = reproject_stage_positions_onto_image2(image, [other_side])
        expected = calculate_reprojected_stage_position2(
            image, _transform_position(other_side, CENTRE)
        )

        assert (drawn.x, drawn.y) == pytest.approx((expected.x, expected.y))

    def test_the_canvas_projection_uses_the_image_s_centre(self, image_with_centre):
        image = image_with_centre
        base = image.metadata.microscope_state.stage_position
        projection = BeamStageProjection.from_image(image)
        assert projection.geometry.rotation_centre == CENTRE

        other_side = deepcopy(base)
        other_side.r = base.r + np.pi
        corrected = BeamStageProjection._compucentric_corrected(
            other_side, base, projection.geometry.rotation_centre
        )

        assert (corrected.x, corrected.y) == pytest.approx(
            (2 * CENTRE[0] - other_side.x, 2 * CENTRE[1] - other_side.y)
        )
