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
        """Images with no recorded centre reflect through LEGACY_ROTATION_CENTRE."""
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.1, t=0.2)
        transformed = _transform_position(pos)

        assert transformed.x == pytest.approx(2 * LEGACY_ROTATION_CENTRE[0] - pos.x)
        assert transformed.y == pytest.approx(2 * LEGACY_ROTATION_CENTRE[1] - pos.y)

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


class _VendorStage:
    """Reports a specimen and a raw position, as xT does for one stage."""

    def __init__(self):
        self.system = None
        self.systems = []

    def set_default_coordinate_system(self, system):
        self.system = system
        self.systems.append(system)

    @property
    def current_position(self):
        if self.system == "Specimen":
            return FibsemStagePosition(x=1e-3 - 513e-6, y=2e-3 + 794e-6, z=0, r=0, t=0)
        return FibsemStagePosition(x=1e-3, y=2e-3, z=0, r=0, t=0)


def _thermo(correction=(0.0, 0.0), monkeypatch=None):
    """A ThermoMicroscope as connect leaves it, without the SDK."""
    import fibsem.drivers.autoscript.microscope as A

    class _CoordinateSystem:
        SPECIMEN = "Specimen"
        RAW = "Raw"

    monkeypatch.setattr(A, "CoordinateSystem", _CoordinateSystem, raising=False)
    monkeypatch.setattr(A, "stage_position_from_autoscript", lambda p: p)
    microscope = object.__new__(A.ThermoMicroscope)
    microscope.system = utils.setup_session(manufacturer="Demo")[0].system
    microscope.system.stage.rotation_centre_correction = correction
    microscope._vendor_stage = _VendorStage()
    microscope._default_stage_coordinate_system = "Raw"
    microscope._compucentric_offset = microscope._read_compucentric_offset()
    return microscope


class TestThermoFisher:
    """xT's centre, read once at connect, plus the calibrated correction (FIB-655)."""

    def test_the_centre_is_minus_the_specimen_offset(self, monkeypatch):
        microscope = _thermo(monkeypatch=monkeypatch)

        assert microscope.rotation_centre == pytest.approx((513e-6, -794e-6))
        assert microscope._vendor_stage.systems[-1] == "Raw"

    def test_the_correction_is_added(self, monkeypatch):
        microscope = _thermo((19e-6, 11e-6), monkeypatch=monkeypatch)

        assert microscope.rotation_centre == pytest.approx((532e-6, -783e-6))

    def test_moves_and_images_use_the_same_centre(self, monkeypatch):
        microscope = _thermo((19e-6, 11e-6), monkeypatch=monkeypatch)
        microscope.system.stage.rotation = True
        pos = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0)
        target = microscope._get_compucentric_rotation_position(pos)

        assert (target.x, target.y) == pytest.approx(
            (2 * 532e-6 - pos.x, 2 * -783e-6 - pos.y)
        )
        drawn = _transform_position(pos, microscope.rotation_centre)
        assert (drawn.x, drawn.y) == pytest.approx((target.x, target.y))

    def test_a_compustage_reports_no_centre(self, monkeypatch):
        microscope = _thermo(monkeypatch=monkeypatch)
        microscope._compucentric_offset = None

        assert microscope.rotation_centre is None


class TestCorrectionConfiguration:
    def test_it_defaults_to_zero_and_is_not_written(self):
        system = utils.setup_session(manufacturer="Demo")[0].system

        assert system.stage.rotation_centre_correction == (0.0, 0.0)
        assert "rotation_centre_correction" not in system.to_dict()["calibration"]

    def test_it_round_trips_under_calibration(self):
        from fibsem.structures import SystemSettings

        system = utils.setup_session(manufacturer="Demo")[0].system
        system.stage.rotation_centre_correction = (19e-6, 11e-6)
        ddict = system.to_dict()

        assert ddict["calibration"]["rotation_centre_correction"] == [19e-6, 11e-6]
        loaded = SystemSettings.from_dict(ddict)
        assert loaded.stage.rotation_centre_correction == (19e-6, 11e-6)


class TestMoveToOrientation:
    """A move to a named orientation turns about the same centre a saved position is
    converted about, so the correction reaches the Move to FIB button (FIB-655)."""

    def test_a_half_turn_carries_xy_round_the_centre(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, y=2e-3))

        at_fib = microscope.move_to_orientation("FIB")

        assert (at_fib.x, at_fib.y) == pytest.approx(
            (2 * CENTRE[0] - 1e-3, 2 * CENTRE[1] - 2e-3)
        )
        assert at_fib.r == pytest.approx(microscope.get_orientation("FIB").r)

    def test_there_and_back_returns_to_the_feature(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, y=2e-3))

        microscope.move_to_orientation("FIB")
        back = microscope.move_to_orientation("SEM")

        assert (back.x, back.y) == pytest.approx((1e-3, 2e-3))

    def test_staying_at_an_orientation_keeps_xy(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = CENTRE
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, y=2e-3))

        again = microscope.move_to_orientation("SEM")

        assert (again.x, again.y) == pytest.approx((1e-3, 2e-3))

    def test_without_a_reported_centre_the_vendor_places_xy(self):
        """Tescan and Odemis report no centre: their own rotation decides, as before."""
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.rotation_centre = None
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, y=2e-3))

        at_fib = microscope.move_to_orientation("FIB")

        assert (at_fib.x, at_fib.y) == pytest.approx((1e-3, 2e-3))
