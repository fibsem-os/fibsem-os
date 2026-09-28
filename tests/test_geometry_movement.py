"""The pure movement geometry, and the stage moves every backend but Tescan shares.

`fibsem.geometry.movement` holds the maths the view-corrected moves used to carry
inline in `ThermoMicroscope`, which Demo and Odemis borrowed by calling
`ThermoMicroscope.stable_move(self, ...)`. The moves now live once on
`FibsemMicroscope`, so these pin two things: that the commands a move sends are exactly
what the pure functions return, and that the backends share the one implementation.

The pre-extraction formulae themselves are pinned in test_view_corrected_movement.py,
and the vertical-move physics in test_vertical_move.py.
"""

import itertools

import numpy as np
import pytest

from fibsem import utils
from fibsem.geometry.movement import (
    apply_delta,
    fib_offset_after_sem_move,
    image_to_stage_delta,
    undo_scan_rotation,
    vertical_move_delta,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.autoscript import ThermoMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import BeamType, FibsemHardwareGeometry, FibsemStagePosition

# -128 is the compustage FIB orientation at the default pretilt of 0
TILTS_DEG = [-128, -50, 0, 18, 52]
ROTATIONS_DEG = [0, 180]
SCAN_ROTATIONS = [0.0, np.pi]
POSES = list(itertools.product(TILTS_DEG, ROTATIONS_DEG, SCAN_ROTATIONS))

SHARED_MOVES = (
    "stable_move",
    "vertical_move",
    "project_stable_move",
    "_vertical_move_from_fib",
    "_vertical_move_from_sem",
    "_y_corrected_stage_movement",
    "_inverse_y_corrected_stage_movement",
    "safe_absolute_stage_movement",
    "_safe_rotation_movement",
)


@pytest.fixture(params=[False, True], ids=["offset-stage", "compustage"])
def microscope(request):
    scope, _ = utils.setup_session(manufacturer="Demo")
    scope.stage_is_compustage = request.param
    scope._update_orientations()
    scope.sent = []
    move_stage_relative = scope.move_stage_relative

    def recording(position):
        scope.sent.append(position)
        return move_stage_relative(position)

    scope.move_stage_relative = recording
    return scope


def _pose(microscope, tilt_deg, rotation_deg, scan_rotation):
    position = FibsemStagePosition(
        x=1e-4,
        y=2e-4,
        z=3e-4,
        r=np.deg2rad(rotation_deg),
        t=np.deg2rad(tilt_deg),
        coordinate_system="RAW",
    )
    microscope.stage_system.position = position
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        microscope.set("scan_rotation", scan_rotation, beam_type)
    return microscope.get_stage_position()


def _is_fib_orientation(microscope):
    if not microscope.stage_is_compustage:
        return None
    return microscope.get_stage_orientation() == "FIB"


@pytest.mark.parametrize("tilt_deg, rotation_deg, scan_rotation", POSES)
@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_stable_move_sends_the_pure_delta(
    microscope, beam_type, tilt_deg, rotation_deg, scan_rotation
):
    pose = _pose(microscope, tilt_deg, rotation_deg, scan_rotation)
    dx, dy = undo_scan_rotation(12e-6, -7e-6, scan_rotation)
    expected = image_to_stage_delta(
        dx,
        dy,
        view_tilt=microscope._beam_view_tilt(beam_type),
        geometry=microscope.hardware_geometry(),
        stage_rotation=pose.r,
        stage_tilt=pose.t,
        is_fib_orientation=_is_fib_orientation(microscope),
    )

    microscope.stable_move(12e-6, -7e-6, beam_type)

    assert microscope.sent == [expected]


@pytest.mark.parametrize("tilt_deg, rotation_deg, scan_rotation", POSES)
@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_project_stable_move_lands_where_stable_move_goes(
    microscope, beam_type, tilt_deg, rotation_deg, scan_rotation
):
    pose = _pose(microscope, tilt_deg, rotation_deg, scan_rotation)
    projected = microscope.project_stable_move(12e-6, -7e-6, beam_type, pose)

    microscope.stable_move(12e-6, -7e-6, beam_type)

    assert projected == apply_delta(pose, microscope.sent[0])
    assert microscope.sent[0].x != 0.0  # it did move


@pytest.mark.parametrize("tilt_deg, rotation_deg, scan_rotation", POSES)
@pytest.mark.parametrize("relaxation", [1.0, 0.5])
def test_vertical_move_from_fib_sends_the_pure_delta(
    microscope, tilt_deg, rotation_deg, scan_rotation, relaxation
):
    pose = _pose(microscope, tilt_deg, rotation_deg, scan_rotation)
    expected = vertical_move_delta(
        dx=3e-6,
        dy=5e-6,
        scan_rotation=scan_rotation,
        fib_column_tilt=microscope.system.ion.column_tilt,
        stage_tilt=pose.t,
        is_compustage=microscope.stage_is_compustage,
        relaxation=relaxation,
    )

    microscope.vertical_move(
        dy=5e-6, dx=3e-6, beam_type=BeamType.ION, relaxation=relaxation
    )

    assert microscope.sent == [expected]


@pytest.mark.parametrize("tilt_deg, rotation_deg, scan_rotation", POSES)
def test_vertical_move_from_sem_is_a_stable_move_then_a_fib_correction(
    microscope, tilt_deg, rotation_deg, scan_rotation
):
    """Across poses, so both the milling-angle scaling (SEM and MILLING orientations)
    and its absence (elsewhere), and both scan-rotation signs, are reached."""
    pose = _pose(microscope, tilt_deg, rotation_deg, scan_rotation)

    microscope.vertical_move(dy=5e-6, dx=3e-6, beam_type=BeamType.ELECTRON)

    stable, vertical = microscope.sent
    milling_angle = None
    if microscope.get_stage_orientation(pose) in ("SEM", "MILLING"):
        milling_angle = microscope.get_current_milling_angle(pose)
    dy = fib_offset_after_sem_move(
        stage_dy=stable.y, ion_scan_rotation=scan_rotation, milling_angle=milling_angle
    )
    expected = vertical_move_delta(
        dx=0,
        dy=dy,
        scan_rotation=scan_rotation,
        fib_column_tilt=microscope.system.ion.column_tilt,
        stage_tilt=pose.t,
        is_compustage=microscope.stage_is_compustage,
    )
    # approx: the move reads its dy back off the stage, as position minus position
    assert (vertical.x, vertical.y, vertical.z) == pytest.approx(
        (expected.x, expected.y, expected.z), rel=1e-9, abs=1e-18
    )
    assert (vertical.y, vertical.z) != (0.0, 0.0)


class TestTheOrientationOverride:
    """The live move asks the orientation table; the saved-image path derives it."""

    GEOMETRY = FibsemHardwareGeometry(is_compustage=True)
    FIB_TILT = np.deg2rad(-128)

    def _delta(self, rotation, is_fib_orientation, geometry=GEOMETRY):
        return image_to_stage_delta(
            0.0,
            10e-6,
            view_tilt=0.0,
            geometry=geometry,
            stage_rotation=rotation,
            stage_tilt=self.FIB_TILT,
            is_fib_orientation=is_fib_orientation,
        )

    def test_none_derives_it_from_the_pose(self):
        assert self._delta(0.0, None) == self._delta(0.0, True)
        assert self._delta(np.pi / 2, None) == self._delta(np.pi / 2, False)

    def test_an_answer_overrides_the_pose(self):
        assert self._delta(np.pi / 2, True) == self._delta(0.0, True)
        assert self._delta(0.0, False) != self._delta(0.0, True)

    def test_it_is_ignored_off_a_compustage(self):
        offset = FibsemHardwareGeometry(is_compustage=False)
        assert self._delta(0.0, True, offset) == self._delta(0.0, None, offset)


def test_undo_scan_rotation_inverts_only_a_half_turn():
    assert undo_scan_rotation(1.0, 2.0, 0.0) == (1.0, 2.0)
    assert undo_scan_rotation(1.0, 2.0, np.pi) == (-1.0, -2.0)
    assert undo_scan_rotation(1.0, 2.0, np.pi / 2) == (1.0, 2.0)


def test_apply_delta_leaves_the_base_position_alone():
    base = FibsemStagePosition(x=1.0, y=2.0, z=3.0, r=0.5, t=0.25, name="base")
    landed = apply_delta(base, FibsemStagePosition(x=0.1, y=0.2, z=0.3))

    assert (base.x, base.y, base.z) == (1.0, 2.0, 3.0)
    assert (landed.x, landed.y, landed.z) == pytest.approx((1.1, 2.2, 3.3))
    assert (landed.r, landed.t, landed.name) == (0.5, 0.25, "base")


@pytest.mark.parametrize("backend", [ThermoMicroscope, DemoMicroscope])
@pytest.mark.parametrize("name", SHARED_MOVES)
def test_the_backends_share_one_implementation(backend, name):
    """FIB-643: Demo and Odemis called ThermoMicroscope's methods on themselves.

    Odemis is held to the same in test_odemis_vertical_move.py, which needs stubs.
    """
    assert getattr(backend, name) is getattr(FibsemMicroscope, name)
