"""The pure movement geometry, and the stage moves every backend but Tescan shares.

`fibsem.geometry.movement` holds the maths the view-corrected moves used to carry
inline in `ThermoMicroscope`, which Demo and Odemis borrowed by calling
`ThermoMicroscope.stable_move(self, ...)`. The moves now live once on
`FibsemMicroscope`, so these pin two things: that the commands a move sends are exactly
what the pure functions return, and that the backends share the one implementation.

The pre-extraction formulae themselves are pinned in test_view_corrected_movement.py,
and the vertical-move physics in test_vertical_move.py.
"""

import dataclasses
import itertools
import os
from copy import deepcopy

import numpy as np
import pytest

import fibsem.config as cfg
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
    scope.stage_device.compustage = request.param
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
    microscope.stage_device.sim_position = position
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        microscope.set("scan_rotation", scan_rotation, beam_type)
    return microscope.get_stage_position()


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
        turned_over=microscope._stage_turned_over(pose.t),
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
        turned_over=microscope._stage_turned_over(pose.t),
    )
    # approx: the move reads its dy back off the stage, as position minus position
    assert (vertical.x, vertical.y, vertical.z) == pytest.approx(
        (expected.x, expected.y, expected.z), rel=1e-9, abs=1e-18
    )
    assert (vertical.y, vertical.z) != (0.0, 0.0)


class TestTheBackViewMirror:
    """A compustage mirrors a view of the back of the grid; the pose decides it."""

    GEOMETRY = FibsemHardwareGeometry(is_compustage=True)

    def _delta(self, tilt_deg, view_tilt=0.0, geometry=GEOMETRY):
        return image_to_stage_delta(
            0.0,
            10e-6,
            view_tilt=view_tilt,
            geometry=geometry,
            stage_rotation=0.0,
            stage_tilt=np.deg2rad(tilt_deg),
        )

    @pytest.mark.parametrize(
        "tilt_deg, view_tilt_deg, mirrored",
        [(-10, 0, False), (-170, 0, True), (-30, 52, False), (-128, 52, True)],
    )
    def test_only_a_back_view_is_mirrored(self, tilt_deg, view_tilt_deg, mirrored):
        offset = dataclasses.replace(self.GEOMETRY, is_compustage=False)
        view_tilt = np.deg2rad(view_tilt_deg)
        compustage = self._delta(tilt_deg, view_tilt)
        plain = self._delta(tilt_deg, view_tilt, geometry=offset)
        expected = -plain.y if mirrored else plain.y
        assert compustage.y == pytest.approx(expected)

    def test_the_live_move_does_not_read_the_orientation_table(self, monkeypatch):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_device.compustage = True

        def unread(*args, **kwargs):
            raise AssertionError("the live move asked the orientation table")

        monkeypatch.setattr(microscope, "get_stage_orientation", unread)
        microscope._view_stage_delta(0.0, 10e-6, view_tilt=0.0)


def test_a_stage_turns_the_sample_over_past_minus_90_by_default(microscope):
    turned_over = microscope.stage_device.turned_over
    assert not turned_over(np.deg2rad(-89.0))
    assert not turned_over(np.deg2rad(-90.0))
    assert turned_over(np.deg2rad(-91.0))
    assert turned_over(np.deg2rad(-180.0))


def test_vertical_move_asks_the_stage_device_where_the_sample_is_turned_over(
    microscope,
):
    """Not the stage type: a stage that says it is turned over at 0 gets the
    reversed move there, and one that never is gets the plain move at -128."""
    _pose(microscope, 0, 0, 0.0)
    microscope.vertical_move(dy=1e-6)
    plain = microscope.sent[-1]

    microscope.stage_device.turned_over = lambda tilt: True
    microscope.vertical_move(dy=1e-6)
    reversed_ = microscope.sent[-1]
    assert (reversed_.y, reversed_.z) == pytest.approx((-plain.y, -plain.z))

    microscope.stage_device.turned_over = lambda tilt: False
    _pose(microscope, -128, 0, 0.0)
    microscope.vertical_move(dy=1e-6)
    tilted = microscope.sent[-1]
    t = np.deg2rad(-128)
    chamber_vertical = tilted.y * np.sin(t) + tilted.z * np.cos(t)
    assert chamber_vertical == pytest.approx(1e-6 / np.sin(np.deg2rad(52)))


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


# FIB-1124: the shared moves ask the stage what it can do, not whether it is a
# compustage. These build real sessions from configuration -- the `microscope` fixture
# re-poses after connect, which does not select either behaviour.
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _session(compustage: bool):
    if compustage:
        microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    else:
        microscope, _ = utils.setup_session(manufacturer="Demo")
    assert microscope._fm_is_a_pose() is compustage
    return microscope


def _record(microscope, name):
    calls = []
    original = getattr(microscope, name)

    def recording(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    setattr(microscope, name, recording)
    return calls


class TestStableMoveRestoresTheWorkingDistanceOfALinkedStage:
    @pytest.mark.parametrize("compustage", [False, True])
    def test_as_before_on_each_mount(self, compustage):
        """An offset stage boots linked and a compustage can't link, so each mount
        keeps the answer the compustage branch gave it."""
        microscope = _session(compustage)
        restored = _record(microscope, "set_working_distance")

        microscope.stable_move(10e-6, 5e-6, BeamType.ELECTRON)

        assert microscope.get("stage_linked") is not compustage
        assert len(restored) == (0 if compustage else 1)

    def test_an_unlinked_offset_stage_is_left_alone(self):
        """The one behaviour change: z doesn't carry the working distance, so there is
        nothing to put back."""
        microscope = _session(compustage=False)
        microscope.stage_device.sim_linked = False
        restored = _record(microscope, "set_working_distance")

        microscope.stable_move(10e-6, 5e-6, BeamType.ELECTRON, static_wd=True)

        assert restored == []


class TestSafeMovementRotatesOnlyAStageThatRotates:
    def _rotations(self, microscope, target):
        """The moves that only rotate: the safe sequence's compucentric step."""
        sent = _record(microscope, "move_stage_absolute")
        microscope.safe_absolute_stage_movement(target)
        return [p for (p,) in sent if p.r is not None and p.x is None and p.t is None]

    @pytest.mark.parametrize("compustage", [False, True])
    def test_as_before_on_each_mount(self, compustage):
        microscope = _session(compustage)
        target = deepcopy(microscope.get_stage_position())
        target.x = (target.x or 0.0) + 1e-5

        rotations = self._rotations(microscope, target)

        assert microscope.system.stage.rotation is not compustage
        assert len(rotations) == (0 if compustage else 1)

    def test_the_capability_decides_not_the_flag(self):
        microscope = _session(compustage=False)
        microscope.system.stage.rotation = False
        target = deepcopy(microscope.get_stage_position())

        assert self._rotations(microscope, target) == []
