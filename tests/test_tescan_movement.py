"""Tescan's own coincidence move from the SEM view.

Every other Tescan move runs the shared maths through TescanStage's frame conversion
(FIB-1114); this one is kept as Tescan's because it is the one verified on hardware
(2026-08-26: the coincident move from the SEM leaves the FIB image untouched). It slides
the sample along the FIB line of sight, computed in Tescan's own frame: y rides the
tilt module, z is chamber-vertical with +z down. See
https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414 for the derivation.

No hardware or Tescan SDK required: the microscope object is created without
__init__, the stage state is stubbed and there is no stage device, so the move is
recorded in Tescan's frame.
"""

import os
import threading

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemStagePosition

TESCAN_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")

# rotation conventions from tescan-configuration.yaml
ROTATION_FLAT_TO_EB = np.deg2rad(180)  # stage.rotation_reference
ROTATION_FLAT_TO_ION = np.deg2rad(0)  # stage.rotation_180

FIB_COLUMN_TILT = np.deg2rad(55)


def make_microscope(
    pretilt_deg: float = 35.0,
    stage_position: FibsemStagePosition = None,
    scan_rotation_deg: float = 0.0,
) -> TescanMicroscope:
    """Create a TescanMicroscope without the SDK, with stubbed stage state."""
    system = utils.load_microscope_configuration(TESCAN_CONFIG_PATH).system
    system.stage.shuttle_pre_tilt = pretilt_deg

    microscope = object.__new__(TescanMicroscope)  # skip __init__ (requires SDK)
    microscope._connection_lock = threading.RLock()
    microscope.system = system

    if stage_position is None:
        stage_position = FibsemStagePosition(
            x=0, y=0, z=0, r=ROTATION_FLAT_TO_EB, t=0, coordinate_system="RAW"
        )
    microscope._test_stage_position = stage_position
    microscope._recorded_moves = []
    microscope.get_stage_position = lambda: microscope._test_stage_position
    # get_scan_rotation returns radians (codebase convention); the test param is degrees
    microscope.get_scan_rotation = lambda beam_type: np.deg2rad(scan_rotation_deg)
    microscope.move_stage_relative = lambda position: microscope._recorded_moves.append(
        position
    )
    return microscope


def stage_at(
    tilt_deg: float, rotation: float = ROTATION_FLAT_TO_EB
) -> FibsemStagePosition:
    return FibsemStagePosition(
        x=0, y=0, z=0, r=rotation, t=np.deg2rad(tilt_deg), coordinate_system="RAW"
    )


# ---------------------------------------------------------------------------
# move_coincident_from_sem: slide along the FIB axis, so the FIB image is
# untouched while the SEM offset closes
# ---------------------------------------------------------------------------


def test_tescan_declares_the_sem_view_as_supported():
    """The movement widget's vertical gate and the alignment STAGE_VERTICAL path both
    ask supports_vertical_move; declaring the view is what lights them up (FIB-785)."""
    m = make_microscope()

    assert m.supports_vertical_move(BeamType.ELECTRON)
    assert m.supports_vertical_move(BeamType.ION)


def test_the_deprecated_name_is_the_electron_branch():
    """``move_coincident_from_sem`` is kept for one release for custom scripts. It must
    stay exactly the ELECTRON branch, not a second implementation of it."""
    dx, dy = 1e-6, 10e-6

    m_new = make_microscope(stage_position=stage_at(-15.0))
    m_new.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    m_old = make_microscope(stage_position=stage_at(-15.0))
    m_old.move_coincident_from_sem(dx=dx, dy=dy)

    assert len(m_new._recorded_moves) == 1
    assert m_new._recorded_moves[0].is_close2(m_old._recorded_moves[0], tol=1e-12)


def test_relaxation_is_accepted_but_not_applied_from_the_sem_view():
    """The automated coincidence alignment passes relaxation to every backend.
    Tescan's own SEM-view move accepts it and does not apply it. The FIB view takes
    the shared move, which does."""
    beam_type = BeamType.ELECTRON
    dx, dy = 1e-6, 10e-6

    m_default = make_microscope(pretilt_deg=40.0, stage_position=stage_at(20.0))
    m_default.vertical_move(dy=dy, dx=dx, beam_type=beam_type)

    m_relaxed = make_microscope(pretilt_deg=40.0, stage_position=stage_at(20.0))
    m_relaxed.vertical_move(dy=dy, dx=dx, beam_type=beam_type, relaxation=0.5)

    (default,) = m_default._recorded_moves
    (relaxed,) = m_relaxed._recorded_moves
    assert relaxed.is_close2(default, tol=1e-12)


def test_coincident_from_sem_explicit_values_flat():
    """At zero tilt: y = dy, z = dy*cot(55) -- NOT a plain lateral move even flat."""
    m = make_microscope(pretilt_deg=40.0, stage_position=stage_at(0.0))
    dy = 10e-6

    m.move_coincident_from_sem(dx=0.0, dy=dy)

    (move,) = m._recorded_moves
    assert move.y == pytest.approx(-10.00e-6, abs=0.01e-6)  # inverted, like stable_move
    assert move.z == pytest.approx(7.00e-6, abs=0.01e-6)  # dy/tan(55 deg), +z down


def test_coincident_from_sem_logged_milling_pose_regression():
    """Pinned at the 2026-07-22 session's milling pose (stage tilt 20 deg).

    tan(t) + cot(55) == 1/cos(t) exactly at t = 2*55 - 90 = 20 deg, so the y and
    z commands come out equal at this pose -- not a typo.
    """
    m = make_microscope(pretilt_deg=40.0, stage_position=stage_at(20.0))
    dy = 33.3e-6

    m.move_coincident_from_sem(dx=0.0, dy=dy)

    (move,) = m._recorded_moves
    assert move.y == pytest.approx(-35.44e-6, abs=0.01e-6)
    assert move.z == pytest.approx(35.44e-6, abs=0.01e-6)


@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 60.0])
def test_coincident_from_sem_is_pretilt_independent(tilt_deg):
    """The FIB axis is chamber-fixed, so the sample plane (and the pre-tilt)
    cancels out of this move entirely -- unlike stable_move."""
    dy = 12e-6
    moves = []
    for pretilt_deg in (0.0, 40.0):
        m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))
        m.move_coincident_from_sem(dx=0.0, dy=dy)
        moves.append(m._recorded_moves[0])

    assert moves[0].y == pytest.approx(moves[1].y)
    assert moves[0].z == pytest.approx(moves[1].z)


@pytest.mark.parametrize("pretilt_deg", [0.0, 40.0])
@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 60.0])
def test_coincident_move_is_invisible_in_fib_view(tilt_deg, pretilt_deg):
    """The defining property, tested as such: reconstruct the chamber displacement
    from the commanded move (y rides the tilt, z is chamber-vertical with +z down)
    and read it through each view's image-y direction e_y(eta) = (cos eta, sin eta).
    The FIB must read zero (the move is along its line of sight); the SEM must
    read exactly dy (the offset closes). Either term of the formula reversed
    fails one of the two assertions."""
    dy = 12e-6
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))

    m.move_coincident_from_sem(dx=0.0, dy=dy)

    (move,) = m._recorded_moves
    tilt = np.deg2rad(tilt_deg)
    y_move = -move.y  # undo the stage-axis inversion
    chamber_h = y_move * np.cos(tilt)
    chamber_v = y_move * np.sin(tilt) - move.z

    fib_reading = chamber_h * np.cos(FIB_COLUMN_TILT) + chamber_v * np.sin(
        FIB_COLUMN_TILT
    )
    sem_reading = chamber_h  # SEM column is vertical (column_tilt 0)

    assert fib_reading == pytest.approx(0.0, abs=1e-12)
    assert sem_reading == pytest.approx(dy)


def test_coincident_from_sem_scan_rotation_180_flips_all_axes():
    """At 180 deg scan rotation dx and dy flip on the way in, so every commanded
    axis (x, y, and the z that follows dy) changes sign."""
    dx, dy = 2e-6, 10e-6
    m0 = make_microscope(pretilt_deg=40.0, stage_position=stage_at(20.0))
    m180 = make_microscope(
        pretilt_deg=40.0, stage_position=stage_at(20.0), scan_rotation_deg=180.0
    )

    m0.move_coincident_from_sem(dx=dx, dy=dy)
    m180.move_coincident_from_sem(dx=dx, dy=dy)

    (move0,) = m0._recorded_moves
    (move180,) = m180._recorded_moves
    assert move180.x == pytest.approx(-move0.x)
    assert move180.y == pytest.approx(-move0.y)
    assert move180.z == pytest.approx(-move0.z)
