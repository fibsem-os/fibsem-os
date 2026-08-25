"""Tests for the Tescan sample-plane stage movement geometry.

Tescan stage axes: the y-axis rides ON the tilt module (a y command travels along
the tilted stage plate) while z stays chamber-vertical (+z down, verified on
hardware 2026-07-23) -- the translation axes are non-orthogonal at tilt. The
forward decomposition is therefore

    d = dy / cos(inclination - beam_tilt)          # perspective correction
    y = d * cos(inclination) / cos(stage_tilt)     # along the plate
    z = d * sin(corrected_pretilt) / cos(stage_tilt)

where inclination = stage_tilt - corrected_pretilt. Corrected 2026-08-25 after the
2026-07-22 session log showed the previous chamber-fixed-y model overshooting the
ion view ~1.65x at the milling pose (a chamber-fixed model cannot overshoot at
all, so the observation refuted it). See
https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414 for the derivation.

These tests lock in the derived math and the internal sign contracts
(stable_move <-> inverse round trips). Hardware-confirmed 2026-08-26: stable
moves at tilt centre the feature and hold focus in both views, and the
coincident move from the SEM leaves the FIB image untouched.

No hardware or Tescan SDK required: the microscope object is created without
__init__ and the stage state is stubbed.
"""

import os
import threading

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.imaging.tiled import (
    _inverse_y_corrected_stage_movement,
    _inverse_y_corrected_stage_movement_tescan,
    calculate_reprojected_stage_position2,
)
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
    SystemInfo,
)

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
    microscope.stage_is_compustage = False

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


def make_image(
    system,
    stage_position: FibsemStagePosition,
    beam_type: BeamType = BeamType.ELECTRON,
    pixel_size: float = 1e-7,
    shape=(1024, 1536),
    scan_rotation: float = 0.0,
) -> FibsemImage:
    """Create a FibsemImage with the metadata needed for reprojection.

    scan_rotation is in radians (codebase convention), matching what
    get_beam_settings stores in the image metadata.
    """
    state = MicroscopeState(
        stage_position=stage_position,
        electron_beam=BeamSettings(
            beam_type=BeamType.ELECTRON, scan_rotation=scan_rotation
        ),
        ion_beam=BeamSettings(beam_type=BeamType.ION, scan_rotation=scan_rotation),
    )
    # v6 (FIB-481) split the old `system` metadata into system_info (identity, carries
    # manufacturer) + hardware_geometry (the tilt/rotation geometry the inverse needs).
    system_info = SystemInfo(
        name="test",
        ip_address="localhost",
        manufacturer="Tescan",
        model="test",
        serial_number="test",
        hardware_version="test",
        software_version="test",
    )
    md = FibsemImageMetadata(
        image_settings=ImageSettings(
            resolution=(shape[1], shape[0]), beam_type=beam_type
        ),
        pixel_size=Point(pixel_size, pixel_size),
        microscope_state=state,
        system_info=system_info,
        hardware_geometry=FibsemHardwareGeometry.from_system_settings(system),
    )
    return FibsemImage(data=np.zeros(shape, dtype=np.uint8), metadata=md)


def stage_at(
    tilt_deg: float, rotation: float = ROTATION_FLAT_TO_EB
) -> FibsemStagePosition:
    return FibsemStagePosition(
        x=0, y=0, z=0, r=rotation, t=np.deg2rad(tilt_deg), coordinate_system="RAW"
    )


# ---------------------------------------------------------------------------
# _y_corrected_stage_movement (forward)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 17.0, 30.0, 45.0])
@pytest.mark.parametrize("pretilt_deg", [0.0, 20.0, 35.0])
def test_sem_y_move_is_dy_over_cos_tilt(tilt_deg, pretilt_deg):
    """For the vertical SEM beam the perspective and inclination cosines cancel:
    y = dy / cos(tilt) along the plate (dy exactly, measured horizontally), plus
    the pre-tilt z compensation."""
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ELECTRON)

    tilt = np.deg2rad(tilt_deg)
    pretilt = np.deg2rad(pretilt_deg)
    d = dy / np.cos(tilt - pretilt)
    assert move.y == pytest.approx(dy / np.cos(tilt))
    assert move.z == pytest.approx(d * np.sin(pretilt) / np.cos(tilt))


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 55.0])
def test_no_pretilt_move_is_pure_y(beam_type, tilt_deg):
    """With no shuttle pre-tilt the plate IS the sample plane: the whole move is a
    single y command of the sample-plane distance, and z is exactly zero at any
    tilt (the stage does the geometry itself)."""
    m = make_microscope(pretilt_deg=0.0, stage_position=stage_at(tilt_deg))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=beam_type)

    beam_tilt = 0.0 if beam_type is BeamType.ELECTRON else FIB_COLUMN_TILT
    d = dy / np.cos(np.deg2rad(tilt_deg) - beam_tilt)
    assert move.y == pytest.approx(d)
    assert move.z == pytest.approx(0.0, abs=1e-12)


def test_fib_move_explicit_values():
    """Explicit numeric check of the FIB case against the derivation:
    d = dy / cos(incl - column_tilt); y = d*cos(incl)/cos(tilt);
    z = d*sin(pretilt)/cos(tilt)."""
    tilt_deg, pretilt_deg = 17.0, 35.0
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ION)

    tilt = np.deg2rad(tilt_deg)
    inclination = np.deg2rad(tilt_deg - pretilt_deg)  # -18 deg
    d = dy / np.cos(inclination - FIB_COLUMN_TILT)
    assert move.y == pytest.approx(d * np.cos(inclination) / np.cos(tilt))
    assert move.z == pytest.approx(d * np.sin(np.deg2rad(pretilt_deg)) / np.cos(tilt))


def test_logged_milling_pose_regression():
    """The 2026-07-22 hardware session, ion view at the milling pose: stage tilt
    20 deg, shuttle pre-tilt 40 deg, inclination -20 deg (15 deg grazing), a
    +33.3 um click. The chamber-fixed model commanded y=121.0/z=+44.0 um and the
    feature overshot ~1.65x; the corrected command is y=128.7/z=+88.0 um -- z on
    the same side, twice the size. Pinned in absolute microns so a regression in
    any angle term fails loudly here first."""
    m = make_microscope(pretilt_deg=40.0, stage_position=stage_at(20.0))
    dy = 33.3e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ION)

    d = dy / np.cos(np.deg2rad(-20.0) - FIB_COLUMN_TILT)  # 128.66 um
    assert move.y == pytest.approx(d)  # cos(incl) == cos(tilt) at this exact pose
    assert move.y == pytest.approx(128.66e-6, rel=1e-3)
    assert move.z == pytest.approx(88.01e-6, rel=1e-3)
    assert move.z > 0


def test_pretilt_sign_flips_when_facing_ion():
    """Rotating 180 deg (facing the FIB) flips the pre-tilt sign: the sample
    inclination changes from (tilt - pretilt) to (tilt + pretilt) and the z
    compensation changes side with it."""
    dy = 2e-6
    tilt_deg, pretilt_deg = 10.0, 35.0
    tilt, pretilt = np.deg2rad(tilt_deg), np.deg2rad(pretilt_deg)

    m_eb = make_microscope(pretilt_deg, stage_at(tilt_deg, ROTATION_FLAT_TO_EB))
    m_ion = make_microscope(pretilt_deg, stage_at(tilt_deg, ROTATION_FLAT_TO_ION))

    move_eb = m_eb._y_corrected_stage_movement(dy, BeamType.ELECTRON)
    move_ion = m_ion._y_corrected_stage_movement(dy, BeamType.ELECTRON)

    d_eb = dy / np.cos(tilt - pretilt)
    d_ion = dy / np.cos(tilt + pretilt)
    assert move_eb.z == pytest.approx(d_eb * np.sin(pretilt) / np.cos(tilt))
    assert move_ion.z == pytest.approx(-d_ion * np.sin(pretilt) / np.cos(tilt))
    assert move_eb.z > 0 > move_ion.z


# ---------------------------------------------------------------------------
# stable_move / project_stable_move
# ---------------------------------------------------------------------------


def test_stable_move_applies_axis_inversion():
    """stable_move applies the empirical stage-axis inversion (x=-dx, y=-y_move)
    after the trig, leaving z independent."""
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_at(17.0))
    dx, dy = 1e-6, 2e-6

    m.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

    assert len(m._recorded_moves) == 1
    move = m._recorded_moves[0]
    tilt, pretilt = np.deg2rad(17.0), np.deg2rad(35.0)
    d = dy / np.cos(tilt - pretilt)
    assert move.x == pytest.approx(-dx)
    assert move.y == pytest.approx(-dy / np.cos(tilt))  # SEM y, inverted
    assert move.z == pytest.approx(d * np.sin(pretilt) / np.cos(tilt))  # z not inverted


def test_stable_move_scan_rotation_180_flips_xy():
    """At 180 deg scan rotation the image axes flip, cancelling the inversion."""
    m = make_microscope(
        pretilt_deg=35.0, stage_position=stage_at(17.0), scan_rotation_deg=180.0
    )
    dx, dy = 1e-6, 2e-6

    m.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

    move = m._recorded_moves[0]
    tilt, pretilt = np.deg2rad(17.0), np.deg2rad(35.0)
    d = dy / np.cos(tilt - pretilt)
    assert move.x == pytest.approx(dx)
    assert move.y == pytest.approx(dy / np.cos(tilt))
    assert move.z == pytest.approx(-d * np.sin(pretilt) / np.cos(tilt))


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_project_stable_move_matches_stable_move(beam_type):
    """project_stable_move is the pure-math equivalent of stable_move."""
    base = stage_at(17.0)
    m = make_microscope(pretilt_deg=35.0, stage_position=base)
    dx, dy = 1e-6, 2e-6

    projected = m.project_stable_move(
        dx=dx, dy=dy, beam_type=beam_type, base_position=base
    )
    m.stable_move(dx=dx, dy=dy, beam_type=beam_type)
    applied = m._recorded_moves[0]

    assert projected.x - base.x == pytest.approx(applied.x)
    assert projected.y - base.y == pytest.approx(applied.y)
    assert projected.z - base.z == pytest.approx(applied.z)


# ---------------------------------------------------------------------------
# inverse round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("rotation", [ROTATION_FLAT_TO_EB, ROTATION_FLAT_TO_ION])
@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 17.0, 30.0])
@pytest.mark.parametrize("pretilt_deg", [0.0, 20.0, 35.0])
def test_microscope_inverse_round_trip(beam_type, rotation, tilt_deg, pretilt_deg):
    """The inverse recovers the image dy from the raw stage deltas applied by
    stable_move (including the stage-axis inversion)."""
    m = make_microscope(pretilt_deg, stage_at(tilt_deg, rotation))
    dy = 2e-6

    chamber = m._y_corrected_stage_movement(expected_y=dy, beam_type=beam_type)
    dy_raw, dz_raw = -chamber.y, chamber.z  # as applied by stable_move

    recovered = m._inverse_y_corrected_stage_movement(
        dy=dy_raw, dz=dz_raw, beam_type=beam_type
    )

    assert recovered == pytest.approx(dy)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_inverse_uses_z_branch_when_pretilt_dominates(beam_type):
    """Round trip through the |sin(pretilt)| > |cos(inclination)| branch of the inverse.

    The inverse recovers the sample-plane distance from whichever forward component
    is better conditioned: y carries cos(inclination), z carries sin(pretilt). The
    z branch only engages when the inclination approaches vertical — facing the ion
    beam flips the pre-tilt sign, making the inclination (tilt + pretilt) = 95 deg
    here. That branch encodes the z convention independently of the y branch, so
    without this case a z-convention change can pass every other test while
    silently inverting the recovered dy.
    """
    tilt_deg, pretilt_deg = 60.0, 35.0
    m = make_microscope(pretilt_deg, stage_at(tilt_deg, ROTATION_FLAT_TO_ION))
    dy = 2e-6

    chamber = m._y_corrected_stage_movement(expected_y=dy, beam_type=beam_type)
    inclination = np.deg2rad(tilt_deg + pretilt_deg)
    assert abs(np.sin(np.deg2rad(pretilt_deg))) > abs(np.cos(inclination)), (
        "fixture must hit the z branch"
    )

    recovered = m._inverse_y_corrected_stage_movement(
        dy=-chamber.y, dz=chamber.z, beam_type=beam_type
    )

    assert recovered == pytest.approx(dy)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("tilt_deg", [0.0, 17.0, 30.0])
def test_standalone_inverse_matches_microscope(beam_type, tilt_deg):
    """The metadata-based standalone inverse (reprojection.py) matches the microscope method."""
    stage_position = stage_at(tilt_deg)
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_position)
    image = make_image(m.system, stage_position, beam_type=beam_type)
    dy_raw, dz_raw = -1.5e-6, 0.5e-6

    from_microscope = m._inverse_y_corrected_stage_movement(
        dy=dy_raw, dz=dz_raw, beam_type=beam_type
    )
    from_metadata = _inverse_y_corrected_stage_movement_tescan(
        image, dy=dy_raw, dz=dz_raw, beam_type=beam_type
    )

    assert from_metadata == pytest.approx(from_microscope)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_standalone_inverse_matches_microscope_z_branch(beam_type):
    """The standalone must match the microscope method in the z branch too.

    The z branch encodes the z convention independently of the y branch, and the two
    copies of the inverse once diverged in exactly that branch (dz vs -dz), silently
    placing reprojected positions at the opposite corner. The microscope method now
    delegates to the reprojection core, so this pins the delegation and the
    metadata-path geometry staying equivalent.
    """
    tilt_deg, pretilt_deg = 60.0, 35.0
    stage_position = stage_at(tilt_deg, ROTATION_FLAT_TO_ION)
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_position)
    image = make_image(m.system, stage_position, beam_type=beam_type)

    inclination = np.deg2rad(tilt_deg + pretilt_deg)  # 95 deg -> z branch
    assert abs(np.sin(np.deg2rad(pretilt_deg))) > abs(np.cos(inclination)), (
        "fixture must hit the z branch"
    )

    dy_raw, dz_raw = -1.5e-6, 0.5e-6
    from_microscope = m._inverse_y_corrected_stage_movement(
        dy=dy_raw, dz=dz_raw, beam_type=beam_type
    )
    from_metadata = _inverse_y_corrected_stage_movement_tescan(
        image, dy=dy_raw, dz=dz_raw, beam_type=beam_type
    )

    assert from_metadata == pytest.approx(from_microscope)


@pytest.mark.parametrize("manufacturer", ["Tescan", "TESCAN"])
def test_inverse_dispatches_on_manufacturer(manufacturer):
    """The generic tiled.py inverse routes Tescan images to the tescan version.

    A live scope reports "TESCAN" (all caps) while the config reports "Tescan";
    both must dispatch to the tescan inverse. An exact "Tescan" check fell back to
    the Thermo inverse for live images, flipping added minimap positions."""
    stage_position = stage_at(17.0)
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_position)
    image = make_image(m.system, stage_position)
    image.metadata.system_info.manufacturer = manufacturer
    dy_raw, dz_raw = -1.5e-6, 0.5e-6

    generic = _inverse_y_corrected_stage_movement(
        image, dy=dy_raw, dz=dz_raw, beam_type=BeamType.ELECTRON
    )
    tescan = _inverse_y_corrected_stage_movement_tescan(
        image, dy=dy_raw, dz=dz_raw, beam_type=BeamType.ELECTRON
    )

    assert generic == pytest.approx(tescan)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("manufacturer", ["Tescan", "TESCAN"])
def test_reprojection_round_trip(beam_type, manufacturer):
    """End-to-end: a position projected with project_stable_move reprojects back
    onto the image at the original (dx, dy) offset. Covers both manufacturer
    casings ("TESCAN" live, "Tescan" config) — a mismatch reprojects the added
    position to the opposite corner."""
    base = stage_at(17.0)
    m = make_microscope(pretilt_deg=35.0, stage_position=base)
    pixel_size = 1e-7
    image = make_image(m.system, base, beam_type=beam_type, pixel_size=pixel_size)
    image.metadata.system_info.manufacturer = manufacturer
    dx, dy = 1e-6, 2e-6

    pos = m.project_stable_move(dx=dx, dy=dy, beam_type=beam_type, base_position=base)
    point = calculate_reprojected_stage_position2(image, pos)

    centre_x = image.data.shape[1] / 2
    centre_y = image.data.shape[0] / 2
    dx_recovered = (point.x - centre_x) * pixel_size
    dy_recovered = -(point.y - centre_y) * pixel_size

    assert dx_recovered == pytest.approx(dx)
    assert dy_recovered == pytest.approx(dy)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_reprojection_round_trip_scan_rotation_180(beam_type):
    """Regression: at 180 deg scan rotation the click projection (project_stable_move,
    which flips dx/dy) and the reprojection (calculate_reprojected_stage_position2,
    which flips px_delta) must agree. If the stored scan_rotation units disagree with
    the microscope's get_scan_rotation, the marker lands at the opposite corner."""
    base = stage_at(17.0)
    m = make_microscope(pretilt_deg=35.0, stage_position=base, scan_rotation_deg=180.0)
    pixel_size = 1e-7
    # image metadata stores scan_rotation in radians (as get_beam_settings does)
    image = make_image(
        m.system, base, beam_type=beam_type, pixel_size=pixel_size, scan_rotation=np.pi
    )
    dx, dy = 1e-6, 2e-6

    pos = m.project_stable_move(dx=dx, dy=dy, beam_type=beam_type, base_position=base)
    point = calculate_reprojected_stage_position2(image, pos)

    centre_x = image.data.shape[1] / 2
    centre_y = image.data.shape[0] / 2
    dx_recovered = (point.x - centre_x) * pixel_size
    dy_recovered = -(point.y - centre_y) * pixel_size

    assert dx_recovered == pytest.approx(dx)
    assert dy_recovered == pytest.approx(dy)


# ---------------------------------------------------------------------------
# move_coincident_from_sem: slide along the FIB axis, so the FIB image is
# untouched while the SEM offset closes
# ---------------------------------------------------------------------------


def test_tescan_has_move_coincident_from_sem():
    """The movement widget's SEM-vertical gate and the alignment STAGE_VERTICAL
    path both dispatch on hasattr; adding the method is what lights them up."""
    assert hasattr(TescanMicroscope, "move_coincident_from_sem")


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
