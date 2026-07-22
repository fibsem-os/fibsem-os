"""Tests for the Tescan sample-plane stage movement geometry.

Tescan stages have the z-axis below the tilt axis (chamber-fixed y/z translation
axes), so stable_move decomposes the sample-plane move using the full chamber-frame
sample inclination. See docs/design/tescan-stable-move.md for the derivation.

These tests lock in the derived math and the internal sign contracts
(stable_move <-> inverse round trips). The absolute hardware sign conventions
(tilt sense, +z direction, stage x/y inversion) are flagged with
TODO(hardware-verify) in code and can only be confirmed on an instrument.

No hardware or Tescan SDK required: the microscope object is created without
__init__ and the stage state is stubbed.
"""

import os

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
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

TESCAN_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")

# rotation conventions from tescan-configuration.yaml
ROTATION_FLAT_TO_EB = np.deg2rad(180)  # stage.rotation_reference
ROTATION_FLAT_TO_ION = np.deg2rad(0)   # stage.rotation_180

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
    microscope.move_stage_relative = lambda position: microscope._recorded_moves.append(position)
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
        electron_beam=BeamSettings(beam_type=BeamType.ELECTRON, scan_rotation=scan_rotation),
        ion_beam=BeamSettings(beam_type=BeamType.ION, scan_rotation=scan_rotation),
    )
    md = FibsemImageMetadata(
        image_settings=ImageSettings(resolution=(shape[1], shape[0]), beam_type=beam_type),
        pixel_size=Point(pixel_size, pixel_size),
        microscope_state=state,
        system=system,
    )
    return FibsemImage(data=np.zeros(shape, dtype=np.uint8), metadata=md)


def stage_at(tilt_deg: float, rotation: float = ROTATION_FLAT_TO_EB) -> FibsemStagePosition:
    return FibsemStagePosition(
        x=0, y=0, z=0, r=rotation, t=np.deg2rad(tilt_deg), coordinate_system="RAW"
    )


# ---------------------------------------------------------------------------
# _y_corrected_stage_movement (forward)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 17.0, 30.0, 45.0])
@pytest.mark.parametrize("pretilt_deg", [0.0, 20.0, 35.0])
def test_sem_y_move_equals_expected_y(tilt_deg, pretilt_deg):
    """For the vertical SEM beam the y stage move equals the image dy exactly
    (chamber-fixed y axis), and z compensates with dy * tan(sample_inclination)."""
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ELECTRON)

    inclination = np.deg2rad(tilt_deg) - np.deg2rad(pretilt_deg)
    assert move.y == pytest.approx(dy)
    assert move.z == pytest.approx(dy * np.tan(inclination))


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_flat_sample_has_no_z_move(beam_type):
    """When the sample is flat to the SEM (tilt == pre-tilt) the move stays in-plane,
    matching the previous placeholder implementation (y only, z = 0)."""
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_at(35.0))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=beam_type)

    assert move.z == pytest.approx(0.0, abs=1e-12)
    if beam_type is BeamType.ELECTRON:
        assert move.y == pytest.approx(dy)
    else:
        # FIB views the flat sample at the column tilt -> perspective stretch
        assert move.y == pytest.approx(dy / np.cos(FIB_COLUMN_TILT))


def test_fib_move_explicit_values():
    """Explicit numeric check of the FIB case against the derivation:
    d = dy / cos(incl - column_tilt); y = d*cos(incl); z = d*sin(incl)."""
    tilt_deg, pretilt_deg = 17.0, 35.0
    m = make_microscope(pretilt_deg=pretilt_deg, stage_position=stage_at(tilt_deg))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ION)

    inclination = np.deg2rad(tilt_deg - pretilt_deg)  # -18 deg
    d = dy / np.cos(inclination - FIB_COLUMN_TILT)
    assert move.y == pytest.approx(d * np.cos(inclination))
    assert move.z == pytest.approx(d * np.sin(inclination))


def test_move_magnitude_equals_sample_plane_distance():
    """The (y, z) stage move has the same magnitude as the sample-plane distance."""
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_at(17.0))
    dy = 2e-6

    move = m._y_corrected_stage_movement(expected_y=dy, beam_type=BeamType.ION)

    inclination = np.deg2rad(17.0 - 35.0)
    d = dy / np.cos(inclination - FIB_COLUMN_TILT)
    assert np.hypot(move.y, move.z) == pytest.approx(abs(d))


def test_pretilt_sign_flips_when_facing_ion():
    """Rotating 180 deg (facing the FIB) flips the pre-tilt sign, changing the
    sample inclination from (tilt - pretilt) to (tilt + pretilt)."""
    dy = 2e-6
    tilt_deg, pretilt_deg = 10.0, 35.0

    m_eb = make_microscope(pretilt_deg, stage_at(tilt_deg, ROTATION_FLAT_TO_EB))
    m_ion = make_microscope(pretilt_deg, stage_at(tilt_deg, ROTATION_FLAT_TO_ION))

    move_eb = m_eb._y_corrected_stage_movement(dy, BeamType.ELECTRON)
    move_ion = m_ion._y_corrected_stage_movement(dy, BeamType.ELECTRON)

    assert move_eb.z == pytest.approx(dy * np.tan(np.deg2rad(tilt_deg - pretilt_deg)))
    assert move_ion.z == pytest.approx(dy * np.tan(np.deg2rad(tilt_deg + pretilt_deg)))


# ---------------------------------------------------------------------------
# stable_move / project_stable_move
# ---------------------------------------------------------------------------

def test_stable_move_applies_axis_inversion():
    """stable_move applies the empirical stage-axis inversion (x=-dx, y=-y_chamber)
    after the trig, leaving z independent."""
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_at(17.0))
    dx, dy = 1e-6, 2e-6

    m.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

    assert len(m._recorded_moves) == 1
    move = m._recorded_moves[0]
    inclination = np.deg2rad(17.0 - 35.0)
    assert move.x == pytest.approx(-dx)
    assert move.y == pytest.approx(-dy)  # SEM: y_chamber == dy, inverted
    assert move.z == pytest.approx(dy * np.tan(inclination))  # z not inverted


def test_stable_move_scan_rotation_180_flips_xy():
    """At 180 deg scan rotation the image axes flip, cancelling the inversion."""
    m = make_microscope(
        pretilt_deg=35.0, stage_position=stage_at(17.0), scan_rotation_deg=180.0
    )
    dx, dy = 1e-6, 2e-6

    m.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

    move = m._recorded_moves[0]
    inclination = np.deg2rad(17.0 - 35.0)
    assert move.x == pytest.approx(dx)
    assert move.y == pytest.approx(dy)
    assert move.z == pytest.approx(-dy * np.tan(inclination))


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_project_stable_move_matches_stable_move(beam_type):
    """project_stable_move is the pure-math equivalent of stable_move."""
    base = stage_at(17.0)
    m = make_microscope(pretilt_deg=35.0, stage_position=base)
    dx, dy = 1e-6, 2e-6

    projected = m.project_stable_move(dx=dx, dy=dy, beam_type=beam_type, base_position=base)
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

    recovered = m._inverse_y_corrected_stage_movement(dy=dy_raw, dz=dz_raw, beam_type=beam_type)

    assert recovered == pytest.approx(dy)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("tilt_deg", [0.0, 17.0, 30.0])
def test_standalone_inverse_matches_microscope(beam_type, tilt_deg):
    """The metadata-based standalone inverse (tiled.py) matches the microscope method."""
    stage_position = stage_at(tilt_deg)
    m = make_microscope(pretilt_deg=35.0, stage_position=stage_position)
    image = make_image(m.system, stage_position, beam_type=beam_type)
    dy_raw, dz_raw = -1.5e-6, 0.5e-6

    from_microscope = m._inverse_y_corrected_stage_movement(dy=dy_raw, dz=dz_raw, beam_type=beam_type)
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
    image.metadata.system.info.manufacturer = manufacturer
    dy_raw, dz_raw = -1.5e-6, 0.5e-6

    generic = _inverse_y_corrected_stage_movement(image, dy=dy_raw, dz=dz_raw, beam_type=BeamType.ELECTRON)
    tescan = _inverse_y_corrected_stage_movement_tescan(image, dy=dy_raw, dz=dz_raw, beam_type=BeamType.ELECTRON)

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
    image.metadata.system.info.manufacturer = manufacturer
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
    image = make_image(m.system, base, beam_type=beam_type,
                       pixel_size=pixel_size, scan_rotation=np.pi)
    dx, dy = 1e-6, 2e-6

    pos = m.project_stable_move(dx=dx, dy=dy, beam_type=beam_type, base_position=base)
    point = calculate_reprojected_stage_position2(image, pos)

    centre_x = image.data.shape[1] / 2
    centre_y = image.data.shape[0] / 2
    dx_recovered = (point.x - centre_x) * pixel_size
    dy_recovered = -(point.y - centre_y) * pixel_size

    assert dx_recovered == pytest.approx(dx)
    assert dy_recovered == pytest.approx(dy)
