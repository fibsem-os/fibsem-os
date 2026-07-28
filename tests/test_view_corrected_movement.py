"""Tests for the view-parameterised stage projection (FIB-133).

`_y_corrected_stage_movement` used to hardcode a two-branch choice between the
electron and ion columns. It is now a thin wrapper over
`_view_corrected_stage_movement(expected_y, view_tilt)`, where `view_tilt` is how
far the viewing axis sits from the electron column -- 0 for the electron column,
`column_tilt` for the ion column, and `camera_tilt` for a fluorescence camera.

Two different properties are pinned here, and they are deliberately kept apart:

*Inert* -- the view parameterisation itself changed nothing. `_reference_y_corrected`
and `_reference_inverse_y_corrected` are the original formulae copied verbatim, and
the sweeps assert the current code agrees with them (for the inverse, on the
non-compustage path, which the fix below leaves untouched).

*Deliberately changed* -- the inverse was never a faithful inverse of the forward on
a compustage: the forward flips `expected_y` again at the FIB orientation, while the
inverse used a stage-tilt threshold and never consulted orientation, so it returned
the wrong sign there. The correctness criterion is round-trip identity, so those
tests assert the round-trip rather than agreement with the old formula, and one test
pins the old formula's wrong answer so the defect cannot creep back.
"""

import numpy as np
import pytest

from fibsem import movement, utils
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import CameraImageTransform
from fibsem.structures import BeamType, FibsemStagePosition

# sweep: stage tilt x pretilt x rotation x beam x compustage
STAGE_TILTS_DEG = [-50, -23, 0, 15, 35, 52]
PRETILTS_DEG = [0, 35, 45]
ROTATIONS_DEG = [0, 180]
BEAM_TYPES = [BeamType.ELECTRON, BeamType.ION]
COMPUSTAGE = [False, True]


def _reference_y_corrected(microscope, expected_y: float, beam_type: BeamType):
    """Pre-refactor `_y_corrected_stage_movement`, copied verbatim."""
    sem_column_tilt = np.deg2rad(microscope.system.electron.column_tilt)
    fib_column_tilt = np.deg2rad(microscope.system.ion.column_tilt)

    stage_pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(
        microscope.system.stage.rotation_reference
    ) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(
        microscope.system.stage.rotation_180
    ) % (2 * np.pi)

    current_stage_position = microscope.get_stage_position()
    stage_rotation = current_stage_position.r % (2 * np.pi)
    stage_tilt = current_stage_position.t

    if microscope.stage_is_compustage:
        expected_y *= -1.0
        stage_tilt += np.pi

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
        PRETILT_SIGN = -1.0

    if microscope.stage_is_compustage and microscope.get_stage_orientation() == "FIB":
        expected_y *= -1.0
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

    if beam_type == BeamType.ELECTRON:
        perspective_tilt_adjustment = -corrected_pretilt_angle
    elif beam_type == BeamType.ION:
        perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

    y_sample_move = expected_y / np.cos(stage_tilt + perspective_tilt_adjustment)

    y_move = y_sample_move * np.cos(corrected_pretilt_angle)
    z_move = -y_sample_move * np.sin(corrected_pretilt_angle)

    return FibsemStagePosition(x=0, y=y_move, z=z_move)


def _reference_inverse_y_corrected(microscope, dy: float, dz: float, beam_type: BeamType):
    """Pre-*fix* `_inverse_y_corrected_stage_movement`, copied verbatim.

    Retained to pin that the non-compustage path is untouched, and to document the
    compustage behaviour this branch deliberately changes (see
    TestCompustageInverseRoundTrip): the old form used a stage-tilt threshold and
    never consulted orientation, so it inverted the sign at the FIB pose.
    """
    sem_column_tilt = np.deg2rad(microscope.system.electron.column_tilt)
    fib_column_tilt = np.deg2rad(microscope.system.ion.column_tilt)

    stage_pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(
        microscope.system.stage.rotation_reference
    ) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(
        microscope.system.stage.rotation_180
    ) % (2 * np.pi)

    current_stage_position = microscope.get_stage_position()
    stage_rotation = current_stage_position.r % (2 * np.pi) if current_stage_position.r is not None else 0.0
    stage_tilt = current_stage_position.t if current_stage_position.t is not None else 0.0

    compustage_sign = 1.0
    if microscope.stage_is_compustage:
        if stage_tilt <= 0:
            compustage_sign = -1.0
        stage_tilt += np.pi

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

    if beam_type == BeamType.ELECTRON:
        perspective_tilt_adjustment = -corrected_pretilt_angle
    elif beam_type == BeamType.ION:
        perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

    cos_pretilt = np.cos(corrected_pretilt_angle)
    sin_pretilt = np.sin(corrected_pretilt_angle)

    if abs(cos_pretilt) > abs(sin_pretilt):
        y_sample_move = dy / cos_pretilt
    else:
        y_sample_move = -dz / sin_pretilt

    expected_y = y_sample_move * np.cos(stage_tilt + perspective_tilt_adjustment)

    if microscope.stage_is_compustage:
        expected_y *= compustage_sign

    return expected_y


@pytest.fixture()
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


def _configure(microscope, *, pretilt_deg, rotation_deg, tilt_deg, compustage):
    """Pin the microscope into a known geometry (pure math, no stage motion)."""
    microscope.stage_is_compustage = compustage
    microscope.system.stage.shuttle_pre_tilt = pretilt_deg
    microscope._update_orientations()

    position = FibsemStagePosition(
        x=0.0, y=0.0, z=0.0,
        r=np.deg2rad(rotation_deg),
        t=np.deg2rad(tilt_deg),
    )
    microscope.get_stage_position = lambda: position  # type: ignore[method-assign]
    return position


class TestRefactorParity:
    """The extracted projection must reproduce the pre-refactor formulae exactly."""

    @pytest.mark.parametrize("tilt_deg", STAGE_TILTS_DEG)
    @pytest.mark.parametrize("pretilt_deg", PRETILTS_DEG)
    @pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
    @pytest.mark.parametrize("beam_type", BEAM_TYPES)
    @pytest.mark.parametrize("compustage", COMPUSTAGE)
    def test_forward_parity(
        self, microscope, tilt_deg, pretilt_deg, rotation_deg, beam_type, compustage
    ):
        _configure(
            microscope,
            pretilt_deg=pretilt_deg,
            rotation_deg=rotation_deg,
            tilt_deg=tilt_deg,
            compustage=compustage,
        )
        expected_y = 25e-6

        got = microscope._y_corrected_stage_movement(expected_y, beam_type=beam_type)
        want = _reference_y_corrected(microscope, expected_y, beam_type=beam_type)

        assert got.y == pytest.approx(want.y, rel=1e-12, abs=1e-18)
        assert got.z == pytest.approx(want.z, rel=1e-12, abs=1e-18)

    @pytest.mark.parametrize("tilt_deg", STAGE_TILTS_DEG)
    @pytest.mark.parametrize("pretilt_deg", PRETILTS_DEG)
    @pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
    @pytest.mark.parametrize("beam_type", BEAM_TYPES)
    def test_inverse_parity_non_compustage(
        self, microscope, tilt_deg, pretilt_deg, rotation_deg, beam_type
    ):
        """Pre-tilted shuttle stages are untouched by the compustage sign fix."""
        _configure(
            microscope,
            pretilt_deg=pretilt_deg,
            rotation_deg=rotation_deg,
            tilt_deg=tilt_deg,
            compustage=False,
        )
        dy, dz = 12e-6, -3e-6

        got = microscope._inverse_y_corrected_stage_movement(dy=dy, dz=dz, beam_type=beam_type)
        want = _reference_inverse_y_corrected(microscope, dy=dy, dz=dz, beam_type=beam_type)

        assert got == pytest.approx(want, rel=1e-12, abs=1e-18)


class TestViewTiltEquivalence:
    """A beam is just a view with a particular axis tilt."""

    def test_zero_view_tilt_is_the_electron_column(self, microscope):
        _configure(microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False)

        by_view = microscope._view_corrected_stage_movement(25e-6, view_tilt=0.0)
        by_beam = microscope._y_corrected_stage_movement(25e-6, beam_type=BeamType.ELECTRON)

        assert by_view.y == pytest.approx(by_beam.y)
        assert by_view.z == pytest.approx(by_beam.z)

    def test_column_tilt_view_is_the_ion_column(self, microscope):
        _configure(microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False)
        view_tilt = np.deg2rad(microscope.system.ion.column_tilt)

        by_view = microscope._view_corrected_stage_movement(25e-6, view_tilt=view_tilt)
        by_beam = microscope._y_corrected_stage_movement(25e-6, beam_type=BeamType.ION)

        assert by_view.y == pytest.approx(by_beam.y)
        assert by_view.z == pytest.approx(by_beam.z)

    def test_beam_view_tilt_values(self, microscope):
        assert microscope._beam_view_tilt(BeamType.ELECTRON) == 0.0
        assert microscope._beam_view_tilt(BeamType.ION) == pytest.approx(
            np.deg2rad(microscope.system.ion.column_tilt)
        )

    @pytest.mark.parametrize("view_tilt_deg", [0, 38, 52, 180])
    def test_inverse_round_trips(self, microscope, view_tilt_deg):
        _configure(microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False)
        view_tilt = np.deg2rad(view_tilt_deg)
        expected_y = 25e-6

        move = microscope._view_corrected_stage_movement(expected_y, view_tilt=view_tilt)
        recovered = microscope._inverse_view_corrected_stage_movement(
            dy=move.y, dz=move.z, view_tilt=view_tilt
        )

        assert recovered == pytest.approx(expected_y, rel=1e-9)


class TestCompustageInverseRoundTrip:
    """The inverse must mirror the forward at every compustage orientation.

    The forward flips `expected_y` for compustage and again at the FIB orientation;
    the inverse previously used a stage-tilt threshold and never consulted
    orientation, so it returned the wrong sign at the FIB pose.
    """

    # (tilt, expected orientation) for the compustage: SEM flat, milling, FIB, FM pose
    COMPUSTAGE_POSES = [(0, "SEM"), (-30, "MILLING"), (-128, "FIB"), (-180, "FM")]

    @pytest.mark.parametrize("tilt_deg, orientation", COMPUSTAGE_POSES)
    @pytest.mark.parametrize("view_tilt_deg", [0, 52, 180])
    def test_round_trips_at_every_compustage_orientation(
        self, microscope, tilt_deg, orientation, view_tilt_deg
    ):
        _configure(microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=tilt_deg, compustage=True)
        assert microscope.get_stage_orientation() == orientation, "test pose drifted"

        view_tilt = np.deg2rad(view_tilt_deg)
        expected_y = 25e-6

        move = microscope._view_corrected_stage_movement(expected_y, view_tilt=view_tilt)
        recovered = microscope._inverse_view_corrected_stage_movement(
            dy=move.y, dz=move.z, view_tilt=view_tilt
        )

        assert recovered == pytest.approx(expected_y, rel=1e-9)

    @pytest.mark.parametrize("tilt_deg, orientation", COMPUSTAGE_POSES)
    def test_beam_round_trips_at_every_compustage_orientation(
        self, microscope, tilt_deg, orientation
    ):
        """The same property through the public beam-typed wrappers."""
        _configure(microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=tilt_deg, compustage=True)
        expected_y = 25e-6

        for beam_type in BEAM_TYPES:
            move = microscope._y_corrected_stage_movement(expected_y, beam_type=beam_type)
            recovered = microscope._inverse_y_corrected_stage_movement(
                dy=move.y, dz=move.z, beam_type=beam_type
            )
            assert recovered == pytest.approx(expected_y, rel=1e-9), (
                f"{orientation} / {beam_type.name} does not round-trip"
            )

    def test_fib_pose_sign_was_inverted_before_the_fix(self, microscope):
        """Pin the specific defect: the pre-fix formula returns the wrong sign at FIB.

        This is the behaviour change this branch makes, stated explicitly so it cannot
        be reintroduced silently.
        """
        _configure(microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-128, compustage=True)
        assert microscope.get_stage_orientation() == "FIB"
        expected_y = 25e-6

        move = microscope._y_corrected_stage_movement(expected_y, beam_type=BeamType.ELECTRON)
        pre_fix = _reference_inverse_y_corrected(
            microscope, dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON
        )
        fixed = microscope._inverse_y_corrected_stage_movement(
            dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON
        )

        assert pre_fix == pytest.approx(-expected_y, rel=1e-9), "pre-fix formula was sign-inverted"
        assert fixed == pytest.approx(expected_y, rel=1e-9), "fixed formula round-trips"

    def test_non_compustage_round_trips_across_orientations(self, microscope):
        """Unchanged behaviour on a pre-tilted shuttle stage."""
        for rotation_deg, tilt_deg in [(0, 35), (0, 12), (180, -17)]:
            _configure(
                microscope,
                pretilt_deg=35,
                rotation_deg=rotation_deg,
                tilt_deg=tilt_deg,
                compustage=False,
            )
            for beam_type in BEAM_TYPES:
                move = microscope._y_corrected_stage_movement(25e-6, beam_type=beam_type)
                recovered = microscope._inverse_y_corrected_stage_movement(
                    dy=move.y, dz=move.z, beam_type=beam_type
                )
                assert recovered == pytest.approx(25e-6, rel=1e-9)


class TestTiledInverseMatchesMicroscope:
    """`tiled.py` keeps its own microscope-free copy for offline reprojection.

    It reads geometry from image metadata instead of a live microscope, so it can
    reproject saved images. It must still agree with the microscope method -- if it
    doesn't, a reprojected point lands somewhere other than where the stage would
    actually move.
    """

    @pytest.fixture()
    def image(self, microscope):
        from fibsem.structures import ImageSettings

        return microscope.acquire_image(
            ImageSettings(
                hfw=80e-6, resolution=[64, 64], beam_type=BeamType.ELECTRON, save=False
            )
        )

    @pytest.mark.parametrize(
        "tilt_deg, orientation", TestCompustageInverseRoundTrip.COMPUSTAGE_POSES
    )
    @pytest.mark.parametrize("beam_type", BEAM_TYPES)
    def test_agrees_with_microscope_at_every_compustage_orientation(
        self, microscope, image, tilt_deg, orientation, beam_type
    ):
        from fibsem.imaging.tiled import _inverse_y_corrected_stage_movement as tiled_inverse

        position = _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=tilt_deg, compustage=True
        )
        image.metadata.system = microscope.system
        image.metadata.microscope_state.stage_position = position

        dy, dz = 12e-6, -3e-6
        from_microscope = microscope._inverse_y_corrected_stage_movement(
            dy=dy, dz=dz, beam_type=beam_type
        )
        from_image = tiled_inverse(image, dy=dy, dz=dz, beam_type=beam_type)

        assert from_image == pytest.approx(from_microscope, rel=1e-9), (
            f"tiled.py disagrees with the microscope at {orientation}"
        )

    @pytest.mark.parametrize(
        "tilt_deg, orientation", TestCompustageInverseRoundTrip.COMPUSTAGE_POSES
    )
    def test_round_trips_against_the_forward(self, microscope, image, tilt_deg, orientation):
        from fibsem.imaging.tiled import _inverse_y_corrected_stage_movement as tiled_inverse

        position = _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=tilt_deg, compustage=True
        )
        image.metadata.system = microscope.system
        image.metadata.microscope_state.stage_position = position
        expected_y = 25e-6

        move = microscope._y_corrected_stage_movement(expected_y, beam_type=BeamType.ELECTRON)
        recovered = tiled_inverse(image, dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON)

        assert recovered == pytest.approx(expected_y, rel=1e-9)

    def test_non_compustage_is_untouched(self, microscope, image):
        """The pre-tilted shuttle path through tiled.py is unchanged."""
        from fibsem.imaging.tiled import _inverse_y_corrected_stage_movement as tiled_inverse

        position = _configure(
            microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False
        )
        image.metadata.system = microscope.system
        image.metadata.system.sim = dict(image.metadata.system.sim or {})
        image.metadata.system.sim["is_compustage"] = False
        image.metadata.microscope_state.stage_position = position

        dy, dz = 12e-6, -3e-6
        for beam_type in BEAM_TYPES:
            from_microscope = microscope._inverse_y_corrected_stage_movement(
                dy=dy, dz=dz, beam_type=beam_type
            )
            from_image = tiled_inverse(image, dy=dy, dz=dz, beam_type=beam_type)
            assert from_image == pytest.approx(from_microscope, rel=1e-9)


class TestCameraTilt:
    """The FM optical axis is derived from the mount, not configured."""

    def test_offset_mount_matches_the_ion_column(self, microscope):
        # METEOR / iFLM: optical axis parallel to the FIB column, offset along x
        microscope.stage_is_compustage = False
        fm = FluorescenceMicroscope(parent=microscope)
        assert fm.camera_tilt == pytest.approx(microscope.system.ion.column_tilt)

    def test_under_grid_mount_is_a_half_turn(self, microscope):
        # Arctis: camera mounted under the grid, looking up
        microscope.stage_is_compustage = True
        fm = FluorescenceMicroscope(parent=microscope)
        assert fm.camera_tilt == pytest.approx(180.0)

    def test_no_parent_is_zero(self):
        assert FluorescenceMicroscope().camera_tilt == 0.0

    def test_under_grid_mount_cancels_the_stage_flip(self, microscope):
        """Arctis images FM with the stage flipped; the half-turn camera_tilt cancels it.

        compustage has no pre-tilt, so the projection should come out flat-on to the
        camera: no foreshortening, pure y move.
        """
        _configure(microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True)
        microscope.system.electron.column_tilt = 0.0

        fm = FluorescenceMicroscope(parent=microscope)
        move = microscope._view_corrected_stage_movement(
            25e-6, view_tilt=np.deg2rad(fm.camera_tilt)
        )

        assert abs(move.y) == pytest.approx(25e-6, rel=1e-9)
        assert move.z == pytest.approx(0.0, abs=1e-18)


class TestMountTransform:
    """The mount correction is applied before the user's transform."""

    def test_defaults_to_no_correction(self):
        assert FluorescenceMicroscope().mount_transform is CameraImageTransform.NONE

    def test_default_pipeline_is_unchanged(self):
        fm = FluorescenceMicroscope()
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        assert np.array_equal(fm._apply_image_transform(data), data)

    @pytest.mark.parametrize(
        "transform, expected",
        [
            (CameraImageTransform.NONE, lambda d: d),
            (CameraImageTransform.FLIP_X, np.fliplr),
            (CameraImageTransform.FLIP_Y, np.flipud),
            (CameraImageTransform.ROTATE_180, lambda d: np.rot90(d, k=2)),
        ],
    )
    def test_user_transform_still_applies(self, transform, expected):
        fm = FluorescenceMicroscope()
        fm.set_image_transform(transform)
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        assert np.array_equal(fm._apply_image_transform(data), expected(data))

    def test_mount_correction_composes_before_user_transform(self, monkeypatch):
        """A driver overriding mount_transform gets it applied first."""
        fm = FluorescenceMicroscope()
        monkeypatch.setattr(
            type(fm), "mount_transform",
            property(lambda self: CameraImageTransform.FLIP_Y),
        )
        fm.set_image_transform(CameraImageTransform.FLIP_X)

        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        assert np.array_equal(
            fm._apply_image_transform(data), np.fliplr(np.flipud(data))
        )
