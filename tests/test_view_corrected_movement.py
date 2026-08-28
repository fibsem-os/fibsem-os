"""Tests for the view-parameterised stage projection (FIB-133).

`_y_corrected_stage_movement` used to hardcode a two-branch choice between the
electron and ion columns. It is now a thin wrapper over
`_view_corrected_stage_movement(expected_y, view_tilt)`, where `view_tilt` is how
far the viewing axis sits from the electron column -- 0 for the electron column,
`column_tilt` for the ion column, and `camera_tilt` for a fluorescence camera.

Two different properties are pinned here, and they are deliberately kept apart:

*Inert* -- the view parameterisation itself changed nothing. `_reference_y_corrected`
and `_reference_inverse_y_corrected` are the original formulae copied verbatim, and
the sweeps assert the current code agrees with them. For the inverse that agreement
now holds **on in-plane movements only**, which is the whole domain the forward can
produce; see the second deliberate change below.

*Deliberately changed (1)* -- the inverse was never a faithful inverse of the forward
on a compustage: the forward flips `expected_y` again at the FIB orientation, while
the inverse used a stage-tilt threshold and never consulted orientation, so it
returned the wrong sign there. The correctness criterion is round-trip identity, so
those tests assert the round-trip rather than agreement with the old formula, and one
test pins the old formula's wrong answer so the defect cannot creep back.

*Deliberately changed (2), FIB-766* -- the inverse is also fed movements the forward
cannot produce. A click slides the sample along its own surface, but two saved
positions can differ in height and a coincidence correction is chamber-vertical. The
old form read stage `dy` and discarded `dz`, so it was exact on the forward's line
and wrong off it, on **both** stage kinds. Same treatment as (1): the parity sweep is
held to in-plane movements, round-trip identity is asserted separately, and
`TestTheInverseCarriesHeight` pins the old formula's wrong answer off-plane. The full
derivation and its oracle live in `tests/test_projection_height.py`.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from fibsem import movement, utils
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import (
    AutoFocusMode,
    CameraImageTransform,
    ChannelSettings,
    OverviewParameters,
)
from fibsem.structures import BeamType, FibsemHardwareGeometry, FibsemStagePosition

# sweep: stage tilt x pretilt x rotation x beam x compustage
#
# -128 is the compustage FIB orientation for pretilt 0 (fib_column_tilt - pretilt - 180
# = 52 - 0 - 180). Without it `is_fib_orientation` was False in all 36 combinations, so
# the compustage sign flip and the pretilt-sign override it drives went untested
# despite compustage being parametrised. Added with FIB-481.
STAGE_TILTS_DEG = [-128, -50, -23, 0, 15, 35, 52]
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
    stage_rotation_flat_to_ion = np.deg2rad(microscope.system.stage.rotation_180) % (
        2 * np.pi
    )

    current_stage_position = microscope.get_stage_position()
    stage_rotation = current_stage_position.r % (2 * np.pi)
    stage_tilt = current_stage_position.t

    if microscope.stage_is_compustage:
        expected_y *= -1.0
        stage_tilt += np.pi

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_eb, atol=5
    ):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_ion, atol=5
    ):
        PRETILT_SIGN = -1.0

    if microscope.stage_is_compustage and microscope.get_stage_orientation() == "FIB":
        expected_y *= -1.0
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

    if beam_type == BeamType.ELECTRON:
        perspective_tilt_adjustment = -corrected_pretilt_angle
    elif beam_type == BeamType.ION:
        perspective_tilt_adjustment = -corrected_pretilt_angle - fib_column_tilt

    y_sample_move = expected_y / np.cos(stage_tilt + perspective_tilt_adjustment)

    y_move = y_sample_move * np.cos(corrected_pretilt_angle)
    z_move = -y_sample_move * np.sin(corrected_pretilt_angle)

    return FibsemStagePosition(x=0, y=y_move, z=z_move)


def _reference_inverse_y_corrected(
    microscope, dy: float, dz: float, beam_type: BeamType
):
    """Pre-*fix* `_inverse_y_corrected_stage_movement`, copied verbatim.

    Retained to pin the two behaviour changes since, and the domain on which nothing
    changed at all:

    * agreement still holds exactly for **in-plane** movements on a pre-tilted
      shuttle -- the whole domain the forward map can produce
    * the compustage sign at the FIB pose (see `TestCompustageInverseRoundTrip`): the
      old form used a stage-tilt threshold and never consulted orientation
    * the discarded height component (see `TestTheInverseCarriesHeight`): the
      `dy / cos` branch below drops `dz` whenever the pretilt cosine dominates, which
      on the Arctis (pretilt 0) is every pose
    """
    sem_column_tilt = np.deg2rad(microscope.system.electron.column_tilt)
    fib_column_tilt = np.deg2rad(microscope.system.ion.column_tilt)

    stage_pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)

    stage_rotation_flat_to_eb = np.deg2rad(
        microscope.system.stage.rotation_reference
    ) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(microscope.system.stage.rotation_180) % (
        2 * np.pi
    )

    current_stage_position = microscope.get_stage_position()
    stage_rotation = (
        current_stage_position.r % (2 * np.pi)
        if current_stage_position.r is not None
        else 0.0
    )
    stage_tilt = (
        current_stage_position.t if current_stage_position.t is not None else 0.0
    )

    compustage_sign = 1.0
    if microscope.stage_is_compustage:
        if stage_tilt <= 0:
            compustage_sign = -1.0
        stage_tilt += np.pi

    PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_eb, atol=5
    ):
        PRETILT_SIGN = 1.0
    if movement.rotation_angle_is_smaller(
        stage_rotation, stage_rotation_flat_to_ion, atol=5
    ):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

    if beam_type == BeamType.ELECTRON:
        perspective_tilt_adjustment = -corrected_pretilt_angle
    elif beam_type == BeamType.ION:
        perspective_tilt_adjustment = -corrected_pretilt_angle - fib_column_tilt

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
        x=0.0,
        y=0.0,
        z=0.0,
        r=np.deg2rad(rotation_deg),
        t=np.deg2rad(tilt_deg),
    )
    microscope.get_stage_position = lambda: position  # type: ignore[method-assign]
    return position


def _sync_image_metadata(image, microscope, position, *, is_compustage: bool) -> None:
    """Point an image's metadata at the geometry the microscope is configured for.

    `tiled.py` reads geometry from image metadata rather than from a microscope, so the
    compustage marker has to be set explicitly -- otherwise the test would silently
    depend on whatever the ambient default configuration happens to advertise, which
    differs between machines and CI.

    Since FIB-481 that is a recorded field rather than a `system.sim` entry standing in
    for one, so this sets what the projection actually reads.
    """
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = FibsemHardwareGeometry.from_system_settings(
        microscope.system, is_compustage=is_compustage
    )
    image.metadata.microscope_state.stage_position = position


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
    def test_inverse_parity_non_compustage_in_plane(
        self, microscope, tilt_deg, pretilt_deg, rotation_deg, beam_type
    ):
        """Pre-tilted shuttle stages are untouched, on in-plane movements.

        The movement is taken from the forward map rather than written down, which is
        what makes it in-plane: the forward only ever slides the sample along its own
        surface, and that is the entire domain a click or a stable move can reach. On
        it, the current inverse and the pre-fix formula agree to ~1e-20 at every pose.

        Off that domain they deliberately disagree, and the old answer is the wrong
        one -- FIB-766, pinned in `TestTheInverseCarriesHeight`. This test used to use
        a hardcoded `(12e-6, -3e-6)`, which is off-plane, so it was asserting the
        discarded-height behaviour without meaning to.
        """
        _configure(
            microscope,
            pretilt_deg=pretilt_deg,
            rotation_deg=rotation_deg,
            tilt_deg=tilt_deg,
            compustage=False,
        )
        move = microscope._y_corrected_stage_movement(25e-6, beam_type=beam_type)
        if abs(move.y) > 1e-2:
            pytest.skip("near-singular pose for this view; the forward blows up")

        got = microscope._inverse_y_corrected_stage_movement(
            dy=move.y, dz=move.z, beam_type=beam_type
        )
        want = _reference_inverse_y_corrected(
            microscope, dy=move.y, dz=move.z, beam_type=beam_type
        )

        assert got == pytest.approx(want, rel=1e-12, abs=1e-18)
        assert got == pytest.approx(25e-6, rel=1e-9), "and it round-trips"


class TestTheInverseCarriesHeight:
    """FIB-766. The deliberate off-plane change, on the pre-tilted shuttle path.

    The old form recovered a movement from stage `dy` alone -- `dy / cos(pretilt)`
    whenever the cosine dominated, `-dz / sin(pretilt)` otherwise -- so it read one
    axis and discarded the other. That is exact for the in-plane movements the forward
    produces and wrong for everything else, and the inverse is fed everything else:
    two saved positions can differ in height, and a coincidence correction is
    chamber-vertical.

    Held here as physics, so it does not depend on this codebase's sign conventions
    being right. The conventions themselves are pinned against a calibrated 3D model
    in `tests/test_projection_height.py`.
    """

    def test_the_old_form_discarded_the_height_component(self, microscope):
        """Pin the specific defect, so it cannot be reintroduced silently.

        A pure stage-z movement projected to exactly nothing, in every view, at every
        pose -- which is the failure that leaves no trace: the overview desynchronises
        from the sample and no marker moves to say so.
        """
        for tilt_deg in (-128, -50, -23, 0, 15, 35):
            _configure(
                microscope,
                pretilt_deg=0,
                rotation_deg=0,
                tilt_deg=tilt_deg,
                compustage=False,
            )
            for beam_type in BEAM_TYPES:
                old = _reference_inverse_y_corrected(
                    microscope, dy=0.0, dz=20e-6, beam_type=beam_type
                )
                assert old == pytest.approx(0.0, abs=1e-18), (
                    f"the pre-fix formula moved at t={tilt_deg} / {beam_type.name}; "
                    "this test no longer pins the defect it was written for"
                )

    @pytest.mark.parametrize("tilt_deg", [-128, -50, -23, 0, 15, 35, 52])
    def test_a_bare_height_change_now_projects_somewhere(self, microscope, tilt_deg):
        """And the current form does not discard it.

        Stated as "at least one view sees it", which is the invariant that actually
        holds: the two view axes are 52 deg apart, so a height change cannot be
        invisible to both. It *can* be invisible to one -- at t = -128 stage z lies
        exactly along the ion column's line of sight, and the ion view is correctly
        blind to it there, as the electron view is at t = 0. Asserting "the ion beam
        always sees it" looks stronger and is simply false; that version of this test
        failed at t = -128 and the code was right.

        Not asserted against values, which is the calibrated suite's job. The point
        here is only that the axis is no longer thrown away -- the old form showed
        nothing in *either* view at *every* pose.
        """
        _configure(
            microscope,
            pretilt_deg=0,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=False,
        )
        shifts = {
            beam_type.name: microscope._inverse_y_corrected_stage_movement(
                dy=0.0, dz=20e-6, beam_type=beam_type
            )
            for beam_type in BEAM_TYPES
        }
        assert any(abs(shift) > 1e-9 for shift in shifts.values()), (
            f"height discarded in every view at t={tilt_deg}: {shifts}"
        )

    @pytest.mark.parametrize("pretilt_deg", PRETILTS_DEG)
    @pytest.mark.parametrize("tilt_deg", [-50, -23, 0, 15, 35])
    def test_the_projection_is_the_view_frame_horizontal(
        self, microscope, tilt_deg, pretilt_deg
    ):
        """What the new form reduces to, and why it is credible.

        On a pre-tilted shuttle it collapses to the horizontal component of the stage
        movement measured in the view's own frame:

            expected_y = dy*cos(t - view_tilt) - dz*sin(t - view_tilt)

        The pretilt cancels out entirely -- which is the right shape, because where a
        *chamber* displacement lands in an orthographic view depends only on the view's
        angle from vertical and on the stage tilt that orients the stage axes in the
        chamber. The sample's own pretilt says how the stage axes relate to the
        surface, which this question does not ask.

        Recorded because FIB-766 rejected `dy*cos(t) - dz*sin(t)` as a candidate on the
        grounds that it disagreed with the old answer for in-plane movements. It does --
        the term it was missing is the view tilt, and with that the disagreement is
        confined to exactly the off-plane movements the old form got wrong.
        """
        _configure(
            microscope,
            pretilt_deg=pretilt_deg,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=False,
        )
        tilt = np.deg2rad(tilt_deg)
        for beam_type, view_tilt in (
            (BeamType.ELECTRON, 0.0),
            (BeamType.ION, np.deg2rad(microscope.system.ion.column_tilt)),
        ):
            for dy, dz in ((12e-6, -3e-6), (0.0, 20e-6), (-7e-6, 9e-6)):
                got = microscope._inverse_y_corrected_stage_movement(
                    dy=dy, dz=dz, beam_type=beam_type
                )
                want = dy * np.cos(tilt - view_tilt) - dz * np.sin(tilt - view_tilt)
                assert got == pytest.approx(want, rel=1e-9, abs=1e-18), (
                    f"pretilt {pretilt_deg} did not cancel at t={tilt_deg} / "
                    f"{beam_type.name}"
                )


class TestViewTiltEquivalence:
    """A beam is just a view with a particular axis tilt."""

    def test_zero_view_tilt_is_the_electron_column(self, microscope):
        _configure(
            microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False
        )

        by_view = microscope._view_corrected_stage_movement(25e-6, view_tilt=0.0)
        by_beam = microscope._y_corrected_stage_movement(
            25e-6, beam_type=BeamType.ELECTRON
        )

        assert by_view.y == pytest.approx(by_beam.y)
        assert by_view.z == pytest.approx(by_beam.z)

    def test_column_tilt_view_is_the_ion_column(self, microscope):
        _configure(
            microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False
        )
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
        _configure(
            microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False
        )
        view_tilt = np.deg2rad(view_tilt_deg)
        expected_y = 25e-6

        move = microscope._view_corrected_stage_movement(
            expected_y, view_tilt=view_tilt
        )
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
        _configure(
            microscope,
            pretilt_deg=0,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=True,
        )
        assert microscope.get_stage_orientation() == orientation, "test pose drifted"

        view_tilt = np.deg2rad(view_tilt_deg)
        expected_y = 25e-6

        move = microscope._view_corrected_stage_movement(
            expected_y, view_tilt=view_tilt
        )
        recovered = microscope._inverse_view_corrected_stage_movement(
            dy=move.y, dz=move.z, view_tilt=view_tilt
        )

        assert recovered == pytest.approx(expected_y, rel=1e-9)

    @pytest.mark.parametrize("tilt_deg, orientation", COMPUSTAGE_POSES)
    def test_beam_round_trips_at_every_compustage_orientation(
        self, microscope, tilt_deg, orientation
    ):
        """The same property through the public beam-typed wrappers."""
        _configure(
            microscope,
            pretilt_deg=0,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=True,
        )
        expected_y = 25e-6

        for beam_type in BEAM_TYPES:
            move = microscope._y_corrected_stage_movement(
                expected_y, beam_type=beam_type
            )
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
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-128, compustage=True
        )
        assert microscope.get_stage_orientation() == "FIB"
        expected_y = 25e-6

        move = microscope._y_corrected_stage_movement(
            expected_y, beam_type=BeamType.ELECTRON
        )
        pre_fix = _reference_inverse_y_corrected(
            microscope, dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON
        )
        fixed = microscope._inverse_y_corrected_stage_movement(
            dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON
        )

        assert pre_fix == pytest.approx(-expected_y, rel=1e-9), (
            "pre-fix formula was sign-inverted"
        )
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
                move = microscope._y_corrected_stage_movement(
                    25e-6, beam_type=beam_type
                )
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
        from fibsem.imaging.tiled import (
            _inverse_y_corrected_stage_movement as tiled_inverse,
        )

        position = _configure(
            microscope,
            pretilt_deg=0,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=True,
        )
        _sync_image_metadata(image, microscope, position, is_compustage=True)

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
    def test_round_trips_against_the_forward(
        self, microscope, image, tilt_deg, orientation
    ):
        from fibsem.imaging.tiled import (
            _inverse_y_corrected_stage_movement as tiled_inverse,
        )

        position = _configure(
            microscope,
            pretilt_deg=0,
            rotation_deg=0,
            tilt_deg=tilt_deg,
            compustage=True,
        )
        _sync_image_metadata(image, microscope, position, is_compustage=True)
        expected_y = 25e-6

        move = microscope._y_corrected_stage_movement(
            expected_y, beam_type=BeamType.ELECTRON
        )
        recovered = tiled_inverse(
            image, dy=move.y, dz=move.z, beam_type=BeamType.ELECTRON
        )

        assert recovered == pytest.approx(expected_y, rel=1e-9)

    def test_non_compustage_is_untouched(self, microscope, image):
        """The pre-tilted shuttle path through tiled.py is unchanged."""
        from fibsem.imaging.tiled import (
            _inverse_y_corrected_stage_movement as tiled_inverse,
        )

        position = _configure(
            microscope, pretilt_deg=35, rotation_deg=0, tilt_deg=20, compustage=False
        )
        _sync_image_metadata(image, microscope, position, is_compustage=False)

        dy, dz = 12e-6, -3e-6
        for beam_type in BEAM_TYPES:
            from_microscope = microscope._inverse_y_corrected_stage_movement(
                dy=dy, dz=dz, beam_type=beam_type
            )
            from_image = tiled_inverse(image, dy=dy, dz=dz, beam_type=beam_type)
            assert from_image == pytest.approx(from_microscope, rel=1e-9)


class TestCameraImageTransformIsFlipOnly:
    """Restricting the transform to flips is what makes the mapping two sign flips.

    Rotations moved into the driver's mount correction, so the remaining members form
    the Klein four-group: shape-preserving, self-inverse, no axis swap.
    """

    def test_no_rotations_are_offered(self):
        names = {t.name for t in CameraImageTransform}
        assert "ROTATE_90_CW" not in names
        assert "ROTATE_90_CCW" not in names

    @pytest.mark.parametrize("transform", list(CameraImageTransform))
    def test_every_transform_is_its_own_inverse(self, transform):
        dx, dy = 3.0, -7.0
        once = transform.apply_to_delta(dx, dy)
        twice = transform.apply_to_delta(*once)
        assert twice == (dx, dy)

    @pytest.mark.parametrize("transform", list(CameraImageTransform))
    def test_delta_mapping_matches_the_array_transform(self, transform):
        """The coordinate mapping and the pixel mapping must agree.

        Build an index grid, transform it as an image, and check that the pixel which
        ends up at a probe offset from centre is the one the delta mapping predicts.
        """
        fm = FluorescenceMicroscope()
        size = 9
        centre = size // 2
        ys, xs = np.mgrid[0:size, 0:size]
        coded = (ys * 100 + xs).astype(np.int32)

        fm.set_image_transform(transform)
        moved = fm._transform_array(coded, transform)

        for probe in [(2, 0), (0, 2), (2, 3), (-1, 4)]:
            dx, dy = probe
            value = moved[centre + dy, centre + dx]
            src_y, src_x = divmod(int(value), 100)
            # where the delta mapping says that displayed offset came from
            exp_dx, exp_dy = transform.apply_to_delta(dx, dy)
            assert (src_x - centre, src_y - centre) == (exp_dx, exp_dy), (
                f"{transform.name}: array and delta mappings disagree at {probe}"
            )

    def test_no_axis_swapping(self):
        """A pure x displacement stays on x for every remaining transform."""
        for transform in CameraImageTransform:
            dx, dy = transform.apply_to_delta(5.0, 0.0)
            assert dy == 0.0
            assert abs(dx) == 5.0

    def test_one_spelling_per_transform(self):
        """No two members may describe the same operation.

        A half turn used to exist as both FLIP_XY and ROTATE_180, which made
        `transform is FLIP_XY` silently false for a value that behaved identically.
        """
        seen = {}
        for transform in CameraImageTransform:
            signature = transform.apply_to_delta(1.0, 1.0)
            assert signature not in seen, (
                f"{transform.name} and {seen[signature].name} are the same operation"
            )
            seen[signature] = transform

    def test_removed_values_load_as_none_with_a_warning(self, caplog):
        """A configuration written before the rotations were removed must still load."""
        from fibsem.fm.structures import CameraSettings

        with caplog.at_level("WARNING"):
            settings = CameraSettings.from_dict(
                {"gain": 0.1, "offset": 0.0, "binning": 1, "transform": "rotate-90-cw"}
            )

        assert settings.transform is CameraImageTransform.NONE
        assert "rotate-90-cw" in caplog.text

    def test_a_stored_half_turn_migrates_rather_than_being_dropped(self):
        """`rotate-180` is the same operation as a double flip, so the setting survives.

        Falling back to NONE here would silently change the image orientation for
        anyone who had it configured.
        """
        from fibsem.fm.structures import CameraSettings

        settings = CameraSettings.from_dict(
            {"gain": 0.1, "offset": 0.0, "binning": 1, "transform": "rotate-180"}
        )

        assert settings.transform is CameraImageTransform.FLIP_XY
        # and it re-saves under the canonical spelling
        assert settings.to_dict()["transform"] == "flip-xy"

    def test_supported_values_still_round_trip(self):
        from fibsem.fm.structures import CameraSettings

        for transform in CameraImageTransform:
            restored = CameraSettings.from_dict(
                CameraSettings(transform=transform).to_dict()
            )
            assert restored.transform is transform


class TestFmStableMove:
    """Click a point in the FM image, the stage goes there, focus is held."""

    def test_requires_a_fluorescence_microscope(self, microscope):
        microscope.fm = None
        with pytest.raises(ValueError):
            microscope.fm_stable_move(1e-6, 1e-6)

    def test_uses_the_camera_tilt_projection(self, microscope):
        """The move must match the shared projection at the camera's own axis tilt."""
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
        microscope.fm = FluorescenceMicroscope(parent=microscope)
        moved = []
        microscope.move_stage_relative = lambda p: moved.append(p)  # type: ignore[method-assign]

        microscope.fm_stable_move(dx=4e-6, dy=25e-6)

        expected = microscope._view_corrected_stage_movement(
            expected_y=25e-6, view_tilt=np.deg2rad(microscope.fm.camera_tilt)
        )
        assert len(moved) == 1
        assert moved[0].x == pytest.approx(4e-6)
        assert moved[0].y == pytest.approx(expected.y)
        assert moved[0].z == pytest.approx(expected.z)

    def test_undoes_the_display_transform(self, microscope):
        """A flipped display must not send the stage the wrong way."""
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
        microscope.fm = FluorescenceMicroscope(parent=microscope)
        moved = []
        microscope.move_stage_relative = lambda p: moved.append(p)  # type: ignore[method-assign]

        microscope.fm.set_image_transform(CameraImageTransform.NONE)
        microscope.fm_stable_move(dx=4e-6, dy=25e-6)
        microscope.fm.set_image_transform(CameraImageTransform.FLIP_X)
        microscope.fm_stable_move(dx=4e-6, dy=25e-6)

        assert moved[1].x == pytest.approx(-moved[0].x), (
            "flip in x should reverse the x move"
        )
        assert moved[1].y == pytest.approx(moved[0].y), "flip in x should leave y alone"

    def test_reads_the_transform_live(self, microscope):
        """Changing the dropdown mid-session takes effect on the next move."""
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
        microscope.fm = FluorescenceMicroscope(parent=microscope)
        moved = []
        microscope.move_stage_relative = lambda p: moved.append(p)  # type: ignore[method-assign]

        for transform in (CameraImageTransform.NONE, CameraImageTransform.FLIP_Y):
            microscope.fm.set_image_transform(transform)
            microscope.fm_stable_move(dx=0.0, dy=25e-6)

        assert moved[1].y == pytest.approx(-moved[0].y)

    def test_does_not_touch_working_distance(self, microscope):
        """Working distance is beam bookkeeping; the FM move must leave it alone."""
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
        microscope.fm = FluorescenceMicroscope(parent=microscope)
        microscope.move_stage_relative = lambda p: None  # type: ignore[method-assign]
        calls = []
        microscope.set_working_distance = lambda *a, **k: calls.append((a, k))  # type: ignore[method-assign]

        microscope.fm_stable_move(dx=0.0, dy=25e-6)

        assert calls == []


# NOTE: these build their own sessions rather than taking the shared `microscope`
# fixture. A test that contrasts the two mounts needs two distinct microscopes -- with a
# common parent fixture they alias to one object, the second _configure silently wins,
# and the contrast the test is written to check quietly stops existing.


@pytest.fixture()
def compustage_fm():
    """The Arctis geometry: under-grid camera, no pre-tilt, FM pose at t = -180."""
    scope, _ = utils.setup_session(manufacturer="Demo")
    _configure(scope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True)
    scope.fm = FluorescenceMicroscope(parent=scope)
    return scope


@pytest.fixture()
def offset_fm():
    """A non-compustage FM mount: optical axis parallel to the FIB column, real pre-tilt.

    The suite otherwise only exercises the compustage geometry, where the absence of a
    pre-tilt pins z at zero and so hides a whole class of error -- see
    `test_z_is_only_exercised_by_the_offset_mount`.
    """
    scope, _ = utils.setup_session(manufacturer="Demo")
    _configure(scope, pretilt_deg=35, rotation_deg=0, tilt_deg=0, compustage=False)
    scope.fm = FluorescenceMicroscope(parent=scope)
    return scope


class TestProjectFmStableMove:
    """Where an FM displacement lands, without moving the stage."""

    def test_requires_a_fluorescence_microscope(self, microscope):
        microscope.fm = None
        with pytest.raises(ValueError):
            microscope.project_fm_stable_move(1e-6, 1e-6, FibsemStagePosition())

    def test_does_not_move_the_stage(self, compustage_fm):
        """It is a projection. Nothing may reach the stage."""
        moved = []
        compustage_fm.move_stage_relative = lambda p: moved.append(p)  # type: ignore[method-assign]
        compustage_fm.move_stage_absolute = lambda p: moved.append(p)  # type: ignore[method-assign]

        compustage_fm.project_fm_stable_move(
            4e-6, 25e-6, FibsemStagePosition(x=0, y=0, z=0)
        )

        assert moved == []

    @pytest.mark.parametrize("mount", ["compustage_fm", "offset_fm"])
    @pytest.mark.parametrize("transform", list(CameraImageTransform))
    def test_agrees_with_fm_stable_move(self, request, mount, transform):
        """The projection and the real move must land in the same place.

        Both go through `_fm_stage_delta`, so this pins them together rather than
        merely checking each against a hand-computed number. Swept over the display
        transform as well as the mount because the two now share one input convention:
        if either stopped undoing the transform, or started undoing it twice, this
        breaks.
        """
        scope = request.getfixturevalue(mount)
        scope.fm.set_image_transform(transform)
        base = FibsemStagePosition(
            x=1e-3, y=-2e-3, z=5e-4, r=0, t=0, coordinate_system="RAW"
        )

        moved = []
        scope.move_stage_relative = lambda p: moved.append(p)  # type: ignore[method-assign]
        scope.fm_stable_move(dx=4e-6, dy=25e-6)

        projected = scope.project_fm_stable_move(4e-6, 25e-6, base)

        assert projected.x == pytest.approx(base.x + moved[0].x)
        assert projected.y == pytest.approx(base.y + moved[0].y)
        assert projected.z == pytest.approx(base.z + moved[0].z)

    def test_is_absolute_not_relative(self, compustage_fm):
        """The result is measured from the base position, not from zero."""
        origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0)
        offset = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3)

        a = compustage_fm.project_fm_stable_move(4e-6, 25e-6, origin)
        b = compustage_fm.project_fm_stable_move(4e-6, 25e-6, offset)

        assert b.x - a.x == pytest.approx(offset.x)
        assert b.y - a.y == pytest.approx(offset.y)
        assert b.z - a.z == pytest.approx(offset.z)

    def test_leaves_the_base_position_untouched(self, compustage_fm):
        """It returns a new position rather than mutating the caller's."""
        base = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3)

        compustage_fm.project_fm_stable_move(4e-6, 25e-6, base)

        assert (base.x, base.y, base.z) == (1e-3, 2e-3, 3e-3)

    def test_undoes_the_display_transform(self, compustage_fm):
        """Same input convention as fm_stable_move: the frame the user is looking at.

        The counterpart of project_stable_move undoing the beam's scan rotation, and it
        applies to synthesised displacements just as much as to clicks -- `tiled.py`
        hands project_stable_move raw grid offsets and relies on that undo. A tile step
        is in the same frame as the tile it positions, and the mosaic canvas is in
        display space because stitch_tileset pastes content that already carries the
        transform. Arrangement and content have to agree or every seam breaks.
        """
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0)

        def project(transform):
            compustage_fm.fm.set_image_transform(transform)
            return compustage_fm.project_fm_stable_move(4e-6, 25e-6, base)

        none = project(CameraImageTransform.NONE)
        flip_x = project(CameraImageTransform.FLIP_X)
        flip_y = project(CameraImageTransform.FLIP_Y)

        assert flip_x.x == pytest.approx(-none.x), "flip in x must reverse the x move"
        assert flip_x.y == pytest.approx(none.y), "flip in x must leave y alone"
        assert flip_y.y == pytest.approx(-none.y), "flip in y must reverse the y move"
        assert flip_y.x == pytest.approx(none.x), "flip in y must leave x alone"

    def test_z_is_only_exercised_by_the_offset_mount(self, compustage_fm, offset_fm):
        """Why the offset fixture exists.

        Compustage has no pre-tilt, so z stays 0 and relative-step accumulation only
        drifts in y. An offset mount puts a real z on every step, which is what makes
        accumulating them a focus problem rather than a positioning one.
        """
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0)

        flat = compustage_fm.project_fm_stable_move(0.0, 25e-6, base)
        tilted = offset_fm.project_fm_stable_move(0.0, 25e-6, base)

        assert flat.z == pytest.approx(0.0, abs=1e-18)
        assert abs(tilted.z) > 1e-9

    def test_offset_mount_uses_the_ion_column_tilt(self, offset_fm):
        """The offset optical axis is parallel to the FIB column, so it must project like one."""
        assert offset_fm.fm.camera_tilt == pytest.approx(
            offset_fm.system.ion.column_tilt
        )

        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0)
        projected = offset_fm.project_fm_stable_move(0.0, 25e-6, base)
        expected = offset_fm._view_corrected_stage_movement(
            expected_y=25e-6, view_tilt=np.deg2rad(offset_fm.system.ion.column_tilt)
        )

        assert projected.y == pytest.approx(expected.y)
        assert projected.z == pytest.approx(expected.z)


class TestFmConsumersUseTheFmProjection:
    """Everything that moves from a fluorescence image must use the FM projection.

    Stepping with a beam type used the wrong view's foreshortening: on an offset mount
    the camera's axis parallels the ion column, so the SEM projection mis-scales the
    y pitch by roughly 1/cos(column_tilt).
    """

    def test_tileset_projects_through_the_camera_not_a_beam(self, microscope):
        """The tileset must position tiles with the FM projection, never a beam's.

        Asserted by spying on the two projections rather than by reading the source.
        The previous version grepped `inspect.getsource(acquire_tileset)` for
        "fm_stable_move", which was fragile twice over: it passed after the switch to
        `project_fm_stable_move` only because that name *contains* the string it looked
        for, and it broke the moment the movement moved into a runner class without
        anything about the behaviour changing.
        """
        from fibsem.fm import acquisition

        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
        microscope.fm = FluorescenceMicroscope(parent=microscope)
        # A fresh objective is retracted, and a run refuses that (FIB-417) before it
        # projects anything. This test is about which projection tiles are placed with,
        # so it needs a run that gets as far as placing them.
        microscope.fm.objective.insert()

        fm_calls, beam_calls = [], []
        real_project_fm = microscope.project_fm_stable_move
        microscope.project_fm_stable_move = (  # type: ignore[method-assign]
            lambda **kw: (fm_calls.append(kw), real_project_fm(**kw))[1]
        )
        microscope.project_stable_move = (  # type: ignore[method-assign]
            lambda **kw: beam_calls.append(kw)
        )
        microscope.safe_absolute_stage_movement = lambda p: None  # type: ignore[method-assign]
        microscope.fm_stable_move = lambda **kw: beam_calls.append(kw)  # type: ignore[method-assign]

        with patch.object(acquisition, "acquire_image", return_value=Mock()):
            acquisition.acquire_tileset(
                microscope=microscope,
                channel_settings=ChannelSettings(
                    name="DAPI",
                    excitation_wavelength=358,
                    emission_wavelength=461,
                    power=0.1,
                    exposure_time=0.1,
                ),
                overview_parameters=OverviewParameters(
                    rows=2,
                    cols=2,
                    overlap=0.1,
                    use_zstack=False,
                    autofocus_mode=AutoFocusMode.NONE,
                ),
            )

        assert len(fm_calls) == 4, "one FM projection per tile"
        assert beam_calls == [], "nothing may position a tile via a beam projection"

    def test_fm_projection_differs_from_the_sem_projection_when_it_matters(
        self, microscope
    ):
        """Guard the reason for the switch: the two projections really do differ.

        At the FIB-flat orientation on an offset mount, stepping via the electron
        column foreshortens differently from the camera. If this ever became a
        no-op the switch would be pointless -- and a regression would be invisible.
        """
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=0, compustage=False
        )
        microscope.system.electron.column_tilt = 0.0
        microscope.system.ion.column_tilt = 52.0
        fm = FluorescenceMicroscope(parent=microscope)

        by_sem = microscope._view_corrected_stage_movement(25e-6, view_tilt=0.0)
        by_fm = microscope._view_corrected_stage_movement(
            25e-6, view_tilt=np.deg2rad(fm.camera_tilt)
        )

        assert by_fm.y != pytest.approx(by_sem.y, rel=1e-3)
        # the FM view here parallels the ion column, so it must match that projection
        by_ion = microscope._y_corrected_stage_movement(25e-6, beam_type=BeamType.ION)
        assert by_fm.y == pytest.approx(by_ion.y)
        assert by_fm.z == pytest.approx(by_ion.z)


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
        _configure(
            microscope, pretilt_deg=0, rotation_deg=0, tilt_deg=-180, compustage=True
        )
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
            (CameraImageTransform.FLIP_XY, lambda d: np.rot90(d, k=2)),
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
            type(fm),
            "mount_transform",
            property(lambda self: CameraImageTransform.FLIP_Y),
        )
        fm.set_image_transform(CameraImageTransform.FLIP_X)

        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        assert np.array_equal(
            fm._apply_image_transform(data), np.fliplr(np.flipud(data))
        )
