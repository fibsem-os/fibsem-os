"""The FIB/SEM stage projection: agreement with both halves it replaces.

`BeamStageProjection` exists because the beam side currently runs *two* derivations of
one formula. Where a stage position lands on an overview comes from
`reproject_stage_positions_onto_image2`, off image metadata. Where a click sends the
stage comes from `microscope.project_stable_move`, off the live instrument. Nothing
holds them together, and a disagreement between them draws every marker consistently
slightly wrong -- which reads as a calibration problem, not a bug.

So the tests that matter here are not "does it compute something plausible" but three
agreements:

* forward, against `reproject_stage_positions_onto_image2` on a real image
* inverse, against `microscope.project_stable_move` on a live (simulated) instrument
* with itself, round-trip, which is the property neither of the above can give

Run directly:
    PYTHONPATH=$PWD python -m pytest tests/test_beam_stage_projection.py
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.imaging.tiling.reprojection import (
    reproject_stage_positions_onto_image2,
    y_corrected_stage_movement_tescan_from_geometry,
)
from fibsem.structures import (
    BeamType,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.projection import BeamStageProjection, surface_foreshortening

SHAPE = (1024, 1536)  # (height, width) -- deliberately non-square, so an axis swap shows
PIXEL_SIZE = 2e-7


@pytest.fixture(scope="module")
def aquilos():
    """A 35 degree pre-tilted shuttle with a 52 degree ion column.

    Module level rather than a fixture on the class that uses it. A class-scoped fixture
    written as an instance method is removed in pytest 10 -- it runs once per class while
    each test gets a fresh instance, so anything set on `self` would be invisible -- and
    the repo runs `filterwarnings = ["error"]`, so it fails CI on any pytest new enough to
    emit the deprecation while passing on an older one locally. The suggested `staticmethod`
    form trades that for a different hazard: staticmethod objects are not callable before
    Python 3.10, and CI builds on 3.8. Out here it is neither.
    """
    scope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "tfs-aquilos2-configuration.yaml"),
    )
    return scope


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


def _pose(microscope, *, compustage, pretilt_deg, rotation_deg, tilt_deg):
    """Pin the microscope into a known geometry without moving the stage."""
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


# A compustage cannot rotate, so only the reference rotation is a reachable pose for it
# -- the same restriction `tests/fm/test_reprojection.py` applies, and for the same
# reason (FIB-500 documents why the unreachable poses still matter).
POSES = [
    pytest.param(True, 0, 0, tilt, id=f"compustage-t{tilt}")
    for tilt in (0, -30, -128, -180)
] + [
    pytest.param(False, pretilt, rot, tilt, id=f"stage-p{pretilt}-r{rot}-t{tilt}")
    for pretilt in (0, 35)
    for rot in (0, 180)
    for tilt in (0, 25, -30)
]

BEAMS = [BeamType.ELECTRON, BeamType.ION]
SCAN_ROTATIONS = [0.0, np.pi]

# A Tescan-shaped instrument, stated rather than read off the shared fixture — see
# `test_tescan_survives_a_round_trip_too`.
_TESCAN_GEOMETRY = FibsemHardwareGeometry(
    column_tilt=0.0,
    fib_column_tilt=55.0,
    shuttle_pre_tilt=0.0,
    rotation_reference=0.0,
    rotation_180=180.0,
    is_compustage=False,
)
# Chosen to straddle 45 degrees of sample inclination, which is where the Tescan
# inverse switches from reading dy to reading dz — and so where a round trip stops
# being able to see the stage-y sign at all.
TESCAN_TILTS = [0, 15, 40, 50, 75]


def _geometry(microscope, compustage: bool) -> FibsemHardwareGeometry:
    return FibsemHardwareGeometry.from_system_settings(
        microscope.system, is_compustage=compustage
    )


class TestParityWithTheLiveMicroscope:
    """`from_plane` must reproduce `project_stable_move` exactly.

    That method is what actually drives the stage today, so anything this class lets
    through is a click that lands somewhere other than where the user pointed.
    """

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("beam_type", BEAMS)
    @pytest.mark.parametrize("scan_rotation", SCAN_ROTATIONS)
    def test_the_inverse_matches_project_stable_move(
        self, microscope, compustage, pretilt, rot, tilt, beam_type, scan_rotation,
        monkeypatch,
    ):
        base = _pose(
            microscope, compustage=compustage, pretilt_deg=pretilt,
            rotation_deg=rot, tilt_deg=tilt,
        )
        monkeypatch.setattr(
            microscope, "get_scan_rotation", lambda beam_type: scan_rotation
        )

        dx, dy_up = 31e-6, -17e-6  # dy_up is the microscope's image y, pointing up
        live = microscope.project_stable_move(
            dx=dx, dy=dy_up, beam_type=beam_type, base_position=base
        )

        projection = BeamStageProjection.from_microscope(microscope, beam_type)
        assert projection is not None
        # The canvas plane's y runs down, so it is handed the negation of the
        # microscope's image y -- the one conversion this class is asserting about.
        pure = projection.from_plane(dx, -dy_up, base)

        assert pure.x == pytest.approx(live.x, abs=1e-15)
        assert pure.y == pytest.approx(live.y, abs=1e-15)
        assert pure.z == pytest.approx(live.z, abs=1e-15)

    def test_the_scan_rotation_is_read_once_not_per_call(self, microscope, monkeypatch):
        """The reason this class exists at all, beyond agreement.

        `project_stable_move` calls `get_scan_rotation` on every invocation, and on a
        TFS system that is a set-then-read on the shared imaging channel -- fired from
        the GUI thread on a mouse event. Building the projection reads it once; using it
        reads nothing.
        """
        _pose(microscope, compustage=False, pretilt_deg=35, rotation_deg=0, tilt_deg=0)
        calls = []

        def counting(beam_type):
            calls.append(beam_type)
            return 0.0

        monkeypatch.setattr(microscope, "get_scan_rotation", counting)
        projection = BeamStageProjection.from_microscope(microscope, BeamType.ELECTRON)
        assert len(calls) == 1, "construction should read the scan rotation exactly once"

        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
        for _ in range(20):
            projection.from_plane(10e-6, 10e-6, base)
            projection.to_plane(base, base)
        assert len(calls) == 1, "using the projection read hardware"


class TestParityWithTheTiledReprojection:
    """`to_plane` must reproduce `reproject_stage_positions_onto_image2`.

    That function is what draws every marker on the Overview tab today, so this is the
    test that says the new canvas will put them in the same places as the old one.
    """

    @staticmethod
    def _image(microscope, base, beam_type, scan_rotation) -> FibsemImage:
        state = microscope.get_microscope_state(beam_type=beam_type)
        state.stage_position = base
        beam = state.electron_beam if beam_type is BeamType.ELECTRON else state.ion_beam
        beam.scan_rotation = scan_rotation
        image = FibsemImage.generate_blank_image(
            resolution=(SHAPE[1], SHAPE[0]), hfw=SHAPE[1] * PIXEL_SIZE
        )
        image.metadata.image_settings = ImageSettings(
            hfw=SHAPE[1] * PIXEL_SIZE, beam_type=beam_type
        )
        image.metadata.microscope_state = state
        image.metadata.system_info = microscope.system.info
        image.metadata.hardware_geometry = microscope.hardware_geometry()
        return image

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("beam_type", BEAMS)
    @pytest.mark.parametrize("scan_rotation", SCAN_ROTATIONS)
    def test_the_forward_matches_the_tiled_reprojection(
        self, microscope, compustage, pretilt, rot, tilt, beam_type, scan_rotation,
    ):
        base = _pose(
            microscope, compustage=compustage, pretilt_deg=pretilt,
            rotation_deg=rot, tilt_deg=tilt,
        )
        image = self._image(microscope, base, beam_type, scan_rotation)

        target = FibsemStagePosition(
            x=base.x + 120e-6, y=base.y - 45e-6, z=base.z + 8e-6, r=base.r, t=base.t,
        )
        (point,) = reproject_stage_positions_onto_image2(image, [target])
        # The tiled function answers in pixels from the image corner; the projection
        # answers in metres from the image centre. Same place, different units.
        expected = (
            (point.x - SHAPE[1] / 2) * PIXEL_SIZE,
            (point.y - SHAPE[0] / 2) * PIXEL_SIZE,
        )

        projection = BeamStageProjection.from_image(image)
        assert projection is not None
        got = projection.to_plane(target, base)

        assert got[0] == pytest.approx(expected[0], abs=1e-12)
        assert got[1] == pytest.approx(expected[1], abs=1e-12)

    def test_it_reads_the_image_not_the_instrument(self, microscope, monkeypatch):
        """A saved overview must project as it was taken.

        The scan rotation is the visible case: change it on the instrument after
        acquisition and the markers on a loaded image must not all invert. Reading it
        live -- which `project_stable_move` does -- gets this wrong, and the wrongness
        is a 180-degree flip, so it is not subtle once it happens.
        """
        base = _pose(
            microscope, compustage=False, pretilt_deg=35, rotation_deg=0, tilt_deg=0
        )
        image = self._image(microscope, base, BeamType.ELECTRON, scan_rotation=0.0)
        monkeypatch.setattr(microscope, "get_scan_rotation", lambda beam_type: np.pi)

        from_image = BeamStageProjection.from_image(image)
        from_live = BeamStageProjection.from_microscope(microscope, BeamType.ELECTRON)

        target = FibsemStagePosition(
            x=100e-6, y=0.0, z=0.0, r=base.r, t=base.t
        )
        assert from_image.to_plane(target, base)[0] == pytest.approx(100e-6)
        assert from_live.to_plane(target, base)[0] == pytest.approx(-100e-6), (
            "the live projection should have flipped -- otherwise this test proves nothing"
        )


class TestEveryOrientationPair:
    """An overview acquired at one orientation, markers recorded at another.

    This is the case the tab is actually used in: lamellae are recorded at the milling
    pose, and an overview may be acquired at SEM, FIB or MILLING. All nine pairs have to
    place markers where the tab being replaced places them.

    It is also where the first version of this class was wrong. On a stage that is not a
    compustage, reaching the ion beam means rotating 180 degrees, and
    `reproject_stage_positions_onto_image2` applies a compucentric correction for that
    which `calculate_reprojected_stage_position2` — the function this class was
    transcribed from — does not. Four of the nine pairs were out by **1.3 to 2.1 mm**.

    Invisible on a compustage, where every orientation shares one rotation. Hence both
    stage types here.
    """

    SHAPE = (1024, 1536)
    PIXEL_SIZE = 2e-7

    @staticmethod
    def _configured(filename):
        import fibsem.config as fibsem_config

        path = os.path.join(os.path.dirname(fibsem_config.__file__), "config", filename)
        scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
        return scope

    def _image(self, scope, base, beam_type):
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = base
        h, w = self.SHAPE
        image = FibsemImage.generate_blank_image(
            resolution=(w, h), hfw=w * self.PIXEL_SIZE
        )
        image.metadata.image_settings = ImageSettings(
            hfw=w * self.PIXEL_SIZE, beam_type=beam_type
        )
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
        return image

    @pytest.mark.parametrize(
        "config, compustage",
        [
            ("sim-arctis-configuration.yaml", True),
            ("tfs-aquilos2-configuration.yaml", False),
        ],
    )
    @pytest.mark.parametrize("overview_at", ["SEM", "FIB", "MILLING"])
    @pytest.mark.parametrize("marker_at", ["SEM", "FIB", "MILLING"])
    @pytest.mark.parametrize("beam_type", BEAMS)
    def test_it_agrees_with_the_tab_it_replaces(
        self, config, compustage, overview_at, marker_at, beam_type
    ):
        scope = self._configured(config)
        assert scope.stage_is_compustage is compustage, "config assumption drifted"

        overview_pose = scope.get_orientation(overview_at)
        marker_pose = scope.get_orientation(marker_at)
        base = FibsemStagePosition(
            x=0.0, y=0.0, z=0.0, r=overview_pose.r, t=overview_pose.t
        )
        scope.get_stage_position = lambda: base  # type: ignore[method-assign]
        image = self._image(scope, base, beam_type)

        target = FibsemStagePosition(
            x=150e-6, y=-60e-6, z=0.0, r=marker_pose.r, t=marker_pose.t
        )
        (point,) = reproject_stage_positions_onto_image2(image, [target])
        expected = (
            (point.x - self.SHAPE[1] / 2) * self.PIXEL_SIZE,
            (point.y - self.SHAPE[0] / 2) * self.PIXEL_SIZE,
        )

        projection = BeamStageProjection.from_image(image)
        assert projection is not None
        got = projection.to_plane(target, base)

        assert got[0] == pytest.approx(expected[0], abs=1e-12)
        assert got[1] == pytest.approx(expected[1], abs=1e-12)

    def test_the_correction_only_fires_across_a_half_turn(self):
        """It is keyed on rotation alone. A tilt-only difference — every pair on a
        compustage, and SEM/MILLING everywhere — must be left alone, or positions that
        need no correction get one."""
        scope = self._configured("tfs-aquilos2-configuration.yaml")
        sem = scope.get_orientation("SEM")
        milling = scope.get_orientation("MILLING")
        assert sem.r == pytest.approx(milling.r), "these differ in rotation here"

        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t)
        same_rotation = FibsemStagePosition(
            x=90e-6, y=0.0, z=0.0, r=milling.r, t=milling.t
        )
        corrected = BeamStageProjection._compucentric_corrected(same_rotation, base)
        assert corrected is same_rotation, "a tilt-only difference was 'corrected'"


class TestRoundTrip:
    """A click on a marker must resolve to the position that marker was drawn from.

    Neither parity test above can establish this: they each pin one direction against
    one existing implementation, and the two existing implementations have never been
    checked against *each other*. This is the property the class was created for.
    """

    @pytest.mark.parametrize("compustage, pretilt, rot, tilt", POSES)
    @pytest.mark.parametrize("beam_type", BEAMS)
    @pytest.mark.parametrize("scan_rotation", SCAN_ROTATIONS)
    def test_a_position_survives_a_round_trip(
        self, microscope, compustage, pretilt, rot, tilt, beam_type, scan_rotation,
    ):
        base = _pose(
            microscope, compustage=compustage, pretilt_deg=pretilt,
            rotation_deg=rot, tilt_deg=tilt,
        )
        projection = BeamStageProjection(
            geometry=_geometry(microscope, compustage),
            beam_type=beam_type,
            scan_rotation=scan_rotation,
        )

        for dx, dy in itertools.product((-250e-6, 0.0, 90e-6), (-40e-6, 0.0, 310e-6)):
            recovered = projection.from_plane(dx, dy, base)
            back = projection.to_plane(recovered, base)
            assert back[0] == pytest.approx(dx, abs=1e-12), f"x drifted at {(dx, dy)}"
            assert back[1] == pytest.approx(dy, abs=1e-12), f"y drifted at {(dx, dy)}"

    @pytest.mark.parametrize("beam_type", BEAMS)
    @pytest.mark.parametrize("tilt_deg", TESCAN_TILTS)
    def test_tescan_survives_a_round_trip_too(self, beam_type, tilt_deg):
        """Tescan gets its own case because it is a genuinely different derivation --
        z below the tilt axis, and stage x inverted against image x -- so it shares
        none of the arithmetic the cases above exercise.

        Built from an explicit geometry rather than the shared microscope fixture. That
        fixture is module-scoped and every pose test mutates its pre-tilt and
        orientations, so a Tescan case reading it was really testing whatever the
        previous parameter left behind -- which is how it ended up at a sample
        inclination past 45 degrees, where `test_the_chamber_frame_inversion_is_pinned`
        explains this round trip goes blind.
        """
        projection = BeamStageProjection(
            geometry=_TESCAN_GEOMETRY, beam_type=beam_type, scan_rotation=0.0,
            is_tescan=True,
        )
        base = FibsemStagePosition(
            x=1e-3, y=-2e-3, z=5e-3, r=0.0, t=np.deg2rad(tilt_deg)
        )

        for dx, dy in itertools.product((-250e-6, 90e-6), (-40e-6, 310e-6)):
            back = projection.to_plane(projection.from_plane(dx, dy, base), base)
            assert back[0] == pytest.approx(dx, abs=1e-12)
            assert back[1] == pytest.approx(dy, abs=1e-12)

    @pytest.mark.parametrize("tilt_deg", TESCAN_TILTS)
    def test_the_chamber_frame_inversion_is_pinned(self, tilt_deg):
        """`stable_move` inverts stage y against the chamber frame; assert it directly.

        A round trip cannot be trusted to catch this. The Tescan inverse recovers the
        sample-plane move from whichever of dy/dz is the larger component, so once the
        sample inclination passes 45 degrees it reads **dz alone** and never looks at
        dy -- at which point flipping the y sign changes nothing that comes back, and
        the round-trip test passes over a projection that would drive the stage the
        wrong way. Verified: this assertion fails under that flip at every tilt here,
        while the round trip above only fails below 45 degrees.
        """
        projection = BeamStageProjection(
            geometry=_TESCAN_GEOMETRY, beam_type=BeamType.ELECTRON, scan_rotation=0.0,
            is_tescan=True,
        )
        base = FibsemStagePosition(
            x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg)
        )
        # The canvas plane's y runs down, so this is +30 um in the microscope's image y.
        y_chamber, z_chamber = y_corrected_stage_movement_tescan_from_geometry(
            geometry=_TESCAN_GEOMETRY, stage_position=base, expected_y=30e-6,
            beam_type=BeamType.ELECTRON,
        )

        got = projection.from_plane(0.0, -30e-6, base)

        assert got.y == pytest.approx(-y_chamber, abs=1e-15), "stage y is not inverted"
        assert got.z == pytest.approx(z_chamber, abs=1e-15), "stage z should not invert"

    def test_tescan_inverts_stage_x_where_thermo_does_not(self):
        """The one Tescan difference visible without doing the trig, kept as its own
        assertion so a lost `is_tescan` branch fails loudly rather than shifting every
        marker to the opposite side of the image (the FIB-300 symptom)."""
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
        target = FibsemStagePosition(x=75e-6, y=0.0, z=0.0, r=0.0, t=0.0)
        kw = dict(
            geometry=_TESCAN_GEOMETRY, beam_type=BeamType.ELECTRON, scan_rotation=0.0
        )

        thermo = BeamStageProjection(**kw, is_tescan=False).to_plane(target, base)[0]
        tescan = BeamStageProjection(**kw, is_tescan=True).to_plane(target, base)[0]

        assert thermo == pytest.approx(75e-6)
        assert tescan == pytest.approx(-75e-6)


class TestRefusesRatherThanGuesses:
    """An image that cannot be projected has to say so.

    A projection built from partial metadata draws markers that look right and are not,
    and there is nothing on screen to tell the difference -- the same argument
    `fm/reprojection.py` makes for raising on a missing geometry.
    """

    def test_an_image_with_no_geometry_gives_no_projection(self, microscope):
        _pose(microscope, compustage=False, pretilt_deg=0, rotation_deg=0, tilt_deg=0)
        image = FibsemImage.generate_blank_image(resolution=(256, 256), hfw=1e-4)
        image.metadata.hardware_geometry = None
        assert BeamStageProjection.from_image(image) is None

    def test_an_image_with_no_beam_state_gives_no_projection(self, microscope):
        base = _pose(
            microscope, compustage=False, pretilt_deg=0, rotation_deg=0, tilt_deg=0
        )
        image = TestParityWithTheTiledReprojection._image(
            microscope, base, BeamType.ELECTRON, scan_rotation=0.0
        )
        image.metadata.microscope_state.electron_beam = None
        assert BeamStageProjection.from_image(image) is None


class TestSurfaceForeshortening:
    """A length *along the sample surface*, seen from a view, is `cos(theta)` of itself.

    The quantity an overlay lying on the sample needs -- a grid's boundary circle -- as
    opposed to one defined in the image, like the tile lattice, which needs nothing.

    Pinned as the six exact cosines a **35 degree shuttle** produces, against the
    shipped Aquilos2 configuration rather than a hand-set pre-tilt, so the numbers stay
    tied to a geometry that exists. The bug this replaced was invisible at a pre-tilt of
    **0** -- which is every Arctis, the default simulator config, and therefore every
    test written before this one. Stepping stage y with no z is a move *through* the
    tilted surface rather than along it, and inflated every span by `1 / cos(pre_tilt)`:
    exactly 1.000 at 0 degrees, 1.221 at 35. It reached an instrument before anything
    caught it (FIB-657).

    The two views where the beam looks straight down the surface normal are what make
    it obvious: a grid there must render as a circle, and did not.
    """

    # (orientation, beam, angle between the beam and the sample-surface normal)
    VIEWS = [
        pytest.param("SEM", BeamType.ELECTRON, 0, id="SEM-electron-normal"),
        pytest.param("SEM", BeamType.ION, 52, id="SEM-ion"),
        pytest.param("FIB", BeamType.ELECTRON, 52, id="FIB-electron"),
        pytest.param("FIB", BeamType.ION, 0, id="FIB-ion-normal"),
        pytest.param("MILLING", BeamType.ELECTRON, 23, id="MILLING-electron"),
        pytest.param("MILLING", BeamType.ION, 75, id="MILLING-ion"),
    ]

    @staticmethod
    def _projection(scope, beam_type):
        return BeamStageProjection(
            geometry=scope.hardware_geometry(),
            beam_type=beam_type,
            scan_rotation=0.0,
            is_tescan=False,
        )

    @pytest.mark.parametrize("orientation, beam_type, theta_deg", VIEWS)
    def test_it_is_the_cosine_of_the_beam_to_normal_angle(
        self, aquilos, orientation, beam_type, theta_deg
    ):
        pose = aquilos.get_orientation(orientation)
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)

        assert surface_foreshortening(
            self._projection(aquilos, beam_type), base
        ) == pytest.approx(np.cos(np.deg2rad(theta_deg)), abs=1e-3)

    @pytest.mark.parametrize(
        "orientation, beam_type",
        [("SEM", BeamType.ELECTRON), ("FIB", BeamType.ION)],
    )
    def test_the_normal_views_are_exactly_one(self, aquilos, orientation, beam_type):
        """The statement the bug broke: where the beam looks down the surface normal
        there is no foreshortening at all, so a disc on the sample is a circle. Both of
        these read 1.221 before -- `1 / cos(35)`."""
        pose = aquilos.get_orientation(orientation)
        base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)

        assert surface_foreshortening(
            self._projection(aquilos, beam_type), base
        ) == pytest.approx(1.0, abs=1e-6)

    def test_a_flat_shuttle_hides_the_whole_question(self, microscope):
        """Why this went unnoticed. At a pre-tilt of 0 the stage plane *is* the sample
        plane, so a stage-axis span and a surface span agree and the factor this
        corrects is 1. Every simulator config and every earlier test is this case."""
        base = _pose(
            microscope, compustage=False, pretilt_deg=0, rotation_deg=0, tilt_deg=0
        )
        projection = BeamStageProjection(
            geometry=_geometry(microscope, compustage=False),
            beam_type=BeamType.ELECTRON, scan_rotation=0.0, is_tescan=False,
        )
        assert surface_foreshortening(projection, base) == pytest.approx(1.0, abs=1e-6)
