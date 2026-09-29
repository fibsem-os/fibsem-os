"""The inverse projection answers height, not just slide (FIB-766).

`view_corrected_stage_movement` (the forward) only ever *produces* in-plane
movements -- a click slides the sample along its own surface. Its inverse is fed
arbitrary position deltas: saved positions that differ in height, and coincidence
corrections, which are chamber-vertical. The old inverse read stage `dy` and
discarded `dz`, so it was exact on the forward's line and wrong off it -- and the
existing parity tests, which round-trip through the forward, were structurally
blind to that (they only ever generate the inputs where every candidate agrees).

These tests hold the inverse to an **independently calibrated 3D model** instead.
The model is plain geometry -- chamber frame, orthographic views, stage axes
rotated by the tilt -- with every sign convention left free, then *fitted* by
requiring it to reproduce the existing forward map exactly over the full pose
matrix, both configured geometries, both beams. The forward is the anchor because
it drives real stage moves from clicks every day; a wrong convention there would
walk the stage the wrong way and could not survive. Calibration is part of the
test: if the forward's conventions ever change, `test_the_oracle_calibrates`
fails first and says why everything else is failing.

Run directly:
    python -m pytest tests/test_projection_height.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fibsem.structures import FibsemHardwareGeometry
from fibsem.transformations import (
    inverse_view_corrected_dy,
    view_corrected_stage_movement,
)

# The two geometries the app actually runs: the compustage (Arctis) and the
# pre-tilted autoloader shuttle. The autoloader matters even though the reported
# defect was found on the Arctis: its 35 deg pretilt exercises the in-plane /
# normal resolution that the Arctis (pretilt 0) never reaches.
ARCTIS = FibsemHardwareGeometry(
    column_tilt=0,
    fib_column_tilt=52,
    shuttle_pre_tilt=0.0,
    rotation_reference=0,
    rotation_180=0,
    is_compustage=True,
    camera_tilt=0.0,
)
AUTOLOADER = FibsemHardwareGeometry(
    column_tilt=0,
    fib_column_tilt=52,
    shuttle_pre_tilt=35.0,
    rotation_reference=0,
    rotation_180=180,
    is_compustage=False,
    camera_tilt=0.0,
)

VIEW_TILTS = {"SEM": 0.0, "FIB": np.deg2rad(52.0)}

# Sweep the configured orientations plus fillers, never just round numbers: both
# geometry defects fixed in this module's history were exact at 0 and 180 and
# wrong at -128 and -23, which nobody picks by hand.
TILTS_DEG = (
    -180.0,
    -160.0,
    -128.0,
    -90.0,
    -60.0,
    -23.0,
    -10.0,
    0.0,
    10.0,
    20.0,
    38.0,
    52.0,
)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])  # rotates (y, z) by theta


def _pose_matrix():
    for geometry in (ARCTIS, AUTOLOADER):
        rotations = (0.0,) if geometry.is_compustage else (0.0, np.pi)
        for rotation in rotations:
            for tilt_deg in TILTS_DEG:
                yield geometry, rotation, np.deg2rad(tilt_deg)


class _Oracle:
    """Orthographic projection of a stage delta into a view, in the chamber y-z plane.

    ``d = dy * y_stage + dz * z_stage`` in chamber coordinates; the image shift is
    ``flip * (y_image . d)``. Six discrete conventions are left free and fitted:

    * ``s_t``    -- which way the stage tilt rotates
    * ``s_v``    -- which side of vertical the tilted view leans
    * ``s_img``  -- the image-y handedness
    * ``k_fold`` -- whether the compustage half-turn adds pi to the physical tilt
    * ``m_r``    -- whether a 180 deg stage rotation flips the raw y-axis
    * ``m_cs``   -- whether the compustage sign flips the image frame
    """

    def __init__(self, s_t, s_v, s_img, k_fold, m_r, m_cs):
        self.s_t, self.s_v, self.s_img = s_t, s_v, s_img
        self.k_fold, self.m_r, self.m_cs = k_fold, m_r, m_cs

    def _stage_axes(self, geometry, rotation, tilt):
        tau = self.s_t * tilt + (np.pi * self.k_fold if geometry.is_compustage else 0.0)
        y_stage = _rot(tau) @ np.array([1.0, 0.0])
        z_stage = _rot(tau) @ np.array([0.0, 1.0])
        near_180 = abs((np.rad2deg(rotation) % 360.0) - 180.0) < 5.0
        if self.m_r and near_180:
            y_stage = -y_stage
        return y_stage, z_stage

    def _flip(self, geometry, rotation, tilt) -> float:
        if not (geometry.is_compustage and self.m_cs):
            return 1.0
        from fibsem.transformations import _projection_terms

        sign, _, _ = _projection_terms(geometry, rotation, tilt)
        return sign

    def project(self, geometry, rotation, tilt, view_tilt, dy, dz) -> float:
        y_stage, z_stage = self._stage_axes(geometry, rotation, tilt)
        d = dy * y_stage + dz * z_stage
        y_image = self.s_img * (_rot(self.s_v * view_tilt) @ np.array([1.0, 0.0]))
        return float(y_image @ d) * self._flip(geometry, rotation, tilt)

    def vertical_as_stage_delta(self, geometry, rotation, tilt, height):
        """A chamber-vertical displacement, expressed as a stage (dy, dz)."""
        y_stage, z_stage = self._stage_axes(geometry, rotation, tilt)
        up = np.array([0.0, 1.0])
        return float(height * (up @ y_stage)), float(height * (up @ z_stage))


def _calibrate():
    survivors = []
    for combo in itertools.product([1, -1], [1, -1], [1, -1], [0, 1], [0, 1], [0, 1]):
        oracle = _Oracle(*combo)
        if all(
            abs(
                oracle.project(
                    g,
                    r,
                    t,
                    view_tilt,
                    *view_corrected_stage_movement(1.0, view_tilt, g, r, t),
                )
                - 1.0
            )
            < 1e-9
            for g, r, t in _pose_matrix()
            for view_tilt in VIEW_TILTS.values()
            if abs(view_corrected_stage_movement(1.0, view_tilt, g, r, t)[0]) < 50
        ):
            survivors.append(oracle)
    return survivors


@pytest.fixture(scope="module")
def oracle():
    survivors = _calibrate()
    assert len(survivors) == 1, (
        f"{len(survivors)} sign conventions reproduce the forward map; the "
        "calibration no longer pins the geometry and nothing below means anything"
    )
    return survivors[0]


def test_the_oracle_calibrates(oracle):
    """One and only one convention tuple reproduces the forward map.

    This is the load-bearing assertion: the fitted model is only an authority on
    off-plane deltas because the forward map -- which drives real stage moves from
    clicks -- pins every one of its sign choices.
    """
    assert (
        oracle.s_t,
        oracle.s_v,
        oracle.s_img,
        oracle.k_fold,
        oracle.m_r,
        oracle.m_cs,
    ) == (1, 1, 1, 1, 0, 1)


def _inverse(geometry, rotation, tilt, view_tilt, dy, dz) -> float:
    return inverse_view_corrected_dy(
        dy=dy,
        dz=dz,
        view_tilt=view_tilt,
        geometry=geometry,
        stage_rotation=rotation,
        stage_tilt=tilt,
    )


class TestAgainstTheOracle:
    def test_every_delta_everywhere(self, oracle):
        """The full contract: the code and the model are the same linear map --
        on-plane, off-plane, both beams, both geometries, every pose."""
        rng = np.random.default_rng(766)
        for geometry, rotation, tilt in _pose_matrix():
            for view_tilt in VIEW_TILTS.values():
                for _ in range(8):
                    dy, dz = rng.uniform(-50e-6, 50e-6, size=2)
                    want = oracle.project(geometry, rotation, tilt, view_tilt, dy, dz)
                    got = _inverse(geometry, rotation, tilt, view_tilt, dy, dz)
                    assert got == pytest.approx(want, abs=1e-15), (
                        f"compustage={geometry.is_compustage} r={np.rad2deg(rotation):.0f} "
                        f"t={np.rad2deg(tilt):.1f} view={np.rad2deg(view_tilt):.0f} "
                        f"delta=({dy:.2e}, {dz:.2e})"
                    )

    def test_in_plane_deltas_round_trip_the_forward(self):
        """The domain statement: on the forward's own outputs, the new inverse
        answers exactly as inverting the forward -- so every marker whose position
        delta came from a click or a stable move draws precisely where it did
        before this fix. The visible change is confined to off-plane deltas."""
        for geometry, rotation, tilt in _pose_matrix():
            for view_tilt in VIEW_TILTS.values():
                for expected_y in (25e-6, -60e-6):
                    dy, dz = view_corrected_stage_movement(
                        expected_y, view_tilt, geometry, rotation, tilt
                    )
                    if abs(dy) > 50:  # near-singular pose for this view
                        continue
                    assert _inverse(
                        geometry, rotation, tilt, view_tilt, dy, dz
                    ) == pytest.approx(expected_y, rel=1e-12)


class TestTheTwoPhysicalInvariants:
    """The convention-free facts the old code got wrong, stated as physics."""

    @pytest.mark.parametrize("height", [20e-6, -35e-6])
    def test_the_sem_is_blind_to_a_chamber_vertical_move(self, oracle, height):
        """The electron column looks straight down, so a vertical move shifts
        nothing in its image. The delta's u and n image shifts cancel exactly --
        the old code kept only the u half and showed a phantom shift of up to
        9.7 um per 20 um of coincidence correction."""
        for geometry, rotation, tilt in _pose_matrix():
            dy, dz = oracle.vertical_as_stage_delta(geometry, rotation, tilt, height)
            assert _inverse(geometry, rotation, tilt, VIEW_TILTS["SEM"], dy, dz) == (
                pytest.approx(0.0, abs=1e-18)
            ), f"phantom SEM shift at t={np.rad2deg(tilt):.1f}"

    @pytest.mark.parametrize("height", [20e-6, -35e-6])
    def test_the_fib_sees_sin52_of_a_chamber_vertical_move(self, oracle, height):
        """A view tilted 52 deg from vertical sees sin(52 deg) of a vertical
        displacement -- at every pose, because the view axis is fixed in the
        chamber and does not care where the stage is tilted to. The old code
        showed nothing for a bare height change (dz discarded) and a wrong,
        pose-dependent value for a coincidence move."""
        expected = abs(height) * np.sin(np.deg2rad(52.0))
        for geometry, rotation, tilt in _pose_matrix():
            dy, dz = oracle.vertical_as_stage_delta(geometry, rotation, tilt, height)
            shift = _inverse(geometry, rotation, tilt, VIEW_TILTS["FIB"], dy, dz)
            assert abs(shift) == pytest.approx(expected, rel=1e-12), (
                f"t={np.rad2deg(tilt):.1f}"
            )


class TestGoldenValuesAtTheConfiguredOrientations:
    """Signed values at the poses the app actually uses, pinned so a future sign
    slip cannot pass the magnitude checks by flipping direction. From the
    calibrated oracle, on the Arctis, for a +20 um chamber-vertical move."""

    @pytest.mark.parametrize(
        "tilt_deg, fib_shift_um",
        [
            (-128.0, +15.760254),  # FIB orientation
            (-23.0, -15.760254),  # MILLING orientation
            (0.0, -15.760254),  # SEM orientation
            (-180.0, -15.760254),  # FM orientation
        ],
    )
    def test_fib_view_response(self, oracle, tilt_deg, fib_shift_um):
        tilt = np.deg2rad(tilt_deg)
        dy, dz = oracle.vertical_as_stage_delta(ARCTIS, 0.0, tilt, 20e-6)
        shift = _inverse(ARCTIS, 0.0, tilt, VIEW_TILTS["FIB"], dy, dz)
        assert shift * 1e6 == pytest.approx(fib_shift_um, abs=1e-4)
