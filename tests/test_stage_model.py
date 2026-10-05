"""The 3D stage model agrees with today's projection and vertical move (FIB-1101).

`fibsem.geometry.frames` is plain geometry: the stage tilting about x, the shuttle and
its pre-tilt turning with r on top of the translation axes, and views fixed in the
chamber. Nothing reads it yet. These tests hold it to what the readers do today, on
every stage type the pins cover, at every declared pose and a sweep of tilts between,
so a reader can move onto it without a pinned value changing.

Where they differ is written down here rather than fitted away:
- a compustage at its FIB pose, where today's projection reverses image y (both views)
  and plain geometry does not;
- vertical_move on a stage tilted past -90 degrees, which reverses the same way
  everywhere there (`Stage.turned_over`), not only at the FIB pose;
- a rotating stage between its SEM and FIB rotations, where today's projection takes
  the pre-tilt sign from the nearer of the two (or + if neither is within 5 degrees),
  and the model takes the shuttle's actual lean.
"""

import numpy as np
import pytest

from fibsem.geometry.frames import (
    StageModel,
    image_y_shift,
    in_plane_move,
    vertical_move,
)
from fibsem.geometry.movement import vertical_move_delta
from fibsem.transformations import (
    inverse_view_corrected_dy,
    view_corrected_stage_movement,
)
from tests.test_stage_poses_pinned import STAGE_TYPES, _microscope

TILTS_DEG = (-180, -160, -128, -100, -90, -60, -23, -10, 0, 10, 20, 38, 52)


@pytest.fixture(scope="module", params=list(STAGE_TYPES), ids=list(STAGE_TYPES))
def stage(request):
    """Every configuration the pins cover, as the stage type they pin."""
    microscope = _microscope(request.param)
    geometry = microscope.hardware_geometry()
    poses = geometry.declared_poses()
    return geometry, poses


def _views(geometry):
    return {"SEM": 0.0, "FIB": float(np.deg2rad(geometry.fib_column_tilt))}


def _poses_to_check(geometry, poses):
    """Every declared pose, and each declared rotation swept through the tilts."""
    rotations = sorted({round(p.r, 9) for p in poses.values()})
    checked = [(name, p.r, p.t) for name, p in poses.items()]
    checked += [
        (f"t={tilt}", r, float(np.deg2rad(tilt)))
        for r in rotations
        for tilt in TILTS_DEG
    ]
    return checked


def _at_compustage_fib(geometry, poses, r, t) -> bool:
    fib = poses["FIB"]
    return geometry.is_compustage and bool(np.isclose(t, fib.t, atol=0.1))


def test_the_model_is_built_from_an_images_geometry(stage):
    geometry, _ = stage
    model = StageModel.from_geometry(geometry)
    assert model.shuttle_pre_tilt == pytest.approx(
        np.deg2rad(geometry.shuttle_pre_tilt)
    )
    assert model.rotation_reference == pytest.approx(
        np.deg2rad(geometry.rotation_reference)
    )


def test_a_click_moves_the_stage_as_today(stage):
    """`view_corrected_stage_movement`, both views, every pose: identical, except
    the image-y reversal today's projection applies at a compustage's FIB pose."""
    geometry, poses = stage
    model = StageModel.from_geometry(geometry)
    for name, r, t in _poses_to_check(geometry, poses):
        for view, view_tilt in _views(geometry).items():
            today = view_corrected_stage_movement(20e-6, view_tilt, geometry, r, t)
            if abs(today[0]) > 1:  # the view sees the surface edge-on here
                continue
            flip = -1.0 if _at_compustage_fib(geometry, poses, r, t) else 1.0
            model_move = in_plane_move(model, flip * 20e-6, view_tilt, r, t)
            assert model_move == pytest.approx(today, rel=1e-9, abs=1e-15), (
                f"{name} in the {view} view"
            )


def test_a_stage_move_shows_in_the_image_as_today(stage):
    """`inverse_view_corrected_dy`, in-plane and off-plane moves alike, with the same
    compustage FIB-pose exception."""
    geometry, poses = stage
    model = StageModel.from_geometry(geometry)
    rng = np.random.default_rng(1101)
    for name, r, t in _poses_to_check(geometry, poses):
        for view, view_tilt in _views(geometry).items():
            flip = -1.0 if _at_compustage_fib(geometry, poses, r, t) else 1.0
            for dy, dz in rng.uniform(-50e-6, 50e-6, size=(4, 2)):
                today = inverse_view_corrected_dy(dy, dz, view_tilt, geometry, r, t)
                shown = flip * image_y_shift(model, dy, dz, view_tilt, t)
                assert shown == pytest.approx(today, rel=1e-9, abs=1e-18), (
                    f"{name} in the {view} view"
                )


def test_vertical_move_goes_straight_up_as_today(stage):
    """`vertical_move_delta`, reversed today where the sample is turned over."""
    geometry, poses = stage
    model = StageModel.from_geometry(geometry)
    column = np.deg2rad(geometry.fib_column_tilt)
    for name, r, t in _poses_to_check(geometry, poses):
        turned_over = t < np.deg2rad(-90)
        today = vertical_move_delta(
            dx=0.0,
            dy=1e-6,
            scan_rotation=0.0,
            fib_column_tilt=geometry.fib_column_tilt,
            stage_tilt=t,
            is_compustage=True,
        )
        flip = -1.0 if turned_over else 1.0
        assert vertical_move(model, flip * 1e-6, column, t) == pytest.approx(
            (today.y, today.z), rel=1e-9, abs=1e-18
        ), name


class TestPhysicsTheReadersCannotState:
    """What the model gives for free, convention-free."""

    def test_the_sem_is_blind_to_a_chamber_vertical_move(self, stage):
        geometry, _ = stage
        model = StageModel.from_geometry(geometry)
        for tilt in TILTS_DEG:
            t = np.deg2rad(tilt)
            up_y, up_z = model.chamber_vertical(t)
            assert image_y_shift(model, up_y, up_z, 0.0, t) == pytest.approx(
                0.0, abs=1e-15
            )

    def test_the_fib_sees_sin_column_of_a_chamber_vertical_move(self, stage):
        geometry, _ = stage
        model = StageModel.from_geometry(geometry)
        column = np.deg2rad(geometry.fib_column_tilt)
        for tilt in TILTS_DEG:
            t = np.deg2rad(tilt)
            up_y, up_z = model.chamber_vertical(t)
            assert image_y_shift(model, up_y, up_z, column, t) == pytest.approx(
                np.sin(column)
            )

    def test_the_pre_tilt_leans_the_other_way_half_a_turn_round(self, stage):
        geometry, _ = stage
        model = StageModel.from_geometry(geometry)
        ref = model.rotation_reference
        at_ref = model.surface_normal(ref, 0.0)
        half_turn = model.surface_normal(ref + np.pi, 0.0)
        assert half_turn[1] == pytest.approx(-at_ref[1])
        assert half_turn[2] == pytest.approx(at_ref[2])

    def test_between_the_poses_the_pre_tilt_is_the_shuttles_actual_lean(self):
        """Today's projection rounds to the nearer of SEM and FIB (or + beyond 5
        degrees of both); the model follows the shuttle. At a quarter turn the lean
        is all along x, so a y-z slide stays flat."""
        model = StageModel(shuttle_pre_tilt=np.deg2rad(35.0))
        direction_y, direction_z = model.in_plane_direction(np.pi / 2)
        assert direction_y > 0
        assert direction_z == pytest.approx(0.0, abs=1e-15)


def test_a_click_between_the_poses_follows_the_shuttles_lean():
    """A quarter turn from the reference, the pre-tilt leans along x, so a click in
    the SEM slides the stage in y alone. The pre-tilt sign buckets gave a slide
    tilted by the full pre-tilt here, as if at the reference rotation."""
    from fibsem.structures import FibsemHardwareGeometry

    geometry = FibsemHardwareGeometry(
        column_tilt=0, fib_column_tilt=52, shuttle_pre_tilt=35.0, rotation_reference=0
    )
    dy, dz = view_corrected_stage_movement(20e-6, 0.0, geometry, np.pi / 2, 0.0)
    assert dy == pytest.approx(20e-6)
    assert dz == pytest.approx(0.0, abs=1e-18)


def test_the_surface_slope_is_todays_corrected_pre_tilt(stage):
    """What coincidence's tilt walk and the sim scene read: the pre-tilt with the
    sign `_projection_terms` gave it, at every pose."""
    from fibsem.transformations import _projection_terms

    geometry, poses = stage
    model = StageModel.from_geometry(geometry)
    for name, r, t in _poses_to_check(geometry, poses):
        _, pretilt, _ = _projection_terms(geometry, r, t)
        assert model.surface_slope(r) == pytest.approx(pretilt, abs=1e-12), name


def test_turned_round_is_the_half_turn_from_the_reference():
    model = StageModel(rotation_reference=np.deg2rad(180.0))
    assert not model.turned_round(np.deg2rad(180.0))
    assert not model.turned_round(np.deg2rad(-95.0))
    assert not model.turned_round(np.deg2rad(90.0))
    assert model.turned_round(np.deg2rad(85.0))
    assert model.turned_round(0.0)
