"""Turning a selection on a beam overview into an FM overview run.

The arithmetic between "the user dragged a box on the FIB image" and "the runner visits
these tiles" crosses two projections and a change of pose, and every way it can be wrong
produces a grid that still looks like a grid. So the tests that matter here do not check
the plan against itself: they push it through the same layout and projection calls
`FMTiledAcquisitionRunner._setup` makes, and ask whether the tiles it would visit land on
the ground that was selected.
"""

import numpy as np
import pytest

from fibsem import utils
from fibsem.fm.planning import (
    plan_sparse_fm_overview,
    project_regions_to_fm_plane,
    to_fm_pose,
)
from fibsem.imaging.tiling.geometry import PlaneRegion, compute_tile_grid_from_fov
from fibsem.projection import (
    BeamStageProjection,
    FMStageProjection,
    surface_foreshortening,
)
from fibsem.structures import BeamType, FibsemStagePosition


@pytest.fixture
def fm_microscope():
    """A demo microscope posed for fluorescence, as the FM overview tab leaves it."""
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope.stage_is_compustage = True
    microscope._update_orientations()
    microscope.move_to_microscope("FM")
    return microscope


@pytest.fixture
def geometry(fm_microscope):
    """The three things a plan needs, read off the instrument once."""
    m = fm_microscope
    fm_projection = FMStageProjection.from_microscope(m)
    assert fm_projection is not None
    width, height = m.fm.camera.resolution
    pixel_x, pixel_y = m.fm.camera.pixel_size
    return dict(
        microscope=m,
        fm_projection=fm_projection,
        fm_orientation=m.get_orientation("FM"),
        fov_x=width * pixel_x,
        fov_y=height * pixel_y,
        resolution=(width, height),
    )


def beam_view(microscope, beam_type=BeamType.ION, orientation="MILLING"):
    """A projection and a base position for an overview taken at a beam pose."""
    projection = BeamStageProjection.from_microscope(microscope, beam_type)
    assert projection is not None
    pose = microscope.get_orientation(orientation)
    base = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    return projection, base


def region(x0, y0, x1, y1):
    return PlaneRegion.from_points([(x0, y0), (x1, y1)])


# ── the pose change ──────────────────────────────────────────────────────


def test_re_posing_keeps_the_translation_and_writes_the_orientation():
    """The whole plan rests on x, y and z naming the same ground at either pose."""
    fm = FibsemStagePosition(r=0.0, t=np.deg2rad(-180))
    posed = to_fm_pose(
        FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=0.0, t=np.deg2rad(-23)),
        fm,
        is_compustage=True,
    )
    assert (posed.x, posed.y, posed.z) == (1e-3, 2e-3, 3e-3)
    assert (posed.r, posed.t) == (fm.r, fm.t)


def test_re_posing_does_not_mutate_what_it_was_given():
    original = FibsemStagePosition(x=1e-3, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-23))
    to_fm_pose(original, FibsemStagePosition(r=0.0, t=np.deg2rad(-180)), is_compustage=True)
    assert original.t == np.deg2rad(-23)


def test_a_non_compustage_stage_is_refused(geometry):
    """There is no FM pose to re-express into; a translating stage would be off by the
    offset between the two instruments, and still produce a plausible grid."""
    projection, base = beam_view(geometry["microscope"])
    with pytest.raises(ValueError, match="non-compustage"):
        plan_sparse_fm_overview(
            [region(0.0, 0.0, 1e-4, 1e-4)],
            base,
            projection,
            geometry["fm_projection"],
            geometry["fm_orientation"],
            fov_x=geometry["fov_x"],
            fov_y=geometry["fov_y"],
            overlap=0.0,
            is_compustage=False,
        )


# ── across the projections ───────────────────────────────────────────────


def test_the_ion_view_at_the_milling_pose_is_foreshortened_against_the_camera(geometry):
    """The reason the selection cannot simply be copied across.

    A square dragged on a FIB overview taken at the milling orientation is not square
    ground, so it is not a square region for the camera either.
    """
    m = geometry["microscope"]
    projection, base = beam_view(m, BeamType.ION, "MILLING")
    square = region(-5e-5, -5e-5, 5e-5, 5e-5)

    [projected] = project_regions_to_fm_plane(
        [square], base, projection, geometry["fm_projection"],
        geometry["fm_orientation"], is_compustage=True,
    )

    # 1 / cos(stage_tilt - pre-tilt - view_tilt) at a -23 degree milling angle. Only the
    # tilt axis is squashed, which is why no single scale factor can carry a selection
    # across and the two panes cannot be the same picture.
    assert projected.width == pytest.approx(square.width, rel=1e-6)
    assert projected.height / square.height == pytest.approx(3.864, abs=1e-3)


def test_the_projected_shape_agrees_with_the_measured_foreshortening(geometry):
    """The factor is not restated here -- it is read off the same projection, through
    its own inverse, so the two cannot drift."""
    m = geometry["microscope"]
    projection, base = beam_view(m, BeamType.ION, "MILLING")
    square = region(-5e-5, -5e-5, 5e-5, 5e-5)

    [projected] = project_regions_to_fm_plane(
        [square], base, projection, geometry["fm_projection"],
        geometry["fm_orientation"], is_compustage=True,
    )

    assert projected.height / square.height == pytest.approx(
        1.0 / surface_foreshortening(projection, base), rel=1e-9
    )


def test_the_electron_view_at_its_own_pose_is_not_foreshortened(geometry):
    """The control for the test above: same code path, no distortion to find."""
    m = geometry["microscope"]
    projection, base = beam_view(m, BeamType.ELECTRON, "SEM")
    square = region(-5e-5, -5e-5, 5e-5, 5e-5)

    [projected] = project_regions_to_fm_plane(
        [square], base, projection, geometry["fm_projection"],
        geometry["fm_orientation"], is_compustage=True,
    )

    assert projected.width == pytest.approx(square.width, rel=1e-6)
    assert projected.height == pytest.approx(square.height, rel=1e-6)


# ── the plan ─────────────────────────────────────────────────────────────


def make_plan(geometry, regions, orientation="MILLING", beam_type=BeamType.ION,
              overlap=0.0):
    projection, base = beam_view(geometry["microscope"], beam_type, orientation)
    plan = plan_sparse_fm_overview(
        regions, base, projection, geometry["fm_projection"],
        geometry["fm_orientation"], fov_x=geometry["fov_x"],
        fov_y=geometry["fov_y"], overlap=overlap, is_compustage=True,
    )
    fm_regions = project_regions_to_fm_plane(
        regions, base, projection, geometry["fm_projection"],
        geometry["fm_orientation"], is_compustage=True,
    )
    return plan, fm_regions


def test_a_selection_inside_one_tile_is_one_tile(geometry):
    plan, _ = make_plan(geometry, [region(0.0, 0.0, geometry["fov_x"] / 4, 1e-6)])
    assert (plan.rows, plan.cols, plan.mask) == (1, 1, [[True]])
    assert plan.enabled_tiles == 1


def test_the_plan_is_posed_for_fluorescence(geometry):
    """The runner drives the stage to this; at the beam pose it would drive it into the
    pole piece rather than under the camera."""
    plan, _ = make_plan(geometry, [region(0.0, 0.0, 1e-4, 1e-4)])
    fm = geometry["fm_orientation"]
    assert plan.centre_position.r == pytest.approx(fm.r)
    assert plan.centre_position.t == pytest.approx(fm.t)


def test_the_centre_lands_where_the_selection_is(geometry):
    """`centre_position` is a stage position; it has to project back to the middle of
    the ground it was derived from."""
    plan, fm_regions = make_plan(geometry, [region(0.0, 0.0, 3e-4, 2e-4)])
    fm_base = to_fm_pose(
        beam_view(geometry["microscope"])[1],
        geometry["fm_orientation"],
        is_compustage=True,
    )
    bounds = PlaneRegion.from_points(
        [(r.x0, r.y0) for r in fm_regions] + [(r.x1, r.y1) for r in fm_regions]
    )
    back = geometry["fm_projection"].to_plane(plan.centre_position, fm_base)
    assert back[0] == pytest.approx((bounds.x0 + bounds.x1) / 2, abs=1e-9)
    assert back[1] == pytest.approx((bounds.y0 + bounds.y1) / 2, abs=1e-9)


def test_the_stored_regions_travel_with_the_plan(geometry):
    """Measured from the grid centre, not from whatever origin the display drew them
    against -- a stored selection has to mean the same thing when it is reopened."""
    plan, _ = make_plan(geometry, [region(0.0, 0.0, 3e-4, 2e-4)])
    bounds = PlaneRegion.from_points(
        [(r.x0, r.y0) for r in plan.regions] + [(r.x1, r.y1) for r in plan.regions]
    )
    assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(0.0, abs=1e-12)
    assert (bounds.y0 + bounds.y1) / 2 == pytest.approx(0.0, abs=1e-12)


def test_the_tile_axes_are_not_interchangeable(geometry):
    """The FM camera is square, so no plan built from it can tell fov_x from fov_y.

    The parameters admit a camera that is not -- `pixel_size` is a pair and binning is
    per axis -- so the axes are passed deliberately unequal here. Taken through the
    unforeshortened electron view, where the expected grid is arithmetic rather than a
    projection result.
    """
    unit = 1e-4
    projection, base = beam_view(geometry["microscope"], BeamType.ELECTRON, "SEM")
    plan = plan_sparse_fm_overview(
        [region(0.0, 0.0, 4 * unit, 4 * unit)],
        base,
        projection,
        geometry["fm_projection"],
        geometry["fm_orientation"],
        fov_x=4 * unit,
        fov_y=unit,
        overlap=0.0,
        is_compustage=True,
    )
    assert (plan.rows, plan.cols) == (4, 1)


def test_two_selections_share_one_grid_and_skip_the_gap(geometry):
    fov = geometry["fov_x"]
    plan, _ = make_plan(
        geometry,
        [region(0.0, 0.0, fov / 2, 1e-6), region(4 * fov, 0.0, 4.5 * fov, 1e-6)],
    )
    assert plan.cols >= 5
    assert plan.mask[0][0] and plan.mask[0][-1]
    assert not any(plan.mask[0][1:-1])


# ── the plan against the run it becomes ──────────────────────────────────
#
# Everything above checks the plan. This checks the acquisition: the tile positions are
# built the way `FMTiledAcquisitionRunner._setup` builds them -- the shared layout, the
# same centring offsets, `project_fm_stable_move` -- and then projected back into the
# plane the selection was made in. A sign error anywhere between the two shows up as
# tiles landing somewhere other than on the selected ground.


def visited_tiles(geometry, plan, overlap=0.0):
    """Where the run's tiles land in the FM plane, keyed by (row, col)."""
    m = geometry["microscope"]
    fov_x, fov_y = geometry["fov_x"], geometry["fov_y"]
    width, height = geometry["resolution"]
    step_x, step_y = fov_x * (1 - overlap), fov_y * (1 - overlap)

    grid = compute_tile_grid_from_fov(
        nrows=plan.rows, ncols=plan.cols, fov_x=fov_x, fov_y=fov_y,
        image_width=width, image_height=height, overlap=overlap, mask=plan.mask,
    )
    offset_x = (plan.cols - 1) * step_x / 2
    offset_y = (plan.rows - 1) * step_y / 2
    fm_base = to_fm_pose(
        beam_view(m)[1], geometry["fm_orientation"], is_compustage=True
    )

    out = {}
    for tile in grid:
        position = m.project_fm_stable_move(
            dx=tile.dx - offset_x,
            dy=tile.dy + offset_y,
            base_position=plan.centre_position,
        )
        cx, cy = geometry["fm_projection"].to_plane(position, fm_base)
        out[(tile.row, tile.col)] = (
            PlaneRegion(cx - fov_x / 2, cy - fov_y / 2, cx + fov_x / 2, cy + fov_y / 2),
            tile.enabled,
        )
    return out


@pytest.mark.parametrize("overlap", [0.0, 0.1])
def test_the_run_visits_exactly_the_tiles_that_touch_the_selection(geometry, overlap):
    fov = geometry["fov_x"]
    regions = [region(0.0, 0.0, 1.5 * fov, 2.5 * fov),
               region(4 * fov, 3 * fov, 5 * fov, 4 * fov)]
    plan, fm_regions = make_plan(geometry, regions, overlap=overlap)

    for rc, (footprint, enabled) in visited_tiles(geometry, plan, overlap).items():
        touches = any(footprint.overlaps(r) for r in fm_regions)
        assert enabled == touches, rc


def test_the_run_covers_every_selected_region(geometry):
    """Undersizing by one row drops ground the user asked for and still looks like a
    sparse acquisition that worked."""
    fov = geometry["fov_x"]
    regions = [region(0.0, 0.0, 1.5 * fov, 2.5 * fov)]
    plan, fm_regions = make_plan(geometry, regions)

    footprints = [f for f, _ in visited_tiles(geometry, plan).values()]
    covered = PlaneRegion.from_points(
        [(f.x0, f.y0) for f in footprints] + [(f.x1, f.y1) for f in footprints]
    )
    for r in fm_regions:
        assert covered.x0 <= r.x0 + 1e-12 and covered.x1 >= r.x1 - 1e-12
        assert covered.y0 <= r.y0 + 1e-12 and covered.y1 >= r.y1 - 1e-12


def test_the_selection_is_not_mirrored_on_its_way_to_the_run(geometry):
    """An asymmetric selection, so a flip in either axis cannot look like a match.

    The tiles the run visits must sit on the same side of the grid centre as the ground
    that was selected -- the failure a symmetric test case cannot see.
    """
    fov = geometry["fov_x"]
    # Up and to the left of the origin in the plane, and only there.
    regions = [region(-4 * fov, -4 * fov, -3 * fov, -1 * fov)]
    plan, fm_regions = make_plan(geometry, regions)

    enabled = [f for f, on in visited_tiles(geometry, plan).values() if on]
    assert enabled
    for footprint in enabled:
        assert any(footprint.overlaps(r) for r in fm_regions)
