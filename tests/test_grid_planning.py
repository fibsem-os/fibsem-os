"""Sizing a tile grid to selected ground, and masking it to what was selected.

The inverse of `compute_tile_grid_from_fov`: that answers where rows and columns land,
this answers what grid covers a region. They have to agree, so the last group here
checks the mask against tiles the *layout* produced rather than against the arithmetic
that produced the mask.

Two properties carry the feature:

* **the grid covers the selection** -- undersizing by one row silently drops ground the
  user asked for, and looks like a sparse acquisition that worked
* **enabled if and only if the tile touches a region** -- a tile enabled by mistake
  costs acquisition time; one disabled by mistake leaves a hole that stitches as zeros,
  indistinguishable from dark sample.
"""

import pytest

from fibsem.imaging.tiling.geometry import (
    PlaneRegion,
    compute_tile_grid_from_fov,
    plan_grid_over_regions,
    tiles_to_span,
)

FOV = 1.0


def region(x0, y0, x1, y1):
    return PlaneRegion.from_points([(x0, y0), (x1, y1)])


def plan(regions, fov_x=FOV, fov_y=FOV, overlap=0.0):
    return plan_grid_over_regions(regions, fov_x=fov_x, fov_y=fov_y, overlap=overlap)


# ── the region itself ────────────────────────────────────────────────────


def test_a_region_is_normalised_whichever_way_it_was_dragged():
    """A drag can end up and to the left of where it started."""
    assert region(1.0, 1.0, 0.0, 0.0) == region(0.0, 0.0, 1.0, 1.0)


def test_a_region_bounds_all_the_points_it_is_given():
    """Four projected corners, not two: the projection can reorder them."""
    r = PlaneRegion.from_points([(3.0, 1.0), (0.0, 4.0), (2.0, 0.0), (1.0, 2.0)])
    assert (r.x0, r.y0, r.x1, r.y1) == (0.0, 0.0, 3.0, 4.0)


def test_a_region_needs_at_least_one_point():
    with pytest.raises(ValueError, match="at least one point"):
        PlaneRegion.from_points([])


def test_touching_edges_do_not_overlap():
    """A tile whose edge grazes a region contributes none of it."""
    assert not region(0.0, 0.0, 1.0, 1.0).overlaps(region(1.0, 0.0, 2.0, 1.0))
    assert region(0.0, 0.0, 1.0, 1.0).overlaps(region(0.99, 0.0, 2.0, 1.0))


def test_separation_in_either_axis_alone_is_enough_to_miss():
    assert not region(0.0, 0.0, 1.0, 1.0).overlaps(region(0.0, 5.0, 1.0, 6.0))


# ── how many tiles ───────────────────────────────────────────────────────


def test_a_span_within_one_tile_needs_one_tile():
    assert tiles_to_span(0.5, FOV, 0.5) == 1
    assert tiles_to_span(FOV, FOV, 0.5) == 1


def test_an_exact_fit_does_not_add_a_row():
    """Three tiles at 0.5 apart cover 2.0 exactly; a fourth would be wasted."""
    assert tiles_to_span(FOV + 2 * 0.5, FOV, 0.5) == 3


@pytest.mark.parametrize("fov", [1.024e-4, 4.512e-4, 9.0e-5])
@pytest.mark.parametrize("overlap", [0.0, 0.05, 0.1, 0.25])
def test_an_exact_fit_at_camera_scale_does_not_add_a_row(fov, overlap):
    """The same property where it actually bites.

    `(extent - fov) / step` is exact in decimal and a hair over an integer in binary, so
    `ceil` alone adds a whole unwanted row and column. Round numbers like the case above
    never show it -- these are real FM fields of view, where it trips for 126 of the
    grid sizes swept here.
    """
    step = fov * (1 - overlap)
    for n in range(1, 40):
        assert tiles_to_span((n - 1) * step + fov, fov, step) == n


def test_a_span_a_hair_over_one_tile_needs_two():
    assert tiles_to_span(FOV * 1.0000001, FOV, 0.5) == 2


def test_the_count_actually_spans_what_it_was_asked_to():
    """The property the arithmetic exists for, over a range of awkward extents."""
    step = 0.5
    for hundredths in range(1, 400):
        extent = hundredths / 100.0
        n = tiles_to_span(extent, FOV, step)
        assert (n - 1) * step + FOV >= extent - 1e-9, extent
        assert (n - 2) * step + FOV < extent or n == 1, extent


def test_a_non_positive_step_is_refused():
    with pytest.raises(ValueError, match="step must be positive"):
        tiles_to_span(5.0, FOV, 0.0)


def test_a_non_positive_field_of_view_is_refused():
    with pytest.raises(ValueError, match="field of view must be positive"):
        tiles_to_span(5.0, 0.0, 0.5)


# ── the plan ─────────────────────────────────────────────────────────────


def test_a_region_inside_one_tile_gives_a_single_enabled_tile():
    p = plan([region(0.0, 0.0, 0.5, 0.5)])
    assert (p.rows, p.cols, p.mask) == (1, 1, [[True]])


def test_the_grid_centres_on_the_selection_not_the_origin():
    p = plan([region(10.0, 20.0, 11.0, 22.0)])
    assert (p.centre_dx, p.centre_dy) == (10.5, 21.0)


def test_two_regions_share_one_grid_spanning_both():
    """One selection is one acquisition; the mask is what skips the gap."""
    p = plan([region(0.0, 0.0, 1.0, 1.0), region(4.0, 0.0, 5.0, 1.0)])
    assert (p.rows, p.cols) == (1, 5)
    assert p.mask == [[True, False, False, False, True]]


def test_the_ground_between_two_regions_is_not_visited():
    p = plan([region(0.0, 0.0, 1.0, 1.0), region(0.0, 4.0, 1.0, 5.0)])
    assert [row[0] for row in p.mask] == [True, False, False, False, True]


def test_overlap_packs_more_tiles_into_the_same_ground():
    dense = plan([region(0.0, 0.0, 4.0, 1.0)], overlap=0.5)
    sparse = plan([region(0.0, 0.0, 4.0, 1.0)], overlap=0.0)
    assert dense.cols > sparse.cols


def test_a_non_square_tile_is_counted_per_axis():
    p = plan([region(0.0, 0.0, 4.0, 4.0)], fov_x=4.0, fov_y=1.0)
    assert (p.rows, p.cols) == (4, 1)


def test_a_region_with_no_area_selects_nothing():
    """A click that never became a drag."""
    with pytest.raises(ValueError, match="No region with any area"):
        plan([region(1.0, 1.0, 1.0, 1.0)])


def test_a_zero_area_region_does_not_stretch_a_real_one():
    """Ignored outright, not merged into the bounding box."""
    p = plan([region(0.0, 0.0, 1.0, 1.0), region(9.0, 9.0, 9.0, 9.0)])
    assert (p.rows, p.cols) == (1, 1)


def test_no_regions_at_all_is_refused():
    with pytest.raises(ValueError, match="No region with any area"):
        plan([])


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_an_impossible_overlap_is_refused(overlap):
    with pytest.raises(ValueError, match="Overlap must be"):
        plan([region(0.0, 0.0, 1.0, 1.0)], overlap=overlap)


# ── the plan against the layout it will be run through ───────────────────
#
# `plan_grid_over_regions` derives rows, cols and the mask; `compute_tile_grid_from_fov`
# decides where those tiles actually land. Everything above tests the first against
# itself. These test it against the second, which is what the acquisition uses.


def tile_footprints(p, fov_x=FOV, fov_y=FOV, overlap=0.0):
    """Where the planned grid's tiles land, in the plane the regions were given in.

    Derived from the layout's own `dx`/`dy` rather than restated, so a disagreement
    between the two shows up here. The layout measures y upward and centres the grid on
    the run's start position -- the same two adjustments `FMTiledAcquisitionRunner`
    makes before projecting each tile.
    """
    step_x, step_y = fov_x * (1 - overlap), fov_y * (1 - overlap)
    grid = compute_tile_grid_from_fov(
        nrows=p.rows, ncols=p.cols, fov_x=fov_x, fov_y=fov_y,
        image_width=100, image_height=100, overlap=overlap, mask=p.mask,
    )
    out = {}
    for tile in grid:
        cx = p.centre_dx + tile.dx - (p.cols - 1) * step_x / 2
        cy = p.centre_dy - (tile.dy + (p.rows - 1) * step_y / 2)
        out[(tile.row, tile.col)] = (
            PlaneRegion(cx - fov_x / 2, cy - fov_y / 2, cx + fov_x / 2, cy + fov_y / 2),
            tile.enabled,
        )
    return out


@pytest.mark.parametrize("overlap", [0.0, 0.1, 0.5])
def test_every_enabled_tile_lands_on_a_region(overlap):
    regions = [region(0.0, 0.0, 1.5, 2.5), region(4.0, 3.0, 5.0, 4.0)]
    p = plan(regions, overlap=overlap)
    for rc, (footprint, enabled) in tile_footprints(p, overlap=overlap).items():
        touches = any(footprint.overlaps(r) for r in regions)
        assert enabled == touches, rc


@pytest.mark.parametrize("overlap", [0.0, 0.1, 0.5])
def test_the_planned_grid_covers_every_region(overlap):
    """The grid the layout produces reaches every corner of the selection."""
    regions = [region(0.0, 0.0, 1.5, 2.5), region(4.0, 3.0, 5.0, 4.0)]
    p = plan(regions, overlap=overlap)
    footprints = [f for f, _ in tile_footprints(p, overlap=overlap).values()]
    covered = PlaneRegion.from_points(
        [(f.x0, f.y0) for f in footprints] + [(f.x1, f.y1) for f in footprints]
    )
    for r in regions:
        assert covered.x0 <= r.x0 + 1e-9 and covered.x1 >= r.x1 - 1e-9
        assert covered.y0 <= r.y0 + 1e-9 and covered.y1 >= r.y1 - 1e-9


def test_the_grid_straddles_its_own_centre():
    """Row 0 is the top of the plane, not the bottom -- the sign that decides whether a
    selection's mask lands on the ground it was drawn over or mirrored across it."""
    p = plan([region(0.0, 0.0, 1.0, 3.0)])
    footprints = tile_footprints(p)
    assert p.rows == 3
    ys = [footprints[(row, 0)][0].y0 for row in range(p.rows)]
    assert ys == sorted(ys), "row index must increase with plane y"


def test_a_mask_that_would_acquire_nothing_cannot_be_planned():
    """`FMTiledAcquisitionRunner` refuses an all-disabled mask; planning cannot make one."""
    p = plan([region(0.0, 0.0, 1.0, 1.0), region(4.0, 4.0, 5.0, 5.0)])
    assert any(any(row) for row in p.mask)
