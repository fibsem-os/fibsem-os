"""Plan a sparse FM overview from regions selected on a beam overview.

A full fluorescence overview of a grid is expensive and mostly wasted -- the interesting
ground is a few regions, and the user can already see where they are on a FIB/SEM
overview. This turns rectangles drawn on that overview into the three values
`FMTiledAcquisitionRunner` needs to acquire only that ground: where to centre the grid,
how big it is, and which of its tiles to visit.

Pure arithmetic over two projections. No microscope, no Qt, no acquisition -- so the
part that is easy to get silently wrong can be tested without either.

**Regions define the grid here**, unlike the ordinary overview path where rows and
columns are settings and the grid centres on the stage. Painting a mask onto an
already-configured grid would save acquisition time and no memory: `stitch_tileset`
sizes the mosaic from the grid's shape whatever the mask says, so a small selection on a
full grid still allocates a full-grid mosaic. Deriving the grid is what makes a small
selection cheap in both.

The caller supplies regions in the beam overview's own displayed plane, in metres from a
base stage position -- what :meth:`StageFrame.offset` answers. Pixels are the display's
business and cancel immediately; taking them here would mean being told which pixels.

**Anything drawing this plan anchors its frame at the fluorescence orientation**, and
:attr:`SparseOverviewPlan.centre_position` is already there, so it can serve as the frame
origin directly. This is not a formality. A canvas frame takes its r and t from its
origin, and `FMOverviewWidget._posed()` reads them off the *live* stage -- which, when
this dialog opens, is at the milling pose, because that is where the beam overview was
just acquired. Anchoring on the live position draws every FM tile 1.086x too tall
(1.624x from the FIB pose): small enough to look like a grid, large enough to plan the
wrong one. `tests/fm/test_planning.py` pins both numbers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

from fibsem.imaging.tiling.geometry import PlaneRegion, plan_grid_over_regions
from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.projection import StageProjection

__all__ = [
    "SparseOverviewPlan",
    "plan_sparse_fm_overview",
    "project_regions_to_fm_plane",
    "to_fm_pose",
]


@dataclass(frozen=True)
class SparseOverviewPlan:
    """Everything an FM overview run needs to cover a selection.

    Attributes:
        centre_position: Stage position the grid straddles, at the FM orientation.
            `FMTiledAcquisitionRunner` takes this directly. It is also at the pose a
            canvas frame drawing the plan must be anchored at -- though a widget wants a
            *fixed* origin rather than one that moves with the selection, so
            `FMTilePreviewWidget` re-poses the stage position once instead. What matters
            is the pose; see the module docstring on why it must not be the live one.
        rows: Grid rows.
        cols: Grid columns.
        mask: Per-tile enable mask, `mask[row][col]`.
        regions: The selection, projected into the FM plane and measured from
            :attr:`centre_position`. For drawing, and for telling a tile that came from
            a region from one toggled by hand -- both of which only matter while the
            selection is being made. **Not persisted:** what an acquisition needs is the
            mask, and storing the regions beside it would mean a second representation
            of the same thing, able to disagree with it. Selecting again starts over.
    """

    centre_position: FibsemStagePosition
    rows: int
    cols: int
    mask: List[List[bool]]
    regions: Tuple[PlaneRegion, ...]

    @property
    def enabled_tiles(self) -> int:
        """How many tiles the run will actually visit."""
        return sum(sum(1 for on in row if on) for row in self.mask)


def to_fm_pose(
    position: FibsemStagePosition,
    fm_orientation: FibsemStagePosition,
    *,
    is_compustage: bool,
) -> FibsemStagePosition:
    """Re-pose a stage position for the fluorescence orientation.

    On a compustage this is a re-tilt and nothing else: x, y and z name the same ground
    under the camera as under the beam, so `FibsemMicroscope.get_target_position` keeps
    them and writes only the target orientation's r and t. Everything here rests on
    that, which is why the guard is a raise rather than a comment -- on a stage that
    reaches fluorescence by translating, every position would be wrong by the offset
    between the two instruments, and wrong in a way that still looks like a grid.

    Args:
        position: Stage position at any orientation.
        fm_orientation: The FM orientation's (r, t), from `get_orientation("FM")`.
        is_compustage: Whether the stage is a compustage.

    Returns:
        The same translation, posed for fluorescence.

    Raises:
        ValueError: on a non-compustage system.
    """
    if not is_compustage:
        raise ValueError("Cannot move to FM position on non-compustage systems.")
    posed = deepcopy(position)
    posed.r = fm_orientation.r
    posed.t = fm_orientation.t
    return posed


def project_regions_to_fm_plane(
    regions: Sequence[PlaneRegion],
    beam_base: FibsemStagePosition,
    beam_projection: "StageProjection",
    fm_projection: "StageProjection",
    fm_orientation: FibsemStagePosition,
    *,
    is_compustage: bool,
) -> List[PlaneRegion]:
    """Carry regions from the beam overview's plane into the FM's.

    Both projections are asked about the same ground from their own pose, so the shape
    changes on the way across: a FIB overview taken at the milling orientation is
    foreshortened by `1 / cos(stage_tilt - pre-tilt - view_tilt)` -- 3.9x at a -23
    degree milling angle -- while the FM at its own pose is not foreshortened at all.
    A rectangle in one plane is a differently-proportioned rectangle in the other.

    All four corners are carried, not two. Two would do while scan rotation is 0 or pi,
    which is all either projection models, but re-bounding the corners costs nothing and
    does not quietly assume it.

    The corners are handed to the FM projection still carrying the beam pose's r and t,
    and that is not an oversight: `project_stage_position` works from `position - base`
    and reads the orientation off the **base** alone, so only *fm_base* needs re-posing.
    Re-posing each corner as well is a no-op -- confirmed by removing it and watching
    nothing change -- and a no-op that reads as load-bearing is worse than none. What
    makes it sound is the compustage invariant: x, y and z name the same ground at
    either pose, which is exactly what :func:`to_fm_pose` refuses to assume elsewhere.

    Args:
        regions: Rectangles in the beam overview's plane, in metres from *beam_base*.
        beam_base: The stage position those metres are measured from.
        beam_projection: Projection for the beam overview, from its own metadata.
        fm_projection: Projection for the fluorescence camera.
        fm_orientation: The FM orientation's (r, t).
        is_compustage: Whether the stage is a compustage.

    Returns:
        The same regions in the FM plane, measured from the FM-posed *beam_base*.
    """
    fm_base = to_fm_pose(beam_base, fm_orientation, is_compustage=is_compustage)
    projected = []
    for region in regions:
        corners = [
            (region.x0, region.y0),
            (region.x1, region.y0),
            (region.x1, region.y1),
            (region.x0, region.y1),
        ]
        projected.append(
            PlaneRegion.from_points(
                fm_projection.to_plane(
                    beam_projection.from_plane(dx, dy, beam_base), fm_base
                )
                for dx, dy in corners
            )
        )
    return projected


def plan_sparse_fm_overview(
    regions: Sequence[PlaneRegion],
    beam_base: FibsemStagePosition,
    beam_projection: "StageProjection",
    fm_projection: "StageProjection",
    fm_orientation: FibsemStagePosition,
    fov_x: float,
    fov_y: float,
    overlap: float,
    *,
    is_compustage: bool,
) -> SparseOverviewPlan:
    """Turn a selection on a beam overview into an FM overview run.

    Args:
        regions: Rectangles in the beam overview's plane, in metres from *beam_base*.
        beam_base: The stage position those metres are measured from -- the origin the
            display drew them against, not necessarily the overview's own position.
        beam_projection: Projection for the beam overview, from its own metadata, so the
            selection resolves against the image as it was taken.
        fm_projection: Projection for the fluorescence camera, read live: the geometry
            is system configuration rather than pose, and an acquired FM tile carries
            none to read.
        fm_orientation: The FM orientation's (r, t).
        fov_x: FM tile width, in metres -- camera resolution times pixel size, not a
            horizontal field width with the other axis inferred.
        fov_y: FM tile height, in metres.
        overlap: Fractional overlap between adjacent tiles, in [0, 1).
        is_compustage: Whether the stage is a compustage.

    Returns:
        A :class:`SparseOverviewPlan`.

    Raises:
        ValueError: if no region has any area, *overlap* is out of range, or the stage
            is not a compustage.
    """
    fm_base = to_fm_pose(beam_base, fm_orientation, is_compustage=is_compustage)
    fm_regions = project_regions_to_fm_plane(
        regions,
        beam_base,
        beam_projection,
        fm_projection,
        fm_orientation,
        is_compustage=is_compustage,
    )
    plan = plan_grid_over_regions(fm_regions, fov_x, fov_y, overlap)
    centre_position = fm_projection.from_plane(plan.centre_dx, plan.centre_dy, fm_base)
    # Re-expressed against the grid centre rather than the base the caller happened to
    # draw against, so the regions and the tiles are in one frame: the display draws both
    # together, and the drawing origin is the caller's business, not the plan's.
    centred = tuple(
        PlaneRegion(
            r.x0 - plan.centre_dx,
            r.y0 - plan.centre_dy,
            r.x1 - plan.centre_dx,
            r.y1 - plan.centre_dy,
        )
        for r in fm_regions
    )
    return SparseOverviewPlan(
        centre_position=centre_position,
        rows=plan.rows,
        cols=plan.cols,
        mask=plan.mask,
        regions=centred,
    )
