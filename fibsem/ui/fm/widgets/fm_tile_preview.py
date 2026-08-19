"""The FM tiles a selection of ground produces, drawn where they will be acquired.

The right-hand pane of the sparse selection dialog. Regions are dragged on a beam
overview in the other pane; this one answers what the fluorescence microscope would
actually visit, at the fluorescence pose, in stage coordinates.

Two panes rather than one picture, because the same ground is two different shapes: a
FIB overview taken at the milling orientation is foreshortened against the camera, and
at selection time there is no FM image to layer the beam one onto anyway.

Deliberately inert. It draws, it reports, and it never touches the microscope beyond
reading geometry — see :meth:`FMTilePreviewWidget.__init__` on why the stage cannot be
driven from here.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.fm.planning import SparseOverviewPlan, plan_sparse_fm_overview, to_fm_pose
from fibsem.imaging.tiling.geometry import (
    PlaneRegion,
    compute_tile_grid_from_fov,
    order_tiles,
    validate_tile_stage_positions,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.projection import FMStageProjection, StageProjection
from fibsem.structures import FibsemStagePosition, TileOrderStrategy
from fibsem.ui.tokens import GRID_BOUNDARY_COLOUR, STAGE_LIMITS_COLOUR
from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    GRID_BOUNDARY_RADIUS_M,
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.canvas.stage_frame import StageFrame

# `ENABLED_ALPHA` is 0 on the canvas: enabled tiles are outlines, which reads over an
# image and does not read over nothing. This pane is nothing but the grid.
_SELECTED_FILL_ALPHA = 0.30

# The centre of the sample holder, and so of the travel limits, the grid boundary and
# the framing. Distinct from the frame origin, which is wherever the stage is standing.
STAGE_ORIGIN = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)


class FMTilePreviewWidget(QWidget):
    """The FM grid a set of regions produces, on a real-space canvas.

    Signals:
        selection_changed: the plan or the mask changed. One signal for both, because a
            host showing a count cannot usefully tell them apart.
    """

    selection_changed = pyqtSignal()

    def __init__(
        self, microscope: FibsemMicroscope, parent: Optional[QWidget] = None
    ) -> None:
        """Build the pane against *microscope*'s fluorescence geometry.

        Raises rather than degrading when the geometry cannot be read. A pane that draws
        a grid it could not place would be indistinguishable from one that placed it
        correctly, which is the failure this whole feature exists to avoid; the button
        that opens the dialog is where an unavailable FM should be caught.

        **Nothing here can move the stage.** The canvas only *emits*
        ``canvas_clicked`` / ``canvas_double_clicked``; every stage move in the FM
        overview tab lives in that tab's own handlers. Connecting neither is what keeps
        this inert, so do not copy the tab's wiring block in.
        """
        super().__init__(parent)
        self.microscope = microscope

        projection = FMStageProjection.from_microscope(microscope)
        if projection is None:
            raise ValueError(
                "Cannot read the fluorescence geometry; the tile preview needs it to "
                "place a grid."
            )
        self._projection: StageProjection = projection
        self._fm_orientation = microscope.get_orientation("FM")

        width, height = microscope.fm.camera.resolution
        pixel_x, pixel_y = microscope.fm.camera.pixel_size
        self._resolution: Tuple[int, int] = (width, height)
        self._pixel_size = pixel_x
        self._fov: Tuple[float, float] = (width * pixel_x, height * pixel_y)

        # Fixed for the life of the pane, and posed for fluorescence whatever the stage
        # is doing. A canvas frame takes its rotation and tilt from its origin, and when
        # this dialog opens the stage sits at the milling orientation -- that is where
        # the beam overview was just acquired. Anchoring on the live position draws every
        # tile 1.086x too tall: small enough to look like a grid, large enough to plan
        # the wrong one. `tests/fm/test_planning.py` pins that number.
        #
        # Fixed rather than re-read, and rather than taken from the plan's own centre --
        # both are at the fluorescence orientation, but a frame whose origin moves takes
        # the whole scene with it, and this pane's whole job is to hold still while the
        # selection changes underneath.
        self._origin = to_fm_pose(
            microscope.get_stage_position(),
            self._fm_orientation,
            is_compustage=microscope.stage_is_compustage,
        )

        self._plan: Optional[SparseOverviewPlan] = None
        self._mask: List[List[bool]] = []
        self._overlap = 0.0
        self._unreachable: List[Tuple[int, int]] = []

        self.canvas = FMRealSpaceCanvasWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.tile_grid_overlay = TileGridOverlay()
        self.tile_grid_overlay.set_fill_alpha(_SELECTED_FILL_ALPHA)
        # The regions own the geometry here. A grid dragged in this pane would spring
        # back the moment the plan was recomputed, which reads as a broken control.
        self.tile_grid_overlay.set_grid_editable(False)
        self.canvas.canvas.add_overlay(self.tile_grid_overlay)
        self.tile_grid_overlay.tile_toggled.connect(self._on_tile_toggled)

        self.stage_overlay = MinimapShapesOverlay(zorder=4.0, crosshair_half_px=24)
        self.canvas.canvas.add_overlay(self.stage_overlay)

        self.canvas.canvas.set_reference_pixel_size(self._pixel_size)
        # Declared once, to what the stage can reach, and never again. `refit=False`
        # does not hold a frame on a canvas with no image in it: it sets the fitted
        # extent to `_fit_extent()`, which falls back to the working area when nothing
        # is placed, so re-declaring per selection still moves the view. Nothing here
        # streams in and the grid is not draggable, so there is nothing to re-declare
        # for -- and a frame that holds still is what lets a growing selection be read
        # as growing rather than as the world shrinking around it.
        #
        # Centred on the stage *origin*, not on canvas (0, 0). Canvas (0, 0) is the frame
        # origin, which is wherever the stage happened to be when the dialog opened;
        # everything drawn here -- the travel limits, the grid boundary, and a plan that
        # follows the sample -- is placed about the stage origin instead. Centring on the
        # frame origin therefore shoved the whole scene by the live stage displacement,
        # which on a compustage is up to a full grid radius, and clipped it off the edge.
        self.canvas.set_world_extent(
            2 * GRID_BOUNDARY_RADIUS_M,
            2 * GRID_BOUNDARY_RADIUS_M,
            self._stage_origin_offset(),
            refit=True,
        )
        self._draw_stage_context()

    def _stage_origin_offset(self) -> Tuple[float, float]:
        """Where stage (0, 0) sits in this canvas's metres, or the frame origin if the
        projection cannot answer."""
        try:
            return self._frame().offset(STAGE_ORIGIN)
        except Exception as e:  # pragma: no cover - geometry the projection rejects
            logging.debug(f"Cannot place the stage origin; framing on the pose: {e}")
            return (0.0, 0.0)

    # ── the selection ────────────────────────────────────────────────────

    def set_selection(
        self,
        regions: Sequence[PlaneRegion],
        beam_base: FibsemStagePosition,
        beam_projection: StageProjection,
        overlap: float,
    ) -> Optional[SparseOverviewPlan]:
        """Draw the grid *regions* produce, and return the plan.

        Any tiles toggled by hand are discarded: the grid is re-derived, so a toggle
        recorded against the old one would land on a different tile or on none. Nothing
        marks a surviving toggle -- the overlay draws every enabled tile the same -- so
        keeping them would leave tiles on with no visible reason, which reads as a bug.

        Args:
            regions: Rectangles in the beam overview's displayed plane, in metres from
                *beam_base*.
            beam_base: The stage position those metres are measured from.
            beam_projection: The beam overview's own projection.
            overlap: Fractional overlap between adjacent FM tiles.

        Returns:
            The plan, or None if the regions select nothing.
        """
        self._overlap = overlap
        try:
            plan = plan_sparse_fm_overview(
                regions,
                beam_base,
                beam_projection,
                self._projection,
                self._fm_orientation,
                fov_x=self._fov[0],
                fov_y=self._fov[1],
                overlap=overlap,
                is_compustage=self.microscope.stage_is_compustage,
            )
        except ValueError as e:
            # Selecting nothing is a state, not a failure -- a region dragged to zero
            # area, or none drawn yet.
            logging.debug(f"No grid to plan from this selection: {e}")
            self.clear()
            return None

        self._plan = plan
        self._mask = [list(row) for row in plan.mask]
        self._redraw()
        self.selection_changed.emit()
        return plan

    def clear(self) -> None:
        """Drop the selection, leaving the stage context drawn."""
        if self._plan is None and not self._mask:
            return
        self._plan = None
        self._mask = []
        self._unreachable = []
        self.tile_grid_overlay.clear()
        self.selection_changed.emit()

    # ── what the host reads ──────────────────────────────────────────────

    @property
    def plan(self) -> Optional[SparseOverviewPlan]:
        """The plan as last computed, before any hand toggles."""
        return self._plan

    @property
    def mask(self) -> List[List[bool]]:
        """The mask as it stands, hand toggles included. A copy."""
        return [list(row) for row in self._mask]

    @property
    def enabled_tiles(self) -> int:
        return sum(sum(1 for on in row if on) for row in self._mask)

    @property
    def unreachable_tiles(self) -> List[Tuple[int, int]]:
        """Enabled tiles the stage cannot travel to, as (row, col)."""
        return list(self._unreachable)

    @property
    def total_tiles(self) -> int:
        return sum(len(row) for row in self._mask)

    # ── drawing ──────────────────────────────────────────────────────────

    def _frame(self) -> StageFrame:
        return StageFrame(self.canvas.canvas, self._origin, self._projection)

    def _redraw(self) -> None:
        if self._plan is None:
            self.tile_grid_overlay.clear()
            return
        width, height = self._resolution
        tiles = compute_tile_grid_from_fov(
            nrows=self._plan.rows,
            ncols=self._plan.cols,
            fov_x=self._fov[0],
            fov_y=self._fov[1],
            image_width=width,
            image_height=height,
            overlap=self._overlap,
            mask=self._mask,
        )
        offset = self._frame().offset(self._plan.centre_position)
        self.tile_grid_overlay.set_anchor(self.canvas.canvas.metres_to_canvas(*offset))
        # No `display_pixel_size`: the overlay reads it from the canvas at draw time.
        self.tile_grid_overlay.set_grid(
            tiles, (height, width), self._pixel_size, overlap=self._overlap
        )
        self._unreachable = self._out_of_reach(tiles)

    def _out_of_reach(self, tiles) -> List[Tuple[int, int]]:
        """Which of the tiles about to be acquired the stage cannot travel to.

        Through the same two helpers the runner uses -- `order_tiles` to drop the
        disabled ones, `validate_tile_stage_positions` to test what is left -- so the
        dialog and the run cannot disagree about which grids are acquirable. Masking off
        an unreachable corner is a legitimate way to fix one, and that only works if both
        sides ask the question of the same tiles.

        Worth asking here rather than leaving to `raise_if_outside_stage_limits`, which
        refuses at acquire time. On a compustage the travel is +/-999.9 um in x and only
        +/-377.8 um in y, against a grid boundary of 1000 um radius, so a selection can
        sit well inside the grid and still be unreachable -- and it is a great deal
        cheaper to hear that while the region is still under the cursor.
        """
        step_x, step_y = self._step()
        offset_x = (self._plan.cols - 1) * step_x / 2
        offset_y = (self._plan.rows - 1) * step_y / 2
        # Through the held projection, not `microscope.project_fm_stable_move`. The two
        # agree to 1.4e-20 m -- the same arithmetic over the same geometry -- but the
        # microscope's reads `fm.camera_tilt` and the camera transform on every call, at
        # **216 ms each**. One per tile on a region change is half a minute of frozen UI
        # for a 150 tile grid, and instrument traffic on a mouse event besides.
        #
        # The negated y is the convention crossing over: the layout above measures y
        # upward, as `project_fm_stable_move` takes it, and a displayed plane measures it
        # down. Verified against the live call rather than reasoned about.
        positions = {
            (tile.row, tile.col): self._projection.from_plane(
                tile.dx - offset_x,
                -(tile.dy + offset_y),
                self._plan.centre_position,
            )
            for tile in tiles
        }
        limits = getattr(self.microscope._stage, "limits", None)
        if not limits:
            return []
        # Any strategy: `order_tiles` changes the sequence and drops the disabled
        # ones, and only the second half bears on whether a tile is reachable. Going
        # through it anyway rather than filtering by hand, so the set asked about here
        # is the set the runner will visit, by construction.
        ordered = order_tiles(tiles, TileOrderStrategy.TYPEWRITER)
        return validate_tile_stage_positions(
            ordered, [positions[(t.row, t.col)] for t in ordered], limits
        )

    def _step(self) -> Tuple[float, float]:
        """Distance between adjacent tile centres, in metres."""
        return (
            self._fov[0] * (1.0 - self._overlap),
            self._fov[1] * (1.0 - self._overlap),
        )

    def _draw_stage_context(self) -> None:
        """Where the stage can go, drawn once.

        The origin is fixed and the limits do not move, so this is not refreshed. It is
        also the only reachability the dialog offers: a selection that reaches past the
        boundary is visibly past it, without a per-tile check being built.
        """
        frame = self._frame()
        try:
            centre = frame.to_canvas(STAGE_ORIGIN)
        except Exception as e:
            logging.debug(f"Cannot place the grid centre: {e}")
            return

        limits = getattr(self.microscope._stage, "limits", None)
        if not limits or not self.microscope.stage_is_compustage:
            return

        # No `surface_foreshortening` term: these lie on the sample, and the sample is
        # square-on to the camera at the fluorescence pose. The beam pane, drawn from a
        # tilted view, does need it.
        self.stage_overlay.set_shapes([
            ShapeSpec(
                kind="rect", cx=centre[0], cy=centre[1], color=STAGE_LIMITS_COLOUR,
                width=frame.length(limits["x"].max - limits["x"].min),
                height=frame.length(limits["y"].max - limits["y"].min),
                label="Stage limits",
            ),
            ShapeSpec(
                kind="circle", cx=centre[0], cy=centre[1], color=GRID_BOUNDARY_COLOUR,
                radius=frame.length(GRID_BOUNDARY_RADIUS_M), label="Grid boundary",
            ),
        ])

    # ── edits ────────────────────────────────────────────────────────────

    def _on_tile_toggled(self, row: int, col: int, enabled: bool) -> None:
        """One tile, added or dropped by hand.

        Guarded on the current shape rather than trusted: the overlay emits against the
        grid it was last given, and a selection change between the press and the release
        would otherwise write outside the mask.
        """
        if row >= len(self._mask) or col >= len(self._mask[row]):
            return
        if self._mask[row][col] == enabled:
            return
        self._mask[row][col] = enabled
        self._redraw()
        self.selection_changed.emit()
