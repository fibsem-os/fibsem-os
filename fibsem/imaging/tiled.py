from __future__ import annotations

import datetime
import logging
import os
import threading
from copy import deepcopy
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import acquire, conversions
from fibsem.constants import DATETIME_FILE
from fibsem.microscope import FibsemMicroscope
from dataclasses import dataclass

from fibsem.structures import (
    AutoFocusMode,
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
    Point,
    TileOrderStrategy,
)

POSITION_COLOURS = ["lime", "blue", "cyan", "magenta", "hotpink", "yellow", "orange", "red"]


##### TILE GRID

@dataclass
class TilePosition:
    """Physical and canvas coordinates for one tile in a tiled acquisition grid.

    Attributes:
        row: Grid row index (0 = top).
        col: Grid column index (0 = left).
        dx: X offset from start_position in metres; positive = right.
        dy: Y offset from start_position in metres; negative = down (stage y inverted).
        canvas_x: Pixel left edge in the stitched canvas array.
        canvas_y: Pixel top edge in the stitched canvas array.
    """
    row: int
    col: int
    dx: float
    dy: float
    canvas_x: int
    canvas_y: int


def compute_tile_grid(settings: OverviewAcquisitionSettings) -> list[TilePosition]:
    """Compute physical and canvas positions for every tile in the grid.

    Pure function — no microscope, no side effects.

    Args:
        settings: Overview acquisition settings (hfw, resolution, nrows, ncols, overlap).
    Returns:
        Flat list of TilePosition objects in row-major order (top-left first).
    """
    image_width, image_height = settings.image_settings.resolution
    tile_fov_x = settings.image_settings.hfw
    tile_fov_y = tile_fov_x * (image_height / image_width)
    overlap = settings.overlap

    dx_step = tile_fov_x * (1 - overlap)
    dy_step = tile_fov_y * (1 - overlap)

    eff_w = max(1, int(round(image_width  * (1 - overlap))))
    eff_h = max(1, int(round(image_height * (1 - overlap))))

    tiles = []
    for i in range(settings.nrows):
        for j in range(settings.ncols):
            tiles.append(TilePosition(
                row=i, col=j,
                dx=j * dx_step,
                dy=-(i * dy_step),   # negate: stage y axis is inverted
                canvas_x=j * eff_w,
                canvas_y=i * eff_h,
            ))
    return tiles


def _spiral_order(nrows: int, ncols: int) -> list[tuple[int, int]]:
    """Return (row, col) pairs in a clockwise outward spiral from the centre tile.

    Works for any grid shape, including non-square and single-row/column grids.
    The traversal position may temporarily leave the grid bounds while stepping;
    only cells inside [0, nrows) × [0, ncols) are included in the result.
    """
    cr, cc = nrows // 2, ncols // 2
    result: list[tuple[int, int]] = [(cr, cc)]
    r, c = cr, cc
    # right, down, left, up
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0
    steps = 1
    total = nrows * ncols
    # Upper bound on iterations: spiral arms can't exceed grid perimeter
    max_steps = nrows + ncols + 2

    while len(result) < total and steps <= max_steps:
        for _ in range(2):
            dr, dc = dirs[dir_idx % 4]
            for _ in range(steps):
                r += dr
                c += dc
                if 0 <= r < nrows and 0 <= c < ncols:
                    result.append((r, c))
            dir_idx += 1
            if len(result) >= total:
                return result
        steps += 1

    return result


def order_tiles(tiles: list[TilePosition], strategy: TileOrderStrategy) -> list[TilePosition]:
    """Reorder tiles according to the movement strategy.

    Pure function — no microscope, no side effects.

    Args:
        tiles: Flat list of TilePosition objects (any order).
        strategy: TYPEWRITER, SERPENTINE, or SPIRAL.
    Returns:
        New list with tiles in traversal order.
    """
    if strategy is TileOrderStrategy.SPIRAL:
        nrows = max(t.row for t in tiles) + 1
        ncols = max(t.col for t in tiles) + 1
        tile_map = {(t.row, t.col): t for t in tiles}
        return [tile_map[rc] for rc in _spiral_order(nrows, ncols) if rc in tile_map]

    rows = sorted(set(t.row for t in tiles))
    result = []
    for row_idx, row in enumerate(rows):
        row_tiles = sorted([t for t in tiles if t.row == row], key=lambda t: t.col)
        if strategy is TileOrderStrategy.SERPENTINE and row_idx % 2 == 1:
            row_tiles = list(reversed(row_tiles))
        result.extend(row_tiles)
    return result


def plot_tile_positions(
    tiles: list[TilePosition],
    settings: OverviewAcquisitionSettings,
    ax: Optional[plt.Axes] = None,
    stage_positions: Optional[list[FibsemStagePosition]] = None,
) -> Figure:
    """Plot the tile grid with traversal order, for debugging and validation.

    Args:
        tiles: Ordered list of TilePosition objects (acquisition order).
        settings: Overview acquisition settings (for FOV dimensions and labels).
        ax: Optional existing axes to draw on; creates a new figure if None.
        stage_positions: Optional list of pre-computed FibsemStagePosition objects
            (same length as tiles). When provided, the actual projected positions are
            overlaid as white crosses + dotted path so you can compare the ideal grid
            against the real stage coordinates returned by project_stable_move.
    Returns:
        The matplotlib Figure.
    """
    import matplotlib.patches as mpatches
    from fibsem import constants

    image_width, image_height = settings.image_settings.resolution
    tile_fov_x = settings.image_settings.hfw * constants.SI_TO_MICRO  # µm
    tile_fov_y = tile_fov_x * (image_height / image_width)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # draw tiles
    for order_idx, tile in enumerate(tiles):
        cx = tile.dx * constants.SI_TO_MICRO  # µm
        cy = tile.dy * constants.SI_TO_MICRO
        x0 = cx - tile_fov_x / 2
        y0 = cy - tile_fov_y / 2
        colour = POSITION_COLOURS[tile.row % len(POSITION_COLOURS)]
        rect = mpatches.FancyBboxPatch(
            (x0, y0), tile_fov_x, tile_fov_y,
            boxstyle="round,pad=0.01",
            linewidth=1, edgecolor="white", facecolor=colour, alpha=0.4,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, str(order_idx), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # draw traversal path
    if len(tiles) > 1:
        xs = [t.dx * constants.SI_TO_MICRO for t in tiles]
        ys = [t.dy * constants.SI_TO_MICRO for t in tiles]
        for k in range(len(tiles) - 1):
            ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.0))

    # overlay actual projected stage positions (if provided)
    if stage_positions is not None and len(stage_positions) > 0:
        ref = stage_positions[0]
        sxs = [(sp.x - ref.x) * constants.SI_TO_MICRO for sp in stage_positions]
        sys_ = [(sp.y - ref.y) * constants.SI_TO_MICRO for sp in stage_positions]
        ax.plot(sxs, sys_, linestyle=":", color="white", lw=0.8, alpha=0.6)
        ax.plot(sxs, sys_, marker="x", color="white", ms=6, markeredgewidth=1.5, linestyle="none")

    sym = constants.MICRON_SYMBOL
    ax.set_xlabel(f"X ({sym})")
    ax.set_ylabel(f"Y ({sym})")
    ax.set_aspect("equal")
    ax.set_facecolor("#1e2027")
    fig.patch.set_facecolor("#1e2027")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    strategy_name = settings.tile_order.value.title()
    ax.set_title(
        f"{strategy_name} — {settings.nrows}×{settings.ncols} tiles, "
        f"{settings.overlap*100:.0f}% overlap"
    )
    ax.autoscale_view()
    fig.tight_layout()
    return fig


def _check_cancelled(stop_event: Optional[threading.Event]) -> None:
    """Raise if the stop event has been set by the caller."""
    if stop_event and stop_event.is_set():
        raise Exception("User Stopped Acquisition")


##### TILED ACQUISITION

class TiledAcquisitionRunner:
    """Orchestrates a tiled image acquisition as a series of discrete phases.

    State accumulated across phases is held on ``self``, making it straightforward
    to extend the acquisition with new pre- or post-tile steps (e.g. a focus map)
    without modifying the main loop.

    Typical usage::

        result = TiledAcquisitionRunner(microscope, settings, stop_event).run()
    """

    def __init__(
        self,
        microscope: FibsemMicroscope,
        settings: OverviewAcquisitionSettings,
        stop_event: Optional[threading.Event] = None,
    ):
        self.microscope = microscope
        self.settings = settings
        self.stop_event = stop_event
        # _setup()        → _image_settings, _cryo, _prev_path, _prev_label,
        #                   _focus_stack_settings, _af_mode
        # _compute_grid() → _tiles, _ordered, _centre_position, _start_state,
        #                   _tile_stage_positions, _canvas, _dx_step, _dy_step
        # _run_tile_loop()→ _first_image, _n_tiles_acquired

    # ── public entry point ───────────────────────────────────────────────

    def run(self) -> None:
        """Acquire all tiles."""
        self._setup()
        self._compute_grid()
        try:
            self._autofocus_if_mode(AutoFocusMode.ONCE)
            self._run_tile_loop()
        except Exception as e:
            logging.error(f"Tiled acquisition failed: {e}")
            raise
        finally:
            logging.info(
                f"Tiled acquisition complete, restoring initial position: "
                f"{self._start_state.stage_position.pretty}"
            )
            self.microscope.set_microscope_state(self._start_state)
        self._image_settings.path = self._prev_path

    def run_and_stitch(self) -> FibsemImage:
        """Acquire all tiles and return the stitched FibsemImage."""
        self.run()
        return self._stitch()

    # ── phases ───────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Prepare image settings and paths; emit initial progress signal."""
        image_settings = self.settings.image_settings
        self._cryo = image_settings.autogamma  # capture before clearing below
        self._focus_stack_settings = self.settings.focus_stack_settings
        self._af_mode = self.settings.autofocus_settings.mode

        # autogamma is applied post-stitch for cryo; disable during tile acquisition
        image_settings.autogamma = False
        image_settings.autocontrast = False
        image_settings.save = True

        # save tile intermediates into a sub-folder; stitched image goes in the parent
        self._prev_path = image_settings.path
        self._prev_label = image_settings.filename
        image_settings.path = os.path.join(self._prev_path, self._prev_label)  # type: ignore
        os.makedirs(image_settings.path, exist_ok=True)  # type: ignore
        image_settings.filename = self._prev_label

        self._image_settings = image_settings

        # notify the UI immediately so the progress bar appears before the first move
        self.microscope.tiled_acquisition_signal.emit({
            "msg": "Computing Tile Positions",
            "counter": 0,
            "total": self.settings.nrows * self.settings.ncols,
        })

    def _compute_grid(self) -> None:
        """Compute tile order and pre-project every stage position from the grid centre.

        Records the current stage position as the grid centre, then computes each tile's
        absolute stage position by projecting the tile's (dx, dy) offset — adjusted by
        the grid offset so they are relative to centre rather than the top-left corner —
        using ``project_stable_move``.

        Previously the code moved the stage to the top-left corner before projecting.
        That move was unnecessary because ``project_stable_move`` is a pure mathematical
        computation that depends only on tilt and rotation (which are constant throughout
        a tiled acquisition), not on the absolute x/y/z coordinates.  Projecting from
        centre with adjusted offsets gives identical results and saves one stage movement.

        Mathematical equivalence:
            project(tile.dx, tile.dy, base=top_left)
            == project(tile.dx - grid_offset_x, tile.dy + grid_offset_y, base=centre)
        """
        image_settings = self._image_settings
        settings = self.settings

        image_width, image_height = image_settings.resolution
        tile_fov_x = image_settings.hfw
        tile_fov_y = tile_fov_x * (image_height / image_width)
        overlap = settings.overlap
        self._dx_step = tile_fov_x * (1 - overlap)
        self._dy_step = tile_fov_y * (1 - overlap)

        self._tiles = compute_tile_grid(settings)
        self._ordered = order_tiles(self._tiles, settings.tile_order)

        self._start_state = self.microscope.get_microscope_state()
        self._centre_position = self.microscope.get_stage_position()

        # offset from centre to top-left corner of the grid (used only for projection)
        grid_offset_x = (settings.ncols - 1) * self._dx_step / 2
        grid_offset_y = (settings.nrows - 1) * self._dy_step / 2

        # stitched canvas
        eff_w = max(1, int(round(image_width  * (1 - overlap))))
        eff_h = max(1, int(round(image_height * (1 - overlap))))
        full_w = eff_w * (settings.ncols - 1) + image_width
        full_h = eff_h * (settings.nrows - 1) + image_height
        self._canvas = np.zeros((full_h, full_w), dtype=np.uint8)

        logging.info(f"Tiled acquisition centre position: {self._centre_position.pretty}")

        self._tile_stage_positions = [
            self.microscope.project_stable_move(
                dx=tile.dx - grid_offset_x,
                dy=tile.dy + grid_offset_y,
                beam_type=image_settings.beam_type,
                base_position=self._centre_position,
            )
            for tile in self._ordered
        ]
        for tile, sp in zip(self._ordered, self._tile_stage_positions):
            logging.info(f"Tile ({tile.row}, {tile.col}) projected: {sp.pretty}")

        # EVERY_ROW is not well-defined for SPIRAL (rows are revisited non-sequentially),
        # so promote it to EVERY_TILE so focus is always fresh.
        if self._af_mode is AutoFocusMode.EVERY_ROW and settings.tile_order is TileOrderStrategy.SPIRAL:
            self._af_mode = AutoFocusMode.EVERY_TILE
            logging.info("EVERY_ROW autofocus upgraded to EVERY_TILE for SPIRAL tile order")

    def _run_tile_loop(self) -> None:
        """Move to each tile, autofocus as configured, acquire, and stitch into the canvas."""
        image_settings = self._image_settings
        image_width, image_height = image_settings.resolution
        total_tiles = self.settings.nrows * self.settings.ncols
        self._first_image: Optional[FibsemImage] = None
        self._n_tiles_acquired: int = 0
        prev_row = -1

        for tile, stage_pos in zip(self._ordered, self._tile_stage_positions):
            # check before moving so we skip the stage movement entirely
            _check_cancelled(self.stop_event)

            image_settings.filename = f"tile_{tile.row}_{tile.col}"

            logging.info(f"Tile ({tile.row}, {tile.col}) — target: {stage_pos.pretty}")
            self.microscope.safe_absolute_stage_movement(stage_pos)
            logging.info(f"Tile ({tile.row}, {tile.col}) — actual: {self.microscope.get_stage_position().pretty}")

            # check after moving in case cancel was requested during the move
            _check_cancelled(self.stop_event)

            if tile.row != prev_row:
                prev_row = tile.row
                self._autofocus_if_mode(AutoFocusMode.EVERY_ROW)

            self._autofocus_if_mode(AutoFocusMode.EVERY_TILE)

            # apply per-tile focus offset (no-op until focus map is implemented)
            self._apply_focus_offset(tile)

            logging.info(f"Acquiring Tile ({tile.row}, {tile.col})")
            image = self._acquire_tile(tile)

            if self._first_image is None:
                self._first_image = image

            # stitch tile into canvas (overlapping regions are overwritten by later tiles)
            self._canvas[
                tile.canvas_y:tile.canvas_y + image_height,
                tile.canvas_x:tile.canvas_x + image_width,
            ] = image.filtered_data

            self._n_tiles_acquired += 1
            self.microscope.tiled_acquisition_signal.emit({
                "msg": "Tile Collected",
                "i": tile.row,
                "j": tile.col,
                "n_rows": self.settings.nrows,
                "n_cols": self.settings.ncols,
                "image": self._canvas,
                "counter": self._n_tiles_acquired,
                "total": total_tiles,
            })

    def _acquire_tile(self, tile: TilePosition) -> FibsemImage:
        """Acquire one tile — focus-stack or plain image."""
        if self._focus_stack_settings.enabled:
            return acquire.acquire_focus_stacked_image(
                microscope=self.microscope,
                image_settings=self._image_settings,
                n_steps=self._focus_stack_settings.n_steps,
                auto_focus=self._focus_stack_settings.auto_focus,
            )
        return acquire.acquire_image(self.microscope, self._image_settings)

    def _stitch(self) -> FibsemImage:
        """Assemble the stitched FibsemImage, save it to disk, and emit completion signal."""
        if self._first_image is None:
            raise ValueError("No tiles were acquired; cannot stitch.")

        signal = self.microscope.tiled_acquisition_signal
        total_tiles = self.settings.nrows * self.settings.ncols
        signal.emit({"msg": "Stitching Tiles", "counter": total_tiles, "total": total_tiles})
        image = FibsemImage(data=self._canvas, metadata=self._first_image.metadata)
        if image.metadata is None:
            raise ValueError("Image metadata is not set. Cannot update metadata for stitched image.")
        image.metadata.microscope_state = deepcopy(self._start_state)
        image.metadata.image_settings = self._image_settings
        image.metadata.image_settings.hfw = deepcopy(float(self.settings.total_fov_x))
        image.metadata.image_settings.resolution = deepcopy((self._canvas.shape[0], self._canvas.shape[1]))

        filename = os.path.join(image.metadata.image_settings.path, self._prev_label)  # type: ignore
        image.save(filename)

        if self._cryo:
            from fibsem.imaging.autogamma import auto_gamma
            image = auto_gamma(image, method="autogamma")
            filename = os.path.join(image.metadata.image_settings.path, f"{self._prev_label}-autogamma")  # type: ignore
            image.save(filename)

        signal.emit({"msg": "Done", "counter": total_tiles, "total": total_tiles, "finished": True})
        return image

    # ── helpers ──────────────────────────────────────────────────────────

    def _autofocus_if_mode(self, mode: AutoFocusMode) -> None:
        """Run autofocus and check for cancellation if the current af_mode matches."""
        if self._af_mode is mode:
            self.microscope.auto_focus(
                beam_type=self._image_settings.beam_type,
                reduced_area=self._image_settings.reduced_area,
            )
            _check_cancelled(self.stop_event)

    # ── future feature hooks ─────────────────────────────────────────────

    def _measure_focus_map(self) -> None:
        """Pre-acquisition pass: visit anchor tiles, measure focus, build interpolated map.

        Called between _compute_grid() and _run_tile_loop() when a focus map is enabled.
        Populates ``self._focus_map: dict[tuple[int, int], float]`` — a per-tile
        working-distance offset (metres) derived from bilinear or plane-fit interpolation
        of measured anchor values.

        Not yet implemented.
        """
        pass

    def _apply_focus_offset(self, tile: TilePosition) -> None:
        """Apply the per-tile working-distance offset from the focus map before imaging.

        No-op until ``_measure_focus_map`` is implemented.
        """
        pass


def tiled_image_acquisition_and_stitch(
    microscope: FibsemMicroscope,
    settings: OverviewAcquisitionSettings,
    stop_event: Optional[threading.Event] = None,
) -> FibsemImage:
    """Acquire a tiled image and stitch it together.
    Args:
        microscope: The microscope connection.
        settings: Overview acquisition settings (image_settings, nrows, ncols, overlap).
        stop_event: Optional threading.Event to cancel acquisition.
    Returns:
        The stitched image."""
    # add datetime to filename for uniqueness
    filename = settings.image_settings.filename
    timestamp = datetime.datetime.now().strftime(DATETIME_FILE)
    settings.image_settings.filename = f"{filename}-{timestamp}"

    return TiledAcquisitionRunner(microscope, settings, stop_event).run_and_stitch()

##### REPROJECTION
# TODO: move these to fibsem.imaging.reprojection?
def calculate_reprojected_stage_position(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    # difference between current position and image position
    delta = pos - image.metadata.stage_position

    # projection of the positions onto the image
    dx = delta.x
    dy = np.sqrt(delta.y**2 + delta.z**2) # TODO: correct for perspective here
    dy = dy if (delta.y<0) else -dy

    pt_delta = Point(dx, dy)
    px_delta = pt_delta._to_pixels(image.metadata.pixel_size.x)

    beam_type = image.metadata.image_settings.beam_type
    if beam_type is BeamType.ELECTRON:
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation
    
    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    # account for compustage tilt, when mounted upside down
    if np.isclose(image.metadata.stage_position.t, np.radians(-180), atol=np.radians(5)):
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point

def reproject_stage_positions_onto_image(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds

    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:


        # hotfix (pat): demo returns None positions #240
        if image.metadata.microscope_state.stage_position.x is None:
            image.metadata.microscope_state.stage_position.x = 0
        if image.metadata.microscope_state.stage_position.y is None:
            image.metadata.microscope_state.stage_position.y = 0
        if image.metadata.microscope_state.stage_position.z is None:
            image.metadata.microscope_state.stage_position.z = 0
        if image.metadata.microscope_state.stage_position.r is None:
            image.metadata.microscope_state.stage_position.r = 0
        if image.metadata.microscope_state.stage_position.t is None:
            image.metadata.microscope_state.stage_position.t = 0      
                
        # automate logic for transforming positions
        # assume only two valid positions are when stage is flat to either beam...  
        # r needs to be 180 degrees different
        # currently only one way: Flat to Ion -> Flat to Electron
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):     
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position(image, pos)
        pt.name = pos.name
        
        if bound and not is_inside_image_bounds([pt.y, pt.x], image.data.shape):
            continue
        
        points.append(pt)
    
    return points

def calculate_reprojected_stage_position2(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")

    if image.metadata.microscope_state.stage_position is None:
        raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")


    beam_type = image.metadata.image_settings.beam_type
    base_stage_position = image.metadata.microscope_state.stage_position 
    pixel_size = image.metadata.pixel_size.x

    scan_rotation = None
    if beam_type is BeamType.ELECTRON:
        if image.metadata.microscope_state.electron_beam is None:
            raise ValueError("Image metadata does not contain a valid electron beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        if image.metadata.microscope_state.ion_beam is None:
            raise ValueError("Image metadata does not contain a valid ion beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation

    if scan_rotation is None:
        raise ValueError("Image metadata does not contain a valid scan rotation. Cannot reproject stage position.")

    # difference between current position and image position
    delta = pos - base_stage_position

    # projection of the positions onto the image
    dx = delta.x
    if dx is None:
        raise ValueError("Stage position x coordinate is None. Cannot reproject stage position.")
    # dy = microscope._inverse_y_corrected_stage_movement(dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore
    dy = _inverse_y_corrected_stage_movement(image, dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore

    pt_delta = Point(dx, -dy)
    px_delta = pt_delta._to_pixels(pixel_size)

    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    return point

def reproject_stage_positions_onto_image2(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds

    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:

        # compucentric rotation correction
        if image.metadata is None or image.metadata.microscope_state is None:
            raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position.r is None:
            raise ValueError("Image metadata does not contain a valid stage position r coordinate. Cannot reproject stage position.")
        if pos.r is None:
            raise ValueError("Stage position r coordinate is None. Cannot reproject stage position.")
        # automate logic for transforming positions
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position2(image, pos)
        pt.name = pos.name

        if bound and not is_inside_image_bounds((pt.y, pt.x), image.data.shape):
            continue

        points.append(pt)

    return points


def plot_stage_positions_on_image(
        image: FibsemImage,
        positions: List[FibsemStagePosition],
        show: bool = False,
        bound: bool = True,
        color: Optional[str] = None,
        show_scalebar: bool = False,
        show_names: bool = True,
        figsize: Optional[Tuple[int, int]] = (15, 15)) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points. (None -> default colour cycle)
    Returns:
        The matplotlib figure."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage positions.")

    # reproject stage positions onto image 
    points = reproject_stage_positions_onto_image2(image=image, positions=positions)

    # construct matplotlib figure
    fig = plt.figure(figsize = figsize)
    plt.imshow(image.data, cmap="gray")

    for i, pt in enumerate(points):

        # if points outside image, don't plot
        if bound and not is_inside_image_bounds((pt.y, pt.x), (image.data.shape[0], image.data.shape[1])):
            continue

        if color is None:
            c = POSITION_COLOURS[i%len(POSITION_COLOURS)]
        else:
            c = color
        plt.plot(pt.x, pt.y, ms=20, c=c, marker="+", markeredgewidth=2, label=f"{pt.name}")

        if show_names:
            # draw position name next to point
            plt.text(pt.x, pt.y-50, pt.name, fontsize=14, color=c, alpha=0.75)

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )
            plt.gca().add_artist(scalebar)
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    plt.axis("off")
    if show:
        plt.show()

    return fig

def plot_minimap(
        image: FibsemImage,
        positions: List[FibsemStagePosition],
        current_position: Optional[FibsemStagePosition] = None,
        grid_positions: Optional[List[FibsemStagePosition]] = None,
        show: bool = False,
        bound: bool = True,
        color: str = "cyan",
        show_scalebar: bool = False,
        show_names: bool = True,
        show_grid_radius: bool = False,
        fontsize: int = 12,
        markersize: int = 20,
        figsize: Optional[Tuple[int, int]] = (15, 15),
        ax: Optional[plt.Axes] = None) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        current_position: Optional current position to highlight
        grid_positions: Optional grid positions to show
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points.
        show_scalebar: Whether to show a scalebar
        show_names: Whether to show position names as labels
        fontsize: Font size for position name labels (default: 14)
        figsize: Figure size in inches (default: (15, 15))
    Returns:
        The matplotlib figure."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage positions.")

    all_positions = list(positions)
    if current_position is not None:
        all_positions.append(current_position)
    if grid_positions is not None:
        all_positions.extend(grid_positions)

    # construct matplotlib figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(image.data, cmap="gray")

    # reproject stage positions onto image 
    points = reproject_stage_positions_onto_image2(image=image, positions=all_positions)

    marker_entries: List[dict] = []
    for i, pt in enumerate(points):

        # if points outside image, don't plot
        if bound and not is_inside_image_bounds((pt.y, pt.x), (image.data.shape[0], image.data.shape[1])):
            continue
        
        if pt.name is None:
            pt.name = f"Position {i:02d}"

        c = color
        if "Grid" in pt.name:
            c = "red"
        elif "Current Position" in pt.name:
            c = "yellow"

        marker_entries.append(
            {
                "point": (pt.x, pt.y),
                "color": c,
                "label": pt.name,
            }
        )

        # show grid radius
        if c == "red" and show_grid_radius:
            r_pixels = 1000e-6 / image.metadata.pixel_size.x 
            ax.add_artist(plt.Circle((pt.x, pt.y), radius=r_pixels, color=c, fill=False, linewidth=5)
            )

    if marker_entries:
        scatter_array = np.array([entry["point"] for entry in marker_entries])
        scatter_colors = [entry["color"] for entry in marker_entries]
        ax.scatter(
            scatter_array[:, 0],
            scatter_array[:, 1],
            c=scatter_colors,
            marker="+",
            s=markersize ** 2,
            linewidths=2,
        )

        if show_names:
            for entry in marker_entries:
                x, y = entry["point"]
                ax.text(
                    x + 10,
                    y - 10,
                    entry["label"],
                    fontsize=fontsize,
                    color=entry["color"],
                    alpha=0.75,
                    clip_on=True,
                )

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar
            ax.add_artist(
                ScaleBar(
                    dx=image.metadata.pixel_size.x,
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                )
            )
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    ax.axis("off")
    if show:
        plt.show()

    return fig

def convert_image_coord_to_stage_position(
    microscope: FibsemMicroscope, image: FibsemImage, coord: Tuple[float, float]
) -> FibsemStagePosition:
    """Convert a coordinate in the image to a stage position. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coord: The coordinate in the image (y,x).
    Returns:
        The stage position.
    """
    # convert image to microscope image coordinates
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=coord[1], y=coord[0]),
        image=image.data,
        pixelsize=image.metadata.pixel_size.x,
    )
    # project as stage position
    stage_position = microscope.project_stable_move(
        dx=point.x,
        dy=point.y,
        beam_type=image.metadata.image_settings.beam_type,
        base_position=image.metadata.microscope_state.stage_position,
    )

    return stage_position

def convert_image_coordinates_to_stage_positions(
    microscope: FibsemMicroscope, image: FibsemImage, coords: List[Tuple[float, float]]
) -> List[FibsemStagePosition]:
    """Convert a list of coordinates in the image to a list of stage positions. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coords: The coordinates in the image (y,x).
    Returns:
        The stage positions."""

    stage_positions = []
    for i, coord in enumerate(coords):
        stage_position = convert_image_coord_to_stage_position(
            microscope=microscope, image=image, coord=coord
        )
        stage_position.name = f"Position {i:02d}"
        stage_positions.append(stage_position)
    return stage_positions

##### THERMO ONLY

X_OFFSET = -0.0005127403888932854 
Y_OFFSET = 0.0007937916666666666

def _to_specimen_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    specimen_position = pos - specimen_offset

    return specimen_position

def _to_raw_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    raw_position = pos + specimen_offset

    return raw_position


def _transform_position(pos: FibsemStagePosition) -> FibsemStagePosition:
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam).
    Args:
        pos: The position flat to the beam.
    Returns:
        The position flat to the other beam."""

    specimen_position = _to_specimen_coordinate_system(pos)
    # print("raw      pos: ", pos)
    # print("specimen pos: ", specimen_position)

    # # inverse xy (rotate 180 degrees)
    specimen_position.x = -specimen_position.x
    specimen_position.y = -specimen_position.y

    # movement offset (calibration for compucentric rotation error)
    specimen_position.x += 50e-6
    specimen_position.y += 25e-6

    # print("rotated pos: ", specimen_position)

    # _to_raw_coordinates
    transformed_position = _to_raw_coordinate_system(specimen_position)
    transformed_position.name = pos.name

    # print("trans   pos: ", transformed_position)
    logging.info(f"Initial position {pos} was transformed to {transformed_position}")

    return transformed_position

def _inverse_y_corrected_stage_movement(
    image: FibsemImage,
    dy: float,
    dz: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> float:
        """
        Calculate the expected_y input from dy, dz stage movements and beam_type.
        This is the inverse of _y_corrected_stage_movement.

        Args:
            dy (float): actual y stage movement
            dz (float): actual z stage movement  
            beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

        Returns:
            float: expected_y input that would produce the given dy, dz movements
        """
        if image.metadata is None or image.metadata.system is None:
            raise ValueError("Image metadata or system metadata is not set. Cannot calculate inverse y corrected stage movement.")

        # all angles in radians
        sem_column_tilt = np.deg2rad(image.metadata.system.electron.column_tilt)
        fib_column_tilt = np.deg2rad(image.metadata.system.ion.column_tilt)

        stage_pretilt = np.deg2rad(image.metadata.system.stage.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(image.metadata.system.stage.rotation_reference) % (2 * np.pi)
        stage_rotation_flat_to_ion = np.deg2rad(image.metadata.system.stage.rotation_180) % (2 * np.pi)

        # current stage position
        current_stage_position = image.metadata.stage_position
        stage_rotation = current_stage_position.r % (2 * np.pi) if current_stage_position.r is not None else 0.0
        stage_tilt = current_stage_position.t if current_stage_position.t is not None else 0.0

        # Handle compustage case
        compustage_sign = 1.0
        stage_is_compustage = "Arctis" in image.metadata.system.info.model or image.metadata.system.sim.get("is_compustage", False)
        if stage_is_compustage: # TODO: add compustage to metadata
            if stage_tilt <= 0 and stage_tilt > np.radians(-90):
                compustage_sign = -1.0
            stage_tilt += np.pi

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        from fibsem import movement
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

        # perspective tilt adjustment
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

        # Reverse the calculations from the forward function:
        # Forward: y_move = y_sample_move * cos(corrected_pretilt_angle)
        # Forward: z_move = -y_sample_move * sin(corrected_pretilt_angle)
        # Therefore: y_sample_move can be calculated from either dy or dz

        # Calculate y_sample_move from dy and dz (should be consistent)
        cos_pretilt = np.cos(corrected_pretilt_angle)
        sin_pretilt = np.sin(corrected_pretilt_angle)
        
        if abs(cos_pretilt) > abs(sin_pretilt):
            # Use dy calculation when cos component is larger
            y_sample_move = dy / cos_pretilt
        else:
            # Use dz calculation when sin component is larger
            y_sample_move = -dz / sin_pretilt

        # Reverse: expected_y = y_sample_move * cos(stage_tilt + perspective_tilt_adjustment)
        expected_y = y_sample_move * np.cos(stage_tilt + perspective_tilt_adjustment)

        # Apply compustage correction if needed
        if stage_is_compustage:
            expected_y *= compustage_sign

        return expected_y

