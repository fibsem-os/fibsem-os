from __future__ import annotations

import datetime
import logging
import os
import threading
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import acquire, conversions
from fibsem.cancellation import OperationCancelledError, raise_if_cancelled
from fibsem.conversions import is_inside_image_bounds

# Moved to the `tiling` package (FIB-390); re-exported so existing importers and the
# public API are unaffected. `fibsem/imaging/__init__.py` star-imports this module.
from fibsem.imaging.tiling.geometry import (  # noqa: E402,F401
    TilePosition,
    compute_tile_grid,
    order_tiles,
    raise_if_outside_stage_limits,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling.geometry import _spiral_order  # noqa: E402,F401
from fibsem.imaging.tiling.plotting import (  # noqa: E402,F401
    POSITION_COLOURS,
    plot_minimap,
    plot_stage_positions_on_image,
    plot_tile_positions,
)
from fibsem.imaging.tiling.reprojection import (  # noqa: E402,F401
    _inverse_y_corrected_stage_movement,
    _inverse_y_corrected_stage_movement_tescan,
    _to_raw_coordinate_system,
    _to_specimen_coordinate_system,
    _transform_position,
    calculate_reprojected_stage_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image,
    reproject_stage_positions_onto_image2,
)
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



##### TILE GRID


def _check_cancelled(stop_event: Optional[threading.Event]) -> None:
    """Raise if the stop event has been set by the caller.

    Raises `OperationCancelledError` rather than a bare `Exception`, so callers can
    tell a user cancel from a genuine failure and report it as "cancelled" rather
    than "failed" -- which is what the rest of the codebase already does for milling
    and autofocus. It subclasses `Exception`, so anything that was catching the
    previous bare raise still catches this.
    """
    raise_if_cancelled(stop_event, "Tiled acquisition cancelled by user.")


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
        # _setup()        → _image_settings, _prev_path, _prev_label,
        #                   _focus_stack_settings, _af_mode
        # _compute_grid() → _tiles, _ordered, _centre_position, _start_state,
        #                   _tile_stage_positions, _canvas, _dx_step, _dy_step
        # _run_tile_loop()→ _first_image, _n_tiles_acquired

    # ── public entry point ───────────────────────────────────────────────

    def run(self) -> None:
        """Acquire all tiles.

        Always emits a terminal progress update, whatever the outcome. Previously the
        only one came from `_stitch`, so a caller using `run()` on its own, or any run
        that was cancelled or failed, left consumers with no way to tell "done" from
        "still going" -- the progress bar simply stopped moving.
        """
        self._setup()
        self._compute_grid()
        outcome, message = "finished", "Acquisition Complete"
        try:
            self._autofocus_if_mode(AutoFocusMode.ONCE)
            self._run_tile_loop()
        except OperationCancelledError:
            outcome, message = "cancelled", "Acquisition Cancelled"
            logging.info("Tiled acquisition cancelled")
            raise
        except Exception as e:
            outcome, message = "failed", "Acquisition Failed"
            logging.error(f"Tiled acquisition failed: {e}")
            raise
        finally:
            logging.info(
                f"Tiled acquisition complete, restoring initial position: "
                f"{self._start_state.stage_position.pretty}"
            )
            self.microscope.set_microscope_state(self._start_state)
            self._emit_terminal(outcome, message)
        self._image_settings.path = self._prev_path

    def _emit_terminal(self, outcome: str, message: str) -> None:
        """Emit the final progress update for the acquisition.

        Carries `counter`/`total`/`msg` because consumers read those unconditionally --
        `FibsemMinimapWidget.handle_tile_acquisition_progress` indexes them directly
        and would raise on a payload without them. `finished` is what
        `AutoLamellaMainUI._on_tile_acquisition_progress` already branches on; it was
        previously only ever set by `_stitch`. `outcome` is the addition, so a consumer can distinguish a
        cancel from a failure rather than seeing both as "stopped". Deliberately not
        called `state`: the fluorescence progress signal already uses that key for the
        current *phase* (moving / acquiring / finished), and reusing it here for a
        terminal *outcome* would collide on "finished" while meaning something else.
        """
        total_tiles = self.settings.n_enabled_tiles
        self.microscope.tiled_acquisition_signal.emit({
            "msg": message,
            "counter": getattr(self, "_n_tiles_acquired", 0),
            "total": total_tiles,
            "finished": True,
            "outcome": outcome,
        })

    def run_and_stitch(self) -> FibsemImage:
        """Acquire all tiles and return the stitched FibsemImage."""
        self.run()
        return self._stitch()

    # ── phases ───────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Prepare image settings and paths; emit initial progress signal."""
        image_settings = self.settings.image_settings
        self._focus_stack_settings = self.settings.focus_stack_settings
        self._af_mode = self.settings.autofocus_settings.mode

        image_settings.autocontrast = False
        image_settings.save = True
        image_settings.reduced_area = None

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
            "total": self.settings.n_enabled_tiles,
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

        # The mask reaches the layout, not the traversal: `compute_tile_grid` still
        # returns the disabled tiles so the grid keeps its shape, and `order_tiles`
        # drops them *after* ordering -- so a sparse run walks the path the dense one
        # would have and misses stops along it, rather than re-deriving a pattern over
        # the holes.
        self._tiles = compute_tile_grid(settings, mask=settings.tile_mask)
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

        raise_if_outside_stage_limits(
            self._ordered, self._tile_stage_positions, self.microscope._stage.limits
        )

        # EACH_ROW is not well-defined for SPIRAL (rows are revisited non-sequentially),
        # so promote it to EACH_TILE so focus is always fresh.
        if self._af_mode is AutoFocusMode.EACH_ROW and settings.tile_order is TileOrderStrategy.SPIRAL:
            self._af_mode = AutoFocusMode.EACH_TILE
            logging.info("EACH_ROW autofocus upgraded to EACH_TILE for SPIRAL tile order")

    def _run_tile_loop(self) -> None:
        """Move to each tile, autofocus as configured, acquire, and stitch into the canvas."""
        image_settings = self._image_settings
        image_width, image_height = image_settings.resolution
        total_tiles = self.settings.n_enabled_tiles
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
                self._autofocus_if_mode(AutoFocusMode.EACH_ROW)

            self._autofocus_if_mode(AutoFocusMode.EACH_TILE)

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
                # The tile itself, alongside the growing stitch buffer above. A
                # real-space display places each tile where it was acquired, and the
                # buffer cannot say where that is: it holds integer pixel offsets, so
                # the error against the true stage position accumulates across the grid
                # (FIB-399). The tile carries its own stage position, pixel size and
                # geometry, which is everything a placement needs -- and it is the
                # position the stage actually reached, not the one it was asked for.
                #
                # Additive: every existing consumer reads `counter`/`total`/`msg`/
                # `image`, so nothing has to change to ignore this.
                "tile": image,
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
        total_tiles = self.settings.n_enabled_tiles
        signal.emit({"msg": "Stitching Tiles", "counter": total_tiles, "total": total_tiles})
        # deepcopy so the stitched image gets its OWN metadata snapshot — the edits below
        # (hfw → total FOV, stitched resolution) must not mutate the caller's shared
        # settings object or the first tile's metadata.
        image = FibsemImage(data=self._canvas, metadata=deepcopy(self._first_image.metadata))
        if image.metadata is None:
            raise ValueError("Image metadata is not set. Cannot update metadata for stitched image.")
        image.metadata.microscope_state = deepcopy(self._start_state)
        image.metadata.image_settings = deepcopy(self._image_settings)
        image.metadata.image_settings.hfw = float(self.settings.total_fov_x)
        # resolution is (width, height); numpy canvas.shape is (height, width).
        image.metadata.image_settings.resolution = (self._canvas.shape[1], self._canvas.shape[0])

        filename = os.path.join(image.metadata.image_settings.path, self._prev_label)  # type: ignore
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


