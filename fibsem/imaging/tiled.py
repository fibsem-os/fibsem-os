from __future__ import annotations

import datetime
import logging
import math
import os
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import acquire, conversions
from fibsem.cancellation import OperationCancelledError, raise_if_cancelled
from fibsem.constants import DATETIME_FILE
from fibsem.conversions import is_inside_image_bounds

# Moved to the `tiling` package (FIB-390); re-exported so existing importers and the
# public API are unaffected. `fibsem/imaging/__init__.py` star-imports this module.
from fibsem.imaging.reduce import PreviewMosaic, downsample
from fibsem.imaging.tiling.geometry import (  # noqa: E402,F401
    TilePosition,
    _spiral_order,  # noqa: E402,F401
    compute_tile_grid,
    order_tiles,
    raise_if_outside_stage_limits,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling.plotting import (  # noqa: E402,F401
    POSITION_COLOURS,
    plot_minimap,
    plot_stage_positions_on_image,
    plot_tile_positions,
)
from fibsem.imaging.tiling.progress import MODALITY_BEAM
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
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    AutoFocusMode,
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
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
        centre_position: Optional[FibsemStagePosition] = None,
    ):
        self.microscope = microscope
        self.settings = settings
        self.stop_event = stop_event
        # Where the grid is centred. None means "wherever the stage is", resolved in
        # `_compute_grid` rather than here so it is read when the run starts rather
        # than when the runner is built. The stage still returns to where it started
        # afterwards: this is the grid's centre, not a new home. Matches
        # `FMTiledAcquisitionRunner`, which has taken one since FIB-393.
        self.centre_position = centre_position
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
        # Before `_setup`, which emits progress and makes a directory. A run with no
        # tiles is a configuration error, not an acquisition that failed: left to run
        # it walked zero tiles, emitted a *successful* terminal payload, restored the
        # stage and only then died in `_stitch` with "No tiles were acquired" -- so a
        # consumer saw the run finish and then saw it fail.
        if self.settings.n_enabled_tiles == 0:
            raise ValueError(
                "No tiles are selected, so there is nothing to acquire. "
                "Enable at least one tile in the grid."
            )
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
            "modality": MODALITY_BEAM,
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
            "modality": MODALITY_BEAM,
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
        # `_start_state` is where the stage goes home to; `_centre_position` is only
        # what the grid is measured from. They are the same unless a caller planned the
        # run around somewhere else -- a grid dragged off the stage on the overview.
        self._centre_position = (
            self.centre_position
            if self.centre_position is not None
            else self.microscope.get_stage_position()
        )

        # offset from centre to top-left corner of the grid (used only for projection)
        grid_offset_x = (settings.ncols - 1) * self._dx_step / 2
        grid_offset_y = (settings.nrows - 1) * self._dy_step / 2

        # stitched canvas
        eff_w = max(1, int(round(image_width  * (1 - overlap))))
        eff_h = max(1, int(round(image_height * (1 - overlap))))
        full_w = eff_w * (settings.ncols - 1) + image_width
        full_h = eff_h * (settings.nrows - 1) + image_height
        self._canvas = np.zeros((full_h, full_w), dtype=np.uint8)
        self._mosaic_metadata = self._build_mosaic_metadata(full_w, full_h)
        self._init_preview(full_w, full_h)

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

    def _build_mosaic_metadata(self, full_w: int, full_h: int) -> FibsemImageMetadata:
        """The mosaic's metadata, built before the first tile rather than after the last.

        It used to be a deepcopy of whichever image happened to arrive first, patched
        at stitch time. Building it up front buys three things:

        * **A partial mosaic is a real image.** A cancelled run left a bare array with
          no position, pixel size or geometry, so nothing could place it or save it.
        * **The live preview can be placed like anything else**, from its own metadata,
          rather than by a display being told the geometry out of band.
        * **The position is the grid's centre**, which is where the mosaic actually is.
          Stitching took it from `_start_state` -- where the stage *began* -- which was
          the same thing until a caller could plan the run somewhere else. A run centred
          500 um away recorded 0, so the saved overview reloaded half a millimetre out.

        The pixel size is the requested one (`hfw / width`); if the column quantises the
        field of view, `_correct_metadata_from` replaces it when the first tile lands.
        """
        state = deepcopy(self._start_state)
        state.stage_position = deepcopy(self._centre_position)

        image_settings = deepcopy(self._image_settings)
        image_settings.hfw = float(self.settings.total_fov_x)
        image_settings.resolution = (full_w, full_h)

        pixel_size = self._image_settings.hfw / self._image_settings.resolution[0]
        return FibsemImageMetadata(
            image_settings=image_settings,
            pixel_size=Point(x=pixel_size, y=pixel_size),
            microscope_state=state,
            system_info=deepcopy(self.microscope.system.info),
            hardware_geometry=deepcopy(self.microscope.hardware_geometry()),
        )

    def _correct_metadata_from(self, image: FibsemImage) -> None:
        """Take the pixel size the instrument actually delivered, once one exists.

        The planned `hfw / width` is right on every simulator and can be a fraction out
        on a column that quantises the field of view. Everything placed from this
        metadata scales by it, so a fraction out is a fraction of the whole mosaic.
        """
        actual = getattr(getattr(image, "metadata", None), "pixel_size", None)
        if actual is None or not actual.x:
            return
        self._mosaic_metadata.pixel_size = deepcopy(actual)

    def _init_preview(self, full_w: int, full_h: int) -> None:
        """A decimated copy of the mosaic, for a display to show as it fills.

        Decimated because the full one is not something to hand a display on every
        tile: a 10x10 of 1536x1024 is a 157 MB array, and a real-space canvas would
        reduce all of it for display each time. Painted per tile from that tile's own
        thumbnail, so the cost is the tile rather than the mosaic.
        """
        self._preview = PreviewMosaic(full_w, full_h, dtype=np.uint8)
        logging.debug(f"{self._preview.describe()} (full {full_h}x{full_w})")

    def _paint_preview(self, tile: TilePosition, image: FibsemImage) -> None:
        """Paint one acquired tile into the live preview.

        Best effort: a preview that cannot be built is not a reason to fail an
        acquisition that is otherwise fine, so this never raises into the tile loop.
        """
        try:
            self._preview.paint(image.filtered_data, tile.canvas_x, tile.canvas_y)
        except Exception as e:
            logging.debug(f"Could not paint the live preview: {e}")

    def _preview_image(self) -> Optional[FibsemImage]:
        """The mosaic so far, as a placeable image.

        Coarser pixels over a smaller count cover the same ground, so saying the pixel
        size is all a real-space display needs to put it in the right place at the right
        size. A fresh metadata object per emit, because the array is the live one and a
        consumer that kept the last payload would otherwise see its dimensions change
        underneath it.
        """
        metadata = deepcopy(self._mosaic_metadata)
        stride = self._preview.stride
        metadata.pixel_size = Point(
            x=self._mosaic_metadata.pixel_size.x * stride,
            y=self._mosaic_metadata.pixel_size.y * stride,
        )
        metadata.image_settings = deepcopy(self._mosaic_metadata.image_settings)
        metadata.image_settings.resolution = (
            self._preview.canvas.shape[1], self._preview.canvas.shape[0],
        )
        return FibsemImage(data=self._preview.canvas.copy(), metadata=metadata)

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
                self._correct_metadata_from(image)

            # stitch tile into canvas (overlapping regions are overwritten by later tiles)
            self._canvas[
                tile.canvas_y:tile.canvas_y + image_height,
                tile.canvas_x:tile.canvas_x + image_width,
            ] = image.filtered_data

            self._paint_preview(tile, image)
            self._n_tiles_acquired += 1
            self.microscope.tiled_acquisition_signal.emit({
                "modality": MODALITY_BEAM,
                "msg": "Tile Collected",
                "i": tile.row,
                "j": tile.col,
                "n_rows": self.settings.nrows,
                "n_cols": self.settings.ncols,
                "image": self._canvas,
                # The mosaic so far, decimated, and carrying metadata -- so a
                # real-space display can place it as one image rather than assembling
                # tiles of its own. One artist per run instead of one per tile, which
                # is what stops the canvas slowing down as tilesets accumulate
                # (FIB-627).
                #
                # Additive, and `image` above is deliberately untouched: the napari
                # minimap assigns it straight into a layer, so it has to stay a bare
                # array until that tab goes.
                "preview": self._preview_image(),
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
        # The metadata `_compute_grid` built, not a patched copy of the first tile's.
        # deepcopy so the stitched image gets its own snapshot rather than sharing the
        # runner's, which the preview also hands out.
        #
        # The position it carries is the grid's *centre*. This used to be
        # `_start_state`, where the stage began, which was the same thing until a run
        # could be planned somewhere else -- a grid dragged 500 um away recorded 0, and
        # the saved overview reloaded half a millimetre out.
        image = FibsemImage(data=self._canvas, metadata=deepcopy(self._mosaic_metadata))
        # The path is a per-run detail of the request, and the mosaic goes in the parent
        # of the tile folder, so it is taken from the live settings rather than the
        # snapshot taken before `_setup` rewrote them.
        image.metadata.image_settings.path = self._image_settings.path
        image.metadata.image_settings.filename = self._prev_label

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
    centre_position: Optional[FibsemStagePosition] = None,
) -> FibsemImage:
    """Acquire a tiled image and stitch it together.
    Args:
        microscope: The microscope connection.
        settings: Overview acquisition settings (image_settings, nrows, ncols, overlap).
        stop_event: Optional threading.Event to cancel acquisition.
        centre_position: Where to centre the grid. None means wherever the stage is.
            The stage still returns to where it started afterwards.
    Returns:
        The stitched image."""
    # add datetime to filename for uniqueness
    filename = settings.image_settings.filename
    timestamp = datetime.datetime.now().strftime(DATETIME_FILE)
    settings.image_settings.filename = f"{filename}-{timestamp}"

    return TiledAcquisitionRunner(
        microscope, settings, stop_event, centre_position=centre_position
    ).run_and_stitch()

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


