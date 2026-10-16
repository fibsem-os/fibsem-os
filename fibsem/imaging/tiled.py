from __future__ import annotations

import logging
import os
import threading
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fibsem import acquire, conversions
from fibsem.autofunctions.autofocus import run_auto_focus
from fibsem.cancellation import (
    AnyStopEvent,
    OperationCancelledError,
    raise_if_cancelled,
)

# Moved to the `tiling` package (FIB-390); re-exported so existing importers and the
# public API are unaffected. `fibsem/imaging/__init__.py` star-imports this module.
from fibsem.imaging.reduce import PreviewMosaic
from fibsem.imaging.tiling.geometry import (  # noqa: E402,F401
    TilePosition,
    _spiral_order,  # noqa: E402,F401
    compute_tile_grid,
    order_tiles,
    raise_if_outside_stage_limits,
    validate_tile_stage_positions,
)
from fibsem.imaging.tiling.layout_check import LAYOUT_FILE, TileLayoutCheck
from fibsem.imaging.tiling.plotting import (  # noqa: E402,F401
    POSITION_COLOURS,
    plot_minimap,
    plot_stage_positions_on_image,
    plot_tile_positions,
)
from fibsem.imaging.tiling.progress import (
    MODALITY_BEAM,
    TiledProgress,
    TiledStatus,
)
from fibsem.imaging.tiling.registration import (
    compose_mosaic,
    layout_summary,
    registered_positions,
)
from fibsem.imaging.tiling.reprojection import (  # noqa: E402,F401
    _inverse_y_corrected_stage_movement,
    _transform_position,
    calculate_reprojected_stage_position,
    calculate_reprojected_stage_position2,
    reproject_stage_positions_onto_image,
    reproject_stage_positions_onto_image2,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    AutoContrastMode,
    AutoFocusMode,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
    Point,
    TileOrderStrategy,
)
from fibsem.utils import current_timestamp_v3

##### TILE GRID


def centred_half_frame() -> FibsemRectangle:
    """The middle half of the frame, each way: the area the Image tab's Auto Focus
    and Auto Contrast buttons score. An overview tile's edges are the mosaic's seams,
    grid bars and the neighbour's overlap, and scoring the whole frame lets them
    drag the result; the centre is the tile's own picture. A new one each call:
    a shared rectangle would be one edit away from moving every caller."""
    return FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)


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


def stamped_overview_name(name: str) -> str:
    """`overview-image` -> `overview-image-14-23-05`, the time the run was started.

    The name is not a label, it is a location: both overview runners make the tile
    sub-folder from it and write the stitch keyed on the same name. Two runs called
    the same thing therefore land on each other -- the second overwrites the first's
    tiles *and* its mosaic. Time rather than date and time: the experiment directory
    is already dated, so inside it the time of day is what distinguishes one run
    from another. Applied to whatever a caller asks for, not only to a default: a
    name someone typed is no less prone to being reused.
    """
    return f"{name}-{current_timestamp_v3(timeonly=True)}"


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
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
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
        # _run_tile_loop()→ _first_image, _n_tiles_acquired, tile_paths, _layout_check

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
        status, error = TiledStatus.FINISHED, None
        try:
            self._autocontrast_at_centre()
            self._autofocus_if_mode(AutoFocusMode.ONCE)
            self._run_tile_loop()
        except OperationCancelledError:
            status, error = TiledStatus.CANCELLED, None
            logging.info("Tiled acquisition cancelled")
            raise
        except Exception as e:
            # The reason travels with the report now. It was caught, logged and
            # thrown away, so a failed run told the UI "Acquisition Failed" and nothing
            # about why -- and a stage-limits rejection names every offending tile.
            status, error = TiledStatus.FAILED, str(e)
            logging.error(f"Tiled acquisition failed: {e}")
            raise
        finally:
            logging.info(
                f"Tiled acquisition complete, restoring initial position: "
                f"{self._start_state.stage_position.pretty}"
            )
            self.microscope.set_microscope_state(self._start_state)
            self._emit_terminal(status, error)
        self._image_settings.path = self._prev_path

    def _emit_terminal(self, status: TiledStatus, error: Optional[str]) -> None:
        """Report how the acquisition ended.

        Counts ride along so a cancel or a failure keeps the progress it did make: the
        run stopped, it did not un-happen. `status` distinguishes the three, so a
        consumer can paint a failure red and leave a cancel alone rather than calling
        both "stopped".
        """
        total_tiles = self.settings.n_enabled_tiles
        self.microscope.tiled_acquisition_signal.emit(
            TiledProgress(
                status=status,
                modality=MODALITY_BEAM,
                completed=getattr(self, "_n_tiles_acquired", 0),
                total=total_tiles,
                error=error,
            )
        )

    def run_and_stitch(self) -> FibsemImage:
        """Acquire all tiles and return the stitched FibsemImage."""
        self.run()
        return self._stitch()

    # ── phases ───────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Prepare image settings and paths; emit initial progress signal."""
        image_settings = self.settings.image_settings
        self._focus_stack_settings = self.settings.focus_stack_settings
        self._af_mode = self.settings.autofocus_mode
        # The sweep scores the centred half-frame unless the settings name an area,
        # as the Image tab's Auto Focus button does. Focusing on the full tile
        # frame scored the seams as much as the picture, and read as soft tiles.
        # A copy: the caller's settings are not rewritten with the default.
        af_settings = self.settings.autofocus_settings
        if af_settings.reduced_area is None:
            af_settings = replace(af_settings, reduced_area=centred_half_frame())
        self._af_settings = af_settings
        # Once a working distance turns out not to be settable, say so once rather than
        # per tile: on a 5 x 5 at EACH_TILE the per-tile version is 25 identical lines.
        self._af_unavailable_logged = False

        # Refused here rather than at the first tile. A sweep with every pass disabled
        # raises inside `run_auto_focus`, and a run that dies on tile 1 of 25 has already
        # moved the stage, made a folder and emitted progress. The fluorescence runner
        # validates in its own `_setup_autofocus` for the same reason.
        if self._af_mode is not AutoFocusMode.NONE and not self._af_settings.enabled:
            raise ValueError(
                f"Autofocus mode is {self._af_mode.value} but every sweep pass is "
                f"disabled, so there is nothing to focus with."
            )

        # The mode drives the per-image flag. ONCE is one detector setting for
        # the whole mosaic, set at the grid centre before the first tile;
        # EACH_TILE is what the flag means for any single image, and
        # `acquire_image` honours it per tile. The flag was forced off here for
        # years, so the box on the overview settings read one thing and the run
        # did another.
        mode = self.settings.autocontrast_mode
        self._autocontrast_once = mode is AutoContrastMode.ONCE
        image_settings.autocontrast = mode is AutoContrastMode.EACH_TILE
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
        self.microscope.tiled_acquisition_signal.emit(
            TiledProgress(
                status=TiledStatus.STARTING,
                modality=MODALITY_BEAM,
                completed=0,
                total=self.settings.n_enabled_tiles,
            )
        )

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
        eff_w = max(1, int(round(image_width * (1 - overlap))))
        eff_h = max(1, int(round(image_height * (1 - overlap))))
        full_w = eff_w * (settings.ncols - 1) + image_width
        full_h = eff_h * (settings.nrows - 1) + image_height
        self._canvas = np.zeros((full_h, full_w), dtype=np.uint8)
        self._mosaic_metadata = self._build_mosaic_metadata(full_w, full_h)
        self._init_preview(full_w, full_h)

        logging.info(
            f"Tiled acquisition centre position: {self._centre_position.pretty}"
        )

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
            logging.debug(f"Tile ({tile.row}, {tile.col}) projected: {sp.pretty}")

        stage = self.microscope.stage
        raise_if_outside_stage_limits(
            self._ordered,
            self._tile_stage_positions,
            (stage.position.limits if stage is not None else None) or {},
        )

        # EACH_ROW is not well-defined for SPIRAL (rows are revisited non-sequentially),
        # so promote it to EACH_TILE so focus is always fresh.
        if (
            self._af_mode is AutoFocusMode.EACH_ROW
            and settings.tile_order is TileOrderStrategy.SPIRAL
        ):
            self._af_mode = AutoFocusMode.EACH_TILE
            logging.info(
                "EACH_ROW autofocus upgraded to EACH_TILE for SPIRAL tile order"
            )

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
        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            pixel_size=Point(x=pixel_size, y=pixel_size),
            microscope_state=state,
            system_info=deepcopy(self.microscope.system.info),
            hardware_geometry=deepcopy(self.microscope.hardware_geometry()),
        )
        # Who and which run, as `Session.stamp` stamps on every single
        # image. The mosaic is built here rather than acquired through that path,
        # and without these a saved overview could not say which grid it is of.
        user = getattr(self.microscope, "user", None)
        if user is not None:
            metadata.user = deepcopy(user)
        experiment = getattr(self.microscope, "experiment", None)
        if experiment is not None:
            metadata.experiment = deepcopy(experiment)
        return metadata

    def _correct_metadata_from(self, image: FibsemImage) -> None:
        """Take what the first tile says that the plan could not: when the mosaic was
        acquired, and the pixel size the instrument actually delivered.

        The mosaic is built here rather than acquired through `Beam.acquire`, so it
        has no acquisition time of its own; it was acquired from when its first tile
        was (FIB-1190).

        The planned `hfw / width` is right on every simulator and can be a fraction out
        on a column that quantises the field of view. Everything placed from this
        metadata scales by it, so a fraction out is a fraction of the whole mosaic.
        """
        metadata = getattr(image, "metadata", None)
        acquired = getattr(metadata, "acquisition_datetime", None)
        if acquired is not None:
            self._mosaic_metadata.acquisition_datetime = acquired
        actual = getattr(metadata, "pixel_size", None)
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
            self._preview.canvas.shape[1],
            self._preview.canvas.shape[0],
        )
        return FibsemImage(data=self._preview.canvas.copy(), metadata=metadata)

    def _run_tile_loop(self) -> None:
        """Move to each tile, autofocus as configured, acquire, and stitch into the canvas."""
        image_settings = self._image_settings
        image_width, image_height = image_settings.resolution
        total_tiles = self.settings.n_enabled_tiles
        self._first_image: Optional[FibsemImage] = None
        self._n_tiles_acquired: int = 0
        # Each tile as saved, in the order acquired. The stitched image is what is
        # shown; the tiles are what a model trained on single frames reads.
        self.tile_paths: List[str] = []
        # What the layout check found and what `register_tiles` did, once
        # stitched (`layout_summary`).
        self.layout_warning: Optional[str] = None
        self.registration: Optional[dict] = None
        # Measures how far apart neighbouring tiles really are, from the pictures, as
        # they arrive. It only reports (`_report_layout`); the canvas is painted from
        # the plan either way.
        self._layout_check: Optional[TileLayoutCheck] = TileLayoutCheck(
            {(t.row, t.col): (t.canvas_y, t.canvas_x) for t in self._ordered}
        )
        # Only with `register_tiles`: every tile is needed again at the end, to place
        # it where its picture says. The canvas is still painted as planned meanwhile,
        # which is what a run falls back to.
        self._kept_tiles: Dict[Tuple[int, int], np.ndarray] = {}
        prev_row = -1

        for tile, stage_pos in zip(self._ordered, self._tile_stage_positions):
            # check before moving so we skip the stage movement entirely
            _check_cancelled(self.stop_event)

            image_settings.filename = f"tile_{tile.row}_{tile.col}"

            logging.debug(f"Tile ({tile.row}, {tile.col}) — target: {stage_pos.pretty}")
            self.microscope.safe_absolute_stage_movement(stage_pos)
            logging.debug(
                f"Tile ({tile.row}, {tile.col}) — actual: {self.microscope.get_stage_position().pretty}"
            )

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
            if image.filepath:
                self.tile_paths.append(image.filepath)

            if self._first_image is None:
                self._first_image = image
                self._correct_metadata_from(image)

            # stitch tile into canvas (overlapping regions are overwritten by later tiles)
            self._canvas[
                tile.canvas_y : tile.canvas_y + image_height,
                tile.canvas_x : tile.canvas_x + image_width,
            ] = image.filtered_data

            self._paint_preview(tile, image)
            self._measure_layout(tile, image)
            if self.settings.register_tiles:
                self._kept_tiles[(tile.row, tile.col)] = image.filtered_data
            self._n_tiles_acquired += 1
            self.microscope.tiled_acquisition_signal.emit(
                TiledProgress(
                    status=TiledStatus.TILE_COLLECTED,
                    modality=MODALITY_BEAM,
                    row_index=tile.row,
                    column_index=tile.col,
                    rows=self.settings.nrows,
                    columns=self.settings.ncols,
                    # The mosaic so far, decimated, and carrying metadata -- so a
                    # real-space display can place it as one image rather than assembling
                    # tiles of its own. One artist per run instead of one per tile, which
                    # is what stops the canvas slowing down as tilesets accumulate
                    # (FIB-627).
                    #
                    # The live `self._canvas` used to go out beside this, as a bare array,
                    # because the napari minimap assigned it straight into a layer. That
                    # array is the one this loop keeps painting into, so the tab was
                    # rendering half-written tiles; it reads `preview.data` now.
                    preview=self._preview_image(),
                    completed=self._n_tiles_acquired,
                    total=total_tiles,
                )
            )

    def _acquire_tile(self, tile: TilePosition) -> FibsemImage:
        """Acquire one tile — focus-stack or plain image."""
        if self._focus_stack_settings.enabled:
            return self.microscope.imaging.focus_stacked_image(
                self._image_settings,
                n_steps=self._focus_stack_settings.n_steps,
                auto_focus=self._focus_stack_settings.auto_focus,
                stop_event=self.stop_event,
            )
        return acquire.acquire_image(self.microscope, self._image_settings)

    def _stitch(self) -> FibsemImage:
        """Assemble the stitched FibsemImage, save it to disk, and emit completion signal."""
        if self._first_image is None:
            raise ValueError("No tiles were acquired; cannot stitch.")

        signal = self.microscope.tiled_acquisition_signal
        total_tiles = self.settings.n_enabled_tiles
        signal.emit(
            TiledProgress(
                status=TiledStatus.STITCHING,
                modality=MODALITY_BEAM,
                completed=total_tiles,
                total=total_tiles,
            )
        )
        # The metadata `_compute_grid` built, not a patched copy of the first tile's.
        # deepcopy so the stitched image gets its own snapshot rather than sharing the
        # runner's, which the preview also hands out.
        #
        # The position it carries is the grid's *centre*. This used to be
        # `_start_state`, where the stage began, which was the same thing until a run
        # could be planned somewhere else -- a grid dragged 500 um away recorded 0, and
        # the saved overview reloaded half a millimetre out.
        data, registration = self._canvas, None
        if self.settings.register_tiles:
            data, registration = self._registered_canvas()
        self.registration = registration
        image = FibsemImage(data=data, metadata=deepcopy(self._mosaic_metadata))
        # The path is a per-run detail of the request, and the mosaic goes in the parent
        # of the tile folder, so it is taken from the live settings rather than the
        # snapshot taken before `_setup` rewrote them.
        image.metadata.image_settings.path = self._image_settings.path
        image.metadata.image_settings.filename = self._prev_label

        filename = os.path.join(image.metadata.image_settings.path, self._prev_label)  # type: ignore
        image.save(filename)

        self.layout_warning = self._report_layout(
            tile_dir=filename, registration=registration
        )
        signal.emit(
            TiledProgress(
                status=TiledStatus.FINISHED,
                modality=MODALITY_BEAM,
                completed=total_tiles,
                total=total_tiles,
                warning=self.layout_warning,
            )
        )
        return image

    def _measure_layout(self, tile: TilePosition, image: FibsemImage) -> None:
        """Feed one tile to the layout check. Best effort: a check that cannot run is
        not a reason to fail an acquisition, so it switches itself off instead."""
        if self._layout_check is None:
            return
        try:
            self._layout_check.add((tile.row, tile.col), image.filtered_data)
        except Exception as e:
            logging.warning(f"Tile layout check stopped: {e}")
            self._layout_check = None

    def _registered_canvas(self) -> Tuple[np.ndarray, dict]:
        """The mosaic with each tile where its picture says it is (`register_tiles`),
        or the planned canvas when the layout check says not to, and a record of which.

        Refused when the check warns: then the stage model is out of scale, and a
        registered mosaic would put positions out by that scale times their distance
        from the centre, where the planned one is out by half a seam at most.
        """
        tiles, self._kept_tiles = self._kept_tiles, {}
        if self._layout_check is None:
            note = "Tiles were left as planned: the layout check did not run."
            logging.info(note)
            return self._canvas, {"applied": False, "note": note}
        try:
            planned = {
                (t.row, t.col): (float(t.canvas_y), float(t.canvas_x))
                for t in self._ordered
                if (t.row, t.col) in tiles
            }
            positions, note = registered_positions(
                planned,
                self._layout_check.pairs,
                self._layout_check.summary(),
                tile_size=max(self._image_settings.resolution),
            )
            logging.info(note)
            if positions is None:
                return self._canvas, {"applied": False, "note": note}
            data = compose_mosaic(tiles, positions, self._canvas.shape, np.uint8)
            shifts = {
                f"{r},{c}": [y - planned[(r, c)][0], x - planned[(r, c)][1]]
                for (r, c), (y, x) in positions.items()
            }
            return data, {"applied": True, "note": note, "shifts": shifts}
        except Exception as e:
            note = f"Tiles were left as planned: aligning them failed ({e})."
            logging.warning(note)
            return self._canvas, {"applied": False, "note": note}

    def _report_layout(
        self, tile_dir: str, registration: Optional[dict] = None
    ) -> Optional[str]:
        """Log the layout check, write it into the run's tile folder, and return its
        warning. The folder is the mosaic's path without the extension, and is named
        for the run, so one run's record never lands on another's."""
        if self._layout_check is None:
            return None
        try:
            summary = self._layout_check.summary()
            logging.info(summary.describe())
            if summary.warning:
                logging.warning(summary.warning)
            self._layout_check.save(
                os.path.join(tile_dir, LAYOUT_FILE),
                pixel_size=self._mosaic_metadata.pixel_size.x,
                registration=registration,
            )
            if summary.warning and registration is not None:
                # Asked to align and did not: say so, or the operator would take the
                # mosaic for an aligned one.
                return f"{summary.warning} {registration['note']}"
            return summary.warning
        except Exception as e:
            logging.warning(f"Could not report the tile layout check: {e}")
            return None

    # ── helpers ──────────────────────────────────────────────────────────

    def _autocontrast_at_centre(self) -> None:
        """ONCE: one contrast for the whole mosaic, set at its centre before the tiles.

        The centre rather than the first tile: a typewriter order starts in a
        corner, which on a grid is as likely to be a bar or the edge of the hole as
        the picture. Scored on the centred half-frame, as the Image tab's Auto
        Contrast button is.
        """
        # getattr, as `_emit_terminal` reads its count: a runner built around
        # `_setup` (the signal tests) has no flag, and no request.
        if not getattr(self, "_autocontrast_once", False):
            return
        _check_cancelled(self.stop_event)
        logging.info(
            f"Auto contrast at the grid centre: {self._centre_position.pretty}"
        )
        self.microscope.safe_absolute_stage_movement(self._centre_position)
        _check_cancelled(self.stop_event)
        self.microscope.autocontrast(
            self._image_settings.beam_type, reduced_area=centred_half_frame()
        )

    def _autofocus_if_mode(self, mode: AutoFocusMode) -> None:
        """Run the configured focus sweep, if the current af_mode matches.

        `run_auto_focus` rather than `microscope.auto_focus`: the vendor routine takes a
        beam and a reduced area and nothing else, so this was the one acquisition in the
        codebase that could be told *when* to focus but not *how* (FIB-646). It is also
        the only autofocus path that was still vendor-specific -- `ThermoMicroscope`
        forwards to `connection.auto_functions.run_auto_focus()`, and on Tescan and
        Odemis that path had never been exercised for a tiled run at all.

        `hfw` is the tile's own field of view, not the parameter default of 150 um. The
        probe images have to frame what the tile frames, or the sweep scores a different
        picture from the one being focused.

        `reduced_area` comes from the sweep settings; `_setup` fills it with the centred
        half-frame when the settings leave it None, so the sweep scores the middle of
        the tile rather than its seams.
        """
        if self._af_mode is not mode:
            return

        result = run_auto_focus(
            self.microscope,
            beam_type=self._image_settings.beam_type,
            hfw=self._image_settings.hfw,
            settings=self._af_settings,
            stop_event=self.stop_event,
        )

        # None means the backend cannot set the working distance for this beam, so the
        # sweep declined to run rather than scoring images against a focus that never
        # moved and reporting a WD it never applied (FIB-508, TESCAN ION). The tiles are
        # still worth acquiring -- unfocused is not the same as wrong -- so this is a
        # warning and not a stop. Cancellation is the case that *does* stop, and it
        # arrives as OperationCancelledError from inside the sweep.
        if result is None and not self._af_unavailable_logged:
            self._af_unavailable_logged = True
            logging.warning(
                "Autofocus is unavailable on the %s beam for this microscope; the "
                "overview will be acquired at the current working distance.",
                self._image_settings.beam_type.name,
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


def acquire_tiled_overview(
    microscope: FibsemMicroscope,
    settings: OverviewAcquisitionSettings,
    path: str,
    name: str,
    centre_position: Optional[FibsemStagePosition] = None,
    stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    tile_paths: Optional[List[str]] = None,
    layout: Optional[Dict[str, Any]] = None,
) -> FibsemImage:
    """Acquire a tiled SEM or FIB overview and stitch it: the tiles in
    ``<path>/<name>/`` and the stitched image as ``<path>/<name>``.

    The one entry point to `TiledAcquisitionRunner`, for the Overview tab, the
    beam overview grid task and the imaging service's ``acquire_overview``. The
    beam is ``settings.image_settings.beam_type``. ``name`` is used as given;
    `stamped_overview_name` makes one that two runs cannot share. A copy of
    ``settings`` is used, so the caller's object is not rewritten with the
    path and name.

    ``centre_position`` is the grid's centre, None for wherever the stage is;
    the stage returns to where it started either way. A ``stop_event`` set
    while it runs raises `OperationCancelledError`. A ``tile_paths`` list is
    extended with each saved tile's path, in the order acquired, also when the
    run is stopped or fails part way. A ``layout`` dict is filled with the
    finished run's `layout_summary`.
    """
    settings = deepcopy(settings)
    settings.image_settings.save = True
    settings.image_settings.path = str(path)
    settings.image_settings.filename = name
    os.makedirs(path, exist_ok=True)
    runner = TiledAcquisitionRunner(
        microscope, settings, stop_event=stop_event, centre_position=centre_position
    )
    try:
        image = runner.run_and_stitch()
    finally:
        if tile_paths is not None:
            tile_paths.extend(getattr(runner, "tile_paths", []))
    if layout is not None:
        layout.update(layout_summary(runner))
    return image


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
