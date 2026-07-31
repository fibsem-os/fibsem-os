import logging
import os
import threading
from dataclasses import replace
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fibsem import utils
from fibsem.cancellation import OperationCancelledError, raise_if_cancelled
from fibsem.fm.calibration import run_autofocus, run_coarse_fine_autofocus
from fibsem.imaging.tiling.geometry import (
    TilePosition,
    compute_tile_grid_from_fov,
    order_tiles,
)
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import (
    AutoFocusMode,
    ChannelSettings,
    FluorescenceImage,
    OverviewParameters,
    ZParameters,
    ZStackOrder,
    FMStagePosition,
    AutoFocusSettings,
)
from fibsem.fm.timing import estimate_tileset_acquisition_time, estimate_positions_acquisition_time
from fibsem.structures import BeamType, FibsemStagePosition, TileOrderStrategy
if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

def acquire_channels(
    microscope: FluorescenceMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    stop_event: Optional[threading.Event] = None,
) -> Optional[FluorescenceImage]:
    """Acquire images for multiple channels."""

    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]  # Ensure settings is a list

    images: List[FluorescenceImage] = []
    for i, channel in enumerate(channel_settings):
        # Same payload shape `acquire_z_stack` emits, minus the z fields. Without it a
        # multi-channel acquisition was silent, so anything watching progress within a
        # tile had nothing to show unless a z-stack happened to be enabled.
        microscope.acquisition_progress_signal.emit({
            "state": "acquiring",
            "task": "channels",
            "channel": channel.name,
            "channel_index": i + 1,
            "total_channels": len(channel_settings),
        })

        # Check for cancellation before each channel
        if stop_event and stop_event.is_set():
            logging.info("Multi-channel acquisition cancelled")
            return None

        image = microscope.acquire_image(channel)
        images.append(image)
    return FluorescenceImage.create_multi_channel_image(images)


def acquire_z_stack(
    microscope: FluorescenceMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: ZParameters,
    stop_event: Optional["threading.Event"] = None,
) -> Optional[FluorescenceImage]:
    """Acquire a Z-stack of images for a given channel."""

    z_init = microscope.objective.position  # initial z position of the objective
    z_positions = zparams.generate_positions(z_init=z_init)
    images: List[FluorescenceImage] = []

    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    if zparams.order == ZStackOrder.Z_LEVEL:
        # Z-level-wise: for each z-plane, acquire all channels
        z_level_images: List[List[FluorescenceImage]] = []
        for j, z in enumerate(z_positions):
            if stop_event and stop_event.is_set():
                logging.info("Z-stack acquisition cancelled before z-level acquisition")
                microscope.objective.move_absolute(z_init)
                return None

            microscope.objective.move_absolute(z)
            ch_images: List[FluorescenceImage] = []
            for i, ch in enumerate(channel_settings):
                microscope.acquisition_progress_signal.emit({
                    "state": "acquiring",
                    "task": "z-stack",
                    "channel": ch.name,
                    "channel_index": i + 1,
                    "total_channels": len(channel_settings),
                    "zlevel": j + 1,
                    "total_zlevels": len(z_positions),
                })
                if stop_event and stop_event.is_set():
                    logging.info("Z-stack acquisition cancelled during z-level acquisition")
                    microscope.objective.move_absolute(z_init)
                    return None
                ch_images.append(microscope.acquire_image(channel_settings=ch))
            z_level_images.append(ch_images)

        # Transpose [z][ch] -> per-channel z-stacks
        for i in range(len(channel_settings)):
            ch_z_imgs = [z_level_images[j][i] for j in range(len(z_positions))]
            images.append(FluorescenceImage.create_z_stack(ch_z_imgs))
    else:
        # Channel-wise (default): for each channel, acquire all z-planes
        for i, ch in enumerate(channel_settings):
            if stop_event and stop_event.is_set():
                logging.info("Z-stack acquisition cancelled before channel acquisition")
                microscope.objective.move_absolute(z_init)
                return None

            ch_images = []
            for j, z in enumerate(z_positions):
                microscope.acquisition_progress_signal.emit({
                    "state": "acquiring",
                    "task": "z-stack",
                    "channel": ch.name,
                    "channel_index": i + 1,
                    "total_channels": len(channel_settings),
                    "zlevel": j + 1,
                    "total_zlevels": len(z_positions),
                })
                if stop_event and stop_event.is_set():
                    logging.info("Z-stack acquisition cancelled during z-stack")
                    microscope.objective.move_absolute(z_init)
                    return None

                microscope.objective.move_absolute(z)
                ch_images.append(microscope.acquire_image(channel_settings=ch))

            images.append(FluorescenceImage.create_z_stack(ch_images))

    # restore objective to initial position
    microscope.objective.move_absolute(z_init)

    return FluorescenceImage.create_multi_channel_image(images)


def acquire_image(
    microscope: FluorescenceMicroscope,
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: Optional[ZParameters] = None,
    stop_event: Optional[threading.Event] = None,
    filename: Optional[str] = None,
) -> Optional[FluorescenceImage]:
    """Acquire a fluroescence image for a single channel or multiple channels.
    If zparams is provided, a Z-stack will be acquired instead.
    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel or list of channels to acquire
        zparams: ZParameters for Z-stack acquisition (optional)
        stop_event: Threading event for cancellation (optional)
        filename: Full file path to save the image (optional)
    Returns:
            FluorescenceImage object containing the acquired image(s)"""
    if microscope.parent is None:
        raise ValueError("Microscope parent is not set. Cannot start acquisition.")

    if not microscope.has_valid_orientation():
        raise ValueError(f"Stage is not in valid orientation ({microscope.parent.get_stage_orientation()!r}). Cannot start acquisition.")

    if zparams is not None:
        # Acquire Z-stack if zparams is provided
        image = acquire_z_stack(microscope, channel_settings, zparams, stop_event)
    else:
        # Acquire single image(s) for specified channel(s)
        image = acquire_channels(microscope, channel_settings, stop_event)

    # Save image if filename is provided and acquisition was successful
    if image is not None and filename is not None:
        try:
            # Set description from filename (without extension)
            image.metadata.description = os.path.basename(filename).removesuffix(
                ".ome.tiff"
            )
            image.metadata.filename = filename
            image.save(filename)
        except Exception as e:
            logging.error(f"Failed to save image to {filename}: {e}")

    microscope.acquisition_progress_signal.emit({"state": "finished"})

    return image

def acquire_at_positions(
    microscope: 'FibsemMicroscope',
    positions: List[FMStagePosition],
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    zparams: Optional[ZParameters] = None,
    use_autofocus: bool = False,
    save_directory: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> List[FluorescenceImage]:
    """Acquire fluorescence images at specified FMStagePosition locations.
    This function moves both the stage and objective to each specified position and
    acquires images for the given channel settings. If zparams is provided, a Z-stack
    will be acquired at each position.
    Args:
        microscope: The fluorescence microscope instance
        positions: List of FMStagePosition objects defining where to acquire images
        channel_settings: Single channel or list of channels to acquire
        zparams: ZParameters for Z-stack acquisition (optional)
        use_autofocus: Whether to run autofocus at each position (default: False)
        save_directory: Directory to save images in. If provided,
                       creates subdirectories for each position (default: None)
        stop_event: Threading event to signal cancellation (optional)
    Returns:
        List of FluorescenceImage objects containing the acquired images
    Raises:
        ValueError: If positions is empty or contains invalid stage positions
    Example:
        >>> stage_pos = FibsemStagePosition(x=0, y=0, z=0, name="pos1")
        >>> fm_pos = FMStagePosition(name="Position-01", stage_position=stage_pos, objective_position=0.012)
        >>> positions = [fm_pos]
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                          emission_wavelength=450, power=0.5, exposure_time=0.1)
        >>> images = acquire_at_positions(microscope, positions, channel,
        ...                              save_directory="/data/experiment")
    """
    if microscope.fm is None:
        raise ValueError(
            "Fluorescence microscope not initialized in the FibsemMicroscope instance"
        )
    if not microscope.fm.has_valid_orientation():
        raise ValueError(f"Stage is not in valid orientation ({microscope.get_stage_orientation()!r}). Cannot start acquisition.")

    if not positions:
        raise ValueError("Positions list cannot be empty")
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    # Calculate time estimates for the entire multi-position acquisition
    time_estimates = estimate_positions_acquisition_time(
        channel_settings, len(positions), zparams, use_autofocus
    )
    total_estimated_time = time_estimates["total_time"]
    acquisition_start_time = time.time()

    # Emit initial acquisition progress signal
    microscope.fm.acquisition_progress_signal.emit({"state": "acquiring", 
                                                    "task": "multi-position",
                                                    "position": "null",
                                                    "current": 1,
                                                    "total": len(positions),
                                                    "estimated_total_time": total_estimated_time,
                                                    "estimated_remaining_time": total_estimated_time,})

    images: List[FluorescenceImage] = []
    for i, fm_pos in enumerate(positions):

        # Check for cancellation before each position
        if stop_event and stop_event.is_set():
            logging.info("Multi-position acquisition cancelled")
            return images

        logging.info(f"Acquiring at position {i + 1}/{len(positions)}: {fm_pos.name}")

        # Calculate remaining time estimate based on progress
        current_position = i + 1
        total_positions = len(positions)
        # Emit progress signal before starting position acquisition
        microscope.fm.acquisition_progress_signal.emit({"state": "acquiring", 
                                                        "task": "multi-position",
                                                       "position": fm_pos.name,
                                                       "current": current_position,
                                                       "total": total_positions,
                                                    #    "estimated_total_time": total_estimated_time,
                                                    #    "estimated_remaining_time": estimated_remaining_time,
                                                    #    "elapsed_time": elapsed_time,
                                                       })

        # Move stage to the saved stage position and objective position
        if microscope.get_stage_orientation(fm_pos.stage_position) not in ["SEM", "FM"]:
            raise ValueError(f"Stage Position {fm_pos.name} is not in valid orientation: {fm_pos.stage_position}")
        microscope.fm.acquisition_progress_signal.emit({"state": "moving", "task": "multi-position"})
        microscope.safe_absolute_stage_movement(fm_pos.stage_position)
        microscope.fm.objective.move_absolute(fm_pos.objective_position)

        # Run autofocus if requested
        if use_autofocus:
            result = run_autofocus(microscope.fm, channel_settings[0], stop_event=stop_event)
            if result is None:
                logging.info("Multi-position acquisition cancelled during autofocus")
                return images

        # create filename/path if requested
        filename = None
        if save_directory is not None:
            # Create position-specific subdirectory
            position_name = fm_pos.name or f"position_{i + 1:03d}"
            position_dir = os.path.join(save_directory, position_name)
            os.makedirs(position_dir, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = utils.current_timestamp_v3(timeonly=True)
            basename = f"{position_name}-zstack-{timestamp}.ome.tiff"
            filename = os.path.join(position_dir, basename)

        # Acquire image
        image = acquire_image(microscope=microscope.fm,
                              channel_settings=channel_settings,
                              zparams=zparams,
                              stop_event=stop_event,
                              filename=filename)

        # Check if acquisition was cancelled
        if image is None:
            logging.info("Multi-position acquisition cancelled during image acquisition")
            return images

        image.metadata.description = f"{fm_pos.name}-{image.metadata.acquisition_date}"
        images.append(image)

        # Calculate remaining time estimate based on progress
        elapsed_time = time.time() - acquisition_start_time
        if current_position > 1:
            time_per_position = elapsed_time / (current_position - 1)
            estimated_remaining_time = time_per_position * (total_positions - current_position)
        else:
            estimated_remaining_time = total_estimated_time

        microscope.fm.acquisition_progress_signal.emit({"state": "acquiring", 
                                                        "task": "multi-position",
                                                       "position": fm_pos.name,
                                                       "current": current_position,
                                                       "total": total_positions,
                                                       "estimated_total_time": total_estimated_time,
                                                       "estimated_remaining_time": estimated_remaining_time,
                                                       "elapsed_time": elapsed_time,})

    microscope.fm.acquisition_progress_signal.emit({"state": "finished"})

    return images

def run_tileset_autofocus(
    microscope: 'FibsemMicroscope',
    channel_settings: Optional[ChannelSettings],
    autofocus_settings: AutoFocusSettings,
    stop_event: Optional[threading.Event] = None,
) -> bool:
    """Run autofocus during tileset acquisition with error handling and logging.

    A thin wrapper over `run_coarse_fine_autofocus` -- the same sweep the live
    auto-focus button runs -- so a tileset focuses exactly the way focusing by hand
    does. It used to iterate passes itself, which made it a second implementation of
    a loop that already existed and could drift from it.

    Args:
        microscope: The FIBSEM microscope instance
        channel_settings: Channel settings for autofocus
        autofocus_settings: Sweep passes, method and channel. Every enabled pass runs,
            each centred on where the previous one left the objective.
        stop_event: Threading event to check for cancellation (optional)

    Returns:
        True if autofocus completed successfully, False if cancelled
    """
    if microscope.fm is None:
        logging.error(
            "Fluorescence microscope not initialized in the FibsemMicroscope instance"
        )
        return False
    try:
        result = run_coarse_fine_autofocus(
            microscope=microscope.fm,
            autofocus_settings=autofocus_settings,
            channel_settings=channel_settings,
            stop_event=stop_event,
        )
        if result is None:
            logging.info("Auto-focus cancelled")
            return False
        return True
    except Exception as e:
        logging.warning(f"Auto-focus failed: {e}")
        return False


PREVIEW_MAX_DIMENSION = 2048
"""Long-edge size of the live mosaic preview, in pixels.

A full-resolution fluorescence mosaic is multi-channel 16-bit -- a 5x5 of 1024px tiles
is ~88 MB -- and the whole canvas is pushed through a Qt signal on every tile. The
preview is decimated to this so watching a run stays cheap; the final stitch is
untouched and full resolution.
"""


def _to_channel_planes(data: np.ndarray, n_channels: int) -> np.ndarray:
    """Normalise tile data to (C, Y, X), projecting z the way the final stitch does.

    Tiles arrive as 2D (Y, X), 3D (either (C, Y, X) or (Z, Y, X)) or 4D (C, Z, Y, X).
    The 3D case is genuinely ambiguous from the shape alone, so it is resolved against
    the channel count -- a single channel with a z-stack is (Z, Y, X) and gets
    projected, several channels without one is (C, Y, X) and does not.
    """
    if data.ndim == 2:
        return data[np.newaxis]
    if data.ndim == 4:
        return np.max(data, axis=1)
    if data.ndim == 3:
        if n_channels > 1 and data.shape[0] == n_channels:
            return data
        return np.max(data, axis=0)[np.newaxis]
    raise ValueError(f"Unsupported tile data dimensions: {data.ndim}")


class FMTiledAcquisitionRunner:
    """Orchestrates a fluorescence tileset acquisition as a series of discrete phases.

    The fluorescence counterpart of `imaging.tiled.TiledAcquisitionRunner`, and
    deliberately the same shape: `run()` drives `_setup` -> `_compute_grid` ->
    once-only autofocus -> `_run_tile_loop`, restoring the starting position in a
    `finally`. The two differ only in what happens at each tile -- channels, z-stacks
    and objective autofocus here; beam settings and focus stacks there -- which is why
    they are separate runners over one shared geometry rather than a single class with
    a mode flag.

    State accumulated across phases lives on `self`, so partial results survive a
    cancellation: `run()` raises `OperationCancelledError`, and whatever was acquired
    up to that point remains on `.tileset`.
    """

    def __init__(
        self,
        microscope: 'FibsemMicroscope',
        channel_settings: Union[ChannelSettings, List[ChannelSettings]],
        overview_parameters: OverviewParameters,
        zparams: Optional[ZParameters] = None,
        autofocus_settings: Optional[AutoFocusSettings] = None,
        save_directory: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.microscope = microscope
        self.channel_settings = channel_settings
        self.overview_parameters = overview_parameters
        self.zparams = zparams
        self.autofocus_settings = autofocus_settings
        self.save_directory = save_directory
        self.stop_event = stop_event

        # Sized in _setup once the grid is known, and written by index. Every cell that
        # was not reached -- skipped by the mask, or not yet visited when a cancel came
        # in -- stays None, so a partial result keeps every acquired tile in its right
        # place instead of collapsing toward the origin.
        self.tileset: List[List[Optional[FluorescenceImage]]] = []

    # ── public entry points ──────────────────────────────────────────────

    def run(self) -> List[List[Optional[FluorescenceImage]]]:
        """Acquire every tile, returning them as [row][col].

        Raises:
            OperationCancelledError: if the stop event is set. Tiles acquired before
                that point remain available on `.tileset`.
        """
        self._setup()
        self._compute_grid()
        try:
            self._autofocus_if_mode(AutoFocusMode.ONCE)
            self._run_tile_loop()
            self._save_parameters()
        except OperationCancelledError:
            logging.info("Tileset acquisition cancelled")
            raise
        except Exception as e:
            logging.error(f"Error during tileset acquisition: {e}")
            raise
        finally:
            self._restore()

        logging.info(
            f"Tileset acquisition complete: {self._rows}x{self._cols} tiles acquired"
        )
        return self.tileset

    def run_and_stitch(self) -> FluorescenceImage:
        """Acquire every tile and return the stitched mosaic.

        The mosaic is an ordinary fluorescence image that happens to be large, and it
        reports where it is from what the run captured rather than from what the
        stitcher can infer: the grid is centred on the starting stage position, and
        every tile is taken at the starting objective position.
        """
        self.run()
        return stitch_tileset(
            self.tileset,
            self.overview_parameters.overlap,
            centre_position=self._initial_position,
            objective_position=self._initial_objective_position,
        )

    # ── phases ───────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Validate inputs, capture what has to be restored, and size the grid."""
        microscope = self.microscope

        if microscope.fm is None:
            raise ValueError(
                "Fluorescence microscope not initialized in the FibsemMicroscope instance"
            )

        orientation = microscope.get_stage_orientation()
        if orientation not in ["SEM", "FM"]:
            raise ValueError(
                f"Stage is not in SEM, or FM orientation {orientation}. Cannot start acquisition."
            )

        if not isinstance(self.channel_settings, list):
            self.channel_settings = [self.channel_settings]

        self._rows = self.overview_parameters.rows
        self._cols = self.overview_parameters.cols
        self._overlap = self.overview_parameters.overlap
        self._autofocus_mode = self.overview_parameters.autofocus_mode
        self._tile_order = self.overview_parameters.tile_order
        self._tile_mask = self.overview_parameters.tile_mask

        if self._rows <= 0 or self._cols <= 0:
            raise ValueError("Grid size must contain positive values")

        if not 0.0 <= self._overlap < 1.0:
            raise ValueError("Tile overlap must be between 0.0 and 1.0 (exclusive)")

        # A spiral revisits rows non-sequentially, so "autofocus on each new row" would
        # fire almost every tile and still leave stale focus in between. Promote it, as
        # the beam runner does.
        if (self._autofocus_mode is AutoFocusMode.EACH_ROW
                and self._tile_order is TileOrderStrategy.SPIRAL):
            self._autofocus_mode = AutoFocusMode.EACH_TILE
            logging.info("EACH_ROW autofocus upgraded to EACH_TILE for SPIRAL tile order")

        # Full grid, holes included: the canvas is the whole grid whatever the mask.
        self.tileset = [[None] * self._cols for _ in range(self._rows)]

        n_enabled = self.overview_parameters.n_enabled_tiles
        logging.info(
            f"Starting tileset acquisition: {self._rows}x{self._cols} grid "
            f"with {self._overlap:.1%} overlap, {self._tile_order.value} order, "
            f"{n_enabled}/{self._rows * self._cols} tiles enabled"
        )

        # Captured for the restore in _restore(), and as the grid's centre.
        self._initial_position = microscope.get_stage_position()
        self._initial_objective_position = microscope.fm.objective.position

        pixel_size_x, pixel_size_y = microscope.fm.camera.pixel_size
        self._image_width, self._image_height = microscope.fm.camera.resolution
        self._fov_x = self._image_width * pixel_size_x
        self._fov_y = self._image_height * pixel_size_y
        self._step_x = self._fov_x * (1.0 - self._overlap)
        self._step_y = self._fov_y * (1.0 - self._overlap)

        self._setup_autofocus()

        time_estimates = estimate_tileset_acquisition_time(
            self.channel_settings, (self._rows, self._cols), self.zparams,
            self._autofocus_mode, tile_mask=self._tile_mask,
        )
        self._total_estimated_time = time_estimates["total_time"]
        self._acquisition_start_time = time.time()

    def _setup_autofocus(self) -> None:
        """Resolve the autofocus channel, method and z-range once, up front."""
        self._autofocus_channel = None
        self._autofocus_method = None
        self._autofocus_zparams = None

        if self._autofocus_mode == AutoFocusMode.NONE:
            return

        if self.autofocus_settings is None:
            raise ValueError(
                f"Auto focus settings must be provided when autofocus mode "
                f"{self._autofocus_mode.value} is enabled"
            )

        if self.autofocus_settings.channel_name:
            for ch in self.channel_settings:
                if ch.name == self.autofocus_settings.channel_name:
                    self._autofocus_channel = ch
                    break

        if self._autofocus_channel is None:
            logging.warning(
                f"Autofocus channel '{self.autofocus_settings.channel_name}' not found, "
                f"using first channel"
            )
            self._autofocus_channel = self.channel_settings[0]

        enabled_passes = [p for p in self.autofocus_settings.passes if p.enabled]
        if not enabled_passes:
            raise ValueError("Auto-focus settings has no enabled passes")

        logging.info(f"Auto-focus mode: {self._autofocus_mode.value}")
        logging.info(
            f"Auto-focus settings: {self.autofocus_settings.method.value}, "
            f"{len(enabled_passes)} enabled pass(es)"
        )

        # The once-before-acquisition focus runs every enabled pass, coarse to fine:
        # each sweep centres on where the previous one left the objective, which is
        # the entire point of configuring more than one. Per-row and per-tile focus
        # runs only the narrowest -- they refine an already-good position, and paying
        # for a wide sweep at every tile would dominate the acquisition.
        #
        # Previously *all* modes used the narrowest pass alone, so a configured coarse
        # pass was silently ignored even for the case it exists for.
        self._autofocus_full = replace(self.autofocus_settings, passes=enabled_passes)
        self._autofocus_fine = replace(
            self.autofocus_settings, passes=[enabled_passes[-1]]
        )

    def _compute_grid(self) -> None:
        """Lay the grid out once and project every tile to an absolute position.

        Nothing moves here. Walking the grid with relative steps accumulated error
        over the traversal, and on an offset mount every step carries a real z
        component -- so the drift was in the focal plane, not just the position. A
        compustage hides that, having no pre-tilt and therefore z == 0 throughout.

        Using the shared layout also keeps the row direction out of the tile loop.
        Row 0 is the top of the mosaic and later rows step down; `stitch_tileset`
        paints row 0 at canvas y=0 regardless, so the two have to agree. They
        disagreed once already (#226).
        """
        self._grid = compute_tile_grid_from_fov(
            nrows=self._rows,
            ncols=self._cols,
            fov_x=self._fov_x,
            fov_y=self._fov_y,
            image_width=self._image_width,
            image_height=self._image_height,
            overlap=self._overlap,
            mask=self._tile_mask,
        )
        # Ordered, and disabled tiles dropped. The traversal comes from the full grid
        # extent, so a sparse run follows the dense path with stops missing rather than
        # a pattern re-derived over the holes -- see order_tiles.
        self._ordered = order_tiles(self._grid, self._tile_order)

        if not self._ordered:
            raise ValueError("Tile mask disables every tile; nothing to acquire.")

        # The grid measures from its top-left tile; shift so it is centred on where
        # the stage already is. Same convention as TiledAcquisitionRunner.
        grid_offset_x = (self._cols - 1) * self._step_x / 2
        grid_offset_y = (self._rows - 1) * self._step_y / 2

        self._tile_stage_positions = {
            (tile.row, tile.col): self.microscope.project_fm_stable_move(
                dx=tile.dx - grid_offset_x,
                dy=tile.dy + grid_offset_y,
                base_position=self._initial_position,
            )
            for tile in self._grid
        }

        self._init_preview_canvas()

        first = self._ordered[0]
        self._emit({"state": "moving", "task": "tileset"})
        self._emit({
            "state": "acquiring", "task": "tileset",
            "row": first.row + 1, "col": first.col + 1,
            "total_rows": self._rows, "total_cols": self._cols,
            # `total` counts tiles that will actually be acquired, not grid cells --
            # a progress bar that stops at 9/25 on a successful sparse run reads as a
            # failure.
            "current": 1, "total": len(self._ordered),
            "estimated_total_time": self._total_estimated_time,
            "estimated_remaining_time": self._total_estimated_time,
        })

    def _init_preview_canvas(self) -> None:
        """Allocate the live mosaic that tiles are painted into as they arrive.

        The beam tiler does the same thing (`TiledAcquisitionRunner._canvas`) and emits
        the whole canvas with each tile, leaving the widget to simply redisplay it. The
        difference here is size: a full-resolution fluorescence mosaic is multi-channel
        and 16-bit, so a 5x5 would be ~88 MB pushed through a Qt signal per tile. The
        preview is therefore decimated to `PREVIEW_MAX_DIMENSION` on its long edge --
        enough to watch the mosaic fill in, and the real stitch still happens at full
        resolution when the run finishes.
        """
        # Taken from the grid rather than recomputed, so the preview cannot drift from
        # where `stitch_tileset` will actually paint these tiles.
        full_w = max(t.canvas_x for t in self._grid) + self._image_width
        full_h = max(t.canvas_y for t in self._grid) + self._image_height

        self._preview_stride = max(
            1, int(np.ceil(max(full_w, full_h) / PREVIEW_MAX_DIMENSION))
        )
        self._preview_canvas = np.zeros(
            (len(self.channel_settings),
             int(np.ceil(full_h / self._preview_stride)),
             int(np.ceil(full_w / self._preview_stride))),
            dtype=np.uint16,
        )
        logging.debug(
            f"Live preview canvas: {self._preview_canvas.shape} "
            f"(stride {self._preview_stride}, full {full_h}x{full_w})"
        )

    def _paint_preview(self, tile: TilePosition, image: FluorescenceImage) -> None:
        """Paint one acquired tile into the live preview mosaic.

        Best effort: a preview that cannot be built is not a reason to fail an
        acquisition that is otherwise fine, so this never raises into the tile loop.
        """
        try:
            stride = self._preview_stride
            data = _to_channel_planes(image.data, len(self.channel_settings))
            if data.shape[0] != self._preview_canvas.shape[0]:
                data = data[: self._preview_canvas.shape[0]]

            thumb = data[:, ::stride, ::stride]
            y0 = tile.canvas_y // stride
            x0 = tile.canvas_x // stride
            h = min(thumb.shape[1], self._preview_canvas.shape[1] - y0)
            w = min(thumb.shape[2], self._preview_canvas.shape[2] - x0)
            if h > 0 and w > 0:
                self._preview_canvas[:, y0:y0 + h, x0:x0 + w] = thumb[:, :h, :w]
        except Exception as e:  # pragma: no cover - preview is never load-bearing
            logging.debug(f"Could not paint tile ({tile.row}, {tile.col}) into preview: {e}")

    def _autofocus_if_mode(self, mode: AutoFocusMode) -> None:
        """Run autofocus when the configured mode matches.

        Raises:
            OperationCancelledError: if autofocus was cancelled.
        """
        if self._autofocus_mode != mode:
            return

        settings = (
            self._autofocus_full if mode is AutoFocusMode.ONCE else self._autofocus_fine
        )
        if not run_tileset_autofocus(
            self.microscope,
            self._autofocus_channel,
            settings,
            stop_event=self.stop_event,
        ):
            raise OperationCancelledError(
                f"Tileset autofocus ({mode.value}) cancelled by user."
            )

    def _run_tile_loop(self) -> None:
        """Visit the enabled tiles in traversal order.

        Tiles are written into the pre-allocated `self.tileset` by index rather than
        appended. Appending assumed a row-major visit and a full grid -- neither holds
        once the order is configurable or tiles can be skipped, and the failure is
        silent: the mosaic would still stitch, with tiles in the wrong places.
        """
        prev_row = -1
        self._n_acquired = 0
        for tile in self._ordered:
            # Check before moving, so a cancel skips the travel entirely.
            raise_if_cancelled(self.stop_event, "Tileset acquisition cancelled by user.")

            if tile.row != prev_row:
                prev_row = tile.row
                self._autofocus_if_mode(AutoFocusMode.EACH_ROW)

            image = self._acquire_tile(tile.row, tile.col)
            self.tileset[tile.row][tile.col] = image
            self._n_acquired += 1
            self._paint_preview(tile, image)
            self._emit_preview(tile)

            if tile is not self._ordered[-1]:
                self._emit({"state": "moving", "task": "tileset"})

    def _acquire_tile(self, row: int, col: int) -> FluorescenceImage:
        """Move to one tile and acquire it.

        Raises:
            OperationCancelledError: if cancelled during autofocus or acquisition.
        """
        microscope = self.microscope

        # Absolute, from the grid computed up front -- no accumulation.
        microscope.safe_absolute_stage_movement(self._tile_stage_positions[(row, col)])
        microscope.fm.objective.move_absolute(self._initial_objective_position)

        self._autofocus_if_mode(AutoFocusMode.EACH_TILE)

        logging.info(f"Acquiring tile [{row + 1}/{self._rows}][{col + 1}/{self._cols}]")
        self._emit_tile_progress(row, col)

        filename = None
        if self.save_directory is not None:
            filename = os.path.join(
                self.save_directory, f"tile-{row:02d}-{col:02d}.ome.tiff"
            )

        tile_image = acquire_image(
            microscope=microscope.fm,
            channel_settings=self.channel_settings,
            zparams=self.zparams,
            stop_event=self.stop_event,
            filename=filename,
        )

        # acquire_image returns None only when it was cancelled part-way.
        if tile_image is None:
            raise OperationCancelledError(
                "Tileset acquisition cancelled by user during tile acquisition."
            )

        if self.zparams is not None:
            # TODO(FIB-394): make the projection method selectable; focus_stack()
            # already exists and is simply not reachable from here.
            tile_image = tile_image.max_intensity_projection()

        return tile_image

    def _save_parameters(self) -> None:
        """Write the acquisition parameters alongside the tiles."""
        if self.save_directory is None:
            return

        params = {
            "grid_size": (self._rows, self._cols),
            "tile_overlap": self._overlap,
            "overview_parameters": self.overview_parameters.to_dict(),
            "zparameters": self.zparams.to_dict() if self.zparams is not None else None,
            "autofocus_settings": (
                self.autofocus_settings.to_dict()
                if self.autofocus_settings is not None else None
            ),
            "channel_settings": [ch.to_dict() for ch in self.channel_settings],
        }
        params_filename = os.path.join(self.save_directory, "overview-parameters.json")
        utils.save_json(params_filename, params)
        logging.info(f"Saved tileset acquisition parameters to {params_filename}")

    def _restore(self) -> None:
        """Put the stage and objective back where they started."""
        logging.info("Returning to initial position")
        self.microscope.safe_absolute_stage_movement(self._initial_position)
        self.microscope.fm.objective.move_absolute(self._initial_objective_position)
        self._emit({"state": "finished"})

    # ── helpers ──────────────────────────────────────────────────────────

    def _emit(self, payload: dict) -> None:
        self.microscope.fm.acquisition_progress_signal.emit(payload)

    def _emit_preview(self, tile: TilePosition) -> None:
        """Publish the mosaic-so-far, so a viewer can watch it fill in.

        Keyed `image`, matching what `TiledAcquisitionRunner` emits and what
        `FibsemMinimapWidget.handle_tile_acquisition_progress` already reads, so a
        consumer of one reads the other. The whole canvas goes out rather than the
        single tile: the receiver then needs no state of its own and simply redisplays
        what it is given, which is also what makes a late subscriber correct.

        A *copy* goes out, unlike the beam tiler, which emits its live canvas directly.
        The acquisition runs on a worker thread, so the signal is queued and the slot
        runs later on the GUI thread -- by which time the shared array has been painted
        into again. Sending the buffer itself is a read/write race for the sake of the
        one thing that does not need to be fast: at ~10 MB against a tile exposure, the
        copy does not show.
        """
        self._emit({
            "state": "tile", "task": "tileset",
            "row": tile.row, "col": tile.col,
            "total_rows": self._rows, "total_cols": self._cols,
            "current": self._n_acquired, "total": len(self._ordered),
            "image": self._preview_canvas.copy(),
            "preview_stride": self._preview_stride,
        })

    def _emit_tile_progress(self, row: int, col: int) -> None:
        # Counted by visits, not by grid index. `row * cols + col` assumed a full
        # row-major traversal; under a spiral it jumps around, and under a mask it
        # overcounts by every skipped tile.
        current_tile = self._n_acquired + 1
        total_tiles = len(self._ordered)
        elapsed_time = time.time() - self._acquisition_start_time

        if current_tile > 1:
            time_per_tile = elapsed_time / (current_tile - 1)
            estimated_remaining_time = time_per_tile * (total_tiles - current_tile + 1)
        else:
            estimated_remaining_time = self._total_estimated_time

        self._emit({
            "state": "acquiring", "task": "tileset",
            "row": row + 1, "col": col + 1,
            "total_rows": self._rows, "total_cols": self._cols,
            "current": current_tile, "total": total_tiles,
            "estimated_total_time": self._total_estimated_time,
            "estimated_remaining_time": estimated_remaining_time,
            "elapsed_time": elapsed_time,
        })


def acquire_tileset(
    microscope: 'FibsemMicroscope',
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    overview_parameters: OverviewParameters,
    zparams: Optional[ZParameters] = None,
    beam_type: BeamType = BeamType.ELECTRON,
    autofocus_settings: Optional[AutoFocusSettings] = None,
    save_directory: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> List[List[Optional[FluorescenceImage]]]:
    """Acquire a tileset of fluorescence images across a grid pattern.

    Thin wrapper over :class:`FMTiledAcquisitionRunner`. Use the runner directly when
    you need the partial tiles after a cancellation, or want to hold the object to
    poll and stop it -- which is what the acquisition UI wants.

    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel or list of channels to acquire
        overview_parameters: Overview parameters containing grid size, overlap, autofocus mode
        zparams: Optional Z parameters for z-stack acquisition (overrides overview_parameters.use_zstack)
        beam_type: Unused for stage movement. Tiles are projected through the camera's
            own axis tilt; kept for signature compatibility with callers that still pass it.
        autofocus_settings: Optional AutoFocusSettings for autofocus configuration
        save_directory: Optional directory path to save individual tile images (default: None)
        stop_event: Threading event to signal cancellation (optional)

    Returns:
        List of lists containing FluorescenceImage objects organized as [row][col]

    Raises:
        ValueError: If grid_size contains non-positive values or overlap is invalid
        OperationCancelledError: If the acquisition was cancelled by the user.

    Example:
        >>> # Acquire a 3x3 grid with 10% overlap
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                          emission_wavelength=450, power=50, exposure_time=0.1)
        >>> tileset = acquire_tileset(microscope, channel, grid_size=(3, 3), tile_overlap=0.1)
        >>> print(f"Acquired {len(tileset)}x{len(tileset[0])} tiles")
    """
    return FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=channel_settings,
        overview_parameters=overview_parameters,
        zparams=zparams,
        autofocus_settings=autofocus_settings,
        save_directory=save_directory,
        stop_event=stop_event,
    ).run()


def stitch_tileset(
    tileset: List[List[Optional[FluorescenceImage]]],
    tile_overlap: float,
    centre_position: FibsemStagePosition,
    objective_position: Optional[float] = None,
) -> FluorescenceImage:
    """Stitch a tileset of fluorescence images into a single mosaic image.

    Supports both single-channel and multi-channel images. For multi-channel images,
    each channel is stitched separately and combined into a single mosaic with
    dimensions (nc_channel, 1, ny, nx). Overlapping regions are handled by taking
    pixels from the rightmost/bottommost tile.

    The result is an ordinary fluorescence image that happens to be large: the same
    metadata a single acquisition carries, describing the whole mosaic. Nothing about
    it is tileset-specific, so anything that consumes an FM image consumes this.

    Args:
        tileset: List of lists containing FluorescenceImage objects [row][col].
            `None` entries are tiles that were not acquired; they are left as canvas
            zeros.
        tile_overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)
        centre_position: Stage position of the mosaic centre. Required: where the
            mosaic sits is what places it on the canvas, so it has to be known rather
            than inferred. The grid is laid out centred on wherever the stage was when
            the run started, so the runner passes that position through and it is exact
            by construction. Deriving it instead -- by averaging the acquired tiles --
            is only right when they are symmetric about the centre: true of a full
            grid, false of a masked or cancelled one, and out by up to a whole tile.
        objective_position: Objective z for the mosaic, recorded on every channel.
            Every tile is acquired at the run's initial objective position, so that is
            the mosaic's. Without it the first acquired tile's value is kept.

    Returns:
        Single FluorescenceImage containing the stitched mosaic

    Raises:
        ValueError: If tileset is empty or irregular

    Example:
        >>> tileset = acquire_tileset(microscope, channel, grid_size=(3, 3))
        >>> mosaic = stitch_tileset(tileset, tile_overlap=0.1)
        >>> print(f"Mosaic size: {mosaic.data.shape}")
    """
    if not tileset or not tileset[0]:
        raise ValueError("Tileset cannot be empty")

    rows = len(tileset)
    cols = len(tileset[0])

    # Validate tileset is rectangular
    for row in tileset:
        if len(row) != cols:
            raise ValueError("Tileset must be rectangular (all rows same length)")

    # Gaps are expected, not exceptional: a sparse acquisition leaves every skipped
    # tile as None, and a cancelled one leaves everything it never reached. Both stitch
    # into a full-size canvas with the missing tiles left as zeros, so acquired tiles
    # keep the canvas coordinates they would have had in a dense mosaic.
    acquired = [t for row in tileset for t in row if t is not None]
    if not acquired:
        raise ValueError("Tileset contains no acquired tiles")

    logging.info(
        f"Stitching {rows}x{cols} tileset with {tile_overlap:.1%} overlap "
        f"({len(acquired)}/{rows * cols} tiles acquired)"
    )

    # Get reference tile for dimensions -- the first acquired one, which is not
    # necessarily (0, 0).
    ref_tile = acquired[0]

    # Handle both 2D and multi-dimensional data
    if ref_tile.data.ndim == 2:
        # 2D data (Y, X)
        tile_height, tile_width = ref_tile.data.shape
        nc_channels = 1
        nz_planes = 1
    elif ref_tile.data.ndim == 3:
        # 3D data (Z, Y, X) or (C, Y, X)
        nz_planes, tile_height, tile_width = ref_tile.data.shape
        nc_channels = 1
    elif ref_tile.data.ndim == 4:
        # 4D data (C, Z, Y, X)
        nc_channels, nz_planes, tile_height, tile_width = ref_tile.data.shape
    else:
        raise ValueError(f"Unsupported data dimensions: {ref_tile.data.ndim}")

    # Calculate overlap in pixels
    overlap_pixels_x = int(tile_width * tile_overlap)
    overlap_pixels_y = int(tile_height * tile_overlap)

    # Calculate final mosaic dimensions
    effective_tile_width = tile_width - overlap_pixels_x
    effective_tile_height = tile_height - overlap_pixels_y

    mosaic_width = effective_tile_width * (cols - 1) + tile_width
    mosaic_height = effective_tile_height * (rows - 1) + tile_height

    logging.info(f"Tile size: {tile_height}x{tile_width}, Mosaic size: {mosaic_height}x{mosaic_width}")
    logging.info(f"Channels: {nc_channels}, Z-planes: {nz_planes}")

    # Initialize mosaic array with proper dimensions (C, Z, Y, X)
    mosaic_data = np.zeros(
        (nc_channels, 1, mosaic_height, mosaic_width), dtype=ref_tile.data.dtype
    )

    # Place each tile
    for row in range(rows):
        for col in range(cols):
            tile = tileset[row][col]
            if tile is None:
                continue  # not acquired: leave the canvas zeros

            # Calculate position in mosaic
            y_start = row * effective_tile_height
            x_start = col * effective_tile_width
            y_end = y_start + tile_height
            x_end = x_start + tile_width

            # Ensure we don't exceed mosaic boundaries
            y_end = min(y_end, mosaic_height)
            x_end = min(x_end, mosaic_width)

            # Calculate actual tile region to use
            tile_y_end = y_end - y_start
            tile_x_end = x_end - x_start

            # Normalize tile data to CZYX format for consistent processing
            if tile.data.ndim == 2:
                # 2D data (Y, X) -> (1, 1, Y, X)
                tile_data = tile.data[np.newaxis, np.newaxis, :tile_y_end, :tile_x_end]
            elif tile.data.ndim == 3:
                # 3D data (Z, Y, X) -> (1, Z, Y, X) or (C, Y, X) -> (C, 1, Y, X)
                if nz_planes > 1:  # Z-stack
                    tile_data = tile.data[np.newaxis, :, :tile_y_end, :tile_x_end]
                else:  # Multi-channel
                    tile_data = tile.data[:, np.newaxis, :tile_y_end, :tile_x_end]
            elif tile.data.ndim == 4:
                # 4D data (C, Z, Y, X) - already in correct format
                tile_data = tile.data[:, :, :tile_y_end, :tile_x_end]
            else:
                raise ValueError(f"Unsupported tile data dimensions: {tile.data.ndim}")

            # For multi-channel and/or z-stack, we take max intensity projection along Z
            # to create the final mosaic with dimensions (C, 1, Y, X)
            if tile_data.shape[1] > 1:  # Multiple Z planes
                tile_data = np.max(tile_data, axis=1, keepdims=True)

            # Place tile data for each channel
            mosaic_data[:, 0, y_start:y_end, x_start:x_end] = tile_data[:, 0, :, :]

    # Create updated metadata for stitched image
    stitched_metadata = deepcopy(ref_tile.metadata)

    # Update resolution to reflect new mosaic size
    stitched_metadata.resolution = (mosaic_width, mosaic_height)

    # Where the mosaic is. The grid is laid out centred on wherever the stage was when
    # the run started, so that position *is* the mosaic centre -- exact, and independent
    # of which tiles were acquired.
    stitched_metadata.stage_position = deepcopy(centre_position)
    stitched_metadata.stage_position.name = f"stitched_mosaic_{rows}x{cols}"

    # Every tile is acquired at the run's initial objective position, so it is the
    # mosaic's too. Recorded per channel, as it is on a single acquisition.
    if objective_position is not None:
        for channel in stitched_metadata.channels:
            channel.objective_position = objective_position

    # Create stitched FluorescenceImage
    stitched_image = FluorescenceImage(data=mosaic_data, metadata=stitched_metadata)

    logging.info(f"Stitching complete: {mosaic_height}x{mosaic_width} mosaic created")
    return stitched_image

def acquire_and_stitch_tileset(
    microscope: 'FibsemMicroscope',
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    overview_parameters: OverviewParameters,
    zparams: Optional[ZParameters] = None,
    beam_type: BeamType = BeamType.ELECTRON,
    autofocus_settings: Optional[AutoFocusSettings] = None,
    save_directory: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> Optional[FluorescenceImage]:
    """Acquire a tileset and stitch it into a single mosaic image.

    Args:
        microscope: The fluorescence microscope instance
        channel_settings: Single channel or list of channels to acquire
        overview_parameters: Overview parameters containing grid size, overlap, autofocus mode
        zparams: Optional Z parameters for z-stack acquisition (overrides overview_parameters.use_zstack)
        beam_type: Unused for stage movement. Tiles are stepped with fm_stable_move,
            which projects through the camera's own axis tilt; kept for signature
            compatibility with callers that still pass it.
        autofocus_settings: Optional AutoFocusSettings for autofocus configuration
        save_directory: Optional directory path to save individual tile images (default: None)
        stop_event: Optional threading event to signal cancellation

    Returns:
        Single stitched FluorescenceImage with dimensions (nc_channel, 1, ny, nx)
    """

    # TODO: support different projection methods (e.g. max intensity, focus stacking)

    # Auto-convert channel_settings to list for consistent processing
    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    # Create timestamp and subdirectory for tiles
    if save_directory is not None:
        timestamp = utils.current_timestamp_v3(timeonly=True)
        basename = f"overview-{timestamp}"
        tiles_directory = os.path.join(save_directory, basename)
        os.makedirs(tiles_directory, exist_ok=True)
    else:
        tiles_directory = None
    
    # Check if zparams is provided when z-stack is requested
    if zparams is None and overview_parameters.use_zstack:
        raise ValueError("Z-stack requested in overview parameters but no zparams provided")

    # acquire the tileset
    # A cancel is now signalled explicitly rather than inferred from the result. The
    # previous check -- empty tileset, or any tile None -- could not tell a user
    # cancel from a tile that genuinely failed, and reported both the same way.
    try:
        # Driven through the runner rather than the `acquire_tileset` wrapper, so the
        # stitched mosaic can be given the position and objective the run captured
        # instead of values inferred from whichever tiles came back.
        overview_image = FMTiledAcquisitionRunner(
            microscope=microscope,
            channel_settings=channel_settings,
            overview_parameters=overview_parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            save_directory=tiles_directory,
            stop_event=stop_event,
        ).run_and_stitch()
    except OperationCancelledError:
        logging.info("Tileset acquisition was cancelled, cannot stitch")
        return None

    # Save overview to experiment directory
    if save_directory is not None:
        filepath = os.path.join(save_directory, f"{basename}.ome.tiff")
        overview_image.metadata.description = basename

        try:
            overview_image.save(filepath)
            logging.info(f"Overview saved to: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save overview to {filepath}: {e}")

    return overview_image


def acquire_multiple_overviews(
    microscope: 'FibsemMicroscope',
    positions: List[FMStagePosition],
    channel_settings: Union[ChannelSettings, List[ChannelSettings]],
    overview_parameters: OverviewParameters,
    zparams: Optional[ZParameters] = None,
    beam_type: BeamType = BeamType.ELECTRON,
    autofocus_settings: Optional[AutoFocusSettings] = None,
    save_directory: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> List[Optional[FluorescenceImage]]:
    """Acquire multiple overview images at different fluorescence microscopy positions.

    This function moves to each specified FMStagePosition (both stage and objective)
    and acquires a stitched overview image using the same parameters as
    acquire_and_stitch_tileset.

    Args:
        microscope: The FIBSEM microscope instance with fluorescence microscope
        positions: List of FMStagePosition objects defining where to acquire overviews
        channel_settings: Single channel or list of channels to acquire
        overview_parameters: Overview parameters containing grid size, overlap, autofocus mode
        zparams: Optional Z parameters for z-stack acquisition (overrides overview_parameters.use_zstack)
        beam_type: Unused for stage movement. Tiles are stepped with fm_stable_move,
            which projects through the camera's own axis tilt; kept for signature
            compatibility with callers that still pass it.
        autofocus_settings: Optional AutoFocusSettings for autofocus configuration
        save_directory: Optional directory path to save overview images. Creates subdirectories
                       for each position (default: None)
        stop_event: Threading event to signal cancellation (optional)

    Returns:
        List of stitched FluorescenceImage objects (or None if acquisition was cancelled)
        corresponding to each position

    Raises:
        ValueError: If positions is empty or microscope.fm is None

    Example:
        >>> # Define positions
        >>> pos1 = FMStagePosition(name="Region1", stage_position=stage_pos1, objective_position=0.012)
        >>> pos2 = FMStagePosition(name="Region2", stage_position=stage_pos2, objective_position=0.013)
        >>> positions = [pos1, pos2]
        >>>
        >>> # Define channel settings and overview parameters  
        >>> channel = ChannelSettings(name="DAPI", excitation_wavelength=365,
        ...                          emission_wavelength=450, power=50, exposure_time=0.1)
        >>> overview_params = OverviewParameters(rows=3, cols=3, overlap=0.1)
        >>>
        >>> # Acquire 3x3 overview grids at each position
        >>> overviews = acquire_multiple_overviews(microscope, positions, channel, overview_params,
        ...                                       save_directory="/data/experiment")
        >>> print(f"Acquired {len(overviews)} overview images")
    """
    if microscope.fm is None:
        raise ValueError(
            "Fluorescence microscope not initialized in the FibsemMicroscope instance"
        )
    if not positions:
        raise ValueError("Positions list cannot be empty")

    # Store initial positions to restore later
    initial_stage_position = microscope.get_stage_position()
    initial_objective_position = microscope.fm.objective.position

    overview_images: List[Optional[FluorescenceImage]] = []

    try:
        for i, fm_pos in enumerate(positions):
            # Check for cancellation before each position
            if stop_event and stop_event.is_set():
                logging.info("Multiple overview acquisition cancelled")
                return overview_images  # Return images acquired so far

            logging.info(
                f"Acquiring overview at position {i + 1}/{len(positions)}: {fm_pos.name}"
            )

            # Move stage to the specified position
            logging.info(f"Moving stage to: {fm_pos.stage_position}")
            microscope.safe_absolute_stage_movement(fm_pos.stage_position)

            # Move objective to the specified position
            logging.info(
                f"Moving objective to: {fm_pos.objective_position * 1e3:.2f} mm"
            )
            microscope.fm.objective.move_absolute(fm_pos.objective_position)

            # Create position-specific save directory if requested
            position_save_directory = None
            if save_directory is not None:
                position_name = fm_pos.name or f"position_{i + 1:03d}"
                position_save_directory = os.path.join(save_directory, position_name)
                os.makedirs(position_save_directory, exist_ok=True)

            # Acquire overview at current position
            overview_image = acquire_and_stitch_tileset(
                microscope=microscope,
                channel_settings=channel_settings,
                overview_parameters=overview_parameters,
                zparams=zparams,
                beam_type=beam_type,
                autofocus_settings=autofocus_settings,
                save_directory=position_save_directory,
                stop_event=stop_event,
            )

            # Check if acquisition was cancelled
            if overview_image is None:
                logging.info("Overview acquisition cancelled")
                overview_images.append(None)
                return overview_images  # Return images acquired so far

            # Update metadata with position information
            if overview_image is not None:
                overview_image.metadata.description = (
                    f"{fm_pos.name}-overview-{overview_image.metadata.acquisition_date}"
                )
                logging.info(f"Acquired overview at position: {fm_pos.name}")

            overview_images.append(overview_image)

    except Exception as e:
        logging.error(f"Error during multiple overview acquisition: {e}")
        raise

    finally:
        # Return to initial positions
        logging.info("Returning to initial positions")
        microscope.safe_absolute_stage_movement(initial_stage_position)
        microscope.fm.objective.move_absolute(initial_objective_position)

    logging.info(
        f"Multiple overview acquisition complete: {len(overview_images)} overviews acquired"
    )
    return overview_images


def generate_grid_positions(
    ncols: int, nrows: int, fov_x: float, fov_y: float, overlap: float = 0.1
) -> List[Tuple[float, float]]:
    """Generate a grid of positions, centered around the origin, for acquiring tiles.

    Creates a regular grid of (x, y) positions that are properly centered around the origin
    (0, 0) for both odd and even numbers of columns and rows. The spacing between positions
    accounts for the specified field of view and tile overlap.

    Args:
        ncols: Number of columns in the grid (must be positive)
        nrows: Number of rows in the grid (must be positive)
        fov_x: Horizontal field of view size in meters (physical dimension of each tile)
        fov_y: Vertical field of view size in meters (physical dimension of each tile)
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)

    Returns:
        List of (x, y) tuples representing grid positions in meters, centered around origin

    Example:
        >>> # 3x3 grid with 10μm x 8μm FOV and 10% overlap
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> len(positions)
        9
        >>> positions[4]  # Center position
        (0.0, 0.0)

        >>> # 4x4 grid is also properly centered
        >>> positions = generate_grid_positions(4, 4, 10e-6, 10e-6, 0.0)
        >>> import numpy as np
        >>> np.mean([pos[0] for pos in positions])  # Mean x should be ~0
        0.0
    """
    positions = []
    for i in range(ncols):
        for j in range(nrows):
            x = (i - (ncols - 1) / 2) * (fov_x * (1 - overlap))
            # Negated so that row 0 is the top of the mosaic, matching
            # `acquire_tileset`'s stepping and `TilePosition` in imaging/tiled.py.
            # The grid is symmetric about zero, so this changes the row ordering
            # without changing the set of positions visited.
            y = -(j - (nrows - 1) / 2) * (fov_y * (1 - overlap))
            positions.append((x, y))

    return positions


def convert_grid_positions_to_stage_positions(
    microscope: 'FibsemMicroscope',
    positions: List[Tuple[float, float]],
    beam_type: BeamType = BeamType.ELECTRON,
    base_position: Optional[FibsemStagePosition] = None,
) -> List[FibsemStagePosition]:
    """Convert grid positions to stage positions using microscope projection.

    Takes a list of (x, y) grid positions and converts them to FibsemStagePosition
    objects using the microscope's project_stable_move method. This accounts for
    the microscope's coordinate system and current stage configuration.

    Args:
        microscope: The FibsemMicroscope instance to use for projection
        positions: List of (x, y) tuples representing grid positions in meters
        beam_type: Beam type to use for projection (default: ELECTRON)
        base_position: Base stage position to project from (default: current position)

    Returns:
        List of FibsemStagePosition objects representing the projected stage positions

    Example:
        >>> # Generate grid positions
        >>> positions = generate_grid_positions(3, 3, 10e-6, 0.1)
        >>> # Convert to stage positions
        >>> stage_positions = convert_grid_positions_to_stage_positions(
        ...     microscope, positions, BeamType.ELECTRON
        ... )
        >>> len(stage_positions)
        9
    """
    if base_position is None:
        base_position = microscope.get_stage_position()

    stage_positions = []
    for pos in positions:
        x, y = pos
        stage_position = microscope.project_stable_move(
            dx=x, dy=y, beam_type=beam_type, base_position=base_position
        )
        stage_positions.append(stage_position)
    return stage_positions


def calculate_grid_size_for_area(
    area_width: float,
    area_height: float,
    fov_x: float,
    fov_y: float,
    overlap: float = 0.1,
) -> Tuple[int, int]:
    """Calculate the number of rows and columns needed to cover a given area.

    Determines the minimum grid dimensions required to fully cover a rectangular area
    with the specified field of view and overlap between adjacent tiles.

    Args:
        area_width: Width of the area to cover in meters
        area_height: Height of the area to cover in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)

    Returns:
        Tuple of (ncols, nrows) representing the minimum grid dimensions needed

    Raises:
        ValueError: If area dimensions, FOV, or overlap are invalid

    Example:
        >>> # Cover a 100x80 μm area with 10x8 μm FOV and 10% overlap
        >>> ncols, nrows = calculate_grid_size_for_area(100e-6, 80e-6, 10e-6, 8e-6, 0.1)
        >>> print(f"Need {ncols}x{nrows} grid")
        Need 12x11 grid

        >>> # Cover a 50x50 μm area with 20x20 μm FOV and no overlap
        >>> ncols, nrows = calculate_grid_size_for_area(50e-6, 50e-6, 20e-6, 20e-6, 0.0)
        >>> print(f"Need {ncols}x{nrows} grid")
        Need 3x3 grid
    """
    # Validate inputs
    if area_width <= 0 or area_height <= 0:
        raise ValueError("Area dimensions must be positive")
    if fov_x <= 0 or fov_y <= 0:
        raise ValueError("FOV dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be between 0.0 and 1.0 (exclusive)")

    # Calculate effective step size (distance between tile centers)
    step_x = fov_x * (1.0 - overlap)
    step_y = fov_y * (1.0 - overlap)

    # Calculate number of tiles needed
    # For n tiles, we need: (n-1) * step + fov >= area
    # Solving for n: n >= (area - fov) / step + 1

    if area_width <= fov_x:
        # Area fits in single tile horizontally
        ncols = 1
    else:
        # Need multiple tiles
        ncols = int(np.ceil((area_width - fov_x) / step_x)) + 1

    if area_height <= fov_y:
        # Area fits in single tile vertically
        nrows = 1
    else:
        # Need multiple tiles
        nrows = int(np.ceil((area_height - fov_y) / step_y)) + 1

    return ncols, nrows


def calculate_grid_coverage_area(
    ncols: int, nrows: int, fov_x: float, fov_y: float, overlap: float = 0.1
) -> Tuple[float, float]:
    """Calculate the total area covered by a grid of tiles.

    Determines the total width and height of the area covered by a grid of tiles
    with specified dimensions, field of view, and overlap between adjacent tiles.

    Args:
        ncols: Number of columns in the grid (must be positive)
        nrows: Number of rows in the grid (must be positive)
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)

    Returns:
        Tuple of (total_width, total_height) in meters representing the covered area

    Raises:
        ValueError: If grid dimensions, FOV, or overlap are invalid

    Example:
        >>> # Calculate area covered by 3x4 grid with 10x8 μm FOV and 10% overlap
        >>> width, height = calculate_grid_coverage_area(3, 4, 10e-6, 8e-6, 0.1)
        >>> print(f"Covers {width*1e6:.1f}x{height*1e6:.1f} μm")
        Covers 28.0x30.4 μm

        >>> # Calculate area covered by 2x2 grid with 20x20 μm FOV and no overlap
        >>> width, height = calculate_grid_coverage_area(2, 2, 20e-6, 20e-6, 0.0)
        >>> print(f"Covers {width*1e6:.1f}x{height*1e6:.1f} μm")
        Covers 40.0x40.0 μm
    """
    # Validate inputs
    if ncols <= 0 or nrows <= 0:
        raise ValueError("Grid dimensions must be positive")
    if fov_x <= 0 or fov_y <= 0:
        raise ValueError("FOV dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be between 0.0 and 1.0 (exclusive)")

    # Calculate step size (distance between tile centers)
    step_x = fov_x * (1.0 - overlap)
    step_y = fov_y * (1.0 - overlap)

    # Calculate total coverage area
    # For n tiles: total_coverage = (n-1) * step + fov
    if ncols == 1:
        total_width = fov_x
    else:
        total_width = (ncols - 1) * step_x + fov_x

    if nrows == 1:
        total_height = fov_y
    else:
        total_height = (nrows - 1) * step_y + fov_y

    return total_width, total_height


def calculate_grid_dimensions(positions: List[Tuple[float, float]]) -> Tuple[int, int]:
    """Calculate the number of rows and columns from grid positions.

    Analyzes the grid positions to determine the number of unique rows and columns
    in the grid layout. Works with regular grids where positions are arranged in
    a rectangular pattern.

    Args:
        positions: List of (x, y) tuples representing grid positions in meters

    Returns:
        Tuple of (ncols, nrows) representing the grid dimensions
        Returns (0, 0) if positions is empty

    Example:
        >>> positions = generate_grid_positions(3, 4, 10e-6, 8e-6, 0.1)
        >>> ncols, nrows = calculate_grid_dimensions(positions)
        >>> print(f"Grid: {ncols}x{nrows}")
        Grid: 3x4
    """
    if not positions:
        return 0, 0

    # Extract unique x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Count unique coordinates with tolerance for floating point precision
    tolerance = 1e-10

    # Find unique x coordinates (columns)
    unique_x = []
    for x in x_coords:
        if not any(abs(x - ux) < tolerance for ux in unique_x):
            unique_x.append(x)

    # Find unique y coordinates (rows)
    unique_y = []
    for y in y_coords:
        if not any(abs(y - uy) < tolerance for uy in unique_y):
            unique_y.append(y)

    ncols = len(unique_x)
    nrows = len(unique_y)

    return ncols, nrows


def calculate_grid_overlap(
    positions: List[Tuple[float, float]], fov_x: float, fov_y: float
) -> Tuple[float, float]:
    """Calculate the overlap between grid positions given the FOV dimensions.

    Analyzes the spacing between adjacent grid positions to determine the overlap
    fraction in both horizontal and vertical directions.

    Args:
        positions: List of (x, y) tuples representing grid positions in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters

    Returns:
        Tuple of (horizontal_overlap, vertical_overlap) as fractions (0.0 to 1.0)
        Returns (0.0, 0.0) if overlap cannot be determined

    Example:
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> overlap_x, overlap_y = calculate_grid_overlap(positions, 10e-6, 8e-6)
        >>> print(f"Horizontal overlap: {overlap_x:.1%}, Vertical overlap: {overlap_y:.1%}")
        Horizontal overlap: 10.0%, Vertical overlap: 10.0%
    """
    if len(positions) < 2:
        return 0.0, 0.0

    # We'll analyze all pairs of positions to find minimum steps

    # Find minimum horizontal and vertical step distances
    min_x_step = float("inf")
    min_y_step = float("inf")

    # Check all pairs of positions to find minimum steps
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i >= j:
                continue

            x1, y1 = pos1
            x2, y2 = pos2

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Look for horizontal steps (same y, different x)
            if abs(dy) < 1e-10 and dx > 1e-10:  # Same y coordinate
                min_x_step = min(min_x_step, dx)

            # Look for vertical steps (same x, different y)
            if abs(dx) < 1e-10 and dy > 1e-10:  # Same x coordinate
                min_y_step = min(min_y_step, dy)

    # Calculate overlaps
    overlap_x = 0.0
    overlap_y = 0.0

    if min_x_step != float("inf") and min_x_step > 0:
        overlap_x = max(0.0, min(1.0, (fov_x - min_x_step) / fov_x))

    if min_y_step != float("inf") and min_y_step > 0:
        overlap_y = max(0.0, min(1.0, (fov_y - min_y_step) / fov_y))

    return overlap_x, overlap_y


def plot_grid_positions(
    positions: List[Tuple[float, float]],
    fov_x: float,
    fov_y: float,
    title: str = "Grid Positions with FOV",
    figsize: Tuple[float, float] = (8, 8),
    show_fov_boxes: bool = True,
    show_grid_lines: bool = True,
    show_center_lines: bool = True,
    show_overlap_info: bool = True,
) -> None:
    """Plot grid positions with field of view bounding boxes.

    Creates a visualization of the grid positions showing:
    - Grid positions as red circles
    - FOV bounding boxes as dashed rectangles around each position
    - Center lines (optional)
    - Grid lines (optional)
    - Calculated overlap information (optional)

    Args:
        positions: List of (x, y) tuples representing grid positions in meters
        fov_x: Horizontal field of view size in meters
        fov_y: Vertical field of view size in meters
        title: Plot title
        figsize: Figure size tuple (width, height)
        show_fov_boxes: Whether to show FOV bounding boxes around each position
        show_grid_lines: Whether to show grid lines
        show_center_lines: Whether to show center axis lines
        show_overlap_info: Whether to calculate and display overlap information

    Example:
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> plot_grid_positions(positions, 10e-6, 8e-6)
    """
    _, ax = plt.subplots(figsize=figsize)

    # Plot grid positions as red circles
    for pos in positions:
        x, y = pos
        ax.plot(
            x,
            y,
            "ro",
            markersize=8,
            label="Grid Position" if pos == positions[0] else "",
        )

        # Draw FOV bounding box around each position
        if show_fov_boxes:
            # Create rectangle centered at position
            rect = patches.Rectangle(
                (x - fov_x / 2, y - fov_y / 2),  # Bottom-left corner
                fov_x,
                fov_y,  # Width and height
                linewidth=1,
                edgecolor="blue",
                facecolor="none",
                linestyle="--",
                alpha=0.7,
                label="FOV Boundary" if pos == positions[0] else "",
            )
            ax.add_patch(rect)

    # Calculate plot limits from positions and FOV dimensions
    if positions:
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # Find the extent of positions
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add FOV/2 to account for the bounding boxes around each position
        # Plus some padding for better visualization
        padding_x = fov_x * 0.25
        padding_y = fov_y * 0.25

        x_extent_min = x_min - fov_x / 2 - padding_x
        x_extent_max = x_max + fov_x / 2 + padding_x
        y_extent_min = y_min - fov_y / 2 - padding_y
        y_extent_max = y_max + fov_y / 2 + padding_y

        ax.set_xlim(x_extent_min, x_extent_max)
        ax.set_ylim(y_extent_min, y_extent_max)
    else:
        # Fallback for empty positions
        ax.set_xlim(-fov_x, fov_x)
        ax.set_ylim(-fov_y, fov_y)

    # Add center lines
    if show_center_lines:
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    # Add grid lines
    if show_grid_lines:
        ax.grid(True, alpha=0.3)

    # Labels and title
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(title)

    # Add legend
    ax.legend(loc="upper right")

    # Equal aspect ratio for proper visualization
    ax.set_aspect("equal", adjustable="box")

    # Add text annotation with grid info
    ncols, nrows = calculate_grid_dimensions(positions)
    if ncols > 0 and nrows > 0:
        info_text = (
            f"Grid: {ncols}x{nrows}\nFOV: {fov_x * 1e6:.1f}x{fov_y * 1e6:.1f} μm"
        )

        # Calculate and add total grid area
        overlap_x, overlap_y = (
            calculate_grid_overlap(positions, fov_x, fov_y)
            if show_overlap_info and len(positions) > 1
            else (0.0, 0.0)
        )
        overlap = max(overlap_x, overlap_y)
        total_width, total_height = calculate_grid_coverage_area(
            ncols, nrows, fov_x, fov_y, overlap
        )
        info_text += f"\nArea: {total_width * 1e6:.1f}x{total_height * 1e6:.1f} μm"
    else:
        info_text = (
            f"Positions: {len(positions)}\nFOV: {fov_x * 1e6:.1f}x{fov_y * 1e6:.1f} μm"
        )

    # Optionally add overlap information
    if show_overlap_info and len(positions) > 1:
        overlap_x, overlap_y = calculate_grid_overlap(positions, fov_x, fov_y)
        if overlap_x > 0 or overlap_y > 0:
            # Use the maximum overlap value (they should be the same for regular grids)
            overlap = max(overlap_x, overlap_y)
            info_text += f"\nOverlap: {overlap:.1%}"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
