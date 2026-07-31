"""Acquire a fluorescence overview: settings, run control, and live result.

Standalone, and embeddable as a tab. It owns nothing the acquisition needs -- it is
handed a microscope and drives `FMTiledAcquisitionRunner` -- so it can be dropped into
the AutoLamella UI or opened on its own against a simulator.

Layout follows the house convention: canvas on the left, controls on the right,
actions along the bottom.
"""

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.acquisition import FMTiledAcquisitionRunner, stitch_tileset
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceImage,
    OverviewParameters,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets.fm_multi_channel_widget import FluorescenceMultiChannelWidget
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    FMOverviewConfirmationDialog,
)
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.fm.widgets.tile_grid_options_panel import TileGridOptionsPanel
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.fm.reprojection import project_stage_position
from fibsem.imaging.tiling.geometry import compute_tile_grid_from_fov
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.progress_widget import (
    FibsemProgressWidget,
    ProgressUpdate,
)
from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget

TEXT_MUTED = "#868e93"
PROGRESS_WIDTH = 260
PROGRESS_HEIGHT = 22
PROGRESS_FONT_PX = 10


def progress_slot(progress: FibsemProgressWidget) -> QWidget:
    """A fixed-size holder that keeps its space whether the bar is showing or not.

    `FibsemProgressWidget.reset()` hides itself, which in a plain layout collapses the
    row and shifts everything beside it -- so a bar going away at the end of a run
    would drag the buttons across. Reserving the space decouples them.
    """
    # Via a stylesheet on the holder rather than `setFont`, which does not reach the
    # bar, and rather than touching the widget's private `_bar`. The bar's own sheet
    # sets no font-size, so this applies without disturbing the chunk colouring it
    # relies on to tell finished from failed.
    progress.setStyleSheet(f"QProgressBar {{ font-size: {PROGRESS_FONT_PX}px; }}")

    slot = QWidget()
    slot.setFixedSize(PROGRESS_WIDTH, PROGRESS_HEIGHT)
    layout = QHBoxLayout(slot)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(progress)
    return slot


# Key the in-progress mosaic is drawn under. Distinct from a finished overview's
# position-derived key, so the preview can be dropped when the real stitch lands.
PREVIEW_KEY = "fm-preview"


class FMOverviewWidget(QWidget):
    """Configure, run and view a fluorescence overview acquisition."""

    overview_acquired = pyqtSignal(FluorescenceImage)

    # Internal hop from the acquisition thread to the GUI thread. The microscope's
    # progress signal is a psygnal, which calls its callbacks synchronously on
    # whichever thread emitted -- here, the worker. Touching widgets from there is a
    # cross-thread GUI access (Qt says so: "Cannot set parent, new parent is in a
    # different thread"). Re-emitting as a Qt signal gets it queued onto the GUI
    # thread, because this widget lives there.
    _progress_received = pyqtSignal(dict)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        channel_settings: Optional[List[ChannelSettings]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if microscope.fm is None:
            raise ValueError("This microscope has no fluorescence detector.")
        self.microscope = microscope
        self.fm = microscope.fm

        self._stop_event = threading.Event()
        self._worker: Optional[FunctionWorker] = None
        self._runner: Optional[FMTiledAcquisitionRunner] = None
        self._mosaic: Optional[FluorescenceImage] = None
        self._displayed_image: Optional[FluorescenceImage] = None
        # Stage position the canvas frame is built around. Everything shown -- images,
        # the planned grid, position markers -- is placed relative to it, so it has to
        # be fixed once and kept: re-deriving it from the newest image would shift the
        # whole scene each time one arrived. Taken from the first image displayed.
        self._origin: Optional[FibsemStagePosition] = None
        self._positions: List[FibsemStagePosition] = []
        self._grid_footprint: Optional[tuple] = None
        self._enabled_channels: Optional[List[ChannelSettings]] = None

        self._init_ui(channel_settings or self._default_channels())
        self._sync_tile_fov()
        self._on_settings_changed()

        self._progress_received.connect(self._apply_progress)
        self.fm.acquisition_progress_signal.connect(self._on_progress)

    def _default_channels(self) -> List[ChannelSettings]:
        """The saved FM configuration if there is one, otherwise a single channel."""
        try:
            from fibsem.fm.config import load_fm_configuration

            config = load_fm_configuration()
            if config is not None and config.channel_settings:
                return list(config.channel_settings)
        except Exception as e:
            logging.debug(f"Could not load the saved FM configuration: {e}")
        return [ChannelSettings(name="Channel-01")]

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self, channels: List[ChannelSettings]) -> None:
        self.canvas = FMRealSpaceCanvasWidget()

        # The planned grid, drawn on the canvas and clickable. A second view of the
        # mask `TileMaskWidget` owns, not a second copy: clicks are routed through the
        # settings widget so there is one place the selection lives.
        self.tile_grid_overlay = TileGridOverlay()
        self.canvas.canvas.add_overlay(self.tile_grid_overlay)

        # Grid display options live on the canvas toolbar, beside the layers control:
        # they are about reading the image, not about what gets acquired, so they do
        # not belong in the settings column with the parameters of the run.
        self.btn_tile_grid = self.canvas.canvas.add_toolbar_button(
            "mdi:grid", "Tile grid", self._toggle_tile_grid_panel, checkable=True
        )
        self.canvas.canvas._reposition_overlay_buttons()
        self.tile_grid_panel = TileGridOptionsPanel(self)
        self.tile_grid_panel.hide()
        self.tile_grid_panel.visibility_changed.connect(
            self.tile_grid_overlay.set_grid_visible
        )
        self.tile_grid_panel.color_changed.connect(self.tile_grid_overlay.set_color)
        self.tile_grid_panel.fill_alpha_changed.connect(
            self.tile_grid_overlay.set_fill_alpha
        )

        # Stage positions -- the current pose, lamellae, anything a host hands over --
        # projected onto whatever image is displayed. Non-interactive: this shows where
        # things are, it does not move them.
        self.position_overlay = PointsOverlay(color="#ffb300", marker="o", size=7)
        self.canvas.canvas.add_overlay(self.position_overlay)

        # The list alone shows only name/excitation/emission, with no way to set the
        # exposure, power or gain a tile is actually acquired at. This composes the
        # list with the detail panel for the selected channel, and is a drop-in for it.
        self.channel_widget = FluorescenceMultiChannelWidget(self.fm, channels)
        # Every overview setting lives in one widget, z-stack included, so their order
        # is decided in one place rather than split across two.
        self.settings_widget = FMOverviewSettingsWidget(
            channel_settings=channels
        )

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._section("Channels", self.channel_widget))
        controls_layout.addWidget(self.settings_widget)
        controls_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        # A channel row needs 446px before its name field hits its own minimum and the
        # excitation combo falls off the right edge; group-box margins and the vertical
        # scrollbar eat ~50px on the way in, so the column itself needs ~500. 510 leaves
        # a little room for wider fonts. The horizontal bar is off below, so anything
        # that overflows here is unreachable rather than merely cramped.
        scroll.setMinimumWidth(510)
        scroll.setMaximumWidth(560)
        # Vertical only: a horizontal bar here means a control is refusing to shrink,
        # and scrolling sideways to reach a spinbox is worse than a cramped one.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # Two bars: across the grid, and within the tile being acquired. Both are
        # `FibsemProgressWidget`, which distinguishes finished from failed -- a failed
        # run used to paint the same full green bar as a successful one.
        self.progress_tiles = FibsemProgressWidget()
        self.progress_tile_detail = FibsemProgressWidget()

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        self.button_acquire = QPushButton("Acquire Overview")
        self.button_acquire.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_acquire.setMinimumHeight(30)
        self.button_acquire.clicked.connect(self.acquire)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.setMinimumHeight(30)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setEnabled(False)

        # One row: status, both bars, then the actions. The bars sit in fixed slots, so
        # the buttons hold their position whatever the bars are doing.
        # The two bars are one readout and sit tight together; the gap that matters is
        # the one separating them from the buttons.
        bars = QWidget()
        bars_layout = QHBoxLayout(bars)
        bars_layout.setContentsMargins(0, 0, 0, 0)
        bars_layout.setSpacing(3)
        bars_layout.addWidget(progress_slot(self.progress_tiles))
        bars_layout.addWidget(progress_slot(self.progress_tile_detail))

        self.status_row = QWidget()
        status_layout = QHBoxLayout(self.status_row)
        status_layout.setContentsMargins(8, 4, 8, 8)
        status_layout.setSpacing(10)
        status_layout.addWidget(self.status, stretch=1)
        status_layout.addWidget(bars)
        status_layout.addWidget(self.button_cancel)
        status_layout.addWidget(self.button_acquire)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.status_row)

        self.channel_widget.settings_changed.connect(self._on_channels_changed)
        self.channel_widget.enabled_changed.connect(self._on_enabled_changed)
        self.settings_widget.changed.connect(self._on_settings_changed)
        self.tile_grid_overlay.tile_toggled.connect(self._on_tile_toggled)
        self.tile_grid_overlay.grid_resize_requested.connect(self._on_grid_resize)

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def channels(self) -> List[ChannelSettings]:
        """The channels that will be acquired: those ticked, or all of them.

        `ChannelListWidget` reports ticks through `enabled_changed` rather than
        exposing them, so they are tracked here. Before any tick has been touched the
        list is every channel, which is what the widget shows.
        """
        if self._enabled_channels is None:
            return list(self.channel_widget.channel_settings)
        return list(self._enabled_channels)

    def _sync_tile_fov(self) -> None:
        """Tell the settings widget one tile's field of view, for the area readout."""
        try:
            pixel_size_x, pixel_size_y = self.fm.camera.pixel_size
            width, height = self.fm.camera.resolution
            self.settings_widget.set_tile_fov(width * pixel_size_x, height * pixel_size_y)
        except Exception as e:
            logging.debug(f"Could not read the camera field of view: {e}")

    def _on_channels_changed(self, channels: List[ChannelSettings]) -> None:
        self.settings_widget.set_channel_settings(channels)
        self._on_settings_changed()

    def _on_enabled_changed(self, channels: List[ChannelSettings]) -> None:
        self._enabled_channels = list(channels)
        self._on_settings_changed()

    def _update_grid_summary(self) -> None:
        """Describe the grid on the canvas-side panel.

        Built from the settings widget's own formatted values rather than recomputed,
        so the two cannot end up quoting different numbers for the same grid.
        """
        parameters = self.settings_widget.parameters
        total = parameters.rows * parameters.cols
        parts = [
            f"{parameters.rows} × {parameters.cols}",
            f"{parameters.overlap:.0%} overlap",
        ]
        parts.append(f"{parameters.n_enabled_tiles}/{total} tiles")

        # The field of view goes on its own line rather than into the join: the panel
        # is narrow, and wrapping a "287 × 287 µm" mid-value reads as two numbers.
        summary = "  ·  ".join(parts)
        fov = self.settings_widget.label_total_fov.text()
        if fov and fov != "—":
            summary = f"{summary}\n{fov}"
        self.tile_grid_panel.set_summary(summary)

    def _refresh_tile_grid(self) -> None:
        """Redraw the planned grid on the canvas.

        Needs a tile field of view and the camera geometry. It does *not* need an image:
        the grid is anchored to the canvas origin -- the stage position the tileset is
        planned around -- so it can be drawn before anything is acquired, which is when
        a planned grid is worth the most.
        """
        fov = self.settings_widget._tile_fov
        if fov is None:
            self.tile_grid_overlay.clear()
            return

        try:
            width, height = self.fm.camera.resolution
            pixel_size = self.fm.camera.pixel_size[0]
        except Exception as e:
            logging.debug(f"Could not read the camera geometry for the tile grid: {e}")
            self.tile_grid_overlay.clear()
            return

        parameters = self.settings_widget.parameters
        tiles = compute_tile_grid_from_fov(
            nrows=parameters.rows,
            ncols=parameters.cols,
            fov_x=fov[0],
            fov_y=fov[1],
            image_width=width,
            image_height=height,
            overlap=parameters.overlap,
            mask=parameters.tile_mask,
        )
        # Give the canvas a scale before the first image, so the grid has a real frame
        # to be drawn in rather than arbitrary units. No-op once an image has landed --
        # by then the image has set it, and changing it would move what is already drawn.
        self.canvas.canvas.set_reference_pixel_size(pixel_size)
        # Centred on the origin: the tileset is planned around a stage position, not
        # around whatever happens to be displayed.
        self.tile_grid_overlay.set_anchor(self.canvas.canvas.metres_to_canvas(0.0, 0.0))

        # No `display_pixel_size`: the overlay reads it from the canvas at draw time.
        # Pinning it here would freeze the scale at whatever was displayed when the
        # settings last changed, and the image underneath changes without them -- the
        # live preview swaps in a decimated mosaic mid-run.
        self.tile_grid_overlay.set_grid(
            tiles, (height, width), pixel_size, overlap=parameters.overlap
        )

        span_x = parameters.cols * fov[0] * (1 - parameters.overlap) + fov[0]
        span_y = parameters.rows * fov[1] * (1 - parameters.overlap) + fov[1]

        # Refit only when the grid's footprint actually changes. Toggling a tile also
        # comes through here, and refitting on that threw away whatever zoom and pan
        # the user had set -- so clicking a tile appeared to zoom the view.
        # Recorded whether or not the view is refitted, so that a drag -- which
        # suppresses the refit on every step -- does not leave the footprint looking
        # stale and jump the view on the next unrelated settings change.
        footprint = (parameters.rows, parameters.cols, parameters.overlap)
        changed = footprint != self._grid_footprint
        self._grid_footprint = footprint
        if changed and not self.tile_grid_overlay.is_resizing:
            # Frame the planned area, so an empty canvas shows where the run will go
            # rather than nothing at all, and so clicks land in a meaningful frame.
            # Behind the same guard as the refit below, and for the same reason: a tile
            # toggle comes through here too, and re-framing on one threw away the
            # user's zoom -- which is what made clicking a tile look like zooming.
            self.canvas.set_world_extent(span_x, span_y)
            self.tile_grid_overlay.fit_view()

    def _toggle_tile_grid_panel(self) -> None:
        if not self.btn_tile_grid.isChecked():
            self.tile_grid_panel.hide()
            return

        self.tile_grid_panel.adjustSize()
        # Anchored in global coordinates: the panel is a top-level tool window, so it
        # cannot be placed relative to the canvas in widget coordinates.
        canvas = self.canvas.canvas
        anchor = canvas.mapToGlobal(QPoint(canvas.width() - 8, 44))
        self.tile_grid_panel.move(anchor.x() - self.tile_grid_panel.width(), anchor.y())
        self.tile_grid_panel.show()
        self.tile_grid_panel.raise_()

    def set_image(self, image: FluorescenceImage) -> None:
        """Show an image at the stage position it was acquired at.

        Overlays are positioned against the canvas frame, not against whatever is
        currently displayed, so the image has to be recorded when it is set rather than
        fetched from the canvas later -- the canvas keeps channel stacks, not the
        `FluorescenceImage` and its metadata, and the metadata is what carries the pose.
        """
        self._displayed_image = image
        if self._origin is None:
            self._origin = self._position_of(image)
        self.canvas.set_composite_key(self._key_for(image))
        self.canvas.set_placement(self._offset_of(image))
        self.canvas.set_fm_image(image)
        self._refresh_positions()
        self._refresh_tile_grid()

    @staticmethod
    def _position_of(image: FluorescenceImage) -> Optional[FibsemStagePosition]:
        return getattr(image.metadata, "stage_position", None)

    @staticmethod
    def _key_for(image: FluorescenceImage) -> str:
        """Identify an overview by when it was acquired.

        Not by position: a small overview and a wider one taken over the same area at
        different times are both worth keeping, and keying on position would silently
        drop the first. `stitch_tileset` carries the first tile's acquisition date onto
        the mosaic, so this is unique per run.

        A property of the image rather than a counter, so showing the same image twice
        replaces it instead of drawing a second copy on top of itself.
        """
        stamp = getattr(image.metadata, "acquisition_date", None)
        return f"overview@{stamp}" if stamp else "overview"

    def _offset_of(self, image: FluorescenceImage) -> Tuple[float, float]:
        """Where *image*'s centre sits relative to the canvas origin, in metres.

        The stage-to-plane projection the canvas deliberately knows nothing about. Falls
        back to the origin when the image cannot be projected -- acquired before the
        geometry was recorded, or with no pose at all -- which puts it in the middle of
        the view rather than refusing to show it.
        """
        position = self._position_of(image)
        geometry = getattr(image.metadata, "geometry", None)
        if self._origin is None or position is None or geometry is None:
            return (0.0, 0.0)
        try:
            pixel_size = image.metadata.pixel_size_x
            shape = np.asarray(image.data).shape[-2:]
            point = project_stage_position(
                position, self._origin, pixel_size, shape, geometry
            )
            # project_stage_position answers in pixels of an image acquired at the
            # origin; the canvas wants metres from it, so measure from the centre out.
            return (
                (point.x - shape[1] / 2) * pixel_size,
                (point.y - shape[0] / 2) * pixel_size,
            )
        except Exception as e:
            logging.debug(f"Could not place the image in stage space: {e}")
            return (0.0, 0.0)

    def set_positions(self, positions: List[FibsemStagePosition]) -> None:
        """Stage positions to mark on the overview, e.g. saved lamella positions.

        Names are carried onto the markers, so a caller does not have to keep a
        parallel list of labels in the order it happened to pass them.
        """
        self._positions = list(positions)
        self._refresh_positions()

    def _refresh_positions(self) -> None:
        """Mark the stage positions in the canvas frame.

        Projected against the canvas origin rather than onto whichever image happens to
        be displayed, so a marker names the same piece of sample no matter what is on
        screen -- and markers survive the image being swapped, or there being none yet.

        The projection uses the recorded geometry rather than the live microscope: after
        an overview is acquired the stage has moved on, and following it would drift
        every marker off the feature it names.
        """
        if not self._positions or self._origin is None:
            self.position_overlay.set_points([])
            return

        geometry = None
        pixel_size = None
        if self._displayed_image is not None:
            geometry = getattr(self._displayed_image.metadata, "geometry", None)
            pixel_size = getattr(self._displayed_image.metadata, "pixel_size_x", None)
        if geometry is None or not pixel_size:
            # Nothing to project through -- an image acquired before the geometry was
            # recorded, or none displayed yet.
            self.position_overlay.set_points([])
            return

        points, labels = [], []
        shape = np.asarray(self._displayed_image.data).shape[-2:]
        for position in self._positions:
            try:
                point = project_stage_position(
                    position, self._origin, pixel_size, shape, geometry
                )
            except Exception as e:
                logging.debug(f"Cannot mark {position.name!r}: {e}")
                continue
            metres = (
                (point.x - shape[1] / 2) * pixel_size,
                (point.y - shape[0] / 2) * pixel_size,
            )
            points.append(self.canvas.canvas.metres_to_canvas(*metres))
            labels.append(position.name or "")

        self.position_overlay.set_points(points, labels=labels)

    def _on_grid_resize(self, rows: int, cols: int) -> None:
        """An edge of the grid was dragged.

        Writes to the settings widget, which owns rows and columns, for the same reason
        a tile click does: the canvas is a view of that state, not a second copy.
        """
        # One call rather than two spin boxes: a drag emits on every motion event, and
        # setting them separately would refresh twice per step and pass through a grid
        # size that was never requested.
        self.settings_widget.set_grid_size(rows, cols)

    def _on_tile_toggled(self, row: int, col: int, enabled: bool) -> None:
        """A tile was clicked on the canvas.

        The mask belongs to the settings widget, so this writes there and lets the
        resulting `changed` redraw the overlay -- rather than updating the overlay
        directly, which would leave the two views to drift apart on any path that
        touched only one of them.
        """
        mask = self.settings_widget.tile_mask.mask
        parameters = self.settings_widget.parameters
        if mask is None:
            mask = [[True] * parameters.cols for _ in range(parameters.rows)]
        if not (0 <= row < len(mask) and 0 <= col < len(mask[row])):
            return

        mask[row][col] = enabled
        self.settings_widget.tile_mask.mask = mask

    def _on_settings_changed(self) -> None:
        if self.is_acquiring:
            return
        self._refresh_tile_grid()
        self._update_grid_summary()
        parameters = self.settings_widget.parameters
        enabled = parameters.n_enabled_tiles > 0
        self.button_acquire.setEnabled(enabled)
        # Only own the status line while it has something of its own to say. Re-enabling
        # the controls at the end of a run makes children emit `changed`, which landed
        # here and wiped the result message the moment it was set.
        if not enabled:
            self.status.setText("No tiles selected.")
        elif self.status.text() == "No tiles selected.":
            self.status.setText("")

    # ── acquisition ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Confirm, then run the overview on a worker thread."""
        if self.is_acquiring:
            logging.warning("An overview acquisition is already running.")
            return
        if self.fm.is_acquiring:
            logging.warning("Stop live acquisition before acquiring an overview.")
            return

        parameters = self.settings_widget.parameters
        channels = self.channels
        if not channels:
            self.status.setText("No channels enabled.")
            return

        zparams = self.settings_widget.z_parameters
        # Method, channel and sweep passes all come from the settings panel now,
        # rather than being defaulted with only the channel filled in. Resolved before
        # the dialog, because the dialog reports what will run.
        autofocus_settings = self.settings_widget.autofocus_settings

        dialog = FMOverviewConfirmationDialog(
            parameters=parameters,
            channel_settings=channels,
            zparams=zparams,
            tile_fov=self.settings_widget._tile_fov,
            autofocus_settings=autofocus_settings,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled before starting")
            return

        self._stop_event.clear()
        self._set_running(True)
        self._runner = FMTiledAcquisitionRunner(
            microscope=self.microscope,
            channel_settings=channels,
            overview_parameters=parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            stop_event=self._stop_event,
        )
        self._worker = FunctionWorker(self._acquire_worker)
        self._worker.start()

    def _acquire_worker(self) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        from fibsem.cancellation import OperationCancelledError

        try:
            runner = self._runner
            runner.run()
            mosaic = stitch_tileset(
                runner.tileset,
                runner.overview_parameters.overlap,
                centre_position=runner._initial_position,
                objective_position=runner._initial_objective_position,
            )
            self._mosaic = mosaic
            self.overview_acquired.emit(mosaic)
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-finished", "task": "tileset"}
            )
        except OperationCancelledError:
            logging.info("Overview acquisition cancelled")
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-cancelled", "task": "tileset"}
            )
        except Exception as e:
            logging.error(f"Overview acquisition failed: {e}", exc_info=True)
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-failed", "task": "tileset", "error": str(e)}
            )

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logging.info("Cancelling overview acquisition")
        self._stop_event.set()
        self.button_cancel.setEnabled(False)
        self.status.setText("Cancelling…")

    def _set_running(self, running: bool) -> None:
        self.button_acquire.setEnabled(not running)
        self.button_cancel.setEnabled(running)
        self.settings_widget.setEnabled(not running)
        self.channel_widget.setEnabled(not running)
        if running:
            # Left empty rather than shown as indeterminate, which would paint a full
            # bar before a single tile had been acquired.
            self.progress_tiles.reset()
            self.progress_tile_detail.reset()
            self.status.setText("Starting…")

    # ── progress ─────────────────────────────────────────────────────────

    def _on_progress(self, payload: dict) -> None:
        """Called by psygnal, on whichever thread emitted. Touches no widgets."""
        self._progress_received.emit(payload)

    def _apply_progress(self, payload: dict) -> None:
        """Runs on the GUI thread, queued via `_progress_received`.

        One signal carries both scales -- the tileset runner's and, from inside each
        tile, `acquire_z_stack`/`acquire_channels` -- so `task` decides which bar a
        payload belongs to. Anything else is ignored rather than shown twice.
        """
        state = payload.get("state")

        if state in ("overview-finished", "overview-cancelled", "overview-failed"):
            self._finish(state, payload.get("error"))
            return

        task = payload.get("task")
        if task == "tileset":
            self._apply_tile_progress(payload, state)
        elif task in ("z-stack", "channels", "autofocus"):
            self.progress_tile_detail.update_progress(self._tile_detail_update(payload))

    def _apply_tile_progress(self, payload: dict, state: Optional[str]) -> None:
        if state == "moving":
            # Deliberately not `indeterminate`: that paints a *full* bar with a
            # spinner, so every stage move looked like the run had just completed.
            # The bar keeps the last tile count -- which is still true between tiles --
            # and the transient state goes to the status label instead.
            self.status.setText("Moving stage…")
            return

        self.status.setText("")

        if state == "tile":
            # Deliberately does *not* clear the within-tile bar. It used to, which made
            # it vanish and reappear at every tile boundary -- a flicker for the whole
            # run. The next tile's first payload overwrites it a moment later anyway.
            self._show_preview(payload)

        current, total = payload.get("current", 0), payload.get("total", 1)
        remaining = payload.get("estimated_remaining_time")
        # The widget renders the count itself, so the message says what is being
        # counted and nothing more -- otherwise it reads "Tile 4/9 — 4/9".
        message = "Tiles"
        if remaining:
            self.progress_tiles.update_progress(ProgressUpdate.combined(
                current=current, total=total,
                remaining_seconds=remaining,
                total_seconds=payload.get("estimated_total_time", 0.0),
                message=message,
            ))
        else:
            self.progress_tiles.update_progress(
                ProgressUpdate.numeric(current=current, total=total, message=message)
            )

    def _tile_detail_update(self, payload: dict) -> ProgressUpdate:
        """Progress within the tile currently being acquired.

        A z-stack counts planes and a plain multi-channel acquisition counts channels,
        so the same bar reads sensibly either way rather than sitting empty whenever
        z-stacking happens to be off.
        """
        channel = payload.get("channel", "")
        zlevel, total_z = payload.get("zlevel"), payload.get("total_zlevels")
        if zlevel and total_z:
            if payload.get("task") == "autofocus":
                # Say which pass, so a coarse sweep followed by a fine one does not
                # look like the same bar inexplicably starting over.
                total_passes = payload.get("total_passes", 1)
                which = (f" {payload.get('pass_index', 1)}/{total_passes}"
                         if total_passes > 1 else "")
                return ProgressUpdate.numeric(
                    current=zlevel, total=total_z, message=f"{channel} focus{which}"
                )
            return ProgressUpdate.numeric(
                current=zlevel, total=total_z, message=f"{channel} z-stack"
            )
        index = payload.get("channel_index", 1)
        total = payload.get("total_channels", 1)
        return ProgressUpdate.numeric(
            current=index, total=total, message=f"{channel} channels"
        )

    def _show_preview(self, payload: dict) -> None:
        """Paint the mosaic-so-far onto the canvas.

        The runner publishes the whole preview canvas each tile, so this stays
        stateless -- it redisplays what it is given rather than accumulating tiles of
        its own, which is also what makes it correct if a frame is dropped.
        """
        image = payload.get("image")
        if image is None:
            return
        try:
            planes = np.asarray(image)
            if planes.ndim == 2:
                planes = planes[np.newaxis]

            # Where, and at what scale, *before* any pixels: `set_channel` composites
            # and places immediately, so anything established after it applies a tick
            # late -- the first frame of a run would land under the previous run's key
            # at the previous run's pixel size, which drew it at the wrong size on top
            # of a finished overview.
            #
            # Its own key, so the in-progress preview neither replaces a finished
            # overview nor survives as one: it is swapped for the real stitch at the end.
            self.canvas.set_composite_key(PREVIEW_KEY)
            # The preview mosaic spans the whole planned grid, which is centred on the
            # position the run started from -- the canvas origin.
            self.canvas.set_placement((0.0, 0.0))
            # The preview is decimated to keep it a sane size, so its pixels are
            # `preview_stride` times coarser than a tile's. Placement is by pixel size,
            # so saying so is all that is needed: coarser pixels over the same count
            # cover the same ground, and the mosaic lands at the size it represents.
            stride = payload.get("preview_stride", 1) or 1
            self.canvas.set_pixel_size(self.fm.camera.pixel_size[0] * stride)

            for channel, plane in zip(self.channels, planes):
                self.canvas.set_channel(channel.name, plane, channel.color)
        except Exception as e:
            logging.debug(f"Could not display the overview preview: {e}")

    def _finish(self, state: str, error: Optional[str]) -> None:
        self._set_running(False)
        self._worker = None
        self.progress_tile_detail.reset()

        if state == "overview-finished" and self._mosaic is not None:
            # Swap the decimated preview for the real thing. Dropped rather than left
            # underneath: it covers the same ground at a coarser scale, so keeping it
            # would only be a blurred copy hidden behind the stitch.
            self.set_image(self._mosaic)
            self.canvas.canvas.remove_image(PREVIEW_KEY)
            shape = self._mosaic.data.shape
            self.status.setText(f"Overview acquired — {shape[-1]} × {shape[-2]} px")
            self.progress_tiles.update_progress(ProgressUpdate.done())
        elif state == "overview-cancelled":
            self.status.setText("Cancelled. Tiles acquired so far are still shown.")
            self.progress_tiles.reset()
        else:
            self.status.setText(f"Failed: {error}" if error else "Failed.")
            # Failed, not done: the widget paints these differently, and a failure
            # showing a full green bar reads as success in everything but the text.
            self.progress_tiles.update_progress(
                ProgressUpdate.failed(error or "Acquisition failed")
            )

    # ── lifecycle ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self._stop_event.set()
        try:
            self.fm.acquisition_progress_signal.disconnect(self._on_progress)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)
