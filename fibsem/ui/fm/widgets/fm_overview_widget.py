"""Acquire a fluorescence overview: settings, run control, and live result.

Standalone, and embeddable as a tab. It owns nothing the acquisition needs -- it is
handed a microscope and runs the imaging service's `fm_acquire_overview` -- so it can
be dropped into the AutoLamella UI or opened on its own against a simulator.

Layout follows the house convention: canvas on the left, controls on the right,
actions along the bottom.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem import constants
from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.fm.structures import (
    ChannelSettings,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ObjectiveStartPosition,
    OverviewParameters,
)
from fibsem.imaging.overview_session import (
    FLUORESCENCE,
    POSITIONS,
    SELECTION,
    OverviewSession,
)
from fibsem.imaging.tiling import unreachable_tiles
from fibsem.imaging.tiling.geometry import TilePosition, compute_tile_grid_from_fov
from fibsem.imaging.tiling.progress import (
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.structures import (
    DeviceImagingState,
    FibsemStagePosition,
    InsertableDeviceState,
)
from fibsem.ui import notification_service, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.control import control_of
from fibsem.ui.fm.widgets.fm_multi_channel_widget import FluorescenceMultiChannelWidget
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    FMOverviewConfirmationDialog,
)
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.tokens import (
    CURRENT_POSITION_COLOUR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
)
from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget
from fibsem.ui.widgets.canvas.overlay_controls import (
    CanvasOverlayControls,
    CanvasPopover,
)
from fibsem.ui.widgets.canvas.overlays import stage_context
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import (
    FieldOfViewOverlay,
    PointsOverlay,
)
from fibsem.ui.widgets.canvas.overlays.tile_grid_options_panel import (
    TileGridOptionsPanel,
)
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.canvas.stage_frame import FMStageProjection, StageFrame
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    ElidedLabel,
    IconToolButton,
    TitledPanel,
)
from fibsem.ui.widgets.overview_list_widget import OverviewListWidget
from fibsem.ui.widgets.progress_widget import (
    FibsemProgressWidget,
    ProgressUpdate,
)

TEXT_MUTED = stylesheets.TEXT_MUTED_COLOR
PROGRESS_FONT_PX = 10


def shrink_progress_text(progress: FibsemProgressWidget) -> FibsemProgressWidget:
    """Fit a progress bar to a narrow column: smaller text, and no say in the width.

    The font goes on via a stylesheet rather than `setFont`, which does not reach the
    bar, and rather than touching the widget's private `_bar`. The bar's own sheet sets
    no font-size, so this applies without disturbing the chunk colouring it relies on to
    tell finished from failed.

    The size policy is the same trap as :class:`ElidedLabel`, one level down: a bar
    reports its *message* width as its hint, so a long failure ran off the end of the
    column and out of the window. Ignored horizontally, it takes the width it is given.
    """
    progress.setStyleSheet(f"QProgressBar {{ font-size: {PROGRESS_FONT_PX}px; }}")
    progress.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    return progress


# Key the in-progress mosaic is drawn under. Distinct from a finished overview's
# position-derived key, so the preview can be dropped when the real stitch lands.
PREVIEW_KEY = "fm-preview"

# The three position markers (`CURRENT_`/`SAVED_`/`SELECTED_POSITION_COLOUR`) are
# imported from `tokens` rather than defined here: all three are crosshairs, so colour
# is the only thing telling them apart, and the FIB/SEM overview draws the same three
# things. A user reads both tabs, and the markers meaning different things on each is
# worse than any one choice of colour.

# Icon buttons in a `TitledPanel` header. Not smaller than this: `TOOLBUTTON_ICON_STYLESHEET`
# pads 6px each side, so a 22px button leaves 10px of width for a 16px icon and clips it --
# the frame's sides vanish and what is left reads as a rendering fault. Matches the size the
# row buttons use.
_HEADER_BTN_SIZE = 26


@dataclass
class PlacedOverviewImageRecord:
    """One overview the canvas is holding, and what is known about it.

    Named at length to stay distinct from the two neighbours it would otherwise be
    confused with: the canvas's own `PlacedImage`, which is an artist and an extent,
    and the `FluorescenceImage` this was built from.

    The canvas can hold several overviews at once; this keeps each one's provenance
    for as long as its pixels are on screen.

    Holds the *metadata*, not the image. What a list needs -- where it was, when, at
    what scale -- is all in there, and keeping N full-resolution mosaics alive to
    reach it would cost hundreds of megabytes on a canvas already short of them.
    The pixels the canvas redraws from are its own decimated copies.
    """

    id: str
    label: str
    metadata: "FluorescenceImageMetadata"
    visible: bool = True

    @property
    def detail(self) -> str:
        """The grid and scale, for the second half of a list row."""
        parts = []
        # `stitch_tileset` names the mosaic's position `stitched_mosaic_3x3`, so the
        # grid is already recorded -- reading it back beats plumbing it separately.
        name = getattr(getattr(self.metadata, "stage_position", None), "name", "") or ""
        grid = name.rsplit("_", 1)[-1] if "_" in name else ""
        if grid and grid[0].isdigit():
            parts.append(grid.replace("x", "×"))
        pixel_size = getattr(self.metadata, "pixel_size_x", None)
        if pixel_size:
            parts.append(f"{pixel_size * constants.SI_TO_MICRO:.2f} µm/px")
        return " · ".join(parts)


# What each countless phase of a run is called on screen. All three are handled the same
# way and for the same reason: the bar keeps its last count, which stays true
# throughout, and the phase goes to the status label instead.


class FMOverviewWidget(QWidget):
    """Configure, run and view a fluorescence overview acquisition."""

    overview_acquired = pyqtSignal(FluorescenceImage)
    # A run that failed before its run could report it on `tiled_progress`: the
    # service refused it, or its folder could not be made. Crosses from the worker.
    _run_failed = pyqtSignal(str)
    # Whether a run is in progress here. A *state*, re-emitted on every change
    # rather than an edge, so a host that connects late or recomputes from
    # several facts cannot end up holding a stale answer. The host that matters
    # is the window, which must stop the other overview driving the stage while
    # this one is mid-tileset.
    acquiring_changed = pyqtSignal(bool)

    # A user right-clicked the canvas and asked for a position there. Both carry a
    # *fluorescence* stage position -- the canvas is anchored on the FM side, and both
    # are only ever emitted with the stage in a valid FM orientation, so the rotation
    # and tilt they carry are the fluorescence ones.
    #
    # Requests, not commands, and deliberately: this widget knows nothing about
    # lamellae or experiments, which is what lets it open standalone against a
    # simulator. A host that owns an experiment decides what a position *means*, whether
    # the user should be asked first, and what else moves with it.
    position_add_requested = pyqtSignal(object)  # FibsemStagePosition
    position_move_requested = pyqtSignal(str, object)  # name, FibsemStagePosition
    # A marked position was clicked. Emitted with the name it was marked under, so a
    # host can select whatever that name means to it.
    position_selected = pyqtSignal(str)

    # Internal hop from the acquisition thread to the GUI thread. The microscope's
    # progress signals are psygnals, which call their callbacks synchronously on
    # whichever thread emitted -- here, the worker. Touching widgets from there is a
    # cross-thread GUI access (Qt says so: "Cannot set parent, new parent is in a
    # different thread"). Re-emitting as a Qt signal gets it queued onto the GUI
    # thread, because this widget lives there.
    #
    # One per source, because the two describe different things at different scales and
    # each drives its own bar: the run as a whole, and the tile currently being taken.
    # `object`, not `TiledProgress`: psygnal hands a slot whatever was emitted, and a
    # plugin-loaded producer is not obliged to emit the typed record.
    _tile_progress_received = pyqtSignal(object)
    # `object`, not the typed record, for the same reason.
    _fm_progress_received = pyqtSignal(object)
    # Same hop for stage moves: `stage.position.changed` is a psygnal, so it fires
    # on whichever thread moved the stage -- a worker, during an acquisition.
    _stage_moved = pyqtSignal(object)
    # And for something starting or finishing on the FM, which is reported from
    # whichever thread finished with it -- for our own runs, the acquisition worker.
    _fm_acquiring_changed = pyqtSignal(bool)
    _session_changed = pyqtSignal(str)

    def __init__(
        self,
        session: OverviewSession,
        channel_settings: Optional[List[ChannelSettings]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if session.kind != FLUORESCENCE:
            raise ValueError("This widget draws a fluorescence overview session.")
        # The positions, selection, lock, run state, mosaic and cached stage
        # position: the overview's, which a host writes.
        self.session = session
        self.microscope = session.microscope
        self.fm = session.microscope.fm

        # Cancel has been pressed on the run under way; the imaging service stops it.
        self._cancelling = False
        self._worker: Optional[FunctionWorker] = None
        # What the run in progress was started with, for the worker: the channels,
        # parameters, z-stack, autofocus and centre `acquire` read before it began.
        self._run_settings: Optional[dict] = None
        # Every overview the canvas is holding, oldest first, keyed by the id each was
        # given. Replaces a single `_displayed_image` that each acquisition overwrote,
        # which left earlier overviews on screen with nothing describing them.
        self._records: Dict[str, PlacedOverviewImageRecord] = {}
        # Ids are minted rather than derived from the acquisition date. The date is
        # unique enough, but `_key_for` fell back to a bare "overview" whenever one was
        # missing -- so every dateless overview shared one key and silently replaced the
        # last. Matching on the date is still done, below, to keep re-showing one image
        # a replacement rather than a second copy of itself.
        self._record_count = 0
        # Stage position the canvas frame is built around. Everything shown -- images,
        # the planned grid, position markers -- is placed relative to it, so it has to
        # be fixed once and kept: re-deriving it from the newest image would shift the
        # whole scene each time one arrived. Taken from the first image displayed.
        self._origin: Optional[FibsemStagePosition] = None
        # The camera half of the frame, kept rather than re-read -- see `_projection`.
        # None until first resolved, and set back to None to force a re-read.
        self._cached_projection: Optional[FMStageProjection] = None
        self._grid_footprint: Optional[tuple] = None
        self._enabled_channels: Optional[List[ChannelSettings]] = None
        # The last answer to "can the FM image from here" -- see `_on_stage_moved`,
        # which re-derives the gate and banner only when this changes.
        self._last_imaging_state: Optional[DeviceImagingState] = None
        # The objective half of the canvas info bar, cached. Read from hardware only
        # when something may have moved it -- see `_refresh_objective_info`.
        self._objective_info: Optional[str] = None
        # Where the next overview will be centred, once a user has dragged the grid
        # somewhere. None means "wherever the stage is", which is what the runner does
        # by default -- so an untouched grid describes exactly what would be acquired.
        self._target: Optional[FibsemStagePosition] = None
        # Set only while `_on_drag_finished` is running the refresh a drag deferred.
        # See there: that refresh must not re-frame the view.
        self._finishing_drag = False
        # Where acquired overviews are written: the Output panel's folder (see
        # `set_save_directory`), else the working directory, as on the beam tab.
        self._saved_path: Optional[str] = None
        # Set by `_on_move_errored` so `_on_move_finished`, which always runs, knows
        # not to clear a message the failure has just put up.
        self._move_failed: bool = False

        self._init_ui(channel_settings or self._default_channels())
        self._sync_tile_fov()
        self._on_settings_changed()

        self._tile_progress_received.connect(self._apply_tile_progress)
        self._run_failed.connect(self._on_run_failed)
        self._fm_progress_received.connect(self._apply_fm_progress)
        # Two scales, two signals, two handlers. The tileset -- tile n of N, the mosaic
        # so far, and how the run ended -- reports on the imaging service's
        # `tiled_progress` beside the beam tiler's, because it is the same event
        # The work *inside* a tile stays on the detector's own progress signal, which
        # is where a z-stack, a channel acquisition and an autofocus sweep report from
        # whether or not a tileset is running.
        # On the imaging service, which every microscope with an FM has.
        self.microscope.imaging.tiled_progress.connect(self._on_tile_progress)
        self.microscope.imaging.overview_acquired.connect(self._on_acquired)
        self.fm.acquisition_progress_signal.connect(self._on_fm_progress)
        self._stage_moved.connect(self._on_stage_moved)
        self._session_changed.connect(self._on_session_changed)
        self.session.changed.connect(self._on_session_signal)
        # A plain bound method, not `self._stage_moved.emit`, matching how the progress
        # signal is subscribed just above. psygnal holds bound methods weakly and drops
        # them when the owner is collected, which a Qt signal's `emit` does not get: PyQt
        # builds a new wrapper on every access, so psygnal cannot weakref it, cannot
        # match it at disconnect time -- and says nothing when the disconnect therefore
        # removes nothing. Emitting into a widget Qt had already torn down was a segfault.
        if self.microscope.stage is not None:
            self.microscope.stage.position.changed.connect(self._on_stage_signal)
        self._fm_acquiring_changed.connect(self._on_fm_acquiring_changed)
        self.fm.acquiring_changed.connect(self._on_fm_acquiring_signal)
        # Marshalled by `@ensure_main_thread` on the slot rather than by relaying
        # through a Qt signal like the three above. That decorator is the contract
        # `FibsemMicroscope` states for its psygnals and what most of the UI does; the
        # relay is this widget's own dialect. Still a plain bound method, so psygnal can
        # hold it weakly and match it at disconnect time.
        self.fm.objective.position_changed.connect(self._on_objective_moved)
        # Same contract as the objective's, and for the same reason: the transform is an
        # input to every projection this widget draws through, and the projection is kept
        # rather than re-read, so nothing else would notice it moved.
        self.fm.transform_changed.connect(self._on_transform_changed)
        # Same contract again. The holder's slots are drawn from its calibration, and
        # a stage move does not redraw them, so a recalibration has to say so. On
        # the sample handling service, which a microscope with no stage doesn't have.
        if self.microscope.sample_handling is not None:
            self.microscope.sample_handling.holder_changed.connect(
                self._on_holder_changed
            )
        self._refresh_current_position()
        self._refresh_objective_info()
        self._refresh_orientation_banner()

    def _default_channels(self) -> List[ChannelSettings]:
        """The saved FM configuration if there is one, otherwise a single channel."""
        try:
            from fibsem.fm.config import load_fm_configuration
            from fibsem.session_state import session_state_for

            config = load_fm_configuration(session_state_for(self.microscope))
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
        # Stand aside for a marked position. A press on one belongs to the marker, not
        # to the tile under it -- see `TileGridOverlay.set_reserved`. Wired to the same
        # hit test a click uses, so the grid stands aside for exactly what a click would
        # have selected, rather than for a second opinion about where the markers are.
        #
        # The crosshair, though, not the field-of-view box. The box is a large thing to
        # reserve -- a whole tile on the fluorescence tab, and a whole tile here at any
        # HFW of 100 um or under -- and reserving it leaves tiles that cannot be toggled
        # at all with nothing on screen to say why. Reserving the crosshair costs at
        # worst a click that toggles a tile you meant to select, which greys out visibly
        # and undoes with one more click. A wrong action that announces itself beats a
        # dead end that does not.
        self.tile_grid_overlay.set_reserved(
            lambda x, y: self._position_at(x, y, crosshair_only=True) is not None
        )
        self.canvas.canvas.add_overlay(self.tile_grid_overlay)

        # The same three switches the beam tab offers, over the same three shapes, from
        # the same entries -- so neither tab can offer a switch the other does not, or
        # word it differently. This tab draws no saved-position or gridbar overlay, so
        # it takes only the stage-context set.
        self.overlay_controls = CanvasOverlayControls(
            stage_context.context_overlay_entries(self.microscope)
        )
        self.overlay_controls.toggled.connect(lambda *_: self._refresh_stage_metadata())
        self._context_defaults_for_calibrated = stage_context.holder_is_calibrated(
            self.microscope
        )
        self.btn_overlays = self.canvas.canvas.add_toolbar_button(
            "mdi:eye-outline",
            "Overlays",
            self._toggle_overlays,
            checkable=True,
        )
        self.overlay_popover = CanvasPopover(
            self.overlay_controls, parent=self.canvas.canvas
        )

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
        self.tile_grid_panel.visibility_changed.connect(self._refresh_grid_hint)
        self._refresh_grid_hint(self.tile_grid_overlay.is_grid_visible)
        self.tile_grid_panel.color_changed.connect(self.tile_grid_overlay.set_color)
        self.tile_grid_panel.fill_alpha_changed.connect(
            self.tile_grid_overlay.set_fill_alpha
        )
        self.tile_grid_panel.centre_requested.connect(self.clear_target)

        # Stage positions -- the current pose, lamellae, anything a host hands over --
        # projected onto whatever image is displayed. Non-interactive: this shows where
        # things are, it does not move them.
        # Where the sample and the stage can physically go -- the context an overview
        # is read against. Added before the position markers so it sits beneath them.
        self.stage_overlay = MinimapShapesOverlay(zorder=4.0, crosshair_half_px=24)
        self.canvas.canvas.add_overlay(self.stage_overlay)

        # Where the stage is now. Distinct from the red origin marker the canvas
        # draws: the origin explains why everything sits where it does, this is what
        # you steer by. They coincide until the stage moves, then diverge.
        self.current_position_overlay = PointsOverlay(
            color=CURRENT_POSITION_COLOUR, marker="+", size=15, edge_width=2.8
        )
        self.canvas.canvas.add_overlay(self.current_position_overlay)

        # Crosshairs rather than dots: a marked position is a point on the sample, and
        # a filled dot covers the feature it is naming. The gap in the middle is the
        # whole reason -- you can see what you marked.
        #
        # Boxed with the camera's field of view, set in `_refresh_positions` once the
        # projection is known, so a marker also says how much sample one frame covers.
        # The current stage position above stays unboxed: it is where you are rather
        # than something you are sizing up, and boxing it would double every lamella's
        # box the moment you drove to one.
        self.position_overlay = FieldOfViewOverlay(
            color=SAVED_POSITION_COLOUR, marker="+", size=11
        )
        self.canvas.canvas.add_overlay(self.position_overlay)

        # The selected position, on its own layer rather than as a colour within the
        # one above: `PointsOverlay` paints every point the same, and one selected
        # marker is not worth teaching it per-point colours for. Added last, so it
        # draws over its unselected neighbours where markers crowd together.
        self.selected_position_overlay = FieldOfViewOverlay(
            color=SELECTED_POSITION_COLOUR, marker="+", size=15
        )
        self.canvas.canvas.add_overlay(self.selected_position_overlay)

        # The list alone shows only name/excitation/emission, with no way to set the
        # exposure, power or gain a tile is actually acquired at. This composes the
        # list with the detail panel for the selected channel, and is a drop-in for it.
        self.channel_widget = FluorescenceMultiChannelWidget(self.fm, channels)
        # Every overview setting lives in one widget, z-stack included, so their order
        # is decided in one place rather than split across two.
        self.settings_widget = FMOverviewSettingsWidget(channel_settings=channels)

        controls = QWidget()
        self._controls_layout = QVBoxLayout(controls)
        self._controls_layout.setContentsMargins(8, 8, 8, 8)
        self._controls_layout.setSpacing(10)
        # First: what is already on the canvas, then the settings for the next run.
        # Reading the column top to bottom then follows the same order as using the tab
        # -- look at what you have, then set up what comes next -- where putting it
        # under the channels split the two halves of "the next acquisition" around it.
        # A host's own section still goes above this, being the subject of the tab.
        self.overview_list = OverviewListWidget()
        self.overview_list.visibility_toggled.connect(self.set_overview_visible)
        self.overview_list.remove_requested.connect(self.remove_overview)
        overviews_panel = self._section("Overviews", self.overview_list)
        # In the panel header rather than under the list: loading acts on the section
        # as a whole, not on any row in it, and a full-width button below the rows read
        # as a fourth row. It also keeps the list widget free of the filesystem -- the
        # button belongs to the section, which this widget builds.
        # An image being added, rather than a folder being opened: the folder is where
        # it came from, not what you end up with. Outline weight to match the eye and
        # trash on the rows below it.
        self.btn_load_overview = IconToolButton(
            icon="mdi:image-plus-outline",
            tooltip="Load a saved overview",
            size=_HEADER_BTN_SIZE,
        )
        self.btn_load_overview.clicked.connect(self._prompt_for_overview)
        overviews_panel.add_header_widget(self.btn_load_overview)
        self._controls_layout.addWidget(overviews_panel)
        self._controls_layout.addWidget(self._section("Channels", self.channel_widget))
        self._controls_layout.addWidget(self.settings_widget)
        self._controls_layout.addStretch()

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

        # Two bars: across the grid, and within the tile being acquired. Both are
        # `FibsemProgressWidget`, which distinguishes finished from failed.
        # Hidden until a run starts: they are empty most of the time, and stacked in a
        # column there is nothing beside them to shift when they come and go.
        self.progress_tiles = shrink_progress_text(FibsemProgressWidget())
        self.progress_tile_detail = shrink_progress_text(FibsemProgressWidget())

        self.status = ElidedLabel("")
        self.status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        # Shown only when the stage is somewhere an overview cannot be acquired from.
        # A line saying what is wrong and a button that fixes it, rather than a greyed
        # Acquire button with no explanation.
        self.orientation_notice = ElidedLabel("")
        self.orientation_notice.setStyleSheet(
            f"color: {stylesheets.WARN_COLOR}; font-size: 11px;"
        )
        # Labelled in `_refresh_orientation_banner`, from the same `default_orientation`
        # the move targets -- the control widget lets that be changed at runtime, and a
        # button naming one orientation while going to another is worse than no button.
        self.button_move_to_fm = QPushButton()
        self.button_move_to_fm.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_move_to_fm.clicked.connect(self.move_to_fm_device)

        self.orientation_banner = QWidget()
        banner_layout = QHBoxLayout(self.orientation_banner)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.setSpacing(6)
        banner_layout.addWidget(self.orientation_notice, stretch=1)
        banner_layout.addWidget(self.button_move_to_fm)
        self.orientation_banner.setVisible(False)

        self.button_acquire = QPushButton("Acquire Overview")
        self.button_acquire.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_acquire.setMinimumHeight(30)
        self.button_acquire.clicked.connect(self.acquire)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.setMinimumHeight(30)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setEnabled(False)

        # In the Grid panel's header, not stacked above Acquire. It sets rows, columns
        # and the mask, so it belongs where those live -- and full width beside the run
        # controls gave it the same weight as the button that starts a twenty-minute
        # acquisition. The count it produces is read off the tile mask directly below.
        self.button_select_ground = IconToolButton(
            icon="mdi:crop-free",
            tooltip="Select the ground to image on a FIB/SEM overview",
            size=_HEADER_BTN_SIZE,
        )
        self.button_select_ground.clicked.connect(self.select_ground_to_image)
        self.settings_widget.add_grid_header_widget(self.button_select_ground)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)
        buttons_layout.addWidget(self.button_cancel)
        buttons_layout.addWidget(self.button_acquire, stretch=1)

        # The actions live at the foot of the settings column, not in a row spanning the
        # whole widget. Spanning cost 1030 px of minimum width -- all of it, the canvas
        # asked for 10 -- because a horizontal row cannot give anything up. Stacked in
        # the column the floor is the column's own, and the canvas can be as narrow as
        # the window allows. Same shape as its beam-side twin, which stacks its run
        # controls under its settings for the same reason.
        self.status_row = QWidget()
        actions_layout = QVBoxLayout(self.status_row)
        actions_layout.setContentsMargins(8, 4, 8, 8)
        actions_layout.setSpacing(5)
        actions_layout.addWidget(self.status)
        actions_layout.addWidget(self.orientation_banner)
        actions_layout.addWidget(self.progress_tiles)
        actions_layout.addWidget(self.progress_tile_detail)
        actions_layout.addWidget(buttons)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(scroll, stretch=1)
        side_layout.addWidget(self.status_row)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, stretch=1)

        self._set_progress_visible(False)

        self.channel_widget.settings_changed.connect(self._on_channels_changed)
        self.channel_widget.enabled_changed.connect(self._on_enabled_changed)
        self.settings_widget.changed.connect(self._on_settings_changed)
        self.tile_grid_overlay.tile_toggled.connect(self._on_tile_toggled)
        self.tile_grid_overlay.grid_resize_requested.connect(self._on_grid_resize)
        self.tile_grid_overlay.grid_move_requested.connect(self._on_grid_move)
        self.tile_grid_overlay.drag_finished.connect(self._on_drag_finished)
        self.canvas.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.canvas.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)
        self.canvas.canvas.cursor_moved.connect(self._on_cursor_moved)

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        """Whether an overview acquisition is running here.

        `_running` as well as the worker, and the order is why: `acquire` calls
        `_set_running(True)` *before* it builds the worker, so for the width of that
        gap a worker-only answer says no while a run is starting. That gap is exactly
        when `acquiring_changed` is emitted, so a host locking the other overview off
        this property would get False and lock nothing.

        Keeping the worker check as well as the flag, rather than replacing it: the
        union is true over a superset of the interval either is, and every caller is a
        guard, so being early and late is the safe direction to be wrong in.
        """
        return self.session.running or (
            self._worker is not None and self._worker.is_alive()
        )

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
            self.settings_widget.set_tile_fov(
                width * pixel_size_x, height * pixel_size_y
            )
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
        fov = self.settings_widget.grid.label_total_fov.text()
        if fov and fov != "—":
            summary = f"{summary}\n{fov}"
        # Said in words as well as drawn, because a grid sitting away from the stage
        # marker is only obvious while both are on screen -- zoom into the grid and
        # the one thing that would tell you it is offset scrolls out of view.
        offset = self._target_offset()
        if offset is not None:
            summary = (
                f"{summary}\nOffset {offset[0] * constants.SI_TO_MICRO:+.0f}, "
                f"{offset[1] * constants.SI_TO_MICRO:+.0f} µm from the stage"
            )
        self.tile_grid_panel.set_summary(summary)
        # The text just changed, and it wraps -- so the panel may need to be a different
        # height than it was. Only worth doing while it is up; opening it fits it anyway.
        if self.tile_grid_panel.isVisible():
            self._fit_tile_grid_panel()

    def _target_offset(self) -> Optional[Tuple[float, float]]:
        """How far the target sits from the stage, in metres, or None if not set.

        Measured in the displayed plane, not by subtracting stage axes. Everything on
        this canvas is drawn in that plane, so a readout in stage coordinates describes
        the same gap in a frame the user cannot see: on a compustage at t = -180 it
        reports the y offset with the sign reversed, and on a pre-tilted mount it hides
        the z entirely -- a 100 µm step down the screen carried 70 µm of z and the
        readout said nothing.
        """
        if self._target is None:
            return None
        frame = self._frame()
        position = self._current_stage_position()
        if frame is None or position is None:
            return None
        try:
            here = frame.offset(position)
            there = frame.offset(self._target)
        except Exception as e:
            logging.debug(f"Could not measure the target offset: {e}")
            return None
        return (there[0] - here[0], there[1] - here[1])

    def _refresh_tile_grid(self, redraw_context: bool = True) -> None:
        """Redraw the planned grid on the canvas.

        Needs a tile field of view and the camera geometry. It does *not* need an image:
        the grid is anchored to the canvas origin -- the stage position the tileset is
        planned around -- so it can be drawn before anything is acquired, which is when
        a planned grid is worth the most.

        *redraw_context* False leaves the stage context alone unless the canvas scale
        changed under it. For a stage move, which redraws the context itself when the
        pose changes and has nothing to redraw when it does not.
        """
        fov = self.settings_widget._tile_fov
        if fov is None:
            self.tile_grid_overlay.clear()
            return

        # Off the kept projection, which carries exactly these two. Read straight from
        # the camera they are two `active_channel()` scopes, and this runs on every
        # motion event of a grid drag -- so each scope set the shared view to the FM and
        # put it back, per mouse-move, which is visible as the active view flicking in
        # the microscope's own UI. See `_projection` for why keeping it is safe.
        projection = self._projection()
        if projection is None:
            logging.debug("Could not read the camera geometry for the tile grid")
            self.tile_grid_overlay.clear()
            return
        height, width = projection.shape
        pixel_size = projection.pixel_size

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
        rescaled = self.canvas.canvas.set_reference_pixel_size(pixel_size)
        # Centred on the position the run will be centred on -- a dragged target, or
        # the stage itself -- not on whatever happens to be displayed, nor on the canvas
        # origin, which is where the stage was the *first* time anything was drawn.
        offset = self._grid_offset()

        # Anchor and flags handed over with the grid rather than set separately: this
        # runs on every motion event of a drag, and each setter repaints every tile
        # patch.
        #
        # No `display_pixel_size`: the overlay reads it from the canvas at draw time.
        # Pinning it here would freeze the scale at whatever was displayed when the
        # settings last changed, and the image underneath changes without them -- the
        # live preview swaps in a decimated mosaic mid-run.
        self.tile_grid_overlay.set_grid(
            tiles,
            (height, width),
            pixel_size,
            overlap=parameters.overlap,
            unreachable=self._unreachable(parameters, tiles=tiles),
            anchor=self.canvas.canvas.metres_to_canvas(*offset),
        )
        # Same camera geometry, so it can be drawn at the same time rather than
        # waiting for an image the tile grid does not wait for either.
        #
        # Not while the grid is being dragged, though. None of these shapes move when
        # the grid does -- the travel limits, the grid boundary and the holder slots are
        # all fixed to the stage -- so redrawing them on every motion event is wasted,
        # and it is a *full* canvas repaint, which costs far more than the grid it was
        # called alongside. The beam tab skips its equivalent for the same reason.
        #
        # Nor on a stage move, unless the scale just changed: a translation moves none
        # of them either, and this runs on every stage update. Redrawing here put four
        # frame conversions and a full repaint on each one, for shapes that had not
        # moved -- `_on_stage_moved` redraws them itself when the pose changes.
        if not self.tile_grid_overlay.is_dragging and (redraw_context or rescaled):
            self._refresh_stage_metadata()

        span_x = parameters.cols * fov[0] * (1 - parameters.overlap) + fov[0]
        span_y = parameters.rows * fov[1] * (1 - parameters.overlap) + fov[1]

        # Measured against the area as it stands, before the line below moves it.
        left_area = self._grid_has_left_the_working_area(offset, span_x, span_y)

        # Keep the declared working area on the grid, always and silently. It is what
        # the zoom limiter measures against, so an area left behind stretches the
        # content across the gap and caps how far the view can zoom in. `refit=False`
        # is what makes doing this safe: without it dragging the grid would snap the
        # view onto it on every motion event.
        #
        # Not *during* a drag, though. Re-declaring asks the canvas to repaint, and a
        # repaint is what the blitted grid is drawn over, so it would put two full
        # canvas draws on every motion event. Deferred to `drag_finished`, which
        # arrives once. Nobody is measuring zoom limits against
        # it mid-gesture.
        if not self.tile_grid_overlay.is_dragging:
            self.canvas.set_world_extent(span_x, span_y, offset, refit=False)

        # Re-frame only for a reason the user would expect to move the view: the grid's
        # footprint changed, or it has ended up outside what was framed (a stage move to
        # an offset FM mount travels 48 mm). Never mid-gesture -- a drag emits on every
        # motion event, and re-framing each time zooms the canvas out from under the
        # cursor. Toggling a tile comes through here too, and re-framing on one threw
        # away the user's zoom, which is what made clicking a tile look like zooming.
        # The footprint is recorded either way, so a suppressed refit does not leave it
        # stale and jump the view on the next unrelated settings change.
        footprint = (parameters.rows, parameters.cols, parameters.overlap)
        changed = footprint != self._grid_footprint
        self._grid_footprint = footprint
        if (
            (changed or left_area)
            and not self.tile_grid_overlay.is_dragging
            and not self._finishing_drag
        ):
            self.tile_grid_overlay.fit_view()

    def _on_drag_finished(self) -> None:
        """Do the work a drag deferred, without moving the camera.

        The declared working area has to catch up with where the grid ended and the
        stage context has to be redrawn -- both repaint, which is why they waited.

        The refit must *not* fire, though, and would: `left_area` compares the grid
        against a working area that has deliberately stopped following it, so by the
        end of any drag the grid has "left" an area that simply stayed put. That made
        every drag end by fitting the view to the grid.

        A drag must not move the view, and the end of a drag is still the drag -- the
        same rule the mid-gesture guard already states, applied to the moment the
        gesture finishes.
        """
        self._finishing_drag = True
        try:
            self._refresh_tile_grid()
        finally:
            self._finishing_drag = False

    def _grid_has_left_the_working_area(
        self, offset: Tuple[float, float], span_x: float, span_y: float
    ) -> bool:
        """Whether the declared working area has stopped describing where the grid is.

        The working area is centred on the grid rather than on the canvas origin. The
        two coincide until the stage travels far from wherever the frame was anchored
        -- moving to an offset FM mount steps 48 mm -- and an area left behind at the
        origin then frames empty space *and* stretches the content extent across the
        gap, which is what caps how far the view can zoom in: the limiter is relative
        to the content, so it read the span as 48 mm when what mattered was 0.1 mm.

        Re-declaring it re-frames the view, though, so it cannot simply follow every
        move -- a double-click to somewhere 100 µm away would throw the zoom away. It
        follows only once the grid has left the area outright, at which point what is
        framed and what is planned no longer overlap at all.
        """
        declared = self.canvas.canvas.world_extent
        if declared is None:
            return True  # nothing declared yet, so nothing describes the grid
        width, height, cx, cy = declared
        return abs(offset[0] - cx) > width / 2 or abs(offset[1] - cy) > height / 2

    def _toggle_tile_grid_panel(self) -> None:
        if not self.btn_tile_grid.isChecked():
            self.tile_grid_panel.hide()
            return

        self._fit_tile_grid_panel()
        self.tile_grid_panel.show()
        self.tile_grid_panel.raise_()

    def _fit_tile_grid_panel(self) -> None:
        """Size the tile grid panel to its contents and put it back in its corner.

        Both together, and both whenever the summary changes rather than only when the
        panel opens. The panel is a fixed width with a word-wrapped summary, so its
        height is however many lines that text takes -- and the text grows: dragging the
        grid adds a line saying how far it now sits from the stage. Sized once at open
        time, the panel would clip the extra lines.

        The move is not optional after a resize. The panel is anchored by its top-*right*
        corner, so its x is computed from its own width; leaving it alone would be fine
        for a height change and wrong the moment the width followed.
        """
        self.tile_grid_panel.adjustSize()
        # Anchored in global coordinates: the panel is a top-level tool window, so it
        # cannot be placed relative to the canvas in widget coordinates.
        canvas = self.canvas.canvas
        anchor = canvas.mapToGlobal(QPoint(canvas.width() - 8, 44))
        self.tile_grid_panel.move(anchor.x() - self.tile_grid_panel.width(), anchor.y())

    def set_image(self, image: FluorescenceImage) -> None:
        """Show an image at the stage position it was acquired at.

        Overlays are positioned against the canvas frame, not against whatever is
        currently displayed, so the image is recorded when it is set rather than fetched
        from the canvas later -- the canvas keeps channel stacks, not the
        `FluorescenceImage` and its metadata, and the metadata is what carries the pose.

        Each image gets a record, so several overviews on the canvas stay individually
        addressable. The in-progress preview never comes through here -- it goes
        straight to `canvas.set_channel` under its own key -- so it gets no record, and
        never appears in a list of what has been acquired.
        """
        record = self._record_for(image)
        if self._origin is None:
            self._origin = self._device_origin()
        self.canvas.set_composite_key(record.id)
        self.canvas.set_placement(self._offset_of(image))
        self.canvas.set_fm_image(image)
        # Re-asserted, not assumed: a reshaped frame is placed as a *new* artist, which
        # starts visible. Re-showing a hidden overview would otherwise bring it back
        # while its row still said it was hidden.
        self.canvas.canvas.set_image_visible(record.id, record.visible)
        self.overview_list.set_records(self.overviews())
        self._refresh_positions()
        self._refresh_stage_metadata()
        self._refresh_tile_grid()

    @staticmethod
    def _position_of(image: FluorescenceImage) -> Optional[FibsemStagePosition]:
        return getattr(image.metadata, "stage_position", None)

    def _record_for(self, image: FluorescenceImage) -> PlacedOverviewImageRecord:
        """The record *image* belongs to, creating one if it is new.

        Matched on the acquisition date when there is one, so showing the same image
        twice replaces it rather than drawing a second copy on top of itself -- the one
        property the old date-derived key had that is worth keeping. An image without a
        date gets a record of its own instead of joining every other dateless overview
        under a shared key, which is what the old fallback did.
        """
        stamp = getattr(image.metadata, "acquisition_datetime", None)
        if stamp:
            for record in self._records.values():
                if getattr(record.metadata, "acquisition_datetime", None) == stamp:
                    record.metadata = image.metadata
                    return record

        self._record_count += 1
        record = PlacedOverviewImageRecord(
            id=f"overview-{self._record_count}",
            label=self._label_for(image, self._record_count),
            metadata=image.metadata,
        )
        self._records[record.id] = record
        return record

    @staticmethod
    def _label_for(image: FluorescenceImage, index: int) -> str:
        """What a list calls this overview.

        `OverviewDestination` stamps the run's directory name onto the mosaic it saves,
        so a saved overview is named for the folder its tiles are in -- which is what
        someone looking for it on disk would search for. An unsaved one has no such
        name, and a number beats an empty row.
        """
        description = getattr(image.metadata, "description", None)
        return description or f"Overview {index}"

    def overviews(self) -> List[PlacedOverviewImageRecord]:
        """Every overview on the canvas, oldest first."""
        return list(self._records.values())

    def load_overview(self, path: str) -> Optional[PlacedOverviewImageRecord]:
        """Place a saved overview from *path*. None if it could not be read.

        Almost all of this is already done: `set_image` places an image by projecting
        through the image's *own* recorded geometry, so that one acquired under a
        configuration the instrument is no longer in lands where it was taken. A file
        off disk is that case exactly, and everything the projection needs -- the
        stage position's r and t, the hardware geometry, the pixel size -- survives the
        OME-TIFF intact.

        That includes `stage_position.name`, which `stitch_tileset` uses to carry the
        grid size, so a loaded overview's row shows its grid as well as its scale.

        Separate from the dialog so it can be called with a path -- by a test, by a
        host restoring a session, or by whatever knows an experiment's own overviews.
        """
        try:
            image = FluorescenceImage.load(path)
        except Exception as e:
            logging.error(f"Could not load an overview from {path}: {e}")
            notification_service.show_toast(
                f"Could not load that overview.\n{e}", "error"
            )
            return None

        # Placed even without a geometry, but said out loud. `_offset_of` falls back to
        # the origin, which puts it in the middle of the view rather than where it was
        # taken -- and an overview silently in the wrong place is worse than one you
        # have been told to distrust. An overview saved before geometry was recorded
        # is this case.
        if FMStageProjection.from_image(image) is None:
            logging.warning(
                f"Overview {path} has no recorded geometry; placing at the origin."
            )
            notification_service.show_toast(
                "That overview has no recorded geometry, so it cannot be placed where "
                "it was taken. Showing it at the canvas origin.",
                "warning",
            )

        self.set_image(image)
        record = self._record_for(image)
        logging.info(f"Loaded overview {record.label} from {path}")
        return record

    def _prompt_for_overview(self) -> None:
        """Ask for a file and load it. The dialog half of `load_overview`."""
        path = ui_utils.open_existing_file_dialog(
            msg="Select an overview to load",
            # Where this tab writes them, when a host has said. Opening at the last
            # place anything was saved beats opening at the home directory.
            path=self._save_directory or os.getcwd(),
            _filter="Fluorescence images (*.ome.tiff *.ome.tif *.tiff *.tif)",
            parent=self,
        )
        if not path:
            return
        self.load_overview(path)

    def set_overview_visible(self, record_id: str, visible: bool) -> bool:
        """Show or hide one overview. False if there is no such record.

        Hidden rather than removed, so it can come back without being re-acquired. The
        canvas skips drawing a hidden image entirely, so this also takes it out of the
        redraw cost -- which is the point on a canvas holding several.
        """
        record = self._records.get(record_id)
        if record is None:
            return False
        record.visible = visible
        self.canvas.canvas.set_image_visible(record_id, visible)
        # In place rather than a rebuild: this is usually the list telling us about its
        # own click, and replacing the row under the cursor mid-click would drop the
        # hover the buttons are showing for. The rows ignore a value they already have,
        # so the click that started this does not come back round.
        self.overview_list.refresh(self.overviews())
        return True

    def remove_overview(self, record_id: str) -> bool:
        """Drop one overview from the canvas for good. False if there is no such record.

        The record and the canvas artist go together: leaving either behind means a row
        with nothing under it, or pixels nothing can reach.
        """
        if self._records.pop(record_id, None) is None:
            return False
        self.canvas.canvas.remove_image(record_id)
        self.canvas.forget_overview(record_id)
        self.overview_list.set_records(self.overviews())
        return True

    def _offset_of(self, image: FluorescenceImage) -> Tuple[float, float]:
        """Where *image*'s centre sits relative to the canvas origin, in metres.

        Projected through the image's *own* geometry rather than the live one, which is
        the whole reason this is not :meth:`_offset_from_origin`: an image may have been
        acquired under a configuration the instrument is no longer in, and it has to be
        placed as it was taken.

        Falls back to the origin when it cannot be projected -- acquired before the
        geometry was recorded, or with no pose at all -- which puts it in the middle of
        the view rather than refusing to show it.
        """
        position = self._position_of(image)
        projection = FMStageProjection.from_image(image)
        if self._origin is None or position is None or projection is None:
            return (0.0, 0.0)
        try:
            # The image's own *geometry*, but the shared *pose* -- the same base
            # `_frame` places everything else against. Two different bases put the
            # image and the grid describing it in different places: a stale t = 0
            # origin against a stage at t = -180 landed them 300 um apart, mirrored
            # in y, which is a mosaic that disagrees with the grid that planned it.
            return projection.to_plane(position, self._posed(self._origin))
        except Exception as e:
            logging.debug(f"Could not place the image in stage space: {e}")
            return (0.0, 0.0)

    def _offset_from_origin(self, position: FibsemStagePosition) -> Tuple[float, float]:
        """Where a stage position sits relative to the canvas origin, in metres.

        The live-position counterpart of :meth:`_offset_of`, which answers the same
        question for an image out of its own metadata -- and has to, since an image may
        have been acquired under a configuration the instrument is no longer in.
        """
        frame = self._frame()
        if frame is None:
            return (0.0, 0.0)
        try:
            return frame.offset(position)
        except Exception as e:
            logging.debug(f"Could not place {position} in the canvas frame: {e}")
            return (0.0, 0.0)

    def add_settings_section(
        self, title: str, widget: QWidget, first: bool = True
    ) -> None:
        """Put a host's own controls in the settings column, under *title*.

        Part of the host contract, alongside `set_positions` and `set_save_directory`,
        and for the same reason: some of what belongs on this tab needs to know things
        this widget must not. A list of the experiment's lamellae is the case in point --
        its rows subscribe to `lamella.events.description` and read task and defect
        state, so it cannot be built here without dragging the whole application layer
        in behind it.

        A slot rather than the host laying out its own column beside this one, because
        the alternative reads as a third pane: the settings already have a column, and
        these controls belong *in* it, next to everything else that is about the run.

        `first` because a host's section is usually the subject of the tab rather than a
        footnote to it -- what you came to look at, above the parameters of the next
        acquisition. Appended before the trailing stretch either way, so the column
        stays top-aligned.
        """
        section = self._section(title, widget)
        if first:
            self._controls_layout.insertWidget(0, section)
        else:
            # -1 is the stretch added in `_init_ui`; stay above it.
            self._controls_layout.insertWidget(
                self._controls_layout.count() - 1, section
            )

    def set_positions(self, positions: List[FibsemStagePosition]) -> None:
        """Stage positions to mark on the overview, e.g. saved lamella positions.

        Names are carried onto the markers, so a caller does not have to keep a
        parallel list of labels in the order it happened to pass them.

        These are *fluorescence* poses. A lamella carries one of each, and the beam-side
        pose is a different place on this canvas -- on a compustage the stage flips
        between them -- so a host passing the wrong one would mark every position
        somewhere the sample is not. See `build_lamella_poses`.
        """
        self.session.set_positions(positions)

    def set_selected_position(self, name: Optional[str]) -> None:
        """Pick out one of the marked positions, or None for none.

        By name rather than index: the host's list and this one are rebuilt
        independently from the same experiment, and an index would silently point at a
        different lamella the moment one was removed.
        """
        self.session.select(name or None)

    @property
    def selected_position(self) -> Optional[str]:
        return self.session.selected

    def set_save_directory(self, path: Optional[str]) -> None:
        """Write acquired overviews under *path*, or None to keep them in memory only.

        Each run claims its own subdirectory under this one, named for when it started,
        so consecutive runs accumulate instead of overwriting each other -- pointing
        straight at an experiment directory and letting the runner write `tile-00-00`
        into it would mean every overview replaced the last.

        Told rather than discovered, because this widget knows nothing about
        experiments: it is handed a microscope, and its tests construct it that way.

        The host still decides the default, and the Output panel is where it lands --
        so being told a folder and choosing one are the same field, and a later
        `set_save_directory` (loading another experiment) replaces a choice rather than
        being ignored behind it. That is what the FIB/SEM tab does.
        """
        self.settings_widget.set_save_directory(path)

    @property
    def _save_directory(self) -> Optional[str]:
        """Where a run would write, as the Output panel has it."""
        return self.settings_widget.save_directory

    def select_ground_to_image(self) -> None:
        """Draw regions on a saved beam overview, and take the grid they produce.

        The overviews come off disk rather than from the FIB/SEM tab: that tab keeps
        pixels and a position, not the metadata a projection is read from, and going
        through the files means this works in a session that did not acquire them.
        """
        from fibsem.ui.fm.widgets.fm_sparse_selection_dialog import (
            FMSparseSelectionDialog,
            beam_overviews_in,
        )

        # Two different nothings, and telling them apart is the whole point of the
        # split. With no output folder there is nowhere to *look*, so "acquire one
        # first" is not merely unhelpful -- it is wrong advice, and it sends the reader
        # off to acquire an overview that will land somewhere this still cannot see.
        # Reported as a bug on exactly that confusion, from an app started without an
        # experiment.
        directory = self._save_directory
        if not directory:
            notification_service.show_toast(
                "No output folder set, so there is nowhere to look for FIB/SEM "
                "overviews. Open an experiment, or choose a folder in Output.",
                "warning",
            )
            return

        views = beam_overviews_in(directory, self.microscope)
        if not views:
            # Named, because the folder being the wrong one is the likeliest reason a
            # user who *has* acquired an overview is being told there are none.
            notification_service.show_toast(
                f"No FIB/SEM overviews found in {os.path.basename(directory.rstrip(os.sep))}. "
                "Acquire one on the FIB/SEM overview first.",
                "warning",
            )
            return

        selection = FMSparseSelectionDialog.choose(
            self.microscope,
            views,
            self.settings_widget.parameters,
            # The same two the confirmation dialog costs a run with, so the estimate
            # here and the one at Acquire cannot disagree.
            channel_settings=self.channels,
            zparams=self.settings_widget.z_parameters,
            parent=self,
        )
        if selection is None:
            return
        self.apply_sparse_selection(selection)

    def apply_sparse_selection(self, selection) -> None:
        """Take a completed selection: the grid, the mask, and where it is centred.

        The centre goes to `_target`, which already exists for a grid dragged off the
        stage position and is what `_grid_centre` hands the runner -- so the drawn grid
        and the acquisition agree without either being told about this feature.
        """
        self.settings_widget.parameters = selection.parameters
        self._target = selection.centre_position
        enabled = selection.parameters.n_enabled_tiles
        total = selection.parameters.rows * selection.parameters.cols
        # The Grid header already shows "28/66 tiles" whenever a run is sparse, so the
        # count is not repeated here -- what it cannot say is *where the mask came from*,
        # and that selecting again replaces rather than adds to it. There is no restore:
        # the regions end with the dialog.
        self.button_select_ground.setToolTip(
            f"{enabled} of {total} tiles came from a selection. "
            "Selecting again replaces it."
        )
        self._refresh_tile_grid()

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        """Allow or forbid starting work, without touching a run already in progress.

        For a host that owns the instrument for a while -- an AutoLamella workflow --
        and needs the tab to stop competing for it. Cancel stays live regardless, so a
        lock arriving mid-run cannot strand an acquisition with no way to stop it.

        Deliberately not `setEnabled(False)` on the whole widget: greying out the canvas
        would also stop you *reading* the overview you just acquired, and looking at it
        costs the workflow nothing.

        *reason* completes the sentence "Cannot move the stage while ___" when a move is
        refused. The widget cannot know why it was locked -- a workflow owning the
        instrument and the other overview being mid-tileset are the same `False` here --
        and a refusal naming the wrong one is barely better than one naming nothing.
        """
        self.session.set_interactive(enabled, reason)

    def set_origin(self, position: Optional[FibsemStagePosition]) -> None:
        """Anchor the canvas frame at *position*, or None to go back to automatic.

        The origin decides only where canvas zero sits -- everything is drawn relative
        to it, so any stage position serves. Left alone it is the FM device's origin
        (`_device_origin`): the place the objective looks at, configured rather than
        wherever the stage happened to be standing when the canvas first drew.

        Raises:
            ValueError: if anything has been placed. Images are positioned relative to
                the origin as it stood when they were added, so re-anchoring afterwards
                would leave every one of them describing a position it was never
                acquired at. Clear them first if that is really what you want --
                silently reprojecting them would be a different feature.
        """
        placed = self.canvas.placed_overviews()
        if placed:
            raise ValueError(
                f"Cannot re-anchor the canvas: {len(placed)} image(s) are already "
                f"placed against the current origin. Call clear_overviews() first."
            )
        self._origin = position
        self._refresh_current_position()
        self._refresh_positions()
        self._refresh_tile_grid()

    @property
    def origin(self) -> Optional[FibsemStagePosition]:
        """The stage position canvas zero corresponds to, once one has been fixed."""
        return self._origin

    def _projection(self, fresh: bool = False) -> Optional[FMStageProjection]:
        """The camera half of the frame, read once and kept.

        `FMStageProjection.from_microscope` reads `camera.resolution`, `camera.pixel_size`
        and `fm_image_geometry()`. On a TFS system the first two go through
        `camera.binning`, which is a device read inside an `active_channel()` scope -- so
        resolving a projection is several AutoScript round trips, and on the shared
        connection at that.

        `_frame` is called from the cursor readout and from the grid drag, both of which
        fire on **every mouse-move event**. Reading the instrument once per pixel of
        pointer travel, to render a text label, is what made the grid drag stutter on
        hardware; the tile-grid overlay's own docstring asks the host to keep that
        response cheap, and it had stopped being cheap.

        Caching is safe because nothing in it varies the way the pointer does.
        `FibsemHardwareGeometry` is system configuration -- camera tilt, shuttle
        pre-tilt, the flip, compustage or not -- and does not change within a session.
        Resolution and pixel size vary only with binning, which changes when somebody
        changes it.

        *fresh* re-reads, for the callers that need the current value rather than a
        current-enough one. The split is the same one `ObjectiveLens.position_changed`
        draws: **displays use the kept value, anything that commands the instrument
        reads the device.** Of the ten callers here exactly one commands anything --
        `_stage_position_at`, which turns a double-click into a stage move.
        """
        if fresh or self._cached_projection is None:
            self._cached_projection = FMStageProjection.from_microscope(self.microscope)
        return self._cached_projection

    def invalidate_projection(self) -> None:
        """Forget the kept projection, so the next frame re-reads it.

        Called wherever the camera geometry could have moved under us -- a reconnect, a
        binning change, the start of a run. Cheap: the next `_frame` pays one read.
        """
        self._cached_projection = None

    def _frame(self, fresh: bool = False) -> Optional[StageFrame]:
        """The mapping between stage space and the canvas, or None if it cannot be had.

        The one place the two meet. Everything drawn from a stage position goes through
        it -- marked positions, the stage and grid limits, the current pose, the planned
        grid -- and so does every click, so they share a frame by construction rather
        than by six callers agreeing to do the same arithmetic.

        Anchored on the canvas origin rather than on whichever image happens to be
        displayed, so a marker names the same piece of sample no matter what is on
        screen. The origin's rotation and tilt are what put the drawing in the current
        stage orientation -- t = -180 for FM.

        *fresh* forces the camera geometry to be re-read -- see :meth:`_projection`.
        """
        projection = self._projection(fresh=fresh)
        if projection is None:
            return None

        origin = self._origin or self._device_origin()
        if origin is None:
            return None
        self._origin = origin  # fix it now, so later drawing shares this frame

        return StageFrame(self.canvas.canvas, self._posed(origin), projection)

    def _device_origin(self) -> Optional[FibsemStagePosition]:
        """Where canvas zero goes when nothing has said: the FM device's origin.

        This canvas is the FM's view of the sample, so its zero is the place the
        objective looks at when the stage is at the FM -- `stage.devices.FM.origin`,
        which is configured, and the same every session, rather than wherever the stage
        was standing the first time anything drew, which on an offset mount is almost
        always the beams, far from anything this canvas shows. On a compustage the
        device origin is the objective's offset from the beams' coincidence point (zero
        unless configured), so zero is the holder's centre as the objective sees it.

        Only a *place*. The configuration declares x, and may declare y and z; an
        axis it leaves out reads as 0. The pose is not part of the anchor at all --
        `_posed` restates it from the stage on every draw, which is what keeps a
        click after Move to FM resolving through the FIB tilt rather than the one the
        stage had when the tab was built.

        Falls back to the stage's position for a microscope with no FM device
        declared, which is how widget tests built on a bare FM arrive here.
        """
        current = self._current_stage_position()
        try:
            origin = self.microscope.system.stage.devices["FM"].origin
        except (AttributeError, KeyError, TypeError):
            return current
        if current is None:
            return None
        return FibsemStagePosition(
            name="FM",
            x=origin.x or 0.0,
            y=origin.y or 0.0,
            z=origin.z or 0.0,
            r=current.r,
            t=current.t,
            coordinate_system=current.coordinate_system,
        )

    def _posed(self, origin: FibsemStagePosition) -> FibsemStagePosition:
        """The origin, re-stated in the pose the stage is actually in.

        Canvas zero is a *place* -- x, y, z -- and stays fixed, which is what lets
        images accumulate against a common frame. The rotation and tilt are not part of
        that anchor: they are how stage space maps onto the canvas at all, and they
        change when the stage re-poses.

        Carrying the pose the origin happened to be captured in was a real bug, and a
        dangerous one. The tab is built when a microscope connects, so the origin
        recorded whatever pose the stage was in then -- usually t = 0. Moving to the FM
        position afterwards left every click resolved through the old pose: the y
        component came out with the wrong sign, so the stage went to the wrong place,
        and the target inherited t = 0, so it also flipped 180 degrees getting there.
        """
        current = self._current_stage_position()
        if current is None:
            return origin
        return FibsemStagePosition(
            x=origin.x,
            y=origin.y,
            z=origin.z,
            r=current.r,
            t=current.t,
            coordinate_system=origin.coordinate_system,
        )

    def _toggle_overlays(self) -> None:
        """Show or hide the overlays popover, anchored under its button."""
        self.overlay_popover.set_open(self.btn_overlays.isChecked(), self.btn_overlays)

    def reset_context_overlay_defaults(self) -> None:
        """Re-resolve the holder-dependent overlay defaults, as the beam tab does:
        a slot calibrated mid-session turns the boundary and slot markers on
        without a reconnect, and only when the answer changed."""
        calibrated = stage_context.holder_is_calibrated(self.microscope)
        if calibrated == self._context_defaults_for_calibrated:
            return
        self._context_defaults_for_calibrated = calibrated
        for key, _label, shown in stage_context.context_overlay_entries(
            self.microscope
        ):
            self.overlay_controls.set_visible(key, shown)

    @ensure_main_thread
    def _on_holder_changed(self, _holder) -> None:
        """Slots calibrated, or another holder fitted: redraw where they are.

        Always, not only when the overlay defaults flip -- recalibrating a slot on a
        holder that was already calibrated changes nothing `reset_context_overlay_defaults`
        looks at, and moves the slot all the same.
        """
        self.reset_context_overlay_defaults()
        self._refresh_stage_metadata()

    def _refresh_stage_metadata(self) -> None:
        """Draw where the sample and the stage can physically go.

        The context an overview is read against: which grid you are on, how far from its
        centre, and how much travel is left. The same shapes the beam tab draws, from the
        same functions.
        """
        frame = self._frame()
        if frame is None:
            self.stage_overlay.set_shapes([])
            return
        self.stage_overlay.set_shapes(
            stage_context.context_shapes(
                self.microscope,
                frame,
                limits=self.overlay_controls.is_visible(stage_context.OVERLAY_LIMITS),
                boundaries=self.overlay_controls.is_visible(
                    stage_context.OVERLAY_BOUNDARIES
                ),
                slots=self.overlay_controls.is_visible(stage_context.OVERLAY_SLOTS),
                # This canvas is the FM's view: its markers are fluorescence poses,
                # so the holder's places are drawn where the FM sees them too.
                device="FM",
            )
        )

    def _refresh_current_position(self) -> None:
        """Mark where the stage is now."""
        frame = self._frame()
        position = self._current_stage_position()
        if frame is None or position is None:
            self.current_position_overlay.set_points([])
            return
        try:
            point = frame.to_canvas(position)
        except Exception as e:
            logging.debug(f"Could not mark the current stage position: {e}")
            self.current_position_overlay.set_points([])
            return
        # Unlabelled on purpose: the crosshair sits on top of the data, and a caption
        # riding along with it obscures the sample it is pointing at. What it is gets
        # said in the info bar below instead, which costs the image nothing.
        self.current_position_overlay.set_points([point], labels=[""])
        self._refresh_stage_info()

    def _refresh_stage_info(self) -> None:
        """Say where the stage is, in the canvas's bottom-left info bar.

        The marker shows *where*; this says the numbers, which is what you need to
        write one down or check you are where you meant to be. On the canvas rather
        than in the settings column because it describes what is being looked at.

        The objective position rides along because on a fluorescence system it is part
        of the pose -- focus depends on it, and it is not visible anywhere else on this
        tab. The milling angle deliberately does not: it is a beam-side quantity,
        meaningless while looking through the camera, and it belongs on the FIB/SEM
        overview if anywhere.

        The objective may be unavailable -- a microscope with none, or one that will not
        answer -- and is dropped on its own rather than losing the stage position with it.
        """
        position = self._current_stage_position()
        if position is None:
            self.canvas.canvas.set_info_text(None)
            return

        parts = [position.pretty]
        if self._objective_info is not None:
            parts.append(self._objective_info)

        self.canvas.canvas.set_info_text("   |   ".join(parts))

    def _set_objective_info(
        self, position: float, state: InsertableDeviceState
    ) -> None:
        """Render the info bar from values already in hand -- no device read.

        The path the signal takes. Split out of `_refresh_objective_info` so a move can
        update the bar from what it announced rather than by asking again.
        """
        self._objective_info = (
            f"objective {position * constants.SI_TO_MILLI:.3f} mm ({state.value})"
        )
        self._refresh_stage_info()
        self.settings_widget.set_objective_positions(
            position, self.fm.objective.focus_position
        )

    @ensure_main_thread
    def _on_objective_moved(
        self, position: float, state: InsertableDeviceState
    ) -> None:
        """Something moved the objective, and said where it ended up.

        Reached from psygnal, which fires on whichever thread did the moving -- a
        z-stack or an autofocus sweep runs on a worker -- so this touches widgets only
        because the decorator has already marshalled it.

        The values come from the signal rather than a fresh read, so this costs nothing
        and cannot disagree with the move that prompted it. Two device reads on every
        stage poll would each take the imaging channel on a shared connection.
        """
        self._set_objective_info(position, state)

    @ensure_main_thread
    def _on_transform_changed(self, transform) -> None:
        """The camera's image transform changed, so every projection here is stale.

        Not the tile *size*: `CameraImageTransform` is flips only and flips preserve the
        array shape, so `camera.resolution` and the field of view derived from it are
        untouched. The issue this closes assumed otherwise; what actually moves is the
        mapping. `fm_image_geometry()` carries the transform, `FMStageProjection` carries
        that geometry, and everything placed from a stage position goes through it --
        the current-position marker, marked positions, the stage and grid limits, the
        planned grid, and the canvas-to-stage direction a double-click moves on.

        The projection is kept rather than re-read on every use, so nothing else would
        pick the new transform up; hence a subscription.

        The grid is redrawn whatever it is pinned to, unlike on a stage move: that gating
        exists because a translation leaves a pinned grid where it was, and a transform
        change moves the mapping itself, so it moves everything.
        """
        self.invalidate_projection()
        self._refresh_current_position()
        self._refresh_positions()
        self._refresh_stage_metadata()
        self._refresh_tile_grid()

    def _refresh_objective_info(self) -> None:
        """Re-read the objective for the info bar. Touches hardware -- call sparingly.

        Not inside `_refresh_stage_info`, which runs on every stage poll: on a real
        system the FM and the beams share one connection, so reading the objective
        points it at the FM.

        A backstop rather than the usual path: `ObjectiveLens.position_changed`
        carries the values whenever *we* move the objective, and this covers the moves we
        do not make -- the microscope's own UI, or a routine driving the device -- at the
        two moments something external may have acted, on entry and when an acquisition
        or autofocus sweep finishes.
        """
        try:
            position, state = self.fm.objective.position, self.fm.objective.state
        except Exception as e:
            logging.debug(f"No objective position for the info bar: {e}")
            self._objective_info = None
            self._refresh_stage_info()
            return
        # `_set_objective_info` also feeds the start-position options, which say what
        # each one currently means -- one read serving both, since on a shared
        # connection two reads are two takes of the imaging channel.
        self._set_objective_info(position, state)

    def _objective_state(self) -> Optional[InsertableDeviceState]:
        """Whether the objective is inserted, or None if it cannot be asked.

        None rather than a guess: a driver that cannot answer is not the same as one
        answering "retracted", and refusing a run over a failed read would be the wrong
        way round -- the runner checks again, where it can raise.
        """
        try:
            return self.fm.objective.state
        except Exception as e:
            logging.debug(f"Could not read the objective state: {e}")
            return None

    def _refresh_objective_start_options(self) -> None:
        """Tell the settings panel where the objective is and where its saved focus is.

        Read here rather than by the settings widget, which takes no microscope -- that
        is what lets it be built and tested on its own.
        """
        try:
            self.settings_widget.set_objective_positions(
                self.fm.objective.position, self.fm.objective.focus_position
            )
        except Exception as e:
            logging.debug(f"No objective positions for the start options: {e}")

    def _current_stage_position(self) -> Optional[FibsemStagePosition]:
        """Where the stage is, polling the microscope only when nothing has said yet.

        `stage.position.changed` fires from a read or a move of the stage device, so
        it means "a read found it had moved" rather than "the stage moved". Polling on
        every use would therefore add a round trip per keystroke on a spin box without
        making the answer any fresher than the last poll anyone did -- so this reads
        what the signal delivered, and everything drawn from the stage position stays
        in agreement instead of drifting between callers who polled at different
        moments.
        """
        if self.session.stage_position is None:
            self.session.read_stage_position()
        return self.session.stage_position

    def _grid_centre(self) -> Optional[FibsemStagePosition]:
        """The stage position the next overview will be centred on.

        A target if one has been set by dragging the grid, otherwise wherever the
        stage is -- which is what the runner falls back to, so the drawn grid and the
        acquisition agree without either having to be told about the other.
        """
        return self._target or self._current_stage_position()

    def _grid_offset(self) -> Tuple[float, float]:
        """Where the planned grid's centre sits relative to the canvas origin, in metres.

        Falls back to the origin when the centre cannot be projected, which draws the
        grid in the middle of the frame rather than not at all.
        """
        frame = self._frame()
        centre = self._grid_centre()
        if frame is not None and centre is not None:
            try:
                return frame.offset(centre)
            except Exception as e:
                logging.debug(f"Could not place the planned grid: {e}")
        return (0.0, 0.0)

    def _sync_position_fov(self) -> None:
        """Size the marked positions' boxes to one camera frame.

        Off the kept projection rather than the camera: this runs on every position
        refresh, and reading `camera.resolution` / `camera.pixel_size` is two
        `active_channel()` scopes on the shared connection -- see :meth:`_projection`.
        Neither changes except with binning, which a refresh does not do.

        Leaves the boxes as they are when the camera geometry cannot be had, which
        before the first read means no box at all: a crosshair with no frame around it
        is honest, a frame at a guessed size is not.
        """
        projection = self._projection()
        if projection is None:
            return
        height, width = projection.shape
        pixel_size = projection.pixel_size
        for overlay in (self.position_overlay, self.selected_position_overlay):
            overlay.set_extent(width * pixel_size, height * pixel_size)

    def _refresh_positions(self) -> None:
        """Mark the stage positions in the canvas frame."""
        self._sync_position_fov()
        frame = self._frame()
        if not self.session.positions or frame is None:
            self.position_overlay.set_points([])
            self.selected_position_overlay.set_points([])
            return

        points, labels = [], []
        selected_points, selected_labels = [], []
        for position in self.session.positions:
            try:
                point = frame.to_canvas(position)
            except Exception as e:
                logging.debug(f"Cannot mark {position.name!r}: {e}")
                continue
            name = position.name or ""
            if name and name == self.session.selected:
                selected_points.append(point)
                selected_labels.append(name)
            else:
                points.append(point)
                labels.append(name)

        self.position_overlay.set_points(points, labels=labels)
        self.selected_position_overlay.set_points(
            selected_points, labels=selected_labels
        )

    # ── stage orientation ────────────────────────────────────────────────

    def at_acquisition_orientation(self) -> bool:
        """Whether the FM can image the sample from where the stage is.

        `READY` strictly -- place and pose both right -- because everything asking is
        about to drive the stage through, or write down, a frame built from the pose:
        the acquire gate walks a grid and stitches through it, marking writes the pose
        into a lamella. From a pose the objective cannot image from, nothing has
        checked those numbers.
        """
        return (
            self.microscope.get_device_imaging_state("FM") is DeviceImagingState.READY
        )

    def move_to_fm_device(self) -> None:
        """Drive the stage to where the FM can image, having asked first.

        The target is the *device*, not an orientation. On an offset mount the move is a
        traverse across the chamber, possibly with a re-pose at the beams first; the
        confirmation describes the actual route, since one button press should not grow
        into a multi-leg motion silently.

        A real stage move, so it gets the same confirmation as the other moves in this
        widget -- and the same worker, since `move_to_device` blocks for as long as
        the stage takes.
        """
        if self.session.running or self.fm.is_acquiring:
            notification_service.show_toast(
                "Cannot move the stage during an acquisition.", "warning"
            )
            return
        reply = QMessageBox.question(
            self,
            "Confirm Movement",
            f"Move the stage to the fluorescence microscope?\n\n"
            f"{self.microscope.explain_imaging_state('FM')}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            logging.info("Move to FM device cancelled by user")
            return

        self.status.setText("Moving to the FM…")
        worker = FunctionWorker(self._move_to_device_worker, "FM")
        worker.start(control=control_of(self.microscope), action="move stage to FM")

    def _move_to_device_worker(self, device: str) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        try:
            self.microscope.move_to_device(device)
            logging.info(f"Moved to the {device} device")
        except Exception as e:
            logging.error(f"Failed to move to the {device} device: {e}")
        # Not followed by a refresh: the move raises `stage.position.changed`, which
        # arrives at `_on_stage_moved` and re-derives everything the pose feeds.

    def _where_the_stage_is(self) -> str:
        """The current orientation as a noun phrase, for a sentence.

        "NONE" is what `get_stage_orientation` returns for a pose it cannot name, and
        "the NONE orientation" is not a thing to say to anyone.
        """
        orientation = self.microscope.get_stage_orientation()
        if orientation == "NONE":
            return "a position that is not a recognised orientation"
        return f"the {orientation} orientation"

    def _where_the_stage_needs_to_be(self) -> str:
        """The orientations an overview may be acquired from, as a noun phrase.

        Every one of them, not only the one the button goes to: on a system configured
        for more than one the stage may already be a shorter move from a different one,
        and naming only `default_orientation` would send the user further than they have
        to go.
        """
        allowed = self.microscope.system.stage.devices["FM"].available_orientations
        if not allowed:
            # An unconstrained device -- the pose is not what is wrong, so this string
            # is never shown for one; named defensively all the same.
            return "a pose the objective can image from"
        if len(allowed) == 1:
            named = allowed[0]
        else:
            named = f"{', '.join(allowed[:-1])} or {allowed[-1]}"
        return f"the {named} orientation"

    def _refresh_orientation_banner(self) -> None:
        """Say where the stage is, when it is not somewhere an overview can be acquired."""
        wrong = not self.at_acquisition_orientation()
        self.orientation_banner.setVisible(wrong)
        if wrong:
            self.orientation_notice.setText(
                f"Stage is at {self._where_the_stage_is()} — an overview needs to be at "
                f"{self._where_the_stage_needs_to_be()}."
            )
            # The device, by name: the button travels to the FM, whatever pose that
            # takes on this mounting.
            self.button_move_to_fm.setText("Move to FM")

    def _on_fm_acquiring_signal(self, acquiring: bool) -> None:
        """Called by psygnal, on whichever thread started or stopped. No widgets here."""
        self._fm_acquiring_changed.emit(acquiring)

    def _on_fm_acquiring_changed(self, acquiring: bool) -> None:
        """Something started driving the fluorescence microscope, or stopped.

        Only the controls need to know. What the FM is doing is not this widget's state
        -- it is a fact about the microscope that `_apply_enabled_state` reads, alongside
        the two that *are* this widget's -- so there is nothing to store here, only a
        reason to re-derive.
        """
        self._apply_enabled_state()
        # Something that just finished may have moved the objective -- a z-stack and an
        # autofocus sweep both do. One read when it stops, rather than one per stage poll
        # while it runs.
        if not acquiring:
            self._refresh_objective_info()

    # ── canvas interaction ───────────────────────────────────────────────

    def _on_session_signal(self, part: str) -> None:
        """Called by psygnal, on whichever thread changed the session."""
        self._session_changed.emit(part)

    def _on_session_changed(self, part: str) -> None:
        if part in (POSITIONS, SELECTION):
            self._refresh_positions()
        else:
            self._apply_enabled_state()

    def _on_stage_signal(self, position: FibsemStagePosition) -> None:
        """Called by psygnal, on whichever thread polled. Touches no widgets.

        The counterpart of :meth:`_on_tile_progress`, and a real method rather than the
        signal's `emit` for a reason beyond symmetry -- see the note where it is
        connected.
        """
        self._stage_moved.emit(position)

    def _on_stage_moved(self, position: FibsemStagePosition) -> None:
        """A poll found the stage somewhere new.

        The position comes from the signal rather than a fresh read: it is the value
        the poll that raised the signal saw, so taking it costs nothing and cannot
        disagree with what prompted the update.
        """
        reposed = self._pose_changed(position)
        self.session.stage_position = position
        self._refresh_current_position()
        # The gate and the banner answer "can the FM image from here", and on an offset
        # mount that changes with pure *translation* -- the 48.8 mm traverse to the FM
        # alters no rotation or tilt at all -- so they cannot hide behind `reposed`.
        # Gating them on it left the acquire button and the banner stale across the
        # whole traverse. Asked with the polled position, and re-derived only when the
        # answer actually changes.
        state = self.microscope.get_device_imaging_state("FM", position)
        imaging_changed = state is not self._last_imaging_state
        self._last_imaging_state = state
        # Everything else on the canvas is placed through a frame whose rotation and
        # tilt come from wherever the stage is (see `_posed`), so a re-pose moves all of
        # it. Redrawn only when the pose actually changes: the stage is polled
        # constantly, and a translation leaves every one of these exactly where it was --
        # the origin they are measured from moves with nothing.
        if reposed:
            self._refresh_positions()
            self._refresh_stage_metadata()
        if reposed or imaging_changed:
            self._refresh_orientation_banner()
            self._apply_enabled_state()
        # The grid follows the stage only when it is not pinned to a target and not
        # mid-run. During a run the stage visits every tile in turn, and a grid that
        # followed would crawl across the canvas describing nothing.
        if self._target is None and not self.is_acquiring:
            self._refresh_tile_grid(redraw_context=False)

    def _pose_changed(self, position: FibsemStagePosition) -> bool:
        """Whether *position* is at a different rotation or tilt from the last one.

        Only r and t, because only they enter the frame. Where the stage has travelled
        to is not part of how stage space maps onto the canvas -- canvas zero is the
        origin, and it does not move.
        """
        previous = self.session.stage_position
        if previous is None:
            return position.r is not None or position.t is not None
        for now, before in ((position.r, previous.r), (position.t, previous.t)):
            if now is None or before is None:
                if now is not before:
                    return True
                continue
            # Well below anything a user could mean and well above float noise on a
            # value that has been through a couple of transforms.
            if abs(now - before) > 1e-9:
                return True
        return False

    def _on_canvas_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Select a marked position if the click landed on one.

        A miss leaves the selection alone rather than clearing it, which is where this
        parts company with the minimap's interactive overlay. Clearing would have to
        travel: the host syncs its other lists from the selection, and their handlers
        ignore None -- so an empty click would blank this canvas and nothing else, which
        is worse than a selection that outstays its welcome. Nothing here is destroyed
        by keeping it.
        """
        name = self._position_at(x, y)
        if name is None:
            return
        self.set_selected_position(name)
        self.position_selected.emit(name)

    # Screen-space hit radius, matching `PointOverlay`'s. In pixels rather than stage
    # microns so that how close you have to click does not change with the zoom.
    PICK_RADIUS_PX = 12

    def _position_at(
        self, x: float, y: float, crosshair_only: bool = False
    ) -> Optional[str]:
        """The marked position under a canvas point, or None.

        A click hits a position if it lands inside that position's field-of-view box
        **or** within `PICK_RADIUS_PX` of its crosshair. The union rather than either
        alone, because neither is reliably the bigger target: the box wins once you are
        zoomed into a region, the fixed radius wins at whole-grid zoom where the box
        shrinks below it -- see `FieldOfViewOverlay.covers`.

        The radius is measured on screen, not in data units: at a wide zoom every marker
        would be within any sensible micron radius of the click, and at a tight one none
        would be. Nearest crosshair wins among the hits, which also settles overlapping
        boxes -- and lamellae closer together than one camera frame do overlap.
                *crosshair_only* drops the box and leaves the radius, for a caller that has to
        share the canvas with something else. The tile grid stands aside wherever this
        answers a name, and a field-of-view box is a large thing to reserve:
        it is a whole tile on the fluorescence tab by construction, and a whole tile on
        this one at any HFW of 100 um or under. Reserving it would leave tiles that
        cannot be toggled at all, with nothing on screen to say why -- where reserving
        only the crosshair costs at worst a click that toggles a tile you meant to
        select, which greys out visibly and undoes with one more click.
        """
        frame = self._frame()
        ax = getattr(self.canvas.canvas, "_ax", None)
        if frame is None or ax is None or not self.session.positions:
            return None
        try:
            transform = ax.transData
            click = transform.transform((x, y))
        except Exception as e:
            logging.debug(f"Could not resolve the click for picking: {e}")
            return None

        best_name, best_distance = None, float("inf")
        for position in self.session.positions:
            name = position.name
            if not name:
                continue
            try:
                centre = frame.to_canvas(position)
                point = transform.transform(centre)
            except Exception:
                continue
            distance = ((click[0] - point[0]) ** 2 + (click[1] - point[1]) ** 2) ** 0.5
            # `centre` is in canvas units and `point` in screen pixels: the box is a
            # fixed piece of sample, the radius a fixed piece of screen.
            if distance >= self.PICK_RADIUS_PX and (
                crosshair_only or not self.position_overlay.covers(centre, x, y)
            ):
                continue
            if distance < best_distance:
                best_name, best_distance = name, distance
        return best_name

    def _on_canvas_double_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Move the stage to the double-clicked point.

        Double-click rather than a single click, matching the minimap and the
        coincidence viewer: a single click is how the canvas is explored, and a stage
        that moved on every stray click would be unusable.
        """
        target = self._stage_position_at(x, y) if self._may_move() else None
        if target is None:
            return
        self.move_to(target)

    def _may_move(self) -> bool:
        """Whether driving the stage from this tab is allowed right now, and say if not.

        A click is a stage move computed through a frame built from the current pose,
        and from a pose the objective cannot image from there is nothing that has
        checked where it would send the stage.
        """
        # The lock covers the canvas as well as the buttons, as on the beam overview.
        refusal = (
            "Cannot move the stage during an acquisition."
            if self.is_acquiring
            else self.session.move_refusal()
        )
        if refusal is not None:
            notification_service.show_toast(refusal, "warning")
            return False
        if not self.at_acquisition_orientation():
            notification_service.show_toast(
                f"Cannot move the stage from {self._where_the_stage_is()} — the "
                f"overview needs to be at {self._where_the_stage_needs_to_be()}.",
                "warning",
            )
            return False
        return True

    def move_to(self, position: FibsemStagePosition) -> None:
        """Drive the stage to *position*, off the GUI thread.

        Public because a host drives it too -- picking a saved position out of a list is
        the same act as double-clicking where it is drawn, and they should not be able to
        disagree about whether a move is allowed or how it is reported.

        The limits are checked here rather than trusted from the caller: a stored
        position can be outside them on a system it was not recorded on.
        """
        if not self._may_move():
            return
        stage = self.microscope.stage
        limits = stage.position.limits if stage is not None else None
        if limits and not position.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return

        self.status.setText(f"Moving to {self._describe(position)}…")
        worker = FunctionWorker(self.session.move_to, position)
        worker.errored.connect(self._on_move_errored)
        worker.finished.connect(self._on_move_finished)
        worker.start(control=control_of(self.microscope), action="move stage")

    def _on_canvas_right_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Offer to put a position at the right-clicked point."""
        config = self._position_menu(x, y)
        if config is None:
            return
        ContextMenu(config, parent=self).show_at_cursor()

    def _position_menu(self, x: float, y: float) -> Optional[ContextMenuConfig]:
        """What right-clicking at a canvas point offers, or None to offer nothing.

        Separate from showing it because `show_at_cursor` runs a modal event loop, and
        everything worth checking -- whether the menu appears at all, what it offers,
        and which position each entry would request -- is decided here. A test that had
        to open the menu could only hang.

        The widget creates nothing itself: each entry emits a request and lets a host
        that owns an experiment decide. So the menu appears whenever the point is
        somewhere the stage could go, even with nobody listening -- there is nothing
        here that could tell the difference, and a menu that stayed shut because no host
        was connected would make the standalone app behave differently for no visible
        reason.
        """
        if self.session.running or self.is_acquiring or not self.session.interactive:
            # `_running` and `_interactive` are the two facts `_apply_enabled_state`
            # gates the controls on, and this is gated on the same two for the same
            # reason: marking is cheap in itself, but a host that has taken the
            # instrument is usually a workflow iterating `experiment.positions`, and
            # adding one underneath it is not a thing to do quietly. `is_acquiring`
            # comes along because the double-click asks it, and a right-click that was
            # allowed where a double-click was refused would just be confusing.
            notification_service.show_toast(
                "Cannot mark positions while an acquisition is running.", "warning"
            )
            return None
        # Marking has to survive being written down: the position becomes a lamella's
        # *fluorescence* pose, so it must be one the objective actually images from --
        # however right it looked on screen.
        if not self.at_acquisition_orientation():
            notification_service.show_toast(
                f"Move to the fluorescence position before marking on the overview "
                f"(the stage is at {self.microscope.get_stage_orientation()}).",
                "warning",
            )
            return None

        target = self._stage_position_at(x, y)
        if target is None:
            return None

        config = ContextMenuConfig()
        config.add_action(
            "Add Position Here",
            callback=lambda: self.position_add_requested.emit(target),
            tooltip=f"Add a position at {self._describe(target)}",
        )
        selected = self.session.selected
        if selected:
            config.add_action(
                f"Move Selected Position Here ({selected})",
                callback=lambda: self.position_move_requested.emit(selected, target),
                tooltip=f"Move {selected} to {self._describe(target)}",
            )
        return config

    def _stage_position_at(self, x: float, y: float) -> Optional[FibsemStagePosition]:
        """The stage position a canvas point names, or None if it is not usable.

        Shared by clicking to move and right-clicking to mark, so the two cannot
        disagree about where a point is -- which they would eventually, being the same
        arithmetic written twice.

        Outside the limits counts as not usable for both: the stage cannot go there, so
        it is neither somewhere to move nor somewhere to put a lamella.

        Reads the camera geometry rather than using the kept value: this is the one
        caller whose answer drives the instrument, and a stale binning here would send
        the stage somewhere other than where the click was.
        """
        frame = self._frame(fresh=True)
        if frame is None:
            return None
        try:
            target = frame.to_stage(x, y)
        except Exception as e:
            logging.debug(f"Could not resolve the clicked position: {e}")
            return None

        stage = self.microscope.stage
        limits = stage.position.limits if stage is not None else None
        if limits and not target.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return None
        return target

    def _on_move_errored(self, error: object) -> None:
        """The stage did not get there. Say that, rather than leaving "Moving to …" up."""
        self._move_failed = True
        self.status.setText(f"Could not move the stage: {error}")
        notification_service.show_toast("Could not move the stage.", "error")

    def _on_move_finished(self) -> None:
        """Always runs, after `errored` when there was one.

        A failure has already put its own message up and that should stand; only a
        success replaces the "Moving to …" line. What it is replaced with is where the
        stage actually got to, read back by the worker rather than assumed from the
        target.
        """
        if self._move_failed:
            self._move_failed = False
            return
        position = self._current_stage_position()
        if position is None:
            self.status.setText("Moved.")
            return
        self.status.setText(f"At {self._describe(position)}")

    def _on_grid_move(self, x: float, y: float) -> None:
        """The grid was dragged: plan the next overview around the point it landed on.

        Deliberately does *not* move the stage. Setting up a run and driving the
        instrument are separate acts, and a drag is exploratory -- you push the grid
        around to see what it would cover. The stage goes there when the acquisition
        does.
        """
        frame = self._frame()
        if frame is None:
            return
        try:
            self._target = frame.to_stage(x, y)
        except Exception as e:
            logging.debug(f"Could not resolve the dragged grid position: {e}")
            return
        self.tile_grid_panel.set_centre_enabled(True)
        self._refresh_tile_grid()
        self._update_grid_summary()

    def clear_target(self) -> None:
        """Plan the next overview around the stage position again."""
        if self._target is None:
            return
        self._target = None
        self.tile_grid_panel.set_centre_enabled(False)
        self._refresh_tile_grid()
        self._update_grid_summary()

    def _on_cursor_moved(self, x: Optional[float], y: Optional[float]) -> None:
        """Report the stage position under the cursor, or hide once it leaves."""
        if x is None or y is None:
            self._set_cursor_readout("")
            return
        frame = self._frame()
        if frame is None:
            self._set_cursor_readout("")
            return
        try:
            self._set_cursor_readout(self._describe(frame.to_stage(x, y)))
        except Exception as e:
            logging.debug(f"Could not resolve the cursor position: {e}")
            self._set_cursor_readout("")

    def _set_cursor_readout(self, text: str) -> None:
        """Put the readout in the canvas status zone, or give the zone back when empty.

        Where a double-click would take the stage, read off continuously rather than on
        demand: the canvas is a map, and a map you have to click to get a coordinate out
        of cannot be used to decide whether to click.

        Through the canvas's status zone rather than a label of this widget's own. The
        zone shows one thing at a time, so they cannot collide: the
        readout outranks the grid hint while the pointer is over the canvas, and a focus
        flash outranks both.

        Empty means the pointer has left the canvas, and the zone then falls back to the
        grid hint on its own -- so an empty readout is a release rather than a blank
        plaque left floating over the data.
        """
        self.canvas.canvas.set_status_readout(text or None)

    @staticmethod
    def _describe(position: FibsemStagePosition) -> str:
        """A stage position as microns, for a readout.

        z included because it is not decoration: on an offset mount a sideways move in
        the image carries a real z component, and a readout that hid it would make the
        focal-plane change invisible.
        """
        return (
            f"x {position.x * constants.SI_TO_MICRO:.1f}  "
            f"y {position.y * constants.SI_TO_MICRO:.1f}  "
            f"z {position.z * constants.SI_TO_MICRO:.1f} µm"
        )

    def _on_grid_resize(self, rows: int, cols: int) -> None:
        """An edge of the grid was dragged.

        Writes to the settings widget, which owns rows and columns, for the same reason
        a tile click does: the canvas is a view of that state, not a second copy.
        """
        # One call rather than two spin boxes: a drag emits on every motion event, and
        # setting them separately would refresh twice per step and pass through a grid
        # size that was never requested.
        self.settings_widget.set_grid_size(rows, cols)

    def _refresh_grid_hint(self, visible: bool) -> None:
        """Say what the grid can be done to, while it is on screen.

        The gestures are otherwise only discoverable by accident: the cursor names the
        edge drag and the paint modifier, but nothing suggests trying either. Tied to
        the grid's visibility rather than set once, so hiding the grid takes its
        instructions with it. A focus flash borrows the same zone and gives it back.
        """
        self.canvas.canvas.set_hint(
            "click a tile to skip it  ·  Shift+drag to sweep" if visible else None
        )

    def _on_tile_toggled(self, row: int, col: int, enabled: bool) -> None:
        """A tile was clicked or swept on the canvas.

        The mask belongs to the settings widget, so this writes there and lets the
        resulting `changed` redraw the overlay -- rather than updating the overlay
        directly, which would leave the two views to drift apart on any path that
        touched only one of them.
        """
        mask = self.settings_widget.tile_mask
        parameters = self.settings_widget.parameters
        if mask is None:
            mask = [[True] * parameters.cols for _ in range(parameters.rows)]
        if not (0 <= row < len(mask) and 0 <= col < len(mask[row])):
            return

        mask[row][col] = enabled
        self.settings_widget.set_mask(mask)

    def _on_settings_changed(self) -> None:
        if self.is_acquiring:
            return
        self._refresh_tile_grid()
        self._update_grid_summary()
        self._apply_enabled_state()
        # Only own the status line while it has something of its own to say. Re-enabling
        # the controls at the end of a run makes children emit `changed`, which landed
        # here and wiped the result message the moment it was set.
        if self.settings_widget.parameters.n_enabled_tiles == 0:
            self.status.setText("No tiles selected.")
        elif self.status.text() == "No tiles selected.":
            self.status.setText("")

    def _apply_enabled_state(self) -> None:
        """One place decides what is clickable, from five independent facts.

        A run in progress, a host that has locked the tab, something else driving the
        FM, a stage posed somewhere an overview cannot be acquired from, and a grid with
        nothing selected each disable different things, and they overlap: a workflow
        finishing while an overview runs must not re-enable Acquire, and a run finishing
        inside a locked tab must not either. Two places each setting the buttons from
        their own half of the truth is how a control gets stuck; deriving all of it from
        all five every time is the version that cannot.

        `fm.is_acquiring` covers this widget's own run too, since it marks the FM for
        the length of one -- harmless, because `_running` is true throughout that.
        """
        idle = (
            self.session.interactive
            and not self.session.running
            and not self.fm.is_acquiring
        )
        has_tiles = self.settings_widget.parameters.n_enabled_tiles > 0
        # Acquisition only. Display and navigation stay live from any pose: looking at an
        # overview that has already been acquired is harmless wherever the stage is, and
        # the settings panel stays editable so a run can be set up while the stage is on
        # its way.
        self.button_acquire.setEnabled(
            idle and has_tiles and self.at_acquisition_orientation()
        )
        # Cancel is deliberately not gated on `_interactive`: a host locking the tab
        # mid-run must not take away the only way to stop the stage. Once a cancel has
        # been asked for, the button goes and stays away.
        self.button_cancel.setEnabled(self.session.running and not self._cancelling)
        self.settings_widget.setEnabled(idle)
        self.channel_widget.setEnabled(idle)

    # ── acquisition ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Confirm, then run the overview on a worker thread."""
        if self.is_acquiring:
            logging.warning("An overview acquisition is already running.")
            return
        # A run plans a grid and drives the stage from the camera geometry, so it starts
        # from a freshly read one rather than whatever the last drawing kept. The natural
        # boundary: binning cannot change mid-run, and it is the only thing that moves.
        self.invalidate_projection()
        # `is_acquiring`, not `is_streaming`: a z-stack or an autofocus sweep in the
        # control widget is not a stream, and this widget's own tileset must be visible
        # to that one.
        if self.fm.is_acquiring:
            reason = self.fm.acquiring_reason or "another acquisition"
            logging.warning(f"Cannot acquire an overview: the FM is in use ({reason}).")
            notification_service.show_toast(
                f"The fluorescence microscope is in use ({reason}). "
                f"Wait for it to finish before acquiring an overview.",
                "warning",
            )
            return
        # Not only on the button, which `_apply_enabled_state` disables from the wrong
        # pose -- the button is not the guard, and a host calling this directly, or a
        # stage that moved between the click and here, is exactly what this is for.
        if not self.at_acquisition_orientation():
            where, needed = (
                self._where_the_stage_is(),
                self._where_the_stage_needs_to_be(),
            )
            logging.warning(
                f"Cannot acquire an overview: the stage is at {where}, and an overview "
                f"needs to be at {needed}."
            )
            notification_service.show_toast(
                f"Move the stage to {needed} before acquiring an overview "
                f"(it is at {where}).",
                "warning",
            )
            return

        parameters = self.settings_widget.parameters
        channels = self.channels
        if not channels:
            self.status.setText("No channels enabled.")
            return

        # Both of these are checked in the runner too, for a caller driving it directly.
        # Here as well because by then the dialog has said yes and the stage is moving,
        # so the message arrives as a failed run rather than as an answer to the
        # question that was just asked.
        state = self._objective_state()
        if state is not None and state is not InsertableDeviceState.INSERTED:
            message = (
                f"The objective is {state.value}, so an overview cannot be acquired. "
                f"Insert it and try again."
            )
            logging.warning(message)
            notification_service.show_toast(message, "warning")
            self.status.setText(f"Objective is {state.value}.")
            return

        if (
            parameters.objective_start is ObjectiveStartPosition.FOCUS
            and self.fm.objective.focus_position is None
        ):
            message = (
                "This overview is set to start from the saved focus position, but none "
                "has been saved. Set one with 'Set Focus Position', or start from the "
                "current position instead."
            )
            logging.warning(message)
            notification_service.show_toast(message, "warning")
            self.status.setText("No focus position saved.")
            return

        zparams = self.settings_widget.z_parameters
        # Method, channel and sweep passes all come from the settings panel now,
        # rather than being defaulted with only the channel filled in. Resolved before
        # the dialog, because the dialog reports what will run.
        autofocus_settings = self.settings_widget.autofocus_settings

        # Read now rather than reusing whatever the settings panel last showed: the
        # objective may have been driven somewhere since, and this dialog is the last
        # thing said before the run commits.
        self._refresh_objective_start_options()

        dialog = FMOverviewConfirmationDialog(
            parameters=parameters,
            channel_settings=channels,
            zparams=zparams,
            tile_fov=self.settings_widget._tile_fov,
            autofocus_settings=autofocus_settings,
            objective_current=self.settings_widget._objective_current,
            objective_focus=self.settings_widget._objective_focus,
            # Read the same way `_sync_tile_fov` already does, and only when the dialog
            # opens rather than on every refresh. Best effort: a disk estimate is not a
            # reason to refuse to show the dialog.
            tile_resolution=self._camera_resolution(),
            save_directory=self._save_directory,
            unreachable=self._unreachable(parameters),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled before starting")
            return

        self._cancelling = False
        self._set_running(True)
        self._run_settings = dict(
            channels=channels,
            parameters=parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            # None means "where the stage is", which the runner resolves itself. The
            # stage still returns to where it started afterwards -- the target is the
            # grid's centre, not a new home.
            centre_position=self._target,
            path=self._save_directory or os.getcwd(),
            name=self.settings_widget.basename,
        )
        self._worker = FunctionWorker(self._acquire_worker)
        self._worker.start(
            control=control_of(self.microscope), action="acquire FM overview"
        )

    def _unreachable(
        self,
        parameters: OverviewParameters,
        tiles: Optional[List[TilePosition]] = None,
    ) -> List[Tuple[int, int]]:
        """Which tiles of the planned run the stage cannot travel to.

        The runner asks this too, but only once the worker has started -- after the
        dialog has been accepted, a directory has been made and the stage has begun
        moving. Asked here it is refused while the grid can still be moved or the
        offending tiles masked off. It matters more on this side than on the beam one:
        a compustage travels +/-999.9 um in x and only +/-377.8 um in y, so a grid that
        looks square and central can be unreachable top and bottom.

        Through the kept projection, for the same reason `_refresh_tile_grid` uses it
        rather than the camera -- reading the geometry sets the shared imaging view and
        puts it back, which is visible in the microscope's own UI.

        Best effort, like `_camera_resolution` beside it: a check that cannot run leaves
        the dialog without the warning rather than refusing to open it. The runner keeps
        the authoritative refusal, so being wrong here fails open. The tile field of
        view comes from the settings panel while the runner reads its own -- they track
        the same camera, and a disagreement between them is exactly the mild direction.

        Also asked on the drag path, where the grid it flags can still be moved --
        `_refresh_tile_grid` passes the tiles it has just built rather than having them
        computed twice.
        """
        try:
            fov = self.settings_widget._tile_fov
            projection = self._projection()
            centre = self._grid_centre()
            stage = self.microscope.stage
            limits = stage.position.limits if stage is not None else None
            if fov is None or projection is None or centre is None or not limits:
                return []
            height, width = projection.shape
            if tiles is None:
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
            return unreachable_tiles(
                tiles,
                parameters.tile_order,
                lambda dx, dy: projection.from_plane(dx, dy, centre),
                limits,
            )
        except Exception as e:
            logging.debug(f"Could not check the grid against the stage limits: {e}")
            return []

    def _camera_resolution(self) -> Optional[tuple]:
        """The camera's pixel dimensions, or None if it cannot be asked right now."""
        try:
            return tuple(self.fm.camera.resolution)
        except Exception as e:
            logging.debug(f"Could not read the camera resolution: {e}")
            return None

    def _acquire_worker(self) -> None:
        """Runs off the GUI thread. Only signals may cross back.

        Through the imaging service, the path the agent's overviews take: it holds
        the FM for the run, stamps the name, and reports the save and the end on
        ``tiled_progress``. Each tile is written as it lands, so a cancelled run
        keeps what it got.
        """
        try:
            answer = self.microscope.imaging.fm_acquire_overview(**self._run_settings)
            if answer["status"] == "cancelled":
                logging.info("Overview acquisition cancelled")
        except Exception as e:
            logging.error(f"Overview acquisition failed: {e}", exc_info=True)
            self._run_failed.emit(str(e))

    def _on_run_failed(self, error: str) -> None:
        # Already finished when the run got far enough to report its failure.
        if self.session.running:
            self._finish(TiledStatus.FAILED, error)

    def _on_acquired(self, mosaic: Any, path: Optional[str]) -> None:
        """Keep the saved mosaic and say it arrived, before the run reports finished:
        the finished report places `_mosaic` and names `_saved_path`, and a consumer
        of `overview_acquired` may want the file. Called by psygnal on the acquiring
        thread, for every overview the imaging service takes: only this tab's
        fluorescence run is kept."""
        if not self.session.running or not isinstance(mosaic, FluorescenceImage):
            return
        self.session.mosaic = mosaic
        self._saved_path = path
        self.overview_acquired.emit(mosaic)

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logging.info("Cancelling overview acquisition")
        self._cancelling = True
        self.microscope.imaging.stop()
        self._apply_enabled_state()
        self.status.setText("Cancelling…")

    def _set_progress_visible(self, visible: bool) -> None:
        """Show the progress bars only while there is progress to show.

        They spend most of their life empty. Stacked in the settings column nothing sits
        beside them, so they can come and go without moving anything.
        """
        self.progress_tiles.setVisible(visible)
        self.progress_tile_detail.setVisible(visible)

    def _set_running(self, running: bool) -> None:
        self.session.set_running(running)
        self.acquiring_changed.emit(running)
        if running:
            # Cleared at the start, so a run that fails to save cannot inherit the
            # previous run's path and report itself saved.
            self._saved_path = None
            # Left empty rather than shown as indeterminate, which would paint a full
            # bar before a single tile had been acquired.
            self.progress_tiles.reset()
            self.progress_tile_detail.reset()
            self.status.setText("Starting…")
            # The framing you pressed Acquire with is the framing you keep. This tab
            # meets it once rather than twice -- one composite under one key while the
            # run goes -- but the end is the same swap: the stitch is placed and the
            # preview removed, two extent changes at the moment there is finally
            # something worth looking at. Taken back by "reset view".
            self.canvas.canvas.auto_fit = False

        # After the resets above, not before: `FibsemProgressWidget.reset()` hides
        # itself, so showing the bars first and resetting them second leaves them
        # hidden for the whole run.
        self._set_progress_visible(running)

    # ── progress ─────────────────────────────────────────────────────────

    def _on_tile_progress(self, payload: TiledProgress) -> None:
        """Called by psygnal, on whichever thread emitted. Touches no widgets."""
        self._tile_progress_received.emit(payload)

    def _on_fm_progress(self, payload: FluorescenceAcquisitionProgress) -> None:
        """Called by psygnal, on whichever thread emitted. Touches no widgets."""
        self._fm_progress_received.emit(payload)

    # The states this bar can render. `FINISHED` is deliberately absent: it describes
    # the acquisition ending rather than progress within it, and drawing it here would
    # be a bar with nothing in it.
    _DETAIL_STATUSES = (
        FluorescenceAcquisitionStatus.ACQUIRING_CHANNELS,
        FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
        FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
    )

    def _apply_fm_progress(self, report: FluorescenceAcquisitionProgress) -> None:
        """The tile currently being taken, on the detail bar.

        Runs on the GUI thread, queued via `_fm_progress_received`. The detector's
        signal reports whatever it is doing, tileset or not, so the states this bar can
        render are named rather than assumed.
        """
        if report.status in self._DETAIL_STATUSES:
            self.progress_tile_detail.update_progress(self._tile_detail_update(report))

    @staticmethod
    def _tile_detail_update(
        report: FluorescenceAcquisitionProgress,
    ) -> ProgressUpdate:
        """Progress within the tile currently being acquired.

        A z-stack counts planes and a plain multi-channel acquisition counts channels,
        so the same bar reads sensibly either way rather than sitting empty whenever
        z-stacking happens to be off.

        The z branch is taken by *two* statuses, which is the whole reason this contract
        is one record rather than a type per acquisition routine.
        """
        channel = report.channel or ""
        if report.zlevel and report.total_zlevels:
            if report.status is FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS:
                # Say which pass, so a coarse sweep followed by a fine one does not
                # look like the same bar inexplicably starting over.
                total_passes = report.total_passes or 1
                which = (
                    f" {report.pass_index or 1}/{total_passes}"
                    if total_passes > 1
                    else ""
                )
                return ProgressUpdate.numeric(
                    current=report.zlevel,
                    total=report.total_zlevels,
                    message=f"{channel} focus{which}",
                )
            return ProgressUpdate.numeric(
                current=report.zlevel,
                total=report.total_zlevels,
                message=f"{channel} z-stack",
            )
        return ProgressUpdate.numeric(
            current=report.channel_index or 1,
            total=report.total_channels or 1,
            message=f"{channel} channels",
        )

    # This widget's words for the states a fluorescence *tileset* run passes through.
    # The states with nothing worth saying are absent on purpose: `.get` returning None
    # is what clears the label rather than leaving a stale one up.
    _STATUS_LABELS = {
        TiledStatus.MOVING: "Moving stage…",
        TiledStatus.STITCHING: "Stitching tiles…",
        TiledStatus.SAVING: "Saving overview…",
    }

    def _apply_tile_progress(self, event: TiledProgress) -> None:
        """The run as a whole, on the tile bar.

        Runs on the GUI thread, queued via `_tile_progress_received`.
        """
        if event.modality != MODALITY_FLUORESCENCE:
            # A beam run, on the signal this shares with the beam tiler. Checked before
            # anything else, terminal states included: the two tilers write into
            # different canvases and different bars.
            return

        if event.status.is_terminal:
            if event.status is TiledStatus.FINISHED and event.warning:
                # Kept in the notification history: it is about the instrument's
                # calibration, not this one image, and outlives the toast.
                notification_service.show(event.warning, "info")
            self._finish(event.status, event.error)
            return

        label = self._STATUS_LABELS.get(event.status)
        if label is not None:
            # Deliberately not `indeterminate`: that paints a *full* bar with a spinner,
            # so every stage move looked like the run had just completed. The bar keeps
            # the last tile count -- still true between tiles, and while the mosaic is
            # being stitched and written -- and the transient state goes to the label.
            self.status.setText(label)
            return

        self.status.setText("")

        if event.status is TiledStatus.TILE_COLLECTED and event.preview is not None:
            # Deliberately does *not* clear the within-tile bar, which would flicker at
            # every tile boundary. The next tile's first report overwrites it anyway.
            self._show_preview(event.preview)

        if event.completed is None or not event.total:
            # An announcement rather than a progress report: the report that says which
            # tile is starting carries no counts, because emitted *before* the tile it
            # could only ever be one short. It has already done its job above, clearing
            # the label the stage move put up.
            return

        # The final report of a run gives 0 remaining and so renders as a plain count --
        # `FibsemProgressWidget` picks the shape from `remaining_seconds > 0`, not from
        # the caller. That is the run finishing rather than a flicker.
        remaining = event.estimated_remaining_seconds
        # The widget renders the count itself, so the message says what is being counted
        # and nothing more -- otherwise it reads "Tile 4/9 — 4/9".
        message = "Tiles"
        if remaining:
            self.progress_tiles.update_progress(
                ProgressUpdate.combined(
                    current=event.completed,
                    total=event.total,
                    remaining_seconds=remaining,
                    total_seconds=event.estimated_total_seconds,
                    message=message,
                )
            )
        else:
            self.progress_tiles.update_progress(
                ProgressUpdate.numeric(
                    current=event.completed, total=event.total, message=message
                )
            )

    def _show_preview(self, preview: FluorescenceImage) -> None:
        """Paint the mosaic-so-far onto the canvas.

        The preview is a `FluorescenceImage` rather than a bare array plus a stride, so
        everything needed to place it arrives with it:

        * where -- `metadata.stage_position`, instead of reaching into the runner for
          the centre it resolved when the run started;
        * at what scale -- `metadata.pixel_size_x`, with the preview's decimation already
          folded in by the producer, instead of `self._projection().pixel_size * stride`;
        * which channels -- `metadata.channels`, instead of this widget's own list.

        Dropping the `_projection()` call is the part that matters beyond tidiness. It
        reads from the camera under an `active_channel()` scope, and this runs once a
        tile *during* a run, so the display was taking the shared channel from the
        acquisition that was using it.
        """
        try:
            planes = np.asarray(preview.data)
            if planes.ndim == 2:
                planes = planes[np.newaxis]

            # Where, and at what scale, *before* any pixels: `set_channels` composites
            # and places immediately, so anything established after it applies a tick
            # late -- the first frame of a run would land under the previous run's key
            # at the previous run's pixel size.
            #
            # Its own key, so the in-progress preview neither replaces a finished
            # overview nor survives as one: it is swapped for the real stitch at the end.
            self.canvas.set_composite_key(PREVIEW_KEY)
            centre = preview.metadata.stage_position
            self.canvas.set_placement(
                self._offset_from_origin(centre) if centre is not None else (0.0, 0.0)
            )
            pixel_size = preview.metadata.pixel_size_x
            if not pixel_size:
                return
            self.canvas.set_pixel_size(pixel_size)

            # This run's channels, and only these. `set_channel` upserts, so a channel
            # switched off since the last run would otherwise keep its layer -- still
            # holding the previous overview's pixels -- and be blended into this one.
            channels = preview.metadata.channels
            self.canvas.retain_channels([channel.name for channel in channels])
            # One composite for the whole update, not one per channel.
            self.canvas.set_channels(
                [
                    (channel.name, plane, channel.color)
                    for channel, plane in zip(channels, planes)
                ]
            )
        except Exception as e:
            logging.debug(f"Could not display the overview preview: {e}")

    def _describe_save(self) -> Tuple[str, str]:
        """Status suffix and tooltip for whether the overview reached disk.

        A mosaic that did not land is worth saying: otherwise a failed save reads
        exactly like a successful one.
        """
        if self._saved_path:
            return "  ·  saved", self._saved_path
        return "  ·  not saved", "Could not be written to disk — see the log."

    def _finish(self, status: TiledStatus, error: Optional[str]) -> None:
        # Hides both bars. The per-tile one stays hidden -- there is no tile in progress
        # to describe -- but the overall bar is shown again below whenever it has a
        # terminal state worth reading: `FibsemProgressWidget` paints finished and
        # failed differently, and that colour is the at-a-glance answer.
        self._set_running(False)
        self._worker = None
        self.progress_tile_detail.reset()

        if status is TiledStatus.FINISHED and self.session.mosaic is not None:
            # Swap the decimated preview for the real thing. Dropped rather than left
            # underneath: it covers the same ground at a coarser scale, so keeping it
            # would only be a blurred copy hidden behind the stitch.
            self.set_image(self.session.mosaic)
            self.canvas.canvas.remove_image(PREVIEW_KEY)
            shape = self.session.mosaic.data.shape
            suffix, tooltip = self._describe_save()
            self.status.setText(
                f"Overview acquired — {shape[-1]} × {shape[-2]} px{suffix}"
            )
            self.status.setToolTip(tooltip)
            self.progress_tiles.update_progress(ProgressUpdate.done())
            self.progress_tiles.show()
        elif status is TiledStatus.CANCELLED:
            self.status.setText("Cancelled. Tiles acquired so far are still shown.")
            # Nothing to show: a cancel has no terminal state of its own, and the status
            # line already says what happened.
            self.progress_tiles.reset()
        else:
            self.status.setText(f"Failed: {error}" if error else "Failed.")
            # Failed, not done: the widget paints these differently, and a failure
            # showing a full green bar reads as success in everything but the text.
            # The bar carries the *fact*, not the reason -- a stage-limits rejection
            # names every offending tile, and a bar is a worse place to read that than
            # the status line above it, which elides and keeps the whole thing in its
            # tooltip.
            self.progress_tiles.update_progress(ProgressUpdate.failed("Failed"))
            self.progress_tiles.show()

    # ── lifecycle ────────────────────────────────────────────────────────

    def hideEvent(self, event) -> None:
        # The tile grid panel is a top-level window, so it does not go with this
        # widget on its own, and would float over the next tab. The button is
        # unchecked with it, so one click opens it again.
        # Reached from C++ during teardown too, where raising is fatal under PyQt5,
        # so looked up tolerantly and never raises.
        super().hideEvent(event)
        panel = self.__dict__.get("tile_grid_panel")
        button = self.__dict__.get("btn_tile_grid")
        try:
            if panel is not None:
                panel.hide()
            if button is not None:
                button.setChecked(False)
        except RuntimeError:  # wrapped C/C++ object already deleted
            pass

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self.microscope.imaging.stop()
        # Every psygnal this widget subscribed to, without exception. They outlive the
        # widget -- they belong to the microscope -- and they hold bound methods of Qt
        # objects that `close` has already torn down on the C++ side, so the next emit
        # writes into freed memory: a hard segfault, not an exception.
        subscriptions = [
            (self.session.changed, self._on_session_signal),
            (self.fm.acquisition_progress_signal, self._on_fm_progress),
            (self.fm.objective.position_changed, self._on_objective_moved),
            (self.fm.acquiring_changed, self._on_fm_acquiring_signal),
            (self.fm.transform_changed, self._on_transform_changed),
        ]
        if self.microscope.sample_handling is not None:
            subscriptions.append(
                (
                    self.microscope.sample_handling.holder_changed,
                    self._on_holder_changed,
                )
            )
        subscriptions.append(
            (self.microscope.imaging.tiled_progress, self._on_tile_progress)
        )
        subscriptions.append(
            (self.microscope.imaging.overview_acquired, self._on_acquired)
        )
        if self.microscope.stage is not None:
            subscriptions.append(
                (self.microscope.stage.position.changed, self._on_stage_signal)
            )
        for signal, slot in subscriptions:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError, ValueError, KeyError):
                pass
        super().closeEvent(event)
