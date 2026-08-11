"""Acquire a fluorescence overview: settings, run control, and live result.

Standalone, and embeddable as a tab. It owns nothing the acquisition needs -- it is
handed a microscope and drives `FMTiledAcquisitionRunner` -- so it can be dropped into
the AutoLamella UI or opened on its own against a simulator.

Layout follows the house convention: canvas on the left, controls on the right,
actions along the bottom.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
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
from fibsem.fm.acquisition import FMTiledAcquisitionRunner, OverviewDestination
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ObjectiveStartPosition,
    OverviewParameters,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui import notification_service, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.fm.widgets.fm_multi_channel_widget import FluorescenceMultiChannelWidget
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    FMOverviewConfirmationDialog,
)
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.fm.widgets.overview_list_widget import OverviewListWidget
from fibsem.ui.fm.widgets.tile_grid_options_panel import TileGridOptionsPanel
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.imaging.tiling.geometry import compute_tile_grid_from_fov
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    ElidedLabel,
    IconToolButton,
    TitledPanel,
)
from fibsem.ui.widgets.progress_widget import (
    FibsemProgressWidget,
    ProgressUpdate,
)
from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget
from fibsem.ui.widgets.canvas.stage_frame import FMStageProjection, StageFrame

TEXT_MUTED = stylesheets.TEXT_MUTED_COLOR
PROGRESS_FONT_PX = 10
# Inset for chrome drawn over the canvas, matching the toolbar buttons' own margin so
# the cursor readout in the top left lines up with the buttons in the top right.
CANVAS_CHROME_MARGIN = 4


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

# Radius of the specimen grid, for the boundary circle. Matches the minimap's.
GRID_RADIUS_M = 1000e-6

# All three position markers are crosshairs, so colour is the only thing telling them
# apart -- they have to stay far enough apart to read at a glance, and away from the red
# the canvas draws its origin in.
#
# Yellow for "you are here".
CURRENT_POSITION_COLOUR = "#ffee58"
# Cyan for positions a user saved, against the yellow of where the stage is.
SAVED_POSITION_COLOUR = "#26c6da"
# Lime for the one selected, matching the minimap so the same position reads the same
# way on both. Bright enough to find at a glance among a grid full of cyan.
SELECTED_POSITION_COLOUR = "#76ff03"
# Muted, because the holder slots are structural context like the limits rather
# than something anyone marked -- and they would otherwise read as saved positions.
SLOT_COLOUR = "#90a4ae"

# Wide enough for millimetre-scale coordinates without the row re-laying out as the
# cursor moves -- a readout that shoves the buttons sideways is worse than none.
CURSOR_READOUT_WIDTH = 210

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

    The canvas can hold several overviews at once and, until this existed, nothing
    tracked them: the widget kept a single `_displayed_image` that each new
    acquisition overwrote, so an earlier overview's pixels stayed on screen with its
    provenance already discarded (FIB-543).

    Holds the *metadata*, not the image. What a list needs -- where it was, when, at
    what scale -- is all in there, and keeping N full-resolution mosaics alive to
    reach it would cost hundreds of megabytes on a canvas already short of them
    (FIB-414). The pixels the canvas redraws from are its own decimated copies.
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


class FMOverviewWidget(QWidget):
    """Configure, run and view a fluorescence overview acquisition."""

    overview_acquired = pyqtSignal(FluorescenceImage)

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
    # progress signal is a psygnal, which calls its callbacks synchronously on
    # whichever thread emitted -- here, the worker. Touching widgets from there is a
    # cross-thread GUI access (Qt says so: "Cannot set parent, new parent is in a
    # different thread"). Re-emitting as a Qt signal gets it queued onto the GUI
    # thread, because this widget lives there.
    _progress_received = pyqtSignal(dict)
    # Same hop for stage moves: `stage_position_changed` is a psygnal, so it fires
    # on whichever thread moved the stage -- a worker, during an acquisition.
    _stage_moved = pyqtSignal(object)
    # And for something starting or finishing on the FM, which is reported from
    # whichever thread finished with it -- for our own runs, the acquisition worker.
    _fm_acquiring_changed = pyqtSignal(bool)

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
        self._positions: List[FibsemStagePosition] = []
        # Which marked position is selected, by name. See `set_selected_position`.
        self._selected_position: Optional[str] = None
        self._grid_footprint: Optional[tuple] = None
        self._enabled_channels: Optional[List[ChannelSettings]] = None
        # Where the stage is, as far as anyone has told us. Cached rather than polled
        # per use so that everything drawn from it -- the marker, the planned grid --
        # agrees by construction instead of because two callers happened to poll
        # together. See `_current_stage_position` for why polling is the odd one out.
        self._stage_position: Optional[FibsemStagePosition] = None
        # The objective half of the canvas info bar, cached. Read from hardware only
        # when something may have moved it -- see `_refresh_objective_info`.
        self._objective_info: Optional[str] = None
        # Where the next overview will be centred, once a user has dragged the grid
        # somewhere. None means "wherever the stage is", which is what the runner does
        # by default -- so an untouched grid describes exactly what would be acquired.
        self._target: Optional[FibsemStagePosition] = None
        # Where acquired overviews are written. None means nowhere: the widget opens
        # standalone against a simulator as often as it runs inside an experiment, and
        # inventing a directory for those runs would scatter files through whatever
        # working directory it happened to be launched from. A host that has somewhere
        # to put them says so -- see `set_save_directory`.
        self._save_directory: Optional[str] = None
        self._destination: Optional[OverviewDestination] = None
        self._saved_path: Optional[str] = None
        # Whether a run is under way, and whether a host is allowing one to be started.
        # Two independent facts kept apart on purpose: a workflow ending must not
        # re-enable a tab whose acquisition is still going, nor the reverse. See
        # `_apply_enabled_state`, which is the only thing that reads them.
        self._running = False
        self._interactive = True

        self._init_ui(channel_settings or self._default_channels())
        self._sync_tile_fov()
        self._on_settings_changed()

        self._progress_received.connect(self._apply_progress)
        self.fm.acquisition_progress_signal.connect(self._on_progress)
        self._stage_moved.connect(self._on_stage_moved)
        # A plain bound method, not `self._stage_moved.emit`, matching how the progress
        # signal is subscribed just above. psygnal holds bound methods weakly and drops
        # them when the owner is collected, which a Qt signal's `emit` does not get: PyQt
        # builds a new wrapper on every access, so psygnal cannot weakref it, cannot
        # match it at disconnect time -- and says nothing when the disconnect therefore
        # removes nothing. Emitting into a widget Qt had already torn down was a segfault.
        self.microscope.stage_position_changed.connect(self._on_stage_signal)
        self._fm_acquiring_changed.connect(self._on_fm_acquiring_changed)
        self.fm.acquiring_changed.connect(self._on_fm_acquiring_signal)
        # Marshalled by `@ensure_main_thread` on the slot rather than by relaying
        # through a Qt signal like the three above. That decorator is the contract
        # `FibsemMicroscope` states for its psygnals and what most of the UI does; the
        # relay is this widget's own dialect. Still a plain bound method, so psygnal can
        # hold it weakly and match it at disconnect time.
        self.fm.objective.position_changed.connect(self._on_objective_moved)
        self._refresh_current_position()
        self._refresh_objective_info()
        self._refresh_orientation_banner()

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
            color=CURRENT_POSITION_COLOUR, marker="+", size=13
        )
        self.canvas.canvas.add_overlay(self.current_position_overlay)

        # Crosshairs rather than dots: a marked position is a point on the sample, and
        # a filled dot covers the feature it is naming. The gap in the middle is the
        # whole reason -- you can see what you marked.
        self.position_overlay = PointsOverlay(
            color=SAVED_POSITION_COLOUR, marker="+", size=11
        )
        self.canvas.canvas.add_overlay(self.position_overlay)

        # The selected position, on its own layer rather than as a colour within the
        # one above: `PointsOverlay` paints every point the same, and one selected
        # marker is not worth teaching it per-point colours for. Added last, so it
        # draws over its unselected neighbours where markers crowd together.
        self.selected_position_overlay = PointsOverlay(
            color=SELECTED_POSITION_COLOUR, marker="+", size=15
        )
        self.canvas.canvas.add_overlay(self.selected_position_overlay)

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

        # Where a double-click would take the stage, read off continuously rather than
        # on demand: the canvas is a map, and a map you have to click to get a
        # coordinate out of cannot be used to decide whether to click.
        #
        # Drawn over the canvas rather than beside it, in the one free corner -- the
        # toolbar owns the top right, the scalebar the bottom right, and the stage info
        # bar the bottom left. A Qt label rather than the canvas's own text chrome
        # because this updates on every mouse motion, and `set_info_text` costs ~4.9 ms
        # against a label's 0.08 ms; at motion-event rates that is the difference
        # between a readout and a stutter.
        self.cursor_readout = QLabel(self.canvas.canvas)
        self.cursor_readout.setAttribute(Qt.WA_TransparentForMouseEvents)
        # Monospaced so the digits sit still while the cursor moves. A proportional
        # font makes the whole readout shuffle on every pixel of travel.
        self.cursor_readout.setStyleSheet(
            "color: #e8e8e8; font-size: 10px; font-family: monospace;"
            "background: rgba(26, 26, 26, 160); border-radius: 3px; padding: 2px 5px;"
        )
        self.cursor_readout.move(CANVAS_CHROME_MARGIN, CANVAS_CHROME_MARGIN)
        self.cursor_readout.hide()  # nothing to say until the pointer is over the canvas

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)

        # Two bars: across the grid, and within the tile being acquired. Both are
        # `FibsemProgressWidget`, which distinguishes finished from failed -- a failed
        # run used to paint the same full green bar as a successful one.
        # Hidden until a run starts: they are empty most of the time, and stacked in a
        # column there is nothing beside them to shift when they come and go.
        self.progress_tiles = shrink_progress_text(FibsemProgressWidget())
        self.progress_tile_detail = shrink_progress_text(FibsemProgressWidget())

        self.status = ElidedLabel("")
        self.status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        # Shown only when the stage is somewhere an overview cannot be acquired from.
        # A line saying what is wrong and a button that fixes it, rather than a greyed
        # Acquire button with no explanation -- which is what the old FM tab did, and
        # which leaves you looking at a dead panel (FIB-436).
        self.orientation_notice = ElidedLabel("")
        self.orientation_notice.setStyleSheet(
            f"color: {stylesheets.WARN_COLOR}; font-size: 11px;"
        )
        # Labelled in `_refresh_orientation_banner`, from the same `default_orientation`
        # the move targets -- the control widget lets that be changed at runtime, and a
        # button naming one orientation while going to another is worse than no button.
        self.button_move_to_fm = QPushButton()
        self.button_move_to_fm.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_move_to_fm.clicked.connect(self.move_to_fm_orientation)

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
        # the window allows. Same shape as `FibsemMinimapWidget`, which stacks its run
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
        self.canvas.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.canvas.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)
        self.canvas.canvas.cursor_moved.connect(self._on_cursor_moved)

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
        # Centred on the position the run will be centred on -- a dragged target, or
        # the stage itself -- not on whatever happens to be displayed. It used to be
        # pinned to the canvas origin, which is where the stage was the *first* time
        # anything was drawn: correct until the stage moved, and then quietly not.
        offset = self._grid_offset()
        self.tile_grid_overlay.set_anchor(
            self.canvas.canvas.metres_to_canvas(*offset)
        )

        # No `display_pixel_size`: the overlay reads it from the canvas at draw time.
        # Pinning it here would freeze the scale at whatever was displayed when the
        # settings last changed, and the image underneath changes without them -- the
        # live preview swaps in a decimated mosaic mid-run.
        self.tile_grid_overlay.set_grid(
            tiles, (height, width), pixel_size, overlap=parameters.overlap
        )
        # Same camera geometry, so it can be drawn at the same time rather than
        # waiting for an image the tile grid does not wait for either.
        self._refresh_stage_metadata()

        span_x = parameters.cols * fov[0] * (1 - parameters.overlap) + fov[0]
        span_y = parameters.rows * fov[1] * (1 - parameters.overlap) + fov[1]

        # Measured against the area as it stands, before the line below moves it.
        left_area = self._grid_has_left_the_working_area(offset, span_x, span_y)

        # Keep the declared working area on the grid, always and silently. It is what
        # the zoom limiter measures against, so an area left behind stretches the
        # content across the gap and caps how far the view can zoom in. `refit=False`
        # is what makes doing this continuously safe: re-declaring used to re-frame,
        # so dragging the grid snapped the view onto it on every motion event.
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
        if (changed or left_area) and not self.tile_grid_overlay.is_dragging:
            self.tile_grid_overlay.fit_view()

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
        move -- a double-click to somewhere 100 µm away would throw the zoom away, the
        same failure a tile toggle used to cause. It follows only once the grid has
        left the area outright, at which point what is framed and what is planned no
        longer overlap at all.
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
        time, the panel kept that height and clipped the extra lines top and bottom
        (FIB-510).

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
            self._origin = self._position_of(image)
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
        stamp = getattr(image.metadata, "acquisition_date", None)
        if stamp:
            for record in self._records.values():
                if getattr(record.metadata, "acquisition_date", None) == stamp:
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

        What does not survive is `stage_position.name`, which `stitch_tileset` uses to
        carry the grid size. A loaded overview's row therefore shows its scale without
        its grid. Accepted rather than worked around: the alternative is teaching the
        OME serialization to carry a field only the list reads (FIB-438).

        Separate from the dialog so it can be called with a path -- by a test, by a
        host restoring a session, or later by whatever knows an experiment's own
        overviews (FIB-547).
        """
        try:
            image = FluorescenceImage.load(path)
        except Exception as e:
            logging.error(f"Could not load an overview from {path}: {e}")
            notification_service.show_toast(f"Could not load that overview.\n{e}", "error")
            return None

        # Placed even without a geometry, but said out loud. `_offset_of` falls back to
        # the origin, which puts it in the middle of the view rather than where it was
        # taken -- and an overview silently in the wrong place is worse than one you
        # have been told to distrust. Anything acquired before FIB-416 is this case.
        if FMStageProjection.from_image(image) is None:
            logging.warning(f"Overview {path} has no recorded geometry; placing at the origin.")
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
        redraw cost -- which is the point on a canvas holding several (FIB-414).
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

    def _offset_from_origin(
        self, position: FibsemStagePosition
    ) -> Tuple[float, float]:
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

    def add_settings_section(self, title: str, widget: QWidget, first: bool = True) -> None:
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
            self._controls_layout.insertWidget(self._controls_layout.count() - 1, section)

    def set_positions(self, positions: List[FibsemStagePosition]) -> None:
        """Stage positions to mark on the overview, e.g. saved lamella positions.

        Names are carried onto the markers, so a caller does not have to keep a
        parallel list of labels in the order it happened to pass them.

        These are *fluorescence* poses. A lamella carries one of each, and the beam-side
        pose is a different place on this canvas -- on a compustage the stage flips
        between them -- so a host passing the wrong one would mark every position
        somewhere the sample is not. See `build_lamella_poses`.
        """
        self._positions = list(positions)
        self._refresh_positions()

    def set_selected_position(self, name: Optional[str]) -> None:
        """Pick out one of the marked positions, or None for none.

        By name rather than index: the host's list and this one are rebuilt
        independently from the same experiment, and an index would silently point at a
        different lamella the moment one was removed.
        """
        self._selected_position = name or None
        self._refresh_positions()

    @property
    def selected_position(self) -> Optional[str]:
        return self._selected_position

    def set_save_directory(self, path: Optional[str]) -> None:
        """Write acquired overviews under *path*, or None to keep them in memory only.

        Each run claims its own subdirectory under this one, named for when it started,
        so consecutive runs accumulate instead of overwriting each other -- pointing
        straight at an experiment directory and letting the runner write `tile-00-00`
        into it would mean every overview replaced the last.

        Told rather than discovered, because this widget knows nothing about
        experiments: it is handed a microscope, and its tests construct it that way.
        """
        self._save_directory = path

    @property
    def saved_path(self) -> Optional[str]:
        """Where the last acquired overview was written, if it was."""
        return self._saved_path

    def set_interactive(self, enabled: bool) -> None:
        """Allow or forbid starting work, without touching a run already in progress.

        For a host that owns the instrument for a while -- an AutoLamella workflow --
        and needs the tab to stop competing for it. Cancel stays live regardless, so a
        lock arriving mid-run cannot strand an acquisition with no way to stop it.

        Deliberately not `setEnabled(False)` on the whole widget: greying out the canvas
        would also stop you *reading* the overview you just acquired, and looking at it
        costs the workflow nothing.
        """
        self._interactive = enabled
        self._apply_enabled_state()

    def set_origin(self, position: Optional[FibsemStagePosition]) -> None:
        """Anchor the canvas frame at *position*, or None to go back to automatic.

        The origin decides only where canvas zero sits -- everything is drawn relative
        to it, so any stage position serves. Left alone it is fixed to wherever the
        stage was the first time anything was drawn, which is right for a widget opened
        where the work is and wrong for one that should always describe the same place:
        an FM canvas anchored at the offset mount while a FIB/SEM canvas is anchored at
        the column, 48 mm away, each correctly framed (FIB-418).

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

    def _frame(self) -> Optional[StageFrame]:
        """The mapping between stage space and the canvas, or None if it cannot be had.

        The one place the two meet. Everything drawn from a stage position goes through
        it -- marked positions, the stage and grid limits, the current pose, the planned
        grid -- and so does every click, so they share a frame by construction rather
        than by six callers agreeing to do the same arithmetic.

        Anchored on the canvas origin rather than on whichever image happens to be
        displayed, so a marker names the same piece of sample no matter what is on
        screen. The origin's rotation and tilt are what put the drawing in the current
        stage orientation -- t = -180 for FM.
        """
        projection = FMStageProjection.from_microscope(self.microscope)
        if projection is None:
            return None

        origin = self._origin or self._current_stage_position()
        if origin is None:
            return None
        self._origin = origin  # fix it now, so later drawing shares this frame

        return StageFrame(self.canvas.canvas, self._posed(origin), projection)

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
            x=origin.x, y=origin.y, z=origin.z,
            r=current.r, t=current.t,
            coordinate_system=origin.coordinate_system,
        )

    def _refresh_stage_metadata(self) -> None:
        """Draw where the sample and the stage can physically go.

        The context an overview is read against: which grid you are on, how far from its
        centre, and how much travel is left. Same shapes the minimap draws, but placed
        straight into the canvas frame -- on a real-space canvas there is no stitched
        image to reproject onto, so the indirection disappears.
        """
        frame = self._frame()
        if frame is None:
            self.stage_overlay.set_shapes([])
            return

        specs = []
        try:
            centre = frame.to_canvas(
                FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
            )
        except Exception as e:
            logging.debug(f"Cannot place the grid centre: {e}")
            self.stage_overlay.set_shapes([])
            return

        limits = getattr(self.microscope._stage, "limits", None)
        if limits and self.microscope.stage_is_compustage:
            # The travel envelope, as a box around the grid centre. Sized from the
            # limits rather than projected corner-by-corner: the projection is flips
            # and a tilt, which keep an axis-aligned box axis-aligned.
            specs.append(ShapeSpec(
                kind="rect", cx=centre[0], cy=centre[1], color="yellow",
                width=frame.length(limits["x"].max - limits["x"].min),
                height=frame.length(limits["y"].max - limits["y"].min),
                label="Stage limits",
            ))
            specs.append(ShapeSpec(
                kind="circle", cx=centre[0], cy=centre[1], color="red",
                radius=frame.length(GRID_RADIUS_M), label="Grid boundary",
            ))

        holder = getattr(self.microscope._stage, "holder", None)
        for slot in getattr(holder, "slots", {}).values():
            position = getattr(slot, "position", None)
            if position is None:
                continue
            try:
                point = frame.to_canvas(position)
            except Exception as e:
                logging.debug(f"Cannot place slot {position.name!r}: {e}")
                continue
            specs.append(ShapeSpec(
                kind="crosshair", cx=point[0], cy=point[1], color=SLOT_COLOUR,
                label=position.name or "",
            ))

        self.stage_overlay.set_shapes(specs)

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

    def _set_objective_info(self, position: float, state: str) -> None:
        """Render the info bar from values already in hand -- no device read.

        The path the signal takes. Split out of `_refresh_objective_info` so a move can
        update the bar from what it announced rather than by asking again.
        """
        self._objective_info = (
            f"objective {position * constants.SI_TO_MILLI:.3f} mm ({state.lower()})"
        )
        self._refresh_stage_info()
        self.settings_widget.set_objective_positions(
            position, self.fm.objective.focus_position
        )

    @ensure_main_thread
    def _on_objective_moved(self, position: float, state: str) -> None:
        """Something moved the objective, and said where it ended up.

        Reached from psygnal, which fires on whichever thread did the moving -- a
        z-stack or an autofocus sweep runs on a worker -- so this touches widgets only
        because the decorator has already marshalled it.

        The values come from the signal rather than a fresh read, so this costs nothing
        and cannot disagree with the move that prompted it. That is the whole point:
        this label used to be rebuilt from two device reads on every stage poll, and on
        a shared connection each of those took the imaging channel (FIB-534).
        """
        self._set_objective_info(position, state)

    def _refresh_objective_info(self) -> None:
        """Re-read the objective for the info bar. Touches hardware -- call sparingly.

        This used to happen inside `_refresh_stage_info`, which runs on every stage poll.
        On a real system that is not a cheap read: the FM and the beams share one
        connection, so reading the objective points it at the FM, and every poll was
        doing it. It stopped a workflow task (FIB-517).

        Now a backstop rather than the usual path: `ObjectiveLens.position_changed`
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
        # connection two reads are two takes of the imaging channel (FIB-517).
        self._set_objective_info(position, state)

    def _objective_state(self) -> Optional[str]:
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

        `stage_position_changed` fires from *inside* `get_stage_position`, so it means
        "a poll found it had moved" rather than "the stage moved". Polling here on
        every use would therefore add a round trip per keystroke on a spin box without
        making the answer any fresher than the last poll anyone did -- so this reads
        what the signal delivered, and everything drawn from the stage position stays
        in agreement instead of drifting between callers who polled at different
        moments.
        """
        if self._stage_position is None:
            try:
                self._stage_position = self.microscope.get_stage_position()
            except Exception as e:
                logging.debug(f"Could not read the stage position: {e}")
        return self._stage_position

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

    def _grid_anchor(self) -> Tuple[float, float]:
        """Where on the canvas the planned grid is centred.

        The same place :meth:`_grid_offset` names, in canvas pixels rather than metres
        -- derived from it rather than projected again, so the drawn grid and the
        declared working area cannot end up centred on different points.
        """
        return self.canvas.canvas.metres_to_canvas(*self._grid_offset())

    def _refresh_positions(self) -> None:
        """Mark the stage positions in the canvas frame."""
        frame = self._frame()
        if not self._positions or frame is None:
            self.position_overlay.set_points([])
            self.selected_position_overlay.set_points([])
            return

        points, labels = [], []
        selected_points, selected_labels = [], []
        for position in self._positions:
            try:
                point = frame.to_canvas(position)
            except Exception as e:
                logging.debug(f"Cannot mark {position.name!r}: {e}")
                continue
            name = position.name or ""
            if name and name == self._selected_position:
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
        """Whether the stage is somewhere the objective can image the sample from.

        `fm.acquisition_orientations`, and not a single pose of our own choosing: a
        system whose objective sees the sample from more than one orientation says so
        there, and hard-coding one here would lock the other out.

        Not `fm.has_valid_orientation()`, which asks the looser question of whether FM
        *control* is allowed -- true at SEM and MILLING, where a compustage has flipped
        the sample away from the objective -- and which `ALLOW_UNKNOWN_ORIENTATIONS`
        answers yes to unconditionally anyway. The canvas frame is built from the pose
        (see `_posed`), so tiles acquired somewhere the code cannot name are placed
        against a projection nobody has checked.
        """
        return self.fm.is_acquisition_orientation()

    def at_fluorescence_pose(self) -> bool:
        """Whether the stage is at *the* fluorescence orientation, not merely a workable one.

        Stricter than `at_acquisition_orientation`, and asked only where something is written
        down. Asked of `fm.default_orientation` because that is the orientation
        `build_lamella_poses` derives a fluorescence pose *into*: two names for the same
        thing could drift, one cannot.
        """
        return self.microscope.get_stage_orientation() == self.fm.default_orientation

    def move_to_fm_orientation(self) -> None:
        """Drive the stage to the fluorescence orientation, having asked first.

        A real stage move, so it gets the same confirmation as the other moves in this
        widget -- and the same worker, since `move_to_microscope` blocks for as long as
        the stage takes.
        """
        if self._running or self.fm.is_acquiring:
            notification_service.show_toast(
                "Cannot move the stage during an acquisition.", "warning"
            )
            return
        orientation = self.fm.default_orientation
        reply = QMessageBox.question(
            self,
            "Confirm Movement",
            f"Move the stage to the {orientation} orientation?\n\n"
            f"The stage is currently at {self.microscope.get_stage_orientation()}.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            logging.info("Move to FM orientation cancelled by user")
            return

        self.status.setText(f"Moving to {orientation}…")
        worker = FunctionWorker(self._move_to_orientation_worker, orientation)
        worker.start()

    def _move_to_orientation_worker(self, orientation: str) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        try:
            self.microscope.move_to_microscope(orientation)
            logging.info(f"Moved to {orientation} orientation")
        except Exception as e:
            logging.error(f"Failed to move to {orientation} orientation: {e}")
        # Not followed by a refresh: the move raises `stage_position_changed`, which
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
        allowed = self.fm.acquisition_orientations
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
            self.button_move_to_fm.setText(f"Move to {self.fm.default_orientation}")

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

    def _on_stage_signal(self, position: FibsemStagePosition) -> None:
        """Called by psygnal, on whichever thread polled. Touches no widgets.

        The counterpart of :meth:`_on_progress`, and a real method rather than the Qt
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
        self._stage_position = position
        self._refresh_current_position()
        # Everything else on the canvas is placed through a frame whose rotation and
        # tilt come from wherever the stage is (see `_posed`), so a re-pose moves all of
        # it. Redrawn only when the pose actually changes: the stage is polled
        # constantly, and a translation leaves every one of these exactly where it was --
        # the origin they are measured from moves with nothing.
        if reposed:
            self._refresh_positions()
            self._refresh_stage_metadata()
            # The orientation is read off rotation and tilt, so it can only have changed
            # when they did -- the same reason the redraws above are gated on it.
            self._refresh_orientation_banner()
            self._apply_enabled_state()
        # The grid follows the stage only when it is not pinned to a target and not
        # mid-run. During a run the stage visits every tile in turn, and a grid that
        # followed would crawl across the canvas describing nothing.
        if self._target is None and not self.is_acquiring:
            self._refresh_tile_grid()

    def _pose_changed(self, position: FibsemStagePosition) -> bool:
        """Whether *position* is at a different rotation or tilt from the last one.

        Only r and t, because only they enter the frame. Where the stage has travelled
        to is not part of how stage space maps onto the canvas -- canvas zero is the
        origin, and it does not move.
        """
        previous = self._stage_position
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

    def _position_at(self, x: float, y: float) -> Optional[str]:
        """The marked position under a canvas point, or None.

        Measured on screen, not in data units: at a wide zoom every marker would be
        within any sensible micron radius of the click, and at a tight one none would be.
        """
        frame = self._frame()
        ax = getattr(self.canvas.canvas, "_ax", None)
        if frame is None or ax is None or not self._positions:
            return None
        try:
            transform = ax.transData
            click = transform.transform((x, y))
        except Exception as e:
            logging.debug(f"Could not resolve the click for picking: {e}")
            return None

        best_name, best_distance = None, float(self.PICK_RADIUS_PX)
        for position in self._positions:
            name = position.name
            if not name:
                continue
            try:
                point = transform.transform(frame.to_canvas(position))
            except Exception:
                continue
            distance = (
                (click[0] - point[0]) ** 2 + (click[1] - point[1]) ** 2
            ) ** 0.5
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

        The orientation check here used to be `fm.has_valid_orientation()`, which
        `ALLOW_UNKNOWN_ORIENTATIONS` answers yes to unconditionally -- so it refused
        nothing. It now asks the same question with the escape hatch off, which is what
        it was always meant to mean: a click is a stage move computed through a frame
        built from the current pose, and from a pose the code cannot name there is
        nothing that has checked where it would send the stage (FIB-436).
        """
        if self.is_acquiring:
            notification_service.show_toast(
                "Cannot move the stage during an acquisition.", "warning"
            )
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
        limits = getattr(self.microscope._stage, "limits", None)
        if limits and not position.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return

        self.status.setText(f"Moving to {self._describe(position)}…")
        worker = FunctionWorker(self._move_worker, position)
        worker.start()

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
        if self._running or self.is_acquiring or not self._interactive:
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
        # Stricter than the double-click's check, and deliberately so. Moving needs only
        # a pose the FM can work from; marking has to survive being written down, since
        # the position becomes a lamella's *fluorescence* pose -- and one carrying SEM
        # rotation and tilt is not one, however right it looked on screen.
        if not self.at_fluorescence_pose():
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
        selected = self._selected_position
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
        """
        frame = self._frame()
        if frame is None:
            return None
        try:
            target = frame.to_stage(x, y)
        except Exception as e:
            logging.debug(f"Could not resolve the clicked position: {e}")
            return None

        limits = getattr(self.microscope._stage, "limits", None)
        if limits and not target.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return None
        return target

    def _move_worker(self, target: FibsemStagePosition) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        try:
            self.microscope.safe_absolute_stage_movement(target)
        except Exception as e:
            logging.error(f"Could not move the stage: {e}", exc_info=True)
        # Publishes the new position through `stage_position_changed`, which is what
        # re-marks it -- rather than assuming the stage arrived exactly where it was
        # asked to, which on a real instrument it does not.
        try:
            self.microscope.get_stage_position()
        except Exception as e:
            logging.debug(f"Could not confirm the stage position after moving: {e}")

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
        """Show the readout over the canvas, sized to its text, or hide it when empty.

        Hidden rather than blanked: it sits on top of the image, and an empty plaque
        floating over the data is worse than nothing there. `adjustSize` because the
        label is positioned rather than laid out, so nothing else will size it.
        """
        self.cursor_readout.setText(text)
        if not text:
            self.cursor_readout.hide()
            return
        self.cursor_readout.adjustSize()
        self.cursor_readout.show()

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
        idle = self._interactive and not self._running and not self.fm.is_acquiring
        has_tiles = self.settings_widget.parameters.n_enabled_tiles > 0
        # Acquisition only. Display and navigation stay live from any pose: looking at an
        # overview that has already been acquired is harmless wherever the stage is, and
        # the settings panel stays editable so a run can be set up while the stage is on
        # its way (FIB-436).
        self.button_acquire.setEnabled(
            idle and has_tiles and self.at_acquisition_orientation()
        )
        # Cancel is deliberately not gated on `_interactive`: a host locking the tab
        # mid-run must not take away the only way to stop the stage. Once a cancel has
        # been asked for, the button goes and stays away -- the stop event is the record
        # of that, so asking it beats a second flag that could disagree.
        self.button_cancel.setEnabled(self._running and not self._stop_event.is_set())
        self.settings_widget.setEnabled(idle)
        self.channel_widget.setEnabled(idle)

    # ── acquisition ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Confirm, then run the overview on a worker thread."""
        if self.is_acquiring:
            logging.warning("An overview acquisition is already running.")
            return
        # `is_acquiring`, not `is_streaming`: a z-stack or an autofocus sweep in the
        # control widget is not a stream, and this widget's own tileset was invisible to
        # that one, which is the direction that mattered (FIB-441).
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
            where, needed = self._where_the_stage_is(), self._where_the_stage_needs_to_be()
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
        # question that was just asked (FIB-417).
        state = self._objective_state()
        if state is not None and state != "Inserted":
            message = (
                f"The objective is {state.lower()}, so an overview cannot be acquired. "
                f"Insert it and try again."
            )
            logging.warning(message)
            notification_service.show_toast(message, "warning")
            self.status.setText(f"Objective is {state.lower()}.")
            return

        if (parameters.objective_start is ObjectiveStartPosition.FOCUS
                and self.fm.objective.focus_position is None):
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
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled before starting")
            return

        # Claimed on the GUI thread, before the worker starts: it makes a directory, so
        # an unwritable path is reported now rather than after the stage has moved.
        self._destination = self._claim_destination()

        # Held for the length of the run, so nothing else drives the FM while the stage
        # is walking the grid. Cleared in the worker's `finally` (FIB-441).
        self.fm.set_acquiring(True, "overview acquisition")

        self._stop_event.clear()
        self._set_running(True)
        self._runner = FMTiledAcquisitionRunner(
            microscope=self.microscope,
            channel_settings=channels,
            overview_parameters=parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            stop_event=self._stop_event,
            # None means "where the stage is", which the runner resolves itself. The
            # stage still returns to where it started afterwards -- the target is the
            # grid's centre, not a new home.
            centre_position=self._target,
            # None when no save directory has been set -- the standalone default. The
            # runner writes each tile as it lands, so a cancelled run keeps what it got.
            save_directory=self._destination.tiles_directory if self._destination else None,
        )
        self._worker = FunctionWorker(self._acquire_worker)
        self._worker.start()

    def _claim_destination(self) -> Optional["OverviewDestination"]:
        """Where this run's files go, or None if nowhere.

        Failing to claim one is not fatal: an overview you can see is worth more than
        no overview at all, so the run goes ahead unsaved and says so.
        """
        if self._save_directory is None:
            return None
        try:
            return OverviewDestination.create(self._save_directory)
        except OSError as e:
            logging.error(f"Cannot save into {self._save_directory}: {e}")
            notification_service.show_toast(
                f"Could not prepare a save directory; this overview will not be "
                f"saved.\n{e}",
                "warning",
            )
            return None

    def _acquire_worker(self) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        from fibsem.cancellation import OperationCancelledError

        try:
            runner = self._runner
            mosaic = runner.run_and_stitch()
            self._mosaic = mosaic
            # Saved here rather than after the signal: a consumer of `overview_acquired`
            # may want the file, and the mosaic carries the run's name once written.
            if self._destination is not None:
                self._saved_path = self._destination.save_mosaic(mosaic)
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
        finally:
            # A run that ends still marked as acquiring leaves the FM unusable for
            # the rest of the session, with nothing on screen to say why -- so this goes
            # in a `finally` rather than after the emits, and stays right if a future
            # edit narrows one of the `except` clauses above.
            self.fm.set_acquiring(False)

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logging.info("Cancelling overview acquisition")
        self._stop_event.set()
        self._apply_enabled_state()
        self.status.setText("Cancelling…")

    def _set_progress_visible(self, visible: bool) -> None:
        """Show the progress bars only while there is progress to show.

        They spend most of their life empty. Stacked in the settings column nothing sits
        beside them, so they can come and go without moving anything -- which is what
        made this possible: they used to be inline with the buttons, where hiding one
        slid the actions sideways, and a fixed-size holder existed purely to stop that.
        """
        self.progress_tiles.setVisible(visible)
        self.progress_tile_detail.setVisible(visible)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._apply_enabled_state()
        if running:
            # Cleared at the start, so a run that fails to save cannot inherit the
            # previous run's path and report itself saved.
            self._saved_path = None
            # Left empty rather than shown as indeterminate, which would paint a full
            # bar before a single tile had been acquired.
            self.progress_tiles.reset()
            self.progress_tile_detail.reset()
            self.status.setText("Starting…")

        # After the resets above, not before: `FibsemProgressWidget.reset()` hides
        # itself, so showing the bars first and resetting them second leaves them
        # hidden for the whole run.
        self._set_progress_visible(running)

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
            # The preview mosaic spans the whole planned grid, so it goes wherever the
            # grid's centre is. Read off the runner rather than recomputed: the runner
            # resolved "wherever the stage is" to a concrete position when it started,
            # and the stage has been moving from tile to tile ever since.
            centre = getattr(self._runner, "centre_position", None)
            self.canvas.set_placement(
                self._offset_from_origin(centre) if centre is not None else (0.0, 0.0)
            )
            # The preview is decimated to keep it a sane size, so its pixels are
            # `preview_stride` times coarser than a tile's. Placement is by pixel size,
            # so saying so is all that is needed: coarser pixels over the same count
            # cover the same ground, and the mosaic lands at the size it represents.
            stride = payload.get("preview_stride", 1) or 1
            self.canvas.set_pixel_size(self.fm.camera.pixel_size[0] * stride)

            # This run's channels, and only these. `set_channel` upserts, so a channel
            # switched off since the last run would otherwise keep its layer -- still
            # holding the previous overview's pixels -- and be blended into this one.
            channels = self.channels
            self.canvas.retain_channels([channel.name for channel in channels])
            for channel, plane in zip(channels, planes):
                self.canvas.set_channel(channel.name, plane, channel.color)
        except Exception as e:
            logging.debug(f"Could not display the overview preview: {e}")

    def _describe_save(self) -> Tuple[str, str]:
        """Status suffix and tooltip for whether the overview reached disk.

        Silent when nothing was meant to be saved, so the standalone app does not report
        a failure every run. When a destination *was* claimed and the mosaic did not
        land, that is worth saying: otherwise a failed save reads exactly like a
        successful one, and the difference only turns up when someone goes looking for
        the file.
        """
        if self._saved_path:
            return "  ·  saved", self._saved_path
        if self._destination is not None:
            return "  ·  not saved", "Could not be written to disk — see the log."
        return "", ""

    def _finish(self, state: str, error: Optional[str]) -> None:
        # Hides both bars. The per-tile one stays hidden -- there is no tile in progress
        # to describe -- but the overall bar is shown again below whenever it has a
        # terminal state worth reading: `FibsemProgressWidget` paints finished and
        # failed differently, and that colour is the at-a-glance answer.
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
            suffix, tooltip = self._describe_save()
            self.status.setText(f"Overview acquired — {shape[-1]} × {shape[-2]} px{suffix}")
            self.status.setToolTip(tooltip)
            self.progress_tiles.update_progress(ProgressUpdate.done())
            self.progress_tiles.show()
        elif state == "overview-cancelled":
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

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self._stop_event.set()
        # Every psygnal this widget subscribed to, without exception. They outlive the
        # widget -- they belong to the microscope -- and they hold bound methods of Qt
        # objects that `close` has already torn down on the C++ side, so the next emit
        # writes into freed memory. Closing the tab and then moving the stage from
        # anywhere else in the application was a hard segfault, not an exception.
        for signal, slot in (
            (self.fm.acquisition_progress_signal, self._on_progress),
            (self.microscope.stage_position_changed, self._on_stage_signal),
            (self.fm.objective.position_changed, self._on_objective_moved),
            # Subscribed since FIB-441 and never in this list, though the comment above
            # has always claimed "without exception".
            (self.fm.acquiring_changed, self._on_fm_acquiring_signal),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError, ValueError, KeyError):
                pass
        super().closeEvent(event)
