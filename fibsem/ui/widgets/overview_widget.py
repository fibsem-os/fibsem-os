"""The FIB/SEM overview: acquire a tiled overview, and steer the stage from it.

The beam counterpart of :class:`~fibsem.ui.fm.widgets.fm_overview_widget.FMOverviewWidget`,
and deliberately ignorant of lamellae and experiments in the same way. It says *where* a
user pointed and emits a request; a host that owns an experiment decides what a position
means. That is what lets this open standalone against a simulator and be built in a test
from a microscope alone.

The inversion
-------------
The tab this replaces is a real-space view, faked. It stitched one giant overview and
then reprojected stage positions onto its pixels, so everything downstream was gated on
the stitch having happened -- no stitch, nothing on screen.

Here that inverts. Each tile is placed at the stage position it was acquired at, as it
arrives, on a canvas whose coordinates *are* stage space. Stitching becomes a display
nicety: it is still what gets written to disk, but nothing on screen waits for it.

The placement is also more honest than the stitch buffer's. That buffer copies each tile
to an integer pixel offset, so the error against the true stage position accumulates
across the grid (FIB-399); a tile placed from its own recorded position has no such
drift, and a stage that did not land exactly where it was asked to shows up as a tile
that is slightly off its neighbours rather than as a silently wrong mosaic.

What it does not read
---------------------
No hardware on a UI event. Where the stage is arrives by subscription and is cached; the
projection is built once and kept. The widget this replaces polled
``get_microscope_state()`` whenever an experiment was loaded and ``get_scan_rotation()``
on every click-to-move -- both of which, on a TFS system, take the shared imaging
channel (FIB-544, FIB-600).
"""

from __future__ import annotations

import logging
import os
import threading
from copy import deepcopy
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt
from superqt import ensure_main_thread

from fibsem import constants
from fibsem.imaging import tiled
from fibsem.imaging.reduce import downsample
from fibsem.microscope import FibsemMicroscope
from fibsem.projection import BeamStageProjection
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
)
from fibsem.ui import notification_service, stylesheets
from fibsem.ui import utils as ui_utils
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.tokens import (
    CURRENT_POSITION_COLOUR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
    SEMANTIC_WARNING_COLOR,
)
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.gridbar_overlay import GridBarOverlay
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas
from fibsem.ui.widgets.canvas.stage_frame import StageFrame
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    TitledPanel,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    OverviewAcquisitionSettingsWidget,
)
from fibsem.ui.widgets.overview_list_widget import OverviewListWidget
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate

logger = logging.getLogger(__name__)

# Structural context rather than anything a user marked, so muted -- the same argument
# the FM overview makes for its holder slots, and the same colour, so the two tabs draw
# the sample holder identically.
SLOT_COLOUR = "#90a4ae"
LIMITS_COLOUR = "#ffca28"
GRID_BOUNDARY_COLOUR = "#ff5252"

# The canvas key the in-progress mosaic is drawn under. Its own, so a run that dies
# leaves no half-filled overview behind pretending to be a finished one.
PREVIEW_KEY = "acquisition-preview"

# The grid boundary a cryo holder's slot describes, as a radius in metres. Carried over
# from the widget this replaces, where it was written inline as `1000e-6 / pixelsize`.
GRID_BOUNDARY_RADIUS = 1000e-6
# Cryo grid bar defaults, in microns -- the values the tab this replaces carried in
# `GRIDBAR_IMAGE_LAYER_PROPERTIES`, which went with the napari layer they configured.
DEFAULT_GRIDBAR_SPACING_UM = 100.0
DEFAULT_GRIDBAR_WIDTH_UM = 20.0

# Screen-space hit radius for picking a marker, matching the FM overview's. In pixels
# rather than stage microns so how close you have to click does not change with zoom.
PICK_RADIUS_PX = 12


class _PlacedTile(NamedTuple):
    """One image a record placed: the pixels, where it was taken, and what it covers.

    `covers` is the ground in metres, measured from the image's *original* shape before
    reduction. It has to be carried rather than re-derived, because `data` is decimated
    for display: the canvas sizes an image from `shape x pixel_size` unless told
    otherwise, so a 1024 px tile stored at 512 px was drawn covering half the ground it
    actually images -- a mosaic with black gaps between every tile, at the right
    positions and the wrong size.
    """

    data: object  # np.ndarray, display-reduced
    position: FibsemStagePosition
    pixel_size: float
    covers: Tuple[float, float]  # (width, height) in metres


class OverviewView(NamedTuple):
    """One way of looking at the sample: a beam, and a stage orientation.

    Two images register with each other only if they were taken through the same beam
    with the stage at the same orientation. Otherwise they are pictures from different
    directions -- foreshortened relative to each other, and on a stage that rotates,
    mirrored as well -- so compositing them on one canvas says something untrue.

    Derived, never asked for. An image records its own beam in
    `image_settings.beam_type` and its own pose in `microscope_state.stage_position`,
    and `get_stage_orientation` turns the pose into a name using nothing but arithmetic
    over the configured orientations. Verified: supplying the pose costs **zero**
    hardware reads, which is what lets a view be derived on every overlay refresh.
    """

    beam_type: BeamType
    orientation: str  # "SEM" / "FIB" / "MILLING" / "NONE"

    @property
    def label(self) -> str:
        return f"{self.orientation} · {self.beam_type.name.title()}"


class OverviewRecord:
    """One overview on the canvas, and the canvas keys holding it.

    A run places one image per tile, so "the overview" is a set of keys rather than a
    single one -- which is why this exists at all. Showing, hiding and removing act on
    the run, because that is the thing a user acquired and the thing they mean.

    Structurally compatible with the fluorescence side's `PlacedOverviewImageRecord`:
    `OverviewListWidget` reads `id`, `label`, `detail` and `visible` off whatever it is
    given, and both overviews use the same list.
    """

    def __init__(self, record_id: str, label: str, keys: List[str],
                 view: Optional["OverviewView"] = None) -> None:
        self.id = record_id
        self.label = label
        self.keys = list(keys)
        self.visible = True
        self.pixel_size: Optional[float] = None
        # Which way of looking at the sample this was acquired in. Records outlive the
        # canvas's contents -- switching view clears the images and re-places only the
        # ones belonging to the new view -- so this is what says which those are.
        self.view: Optional["OverviewView"] = view
        # What was placed, kept so a view switch can re-place it. Display-reduced --
        # see `FibsemOverviewWidget._stored_tile`.
        self.images: List["_PlacedTile"] = []
        # How many tiles the run has acquired, once something says. An overview is one
        # image on the canvas now, so the images it holds no longer count them, and a
        # mosaic loaded from disk cannot say how many it was made of -- which is why
        # this is None rather than 1.
        self.tiles: Optional[int] = None

    @property
    def detail(self) -> str:
        """The tile count and scale, for the second half of a list row.

        Tiles rather than a grid shape: a run can be cancelled part way, and saying
        "3x3" for an overview holding four tiles would describe what was asked for
        rather than what is on the canvas.
        """
        parts = []
        # Counted by the run, not derived from what is on the canvas: an overview is a
        # single placed image, so there is nothing on the canvas left to count, and the
        # keys are cleared anyway while another view is displayed.
        count = self.tiles
        if count:
            parts.append(f"{count} tile{'s' if count != 1 else ''}")
        if self.pixel_size:
            parts.append(f"{self.pixel_size * constants.SI_TO_MICRO:.2f} µm/px")
        if self.view is not None:
            parts.append(self.view.label)
        return " · ".join(parts)


class FibsemOverviewWidget(QWidget):
    """Configure, run and view a tiled FIB/SEM overview on a real-space canvas."""

    overview_acquired = pyqtSignal(object)  # FibsemImage (the stitched mosaic)

    # A user right-clicked the canvas and asked for a position there. Requests, not
    # commands: this widget knows nothing about lamellae, so a host that owns an
    # experiment decides what a position means and whether to ask first.
    position_add_requested = pyqtSignal(object)  # FibsemStagePosition
    position_move_requested = pyqtSignal(str, object)  # name, FibsemStagePosition
    # A marked position was clicked, by the name it was marked under.
    position_selected = pyqtSignal(str)

    # Internal hops from a worker thread to the GUI thread. `tiled_acquisition_signal`
    # and `stage_position_changed` are psygnals, which call their callbacks
    # synchronously on whichever thread emitted -- during a run, the acquisition
    # worker. Touching widgets from there is a cross-thread GUI access; re-emitting as
    # a Qt signal gets it queued onto the GUI thread, because this widget lives there.
    _progress_received = pyqtSignal(dict)
    _stage_moved = pyqtSignal(object)
    _acquisition_finished = pyqtSignal(dict)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope

        self._stop_event = threading.Event()
        self._worker: Optional[FunctionWorker] = None
        self._records: Dict[str, OverviewRecord] = {}
        self._record_count = 0
        # Stage position the canvas frame is built around. Fixed once and kept:
        # re-deriving it from whatever arrived last would shift the whole scene each
        # time a tile landed. Taken from the first image placed.
        # One origin and one projection *per view*. A view is a direction the sample
        # is seen from, and everything placed in it is placed relative to that view's
        # own anchor -- so a single origin would put the FIB overview's tiles wherever
        # the SEM one happened to start.
        self._origins: Dict["OverviewView", FibsemStagePosition] = {}
        self._projections: Dict["OverviewView", BeamStageProjection] = {}
        # Views anchored on the stage rather than on an image, so the tab could draw
        # before anything was acquired. The first image in one replaces its anchor --
        # see `_seed_frame` and `_set_origin_from`.
        self._provisional: Set["OverviewView"] = set()
        # The view the canvas is showing. None only before the stage position and the
        # settings are both known; `_seed_frame` fills it in from where the stage is.
        self._current_view: Optional["OverviewView"] = None
        # The beam the settings last named, so a change of beam can be told from any
        # other settings change. See `_follow_the_planned_beam`.
        self._planned_beam: Optional[BeamType] = None
        # Whether the displayed view has been decided by something other than where the
        # stage happens to be -- an image arriving, or the user picking one. Until then
        # the canvas is a live plan of the next run and follows the stage.
        self._view_pinned = False
        # Where the next run is planned around, if the grid has been dragged off the
        # stage. None means "wherever the stage is", which is also what the runner
        # falls back to -- so the drawn grid and the acquisition agree by default.
        self._target: Optional[FibsemStagePosition] = None
        self._positions: List[FibsemStagePosition] = []
        self._selected_position: Optional[str] = None
        # Positions a host has flagged, by name. What "flagged" means is the host's
        # business -- for an experiment it is a lamella marked defective. Kept as names
        # rather than as a predicate on the position so this stays ignorant of whatever
        # the host is really looking at.
        self._flagged: Set[str] = set()
        # Where the stage is, as far as anyone has told us. Cached rather than polled
        # so that everything drawn from it agrees by construction rather than because
        # two callers happened to poll together.
        self._stage_position: Optional[FibsemStagePosition] = None
        self._save_directory: Optional[str] = None
        self._mosaic: Optional[FibsemImage] = None
        # Whether a run is under way, and whether a host is allowing one to be started.
        # Two independent facts kept apart on purpose: a workflow ending must not
        # re-enable a tab whose acquisition is still going, nor the reverse.
        self._running = False
        self._interactive = True
        self._tiles_acquired = 0

        self._init_ui()

        self._progress_received.connect(self._apply_progress)
        self._stage_moved.connect(self._on_stage_moved)
        self._acquisition_finished.connect(self._on_finished)
        # Plain bound methods, never a Qt signal's `.emit`. psygnal holds bound methods
        # weakly and drops them when the owner is collected, which a Qt signal's `emit`
        # does not get: PyQt builds a new wrapper on every access, so psygnal cannot
        # weakref it, cannot match it at disconnect time -- and says nothing when the
        # disconnect therefore removes nothing. Emitting into a widget Qt had already
        # torn down was a segfault.
        self.microscope.tiled_acquisition_signal.connect(self._on_progress)
        self.microscope.stage_position_changed.connect(self._on_stage_signal)

        self._refresh_current_position()
        self._refresh_context_overlays()
        self._apply_enabled_state()

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.canvas = FibsemRealSpaceCanvas()
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)

        # The grid's bars, where they should be. Off by default: it is a reference you
        # turn on to check the overview against, not something to read the sample
        # through. Added first so it sits under everything else.
        self.gridbar_overlay = GridBarOverlay()
        self.canvas.add_overlay(self.gridbar_overlay)

        # What the next run would acquire, tile by tile. Clickable: a tile toggles in
        # or out, an edge resizes the grid, the interior drags it somewhere else.
        self.tile_grid_overlay = TileGridOverlay()
        self.tile_grid_overlay.tile_toggled.connect(self._on_tile_toggled)
        self.tile_grid_overlay.grid_resize_requested.connect(self._on_grid_resized)
        self.tile_grid_overlay.grid_move_requested.connect(self._on_grid_moved)
        self.canvas.add_overlay(self.tile_grid_overlay)

        # Where the sample and the stage can physically go -- the context an overview is
        # read against. Added before the position markers so it sits beneath them.
        self.context_overlay = MinimapShapesOverlay(zorder=4.0, crosshair_half_px=24)
        self.canvas.add_overlay(self.context_overlay)

        # Where the stage is now. Distinct from the red origin marker the canvas draws:
        # the origin explains why everything sits where it does, this is what you steer
        # by. They coincide until the stage moves, then diverge.
        self.current_position_overlay = PointsOverlay(
            color=CURRENT_POSITION_COLOUR, marker="+", size=13
        )
        self.canvas.add_overlay(self.current_position_overlay)

        # Crosshairs rather than dots: a marked position is a point on the sample, and a
        # filled dot covers the feature it is naming.
        self.position_overlay = PointsOverlay(
            color=SAVED_POSITION_COLOUR, marker="+", size=11
        )
        self.canvas.add_overlay(self.position_overlay)
        # Flagged positions, on their own layer for the same reason the selection has
        # one: `PointsOverlay` paints every point the same colour. Worth the third
        # layer here rather than collapsing it into the others -- a lamella marked
        # defective is one you should not be re-targeting, and the tab this replaces
        # said so in colour.
        self.flagged_position_overlay = PointsOverlay(
            color=SEMANTIC_WARNING_COLOR, marker="+", size=11
        )
        self.canvas.add_overlay(self.flagged_position_overlay)
        # The selected position on its own layer rather than as a colour within the one
        # above: `PointsOverlay` paints every point the same. Added last, so it draws
        # over its unselected neighbours where markers crowd together.
        self.selected_position_overlay = PointsOverlay(
            color=SELECTED_POSITION_COLOUR, marker="+", size=15
        )
        self.canvas.add_overlay(self.selected_position_overlay)

        self.settings_widget = OverviewAcquisitionSettingsWidget(self)
        self.settings_widget.settings_changed.connect(self._on_settings_changed)

        # Which way of looking at the sample the canvas is showing. Populated from what
        # has actually been placed -- an empty selector means nothing is on the canvas
        # yet, which is the honest state rather than a list of hypotheticals.
        self.combo_view = ValueComboBox()
        self.combo_view.currentIndexChanged.connect(self._on_view_selected)
        self.label_view_note = QLabel("")
        self.label_view_note.setWordWrap(True)
        self.label_view_note.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)

        self.checkbox_gridbars = QCheckBox("Show grid bars")
        self.checkbox_gridbars.toggled.connect(self._on_gridbars_toggled)
        self.spin_gridbar_spacing = ValueSpinBox(
            suffix=" um", minimum=1.0, maximum=10000.0, step=10.0, decimals=1
        )
        self.spin_gridbar_spacing.setValue(DEFAULT_GRIDBAR_SPACING_UM)
        self.spin_gridbar_width = ValueSpinBox(
            suffix=" um", minimum=0.1, maximum=10000.0, step=5.0, decimals=1
        )
        self.spin_gridbar_width.setValue(DEFAULT_GRIDBAR_WIDTH_UM)
        for _spin in (self.spin_gridbar_spacing, self.spin_gridbar_width):
            _spin.valueChanged.connect(self._refresh_gridbars)

        self.button_acquire = QPushButton("Run Tile Collection")
        self.button_acquire.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_acquire.clicked.connect(self.acquire)
        self.button_cancel = QPushButton("Cancel Acquisition")
        self.button_cancel.setStyleSheet(stylesheets.DANGER_BUTTON_STYLESHEET)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setVisible(False)
        self.button_load = QPushButton("Load Overview")
        self.button_load.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_load.clicked.connect(self._prompt_for_overview)

        self.label_status = QLabel("")
        self.label_status.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.label_status.setWordWrap(True)

        # Empty for most of its life, and hidden while it is: it sits at the bottom of
        # a stack with nothing beside it, so it can come and go without moving anything.
        self.progress = FibsemProgressWidget()
        self.progress.reset()

        # What is already on the canvas, above the settings for the next run: reading
        # the column top to bottom then follows the same order as using the tab -- look
        # at what you have, then set up what comes next. Same order as the FM overview.
        self.overview_list = OverviewListWidget()
        self.overview_list.visibility_toggled.connect(self.set_overview_visible)
        self.overview_list.remove_requested.connect(self.remove_overview)
        overviews_panel = self._section("Overviews", self.overview_list)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self._controls_layout = controls_layout
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(overviews_panel)
        controls_layout.addWidget(self.settings_widget)
        controls_layout.addWidget(self._section("Display", self._display_panel()))
        controls_layout.addWidget(self._section("Overview", self._acquisition_panel()))
        controls_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # No horizontal scrollbar, so the column cannot be squeezed narrower than its
        # controls and hide half of them off the edge -- it holds its width and the
        # canvas gives way instead. Which means each control's own minimum is what
        # decides how narrow this can get: the tile-count spinboxes carry one, because
        # below it they stop drawing their digits (see the settings widget).
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(controls)

        # A splitter rather than a fixed column, so a user can give the canvas the whole
        # window on a small screen. The canvas takes the extra room as the window grows.
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _acquisition_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.button_acquire)
        layout.addWidget(self.button_cancel)
        layout.addWidget(self.button_load)
        layout.addWidget(self.progress)
        layout.addWidget(self.label_status)
        return panel

    def _display_panel(self) -> QWidget:
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addRow("View", self.combo_view)
        layout.addRow(self.label_view_note)
        layout.addRow(self.checkbox_gridbars)
        layout.addRow("Bar spacing", self.spin_gridbar_spacing)
        layout.addRow("Bar width", self.spin_gridbar_width)
        return panel

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── grid bars ────────────────────────────────────────────────────────

    # ── the view selector ────────────────────────────────────────────────

    def _on_view_selected(self, _index: int) -> None:
        view = self.combo_view.value()
        if isinstance(view, OverviewView):
            # Chosen, so the canvas stops following the stage -- otherwise re-posing
            # would silently undo the choice.
            self._view_pinned = True
            self.show_view(view)

    def _refresh_view_selector(self) -> None:
        """List the views worth switching to, and say what is not shown.

        Everything something has been placed in, plus the one being shown, plus **the
        one the next run would land in** even if nothing has been acquired there. That
        last is what makes a re-posed stage usable: the planned grid refuses to draw on
        a view the run will not appear in, so after moving to the milling pose the plan
        is somewhere else -- and without an entry for it there is no way to go and look.

        Signals blocked while repopulating: setting the items fires
        `currentIndexChanged`, which would call `show_view` and switch the canvas to
        whatever landed at index 0.
        """
        views = list(self.views)
        for view in (self._current_view, self.acquisition_view):
            if view is not None and view not in views:
                views.append(view)

        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        for view in views:
            self.combo_view.addItem(view.label, view)
        if self._current_view is not None and self._current_view in views:
            self.combo_view.setCurrentIndex(views.index(self._current_view))
        self.combo_view.blockSignals(False)
        self.combo_view.setEnabled(len(views) > 1)
        self._refresh_view_note()

    def _refresh_view_note(self) -> None:
        """Say when the canvas is not showing where the stage is pointing.

        Silent when they agree, which is the normal case -- a note that is always there
        stops being read. The one it has to make is the confusing one: the planned
        footprint is missing because the next run would land in a different view.
        """
        acquisition = self.acquisition_view
        if (
            acquisition is None
            or self._current_view is None
            or acquisition == self._current_view
        ):
            self.label_view_note.clear()
            self.label_view_note.setVisible(False)
            return
        self.label_view_note.setText(
            f"Showing {self._current_view.label}; the stage is at "
            f"{acquisition.label}, so the next acquisition would not appear here."
        )
        self.label_view_note.setVisible(True)

    def _on_gridbars_toggled(self, checked: bool) -> None:
        # The pitch controls only mean anything while the bars are drawn.
        self.spin_gridbar_spacing.setEnabled(checked)
        self.spin_gridbar_width.setEnabled(checked)
        self.gridbar_overlay.set_visible(checked)
        if checked:
            self._refresh_gridbars()

    def _refresh_gridbars(self) -> None:
        """Re-measure the lattice against the canvas scale.

        In metres through the frame rather than in pixels: the bars are a physical
        feature of the holder, so a spacing set in microns has to stay that spacing
        whatever the overview was acquired at. Silently does nothing before there is a
        frame -- the controls are usable from the moment the tab opens, and nothing is
        on screen to reference yet.

        The pitch is still `frame.length()`, and still one number for both axes, which
        is wrong in the same way the travel envelope was: a square lattice on the sample
        is not square in a tilted view, so at the milling pose the horizontal bars sit
        about four times too far apart. Deliberately left (FIB-615) -- the fix is
        `_canvas_span` and a per-axis `set_lattice`, and it is not what anyone is
        waiting on.

        The lattice *centre* is fixed here, because it was a different bug: grid centre
        is a place, and built with its own rotation it read as a position recorded half
        a turn away. See :meth:`_landmark`.
        """
        if not self.checkbox_gridbars.isChecked():
            return
        frame = self._frame()
        if frame is None:
            return
        try:
            centre = frame.to_canvas(self._landmark(frame, 0.0, 0.0, "Grid Centre"))
            pitch = frame.length(
                self.spin_gridbar_spacing.value() * constants.MICRO_TO_SI
            )
            width = frame.length(
                self.spin_gridbar_width.value() * constants.MICRO_TO_SI
            )
        except Exception as e:
            logger.debug(f"Could not place the grid bars: {e}")
            return
        self.gridbar_overlay.set_lattice(centre, pitch, width)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _settings(self) -> Optional[OverviewAcquisitionSettings]:
        """The planned acquisition, read from the settings widget.

        Note what `OverviewAcquisitionSettingsWidget.get_settings()` actually does: it
        *mutates and returns the widget's own* `ImageSettings`, resetting `path` from
        its text box among other fields. So the object it hands back is shared, and
        every later call rewrites it.

        That is safe to read from here only because `acquire` deep-copies before giving
        anything to the runner. Before it did, this exact call -- reached from an
        overlay refresh when the stage moved between tiles -- reset a running
        acquisition's output path to None, and the second tile died in
        `os.path.join(None, filename)`. The first tile had already succeeded, so it
        looked like a failure part-way through a run rather than a configuration
        problem.

        Deliberately not cached. A cache would go stale the moment a host called
        `update_from_settings`, which suppresses `settings_changed` while it populates
        -- and a silently stale plan is worse than re-reading a few spinboxes. These
        are widget reads, not hardware.
        """
        try:
            return self.settings_widget.get_settings()
        except Exception as e:
            logger.debug(f"Could not read the overview settings: {e}")
            return None

    @property
    def beam_type(self) -> BeamType:
        """The beam the overview is acquired and projected in.

        Read rather than kept, so the projection cannot describe one beam while the
        acquisition uses the other.
        """
        settings = self._settings()
        if settings is None:
            return BeamType.ELECTRON
        return settings.image_settings.beam_type

    def add_settings_section(self, title: str, widget: QWidget, first: bool = True) -> None:
        """Let a host put its own section in the settings column.

        In the column rather than beside it: a host's section is usually the subject of
        the tab -- the lamella positions, here -- and a column of its own would read as
        a third pane. `first` puts it at the top, above what this widget owns, because
        the host's subject outranks the settings for the next acquisition.
        """
        section = self._section(title, widget)
        if first:
            self._controls_layout.insertWidget(0, section)
        else:
            # Before the trailing stretch, or it lands below the blank space.
            self._controls_layout.insertWidget(
                self._controls_layout.count() - 1, section
            )

    def set_save_directory(self, path: Optional[str]) -> None:
        """Where acquired overviews are written. None means nowhere.

        The widget opens standalone against a simulator as often as it runs inside an
        experiment, and inventing a directory for those runs would scatter files through
        whatever working directory it happened to be launched from.

        Written into the settings widget rather than only kept here, so a user can see
        where a run will go and change it. It is also what stops `get_settings()`
        reading an empty box back as `path=None`.
        """
        self._save_directory = path
        if path:
            try:
                self.settings_widget.image_settings_widget.path_edit.setText(str(path))
            except Exception as e:
                logger.debug(f"Could not show the save directory: {e}")

    def set_interactive(self, enabled: bool) -> None:
        """Allow or forbid starting work, for a host that has taken the instrument."""
        self._interactive = bool(enabled)
        self._apply_enabled_state()

    def _apply_enabled_state(self) -> None:
        running = self._running
        # Nothing selected is a real state now that tiles can be clicked off, and it
        # is one the runner cannot do anything with. Refused here as well as in
        # `acquire`, so it reads as unavailable rather than failing when pressed.
        has_tiles = self._planned_tile_count() > 0
        self.button_acquire.setEnabled(self._interactive and not running and has_tiles)
        self.button_acquire.setToolTip(
            "" if has_tiles else "No tiles are selected. Click a tile to include it."
        )
        self.button_acquire.setText(
            "Running Tile Collection…" if running else "Run Tile Collection"
        )
        self.button_cancel.setVisible(running)
        self.settings_widget.setEnabled(self._interactive and not running)

    def _planned_tile_count(self) -> int:
        """How many tiles the next run would acquire. 0 if the settings cannot be read."""
        settings = self._settings()
        return 0 if settings is None else settings.n_enabled_tiles

    def _on_settings_changed(self) -> None:
        """The planned overview changed shape, so redraw what it would cover."""
        self._follow_the_planned_beam()
        if self.tile_grid_overlay.is_dragging:
            # Dragging an edge writes rows and columns to the settings widget, so this
            # runs on every motion event of a resize too -- and the grid is the only
            # thing that changed. `TileGridOverlay` asks its host to keep this cheap
            # for exactly this reason.
            self._refresh_tile_grid()
            # Cheap, and it has to keep up: resizing drops a mask that no longer fits
            # the grid, which can take the tile count from zero back to full. Left to
            # the end of the drag the button would stay disabled with tiles selected.
            self._apply_enabled_state()
            return
        self._refresh_context_overlays()
        # Masking the last tile off has to reach the button, and a tile is toggled from
        # the canvas rather than from the controls that already refresh it.
        self._apply_enabled_state()

    def _follow_the_planned_beam(self) -> None:
        """Show the view the next run would land in when the beam changes.

        Choosing a beam is a statement about the *next acquisition*, so the canvas has
        to show where that run will appear. Without this the planned grid vanishes at
        exactly the moment you are planning: it refuses to draw on a view the run will
        not land in, and switching beam is precisely what makes the displayed view stop
        being that one.

        Deliberately not the same treatment a stage move gets. Moving the stage is not
        a statement about what to look at, and following it would drag the reader off
        the overview they are reading -- which is what the view pin exists to prevent.
        Choosing a beam has no other meaning.
        """
        beam = self.beam_type
        if beam == self._planned_beam:
            return
        self._planned_beam = beam
        view = self.acquisition_view
        if view is not None and view != self._current_view:
            self.show_view(view)

    # ── the frame ────────────────────────────────────────────────────────

    def _projection(
        self, view: Optional["OverviewView"] = None, fresh: bool = False
    ) -> Optional[BeamStageProjection]:
        """The projection for a view, built once per view and kept.

        Kept because building it reads the scan rotation off the instrument, and doing
        that per use would be a hardware read on a mouse move. `fresh=True` forces a
        re-read for the one caller whose answer drives the stage, where a stale scan
        rotation would send it somewhere other than where the click was.

        Keyed by view rather than held singly. A single cache went stale the moment the
        beam selector changed: drawing kept using the electron projection (view tilt 0)
        while a click resolved through the ion one (0.91 rad), so markers were drawn in
        one projection and clicks answered in another.
        """
        view = view or self._current_view
        if view is None:
            return None
        if fresh or view not in self._projections:
            projection = BeamStageProjection.from_microscope(
                self.microscope, view.beam_type
            )
            if projection is None:
                return self._projections.get(view)
            self._projections[view] = projection
        return self._projections[view]

    def invalidate_projection(self) -> None:
        """Force every projection to be re-read. For a host that changed the instrument."""
        self._projections.clear()

    def _seed_frame(self) -> None:
        """Give the canvas a frame before anything has been acquired.

        A frame needs three things: a projection, a scale and an origin. The projection
        is read from the instrument, but the other two used to arrive only with the
        first image -- so the tab opened blank. No travel limits, no grid boundary, no
        holder slots, no lamella markers, not even the stage. Everything appeared at
        once when the first tile landed, which is the moment it stopped being the most
        useful. The fluorescence overview has never behaved this way.

        Both are free here. The scale is `hfw / width` off the settings widget -- a
        widget read, not a device one -- and it is only "how many metres a canvas pixel
        is worth", so the first image being a little coarser or finer than planned does
        not matter: `add_image` scales each image by its own pixel size against this
        one. The origin is the stage position already cached for the markers.

        Anchored rather than re-derived on each call. Fixing it means the travel
        envelope stays put and the stage marker moves inside it, which is the way round
        that matches what the overlays describe; following the stage would pin the
        marker to the middle and slide the grid past it.

        Provisional, though, and marked as such: the first real image in the view
        replaces it, so a canvas that ends up holding data is anchored on the data
        rather than on wherever the stage happened to be when the tab was opened.
        """
        settings = self._settings()
        if settings is None:
            return

        if self.canvas.reference_pixel_size is None:
            try:
                width = settings.image_settings.resolution[0]
                pixel_size = settings.image_settings.hfw / width
            except Exception as e:
                logger.debug(f"Could not work out a canvas scale: {e}")
                pixel_size = None
            if pixel_size:
                self.canvas.set_reference_pixel_size(pixel_size)

        view = self.acquisition_view
        if view is None or self._stage_position is None:
            return

        # Follow the stage only while nothing has decided otherwise. Once an image has
        # been placed, or the user has picked a view, the displayed view is theirs and
        # re-posing the stage must not drag them off it -- that is what
        # `_refresh_view_note` is for.
        #
        # A pin rather than "is the canvas empty": `show_view` clears the canvas before
        # re-placing, and this runs from inside it, so an emptiness test flips the view
        # back to the stage's half way through switching to another one.
        if not self._view_pinned and view != self._current_view:
            self._current_view = view
            self._refresh_view_selector()

        if view not in self._origins:
            self._origins[view] = deepcopy(self._stage_position)
            self._provisional.add(view)

    def _frame(
        self, view: Optional["OverviewView"] = None, fresh: bool = False
    ) -> Optional[StageFrame]:
        """Stage positions and canvas coordinates, about a view's own origin.

        None until the view has an origin *and* the canvas has a scale. Both are seeded
        for the view the *next run* would produce (see :meth:`_seed_frame`), so this is
        None in practice only for a view nothing has been acquired in and the stage is
        not pointing at -- which has no honest origin to invent.

        A pure read: the seeding is done by `_refresh_context_overlays`, so the many
        callers that only want to draw or resolve a point cannot quietly re-anchor the
        canvas by asking.
        """
        view = view or self._current_view
        if view is None:
            return None
        origin = self._origins.get(view)
        projection = self._projection(view, fresh=fresh)
        if projection is None or origin is None:
            return None
        if self.canvas.reference_pixel_size is None:
            return None
        return StageFrame(self.canvas, origin, projection)

    def _set_origin_from(self, image: FibsemImage, view: "OverviewView") -> None:
        """Anchor a view on the first image placed in it, if it is not anchored yet.

        A *provisional* anchor -- one `_seed_frame` invented so the tab could draw
        before anything was acquired -- is replaced rather than kept. Nothing is placed
        in the view at that point, so re-anchoring moves nothing; it just stops the
        canvas being centred on wherever the stage happened to be when the tab opened,
        which can be millimetres from the data.
        """
        if view in self._origins and view not in self._provisional:
            return
        position = self._position_of(image)
        if position is None:
            return
        self._origins[view] = deepcopy(position)
        self._provisional.discard(view)

    # ── views ────────────────────────────────────────────────────────────

    def _view_of(self, image: FibsemImage) -> Optional["OverviewView"]:
        """The view an image was acquired in, from its own metadata alone."""
        metadata = getattr(image, "metadata", None)
        position = self._position_of(image)
        if metadata is None or position is None:
            return None
        try:
            beam_type = metadata.image_settings.beam_type
            orientation = self.microscope.get_stage_orientation(stage_position=position)
        except Exception as e:
            logger.debug(f"Could not tell which view an image belongs to: {e}")
            return None
        return OverviewView(beam_type=beam_type, orientation=orientation)

    @property
    def stage_orientation(self) -> Optional[str]:
        """The orientation the stage is in now, from the cached position.

        No hardware: `get_stage_orientation` with a pose supplied is arithmetic over
        the configured orientations.
        """
        if self._stage_position is None:
            return None
        try:
            return self.microscope.get_stage_orientation(
                stage_position=self._stage_position
            )
        except Exception as e:
            logger.debug(f"Could not tell which orientation the stage is in: {e}")
            return None

    @property
    def acquisition_view(self) -> Optional["OverviewView"]:
        """The view the *next* run would produce.

        Distinct from the displayed view, and deliberately: you can look at the SEM
        overview while the stage sits at FIB. Derived from the cached stage position, so
        it costs no hardware access -- `get_stage_orientation` with a pose supplied is
        arithmetic over the configured orientations.
        """
        orientation = self.stage_orientation
        if orientation is None:
            return None
        return OverviewView(beam_type=self.beam_type, orientation=orientation)

    @property
    def current_view(self) -> Optional["OverviewView"]:
        """The view the canvas is showing."""
        return self._current_view

    @property
    def views(self) -> List["OverviewView"]:
        """Every view something has been placed in, in the order they first appeared."""
        seen: List["OverviewView"] = []
        for record in self._records.values():
            if record.view is not None and record.view not in seen:
                seen.append(record.view)
        return seen

    def show_view(self, view: "OverviewView") -> bool:
        """Show one view, re-placing the images that belong to it.

        The canvas holds one view at a time because images from different views do not
        register. Switching is a re-place rather than N canvases: the images are already
        reduced for display, and one canvas means one set of overlays, one selection and
        one zoom rather than N of each kept in step.
        """
        if view == self._current_view:
            return False
        self._current_view = view
        self.canvas.clear_images()
        for record in self._records.values():
            record.keys = []
            if record.view != view:
                continue
            for image in record.images:
                key = self._place_on_canvas(image, view)
                if key is not None:
                    record.keys.append(key)
            for key in record.keys:
                self.canvas.set_image_visible(key, record.visible)
        self._refresh_view_selector()
        self._refresh_context_overlays()
        self._refresh_overview_list()
        return True

    @staticmethod
    def _position_of(image: FibsemImage) -> Optional[FibsemStagePosition]:
        metadata = getattr(image, "metadata", None)
        if metadata is None:
            return None
        return getattr(metadata, "stage_position", None)

    @staticmethod
    def _pixel_size_of(image: FibsemImage) -> Optional[float]:
        metadata = getattr(image, "metadata", None)
        pixel_size = getattr(metadata, "pixel_size", None)
        return getattr(pixel_size, "x", None)

    # ── placing images ───────────────────────────────────────────────────

    def place_image(self, image: FibsemImage, key: Optional[str] = None,
                    zorder: Optional[float] = None) -> Optional[str]:
        """Put one image on the canvas where it was acquired.

        Switches the canvas to the image's own view first, if it is not already there:
        you placed the image in order to look at it, and it would otherwise be recorded
        into a view nothing is showing.

        Returns the canvas key, or None if the image cannot be placed -- which is the
        case for anything acquired before the stage position and pixel size were
        recorded. Refused rather than placed at the origin: an image in the wrong place
        looks exactly like an image in the right place.
        """
        position = self._position_of(image)
        pixel_size = self._pixel_size_of(image)
        if position is None or not pixel_size:
            logger.debug("Cannot place an image with no stage position or pixel size.")
            return None

        view = self._view_of(image)
        if view is None:
            return None
        # An image decides the view from here on, so the canvas stops following the
        # stage. Set before the switch below, because `show_view` refreshes the
        # overlays and the seeding that runs there would otherwise pull the view back.
        self._view_pinned = True
        if self._current_view is None:
            self._current_view = view
            self._refresh_view_selector()
        elif view != self._current_view:
            self.show_view(view)

        self._set_origin_from(image, view)
        # The canvas needs a scale before a frame can exist, and the frame is what turns
        # a stage position into an offset. Usually seeded from the settings before any
        # image arrives (`_seed_frame`); this is the fallback for a widget that has been
        # handed an image without ever drawing. Which pixel size wins does not matter
        # beyond the units: `add_image` scales each image by its own against this one.
        if self.canvas.reference_pixel_size is None:
            self.canvas.set_reference_pixel_size(pixel_size)

        return self._place_on_canvas(
            self._stored_tile(image), view, key=key, zorder=zorder
        )

    def _place_on_canvas(
        self, tile: "_PlacedTile", view: "OverviewView",
        key: Optional[str] = None, zorder: Optional[float] = None,
    ) -> Optional[str]:
        """Draw a stored tile in *view*. Shared by first placement and re-placement."""
        frame = self._frame(view)
        if frame is None:
            return None
        try:
            centre = frame.offset(tile.position)
        except Exception as e:
            logger.debug(f"Could not place an image: {e}")
            return None
        placed = self.canvas.add_image(
            tile.data, centre=centre, pixel_size=tile.pixel_size,
            key=key, zorder=zorder, covers=tile.covers,
        )
        self._refresh_context_overlays()
        return placed

    def set_image(self, image: FibsemImage) -> Optional[str]:
        """Place a whole overview as one image, and record it as one.

        The path a *loaded* overview takes: a stitched mosaic off disk arrives as a
        single image with a single position. An acquired one arrives tile by tile
        instead -- see `_place_tile`.
        """
        self._record_count += 1
        record_id = f"overview-{self._record_count}"
        view = self._view_of(image)
        key = self.place_image(image, key=record_id)
        if key is None:
            notification_service.show_toast(
                "That image does not record where it was acquired, so it cannot be "
                "placed on the overview.",
                "warning",
            )
            return None
        record = OverviewRecord(record_id, os.path.basename(record_id), [key], view=view)
        record.pixel_size = self._pixel_size_of(image)
        record.images.append(self._stored_tile(image))
        self._records[record_id] = record
        self._refresh_overview_list()
        return record_id

    def _stored_tile(self, image: FibsemImage) -> "_PlacedTile":
        """What a record keeps so it can be re-placed after a view switch.

        The *display-reduced* array, not the original: it is exactly what the canvas
        holds anyway, so keeping it costs no more than the canvas already does -- where
        keeping full-resolution tiles would be hundreds of megabytes for a large
        tileset, to redraw something that is decimated on the way to the screen.
        """
        data = image.filtered_data
        pixel_size = self._pixel_size_of(image)
        height, width = data.shape[0], data.shape[1]
        return _PlacedTile(
            data=downsample(data, self.canvas._display_max_px),
            position=deepcopy(self._position_of(image)),
            pixel_size=pixel_size,
            # From the shape *before* reduction: this is the ground the image images.
            covers=(width * pixel_size, height * pixel_size),
        )

    def _show_preview(self, preview: FibsemImage) -> None:
        """Show the mosaic-so-far, as one image, under its own key.

        Stateless: it redisplays whatever it is handed rather than accumulating tiles
        of its own, which is what makes it correct when a frame is dropped.

        One image rather than one per tile, which is the whole point. Placing tiles
        individually put each one where the stage actually reached, which the stitch
        buffer cannot express (FIB-399) -- but that accuracy never survived the run:
        the buffer is what gets saved, so reloading the overview showed the nominal
        placement anyway. What it did cost was an artist per tile, and the canvas
        repaints every artist on every draw, so dragging anything slowed down by about
        2 ms for every tile ever acquired (FIB-627).

        Its own key, not the record's: an in-progress preview must not survive as a
        finished overview if the run dies, and the real stitch replaces it at the end.
        """
        if self.place_image(preview, key=PREVIEW_KEY) is None:
            logger.debug("Could not place the acquisition preview.")

    def _clear_preview(self) -> None:
        self.canvas.remove_image(PREVIEW_KEY)

    def _place_finished_mosaic(self, mosaic: FibsemImage) -> None:
        """Swap the preview for the stitched overview the run actually produced."""
        record = self._records.get(getattr(self, "_active_record", None) or "")
        self._clear_preview()
        if record is None:
            return
        record.view = self._view_of(mosaic)
        key = self.place_image(mosaic, key=record.id)
        if key is None:
            logger.debug("The finished overview could not be placed.")
            return
        record.keys = [key]
        record.images = [self._stored_tile(mosaic)]
        record.pixel_size = self._pixel_size_of(mosaic)
        self._refresh_overview_list()

    def set_overview_visible(self, record_id: str, visible: bool) -> bool:
        """Show or hide every tile of one overview. False if the id is unknown."""
        record = self._records.get(record_id)
        if record is None:
            return False
        record.visible = bool(visible)
        for key in record.keys:
            self.canvas.set_image_visible(key, record.visible)
        self._refresh_overview_list()
        return True

    def remove_overview(self, record_id: str) -> bool:
        """Take one overview off the canvas entirely. False if the id is unknown.

        Removing the last one puts the tab back where it started, so it goes back to
        following the stage too. Otherwise you would be left looking at an empty canvas
        pinned to a view nothing is in, which is the state this whole seeding exists to
        avoid -- and the pin came from an image that is no longer there to justify it.
        Cleared before the refresh below, so the canvas re-points in the same pass.

        Hidden is not removed: `set_overview_visible` leaves the record, and a view you
        can bring back with a checkbox is still a view you chose to be in.
        """
        record = self._records.pop(record_id, None)
        if record is None:
            return False
        for key in record.keys:
            self.canvas.remove_image(key)
        if not self._records:
            self._view_pinned = False
        self._refresh_context_overlays()
        self._refresh_overview_list()
        return True

    @property
    def overviews(self) -> List[OverviewRecord]:
        """Every overview on the canvas, oldest first."""
        return list(self._records.values())

    def _refresh_overview_list(self) -> None:
        """Rebuild the list from the records.

        Called wherever a record appears, disappears or gains a tile -- the row shows a
        tile count, so a run in progress has to keep up with itself.
        """
        self.overview_list.set_records(self.overviews)

    def load_overview(self, path: str) -> Optional[str]:
        """Load a saved overview from disk and place it. Returns its record id."""
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            notification_service.show_toast(f"Could not load {os.path.basename(path)}.",
                                            "error")
            return None
        return self.set_image(image)

    def _prompt_for_overview(self) -> None:
        path = ui_utils.open_existing_file_dialog(
            msg="Select an overview image to load",
            path=str(self._save_directory or os.getcwd()),
            _filter="Image Files (*.tif *.tiff)",
            parent=self,
        )
        if not path:
            return
        self.load_overview(path)

    # ── overlays ─────────────────────────────────────────────────────────

    def _refresh_context_overlays(self) -> None:
        """Redraw the stage limits, holder slots and planned overview footprint.

        Everything here is derived from configuration and the cached stage position, so
        it costs no hardware access and is safe to call on any UI event.

        The one place that anchors the canvas, so `_frame` stays a pure read: this runs
        on construction, on a stage move, on a settings change and on a view change,
        which is every moment the answer could have changed.
        """
        self._seed_frame()
        frame = self._frame()
        if frame is None:
            self.context_overlay.set_shapes([])
            return

        specs: List[ShapeSpec] = []
        specs.extend(self._limit_shapes(frame))
        specs.extend(self._slot_shapes(frame))
        self.context_overlay.set_shapes(specs)
        self._refresh_tile_grid()
        self._refresh_position_markers()
        self._refresh_gridbars()
        # The selector, not just the note: the list includes the view the next run
        # would land in, and that changes when the stage re-poses -- which does not
        # change the *displayed* view, so nothing else here would refresh it.
        self._refresh_view_selector()

    # ── the planned tileset ──────────────────────────────────────────────

    def _grid_centre(self) -> Optional[FibsemStagePosition]:
        """The stage position the next overview will be centred on.

        A target if the grid has been dragged somewhere, otherwise wherever the stage
        is -- which is what the runner falls back to, so the drawn grid and the
        acquisition agree without either being told about the other.
        """
        return self._target or self._stage_position

    def _refresh_tile_grid(self) -> None:
        """Redraw the planned tileset.

        Drawn as *tiles*, not as one rectangle: the rectangle said how much ground a
        run would cover and nothing else, and the things worth seeing before pressing
        the button are where the seams fall and which tiles are in.

        Only in the view the run would land in, for the same reason the footprint was:
        drawn on another view it promises coverage this canvas will never show. The
        lattice is regular in the displayed frame at any tilt, because tiles are
        *defined* by displayed-frame offsets handed to `project_stable_move` -- the
        tilt is absorbed into the stage moves, not into where the tiles appear, so
        none of the foreshortening in FIB-615 applies here.
        """
        settings = self._settings()
        centre = self._grid_centre()
        frame = self._frame()
        if (
            settings is None
            or centre is None
            or frame is None
            or self.acquisition_view != self._current_view
        ):
            self.tile_grid_overlay.clear()
            return

        width, height = settings.image_settings.resolution
        if not width or not settings.image_settings.hfw:
            self.tile_grid_overlay.clear()
            return

        try:
            tiles = tiled.compute_tile_grid(settings, mask=settings.tile_mask)
            anchor = self.canvas.metres_to_canvas(*frame.offset(centre))
        except Exception as e:
            logger.debug(f"Could not place the planned tileset: {e}")
            self.tile_grid_overlay.clear()
            return

        self.tile_grid_overlay.set_anchor(anchor)
        # No `display_pixel_size`: the overlay reads it off the canvas at draw time, so
        # the grid keeps describing the image underneath it when that image changes --
        # which it does mid-run, as tiles land.
        self.tile_grid_overlay.set_grid(
            tiles,
            (height, width),
            settings.image_settings.hfw / width,
            overlap=settings.overlap,
        )

    def _may_edit_the_plan(self) -> bool:
        """Whether a canvas gesture is allowed to change what the next run does.

        The same two facts the context menu is gated on. A run in progress is reading
        this plan, and a host that has taken the instrument is usually iterating an
        experiment -- neither is a moment to redraw the grid underneath.
        """
        return not (self._running or self.is_acquiring or not self._interactive)

    def _on_grid_resized(self, rows: int, cols: int) -> None:
        """An edge of the grid was dragged.

        Writes to the settings widget, which owns rows and columns: the canvas is a
        view of that state, not a second copy of it. The redraw comes back through
        `settings_changed`.
        """
        if not self._may_edit_the_plan():
            return
        self.settings_widget.set_grid_size(rows, cols)

    def _on_tile_toggled(self, row: int, col: int, enabled: bool) -> None:
        """A tile was clicked. Writes into the mask the settings widget carries."""
        if not self._may_edit_the_plan():
            return
        settings = self._settings()
        if settings is None:
            return
        mask = settings.tile_mask
        if mask is None:
            mask = [[True] * settings.ncols for _ in range(settings.nrows)]
        if not (0 <= row < len(mask) and 0 <= col < len(mask[row])):
            return
        mask[row][col] = enabled
        self.settings_widget.tile_mask = mask

    def _on_grid_moved(self, x: float, y: float) -> None:
        """The grid was dragged: plan the next overview around where it landed.

        Deliberately does *not* move the stage. Setting a run up and driving the
        instrument are separate acts, and a drag is exploratory -- you push the grid
        around to see what it would cover. The stage goes there when the run does.

        The resolved position keeps the stage's own rotation and tilt, like a click
        does, so the run stays in the view it was planned in and does not re-pose the
        stage to reach its own grid.
        """
        if not self._may_edit_the_plan():
            return
        frame = self._frame()
        if frame is None:
            return
        try:
            self._target = self._posed_like_the_stage(frame.to_stage(x, y))
        except Exception as e:
            logger.debug(f"Could not resolve the dragged grid position: {e}")
            return
        # Only the grid. A drag emits on every motion event, and the full context
        # refresh redraws the limits, the slots, every marker and the lattice as well
        # -- none of which move when the grid does. Measured on a canvas holding four
        # acquired tilesets: 90 ms per motion event, growing about 2 ms with every tile
        # ever placed, because each refresh repaints the whole canvas.
        self._refresh_tile_grid()

    def clear_target(self) -> None:
        """Plan the next overview around the stage position again."""
        if self._target is None:
            return
        self._target = None
        self._refresh_tile_grid()

    @property
    def target(self) -> Optional[FibsemStagePosition]:
        """Where the next run is planned around, or None for wherever the stage is."""
        return self._target

    @staticmethod
    def _landmark(frame: StageFrame, x: float, y: float, name: str = "") -> FibsemStagePosition:
        """A stage position that is a *place*, in the frame's own pose.

        Grid centre and the corners of the travel envelope are places, not recorded
        poses, so the rotation has to come from somewhere. Taking the frame's means
        `BeamStageProjection` sees no rotation difference and leaves the compucentric
        correction alone -- it is there to flip positions *recorded* half a turn away,
        and firing it on a synthetic landmark would move the travel limits by the
        instrument's compucentric calibration for no reason. Tilt is not read at all
        (the projection takes it from the base), and is carried only so the position
        is complete.
        """
        origin = frame.origin
        return FibsemStagePosition(
            name=name, x=x, y=y, z=0.0, r=origin.r, t=origin.t
        )

    def _canvas_span(self, frame: StageFrame, length: float) -> Tuple[float, float]:
        """A stage-space step of *length* metres, in canvas pixels, per axis.

        Not `frame.length()` twice. That divides by the canvas scale and stops, which
        is right for x and wrong for y: the view foreshortens stage y by a factor that
        changes with the beam and the pose -- 1.00 looking down the pose a beam is
        named after, 0.26 for the ion beam at the milling pose. A lattice or a boundary
        sized by the scale alone is the same size in every view, and therefore right in
        at most one of them.

        Measured through the frame rather than derived from the geometry, so it cannot
        disagree with where the frame puts a marker. Absolute, because a view can flip
        an axis and a span has no sign.
        """
        anchor = self._landmark(frame, 0.0, 0.0)
        ox, oy = frame.to_canvas(anchor)
        along_x, _ = frame.to_canvas(self._landmark(frame, length, 0.0))
        _, along_y = frame.to_canvas(self._landmark(frame, 0.0, length))
        return abs(along_x - ox), abs(along_y - oy)

    def _limit_shapes(self, frame: StageFrame) -> List[ShapeSpec]:
        """The travel limits, and the grid boundary a cryo holder describes."""
        limits = getattr(self.microscope._stage, "limits", None)
        if not limits or not self.microscope.stage_is_compustage:
            return []
        try:
            cx, cy = frame.to_canvas(self._landmark(frame, 0.0, 0.0, "Grid Centre"))
            # Every corner, not a width and a height: the projection can flip either
            # axis, and reading the envelope off the extremes of the projected corners
            # is true whatever it does to them. The box stays axis-aligned -- the
            # terms are a scale per axis and a possible flip of both.
            corners = [
                frame.to_canvas(self._landmark(frame, x, y))
                for x in (limits["x"].min, limits["x"].max)
                for y in (limits["y"].min, limits["y"].max)
            ]
            xs = [point[0] for point in corners]
            ys = [point[1] for point in corners]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            box_cx, box_cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            # A circle on the sample, so an ellipse on screen everywhere but the two
            # views where the beam looks down the pose it is named after.
            span_x, span_y = self._canvas_span(frame, GRID_BOUNDARY_RADIUS)
        except Exception as e:
            logger.debug(f"Could not draw the stage limits: {e}")
            return []
        return [
            ShapeSpec(kind="rect", cx=box_cx, cy=box_cy, width=width, height=height,
                      color=LIMITS_COLOUR, label="Stage Limits"),
            ShapeSpec(kind="ellipse", cx=cx, cy=cy,
                      width=2 * span_x, height=2 * span_y,
                      color=GRID_BOUNDARY_COLOUR, label="Grid Boundary"),
        ]

    def _slot_shapes(self, frame: StageFrame) -> List[ShapeSpec]:
        """The sample holder's slots, as crosshairs at their configured positions."""
        try:
            slots = self.microscope._stage.holder.slots.values()
        except Exception:
            return []
        specs = []
        for slot in slots:
            try:
                cx, cy = frame.to_canvas(slot.position)
            except Exception:
                continue
            specs.append(ShapeSpec(kind="crosshair", cx=cx, cy=cy,
                                   color=SLOT_COLOUR, label=slot.position.name or ""))
        return specs

    def _refresh_position_markers(self) -> None:
        """Redraw the current stage position and every marked position."""
        frame = self._frame()
        if frame is None:
            return

        if self._stage_position is not None:
            try:
                self.current_position_overlay.set_points([frame.to_canvas(self._stage_position)])
            except Exception as e:
                logger.debug(f"Could not mark the current stage position: {e}")

        unselected, unselected_labels = [], []
        flagged, flagged_labels = [], []
        selected = []
        for position in self._positions:
            try:
                point = frame.to_canvas(position)
            except Exception:
                continue
            name = position.name or ""
            # Selection wins over flagged: it is what the user is looking at right now,
            # and a defective lamella they have just clicked should still read as the
            # selected one.
            if name and name == self._selected_position:
                selected.append(point)
            elif name in self._flagged:
                flagged.append(point)
                flagged_labels.append(name)
            else:
                unselected.append(point)
                unselected_labels.append(name)
        self.position_overlay.set_points(unselected, labels=unselected_labels)
        self.flagged_position_overlay.set_points(flagged, labels=flagged_labels)
        self.selected_position_overlay.set_points(
            selected, labels=[self._selected_position] if selected else None
        )

    # ── the host's API ───────────────────────────────────────────────────

    def set_positions(
        self,
        positions: List[FibsemStagePosition],
        flagged: Optional[Iterable[str]] = None,
    ) -> None:
        """Mark these positions on the canvas, replacing whatever was marked.

        *flagged* names a subset to draw in the warning colour. The widget does not
        know or ask what that means; a host with an experiment passes the lamellae it
        considers defective.
        """
        self._positions = list(positions)
        self._flagged = set(flagged or ())
        self._refresh_position_markers()

    def set_selected_position(self, name: Optional[str]) -> None:
        """Highlight one marked position by name, or None to highlight nothing."""
        self._selected_position = name
        self._refresh_position_markers()

    @property
    def selected_position(self) -> Optional[str]:
        return self._selected_position

    # ── the stage ────────────────────────────────────────────────────────

    def _on_stage_signal(self, position: FibsemStagePosition) -> None:
        """Called by psygnal, on whichever thread moved the stage. Touches no widgets."""
        self._stage_moved.emit(position)

    @ensure_main_thread
    @pyqtSlot(object)
    def _on_stage_moved(self, position: FibsemStagePosition) -> None:
        self._stage_position = deepcopy(position)
        # Not during a run: the stage visits every tile, and re-drawing the planned
        # footprint at each one would drag it across the canvas as the acquisition
        # walked the grid.
        if self._running:
            self._refresh_position_markers()
            return
        self._refresh_context_overlays()

    def _refresh_current_position(self) -> None:
        """Seed the cached stage position once, at construction.

        The one read of the stage this widget does, and it is not on a UI event: without
        it nothing is marked until the stage happens to move, which on a tab that has
        just been opened is exactly when a user is looking.
        """
        try:
            self._stage_position = deepcopy(self.microscope.get_stage_position())
        except Exception as e:
            logger.debug(f"Could not read the stage position: {e}")

    def move_to(self, position: FibsemStagePosition) -> None:
        """Drive the stage to a position, off the GUI thread."""
        if self.is_acquiring:
            notification_service.show_toast(
                "Cannot move the stage during an acquisition.", "warning"
            )
            return
        worker = FunctionWorker(self._move_worker, position)
        worker.start()

    def _move_worker(self, target: FibsemStagePosition) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        try:
            self.microscope.safe_absolute_stage_movement(target)
        except Exception as e:
            logger.error(f"Could not move the stage: {e}", exc_info=True)
        # Publishes the new position through `stage_position_changed`, which is what
        # re-marks it -- rather than assuming the stage arrived exactly where it was
        # asked to, which on a real instrument it does not.
        try:
            self.microscope.get_stage_position()
        except Exception as e:
            logger.debug(f"Could not confirm the stage position after moving: {e}")

    # ── canvas interaction ───────────────────────────────────────────────

    def _on_canvas_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Select a marked position if the click landed on one.

        A miss leaves the selection alone rather than clearing it: a host syncs its own
        lists from the selection and their handlers ignore None, so an empty click would
        blank this canvas and nothing else.
        """
        name = self._position_at(x, y)
        if name is None:
            return
        self.set_selected_position(name)
        self.position_selected.emit(name)

    def _position_at(self, x: float, y: float) -> Optional[str]:
        """The marked position under a canvas point, or None.

        Measured on screen, not in data units: at a wide zoom every marker would be
        within any sensible micron radius of the click, and at a tight one none would be.
        """
        frame = self._frame()
        ax = getattr(self.canvas, "_ax", None)
        if frame is None or ax is None or not self._positions:
            return None
        try:
            transform = ax.transData
            click = transform.transform((x, y))
        except Exception as e:
            logger.debug(f"Could not resolve the click for picking: {e}")
            return None

        best_name, best_distance = None, float(PICK_RADIUS_PX)
        for position in self._positions:
            if not position.name:
                continue
            try:
                point = transform.transform(frame.to_canvas(position))
            except Exception:
                continue
            distance = ((click[0] - point[0]) ** 2 + (click[1] - point[1]) ** 2) ** 0.5
            if distance < best_distance:
                best_name, best_distance = position.name, distance
        return best_name

    def _on_canvas_double_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Move the stage to the double-clicked point.

        Double-click rather than a single click, matching the FM overview and the
        coincidence viewer: a single click is how the canvas is explored, and a stage
        that moved on every stray click would be unusable.
        """
        if not self._may_move():
            return
        target = self._stage_position_at(x, y)
        if target is None:
            return
        self.move_to(target)

    def _may_move(self) -> bool:
        """Whether driving the stage from this tab is allowed right now, and say if not."""
        if self.is_acquiring or self._running:
            notification_service.show_toast(
                "Cannot move the stage during an acquisition.", "warning"
            )
            return False
        if not self._interactive:
            notification_service.show_toast(
                "Cannot move the stage while a workflow is running.", "warning"
            )
            return False
        return True

    def _on_canvas_right_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Offer to put a position at the right-clicked point."""
        config = self._position_menu(x, y)
        if config is None:
            return
        ContextMenu(config, parent=self).show_at_cursor()

    def _position_menu(self, x: float, y: float) -> Optional[ContextMenuConfig]:
        """What right-clicking at a canvas point offers, or None to offer nothing.

        Separate from showing it because `show_at_cursor` runs a modal event loop, and
        everything worth checking -- whether the menu appears at all, what it offers, and
        which position each entry would request -- is decided here. A test that had to
        open the menu could only hang.
        """
        if self._running or self.is_acquiring or not self._interactive:
            notification_service.show_toast(
                "Cannot mark positions while an acquisition is running.", "warning"
            )
            return None

        target = self._stage_position_at(x, y)
        if target is None:
            return None

        config = ContextMenuConfig()
        config.add_action(
            "Add New Position Here",
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

        Reads the projection fresh: this is the one caller whose answer drives the
        instrument, and a stale scan rotation here would send the stage to the point
        rotated 180 degrees from the click.

        **Refuses when the displayed view is not the orientation the stage is in.** A
        click resolved through another view names a point as *that* view sees it, and
        reaching it would rotate and tilt the stage to match -- a far bigger move than
        clicking a picture looks like, and one that silently changes what the beam is
        pointing at. Marking is refused for the same reason: the position would be
        recorded at an orientation the instrument is not in.
        """
        if not self._view_matches_stage():
            return None

        frame = self._frame(fresh=True)
        if frame is None:
            return None
        try:
            target = frame.to_stage(x, y)
        except Exception as e:
            logger.debug(f"Could not resolve the clicked position: {e}")
            return None

        target = self._posed_like_the_stage(target)

        limits = getattr(self.microscope._stage, "limits", None)
        if limits and not target.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return None
        return target

    def _posed_like_the_stage(
        self, target: FibsemStagePosition
    ) -> FibsemStagePosition:
        """A resolved position, wearing the pose the stage is actually in.

        Neither steering nor planning reorients. A point on the canvas says *where on
        the sample*, not which way to look at it, so the stage's own rotation and tilt
        are kept and any move is pure x/y/z. Without this the position carries the view
        *origin's* pose, and a stage sitting at a slightly different tilt within the
        same orientation would be tilted back to it as a side effect.

        Shared by the click and by a dragged tile grid, so a run planned around a
        dragged centre cannot end up asking for a pose a click would have refused.
        """
        current = self._stage_position
        if current is not None:
            if current.r is not None:
                target.r = current.r
            if current.t is not None:
                target.t = current.t
        return target

    def _view_matches_stage(self) -> bool:
        """Whether the canvas is showing the orientation the stage is in, and say if not.

        The message names both ways out, because either can be the right one: you may
        have switched view to look at something, or moved the stage and not switched.
        """
        displayed = self._current_view
        stage = self.stage_orientation
        if displayed is None or stage is None:
            # Nothing placed yet, or the pose cannot be classified. Not a mismatch --
            # refusing here would block the tab on a system whose orientation is simply
            # not one of the named ones.
            return True
        if displayed.orientation == stage:
            return True
        notification_service.show_toast(
            f"This canvas shows the {displayed.orientation} view, but the stage is at "
            f"{stage}. Switch the view to {stage}, or move the stage to "
            f"{displayed.orientation}, to steer from here.",
            "warning",
        )
        return False

    @staticmethod
    def _describe(position: FibsemStagePosition) -> str:
        try:
            return position.pretty_string
        except Exception:
            return "the clicked position"

    # ── acquisition ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Start a tiled overview acquisition."""
        if self.is_acquiring:
            return
        # Deep-copied before anything else touches it. `get_settings()` returns the
        # settings widget's own instance, and handing that to a background runner means
        # any later read of the widget rewrites what the run is using.
        settings = deepcopy(self.settings_widget.get_settings())
        if not settings.image_settings.filename:
            notification_service.show_toast(
                "Please enter a filename for the overview.", "error"
            )
            return
        if settings.n_enabled_tiles == 0:
            notification_service.show_toast(
                "No tiles are selected, so there is nothing to acquire. "
                "Click a tile in the grid to include it.",
                "warning",
            )
            return
        settings.image_settings.save = True
        if not settings.image_settings.path:
            # Nothing in the path box and no host directory: the run still has to go
            # somewhere, and failing at the second tile with `os.path.join(None, ...)`
            # is the worst way to find out.
            settings.image_settings.path = self._save_directory or os.getcwd()

        # A new record before the first tile arrives, so every tile has somewhere to go
        # and the run is on the canvas from the moment it starts.
        self._record_count += 1
        self._active_record = f"overview-{self._record_count}"
        self._records[self._active_record] = OverviewRecord(
            self._active_record, settings.image_settings.filename, []
        )
        self._refresh_overview_list()

        self._stop_event.clear()
        self._set_running(True)
        # Copied with the settings: the target can be dragged again while the run is
        # under way, and the run has to keep the grid it was started with.
        self._worker = FunctionWorker(
            self._acquire_worker, settings, deepcopy(self._target)
        )
        self._worker.start()

    def _acquire_worker(
        self,
        settings: OverviewAcquisitionSettings,
        centre_position: Optional[FibsemStagePosition] = None,
    ) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        from fibsem.cancellation import OperationCancelledError

        result = {"cancelled": False, "error": None}
        try:
            self._mosaic = tiled.tiled_image_acquisition_and_stitch(
                microscope=self.microscope,
                settings=settings,
                stop_event=self._stop_event,
                centre_position=centre_position,
            )
            self.overview_acquired.emit(self._mosaic)
        except OperationCancelledError:
            logger.info("Overview acquisition cancelled")
            result["cancelled"] = True
        except Exception as e:
            logger.error(f"Overview acquisition failed: {e}", exc_info=True)
            result["error"] = str(e)
        finally:
            self._acquisition_finished.emit(result)

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logger.info("Cancelling overview acquisition")
        self._stop_event.set()
        self.label_status.setText("Cancelling…")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._apply_enabled_state()
        if running:
            self._tiles_acquired = 0
            # Cleared rather than set to "Starting…": the progress bar carries the
            # message for the whole run, and a label saying "Starting…" underneath one
            # reading "4 / 9" is the two of them disagreeing in public.
            self.label_status.clear()
            self.progress.reset()

    @ensure_main_thread
    @pyqtSlot(dict)
    def _on_finished(self, result: dict) -> None:
        self._worker = None
        self._stop_event.clear()
        self._running = False
        self._apply_enabled_state()
        # The bar has done its job; the label below it carries the outcome from here.
        self.progress.reset()
        if result.get("error"):
            notification_service.show_toast(str(result["error"]), "error")
            self.label_status.setText("Acquisition failed.")
            self._drop_unfinished_run()
        elif result.get("cancelled"):
            notification_service.show_toast("Tile collection cancelled.", "warning")
            self.label_status.setText(
                f"Cancelled after {self._tiles_acquired} tile(s)."
            )
            self._drop_unfinished_run()
        else:
            notification_service.show_toast("Tile collection finished.")
            self.label_status.setText(f"Acquired {self._tiles_acquired} tile(s).")
            if self._mosaic is not None:
                self._place_finished_mosaic(self._mosaic)
            else:
                self._drop_unfinished_run()
        # The stage went home at the end of the run, so the planned footprint moved.
        self._refresh_context_overlays()

    def _drop_unfinished_run(self) -> None:
        """Take the preview off the canvas, and the run's empty record with it.

        A cancelled or failed run has no stitched overview to swap in. The preview it
        left is a partial mosaic that was never saved, so keeping it would put
        something on the canvas that matches no file -- and leave a record listing an
        overview that does not exist. What was acquired is on disk as tiles either way.
        """
        self._clear_preview()
        record_id = getattr(self, "_active_record", None)
        record = self._records.get(record_id or "")
        if record is not None and not record.keys:
            self._records.pop(record_id, None)
            self._refresh_overview_list()
        self._active_record = None

    # ── progress ─────────────────────────────────────────────────────────

    def _on_progress(self, payload: dict) -> None:
        """Called by psygnal, on whichever thread emitted. Touches no widgets."""
        self._progress_received.emit(payload)

    @ensure_main_thread
    @pyqtSlot(dict)
    def _apply_progress(self, payload: dict) -> None:
        """Runs on the GUI thread, queued via `_progress_received`.

        Reads with `.get`, not indexing: this signal is emitted from several places with
        several shapes, and the terminal update carries none of the per-tile keys.
        """
        counter = payload.get("counter")
        total = payload.get("total")
        if counter is not None and total:
            self._tiles_acquired = counter
            self.progress.update_progress(
                ProgressUpdate.numeric(
                    current=counter,
                    total=total,
                    message=payload.get("msg", "Acquiring"),
                )
            )

        preview = payload.get("preview")
        record = self._records.get(getattr(self, "_active_record", None) or "")
        if preview is not None and record is not None:
            self._show_preview(preview)
            # The row says how many tiles this run has, and it says it while the run is
            # going -- a row reading "0 tiles" beside a filling mosaic is the list
            # contradicting the canvas.
            if counter:
                record.tiles = counter
                record.pixel_size = self._pixel_size_of(preview)
                self._refresh_overview_list()

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
            (self.microscope.tiled_acquisition_signal, self._on_progress),
            (self.microscope.stage_position_changed, self._on_stage_signal),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError, ValueError, KeyError):
                pass
        super().closeEvent(event)


def main():  # pragma: no cover - manual harness
    """Open the overview against a simulator.

    Deliberately does NOT style the QApplication: this widget is hosted in a tab, and a
    harness that styles the app proves nothing about how it looks there. See
    `feedback_harness_friendlier_than_reality` -- a render script that supplied an
    app-level stylesheet hid a defect that shipped.
    """
    import sys

    from PyQt5.QtWidgets import QApplication

    from fibsem import utils

    app = QApplication(sys.argv)
    microscope, _ = utils.setup_session(manufacturer="Demo")
    widget = FibsemOverviewWidget(microscope)
    widget.setStyleSheet(stylesheets.NAPARI_STYLE)
    widget.resize(1400, 900)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    main()
