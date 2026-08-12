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
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
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
)
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointsOverlay
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas
from fibsem.ui.widgets.canvas.stage_frame import StageFrame
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    TitledPanel,
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
OVERVIEW_FOV_COLOUR = "#ce93d8"

# The grid boundary a cryo holder's slot describes, as a radius in metres. Carried over
# from the widget this replaces, where it was written inline as `1000e-6 / pixelsize`.
GRID_BOUNDARY_RADIUS = 1000e-6
# Screen-space hit radius for picking a marker, matching the FM overview's. In pixels
# rather than stage microns so how close you have to click does not change with zoom.
PICK_RADIUS_PX = 12


class OverviewRecord:
    """One overview on the canvas, and the canvas keys holding it.

    A run places one image per tile, so "the overview" is a set of keys rather than a
    single one -- which is why this exists at all. Showing, hiding and removing act on
    the run, because that is the thing a user acquired and the thing they mean.

    Structurally compatible with the fluorescence side's `PlacedOverviewImageRecord`:
    `OverviewListWidget` reads `id`, `label`, `detail` and `visible` off whatever it is
    given, and both overviews use the same list.
    """

    def __init__(self, record_id: str, label: str, keys: List[str]) -> None:
        self.id = record_id
        self.label = label
        self.keys = list(keys)
        self.visible = True
        self.pixel_size: Optional[float] = None

    @property
    def detail(self) -> str:
        """The tile count and scale, for the second half of a list row.

        Tiles rather than a grid shape: a run can be cancelled part way, and saying
        "3x3" for an overview holding four tiles would describe what was asked for
        rather than what is on the canvas.
        """
        parts = []
        if self.keys:
            parts.append(f"{len(self.keys)} tile{'s' if len(self.keys) != 1 else ''}")
        if self.pixel_size:
            parts.append(f"{self.pixel_size * constants.SI_TO_MICRO:.2f} µm/px")
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
        self._origin: Optional[FibsemStagePosition] = None
        # The beam half of the frame, kept rather than re-read -- see `_projection`.
        self._cached_projection: Optional[BeamStageProjection] = None
        self._positions: List[FibsemStagePosition] = []
        self._selected_position: Optional[str] = None
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
        # The selected position on its own layer rather than as a colour within the one
        # above: `PointsOverlay` paints every point the same. Added last, so it draws
        # over its unselected neighbours where markers crowd together.
        self.selected_position_overlay = PointsOverlay(
            color=SELECTED_POSITION_COLOUR, marker="+", size=15
        )
        self.canvas.add_overlay(self.selected_position_overlay)

        self.settings_widget = OverviewAcquisitionSettingsWidget(self)
        self.settings_widget.settings_changed.connect(self._on_settings_changed)

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
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(overviews_panel)
        controls_layout.addWidget(self.settings_widget)
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

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def beam_type(self) -> BeamType:
        """The beam the overview is acquired and projected in.

        Read from the settings rather than kept, so the projection cannot describe one
        beam while the acquisition uses the other.
        """
        try:
            return self.settings_widget.get_settings().image_settings.beam_type
        except Exception:
            return BeamType.ELECTRON

    def set_save_directory(self, path: Optional[str]) -> None:
        """Where acquired overviews are written. None means nowhere.

        The widget opens standalone against a simulator as often as it runs inside an
        experiment, and inventing a directory for those runs would scatter files through
        whatever working directory it happened to be launched from.
        """
        self._save_directory = path

    def set_interactive(self, enabled: bool) -> None:
        """Allow or forbid starting work, for a host that has taken the instrument."""
        self._interactive = bool(enabled)
        self._apply_enabled_state()

    def _apply_enabled_state(self) -> None:
        running = self._running
        self.button_acquire.setEnabled(self._interactive and not running)
        self.button_acquire.setText(
            "Running Tile Collection…" if running else "Run Tile Collection"
        )
        self.button_cancel.setVisible(running)
        self.settings_widget.setEnabled(self._interactive and not running)

    def _on_settings_changed(self) -> None:
        """The planned overview changed shape, so redraw what it would cover."""
        self._refresh_context_overlays()

    # ── the frame ────────────────────────────────────────────────────────

    def _projection(self, fresh: bool = False) -> Optional[BeamStageProjection]:
        """The beam projection, built once and kept.

        Kept because building it reads the scan rotation off the instrument, and doing
        that per use would be a hardware read on a mouse move. `fresh=True` forces a
        re-read for the one caller whose answer drives the stage, where a stale scan
        rotation would send it somewhere other than where the click was.
        """
        if fresh or self._cached_projection is None:
            projection = BeamStageProjection.from_microscope(
                self.microscope, self.beam_type
            )
            if projection is None:
                return self._cached_projection
            self._cached_projection = projection
        return self._cached_projection

    def invalidate_projection(self) -> None:
        """Force the projection to be re-read. For a host that changed the instrument."""
        self._cached_projection = None

    def _frame(self, fresh: bool = False) -> Optional[StageFrame]:
        """Stage positions and canvas coordinates, about the origin.

        None until there is an origin *and* a canvas scale: the canvas takes its scale
        from the first image placed, so nothing can be drawn in stage coordinates before
        then. That is why `_refresh_context_overlays` is safe to call at any time and
        simply does nothing early on.
        """
        projection = self._projection(fresh=fresh)
        if projection is None or self._origin is None:
            return None
        if self.canvas.reference_pixel_size is None:
            return None
        return StageFrame(self.canvas, self._origin, projection)

    def _set_origin_from(self, image: FibsemImage) -> None:
        """Anchor the canvas on the first image placed, if it is not anchored yet."""
        if self._origin is not None:
            return
        position = self._position_of(image)
        if position is None:
            return
        self._origin = deepcopy(position)

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

        self._set_origin_from(image)
        # The canvas needs a scale before a frame can exist, and the frame is what turns
        # a stage position into an offset -- so the first image sets the scale and is
        # placed at the origin by definition.
        if self.canvas.reference_pixel_size is None:
            self.canvas.set_reference_pixel_size(pixel_size)

        frame = self._frame()
        if frame is None:
            return None
        try:
            centre = frame.offset(position)
        except Exception as e:
            logger.debug(f"Could not place an image: {e}")
            return None

        key = self.canvas.add_image(
            image.filtered_data,
            centre=centre,
            pixel_size=pixel_size,
            key=key,
            zorder=zorder,
        )
        self._refresh_context_overlays()
        return key

    def set_image(self, image: FibsemImage) -> Optional[str]:
        """Place a whole overview as one image, and record it as one.

        The path a *loaded* overview takes: a stitched mosaic off disk arrives as a
        single image with a single position. An acquired one arrives tile by tile
        instead -- see `_place_tile`.
        """
        self._record_count += 1
        record_id = f"overview-{self._record_count}"
        key = self.place_image(image, key=record_id)
        if key is None:
            notification_service.show_toast(
                "That image does not record where it was acquired, so it cannot be "
                "placed on the overview.",
                "warning",
            )
            return None
        record = OverviewRecord(record_id, os.path.basename(record_id), [key])
        record.pixel_size = self._pixel_size_of(image)
        self._records[record_id] = record
        self._refresh_overview_list()
        return record_id

    def _place_tile(self, tile: FibsemImage, record_id: str) -> None:
        """Place one acquired tile, and attach it to the run that produced it."""
        record = self._records.get(record_id)
        if record is None:
            return
        key = self.place_image(tile, key=f"{record_id}-tile-{len(record.keys)}")
        if key is not None:
            record.keys.append(key)
            if record.pixel_size is None:
                record.pixel_size = self._pixel_size_of(tile)
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
        """Take one overview off the canvas entirely. False if the id is unknown."""
        record = self._records.pop(record_id, None)
        if record is None:
            return False
        for key in record.keys:
            self.canvas.remove_image(key)
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
        """
        frame = self._frame()
        if frame is None:
            self.context_overlay.set_shapes([])
            return

        specs: List[ShapeSpec] = []
        specs.extend(self._limit_shapes(frame))
        specs.extend(self._slot_shapes(frame))
        specs.extend(self._planned_overview_shape(frame))
        self.context_overlay.set_shapes(specs)
        self._refresh_position_markers()

    def _limit_shapes(self, frame: StageFrame) -> List[ShapeSpec]:
        """The travel limits, and the grid boundary a cryo holder describes."""
        limits = getattr(self.microscope._stage, "limits", None)
        if not limits or not self.microscope.stage_is_compustage:
            return []
        try:
            centre = FibsemStagePosition(name="Grid Centre", x=0, y=0, z=0, r=0, t=0)
            cx, cy = frame.to_canvas(centre)
            width = frame.length(limits["x"].max - limits["x"].min)
            height = frame.length(limits["y"].max - limits["y"].min)
        except Exception as e:
            logger.debug(f"Could not draw the stage limits: {e}")
            return []
        return [
            ShapeSpec(kind="rect", cx=cx, cy=cy, width=width, height=height,
                      color=LIMITS_COLOUR, label="Stage Limits"),
            ShapeSpec(kind="circle", cx=cx, cy=cy,
                      radius=frame.length(GRID_BOUNDARY_RADIUS),
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

    def _planned_overview_shape(self, frame: StageFrame) -> List[ShapeSpec]:
        """What the next run would cover, centred on where the stage is.

        Drawn from the settings rather than from anything acquired, so it answers "if I
        press the button now, what do I get?" before there is anything on screen.
        """
        position = self._stage_position
        if position is None:
            return []
        try:
            settings = self.settings_widget.get_settings()
            cx, cy = frame.to_canvas(position)
            width = frame.length(settings.total_fov_x)
            height = frame.length(settings.total_fov_y)
        except Exception as e:
            logger.debug(f"Could not draw the planned overview footprint: {e}")
            return []
        return [ShapeSpec(kind="rect", cx=cx, cy=cy, width=width, height=height,
                          color=OVERVIEW_FOV_COLOUR, label="Overview FoV")]

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

        unselected, selected, labels = [], [], []
        for position in self._positions:
            try:
                point = frame.to_canvas(position)
            except Exception:
                continue
            if position.name and position.name == self._selected_position:
                selected.append(point)
            else:
                unselected.append(point)
                labels.append(position.name or "")
        self.position_overlay.set_points(unselected, labels=labels)
        self.selected_position_overlay.set_points(
            selected, labels=[self._selected_position] if selected else None
        )

    # ── the host's API ───────────────────────────────────────────────────

    def set_positions(self, positions: List[FibsemStagePosition]) -> None:
        """Mark these positions on the canvas, replacing whatever was marked."""
        self._positions = list(positions)
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
        """
        frame = self._frame(fresh=True)
        if frame is None:
            return None
        try:
            target = frame.to_stage(x, y)
        except Exception as e:
            logger.debug(f"Could not resolve the clicked position: {e}")
            return None

        limits = getattr(self.microscope._stage, "limits", None)
        if limits and not target.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return None
        return target

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
        settings = self.settings_widget.get_settings()
        if not settings.image_settings.filename:
            notification_service.show_toast(
                "Please enter a filename for the overview.", "error"
            )
            return
        settings.image_settings.save = True
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
        self._worker = FunctionWorker(self._acquire_worker, settings)
        self._worker.start()

    def _acquire_worker(self, settings: OverviewAcquisitionSettings) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        from fibsem.cancellation import OperationCancelledError

        result = {"cancelled": False, "error": None}
        try:
            self._mosaic = tiled.tiled_image_acquisition_and_stitch(
                microscope=self.microscope,
                settings=settings,
                stop_event=self._stop_event,
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
        elif result.get("cancelled"):
            notification_service.show_toast("Tile collection cancelled.", "warning")
            self.label_status.setText(
                f"Cancelled after {self._tiles_acquired} tile(s)."
            )
        else:
            notification_service.show_toast("Tile collection finished.")
            self.label_status.setText(f"Acquired {self._tiles_acquired} tile(s).")
        # The stage went home at the end of the run, so the planned footprint moved.
        self._refresh_context_overlays()

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

        tile = payload.get("tile")
        if tile is not None and getattr(self, "_active_record", None):
            self._place_tile(tile, self._active_record)

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
