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
import math
import os
import threading
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem import constants
from fibsem.config import DEFAULT_STANDARD_RESOLUTION_LIST
from fibsem.fm.composite import auto_clim
from fibsem.imaging import tiled
from fibsem.imaging.reduce import downsample, downsample_mask
from fibsem.imaging.tiling import unreachable_tiles
from fibsem.imaging.tiling.progress import MODALITY_BEAM, is_modality
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
    ACCENT_COLOR,
    BORDER_COLOR,
    CURRENT_POSITION_COLOUR,
    GRAY_WHITE_COLOR,
    GRID_BOUNDARY_COLOUR,
    NEUTRAL_300,
    NEUTRAL_750,
    NEUTRAL_800,
    NEUTRAL_900,
    PANEL_COLOR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
    SEMANTIC_WARNING_COLOR,
    SLOT_COLOUR,
    STAGE_LIMITS_COLOUR,
)
from fibsem.ui.widgets.canvas.contrast_gamma_control import ContrastGammaControl
from fibsem.ui.widgets.canvas.overlay_controls import (
    CanvasOverlayControls,
    CanvasPopover,
)
from fibsem.ui.widgets.canvas.overlays import stage_context
from fibsem.ui.widgets.canvas.overlays.gridbar_overlay import GridBarOverlay
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    GRID_BOUNDARY_RADIUS_M,
    MinimapShapesOverlay,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import (
    FieldOfViewOverlay,
    PointsOverlay,
)
from fibsem.ui.widgets.canvas.overlays.tile_grid_options_panel import (
    TileGridOptionsPanel,
)
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import TileGridOverlay
from fibsem.ui.widgets.canvas.real_space_canvas import (
    WHOLE_IMAGE,
    FibsemRealSpaceCanvas,
    ImageRegion,
)
from fibsem.ui.widgets.canvas.stage_frame import StageFrame
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    IconToolButton,
    TitledPanel,
    ValueSpinBox,
)
from fibsem.ui.widgets.fibsem_overview_settings_widget import (
    FibsemOverviewSettingsWidget,
)
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    stamped_overview_name,
)
from fibsem.ui.widgets.overview_confirmation_dialog import OverviewConfirmationDialog
from fibsem.ui.widgets.overview_list_widget import OverviewListWidget
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate

logger = logging.getLogger(__name__)

# The canvas key the in-progress mosaic is drawn under. Its own, so a run that dies
# leaves no half-filled overview behind pretending to be a finished one.
# Overlay keys for `CanvasOverlayControls`. Named rather than inline so the control and
# the thing it gates cannot drift apart under a rename. The three stage-context ones are
# shared with the fluorescence tab and live with the shapes they gate; these two are this
# tab's own.
_OVERLAY_LIMITS = stage_context.OVERLAY_LIMITS
_OVERLAY_BOUNDARIES = stage_context.OVERLAY_BOUNDARIES
_OVERLAY_SLOTS = stage_context.OVERLAY_SLOTS
_OVERLAY_POSITIONS = "positions"
_OVERLAY_GRIDBARS = "gridbars"

PREVIEW_KEY = "acquisition-preview"

# Icon buttons that sit in a `TitledPanel` header. Matches the fluorescence overview,
# so the two tabs' section headers line up.
_HEADER_BTN_SIZE = 26

# Inset from the canvas's left edge for chrome drawn over it. Only x: the y comes from
# the canvas, which owns its top row -- see `chrome_below_toolbar_y`.
_CANVAS_CHROME_MARGIN = 8
# Gap between view chips.
_VIEW_CHIP_SPACING = 4

# Chips as they have always looked -- dark and rounded rather than the app's button
# styling, because they read as a set of states rather than a toolbar. The active one is
# the view the next run would land in, in the accent colour the app uses for a selected
# state.
#
# Opaque now, where these were `rgba(..., 170)`. The translucency was doing real work
# while they sat on the image; on a solid strip it means nothing, and a transparency
# that means nothing is the kind of thing a later redesign preserves for no reason. The
# values are the palette's own steps rather than the blend arithmetic -- within a couple
# of levels of it, and picked rather than mixed (`feedback_ui_widget_style`).
_VIEW_CHIP_STYLE = (
    f"QPushButton {{ color: {NEUTRAL_300}; font-size: 10px; padding: 3px 8px;"
    f" border: none; border-radius: 9px; background: {NEUTRAL_900}; }}"
    f"QPushButton:hover {{ background: {NEUTRAL_800}; }}"
    f"QPushButton:checked {{ color: {GRAY_WHITE_COLOR};"
    f" background: {NEUTRAL_750}; }}"
)
_VIEW_CHIP_STYLE_ACTIVE = (
    f"QPushButton {{ color: {NEUTRAL_300}; font-size: 10px; padding: 3px 8px;"
    f" border: 1px solid {ACCENT_COLOR}; border-radius: 9px;"
    f" background: {NEUTRAL_900}; }}"
    f"QPushButton:hover {{ background: {NEUTRAL_800}; }}"
    f"QPushButton:checked {{ color: {GRAY_WHITE_COLOR}; background: {ACCENT_COLOR}; }}"
)

# The strip the chips live on. Distinct from the canvas rather than seamless: seamless is
# only seamless while the canvas is empty and dark, and the moment an overview is on
# screen the canvas is bright, so the boundary arrives anyway -- better as a decision
# than as an accident. Scoped by object name, or the background would be inherited by
# every chip on it.
_VIEW_STRIP_STYLE = (
    f"#viewStrip {{ background: {PANEL_COLOR};"
    f" border-bottom: 1px solid {BORDER_COLOR}; }}"
)

# Cryo grid bar defaults, in microns -- the values the tab this replaces carried in
# `GRIDBAR_IMAGE_LAYER_PROPERTIES`, which went with the napari layer they configured.
DEFAULT_GRIDBAR_SPACING_UM = 100.0
DEFAULT_GRIDBAR_WIDTH_UM = 20.0

# Screen-space hit radius for picking a marker, matching the FM overview's. In pixels
# rather than stage microns so how close you have to click does not change with zoom.
PICK_RADIUS_PX = 12

# The field of view drawn around each marked position, in metres. A fixed size rather
# than the current imaging settings: the box says how much sample a lamella occupies,
# which does not change when somebody adjusts the HFW for the next overview -- and a box
# that resized itself under a settings change would look like the positions had moved.
# 100 um at the standard resolution's aspect, so it is the frame a normal image is.
POSITION_FOV_WIDTH = 100e-6
POSITION_FOV_HEIGHT = POSITION_FOV_WIDTH * (
    DEFAULT_STANDARD_RESOLUTION_LIST[1] / DEFAULT_STANDARD_RESOLUTION_LIST[0]
)


# How many pixels the contrast limits are taken from. `auto_clim`'s own cap, and for
# the same reason: the percentile over a full mosaic is the expensive part, and it is
# visually identical on a subsample.
_CLIM_SAMPLES = 250_000


def _contrast_limits(values: np.ndarray, acquired: np.ndarray) -> Tuple[float, float]:
    """Where to put black and white, taken from the acquired pixels only.

    The unacquired zeros are most of a part-finished mosaic and are not drawn, so letting
    them win the minimum squeezes what *is* there into the top of the range -- which
    renders a half-done overview visibly bleached.

    Percentiles rather than min/max, and the same 1st/99th `fibsem.fm.composite.auto_clim`
    picks: the boundary between an acquired tile and the nothing beside it is not
    perfectly sharp once a display filter has run over it, and a handful of pixels should
    not decide the whole stretch.
    """
    sample = values[acquired]
    if sample.size == 0:
        # Nothing acquired. Any limits will do -- every pixel is about to be transparent
        # -- but they have to be finite, and `auto_clim` indexes into what it is given.
        return 0.0, 1.0
    if sample.size > _CLIM_SAMPLES:
        # `auto_clim` strides a 2-D frame down before taking percentiles, for the same
        # reason; masking has already flattened this one, so it is done here.
        sample = sample[:: int(np.ceil(sample.size / _CLIM_SAMPLES))]
    return auto_clim(sample)


def _as_colour_and_coverage(
    data: np.ndarray,
    acquired: np.ndarray,
    clim: Tuple[float, float],
    curve: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Re-express a grayscale overview as colour plus where it holds anything.

    An overview is rarely complete: a masked run acquires some tiles, a cancelled one
    stops partway, and either way the rest of the mosaic is the zeros the stitch buffer
    started as. Placed opaquely those zeros paint black *over* whatever is beneath, so a
    second overview hides the first with pixels that hold nothing (FIB-630). Made
    transparent instead, the image underneath shows through.

    Same idea as the fluorescence canvas's `to_rgba` (FIB-519), and deliberately **not**
    the same alpha. There, alpha is signal strength, which works because an FM composite
    sits over black -- matplotlib draws `colour x alpha + (1 - alpha) x beneath`, and the
    second term vanishes. Over another overview it does not: measured on a mid-grey
    region over a textured one, intensity-as-alpha reads **0.772 against a true 0.500**,
    a mean error of 0.27. An overview would brighten wherever something happened to be
    behind it. Beam images are dense grayscale rather than signal over black, so "is
    there anything here" is the honest question, and it is a step function.

    Both *acquired* and *clim* are measured on the **unfiltered** array by the caller,
    and only the colour comes from the filtered one. `filtered_data` is a median then a
    gaussian, which smears the boundary between an acquired tile and the nothing beside
    it in both directions: signal leaks a couple of pixels past it, so testing the
    filtered array for "greater than zero" gives a mask a filter radius too generous;
    and the last rows *inside* the tile are pulled part-way down toward the nothing, far
    enough to take the low limit with them and bleach the whole mosaic. Measured: 172/255
    where 128 was right. A display filter does not get a vote on what was acquired, nor
    on where black is.

    The cost is that a pixel of *exactly* zero inside acquired data reads as a hole.
    Measured at 0.39% of an autocontrast-stretched frame -- but the caller reduces before
    this runs, and `downsample`'s box mean averages an isolated zero away with its
    neighbours: 0.39% -> 0.000% at any reduction factor. Only whole unacquired blocks
    stay exactly zero, which is the thing being asked about. An image small enough to be
    placed unreduced keeps that speckle, and it shows only where another overview lies
    beneath it.

    *curve* is the user's contrast and gamma, applied to the stretched values on the way
    through. Folded in here rather than run over the result because this is called on the
    *patch being drawn* -- a screen's worth -- where a separate pass over the finished
    RGBA would be a second normalise and a second copy of the whole mosaic. **Alpha is
    never curved.** Contrast says how bright what was acquired should look; it must not
    decide what *was* acquired, and passing alpha through the same curve would fade a
    region out as the maximum came down and paint unacquired ground in as it went up.
    """
    values = np.asarray(data, dtype=np.float32)
    out = np.zeros((*values.shape[:2], 4), dtype=np.uint8)
    if not acquired.any():
        return out  # nothing acquired: wholly transparent, which is the truth

    lo, hi = clim
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    if curve is not None:
        norm = np.clip(curve(norm), 0.0, 1.0)
    out[..., :3] = (norm * 255.0).astype(np.uint8)[..., None]
    out[..., 3] = np.where(acquired, 255, 0)
    return out


class _PlacedTile(NamedTuple):
    """One image a record placed: what it holds, where it was taken, what it covers.

    The *ingredients* of a picture rather than a picture. Colouring and the user's
    contrast curve both happen at draw time, on the part being drawn, and that is the
    whole point of the split: a finished RGBA can only be shown at the resolution it was
    finished at, so a zoom has nothing to reveal and a contrast change costs a pass over
    the entire mosaic instead of over a screenful.

    `clim` is measured once over the whole image and then kept. Deliberately not
    re-measured per patch: limits taken from whatever is currently on screen would make
    the picture change brightness as you pan across it, which is a worse fault than the
    one the split is fixing.

    `covers` is the ground in metres, measured from the image's *original* shape before
    reduction. It has to be carried rather than re-derived, because `grey` is decimated:
    the canvas sizes an image from `shape x pixel_size` unless told otherwise, so a
    1024 px tile stored at 512 px was drawn covering half the ground it actually images
    -- a mosaic with black gaps between every tile, at the right positions and the wrong
    size.
    """

    grey: object  # np.ndarray, the filtered image reduced to the store cap
    acquired: object  # np.ndarray of bool, same shape -- where anything was acquired
    clim: Tuple[float, float]
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
        """The beam, then the orientation the stage is in: "FIB @ Milling".

        Both facts, always, in the same shape -- so nothing is implied by leaving one
        out. An earlier version dropped the orientation when it matched the beam
        ("SEM" for electron at the SEM pose) which read well until you met the ones it
        kept, and then "FIB · SEM pose" had to be decoded rather than read.

        Beam first because that is what you are looking *through*, and it is how people
        say it: "the FIB overview". "@" rather than a separator because the second half
        is a place, and reading it as "at" is exactly right.

        The orientations are named after the beams -- "SEM" is the pose where the sample
        faces the electron column -- so "SEM @ SEM" says the electron beam looking at
        the pose meant for it, and repeats itself for a reason. `describe` spells the
        whole thing out for a tooltip, where there is room.
        """
        return f"{self.beam_label} @ {self.orientation_label}"

    @property
    def beam_label(self) -> str:
        """The beam, as the instrument is spoken about: SEM and FIB, not electron/ion."""
        return "SEM" if self.beam_type is BeamType.ELECTRON else "FIB"

    @property
    def orientation_label(self) -> str:
        """The orientation, exactly as the microscope names it.

        Uppercase throughout rather than title case, because these are the instrument's
        own names -- SEM, FIB, MILLING -- and two of them are acronyms. Title-casing
        rendered those as "Sem" and "Fib", which reads as a mistake; casing them by
        rule instead means a rule to remember, and a new orientation name would have to
        be added to it. Showing them verbatim needs neither.
        """
        return self.orientation.upper()

    @property
    def describe(self) -> str:
        """The same thing as a sentence, for a tooltip."""
        beam = "Electron" if self.beam_type is BeamType.ELECTRON else "Ion"
        return f"{beam} beam, stage at the {self.orientation_label} orientation."


class OverviewRecord:
    """One overview on the canvas, and the canvas keys holding it.

    A run places one image per tile, so "the overview" is a set of keys rather than a
    single one -- which is why this exists at all. Showing, hiding and removing act on
    the run, because that is the thing a user acquired and the thing they mean.

    Structurally compatible with the fluorescence side's `PlacedOverviewImageRecord`:
    `OverviewListWidget` reads `id`, `label`, `detail` and `visible` off whatever it is
    given, and both overviews use the same list.
    """

    def __init__(
        self,
        record_id: str,
        label: str,
        keys: List[str],
        view: Optional["OverviewView"] = None,
    ) -> None:
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
    # Whether a run is in progress here. A *state*, re-emitted on every change
    # rather than an edge, so a host that connects late or recomputes from
    # several facts cannot end up holding a stale answer. The host that matters
    # is the window, which must stop the other overview driving the stage while
    # this one is mid-tileset (FIB-706).
    acquiring_changed = pyqtSignal(bool)

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
        # Canvas key -> the *base* tile behind it, the auto-stretched one. Contrast is
        # applied on the way to the canvas and never written back here, so moving a
        # slider twice adjusts the original twice rather than compounding.
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
        # The view the next run would have landed in, last time anything looked. Kept
        # so a *change* can be told from a refresh -- see `_follow_the_acquisition_view`.
        self._planned_view: Optional["OverviewView"] = None
        # Where the next run is planned around, if the grid has been dragged off the
        # stage. None means "wherever the stage is", which is also what the runner
        # falls back to -- so the drawn grid and the acquisition agree by default.
        self._target: Optional[FibsemStagePosition] = None
        # What the run under way is centred on, and None between runs. Its own field
        # rather than a read of `_target`: a run started without one is centred on
        # wherever the stage was when it began, which the stage stops being one tile in.
        self._run_centre: Optional[FibsemStagePosition] = None
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
        # Set by `_on_move_errored` so `_on_move_finished`, which always runs, knows
        # not to clear a message the failure has just put up.
        self._move_failed: bool = False
        self._lock_reason = "a workflow is running"
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
        # Two bounds, answering two different questions:
        #
        #   `_store_budget_bytes` -- how much memory one overview may occupy, and so the
        #                            ceiling on any detail a zoom could ever recover.
        #   `display_max_px`      -- the most the canvas ever hands matplotlib, paid on
        #                            every frame at every zoom. A redraw bound.
        #
        # Held no finer than drawn, "don't downscale the actual image" has no answer at
        # all: the detail would be gone before the canvas saw it. Held finer, the canvas
        # asks for the part on screen at the resolution the screen can use, and a zoom
        # reveals rather than magnifies.
        #
        # The store bound is bytes rather than pixels because bytes are the thing being
        # spent. A pixel cap prices the same setting at 52 MB for a mosaic of 1024 px
        # tiles and 118 MB for one of 3072 px tiles, and silently costs 50% more again on
        # a uint16 detector. It is also a *ceiling*, not an allocation: `downsample`
        # reduces by whole factors, so an overview that fits is held whole and one that
        # does not drops to the next factor. At 128 MB every 1024 px-tile mosaic up to
        # 5x5 is held at full acquired resolution -- which is what the complaint behind
        # FIB-658 was asking for -- and a 10x10 lands at 5120 px for 52 MB.
        #
        # 2048 is the drawing half. At 512 a 5x5 of 1024 px tiles was reduced tenfold and
        # then magnified back onto a ~1100 px wide canvas, so every stored pixel was
        # drawn as a 2x2 block. 2048 is the first power of two past the canvas's own
        # width, which is what stops the magnification; the detail pass keeps well under
        # it in practice, asking only for what the axes can show.
        self._store_budget_bytes = 128_000_000
        self.canvas = FibsemRealSpaceCanvas(display_max_px=2048)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)

        # Contrast and gamma, owned here rather than by the canvas. The canvas has its
        # own `btn_contrast`, and it stays hidden on a real-space canvas because the
        # machinery behind it acts on `imgs[0]` -- meaningful when a canvas holds one
        # image, arbitrary when it holds an overview per key. The fluorescence tab hides
        # it for the sibling reason and drives its own layers popover; this is that,
        # with one grayscale layer instead of several channels.
        #
        # Canvas-wide, not per image: an overview is a mosaic, and per-image contrast
        # would emphasise the seams rather than hide them. Selecting one image to adjust
        # is the later half of FIB-415, and needs a selection the canvas has not got.
        self.btn_contrast = self.canvas.add_toolbar_button(
            "mdi:contrast-circle",
            "Contrast and gamma",
            self._toggle_contrast,
            checkable=True,
        )
        self.contrast_control = ContrastGammaControl(self.canvas)
        self.contrast_control.changed.connect(self._reapply_contrast)

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
        self.canvas.add_overlay(self.tile_grid_overlay)

        # Where the sample and the stage can physically go -- the context an overview is
        # read against. Added before the position markers so it sits beneath them.
        self.context_overlay = MinimapShapesOverlay(zorder=4.0, crosshair_half_px=24)
        self.canvas.add_overlay(self.context_overlay)

        # Where the stage is now. Distinct from the red origin marker the canvas draws:
        # the origin explains why everything sits where it does, this is what you steer
        # by. They coincide until the stage moves, then diverge.
        self.current_position_overlay = PointsOverlay(
            color=CURRENT_POSITION_COLOUR, marker="+", size=15, edge_width=2.8
        )
        self.canvas.add_overlay(self.current_position_overlay)

        # Crosshairs rather than dots: a marked position is a point on the sample, and a
        # filled dot covers the feature it is naming. Boxed with the field of view an
        # image taken there would cover, so the marker carries a sense of scale -- on a
        # canvas spanning millimetres a bare crosshair gives none, and "would these two
        # lamellae land in one frame" is a question you cannot answer by eye without it.
        #
        # The current stage position above is deliberately left unboxed: it is where you
        # are rather than something you are sizing up, and boxing it would double every
        # lamella's box the moment you drove to one.
        self.position_overlay = FieldOfViewOverlay(
            color=SAVED_POSITION_COLOUR,
            marker="+",
            size=11,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.position_overlay)
        # Flagged positions, on their own layer for the same reason the selection has
        # one: `PointsOverlay` paints every point the same colour. Worth the third
        # layer here rather than collapsing it into the others -- a lamella marked
        # defective is one you should not be re-targeting, and the tab this replaces
        # said so in colour.
        self.flagged_position_overlay = FieldOfViewOverlay(
            color=SEMANTIC_WARNING_COLOR,
            marker="+",
            size=11,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.flagged_position_overlay)
        # The selected position on its own layer rather than as a colour within the one
        # above: `PointsOverlay` paints every point the same. Added last, so it draws
        # over its unselected neighbours where markers crowd together.
        self.selected_position_overlay = FieldOfViewOverlay(
            color=SELECTED_POSITION_COLOUR,
            marker="+",
            size=15,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.selected_position_overlay)

        self.canvas.cursor_moved.connect(self._on_cursor_moved)

        self.settings_widget = FibsemOverviewSettingsWidget(self)
        self.settings_widget.settings_changed.connect(self._on_settings_changed)

        # Which way of looking at the sample the canvas is showing. Attached to the
        # canvas rather than filed in the settings column, because it selects what you
        # are looking at -- but on a strip *above* the canvas rather than painted on it
        # (FIB-649).
        #
        # The set is beams x orientations and grows through a session as views are
        # acquired in. Measured with this stylesheet: eight chips end at x=707, and the
        # splitter lets the canvas be narrower than that at any window size, so on the
        # canvas they run off the edge and under the toolbar buttons. There is no offset
        # that rescues a growing set inside a fixed region -- which is also how they came
        # to sit inside the status zone's rectangle (FIB-651). Above the canvas they grow
        # into space nothing else wants, and occlude no data.
        self._view_chip_buttons: Dict["OverviewView", QPushButton] = {}
        chips = QWidget()
        self._view_strip_layout = QHBoxLayout(chips)
        self._view_strip_layout.setContentsMargins(
            _CANVAS_CHROME_MARGIN, 4, _CANVAS_CHROME_MARGIN, 4
        )
        self._view_strip_layout.setSpacing(_VIEW_CHIP_SPACING)
        # Packs the chips left. Added once and kept: the rebuild inserts before it.
        self._view_strip_layout.addStretch(1)

        # In a scroll area, because a row of controls in a plain layout sets the
        # *window's* minimum width: eight chips measure 709 px, so a tab that had been
        # used in every view would refuse to be made narrower than that -- a floor that
        # appears mid-session as views accumulate. A scroll area has a small minimum of
        # its own, so the window goes on shrinking (measured: 354 px, the settings
        # column, with or without chips).
        #
        # No scrollbar, in either direction. One would halve a row this short, and the
        # case it serves is a canvas pane narrower than ~710 px, which is a window
        # narrower than any usable one. The wheel still scrolls the row there, so the
        # far chips are reachable rather than lost.
        self.view_strip = QScrollArea()
        self.view_strip.setObjectName("viewStrip")
        self.view_strip.setStyleSheet(_VIEW_STRIP_STYLE)
        self.view_strip.setWidget(chips)
        self.view_strip.setWidgetResizable(True)
        self.view_strip.setFrameShape(QFrame.NoFrame)
        self.view_strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view_strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view_strip.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Nothing to choose between until a view exists, and an empty bar over the canvas
        # is chrome that has not earned its place.
        self.view_strip.hide()
        self.label_view_note = QLabel("")
        self.label_view_note.setWordWrap(True)
        self.label_view_note.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)

        # One surface for everything drawn *over* the data, rather than a checkbox per
        # overlay appearing wherever its feature happened to land. The gridbar toggle
        # was the first and only one; it moves in here rather than sitting beside it.
        #
        # Ordered as they sit on the canvas, outermost first: where the stage can go,
        # then the holder's grids, then the marks inside them. Grid bars last because
        # they are a lattice over everything and the two controls under them are theirs.
        self.overlay_controls = CanvasOverlayControls(
            [
                *stage_context.CONTEXT_OVERLAY_ENTRIES,
                (_OVERLAY_POSITIONS, "Saved positions", True),
                (_OVERLAY_GRIDBARS, "Grid bars", False),
            ]
        )
        self.overlay_controls.toggled.connect(self._on_overlay_toggled)
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
            # Matched to the checkbox at construction, not only when it is toggled. The
            # handler had never run by this point, so the pitch controls started live
            # over a lattice that was not drawn -- inviting an adjustment that appeared
            # to do nothing.
            _spin.setEnabled(self.overlay_controls.is_visible(_OVERLAY_GRIDBARS))

        # On the canvas toolbar beside contrast, not in the settings column: what is
        # drawn *over* the picture is a looking-at-it question, where the column is for
        # setting up the next run. In the column it also sat at the bottom of a scroll,
        # which is the least reachable place on the tab. Built here rather than with the
        # other toolbar buttons because it holds the controls above, which have to exist
        # first.
        self.btn_overlays = self.canvas.add_toolbar_button(
            "mdi:eye-outline",
            "Overlays",
            self._toggle_overlays,
            checkable=True,
        )
        self.overlay_popover = CanvasPopover(self._overlay_panel(), parent=self.canvas)

        # The planned tileset gets its own button rather than a switch among the others,
        # matching the fluorescence tab: it is the one overlay you *edit* -- drag it,
        # resize it, click tiles out of it -- so it carries colour, fill and a re-centre
        # beside its visibility, which is more than a checkbox row holds. Same panel
        # class as the FM tab now that it lives beside the overlay it configures, so the
        # two tabs cannot drift apart on the one overlay they both draw.
        self.btn_tile_grid = self.canvas.add_toolbar_button(
            "mdi:grid",
            "Tile grid",
            self._toggle_tile_grid_panel,
            checkable=True,
        )
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

        # "Acquire Overview" says what the button produces; "Run Tile Collection" said
        # how it is produced, which is the part the settings above already describe.
        self.button_acquire = QPushButton("Acquire Overview")
        self.button_acquire.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_acquire.setMinimumHeight(30)
        self.button_acquire.clicked.connect(self.acquire)
        # Beside Acquire and always present, disabled rather than hidden -- the FM tab's
        # arrangement. Stop belongs next to go, where it is looked for, and a button
        # that appears only once a run has started moves everything under it at the
        # moment the user is least able to absorb a moving layout.
        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.setMinimumHeight(30)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setEnabled(False)

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
        # In the panel header rather than under the list: loading acts on the section
        # as a whole, not on any row in it, and a full-width button below the rows read
        # as a fourth row. Same icon and size as the fluorescence tab's, because it is
        # the same action on the same kind of section.
        self.button_load = IconToolButton(
            icon="mdi:image-plus-outline",
            tooltip="Load a saved overview",
            size=_HEADER_BTN_SIZE,
        )
        self.button_load.clicked.connect(self._prompt_for_overview)
        overviews_panel.add_header_widget(self.button_load)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self._controls_layout = controls_layout
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(overviews_panel)
        controls_layout.addWidget(self.settings_widget)
        # Kept as a handle: with the overlay switches moved onto the canvas toolbar
        # this section holds only the view note, which is empty whenever there is
        # nothing to say -- and an empty titled panel is chrome that has not earned its
        # place, the same argument the view strip is hidden on.
        self.display_section = self._section("Display", self._display_panel())
        self.display_section.setVisible(False)
        controls_layout.addWidget(self.display_section)
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

        # The actions sit *below* the scroll area, not inside it, so Acquire, Cancel and
        # the progress of a running acquisition are on screen whatever the column is
        # scrolled to. Inside, a host adding its own section (the lamella list) pushes
        # them past the bottom on any window short enough, and a run then reports its
        # progress somewhere nobody is looking. Not in a titled panel either: a
        # collapsible header over two buttons is a control that can fold Acquire and
        # Cancel out of sight.
        column = QWidget()
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        column_layout.addWidget(scroll, stretch=1)
        column_layout.addWidget(self._acquisition_panel())

        # The view strip rides with the canvas rather than spanning the window, so it
        # keeps saying "this selects what *the canvas* is showing" -- and stays the
        # canvas's width when the splitter is dragged.
        canvas_pane = QWidget()
        pane_layout = QVBoxLayout(canvas_pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(0)
        pane_layout.addWidget(self.view_strip)
        pane_layout.addWidget(self.canvas, stretch=1)

        # A splitter rather than a fixed column, so a user can give the canvas the whole
        # window on a small screen. The canvas takes the extra room as the window grows.
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(canvas_pane)
        splitter.addWidget(column)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _acquisition_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        # Left and right match the scrolled column's own 8px, so the buttons line up
        # with the panels above them rather than sitting 4px further out.
        layout.setContentsMargins(8, 4, 8, 8)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)
        buttons_layout.addWidget(self.button_cancel)
        buttons_layout.addWidget(self.button_acquire, stretch=1)

        # Readouts first, the action last. The fluorescence tab stacks its column the
        # same way, and the two saying the same thing in opposite places is the drift
        # this pair keeps producing -- they had already diverged over *which* surface
        # reports a stage move (FIB-765), and this is the same argument about where.
        #
        # Acquire being the last thing in the column is also the ordinary convention for
        # a commit action, and it means the button does not move as the readouts above
        # it appear and disappear through a run.
        layout.addWidget(self.progress)
        layout.addWidget(self.label_status)
        layout.addWidget(buttons)
        return panel

    def _overlay_panel(self) -> QWidget:
        """What the overlays button opens: the switches, then the bars\' own pitch.

        The pitch controls follow the switch that draws them rather than staying in the
        column. They mean nothing while the lattice is off -- which is why they are
        disabled with it -- so several panels away from their checkbox is the one place
        they should not be.
        """
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addRow(self.overlay_controls)
        layout.addRow("Bar spacing", self.spin_gridbar_spacing)
        layout.addRow("Bar width", self.spin_gridbar_width)
        return panel

    def _toggle_tile_grid_panel(self) -> None:
        """Show or hide the tile grid panel, in the canvas's top-right corner.

        A `Qt.Tool` window rather than a child of the canvas, which is how it was built
        for the fluorescence tab, so it is placed in global coordinates rather than by
        `CanvasPopover`. Re-fitted on every open because its summary is word-wrapped and
        grows -- dragging the grid adds a line saying how far it now sits from the stage,
        and a panel sized once at open time clipped it (FIB-510).

        Notably *not* accompanied by the FM tab's gesture hint. That writes to the
        canvas's status zone, which on this tab already carries the "no overview in this
        view" caption (FIB-659); two writers to one zone is what the zone was built to
        prevent, and picking between them is its own small piece of work.
        """
        if not self.btn_tile_grid.isChecked():
            self.tile_grid_panel.hide()
            return
        self.tile_grid_panel.adjustSize()
        corner = self.canvas.mapToGlobal(self.canvas.rect().topRight())
        self.tile_grid_panel.move(
            corner.x() - self.tile_grid_panel.width() - 8,
            corner.y() + self.btn_tile_grid.height() + 8,
        )
        self.tile_grid_panel.show()
        self.tile_grid_panel.raise_()

    def _toggle_overlays(self) -> None:
        """Show or hide the overlays popover, anchored under its button."""
        self.overlay_popover.set_open(self.btn_overlays.isChecked(), self.btn_overlays)

    def _display_panel(self) -> QWidget:
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addRow(self.label_view_note)
        return panel

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── grid bars ────────────────────────────────────────────────────────

    # ── the view selector ────────────────────────────────────────────────

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

        # Rebuilt rather than updated: this runs on every overlay refresh, and a handful
        # of buttons is cheaper to make than to diff.
        for chip in self._view_chip_buttons.values():
            self._view_strip_layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._view_chip_buttons = {}

        # Shown even when there is only one, which was not the first answer: a lone chip
        # says nothing the info bar does not, and chrome has to earn its place. But a
        # control nobody can see does not exist, and the first time there *are* two views
        # is the worst moment to discover that a way of switching between them has been
        # there all along. One chip also states plainly which view the tab opened in,
        # above the canvas rather than in the corner among the coordinates.
        acquisition = self.acquisition_view
        for view in views:
            chip = QPushButton(view.label)
            chip.setCheckable(True)
            chip.setChecked(view == self._current_view)
            chip.setCursor(Qt.PointingHandCursor)
            # The one the next run would land in is marked, not just the one being
            # shown: that is the difference between "what am I looking at" and "where
            # will the next overview appear", and on this tab they come apart.
            chip.setToolTip(
                f"{view.describe}\n"
                + (
                    "Where the next acquisition would appear."
                    if view == acquisition
                    else "Nothing acquired now would appear here."
                )
            )
            chip.setStyleSheet(
                _VIEW_CHIP_STYLE_ACTIVE if view == acquisition else _VIEW_CHIP_STYLE
            )
            chip.clicked.connect(partial(self._on_view_chip_clicked, view))
            # Before the trailing stretch, so they pack left in the order above.
            self._view_strip_layout.insertWidget(len(self._view_chip_buttons), chip)
            # Shown here rather than left to the layout: a widget inserted into a live
            # layout is shown when that layout next activates, which is after this
            # returns -- so the row measured below would size itself to a row of
            # zero-height chips, and the strip would come out 8 px of margin.
            chip.show()
            self._view_chip_buttons[view] = chip

        # Sized to the chips it is holding, here rather than at construction: a scroll
        # area's own height hint is sized for a page of content (93 px), and asking the
        # row before any chip is in it measures the margins (8 px) and squashes them
        # flat. The active chip carries a border, so the row is a pixel or two taller
        # depending on which views exist -- which is why this is re-read per rebuild.
        row = self.view_strip.widget()
        row.adjustSize()
        self.view_strip.setFixedHeight(row.sizeHint().height())

        self.view_strip.setVisible(bool(self._view_chip_buttons))
        self._refresh_view_note()

    def _on_view_chip_clicked(self, view: "OverviewView") -> None:
        self.show_view(view)

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
            self._show_display_section(False)
            return
        self.label_view_note.setText(
            f"Showing {self._current_view.label}; the stage is at "
            f"{acquisition.label}, so the next acquisition would not appear here."
        )
        self.label_view_note.setVisible(True)
        self._show_display_section(True)

    def _show_display_section(self, visible: bool) -> None:
        """The section follows the only thing left in it.

        `getattr` because the view selector refreshes during construction, before the
        section exists -- the panel is built from the controls, so it cannot be built
        before them.
        """
        section = getattr(self, "display_section", None)
        if section is not None:
            section.setVisible(visible)

    def _on_overlay_toggled(self, key: str, checked: bool) -> None:
        """One overlay turned on or off.

        Everything except the grid bars is rebuilt by `_refresh_context_overlays`, which
        reads the controls rather than being told what changed -- so a toggle and a stage
        move take the same path and cannot disagree. The bars are their own overlay with
        its own artists, so they are set directly.
        """
        if key == _OVERLAY_GRIDBARS:
            # The pitch controls only mean anything while the bars are drawn.
            self.spin_gridbar_spacing.setEnabled(checked)
            self.spin_gridbar_width.setEnabled(checked)
            self.gridbar_overlay.set_visible(checked)
            if checked:
                self._refresh_gridbars()
            return
        self._refresh_context_overlays()

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
        if not self.overlay_controls.is_visible(_OVERLAY_GRIDBARS):
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
        """Whether an overview acquisition is running here.

        `_running` as well as the worker, and the order is why: `acquire` calls
        `_set_running(True)` *before* it builds the worker, so for the width of that
        gap a worker-only answer says no while a run is starting. That gap is exactly
        when `acquiring_changed` is emitted, so a host locking the other overview off
        this property got False and locked nothing (FIB-706).

        Keeping the worker check as well as the flag, rather than replacing it: the
        union is true over a superset of the interval either is, and every caller is a
        guard, so being early and late is the safe direction to be wrong in.
        """
        return self._running or (self._worker is not None and self._worker.is_alive())

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

    def add_settings_section(
        self, title: str, widget: QWidget, first: bool = True
    ) -> None:
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
                self.settings_widget.set_save_directory(path)
            except Exception as e:
                logger.debug(f"Could not show the save directory: {e}")

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        """Allow or forbid starting work, for a host that has taken the instrument.

        *reason* completes the sentence "Cannot move the stage while ___" when a move is
        refused. The widget cannot know why it was locked -- a workflow owning the
        instrument and the other overview being mid-tileset are the same `False` here --
        and a refusal naming the wrong one is barely better than one naming nothing
        (FIB-706).
        """
        self._interactive = bool(enabled)
        self._lock_reason = reason or "a workflow is running"
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
            "Acquiring Overview…" if running else "Acquire Overview"
        )
        # Not gated on `_interactive`: a host locking the tab must not take away the
        # only way to stop a run that is already under way. Off once cancellation has
        # been asked for, so a second press cannot read as "it did not work".
        self.button_cancel.setEnabled(running and not self._stop_event.is_set())
        self.settings_widget.setEnabled(self._interactive and not running)

    def _planned_tile_count(self) -> int:
        """How many tiles the next run would acquire. 0 if the settings cannot be read."""
        settings = self._settings()
        return 0 if settings is None else settings.n_enabled_tiles

    def _on_settings_changed(self) -> None:
        """The planned overview changed shape, so redraw what it would cover."""
        self._follow_the_acquisition_view()
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

    def _follow_the_acquisition_view(self) -> None:
        """Show the view the next run would land in, whenever that changes.

        A view is `(beam, orientation)`, and both halves are things a user changes
        deliberately: picking a beam in the settings, or re-posing the stage. Either
        way what they have said is "the next run happens *there*", and the canvas has
        to show there -- otherwise the planned grid vanishes at exactly the moment they
        are planning, since it refuses to draw on a view the run will not land in.

        This was once beam-only, on the theory that moving the stage is not a statement
        about what to look at. That distinction does not survive contact: a stage move
        *within* an orientation does not change the view at all, so the only move this
        reacts to is a change of orientation -- which is as deliberate as choosing a
        beam, and never happens by accident.

        On change, not on every refresh, which is what leaves room for the view
        selector: an explicit choice stands until the next run would land somewhere
        else.
        """
        view = self.acquisition_view
        if view is None or view == self._planned_view:
            return
        self._planned_view = view
        if view != self._current_view:
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

        # Only the very first time, to give the canvas something to draw in. Steering
        # after that belongs to `_follow_the_acquisition_view`, which acts on *changes*
        # -- this runs on every refresh, including from inside `show_view`, so choosing
        # here would undo a switch half way through making it.
        if self._current_view is None:
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

    def _set_origin_from(self, image: FibsemImage, view: "OverviewView") -> bool:
        """Anchor a view on the first image placed in it, if it is not anchored yet.

        A *provisional* anchor -- one `_seed_frame` invented so the tab could draw
        before anything was acquired -- is replaced rather than kept. Nothing is placed
        in the view at that point, so re-anchoring moves nothing; it just stops the
        canvas being centred on wherever the stage happened to be when the tab opened,
        which can be millimetres from the data.

        Returns whether it re-anchored, because that moves every overlay drawn in the
        view: the origin is what a stage position is measured *from*.
        """
        if view in self._origins and view not in self._provisional:
            return False
        position = self._position_of(image)
        if position is None:
            return False
        self._origins[view] = deepcopy(position)
        self._provisional.discard(view)
        return True

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

    def place_image(
        self,
        image: FibsemImage,
        key: Optional[str] = None,
        zorder: Optional[float] = None,
    ) -> Optional[str]:
        """Put one image on the canvas where it was acquired. See :meth:`_place`."""
        return self._place(image, key=key, zorder=zorder)[0]

    def _place(
        self,
        image: FibsemImage,
        key: Optional[str] = None,
        zorder: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional["_PlacedTile"]]:
        """Put one image on the canvas, and hand back the tile that was stored for it.

        The tile comes back because a caller that keeps a *record* needs the same one:
        reducing the image again to fill the record built a second, equal copy of the
        largest arrays this widget holds, and kept both -- the canvas binds its tile into
        the `detail` closure, so nothing dropped the first. At the 128 MB store budget
        that is up to a quarter of a gigabyte per overview instead of an eighth, and the
        reduction itself (measured at 218 MB of temporary and 38 ms for a 10x10 of
        1024 px tiles, see `_stored_tile`) was paid twice.

        Switches the canvas to the image's own view first, if it is not already there:
        you placed the image in order to look at it, and it would otherwise be recorded
        into a view nothing is showing.

        Returns `(key, tile)`, or `(None, None)` if the image cannot be placed -- which
        is the case for anything acquired before the stage position and pixel size were
        recorded. Refused rather than placed at the origin: an image in the wrong place
        looks exactly like an image in the right place.
        """
        position = self._position_of(image)
        pixel_size = self._pixel_size_of(image)
        if position is None or not pixel_size:
            logger.debug("Cannot place an image with no stage position or pixel size.")
            return None, None

        view = self._view_of(image)
        if view is None:
            return None, None
        if self._current_view is None:
            self._current_view = view
            self._refresh_view_selector()
        elif view != self._current_view:
            self.show_view(view)

        reframed = self._set_origin_from(image, view)
        # The canvas needs a scale before a frame can exist, and the frame is what turns
        # a stage position into an offset. Usually seeded from the settings before any
        # image arrives (`_seed_frame`); this is the fallback for a widget that has been
        # handed an image without ever drawing. Which pixel size wins does not matter
        # beyond the units: `add_image` scales each image by its own against this one.
        if self.canvas.reference_pixel_size is None:
            reframed |= self.canvas.set_reference_pixel_size(pixel_size)

        tile = self._stored_tile(image)
        placed = self._place_on_canvas(tile, view, key=key, zorder=zorder)
        # Only when the *frame* moved, which is the origin or the scale and nothing else.
        # An image is drawn in the frame; it does not decide it, so a placement that
        # leaves both alone changes nothing any overlay is derived from -- and every
        # placement after the first is one of those, including every frame of a run's
        # live preview. Refreshing on each of them re-anchored the planned tileset at
        # wherever the stage had reached, so the plan walked the grid alongside the
        # acquisition (FIB-647).
        if reframed:
            self._refresh_context_overlays()
        return placed, tile

    # ── contrast and gamma ────────────────────────────────────────────────

    def _toggle_contrast(self) -> None:
        """Show or hide the floating contrast popover, anchored under its button."""
        self.contrast_control.set_open(self.btn_contrast.isChecked(), self.btn_contrast)

    def _patch(
        self, tile: "_PlacedTile", region: ImageRegion, max_px: int
    ) -> Tuple[np.ndarray, ImageRegion]:
        """The part of *tile* that *region* names, coloured, at most *max_px* across.

        What the canvas asks for as the view moves, and the whole of what phase 2 bought:
        the reduction factor is set by how much of the image is on screen rather than by
        how large the image is, so zooming in reveals detail instead of magnifying stored
        pixels.

        Snapped outward to whole stored pixels, and the region *actually* covered is
        returned rather than the one asked for -- the canvas draws the patch at the
        rectangle this names, and a fractional-pixel disagreement between the two would
        show as a seam every time the view moved.

        The mask is reduced by its own rule rather than alongside the pixels, because
        "was anything acquired in this block" is a different question from "how bright is
        this block" and box-averaging answers only the second.
        """
        grey, acquired = tile.grey, tile.acquired
        height, width = grey.shape[0], grey.shape[1]
        x0 = min(max(0, int(np.floor(region.left * width))), width - 1)
        y0 = min(max(0, int(np.floor(region.top * height))), height - 1)
        x1 = min(width, max(int(np.ceil(region.right * width)), x0 + 1))
        y1 = min(height, max(int(np.ceil(region.bottom * height)), y0 + 1))
        curve = (
            None if self.contrast_control.is_default() else self.contrast_control.apply
        )
        drawn = _as_colour_and_coverage(
            downsample(grey[y0:y1, x0:x1], max_px),
            downsample_mask(acquired[y0:y1, x0:x1], max_px),
            tile.clim,
            curve,
        )
        return drawn, ImageRegion(x0 / width, x1 / width, y0 / height, y1 / height)

    def _for_display(self, tile: "_PlacedTile") -> np.ndarray:
        """The whole of *tile* at display resolution.

        The fallback the canvas draws until the first patch arrives, and whatever it
        falls back to if the source ever declines.
        """
        return self._patch(tile, WHOLE_IMAGE, self.canvas.display_max_px)[0]

    def _reapply_contrast(self) -> None:
        """Redraw every placed image through the current contrast.

        Through the sources rather than by re-placing: re-adding under the same key
        destroys and recreates the artist, which moves it to the top of the draw order --
        so adjusting contrast would silently reorder overlapping overviews.

        Forced, because nothing about the *view* changed and none of the canvas's usual
        tests would notice. The curve now runs over the patch being drawn rather than
        over the whole mosaic, which is what takes a contrast step off the size of the
        overview and onto the size of the window.
        """
        self.canvas.refresh_detail(force=True)

    def _place_on_canvas(
        self,
        tile: "_PlacedTile",
        view: "OverviewView",
        key: Optional[str] = None,
        zorder: Optional[float] = None,
    ) -> Optional[str]:
        """Draw a stored tile in *view*. Shared by first placement and re-placement.

        Draws, and nothing else. The overlays are the caller's business: `place_image`
        refreshes them when the placement moved the frame, and `show_view` once at the
        end rather than once per re-placed image.
        """
        frame = self._frame(view)
        if frame is None:
            return None
        try:
            centre = frame.offset(tile.position)
        except Exception as e:
            logger.debug(f"Could not place an image: {e}")
            return None
        return self.canvas.add_image(
            self._for_display(tile),
            centre=centre,
            pixel_size=tile.pixel_size,
            key=key,
            zorder=zorder,
            covers=tile.covers,
            # Bound to this tile, and the canvas holds it for as long as the image is
            # placed -- so a contrast change reaches every placed image without walking
            # the records, which would miss the acquisition preview, the one thing on
            # the canvas that has no record. Removing the image drops the source and the
            # tile with it, which a dictionary kept alongside had to be told to do.
            detail=lambda region, max_px, tile=tile: self._patch(tile, region, max_px),
        )

    def set_image(self, image: FibsemImage) -> Optional[str]:
        """Place a whole overview as one image, and record it as one.

        The path a *loaded* overview takes: a stitched mosaic off disk arrives as a
        single image with a single position. An acquired one arrives tile by tile
        instead -- see `_place_tile`.
        """
        self._record_count += 1
        record_id = f"overview-{self._record_count}"
        view = self._view_of(image)
        key, tile = self._place(image, key=record_id)
        if key is None or tile is None:
            notification_service.show_toast(
                "That image does not record where it was acquired, so it cannot be "
                "placed on the overview.",
                "warning",
            )
            return None
        record = OverviewRecord(
            record_id, os.path.basename(record_id), [key], view=view
        )
        record.pixel_size = self._pixel_size_of(image)
        record.images.append(tile)
        self._records[record_id] = record
        self._refresh_overview_list()
        return record_id

    def _store_cap(self, dtype: np.dtype) -> int:
        """The largest square an overview of *dtype* may be held at, in pixels a side.

        Derived from the memory budget rather than fixed, because bytes are what is
        being spent and pixels are a poor proxy for them: the same pixel cap costs 52 MB
        on a mosaic of 1024 px tiles and 118 MB on one of 3072 px tiles, and half as much
        again on a uint16 detector as on a uint8 one. Two bytes a pixel here -- the
        grayscale and the coverage mask beside it, one byte each for the usual uint8.

        Square is the assumption, and it errs the safe way: a long thin mosaic reduces
        on its longest side, so it comes in *under* budget rather than over.
        """
        per_pixel = int(np.dtype(dtype).itemsize) + 1  # + the coverage mask
        return max(1, math.isqrt(self._store_budget_bytes // per_pixel))

    def _stored_tile(self, image: FibsemImage) -> "_PlacedTile":
        """What a record keeps so it can be re-placed after a view switch.

        Reduced to this widget's *store* cap rather than the canvas's draw cap. The two
        are the same number today, so this is still exactly the array the canvas draws
        and keeping it costs no more than the canvas already does -- but they are not
        the same question, and which one belongs here is the whole of it: this is the
        finest the record will ever hold, so it is the ceiling on anything a zoom could
        recover. See `_init_ui`.

        Split into colour and coverage here rather than at each placement: a view switch
        re-places every record, and the split is the same answer every time.
        """
        data = image.filtered_data
        pixel_size = self._pixel_size_of(image)
        height, width = data.shape[0], data.shape[1]
        max_px = self._store_cap(data.dtype)
        # What was acquired, and where black is, both come off the *unfiltered* array --
        # `filtered_data` smears the boundary in both directions and would answer either
        # question wrong. Reduced the same way as the pixels so the three line up.
        #
        # In the source's own dtype rather than promoted to float32 first, and the same
        # for the mask: promoting is four bytes per *source* pixel of temporary, so the
        # cost tracks the mosaic rather than the few megabytes that come out of it, and
        # it was being spent twice over. Measured on a 10x10 of 1024 px tiles, the two
        # reductions together were 541 MB and 207 ms; they are 218 MB and 38 ms here,
        # for a bit-identical mask and the same contrast limits. `downsample` preserves
        # dtype and answers for every one; only a dtype cv2 refuses takes the slow path,
        # and beam images are uint8 or uint16.
        source = np.asarray(image.data)
        raw = downsample(source, max_px)
        acquired = downsample_mask(source > 0, max_px)
        clim = _contrast_limits(raw, acquired)
        return _PlacedTile(
            grey=downsample(data, max_px),
            acquired=acquired,
            clim=clim,
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
        key, tile = self._place(mosaic, key=record.id)
        if key is None or tile is None:
            logger.debug("The finished overview could not be placed.")
            return
        record.keys = [key]
        record.images = [tile]
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

        The canvas stays on the view the removed overview was in rather than jumping
        elsewhere: the view still describes where the next run would land if the stage
        is there, and `_follow_the_acquisition_view` moves it when that stops being true.
        """
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
        self._refresh_empty_view_hint()

    def _refresh_empty_view_hint(self) -> None:
        """Say where the overviews went when the displayed view moved off them.

        Re-posing the stage moves the displayed view to wherever the next run would land
        (`_follow_the_acquisition_view`), which is what a planner wants. But if nothing
        was acquired *there*, the canvas goes blank while the records are all still held
        -- and a canvas that empties itself moments after a stage move reads as a fault
        rather than as a view change.

        It was reported as one: the beam had to be "re-selected" to get the picture back,
        which worked only because changing the beam changes the view, and changing the
        view back is what actually restored it. The beam was never the problem (FIB-659).

        A hint rather than something drawn on the canvas. The status zone already ranks a
        standing instruction below the cursor readout and decays the readout so the hint
        underneath is not shut out, and the chips that resolve this sit directly above.
        `_plot_empty` stays silent on purpose -- its blank backdrop means "nothing was
        acquired here", which is a different and still-true statement. This one is
        "nothing was acquired here, but something was somewhere else", which the backdrop
        cannot say.
        """
        elsewhere = {
            record.view.label
            for record in self._records.values()
            if record.view is not None and record.view != self._current_view
        }
        if self.canvas.placed_keys or not elsewhere:
            self.canvas.set_hint(None)
            return
        names = sorted(elsewhere)
        where = names[0] if len(names) == 1 else f"{len(names)} other views"
        self.canvas.set_hint(f"No overview in this view — {where} has one")

    def load_overview(self, path: str) -> Optional[str]:
        """Load a saved overview from disk and place it. Returns its record id."""
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            notification_service.show_toast(
                f"Could not load {os.path.basename(path)}.", "error"
            )
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

        Everything here is derived from configuration and the cached stage position --
        with one exception, which is unavoidable rather than an oversight: the *first*
        refresh in a view builds that view's `BeamStageProjection`, and building one
        reads the scan rotation off the instrument. There is no drawing a view without
        a projection, and no projection without that read.

        Once per view, then cached (`_projection`), so the cost is not per event -- and
        it recurs only after `invalidate_projection`, which a host calls precisely
        because the instrument changed and the old answer is wrong. Worth knowing
        because on TFS that read is a set-then-read on the shared imaging channel
        (FIB-544, FIB-600), and the rest of this class goes to some trouble to keep such
        reads off UI events: `_on_cursor_moved` and `_target_offset` both take the
        cached projection for exactly that reason.

        The one place that anchors the canvas, so `_frame` stays a pure read: this runs
        on construction, on a stage move, on a settings change and on a view change,
        which is every moment the answer could have changed.
        """
        self._seed_frame()
        frame = self._frame()
        if frame is None:
            self.context_overlay.set_shapes([])
            return

        self.context_overlay.set_shapes(
            stage_context.context_shapes(
                self.microscope,
                frame,
                limits=self.overlay_controls.is_visible(_OVERLAY_LIMITS),
                boundaries=self.overlay_controls.is_visible(_OVERLAY_BOUNDARIES),
                slots=self.overlay_controls.is_visible(_OVERLAY_SLOTS),
            )
        )
        self._declare_working_area(frame)
        self._refresh_tile_grid()
        self._refresh_stage_info()
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

        A run in progress owns the answer, because during one "wherever the stage is"
        stops being the same question: the stage is at whichever tile it has reached,
        so the plan would describe the tile being acquired rather than the run doing
        the acquiring. The centre the run was started with is the one thing that is
        true for the whole of it (FIB-647).
        """
        if self._run_centre is not None:
            return self._run_centre
        return self._target or self._stage_position

    def _declare_working_area(self, frame: StageFrame) -> None:
        """Tell the canvas how much ground this tab is describing.

        Without it the canvas frames the images alone, and with none it frames whatever
        matplotlib made of the overlays -- a region of no particular shape. Keeping
        pixels square then costs the *axes*, which shrink to fit, and the canvas draws
        as a band across the middle of the widget with black either side. That is what
        an empty tab looked like, which since FIB-616 is a tab with plenty to show.

        The area is what the overlays actually cover: the grid boundary where there is
        one, and the planned run otherwise -- so opening the tab frames the sample and
        the plan rather than one tile's worth of nothing. A *minimum*: acquired images
        outside it still draw and still pull the view out to include them.

        Declared in metres about the grid centre. `set_world_extent` is idempotent, so
        restating the same area on every refresh is not a change and does not disturb
        the view; a real change refits, which is wanted -- the area only changes when
        the plan does.
        """
        span = 2 * GRID_BOUNDARY_RADIUS_M if self._draws_grid_boundary() else None
        settings = self._settings()
        if settings is not None:
            planned = max(settings.total_fov_x, settings.total_fov_y)
            span = planned if span is None else max(span, planned)
        if not span:
            return
        try:
            centre = frame.offset(self._landmark(frame, 0.0, 0.0))
        except Exception as e:
            logger.debug(f"Could not place the working area: {e}")
            return
        self.canvas.set_world_extent(span, span, centre)

    def _draws_grid_boundary(self) -> bool:
        """Whether the holder's grid boundary is being drawn, which bounds the view."""
        return bool(
            getattr(self.microscope._stage, "limits", None)
            and self.microscope.stage_is_compustage
        )

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

        # Anchor and flags handed over with the grid rather than set separately: this
        # runs on every motion event of a drag, and each setter used to repaint every
        # tile patch, so setting two of them cost two full repaints (FIB-751).
        #
        # No `display_pixel_size`: the overlay reads it off the canvas at draw time, so
        # the grid keeps describing the image underneath it when that image changes --
        # which it does mid-run, as tiles land.
        self.tile_grid_overlay.set_grid(
            tiles,
            (height, width),
            settings.image_settings.hfw / width,
            overlap=settings.overlap,
            unreachable=self._unreachable(settings, tiles=tiles),
            anchor=anchor,
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

    # Delegated to `stage_context`, which owns the drawing both tabs share. Kept as
    # methods because the tab resolves places of its own with them -- the grid centre it
    # declares a working area around, and the origin the tile grid hangs off.

    @staticmethod
    def _landmark(
        frame: StageFrame, x: float, y: float, name: str = ""
    ) -> FibsemStagePosition:
        return stage_context.landmark(frame, x, y, name)

    def _slot_landmark(self, slot: object) -> Optional[FibsemStagePosition]:
        return stage_context.slot_landmark(self.microscope, slot)

    @property
    def target(self) -> Optional[FibsemStagePosition]:
        """Where the next run is planned around, or None for wherever the stage is."""
        return self._target

    def _refresh_stage_info(self) -> None:
        """Say where the stage is, in the canvas's bottom-left info bar.

        The marker shows *where*; this says the numbers, which is what you need to write
        one down, or to check you are where you meant to be. On the canvas rather than
        in the settings column because it describes what is being looked at.

        The view rides along because on this tab it is not a given: the canvas can be
        showing one view while the stage sits in another, and a position without the
        direction it was read from is half an answer.

        So does the milling angle, which is what the stage tilt *means* on the beam
        side -- the angle the ion beam makes with the sample surface, and the number a
        milling pose is actually chosen for. The fluorescence tab deliberately leaves it
        out, on the grounds that it is meaningless through a camera and belongs here if
        anywhere.

        No hardware: `get_current_milling_angle` is arithmetic over the pose it is
        given, the same as `get_stage_orientation`. It can refuse -- a position with no
        tilt has no angle -- and is dropped on its own rather than taking the rest of
        the line with it.
        """
        position = self._stage_position
        if position is None:
            self.canvas.set_info_text(None)
            return
        parts = [position.pretty]
        view = self.acquisition_view
        if view is not None:
            parts.append(view.label)
        angle = self._milling_angle()
        if angle is not None:
            parts.append(f"milling {angle:.1f}°")
        self.canvas.set_info_text("   |   ".join(parts))

    def _milling_angle(self) -> Optional[float]:
        """The milling angle for the cached pose, in degrees, or None if it has none."""
        if self._stage_position is None:
            return None
        try:
            return self.microscope.get_current_milling_angle(
                stage_position=self._stage_position
            )
        except Exception as e:
            logger.debug(f"Could not work out the milling angle: {e}")
            return None

    def _on_cursor_moved(self, x: Optional[float], y: Optional[float]) -> None:
        """Report the stage position under the pointer, or hide once it leaves.

        What makes the canvas legible as a map rather than a picture: you can tell how
        far apart two features are without clicking either.

        Through the kept projection, never a fresh one. This fires on every motion
        event, and reading the instrument once per pixel of pointer travel to render a
        text label is what made the fluorescence grid drag stutter on hardware.
        """
        if x is None or y is None:
            self._set_cursor_readout("")
            return
        frame = self._frame()
        if frame is None:
            self._set_cursor_readout("")
            return
        try:
            self._set_cursor_readout(self._in_microns(frame.to_stage(x, y)))
        except Exception as e:
            logger.debug(f"Could not resolve the cursor position: {e}")
            self._set_cursor_readout("")

    def _set_cursor_readout(self, text: str) -> None:
        """Put the readout in the canvas's status zone, or give the zone back when empty.

        Through the canvas rather than a label of this widget's own. This used to be
        hand-placed below the view chips, in "the one free corner" -- which stopped being
        free when the status zone took that row, and the chips were already inside the
        status label's rectangle without anything showing it, because nothing on this tab
        ever set a hint or a flash (FIB-651).

        The zone shows one thing at a time in a fixed order, so its occupants cannot
        collide: a flash outranks this, and this outranks a standing hint. Empty means
        the pointer has left the canvas, and the zone falls back on its own -- so an
        empty readout releases the corner rather than leaving a blank plaque over the
        data.

        Not `set_info_text`, and no longer for the reason it used to be: that was an axes
        artist whose update repainted every placed image, and FIB-650 made it a `QLabel`
        too, so both are now cheap. What separates them is what they say. The info bar is
        where the *stage* is, standing until the instrument moves; this is what is under
        the *pointer*, true only while it is there. They are read against each other, and
        a canvas that put them in one place would have neither.
        """
        self.canvas.set_status_readout(text or None)

    def _refresh_position_markers(self) -> None:
        """Redraw the current stage position and every marked position.

        The *saved* positions are what the control hides. Where the stage is now stays
        drawn either way: it is not an annotation over the data so much as the one mark
        that says which part of the sample you are looking at, and hiding it makes the
        canvas harder to read rather than cleaner.
        """
        frame = self._frame()
        if frame is None:
            return

        if not self.overlay_controls.is_visible(_OVERLAY_POSITIONS):
            for overlay in (
                self.position_overlay,
                self.flagged_position_overlay,
                self.selected_position_overlay,
            ):
                overlay.set_points([])
            if self._stage_position is not None:
                try:
                    self.current_position_overlay.set_points(
                        [frame.to_canvas(self._stage_position)]
                    )
                except Exception as e:
                    logger.debug(f"Could not mark the current stage position: {e}")
            return

        if self._stage_position is not None:
            try:
                self.current_position_overlay.set_points(
                    [frame.to_canvas(self._stage_position)]
                )
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
        #
        # The two things that describe *where the stage is* still keep up, because they
        # are not the plan: the marker, and the readout under it. The readout is here
        # rather than left out because placing an image used to refresh it as a side
        # effect, so it tracked a run tile by tile -- and it should go on doing that now
        # that placing an image refreshes nothing.
        if self._running:
            self._refresh_position_markers()
            self._refresh_stage_info()
            return
        # A re-pose changes which view the next run lands in, and the canvas follows it
        # the same way it follows a change of beam.
        self._follow_the_acquisition_view()
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
        """Drive the stage to a position, off the GUI thread.

        Public because a host drives it too -- picking a lamella out of a list is the
        same act as double-clicking where it is drawn, so it goes through the same gate.
        It used to ask only whether *this* widget was acquiring, which let a host move
        the stage in cases a user clicking the canvas was refused.
        """
        if not self._may_move():
            return
        # Say so. A double-click that starts a multi-second stage move used to be
        # indistinguishable from one that did nothing -- and because `_may_move`
        # refusals *do* toast, silence was the state where something was actually
        # happening (FIB-765).
        #
        # The status label, not the progress bar, and this tab already says why: the bar
        # "carries the message for the whole run" while a run is on, and "the label
        # below it carries the outcome from here" once it is not. A stage move has no
        # fraction -- `safe_absolute_stage_movement` blocks and emits nothing along the
        # way -- so a bar would be inventing one, and the label is where a thing that
        # merely happens belongs. It is also what the fluorescence tab uses, which is
        # the point: these two drifted apart once already and that is this whole issue.
        self.label_status.setText(f"Moving to {self._describe(position)}…")
        worker = FunctionWorker(self._move_worker, position)
        worker.errored.connect(self._on_move_errored)
        worker.finished.connect(self._on_move_finished)
        worker.start()

    def _move_worker(self, target: FibsemStagePosition) -> None:
        """Runs off the GUI thread. Only signals may cross back.

        The exception is deliberately not caught. `FunctionWorker` logs it with a
        traceback and re-emits it as `errored` on the GUI thread, which is the only way
        the widget can tell a failed move from a finished one -- swallowing it here left
        the two identical, so a stage that never arrived reported success.
        """
        try:
            self.microscope.safe_absolute_stage_movement(target)
        finally:
            # In a `finally`, so it runs on the failing path too: a move that stopped
            # part-way has still left the stage somewhere, and the marker should say
            # where rather than where it set off from.
            #
            # Publishes the new position through `stage_position_changed`, which is what
            # re-marks it -- rather than assuming the stage arrived exactly where it was
            # asked to, which on a real instrument it does not.
            try:
                self.microscope.get_stage_position()
            except Exception as e:
                logger.debug(f"Could not confirm the stage position after moving: {e}")

    def _on_move_errored(self, error: object) -> None:
        """The stage did not get there. Say that, rather than falling quiet."""
        self._move_failed = True
        self.label_status.setText(f"Could not move the stage: {error}")
        notification_service.show_toast("Could not move the stage.", "error")

    def _on_move_finished(self) -> None:
        """Always runs, after `errored` when there was one.

        A failure has already put its own message up and that should stand; only a
        success replaces the "Moving to …" line. What replaces it is where the stage
        actually reached, read back by the worker rather than assumed from the target --
        the same report the fluorescence tab gives, in the same place.
        """
        if self._move_failed:
            self._move_failed = False
            return
        position = self._stage_position
        self.label_status.setText(
            f"At {self._describe(position)}" if position is not None else "Moved."
        )

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
        boxes -- and lamellae closer together than the field of view do overlap.
                *crosshair_only* drops the box and leaves the radius, for a caller that has to
        share the canvas with something else. The tile grid stands aside wherever this
        answers a name (FIB-767), and a field-of-view box is a large thing to reserve:
        it is a whole tile on the fluorescence tab by construction, and a whole tile on
        this one at any HFW of 100 um or under. Reserving it would leave tiles that
        cannot be toggled at all, with nothing on screen to say why -- where reserving
        only the crosshair costs at worst a click that toggles a tile you meant to
        select, which greys out visibly and undoes with one more click.
        """
        if not self.overlay_controls.is_visible(_OVERLAY_POSITIONS):
            # Turned off means gone, not merely invisible. `_refresh_position_markers`
            # reads the same control and draws nothing, so picking here would select a
            # lamella from a click on what looks like bare mosaic -- and the host fans
            # that selection out to every list in the window, with nothing on screen to
            # say why. The one marker that stays drawn when these are off is the current
            # stage position, and it was never pickable.
            return None

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

        best_name, best_distance = None, float("inf")
        for position in self._positions:
            if not position.name:
                continue
            try:
                centre = frame.to_canvas(position)
                point = transform.transform(centre)
            except Exception:
                continue
            distance = ((click[0] - point[0]) ** 2 + (click[1] - point[1]) ** 2) ** 0.5
            # `centre` is in canvas units and `point` in screen pixels: the box is a
            # fixed piece of sample, the radius a fixed piece of screen.
            if distance >= PICK_RADIUS_PX and (
                crosshair_only or not self.position_overlay.covers(centre, x, y)
            ):
                continue
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
                f"Cannot move the stage while {self._lock_reason}.", "warning"
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

    def _posed_like_the_stage(self, target: FibsemStagePosition) -> FibsemStagePosition:
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

    @staticmethod
    def _in_microns(position: FibsemStagePosition) -> str:
        """A stage position as microns, for the cursor readout.

        Microns rather than the millimetres-and-degrees of `pretty_string`: this is read
        while comparing two points on a sample, and at that scale millimetres are three
        zeros and a digit. The rotation and tilt are left out for the same reason -- they
        are the stage's, they do not change as the pointer moves, and they are already in
        the info bar.

        z is *not* left out. On a tilted stage a sideways move in the image carries a
        real z, and a readout that hid it would make the focal-plane change invisible.
        """
        return (
            f"x {(position.x or 0.0) * constants.SI_TO_MICRO:9.1f}  "
            f"y {(position.y or 0.0) * constants.SI_TO_MICRO:9.1f}  "
            f"z {(position.z or 0.0) * constants.SI_TO_MICRO:9.1f} µm"
        )

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
        settings.image_settings.filename = stamped_overview_name(
            settings.image_settings.filename
        )

        # Where this run happens, resolved **once**, before the dialog, and handed to the
        # runner unchanged. Everything about a run then comes from one value: the plan
        # drawn on the canvas, the pose the dialog names, and the ground the tiles are
        # computed from.
        #
        # It used to be resolved twice. The widget planned from its cached pose while the
        # runner was handed `None` unless the grid had been dragged -- and `None` means
        # "read the stage yourself", which it does when the worker starts, a moment later
        # and from a different source. Anything that re-posed the stage in between made
        # the dialog describe one view and the acquisition happen in another, with the
        # canvas agreeing with the dialog and the files agreeing with neither. Reported
        # from an instrument: the dialog read SEM @ MILLING and the overview came back
        # SEM @ SEM.
        #
        # The cached pose is not always fresh either -- `stage_position_changed` is
        # emitted by `get_stage_position`, so a move nobody polls after is a move this
        # tab never hears about (FIB-669). That is a real defect and this does not fix
        # it. What it does fix is the *disagreement*: with one value there is no longer a
        # second reading to differ from, so a stale pose gives a wrong-but-honest run
        # rather than a run that contradicts what it was authorised to do.
        self._run_centre = deepcopy(self._target or self._stage_position)

        if not self._confirm(settings):
            logger.info("Overview acquisition cancelled before starting")
            self._run_centre = None  # or the plan stays pinned to a run that never ran
            return

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
        # Copied, because the target can be dragged again while the run is under way and
        # the run has to keep the grid it was started with. `_run_centre` rather than
        # `_target`: the same value the canvas draws the plan around and the dialog just
        # described, so the three cannot come apart.
        self._worker = FunctionWorker(
            self._acquire_worker, settings, deepcopy(self._run_centre)
        )
        self._worker.start()

    def _confirm(self, settings: OverviewAcquisitionSettings) -> bool:
        """Show what is about to happen, and let it be called off.

        Two things a run carries are set on the canvas rather than in the controls, so
        neither is visible from the settings column at the moment Acquire is pressed:
        where the grid was dragged to (FIB-617) and which tiles are masked off
        (FIB-618). Both survive a tab switch. This is where they announce themselves.
        """
        view = self.acquisition_view
        dialog = OverviewConfirmationDialog(
            settings=settings,
            view_description=view.describe if view is not None else None,
            offset=self._target_offset(),
            unreachable=self._unreachable(settings),
            parent=self,
        )
        return dialog.exec_() == QDialog.Accepted

    def _unreachable(
        self,
        settings: OverviewAcquisitionSettings,
        tiles: Optional[List["tiled.TilePosition"]] = None,
    ) -> List[Tuple[int, int]]:
        """Which tiles of the planned run the stage cannot travel to.

        The runner asks this too, in `_compute_grid` -- but that runs on the worker,
        after the dialog has been accepted, so the sequence a user gets is: read the
        dialog, press Start, watch the run fail with a directory already made and the
        stage already moving. Asked here it is refused while the grid can still be
        moved or the offending tiles masked off.

        Through the tab's cached projection rather than `microscope.project_stable_move`,
        which the runner uses: the two agree to the bit, verified against the live call,
        but the microscope's re-reads the scan rotation every time -- one read per tile,
        on a dialog opening. The house rule against hardware reads on UI events
        (FIB-544, FIB-600) is aimed at things that fire constantly rather than at an
        explicit Acquire press, but there is no reason to pay it when the answer is
        already held.

        Best effort, like the disk estimate on the fluorescence dialog: a check that
        cannot run leaves the dialog without the warning rather than refusing to open
        it. The runner keeps the authoritative refusal, so being wrong here fails open.

        Also asked on the drag path, where the grid it flags can still be moved --
        `_refresh_tile_grid` passes the tiles it has just built rather than having them
        computed twice. Affordable there: 0.45 ms for a 3x3 and 9.3 ms for the largest
        grid the spin boxes allow, against a redraw already costing several times that.
        """
        try:
            projection = self._projection(self.acquisition_view)
            centre = self._grid_centre()
            limits = getattr(self.microscope._stage, "limits", None)
            if projection is None or centre is None or not limits:
                return []
            if tiles is None:
                tiles = tiled.compute_tile_grid(settings, mask=settings.tile_mask)
            return unreachable_tiles(
                tiles,
                settings.tile_order,
                lambda dx, dy: projection.from_plane(dx, dy, centre),
                limits,
            )
        except Exception as e:
            logger.debug(f"Could not check the grid against the stage limits: {e}")
            return []

    def _target_offset(self) -> Optional[Tuple[float, float]]:
        """How far the grid's centre sits from the stage, in metres, or None for on it.

        From the cached stage position, like everything else here -- opening a dialog
        must not reach for the instrument.
        """
        if self._target is None or self._stage_position is None:
            return None
        return (
            self._target.x - self._stage_position.x,
            self._target.y - self._stage_position.y,
        )

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
        # A run stops at the next tile boundary, so the button stays there for a while
        # after it is pressed. Disabled once asked, or a second press reads as the first
        # one not having worked.
        self._apply_enabled_state()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._apply_enabled_state()
        self.acquiring_changed.emit(running)
        if running:
            # The framing you pressed Acquire with is the framing you keep. A run is the
            # worst moment to re-frame: the preview lands under one key and the stitch
            # replaces it under another, so the canvas held still for the whole
            # acquisition and then lurched twice at the end, when there was finally
            # something worth looking at (FIB-648).
            #
            # Not restored afterwards. There is content on the canvas now, and the
            # framing belongs to whoever last set it; "reset view" is how you ask for
            # it back.
            self.canvas.auto_fit = False
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
        self._set_running(False)
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
        # Handed back before the redraw below, so the plan goes back to describing the
        # *next* run rather than the one that has just ended.
        self._run_centre = None
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

        Beam runs only. This widget places the payload's mosaic on its own canvas and
        counts the tiles into its own record, so a fluorescence run reaching here would
        be drawn as one of this tab's overviews (FIB-725).
        """
        if not is_modality(payload, MODALITY_BEAM):
            return

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
