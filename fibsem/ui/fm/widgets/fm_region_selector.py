"""An acquired FIB/SEM overview, with the ground to image drawn on it.

The left-hand pane of the sparse selection dialog. The user looks at a beam overview,
decides which parts of the grid are worth the fluorescence microscope's time, and draws
rectangles over them; this reports those rectangles in the overview's own displayed
plane, which is what :func:`fibsem.fm.planning.plan_sparse_fm_overview` consumes.

On a real-space canvas rather than an image canvas, and the reason is plural: one view
can hold several overviews. Records are placed by stage position and a view switch
re-places all of them, so somebody who has taken three FIB overviews of different parts
of a grid can select across all three. An image canvas shows one image.

**Ground, not tiles.** No tile grid is drawn here and none is consulted: the beam field
of view and the camera's are unrelated sizes, so there is no tile-to-tile correspondence
to preserve. What crosses to the other pane is an area of sample.

**No stage-context overlays.** The travel limits and the grid boundary are drawn in the
*preview* pane, in the frame the acquisition happens in, which is the frame that decides
whether a tile can be reached. Drawing them here as well would duplicate the answer and
carry a hazard to do it: overlays sized from a stage length ignore the view, and this one
is foreshortened up to 3.9x at the milling pose, so a boundary drawn without
`surface_foreshortening()` is wrong by that much and still looks like a boundary.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.tiling.geometry import PlaneRegion
from fibsem.microscope import FibsemMicroscope
from fibsem.projection import BeamStageProjection
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.tokens import WHITE_ICON_COLOR
from fibsem.ui.widgets.canvas.overlays.rect_overlay import RectOverlay
from fibsem.ui.widgets.canvas.overlays.tile_grid_overlay import ENABLED_EDGE
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas
from fibsem.ui.widgets.canvas.stage_frame import StageFrame
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.widgets.overview_widget import (
    _VIEW_CHIP_SPACING,
    _VIEW_CHIP_STYLE,
    _VIEW_CHIP_STYLE_ACTIVE,
    _VIEW_STRIP_STYLE,
    OverviewView,
)

logger = logging.getLogger(__name__)

# The same magenta the tile grid draws in, deliberately: a region on this pane and the
# tiles it produces on the other are one selection seen twice, and colouring them apart
# would suggest otherwise.
REGION_COLOUR = ENABLED_EDGE
REGION_SELECTED_COLOUR = WHITE_ICON_COLOR
# A new region covers a fifth of the frame's short side: big enough to see and grab,
# small enough that it is obviously a starting point rather than an answer.
_NEW_REGION_FRACTION = 0.2


class FMRegionSelectorWidget(QWidget):
    """A beam overview with draggable regions over it.

    Signals:
        regions_changed: a region was added, moved, resized or deleted.
        selection_changed: which region is selected, by index, or None.
    """

    regions_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)

    def __init__(
        self, microscope: FibsemMicroscope, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope

        self._projection: Optional[BeamStageProjection] = None
        self._origin: Optional[FibsemStagePosition] = None
        self._overlays: List[RectOverlay] = []
        self._selected: Optional[int] = None
        self._placed: List[str] = []
        self._views: Dict["OverviewView", List[FibsemImage]] = {}
        self._current_view: Optional["OverviewView"] = None
        # The ground the placed overviews cover, in metres, for sizing a new region.
        self._content_extent: Tuple[float, float] = (1e-3, 1e-3)

        # The bare canvas, as the beam overview tab uses it. `FMRealSpaceCanvasWidget`
        # wraps one in channel and layer controls, which belong to fluorescence.
        self.canvas = FibsemRealSpaceCanvas(display_max_px=2048)

        # Above the canvas, not on it. The chips run to ~650 px against a 640 px canvas,
        # so no corner holds them -- settled in the beam overview tab and followed here.
        self._chip_strip = QScrollArea()
        self._chip_strip.setObjectName("viewStrip")
        self._chip_strip.setStyleSheet(_VIEW_STRIP_STYLE)
        self._chip_strip.setWidgetResizable(True)
        self._chip_strip.setFixedHeight(34)
        self._chip_strip.setFrameShape(QFrame.NoFrame)
        self._chip_strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_strip.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._chip_strip)
        layout.addWidget(self.canvas)

        self.canvas.canvas_clicked.connect(self._on_left_click)
        self.canvas.canvas_right_clicked.connect(self._on_right_click)

    # ── what is being looked at ──────────────────────────────────────────

    def set_views(
        self,
        views: Dict["OverviewView", Sequence[FibsemImage]],
        current: Optional["OverviewView"] = None,
    ) -> None:
        """Offer these views, and show one.

        A view is a beam and a stage orientation. Images from different views do not
        register with each other -- they are pictures of the same sample from different
        directions -- so they are never placed together, and switching is what the chips
        are for. Defaults to the last view offered, which is the one most recently
        acquired.

        The strip stays hidden below two views: a control with one option is a label
        that looks clickable.
        """
        self._views = {view: list(images) for view, images in views.items()}
        chosen = current if current in self._views else None
        if chosen is None and self._views:
            chosen = list(self._views)[-1]
        self._build_chips()
        self.show_view(chosen)

    def show_view(self, view: Optional["OverviewView"]) -> None:
        """Show one of the offered views. Starts a fresh selection."""
        self._current_view = view
        self.set_overviews(self._views.get(view, []) if view is not None else [])
        self._build_chips()

    @property
    def current_view(self) -> Optional["OverviewView"]:
        return self._current_view

    def _build_chips(self) -> None:
        """One chip per view, the current one marked."""
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(_VIEW_CHIP_SPACING)
        for view in self._views:
            chip = QPushButton(view.label)
            chip.setCheckable(True)
            chip.setChecked(view == self._current_view)
            chip.setStyleSheet(
                _VIEW_CHIP_STYLE_ACTIVE
                if view == self._current_view
                else _VIEW_CHIP_STYLE
            )
            chip.clicked.connect(lambda _checked, v=view: self.show_view(v))
            row.addWidget(chip)
        row.addStretch(1)
        self._chip_strip.setWidget(holder)
        self._chip_strip.setVisible(len(self._views) > 1)

    def set_overviews(self, images: Sequence[FibsemImage]) -> None:
        """Show these overviews, and start a fresh selection.

        They are expected to share a view -- a beam and a stage orientation. Images from
        different views do not register with each other, so placing them together would
        say something untrue; the caller picks the view and hands over what belongs to it.

        The projection and the frame origin come from the *first* image rather than from
        the live instrument, so the selection resolves against the overview as it was
        taken. That matters here more than usually: the stage may well have moved since,
        and a saved overview is exactly the case a live projection gets wrong.
        """
        self.clear_regions()
        for key in self._placed:
            self.canvas.remove_image(key)
        self._placed = []
        self._projection = None
        self._origin = None

        if not images:
            return

        projection = BeamStageProjection.from_image(images[0])
        origin = self._position_of(images[0])
        if projection is None or origin is None:
            logger.debug("That overview does not record how it was acquired.")
            return
        self._projection, self._origin = projection, origin

        pixel_size = self._pixel_size_of(images[0])
        if pixel_size:
            self.canvas.set_reference_pixel_size(pixel_size)

        frame = self._frame()
        spans: List[Tuple[float, float, float, float]] = []
        for index, image in enumerate(images):
            position = self._position_of(image)
            size = self._pixel_size_of(image)
            if position is None or not size or frame is None:
                logger.debug(
                    "Skipping an overview that does not say where it was taken."
                )
                continue
            try:
                centre = frame.offset(position)
            except Exception as e:
                logger.debug(f"Could not place an overview: {e}")
                continue
            data = image.filtered_data
            self._placed.append(
                self.canvas.add_image(
                    data,
                    centre=centre,
                    pixel_size=size,
                    key=f"overview-{index}",
                )
            )
            spans.append(
                (
                    centre[0],
                    centre[1],
                    data.shape[1] * size,
                    data.shape[0] * size,
                )
            )

        if spans:
            xs = [cx - w / 2 for cx, _, w, _ in spans] + [
                cx + w / 2 for cx, _, w, _ in spans
            ]
            ys = [cy - h / 2 for _, cy, _, h in spans] + [
                cy + h / 2 for _, cy, _, h in spans
            ]
            self._content_extent = (max(xs) - min(xs), max(ys) - min(ys))

    @property
    def has_overview(self) -> bool:
        """Whether anything is placed and a region could be drawn against it."""
        return bool(self._placed) and self._projection is not None

    # ── regions ──────────────────────────────────────────────────────────

    def add_region(self) -> Optional[int]:
        """Drop a new region in the middle of the view, and select it.

        Placed rather than dragged out. `RectOverlay` starts a drag only on a handle or
        on its own body, so there is no create-by-dragging gesture to reach for, and
        adding one would mean a create mode inside a widget several other callers share.
        Drop-then-resize is also the gesture the vendor's own sparse tileset uses.
        """
        if not self.has_overview:
            return None
        overlay = RectOverlay(
            color=REGION_COLOUR,
            facecolor=REGION_COLOUR,
            alpha=0.30,
            linewidth=2,
            resizable=True,
        )
        self.canvas.add_overlay(overlay)
        overlay.set_rect(*self._new_region_rect())
        overlay.rect_changed.connect(self._on_region_changed)
        self._overlays.append(overlay)
        self._select(len(self._overlays) - 1)
        self.regions_changed.emit()
        return len(self._overlays) - 1

    def delete_selected_region(self) -> None:
        """Remove the selected region, if there is one."""
        if self._selected is None:
            return
        overlay = self._overlays.pop(self._selected)
        self.canvas.remove_overlay(overlay)
        self._select(None)
        self.regions_changed.emit()

    def clear_regions(self) -> None:
        """Remove every region."""
        if not self._overlays:
            return
        for overlay in self._overlays:
            self.canvas.remove_overlay(overlay)
        self._overlays = []
        self._select(None)
        self.regions_changed.emit()

    @property
    def regions(self) -> List[PlaneRegion]:
        """The regions, in the overview's displayed plane, in metres from :attr:`base`.

        What `plan_sparse_fm_overview` takes. Metres rather than pixels because pixels
        belong to a detector and would cancel immediately on the other side.
        """
        out = []
        for overlay in self._overlays:
            rect = overlay.get_rect()
            out.append(
                PlaneRegion.from_points(
                    [
                        self.canvas.canvas_to_metres(rect["x0"], rect["y0"]),
                        self.canvas.canvas_to_metres(rect["x1"], rect["y1"]),
                    ]
                )
            )
        return out

    @property
    def base(self) -> Optional[FibsemStagePosition]:
        """The stage position :attr:`regions` are measured from."""
        return self._origin

    @property
    def projection(self) -> Optional[BeamStageProjection]:
        """The overview's own projection, for resolving those regions to stage space."""
        return self._projection

    @property
    def selected_region(self) -> Optional[int]:
        return self._selected

    def describe_selected(self) -> str:
        """A one-line summary of the selected region, or of the selection as a whole."""
        regions = self.regions
        if not regions:
            return "No regions. Right-click to add one."
        if self._selected is None or self._selected >= len(regions):
            return f"{len(regions)} region{'s' if len(regions) != 1 else ''}."
        region = regions[self._selected]
        return (
            f"Region {self._selected + 1} of {len(regions)} — "
            f"{region.width * 1e6:.0f} x {region.height * 1e6:.0f} um "
            "as drawn"
        )

    # ── interaction ──────────────────────────────────────────────────────

    def _on_right_click(self, x: float, y: float, modifiers=None) -> None:
        """Add or delete, from wherever the pointer is.

        A menu rather than a toolbar: adding is rare enough not to deserve permanent
        chrome, and deleting needs a target, which a right-click already names.
        """
        self._select(self._region_at(x, y))
        config = ContextMenuConfig()
        config.add_action(
            "Add region", callback=self.add_region, enabled=self.has_overview
        )
        config.add_action(
            "Delete region",
            callback=self.delete_selected_region,
            enabled=self._selected is not None,
        )
        config.add_action(
            "Clear all regions",
            callback=self.clear_regions,
            enabled=bool(self._overlays),
        )
        ContextMenu(config, parent=self).show_at_cursor()

    def _on_left_click(self, x: float, y: float, modifiers=None) -> None:
        """Select what was clicked, or nothing.

        Only reaches here for a click the overlays did not claim: `RectOverlay` takes
        the press when it lands on its own body or a handle, and suppresses the click
        that would follow. So this selects a region by its *edge* and deselects on the
        background, while clicking a region's middle selects it through
        :meth:`_on_region_changed` instead -- the two paths together cover the rectangle.
        """
        self._select(self._region_at(x, y))

    def _on_region_changed(self, _rect: dict) -> None:
        """A region moved or resized. Whichever it was becomes the selected one.

        From the sender rather than from the pointer: the overlay that emitted is the
        one being dragged, and asking the position again could name a different region
        where two overlap.
        """
        overlay = self.sender()
        if overlay in self._overlays:
            self._select(self._overlays.index(overlay))
        self.regions_changed.emit()

    def _region_at(self, x: float, y: float) -> Optional[int]:
        """The topmost region under a point, or None.

        Reverse order, so the one drawn last -- and so on top -- wins, matching what
        clicking it would have hit.
        """
        for index in reversed(range(len(self._overlays))):
            rect = self._overlays[index].get_rect()
            if rect["x0"] <= x <= rect["x1"] and rect["y0"] <= y <= rect["y1"]:
                return index
        return None

    def _select(self, index: Optional[int]) -> None:
        if index == self._selected:
            return
        self._selected = index
        for position, overlay in enumerate(self._overlays):
            colour = REGION_SELECTED_COLOUR if position == index else REGION_COLOUR
            overlay.set_color(colour, facecolor=REGION_COLOUR)
        self.selection_changed.emit(index)

    # ── internals ────────────────────────────────────────────────────────

    def _frame(self) -> Optional[StageFrame]:
        if self._projection is None or self._origin is None:
            return None
        return StageFrame(self.canvas, self._origin, self._projection)

    def _new_region_rect(self) -> Tuple[float, float, float, float]:
        """Where a new region starts, in canvas coordinates.

        A square about the frame origin -- the middle of what was placed -- sized from
        the ground the overviews cover, and stepped for each one already down so a second
        Add does not land exactly on the first and look like nothing happened.
        """
        side_m = min(self._content_extent) * _NEW_REGION_FRACTION
        step_m = side_m * 0.35 * (len(self._overlays) % 4)
        side, _ = self.canvas.metres_to_canvas(side_m, 0.0)
        step, _ = self.canvas.metres_to_canvas(step_m, 0.0)
        centre_x, centre_y = self.canvas.metres_to_canvas(0.0, 0.0)
        return (
            centre_x - side / 2 + step,
            centre_y - side / 2 + step,
            side,
            side,
        )

    @staticmethod
    def _position_of(image: FibsemImage) -> Optional[FibsemStagePosition]:
        state = getattr(getattr(image, "metadata", None), "microscope_state", None)
        return getattr(state, "stage_position", None)

    @staticmethod
    def _pixel_size_of(image: FibsemImage) -> Optional[float]:
        pixel_size = getattr(getattr(image, "metadata", None), "pixel_size", None)
        return getattr(pixel_size, "x", None)
