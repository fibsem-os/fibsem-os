"""A canvas of stored overview images, with positions marked on them. Nothing live.

The Overview tab's canvas is the stage: it draws where the stage is, what it can
reach, and what has been acquired there, and a click on it drives the instrument.
This one is a grid's record. It shows the overviews a grid task saved, placed from
each image's own metadata, and it lets positions be marked on them -- for a grid
that is in the magazine, or on another instrument, or was screened last week.
Nothing here reads the microscope and nothing here moves it.

Every image says where it was taken and how it projects (``BeamStageProjection``
and ``FMStageProjection`` both build from an image), so a click on it becomes a
stage position from what the image recorded, the way the old minimap did. Images
of the same *view* -- the same beam at the same pose, or the fluorescence
microscope -- register with each other by their recorded positions and share one
frame. Views cannot be drawn in one frame (the FIB sees the grid foreshortened,
the FM from another pose), so the canvas shows one view at a time and remembers
the others; ``show_view`` switches.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.fm.preview import composite_projection
from fibsem.fm.structures import FluorescenceImage
from fibsem.projection import BeamStageProjection, FMStageProjection
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.tokens import (
    DRAFT_POSITION_COLOUR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import FieldOfViewOverlay
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas
from fibsem.ui.widgets.canvas.stage_frame import StageFrame
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig

logger = logging.getLogger(__name__)

VIEW_FM = "FM"

# The box drawn around a marked position: the field of view a lamella image
# covers, the same size the Overview tab draws.
POSITION_FOV_WIDTH = 100e-6
POSITION_FOV_HEIGHT = POSITION_FOV_WIDTH * (1024 / 1536)


class StoredOverview:
    """One image on the canvas: what it is of, where it sits, how it projects."""

    def __init__(
        self,
        record_id: str,
        label: str,
        view: str,
        position: FibsemStagePosition,
        projection,
        pixel_size: float,
        data: np.ndarray,
        item_id: Optional[str],
        item_name: Optional[str],
    ) -> None:
        self.id = record_id
        self.label = label
        self.view = view
        self.position = position
        self.projection = projection
        self.pixel_size = pixel_size
        self.data = data
        self.item_id = item_id
        self.item_name = item_name
        self.key: Optional[str] = None
        self.extent: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    @property
    def covers(self) -> Tuple[float, float]:
        """The ground the image covers, in metres: (width, height)."""
        height, width = self.data.shape[0], self.data.shape[1]
        return width * self.pixel_size, height * self.pixel_size


def _item_of(image) -> Tuple[Optional[str], Optional[str]]:
    ref = getattr(getattr(image, "metadata", None), "experiment", None)
    return getattr(ref, "item_id", None), getattr(ref, "item_name", None)


def _view_of_beam_image(image: FibsemImage, position: FibsemStagePosition) -> str:
    """Images of the same beam at the same pose register with each other."""
    beam = getattr(getattr(image.metadata, "image_settings", None), "beam_type", None)
    name = getattr(beam, "name", None) or "BEAM"
    name = {"ELECTRON": "SEM", "ION": "FIB"}.get(name, name)
    r = math.degrees(float(position.r or 0.0))
    t = math.degrees(float(position.t or 0.0))
    return f"{name} @ r={r:.0f}° t={t:.0f}°"


class StoredOverviewCanvas(QWidget):
    """Stored overviews, one view at a time, with positions marked on them.

    Signals carry stage positions computed from the images' own metadata:

    * ``position_add_requested(position, record_id)`` -- a right-click's
      "Add position here"; ``record_id`` names the overview the click was on,
      or is None over bare canvas.
    * ``position_move_requested(name, position)`` -- "Move selected position here".
    * ``position_selected(name)`` -- a left click landed on a marked position.
    * ``draft_remove_requested(index)`` -- "Remove position" over a draft mark.

    Two layers of marks, because two different things are drawn at once. The
    **positions** are what the experiment holds; the **drafts** are positions
    somebody has placed and not committed, which is what a review carries
    until it is confirmed. The canvas draws both and reports what was asked
    for; what a mark means, and whether anything is written, is the owner's.
    """

    position_add_requested = pyqtSignal(object, object)
    position_move_requested = pyqtSignal(str, object)
    position_selected = pyqtSignal(str)
    draft_remove_requested = pyqtSignal(int)
    view_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._records: Dict[str, StoredOverview] = {}
        self._count = 0
        self._view: Optional[str] = None
        self._origins: Dict[str, FibsemStagePosition] = {}
        self._positions: List[FibsemStagePosition] = []
        self._movable = True
        self._placing = True
        self._drafts: List[FibsemStagePosition] = []
        self._selected: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FibsemRealSpaceCanvas(display_max_px=2048)
        layout.addWidget(self.canvas)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.canvas_right_clicked.connect(self._on_canvas_right_clicked)

        self.position_overlay = FieldOfViewOverlay(
            color=SAVED_POSITION_COLOUR,
            marker="+",
            size=11,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.position_overlay)
        self.selected_position_overlay = FieldOfViewOverlay(
            color=SELECTED_POSITION_COLOUR,
            marker="+",
            size=15,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.selected_position_overlay)
        # On top: a draft sits over what is already there, because it is the
        # thing being worked on.
        self.draft_overlay = FieldOfViewOverlay(
            color=DRAFT_POSITION_COLOUR,
            marker="+",
            size=13,
            extent=(POSITION_FOV_WIDTH, POSITION_FOV_HEIGHT),
        )
        self.canvas.add_overlay(self.draft_overlay)

    # ── images ───────────────────────────────────────────────────────────

    @property
    def overviews(self) -> List[StoredOverview]:
        return list(self._records.values())

    @property
    def view(self) -> Optional[str]:
        return self._view

    @property
    def views(self) -> List[str]:
        seen: List[str] = []
        for record in self._records.values():
            if record.view not in seen:
                seen.append(record.view)
        return seen

    def set_image(self, image, label: Optional[str] = None) -> Optional[str]:
        """Place a stored overview, beam or fluorescence, from its own metadata.

        Returns the record id, or None for an image that does not say where it
        was taken or how it projects -- refused rather than placed at the
        origin, since an image in the wrong place looks like one in the right
        place. Switches the canvas to the image's view.
        """
        record = self._record_from(image, label)
        if record is None:
            return None
        self._records[record.id] = record
        self._origins.setdefault(record.view, record.position)
        if self._view != record.view:
            self.show_view(record.view)
        else:
            self._place(record)
            self._refresh_positions()
        return record.id

    def _record_from(self, image, label: Optional[str]) -> Optional[StoredOverview]:
        self._count += 1
        record_id = f"stored-{self._count}"
        metadata = getattr(image, "metadata", None)
        position = getattr(metadata, "stage_position", None)
        if metadata is None or position is None:
            logger.debug("A stored overview with no stage position cannot be placed.")
            return None
        item_id, item_name = _item_of(image)
        if isinstance(image, FluorescenceImage):
            projection = FMStageProjection.from_image(image)
            pixel_size = getattr(metadata, "pixel_size_x", None)
            if projection is None or not pixel_size:
                logger.debug(
                    "A fluorescence overview with no geometry cannot be placed."
                )
                return None
            data = composite_projection(image)
            view = VIEW_FM
        else:
            projection = BeamStageProjection.from_image(image)
            pixel_size = getattr(getattr(metadata, "pixel_size", None), "x", None)
            if projection is None or not pixel_size:
                logger.debug("A beam overview with no geometry cannot be placed.")
                return None
            data = np.asarray(image.data)
            view = _view_of_beam_image(image, position)
        return StoredOverview(
            record_id,
            label or record_id,
            view,
            position,
            projection,
            float(pixel_size),
            data,
            item_id,
            item_name,
        )

    def _frame(self, view: Optional[str] = None) -> Optional[StageFrame]:
        view = view or self._view
        origin = self._origins.get(view or "")
        if view is None or origin is None:
            return None
        projection = next(
            (r.projection for r in self._records.values() if r.view == view), None
        )
        if projection is None:
            return None
        return StageFrame(self.canvas, origin, projection)

    def _place(self, record: StoredOverview) -> None:
        frame = self._frame(record.view)
        if frame is None:
            return
        if self.canvas.reference_pixel_size is None:
            self.canvas.set_reference_pixel_size(record.pixel_size)
        try:
            centre = frame.offset(record.position)
        except Exception as e:
            logger.debug(f"Could not place a stored overview: {e}")
            return
        record.key = self.canvas.add_image(
            record.data,
            centre=centre,
            pixel_size=record.pixel_size,
            key=record.id,
            covers=record.covers,
        )
        record.extent = (centre, record.covers)

    def show_view(self, view: str) -> None:
        """Show every stored overview of *view*, and the positions as it sees them."""
        if view not in self.views:
            return
        self._view = view
        self.canvas.clear_images()
        for record in self._records.values():
            record.key = None
            record.extent = None
        for record in self._records.values():
            if record.view == view:
                self._place(record)
        self.canvas.reset_view()
        self._refresh_positions()
        self.view_changed.emit(view)

    def remove_overview(self, record_id: str) -> bool:
        record = self._records.pop(record_id, None)
        if record is None:
            return False
        if record.key is not None:
            self.canvas.remove_image(record.key)
        if record.view not in self.views:
            self._origins.pop(record.view, None)
            if self._view == record.view:
                self._view = None
                remaining = self.views
                if remaining:
                    self.show_view(remaining[0])
        self._refresh_positions()
        return True

    def clear(self) -> None:
        self._records.clear()
        self._origins.clear()
        self._view = None
        self.canvas.clear_images()
        self._refresh_positions()

    def record_at(self, x: float, y: float) -> Optional[StoredOverview]:
        """The shown overview under a canvas point, or None over bare canvas."""
        try:
            px, py = self.canvas.canvas_to_metres(x, y)
        except Exception as e:
            logger.debug(f"Could not resolve the clicked point: {e}")
            return None
        for record in reversed(list(self._records.values())):
            if record.extent is None:
                continue
            (cx, cy), (w, h) = record.extent
            if abs(px - cx) <= w / 2 and abs(py - cy) <= h / 2:
                return record
        return None

    def item_of(self, record_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        record = self._records.get(record_id or "")
        if record is None:
            return None, None
        return record.item_id, record.item_name

    # ── positions ────────────────────────────────────────────────────────

    def set_positions(
        self, positions: List[FibsemStagePosition], movable: bool = True
    ) -> None:
        """The positions to mark; each is drawn where this view's frame puts it,
        and one the frame cannot place is left off rather than guessed.

        ``movable`` False offers no "move it here": a caller that is showing
        these for context rather than editing them, as a review does with the
        lamellae a grid already has.
        """
        self._positions = list(positions)
        self._movable = movable
        self._refresh_positions()

    def set_placing_enabled(self, enabled: bool) -> None:
        """Whether a position can be placed or taken back here.

        Off, the right-click offers neither: a caller showing a record that is
        already decided must not offer an action that would do nothing, which
        is worse than not offering it -- the click looks broken rather than
        refused.
        """
        self._placing = enabled

    def set_draft_positions(self, positions: List[FibsemStagePosition]) -> None:
        """Positions placed but not committed, drawn in their own colour over
        the rest. Nothing here writes them anywhere; the owner decides what a
        draft becomes."""
        self._drafts = list(positions)
        self._refresh_positions()

    @property
    def draft_positions(self) -> List[FibsemStagePosition]:
        return list(self._drafts)

    def set_selected_position(self, name: Optional[str]) -> None:
        self._selected = name
        self._refresh_positions()

    @property
    def selected_position(self) -> Optional[str]:
        return self._selected

    def _refresh_positions(self) -> None:
        frame = self._frame()
        if frame is None:
            self.position_overlay.set_points([])
            self.selected_position_overlay.set_points([])
            self.draft_overlay.set_points([])
            return
        points, labels, selected = [], [], []
        for position in self._positions:
            try:
                point = frame.to_canvas(position)
            except Exception:
                continue
            name = position.name or ""
            if name and name == self._selected:
                selected.append(point)
            else:
                points.append(point)
                labels.append(name)
        self.position_overlay.set_points(points, labels=labels)
        self.selected_position_overlay.set_points(
            selected, labels=[self._selected] if selected else None
        )
        drafts = []
        for position in self._drafts:
            try:
                drafts.append(frame.to_canvas(position))
            except Exception:
                continue
        self.draft_overlay.set_points(drafts)

    def stage_position_at(self, x: float, y: float) -> Optional[FibsemStagePosition]:
        """The stage position a canvas point names, in the shown view's frame."""
        frame = self._frame()
        if frame is None:
            return None
        try:
            return frame.to_stage(x, y)
        except Exception as e:
            logger.debug(f"Could not resolve the clicked position: {e}")
            return None

    def position_at(self, x: float, y: float) -> Optional[str]:
        """The marked position under a canvas point: inside its box, or nearest
        crosshair within half a box; None otherwise."""
        frame = self._frame()
        if frame is None:
            return None
        # A length in canvas units, as a difference of two mapped points so the
        # frame's own offset cancels.
        radius = abs(
            self.canvas.metres_to_canvas(POSITION_FOV_WIDTH / 2, 0.0)[0]
            - self.canvas.metres_to_canvas(0.0, 0.0)[0]
        )
        best, best_distance = None, None
        for position in self._positions:
            name = position.name or ""
            if not name:
                continue
            try:
                px, py = frame.to_canvas(position)
            except Exception:
                continue
            distance = math.hypot(px - x, py - y)
            overlay = (
                self.selected_position_overlay
                if name == self._selected
                else self.position_overlay
            )
            if distance <= radius or overlay.covers((px, py), x, y):
                if best_distance is None or distance < best_distance:
                    best, best_distance = name, distance
        return best

    def draft_at(self, x: float, y: float) -> Optional[int]:
        """Which draft mark a canvas point is on, by index. Drafts have no
        names -- nothing has been created to name them after -- so the index
        is what a caller removes by."""
        frame = self._frame()
        if frame is None:
            return None
        radius = abs(
            self.canvas.metres_to_canvas(POSITION_FOV_WIDTH / 2, 0.0)[0]
            - self.canvas.metres_to_canvas(0.0, 0.0)[0]
        )
        best, best_distance = None, None
        for index, position in enumerate(self._drafts):
            try:
                px, py = frame.to_canvas(position)
            except Exception:
                continue
            distance = math.hypot(px - x, py - y)
            if distance <= radius or self.draft_overlay.covers((px, py), x, y):
                if best_distance is None or distance < best_distance:
                    best, best_distance = index, distance
        return best

    # ── clicks ───────────────────────────────────────────────────────────

    def _on_canvas_clicked(self, x: float, y: float, modifiers=None) -> None:
        name = self.position_at(x, y)
        if name is None:
            return
        self.set_selected_position(name)
        self.position_selected.emit(name)

    def _on_canvas_right_clicked(self, x: float, y: float, modifiers=None) -> None:
        config = self.position_menu(x, y)
        if config is None:
            return
        ContextMenu(config, parent=self).show_at_cursor()

    def position_menu(self, x: float, y: float) -> Optional[ContextMenuConfig]:
        """What a right-click at a canvas point offers, or None to offer nothing.
        Separate from showing it, so a test can read the offer without a modal
        menu."""
        target = self.stage_position_at(x, y)
        if target is None:
            return None
        record = self.record_at(x, y)
        record_id = record.id if record is not None else None
        config = ContextMenuConfig()
        draft = self.draft_at(x, y)
        if not self._placing:
            if not (self._selected and self._movable):
                return None
        elif draft is not None:
            # Over a draft: removing the one under the cursor is the offer, and
            # adding another on top of it is not.
            config.add_action(
                "Remove Position",
                callback=lambda i=draft: self.draft_remove_requested.emit(i),
            )
            return config
        if self._placing:
            config.add_action(
                "Add Position Here",
                callback=lambda: self.position_add_requested.emit(target, record_id),
            )
        if self._selected and self._movable:
            selected = self._selected
            config.add_action(
                f"Move Selected Position Here ({selected})",
                callback=lambda: self.position_move_requested.emit(selected, target),
            )
        return config

    def request_add_at(self, x: float, y: float) -> None:
        """Ask for a position at a canvas point, naming the overview it is on."""
        target = self.stage_position_at(x, y)
        if target is None:
            return
        record = self.record_at(x, y)
        self.position_add_requested.emit(target, record.id if record else None)
