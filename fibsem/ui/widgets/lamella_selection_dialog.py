"""Place lamella positions by clicking a grid's overview image.

Opened from the Grids → Results overview. Shows the grid's saved overview in a
matplotlib canvas (left) beside the grid's lamella list (right). The user can
right-click to add a position or move the selected one, and left-click / list-
select to highlight. On Accept the edits are committed to the experiment, using
the microscope state recorded in the overview image (so positions are placed in
the frame the overview was acquired in).

Colours + markers match the minimap (cyan = unselected, lime = selected).
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.tiled import (
    convert_image_coord_to_stage_position,
    reproject_stage_positions_onto_image2,
)
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    MinimapShapesOverlay,
    ShapeSpec,
)

_SELECTED = "lime"
_UNSELECTED = "cyan"
_FOV_WIDTH = 80e-6  # milling field-of-view width (m), matching the minimap markers
_FOV_ASPECT = 1024 / 1536  # height / width


@dataclass
class _Entry:
    """A working-set position: an existing lamella (to move) or a new draft."""
    name: str
    stage_position: FibsemStagePosition
    lamella: Optional[object] = None  # the existing Lamella, or None for a new draft
    moved: bool = False  # an existing lamella whose position was changed this session

    @property
    def is_new(self) -> bool:
        return self.lamella is None


class OverviewCanvas(QWidget):
    """Overview + lamella markers on the shared matplotlib ``FibsemImageCanvas``.

    Pixel<->stage uses the overview image's own metadata, so it needs a
    microscope only to project right-clicks (``microscope`` may be None ->
    read-only). The canvas downsamples large overviews for display and keeps the
    markers in a separate overlay, so add/select/move only re-draws the markers
    (not the whole image).
    """

    position_selected = pyqtSignal(object)  # index into entries, or None
    add_requested = pyqtSignal(object)       # FibsemStagePosition
    move_requested = pyqtSignal(object)      # FibsemStagePosition (for the selected entry)

    def __init__(self, image: FibsemImage, microscope=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image = image
        self.microscope = microscope
        self._entries: List[_Entry] = []
        self._selected: Optional[int] = None
        self._show_names = True
        self._show_fov = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.canvas = FibsemImageCanvas(self)
        layout.addWidget(self.canvas)

        # lamella markers (FOV rect + crosshair + name) as one overlay, so moving
        # or selecting a marker just re-sets specs — no full-image re-render.
        self._markers = MinimapShapesOverlay(zorder=6)
        self.canvas.add_overlay(self._markers)

        # names + FOV toggles (contrast / scalebar / crosshair / reset-view are
        # built into the canvas toolbar already).
        self._btn_names = self.canvas.add_toolbar_button(
            "mdi:label-outline", "Show names", self._set_show_names, checkable=True)
        self._btn_names.setChecked(True)
        self._btn_fov = self.canvas.add_toolbar_button(
            "mdi:vector-rectangle", "Show FOV", self._set_show_fov, checkable=True)
        self._btn_fov.setChecked(True)

        self.canvas.canvas_clicked.connect(self._on_click)
        self.canvas.canvas_right_clicked.connect(self._on_right_click)

        # downsampled display; metadata drives the scalebar pixel size
        self.canvas.set_image(image)

    # --- state -------------------------------------------------------------

    def set_entries(self, entries: List[_Entry]) -> None:
        self._entries = entries
        self._redraw_markers()

    def set_selected(self, index: Optional[int]) -> None:
        self._selected = index
        self._redraw_markers()

    def _set_show_names(self, show: bool) -> None:
        self._show_names = show
        self._redraw_markers()

    def _set_show_fov(self, show: bool) -> None:
        self._show_fov = show
        self._redraw_markers()

    # --- markers -----------------------------------------------------------

    def _fov_dims(self):
        """(width, height) of the milling-FOV box in pixels, or (None, None)."""
        meta = self.image.metadata
        if meta is None or meta.pixel_size is None:
            return None, None
        w = _FOV_WIDTH / meta.pixel_size.x
        return w, w * _FOV_ASPECT

    def _redraw_markers(self) -> None:
        specs: List[ShapeSpec] = []
        positions = [e.stage_position for e in self._entries]
        if positions:
            points = reproject_stage_positions_onto_image2(self.image, positions)
            fov_w, fov_h = self._fov_dims()
            for i, pt in enumerate(points):
                color = _SELECTED if i == self._selected else _UNSELECTED
                label = self._entries[i].name if self._show_names else ""
                if self._show_fov and fov_w:
                    specs.append(ShapeSpec(kind="rect", cx=pt.x, cy=pt.y,
                                           width=fov_w, height=fov_h,
                                           color=color, label=label))
                    specs.append(ShapeSpec(kind="crosshair", cx=pt.x, cy=pt.y,
                                           color=color))
                else:
                    specs.append(ShapeSpec(kind="crosshair", cx=pt.x, cy=pt.y,
                                           color=color, label=label))
        self._markers.set_shapes(specs)

    # --- interaction (canvas emits image-pixel coords) ---------------------

    def _on_click(self, x: float, y: float, _mods) -> None:
        self._select_nearest(x, y)

    def _select_nearest(self, x: float, y: float) -> None:
        if not self._entries:
            self.position_selected.emit(None)
            return
        points = reproject_stage_positions_onto_image2(
            self.image, [e.stage_position for e in self._entries])
        best, best_d = None, None
        for i, pt in enumerate(points):
            d = (pt.x - x) ** 2 + (pt.y - y) ** 2
            if best_d is None or d < best_d:
                best, best_d = i, d
        radius = max(self.image.data.shape) * 0.03  # ~3% of the image
        self.position_selected.emit(best if best_d ** 0.5 <= radius else None)

    def _on_right_click(self, x: float, y: float, _mods) -> None:
        if self.microscope is None:
            return  # read-only without a microscope to project the click
        try:
            stage_position = convert_image_coord_to_stage_position(
                self.microscope, self.image, (y, x))
        except Exception:
            return
        menu = QMenu(self)
        act_add = menu.addAction("Add lamella position here")
        act_move = menu.addAction("Move selected position here")
        act_move.setEnabled(self._selected is not None)
        chosen = menu.exec_(QCursor.pos())
        if chosen == act_add:
            self.add_requested.emit(stage_position)
        elif chosen == act_move and self._selected is not None:
            self.move_requested.emit(stage_position)


class LamellaSelectionDialog(QDialog):
    """Non-modal dialog: place lamella positions on a grid's overview image."""

    accepted_positions = pyqtSignal()  # emitted after Accept commits

    def __init__(self, experiment, grid_record, image: FibsemImage,
                 microscope=None, host=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.experiment = experiment
        self.grid_record = grid_record
        self.image = image
        self.microscope = microscope
        self.host = host  # the AutoLamellaUI, for the creation plumbing
        self.setWindowTitle(f"Select positions — {grid_record.name}")
        self.setModal(False)
        self.resize(1000, 640)

        # seed the working set from the grid's existing lamellae
        self._entries: List[_Entry] = [
            _Entry(name=lam.name, stage_position=deepcopy(lam.stage_position), lamella=lam)
            for lam in experiment.get_lamellae_for_grid(grid_record)
        ]
        self._selected: Optional[int] = None
        self._to_delete: List = []  # existing lamellae removed this session (applied on Accept)

        self._build_ui()
        self._refresh_views()

    # --- ui ----------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        body = QHBoxLayout()

        self.canvas = OverviewCanvas(self.image, self.microscope)
        self.canvas.position_selected.connect(self._on_canvas_selected)
        self.canvas.add_requested.connect(self._on_add)
        self.canvas.move_requested.connect(self._on_move)
        body.addWidget(self.canvas, 3)

        from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget
        self.list = LamellaNameListWidget()
        self.list.enable_remove_button(True)  # per-row trash (confirms on click)
        self.list.lamella_selected.connect(self._on_list_selected)
        self.list.remove_requested.connect(self._on_list_remove)

        right = QVBoxLayout()
        right.setSpacing(4)
        right.addWidget(self.list, 1)
        hint = QLabel(
            "Right-click the image to add a position, or to move the selected one.\n"
            "Click a marker or row to select · trash to delete.\n"
            "Accept saves the positions to the experiment."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(hint)
        body.addLayout(right, 1)
        root.addLayout(body, 1)  # the canvas/list row takes the vertical space

        controls = QHBoxLayout()
        if self.microscope is None:
            note = QLabel("Connect a microscope to add / move positions.")
            note.setStyleSheet("color: #c08a3e;")
            controls.addWidget(note)
        controls.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        controls.addWidget(self.btn_cancel)
        self.btn_accept = QPushButton("Accept")
        self.btn_accept.clicked.connect(self._on_accept)
        controls.addWidget(self.btn_accept)
        root.addLayout(controls)

    # --- selection sync ----------------------------------------------------

    def _on_canvas_selected(self, index) -> None:
        self._set_selected(index)

    def _on_list_selected(self, lamella) -> None:
        # map the list's selected row back to a working-set index
        idx = self.list.selected_index
        self._set_selected(idx if idx >= 0 else None)

    def _set_selected(self, index: Optional[int]) -> None:
        if index == self._selected:
            return  # already selected — avoid a canvas<->list feedback loop
        self._selected = index
        self.canvas.set_selected(index)
        if index is not None and 0 <= index < len(self._entries):
            self.list.select(self._entries[index].name)

    # --- edits -------------------------------------------------------------

    def _on_add(self, stage_position) -> None:
        name = self._next_name()
        self._entries.append(_Entry(name=name, stage_position=stage_position))
        self._refresh_views()
        self._set_selected(len(self._entries) - 1)

    def _on_move(self, stage_position) -> None:
        if self._selected is not None and 0 <= self._selected < len(self._entries):
            entry = self._entries[self._selected]
            entry.stage_position = stage_position
            entry.moved = True
            self.canvas.set_entries(self._entries)

    def _on_list_remove(self, item) -> None:
        """A list row's trash button (already confirmed) → drop that position."""
        name = getattr(item, "name", None)
        idx = next((i for i, e in enumerate(self._entries) if e.name == name), None)
        if idx is not None:
            self._delete_entry(idx)

    def _delete_entry(self, index: int) -> None:
        """Remove a working-set entry. A draft just vanishes; an existing lamella
        is queued for removal from the experiment on Accept."""
        entry = self._entries.pop(index)
        if entry.lamella is not None:
            self._to_delete.append(entry.lamella)
        self._selected = None
        self._refresh_views()
        self.canvas.set_selected(None)

    def _next_name(self) -> str:
        # petname-based, matching manually-added lamellae; offset by the drafts
        # already placed this session so they number sequentially
        pending = sum(1 for e in self._entries if e.is_new)
        return self.experiment.generate_lamella_name(offset=pending)

    def _refresh_views(self) -> None:
        self.canvas.set_entries(self._entries)
        # a lightweight stage-position list for the row widgets (name carried on it)
        positions = []
        for e in self._entries:
            sp = deepcopy(e.stage_position)
            sp.name = e.name
            positions.append(sp)
        self.list.set_lamella(positions)

    # --- commit ------------------------------------------------------------

    def _on_accept(self) -> None:
        """Commit: new entries -> add_new_lamella with the overview state; moved
        existing entries -> update their position. One refresh at the end."""
        overview_state = self.image.metadata.microscope_state if self.image.metadata else None
        for e in self._entries:
            if e.is_new:
                if self.host is not None:
                    self.host.add_new_lamella(
                        stage_position=e.stage_position,
                        name=e.name,
                        microscope_state=overview_state,
                        grid_id=self.grid_record._id,
                        notify=False,
                    )
            elif e.moved:  # only rewrite the pose of lamellae actually moved
                e.lamella.milling_pose = self._state_for(overview_state, e.stage_position)

        # apply queued deletions of existing lamellae (drafts were never added)
        for lam in self._to_delete:
            try:
                self.experiment.positions.remove(lam)
            except ValueError:
                pass

        self.experiment.save()
        if self.host is not None:
            self.host.update_lamella_combobox()
            self.host.update_ui()
            self.host.lamella_added_signal.emit()
        self.accepted_positions.emit()
        self.accept()

    @staticmethod
    def _state_for(overview_state, stage_position):
        state = deepcopy(overview_state)
        if state is not None:
            state.stage_position = deepcopy(stage_position)
        return state
