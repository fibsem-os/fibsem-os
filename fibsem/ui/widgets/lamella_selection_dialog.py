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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.tiled import (
    convert_image_coord_to_stage_position,
    reproject_stage_positions_onto_image2,
)
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.widgets.contrast_gamma_control import ContrastGammaControl

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

_BG = "#262930"
_SELECTED = "lime"
_UNSELECTED = "cyan"
_FOV_WIDTH = 80e-6  # milling field-of-view width (m), matching the minimap markers
_FOV_ASPECT = 1024 / 1536  # height / width
_ZOOM_SCALE = 1.5
_CLICK_TOL = 3  # px of drag below which a left-press counts as a click (select)


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
    """Matplotlib canvas showing the overview + lamella position markers.

    Pixel<->stage uses the overview image's own metadata, so it needs a
    microscope only to project clicks (``microscope`` may be None → read-only).
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
        self._show_crosshair = False
        self._show_scalebar = False
        self._has_view = False  # preserve zoom/pan across redraws once drawn
        # contrast / gamma popover (display-only; the raw image is never mutated)
        self._contrast = ContrastGammaControl(self)
        self._contrast.changed.connect(self._redraw)
        self._norm = ContrastGammaControl.normalize(image.data)  # cached for apply()
        # pan state
        self._pan = None  # (start_x, start_y, press_px) while a left-drag is active
        self._dragged = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self._build_toolbar())

        self.figure = Figure(figsize=(6, 6), dpi=80)
        self.figure.patch.set_facecolor(_BG)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(300, 300)
        # axes fill the whole figure so the image grows with the canvas
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._redraw()

    # --- toolbar -----------------------------------------------------------

    def _build_toolbar(self):
        from fibsem.ui.widgets.custom_widgets import IconToolButton

        bar = QHBoxLayout()
        bar.setContentsMargins(2, 2, 2, 0)
        bar.setSpacing(2)
        bar.addStretch(1)  # right-align the toolbar buttons

        def btn(icon, tip, slot, checkable=False, checked=False):
            b = IconToolButton(icon=icon, tooltip=tip, checkable=checkable, checked=checked)
            b.clicked.connect(slot) if not checkable else b.toggled.connect(slot)
            bar.addWidget(b)
            return b

        self._btn_contrast = btn(
            "mdi:contrast-box", "Contrast / gamma",
            lambda on: self._contrast.set_open(on, self._btn_contrast),
            checkable=True, checked=False)
        btn("mdi:fit-to-page-outline", "Fit view", lambda *_: self.fit_view())
        btn("mdi:crosshairs", "Show crosshair", self.set_show_crosshair,
            checkable=True, checked=self._show_crosshair)
        btn("mdi:ruler", "Show scalebar", self.set_show_scalebar,
            checkable=True, checked=self._show_scalebar)
        btn("mdi:label-outline", "Show names", self.set_show_names,
            checkable=True, checked=self._show_names)
        btn("mdi:vector-rectangle", "Show FOV", self.set_show_fov,
            checkable=True, checked=self._show_fov)
        return bar

    def fit_view(self) -> None:
        """Reset the zoom/pan to the full image extent."""
        h, w = self.image.data.shape[:2]
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)  # origin upper
        self.canvas.draw_idle()

    def set_show_crosshair(self, show: bool) -> None:
        self._show_crosshair = show
        self._redraw()

    def set_show_scalebar(self, show: bool) -> None:
        self._show_scalebar = show
        self._redraw()

    # --- contrast / gamma --------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._contrast.isVisible():
            self._contrast.reposition()

    def _display_data(self):
        """The image to show: raw at defaults, else contrast/gamma-processed."""
        if self._contrast.is_default():
            return self.image.data, None
        return self._contrast.apply(self._norm), (0.0, 1.0)

    # --- state -------------------------------------------------------------

    def set_entries(self, entries: List[_Entry]) -> None:
        self._entries = entries
        self._redraw()

    def set_selected(self, index: Optional[int]) -> None:
        self._selected = index
        self._redraw()

    def set_show_names(self, show: bool) -> None:
        self._show_names = show
        self._redraw()

    def set_show_fov(self, show: bool) -> None:
        self._show_fov = show
        self._redraw()

    # --- drawing -----------------------------------------------------------

    def _redraw(self) -> None:
        view = (self.ax.get_xlim(), self.ax.get_ylim()) if self._has_view else None
        self.ax.clear()
        self.ax.set_facecolor(_BG)
        data, clim = self._display_data()
        im = self.ax.imshow(data, cmap="gray", origin="upper")
        if clim is not None:
            im.set_clim(*clim)
        self.ax.axis("off")

        if self._show_crosshair:
            self._draw_crosshair()
        if self._show_scalebar:
            self._draw_scalebar()

        positions = [e.stage_position for e in self._entries]
        if positions:
            points = reproject_stage_positions_onto_image2(self.image, positions)
            for i, pt in enumerate(points):
                color = _SELECTED if i == self._selected else _UNSELECTED
                if self._show_fov:
                    rect = self._fov_rect(pt.x, pt.y, color)
                    if rect is not None:
                        self.ax.add_patch(rect)
                self.ax.plot(pt.x, pt.y, marker="+", ms=15, c=color,
                             markeredgewidth=2, alpha=0.7)
                if self._show_names:
                    self.ax.text(pt.x + 10, pt.y - 10, self._entries[i].name,
                                 fontsize=8, color=color, alpha=0.75)

        if view is not None:  # restore the user's zoom/pan
            self.ax.set_xlim(view[0])
            self.ax.set_ylim(view[1])
        self._has_view = True
        self.canvas.draw_idle()

    def _fov_rect(self, px: float, py: float, color: str):
        """A dashed milling-FOV rectangle centred on a position (pixels)."""
        from matplotlib.patches import Rectangle
        meta = self.image.metadata
        if meta is None or meta.pixel_size is None:
            return None
        w = _FOV_WIDTH / meta.pixel_size.x
        h = w * _FOV_ASPECT
        return Rectangle((px - w / 2, py - h / 2), w, h, color=color,
                         fill=False, linewidth=1.5, linestyle="--", alpha=0.7)

    def _draw_crosshair(self) -> None:
        """Yellow crosshair at the image centre (matches the image-canvas)."""
        h, w = self.image.data.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        kw = dict(color="yellow", linewidth=1, alpha=0.8, zorder=7)
        self.ax.plot([cx - w * 0.025, cx + w * 0.025], [cy, cy], **kw)
        self.ax.plot([cx, cx], [cy - h * 0.025, cy + h * 0.025], **kw)

    def _draw_scalebar(self) -> None:
        meta = self.image.metadata
        if meta is None or meta.pixel_size is None:
            return
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            self.ax.add_artist(ScaleBar(
                dx=meta.pixel_size.x, color="black", box_color="white",
                box_alpha=0.5, location="lower right"))
        except Exception:
            pass

    # --- interaction (left-drag = pan, left-click = select, wheel = zoom) ---

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 1:
            self._pan = (event.xdata, event.ydata, (event.x, event.y))
            self._dragged = False
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button == 3:  # right-click: add / move menu
            self._show_menu(event)

    def _on_move(self, event) -> None:
        if self._pan is None or event.inaxes != self.ax or event.xdata is None:
            return
        # past the click tolerance → it's a pan, not a select
        if abs(event.x - self._pan[2][0]) > _CLICK_TOL or abs(event.y - self._pan[2][1]) > _CLICK_TOL:
            self._dragged = True
        cur_x, cur_y = self.ax.get_xlim(), self.ax.get_ylim()
        dx, dy = event.xdata - self._pan[0], event.ydata - self._pan[1]
        self.ax.set_xlim(cur_x[0] - dx, cur_x[1] - dx)
        self.ax.set_ylim(cur_y[0] - dy, cur_y[1] - dy)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._pan is None or event.button != 1:
            return
        was_click = not self._dragged
        start = self._pan
        self._pan = None
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        if was_click and event.inaxes == self.ax and event.xdata is not None:
            self._select_nearest(event.xdata, event.ydata)

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = (1 / _ZOOM_SCALE) if event.button == "up" else _ZOOM_SCALE
        cur_x, cur_y = self.ax.get_xlim(), self.ax.get_ylim()
        relx = (cur_x[1] - event.xdata) / (cur_x[1] - cur_x[0])
        rely = (cur_y[1] - event.ydata) / (cur_y[1] - cur_y[0])
        nw = (cur_x[1] - cur_x[0]) * scale
        nh = (cur_y[1] - cur_y[0]) * scale
        self.ax.set_xlim(event.xdata - nw * (1 - relx), event.xdata + nw * relx)
        self.ax.set_ylim(event.ydata - nh * (1 - rely), event.ydata + nh * rely)
        self.canvas.draw_idle()

    def _select_nearest(self, x: float, y: float) -> None:
        if not self._entries:
            self.position_selected.emit(None)
            return
        points = reproject_stage_positions_onto_image2(
            self.image, [e.stage_position for e in self._entries])
        # nearest within a reasonable pixel radius, else clear
        best, best_d = None, None
        for i, pt in enumerate(points):
            d = (pt.x - x) ** 2 + (pt.y - y) ** 2
            if best_d is None or d < best_d:
                best, best_d = i, d
        radius = max(self.image.data.shape) * 0.03  # ~3% of the image
        self.position_selected.emit(best if best_d ** 0.5 <= radius else None)

    def _show_menu(self, event) -> None:
        if self.microscope is None:
            return  # read-only without a microscope to project the click
        try:
            stage_position = convert_image_coord_to_stage_position(
                self.microscope, self.image, (event.ydata, event.xdata))
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
