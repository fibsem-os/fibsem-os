from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import BeamType, Point
from fibsem.ui.widgets.canvas_state import PointsSpec
from fibsem.ui.widgets.custom_widgets import IconToolButton
from fibsem.applications.autolamella.workflows.tasks.tasks import SpotBurnFiducialTaskConfig


_HEADER_BG = "#1e2124"
_MUTED = "#8a9099"


class _SpotBurnRow(QWidget):
    """A single read-only coordinate row: index + X + Y + a remove button."""

    remove_clicked = pyqtSignal(int)

    def __init__(self, index: int, point: Point, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.index = index
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        idx_lbl = QLabel(str(index + 1))
        idx_lbl.setFixedWidth(20)
        idx_lbl.setAlignment(Qt.AlignCenter)
        idx_lbl.setStyleSheet(f"color: {_MUTED}; background: transparent;")
        layout.addWidget(idx_lbl)

        x_lbl = QLabel(f"{point.x:.3f}")
        x_lbl.setStyleSheet("background: transparent; font-family: monospace;")
        layout.addWidget(x_lbl, stretch=1)

        y_lbl = QLabel(f"{point.y:.3f}")
        y_lbl.setStyleSheet("background: transparent; font-family: monospace;")
        layout.addWidget(y_lbl, stretch=1)

        btn_remove = IconToolButton(
            "mdi:trash-can-outline", tooltip="Remove coordinate", size=24
        )
        btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.index))
        layout.addWidget(btn_remove)


class AutoLamellaSpotBurnCoordinatesWidget(QWidget):
    """Editor for spot-burn coordinates, backed by a canvas points overlay.

    A titled list of read-only coordinate rows (index + x, y in relative 0-1 image
    space), each with a remove button, plus an add button in the header — styled to
    match the app's task / lamella lists. Coordinates are *placed and moved on the
    image* (right-click to add, drag to move, Delete to remove); the list mirrors the
    overlay and its selection both ways, and the on-image markers are numbered to match
    the rows.

    Reusable by any host that owns a ``MicroscopeViewController`` (the protocol editor
    now; the live spot-burn widget in a later slice).
    """

    settings_changed = pyqtSignal(SpotBurnFiducialTaskConfig)
    OVERLAY_ID = "spot_burn"

    def __init__(self,
                 controller,
                 beam: BeamType = BeamType.ION,
                 config: Optional[SpotBurnFiducialTaskConfig] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self.beam = beam
        self.config = config
        self._coordinates: List[Point] = list(config.coordinates) if config else []
        self._image_shape: Optional[Tuple[int, int]] = None  # (h, w) for 0-1 <-> px
        self._updating = False  # guard against re-entrant list/overlay updates
        self._active = False    # overlay armed + shown while the widget is visible
        self._wired = False     # subscribed to controller signals

        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # header: title + add button (matches TaskNameListWidget / LamellaNameListWidget)
        header = QWidget()
        header.setStyleSheet(f"background: {_HEADER_BG};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 3, 4, 3)
        hl.setSpacing(4)
        title = QLabel("Spot Burn Coordinates")
        title.setStyleSheet("font-weight: bold; background: transparent;")
        hl.addWidget(title)
        hl.addStretch()
        self.btn_add = IconToolButton("mdi:plus", tooltip="Add coordinate", size=24)
        self.btn_add.clicked.connect(self._add_coordinate)
        hl.addWidget(self.btn_add)
        outer.addWidget(header)

        # column header
        col = QWidget()
        cl = QHBoxLayout(col)
        cl.setContentsMargins(6, 2, 6, 2)
        cl.setSpacing(8)
        lbl_idx = QLabel("#")
        lbl_idx.setFixedWidth(20)
        lbl_idx.setAlignment(Qt.AlignCenter)
        lbl_x = QLabel("X (0-1)")
        lbl_y = QLabel("Y (0-1)")
        for lbl in (lbl_idx, lbl_x, lbl_y):
            lbl.setStyleSheet(f"color: {_MUTED}; background: transparent; font-size: 11px;")
        cl.addWidget(lbl_idx)
        cl.addWidget(lbl_x, stretch=1)
        cl.addWidget(lbl_y, stretch=1)
        cl.addSpacing(24)  # align with the per-row remove button
        outer.addWidget(col)

        # rows
        self._list = QListWidget()
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setMaximumHeight(180)
        self._list.itemSelectionChanged.connect(self._on_row_selection_changed)
        outer.addWidget(self._list)

        # footer summary + hint
        self.label_summary = QLabel()
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet(f"color: {_MUTED}; padding: 4px 6px;")
        outer.addWidget(self.label_summary)

        self._rebuild_rows()

    # --- public API ---

    def set_image_shape(self, shape) -> None:
        """Set the host FIB image shape (h, w), used for 0-1 <-> pixel conversion."""
        self._image_shape = (int(shape[0]), int(shape[1])) if shape is not None else None
        self._sync_overlay()

    def set_task_config(self, config: SpotBurnFiducialTaskConfig):
        """Set the config and update the rows + overlay."""
        self.config = config
        self._coordinates = list(config.coordinates)
        self._sync_overlay()
        self._rebuild_rows()

    def get_task_config(self) -> Optional[SpotBurnFiducialTaskConfig]:
        """Read the current coordinates back into the config."""
        if self.config is None:
            return None
        self.config.coordinates = list(self._coordinates)
        return self.config

    # --- rows ---

    def _rebuild_rows(self):
        self._updating = True
        self._list.clear()
        for i, pt in enumerate(self._coordinates):
            row = _SpotBurnRow(i, pt)
            row.remove_clicked.connect(self._remove_coordinate)
            item = QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._updating = False
        self._update_summary()

    def _add_coordinate(self):
        """Add a coordinate at the image centre; the user then drags it into place."""
        self._coordinates.append(Point(0.5, 0.5))
        self._sync_overlay()
        self._rebuild_rows()
        self._emit_settings_changed()

    def _remove_coordinate(self, index: int):
        if 0 <= index < len(self._coordinates):
            self._coordinates.pop(index)
            self._sync_overlay()
            self._rebuild_rows()
            self._emit_settings_changed()

    # --- overlay sync ---

    def _points_spec(self, px_points) -> PointsSpec:
        return PointsSpec(
            id=self.OVERLAY_ID,
            points=px_points,
            color="white",
            selected_color="cyan",
            marker="o",
            size=12,
            add_on_right_click=True,
            removable=True,
            modal=True,
            numbered=True,
        )

    def _sync_overlay(self):
        """Push the coordinates onto the canvas overlay (relative -> pixels)."""
        if not self._active or self._image_shape is None:
            return
        h, w = self._image_shape
        px = [(pt.x * w, pt.y * h) for pt in self._coordinates]
        self.controller.set_overlay(self.beam, self._points_spec(px))

    def _on_overlay_edited(self, beam, overlay_id, points):
        """A point was added / moved / removed on the canvas -> refresh the rows."""
        if beam != self.beam or overlay_id != self.OVERLAY_ID or self._image_shape is None:
            return
        h, w = self._image_shape
        # overlay already reflects the edit (incl. renumbering); just mirror it here
        self._coordinates = [Point(float(x / w), float(y / h)) for (x, y) in points]
        self._rebuild_rows()
        self._emit_settings_changed()

    # --- selection sync (list <-> overlay) ---

    def _on_row_selection_changed(self):
        """A row was selected -> highlight the matching point on the canvas."""
        if self._updating or not self._active:
            return
        row = self._list.currentRow()
        self.controller.set_selected_point(
            self.beam, self.OVERLAY_ID, row if row >= 0 else None
        )

    def _on_overlay_point_selected(self, beam, overlay_id, index):
        """A point was selected on the canvas -> select the matching row."""
        if beam != self.beam or overlay_id != self.OVERLAY_ID:
            return
        if 0 <= index < self._list.count():
            self._updating = True
            self._list.setCurrentRow(index)
            self._updating = False

    # --- misc ---

    def _emit_settings_changed(self):
        config = self.get_task_config()
        if config is not None:
            self.settings_changed.emit(config)

    def _update_summary(self):
        n = len(self._coordinates)
        if n == 0:
            self.label_summary.setText(
                "No coordinates defined.  ·  right-click the image to add a burn point."
            )
        else:
            self.label_summary.setText(
                f"{n} coordinate{'s' if n != 1 else ''}  ·  drag to move, Delete to remove."
            )

    # --- activation (arming) driven by visibility ---

    def _set_active(self, active: bool):
        if active == self._active:
            if active:
                self._sync_overlay()  # refresh on re-show
            return
        self._active = active
        if active:
            if not self._wired:
                self.controller.overlay_edited.connect(self._on_overlay_edited)
                self.controller.overlay_point_selected.connect(
                    self._on_overlay_point_selected
                )
                self._wired = True
            self._sync_overlay()
            self.controller.arm_overlay(
                self.beam,
                self.OVERLAY_ID,
                label="Spot Burn",
                icon="mdi:record-circle-outline",
            )
        else:
            self.controller.arm_overlay(self.beam, None)
            self.controller.remove_overlay(self.beam, self.OVERLAY_ID)

    def showEvent(self, event):
        super().showEvent(event)
        self._set_active(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._set_active(False)

    def closeEvent(self, event):
        if self._wired:
            for sig, slot in (
                (self.controller.overlay_edited, self._on_overlay_edited),
                (self.controller.overlay_point_selected, self._on_overlay_point_selected),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._wired = False
        super().closeEvent(event)
