from typing import List, Optional, Tuple

from PyQt5.QtCore import QItemSelectionModel, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.spot import SpotBurnSettings
from fibsem.structures import BeamType, Point
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import CANVAS_BG, GRAY_ICON_COLOR
from fibsem.ui.tokens import (
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.canvas.canvas_state import PointsSpec
from fibsem.ui.widgets.custom_widgets import IconToolButton

_HEADER_BG = CANVAS_BG
_MUTED = TEXT_MUTED_COLOR


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
            "mdi:trash-can-outline",
            tooltip="Remove coordinate (the whole selection, if this row is in it)",
            size=24,
        )
        btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.index))
        layout.addWidget(btn_remove)


class SpotBurnCoordinatesWidget(QWidget):
    """Editor for spot-burn coordinates, backed by a canvas points overlay.

    A titled list of read-only coordinate rows (index + x, y in relative 0-1 image
    space), each with a remove button, plus an add button in the header — styled to
    match the app's task / lamella lists. Coordinates are *placed and moved on the
    image* (right-click to add, drag to move, Delete to remove); the list mirrors the
    overlay and its selection both ways, and the on-image markers are numbered to match
    the rows. Several points can be selected at once — Ctrl/Shift-click in the list or
    on the image, or Shift-drag a box on the image — then moved or removed together.

    Works with a :class:`SpotBurnSettings` payload — it edits ``.coordinates`` and passes
    current/exposure through untouched. Reusable by any host that owns a
    ``MicroscopeViewController`` — the protocol editor and the live spot-burn widget.
    """

    settings_changed = pyqtSignal(SpotBurnSettings)
    OVERLAY_ID = "spot_burn"

    def __init__(
        self,
        controller,
        beam: BeamType = BeamType.ION,
        settings: Optional[SpotBurnSettings] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.controller = controller
        self.beam = beam
        self.settings = settings if settings is not None else SpotBurnSettings()
        self._coordinates: List[Point] = list(self.settings.coordinates)
        self._image_shape: Optional[Tuple[int, int]] = None  # (h, w) for 0-1 <-> px
        # selected rows, ascending. Kept here rather than read off the list because the
        # list is rebuilt on every edit, which would otherwise drop it
        self._selection: List[int] = []
        self._updating = False  # guard against re-entrant list/overlay updates
        self._active = False  # overlay armed + shown while the widget is visible
        self._wired = False  # subscribed to controller signals

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
        # one toggle, drawn as the selection: empty (none), minus (some), ticked (all).
        # Ticked clears, the others select all. Removing the selection is the Delete
        # key, in the list or on the image, or a selected row's trash button.
        self.btn_select_all = IconToolButton(
            "mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            tooltip="Select all coordinates",
            checked_tooltip="Clear the selection",
            size=24,
        )
        self.btn_select_all.clicked.connect(self._toggle_select_all)
        hl.addWidget(self.btn_select_all)
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
            lbl.setStyleSheet(
                f"color: {_MUTED}; background: transparent; font-size: 11px;"
            )
        cl.addWidget(lbl_idx)
        cl.addWidget(lbl_x, stretch=1)
        cl.addWidget(lbl_y, stretch=1)
        cl.addSpacing(24)  # align with the per-row remove button
        outer.addWidget(col)

        # rows — the only part that should absorb spare height. It used to be capped at
        # 180px with no stretch, which scrolled a six-row window while the panel below it
        # sat empty; a fiducial pattern is a dozen-plus points, so the cap bit immediately.
        self._list = QListWidget()
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setMinimumHeight(120)  # still readable in a short host
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_row_selection_changed)
        # Delete is Backspace on a Mac keyboard
        for key in (Qt.Key_Delete, Qt.Key_Backspace):
            shortcut = QShortcut(QKeySequence(key), self._list)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._remove_selected)
        outer.addWidget(self._list, 1)

        # footer summary + hint
        self.label_summary = QLabel()
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet(f"color: {_MUTED}; padding: 4px 6px;")
        outer.addWidget(self.label_summary, 0)

        self._rebuild_rows()

    # --- public API ---

    def set_image_shape(self, shape) -> None:
        """Set the host FIB image shape (h, w), used for 0-1 <-> pixel conversion."""
        self._image_shape = (
            (int(shape[0]), int(shape[1])) if shape is not None else None
        )
        self._sync_overlay()

    def set_settings(self, settings: SpotBurnSettings):
        """Set the settings payload and update the rows + overlay."""
        self.settings = settings
        self._coordinates = list(settings.coordinates)
        self._selection = []
        self._sync_overlay()
        self._rebuild_rows()

    def get_settings(self) -> SpotBurnSettings:
        """Read the current coordinates back into the settings payload."""
        self.settings.coordinates = list(self._coordinates)
        return self.settings

    # --- rows ---

    def _rebuild_rows(self):
        self._updating = True
        # QListWidget.clear() drops the items but leaves the setItemWidget row widgets
        # parented to the viewport. This runs on every edit — including each drag-release
        # on the canvas — so without an explicit delete they accumulate for the lifetime
        # of the widget (measured: 24 row widgets for a single coordinate after 21 edits).
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if row is not None:
                row.setParent(None)
                row.deleteLater()
        self._list.clear()
        for i, pt in enumerate(self._coordinates):
            row = _SpotBurnRow(i, pt)
            row.remove_clicked.connect(self._remove_coordinate)
            item = QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._updating = False
        self._show_selection()

    def _add_coordinate(self):
        """Add a coordinate at the image centre, selected, so the user can drag it into
        place (as a right-click on the image does)."""
        self._coordinates.append(Point(0.5, 0.5))
        self._selection = [len(self._coordinates) - 1]
        self._sync_overlay()
        self._rebuild_rows()
        self._emit_settings_changed()

    def _remove_coordinate(self, index: int):
        """A row's trash button. On a selected row it removes the whole selection, as
        a file manager does; on any other row just that row, so a stray click cannot
        take a selection made elsewhere with it."""
        if index in self._selection:
            self._remove_rows(self._selection)
        elif 0 <= index < len(self._coordinates):
            self._remove_rows([index])

    def _remove_selected(self):
        if self._selection:
            self._remove_rows(self._selection)

    def _remove_rows(self, rows) -> None:
        gone = set(rows)
        self._coordinates = [
            p for i, p in enumerate(self._coordinates) if i not in gone
        ]
        self._selection = [
            i - sum(1 for g in gone if g < i) for i in self._selection if i not in gone
        ]
        self._sync_overlay()
        self._rebuild_rows()
        self._emit_settings_changed()

    # --- overlay sync ---

    def _points_spec(self, px_points) -> PointsSpec:
        return PointsSpec(
            id=self.OVERLAY_ID,
            points=px_points,
            color="cyan",
            selected_color="lime",
            marker="o",
            # markersize, in points. Deliberately small: fiducial patterns put spots
            # ~0.01 of the frame apart, and a larger marker merges the cluster into one
            # blob at full-frame zoom. Picking is unaffected — PointOverlay hit-tests
            # against a fixed screen-space radius, not the marker size — and the
            # selected point still stands out at size * 1.4.
            size=6,
            add_on_right_click=True,
            removable=True,
            modal=True,
            numbered=True,
            multi_select=True,
            selection=tuple(self._selection),
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
        if (
            beam != self.beam
            or overlay_id != self.OVERLAY_ID
            or self._image_shape is None
        ):
            return
        h, w = self._image_shape
        # overlay already reflects the edit (incl. renumbering); just mirror it here.
        # An add or a delete also changed which indices are selected, and the model
        # already holds the post-edit selection.
        self._coordinates = [Point(float(x / w), float(y / h)) for (x, y) in points]
        self._selection = self._valid(
            self.controller.overlay_selection(self.beam, self.OVERLAY_ID)
        )
        self._rebuild_rows()
        self._emit_settings_changed()

    # --- selection sync (list <-> overlay) ---

    def _valid(self, indices) -> List[int]:
        return sorted({i for i in indices if 0 <= i < len(self._coordinates)})

    def _on_row_selection_changed(self):
        """Rows were (de)selected -> select the matching points on the canvas."""
        if self._updating:
            return
        self._selection = sorted(
            self._list.row(it) for it in self._list.selectedItems()
        )
        self._update_summary()
        if self._active:
            self.controller.set_selected_points(
                self.beam, self.OVERLAY_ID, self._selection
            )

    def _on_overlay_selection_changed(self, beam, overlay_id, indices):
        """Points were (de)selected on the canvas -> select the matching rows."""
        if beam != self.beam or overlay_id != self.OVERLAY_ID:
            return
        self._selection = self._valid(indices)
        self._show_selection()

    def _show_selection(self) -> None:
        """Make the list's selection match ``_selection``, without echoing it back."""
        self._selection = self._valid(self._selection)
        self._updating = True
        try:
            self._list.clearSelection()
            for i in self._selection:
                self._list.item(i).setSelected(True)
            if self._selection:
                # the anchor for the next Shift-click, and into view for a long list
                last = self._list.item(self._selection[-1])
                self._list.setCurrentItem(last, QItemSelectionModel.NoUpdate)
                self._list.scrollToItem(last)
        finally:
            self._updating = False
        self._update_summary()

    # --- misc ---

    def _emit_settings_changed(self):
        self.settings_changed.emit(self.get_settings())

    def _toggle_select_all(self):
        n = len(self._coordinates)
        if n and len(self._selection) == n:
            self._list.clearSelection()
        else:
            self._list.selectAll()
        self._update_summary()  # a click on an already-matching state changes nothing

    def _update_summary(self):
        n = len(self._coordinates)
        k = len(self._selection)
        # the button shows the selection, not its own click: Qt flips a checkable
        # button on every click, so set it back to what is true
        all_selected = n > 0 and k == n
        self.btn_select_all.blockSignals(True)
        self.btn_select_all.setChecked(all_selected)
        self.btn_select_all.blockSignals(False)
        self.btn_select_all.set_icon_state(all_selected)
        if 0 < k < n:  # the third state, which the two-state button does not draw
            self.btn_select_all.setIcon(
                fibsem_icon("mdi:minus-box-outline", color=GRAY_ICON_COLOR)
            )
        self.btn_select_all.setEnabled(n > 0)
        if n == 0:
            self.label_summary.setText(
                "No coordinates defined.  ·  right-click the image to add a burn point."
            )
        else:
            selected = f"  ·  {k} selected" if k else ""
            self.label_summary.setText(
                f"{n} coordinate{'s' if n != 1 else ''}{selected}  ·  drag to move, "
                "Shift-drag to box-select, Delete to remove."
            )

    # --- activation (arming) driven by visibility ---

    def set_active(self, active: bool) -> None:
        """Explicitly arm/disarm the overlay — for hosts that drive activation directly
        (e.g. the live spot-burn tab) rather than relying on show/hide. Idempotent with
        the show/hide path."""
        self._set_active(active)

    def _set_active(self, active: bool):
        if active == self._active:
            if active:
                self._sync_overlay()  # refresh on re-show
            return
        self._active = active
        if active:
            if not self._wired:
                self.controller.overlay_edited.connect(self._on_overlay_edited)
                self.controller.overlay_selection_changed.connect(
                    self._on_overlay_selection_changed
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
            # Deactivation runs on teardown too (hideEvent), and Qt gives no ordering
            # guarantee between this widget and the controller. If the controller's C++
            # object is already gone, touching it raises RuntimeError — there is nothing
            # left to disarm, so treat it as already torn down. closeEvent guards its
            # disconnects the same way.
            try:
                self.controller.arm_overlay(self.beam, None)
                self.controller.remove_overlay(self.beam, self.OVERLAY_ID)
            except RuntimeError:
                pass

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
                (
                    self.controller.overlay_selection_changed,
                    self._on_overlay_selection_changed,
                ),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._wired = False
        super().closeEvent(event)
