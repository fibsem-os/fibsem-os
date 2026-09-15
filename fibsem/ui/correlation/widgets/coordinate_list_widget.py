"""CoordinateListWidget -- reorderable list of Coordinate objects.

Each row is a colour dot, the auto-generated name, x/y/z fields and a state
word when the point is not simply placed. Fit and remove appear on the
selected row; the full set of actions is on the row's context menu.

Operates on a flat List[Coordinate]. Callers are responsible for
flattening/reconstructing CorrelationInputData.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import (
    DRAG_HANDLE_HEIGHT,
    DRAG_HANDLE_WIDTH,
    drag_handle_pixmap,
    fibsem_icon,
)
from fibsem.ui.tokens import (
    CAPTION_STYLE,
    CAPTION_VALUE_STYLE,
    NUMBER_STYLE,
    ROW_ALT_COLOR,
    WARN_COLOR,
    state_style,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton, ValueSpinBox

_NAME_FIXED_WIDTH = 100
_SPIN_FIXED_WIDTH = 62
_BTN_SIZE = QSize(24, 24)
_ROW_HEIGHT = 28
# A list shows every row up to this many, then scrolls. Sized to content rather
# than left at Qt's default height, which showed seven rows and hid the eighth
# fiducial under the next panel's header with only the count to say so
# (FIB-978).
_MAX_VISIBLE_ROWS = 12


_POINT_TYPE_COLORS: Dict[PointType, str] = {
    PointType.FIB: "lime",
    PointType.FM: "cyan",
    PointType.POI: "magenta",
    PointType.SURFACE: "red",
    PointType.SURFACE_FM: "yellow",
}


def _make_color_icon(color_name: str, size: int = 16) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor(color_name))
    return QIcon(px)


def _generate_names(coordinates: List[Coordinate]) -> List[str]:
    """Return auto-generated names: 'FIB 1', 'FIB 2', 'FM 1', etc."""
    counters: Dict[PointType, int] = {}
    names = []
    for c in coordinates:
        counters[c.point_type] = counters.get(c.point_type, 0) + 1
        names.append(f"{c.point_type.value} {counters[c.point_type]}")
    return names


def _name_col_width(point_type: Optional[PointType]) -> int:
    """Width of the Name column, sized to the longest name this list can show.

    Names are ``"{point_type} {n}"`` and each list holds a single point type, so
    a fiducial list ("FM 8") needs far less room than a surface ("FM-SURFACE 1").
    Sizing per list removes the wide empty gap that a fixed 100px left after the
    short FIB/FM/POI names. Capped at that former width so surface lists — the
    only ones that filled it — are unchanged.
    """
    label = f"{point_type.value if point_type else 'POINT'} 99"
    font = QApplication.font()  # the app-default family the label inherits...
    font.setPixelSize(11)  # ...at the row/name label font-size
    text_w = QFontMetrics(font).horizontalAdvance(label)
    return max(48, min(text_w + 14, _NAME_FIXED_WIDTH))  # +padding, clamped


# ---------------------------------------------------------------------------
# Draggable list
# ---------------------------------------------------------------------------


class _DraggableCoordinateList(QListWidget):
    """QListWidget with InternalMove drag-and-drop.

    Qt clears itemWidget on move, so the parent must listen to ``reordered``
    and rebuild row widgets after each drop.
    """

    reordered = pyqtSignal(list)  # List[Coordinate]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        coords = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i) is not None
            and self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(coords)


# ---------------------------------------------------------------------------
# Row widget
# ---------------------------------------------------------------------------


def state_text(coord: Coordinate) -> str:
    """The one word a row shows for a point that is not simply placed (FIB-978).

    Empty for a placed point: the default state carries no chrome. The
    vocabulary is :class:`PointStatus`; ``suggested`` is the highlight on the
    predictions worth dragging first.
    """
    status = getattr(coord, "status", "")
    if status == PointStatus.PREDICTED:
        return (
            "predicted \u00b7 start here"
            if getattr(coord, "suggested", False)
            else "predicted"
        )
    if status == PointStatus.FITTED or (
        status == "" and getattr(coord, "fitted", False)
    ):
        return "fitted"
    if status == PointStatus.ACCEPTED:
        return "accepted"
    if status == PointStatus.REJECTED:
        return "removed from fit"
    return ""


def _state_tone(coord: Coordinate) -> str:
    status = getattr(coord, "status", "")
    if status == PointStatus.PREDICTED:
        return "warn" if getattr(coord, "suggested", False) else "muted"
    if status == PointStatus.FITTED or (
        status == "" and getattr(coord, "fitted", False)
    ):
        return "ok"
    return "muted"


class CoordinateRowWidget(QWidget):
    """One point: colour dot, name, x y z, and a state word when there is one.

    Nothing else at rest. The actions (fit, remove) appear only while the row
    is selected; the drag handle only under the pointer; the full set (fit,
    reset to prediction, reject from fit, remove) is on the right-click menu
    (FIB-978 \u00a77).
    """

    row_clicked = pyqtSignal(object)  # Coordinate
    coordinate_changed = pyqtSignal(object, str, float)  # Coordinate, field, value
    remove_clicked = pyqtSignal(object)  # Coordinate
    fit_clicked = pyqtSignal(object)  # Coordinate
    reset_clicked = pyqtSignal(object)  # Coordinate: back to its prediction
    reject_toggled = pyqtSignal(object)  # Coordinate: in / out of the fit

    def __init__(
        self,
        coord: Coordinate,
        name: str,
        parent: Optional[QWidget] = None,
        name_width: int = _NAME_FIXED_WIDTH,
    ) -> None:
        super().__init__(parent)
        self.coord = coord
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(6)

        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        layout.addWidget(self.dot)

        self.name_label = QLabel(name)
        self.name_label.setFixedWidth(name_width)
        # Omit `background: transparent` -- it is the QLabel default, and an
        # unscoped rule bleeds into this label's QToolTip background.
        self.name_label.setStyleSheet(CAPTION_VALUE_STYLE)
        self.name_label.setToolTip("Auto-generated coordinate name")
        layout.addWidget(self.name_label)

        # One decimal: a pixel to a thousandth is noise and widens every field.
        self.x_spin = ValueSpinBox(
            decimals=1, minimum=-1e6, maximum=1e6, step=0.1, no_buttons=True
        )
        self.y_spin = ValueSpinBox(
            decimals=1, minimum=-1e6, maximum=1e6, step=0.1, no_buttons=True
        )
        self.z_spin = ValueSpinBox(
            decimals=1, minimum=-1e6, maximum=1e6, step=0.1, no_buttons=True
        )
        for spin, tip in (
            (self.x_spin, "X (px)"),
            (self.y_spin, "Y (px)"),
            (self.z_spin, "Z (slice)"),
        ):
            spin.setFixedWidth(_SPIN_FIXED_WIDTH)
            spin.setToolTip(tip)
            spin.setStyleSheet(f"{NUMBER_STYLE} padding: 1px 4px;")
            layout.addWidget(spin)

        self.state_label = QLabel("")
        layout.addWidget(self.state_label, stretch=1)

        # Actions: shown only while selected
        self.actions = QWidget()
        actions_layout = QHBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)
        self.btn_fit = IconToolButton(
            icon="mdi:target",
            tooltip="Fit this point to the image (F)",
            size=_BTN_SIZE.width(),
        )
        self.btn_remove = IconToolButton(
            icon="mdi:close",
            tooltip="Remove this point (Delete)",
            size=_BTN_SIZE.width(),
        )
        actions_layout.addWidget(self.btn_fit)
        actions_layout.addWidget(self.btn_remove)
        self.actions.setVisible(False)
        layout.addWidget(self.actions)

        # Drag handle: shown only under the pointer
        self.drag_icon = QLabel()
        self.drag_icon.setFixedSize(DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)
        self.drag_icon.setPixmap(drag_handle_pixmap())
        self.drag_icon.setStyleSheet("background: transparent;")
        self.drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        self.drag_icon.setVisible(False)
        layout.addWidget(self.drag_icon)

        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.installEventFilter(self)

        self._selected = False
        self._connect_signals()
        self.refresh()

    def _connect_signals(self) -> None:
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.coord))
        self.btn_fit.clicked.connect(lambda: self.fit_clicked.emit(self.coord))
        self.x_spin.editingFinished.connect(self._on_x_changed)
        self.y_spin.editingFinished.connect(self._on_y_changed)
        self.z_spin.editingFinished.connect(self._on_z_changed)

    def mousePressEvent(self, event) -> None:
        self.row_clicked.emit(self.coord)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FocusIn:
            self.row_clicked.emit(self.coord)
        return super().eventFilter(obj, event)

    def enterEvent(self, event) -> None:
        self.drag_icon.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.drag_icon.setVisible(False)
        super().leaveEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.row_clicked.emit(self.coord)
        menu = QMenu(self)
        menu.addAction("Fit to image\tF", lambda: self.fit_clicked.emit(self.coord))
        reset = menu.addAction(
            "Reset to prediction", lambda: self.reset_clicked.emit(self.coord)
        )
        reset.setEnabled(
            getattr(self.coord, "provenance", "") == PointProvenance.PROJECTED
            and getattr(self.coord, "status", "") != PointStatus.PREDICTED
        )
        rejected = getattr(self.coord, "status", "") == PointStatus.REJECTED
        menu.addAction(
            "Restore to fit" if rejected else "Reject from fit",
            lambda: self.reject_toggled.emit(self.coord),
        )
        menu.addSeparator()
        menu.addAction("Remove\tDel", lambda: self.remove_clicked.emit(self.coord))
        menu.exec_(event.globalPos())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        self.name_label.setText(name)

    def set_selected(self, selected: bool) -> None:
        """Show the row's actions only while it is the selected row, and paint
        the row's own selected background (the list's item highlight is off)."""
        self._selected = selected
        self.actions.setVisible(selected)
        self.setAutoFillBackground(selected)
        if selected:
            palette = self.palette()
            palette.setColor(self.backgroundRole(), QColor(ROW_ALT_COLOR))
            self.setPalette(palette)

    def set_axis_maxima(
        self,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> None:
        """Constrain spinbox ranges to image shape. Pass None to leave unconstrained."""
        self._axis_max = (x_max, y_max)
        if x_max is not None:
            self.x_spin.setMinimum(0.0)
            self.x_spin.setMaximum(float(x_max))
        if y_max is not None:
            self.y_spin.setMinimum(0.0)
            self.y_spin.setMaximum(float(y_max))
        if z_max is not None:
            self.z_spin.setMinimum(0.0)
            self.z_spin.setMaximum(float(z_max))
        self.refresh()

    def _off_image(self) -> bool:
        """Outside the image's axes: only a projection can put a point there."""
        x_max, y_max = getattr(self, "_axis_max", (None, None))
        p = self.coord.point
        return (x_max is not None and not 0.0 <= p.x <= x_max) or (
            y_max is not None and not 0.0 <= p.y <= y_max
        )

    def refresh(self) -> None:
        """Re-sync the fields and the state word from the coordinate, silently."""
        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.blockSignals(True)
        # A typed value is held to the image. A projection can land outside
        # it; the row then shows the true number, read-only, rather than the
        # clamped 0.0 it used to show. The user drags the ring in from the
        # canvas or projects again after a drop.
        off = self._off_image()
        x_max, y_max = getattr(self, "_axis_max", (None, None))
        for spin, value, top in (
            (self.x_spin, self.coord.point.x, x_max),
            (self.y_spin, self.coord.point.y, y_max),
        ):
            if off:
                spin.setRange(-1e6, 1e6)
            elif top is not None:
                spin.setRange(0.0, float(top))
            spin.setValue(value)
            spin.setEnabled(not off)
            spin.setToolTip(
                "Outside the image. Drag its ring in from the canvas, or "
                "project again after placing a pair."
                if off
                else ""
            )
        # z is a slice: an integer unless the fitter found a sub-slice depth
        # (or the value is fractional anyway, as an older file's may be)
        z = self.coord.point.z
        self.z_spin.setDecimals(
            0 if float(z).is_integer() and state_text(self.coord) != "fitted" else 1
        )
        self.z_spin.setValue(z)
        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.blockSignals(False)
        self._update_state()

    def _update_state(self) -> None:
        colour = _POINT_TYPE_COLORS.get(self.coord.point_type, "gray")
        status = getattr(self.coord, "status", "")
        if status in PointStatus.TENTATIVE:
            edge = WARN_COLOR if getattr(self.coord, "suggested", False) else colour
            self.dot.setStyleSheet(
                f"border: 1.5px solid {edge}; border-radius: 5px; background: transparent;"
            )
        else:
            self.dot.setStyleSheet(f"background: {colour}; border-radius: 5px;")
        text = state_text(self.coord)
        tone = _state_tone(self.coord)
        if text.startswith("predicted") and self._off_image():
            text, tone = "predicted \u00b7 off image", "warn"
        self.state_label.setText(text)
        self.state_label.setStyleSheet(state_style(tone, 11))
        muted = status == PointStatus.REJECTED
        for w in (self.name_label, self.x_spin, self.y_spin, self.z_spin):
            font = w.font()
            font.setStrikeOut(muted)
            w.setFont(font)
        self.setToolTip(
            {
                "predicted": "Predicted from the FIB fiducial; drag it onto the burn.",
                "predicted \u00b7 start here": "Predicted from the FIB fiducial; one of the "
                "three best-spread points, so drag this one first.",
                "fitted": "Position from the image fitter.",
                "accepted": "A prediction you accepted without moving it; it counts in "
                "the fit but is no evidence for the transform.",
                "removed from fit": "Left out of the fit; still on screen and in the file.",
            }.get(text, "")
        )

    # ------------------------------------------------------------------
    # Mutation handlers
    # ------------------------------------------------------------------

    def _on_x_changed(self) -> None:
        value = self.x_spin.value()
        if value == self.coord.point.x:
            return
        self.coord.point.x = value
        self.coordinate_changed.emit(self.coord, "x", value)

    def _on_y_changed(self) -> None:
        value = self.y_spin.value()
        if value == self.coord.point.y:
            return
        self.coord.point.y = value
        self.coordinate_changed.emit(self.coord, "y", value)

    def _on_z_changed(self) -> None:
        value = self.z_spin.value()
        if value == self.coord.point.z:
            return
        self.coord.point.z = value
        self.coordinate_changed.emit(self.coord, "z", value)


# ---------------------------------------------------------------------------
# Main list widget
# ---------------------------------------------------------------------------


class CoordinateListWidget(QWidget):
    """Reorderable list of Coordinate objects.

    No header row: the rows read without column labels, and the actions live
    on the selected row and its context menu (FIB-978 \u00a77).

    Signals
    -------
    coordinate_selected  : Coordinate -- a row was selected
    coordinate_changed   : Coordinate, field, value -- xyz edited
    coordinate_removed   : Coordinate -- a row was removed
    order_changed        : List[Coordinate] -- order after drag-drop
    refit_requested      : Coordinate -- fit asked for on a row
    reset_requested      : Coordinate -- back to its prediction
    reject_toggled       : Coordinate -- in / out of the fit
    """

    coordinate_selected = pyqtSignal(object)  # Coordinate
    coordinate_changed = pyqtSignal(object, str, float)  # Coordinate, field, value
    coordinate_removed = pyqtSignal(object)  # Coordinate
    order_changed = pyqtSignal(list)  # List[Coordinate]
    refit_requested = pyqtSignal(object)  # Coordinate
    reset_requested = pyqtSignal(object)  # Coordinate
    reject_toggled = pyqtSignal(object)  # Coordinate

    def __init__(
        self,
        coordinates: Optional[List[Coordinate]] = None,
        point_type: Optional[PointType] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._coordinates: List[Coordinate] = []
        self._selected_coordinate: Optional[Coordinate] = None
        self._x_max: Optional[float] = None
        self._y_max: Optional[float] = None
        self._z_max: Optional[float] = None
        self._name_width = _name_col_width(point_type)

        self._setup_ui()
        self._connect_signals()

        if coordinates:
            self.coordinates = coordinates

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._list = _DraggableCoordinateList()
        # The item highlight would show through the row's translucent widget
        # only where no child covers it, reading as a stray box beside the
        # numbers; the row paints its own selected background instead.
        self._list.setStyleSheet(
            stylesheets.LIST_WIDGET_STYLESHEET
            + "QListWidget::item:selected { background: transparent; }"
        )
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self._list)
        self._fit_height_to_rows()

        self._empty_label = QLabel("No coordinates")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"{CAPTION_STYLE} font-style: italic; padding: 8px;"
        )
        self._empty_label.setVisible(True)
        layout.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        self._list.reordered.connect(self._on_reordered)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_axis_maxima(
        self,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> None:
        """Constrain xyz spinboxes to image dimensions. Pass None to leave unconstrained."""
        self._x_max = x_max
        self._y_max = y_max
        self._z_max = z_max
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None:
                w = self._list.itemWidget(item)
                if isinstance(w, CoordinateRowWidget):
                    w.set_axis_maxima(x_max, y_max, z_max)

    @property
    def selected_coordinate(self) -> Optional[Coordinate]:
        return self._selected_coordinate

    @property
    def coordinates(self) -> List[Coordinate]:
        return list(self._coordinates)

    @coordinates.setter
    def coordinates(self, value: List[Coordinate]) -> None:
        self._coordinates = list(value)
        self._selected_coordinate = None
        self._rebuild_rows()
        if self._coordinates:
            self._set_selected(self._coordinates[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_rows(self) -> None:
        self._list.clear()
        names = _generate_names(self._coordinates)
        for coord, name in zip(self._coordinates, names):
            self._add_row(coord, name)
        self._empty_label.setVisible(len(self._coordinates) == 0)
        self._fit_height_to_rows()

    def _fit_height_to_rows(self) -> None:
        """Size the list to its rows, up to ``_MAX_VISIBLE_ROWS``, then scroll.

        The tab around these lists scrolls as a whole; a list that scrolls
        inside it at a fixed default height hides rows, and a hidden row is a
        hidden fiducial. With no rows the list takes no space and the empty
        label shows instead.
        """
        n = self._list.count()
        shown = min(n, _MAX_VISIBLE_ROWS)
        self._list.setFixedHeight(shown * _ROW_HEIGHT + (4 if shown else 0))
        self._list.setVisible(n > 0)

    def _add_row(self, coord: Coordinate, name: str) -> None:
        row_widget = CoordinateRowWidget(
            coord=coord, name=name, name_width=self._name_width
        )
        if any(v is not None for v in (self._x_max, self._y_max, self._z_max)):
            row_widget.set_axis_maxima(self._x_max, self._y_max, self._z_max)
        self._connect_row(row_widget)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, coord)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row_widget)
        self._fit_height_to_rows()

    def _connect_row(self, row_widget: CoordinateRowWidget) -> None:
        row_widget.row_clicked.connect(self._on_row_clicked)
        row_widget.coordinate_changed.connect(self.coordinate_changed)
        row_widget.remove_clicked.connect(self._on_remove)
        row_widget.fit_clicked.connect(self.refit_requested)
        row_widget.reset_clicked.connect(self.reset_requested)
        row_widget.reject_toggled.connect(self.reject_toggled)

    def _set_selected(self, coord: Coordinate) -> None:
        self._selected_coordinate = coord
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is coord:
                self._list.setCurrentItem(item)
                break
        self._mark_selected(coord)
        self.coordinate_selected.emit(coord)

    def _mark_selected(self, coord: Optional[Coordinate]) -> None:
        """Show the actions on the selected row only."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            w = self._list.itemWidget(item) if item is not None else None
            if isinstance(w, CoordinateRowWidget):
                w.set_selected(w.coord is coord)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_clicked(self, coord: Coordinate) -> None:
        if coord is not self._selected_coordinate:
            self._set_selected(coord)

    def _on_reordered(self, coords: List[Coordinate]) -> None:
        self._coordinates = coords
        selected_before = self._selected_coordinate
        self._rebuild_rows()
        if selected_before is not None and selected_before in self._coordinates:
            self._set_selected(selected_before)
        self.order_changed.emit(list(self._coordinates))

    def _on_remove(self, coord: Coordinate) -> None:
        if coord not in self._coordinates:
            return
        idx = self._coordinates.index(coord)
        self._coordinates.remove(coord)

        next_coord = None
        if self._coordinates:
            next_idx = min(idx, len(self._coordinates) - 1)
            next_coord = self._coordinates[next_idx]

        if self._selected_coordinate is coord:
            self._selected_coordinate = None

        self._rebuild_rows()

        if next_coord is not None:
            self._set_selected(next_coord)

        self.coordinate_removed.emit(coord)

    def add_coordinate(self, coord: Coordinate) -> None:
        """Append a coordinate, rebuild its row, and select it."""
        self._coordinates.append(coord)
        names = _generate_names(self._coordinates)
        self._add_row(coord, names[-1])
        self._empty_label.setVisible(False)
        self._set_selected(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        """Re-sync spinboxes for one coordinate after an external edit (e.g. canvas drag)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is coord:
                w = self._list.itemWidget(item)
                if isinstance(w, CoordinateRowWidget):
                    w.refresh()
                break

    def select_coordinate_silent(self, coord: Optional[Coordinate]) -> None:
        """Highlight a row without emitting ``coordinate_selected`` (avoids sync loops)."""
        self._selected_coordinate = coord
        if coord is None:
            self._mark_selected(None)
            self._list.clearSelection()
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is coord:
                self._list.setCurrentItem(item)
                break
        self._mark_selected(coord)
