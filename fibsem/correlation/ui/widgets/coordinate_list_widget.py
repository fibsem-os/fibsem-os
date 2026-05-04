"""CoordinateListWidget — reorderable list of Coordinate objects.

Each row shows an auto-generated read-only name, x/y/z spinboxes, and a trash
button. The header row holds a shared color indicator and refit button that
operate on the currently selected coordinate.

Operates on a flat List[Coordinate]. Callers are responsible for
flattening/reconstructing CorrelationInputData.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, ValueSpinBox

_DRAG_HANDLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "ui", "icons", "drag_handle.svg",
)

_NAME_FIXED_WIDTH = 100
_SPIN_FIXED_WIDTH = 75
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
# Spacer in header aligning with drag handle in rows (layout spacing handles the 4px gap)
_ROW_RIGHT_WIDTH = 10

_POINT_TYPE_COLORS: Dict[PointType, str] = {
    PointType.FIB:     "lime",
    PointType.FM:      "cyan",
    PointType.POI:     "magenta",
    PointType.SURFACE: "red",
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
# Header
# ---------------------------------------------------------------------------

class _CoordinateListHeader(QWidget):
    """Sticky dark header with column labels, a color indicator for the
    selected coordinate's point type, and a shared refit button."""

    def __init__(self, parent=None, default_color: Optional[str] = None) -> None:
        self._default_color = default_color
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")
        self.setFixedHeight(_ROW_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        def _lbl(text: str, width: Optional[int] = None) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #aaa; font-size: 11px; background: transparent;")
            if width is not None:
                lbl.setFixedWidth(width)
            return lbl

        layout.addWidget(_lbl("Name", _NAME_FIXED_WIDTH))
        layout.addWidget(_lbl("X", _SPIN_FIXED_WIDTH))
        layout.addWidget(_lbl("Y", _SPIN_FIXED_WIDTH))
        layout.addWidget(_lbl("Z", _SPIN_FIXED_WIDTH))
        layout.addStretch(1)

        # Color indicator — shows selected row's point type color
        self.color_label = QLabel()
        self.color_label.setFixedSize(_BTN_SIZE)
        self.color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.color_label)

        # Refit button — acts on selected coordinate; disabled when nothing selected
        self.btn_refit = IconToolButton(
            icon="mdi:refresh",
            tooltip="Refit selected coordinate",
            size=_BTN_SIZE.width(),
        )
        self.btn_refit.setEnabled(False)
        layout.addWidget(self.btn_refit)

        # Spacer aligning with rows' remove button + drag handle
        layout.addWidget(_lbl("", _ROW_RIGHT_WIDTH))

        self._clear_color()

    def update_selection(self, coord: Optional[Coordinate]) -> None:
        """Update color indicator and refit enable state for the selected coord."""
        if coord is None:
            self._clear_color()
            self.btn_refit.setEnabled(False)
        else:
            color = _POINT_TYPE_COLORS.get(coord.point_type, "gray")
            px = QPixmap(_BTN_SIZE.width() - 8, _BTN_SIZE.height() - 8)
            px.fill(QColor(color))
            self.color_label.setPixmap(px)
            self.color_label.setToolTip(f"Point type: {coord.point_type.value}")
            self.btn_refit.setEnabled(True)

    def _clear_color(self) -> None:
        if self._default_color:
            px = QPixmap(_BTN_SIZE.width() - 8, _BTN_SIZE.height() - 8)
            px.fill(QColor(self._default_color))
            self.color_label.setPixmap(px)
        else:
            self.color_label.setPixmap(QPixmap())
            self.color_label.setToolTip("")


# ---------------------------------------------------------------------------
# Row widget
# ---------------------------------------------------------------------------

class CoordinateRowWidget(QWidget):
    """Single row: read-only name, xyz spinboxes, trash button."""

    row_clicked        = pyqtSignal(object)              # Coordinate
    coordinate_changed = pyqtSignal(object, str, float)  # Coordinate, field, value
    remove_clicked     = pyqtSignal(object)              # Coordinate

    def __init__(
        self,
        coord: Coordinate,
        name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.coord = coord
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        # Name label (read-only)
        self.name_label = QLabel(name)
        self.name_label.setFixedWidth(_NAME_FIXED_WIDTH)
        self.name_label.setStyleSheet(
            "color: #F0F1F2; background: transparent; font-size: 12px;"
        )
        self.name_label.setToolTip("Auto-generated coordinate name")
        layout.addWidget(self.name_label)

        # XYZ spinboxes
        self.x_spin = ValueSpinBox(decimals=3, minimum=-1e6, maximum=1e6, step=0.001, no_buttons=True)
        self.x_spin.setFixedWidth(_SPIN_FIXED_WIDTH)
        self.x_spin.setToolTip("X coordinate")
        layout.addWidget(self.x_spin)

        self.y_spin = ValueSpinBox(decimals=3, minimum=-1e6, maximum=1e6, step=0.001, no_buttons=True)
        self.y_spin.setFixedWidth(_SPIN_FIXED_WIDTH)
        self.y_spin.setToolTip("Y coordinate")
        layout.addWidget(self.y_spin)

        self.z_spin = ValueSpinBox(decimals=3, minimum=-1e6, maximum=1e6, step=0.001, no_buttons=True)
        self.z_spin.setFixedWidth(_SPIN_FIXED_WIDTH)
        self.z_spin.setToolTip("Z coordinate")
        layout.addWidget(self.z_spin)

        layout.addStretch(1)

        # Remove button
        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline",
            tooltip="Remove coordinate",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_remove)

        # Drag handle
        drag_icon = QLabel()
        drag_icon.setFixedSize(10, 16)
        if os.path.exists(_DRAG_HANDLE_PATH):
            drag_icon.setPixmap(
                QPixmap(_DRAG_HANDLE_PATH).scaled(
                    10, 16,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        # Install eventFilter on spinboxes for row-selection on focus
        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.installEventFilter(self)

        self._connect_signals()
        self.refresh()

    def _connect_signals(self) -> None:
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.coord))
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        self.name_label.setText(name)

    def set_axis_maxima(
        self,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> None:
        """Constrain spinbox ranges to image shape. Pass None to leave unconstrained."""
        if x_max is not None:
            self.x_spin.setMinimum(0.0)
            self.x_spin.setMaximum(float(x_max))
        if y_max is not None:
            self.y_spin.setMinimum(0.0)
            self.y_spin.setMaximum(float(y_max))
        if z_max is not None:
            self.z_spin.setMinimum(0.0)
            self.z_spin.setMaximum(float(z_max))

    def refresh(self) -> None:
        """Re-sync spinboxes from coord.point without emitting signals."""
        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.blockSignals(True)
        self.x_spin.setValue(self.coord.point.x)
        self.y_spin.setValue(self.coord.point.y)
        self.z_spin.setValue(self.coord.point.z)
        for w in (self.x_spin, self.y_spin, self.z_spin):
            w.blockSignals(False)

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

    The header holds a shared color indicator and refit button for the
    currently selected row.

    Signals
    -------
    coordinate_selected  : Coordinate — a row was selected
    coordinate_changed   : Coordinate, field, value — xyz edited
    coordinate_removed   : Coordinate — a row was removed
    order_changed        : List[Coordinate] — order after drag-drop
    refit_requested      : Coordinate — header refit button clicked
    """

    coordinate_selected  = pyqtSignal(object)              # Coordinate
    coordinate_changed   = pyqtSignal(object, str, float)  # Coordinate, field, value
    coordinate_removed   = pyqtSignal(object)              # Coordinate
    order_changed        = pyqtSignal(list)                # List[Coordinate]
    refit_requested      = pyqtSignal(object)              # Coordinate

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
        self._default_header_color = _POINT_TYPE_COLORS.get(point_type) if point_type else None

        self._setup_ui()
        self._connect_signals()

        if coordinates:
            self.coordinates = coordinates

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _CoordinateListHeader(default_color=self._default_header_color)
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableCoordinateList()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._list)

        self._empty_label = QLabel("No coordinates")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-style: italic; padding: 8px;")
        self._empty_label.setVisible(True)
        layout.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        self._list.reordered.connect(self._on_reordered)
        self._header.btn_refit.clicked.connect(self._on_header_refit)

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
        self._header.update_selection(None)
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

    def _add_row(self, coord: Coordinate, name: str) -> None:
        row_widget = CoordinateRowWidget(coord=coord, name=name)
        if any(v is not None for v in (self._x_max, self._y_max, self._z_max)):
            row_widget.set_axis_maxima(self._x_max, self._y_max, self._z_max)
        self._connect_row(row_widget)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, coord)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row_widget)

    def _connect_row(self, row_widget: CoordinateRowWidget) -> None:
        row_widget.row_clicked.connect(self._on_row_clicked)
        row_widget.coordinate_changed.connect(self.coordinate_changed)
        row_widget.remove_clicked.connect(self._on_remove)

    def _set_selected(self, coord: Coordinate) -> None:
        self._selected_coordinate = coord
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is coord:
                self._list.setCurrentItem(item)
                break
        self._header.update_selection(coord)
        self.coordinate_selected.emit(coord)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_clicked(self, coord: Coordinate) -> None:
        if coord is not self._selected_coordinate:
            self._set_selected(coord)

    def _on_header_refit(self) -> None:
        self.refit_requested.emit(self._selected_coordinate)

    def _on_reordered(self, coords: List[Coordinate]) -> None:
        self._coordinates = coords
        selected_before = self._selected_coordinate
        self._rebuild_rows()
        if selected_before is not None and selected_before in self._coordinates:
            self._set_selected(selected_before)
        else:
            self._header.update_selection(self._selected_coordinate)
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
        else:
            self._header.update_selection(None)

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
            self._header.update_selection(None)
            self._list.clearSelection()
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is coord:
                self._list.setCurrentItem(item)
                break
        self._header.update_selection(coord)
