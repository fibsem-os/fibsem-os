import logging
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import napari
from napari.layers import Points as NapariPointsLayer

from fibsem.structures import Point
from fibsem.applications.autolamella.workflows.tasks.tasks import SpotBurnFiducialTaskConfig


SPOT_BURN_EDITOR_POINTS_LAYER = "spot-burn-coordinates"


class AutoLamellaSpotBurnCoordinatesWidget(QWidget):
    """Widget for editing spot burn coordinates in the protocol editor.

    Provides a table with x, y columns and add/remove buttons,
    synced with a napari points layer for visual placement.
    """

    settings_changed = pyqtSignal(SpotBurnFiducialTaskConfig)

    def __init__(self,
                 viewer: napari.Viewer,
                 config: Optional[SpotBurnFiducialTaskConfig] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.viewer = viewer
        self.config = config
        self.pts_layer: Optional[NapariPointsLayer] = None
        self._updating = False  # guard against re-entrant updates

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label_info = QLabel("Coordinates are in relative image space (0-1).")
        label_info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label_info)

        # table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["x", "y"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(150)
        self.table.cellChanged.connect(self._on_table_changed)
        self.table.currentCellChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self.table)

        # mode buttons
        mode_layout = QHBoxLayout()
        self.btn_mode_add = QPushButton("Add Mode")
        self.btn_mode_select = QPushButton("Select Mode")
        self.btn_mode_add.setCheckable(True)
        self.btn_mode_select.setCheckable(True)
        self.btn_mode_add.setChecked(True)
        self.btn_mode_add.clicked.connect(lambda: self._set_layer_mode("add"))
        self.btn_mode_select.clicked.connect(lambda: self._set_layer_mode("select"))
        mode_layout.addWidget(self.btn_mode_add)
        mode_layout.addWidget(self.btn_mode_select)
        layout.addLayout(mode_layout)

        # buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        self.add_btn.clicked.connect(self._add_coordinate)
        self.remove_btn.clicked.connect(self._remove_coordinate)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        layout.addLayout(btn_layout)

        # summary label
        self.label_summary = QLabel("No coordinates defined.")
        layout.addWidget(self.label_summary)

    # --- public API (matches fluorescence widget pattern) ---

    def set_task_config(self, config: SpotBurnFiducialTaskConfig):
        """Set the config and update the table + points layer."""
        self.config = config
        self._populate_table(config.coordinates)
        self._sync_points_layer_from_table()
        self._update_summary()

    def get_task_config(self) -> Optional[SpotBurnFiducialTaskConfig]:
        """Read coordinates from the table back into the config."""
        if self.config is None:
            return None
        self.config.coordinates = self._read_table()
        return self.config

    # --- table helpers ---

    def _populate_table(self, coordinates: List[Point]):
        """Fill the table from a list of Points."""
        self.table.blockSignals(True)
        self.table.setRowCount(len(coordinates))
        for i, pt in enumerate(coordinates):
            self.table.setItem(i, 0, QTableWidgetItem(f"{pt.x:.4f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{pt.y:.4f}"))
        self.table.blockSignals(False)

    def _read_table(self) -> List[Point]:
        """Read the table rows as a list of Points."""
        points = []
        for i in range(self.table.rowCount()):
            x_item = self.table.item(i, 0)
            y_item = self.table.item(i, 1)
            if x_item and y_item:
                try:
                    points.append(Point(float(x_item.text()), float(y_item.text())))
                except ValueError:
                    pass
        return points

    def _add_coordinate(self):
        """Add a new coordinate row at (0.5, 0.5)."""
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("0.5000"))
        self.table.setItem(row, 1, QTableWidgetItem("0.5000"))
        self.table.blockSignals(False)
        self._on_coordinates_changed()

    def _remove_coordinate(self):
        """Remove the currently selected coordinate row."""
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._on_coordinates_changed()

    # --- napari points layer ---

    def _ensure_points_layer(self):
        """Create or retrieve the napari points layer."""
        if SPOT_BURN_EDITOR_POINTS_LAYER in self.viewer.layers:
            self.pts_layer = self.viewer.layers[SPOT_BURN_EDITOR_POINTS_LAYER]
        else:
            self.pts_layer = self.viewer.add_points(
                data=[],
                name=SPOT_BURN_EDITOR_POINTS_LAYER,
                visible=True,
                size=20,
            )
            self.pts_layer.events.data.connect(self._on_points_layer_changed)
            self.pts_layer.events.highlight.connect(self._on_points_layer_selection_changed)
            self.pts_layer.mode = "add"

    def _remove_points_layer(self):
        """Remove the points layer from the viewer."""
        if SPOT_BURN_EDITOR_POINTS_LAYER in self.viewer.layers:
            self.viewer.layers.remove(SPOT_BURN_EDITOR_POINTS_LAYER)
        self.pts_layer = None

    def _sync_points_layer_from_table(self):
        """Update the napari points layer to match the table contents."""
        self._ensure_points_layer()

        coordinates = self._read_table()
        if not coordinates:
            self.pts_layer.data = np.empty((0, 2))
            return

        # find the reference image layer to get shape for coordinate conversion
        image_layer = self._get_image_layer()
        if image_layer is None:
            return

        image_shape = image_layer.data.shape
        translate = image_layer.translate

        pts = np.array([[pt.y * image_shape[0] + translate[0],
                         pt.x * image_shape[1] + translate[1]]
                        for pt in coordinates])

        self._updating = True
        self.pts_layer.data = pts
        self._updating = False

    def _sync_table_from_points_layer(self):
        """Update the table to match the napari points layer."""
        if self.pts_layer is None or len(self.pts_layer.data) == 0:
            self._populate_table([])
            return

        image_layer = self._get_image_layer()
        if image_layer is None:
            return

        layer_translated = self.pts_layer.data - image_layer.translate
        image_shape = image_layer.data.shape

        coordinates = [Point(float(pt[1] / image_shape[1]), float(pt[0] / image_shape[0]))
                       for pt in layer_translated]
        self._populate_table(coordinates)

    def _get_image_layer(self):
        """Get the reference image layer from the viewer."""
        if "Reference Image (FIB)" in self.viewer.layers:
            return self.viewer.layers["Reference Image (FIB)"]
        return None

    # --- change handlers ---

    def _on_table_changed(self):
        """Called when the user edits a cell in the table."""
        self._on_coordinates_changed()

    def _on_points_layer_changed(self, event=None):
        """Called when the user moves/adds/removes points in napari."""
        if self._updating:
            return
        self._updating = True
        self._sync_table_from_points_layer()
        self._update_summary()
        self._emit_settings_changed()
        self._updating = False

    def _on_coordinates_changed(self):
        """Common handler: sync points layer from table, emit signal."""
        self._sync_points_layer_from_table()
        self._update_summary()
        self._emit_settings_changed()

    def _emit_settings_changed(self):
        """Update config and emit the settings_changed signal."""
        config = self.get_task_config()
        if config is not None:
            self.settings_changed.emit(config)

    def _update_summary(self):
        """Update the summary label with the current coordinate count."""
        n = self.table.rowCount()
        if n == 0:
            self.label_summary.setText("No coordinates defined.")
        else:
            self.label_summary.setText(f"{n} coordinate{'s' if n != 1 else ''} defined.")

    # --- layer mode ---

    def _set_layer_mode(self, mode: str):
        """Set the points layer mode and update button state."""
        if self.pts_layer is not None:
            self.pts_layer.mode = mode
        self.btn_mode_add.setChecked(mode == "add")
        self.btn_mode_select.setChecked(mode == "select")

    # --- selection sync ---

    def _on_table_selection_changed(self):
        """When a row is selected in the table, select the corresponding point in napari."""
        if self._updating or self.pts_layer is None:
            return
        row = self.table.currentRow()
        if row < 0 or row >= len(self.pts_layer.data):
            return
        self._updating = True
        self.viewer.layers.selection.active = self.pts_layer
        self.pts_layer.selected_data = {row}
        self._updating = False

    def _on_points_layer_selection_changed(self, event=None):
        """When a point is selected in napari, select the corresponding row in the table."""
        if self._updating or self.pts_layer is None:
            return
        selected = self.pts_layer.selected_data
        if len(selected) == 1:
            row = next(iter(selected))
            if 0 <= row < self.table.rowCount():
                self._updating = True
                self.table.selectRow(row)
                self._updating = False

    # --- visibility ---

    def showEvent(self, event):
        """When the widget becomes visible, ensure the points layer exists."""
        super().showEvent(event)
        self._sync_points_layer_from_table()

    def hideEvent(self, event):
        """When the widget is hidden, remove the points layer."""
        super().hideEvent(event)
        self._remove_points_layer()
