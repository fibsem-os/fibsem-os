# TMP: WIP
from __future__ import annotations

import logging
from dataclasses import fields
from copy import deepcopy
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QGridLayout,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscopes._stage import SampleGrid, SampleHolder, Stage
from fibsem.structures import FibsemStagePosition
from fibsem.ui import stylesheets

def _get_field_metadata(model_cls, field_name: str) -> dict:
    for field_def in fields(model_cls):
        if field_def.name == field_name:
            return dict(field_def.metadata)
    return {}


def _apply_spinbox_metadata(spinbox: QDoubleSpinBox, metadata: dict, scale: float = 1.0) -> None:
    if "minimum" in metadata and metadata["minimum"] is not None:
        spinbox.setMinimum(metadata["minimum"] * scale)
    if "maximum" in metadata and metadata["maximum"] is not None:
        spinbox.setMaximum(metadata["maximum"] * scale)
    if "step" in metadata and metadata["step"] is not None:
        spinbox.setSingleStep(metadata["step"] * scale)
    if "decimals" in metadata and metadata["decimals"] is not None:
        spinbox.setDecimals(int(metadata["decimals"]))
    elif not spinbox.decimals():
        spinbox.setDecimals(2)

    spinbox.setKeyboardTracking(False)

    units = metadata.get("units")
    if units:
        spinbox.setSuffix(f" {units}")
    tooltip = metadata.get("tooltip")
    if tooltip:
        spinbox.setToolTip(tooltip)


HOLDER_PRE_TILT_METADATA = _get_field_metadata(SampleHolder, "pre_tilt")
HOLDER_REFERENCE_ROTATION_METADATA = _get_field_metadata(SampleHolder, "reference_rotation")
GRID_RADIUS_METADATA = _get_field_metadata(SampleGrid, "radius")
SCALE_UNIT_MAP = {
    1e2: "cm",
    1e3: "mm",
    1e6: "µm",
    1e9: "nm",
}


class SampleHolderWidget(QWidget):
    """Widget that exposes a form for editing SampleHolder metadata and its grids."""

    holder_changed = pyqtSignal(SampleHolder)
    grid_selected = pyqtSignal(str, object)

    def __init__(self, microscope: Optional['FibsemMicroscope'] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)
        self._microscope = microscope
        self._holder: Optional[SampleHolder] = None
        self._updating_ui = False
        self._setup_ui()
        self._connect_signals()
        self.setEnabled(False)

    def _setup_ui(self) -> None:

        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.pre_tilt_spin = QDoubleSpinBox()
        self.reference_rotation_spin = QDoubleSpinBox()

        _apply_spinbox_metadata(self.pre_tilt_spin, HOLDER_PRE_TILT_METADATA)
        _apply_spinbox_metadata(self.reference_rotation_spin, HOLDER_REFERENCE_ROTATION_METADATA)

        form_layout.addRow("Name", self.name_edit)
        form_layout.addRow("Description", self.description_edit)
        form_layout.addRow("Pre-Tilt", self.pre_tilt_spin)
        form_layout.addRow("Reference Rotation", self.reference_rotation_spin)

        self.grid_list = QListWidget()
        self.grid_list.setSelectionMode(QListWidget.SingleSelection)
        self.position_label = QLabel("")

        layout = QVBoxLayout(self)
        layout.addLayout(form_layout)
        layout.addWidget(QLabel("Grids"))
        layout.addWidget(self.grid_list, 1)
        layout.addWidget(self.position_label)
        layout.addWidget(QLabel("Double-click a grid to edit"))

        button_layout = QGridLayout()
        self.add_grid_button = QPushButton("Add Grid")
        self.remove_grid_button = QPushButton("Remove Grid")
        self.edit_grid_button = QPushButton("Edit Grid")
        self.move_to_grid_button = QPushButton("Move to Grid")

        self.add_grid_button.setToolTip("Add a new sample grid to the holder")
        self.remove_grid_button.setToolTip("Remove the selected sample grid from the holder")
        self.edit_grid_button.setToolTip("Edit the selected sample grid")
        self.move_to_grid_button.setToolTip("Move the stage to the selected grid position")

        self.add_grid_button.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.remove_grid_button.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.edit_grid_button.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        self.move_to_grid_button.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        button_layout.addWidget(self.add_grid_button, 0, 0)
        button_layout.addWidget(self.remove_grid_button, 0, 1)
        button_layout.addWidget(self.edit_grid_button, 1, 0)
        button_layout.addWidget(self.move_to_grid_button, 1, 1)
        layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        self.name_edit.editingFinished.connect(self._handle_holder_changed)
        self.description_edit.editingFinished.connect(self._handle_holder_changed)
        self.pre_tilt_spin.valueChanged.connect(self._handle_holder_changed)
        self.reference_rotation_spin.valueChanged.connect(self._handle_holder_changed)
        self.grid_list.currentItemChanged.connect(self._handle_grid_selection)
        self.grid_list.itemDoubleClicked.connect(self._handle_grid_double_clicked)
        self.grid_selected.connect(self._on_grid_selected)

        self.add_grid_button.clicked.connect(self._handle_add_grid)
        self.remove_grid_button.clicked.connect(self._handle_remove_grid)
        self.edit_grid_button.clicked.connect(self._handle_edit_grid)
        self.move_to_grid_button.clicked.connect(self._handle_move_to_grid)

    def set_holder(self, holder: Optional[SampleHolder]) -> None:
        self._holder = holder
        self.setEnabled(holder is not None)
        self._refresh_grid_list()

        if holder is None:
            self.name_edit.clear()
            self.description_edit.clear()
            self.pre_tilt_spin.setValue(0.0)
            self.reference_rotation_spin.setValue(0.0)
            return

        self.name_edit.setText(holder.name)
        self.description_edit.setText(holder.description)
        self.pre_tilt_spin.setValue(holder.pre_tilt)
        self.reference_rotation_spin.setValue(holder.reference_rotation)

    def current_holder(self) -> Optional[SampleHolder]:
        return self._holder

    def _refresh_grid_list(self) -> None:
        self.grid_list.clear()
        if self._holder is None:
            return

        grids = list(self._holder.grids.items())
        grids.sort(key=lambda item: item[1].index)
        for name, grid in grids:
            item = QListWidgetItem(f"{grid.index:02d} — {grid.name}")
            item.setData(Qt.UserRole, name)
            self.grid_list.addItem(item)

    def _handle_holder_changed(self) -> None:
        if self._holder is None or self._updating_ui:
            return

        self._holder.name = self.name_edit.text()
        self._holder.description = self.description_edit.text()
        self._holder.pre_tilt = self.pre_tilt_spin.value()
        self._holder.reference_rotation = self.reference_rotation_spin.value()
        self.holder_changed.emit(self._holder)

    def _handle_grid_selection(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if self._holder is None or current is None:
            return

        grid_key = current.data(Qt.UserRole)
        if grid_key is None:
            return

        grid = self._holder.grids.get(grid_key)
        if grid is not None:
            self.grid_selected.emit(grid_key, grid)

    def _handle_grid_double_clicked(self, item: QListWidgetItem) -> None:
        grid_key = item.data(Qt.UserRole)
        if grid_key is None:
            return
        
        if self._holder is None:
            return
        self._open_grid_editor(grid_key)

    def _open_grid_editor(self, grid_key: str) -> None:        
        if self._holder is None:
            return
        grid = self._holder.grids.get(grid_key)
        grid_widget = SampleGridWidget()
        grid_widget.set_grid(grid_key, grid)

        # add to dialog
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Grid: {grid_key}")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.addWidget(grid_widget)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog.exec_()

    def update_grid_name(self, old_name: str, new_name: str) -> None:
        """Update list entry for a renamed grid."""
        if self._holder is None:
            return

        grid = self._holder.grids.pop(old_name, None)
        if grid is None:
            return
        self._holder.grids[new_name] = grid
        self._refresh_grid_list()

    def _on_grid_selected(self, grid_key: str, grid: SampleGrid) -> None:
        
        msg = f"{grid_key}: {grid.position.pretty if grid.position else 'Position not set'}"
        self.position_label.setText(msg)

    def _handle_add_grid(self) -> None:
        return
    
    def _handle_remove_grid(self) -> None:
        return

    def _handle_edit_grid(self) -> None:
        current_item = self.grid_list.currentItem()
        if current_item is None:
            return
        grid_key = current_item.data(Qt.UserRole)
        if grid_key is None:
            return
        self._open_grid_editor(grid_key)

    def _handle_move_to_grid(self) -> None:
        if self._microscope is None or self._holder is None:
            return
        
        current_item = self.grid_list.currentItem()
        if current_item is None:
            return
        grid_key = current_item.data(Qt.UserRole)
        if grid_key is None:
            return
        grid = self._holder.grids.get(grid_key)
        if grid is None:
            return
        self._microscope._stage.move_to_grid(grid.name)
        logging.info(f"Moved stage to grid '{grid.name}' at position {grid.position}")

class SampleGridWidget(QWidget):
    """Widget that edits SampleGrid metadata and stage position."""

    grid_changed = pyqtSignal(SampleGrid)
    grid_name_changed = pyqtSignal(str, str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._grid: Optional[SampleGrid] = None
        self._grid_key: Optional[str] = None
        self._updating_ui = False
        self._radius_metadata = GRID_RADIUS_METADATA
        self._radius_scale = self._radius_metadata.get("scale", 1.0) or 1.0
        if self._radius_scale == 0:
            self._radius_scale = 1.0
        self._setup_ui()
        self._connect_signals()
        self.setEnabled(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.index_spin = QSpinBox()
        self.index_spin.setMinimum(0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(2)
        self.radius_spin.setRange(0.0, 50.0)
        self.radius_spin.setSingleStep(0.1)
        _apply_spinbox_metadata(self.radius_spin, self._radius_metadata, scale=self._radius_scale)
        radius_units = self._radius_metadata.get("display_units")
        if not radius_units:
            radius_units = SCALE_UNIT_MAP.get(
                self._radius_scale, self._radius_metadata.get("units")
            )
        if radius_units:
            self.radius_spin.setSuffix(f" {radius_units}")
        self.description_edit = QLineEdit()
        
        self.position_label = QLabel("Stage position not set")
        self.position_label.setWordWrap(True)

        self.form_layout.addRow("Name", self.name_edit)
        self.form_layout.addRow("Index", self.index_spin)
        self.form_layout.addRow("Radius", self.radius_spin)
        self.form_layout.addRow("Description", self.description_edit)
        self.form_layout.addRow("Stage Position", self.position_label)

        layout.addLayout(self.form_layout)

    def _connect_signals(self) -> None:
        pass


    def set_grid(self, grid_key: Optional[str], grid: Optional[SampleGrid]) -> None:
        self._grid = grid
        self._grid_key = grid_key

        self.setEnabled(grid is not None)

        if grid is None:
            self.name_edit.clear()
            self.index_spin.setValue(0)
            self.radius_spin.setValue(0.0)
            self.description_edit.clear()
            self._update_position_label(None)
            return

        self.name_edit.setText(grid.name)
        self.index_spin.setValue(grid.index)
        display_radius = self._to_display_radius(grid.radius)
        self.radius_spin.setValue(display_radius)
        self.description_edit.setText(grid.description)
        self._update_position_label(grid.position)

    def current_grid(self) -> Optional[SampleGrid]:
        return self._grid

    def _update_position_label(
        self, position: Optional[FibsemStagePosition]
    ) -> None:
        if position is None:
            self.position_label.setText("Stage position not set")
        else:
            self.position_label.setText(position.pretty)

    def _to_display_radius(self, radius: Optional[float]) -> float:
        if radius is None:
            return 0.0
        return radius * self._radius_scale

    def _from_display_radius(self, value: float) -> float:
        if self._radius_scale == 0:
            return value
        return value / self._radius_scale



if __name__ == "__main__":


    from fibsem import utils
    import os
    from fibsem.structures import BeamType, FibsemImage, Point, FibsemStagePosition
    from fibsem.imaging.tiled import plot_minimap
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope
    import matplotlib.pyplot as plt
    import glob
    import logging

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    PATH = "/home/patrick/github/fibsem/fibsem/config/microscope-configuration.yaml"
    microscope, settings = utils.setup_session(config_path=None)

    widget = QWidget()
    layout = QVBoxLayout(widget)


    holder_widget = SampleHolderWidget(microscope=microscope)
    # grid_widget = SampleGridWidget()
    layout.addWidget(holder_widget)
    # layout.addWidget(grid_widget)

    holder_widget.set_holder(microscope._stage.holder)
    def on_holder_changed(holder: SampleHolder) -> None:
        print("Holder changed:", holder)
    holder_widget.holder_changed.connect(on_holder_changed) 

    def on_grid_selected(grid_key: str, grid: SampleGrid) -> None:
        print("Grid selected:", grid_key, grid)
        # grid_widget.set_grid(grid_key, grid)
    holder_widget.grid_selected.connect(on_grid_selected)

    # def on_grid_changed(grid: SampleGrid) -> None:
    #     print("Grid changed:", grid)
    # grid_widget.grid_changed.connect(on_grid_changed)


    # holder_widget.show()
    # grid_widget.show()
    widget.show()


    sys.exit(app.exec_())
