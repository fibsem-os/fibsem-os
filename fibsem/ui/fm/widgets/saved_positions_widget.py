import logging
from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from fibsem.fm.structures import FMStagePosition
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)
from fibsem.ui.utils import message_box_ui
from fibsem.applications.autolamella.structures import Lamella

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget


class SavedPositionsWidget(QWidget):

    def __init__(self, parent: Optional['FMAcquisitionWidget'] = None):
        super().__init__(parent)
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        # Combobox for selecting saved positions
        self.label_positions = QLabel("Select Position", self)
        self.comboBox_positions = QComboBox(self)
        self.comboBox_positions.setToolTip("Select a saved position from the list")

        # Checkbox list for selecting multiple positions
        self.label_checkbox_list = QLabel("Select Positions", self)
        self.listWidget_positions = QListWidget(self)
        self.listWidget_positions.setToolTip("Check/uncheck positions for selection")
        self.listWidget_positions.setMaximumHeight(120)  # Limit height to keep widget compact

        # Buttons for managing positions
        self.pushButton_goto_position = QPushButton("Go To", self)
        self.pushButton_goto_position.setToolTip("Move stage to the selected position")
        self.pushButton_delete_position = QPushButton("Delete", self)
        self.pushButton_delete_position.setToolTip("Delete the selected position")

        # Controls for updating objective position
        self.comboBox_objective_source = QComboBox(self)
        self.comboBox_objective_source.addItem("Current Position")
        self.comboBox_objective_source.addItem("Focus Position")
        self.comboBox_objective_source.setToolTip("Select which objective position to use")
        self.pushButton_set_objective = QPushButton("Set Objective", self)
        self.pushButton_set_objective.setToolTip("Update selected position with chosen objective position")

        # Position info label
        self.label_position_info = QLabel("No positions saved", self)
        self.label_position_info.setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        self.label_position_info.setWordWrap(True)

        # Auto Focus checkbox
        self.checkBox_auto_focus = QCheckBox("Auto Focus at each position", self)
        self.checkBox_auto_focus.setToolTip("Run autofocus at each position before acquisition")
        self.checkBox_auto_focus.setChecked(False)

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.label_positions, 0, 0)
        layout.addWidget(self.comboBox_positions, 0, 1)
        layout.addWidget(self.pushButton_goto_position, 1, 0)
        layout.addWidget(self.pushButton_delete_position, 1, 1)
        layout.addWidget(self.comboBox_objective_source, 2, 0)
        layout.addWidget(self.pushButton_set_objective, 2, 1)
        layout.addWidget(self.label_position_info, 3, 0, 1, 2)
        layout.addWidget(self.label_checkbox_list, 4, 0, 1, 2)
        layout.addWidget(self.listWidget_positions, 5, 0, 1, 2)
        layout.addWidget(self.checkBox_auto_focus, 6, 0, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Connect signals
        self.comboBox_positions.currentIndexChanged.connect(self._on_position_selected)
        self.listWidget_positions.itemChanged.connect(self._on_checkbox_changed)
        self.listWidget_positions.currentItemChanged.connect(self._on_list_item_selected)
        self.pushButton_goto_position.clicked.connect(self._goto_selected_position)
        self.pushButton_delete_position.clicked.connect(self._delete_selected_position)
        self.pushButton_set_objective.clicked.connect(self._set_objective_position)

        # Set initial button states
        self._update_widget_state()

        # Set button styles
        self.pushButton_goto_position.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_delete_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_set_objective.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

    def update_positions(self, positions: List[Lamella]):
        """Update the combobox and checkbox list with current saved positions."""
        # Store current selection
        current_text = self.comboBox_positions.currentText()

        # Clear and repopulate combobox and list widget
        self.comboBox_positions.clear()
        self.listWidget_positions.clear()

        if not positions:
            self.comboBox_positions.addItem("No positions saved")
            self.label_position_info.setText("No positions saved")
        else:
            for i, pos in enumerate(positions):
                self.comboBox_positions.addItem(pos.name)

                item = QListWidgetItem(pos.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # Default to checked
                self.listWidget_positions.addItem(item)

            # Try to restore previous selection
            current_index = self.comboBox_positions.findText(current_text)
            if current_index >= 0:
                self.comboBox_positions.setCurrentIndex(current_index)
                self.listWidget_positions.setCurrentRow(current_index)
            
            # Update info for currently selected position
            self._update_position_info()

        self._update_widget_state()
        if self.parent_widget and self.parent_widget.experiment:
            self.parent_widget.experiment.save()

    def _update_widget_state(self):
        """Update button enabled/disabled state based on available positions."""
        has_positions = bool(self.parent_widget and
                             self.parent_widget.experiment and
                             len(self.parent_widget.experiment.positions) > 0)

        self.pushButton_goto_position.setEnabled(has_positions)
        self.pushButton_delete_position.setEnabled(has_positions)
        self.pushButton_set_objective.setEnabled(has_positions)
        self.comboBox_objective_source.setEnabled(has_positions)
        self.comboBox_positions.setEnabled(has_positions)

        if not has_positions:
            self.pushButton_goto_position.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
            self.pushButton_delete_position.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
            self.pushButton_set_objective.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        else:
            self.pushButton_goto_position.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
            self.pushButton_delete_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
            self.pushButton_set_objective.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

    def _update_position_info(self):
        """Update the position info label with details of the selected position."""
        if (not self.parent_widget or
            not self.parent_widget.experiment or
            not self.parent_widget.experiment.positions):
            self.label_position_info.setText("No positions saved")
            return

        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.experiment.positions):
            fm_pos = self.parent_widget.experiment.positions[current_index]
            self.label_position_info.setText(fm_pos.pretty_fm_name)
        else:
            self.label_position_info.setText("Invalid selection")

    def _on_position_selected(self, index: int):
        """Handle position selection in combobox."""
        self._update_position_info()
        if index >= 0:
            # Sync checkbox list selection
            self.listWidget_positions.setCurrentRow(index)
            # Update crosshairs to highlight selected position
            if self.parent_widget:
                self.parent_widget.draw_stage_position_crosshairs()

    def _on_list_item_selected(self, current_item, previous_item):
        """Handle selection change in checkbox list."""
        if current_item is not None:
            # Get the index of the selected item
            index = self.listWidget_positions.row(current_item)
            # Sync combobox selection
            if 0 <= index < self.comboBox_positions.count():
                self.comboBox_positions.setCurrentIndex(index)

    def _on_checkbox_changed(self, item):
        """Handle checkbox state change in list widget."""
        # Update the positions button in the parent widget when checkboxes change
        if self.parent_widget:
            self.parent_widget._update_positions_button()

    def get_checked_positions(self) -> List[FMStagePosition]:
        """Return a list of positions that are currently checked in the checkbox list."""
        if not self.parent_widget or not self.parent_widget.experiment or not self.parent_widget.experiment.positions:
            return []

        checked_positions = []
        for i in range(self.listWidget_positions.count()):
            item = self.listWidget_positions.item(i)
            if item.checkState() == Qt.Checked:
                if i < len(self.parent_widget.experiment.positions):
                    checked_positions.append(self.parent_widget.experiment.positions[i])

        return checked_positions

    def get_auto_focus_enabled(self) -> bool:
        """Return whether auto focus is enabled for position acquisition."""
        return self.checkBox_auto_focus.isChecked()

    def _goto_selected_position(self):
        """Move stage to the selected position."""
        if (not self.parent_widget or
            not self.parent_widget.experiment or
            not self.parent_widget.experiment.positions):
            return

        if not self.parent_widget.fm.parent:
            logging.error("No parent microscope available for stage movement")
            return

        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.experiment.positions):
            fm_position = self.parent_widget.experiment.positions[current_index]

            try:
                logging.info(f"Moving to position: {fm_position.name}")
                self.parent_widget.fm.parent.move_stage_absolute(fm_position.stage_position)

                # Also move objective to saved position
                if fm_position.objective_position is not None:
                    self.parent_widget.fm.objective.move_absolute(fm_position.objective_position)
                    self.parent_widget.objectiveControlWidget.update_objective_position_labels()
            except Exception as e:
                logging.error(f"Error moving to position: {e}")

    def _delete_selected_position(self):
        """Delete the selected position."""
        if (not self.parent_widget or
            not self.parent_widget.experiment or
            not self.parent_widget.experiment.positions):
            return

        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.experiment.positions):
            position = self.parent_widget.experiment.positions[current_index]

            # Confirmation dialog
            ret = message_box_ui(
                title="Delete Position",
                text=f"Are you sure you want to delete position '{position.name}'?",
                parent=self
            )
            if ret:
                # Remove from parent's list
                del self.parent_widget.experiment.positions[current_index]
                self.parent_widget.experiment.save()

                # Update displays
                self.update_positions(self.parent_widget.experiment.positions)
                self.parent_widget.draw_stage_position_crosshairs()
                self.parent_widget._update_positions_button()
                logging.info(f"Deleted saved position: {position.name}")

    def _set_objective_position(self):
        """Update the selected position with the chosen objective position source."""
        if not self.parent_widget or not self.parent_widget.experiment or not self.parent_widget.experiment.positions:
            return

        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.experiment.positions):
            # Get the selected source
            source = self.comboBox_objective_source.currentText()

            if source not in ["Current Position", "Focus Position"]:
                logging.error(f"Unknown objective source: {source}")
                return

            if source == "Current Position":
                new_objective_position = self.parent_widget.fm.objective.position
                source_description = "current position"
            elif source == "Focus Position":
                new_objective_position = self.parent_widget.fm.objective.focus_position
                source_description = "focus position"

                if new_objective_position is None:
                    logging.warning("No focus position available. Run autofocus first.")
                    return

            # Update the saved position
            fm_position = self.parent_widget.experiment.positions[current_index]
            old_objective = fm_position.objective_position
            fm_position.objective_position = new_objective_position

            # Update displays
            self._update_position_info()
            self.parent_widget.experiment.save()

            if old_objective is None:
                old_objective_str = "N/A"
            else:
                old_objective_str = f"{old_objective*1e3:.2f} mm"
            logging.info(f"Updated '{fm_position.name}' objective position from {old_objective_str} to {source_description} {new_objective_position*1e3:.2f} mm")
