import logging
from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
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

if TYPE_CHECKING:
    from fibsem.ui.FMAcquisitionWidget import FMAcquisitionWidget


class SavedPositionsWidget(QWidget):
    position_deleted = pyqtSignal(int)  # Signal emitted when a position is deleted (index)
    position_selected = pyqtSignal(int)  # Signal emitted when a position is selected (index)
    
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

    def update_positions(self, positions: List[FMStagePosition]):
        """Update the combobox and checkbox list with current saved positions."""
        # Store current selection
        current_text = self.comboBox_positions.currentText()
        
        # Clear and repopulate combobox
        self.comboBox_positions.clear()
        
        # Clear and repopulate checkbox list
        self.listWidget_positions.clear()
        
        if not positions:
            self.comboBox_positions.addItem("No positions saved")
            self.label_position_info.setText("No positions saved")
        else:
            for i, pos in enumerate(positions):
                display_text = pos.name  # FMStagePosition.name is required
                self.comboBox_positions.addItem(display_text)
                
                # Add item to checkbox list
                item = QListWidgetItem(display_text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # Default to checked
                self.listWidget_positions.addItem(item)
            
            # Try to restore previous selection
            current_index = self.comboBox_positions.findText(current_text)
            if current_index >= 0:
                self.comboBox_positions.setCurrentIndex(current_index)
                # Sync checkbox list selection
                self.listWidget_positions.setCurrentRow(current_index)
            
            # Update info for currently selected position
            self._update_position_info()
        
        self._update_widget_state()

    def _update_widget_state(self):
        """Update button enabled/disabled state based on available positions."""
        has_positions = self.parent_widget and len(self.parent_widget.stage_positions) > 0
        
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
        if not self.parent_widget or not self.parent_widget.stage_positions:
            self.label_position_info.setText("No positions saved")
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            fm_pos = self.parent_widget.stage_positions[current_index]
            info_text = fm_pos.format_position_info()
            self.label_position_info.setText(info_text)
        else:
            self.label_position_info.setText("Invalid selection")

    def _on_position_selected(self, index: int):
        """Handle position selection in combobox."""
        self._update_position_info()
        if index >= 0:
            # Sync checkbox list selection
            self.listWidget_positions.setCurrentRow(index)
            self.position_selected.emit(index)
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
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return []
        
        checked_positions = []
        for i in range(self.listWidget_positions.count()):
            item = self.listWidget_positions.item(i)
            if item.checkState() == Qt.Checked:
                if i < len(self.parent_widget.stage_positions):
                    checked_positions.append(self.parent_widget.stage_positions[i])
        
        return checked_positions

    def _goto_selected_position(self):
        """Move stage to the selected position."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            fm_position = self.parent_widget.stage_positions[current_index]
            stage_position = fm_position.stage_position
            
            if self.parent_widget.fm.parent:
                try:
                    logging.info(f"Moving to saved position: {fm_position.name} at {stage_position}")
                    self.parent_widget.fm.parent.move_stage_absolute(stage_position)
                    
                    # Also move objective to saved position
                    logging.info(f"Moving objective to: {fm_position.objective_position*1e3:.2f} mm")
                    self.parent_widget.fm.objective.move_absolute(fm_position.objective_position)
                    
                    self.parent_widget.display_stage_position_overlay()
                    self.parent_widget.objectiveControlWidget.update_objective_position_labels()
                except Exception as e:
                    logging.error(f"Error moving to position: {e}")
            else:
                logging.error("No parent microscope available for stage movement")

    def _delete_selected_position(self):
        """Delete the selected position."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return
        
        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
            position = self.parent_widget.stage_positions[current_index]
            
            # Confirmation dialog
            from fibsem.ui.utils import message_box_ui
            ret = message_box_ui(
                title="Delete Position",
                text=f"Are you sure you want to delete position '{position.name}'?",
                parent=self
            )
            if ret:
                # Remove from parent's list
                del self.parent_widget.stage_positions[current_index]
                
                # Update displays
                self.update_positions(self.parent_widget.stage_positions)
                self.parent_widget.draw_stage_position_crosshairs()
                self.parent_widget._update_positions_button()
                
                # Save updated positions to YAML file
                self.parent_widget._save_positions_to_yaml()
                
                # Emit signal
                self.position_deleted.emit(current_index)
                
                logging.info(f"Deleted saved position: {position.name}")


    def _load_positions_from_file(self):
        """Load positions from a user-selected YAML file."""
        if not self.parent_widget:
            return
        
        # Open file dialog to select positions file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Positions File",
            self.parent_widget.experiment_path,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            import yaml
            
            with open(file_path, 'r') as f:
                positions_data = yaml.safe_load(f)
            
            if not positions_data or 'positions' not in positions_data:
                from fibsem.ui.utils import message_box_ui
                message_box_ui(
                    title="Invalid File",
                    text="The selected file does not contain valid position data.",
                    parent=self
                )
                return
            
            # Confirmation dialog showing what will be loaded
            num_positions = len(positions_data['positions'])
            from fibsem.ui.utils import message_box_ui
            ret = message_box_ui(
                title="Load Positions",
                text=f"Load {num_positions} positions from file? This will replace current positions.",
                parent=self
            )
            
            if ret:
                # Clear existing positions
                self.parent_widget.stage_positions.clear()
                
                # Load positions from file
                for pos_dict in positions_data['positions']:
                    try:
                        fm_position = FMStagePosition.from_dict(pos_dict)
                        self.parent_widget.stage_positions.append(fm_position)
                    except Exception as e:
                        logging.warning(f"Failed to load position from file: {e}")
                        continue
                
                # Update UI
                self.update_positions(self.parent_widget.stage_positions)
                self.parent_widget.draw_stage_position_crosshairs()
                self.parent_widget._update_positions_button()
                
                # Save to current experiment's positions.yaml
                self.parent_widget._save_positions_to_yaml()
                
                logging.info(f"Loaded {len(self.parent_widget.stage_positions)} positions from {file_path}")
                
        except Exception as e:
            from fibsem.ui.utils import message_box_ui
            message_box_ui(
                title="Error Loading File",
                text=f"Failed to load positions from file:\n{str(e)}",
                parent=self
            )
            logging.error(f"Error loading positions from {file_path}: {e}")

    def _set_objective_position(self):
        """Update the selected position with the chosen objective position source."""
        if not self.parent_widget or not self.parent_widget.stage_positions:
            return

        current_index = self.comboBox_positions.currentIndex()
        if 0 <= current_index < len(self.parent_widget.stage_positions):
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
            fm_position = self.parent_widget.stage_positions[current_index]
            old_objective = fm_position.objective_position
            fm_position.objective_position = new_objective_position

            # Update displays
            self._update_position_info()
            self.parent_widget._save_positions_to_yaml()

            logging.info(f"Updated '{fm_position.name}' objective position from {old_objective*1e3:.2f} mm to {source_description} {new_objective_position*1e3:.2f} mm")
