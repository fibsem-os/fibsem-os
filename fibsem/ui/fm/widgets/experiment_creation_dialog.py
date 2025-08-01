import os
from datetime import datetime
from typing import Optional

import yaml
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.stylesheets import GRAY_PUSHBUTTON_STYLE, GREEN_PUSHBUTTON_STYLE


class ExperimentCreationDialog(QDialog):
    """Dialog for creating a new experiment or loading an existing one."""

    def __init__(self, initial_directory: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.selected_directory = initial_directory
        self.experiment_name = ""
        self.positions_file_path = ""
        self.setWindowTitle("Experiment Setup")
        self.setModal(True)
        self.initUI()

    def initUI(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Radio buttons for mode selection
        mode_layout = QVBoxLayout()
        self.radio_create = QRadioButton("Create New Experiment")
        self.radio_load = QRadioButton("Load Existing Experiment")
        self.radio_create.setChecked(True)  # Default to create mode

        mode_layout.addWidget(self.radio_create)
        mode_layout.addWidget(self.radio_load)
        layout.addLayout(mode_layout)
        layout.addWidget(QLabel())

        # Create experiment section
        self.create_section = QWidget()
        create_layout = QVBoxLayout()

        # Directory selection for creating
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Directory:")
        dir_layout.addWidget(dir_label)

        self.dir_line_edit = QLineEdit(self.selected_directory)
        self.dir_line_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_line_edit)

        self.browse_dir_button = QToolButton()
        self.browse_dir_button.setText("...")
        self.browse_dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.browse_dir_button)
        create_layout.addLayout(dir_layout)

        # Experiment name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_layout.addWidget(name_label)

        # Generate default name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        default_name = f"AutoLamella-{timestamp}"

        self.name_line_edit = QLineEdit(default_name)
        self.name_line_edit.selectAll()
        name_layout.addWidget(self.name_line_edit)
        create_layout.addLayout(name_layout)

        # Path preview for create mode
        self.path_preview_label = QLabel()
        self.path_preview_label.setStyleSheet("font-size: 10px; color: #666666; margin: 10px 0;")
        self.update_path_preview()
        create_layout.addWidget(self.path_preview_label)

        self.create_section.setLayout(create_layout)
        layout.addWidget(self.create_section)

        # Load experiment section
        self.load_section = QWidget()
        load_layout = QVBoxLayout()

        # Positions file selection
        positions_layout = QHBoxLayout()
        positions_label = QLabel("Positions File:")
        positions_layout.addWidget(positions_label)

        self.positions_file_edit = QLineEdit()
        self.positions_file_edit.setReadOnly(True)
        self.positions_file_edit.setPlaceholderText("Select positions.yaml file...")
        positions_layout.addWidget(self.positions_file_edit)

        self.browse_positions_button = QToolButton()
        self.browse_positions_button.setText("...")
        self.browse_positions_button.clicked.connect(self.browse_positions_file)
        positions_layout.addWidget(self.browse_positions_button)
        load_layout.addLayout(positions_layout)

        # Experiment info display for load mode
        self.experiment_info_label = QLabel()
        self.experiment_info_label.setStyleSheet("font-size: 10px; color: #666666; margin: 10px 0;")
        load_layout.addWidget(self.experiment_info_label)

        self.load_section.setLayout(load_layout)
        self.load_section.setVisible(False)  # Hidden by default
        layout.addWidget(self.load_section)

        # Connect radio button signals
        self.radio_create.toggled.connect(self.on_mode_changed)
        self.radio_load.toggled.connect(self.on_mode_changed)

        # Connect signals for live preview update and validation
        self.name_line_edit.textChanged.connect(self.update_path_preview)
        self.name_line_edit.textChanged.connect(self.validate_input)
        layout.addStretch()

        # Buttons
        button_layout = QGridLayout()
        self.button_ok = QPushButton("OK")
        self.button_ok.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.button_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.button_ok, 0, 0)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.button_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.button_cancel, 0, 1)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Perform initial validation
        self.validate_input()

    def browse_directory(self):
        """Open directory browser dialog."""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Experiment Directory",
            self.selected_directory
        )

        if selected_dir:
            self.selected_directory = selected_dir
            self.dir_line_edit.setText(selected_dir)
            self.update_path_preview()

    def update_path_preview(self):
        """Update the full path preview label."""
        experiment_name = self.name_line_edit.text().strip()
        if experiment_name:
            full_path = os.path.join(self.selected_directory, experiment_name)
            self.path_preview_label.setText(f"Full path: {full_path}")
        else:
            self.path_preview_label.setText("Full path: ")

    def on_mode_changed(self):
        """Handle radio button mode changes."""
        if self.radio_create.isChecked():
            self.create_section.setVisible(True)
            self.load_section.setVisible(False)
        else:
            self.create_section.setVisible(False)
            self.load_section.setVisible(True)
        
        self.validate_input()

    def browse_positions_file(self):
        """Open file dialog to select positions.yaml file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Positions File",
            "",
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )

        if file_path:
            self.positions_file_path = file_path
            self.positions_file_edit.setText(file_path)
            self.update_experiment_info()
            self.validate_input()

    def update_experiment_info(self):
        """Update experiment info display when a positions file is selected."""
        if not self.positions_file_path:
            self.experiment_info_label.setText("")
            return

        try:
            # Get experiment directory from positions file path
            experiment_dir = os.path.dirname(self.positions_file_path)

            # Try to load and parse the positions file
            with open(self.positions_file_path, 'r') as f:
                positions_data = yaml.safe_load(f)

            num_positions = positions_data.get('num_positions', 0)
            created_date = positions_data.get('created_date', 'Unknown')

            info_text = f"Experiment: {os.path.basename(experiment_dir)}\n"
            info_text += f"Positions: {num_positions}\n"
            info_text += f"Created: {created_date}"

            self.experiment_info_label.setText(info_text)

        except Exception as e:
            self.experiment_info_label.setText(f"Error reading file: {str(e)}")

    def validate_input(self):
        """Validate input based on current mode and enable/disable OK button."""
        is_valid = False

        if self.radio_create.isChecked():
            # Validate experiment name for create mode
            experiment_name = self.name_line_edit.text().strip()
            is_valid = bool(experiment_name) and experiment_name not in ['.', '..']

            # Additional validation for invalid filename characters
            if is_valid:
                invalid_chars = '<>:"/\\|?*'
                is_valid = not any(char in experiment_name for char in invalid_chars)

        else:
            # Validate positions file for load mode
            is_valid = bool(self.positions_file_path) and os.path.exists(self.positions_file_path)

        # Enable/disable the OK button
        self.button_ok.setEnabled(is_valid)

        # Update button style based on validity
        if is_valid:
            self.button_ok.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        else:
            self.button_ok.setStyleSheet(GRAY_PUSHBUTTON_STYLE)

    def get_experiment_info(self) -> dict:
        """Get the experiment information based on current mode."""
        if self.radio_create.isChecked():
            # Create mode - return new experiment info
            experiment_name = self.name_line_edit.text().strip()
            return {
                'mode': 'create',
                'directory': self.selected_directory,
                'name': experiment_name,
                'full_path': os.path.join(self.selected_directory, experiment_name) if experiment_name else ""
            }
        else:
            # Load mode - return existing experiment info
            experiment_dir = os.path.dirname(self.positions_file_path) if self.positions_file_path else ""
            return {
                'mode': 'load',
                'positions_file': self.positions_file_path,
                'directory': experiment_dir,
                'name': os.path.basename(experiment_dir) if experiment_dir else "",
                'full_path': experiment_dir
            }