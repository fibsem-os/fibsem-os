"""Widget for creating a new AutoLamella experiment with protocol selection."""

import datetime
import logging
import os
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.ui import utils as fui
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit


class AutoLamellaCreateExperimentWidget(QtWidgets.QDialog):
    """Dialog for creating a new AutoLamella experiment.

    Allows users to:
    - Create a new experiment with name, description, and directory
    - Add optional metadata (user, project, organisation)
    - Select and validate a task protocol file
    - Attach the protocol to the experiment

    Returns:
        Experiment: The created experiment with attached protocol, or None if cancelled
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None
        self.protocol: Optional[AutoLamellaTaskProtocol] = None
        self.protocol_path: Optional[str] = None

        self.setWindowTitle("Create New Experiment")
        self.setMinimumWidth(600)

        self._setup_ui()
        self._connect_signals()

        # Load default protocol (deferred to showEvent to ensure QApplication exists)
        self._default_protocol_loaded = False

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Experiment Information
        exp_group = QtWidgets.QGroupBox("Experiment Information")
        exp_layout = QtWidgets.QVBoxLayout()

        # Experiment form fields
        exp_form_layout = QtWidgets.QFormLayout()

        # Experiment Name
        self.lineEdit_experiment_name = QtWidgets.QLineEdit()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.lineEdit_experiment_name.setText(f"{cfg.EXPERIMENT_NAME}-{current_date}")

        # Experiment Description
        self.lineEdit_experiment_description = QtWidgets.QLineEdit()
        self.lineEdit_experiment_description.setPlaceholderText("Optional description of the experiment...")

        # User (optional)
        self.lineEdit_experiment_user = QtWidgets.QLineEdit()
        self.lineEdit_experiment_user.setPlaceholderText("Optional user name...")

        # Project (optional)
        self.lineEdit_experiment_project = QtWidgets.QLineEdit()
        self.lineEdit_experiment_project.setPlaceholderText("Optional project name...")

        # Organisation (optional)
        self.lineEdit_experiment_organisation = QtWidgets.QLineEdit()
        self.lineEdit_experiment_organisation.setPlaceholderText("Optional organisation name...")

        # Experiment Directory
        self.lineEdit_experiment_directory = QDirectoryLineEdit()
        self.lineEdit_experiment_directory.setText(str(cfg.LOG_PATH))

        exp_form_layout.addRow("Name", self.lineEdit_experiment_name)
        exp_form_layout.addRow("Description", self.lineEdit_experiment_description)
        exp_form_layout.addRow("User", self.lineEdit_experiment_user)
        exp_form_layout.addRow("Project", self.lineEdit_experiment_project)
        exp_form_layout.addRow("Organisation", self.lineEdit_experiment_organisation)
        exp_form_layout.addRow("Directory", self.lineEdit_experiment_directory)

        exp_layout.addLayout(exp_form_layout)

        # Validation warning label
        self.label_validation_warning = QtWidgets.QLabel("")
        self.label_validation_warning.setStyleSheet("color: orange; font-style: italic;")
        self.label_validation_warning.setWordWrap(True)
        exp_layout.addWidget(self.label_validation_warning)

        exp_group.setLayout(exp_layout)
        main_layout.addWidget(exp_group)

        # Protocol Information
        protocol_group = QtWidgets.QGroupBox("Protocol Information")
        protocol_layout = QtWidgets.QVBoxLayout()

        # Select Protocol buttons at top
        protocol_button_layout = QtWidgets.QHBoxLayout()
        protocol_button_layout.addStretch()

        self.btn_select_legacy_protocol = QtWidgets.QPushButton("Select Legacy Protocol")
        self.btn_select_legacy_protocol.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        protocol_button_layout.addWidget(self.btn_select_legacy_protocol)

        self.btn_select_protocol = QtWidgets.QPushButton("Select Protocol")
        self.btn_select_protocol.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        protocol_button_layout.addWidget(self.btn_select_protocol)

        protocol_layout.addLayout(protocol_button_layout)

        # Protocol form fields (all read-only)
        protocol_form_layout = QtWidgets.QFormLayout()

        self.lineEdit_protocol_name = QtWidgets.QLineEdit()
        self.lineEdit_protocol_name.setEnabled(False)
        self.lineEdit_protocol_name.setPlaceholderText("No protocol loaded")

        self.lineEdit_protocol_description = QtWidgets.QLineEdit()
        self.lineEdit_protocol_description.setEnabled(False)
        self.lineEdit_protocol_description.setPlaceholderText("No protocol loaded")

        self.lineEdit_protocol_path = QtWidgets.QLineEdit()
        self.lineEdit_protocol_path.setEnabled(False)
        self.lineEdit_protocol_path.setPlaceholderText("No protocol loaded")
        self.lineEdit_protocol_path.setCursorPosition(0)

        self.lineEdit_protocol_tasks = QtWidgets.QLineEdit()
        self.lineEdit_protocol_tasks.setEnabled(False)
        self.lineEdit_protocol_tasks.setPlaceholderText("0")

        protocol_form_layout.addRow("Name", self.lineEdit_protocol_name)
        protocol_form_layout.addRow("Description", self.lineEdit_protocol_description)
        protocol_form_layout.addRow("Path", self.lineEdit_protocol_path)
        protocol_form_layout.addRow("Tasks", self.lineEdit_protocol_tasks)

        protocol_layout.addLayout(protocol_form_layout)

        # Protocol tooltip/info label
        protocol_info_label = QtWidgets.QLabel(
            "Note: You will be able to edit the protocol after creating the experiment."
        )
        protocol_info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        protocol_info_label.setWordWrap(True)
        protocol_layout.addWidget(protocol_info_label)

        protocol_group.setLayout(protocol_layout)
        main_layout.addWidget(protocol_group)

        # Dialog buttons (Create/Cancel)
        button_box = QtWidgets.QDialogButtonBox()

        self.btn_ok = QtWidgets.QPushButton("Create Experiment")
        self.btn_ok.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_ok.setDefault(True)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_box.addButton(self.btn_ok, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def showEvent(self, event):
        """Called when widget is shown - load default protocol if not already loaded."""
        super().showEvent(event)
        if not self._default_protocol_loaded:
            self._load_default_protocol()
            self._default_protocol_loaded = True

    def _connect_signals(self):
        """Connect UI signals."""
        self.lineEdit_experiment_directory.textChanged.connect(self._validate_experiment_path)
        self.btn_select_protocol.clicked.connect(self._select_protocol)
        self.btn_select_legacy_protocol.clicked.connect(self._select_legacy_protocol)
        self.lineEdit_experiment_name.textChanged.connect(self._validate_experiment_path)
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)

    def _validate_experiment_path(self):
        """Validate the experiment path and show warning if it already exists."""
        directory = self.lineEdit_experiment_directory.text()
        experiment_name = self.lineEdit_experiment_name.text().strip()

        if not directory or not experiment_name:
            self.label_validation_warning.setText("")
            return

        experiment_path = os.path.join(directory, experiment_name)

        if os.path.exists(experiment_path):
            self.label_validation_warning.setText(
                f"⚠ Warning: An experiment named '{experiment_name}' already exists in this directory."
            )
        else:
            self.label_validation_warning.setText("")

    def _load_default_protocol(self):
        """Load the default task protocol."""
        if os.path.exists(cfg.TASK_PROTOCOL_PATH):
            try:
                self.protocol = AutoLamellaTaskProtocol.load(str(cfg.TASK_PROTOCOL_PATH))
                self.protocol_path = str(cfg.TASK_PROTOCOL_PATH)
                self._update_protocol_display()
                logging.info(f"Default protocol loaded from {cfg.TASK_PROTOCOL_PATH}")
            except Exception as e:
                logging.error(f"Failed to load default protocol: {e}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Protocol Load Error",
                    "The default protocol file could not be loaded. It may be corrupted or incorrectly formatted.\n\nPlease select a valid protocol file manually."
                )

    def _select_protocol(self):
        """Open dialog to select a protocol file."""
        # Open file dialog at the directory of the current protocol or default
        start_path = cfg.TASK_PROTOCOL_PATH
        if self.protocol_path:
            start_path = os.path.dirname(self.protocol_path)

        protocol_path = fui.open_existing_file_dialog(
            msg="Select a task protocol file (*.yaml)",
            path=start_path,
            parent=self,
        )

        if not protocol_path or protocol_path == "":
            return

        # Validate the protocol file
        try:
            self.protocol = AutoLamellaTaskProtocol.load(protocol_path)
            self.protocol_path = protocol_path
            self._update_protocol_display()
            logging.info(f"Protocol loaded successfully from {protocol_path}")
        except Exception as e:
            logging.error(f"Failed to load protocol: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Protocol",
                "The selected protocol file is not valid. It may be corrupted, incorrectly formatted, or missing required fields.\n\nPlease select a valid protocol file (*.yaml)."
            )

    def _select_legacy_protocol(self):
        """Open dialog to select a legacy protocol file and convert it."""
        # Open file dialog at the protocol path
        start_path = cfg.PROTOCOL_PATH
        if self.protocol_path:
            start_path = os.path.dirname(self.protocol_path)

        protocol_path = fui.open_existing_file_dialog(
            msg="Select a legacy protocol file (*.yaml)",
            path=start_path,
            parent=self,
        )

        if not protocol_path or protocol_path == "":
            return

        # Validate and convert the legacy protocol file
        try:
            self.protocol = AutoLamellaTaskProtocol.load_from_old_protocol(Path(protocol_path))
            self.protocol_path = protocol_path
            self._update_protocol_display()
            logging.info(f"Legacy protocol loaded and converted successfully from {protocol_path}")
            QtWidgets.QMessageBox.information(
                self,
                "Legacy Protocol Converted",
                "The legacy protocol has been successfully converted to the new task-based format."
            )
        except Exception as e:
            logging.error(f"Failed to load legacy protocol: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Legacy Protocol",
                f"The selected legacy protocol file could not be converted. It may be corrupted, incorrectly formatted, or missing required fields.\n\nError: {e}\n\nPlease select a valid legacy protocol file (*.yaml)."
            )

    def _update_protocol_display(self):
        """Update the protocol information display."""
        if self.protocol is None:
            self.lineEdit_protocol_name.setText("")
            self.lineEdit_protocol_description.setText("")
            self.lineEdit_protocol_path.setText("")
            self.lineEdit_protocol_tasks.setText("0")
            return

        self.lineEdit_protocol_name.setText(self.protocol.name or "")
        self.lineEdit_protocol_description.setText(self.protocol.description or "")
        self.lineEdit_protocol_path.setText(self.protocol_path or "")
        self.lineEdit_protocol_path.setCursorPosition(0)
        self.lineEdit_protocol_tasks.setText(str(len(self.protocol.task_config)))

    def _on_ok_clicked(self):
        """Handle OK button click - validate and create experiment."""
        # Validate experiment name
        experiment_name = self.lineEdit_experiment_name.text().strip()
        if not experiment_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Name",
                "Please enter an experiment name."
            )
            return

        # Validate directory
        directory = self.lineEdit_experiment_directory.text()
        if not directory or not os.path.exists(directory):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Directory",
                "Please select a valid directory."
            )
            return

        # Validate protocol
        if self.protocol is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Protocol",
                "Please select a task protocol file."
            )
            return

        # Check if experiment already exists
        experiment_path = os.path.join(directory, experiment_name)
        if os.path.exists(experiment_path):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Experiment Exists",
                f"An experiment named '{experiment_name}' already exists in this directory.\n\nDo you want to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        # Create the experiment
        try:
            # Collect metadata from optional fields
            metadata = {}
            experiment_description = self.lineEdit_experiment_description.text().strip()
            experiment_user = self.lineEdit_experiment_user.text().strip()
            experiment_project = self.lineEdit_experiment_project.text().strip()
            experiment_organisation = self.lineEdit_experiment_organisation.text().strip()

            if experiment_description:
                metadata["description"] = experiment_description
            if experiment_user:
                metadata["user"] = experiment_user
            if experiment_project:
                metadata["project"] = experiment_project
            if experiment_organisation:
                metadata["organisation"] = experiment_organisation

            self.experiment = Experiment.create(
                path=Path(directory),
                name=experiment_name,
                metadata=metadata if metadata else None
            )

            # Attach the protocol to the experiment
            self.experiment.task_protocol = self.protocol

            # Save the protocol to the experiment directory
            protocol_save_path = os.path.join(self.experiment.path, "protocol.yaml")
            self.experiment.task_protocol.save(protocol_save_path)

            logging.info(f"Experiment '{experiment_name}' created successfully at {self.experiment.path}")
            logging.info(f"Protocol saved to {protocol_save_path}")

            # Accept the dialog
            self.accept()

        except Exception as e:
            logging.error(f"Failed to create experiment: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to create experiment:\n\n{e}"
            )

    def get_experiment(self) -> Optional[Experiment]:
        """Return the created experiment, or None if cancelled."""
        return self.experiment


def create_experiment_dialog(parent: Optional[QtWidgets.QWidget] = None) -> Optional[Experiment]:
    """Create and execute the experiment creation dialog.

    Args:
        parent: Parent widget for the dialog

    Returns:
        Experiment: The created experiment with attached protocol, or None if cancelled
    """
    # Create and show the dialog
    dialog = AutoLamellaCreateExperimentWidget(parent)
    result = dialog.exec_()

    # Handle the result
    if result == QtWidgets.QDialog.Accepted:
        experiment = dialog.get_experiment()
        if experiment:
            logging.info(f"Experiment created: {experiment.name}")
            logging.info(f"Path: {experiment.path}")
            logging.info(f"Protocol: {experiment.task_protocol.name}")
            logging.info(f"Number of tasks: {len(experiment.task_protocol.task_config)}")
        return experiment
    else:
        logging.info("Experiment creation cancelled")
        return None


def main():
    """Test the AutoLamellaCreateExperimentWidget."""
    import napari
    viewer = napari.Viewer()

    qwidget = QtWidgets.QWidget()
    viewer.window.add_dock_widget(qwidget, area='right')

    # Use the standalone function to create experiment
    experiment = create_experiment_dialog(qwidget)

    if experiment:
        print(f"Experiment created: {experiment.name}")
        print(f"Path: {experiment.path}")
        print(f"Protocol: {experiment.task_protocol.name}")
        print(f"Metadata: {experiment.metadata}")
        print(f"Number of tasks: {len(experiment.task_protocol.task_config)}")
    else:
        print("Experiment creation cancelled")

    napari.run()
    # sys.exit()


if __name__ == "__main__":
    main()
