"""Widget for loading an existing AutoLamella experiment with protocol."""

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
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET, SECONDARY_BUTTON_STYLESHEET

# Error message constants
ERROR_PROTOCOL_NOT_FOUND_TITLE = "Protocol Not Found"
ERROR_PROTOCOL_NOT_FOUND_MSG = (
    "The protocol file 'protocol.yaml' was not found in the experiment directory:\n\n{experiment_dir}\n\n"
    "Please ensure the experiment has an associated protocol file."
)

ERROR_INVALID_EXPERIMENT_TITLE = "Invalid Experiment"
ERROR_INVALID_EXPERIMENT_MSG = (
    "The selected experiment file is not valid. It may be corrupted, incorrectly formatted, or missing required fields.\n\n"
    "Error: {error}\n\n"
    "Please select a valid experiment file (experiment.yaml)."
)

ERROR_NO_EXPERIMENT_TITLE = "No Experiment"
ERROR_NO_EXPERIMENT_MSG = "Please select an experiment file."

ERROR_NO_PROTOCOL_TITLE = "No Protocol"
ERROR_NO_PROTOCOL_MSG = "The experiment does not have a valid protocol file."

ERROR_INVALID_PROTOCOL_TITLE = "Invalid Protocol"
ERROR_INVALID_PROTOCOL_MSG = (
    "The selected protocol file is not valid. It may be corrupted, incorrectly formatted, or missing required fields.\n\n"
    "Error: {error}\n\n"
    "Please select a valid protocol file (*.yaml)."
)

ERROR_INVALID_LEGACY_PROTOCOL_TITLE = "Invalid Legacy Protocol"
ERROR_INVALID_LEGACY_PROTOCOL_MSG = (
    "The selected legacy protocol file could not be converted. It may be corrupted, incorrectly formatted, or missing required fields.\n\n"
    "Error: {error}\n\n"
    "Please select a valid legacy protocol file (*.yaml)."
)


class AutoLamellaLoadExperimentWidget(QtWidgets.QDialog):
    """Dialog for loading an existing AutoLamella experiment.

    Allows users to:
    - Select and load an existing experiment file (experiment.yaml)
    - Automatically load the associated protocol file (protocol.yaml)
    - View experiment and protocol information

    Returns:
        Experiment: The loaded experiment with attached protocol, or None if cancelled
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None
        self.protocol: Optional[AutoLamellaTaskProtocol] = None
        self.protocol_path: Optional[str] = None

        self.setWindowTitle("Load Existing Experiment")
        self.setMinimumWidth(600)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Experiment Information
        exp_group = QtWidgets.QGroupBox("Experiment Information")
        exp_layout = QtWidgets.QVBoxLayout()

        # Select Experiment button at top
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.btn_select_experiment = QtWidgets.QPushButton("Select Experiment")
        self.btn_select_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        button_layout.addWidget(self.btn_select_experiment)
        exp_layout.addLayout(button_layout)

        # Experiment form fields (all read-only)
        exp_form_layout = QtWidgets.QFormLayout()

        # Experiment Name
        self.lineEdit_experiment_name = QtWidgets.QLineEdit()
        self.lineEdit_experiment_name.setEnabled(False)
        self.lineEdit_experiment_name.setPlaceholderText("No experiment loaded")

        # Experiment Description
        self.lineEdit_experiment_description = QtWidgets.QLineEdit()
        self.lineEdit_experiment_description.setEnabled(False)
        self.lineEdit_experiment_description.setPlaceholderText("No experiment loaded")

        # User
        self.lineEdit_experiment_user = QtWidgets.QLineEdit()
        self.lineEdit_experiment_user.setEnabled(False)
        self.lineEdit_experiment_user.setPlaceholderText("No experiment loaded")

        # Project
        self.lineEdit_experiment_project = QtWidgets.QLineEdit()
        self.lineEdit_experiment_project.setEnabled(False)
        self.lineEdit_experiment_project.setPlaceholderText("No experiment loaded")

        # Organisation
        self.lineEdit_experiment_organisation = QtWidgets.QLineEdit()
        self.lineEdit_experiment_organisation.setEnabled(False)
        self.lineEdit_experiment_organisation.setPlaceholderText("No experiment loaded")

        # Experiment Directory (Read Only)
        self.lineEdit_experiment_directory = QtWidgets.QLineEdit()
        self.lineEdit_experiment_directory.setEnabled(False)
        self.lineEdit_experiment_directory.setPlaceholderText("No experiment loaded")
        self.lineEdit_experiment_directory.setCursorPosition(0)

        # Number of Lamella (Read Only)
        self.lineEdit_experiment_lamella = QtWidgets.QLineEdit()
        self.lineEdit_experiment_lamella.setEnabled(False)
        self.lineEdit_experiment_lamella.setPlaceholderText("0")

        exp_form_layout.addRow("Name", self.lineEdit_experiment_name)
        exp_form_layout.addRow("Description", self.lineEdit_experiment_description)
        exp_form_layout.addRow("User", self.lineEdit_experiment_user)
        exp_form_layout.addRow("Project", self.lineEdit_experiment_project)
        exp_form_layout.addRow("Organisation", self.lineEdit_experiment_organisation)
        exp_form_layout.addRow("Directory", self.lineEdit_experiment_directory)
        exp_form_layout.addRow("Lamella", self.lineEdit_experiment_lamella)

        exp_layout.addLayout(exp_form_layout)

        exp_group.setLayout(exp_layout)
        main_layout.addWidget(exp_group)

        # Protocol Information
        protocol_group = QtWidgets.QGroupBox("Protocol Information")
        protocol_layout = QtWidgets.QVBoxLayout()

        # Protocol action buttons (hidden unless protocol missing)
        self.protocol_button_layout = QtWidgets.QHBoxLayout()
        self.protocol_button_layout.addStretch()

        self.btn_select_legacy_protocol = QtWidgets.QPushButton("Select Legacy Protocol")
        self.btn_select_legacy_protocol.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.protocol_button_layout.addWidget(self.btn_select_legacy_protocol)

        self.btn_select_protocol = QtWidgets.QPushButton("Select Protocol")
        self.btn_select_protocol.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.protocol_button_layout.addWidget(self.btn_select_protocol)

        protocol_layout.addLayout(self.protocol_button_layout)

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
            "Note: You will be able to edit the protocol after loading the experiment."
        )
        protocol_info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        protocol_info_label.setWordWrap(True)
        protocol_layout.addWidget(protocol_info_label)

        protocol_group.setLayout(protocol_layout)
        main_layout.addWidget(protocol_group)

        # Dialog buttons (OK/Cancel)
        button_box = QtWidgets.QDialogButtonBox()

        self.btn_ok = QtWidgets.QPushButton("Load Experiment")
        self.btn_ok.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_ok.setDefault(True)
        self.btn_ok.setEnabled(False)  # Disabled until experiment is loaded

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_box.addButton(self.btn_ok, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self._set_protocol_buttons_visible(False)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_select_experiment.clicked.connect(self._select_experiment)
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_select_protocol.clicked.connect(self._select_protocol)
        self.btn_select_legacy_protocol.clicked.connect(self._select_legacy_protocol)

    def _select_experiment(self):
        """Open dialog to select an experiment file."""
        experiment_path = fui.open_existing_file_dialog(
            msg="Select an experiment file (experiment.yaml)",
            path=str(cfg.LOG_PATH),
            parent=self,
        )

        if not experiment_path or experiment_path == "":
            return

        # Validate the experiment file
        try:
            # Load the experiment
            self.experiment = Experiment.load(Path(experiment_path))

            # Load the protocol from the same directory
            experiment_dir = os.path.dirname(experiment_path)
            protocol_path = os.path.join(experiment_dir, "protocol.yaml")

            if self.experiment.task_protocol is not None:
                # Protocol already attached to experiment
                # Load the protocol
                self.protocol_path = os.path.join(self.experiment.path, "protocol.yaml")

                # Enable OK button
                self.btn_ok.setEnabled(True)

                logging.info(f"Experiment loaded successfully from {experiment_path}")
                logging.info(f"Protocol loaded successfully from {protocol_path}")
            else:
                # Protocol missing; keep experiment loaded so user can reattach one
                QtWidgets.QMessageBox.warning(
                    self,
                    ERROR_PROTOCOL_NOT_FOUND_TITLE,
                    ERROR_PROTOCOL_NOT_FOUND_MSG.format(experiment_dir=experiment_dir)
                    + "\n\nThe experiment has been loaded without a task protocol. "
                      "Please load a protocol before continuing."
                )
                self.protocol_path = None
                self.btn_ok.setEnabled(False)
                self._set_protocol_buttons_visible(True)
                logging.warning(
                    f"Experiment loaded from {experiment_path} but no protocol.yaml was found."
                )

            # Update the display regardless of protocol presence
            self._update_experiment_display()
            self._update_protocol_display()
            if self.experiment.task_protocol is not None:
                self._set_protocol_buttons_visible(False)

        except Exception as e:
            logging.error(f"Failed to load experiment: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_EXPERIMENT_TITLE,
                ERROR_INVALID_EXPERIMENT_MSG.format(error=e)
            )
            self._clear_display()

    def _update_experiment_display(self):
        """Update the experiment information display."""
        if self.experiment is None:
            self.lineEdit_experiment_name.setText("")
            self.lineEdit_experiment_name.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_description.setText("")
            self.lineEdit_experiment_description.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_user.setText("")
            self.lineEdit_experiment_user.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_project.setText("")
            self.lineEdit_experiment_project.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_organisation.setText("")
            self.lineEdit_experiment_organisation.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_directory.setText("")
            self.lineEdit_experiment_directory.setPlaceholderText("No experiment loaded")
            self.lineEdit_experiment_lamella.setText("")
            self.lineEdit_experiment_lamella.setPlaceholderText("0")
            return

        self.lineEdit_experiment_name.setText(self.experiment.name or "")

        # Clear placeholders and show empty text if no metadata
        self.lineEdit_experiment_description.setPlaceholderText("")
        self.lineEdit_experiment_description.setText(self.experiment.description or "")

        self.lineEdit_experiment_user.setPlaceholderText("")
        self.lineEdit_experiment_user.setText(self.experiment.user or "")

        self.lineEdit_experiment_project.setPlaceholderText("")
        self.lineEdit_experiment_project.setText(self.experiment.project or "")

        self.lineEdit_experiment_organisation.setPlaceholderText("")
        self.lineEdit_experiment_organisation.setText(self.experiment.organisation or "")

        self.lineEdit_experiment_directory.setText(str(self.experiment.path) or "")
        self.lineEdit_experiment_directory.setCursorPosition(0)
        self.lineEdit_experiment_lamella.setText(str(len(self.experiment.positions)))

    def _update_protocol_display(self):
        """Update the protocol information display."""
        if self.experiment is None or self.experiment.task_protocol is None:
            self.lineEdit_protocol_name.setText("")
            self.lineEdit_protocol_name.setPlaceholderText("No protocol loaded")
            self.lineEdit_protocol_description.setText("")
            self.lineEdit_protocol_description.setPlaceholderText("No protocol loaded")
            self.lineEdit_protocol_path.setText("")
            self.lineEdit_protocol_path.setPlaceholderText("No protocol loaded")
            self.lineEdit_protocol_tasks.setText("")
            self.lineEdit_protocol_tasks.setPlaceholderText("0")
            return

        self.lineEdit_protocol_name.setText(self.experiment.task_protocol.name or "")
        self.lineEdit_protocol_description.setText(self.experiment.task_protocol.description or "")
        self.lineEdit_protocol_path.setText(self.protocol_path or "")
        self.lineEdit_protocol_path.setCursorPosition(0)
        self.lineEdit_protocol_tasks.setText(str(len(self.experiment.task_protocol.task_config)))

    def _clear_display(self):
        """Clear all display fields and disable OK button."""
        self.experiment = None
        self.protocol_path = None
        self._update_experiment_display()
        self._update_protocol_display()
        self.btn_ok.setEnabled(False)
        self._set_protocol_buttons_visible(False)

    def _on_ok_clicked(self):
        """Handle OK button click - validate and return experiment."""
        # Validate experiment is loaded
        if self.experiment is None:
            QtWidgets.QMessageBox.warning(
                self,
                ERROR_NO_EXPERIMENT_TITLE,
                ERROR_NO_EXPERIMENT_MSG
            )
            return

        # Validate protocol is loaded
        if self.experiment.task_protocol is None:
            QtWidgets.QMessageBox.warning(
                self,
                ERROR_NO_PROTOCOL_TITLE,
                ERROR_NO_PROTOCOL_MSG
            )
            return

        logging.info(f"Experiment '{self.experiment.name}' loaded successfully")

        # Accept the dialog
        self.accept()

    def get_experiment(self) -> Optional[Experiment]:
        """Return the loaded experiment, or None if cancelled."""
        return self.experiment

    def _select_protocol(self):
        """Let the user pick a protocol file after experiment load."""

        protocol_path = fui.open_existing_file_dialog(
            msg="Select a task protocol file (*.yaml)",
            path=str(cfg.TASK_PROTOCOL_PATH),
            parent=self,
        )

        if not protocol_path:
            return

        try:
            protocol = AutoLamellaTaskProtocol.load(protocol_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_PROTOCOL_TITLE,
                ERROR_INVALID_PROTOCOL_MSG.format(error=e),
            )
            return

        self.protocol_path = protocol_path
        if self.experiment is not None:
            self.experiment.task_protocol = protocol
        self._update_protocol_display()
        self.btn_ok.setEnabled(True)
        self._set_protocol_buttons_visible(False)
        logging.info(f"Protocol loaded manually from {protocol_path}")

    def _select_legacy_protocol(self):
        """Let the user pick and convert a legacy protocol file."""
        
        protocol_path = fui.open_existing_file_dialog(
            msg="Select a legacy protocol file (*.yaml)",
            path=str(cfg.PROTOCOL_PATH),
            parent=self,
        )

        if not protocol_path:
            return

        try:
            protocol = AutoLamellaTaskProtocol.load_from_old_protocol(Path(protocol_path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_LEGACY_PROTOCOL_TITLE,
                ERROR_INVALID_LEGACY_PROTOCOL_MSG.format(error=e),
            )
            return

        self.protocol_path = protocol_path
        if self.experiment is not None:
            self.experiment.task_protocol = protocol
        self._update_protocol_display()
        self.btn_ok.setEnabled(True)
        self._set_protocol_buttons_visible(False)
        logging.info(f"Legacy protocol converted from {protocol_path}")

    def _set_protocol_buttons_visible(self, visible: bool) -> None:
        """Show or hide protocol selection buttons."""
        self.btn_select_protocol.setVisible(visible)
        self.btn_select_protocol.setEnabled(visible)
        self.btn_select_legacy_protocol.setVisible(visible)
        self.btn_select_legacy_protocol.setEnabled(visible)


def load_experiment_dialog(parent: Optional[QtWidgets.QWidget] = None) -> Optional[Experiment]:
    """Create and execute the experiment loading dialog.

    Args:
        parent: Parent widget for the dialog

    Returns:
        Experiment: The loaded experiment with attached protocol, or None if cancelled
    """
    # Create and show the dialog
    dialog = AutoLamellaLoadExperimentWidget(parent)
    result = dialog.exec_()

    # Handle the result
    if result == QtWidgets.QDialog.Accepted:
        experiment = dialog.get_experiment()
        if experiment:
            logging.info(f"Experiment loaded: {experiment.name}")
            logging.info(f"Path: {experiment.path}")
            logging.info(f"Protocol: {experiment.task_protocol.name}")
            logging.info(f"Number of tasks: {len(experiment.task_protocol.task_config)}")
        return experiment
    else:
        logging.info("Experiment loading cancelled")
        return None


def main():
    """Test the AutoLamellaLoadExperimentWidget."""
    import napari
    viewer = napari.Viewer()

    qwidget = QtWidgets.QWidget()
    viewer.window.add_dock_widget(qwidget, area='right')

    # Use the standalone function to load experiment
    experiment = load_experiment_dialog(qwidget)

    if experiment:
        print(f"Experiment loaded: {experiment.name}")
        print(f"Path: {experiment.path}")
        print(f"Protocol: {experiment.task_protocol.name}")
        print(f"Number of tasks: {len(experiment.task_protocol.task_config)}")
    else:
        print("Experiment loading cancelled")

    napari.run()


if __name__ == "__main__":
    main()
