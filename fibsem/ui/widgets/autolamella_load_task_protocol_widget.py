"""Widget for loading a task protocol into an existing AutoLamella experiment."""

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

# Error message constants
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

ERROR_NO_PROTOCOL_TITLE = "No Protocol"
ERROR_NO_PROTOCOL_MSG = "Please select a task protocol file."


class AutoLamellaLoadTaskProtocolWidget(QtWidgets.QDialog):
    """Dialog for loading a task protocol into an existing experiment.

    Allows users to:
    - View current experiment information (read-only)
    - Select and validate a task protocol file
    - Load legacy protocols with automatic conversion
    - Attach the protocol to the experiment

    Returns:
        AutoLamellaTaskProtocol: The loaded protocol, or None if cancelled
    """

    def __init__(
        self,
        experiment: Experiment,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)

        self.experiment = experiment
        self.protocol: Optional[AutoLamellaTaskProtocol] = None
        self.protocol_path: Optional[str] = None
        self._initial_protocol_data: Optional[dict] = None
        self._initial_protocol_path: Optional[str] = None
        self._protocol_changed: bool = False

        self.setWindowTitle("Load Task Protocol")
        self.setMinimumWidth(600)

        self._setup_ui()
        self._connect_signals()
        self._update_experiment_display()
        self._load_existing_protocol()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Experiment Information (Read-only)
        exp_group = QtWidgets.QGroupBox("Experiment Information")
        exp_layout = QtWidgets.QVBoxLayout()

        # Experiment form fields (all read-only)
        exp_form_layout = QtWidgets.QFormLayout()

        self.lineEdit_experiment_name = QtWidgets.QLineEdit()
        self.lineEdit_experiment_name.setEnabled(False)

        self.lineEdit_experiment_description = QtWidgets.QLineEdit()
        self.lineEdit_experiment_description.setEnabled(False)

        self.lineEdit_experiment_directory = QtWidgets.QLineEdit()
        self.lineEdit_experiment_directory.setEnabled(False)
        self.lineEdit_experiment_directory.setCursorPosition(0)

        self.lineEdit_experiment_lamella = QtWidgets.QLineEdit()
        self.lineEdit_experiment_lamella.setEnabled(False)

        exp_form_layout.addRow("Name", self.lineEdit_experiment_name)
        exp_form_layout.addRow("Description", self.lineEdit_experiment_description)
        exp_form_layout.addRow("Directory", self.lineEdit_experiment_directory)
        exp_form_layout.addRow("Lamella", self.lineEdit_experiment_lamella)

        exp_layout.addLayout(exp_form_layout)

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
        self.btn_select_protocol.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
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

        # Protocol warning/info label
        self.label_protocol_warning = QtWidgets.QLabel(
            "⚠ Warning: Loading a new protocol will overwrite the existing protocol in this experiment."
        )
        self.label_protocol_warning.setStyleSheet("color: orange; font-style: italic; font-size: 10px;")
        self.label_protocol_warning.setWordWrap(True)
        self.label_protocol_warning.setVisible(False)
        protocol_layout.addWidget(self.label_protocol_warning)

        protocol_info_label = QtWidgets.QLabel(
            "Note: You will be able to edit the protocol after loading it into the experiment."
        )
        protocol_info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        protocol_info_label.setWordWrap(True)
        protocol_layout.addWidget(protocol_info_label)

        protocol_group.setLayout(protocol_layout)
        main_layout.addWidget(protocol_group)

        # Dialog buttons (OK/Cancel)
        button_box = QtWidgets.QDialogButtonBox()

        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_ok.setDefault(True)
        self.btn_ok.setEnabled(False)  # Disabled until new protocol is loaded

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_box.addButton(self.btn_ok, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_select_protocol.clicked.connect(self._select_protocol)
        self.btn_select_legacy_protocol.clicked.connect(self._select_legacy_protocol)
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)

    def _update_experiment_display(self):
        """Update the experiment information display."""
        self.lineEdit_experiment_name.setText(self.experiment.name or "")

        # Clear placeholder and show empty text if no description
        self.lineEdit_experiment_description.setPlaceholderText("")
        self.lineEdit_experiment_description.setText(self.experiment.description or "")

        self.lineEdit_experiment_directory.setText(str(self.experiment.path) or "")
        self.lineEdit_experiment_directory.setCursorPosition(0)
        self.lineEdit_experiment_lamella.setText(str(len(self.experiment.positions)))

    def _load_existing_protocol(self):
        """Load and display the existing protocol from the experiment if available."""
        if self.experiment.task_protocol is not None:
            existing_protocol = self.experiment.task_protocol
            protocol_file_path = os.path.join(self.experiment.path, "protocol.yaml")

            self.protocol = existing_protocol
            self.protocol_path = protocol_file_path if os.path.exists(protocol_file_path) else None
            self._initial_protocol_path = self.protocol_path
            self._initial_protocol_data = existing_protocol.to_dict()
            self._protocol_changed = False

            self._update_protocol_display()
            self.btn_ok.setEnabled(True)
            logging.info(f"Displaying existing protocol: {existing_protocol.name}")
        else:
            self._update_protocol_display()

    def _select_protocol(self):
        """Open dialog to select a protocol file."""
        # Open file dialog at the directory of the current protocol or default
        protocol_path = fui.open_existing_file_dialog(
            msg="Select a task protocol file (*.yaml)",
            path=str(cfg.TASK_PROTOCOL_PATH),
            parent=self,
        )

        if not protocol_path or protocol_path == "":
            return

        # Validate the protocol file
        try:
            loaded_protocol = AutoLamellaTaskProtocol.load(protocol_path)
        except Exception as e:
            logging.error(f"Failed to load protocol: {e}")
            self.btn_ok.setEnabled(False)  # Keep OK button disabled on error
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_PROTOCOL_TITLE,
                ERROR_INVALID_PROTOCOL_MSG.format(error=e)
            )
            return

        self.protocol = loaded_protocol
        self.protocol_path = protocol_path
        self._protocol_changed = not self._is_protocol_same_as_initial(loaded_protocol)
        if not self._protocol_changed and self._initial_protocol_path:
            self.protocol_path = self._initial_protocol_path
        self.btn_ok.setEnabled(True)  # Enable OK button when new protocol is loaded
        self._update_protocol_display()
        logging.info(f"Protocol loaded successfully from {protocol_path}")

    def _select_legacy_protocol(self):
        """Open dialog to select a legacy protocol file and convert it."""
        # Open file dialog at the protocol path
        protocol_path = fui.open_existing_file_dialog(
            msg="Select a legacy protocol file (*.yaml)",
            path=str(cfg.PROTOCOL_PATH),
            parent=self,
        )

        if not protocol_path or protocol_path == "":
            return

        # Validate and convert the legacy protocol file
        try:
            loaded_protocol = AutoLamellaTaskProtocol.load_from_old_protocol(Path(protocol_path))
        except Exception as e:
            logging.error(f"Failed to load legacy protocol: {e}")
            self.btn_ok.setEnabled(False)  # Keep OK button disabled on error
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_LEGACY_PROTOCOL_TITLE,
                ERROR_INVALID_LEGACY_PROTOCOL_MSG.format(error=e)
            )
            return

        self.protocol = loaded_protocol
        self.protocol_path = protocol_path
        self._protocol_changed = not self._is_protocol_same_as_initial(loaded_protocol)
        if not self._protocol_changed and self._initial_protocol_path:
            self.protocol_path = self._initial_protocol_path
        self.btn_ok.setEnabled(True)  # Enable OK button when new protocol is loaded
        self._update_protocol_display()
        logging.info(f"Legacy protocol loaded and converted successfully from {protocol_path}")
        QtWidgets.QMessageBox.information(
            self,
            "Legacy Protocol Converted",
            "The legacy protocol has been successfully converted to the new task-based format."
        )

    def _update_protocol_display(self):
        """Update the protocol information display."""
        if self.protocol is None:
            self.lineEdit_protocol_name.setText("")
            self.lineEdit_protocol_description.setText("")
            self.lineEdit_protocol_path.setText("")
            self.lineEdit_protocol_tasks.setText("0")
            self._update_warning_visibility()
            return

        self.lineEdit_protocol_name.setText(self.protocol.name or "")
        self.lineEdit_protocol_description.setText(self.protocol.description or "")
        display_path = self.protocol_path if self.protocol_path else "(Current protocol)"
        self.lineEdit_protocol_path.setText(display_path)
        self.lineEdit_protocol_path.setCursorPosition(0)
        self.lineEdit_protocol_tasks.setText(str(len(self.protocol.task_config)))
        self._update_warning_visibility()

    def _on_ok_clicked(self):
        """Handle OK button click - validate and return protocol."""
        # Validate protocol
        if self.protocol is None:
            QtWidgets.QMessageBox.warning(
                self,
                ERROR_NO_PROTOCOL_TITLE,
                ERROR_NO_PROTOCOL_MSG
            )
            return

        protocol_save_path = os.path.join(self.experiment.path, "protocol.yaml")
        should_save = self._protocol_changed or not os.path.exists(protocol_save_path)

        if should_save:
            try:
                self.experiment.task_protocol.save(protocol_save_path)
                logging.info(f"Protocol saved to {protocol_save_path}")
            except Exception as e:
                logging.error(f"Failed to save protocol: {e}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Protocol",
                    f"Failed to save protocol to experiment directory:\n\n{e}"
                )
                return
        else:
            logging.info("Protocol unchanged; skipping save.")
            if self._initial_protocol_path:
                self.protocol_path = self._initial_protocol_path

        # Update internal state to match saved protocol
        self._initial_protocol_data = self.protocol.to_dict()
        self._initial_protocol_path = protocol_save_path if should_save else self._initial_protocol_path
        self._protocol_changed = False
        if should_save:
            self.protocol_path = protocol_save_path
        elif self._initial_protocol_path:
            self.protocol_path = self._initial_protocol_path
        self._update_protocol_display()

        logging.info(f"Protocol '{self.protocol.name}' loaded into experiment '{self.experiment.name}'")

        # Accept the dialog
        self.accept()

    def get_protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        """Return the loaded protocol, or None if cancelled."""
        return self.protocol

    def _is_protocol_same_as_initial(self, protocol: AutoLamellaTaskProtocol) -> bool:
        if self._initial_protocol_data is None:
            return False
        try:
            return protocol.to_dict() == self._initial_protocol_data
        except Exception:
            return False

    def _update_warning_visibility(self) -> None:
        show_warning = self._protocol_changed and self._initial_protocol_data is not None
        self.label_protocol_warning.setVisible(show_warning)


def load_task_protocol_dialog(
    experiment: Experiment,
    parent: Optional[QtWidgets.QWidget] = None
) -> Optional[AutoLamellaTaskProtocol]:
    """Create and execute the task protocol loading dialog.

    Args:
        experiment: The experiment to load the protocol into
        parent: Parent widget for the dialog

    Returns:
        AutoLamellaTaskProtocol: The loaded protocol with protocol attached to experiment, or None if cancelled
    """
    # Create and show the dialog
    dialog = AutoLamellaLoadTaskProtocolWidget(experiment=experiment, parent=parent)
    result = dialog.exec_()

    # Handle the result
    if result == QtWidgets.QDialog.Accepted:
        protocol = dialog.get_protocol()
        if protocol:
            logging.info(f"Protocol loaded: {protocol.name}")
            logging.info(f"Number of tasks: {len(protocol.task_config)}")
            logging.info(f"Protocol attached to experiment: {experiment.name}")
        return protocol
    else:
        logging.info("Protocol loading cancelled")
        return None


def main():
    """Test the AutoLamellaLoadTaskProtocolWidget."""
    import napari
    viewer = napari.Viewer()

    qwidget = QtWidgets.QWidget()
    viewer.window.add_dock_widget(qwidget, area='right')

    # Create a test experiment
    from fibsem.applications.autolamella.structures import Experiment
    experiment = Experiment.create(
        path=Path("/tmp"),
        name="Test Experiment",
        metadata={"description": "Test experiment for protocol loading"}
    )

    # Use the standalone function to load protocol
    protocol = load_task_protocol_dialog(experiment=experiment, parent=qwidget)

    if protocol:
        print(f"Protocol loaded: {protocol.name}")
        print(f"Number of tasks: {len(protocol.task_config)}")
        print(f"Experiment now has protocol: {experiment.task_protocol is not None}")
    else:
        print("Protocol loading cancelled")

    napari.run()


if __name__ == "__main__":
    main()
