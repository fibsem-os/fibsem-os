"""Widget for loading a task protocol into an existing AutoLamella experiment."""

import logging
import os
from typing import Optional

from PyQt5 import QtWidgets

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.protocol.picker import (
    ProtocolSummary,
    pick_protocol,
)
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel

# Error message constants

ERROR_NO_PROTOCOL_TITLE = "No Protocol"
ERROR_NO_PROTOCOL_MSG = "Please select a task protocol file."


class AutoLamellaLoadTaskProtocolWidget(QtWidgets.QDialog):
    """Dialog for loading a task protocol into an existing experiment.

    Allows users to:
    - View current experiment information (read-only)
    - Select and validate a task protocol file
    - Attach the protocol to the experiment

    Returns:
        AutoLamellaTaskProtocol: The loaded protocol, or None if cancelled
    """

    def __init__(
        self, experiment: Experiment, parent: Optional[QtWidgets.QWidget] = None
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
        exp_content = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(exp_content)
        exp_layout.setContentsMargins(0, 0, 0, 0)

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

        exp_group = TitledPanel(
            "Experiment Information", content=exp_content, collapsible=False
        )
        main_layout.addWidget(exp_group)

        # Protocol Information
        protocol_content = QtWidgets.QWidget()
        protocol_layout = QtWidgets.QVBoxLayout(protocol_content)
        protocol_layout.setContentsMargins(0, 0, 0, 0)

        # Select Protocol button at top
        protocol_button_layout = QtWidgets.QHBoxLayout()
        protocol_button_layout.addStretch()

        self.btn_select_protocol = QtWidgets.QPushButton("Select Protocol")
        self.btn_select_protocol.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        protocol_button_layout.addWidget(self.btn_select_protocol)

        protocol_layout.addLayout(protocol_button_layout)

        self.protocol_summary = ProtocolSummary()
        protocol_layout.addWidget(self.protocol_summary)

        # Protocol warning/info label
        self.label_protocol_warning = QtWidgets.QLabel(
            "⚠ Warning: Loading a new protocol will overwrite the existing protocol in this experiment."
        )
        self.label_protocol_warning.setStyleSheet(
            "color: orange; font-style: italic; font-size: 10px;"
        )
        self.label_protocol_warning.setWordWrap(True)
        self.label_protocol_warning.setVisible(False)
        protocol_layout.addWidget(self.label_protocol_warning)

        protocol_info_label = QtWidgets.QLabel(
            "Note: You will be able to edit the protocol after loading it into the experiment."
        )
        protocol_info_label.setStyleSheet(
            "color: gray; font-style: italic; font-size: 10px;"
        )
        protocol_info_label.setWordWrap(True)
        protocol_layout.addWidget(protocol_info_label)

        protocol_group = TitledPanel(
            "Protocol Information", content=protocol_content, collapsible=False
        )
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
            self.protocol_path = (
                protocol_file_path if os.path.exists(protocol_file_path) else None
            )
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
        picked = pick_protocol(self, str(cfg.USER_PROTOCOL_PATH))
        if picked is None:
            return
        protocol_path, loaded_protocol = picked
        self.protocol = loaded_protocol
        self.protocol_path = protocol_path
        self._protocol_changed = not self._is_protocol_same_as_initial(loaded_protocol)
        if not self._protocol_changed and self._initial_protocol_path:
            self.protocol_path = self._initial_protocol_path
        self.btn_ok.setEnabled(True)  # Enable OK button when new protocol is loaded
        self._update_protocol_display()
    def _update_protocol_display(self):
        """Update the protocol information display."""
        self.protocol_summary.show_protocol(
            self.protocol, self.protocol_path or "(Current protocol)"
        )
        self._update_warning_visibility()
    def _on_ok_clicked(self):
        """Accept the chosen protocol. The window adopts and saves it; the
        dialog saves nothing itself."""
        if self.protocol is None:
            QtWidgets.QMessageBox.warning(
                self, ERROR_NO_PROTOCOL_TITLE, ERROR_NO_PROTOCOL_MSG
            )
            return

        logging.info(
            f"Protocol '{self.protocol.name}' loaded into experiment '{self.experiment.name}'"
        )

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
        show_warning = (
            self._protocol_changed and self._initial_protocol_data is not None
        )
        self.label_protocol_warning.setVisible(show_warning)


def load_task_protocol_dialog(
    experiment: Experiment, parent: Optional[QtWidgets.QWidget] = None
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
