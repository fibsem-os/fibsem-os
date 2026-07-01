"""Widget for loading an existing AutoLamella experiment with protocol."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from superqt import QIconifyIcon

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.config import (
    get_recent_experiments,
    load_user_preferences,
    record_recent_experiment,
    save_user_preferences,
)
from fibsem.ui import utils as fui
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET, SECONDARY_BUTTON_STYLESHEET
from fibsem.ui.widgets.custom_widgets import TitledPanel

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

# Width of the recent-experiments quick-select column
RECENT_COLUMN_WIDTH = 260

# Recent-experiment row icons (superqt / material design icons) and colours
RECENT_FOLDER_ICON = "mdi:folder-outline"
RECENT_LAMELLA_ICON = "mdi:layers-triple-outline"
RECENT_ICON_COLOR = "#9aa0ab"
RECENT_PILL_TEXT_COLOR = "#b7bcc6"
RECENT_NAME_COLOR = "#d6d6d6"


class _ElidedLabel(QtWidgets.QLabel):
    """A QLabel that elides (…) its text on the right when too narrow.

    Reports a zero minimum width so it can shrink inside a layout instead of
    forcing the row wider than the list viewport.
    """

    def __init__(self, text: str, color: str, parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self._color = QtGui.QColor(color)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, QtGui.QFontMetrics(self.font()).height())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, QtGui.QFontMetrics(self.font()).height())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self._color)
        painter.setFont(self.font())
        metrics = QtGui.QFontMetrics(self.font())
        elided = metrics.elidedText(self._full_text, QtCore.Qt.ElideMiddle, self.width())
        painter.drawText(
            self.rect(), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, elided
        )

RECENT_LIST_STYLESHEET = """
QListWidget {
    background-color: #1e2027;
    border: 1px solid #3d4251;
    border-radius: 4px;
    outline: none;
    padding: 2px;
}
QListWidget::item {
    border: none;
    border-radius: 4px;
    margin: 1px 0px;
}
QListWidget::item:selected {
    background-color: #2d72c4;
}
QListWidget::item:hover:!selected {
    background-color: #2a2e39;
}
"""


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
        self.setMinimumWidth(860)

        self._setup_ui()
        self._connect_signals()
        self._populate_recent_experiments()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Experiment Information
        exp_content = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(exp_content)
        exp_layout.setContentsMargins(0, 0, 0, 0)

        # Browse for an experiment file (placed beneath the recent list)
        self.btn_select_experiment = QtWidgets.QPushButton("Browse…")
        self.btn_select_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)

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

        exp_group = TitledPanel("Experiment Information", content=exp_content, collapsible=False)

        # Protocol Information
        protocol_content = QtWidgets.QWidget()
        protocol_layout = QtWidgets.QVBoxLayout(protocol_content)
        protocol_layout.setContentsMargins(0, 0, 0, 0)

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

        protocol_group = TitledPanel("Protocol Information", content=protocol_content, collapsible=False)

        # Recent Experiments quick-select (left column)
        recent_content = QtWidgets.QWidget()
        recent_layout = QtWidgets.QVBoxLayout(recent_content)
        recent_layout.setContentsMargins(0, 0, 0, 0)

        self.list_recent_experiments = QtWidgets.QListWidget()
        self.list_recent_experiments.setStyleSheet(RECENT_LIST_STYLESHEET)
        self.list_recent_experiments.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_recent_experiments.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        recent_layout.addWidget(self.list_recent_experiments)

        # Manual browse button sits beneath the recent list
        recent_layout.addWidget(self.btn_select_experiment)

        recent_group = TitledPanel("Recent Experiments", content=recent_content, collapsible=False)
        recent_group.setFixedWidth(RECENT_COLUMN_WIDTH)

        # Right column: experiment + protocol information
        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(exp_group)
        right_layout.addWidget(protocol_group)

        # Assemble the two columns side by side
        columns_layout = QtWidgets.QHBoxLayout()
        columns_layout.addWidget(recent_group)
        columns_layout.addWidget(right_column, stretch=1)
        main_layout.addLayout(columns_layout)

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
        self.list_recent_experiments.itemClicked.connect(self._on_recent_experiment_clicked)
        self.list_recent_experiments.itemDoubleClicked.connect(
            self._on_recent_experiment_double_clicked
        )

    def _select_experiment(self):
        """Open dialog to select an experiment file."""
        # Default to last used experiment directory if available
        default_path = str(cfg.LOG_PATH)
        prefs = load_user_preferences()
        last_path = prefs.experiment.last_experiment_path
        if last_path and os.path.exists(last_path):
            default_path = last_path

        experiment_path = fui.open_existing_file_dialog(
            msg="Select an experiment file (experiment.yaml)",
            path=default_path,
            parent=self,
        )

        if not experiment_path or experiment_path == "":
            return

        self._load_experiment_from_path(experiment_path)

    def _load_experiment_from_path(self, experiment_path: str) -> bool:
        """Load and validate an experiment from a path, updating the display.

        Shared by the manual file dialog and the recent-experiments quick-select.

        Returns:
            True if the experiment was loaded (with or without a protocol),
            False if loading failed.
        """
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

            return True

        except Exception as e:
            logging.error(f"Failed to load experiment: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                ERROR_INVALID_EXPERIMENT_TITLE,
                ERROR_INVALID_EXPERIMENT_MSG.format(error=e)
            )
            self._clear_display()
            return False

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

        # Save last used experiment path
        prefs = load_user_preferences()
        prefs.experiment.last_experiment_path = str(self.experiment.path)
        save_user_preferences(prefs)

        # Record in the recent experiments quick-select list
        record_recent_experiment(os.path.join(self.experiment.path, "experiment.yaml"))

        # Accept the dialog
        self.accept()

    def _populate_recent_experiments(self) -> None:
        """Populate the recent-experiments list from user preferences."""
        self.list_recent_experiments.clear()

        recents = get_recent_experiments()

        if not recents:
            placeholder = QtWidgets.QListWidgetItem("No recent experiments")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            placeholder.setForeground(QtCore.Qt.gray)
            self.list_recent_experiments.addItem(placeholder)
            return

        for info in recents:
            item = QtWidgets.QListWidgetItem(self.list_recent_experiments)
            item.setData(QtCore.Qt.UserRole, info.path)
            item.setToolTip(info.path)
            row = self._make_recent_row_widget(info)
            # Width 0 lets the row track the list viewport width instead of the
            # row's (wide) content hint, so long names elide rather than clip.
            item.setSizeHint(QtCore.QSize(0, row.sizeHint().height()))
            self.list_recent_experiments.addItem(item)
            self.list_recent_experiments.setItemWidget(item, row)

    def _make_recent_row_widget(self, info) -> QtWidgets.QWidget:
        """Build a row widget for a recent experiment.

        Layout: folder icon | name + date (stacked) | lamella-count pill.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(10, 11, 8, 11)
        layout.setSpacing(9)

        # Leading folder icon
        folder_label = QtWidgets.QLabel()
        folder_label.setPixmap(
            QIconifyIcon(RECENT_FOLDER_ICON, color=RECENT_ICON_COLOR).pixmap(18, 18)
        )
        folder_label.setFixedWidth(18)
        layout.addWidget(folder_label, alignment=QtCore.Qt.AlignVCenter)

        # Name + date (stacked)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(5)

        name_label = _ElidedLabel(info.name, RECENT_NAME_COLOR)
        name_font = name_label.font()
        name_font.setPixelSize(13)
        name_font.setWeight(QtGui.QFont.DemiBold)
        name_label.setFont(name_font)
        name_label.setToolTip(info.name)

        if info.created_at:
            date_str = datetime.fromtimestamp(info.created_at).strftime("%Y-%m-%d")
        else:
            date_str = "unknown date"
        date_label = QtWidgets.QLabel(date_str)
        date_label.setStyleSheet("color: #8a8f99; font-size: 11px; background: transparent;")

        text_layout.addWidget(name_label)
        text_layout.addWidget(date_label)
        layout.addLayout(text_layout, stretch=1)

        # Trailing lamella-count pill
        layout.addWidget(
            self._make_lamella_pill(info.num_lamella), alignment=QtCore.Qt.AlignVCenter
        )
        return widget

    def _make_lamella_pill(self, num_lamella: int) -> QtWidgets.QWidget:
        """Build a small rounded pill showing a layers icon and the lamella count."""
        pill = QtWidgets.QFrame()
        pill.setObjectName("lamellaPill")
        pill.setStyleSheet("#lamellaPill { background: #2a2e39; border-radius: 9px; }")

        pill_layout = QtWidgets.QHBoxLayout(pill)
        pill_layout.setContentsMargins(7, 2, 8, 2)
        pill_layout.setSpacing(3)

        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(
            QIconifyIcon(RECENT_LAMELLA_ICON, color=RECENT_PILL_TEXT_COLOR).pixmap(12, 12)
        )
        count_label = QtWidgets.QLabel(str(num_lamella))
        count_label.setStyleSheet(
            f"color: {RECENT_PILL_TEXT_COLOR}; font-size: 11px; "
            "font-weight: 600; background: transparent;"
        )

        pill_layout.addWidget(icon_label)
        pill_layout.addWidget(count_label)
        return pill

    def _on_recent_experiment_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Load the selected recent experiment into the form (single click)."""
        path = item.data(QtCore.Qt.UserRole)
        if not path:
            return
        self._load_experiment_from_path(path)

    def _on_recent_experiment_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Load the selected recent experiment and accept immediately (double click)."""
        path = item.data(QtCore.Qt.UserRole)
        if not path:
            return
        if self._load_experiment_from_path(path):
            # Only accept if the experiment loaded with a valid protocol
            if self.experiment is not None and self.experiment.task_protocol is not None:
                self._on_ok_clicked()

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
