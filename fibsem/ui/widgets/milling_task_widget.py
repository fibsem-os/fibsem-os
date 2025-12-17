from typing import Dict, Optional, Set

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.stylesheets import GREEN_PUSHBUTTON_STYLE, RED_PUSHBUTTON_STYLE
from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget


# TODO: be able to sync the position of different tasks


class TaskNameDialog(QDialog):
    """Dialog for entering a task name with duplicate validation."""

    def __init__(self, existing_names: Set[str], parent: Optional[QWidget] = None):
        """Initialize the task name dialog.

        Args:
            existing_names: Set of existing task names to check for duplicates
            parent: Parent widget
        """
        super().__init__(parent)
        self.existing_names = existing_names
        self.setWindowTitle("Add New Task")
        self._setup_ui()

    def _setup_ui(self):
        """Create and configure UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Form layout for input
        form_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter task name...")
        self.name_input.textChanged.connect(self._validate_name)
        form_layout.addRow("Task Name:", self.name_input)
        layout.addLayout(form_layout)

        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initially disable OK button
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def _validate_name(self, text: str):
        """Validate the task name and update UI accordingly.

        Args:
            text: Current text in the input field
        """
        ok_button = self.button_box.button(QDialogButtonBox.Ok)

        if not text:
            # Empty name
            ok_button.setEnabled(False)
            self.warning_label.setVisible(False)
        elif text in self.existing_names:
            # Duplicate name
            ok_button.setEnabled(False)
            self.warning_label.setText(f"Task name '{text}' already exists")
            self.warning_label.setVisible(True)
        else:
            # Valid name
            ok_button.setEnabled(True)
            self.warning_label.setVisible(False)

    def get_task_name(self) -> str:
        """Get the entered task name.

        Returns:
            The task name entered by the user
        """
        return self.name_input.text().strip()

class FibsemMillingTaskWidget(QWidget):
    """Widget for selecting and configuring milling tasks.

    Contains a list widget for task selection and a configuration widget
    for editing the selected task's settings.
    """

    task_config_updated = pyqtSignal(str, FibsemMillingTaskConfig)  # task_name, config
    task_config_removed = pyqtSignal(str)  # task_name
    task_configs_changed = pyqtSignal(dict)  # all task configs
    task_selection_changed = pyqtSignal(str)  # task_name

    def __init__(
        self,
        microscope: FibsemMicroscope,
        task_configs: Optional[Dict[str, FibsemMillingTaskConfig]] = None,
        milling_enabled: bool = True,
        correlation_enabled: bool = True,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the milling task widget.

        Args:
            microscope: FibsemMicroscope instance
            task_configs: Dictionary of task configurations with task names as keys
            milling_enabled: Whether to show milling controls
            parent: Parent widget
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.microscope = microscope
        self._task_configs = task_configs or {}
        self._milling_enabled = milling_enabled
        self._correlation_enabled = correlation_enabled

        self._setup_ui()
        self._connect_signals()

        # Load initial task configs
        if self._task_configs:
            self._populate_task_list()

    @property
    def _current_task_name(self) -> Optional[str]:
        return self.task_list.currentText()

    def _setup_ui(self):
        """Create and configure all UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Task selection with label and buttons
        grid_layout = QGridLayout()
        self.task_label = QLabel("Milling Task")
        self.task_list = QComboBox()
        self.btn_add_task = QPushButton("+")
        self.btn_add_task.setMaximumWidth(30)
        self.btn_add_task.setToolTip("Add new task")
        self.btn_add_task.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.btn_remove_task = QPushButton("-")
        self.btn_remove_task.setMaximumWidth(30)
        self.btn_remove_task.setToolTip("Remove selected task")
        self.btn_remove_task.setStyleSheet(RED_PUSHBUTTON_STYLE)

        grid_layout.addWidget(self.task_label, 0, 0)
        grid_layout.addWidget(self.task_list, 0, 1)
        grid_layout.addWidget(self.btn_add_task, 0, 2)
        grid_layout.addWidget(self.btn_remove_task, 0, 3)
        layout.addLayout(grid_layout)

        # Task configuration widget
        self.config_widget = MillingTaskConfigWidget(
            microscope=self.microscope,
            milling_enabled=self._milling_enabled,
            correlation_enabled=self._correlation_enabled,
            parent=self,
        )
        layout.addWidget(self.config_widget)

        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.task_list.currentIndexChanged.connect(self._on_task_selection_changed)
        self.config_widget.settings_changed.connect(self._on_config_changed)
        self.btn_add_task.clicked.connect(self._on_add_task)
        self.btn_remove_task.clicked.connect(self._on_remove_task)

    def _populate_task_list(self):
        """Populate the task list with available task configurations."""
        self.task_list.clear()
        for task_name in self._task_configs.keys():
            self.task_list.addItem(task_name)

        # Select the first task if available
        if self.task_list.count() > 0:
            self.task_list.setCurrentIndex(0)

        # Show task selection controls only if there are tasks
        visible = self.task_list.count() > 0
        self.task_label.setVisible(visible)
        self.task_list.setVisible(visible)
        self.btn_add_task.setVisible(True)
        self.btn_remove_task.setVisible(self.task_list.count() > 1)  # need at least one task to remove

        self.task_label.setVisible(False)
        self.task_list.setVisible(False)
        self.btn_remove_task.setVisible(False)
        self.btn_add_task.setVisible(False)

    def _on_task_selection_changed(self, index: int):
        """Handle task selection changes."""
        if index < 0:
            return

        task_name = self.task_list.currentText()

        # Load the selected task configuration
        if task_name in self._task_configs:
            # Update background milling stages with other tasks
            self._update_background_milling_stages()

            # update config widget with selected task settings
            self.config_widget.set_config(self._task_configs[task_name])
            # NOTE: selection seems to cause a double draw in stage editor widget

    def _on_config_changed(self, config: FibsemMillingTaskConfig):
        """Handle configuration changes in the sub-widget."""

        current_task_name = self._current_task_name

        if current_task_name:
            # Update the stored configuration
            self._task_configs[current_task_name] = config
            self.task_config_updated.emit(current_task_name, config)

        print(f"Task Config changed: {self._current_task_name} {config.name}, {len(config.stages)} stages")
        self.task_configs_changed.emit(self._task_configs)

    def _on_add_task(self):
        """Handle adding a new task."""
        # Show custom dialog to get task name
        dialog = TaskNameDialog(existing_names=set(self._task_configs.keys()), parent=self)

        if dialog.exec_() == QDialog.Accepted:
            task_name = dialog.get_task_name()

            if task_name:
                # Create a new default task configuration
                new_config = FibsemMillingTaskConfig(name=task_name)
                self.add_task_config(task_name, new_config)

                # Select the newly added task
                self.select_task(task_name)

    def _on_remove_task(self):
        """Handle removing the selected task."""
        if not self._current_task_name:
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the task '{self._current_task_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.remove_task_config(self._current_task_name)

    def set_task_configs(self, task_configs: Dict[str, FibsemMillingTaskConfig]):
        """Set the available task configurations.

        Args:
            task_configs: Dictionary of task configurations with task names as keys
        """
        self._task_configs = task_configs.copy()
        self._populate_task_list()

    def get_task_configs(self) -> Dict[str, FibsemMillingTaskConfig]:
        """Get all task configurations.

        Returns:
            Dictionary of task configurations with task names as keys
        """
        return self._task_configs.copy()

    def get_current_task_config(self) -> Optional[FibsemMillingTaskConfig]:
        """Get the currently selected task configuration.

        Returns:
            The currently selected task configuration, or None if no task is selected
        """
        if self._current_task_name and self._current_task_name in self._task_configs:
            return self._task_configs[self._current_task_name]
        return None

    def get_current_task_name(self) -> Optional[str]:
        """Get the name of the currently selected task.

        Returns:
            The name of the currently selected task, or None if no task is selected
        """
        return self._current_task_name

    def select_task(self, task_name: str):
        """Programmatically select a task by name.

        Args:
            task_name: Name of the task to select
        """
        if task_name not in self._task_configs:
            return

        # Find and select the item
        index = self.task_list.findText(task_name)
        if index >= 0:
            self.task_list.setCurrentIndex(index)

    def add_task_config(self, task_name: str, config: FibsemMillingTaskConfig):
        """Add a new task configuration.

        Args:
            task_name: Name of the task
            config: Task configuration
        """
        self._task_configs[task_name] = config
        self._populate_task_list()

    def remove_task_config(self, task_name: str):
        """Remove a task configuration.

        Args:
            task_name: Name of the task to remove
        """
        if task_name in self._task_configs:
            del self._task_configs[task_name]
            self._populate_task_list()

            # emit signal that task selection changed
            self.task_configs_changed.emit(self._task_configs)

    def update_task_config(self, task_name: str, config: FibsemMillingTaskConfig):
        """Update an existing task configuration.

        Args:
            task_name: Name of the task to update
            config: New task configuration
        """
        if task_name in self._task_configs:
            self._task_configs[task_name] = config

            # If this is the currently selected task, update the widget
            if self._current_task_name == task_name:
                self.config_widget.update_from_settings(config)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in the configuration widget.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self.config_widget.set_show_advanced(show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self.config_widget.get_show_advanced()

    def _update_background_milling_stages(self):
        """Update background milling stages with stages from other tasks."""
        if not self._current_task_name:
            return

        background_stages = []
        for task_name, task_config in self._task_configs.items():
            if task_name != self._current_task_name:
                background_stages.extend(task_config.stages)

        self.config_widget.set_background_milling_stages(background_stages)

    def set_movement_lock(self, locked: bool):
        """Set whether movement is locked in the milling stage editor.

        Args:
            locked: True to lock movement, False to unlock
        """
        self.config_widget.milling_editor_widget.set_movement_lock(locked)

if __name__ == "__main__":
    import os
    from pathlib import Path

    import napari
    from PyQt5.QtWidgets import QTabWidget, QWidget

    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
    )
    from fibsem.applications.autolamella.config import TASK_PROTOCOL_PATH

    from fibsem import utils
    from PyQt5.QtWidgets import QVBoxLayout

    viewer = napari.Viewer()
    main_widget = QTabWidget()

    # set tab to side
    qwidget = QWidget()
    icon1 = QIconifyIcon("material-symbols:experiment", color="white")
    main_widget.addTab(qwidget, icon1, "Experiment")  # type: ignore
    layout = QVBoxLayout()
    qwidget.setLayout(layout)
    qwidget.setContentsMargins(0, 0, 0, 0)
    layout.setContentsMargins(0, 0, 0, 0)

    microscope, settings = utils.setup_session()

    protocol = AutoLamellaTaskProtocol.load(TASK_PROTOCOL_PATH)
    task_configs = protocol.task_config["Rough Milling"].milling

    task_widget = FibsemMillingTaskWidget(
        microscope=microscope,
        task_configs=task_configs,
        milling_enabled=False
    )
    layout.addWidget(task_widget)

    # Connect to settings change signal
    def on_task_config_changed(task_name: str, task_config: FibsemMillingTaskConfig):
        print(f"Task Config changed: {task_name} {utils.current_timestamp_v3(timeonly=False)}")
        print(f"  name: {task_config.name}")
        print(f"  field_of_view: {task_config.field_of_view}")
        print(f"  alignment: {task_config.alignment}")
        print(f"  acquisition: {task_config.acquisition}")
        print(f"  stages: {len(task_config.stages)} stages")
        for stage in task_config.stages:
            print(f"    Stage Name: {stage.name}")
            print(f"    Milling: {stage.milling}")
            print(f"    Pattern: {stage.pattern}")
            print(f"    Strategy: {stage.strategy.config}")
            print("---------------------" * 3)

    task_widget.task_config_updated.connect(on_task_config_changed)
    main_widget.setWindowTitle("MillingTaskConfig Widget Test")

    viewer.window.add_dock_widget(main_widget, add_vertical_stretch=False, area="right")

    napari.run()
