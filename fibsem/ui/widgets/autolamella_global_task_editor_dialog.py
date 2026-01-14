from __future__ import annotations
import copy
import logging
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.constants import SI_TO_MICRO, MICRO_TO_SI
from fibsem.applications.autolamella.structures import Experiment
from fibsem.structures import ReferenceImageParameters
from fibsem.ui.stylesheets import BLUE_PUSHBUTTON_STYLE
from fibsem.ui.widgets.reference_image_parameters_widget import ReferenceImageParametersWidget


class AutoLamellaGlobalTaskEditDialog(QDialog):
    """Dialog for globally editing reference imaging and milling field of view across all tasks."""

    def __init__(self, experiment: Experiment, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.experiment = experiment
        self.setWindowTitle("Global Edit - Reference Imaging & Milling FoV")
        self.setModal(True)

        self._create_widgets()
        self._setup_layout()
        self._initialize_values()

        # Set minimum width
        self.setMinimumWidth(500)

    def _create_widgets(self):
        """Create the widgets for the dialog."""
        # Reference image parameters widget
        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)

        # Milling field of view group
        self.milling_fov_group = QGroupBox("Milling")
        self.milling_fov_group.setFlat(True)

        self.label_milling_fov = QLabel("Field of View")
        self.label_milling_fov.setToolTip("Field of view for all milling tasks (in microns)")

        self.spinbox_milling_fov = QDoubleSpinBox()
        self.spinbox_milling_fov.setRange(0.001, 10000)
        self.spinbox_milling_fov.setDecimals(1)
        self.spinbox_milling_fov.setSingleStep(5.0)
        self.spinbox_milling_fov.setValue(150.0)
        self.spinbox_milling_fov.setSuffix(" μm")
        self.spinbox_milling_fov.setToolTip("Field of view for all milling tasks (in microns)")
        self.spinbox_milling_fov.setKeyboardTracking(False)

        # Layout for milling group
        milling_fov_layout = QGridLayout()
        milling_fov_layout.addWidget(self.label_milling_fov, 0, 0)
        milling_fov_layout.addWidget(self.spinbox_milling_fov, 0, 1)
        self.milling_fov_group.setLayout(milling_fov_layout)

        # Task selection group
        self.task_selection_group = QGroupBox("Apply Changes To")
        self.task_selection_group.setFlat(True)

        task_selection_layout = QVBoxLayout()

        # Create list widget for task selection
        self.tasks_list = QListWidget()
        self.tasks_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tasks_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        # Add tasks as checkable items
        for task_name in self.experiment.task_protocol.task_config.keys():
            item = QListWidgetItem(task_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.tasks_list.addItem(item)

        # Connect item changed signal to update info label
        self.tasks_list.itemChanged.connect(lambda _: self._update_info_label())

        task_selection_layout.addWidget(self.tasks_list)

        # Add Select All / Deselect All buttons
        tasks_buttons_layout = QHBoxLayout()
        self.pushButton_select_all = QPushButton("Select All Tasks")
        self.pushButton_deselect_all = QPushButton("Deselect All Tasks")

        self.pushButton_select_all.clicked.connect(self._select_all_tasks)
        self.pushButton_deselect_all.clicked.connect(self._deselect_all_tasks)

        tasks_buttons_layout.addWidget(self.pushButton_select_all)
        tasks_buttons_layout.addWidget(self.pushButton_deselect_all)
        task_selection_layout.addLayout(tasks_buttons_layout)

        self.task_selection_group.setLayout(task_selection_layout)

        # Info label
        self.label_info = QLabel()
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.label_info.setWordWrap(True)

        # Checkbox for updating existing lamella configurations
        self.checkbox_update_existing = QCheckBox("Also update existing lamella configurations")
        self.checkbox_update_existing.setToolTip(
            "When enabled, applies these settings to existing lamella positions that have already been created. "
            "This will update the task configurations for all positions in the experiment."
        )

        # Disable checkbox if there are no positions in the experiment
        if not self.experiment.positions or len(self.experiment.positions) == 0:
            self.checkbox_update_existing.setChecked(False)
            self.checkbox_update_existing.setEnabled(False)
            self.checkbox_update_existing.setToolTip(
                "No existing lamella positions found in the experiment. "
                "This option is only available when positions have been created."
            )
        else:
            # Enable by default when positions exist
            self.checkbox_update_existing.setChecked(True)

        # Dialog buttons
        self.button_box = QDialogButtonBox(self)
        self.pushButton_apply = QPushButton("Apply to Selected Tasks")
        self.pushButton_apply.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_apply.setAutoDefault(False)  # Prevent Enter key from triggering
        self.pushButton_cancel = QPushButton("Cancel")
        self.pushButton_cancel.setAutoDefault(False)  # Prevent Enter key from triggering

        self.button_box.addButton(self.pushButton_apply, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)

        self.pushButton_apply.clicked.connect(self.accept)
        self.pushButton_cancel.clicked.connect(self.reject)

        # Ensure no button is default
        self.pushButton_apply.setDefault(False)
        self.pushButton_cancel.setDefault(False)

        # Connect signals for live updates
        self.ref_image_params_widget.settings_changed.connect(self._on_settings_changed)
        self.spinbox_milling_fov.valueChanged.connect(self._update_info_label)

        # Initialize info label after all widgets are created
        self._update_info_label()

    def _setup_layout(self):
        """Setup the dialog layout."""
        main_layout = QVBoxLayout()

        # Add description
        description = QLabel(
            "Configure reference imaging parameters and milling field of view that will be applied "
            "to selected tasks in the protocol."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-style: italic; margin-bottom: 10px;")
        main_layout.addWidget(description)

        # Add reference image parameters widget
        main_layout.addWidget(self.ref_image_params_widget)

        # Add milling FoV group
        main_layout.addWidget(self.milling_fov_group)

        # Add task selection group
        main_layout.addWidget(self.task_selection_group)

        # Add info label
        main_layout.addWidget(self.label_info)

        # Add checkbox for updating existing lamella configurations
        main_layout.addWidget(self.checkbox_update_existing)

        # Add stretch and buttons
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def _initialize_values(self):
        """Initialize widget values from the first task config with milling."""
        initialized = False

        # Try to find a task with milling first
        for task_config in self.experiment.task_protocol.task_config.values():
            if task_config.milling:
                # Set reference imaging parameters
                self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)

                # Set milling FoV from first milling task
                first_milling_key = list(task_config.milling.keys())[0]
                milling_fov_um = task_config.milling[first_milling_key].field_of_view * SI_TO_MICRO
                self.spinbox_milling_fov.setValue(milling_fov_um)
                initialized = True
                break

        # Fallback: if no milling tasks, just use first task for reference imaging
        if not initialized and self.experiment.task_protocol.task_config:
            first_task = next(iter(self.experiment.task_protocol.task_config.values()))
            self.ref_image_params_widget.update_from_settings(first_task.reference_imaging)

    def _on_settings_changed(self, settings: ReferenceImageParameters):
        """Handle changes to reference imaging settings."""
        self._update_info_label()

    def _select_all_tasks(self):
        """Select all tasks in the task list."""
        for i in range(self.tasks_list.count()):
            item = self.tasks_list.item(i)
            if item:
                item.setCheckState(Qt.Checked)  # type: ignore

    def _deselect_all_tasks(self):
        """Deselect all tasks in the task list."""
        for i in range(self.tasks_list.count()):
            item = self.tasks_list.item(i)
            if item:
                item.setCheckState(Qt.Unchecked)  # type: ignore

    def get_selected_tasks(self) -> List[str]:
        """Get list of selected task names."""
        selected = []
        for i in range(self.tasks_list.count()):
            item = self.tasks_list.item(i)
            if item and item.checkState() == Qt.Checked:  # type: ignore
                selected.append(item.text())
        return selected

    def _update_info_label(self):
        """Update the info label with the number of tasks that will be affected."""
        # Get selected tasks
        selected_tasks = self.get_selected_tasks()

        # Filter for tasks with milling
        selected_tasks_with_milling = []
        for task_name in selected_tasks:
            task_config = self.experiment.task_protocol.task_config.get(task_name)
            if task_config and task_config.milling:
                selected_tasks_with_milling.append(task_name)

        # Build info message
        info_parts = []

        if not selected_tasks:
            self.label_info.setText("No tasks selected - please select at least one task")
            self.label_info.setStyleSheet("color: orange; font-style: italic;")
            self.pushButton_apply.setEnabled(False)
            return

        # Reference imaging info (applies to all selected tasks)
        all_task_list = ", ".join(f"'{name}'" for name in selected_tasks)
        info_parts.append(
            f"Reference imaging will be updated for {len(selected_tasks)} task(s): {all_task_list}"
        )

        # Milling FoV info (applies only to selected tasks with milling)
        if selected_tasks_with_milling:
            milling_task_list = ", ".join(f"'{name}'" for name in selected_tasks_with_milling)
            info_parts.append(
                f"Milling FoV will be updated for {len(selected_tasks_with_milling)} task(s) with milling: {milling_task_list}"
            )
        else:
            info_parts.append("No selected tasks have milling configurations - only reference imaging will be updated.")

        # Combine messages
        self.label_info.setText("\n".join(info_parts))
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.pushButton_apply.setEnabled(True)

    def apply_changes(self):
        """Apply the changes to selected milling task configs."""
        # Get selected tasks
        selected_tasks = self.get_selected_tasks()

        if not selected_tasks:
            logging.warning("No tasks selected - no changes applied")
            return 0

        # Get the new values
        new_ref_imaging = self.ref_image_params_widget.get_settings()
        new_milling_fov = self.spinbox_milling_fov.value() * MICRO_TO_SI  # Convert to SI units

        # Apply to selected task configs
        updated_count = 0
        for task_name in selected_tasks:
            task_config = self.experiment.task_protocol.task_config.get(task_name)
            if not task_config:
                continue

            # Update reference imaging for selected tasks
            task_config.reference_imaging = copy.deepcopy(new_ref_imaging)

            # Update milling FoV if task has milling
            if task_config.milling:
                for milling_config in task_config.milling.values():
                    milling_config.field_of_view = new_milling_fov
                updated_count += 1


        return updated_count

    def keyPressEvent(self, event):
        """Override key press event to prevent Enter/Return from accepting the dialog."""
        # Block Enter and Return keys from accepting the dialog
        # Qt.Key_Return is the main Enter key, Qt.Key_Enter is the numpad Enter
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)