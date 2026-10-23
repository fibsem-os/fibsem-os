"""The protocol's identity on the Protocol tab: one line, and a pencil.

Name, description and version are set when the experiment is created and rarely
touched after, so they stay out of the only column that has to fit two task lists.
The line says which protocol is loaded; the dialog behind the pencil does the rest.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.constants import SI_TO_MICRO
from fibsem.structures import (
    FibsemMillingSettings,
    ReferenceImageParameters,
)
from fibsem.theme.tokens import NEUTRAL_200, TEXT_MUTED_COLOR
from fibsem.ui.utils import beam_spec
from fibsem.ui.widgets.checklist import ApplyDialog, TaskChecklist
from fibsem.ui.widgets.custom_widgets import (
    ElidedLabel,
    IconToolButton,
    TitledPanel,
    ValueSpinBox,
)
from fibsem.ui.widgets.form_builder import configure_spinbox
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)


class AutoLamellaGlobalTaskEditDialog(ApplyDialog):
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

        # Milling field of view panel
        milling_content = QWidget()
        milling_fov_layout = QGridLayout(milling_content)
        milling_fov_layout.setContentsMargins(0, 0, 0, 0)

        self.label_milling_fov = QLabel("Field of View")
        self.label_milling_fov.setToolTip(
            "Field of view for all milling tasks (in microns)"
        )

        # shown as the milling field of view is, in the range every beam declares
        self.spinbox_milling_fov = ValueSpinBox()
        configure_spinbox(
            self.spinbox_milling_fov,
            beam_spec(FibsemMillingSettings.params["hfw"]),
        )
        self.spinbox_milling_fov.setValue(150.0)
        self.spinbox_milling_fov.setToolTip(
            "Field of view for all milling tasks (in microns)"
        )
        self.spinbox_milling_fov.setKeyboardTracking(False)

        milling_fov_layout.addWidget(self.label_milling_fov, 0, 0)
        milling_fov_layout.addWidget(self.spinbox_milling_fov, 0, 1)

        self.milling_fov_group = TitledPanel(
            "Milling", content=milling_content, collapsible=False
        )

        # Task selection panel
        self.task_checklist = TaskChecklist(
            "Apply Changes To",
            select_label="Select All Tasks",
            deselect_label="Deselect All Tasks",
        )
        self.task_checklist.set_items(list(self.experiment.task_protocol.task_config))
        self.task_checklist.changed.connect(self._update_info_label)

        # Info label
        self.label_info = QLabel()
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.label_info.setWordWrap(True)

        # Checkbox for updating existing lamella configurations
        self.checkbox_update_existing = QCheckBox(
            "Also update existing lamella configurations"
        )
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

        self.button_box = self.apply_buttons("Apply to Selected Tasks")

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
        main_layout.addWidget(self.task_checklist)

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
                self.ref_image_params_widget.update_from_settings(
                    task_config.reference_imaging
                )

                # Set milling FoV from first milling task
                first_milling_key = list(task_config.milling.keys())[0]
                milling_fov_um = (
                    task_config.milling[first_milling_key].field_of_view * SI_TO_MICRO
                )
                self.spinbox_milling_fov.setValue(milling_fov_um)
                initialized = True
                break

        # Fallback: if no milling tasks, just use first task for reference imaging
        if not initialized and self.experiment.task_protocol.task_config:
            first_task = next(iter(self.experiment.task_protocol.task_config.values()))
            self.ref_image_params_widget.update_from_settings(
                first_task.reference_imaging
            )

    def _on_settings_changed(self, settings: ReferenceImageParameters):
        """Handle changes to reference imaging settings."""
        self._update_info_label()

    def get_selected_tasks(self) -> List[str]:
        """Get list of selected task names."""
        return self.task_checklist.checked()

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
            self.label_info.setText(
                "No tasks selected - please select at least one task"
            )
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
            milling_task_list = ", ".join(
                f"'{name}'" for name in selected_tasks_with_milling
            )
            info_parts.append(
                f"Milling FoV will be updated for {len(selected_tasks_with_milling)} task(s) with milling: {milling_task_list}"
            )
        else:
            info_parts.append(
                "No selected tasks have milling configurations - only reference imaging will be updated."
            )

        # Combine messages
        self.label_info.setText("\n".join(info_parts))
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.pushButton_apply.setEnabled(True)

    def milling_fov(self) -> float:
        """The milling field of view chosen, in metres."""
        return self.spinbox_milling_fov.value() / SI_TO_MICRO


_FIELDS = ("name", "description", "version")


class ProtocolHeaderWidget(QWidget):
    """`AutoLamella Protocol · v1.0`, and a pencil."""

    edit_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 0)
        layout.setSpacing(6)
        self.label = ElidedLabel()
        self.label.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        layout.addWidget(self.label, 1)
        # A plain label: an eliding one beside a stretched name has no width left.
        self.version_label = QLabel()
        self.version_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.version_label)
        self.btn_edit = IconToolButton(
            icon="mdi:pencil-outline",
            tooltip="Edit the protocol's name, description and version",
            size=24,
        )
        self.btn_edit.clicked.connect(self.edit_clicked)
        layout.addWidget(self.btn_edit)

    def update_from_protocol(self, protocol: AutoLamellaTaskProtocol) -> None:
        self.label.setText(protocol.name or "Untitled protocol")
        version = protocol.version or ""
        self.version_label.setText(f"v{version}" if version else "")
        self.setToolTip(protocol.description or "")


class ProtocolDetailsDialog(QDialog):
    """Name, description and version, for the pencil."""

    def __init__(
        self, protocol: AutoLamellaTaskProtocol, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Protocol details")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edits: Dict[str, QLineEdit] = {}
        for field in _FIELDS:
            edit = QLineEdit(getattr(protocol, field, "") or "")
            form.addRow(field.capitalize(), edit)
            self.edits[field] = edit
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, str]:
        return {field: edit.text().strip() for field, edit in self.edits.items()}


class ApplyLamellaConfigDialog(ApplyDialog):
    """Dialog for applying a lamella's task configuration to other lamella."""

    def __init__(
        self,
        source_name: str,
        other_lamella_names: list[str],
        task_names: list[str],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.source_name = source_name

        self.setWindowTitle(f"Apply Config from '{source_name}'")
        self.setModal(True)
        self.setMinimumWidth(450)

        self._other_lamella_names = other_lamella_names
        self._task_names = task_names

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        """Create the dialog widgets."""
        # Description
        self.label_description = QLabel(
            f"Apply task configurations from <b>{self.source_name}</b> "
            f"to other lamella. Existing milling pattern positions will be preserved."
        )
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet("font-style: italic; margin-bottom: 10px;")

        self.lamella_checklist = TaskChecklist("Target Lamella")
        self.lamella_checklist.set_items(self._other_lamella_names)
        self.task_checklist = TaskChecklist("Tasks to Apply")
        self.task_checklist.set_items(self._task_names)

        # --- Base protocol checkbox ---
        self.checkbox_update_base_protocol = QCheckBox("Also update the base protocol")
        self.checkbox_update_base_protocol.setToolTip(
            "When enabled, also updates the base protocol's task configurations "
            "so that new lamella created in the future will use these settings."
        )

        # --- Info label ---
        self.label_info = QLabel()
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.label_info.setWordWrap(True)

        self.button_box = self.apply_buttons("Apply")

        # Update info when selections change (must be after buttons are created)
        self.lamella_checklist.changed.connect(self._update_info_label)
        self.task_checklist.changed.connect(self._update_info_label)
        self.checkbox_update_base_protocol.stateChanged.connect(
            lambda _: self._update_info_label()
        )
        self._update_info_label()

    def _setup_layout(self):
        """Setup the dialog layout."""
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_description)
        main_layout.addWidget(self.lamella_checklist)
        main_layout.addWidget(self.task_checklist)
        main_layout.addWidget(self.checkbox_update_base_protocol)
        main_layout.addWidget(self.label_info)
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def get_selected_lamella_names(self) -> list[str]:
        """Get list of selected target lamella names."""
        return self.lamella_checklist.checked()

    def get_selected_tasks(self) -> list[str]:
        """Get list of selected task names."""
        return self.task_checklist.checked()

    def get_update_base_protocol(self) -> bool:
        """Whether to also update the base protocol."""
        return self.checkbox_update_base_protocol.isChecked()

    def _update_info_label(self):
        """Update the info label with summary of what will be applied."""
        selected_lamella_names = self.get_selected_lamella_names()
        selected_tasks = self.get_selected_tasks()

        if not selected_lamella_names or not selected_tasks:
            missing = []
            if not selected_lamella_names:
                missing.append("target lamella")
            if not selected_tasks:
                missing.append("tasks")
            self.label_info.setText(
                f"Please select at least one {' and '.join(missing)}."
            )
            self.label_info.setStyleSheet("color: orange; font-style: italic;")
            self.pushButton_apply.setEnabled(False)
            return

        parts = []
        task_list = ", ".join(f"'{t}'" for t in selected_tasks)
        lamella_list = ", ".join(selected_lamella_names)
        parts.append(
            f"{len(selected_tasks)} task(s) ({task_list}) will be applied to "
            f"{len(selected_lamella_names)} lamella ({lamella_list})."
        )
        if self.get_update_base_protocol():
            parts.append("The base protocol will also be updated.")

        self.label_info.setText(" ".join(parts))
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.pushButton_apply.setEnabled(True)
