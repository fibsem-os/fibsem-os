
import copy
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import napari
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from superqt import QCollapsible

from fibsem.utils import format_value
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.workflows.tasks.tasks import TASK_REGISTRY
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, ReferenceImageParameters
from fibsem.ui.stylesheets import BLUE_PUSHBUTTON_STYLE, GREEN_PUSHBUTTON_STYLE, RED_PUSHBUTTON_STYLE
from fibsem.ui.widgets.autolamella_global_task_editor_dialog import AutoLamellaGlobalTaskEditDialog
from fibsem.ui.widgets.autolamella_task_config_widget import (
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import AutoLamellaProtocolEditorWidget
from fibsem.ui.widgets.autolamella_workflow_widget import AutoLamellaWorkflowWidget
from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget
from fibsem.ui.widgets.reference_image_parameters_widget import ReferenceImageParametersWidget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


class AddTaskDialog(QDialog):
    """Dialog for adding a new task to the protocol."""

    def __init__(self, existing_task_config: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.existing_task_config = existing_task_config
        self.setWindowTitle("Add New Task")
        self.setModal(True)

        # Create widgets
        self.label_task_type = QLabel("Task Type:")
        self.comboBox_task_type = QComboBox()

        # Populate task types from TASK_REGISTRY
        for task_type, task_cls in TASK_REGISTRY.items():
            display_name = task_cls.config_cls.display_name
            self.comboBox_task_type.addItem(f"{display_name} ({task_type})", task_type)

        self.label_task_name = QLabel("Task Name:")
        self.lineEdit_task_name = QLineEdit()

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange; font-weight: bold;")

        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        form_layout = QGridLayout()
        form_layout.addWidget(self.label_task_type, 0, 0)
        form_layout.addWidget(self.comboBox_task_type, 0, 1)
        form_layout.addWidget(self.label_task_name, 1, 0)
        form_layout.addWidget(self.lineEdit_task_name, 1, 1)
        layout.addLayout(form_layout)
        layout.addWidget(self.label_warning)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Connect signals
        self.lineEdit_task_name.textChanged.connect(self.validate_task_name)
        self.comboBox_task_type.currentIndexChanged.connect(self.update_default_task_name)

        # Set default task name
        self.update_default_task_name()

        # Set minimum width
        self.setMinimumWidth(400)

    def update_default_task_name(self):
        """Update the default task name based on the selected task type."""
        task_type = self.comboBox_task_type.currentData()
        if task_type and task_type in TASK_REGISTRY:
            display_name = TASK_REGISTRY[task_type].config_cls.display_name
            self.lineEdit_task_name.setText(display_name)

    def validate_task_name(self):
        """Validate the task name and show warning if it already exists."""
        task_name = self.lineEdit_task_name.text().strip()

        if not task_name:
            self.label_warning.setText("")
            return False

        if task_name in self.existing_task_config:
            self.label_warning.setText(f"⚠ Warning: Task name '{task_name}' already exists!")
            return False

        self.label_warning.setText("")
        return True

    def validate_and_accept(self):
        """Validate inputs before accepting the dialog."""
        task_name = self.lineEdit_task_name.text().strip()

        if not task_name:
            self.label_warning.setText("⚠ Warning: Task name cannot be empty!")
            return

        if not self.validate_task_name():
            return

        self.accept()

    def get_task_info(self) -> Tuple[str, str]:
        """Get the selected task type and name."""
        task_type = self.comboBox_task_type.currentData()
        task_name = self.lineEdit_task_name.text().strip()
        return task_type, task_name


class AutoLamellaProtocolTaskConfigEditor(QWidget):
    """A widget to edit the AutoLamella protocol."""

    def __init__(self, 
                viewer: napari.Viewer,
                 parent: 'AutoLamellaUI'):
        super().__init__(parent)
        self.parent_widget = parent
        self.viewer = viewer
        if self.parent_widget.microscope is None:
            raise ValueError("Microscope is None, cannot open protocol editor.")
        self.microscope = self.parent_widget.microscope
        if (self.parent_widget.experiment is None or
            self.parent_widget.experiment.task_protocol is None):
            raise ValueError("Experiment is None, cannot open protocol editor.")
        self.experiment: Experiment = self.parent_widget.experiment

        self._create_widgets()
        self._setup_connections()
        self._initialise_widgets()
        self._on_selected_task_changed()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        # Protocol metadata fields
        self.label_protocol_name = QLabel("Name")
        self.lineEdit_protocol_name = QLineEdit()

        self.label_protocol_description = QLabel("Description")
        self.lineEdit_protocol_description = QLineEdit()

        self.label_protocol_version = QLabel("Version")
        self.lineEdit_protocol_version = QLineEdit()

        self.milling_task_collapsible = QCollapsible("Milling Task Parameters", self)
        self.milling_task_editor = FibsemMillingTaskWidget(microscope=self.microscope, 
                                                                         milling_enabled=False,
                                                                         correlation_enabled=False,
                                                                         parent=self)
        self.milling_task_collapsible.addWidget(self.milling_task_editor)


        self.task_params_collapsible = QCollapsible("Task Parameters", self)
        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.task_params_collapsible.addWidget(self.task_parameters_config_widget)

        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)
        self.task_params_collapsible.addWidget(self.ref_image_params_widget)

        # lamella, milling controls
        self.label_selected_milling = QLabel("Task Name")
        self.comboBox_selected_task = QComboBox()

        self.pushButton_add_task = QPushButton("Add Task")
        self.pushButton_add_task.setStyleSheet(GREEN_PUSHBUTTON_STYLE)

        self.pushButton_remove_task = QPushButton("Remove Task")
        self.pushButton_remove_task.setStyleSheet(RED_PUSHBUTTON_STYLE)

        self.pushButton_sync_to_lamella = QPushButton("Sync Config to Existing Lamella")
        self.pushButton_sync_to_lamella.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_sync_to_lamella.setToolTip("Update all existing lamella with the current task configuration")

        self.pushButton_global_edit = QPushButton("Global Edit Imaging and Field of View")
        self.pushButton_global_edit.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_global_edit.setToolTip("Edit reference imaging and milling field of view for all milling tasks")

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange;")
        self.label_warning.setVisible(False)
        self.label_warning.setWordWrap(True)

        self.button_layout = QGridLayout()
        self.button_layout.addWidget(self.pushButton_add_task, 0, 0)
        self.button_layout.addWidget(self.pushButton_remove_task, 0, 1)
        self.button_layout.addWidget(self.pushButton_global_edit, 1, 0, 1, 2)
        self.button_layout.addWidget(self.pushButton_sync_to_lamella, 2, 0, 1, 2)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.label_protocol_name, 0, 0)
        self.grid_layout.addWidget(self.lineEdit_protocol_name, 0, 1)
        self.grid_layout.addWidget(self.label_protocol_description, 1, 0)
        self.grid_layout.addWidget(self.lineEdit_protocol_description, 1, 1)
        self.grid_layout.addWidget(self.label_protocol_version, 2, 0)
        self.grid_layout.addWidget(self.lineEdit_protocol_version, 2, 1)
        self.grid_layout.addWidget(self.label_selected_milling, 3, 0)
        self.grid_layout.addWidget(self.comboBox_selected_task, 3, 1)
        self.grid_layout.addLayout(self.button_layout, 4, 0, 1, 2)
        self.grid_layout.addWidget(self.label_warning, 6, 0, 1, 2)
        # self.grid_layout.setColumnStretch(0, 1)  # Labels column - expandable
        # self.grid_layout.setColumnStretch(1, 1)  # Input widgets column - expandable

        # main layout
        self.main_layout = QVBoxLayout(self)
        self.scroll_content_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # required to resize
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)       # type: ignore
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # type: ignore
        self.scroll_content_layout.addLayout(self.grid_layout)
        self.scroll_content_layout.addWidget(self.task_params_collapsible)      # type: ignore
        self.scroll_content_layout.addWidget(self.milling_task_collapsible)     # type: ignore
        self.scroll_content_layout.addStretch()

        self.scroll_content_widget = QWidget()
        self.scroll_content_widget.setLayout(self.scroll_content_layout)
        self.scroll_area.setWidget(self.scroll_content_widget) # type: ignore
        self.main_layout.addWidget(self.scroll_area) # type: ignore

    def _setup_connections(self):
        """Setup signal connections - called once during initialization."""
        self.comboBox_selected_task.currentIndexChanged.connect(self._on_selected_task_changed)
        self.milling_task_editor.task_configs_changed.connect(self._on_milling_task_configs_changed)
        self.task_parameters_config_widget.parameter_changed.connect(self._on_task_parameters_config_changed)
        self.ref_image_params_widget.settings_changed.connect(self._on_ref_image_settings_changed)
        self.pushButton_add_task.clicked.connect(self._on_add_task_clicked)
        self.pushButton_remove_task.clicked.connect(self._on_remove_task_clicked)
        self.pushButton_global_edit.clicked.connect(self._on_global_edit_clicked)
        self.pushButton_sync_to_lamella.clicked.connect(self._on_sync_to_lamella_clicked)
        self.lineEdit_protocol_name.editingFinished.connect(self._on_protocol_name_changed)
        self.lineEdit_protocol_description.editingFinished.connect(self._on_protocol_description_changed)
        self.lineEdit_protocol_version.editingFinished.connect(self._on_protocol_version_changed)

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""

        if self.experiment is None or self.experiment.task_protocol is None:
            raise ValueError("Experiment or task protocol is None, cannot initialise protocol editor.")

        # Initialize protocol metadata fields
        self.lineEdit_protocol_name.blockSignals(True)
        self.lineEdit_protocol_description.blockSignals(True)
        self.lineEdit_protocol_version.blockSignals(True)

        self.lineEdit_protocol_name.setText(self.experiment.task_protocol.name or "")
        self.lineEdit_protocol_description.setText(self.experiment.task_protocol.description or "")
        self.lineEdit_protocol_version.setText(self.experiment.task_protocol.version or "")

        self.lineEdit_protocol_name.blockSignals(False)
        self.lineEdit_protocol_description.blockSignals(False)
        self.lineEdit_protocol_version.blockSignals(False)

        task_names = list(self.experiment.task_protocol.task_config.keys())

        self.comboBox_selected_task.blockSignals(True)

        selected_task = self.comboBox_selected_task.currentText()
        self.comboBox_selected_task.clear()
        for name in task_names:
            self.comboBox_selected_task.addItem(name)

        if selected_task in task_names:
            self.comboBox_selected_task.setCurrentText(selected_task)
        elif "Rough Milling" in task_names:
            self.comboBox_selected_task.setCurrentText("Rough Milling")
        else:
            self.comboBox_selected_task.setCurrentIndex(0)
        self.comboBox_selected_task.blockSignals(False)


    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.comboBox_selected_task.currentText()

        task_config = self.experiment.task_protocol.task_config[selected_stage_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)

        field_of_view = 150e-6
        if task_config.milling:
            key = list(task_config.milling.keys())[0]
            field_of_view = task_config.milling[key].field_of_view

        # clear existing image layers
        self.viewer.layers.clear()
        self.image = FibsemImage.generate_blank_image(hfw=field_of_view, random=True)
        self.viewer.add_image(self.image.data, name="Reference Image (FIB)", colormap='gray')

        self.milling_task_editor.config_widget.milling_editor_widget.set_image(self.image)
        self.viewer.reset_view()

        # set milling task config
        self.milling_task_editor.set_task_configs(task_config.milling)


        # # TODO: turn this into a helper function to get background milling stages -> background = related stages
        # # depending on selected stage
        # background_milling_stages = []

        # rough_milling_config = None
        # polishing_config = None
        # setup_milling_config = None

        # setup_config = self.experiment.task_protocol.task_config.get("Setup Lamella", None)
        # if setup_config is not None and setup_config.milling:
        #     setup_milling_config = setup_config.milling.get("fiducial", None)
        # polishing = self.experiment.task_protocol.task_config.get("Polishing", None)
        # if polishing is not None and polishing.milling:
        #     polishing_config = polishing.milling.get("mill_polishing", None)
        # rough_milling = self.experiment.task_protocol.task_config.get("Rough Milling", None)
        # if rough_milling is not None and rough_milling.milling:
        #     rough_milling_config = rough_milling.milling.get("mill_rough", None)

        # if selected_stage_name == "Setup Lamella":
        #     if polishing_config is not None:
        #         background_milling_stages.extend(polishing_config.stages)
        #     if rough_milling_config is not None:
        #         background_milling_stages.extend(rough_milling_config.stages)
        # elif selected_stage_name == "Rough Milling":
        #     if polishing_config is not None:
        #         background_milling_stages.extend(polishing_config.stages)
        #     if setup_milling_config is not None:
        #         background_milling_stages.extend(setup_milling_config.stages)
        # elif selected_stage_name == "Polishing":
        #     if rough_milling_config is not None:
        #         background_milling_stages.extend(rough_milling_config.stages)
        #     if setup_milling_config is not None:
        #         background_milling_stages.extend(setup_milling_config.stages)

        # self.milling_task_editor.config_widget.set_background_milling_stages(background_milling_stages)
        # self.milling_task_editor.config_widget.milling_editor_widget.update_milling_stage_display()

        if task_config.milling:
            self._on_milling_fov_changed(task_config.milling)
            self.milling_task_collapsible.setVisible(True)
        else:
            self.milling_task_collapsible.setVisible(False)
            if "Milling Patterns" in self.viewer.layers:
                self.viewer.layers.remove("Milling Patterns") # type: ignore

    def _on_milling_fov_changed(self, config: Dict[str, FibsemMillingTaskConfig]):
        """Display a warning if the milling FoV does not match the image FoV."""
        try:
            key = list(config.keys())[0]
            milling_fov = config[key].field_of_view
            image_hfw = self.milling_task_editor.config_widget.milling_editor_widget.image.metadata.image_settings.hfw
            if not np.isclose(milling_fov, image_hfw):
                milling_fov_um = format_value(milling_fov, unit='m', precision=0)
                image_fov_um = format_value(image_hfw, unit='m', precision=0)
                self.label_warning.setText(f"Milling Task FoV ({milling_fov_um}) does not match image FoV ({image_fov_um}).")
                self.label_warning.setVisible(True)
                return
        except Exception as e:
            logging.warning(f"Could not compare milling FoV and image FoV: {e}")

        self.label_warning.setVisible(False)

    def _on_milling_task_configs_changed(self, configs: Dict[str, FibsemMillingTaskConfig]):
        """Callback when the milling task configs are changed (added/removed)."""

        selected_task_name = self.comboBox_selected_task.currentText()
        self.experiment.task_protocol.task_config[selected_task_name].milling = copy.deepcopy(configs)
        logging.info(f"Updated {selected_task_name} Task, Milling Tasks: {list(configs.keys())} ")

        # save the experiment
        self._save_experiment()

        # update fov warning
        if configs:
            self._on_milling_fov_changed(configs)
        else:
            self.label_warning.setText("")

    def _on_task_parameters_config_changed(self, field_name: str, new_value: Any):
        """Callback when the task parameters config is updated."""
        selected_task_name = self.comboBox_selected_task.currentText()
        logging.info(f"Updated {selected_task_name} Task Parameters: {field_name} = {new_value}")

        # update parameters in the task config
        setattr(self.experiment.task_protocol.task_config[selected_task_name], field_name, new_value)

        # save the experiment
        self._save_experiment()

    def _on_ref_image_settings_changed(self, settings: ReferenceImageParameters):
        """Callback when the image settings are changed."""
        # # Update the image settings in the task config
        selected_task_name = self.comboBox_selected_task.currentText()
        self.experiment.task_protocol.task_config[selected_task_name].reference_imaging = settings

        # # Save the experiment
        self._save_experiment()

    def _save_experiment(self):
        """Save the experiment if available."""
        if self.parent_widget is not None and self.parent_widget.experiment is not None:
            self.parent_widget.experiment.save()
            self.parent_widget.experiment.task_protocol.save(os.path.join(self.experiment.path, "protocol.yaml"))
            self.parent_widget.protocol_editor_widget.workflow_widget._refresh_from_state()

    def _on_add_task_clicked(self):
        """Show dialog to add a new task."""
        dialog = AddTaskDialog(self.experiment.task_protocol.task_config, parent=self) # type: ignore
        if dialog.exec_() == QDialog.Accepted:
            task_type, task_name = dialog.get_task_info()
            if task_type and task_name:
                # Create new task config from registry
                task_cls = TASK_REGISTRY[task_type]
                new_task_config = task_cls.config_cls()
                new_task_config.task_name = task_name

                # Add to experiment
                self.experiment.task_protocol.task_config[task_name] = new_task_config
                self.experiment.task_protocol.workflow_config.add_task(new_task_config)

                # also add task to each existing lamella
                for lamella in self.experiment.positions:
                    lamella.task_config[task_name] = copy.deepcopy(new_task_config)

                # Save experiment
                self._save_experiment()

                # Refresh widgets
                self._initialise_widgets()
                self.comboBox_selected_task.setCurrentText(task_name)

                logging.info(f"Added new task: {task_name} ({task_type})")

    def _on_remove_task_clicked(self):
        """Remove the selected task with confirmation dialog."""
        selected_task_name = self.comboBox_selected_task.currentText()

        if not selected_task_name:
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the task '{selected_task_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Remove from experiment
            if selected_task_name in self.experiment.task_protocol.task_config:
                del self.experiment.task_protocol.task_config[selected_task_name]

                # Save experiment
                self._save_experiment()

                # Refresh widgets
                self._initialise_widgets()

                logging.info(f"Removed task: {selected_task_name}")

    def _on_global_edit_clicked(self):
        """Show dialog for globally editing reference imaging and milling FoV."""
        dialog = AutoLamellaGlobalTaskEditDialog(self.experiment, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            # Apply changes to all tasks
            updated_count = dialog.apply_changes()

            # apply to existing lamella if selected
            update_lamella = dialog.checkbox_update_existing.isChecked()
            if update_lamella:
                selected_task_names = dialog.get_selected_tasks()
                for task_name in selected_task_names:
                    self._sync_existing_lamella_task_config(task_name)

            # Save the experiment
            self._save_experiment()

            # Refresh the current task view to reflect changes
            self._on_selected_task_changed()

            # Show success message
            selected_count = len(dialog.get_selected_tasks())
            msg = f"Successfully updated reference imaging for {selected_count} task(s) and milling FoV for {updated_count} task(s) with milling."
            if update_lamella:
                msg += "\nExisting lamella configurations were also updated."
            QMessageBox.information(
                self,
                "Global Edit Applied",
                msg,
            )

    def _on_protocol_name_changed(self):
        """Callback when the protocol name editing is finished."""
        text = self.lineEdit_protocol_name.text()
        if self.experiment and self.experiment.task_protocol:
            self.experiment.task_protocol.name = text
            self._save_experiment()
            logging.info(f"Updated protocol name: {text}")

    def _on_protocol_description_changed(self):
        """Callback when the protocol description editing is finished."""
        text = self.lineEdit_protocol_description.text()
        if self.experiment and self.experiment.task_protocol:
            self.experiment.task_protocol.description = text
            self._save_experiment()
            logging.info(f"Updated protocol description: {text}")

    def _on_protocol_version_changed(self):
        """Callback when the protocol version editing is finished."""
        text = self.lineEdit_protocol_version.text()
        if self.experiment and self.experiment.task_protocol:
            self.experiment.task_protocol.version = text
            self._save_experiment()
            logging.info(f"Updated protocol version: {text}")

    def _on_sync_to_lamella_clicked(self):
        """Sync the current task configuration to all existing lamella."""
        selected_task_name = self.comboBox_selected_task.currentText()

        if not selected_task_name:
            return

        # Check if there are any positions
        if not self.experiment.positions:
            QMessageBox.information(
                self,
                "No Lamella",
                "There are no existing lamella to sync the configuration to.",
            )
            return


        # Show confirmation dialog
        num_lamella = len(self.experiment.positions)
        reply = QMessageBox.question(
            self,
            "Confirm Sync",
            (
                f"Are you sure you want to sync the task configuration '{selected_task_name}' to all "
                f"{num_lamella} existing lamella?\n\nThis will overwrite their current task configurations, while "
                f"keeping any existing milling pattern positions unchanged."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:

            # Perform the sync
            updated_count = self._sync_existing_lamella_task_config(selected_task_name)

            # Save the experiment
            self._save_experiment()

            # Show success message
            QMessageBox.information(
                self,
                "Sync Complete",
                f"Successfully synced task configuration '{selected_task_name}' to {updated_count} lamella.",
            )

            logging.info(f"Synced task configuration '{selected_task_name}' to {updated_count} lamella")

    def _sync_existing_lamella_task_config(self, selected_task_name: str):
        """Sync the current task configuration to all existing lamella, preserving milling pattern positions.
        Args:
            selected_task_name (str): The name of the task to sync.
        Returns:
            int: The number of lamella updated.
        """
        # TODO: add selective lamella selection instead of all lamella
        # TODO: add a two way sync option to also update the protocol from lamella

        # Get the current task config
        current_task_config = self.experiment.task_protocol.task_config[selected_task_name]

        # Update task_config for each lamella
        updated_count = 0
        for lamella in self.experiment.positions:
            existing_task_config = lamella.task_config.get(selected_task_name)
            new_task_config = copy.deepcopy(current_task_config)

            if existing_task_config is not None:
                for milling_name, new_milling_config in new_task_config.milling.items():
                    existing_milling_config = existing_task_config.milling.get(milling_name)
                    if existing_milling_config is None:
                        continue

                    existing_stage_lookup = {
                        (stage.num, stage.name): stage for stage in existing_milling_config.stages
                    }

                    for new_stage in new_milling_config.stages:
                        existing_stage = existing_stage_lookup.get((new_stage.num, new_stage.name))
                        if existing_stage is None:
                            continue

                        if (
                            type(existing_stage.pattern) is type(new_stage.pattern)
                            and hasattr(existing_stage.pattern, "point")
                        ):
                            new_stage.pattern.point = copy.deepcopy(existing_stage.pattern.point)

            # Deep copy the task config to avoid reference issues
            lamella.task_config[selected_task_name] = new_task_config
            updated_count += 1

        return updated_count


    # TODO: we should integrate both milling and parameter updates into a single config update method
    # TODO: support position sync between milling tasks, e.g. sync trench position between rough milling and polishing


class AutoLamellaProtocolEditorTabWidget(QTabWidget):
    def __init__(self, parent_widget: Optional['AutoLamellaUI'] = None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.workflow_widget: AutoLamellaWorkflowWidget
        self.task_widget: AutoLamellaProtocolTaskConfigEditor
        self.lamella_widget : AutoLamellaProtocolEditorWidget

        # Connect to tab change signal
        self.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        """Handle tab change event."""
        tab_name = self.tabText(index)

        try:
            for i in range(self.count()):
                w: Union[AutoLamellaProtocolTaskConfigEditor, AutoLamellaProtocolEditorWidget] = self.widget(i)
                if w is None or isinstance(w, AutoLamellaWorkflowWidget):
                    continue
                milling_editor = w.milling_task_editor.config_widget.milling_editor_widget

                if i == index:
                    logging.info(f"Tab '{tab_name}' is now focused. Connecting mouse drag callbacks.")
                    milling_editor.is_movement_locked = False  # lock movement when tab is not focused
                    if index == 0:  # Task Config tab
                        w._on_selected_task_changed()  # type: ignore
                    elif index == 2:  # Lamella Config tab
                        w._on_selected_lamella_changed()  # type: ignore
                else:
                    milling_editor.is_movement_locked = True  # lock movement when tab is not focused
        except Exception as e:
            logging.error(f"Error updating tab '{tab_name}': {e}")

    def closeEvent(self, event):
        """Handle widget close event to clean up parent reference."""
        if self.parent_widget is not None:
            self.parent_widget.protocol_editor_widget = None
        super().closeEvent(event)


def show_protocol_editor(parent: 'AutoLamellaUI'):
    """Show the AutoLamella Protocol Editor widget."""
    viewer = napari.Viewer(title="AutoLamella Protocol Editor")

    # create a tab widget and add the editor as a tab
    task_widget = AutoLamellaProtocolTaskConfigEditor(viewer=viewer, parent=parent)
    lamella_widget = AutoLamellaProtocolEditorWidget(viewer=viewer, parent=parent)
    workflow_widget = AutoLamellaWorkflowWidget(experiment=parent.experiment, parent=parent)
    workflow_widget.workflow_config_changed.connect(parent._on_workflow_config_changed)
    workflow_widget.workflow_options_changed.connect(parent._on_workflow_options_changed)
    tab_widget = AutoLamellaProtocolEditorTabWidget(parent_widget=parent)
    tab_widget.addTab(task_widget, "Protocol")
    tab_widget.addTab(workflow_widget, "Workflow")
    tab_widget.addTab(lamella_widget, "Lamella")
    tab_widget._on_tab_changed(0)  # default to Protocol tab

    tab_widget.workflow_widget = workflow_widget
    tab_widget.task_widget = task_widget
    tab_widget.lamella_widget = lamella_widget

    # Store reference in parent for later access
    parent.protocol_editor_widget = tab_widget

    # Connect to viewer window destruction to clean up parent reference
    # This is necessary because closeEvent on the dock widget isn't triggered when napari closes
    def on_viewer_closed():
        if parent.protocol_editor_widget is not None:
            parent.protocol_editor_widget = None

    viewer.window._qt_window.destroyed.connect(on_viewer_closed)

    viewer.window.add_dock_widget(tab_widget, area='right', name='AutoLamella Protocol Editor')
    viewer.window.activate()
    return tab_widget
