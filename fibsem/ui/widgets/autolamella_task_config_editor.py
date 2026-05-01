
import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.ui import stylesheets
from fibsem.ui.widgets.autolamella_global_task_editor_dialog import AutoLamellaGlobalTaskEditDialog
from fibsem.ui.widgets.lamella_template_config_widget import LamellaTemplateConfigWidget
from fibsem.ui.widgets.custom_widgets import TaskNameListWidget
from fibsem.ui.widgets.autolamella_protocol_information_widget import ProtocolInformationWidget
from fibsem.ui.widgets.autolamella_task_config_widget import AutoLamellaTaskParametersConfigWidget
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.reference_image_parameters_widget import ReferenceImageParametersWidget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.structures import ReferenceImageParameters
    from fibsem.milling.tasks import FibsemMillingTaskConfig

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

        # Populate task types from registry (includes plugins and runtime registrations)
        for task_type, task_cls in get_tasks().items():
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
        task_registry = get_tasks()
        if task_type and task_type in task_registry:
            display_name = task_registry[task_type].config_cls.display_name
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

    workflow_config_changed = pyqtSignal(object)  # AutoLamellaWorkflowConfig

    def __init__(self, parent: 'AutoLamellaUI'):
        super().__init__(parent)
        self.parent_widget = parent
        self.setStyleSheet(stylesheets.NAPARI_STYLE)

        self.milling_task_editor: Optional[MillingTaskViewerWidget] = None
        self.microscope = getattr(self.parent_widget, 'microscope', None)  # type: ignore[assignment]
        self.experiment: 'Experiment' = getattr(self.parent_widget, 'experiment', None)  # type: ignore[assignment]
        self._current_milling_key: Optional[str] = None

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)

        self._try_initialize()

    def _try_initialize(self) -> None:
        """Create and populate widgets once both microscope and experiment are available."""
        if self.milling_task_editor is not None:
            return  # already initialized
        if self.microscope is None:
            return
        if self.experiment is None or self.experiment.task_protocol is None:
            return
        self._create_widgets()
        self._setup_connections()
        self._initialise_widgets()
        self._on_selected_task_changed()

    def _on_microscope_connected(self):
        if self.parent_widget.microscope is None:
            raise ValueError("Microscope is None, cannot open protocol editor.")
        self.microscope = self.parent_widget.microscope
        if self.milling_task_editor is not None:
            self.milling_task_editor.microscope = self.microscope
        else:
            self._try_initialize()

    def set_experiment(self, experiment: 'Experiment'):
        """Set the experiment for the protocol editor."""
        self.experiment = experiment
        if self.milling_task_editor is None:
            self._try_initialize()
        else:
            self._initialise_widgets()
            self._on_selected_task_changed()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        # Protocol metadata (Column 1, top)
        self.protocol_info_widget = ProtocolInformationWidget(parent=self)

        # Task parameters (Column 2)
        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)

        # Milling task editor (Column 3 — viewer=None, config panels only)
        self.milling_task_editor = MillingTaskViewerWidget(
            microscope=self.microscope,  # type: ignore[arg-type]
            viewer=None,
            milling_enabled=False,
            parent=self,
        )
        self.milling_task_editor.setMinimumHeight(550)

        # lamella, milling controls (Column 1)
        self.task_list_widget = TaskNameListWidget()

        self.pushButton_sync_to_lamella = QPushButton("Apply Config to Existing Lamella")
        self.pushButton_sync_to_lamella.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_sync_to_lamella.setToolTip("Update all existing lamella with the current task configuration")

        self.pushButton_open_global_editor = QPushButton("Global Edit")
        self.pushButton_open_global_editor.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_open_global_editor.setToolTip("Globally edit reference imaging settings and milling FoV across multiple tasks.")

        self.pushButton_open_lamella_template = QPushButton("Lamella Template")
        self.pushButton_open_lamella_template.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_open_lamella_template.setToolTip("Edit the initial state applied to every new lamella created from this protocol.")

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange;")
        self.label_warning.setVisible(False)
        self.label_warning.setWordWrap(True)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.pushButton_sync_to_lamella)
        self.button_layout.addWidget(self.pushButton_open_global_editor)
        self.button_layout.addWidget(self.pushButton_open_lamella_template)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.task_list_widget, 0, 0, 1, 2)
        self.grid_layout.addLayout(self.button_layout, 1, 0, 1, 2)

        # --- Column 1: Protocol info + task selector ---
        col1_content = QWidget()
        col1_layout = QVBoxLayout(col1_content)
        col1_layout.setContentsMargins(4, 4, 4, 4)
        col1_layout.addWidget(self.protocol_info_widget)
        col1_layout.addLayout(self.grid_layout)
        col1_layout.addWidget(self.label_warning)
        col1_layout.addStretch()
        col1_scroll = QScrollArea()
        col1_scroll.setWidgetResizable(True)
        col1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col1_scroll.setWidget(col1_content)

        # --- Column 2: Task parameters ---
        col2_content = QWidget()
        col2_layout = QVBoxLayout(col2_content)
        col2_layout.setContentsMargins(4, 4, 4, 4)
        col2_layout.addWidget(self.task_parameters_config_widget)
        col2_layout.addWidget(self.ref_image_params_widget)
        col2_layout.addStretch()
        col2_scroll = QScrollArea()
        col2_scroll.setWidgetResizable(True)
        col2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col2_scroll.setWidget(col2_content)

        # --- Column 3: Milling parameters (config panels, no napari canvas) ---
        col3_scroll = QScrollArea()
        col3_scroll.setWidgetResizable(True)
        col3_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col3_scroll.setWidget(self.milling_task_editor)

        # --- 3-column splitter ---
        splitter = QSplitter(Qt.Horizontal)  # type: ignore
        splitter.addWidget(col1_scroll)
        splitter.addWidget(col2_scroll)
        splitter.addWidget(col3_scroll)
        splitter.setSizes([280, 350, 800])

        self._main_layout.addWidget(splitter)

    def _setup_connections(self):
        """Setup signal connections - called once during initialization."""
        self.task_list_widget.task_selected.connect(lambda _: self._on_selected_task_changed())
        self.task_list_widget.add_clicked.connect(self._on_add_task_clicked)
        self.task_list_widget.remove_clicked.connect(self._on_remove_task_clicked)
        self.milling_task_editor.settings_changed.connect(self._on_milling_settings_changed)
        self.task_parameters_config_widget.parameter_changed.connect(self._on_task_parameters_config_changed)
        self.ref_image_params_widget.settings_changed.connect(self._on_ref_image_settings_changed)
        self.pushButton_sync_to_lamella.clicked.connect(self._on_sync_to_lamella_clicked)
        self.pushButton_open_global_editor.clicked.connect(self._on_global_edit_clicked)
        self.pushButton_open_lamella_template.clicked.connect(self._on_lamella_template_clicked)
        self.protocol_info_widget.field_changed.connect(self._on_protocol_field_changed)

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""

        if self.experiment is None or self.experiment.task_protocol is None:
            raise ValueError("Experiment or task protocol is None, cannot initialise protocol editor.")

        self.protocol_info_widget.update_from_protocol(self.experiment.task_protocol)

        task_names = list(self.experiment.task_protocol.task_config.keys())
        self.task_list_widget.set_tasks(task_names)

    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_task_name = self.task_list_widget.selected_task

        task_config = self.experiment.task_protocol.task_config[selected_task_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)

        # set milling task config
        if task_config.milling:
            self._current_milling_key = next(iter(task_config.milling))
            self.milling_task_editor.set_config(task_config.milling[self._current_milling_key])
            self.milling_task_editor.setVisible(True)
        else:
            self._current_milling_key = None
            self.milling_task_editor.clear()
            self.milling_task_editor.setVisible(False)

    def _on_milling_settings_changed(self, config: 'FibsemMillingTaskConfig'):
        """Callback when the milling task config is changed."""

        selected_task_name = self.task_list_widget.selected_task
        key = self._current_milling_key
        if key and selected_task_name in self.experiment.task_protocol.task_config:
            self.experiment.task_protocol.task_config[selected_task_name].milling[key] = config
            logging.info(f"Updated {selected_task_name} Task, milling key '{key}'")

        # save the experiment
        self._save_experiment()

    def _on_task_parameters_config_changed(self, field_name: str, new_value: Any):
        """Callback when the task parameters config is updated."""
        selected_task_name = self.task_list_widget.selected_task
        logging.info(f"Updated {selected_task_name} Task Parameters: {field_name} = {new_value}")

        # update parameters in the task config
        setattr(self.experiment.task_protocol.task_config[selected_task_name], field_name, new_value)

        # save the experiment
        self._save_experiment()

    def _on_ref_image_settings_changed(self, settings: 'ReferenceImageParameters'):
        """Callback when the image settings are changed."""
        # Update the image settings in the task config
        selected_task_name = self.task_list_widget.selected_task
        self.experiment.task_protocol.task_config[selected_task_name].reference_imaging = settings

        # Save the experiment
        self._save_experiment()

    def _save_experiment(self):
        """Save the experiment if available."""
        if self.parent_widget is not None and self.parent_widget.experiment is not None:
            self.parent_widget.experiment.save(save_protocol=True)

    def _on_add_task_clicked(self):
        """Show dialog to add a new task."""
        dialog = AddTaskDialog(self.experiment.task_protocol.task_config, parent=self) # type: ignore
        if dialog.exec_() == QDialog.Accepted:
            task_type, task_name = dialog.get_task_info()
            if task_type and task_name:
                # Create new task config from registry
                task_cls = get_tasks()[task_type]
                new_task_config = task_cls.config_cls()  # type: ignore
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
                self.task_list_widget.select(task_name)
                self.workflow_config_changed.emit(self.experiment.task_protocol.workflow_config)
                logging.info(f"Added new task: {task_name} ({task_type})")

    def _on_remove_task_clicked(self):
        """Remove the selected task with confirmation dialog."""
        selected_task_name = self.task_list_widget.selected_task

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
                self.workflow_config_changed.emit(self.experiment.task_protocol.workflow_config)

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
                all_lamella_names = [p.name for p in self.experiment.positions]
                self.experiment.apply_lamella_config(
                    lamella_names=all_lamella_names,
                    task_names=selected_task_names,
                )

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

    def _on_lamella_template_clicked(self):
        """Show dialog for editing the protocol's LamellaTemplateConfig."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Lamella Template")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        template_widget = LamellaTemplateConfigWidget()
        template_widget.set_template(self.experiment.task_protocol.lamella_template)
        layout.addWidget(template_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.experiment.task_protocol.lamella_template = template_widget.get_template()
            self._save_experiment()

    def _on_protocol_field_changed(self, field: str, value: str) -> None:
        if self.experiment and self.experiment.task_protocol:
            setattr(self.experiment.task_protocol, field, value)
            self._save_experiment()
            logging.info(f"Updated protocol {field}: {value}")

    def _on_sync_to_lamella_clicked(self):
        """Sync the current task configuration to all existing lamella."""
        selected_task_name = self.task_list_widget.selected_task

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

            # Perform the sync (from base protocol to all lamella)
            all_lamella_names = [p.name for p in self.experiment.positions]
            updated_count = self.experiment.apply_lamella_config(
                lamella_names=all_lamella_names,
                task_names=[selected_task_name],
            )

            # Save the experiment
            self._save_experiment()

            # Show success message
            QMessageBox.information(
                self,
                "Sync Complete",
                f"Successfully synced task configuration '{selected_task_name}' to {updated_count} lamella.",
            )

            logging.info(f"Synced task configuration '{selected_task_name}' to {updated_count} lamella")



