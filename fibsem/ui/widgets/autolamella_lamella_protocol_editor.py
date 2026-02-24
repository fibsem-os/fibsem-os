
import copy
import glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import napari
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible
from fibsem import conversions
from fibsem.ui.napari.utilities import is_inside_image_bounds, add_points_layer
from fibsem.utils import format_value
from fibsem.applications.autolamella.structures import (
    DefectState,
    Lamella,
)
import fibsem.applications.autolamella.config as cfg
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, Point, ReferenceImageParameters
from fibsem.ui.widgets.autolamella_task_config_widget import (
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui.stylesheets import BLUE_PUSHBUTTON_STYLE, PRIMARY_BUTTON_STYLESHEET, SECONDARY_BUTTON_STYLESHEET
from fibsem.ui.widgets.autolamella_defect_state_widget import AutoLamellaDefectStateWidget
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI

REFERENCE_IMAGE_LAYER_NAME = "Reference Image (FIB)"
REFERENCE_IMAGE_SEM_LAYER_NAME = "Reference Image (SEM)"


class ApplyLamellaConfigDialog(QDialog):
    """Dialog for applying a lamella's task configuration to other lamella."""

    def __init__(
        self,
        source_name: str,
        other_lamella_names: List[str],
        task_names: List[str],
        parent: Optional[QWidget] = None,
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

        # --- Target lamella selection ---
        self.lamella_group = QGroupBox("Target Lamella")
        self.lamella_group.setFlat(True)

        self.lamella_list = QListWidget()
        self.lamella_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lamella_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        for name in self._other_lamella_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.lamella_list.addItem(item)

        lamella_buttons_layout = QHBoxLayout()
        self.pushButton_select_all_lamella = QPushButton("Select All")
        self.pushButton_deselect_all_lamella = QPushButton("Deselect All")
        self.pushButton_select_all_lamella.clicked.connect(
            lambda: self._set_all_check_state(self.lamella_list, Qt.Checked))  # type: ignore
        self.pushButton_deselect_all_lamella.clicked.connect(
            lambda: self._set_all_check_state(self.lamella_list, Qt.Unchecked))  # type: ignore
        lamella_buttons_layout.addWidget(self.pushButton_select_all_lamella)
        lamella_buttons_layout.addWidget(self.pushButton_deselect_all_lamella)

        lamella_layout = QVBoxLayout()
        lamella_layout.addWidget(self.lamella_list)
        lamella_layout.addLayout(lamella_buttons_layout)
        self.lamella_group.setLayout(lamella_layout)

        # --- Task selection ---
        self.task_group = QGroupBox("Tasks to Apply")
        self.task_group.setFlat(True)

        self.task_list = QListWidget()
        self.task_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        for task_name in self._task_names:
            item = QListWidgetItem(task_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.task_list.addItem(item)

        task_buttons_layout = QHBoxLayout()
        self.pushButton_select_all_tasks = QPushButton("Select All")
        self.pushButton_deselect_all_tasks = QPushButton("Deselect All")
        self.pushButton_select_all_tasks.clicked.connect(
            lambda: self._set_all_check_state(self.task_list, Qt.Checked))  # type: ignore
        self.pushButton_deselect_all_tasks.clicked.connect(
            lambda: self._set_all_check_state(self.task_list, Qt.Unchecked))  # type: ignore
        task_buttons_layout.addWidget(self.pushButton_select_all_tasks)
        task_buttons_layout.addWidget(self.pushButton_deselect_all_tasks)

        task_layout = QVBoxLayout()
        task_layout.addWidget(self.task_list)
        task_layout.addLayout(task_buttons_layout)
        self.task_group.setLayout(task_layout)

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

        # --- Buttons ---
        self.button_box = QDialogButtonBox(self)
        self.pushButton_apply = QPushButton("Apply")
        self.pushButton_apply.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_apply.setAutoDefault(False)
        self.pushButton_cancel = QPushButton("Cancel")
        self.pushButton_cancel.setAutoDefault(False)

        self.button_box.addButton(self.pushButton_apply, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)
        self.pushButton_apply.clicked.connect(self.accept)
        self.pushButton_cancel.clicked.connect(self.reject)
        self.pushButton_apply.setDefault(False)
        self.pushButton_cancel.setDefault(False)

        # Update info when selections change (must be after buttons are created)
        self.lamella_list.itemChanged.connect(lambda _: self._update_info_label())
        self.task_list.itemChanged.connect(lambda _: self._update_info_label())
        self.checkbox_update_base_protocol.stateChanged.connect(lambda _: self._update_info_label())
        self._update_info_label()

    def _setup_layout(self):
        """Setup the dialog layout."""
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_description)
        main_layout.addWidget(self.lamella_group)
        main_layout.addWidget(self.task_group)
        main_layout.addWidget(self.checkbox_update_base_protocol)
        main_layout.addWidget(self.label_info)
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def _set_all_check_state(self, list_widget: QListWidget, state):
        """Set the check state for all items in a list widget."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item:
                item.setCheckState(state)

    def get_selected_lamella_names(self) -> List[str]:
        """Get list of selected target lamella names."""
        selected = []
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item and item.checkState() == Qt.Checked:  # type: ignore
                selected.append(item.text())
        return selected

    def get_selected_tasks(self) -> List[str]:
        """Get list of selected task names."""
        selected = []
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            if item and item.checkState() == Qt.Checked:  # type: ignore
                selected.append(item.text())
        return selected

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
            self.label_info.setText(f"Please select at least one {' and '.join(missing)}.")
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

    def keyPressEvent(self, event):
        """Prevent Enter/Return from accepting the dialog."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)


class AutoLamellaProtocolEditorWidget(QWidget):
    """A widget to edit the AutoLamella protocol."""

    def __init__(self,
                viewer: napari.Viewer,
                parent: 'AutoLamellaUI'):
        super().__init__(parent)
        self.parent_widget = parent
        self.viewer = viewer

        self.image: FibsemImage

        if self.parent_widget.microscope is None:
            return

        self._on_microscope_connected()

    def _on_microscope_connected(self):
        """Callback when the microscope is connected."""
        if self.parent_widget.microscope is None:
            raise ValueError("Microscope in parent widget is None, cannot proceed.")
        self.microscope = self.parent_widget.microscope
        self._create_widgets()
        self._initialise_widgets()

    def set_experiment(self):
        """Set the experiment for the protocol editor."""
        self._refresh_experiment_positions()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        self.milling_task_collapsible = QCollapsible("Milling Task Parameters", self)
        self.milling_task_editor = FibsemMillingTaskWidget(microscope=self.microscope,
                                                            milling_enabled=False,
                                                            parent=self)
        self.milling_task_collapsible.addWidget(self.milling_task_editor)
        self.milling_task_editor.setMinimumHeight(550)

        self.task_params_collapsible = QCollapsible("Task Parameters", self)
        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.task_params_collapsible.addWidget(self.task_parameters_config_widget)

        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)
        self.task_params_collapsible.addWidget(self.ref_image_params_widget)

        # lamella, milling controls
        self.label_selected_lamella = QLabel("Lamella")
        self.comboBox_selected_lamella = QComboBox()
        self.pushButton_refresh_positions = QPushButton("Refresh Experiment Data")
        self.pushButton_refresh_positions.setToolTip("Refresh the list of lamella positions from the experiment (and associated data).")
        self.pushButton_refresh_positions.clicked.connect(self._refresh_experiment_positions)
        self.pushButton_apply_to_other = QPushButton("Apply Config to Other Lamella")
        self.pushButton_apply_to_other.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_apply_to_other.setToolTip(
            "Apply this lamella's task configurations to other lamella in the experiment."
        )
        self.pushButton_apply_to_other.clicked.connect(self._on_apply_to_other_clicked)
        self.pushButton_open_correlation = QPushButton("Open Correlation")
        self.pushButton_open_correlation.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_open_correlation.setToolTip(
            "Open the 3D correlation tool to align the FIB and FM images."
        )
        self.pushButton_open_correlation.clicked.connect(self._open_correlation_widget)
        self.label_selected_milling = QLabel("Task Name")
        self.comboBox_selected_task = QComboBox()

        self.combobox_fm_filenames = QComboBox()
        self.combobox_fm_filenames_label = QLabel("FM Z-Stack")
        self.combobox_fm_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_fib_filenames = QComboBox()
        self.combobox_fib_filenames_label = QLabel("FIB Image")
        self.combobox_fib_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_sem_filenames = QComboBox()
        self.combobox_sem_filenames_label = QLabel("SEM Image")
        self.combobox_sem_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange;")
        self.label_warning.setWordWrap(True)
        self.label_status = QLabel("")
        self.label_status.setStyleSheet("color: lime;")
        self.label_status.setWordWrap(True)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.label_selected_lamella, 0, 0)
        self.grid_layout.addWidget(self.comboBox_selected_lamella, 0, 1)
        self.grid_layout.addWidget(self.combobox_fib_filenames_label, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fib_filenames, 1, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames_label, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames_label, 3, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames, 3, 1, 1, 1)
        self.grid_layout.addWidget(self.label_selected_milling, 4, 0)
        self.grid_layout.addWidget(self.comboBox_selected_task, 4, 1)
        self.grid_layout.addWidget(self.label_status, 5, 0, 1, 2)
        self.grid_layout.addWidget(self.label_warning, 6, 0, 1, 2)
        self.grid_layout.addWidget(self.pushButton_refresh_positions, 7, 0, 1, 2)
        self.grid_layout.addWidget(self.pushButton_apply_to_other, 8, 0, 1, 2)
        self.grid_layout.addWidget(self.pushButton_open_correlation, 9, 0, 1, 2)

        self.lamella_defect_widget = AutoLamellaDefectStateWidget(parent=self)
        self.grid_layout.addWidget(self.lamella_defect_widget, 10, 0, 1, 2)

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

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""
        if self.parent_widget.experiment is not None:
            for pos in self.parent_widget.experiment.positions:
                self.comboBox_selected_lamella.addItem(pos.name, pos)

        self.comboBox_selected_lamella.currentIndexChanged.connect(self._on_selected_lamella_changed)
        self.comboBox_selected_task.currentIndexChanged.connect(self._on_selected_task_changed)
        self.milling_task_editor.task_configs_changed.connect(self._on_milling_task_config_updated)
        self.task_parameters_config_widget.parameter_changed.connect(self._on_task_parameters_config_changed)
        self.ref_image_params_widget.settings_changed.connect(self._on_ref_image_settings_changed)
        self.milling_task_editor.config_widget.correlation_result_updated_signal.connect(self._on_point_of_interest_updated)
        self.lamella_defect_widget.defect_state_changed.connect(self._on_defect_state_changed)
        self.viewer.mouse_drag_callbacks.append(self._on_single_click)

        if self.comboBox_selected_lamella.count() > 0:
            self.comboBox_selected_lamella.setCurrentIndex(0)

        if self.parent_widget.experiment is not None and self.parent_widget.experiment.positions:  # type: ignore
            self._on_selected_lamella_changed()

    def _refresh_experiment_positions(self):
        """Refresh the list of experiment positions in the combobox."""
        # TODO: migrate to using experiment.positions.events for updates, rather than manual refresh

        # Store the currently selected lamella name
        current_lamella_name = None
        if self.comboBox_selected_lamella.currentData() is not None:
            current_lamella_name = self.comboBox_selected_lamella.currentData().name

        # Block signals to prevent triggering changes during refresh
        self.comboBox_selected_lamella.blockSignals(True)
        self.comboBox_selected_lamella.clear()

        # Reload positions from the experiment
        if self.parent_widget.experiment is not None:
            for pos in self.parent_widget.experiment.positions:
                self.comboBox_selected_lamella.addItem(pos.name, pos)

        # Try to restore the previously selected lamella
        if current_lamella_name is not None:
            for i in range(self.comboBox_selected_lamella.count()):
                if self.comboBox_selected_lamella.itemData(i).name == current_lamella_name:
                    self.comboBox_selected_lamella.setCurrentIndex(i)
                    break
        elif self.comboBox_selected_lamella.count() > 0:
            self.comboBox_selected_lamella.setCurrentIndex(0)

        self.comboBox_selected_lamella.blockSignals(False)

        # Trigger the change event to update the UI
        if self.comboBox_selected_lamella.count() > 0:
            self._on_selected_lamella_changed()

    def _on_selected_lamella_changed(self):
        """Callback when the selected lamella changes."""
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()

        task_names = list(selected_lamella.task_config.keys())
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

        # load fluorescence image
        filenames = sorted(glob.glob(os.path.join(selected_lamella.path, "*.ome.tiff")))
        self.combobox_fm_filenames.blockSignals(True)
        self.combobox_fm_filenames.clear()
        for f in filenames:
            self.combobox_fm_filenames.addItem(os.path.basename(f))
        self.combobox_fm_filenames.setCurrentIndex(len(filenames)-1)  # Select latest by default
        self.combobox_fm_filenames.blockSignals(False)

        # load fib reference image
        self.combobox_fib_filenames.blockSignals(True)
        fib_filenames = sorted(glob.glob(os.path.join(selected_lamella.path, "*_ib.tif")))
        fib_filenames = [f for f in fib_filenames if "alignment" not in f] # show all non-alignment images

        # remember selected fib filename
        selected_fib_filename = self.combobox_fib_filenames.currentText()
        self.combobox_fib_filenames.clear()
        for f in fib_filenames:
            self.combobox_fib_filenames.addItem(os.path.basename(f))

        base_filenames = [os.path.basename(f) for f in fib_filenames]

        latest_task_filename = ""
        if selected_lamella.last_completed_task is not None:
            last_task_name = selected_lamella.last_completed_task.name          
            reference_image_filename = f"ref_{last_task_name}_final_res_*_ib.tif"
            matching_filenames = glob.glob(os.path.join(selected_lamella.path, reference_image_filename))
            if len(matching_filenames) > 0:
                # pick the latest by modification time
                latest_task_filename = os.path.basename(sorted(matching_filenames, key=os.path.getmtime)[-1])

        # set default fib image filename
        default_filename = fib_filenames[-1] if len(fib_filenames) > 0 else ""
        if latest_task_filename in base_filenames:
            default_filename = latest_task_filename
        elif selected_fib_filename in base_filenames:
            default_filename = selected_fib_filename

        self.combobox_fib_filenames.setCurrentText(default_filename)
        self.combobox_fib_filenames.blockSignals(False)

        # load sem reference image
        self.combobox_sem_filenames.blockSignals(True)
        sem_filenames = sorted(glob.glob(os.path.join(selected_lamella.path, "*_eb.tif")))
        sem_filenames = [f for f in sem_filenames if "alignment" not in f]

        selected_sem_filename = self.combobox_sem_filenames.currentText()
        self.combobox_sem_filenames.clear()
        for f in sem_filenames:
            self.combobox_sem_filenames.addItem(os.path.basename(f))

        sem_base_filenames = [os.path.basename(f) for f in sem_filenames]

        latest_sem_task_filename = ""
        if selected_lamella.last_completed_task is not None:
            last_task_name = selected_lamella.last_completed_task.name
            reference_sem_filename = f"ref_{last_task_name}_final_res_*_eb.tif"
            matching_sem_filenames = glob.glob(os.path.join(selected_lamella.path, reference_sem_filename))
            if len(matching_sem_filenames) > 0:
                latest_sem_task_filename = os.path.basename(sorted(matching_sem_filenames, key=os.path.getmtime)[-1])

        default_sem_filename = sem_filenames[-1] if sem_filenames else ""
        if latest_sem_task_filename in sem_base_filenames:
            default_sem_filename = latest_sem_task_filename
        elif selected_sem_filename in sem_base_filenames:
            default_sem_filename = selected_sem_filename

        self.combobox_sem_filenames.setCurrentText(default_sem_filename)
        self.combobox_sem_filenames.blockSignals(False)

        # hide if no filenames
        self.combobox_fm_filenames.setVisible(len(filenames) > 0)
        self.combobox_fm_filenames_label.setVisible(len(filenames) > 0)
        self.combobox_fib_filenames.setVisible(len(fib_filenames) > 0)
        self.combobox_fib_filenames_label.setVisible(len(fib_filenames) > 0)
        self.combobox_sem_filenames.setVisible(len(sem_filenames) > 0)
        self.combobox_sem_filenames_label.setVisible(len(sem_filenames) > 0)

        self._on_image_selected(0)
        self._draw_point_of_interest(selected_lamella.poi)
        self.lamella_defect_widget.set_defect_state(selected_lamella.defect)

    def _on_image_selected(self, index):
        """Callback when an image is selected."""
        p: Lamella = self.comboBox_selected_lamella.currentData()
        fib_filename = self.combobox_fib_filenames.currentText()
        sem_filename = self.combobox_sem_filenames.currentText()

        # load the fib reference image
        reference_image_path = os.path.join(p.path, fib_filename)
        if os.path.exists(reference_image_path) and os.path.isfile(reference_image_path):
            self.image = FibsemImage.load(reference_image_path)
        else:
            self.image = FibsemImage.generate_blank_image(hfw=150e-6, random=True)

        # load the sem reference image
        sem_image = None
        if sem_filename:
            sem_image_path = os.path.join(p.path, sem_filename)
            if os.path.exists(sem_image_path) and os.path.isfile(sem_image_path):
                sem_image = FibsemImage.load(sem_image_path)

        self.viewer.layers.clear()
        self.viewer.add_image(
            data=self.image.filtered_data,
            name=REFERENCE_IMAGE_LAYER_NAME,
            colormap="gray",
            blending="additive",
        )

        if sem_image is not None:
            self.viewer.add_image(
                data=sem_image.filtered_data,
                name=REFERENCE_IMAGE_SEM_LAYER_NAME,
                colormap="gray",
                blending="additive",
                translate=(0, -sem_image.data.shape[1]),
            )

        self.milling_task_editor.config_widget.milling_editor_widget.set_image(self.image)
        self._on_selected_task_changed()
        self.viewer.reset_view()

    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()

        task_config = selected_lamella.task_config[selected_stage_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)


        # display milling stages from other tasks as background in the milling task editor for context
        milling_task_config = copy.deepcopy(task_config.milling)
        background_milling_stages = []

        if related_configs := task_config.related_tasks:
            for related_task_name, related_task_config in selected_lamella.task_config.items():
                if isinstance(related_task_config, tuple(related_configs)):
                    cfg = selected_lamella.task_config.get(related_task_name, None)
                    if cfg is not None and cfg.milling:
                        for mcfg in cfg.milling.values():
                            background_milling_stages.extend(mcfg.stages)

        self.milling_task_editor.set_task_configs(milling_task_config)
        self.milling_task_editor.config_widget.set_background_milling_stages(background_milling_stages)
        self.milling_task_editor.config_widget.milling_editor_widget.update_milling_stage_display()

        if task_config.milling:
            self._on_milling_fov_changed(task_config.milling)
            self.milling_task_collapsible.setVisible(True)
        else:
            self.milling_task_editor.remove_all_tasks()
            self.milling_task_collapsible.setVisible(False)
            if "Milling Patterns" in self.viewer.layers:
                self.viewer.layers.remove("Milling Patterns") # type: ignore
            if "Milling Alignment Area" in self.viewer.layers:
                self.viewer.layers.remove("Milling Alignment Area") # type: ignore
        self.milling_task_editor.setEnabled(bool(task_config.milling))

        # display label showing task has been completed
        msg = ""
        if selected_stage_name in [t.name for t in selected_lamella.task_history]:
            msg = f"Task '{selected_stage_name}' has been completed."
        self.label_status.setText(msg)
        self.label_status.setVisible(bool(msg))

        self._draw_point_of_interest(selected_lamella.poi)

    def _on_milling_fov_changed(self, config: Dict[str, FibsemMillingTaskConfig]):
        """Display a warning if the milling FoV does not match the image FoV."""
        try:
            key = list(config.keys())[0]
            milling_fov = config[key].field_of_view
            image_hfw = self.milling_task_editor.config_widget.milling_editor_widget.image.metadata.image_settings.hfw # type: ignore
            if not np.isclose(milling_fov, image_hfw):
                milling_fov_um = format_value(milling_fov, unit='m', precision=0)
                image_fov_um = format_value(image_hfw, unit='m', precision=0)
                self.label_warning.setText(f"Milling Task FoV ({milling_fov_um}) does not match image FoV ({image_fov_um}).")
                self.label_warning.setVisible(True)
                return
        except Exception as e:
            logging.warning(f"Could not compare milling FoV and image FoV: {e}")

        self.label_warning.setVisible(False)

    def _on_milling_task_config_updated(self, configs: Dict[str, FibsemMillingTaskConfig]):
        """Callback when the milling task config is updated."""

        selected_task_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        selected_lamella.task_config[selected_task_name].milling = copy.deepcopy(configs)
        logging.info(f"Updated {selected_lamella.name}, {selected_task_name} Task, Milling Tasks: {list(configs.keys())} ")

        # TODO: support position sync between milling tasks, e.g. sync trench position between rough milling and polishing

        self._save_experiment()

        self._on_milling_fov_changed(configs)

    def _on_task_parameters_config_changed(self, field_name: str, new_value: Any):
        """Callback when the task parameters config is updated."""
        selected_task_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        logging.info(f"Updated {selected_lamella.name}, {selected_task_name} Task Parameters: {field_name} = {new_value}")

        # TODO: we should integrate both milling and parameter updates into a single config update method

        # update parameters in the task config
        setattr(selected_lamella.task_config[selected_task_name], field_name, new_value)

        self._save_experiment()

    def _on_ref_image_settings_changed(self, settings: ReferenceImageParameters):
        """Callback when the image settings are changed."""

        # Update the image settings in the task config
        selected_task_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        selected_lamella.task_config[selected_task_name].reference_imaging = settings

        # Save the experiment
        self._save_experiment()

    def _on_point_of_interest_updated(self, point: Point):
        """Callback when the point of interest is updated."""
        selected_task_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        logging.info(f"Updated {selected_lamella.name}, {selected_task_name} Task, Point of Interest: {point}")

        # update point of interest in the task config
        selected_lamella.poi = point

        # move patterns for tasks with sync_to_poi enabled
        self._sync_task_patterns_to_poi(selected_lamella, point)
        self._draw_point_of_interest(point)

        self._save_experiment()

    def _draw_point_of_interest(self, point: Point):
        """Draw the point of interest on the viewer."""

        if cfg.FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED is False:
            if "Point of Interest" in self.viewer.layers:
                self.viewer.layers.remove("Point of Interest")  # type: ignore
            return

        image_coords = conversions.microscope_image_to_image_coordinates(
            point=point,
            image_shape=self.image.data.shape,
            pixel_size=self.image.metadata.pixel_size.x,
        )

        if "Point of Interest" in self.viewer.layers:
            self.viewer.layers["Point of Interest"].data = np.array([[image_coords.y, image_coords.x]])  # yx format
        else:
            add_points_layer(
                viewer=self.viewer,
                data=np.array([[image_coords.y, image_coords.x]]),  # yx format
                name="Point of Interest",
                size=20,
                face_color='magenta',
                border_color='white',
                symbol='cross',
                blending='additive',
                border_width=None,
                border_width_is_relative=False,
            )

    def _on_single_click(self, viewer: napari.Viewer, event):
        """Handle single click events in the viewer."""

        if REFERENCE_IMAGE_LAYER_NAME not in self.viewer.layers:
            return

        if event.button != 2 or event.type != "mouse_press":  # Right click only
            return

        if self.milling_task_editor.config_widget.milling_editor_widget.is_movement_locked:
            logging.warning("Movement is locked. Cannot move milling patterns.")
            return

        if self.milling_task_editor.config_widget.milling_editor_widget.is_correlation_open:
            logging.info("Correlation tool is open, ignoring click event.")
            return

        event.handled = True

        # convert from image coordinates to microscope coordinates
        coords = self.viewer.layers[REFERENCE_IMAGE_LAYER_NAME].world_to_data(event.position)

        if not is_inside_image_bounds(coords=coords, shape=self.image.data.shape):
            logging.info(f"Clicked outside image bounds: {coords}, not updating point of interest.")
            return

        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        # Show context menu
        config = ContextMenuConfig()
        if cfg.FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED:
            config.add_action(
                "Move Point of Interest Here",
                callback=lambda: self._on_point_of_interest_updated(point_clicked),
            )
        selected_stage_name = self.milling_task_editor.config_widget.milling_editor_widget.selected_stage_name
        num_stages = len(self.milling_task_editor.config_widget.milling_editor_widget._milling_stages)
        if num_stages > 1:
            config.add_action(
                "Move All Patterns Here",
                callback=lambda: self.milling_task_editor.config_widget.milling_editor_widget.move_patterns_to_point(point_clicked),
            )
        if selected_stage_name is not None and num_stages > 0:
            config.add_action(
                label=f"Move {selected_stage_name} Pattern Here",
                callback=lambda: self.milling_task_editor.config_widget.milling_editor_widget.move_patterns_to_point(point_clicked, move_all=False),
            )
        menu = ContextMenu(config, parent=self)
        menu.show_at_cursor()

    def _sync_task_patterns_to_poi(self, lamella: Lamella, point: Point):
        """Move milling patterns to the point of interest for tasks with sync_to_poi enabled."""
        synced_tasks = []

        for task_name, task_config in lamella.task_config.items():
            # check if task has sync_to_poi enabled
            if not getattr(task_config, "sync_to_poi", False):
                continue
            if not task_config.milling:
                continue

            for milling_config in task_config.milling.values():
                if not milling_config.stages:
                    continue
                # calculate offset from the first stage's pattern point
                diff = point - milling_config.stages[0].pattern.point
                for stage in milling_config.stages:
                    stage.pattern.point = stage.pattern.point + diff

            synced_tasks.append(task_name)


        if synced_tasks:
            logging.info(f"Synced patterns to POI for tasks: {synced_tasks}")
            self._on_selected_task_changed()  # refresh milling task editor to show updated pattern positions

    def _open_correlation_widget(self):
        """Open the 3D correlation widget in the viewer."""

        # close existing correlation widget if open
        try:
            if hasattr(self, "correlation_widget") and self.correlation_widget is not None:
                self.correlation_widget.close()
                self.correlation_widget = None
        except Exception as e:
            logging.warning(f"Error closing correlation widget: {e}")

        # snapshot existing layers so we can restore visibility later
        self._existing_layers = [layer.name for layer in self.viewer.layers]
        for layer_name in self._existing_layers:
            self.viewer.layers[layer_name].visible = False

        from fibsem.correlation.app import CorrelationUI
        self.correlation_widget = CorrelationUI(viewer=self.viewer, parent_ui=self)
        self.correlation_widget.continue_pressed_signal.connect(self._handle_correlation_continue_signal)
        self.milling_task_editor.config_widget.milling_editor_widget.set_movement_lock(True)

        # load fib image
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        project_path = str(selected_lamella.path) if selected_lamella is not None else ""
        self.correlation_widget.set_project_path(project_path)

        fib_image: FibsemImage = self.image
        if fib_image is not None and fib_image.metadata is not None:
            fib_filename = fib_image.metadata.image_settings.filename if fib_image.metadata.image_settings.filename else "FIB Image"
            self.correlation_widget.load_fib_image(
                image=fib_image.filtered_data,
                pixel_size=fib_image.metadata.pixel_size.x,
                filename=fib_filename,
            )

        self.viewer.window.add_dock_widget(
            self.correlation_widget,
            area="right",
            name="3DCT Correlation",
            tabify=True,
        )

    def _handle_correlation_continue_signal(self, data: dict):
        """Handle the result from the correlation widget."""
        logging.info(f"correlation-data: {data}")

        poi = data.get("poi", None)
        if poi is not None:
            point = Point(x=poi[0], y=poi[1])
            self._on_point_of_interest_updated(point)

        self._close_correlation_widget()

    def _close_correlation_widget(self):
        """Close the correlation widget and restore layer visibility."""
        # restore layers visibility
        for layer_name in self._existing_layers:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = True

        # remove dock widget and clean up
        self.viewer.window.remove_dock_widget(self.correlation_widget)
        self.correlation_widget = None
        self._existing_layers = []
        self.milling_task_editor.config_widget.milling_editor_widget.set_movement_lock(False)

    def _on_apply_to_other_clicked(self):
        """Open dialog to apply this lamella's config to other lamella."""
        source_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        if source_lamella is None:
            return

        experiment = self.parent_widget.experiment
        if experiment is None or len(experiment.positions) < 2:
            QMessageBox.information(
                self,
                "Not Enough Lamella",
                "There must be at least 2 lamella in the experiment to apply configurations.",
            )
            return

        other_names = [p.name for p in experiment.positions if p._id != source_lamella._id]
        task_names = list(source_lamella.task_config.keys())

        dialog = ApplyLamellaConfigDialog(
            source_name=source_lamella.name,
            other_lamella_names=other_names,
            task_names=task_names,
            parent=self,
        )

        if dialog.exec_() != QDialog.Accepted:
            return

        selected_lamella_names = dialog.get_selected_lamella_names()
        selected_tasks = dialog.get_selected_tasks()
        update_base_protocol = dialog.get_update_base_protocol()

        if not selected_lamella_names or not selected_tasks:
            return

        # Apply configs via experiment method
        updated_count = experiment.apply_lamella_config(
            lamella_names=selected_lamella_names,
            task_names=selected_tasks,
            source_lamella_name=source_lamella.name,
            update_base_protocol=update_base_protocol,
        )

        self._save_experiment()

        # Show success message
        task_list = ", ".join(f"'{t}'" for t in selected_tasks)
        names_str = ", ".join(selected_lamella_names)
        msg = (
            f"Applied {len(selected_tasks)} task(s) ({task_list}) "
            f"to {updated_count} lamella ({names_str})."
        )
        if update_base_protocol:
            msg += "\nBase protocol was also updated."
        QMessageBox.information(self, "Apply Complete", msg)
        logging.info(msg)

    def _on_defect_state_changed(self, defect: DefectState):
        """Callback when the defect state is changed."""
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        if selected_lamella is None:
            return
        selected_lamella.defect = defect
        self._save_experiment()

    def _save_experiment(self):
        """Save the experiment."""
        # save the experiment
        if self.parent_widget is not None and self.parent_widget.experiment is not None:
            self.parent_widget.experiment.save() # TODO: migrate to shared data model


def show_protocol_editor(parent: 'AutoLamellaUI',):
    """Show the AutoLamella protocol editor widget."""
    viewer = napari.Viewer(title="AutoLamella Protocol Editor")
    widget = AutoLamellaProtocolEditorWidget(viewer=viewer, 
                                             parent=parent)
    viewer.window.add_dock_widget(widget, area='right', name='AutoLamella Protocol Editor')
    return widget