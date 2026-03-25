
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
from fibsem import conversions
from fibsem.ui.napari.patterns import MILLING_PATTERN_LAYER_NAME
from fibsem.ui.napari.utilities import add_points_layer
from fibsem.utils import format_value
from fibsem.applications.autolamella.structures import (
    Lamella,
)
import fibsem.applications.autolamella.config as cfg
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, Point, ReferenceImageParameters
from fibsem.ui.widgets.autolamella_task_config_widget import (
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import ContextMenuConfig, IconToolButton, TaskNameListWidget, TitledPanel
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI

REFERENCE_IMAGE_LAYER_NAME = "Reference Image (FIB)"
REFERENCE_IMAGE_SEM_LAYER_NAME = "Reference Image (SEM)"
POINT_OF_INTEREST_LAYER_NAME = "Point of Interest"


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
        lamella_content = QWidget()
        lamella_layout = QVBoxLayout(lamella_content)
        lamella_layout.setContentsMargins(0, 0, 0, 0)

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

        lamella_layout.addWidget(self.lamella_list)
        lamella_layout.addLayout(lamella_buttons_layout)

        self.lamella_group = TitledPanel("Target Lamella", content=lamella_content, collapsible=False)

        # --- Task selection ---
        task_content = QWidget()
        task_layout = QVBoxLayout(task_content)
        task_layout.setContentsMargins(0, 0, 0, 0)

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

        task_layout.addWidget(self.task_list)
        task_layout.addLayout(task_buttons_layout)

        self.task_group = TitledPanel("Tasks to Apply", content=task_content, collapsible=False)

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
        self.pushButton_apply.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
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
        self.show_related_milling_tasks = True
        self.show_sem_image = False
        self._active_lamella_name: Optional[str] = None
        self._active_task_name: Optional[str] = None
        self._selected_lamella: Optional[Lamella] = None

        if self.parent_widget.microscope is None:
            return

        self._on_microscope_connected()

    def _on_microscope_connected(self):
        """Callback when the microscope is connected."""
        if self.parent_widget.microscope is None:
            raise ValueError("Microscope in parent widget is None, cannot proceed.")
        self.microscope = self.parent_widget.microscope
        if hasattr(self, 'milling_task_editor'):
            # Reconnect: widget already initialized, just update microscope reference
            self.milling_task_editor.microscope = self.microscope
        else:
            # First connect: build the full UI
            self._create_widgets()
            self._initialise_widgets()

    def set_experiment(self):
        """Set the experiment for the protocol editor."""
        self._refresh_experiment_positions()

    def set_active_lamella_name(self, lamella_name: Optional[str], task_name: Optional[str] = None) -> None:
        """Track which lamella/task is actively being processed and lock/unlock editing accordingly."""
        self._active_lamella_name = lamella_name
        self._active_task_name = task_name
        self._apply_editing_lock(self._is_editing_locked())

    def _is_editing_locked(self) -> bool:
        if self._active_lamella_name is None:
            return False
        selected_name = self._selected_lamella.name if self._selected_lamella else ""
        selected_task = self.listWidget_selected_task.selected_task
        return (
            selected_name == self._active_lamella_name
            and (self._active_task_name is None or selected_task == self._active_task_name)
        )

    def _apply_editing_lock(self, locked: bool) -> None:
        self.task_parameters_config_widget.setEnabled(not locked)
        self.ref_image_params_widget.setEnabled(not locked)
        self.milling_task_editor.setEnabled(not locked)
        if locked:
            self.label_lamella_warning.setText("This lamella is currently being processed and cannot be edited.")
        else:
            self.label_lamella_warning.setText("")

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        self.milling_task_editor = MillingTaskViewerWidget(
            microscope=self.microscope,
            viewer=self.viewer,
            milling_enabled=False,
            parent=self,
        )
        self.milling_task_editor.setMinimumHeight(550)

        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(parent=self)

        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)

        self.pushButton_refresh_positions = IconToolButton(
            icon="mdi:refresh",
            tooltip="Reload task configs and images for the current lamella.",
        )
        self.pushButton_refresh_positions.clicked.connect(self._refresh_experiment_positions)
        self.pushButton_apply_to_other = IconToolButton(
            icon="mdi:file-transfer",
            tooltip="Apply this lamella's task configurations to other lamella in the experiment.",
        )
        self.pushButton_apply_to_other.clicked.connect(self._on_apply_to_other_clicked)
        self.pushButton_open_correlation = IconToolButton(
            icon="mdi:target",
            tooltip="Open the 3D correlation tool to align the FIB and FM images.",
        )
        self.pushButton_open_correlation.clicked.connect(self._open_correlation_widget)
        self.pushButton_open_correlation.setVisible(False) # TMP: disable until refactored

        self.pushButton_toggle_sem_image = IconToolButton(
            icon="mdi:eye-off",
            checked_icon="mdi:eye",
            tooltip="Show SEM reference image.",
            checked_tooltip="Hide SEM reference image.",
            checkable=True,
            checked=self.show_sem_image,
        )
        self.pushButton_toggle_sem_image.toggled.connect(self._on_toggle_sem_image)

        self.pushButton_toggle_related_tasks = IconToolButton(
            icon="mdi:layers-off",
            checked_icon="mdi:layers",
            tooltip="Show related milling tasks.",
            checked_tooltip="Hide related milling tasks.",
            checkable=True,
            checked=self.show_related_milling_tasks,
        )
        self.pushButton_toggle_related_tasks.toggled.connect(self._on_toggle_related_tasks)

        self.label_lamella_name = QLabel("")
        self.label_lamella_name.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #e0e0e0; background: transparent;"
        )

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.label_lamella_name)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.pushButton_refresh_positions)
        self.button_layout.addWidget(self.pushButton_apply_to_other)
        self.button_layout.addWidget(self.pushButton_toggle_sem_image)
        self.button_layout.addWidget(self.pushButton_toggle_related_tasks)
        self.button_layout.addWidget(self.pushButton_open_correlation)
        self.listWidget_selected_task = TaskNameListWidget()
        self.listWidget_selected_task.set_buttons_visible(add=False, remove=False)

        self.combobox_fm_filenames = QComboBox()
        self.combobox_fm_filenames_label = QLabel("FM Z-Stack")
        self.combobox_fm_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_fib_filenames = QComboBox()
        self.combobox_fib_filenames_label = QLabel("FIB Image")
        self.combobox_fib_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_sem_filenames = QComboBox()
        self.combobox_sem_filenames_label = QLabel("SEM Image")
        self.combobox_sem_filenames.currentIndexChanged.connect(self._on_image_selected)
        self.combobox_sem_filenames.setEnabled(self.show_sem_image)

        self.label_lamella_warning = QLabel("")
        self.label_lamella_warning.setStyleSheet("color: orange;")
        self.label_lamella_warning.setWordWrap(True)
        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange;")
        self.label_warning.setWordWrap(True)
        self.label_status = QLabel("")
        self.label_status.setStyleSheet("color: lime;")
        self.label_status.setWordWrap(True)

        self.grid_layout = QGridLayout()
        self.grid_layout.addLayout(self.button_layout, 0, 0, 1, 2)
        self.grid_layout.addWidget(self.listWidget_selected_task, 1, 0, 1, 2)
        self.grid_layout.addWidget(self.combobox_fib_filenames_label, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fib_filenames, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames_label, 3, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames, 3, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames_label, 4, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames, 4, 1, 1, 1)
        self.grid_layout.addWidget(self.label_lamella_warning, 5, 0, 1, 2)
        self.grid_layout.addWidget(self.label_status, 6, 0, 1, 2)
        self.grid_layout.addWidget(self.label_warning, 7, 0, 1, 2)

        # main layout
        self.main_layout = QVBoxLayout(self)
        self.scroll_content_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # required to resize
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)       # type: ignore
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # type: ignore
        self.scroll_content_layout.addLayout(self.grid_layout)
        self.scroll_content_layout.addWidget(self.task_parameters_config_widget)
        self.scroll_content_layout.addWidget(self.ref_image_params_widget)      # type: ignore
        self.scroll_content_layout.addWidget(self.milling_task_editor)     # type: ignore
        self.scroll_content_layout.addStretch()

        self.scroll_content_widget = QWidget()
        self.scroll_content_widget.setLayout(self.scroll_content_layout)
        self.scroll_area.setWidget(self.scroll_content_widget) # type: ignore
        self.main_layout.addWidget(self.scroll_area) # type: ignore

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""
        self.listWidget_selected_task.task_selected.connect(lambda _: self._on_selected_task_changed())
        self.milling_task_editor.settings_changed.connect(self._on_milling_task_config_updated)
        self.task_parameters_config_widget.parameter_changed.connect(self._on_task_parameters_config_changed)
        self.ref_image_params_widget.settings_changed.connect(self._on_ref_image_settings_changed)
        self.milling_task_editor.set_right_click_menu_actions(self._add_poi_context_menu_action)
        self._correlation_open = False

        if self.parent_widget.experiment is not None and self.parent_widget.experiment.positions:  # type: ignore
            self._selected_lamella = self.parent_widget.experiment.positions[0]
            self._on_selected_lamella_changed()

    def _refresh_experiment_positions(self):
        """Refresh the editor from the current experiment positions."""
        experiment = self.parent_widget.experiment
        if experiment is None:
            return
        current_name = self._selected_lamella.name if self._selected_lamella else ""
        self._selected_lamella = (
            experiment.get_lamella_by_name(current_name)
            or (experiment.positions[0] if experiment.positions else None)
        )
        if self._selected_lamella is not None:
            self._on_selected_lamella_changed()

    def select_lamella(self, name: str) -> None:
        """Select a lamella by name and refresh the editor. Called externally (e.g. card click)."""
        if self.parent_widget.experiment is None or not hasattr(self, "label_lamella_name"):
            return
        lamella = self.parent_widget.experiment.get_lamella_by_name(name)
        if lamella is not None:
            self._selected_lamella = lamella
            self._on_selected_lamella_changed()

    def _on_selected_lamella_changed(self):
        """Callback when the selected lamella changes."""
        selected_lamella: Lamella = self._selected_lamella
        if selected_lamella is None:
            return
        self.label_lamella_name.setText(selected_lamella.name)

        task_names = self._sort_task_names_by_workflow(list(selected_lamella.task_config.keys()))
        self.listWidget_selected_task.set_tasks(task_names)

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
        self.combobox_sem_filenames.setEnabled(self.show_sem_image and len(sem_filenames) > 0)
        self.pushButton_toggle_sem_image.setEnabled(len(sem_filenames) > 0)

        # warnings and widget enablement
        lamella_warnings = []
        if not fib_filenames:
            lamella_warnings.append("No FIB reference images found.")
        if selected_lamella.last_completed_task is None:
            lamella_warnings.append("No tasks have been completed for this lamella.")
        self.label_lamella_warning.setText("  ".join(lamella_warnings))

        has_images = bool(fib_filenames)
        self.task_parameters_config_widget.setEnabled(has_images)
        self.ref_image_params_widget.setEnabled(has_images)
        self.milling_task_editor.setEnabled(has_images)

        # Re-apply lock if this lamella/task is the one currently being processed
        self._apply_editing_lock(self._is_editing_locked())

        self._on_image_selected(0)
        self._draw_point_of_interest(selected_lamella.poi)

    def _on_toggle_sem_image(self, checked: bool):
        """Toggle displaying the sem image and enablement of sem controls."""
        self.show_sem_image = checked
        self.combobox_sem_filenames.setEnabled(self.show_sem_image and self.combobox_sem_filenames.count() > 0)
        self._on_image_selected(0)

    def _on_toggle_related_tasks(self, checked: bool):
        """Toggle displaying related milling tasks."""
        self.show_related_milling_tasks = checked
        self._on_selected_task_changed()

    def _on_image_selected(self, index):
        """Callback when an image is selected."""
        p: Lamella = self._selected_lamella
        fib_filename = self.combobox_fib_filenames.currentText()
        sem_filename = self.combobox_sem_filenames.currentText()

        # load the fib reference image
        reference_image_path = os.path.join(p.path, fib_filename)
        if os.path.exists(reference_image_path) and os.path.isfile(reference_image_path):
            self.image = FibsemImage.load(reference_image_path)
        else:
            self.image = FibsemImage.generate_blank_image(hfw=150e-6, random=False)

        # load the sem reference image
        sem_image = None
        if self.show_sem_image and sem_filename:
            sem_image_path = os.path.join(p.path, sem_filename)
            if os.path.exists(sem_image_path) and os.path.isfile(sem_image_path):
                sem_image = FibsemImage.load(sem_image_path)

        # clear existing layers, except images to maintain the layer parameters (e.g. gamma, contrast limits)
        for layer in list(self.viewer.layers): # type: ignore
            if layer.name not in [REFERENCE_IMAGE_LAYER_NAME, REFERENCE_IMAGE_SEM_LAYER_NAME]:
                self.viewer.layers.remove(layer) # type: ignore

        self._update_fib_image_layer()
        self._update_sem_image_layer(sem_image)

        self._on_selected_task_changed()
        self.viewer.reset_view()

    def _update_fib_image_layer(self):
        """Update the FIB reference image layer with the currently selected image, or create it if it doesn't exist."""
        if REFERENCE_IMAGE_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[REFERENCE_IMAGE_LAYER_NAME].data = self.image.filtered_data # type: ignore
        else:
            self.viewer.add_image(
                data=self.image.filtered_data,
                name=REFERENCE_IMAGE_LAYER_NAME,
                colormap="gray",
                blending="additive",
            )

    def _update_sem_image_layer(self, sem_image: Optional[FibsemImage]):
        """Add or update the SEM reference image layer, or remove it if no SEM image is provided."""
        if sem_image is None:
            if REFERENCE_IMAGE_SEM_LAYER_NAME in self.viewer.layers:
                self.viewer.layers.remove(REFERENCE_IMAGE_SEM_LAYER_NAME) # type: ignore
            return
            
        if REFERENCE_IMAGE_SEM_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[REFERENCE_IMAGE_SEM_LAYER_NAME].data = sem_image.filtered_data # type: ignore
        else:
            self.viewer.add_image(
                data=sem_image.filtered_data,
                name=REFERENCE_IMAGE_SEM_LAYER_NAME,
                colormap="gray",
                blending="additive",
            translate=(0, -sem_image.data.shape[1]),
        )


    # TODO: migrate this to a task_config method that returns task names in workflow order, rather than sorting here in the UI,
    def _sort_task_names_by_workflow(self, task_names: List[str]) -> List[str]:
        """Sort task names by experiment workflow order; keep unknown tasks in original order."""
        experiment = self.parent_widget.experiment
        if experiment is None or experiment.task_protocol is None:
            return task_names

        workflow_names = experiment.task_protocol.workflow_config.workflow
        if not workflow_names:
            return task_names

        workflow_order = {name: i for i, name in enumerate(workflow_names)}
        original_order = {name: i for i, name in enumerate(task_names)}
        default_order = len(workflow_order)

        return sorted(
            task_names,
            key=lambda name: (workflow_order.get(name, default_order), original_order[name]),
        )

    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.listWidget_selected_task.selected_task
        if not selected_stage_name:
            return
        selected_lamella: Lamella = self._selected_lamella

        task_config = selected_lamella.task_config[selected_stage_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)

        # display milling stages from other tasks as background in the milling task editor for context
        milling_task_config = copy.deepcopy(task_config.milling)
        background_milling_stages = []

        if self.show_related_milling_tasks and (related_configs := task_config.related_tasks):
            for related_task_name, related_task_config in selected_lamella.task_config.items():
                if isinstance(related_task_config, tuple(related_configs)):
                    cfg = selected_lamella.task_config.get(related_task_name, None)
                    if cfg is not None and cfg.milling:
                        for mcfg in cfg.milling.values():
                            background_milling_stages.extend(mcfg.enabled_stages)

        if milling_task_config:
            self._current_milling_key = next(iter(milling_task_config))
            image_layer = self.viewer.layers[REFERENCE_IMAGE_LAYER_NAME] if REFERENCE_IMAGE_LAYER_NAME in self.viewer.layers else None
            self.milling_task_editor.set_fib_image(self.image, image_layer)
            self.milling_task_editor.set_config(milling_task_config[self._current_milling_key])
            self.milling_task_editor.set_background_milling_stages(background_milling_stages)
            self._on_milling_fov_changed(milling_task_config)
            self.milling_task_editor.setVisible(True)
        else:
            self._current_milling_key = None
            self.milling_task_editor.clear()
            self.milling_task_editor.setVisible(False)
            if MILLING_PATTERN_LAYER_NAME in self.viewer.layers:
                self.viewer.layers.remove(MILLING_PATTERN_LAYER_NAME) # type: ignore
        self.milling_task_editor.setEnabled(bool(task_config.milling))

        # Re-apply lock if this lamella/task is currently being processed
        self._apply_editing_lock(self._is_editing_locked())

        # display label showing task has been completed
        msg = "Task not yet completed."
        if selected_stage_name in [t.name for t in selected_lamella.task_history]:
            msg = f"Task '{selected_stage_name}' has been completed."
        self.label_status.setText(msg)
        self.label_status.setVisible(bool(msg))

        self._draw_point_of_interest(selected_lamella.poi)

    def _on_milling_fov_changed(self, config: Dict[str, FibsemMillingTaskConfig]):
        """Display a warning if the milling FoV does not match the image FoV."""
        try:
            key = next(iter(config))
            milling_fov = config[key].field_of_view
            image_hfw = self.image.metadata.image_settings.hfw  # type: ignore
            if not np.isclose(milling_fov, image_hfw):
                milling_fov_um = format_value(milling_fov, unit='m', precision=0)
                image_fov_um = format_value(image_hfw, unit='m', precision=0)
                self.label_warning.setText(f"Milling Task FoV ({milling_fov_um}) does not match image FoV ({image_fov_um}).")
                self.label_warning.setVisible(True)
                return
        except Exception as e:
            logging.warning(f"Could not compare milling FoV and image FoV: {e}")

        self.label_warning.setVisible(False)

    def _on_milling_task_config_updated(self, config: FibsemMillingTaskConfig):
        """Callback when the milling task config is updated."""

        selected_task_name = self.listWidget_selected_task.selected_task
        if not selected_task_name:
            return
        selected_lamella: Lamella = self._selected_lamella
        key = getattr(self, '_current_milling_key', None)
        if key:
            selected_lamella.task_config[selected_task_name].milling[key] = config
            logging.info(f"Updated {selected_lamella.name}, {selected_task_name} Task, milling key '{key}'")

        self._save_experiment()

        self._on_milling_fov_changed(selected_lamella.task_config[selected_task_name].milling)

    def _on_task_parameters_config_changed(self, field_name: str, new_value: Any):
        """Callback when the task parameters config is updated."""
        selected_task_name = self.listWidget_selected_task.selected_task
        if not selected_task_name:
            return
        selected_lamella: Lamella = self._selected_lamella
        logging.info(f"Updated {selected_lamella.name}, {selected_task_name} Task Parameters: {field_name} = {new_value}")

        # TODO: we should integrate both milling and parameter updates into a single config update method

        # update parameters in the task config
        setattr(selected_lamella.task_config[selected_task_name], field_name, new_value)

        self._save_experiment()

    def _on_ref_image_settings_changed(self, settings: ReferenceImageParameters):
        """Callback when the image settings are changed."""

        # Update the image settings in the task config
        selected_task_name = self.listWidget_selected_task.selected_task
        if not selected_task_name:
            return
        selected_lamella: Lamella = self._selected_lamella
        selected_lamella.task_config[selected_task_name].reference_imaging = settings

        # Save the experiment
        self._save_experiment()

    def _on_point_of_interest_updated(self, point: Point):
        """Callback when the point of interest is updated."""
        selected_lamella: Lamella = self._selected_lamella
        logging.info(f"Updated {selected_lamella.name}, Point of Interest: {point}")

        # update point of interest in the task config
        selected_lamella.poi = point

        # move patterns for tasks with sync_to_poi enabled
        self._sync_task_patterns_to_poi(selected_lamella, point)
        self._draw_point_of_interest(point)

        self._save_experiment()

    def _draw_point_of_interest(self, point: Point):
        """Draw the point of interest on the viewer."""

        if cfg.FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED is False:
            if POINT_OF_INTEREST_LAYER_NAME in self.viewer.layers:
                self.viewer.layers.remove(POINT_OF_INTEREST_LAYER_NAME)  # type: ignore
            return

        image_coords = conversions.microscope_image_to_image_coordinates(
            point=point,
            image_shape=self.image.data.shape,
            pixel_size=self.image.metadata.pixel_size.x,
        )

        if  POINT_OF_INTEREST_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[POINT_OF_INTEREST_LAYER_NAME].data = np.array([[image_coords.y, image_coords.x]])  # yx format
        else:
            add_points_layer(
                viewer=self.viewer,
                data=np.array([[image_coords.y, image_coords.x]]),  # yx format
                name=POINT_OF_INTEREST_LAYER_NAME,
                size=20,
                face_color='magenta',
                border_color='white',
                symbol='cross',
                blending='additive',
                border_width=None,
                border_width_is_relative=False,
            )

    def _add_poi_context_menu_action(self, config: ContextMenuConfig, point: Point) -> None:
        """Add POI movement action to the right-click context menu."""
        if not cfg.FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED:
            return
        if self._correlation_open:
            return
        config.add_action(
            "Move Point of Interest Here",
            callback=lambda: self._on_point_of_interest_updated(point),
        )

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
        self._correlation_open = True

        # load fib image
        selected_lamella: Lamella = self._selected_lamella
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
        self._correlation_open = False

    def _on_apply_to_other_clicked(self):
        """Open dialog to apply this lamella's config to other lamella."""
        source_lamella: Lamella = self._selected_lamella
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
        task_names = self._sort_task_names_by_workflow(list(source_lamella.task_config.keys()))

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
