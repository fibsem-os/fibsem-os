from __future__ import annotations

import copy
import datetime
import glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions, constants
from fibsem.applications.autolamella.structures import (
    Lamella,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    Point,
    ReferenceImageParameters,
)
from fibsem.ui.widgets.canvas_state import AlignmentSpec, PointsSpec
from fibsem.ui.widgets.quad_view import LamellaEditorView, MicroscopeViewController
from fibsem.ui.widgets.autolamella_apply_protocol_dialog import ApplyLamellaConfigDialog
from fibsem.ui.widgets.autolamella_task_config_widget import (
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui.widgets.custom_widgets import (
    ContextMenuConfig,
    IconToolButton,
    TaskNameListWidget,
)
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)
from fibsem.utils import format_value
from fibsem.applications.autolamella.ui.autolamella_fluorescence_acquisition_task_config_widget import (
    AutoLamellaFluorescenceAcquisitionTaskConfigWidget,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
    SpotBurnFiducialTaskConfig,
)
from fibsem.ui.widgets.spot_burn_coordinates_widget import (
    SpotBurnCoordinatesWidget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI
    from fibsem.correlation.structures import CorrelationResult
    from fibsem.imaging.spot import SpotBurnSettings

# Reducer overlay ids on the FIB (ION) canvas
POI_OVERLAY_ID = "poi"
ALIGNMENT_OVERLAY_ID = "alignment_area"


class AutoLamellaProtocolEditorWidget(QWidget):
    """A widget to edit the AutoLamella protocol."""

    def __init__(self, parent: "AutoLamellaUI"):
        super().__init__(parent)
        self.parent_widget = parent

        # Canvas display: own a controller driving a task-driven stacked view (FIB +
        # SEM beside it + FM page). Created eagerly (no microscope needed) so the host
        # tab can embed ``view_controller.widget`` before the microscope connects.
        self.view_controller = MicroscopeViewController(view=LamellaEditorView())

        self.image: FibsemImage
        self.fm_image: Optional[FluorescenceImage] = None
        self.show_related_milling_tasks = True
        self.show_sem_image = False
        self.show_alignment_area = True
        self.alignment_area_editable = False
        self._active_lamella_name: Optional[str] = None
        self._active_task_name: Optional[str] = None
        self._selected_lamella: Optional[Lamella] = None
        self._overlay_wired = False  # subscribed to controller.overlay_edited

        if self.parent_widget.microscope is None:
            return

        self._on_microscope_connected()

    def _on_microscope_connected(self):
        """Callback when the microscope is connected."""
        if self.parent_widget.microscope is None:
            raise ValueError("Microscope in parent widget is None, cannot proceed.")
        self.microscope = self.parent_widget.microscope
        if hasattr(self, "milling_task_editor"):
            # Reconnect: widget already initialized, just update microscope reference
            self.milling_task_editor.microscope = self.microscope
        else:
            # First connect: build the full UI
            self._create_widgets()
            self._initialise_widgets()

    def set_experiment(self):
        """Set the experiment for the protocol editor."""
        self._refresh_experiment_positions()

    def set_active_lamella_name(
        self, lamella_name: Optional[str], task_name: Optional[str] = None
    ) -> None:
        """Track which lamella/task is actively being processed and lock/unlock editing accordingly."""
        self._active_lamella_name = lamella_name
        self._active_task_name = task_name
        self._apply_editing_lock(self._is_editing_locked())

    def _is_editing_locked(self) -> bool:
        if self._active_lamella_name is None:
            return False
        selected_name = self._selected_lamella.name if self._selected_lamella else ""
        selected_task = self.listWidget_selected_task.selected_task
        return selected_name == self._active_lamella_name and (
            self._active_task_name is None or selected_task == self._active_task_name
        )

    def _apply_editing_lock(self, locked: bool) -> None:
        self.task_parameters_config_widget.setEnabled(not locked)
        self.ref_image_params_widget.setEnabled(not locked)
        self.milling_task_editor.setEnabled(not locked)
        if locked:
            self.label_lamella_warning.setText(
                "This lamella is currently being processed and cannot be edited."
            )
        else:
            self.label_lamella_warning.setText("")

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        self.milling_task_editor = MillingTaskViewerWidget(
            microscope=self.microscope,
            viewer=None,
            milling_enabled=False,
            parent=self,
        )
        self.milling_task_editor.setMinimumHeight(550)
        # drive patterns/reposition on the FIB canvas (not napari)
        self.milling_task_editor.set_controller(self.view_controller)
        self.milling_task_editor.set_alignment_area_visible(False)

        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(
            parent=self
        )

        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)

        self.fluorescence_acquisition_task_config_widget = (
            AutoLamellaFluorescenceAcquisitionTaskConfigWidget(
                microscope=self.microscope, config=None, parent=self
            )
        )

        self.spot_burn_coordinates_widget = SpotBurnCoordinatesWidget(
            controller=self.view_controller,
            beam=BeamType.ION,
            parent=self,
        )

        self.pushButton_refresh_positions = IconToolButton(
            icon="mdi:refresh",
            tooltip="Reload task configs and images for the current lamella.",
        )
        self.pushButton_refresh_positions.clicked.connect(
            self._refresh_experiment_positions
        )
        self.pushButton_apply_to_other = IconToolButton(
            icon="mdi:file-transfer",
            tooltip="Apply this lamella's task configurations to other lamella in the experiment.",
        )
        self.pushButton_apply_to_other.clicked.connect(self._on_apply_to_other_clicked)
        self.pushButton_open_correlation = IconToolButton(
            icon="mdi:target",
            tooltip="Open the 3D correlation tool to align the FIB and FM images.",
        )
        self.pushButton_open_correlation.clicked.connect(self._open_correlation_dialog)

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
        self.pushButton_toggle_related_tasks.toggled.connect(
            self._on_toggle_related_tasks
        )

        self.pushButton_toggle_alignment_area = IconToolButton(
            icon="mdi:crop-free",
            checked_icon="mdi:crop-free",
            tooltip="Show alignment area.",
            checked_tooltip="Hide alignment area.",
            checkable=True,
            checked=self.show_alignment_area,
        )
        self.pushButton_toggle_alignment_area.toggled.connect(
            self._on_toggle_alignment_area
        )

        self.pushButton_edit_alignment_area = IconToolButton(
            icon="mdi:pencil-off",
            checked_icon="mdi:pencil",
            tooltip="Enable alignment area editing.",
            checked_tooltip="Disable alignment area editing.",
            checkable=True,
            checked=self.alignment_area_editable,
        )
        self.pushButton_edit_alignment_area.setEnabled(self.show_alignment_area)
        self.pushButton_edit_alignment_area.toggled.connect(
            self._on_toggle_alignment_area_editable
        )

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
        self.button_layout.addWidget(self.pushButton_toggle_alignment_area)
        self.button_layout.addWidget(self.pushButton_edit_alignment_area)
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
        self.label_status.setStyleSheet("color: cyan;")
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
        self.scroll_area.setWidgetResizable(True)  # required to resize
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        self.scroll_content_layout.addLayout(self.grid_layout)
        self.scroll_content_layout.addWidget(self.task_parameters_config_widget)
        self.scroll_content_layout.addWidget(self.spot_burn_coordinates_widget)
        self.scroll_content_layout.addWidget(self.ref_image_params_widget)  # type: ignore
        self.scroll_content_layout.addWidget(self.milling_task_editor)  # type: ignore
        self.scroll_content_layout.addWidget(
            self.fluorescence_acquisition_task_config_widget
        )  # type: ignore
        self.scroll_content_layout.addStretch()

        self.scroll_content_widget = QWidget()
        self.scroll_content_widget.setLayout(self.scroll_content_layout)
        self.scroll_area.setWidget(self.scroll_content_widget)  # type: ignore
        self.main_layout.addWidget(self.scroll_area)  # type: ignore

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""
        self.listWidget_selected_task.task_selected.connect(
            lambda _: self._on_selected_task_changed()
        )
        self.milling_task_editor.settings_changed.connect(
            self._on_milling_task_config_updated
        )
        self.task_parameters_config_widget.parameter_changed.connect(
            self._on_task_parameters_config_changed
        )
        self.ref_image_params_widget.settings_changed.connect(
            self._on_ref_image_settings_changed
        )
        self.milling_task_editor.set_right_click_menu_actions(
            self._add_poi_context_menu_action
        )
        self._correlation_open = False
        self.fluorescence_acquisition_task_config_widget.settings_changed.connect(
            self._on_fluorescence_acquisition_settings_changed
        )
        self.spot_burn_coordinates_widget.settings_changed.connect(
            self._on_spot_burn_coordinates_changed
        )
        if not self._overlay_wired:
            self.view_controller.overlay_edited.connect(
                self._on_controller_overlay_edited
            )
            self._overlay_wired = True

        if (
            self.parent_widget.experiment is not None
            and self.parent_widget.experiment.positions
        ):  # type: ignore
            self._selected_lamella = self.parent_widget.experiment.positions[0]
            self._on_selected_lamella_changed()

    def _refresh_experiment_positions(self):
        """Refresh the editor from the current experiment positions."""
        experiment = self.parent_widget.experiment
        if experiment is None:
            return
        current_name = self._selected_lamella.name if self._selected_lamella else ""
        self._selected_lamella = experiment.get_lamella_by_name(current_name) or (
            experiment.positions[0] if experiment.positions else None
        )
        if self._selected_lamella is not None:
            self._on_selected_lamella_changed()

    def select_lamella(self, name: str) -> None:
        """Select a lamella by name and refresh the editor. Called externally (e.g. card click)."""
        if self.parent_widget.experiment is None or not hasattr(
            self, "label_lamella_name"
        ):
            return
        lamella = self.parent_widget.experiment.get_lamella_by_name(name)
        if lamella is not None:
            self._selected_lamella = lamella
            self._on_selected_lamella_changed()

    def _on_selected_lamella_changed(self):
        """Callback when the selected lamella changes."""
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return
        self.label_lamella_name.setText(selected_lamella.name)

        task_names = self._sort_task_names_by_workflow(
            list(selected_lamella.task_config.keys())
        )
        self.listWidget_selected_task.set_tasks(task_names)

        # load fluorescence image
        filenames = sorted(glob.glob(os.path.join(selected_lamella.path, "*.ome.tiff")))
        self.combobox_fm_filenames.blockSignals(True)
        self.combobox_fm_filenames.clear()
        for f in filenames:
            self.combobox_fm_filenames.addItem(os.path.basename(f))
        self.combobox_fm_filenames.setCurrentIndex(
            len(filenames) - 1
        )  # Select latest by default
        self.combobox_fm_filenames.blockSignals(False)

        # load fib reference image
        self.combobox_fib_filenames.blockSignals(True)
        fib_filenames = sorted(
            glob.glob(os.path.join(selected_lamella.path, "*_ib.tif"))
        )
        fib_filenames = [
            f for f in fib_filenames if "alignment" not in f
        ]  # show all non-alignment images

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
            matching_filenames = glob.glob(
                os.path.join(selected_lamella.path, reference_image_filename)
            )
            if len(matching_filenames) > 0:
                # pick the latest by modification time
                latest_task_filename = os.path.basename(
                    sorted(matching_filenames, key=os.path.getmtime)[-1]
                )

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
        sem_filenames = sorted(
            glob.glob(os.path.join(selected_lamella.path, "*_eb.tif"))
        )
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
            matching_sem_filenames = glob.glob(
                os.path.join(selected_lamella.path, reference_sem_filename)
            )
            if len(matching_sem_filenames) > 0:
                latest_sem_task_filename = os.path.basename(
                    sorted(matching_sem_filenames, key=os.path.getmtime)[-1]
                )

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
        self.combobox_sem_filenames.setEnabled(
            self.show_sem_image and len(sem_filenames) > 0
        )
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
        self.combobox_sem_filenames.setEnabled(
            self.show_sem_image and self.combobox_sem_filenames.count() > 0
        )
        self._on_image_selected(0)

    def _on_toggle_related_tasks(self, checked: bool):
        """Toggle displaying related milling tasks."""
        self.show_related_milling_tasks = checked
        self._on_selected_task_changed()

    def _on_image_selected(self, index):
        """Callback when an image is selected — load the images and push them to the canvases."""
        p = self._selected_lamella
        if p is None:
            return
        fm_filename = self.combobox_fm_filenames.currentText()
        fib_filename = self.combobox_fib_filenames.currentText()
        sem_filename = self.combobox_sem_filenames.currentText()

        # load the fib reference image
        reference_image_path = os.path.join(p.path, fib_filename)
        if os.path.exists(reference_image_path) and os.path.isfile(
            reference_image_path
        ):
            self.image = FibsemImage.load(reference_image_path)
        else:
            self.image = FibsemImage.generate_blank_image(hfw=100e-6, random=False)

        # load the sem reference image
        sem_image = None
        if self.show_sem_image and sem_filename:
            sem_image_path = os.path.join(p.path, sem_filename)
            if os.path.exists(sem_image_path) and os.path.isfile(sem_image_path):
                sem_image = FibsemImage.load(sem_image_path)

        # load the fluorescence image
        self.fm_image = None
        fm_image_path = os.path.join(p.path, fm_filename) if fm_filename else None
        if fm_image_path and os.path.exists(fm_image_path):
            self.fm_image = FluorescenceImage.load(fm_image_path)

        # push the FIB image to the ION canvas (patterns / POI / alignment / spot-burn host)
        self.view_controller.set_image(BeamType.ION, self.image)
        # keep the spot-burn editor's 0-1 <-> pixel conversion in step with the FIB image
        self.spot_burn_coordinates_widget.set_image_shape(self.image.data.shape[:2])

        # SEM reference: shown beside FIB when toggled on and available
        self._update_sem_display(sem_image)

        self._on_selected_task_changed()

    def _update_sem_display(self, sem_image: Optional[FibsemImage]) -> None:
        """Show/hide the SEM canvas beside FIB and feed it the reference image."""
        show = sem_image is not None
        self.view_controller.widget.set_sem_visible(show)
        if show:
            self.view_controller.set_image(BeamType.ELECTRON, sem_image)

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
            key=lambda name: (
                workflow_order.get(name, default_order),
                original_order[name],
            ),
        )

    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.listWidget_selected_task.selected_task
        if not selected_stage_name:
            return
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return

        task_config = selected_lamella.task_config[selected_stage_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)

        # display milling stages from other tasks as background in the milling task editor for context
        milling_task_config = copy.deepcopy(task_config.milling)
        background_milling_stages = []

        if self.show_related_milling_tasks and (
            related_configs := task_config.related_tasks
        ):
            for (
                related_task_name,
                related_task_config,
            ) in selected_lamella.task_config.items():
                if isinstance(related_task_config, tuple(related_configs)):
                    cfg = selected_lamella.task_config.get(related_task_name, None)
                    if cfg is not None and cfg.milling:
                        for mcfg in cfg.milling.values():
                            background_milling_stages.extend(mcfg.enabled_stages)

        if milling_task_config:
            self._current_milling_key = next(iter(milling_task_config))
            # image_layer is unused on the canvas path (patterns render via the reducer)
            self.milling_task_editor.set_fib_image(self.image, None)
            self.milling_task_editor.set_config(
                milling_task_config[self._current_milling_key]
            )
            self.milling_task_editor.set_background_milling_stages(
                background_milling_stages
            )
            self._on_milling_fov_changed(milling_task_config)
            self.milling_task_editor.setVisible(True)
        else:
            self._current_milling_key = None
            self.milling_task_editor.clear()  # drops the milling overlay via the reducer
            self.milling_task_editor.setVisible(False)
        self.milling_task_editor.setEnabled(bool(task_config.milling))

        # Re-apply lock if this lamella/task is currently being processed
        self._apply_editing_lock(self._is_editing_locked())

        # display label showing task has been completed
        msg = "Task not yet completed."
        if selected_stage_name in [t.name for t in selected_lamella.task_history]:
            msg = f"Task '{selected_stage_name}' has been completed."
        self.label_status.setText(msg)
        self.label_status.setVisible(bool(msg))

        # special handling for fluorescence acquisition task
        is_fluorescence_task = isinstance(task_config, AcquireFluorescenceImageConfig)
        self.fluorescence_acquisition_task_config_widget.setVisible(
            is_fluorescence_task
        )
        self.task_parameters_config_widget.setVisible(not is_fluorescence_task)
        self.ref_image_params_widget.setVisible(not is_fluorescence_task)
        self.combobox_fm_filenames.setEnabled(is_fluorescence_task)
        self.combobox_fm_filenames_label.setEnabled(is_fluorescence_task)
        self.combobox_fm_filenames.setToolTip(
            ""
            if is_fluorescence_task
            else "Only available for Acquire Fluorescence Image tasks"
        )
        if is_fluorescence_task:
            self.fluorescence_acquisition_task_config_widget.set_task_config(
                task_config
            )
            if self.fm_image is not None:
                self.view_controller.set_fm_image(self.fm_image)

        # special handling for spot burn fiducial task
        is_spot_burn_task = isinstance(task_config, SpotBurnFiducialTaskConfig)
        self.spot_burn_coordinates_widget.setVisible(is_spot_burn_task)
        if is_spot_burn_task:
            self.spot_burn_coordinates_widget.set_settings(task_config.to_settings())

        self._draw_point_of_interest(selected_lamella.poi)
        self._draw_alignment_area()

        # task type drives which canvas page is visible (mirrors the old per-task napari
        # layer-visibility toggle): the FM composite for fluorescence, else FIB (+ SEM)
        if is_fluorescence_task:
            self.view_controller.widget.show_fluorescence()
        else:
            self.view_controller.widget.show_beams()

    def _on_milling_fov_changed(self, config: Dict[str, FibsemMillingTaskConfig]):
        """Display a warning if the milling FoV does not match the image FoV."""
        try:
            key = next(iter(config))
            milling_fov = config[key].field_of_view
            image_hfw = self.image.metadata.image_settings.hfw  # type: ignore
            if not np.isclose(milling_fov, image_hfw):
                milling_fov_um = format_value(milling_fov, unit="m", precision=0)
                image_fov_um = format_value(image_hfw, unit="m", precision=0)
                self.label_warning.setText(
                    f"Milling Task FoV ({milling_fov_um}) does not match image FoV ({image_fov_um})."
                )
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
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return
        key = getattr(self, "_current_milling_key", None)
        if key:
            selected_lamella.task_config[selected_task_name].milling[key] = config
            logging.info(
                f"Updated {selected_lamella.name}, {selected_task_name} Task, milling key '{key}'"
            )

        self._save_experiment()

        self._on_milling_fov_changed(
            selected_lamella.task_config[selected_task_name].milling
        )

    def _on_task_parameters_config_changed(self, field_name: str, new_value: Any):
        """Callback when the task parameters config is updated."""
        selected_task_name = self.listWidget_selected_task.selected_task
        if not selected_task_name:
            return
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return
        logging.info(
            f"Updated {selected_lamella.name}, {selected_task_name} Task Parameters: {field_name} = {new_value}"
        )

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

        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return
        selected_lamella.task_config[selected_task_name].reference_imaging = settings

        # Save the experiment
        self._save_experiment()

    def _on_fluorescence_acquisition_settings_changed(
        self, config: "AcquireFluorescenceImageConfig"
    ):
        """Callback when the fluorescence acquisition settings are changed."""

        # Update the task config
        selected_task_name = self.listWidget_selected_task.selected_task
        selected_lamella = self._selected_lamella
        if selected_lamella is None or selected_task_name is None:
            return
        selected_lamella.task_config[selected_task_name] = config
        logging.info(
            f"Updated {selected_lamella.name}, {selected_task_name} Fluorescence Acquisition Settings"
        )

        # Save the experiment
        self._save_experiment()

    def _on_spot_burn_coordinates_changed(self, settings: SpotBurnSettings):
        """Callback when the spot burn coordinates are edited on the canvas.

        Only the coordinates are taken from the editor's settings — milling current and
        exposure stay owned by the generic task-parameters widget.
        """
        selected_task_name = self.listWidget_selected_task.selected_task
        selected_lamella = self._selected_lamella
        if selected_lamella is None or selected_task_name is None:
            return
        task_config = selected_lamella.task_config.get(selected_task_name)
        if isinstance(task_config, SpotBurnFiducialTaskConfig):
            task_config.coordinates = list(settings.coordinates)
        logging.info(
            f"Updated {selected_lamella.name}, {selected_task_name} Spot Burn Coordinates"
        )
        self._save_experiment()

    def _on_point_of_interest_updated(self, point: Point):
        """Callback when the point of interest is updated."""
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return

        logging.info(f"Updated {selected_lamella.name}, Point of Interest: {point}")

        # update point of interest in the task config
        selected_lamella.poi = point

        # update point of interest in the task config, and optionally sync pattern positions to the new point
        selected_lamella.poi = point
        synced_tasks = selected_lamella.sync_tasks_to_poi()
        self._draw_point_of_interest(point)
        if synced_tasks:
            self._on_selected_task_changed()  # refresh milling task editor to show updated pattern positions

        self._save_experiment()

    def _draw_point_of_interest(self, point: Point):
        """Draw the point-of-interest marker on the FIB canvas.

        Display-only (``modal`` + never armed → inert): the POI is moved via the milling
        editor's right-click "Move Point of Interest Here", not by dragging.
        """
        if getattr(self, "image", None) is None or self.image.metadata is None:
            return
        image_coords = conversions.microscope_image_to_image_coordinates(
            point=point,
            image_shape=self.image.data.shape,
            pixel_size=self.image.metadata.pixel_size.x,
        )
        self.view_controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id=POI_OVERLAY_ID,
                points=[(image_coords.x, image_coords.y)],
                color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                add_on_right_click=False,
                removable=False,
                modal=True,
                legend_label="Point of Interest",
            ),
        )

    def _on_toggle_alignment_area(self, checked: bool):
        self.show_alignment_area = checked
        if not checked:
            self.alignment_area_editable = False
            self.pushButton_edit_alignment_area.blockSignals(True)
            self.pushButton_edit_alignment_area.setChecked(False)
            self.pushButton_edit_alignment_area.blockSignals(False)
        self.pushButton_edit_alignment_area.setEnabled(checked)
        self._draw_alignment_area()
        if not checked:
            self.view_controller.arm_overlay(BeamType.ION, None)

    def _on_toggle_alignment_area_editable(self, checked: bool):
        self.alignment_area_editable = checked
        self._draw_alignment_area()
        # arming is an explicit user action here (redraws never touch arming, so they
        # can't steal input focus from the spot-burn overlay)
        self.view_controller.arm_overlay(
            BeamType.ION,
            ALIGNMENT_OVERLAY_ID if checked else None,
            label="Alignment",
            icon="mdi:vector-rectangle",
        )

    def _draw_alignment_area(self):
        """Draw (or remove) the editable alignment-area rectangle on the FIB canvas.

        Overlay only — arming is handled by the edit toggle so a task-change redraw
        never disarms another overlay. A separate id (not the milling widget's shared
        ``alignment``) keeps the milling read-only display from clobbering it.
        """
        if not self.show_alignment_area or self._selected_lamella is None:
            self.view_controller.remove_overlay(BeamType.ION, ALIGNMENT_OVERLAY_ID)
            return
        self.view_controller.set_overlay(
            BeamType.ION,
            AlignmentSpec(
                id=ALIGNMENT_OVERLAY_ID,
                rect=self._selected_lamella.alignment_area,
                display=True,
                editing=self.alignment_area_editable,
            ),
        )

    def _on_controller_overlay_edited(self, beam, overlay_id: str, value) -> None:
        """Fold a committed alignment-area edit on the FIB canvas back to the lamella."""
        if beam == BeamType.ION and overlay_id == ALIGNMENT_OVERLAY_ID:
            self._on_alignment_area_updated(value)

    def _on_alignment_area_updated(self, rect: FibsemRectangle):
        """Callback when the user drags/resizes the alignment area."""
        if self._selected_lamella is None or not self.alignment_area_editable:
            return
        self._selected_lamella.alignment_area = rect
        self._save_experiment()

    def _add_poi_context_menu_action(
        self, config: ContextMenuConfig, point: Point
    ) -> None:
        """Add POI movement action to the right-click context menu."""
        if self._correlation_open:
            return
        config.add_action(
            "Move Point of Interest Here",
            callback=lambda: self._on_point_of_interest_updated(point),
        )

    def _open_correlation_dialog(self) -> None:
        """Open CorrelationTabWidget in a modal dialog."""
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            logging.error("No lamella selected, cannot open correlation dialog.")
            return

        from fibsem.correlation.ui.widgets.correlation_tab_widget import (
            CorrelationTabDialog,
        )

        project_path = os.path.join(
            selected_lamella.path,
            "Correlation",
            datetime.datetime.now().strftime(constants.DATETIME_FILE),
        )
        os.makedirs(project_path, exist_ok=True)

        dialog = CorrelationTabDialog(parent=self)
        dialog.set_project_dir(project_path)

        fib_image = self.image
        if fib_image is not None:
            dialog.set_fib_image(fib_image)

        fm_image = self.fm_image
        if fm_image is not None:
            dialog.set_fm_image(fm_image)

        if dialog.exec_() == QDialog.Accepted and dialog.result is not None:
            self._handle_correlation_dialog_result(dialog.result)

    def _handle_correlation_dialog_result(self, result: "CorrelationResult") -> None:
        """Handle the CorrelationResult returned from CorrelationTabDialog."""
        if result is None or not result.poi:
            logging.warning("Correlation dialog closed with no POI result.")
            return
        logging.info(
            f"correlation-result: rms={result.rms_error:.3f}, poi={result.poi[0].px_m}"
        )
        poi: Point = result.poi[0].px_m  # Point in metres, same format as old signal
        self._on_point_of_interest_updated(poi)

    def _on_apply_to_other_clicked(self):
        """Open dialog to apply this lamella's config to other lamella."""
        source_lamella = self._selected_lamella
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

        other_names = [
            p.name for p in experiment.positions if p._id != source_lamella._id
        ]
        task_names = self._sort_task_names_by_workflow(
            list(source_lamella.task_config.keys())
        )

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
        if self.parent_widget is not None and self.parent_widget.experiment is not None:
            self.parent_widget.experiment.save(save_protocol=True)
