
import copy
import glob
import logging
import os
from typing import TYPE_CHECKING, Any, Dict

import napari
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from fibsem.utils import format_value
from fibsem.applications.autolamella.structures import (
    Lamella,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage, ReferenceImageParameters
from fibsem.ui.widgets.autolamella_task_config_widget import (
    AutoLamellaTaskParametersConfigWidget,
)
from fibsem.ui.widgets.milling_task_widget import FibsemMillingTaskWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


class AutoLamellaProtocolEditorWidget(QWidget):
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

        self._create_widgets()
        self._initialise_widgets()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        self.milling_task_collapsible = QCollapsible("Milling Task Editor")
        self.milling_task_editor = FibsemMillingTaskWidget(microscope=self.microscope,
                                                            milling_enabled=False,
                                                            parent=self)
        self.milling_task_collapsible.addWidget(self.milling_task_editor)
        self.milling_task_editor.setMinimumHeight(550)

        self.task_params_collapsible = QCollapsible("Task Parameters", self)
        self.task_parameters_config_widget = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.task_params_collapsible.addWidget(self.task_parameters_config_widget)

        self.image_params_collapsible = QCollapsible("Imaging Parameters", self)
        self.ref_image_params_widget = ReferenceImageParametersWidget(parent=self)
        self.image_params_collapsible.addWidget(self.ref_image_params_widget)

        # lamella, milling controls
        self.label_selected_lamella = QLabel("Lamella")
        self.comboBox_selected_lamella = QComboBox()
        self.pushButton_refresh_positions = QPushButton("Refresh Experiment Positions")
        self.pushButton_refresh_positions.clicked.connect(self._refresh_experiment_positions)
        self.label_selected_milling = QLabel("Task Name")
        self.comboBox_selected_task = QComboBox()

        self.combobox_fm_filenames = QComboBox()
        self.combobox_fm_filenames_label = QLabel("FM Z-Stack")
        self.combobox_fm_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_fib_filenames = QComboBox()
        self.combobox_fib_filenames_label = QLabel("FIB Image")
        self.combobox_fib_filenames.currentIndexChanged.connect(self._on_image_selected)

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
        self.grid_layout.addWidget(self.combobox_fm_filenames_label, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.label_selected_milling, 3, 0)
        self.grid_layout.addWidget(self.comboBox_selected_task, 3, 1)
        self.grid_layout.addWidget(self.label_status, 5, 0, 1, 2)
        self.grid_layout.addWidget(self.label_warning, 6, 0, 1, 2)
        self.grid_layout.addWidget(self.pushButton_refresh_positions, 7, 0, 1, 2)

        # main layout
        self.main_layout = QVBoxLayout(self)
        self.scroll_content_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # required to resize
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)       # type: ignore
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # type: ignore
        self.scroll_content_layout.addLayout(self.grid_layout)
        self.scroll_content_layout.addWidget(self.task_params_collapsible)      # type: ignore
        self.scroll_content_layout.addWidget(self.image_params_collapsible)     # type: ignore
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

        if self.comboBox_selected_lamella.count() > 0:
            self.comboBox_selected_lamella.setCurrentIndex(0)

        if self.parent_widget.experiment.positions:  # type: ignore
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

        # hide if no filenames
        self.combobox_fm_filenames.setVisible(len(filenames) > 0)
        self.combobox_fm_filenames_label.setVisible(len(filenames) > 0)
        self.combobox_fib_filenames.setVisible(len(fib_filenames) > 0)
        self.combobox_fib_filenames_label.setVisible(len(fib_filenames) > 0)

        self._on_image_selected(0)

    def _on_image_selected(self, index):
        """Callback when an image is selected."""
        p: Lamella = self.comboBox_selected_lamella.currentData()
        fib_filename = self.combobox_fib_filenames.currentText()

        # load the fib reference image
        reference_image_path = os.path.join(p.path, fib_filename)
        if os.path.exists(reference_image_path) and os.path.isfile(reference_image_path):
            self.image = FibsemImage.load(reference_image_path)
        else:
            self.image = FibsemImage.generate_blank_image(hfw=150e-6, random=True)

        self.viewer.layers.clear()
        self.viewer.add_image(self.image.data,
                              name="Reference Image (FIB)",
                              colormap='gray',
                              blending="additive")

        self.milling_task_editor.config_widget.milling_editor_widget.set_image(self.image)
        self._on_selected_task_changed()
        self.viewer.reset_view()

    def _on_selected_task_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.comboBox_selected_task.currentText()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()

        task_config = selected_lamella.task_config[selected_stage_name]
        self.task_parameters_config_widget.set_task_config(task_config)
        self.milling_task_editor.set_task_configs(task_config.milling)
        self.ref_image_params_widget.update_from_settings(task_config.reference_imaging)


        # background_milling_stages = []

        # rough_milling_config = None
        # polishing_config = None
        # setup_milling_config = None

        # setup_config = selected_lamella.task_config.get("Setup Lamella", None)
        # if setup_config is not None and setup_config.milling:
        #     setup_milling_config = setup_config.milling.get("fiducial", None)
        # polishing = selected_lamella.task_config.get("Polishing", None)
        # if polishing is not None and polishing.milling:
        #     polishing_config = polishing.milling.get("mill_polishing", None)
        # rough_milling = selected_lamella.task_config.get("Rough Milling", None)
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
        self.milling_task_editor.setEnabled(bool(task_config.milling))

        # display label showing task has been completed
        msg = ""
        if selected_stage_name in [t.name for t in selected_lamella.task_history]:
            msg = f"Task '{selected_stage_name}' has been completed."
        self.label_status.setText(msg)

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
