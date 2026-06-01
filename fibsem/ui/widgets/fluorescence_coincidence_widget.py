import logging
import threading
from pprint import pformat
from typing import TYPE_CHECKING, List, Optional, Union

import napari
from napari.layers import Shapes as NapariShapesLayer
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingSettings, FibsemMillingStage
from fibsem.milling.patterning import RectanglePattern
from fibsem.milling.strategy import CoincidenceMillingStrategy
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    CrossSectionPattern,
    FibsemRectangle,
)
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets import LinePlotWidget
from fibsem.ui.napari.patterns import (
    convert_reduced_area_to_napari_shape,
    convert_shape_to_image_area,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

# TODO:  more automated workflow

milling_stage = FibsemMillingStage(name="Coincidence Milling Stage",
                                  milling=FibsemMillingSettings(hfw=100e-6, milling_current=0.1e-9),
                                  pattern=RectanglePattern(depth=2.0e-6,
                                                        width=5e-6,
                                                        height=8e-6,
                                                        scan_direction="BottomToTop",
                                                        cross_section=CrossSectionPattern.CleaningCrossSection),
                                  strategy=CoincidenceMillingStrategy())
milling_task_config = FibsemMillingTaskConfig(name="Coincidence Milling Task", 
                                              field_of_view=100e-6, 
                                              stages=[milling_stage])
milling_task_config.alignment.enabled = False
milling_task_config.acquisition.acquire_fib = False
milling_task_config.acquisition.acquire_sem = False
milling_task_config.stages[0].strategy.config.save_rate_limit = 0.0

BOUNDING_BOX_LAYER_CONFIG = {
    "name": "bbox",
    "shape_type": "rectangle",
    "edge_color": "white",
    "edge_width": 2,
    "face_color": "transparent",
    "opacity": 0.7,
}


class FluorescenceCoincidenceMillingWidget(QWidget):
    """Widget for FM Coincidence Milling with FIB image acquisition, FM image acquisition, and start/stop milling controls."""
    bbox_updated_signal = pyqtSignal(FibsemRectangle)
    update_line_plot_signal = pyqtSignal(float)

    def __init__(self, microscope: FibsemMicroscope,
                 viewer: napari.Viewer,
                 parent: 'AutoLamellaUI'):
        super().__init__(parent)

        self.parent_widget = parent
        self.microscope = microscope
        self.viewer = viewer
        self._lock = threading.RLock()

        if self.microscope.fm is None:
            raise ValueError("FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope.")

        if self.parent_widget.image_widget is None or self.parent_widget.image_widget.ib_image is None:
            raise ValueError("FIB image must be initialized in the parent widget's image_widget.")

        if self.parent_widget.milling_task_config_widget is None or self.parent_widget.milling_task_config_widget.milling_widget is None:
            raise ValueError("Milling widget must be initialized in the parent widget's milling_task_config_widget.")

        self.fm_translation = self.parent_widget.image_widget.ib_image.data.shape[0]  # translation between FIB and FM images in napari
        self.fm_resolution = self.microscope.fm.camera.resolution

        self.alignment_layer: Optional[NapariShapesLayer] = None

        self.line_plot_widget = LinePlotWidget(parent=self,
                                               max_length=5000,
                                               rolling_mean_window=25)
        self.line_plot_dock = self.viewer.window.add_dock_widget(self.line_plot_widget,
                                                                 name="Line Plot",
                                                                 area='left',
                                                                 tabify=True)
        self.line_plot_dock.hide()

        self.initUI()
        self.connect_signals()

    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        header = QLabel("Fluorescence Coincidence Milling")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
 
        # buttons
        self.button_toggle_line_plot = QPushButton("Toggle Line Plot")
        self.button_select_fm_bbox = QPushButton("Select FM Region of Interest")
        self.button_load_coincidence_milling = QPushButton("Load Milling Pattern")
        self.button_run_milling = QPushButton("Start Milling")
        self.button_stop_milling = QPushButton("Stop Milling")
        self.button_pause_milling = QPushButton("Pause Milling")
        self.button_pause_fm = QPushButton("Pause FM")

        # Style buttons
        self.button_select_fm_bbox.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.button_load_coincidence_milling.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.button_run_milling.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.button_stop_milling.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.button_pause_milling.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        self.button_pause_fm.setStyleSheet(stylesheets.PURPLE_PUSHBUTTON_STYLE)
        self.button_pause_fm.setEnabled(False)

        # Create button layout
        button_layout = QGridLayout()
        button_layout.addWidget(self.button_toggle_line_plot, 0, 0)
        button_layout.addWidget(self.button_select_fm_bbox, 1, 0)
        button_layout.addWidget(self.button_load_coincidence_milling, 1, 1)
        button_layout.addWidget(self.button_run_milling, 2, 0)
        button_layout.addWidget(self.button_stop_milling, 2, 1)
        button_layout.addWidget(self.button_pause_milling, 3, 0)
        button_layout.addWidget(self.button_pause_fm, 3, 1)

        layout.addWidget(header)
        # layout.addWidget(self.line_plot_widget)
        layout.addStretch()
        layout.addLayout(button_layout)
        self.setLayout(layout)
 
    def connect_signals(self):
        """Connect internal signals."""
        self.update_line_plot_signal.connect(lambda value: self.line_plot_widget.append_value(value))
        self.button_select_fm_bbox.clicked.connect(lambda: self.set_bounding_box_layer(editable=True))
        self.button_run_milling.clicked.connect(self._run_milling)
        self.button_stop_milling.clicked.connect(self.parent_widget.milling_task_config_widget.milling_widget.stop_milling)
        self.button_pause_milling.clicked.connect(self.parent_widget.milling_task_config_widget.milling_widget.pause_resume_milling)
        self.button_load_coincidence_milling.clicked.connect(self._load_coincidence_milling_task_config)
        self.button_toggle_line_plot.clicked.connect(self._toggle_line_plot)
        self.button_pause_fm.clicked.connect(self._toggle_fm_pause)

    def _toggle_line_plot(self):
        """Toggle the visibility of the line plot dock widget."""

        if self.line_plot_dock.isVisible():
            self.line_plot_dock.hide()
            return

        self.line_plot_dock.show()
        self.line_plot_dock.setFocus()

    def set_active(self, active: bool = True):
        """Set the widget active or inactive."""
        self.setEnabled(active)

        if active:
            self.set_bounding_box_layer(editable=True)
            # reset the line plot
            self.line_plot_widget.reset_chart()
            self.line_plot_dock.show()
            self.line_plot_dock.setFocus()
        else:
            self._reset_fm_pause_button()
            self.line_plot_dock.hide()
            if self.alignment_layer is not None and "bbox" in self.viewer.layers:
                self.viewer.layers.remove(self.alignment_layer)
                self.alignment_layer = None

    def _toggle_fm_pause(self):
        """Toggle pause/resume of FM acquisition via stop/start."""
        if self.microscope.fm is None:
            return
        if self.microscope.fm.is_acquiring:
            self.microscope.fm.stop_acquisition()
            self.button_pause_fm.setText("Resume FM")
            self.button_pause_fm.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        else:
            self.microscope.fm.start_acquisition()
            self.button_pause_fm.setText("Pause FM")
            self.button_pause_fm.setStyleSheet(stylesheets.PURPLE_PUSHBUTTON_STYLE)

    def _reset_fm_pause_button(self):
        """Reset the FM pause button to its default state."""
        self.button_pause_fm.setEnabled(False)
        self.button_pause_fm.setText("Pause FM")
        self.button_pause_fm.setStyleSheet(stylesheets.PURPLE_PUSHBUTTON_STYLE)

    def _load_coincidence_milling_task_config(self):
        # TODO: load this from the protocol instead of hardcoded
        self.parent_widget.milling_task_config_widget.set_config(milling_task_config)

    def _run_milling(self):
        """Run the coincidence milling process after confirmation dialog."""

        if self.parent_widget.experiment is None or len(self.parent_widget.experiment.positions) == 0:
            logging.error("No lamella positions defined in the experiment. Cannot run milling.")
            QMessageBox.critical(self, "Error", "No lamella positions defined in the experiment. Cannot run milling.")
            return
        
        if self.microscope.fm is None:
            logging.error("FluorescenceMicroscope is not initialized. Cannot run milling.")
            QMessageBox.critical(self, "Error", "FluorescenceMicroscope is not initialized. Cannot run milling.")
            return
        
        if (self.parent_widget.milling_task_config_widget is None or
            self.parent_widget.milling_task_config_widget.milling_widget is None or
            self.parent_widget.fm_control_widget is None or
            self.parent_widget.milling_task_config_widget.config_widget.acquisition_widget is None):
            logging.error("Milling task configuration widget is not initialized. Cannot run milling.")
            QMessageBox.critical(self, "Error", "Milling task configuration widget is not initialized. Cannot run milling.")
            return

        selected_channel_settings = self.parent_widget.fm_control_widget.channelSettingsWidget.selected_channel

        if selected_channel_settings is None:
            logging.error("No FM channel selected. Cannot run milling.")
            QMessageBox.critical(self, "Error", "No FM channel selected. Cannot run milling.")
            return

        lamella = self.parent_widget.get_selected_lamella()
        if lamella is None:
            logging.error("No lamella selected. Cannot run milling.")
            QMessageBox.critical(self, "Error", "No lamella selected. Cannot run milling.")
            return
        lamella_path = lamella.path

        if lamella is not None:
            self.parent_widget.milling_task_config_widget.config_widget.acquisition_widget.image_settings_widget.path_edit.setText(str(lamella_path))
            self.parent_widget.milling_task_config_widget.config_widget.acquisition_widget._emit_settings_changed()
        milling_task_config = self.parent_widget.milling_task_config_widget.get_config()                   

        # confirm the milling parameters with the user before starting
        first_stage = milling_task_config.stages[0] if milling_task_config.stages else None
        if first_stage is None or not isinstance(first_stage.strategy, CoincidenceMillingStrategy):
            logging.warning("Invalid milling strategy loaded for coincidence milling.")
            QMessageBox.warning(
                self,
                "Invalid Milling Strategy",
                "Load the correct milling strategy for coincidence milling before starting.",
            )
            return

        ret = self._show_confirmation_dialog(
            selected_lamella=lamella,
            milling_task_config=milling_task_config,
            channel_settings=selected_channel_settings
        )

        if not ret:
            return  # User cancelled
        self.microscope.fm.set_channel(selected_channel_settings)
        self.button_pause_fm.setEnabled(True)
        self.parent_widget.milling_task_config_widget.milling_widget.run_milling()

    def _show_confirmation_dialog(self, selected_lamella: Optional['Lamella'],
                                    milling_task_config: 'FibsemMillingTaskConfig',
                                    channel_settings: ChannelSettings) -> bool:
        """Show confirmation dialog before starting milling."""
        lamella_name = selected_lamella.name if selected_lamella else "N/A"
        lamella_path = str(selected_lamella.path) if selected_lamella else "N/A"
        stage_data = []
        for idx, stage in enumerate(milling_task_config.stages):
            stage_data.append(stage.pretty_name)

        summary_lines = [
            f"Lamella: {lamella_name}",
            f"Lamella Path: {lamella_path}",
            f"Channel: {channel_settings.pretty}",
            f"Field of View: {milling_task_config.field_of_view*1e6:.1f} µm",
        ]
        if stage_data:
            summary_lines.append("Stages: " + ", ".join(stage_data))

        confirm_dialog = QMessageBox(self)
        confirm_dialog.setWindowTitle("Confirm Coincidence Milling")
        confirm_dialog.setIcon(QMessageBox.Question)
        confirm_dialog.setText("Review milling parameters before starting.")
        confirm_dialog.setInformativeText("\n".join(summary_lines))
        confirm_dialog.setDetailedText(pformat(milling_task_config.to_dict(), width=80, compact=True))
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        confirm_dialog.setDefaultButton(QMessageBox.Yes)
        confirm_dialog.setMinimumSize(800, 800)      # optional

        if confirm_dialog.exec_() != QMessageBox.Yes:
            return False
        return True

    def set_bounding_box_layer(self, editable: bool = True):
        """Set the alignment area layer in napari."""
        if self.fm_resolution is None:
            logging.warning("FM resolution is not available. Cannot set alignment area.")
            return

        # add alignment area to napari
        if self.alignment_layer is None or BOUNDING_BOX_LAYER_CONFIG["name"] not in self.viewer.layers:
            reduced_area: FibsemRectangle = FibsemRectangle(0.45, 0.45, 0.1, 0.1) 
            data = convert_reduced_area_to_napari_shape(reduced_area=reduced_area, image_shape=self.fm_resolution)
            self.alignment_layer = self.viewer.add_shapes(data=data, 
                                              name=BOUNDING_BOX_LAYER_CONFIG["name"], 
                                    shape_type=BOUNDING_BOX_LAYER_CONFIG["shape_type"], 
                                    edge_color=BOUNDING_BOX_LAYER_CONFIG["edge_color"], 
                                    edge_width=BOUNDING_BOX_LAYER_CONFIG["edge_width"], 
                                    face_color=BOUNDING_BOX_LAYER_CONFIG["face_color"], 
                                    opacity=BOUNDING_BOX_LAYER_CONFIG["opacity"], 
                                    translate=(self.parent_widget.image_widget.eb_image.data.shape[0], 0)
                                    )
            self.alignment_layer.metadata = {"type": "alignment"}

        if editable:
            self.viewer.layers.selection.active = self.alignment_layer
            self.alignment_layer.mode = "select"
        # TODO: prevent rotation of rectangles?
        self.alignment_layer.events.data.connect(self.update_alignment)  # type: ignore
        self.update_alignment(None)  # Initial update to validate the area

    def update_alignment(self, event):
        """Validate the alignment area, and update the parent ui."""
        if event is None:
            return
        try:
            if not event.action == "changed":
                return  

            reduced_area = self.get_bounding_box()

            if reduced_area is None:
                return

            # check if the area is valid
            is_valid = reduced_area.is_valid_reduced_area
            logging.info(f"Updated alignment area: valid: {is_valid}, area: {reduced_area}")
            if not is_valid:
                return

            self.bbox_updated_signal.emit(reduced_area)

        except Exception as e:
            logging.info(f"Error updating alignment area: {e}")

    def get_bounding_box(self) -> Optional[FibsemRectangle]:
        """Get the bounding box from the alignment layer."""
        with self._lock:

            if self.alignment_layer is None or not isinstance(self.alignment_layer, NapariShapesLayer):
                logging.warning("Alignment layer is not set or not a Shapes layer.")
                return None

            data = self.alignment_layer.data
            if data is None or self.fm_resolution is None:
                return None
            data = data[0]
            return convert_shape_to_image_area(data, self.fm_resolution) # type: ignore

    # # def on_bbox_update(self, bbox: FibsemRectangle):
    # #     """Handle updates to the bounding box."""
    # #     logging.info(f"Bounding Box Updated: {bbox}")

    # def on_intensity_drop_signal(self, ddict: dict):
    #     """Handle intensity drop signal."""
    #     # logging.info('-'*80)
    #     # logging.info(f"Intensity Drop Signal Received: {ddict}")
    #     # logging.info('-'*80)

    #     # self.stop_milling()  # stop milling if intensity drop is detected

    #     self.label_fm_intensity.setText(f"Intensity Drop Detected: Mean Intensity: {ddict['mean_intensity']:.2f}")
    #     self.label_fm_intensity.setStyleSheet("color: orange;")


def create_widget(microscope: FibsemMicroscope,
                  viewer: napari.Viewer,
                  parent: Optional[QWidget] = None) -> FluorescenceCoincidenceMillingWidget:
    """Create the FluorescenceCoincidenceMillingWidget with a demo microscope."""

    widget = FluorescenceCoincidenceMillingWidget(
        microscope=microscope,
        viewer=viewer,
        parent=parent
    )
    return widget


def main():
    """Main function to run the widget standalone."""
    from fibsem.microscopes.simulator import DemoMicroscope
    microscope, settings = utils.setup_session()

    if microscope.fm is None:
        logging.error("FluorescenceMicroscope is not initialized. Cannot create FMCoincidenceMillingWidget.")
        raise RuntimeError("FluorescenceMicroscope is not initialized.")
    

    # Ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True

    if isinstance(microscope, DemoMicroscope):
        microscope.move_to_microscope("FM")
    
    viewer = napari.Viewer()
    widget = create_widget(microscope, viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
    return


if __name__ == "__main__":
    main()
