import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Points as NapariPointLayer
from napari.layers import Shapes as NapariShapesLayer
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, pyqtSignal
from superqt import ensure_main_thread

from fibsem import acquire, constants, utils
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    Point,
)
from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.widgets.custom_widgets import WheelBlocker
from fibsem.ui.napari.patterns import (
    convert_reduced_area_to_napari_shape,
    convert_shape_to_image_area,
)
from fibsem.ui.napari.properties import (
    ALIGNMENT_LAYER_PROPERTIES,
    IMAGE_TEXT_LAYER_PROPERTIES,
    IMAGING_RULER_LAYER_PROPERTIES,
    RULER_LAYER_NAME,
    RULER_LINE_LAYER_NAME,
)
from fibsem.ui.napari.utilities import draw_crosshair_in_napari, draw_scalebar_in_napari, add_points_layer
from fibsem.ui.qtdesigner_files import ImageSettingsWidget as ImageSettingsWidgetUI
from fibsem.ui.widgets.custom_widgets import _create_combobox_control


class FibsemImageSettingsWidget(ImageSettingsWidgetUI.Ui_Form, QtWidgets.QWidget):
    viewer_update_signal = pyqtSignal()             # when the viewer is updated
    acquisition_progress_signal = pyqtSignal(dict)  # TODO: add progress indicator
    alignment_area_updated = pyqtSignal(FibsemRectangle)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        image_settings: ImageSettings,
        parent: QtWidgets.QWidget,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)

        if not hasattr(parent, "viewer") and not isinstance(parent.viewer, napari.Viewer):
            raise ValueError("Parent must have a 'viewer' attribute of type napari.Viewer")

        self.parent = parent
        self.microscope = microscope
        self.viewer = parent.viewer
        self.eb_layer: NapariImageLayer = None
        self.ib_layer: NapariImageLayer = None

        # TODO: migrate to this structure
        self.imaging_layers: Dict[BeamType, NapariImageLayer] = {}
        self.imaging_layers[BeamType.ELECTRON] = None
        self.imaging_layers[BeamType.ION] = None

        # generate initial blank images 
        self.eb_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)
        self.ib_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)

        # overlay layers
        self.ruler_layer: Optional[NapariPointLayer] = None
        self.alignment_layer: Optional[NapariShapesLayer] = None

        self.is_acquiring: bool = False

        self.setup_connections()

        if image_settings is not None:
            self.image_settings = image_settings
            beam_settings = self.microscope.get_beam_settings(BeamType.ELECTRON)
            detector_settings = self.microscope.get_detector_settings(BeamType.ELECTRON)
            self.set_ui_from_settings(image_settings = image_settings, beam_type = BeamType.ELECTRON, 
                                      beam_settings=beam_settings, detector_settings=detector_settings)
        self.update_detector_ui() # TODO: can this be removed?
        # self._update_beam_settings_ui()

        # register initial images
        self.eb_layer = self.viewer.add_image(self.eb_image.data, name = BeamType.ELECTRON.name, blending='additive')
        self.ib_layer = self.viewer.add_image(self.ib_image.data, name = BeamType.ION.name, blending='additive')
        self._on_acquire(self.eb_image)
        self._on_acquire(self.ib_image)

    def setup_connections(self) -> None:
        """Set up the connections for the UI components, signals and initialise the UI"""

        # set ui elements
        for beam in self.microscope.get_available_beams():
            self.selected_beam.addItem(beam.name, beam)
        for res_str, res_data in cfg.STANDARD_RESOLUTIONS_ZIP:
            self.comboBox_image_resolution.addItem(res_str, tuple(res_data))
        self.comboBox_image_resolution.setCurrentText(cfg.DEFAULT_STANDARD_RESOLUTION)
        self.doubleSpinBox_beam_current.setRange(0.1, 10000.0) # TODO: convert to combobox

        # TODO: add comboboxes for beam current and beam voltage
        self.comboBox_beam_current.currentIndexChanged.connect(self._on_beam_current_changed)
        self.comboBox_beam_voltage.currentIndexChanged.connect(self._on_beam_voltage_changed)
        self.comboBox_beam_current.setVisible(False) # TODO: hide until we can test with microscope
        self.comboBox_beam_voltage.setVisible(False) # TODO: hide until we can test with microscope

        # buttons
        self.pushButton_acquire_sem_image.clicked.connect(self.acquire_sem_image)
        self.pushButton_acquire_fib_image.clicked.connect(self.acquire_fib_image)
        self.pushButton_take_all_images.clicked.connect(self.acquire_reference_images)

        # image / beam settings
        # self.selected_beam.currentIndexChanged.connect(self._update_beam_settings_ui) # TODO: CHANGE_BEAM_CURRENT_COMBOBOX
        self.selected_beam.currentIndexChanged.connect(self.update_detector_ui)
        self.checkBox_image_save_image.toggled.connect(self.update_ui_saving_settings)
        self.button_set_beam_settings.clicked.connect(self.apply_beam_settings)
        self.doubleSpinBox_working_distance.valueChanged.connect(self._on_working_distance_changed)
        self.doubleSpinBox_working_distance.setKeyboardTracking(False)
        self.doubleSpinBox_shift_x.valueChanged.connect(self._on_shift_changed)
        self.doubleSpinBox_shift_x.setKeyboardTracking(False)
        self.doubleSpinBox_shift_y.valueChanged.connect(self._on_shift_changed)
        self.doubleSpinBox_shift_y.setKeyboardTracking(False)
        self.doubleSpinBox_stigmation_x.valueChanged.connect(self._on_stigmation_changed)
        self.doubleSpinBox_stigmation_x.setKeyboardTracking(False)
        self.doubleSpinBox_stigmation_y.valueChanged.connect(self._on_stigmation_changed)
        self.doubleSpinBox_stigmation_y.setKeyboardTracking(False)

        self.pushButton_beam_is_on.clicked.connect(self._toggle_beam_on)
        self.pushButton_beam_blanked.clicked.connect(self._toggle_beam_blank)
        self.update_beam_ui_components()

        # detector
        self.detector_contrast_slider.valueChanged.connect(self._update_detector_slider_labels)
        self.detector_brightness_slider.valueChanged.connect(self._update_detector_slider_labels)
        self.detector_brightness_slider.valueChanged.connect(self._on_detector_brightness_changed)
        self.detector_contrast_slider.valueChanged.connect(self._on_detector_contrast_changed)
        self.detector_type_combobox.currentTextChanged.connect(self._on_detector_type_changed)
        self.detector_mode_combobox.currentTextChanged.connect(self._on_detector_mode_changed)

        # util
        self.checkBox_enable_ruler.toggled.connect(self.update_ruler)
        self.scalebar_checkbox.toggled.connect(self.update_ui_tools)
        self.crosshair_checkbox.toggled.connect(self.update_ui_tools)

        # signals
        self.acquisition_progress_signal.connect(self.handle_acquisition_progress_update)
        self.viewer_update_signal.connect(self.update_ui_tools)

        # advanced settings
        self.checkBox_advanced_settings.stateChanged.connect(self.toggle_mode)
        self.toggle_mode()

        # auto functions
        self.pushButton_run_autocontrast.clicked.connect(self.run_autocontrast)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)

        # set ui stylesheets
        self.pushButton_acquire_sem_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_fib_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_take_all_images.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_set_beam_settings.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autocontrast.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autofocus.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        self.IS_TESCAN = self.microscope.manufacturer == "TESCAN"
        if self.IS_TESCAN:
            self.label_stigmation.hide()
            self.doubleSpinBox_stigmation_x.hide()
            self.doubleSpinBox_stigmation_y.hide()
            self.doubleSpinBox_stigmation_x.setEnabled(False)
            self.doubleSpinBox_stigmation_y.setEnabled(False)
            available_presets = self.microscope.get_available_values("presets", beam_type=BeamType.ION)
            self.comboBox_presets.addItems(available_presets)
            self.comboBox_presets.currentTextChanged.connect(self.update_presets)
            self.checkBox_image_line_integration.setVisible(False)
            self.checkBox_image_scan_interlacing.setVisible(False)
            self.checkBox_image_frame_integration.setVisible(False)
            self.checkBox_image_drift_correction.setVisible(False)
            self.spinBox_image_line_integration.setVisible(False)
            self.spinBox_image_scan_interlacing.setVisible(False)
            self.spinBox_image_frame_integration.setVisible(False)
        else:
            self.comboBox_presets.hide()
            self.label_presets.hide()

        # advanced imaging settings
        self.spinBox_image_line_integration.setRange(2, 255)
        self.spinBox_image_frame_integration.setRange(2, 512)
        self.spinBox_image_scan_interlacing.setRange(2, 8)
        self.spinBox_image_line_integration.setEnabled(False)
        self.spinBox_image_frame_integration.setEnabled(False)
        self.spinBox_image_scan_interlacing.setEnabled(False)
        self.checkBox_image_line_integration.toggled.connect(self.spinBox_image_line_integration.setEnabled)
        self.checkBox_image_scan_interlacing.toggled.connect(self.spinBox_image_scan_interlacing.setEnabled)
        self.checkBox_image_frame_integration.toggled.connect(self.spinBox_image_frame_integration.setEnabled)
        self.checkBox_image_frame_integration.toggled.connect(self.checkBox_image_drift_correction.setEnabled)

        # save image with selected lamella control
        show_lamella_controls = hasattr(self.parent, "experiment")
        self.checkBox_save_with_selected_lamella.setVisible(show_lamella_controls)
        self.checkBox_save_with_selected_lamella.toggled.connect(self.update_ui_saving_settings)
        self.checkBox_save_with_selected_lamella.setToolTip("Save images to the path of the currently selected lamella position in the experiment.")
        try:
            self.parent.comboBox_current_lamella.currentIndexChanged.connect(self._on_current_lamella_changed)
        except Exception as e:
            logging.debug(f"Error connecting to lamella selection changes: {e}")

        # install wheel blockers on spinboxes and comboboxes
        for widget in [
            # imaging
            self.comboBox_image_resolution,
            self.comboBox_presets,
            self.doubleSpinBox_image_dwell_time,
            self.doubleSpinBox_image_hfw,
            self.spinBox_image_line_integration,
            self.spinBox_image_scan_interlacing,
            self.spinBox_image_frame_integration,
            # beam
            self.selected_beam,
            self.doubleSpinBox_working_distance,
            self.doubleSpinBox_beam_current,
            self.doubleSpinBox_beam_voltage,
            self.doubleSpinBox_stigmation_x,
            self.doubleSpinBox_stigmation_y,
            self.doubleSpinBox_shift_x,
            self.doubleSpinBox_shift_y,
            self.spinBox_beam_scan_rotation,
            # detector
            self.detector_type_combobox,
            self.detector_mode_combobox,
        ]:
            widget.installEventFilter(WheelBlocker(parent=widget))

        # feature flags
        self.pushButton_show_alignment_area.setVisible(False)
        # self.pushButton_show_alignment_area.clicked.connect(lambda: self.toggle_alignment_area(None))

        self.acquisition_buttons: List[QtWidgets.QPushButton] = [
            self.pushButton_take_all_images,
            self.pushButton_acquire_sem_image,
            self.pushButton_acquire_fib_image,
        ]

########### Live Acquisition
        # live acquisition
        LIVE_ACQUISITION_ENABLED = True
        self.pushButton_start_acquisition.setVisible(LIVE_ACQUISITION_ENABLED)
        self.pushButton_start_acquisition.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.microscope.sem_acquisition_signal.connect(self._on_acquire)
        self.microscope.fib_acquisition_signal.connect(self._on_acquire)
        self.pushButton_start_acquisition.clicked.connect(self.toggle_live_acquisition)
        # TODO: support real-time parameter updates too

        # set suffixes for units
        self.spinBox_beam_scan_rotation.setSuffix(f" {constants.DEGREE_SYMBOL}")
        self.doubleSpinBox_shift_x.setSuffix(f" {constants.MICRON_SYMBOL}")
        self.doubleSpinBox_shift_y.setSuffix(f" {constants.MICRON_SYMBOL}")
        self.doubleSpinBox_image_hfw.setSuffix(f" {constants.MICRON_SYMBOL}")
        self.doubleSpinBox_image_dwell_time.setSuffix(f" {constants.MICROSECOND_SYMBOL}")

    @ensure_main_thread
    def _on_acquire(self, image: FibsemImage):
        """Update the viewer from the main thread"""
        try:
            if image.metadata is None:
                raise ValueError("Image metadata is None, cannot update viewer layer without beam type information.")

            # Update existing layer
            layer = self.viewer.layers[image.metadata.beam_type.name]
            layer.data = image.filtered_data
            # update images references
            if self.microscope.is_acquiring:
                if image.metadata.beam_type is BeamType.ELECTRON:
                    self.eb_image = image
                elif image.metadata.beam_type is BeamType.ION:
                    self.ib_image = image
        except Exception as e:
            logging.error(f"Error updating image layer: {e}")

        # dont reset view when live acq
        self._update_layer_positions()
        self.restore_active_layer_for_movement()
        self.viewer_update_signal.emit()

    def toggle_live_acquisition(self, event=None):
        # Start acquisition logic
        if self.microscope.is_acquiring:
            logging.info("Microscope is already acquiring. Stopping acquisition...")
            self.microscope.stop_acquisition()
            self.pushButton_start_acquisition.setText("Start Acquisition")
            self.pushButton_start_acquisition.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            return

        # Start acquisition in the microscope
        beam_type = self.selected_beam.currentData()
        self.microscope.start_acquisition(beam_type)
        self.pushButton_start_acquisition.setText("Stop Acquisition")
        self.pushButton_start_acquisition.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)

###########

    def toggle_mode(self) -> None:
        """Toggle the visibility of the advanced settings"""
        advanced_mode = self.checkBox_advanced_settings.isChecked()

        self.label_detector_type.setVisible(advanced_mode)
        self.detector_type_combobox.setVisible(advanced_mode)
        self.label_detector_mode.setVisible(advanced_mode)
        self.detector_mode_combobox.setVisible(advanced_mode)
        self.label_stigmation.setVisible(advanced_mode)
        self.doubleSpinBox_stigmation_x.setVisible(advanced_mode)
        self.doubleSpinBox_stigmation_y.setVisible(advanced_mode)
        self.doubleSpinBox_shift_x.setVisible(advanced_mode)
        self.doubleSpinBox_shift_y.setVisible(advanced_mode)
        self.label_shift.setVisible(advanced_mode)
        self.doubleSpinBox_beam_voltage.setVisible(advanced_mode)
        self.label_beam_voltage.setVisible(advanced_mode)
        self.label_beam_scan_rotation.setVisible(advanced_mode)
        self.spinBox_beam_scan_rotation.setVisible(advanced_mode)
        self.checkBox_image_use_autocontrast.setVisible(advanced_mode)
        self.checkBox_image_use_autogamma.setVisible(advanced_mode)

        # advanced imaging
        self.checkBox_image_line_integration.setVisible(advanced_mode)
        self.checkBox_image_scan_interlacing.setVisible(advanced_mode)
        self.checkBox_image_frame_integration.setVisible(advanced_mode)
        self.checkBox_image_drift_correction.setVisible(advanced_mode)
        self.spinBox_image_line_integration.setVisible(advanced_mode)
        self.spinBox_image_scan_interlacing.setVisible(advanced_mode)
        self.spinBox_image_frame_integration.setVisible(advanced_mode)

    def update_presets(self) -> None:
        beam_type = self.selected_beam.currentData()
        self.microscope.set("preset", self.comboBox_presets.currentText(), beam_type)
    
    def update_ruler(self):
        """Initialise the ruler in the viewer"""

        # TODO: migrate to using a line layer for everything, and enable 'select vertices' mode to move it.
        # TODO: allow multiple rulers
        # TODO: re-do this and consolidate into napari.utilities

        if not self.checkBox_enable_ruler.isChecked():
            self.label_ruler_value.setText("Ruler: is off")
            self.label_ruler_value.setVisible(False)
            # remove the ruler layers
            try:
                self.viewer.layers.remove(self.ruler_layer)
                self.viewer.layers.remove(RULER_LINE_LAYER_NAME)
            except Exception as e:
                logging.debug(f"Error removing ruler layers: {e}")
            self.ruler_layer = None
            self.restore_active_layer_for_movement()
            return
        
        # enable the ruler
        self.label_ruler_value.setText("Ruler: is on")
        self.label_ruler_value.setVisible(True)

        # create an initial ruler in SEM image
        sem_shape = self.eb_image.data.shape
        pixelsize = self.eb_image.metadata.pixel_size.x
        cy, cx = sem_shape[0] // 2, sem_shape[1] // 2
        length = 400
        data = [[cy, cx - length/2], [cy, cx + length]]
        
        # create text label
        dist_um = length * pixelsize * constants.SI_TO_MICRO
        text = {
            "string": [f"{dist_um:.2f} um", ""],
            "color": IMAGING_RULER_LAYER_PROPERTIES["text"]["color"],
            "translation": IMAGING_RULER_LAYER_PROPERTIES["text"]["translation"],
            "size": IMAGING_RULER_LAYER_PROPERTIES["text"]["size"],
            }

        # create initial layers, and select the ruler layer for interaction 
        self.ruler_layer = add_points_layer(self.viewer, data=data, 
                                                    name=RULER_LAYER_NAME,
                                                    size=IMAGING_RULER_LAYER_PROPERTIES["size"], 
                                                    face_color=IMAGING_RULER_LAYER_PROPERTIES["face_color"], 
                                                    border_color=IMAGING_RULER_LAYER_PROPERTIES["edge_color"], 
                                                    text=text)
        self.viewer.add_shapes(data, shape_type='line', edge_color='lime', name=RULER_LINE_LAYER_NAME,edge_width=5)
        self.ruler_layer.mouse_drag_callbacks.append(self.update_ruler_points)
        self.ruler_layer.mode = 'select'
        self.viewer.layers.selection.active = self.ruler_layer

    def update_ruler_points(self, layer: NapariPointLayer, event):
        """Update the ruler line and value when the mouse is dragged"""

        # TODO: update this to use the new data changed api in napari
        # self.ruler_layer.events.data.connect(self._update_ruler)
        # event.action == "changing", "changed", "added", "removed"...

        dragged = False
        yield

        def check_point_image_in_eb(point: Tuple[int,int]) -> bool:
            if point[1] >= 0 and point[1] <= self.eb_layer.data.shape[1]:
                return True
            else:
                return False

        while event.type == 'mouse_move':

            # if no points are selected, do nothing
            if self.ruler_layer.selected_data is None:
                yield

            data = self.ruler_layer.data

            p1, p2 = data[0], data[1]
            dist_px = np.linalg.norm(p1-p2)
            dx_px = abs(p2[1]-p1[1])
            dy_px = abs(p2[0]-p1[0])
            
            if check_point_image_in_eb(p1):
                pixelsize = self.eb_image.metadata.pixel_size.x
            else: 
                pixelsize = self.ib_image.metadata.pixel_size.x

            dist_um = dist_px * pixelsize * constants.SI_TO_MICRO

            # update the text labels
            dx_um = dx_px * pixelsize * constants.SI_TO_MICRO
            dy_um = dy_px * pixelsize * constants.SI_TO_MICRO
            self.label_ruler_value.setText(f"Ruler: {dist_um:.2f} um  dx: {dx_um:.2f} um  dy: {dy_um:.2f} um")

            text = {
                "string": [f"{dist_um:.2f} um", ""],
                "color": IMAGING_RULER_LAYER_PROPERTIES["text"]["color"],
                "translation": IMAGING_RULER_LAYER_PROPERTIES["text"]["translation"],
                "size": IMAGING_RULER_LAYER_PROPERTIES["text"]["size"],
                }

            self.viewer.layers[RULER_LAYER_NAME].text = text
            self.viewer.layers.selection.active = self.ruler_layer
            self.viewer.layers[RULER_LINE_LAYER_NAME].data = [p1, p2]
            self.viewer.layers[RULER_LINE_LAYER_NAME].refresh()
            
            dragged = True
            yield

    def _update_detector_slider_labels(self) -> None:
        """"Update the labels for the detector contrast and brightness"""
        self.detector_contrast_label.setText(f"{self.detector_contrast_slider.value()}%")
        self.detector_brightness_label.setText(f"{self.detector_brightness_slider.value()}%")

    def _toggle_beam_on(self) -> None:
        """Toggle the beam on/off"""
        beam_type = self.selected_beam.currentData()
        if self.microscope.is_on(beam_type):
            self.microscope.turn_off(beam_type)
        else:
            self.microscope.turn_on(beam_type)

        self.update_beam_ui_components()

    def _toggle_beam_blank(self) -> None:
        """Toggle the beam blanking"""
        beam_type = self.selected_beam.currentData()
        if self.microscope.is_blanked(beam_type):
            self.microscope.unblank(beam_type)
        else:
            self.microscope.blank(beam_type)

        self.update_beam_ui_components()

    def _on_beam_current_changed(self, index: int) -> None:
        """Update the microscope beam current when the combobox value changes"""
        beam_type = self.selected_beam.currentData()
        current = self.comboBox_beam_current.itemData(index)
        self.microscope.set_beam_current(current, beam_type)

    def _on_beam_voltage_changed(self, index: int) -> None:
        """Update the microscope beam voltage when the combobox value changes"""
        beam_type = self.selected_beam.currentData()
        voltage = self.comboBox_beam_voltage.itemData(index)
        self.microscope.set_beam_voltage(voltage, beam_type)

    def _on_working_distance_changed(self, value: float) -> None:
        """Update the microscope working distance when the spinbox value changes."""
        beam_type = self.selected_beam.currentData()
        wd = value * constants.MILLI_TO_SI
        self.microscope.set_working_distance(wd, beam_type) # TODO: check the limits before trying to set
        logging.info({"msg": "_on_working_distance_changed", "beam_type": beam_type.name, "working_distance": wd})

    def _on_shift_changed(self) -> None:
        """Update the microscope beam shift when either shift spinbox value changes."""
        beam_type = self.selected_beam.currentData()
        shift = Point(
            x=self.doubleSpinBox_shift_x.value() * constants.MICRO_TO_SI,
            y=self.doubleSpinBox_shift_y.value() * constants.MICRO_TO_SI,
        )
        self.microscope.set_beam_shift(shift, beam_type)
        logging.info({"msg": "_on_shift_changed", "beam_type": beam_type.name, "shift": shift})

    def _on_stigmation_changed(self) -> None:
        """Update the microscope stigmation when either stigmation spinbox value changes."""
        beam_type = self.selected_beam.currentData()
        stigmation = Point(
            x=self.doubleSpinBox_stigmation_x.value(),
            y=self.doubleSpinBox_stigmation_y.value(),
        )
        self.microscope.set_stigmation(stigmation, beam_type)
        logging.info({"msg": "_on_stigmation_changed", "beam_type": beam_type.name, "stigmation": stigmation})

    def _on_detector_brightness_changed(self, value: int) -> None:
        """Update the microscope detector brightness when the slider value changes"""
        beam_type = self.selected_beam.currentData()
        brightness = value * constants.FROM_PERCENTAGES # percentage int to float (0-1)
        self.microscope.set_detector_brightness(brightness, beam_type)

    def _on_detector_contrast_changed(self, value: int) -> None:
        """Update the microscope detector contrast when the slider value changes"""
        beam_type = self.selected_beam.currentData()
        contrast = value * constants.FROM_PERCENTAGES # percentage int to float (0-1)
        self.microscope.set_detector_contrast(contrast, beam_type)

    def _on_detector_type_changed(self, detector_type: str) -> None:
        """Update the microscope detector type when the combobox value changes"""
        if not detector_type:
            return
        beam_type = self.selected_beam.currentData()
        self.microscope.set_detector_type(detector_type, beam_type)

    def _on_detector_mode_changed(self, mode: str) -> None:
        """Update the microscope detector mode when the combobox value changes"""
        if not mode or mode == "N/A":
            return
        beam_type = self.selected_beam.currentData()
        self.microscope.set_detector_mode(mode, beam_type)

    def run_autocontrast(self) -> None:
        """Run autocontrast for the selected beam type."""
        beam_type = self.selected_beam.currentData()
        self._toggle_interactions(enable=False)
        worker = self._autocontrast_worker(beam_type)
        worker.finished.connect(lambda: self._on_auto_function_finished("AutoContrast", beam_type=beam_type))
        worker.start()

    def run_autofocus(self) -> None:
        """Run autofocus for the selected beam type."""
        beam_type = self.selected_beam.currentData()
        self._toggle_interactions(enable=False)
        worker = self._autofocus_worker(beam_type)
        worker.finished.connect(lambda: self._on_auto_function_finished("AutoFocus", beam_type=beam_type))
        worker.start()

    @thread_worker
    def _autocontrast_worker(self, beam_type: BeamType):
        self.microscope.autocontrast(beam_type, reduced_area=FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5))

    @thread_worker
    def _autofocus_worker(self, beam_type: BeamType):
        self.microscope.auto_focus(beam_type, reduced_area=FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5))

    def _on_auto_function_finished(self, name: str, beam_type: BeamType) -> None:
        self._toggle_interactions(enable=True)
        # update focus / brightness / contrast values for gui when this finishes
        if name == "AutoFocus":
            wd = self.microscope.get_working_distance(beam_type=beam_type)
            self.doubleSpinBox_working_distance.blockSignals(True)
            self.doubleSpinBox_working_distance.setValue(wd * constants.METRE_TO_MILLIMETRE)
            self.doubleSpinBox_working_distance.blockSignals(False)
            napari.utils.notifications.show_info(f"{name} Complete. Best WD: {wd*constants.METRE_TO_MILLIMETRE:.2f}mm")
        if name == "AutoContrast":
            brightness = self.microscope.get_detector_brightness(beam_type=beam_type)
            contrast = self.microscope.get_detector_contrast(beam_type=beam_type)
            self.detector_contrast_slider.blockSignals(True)
            self.detector_brightness_slider.blockSignals(True)
            self.detector_contrast_slider.setValue(int(contrast * 100))
            self.detector_brightness_slider.setValue(int(brightness * 100))
            self._update_detector_slider_labels()
            self.detector_contrast_slider.blockSignals(False)
            self.detector_brightness_slider.blockSignals(False)
            napari.utils.notifications.show_info(f"{name} Complete. Brightness: {brightness}, Contrast: {contrast}")

    def update_beam_ui_components(self):
        """Update beam ui (on/off and blanked/unblanked)"""
        beam_type = self.selected_beam.currentData()
        beam_is_on = self.microscope.is_on(beam_type)
        beam_is_blanked = self.microscope.is_blanked(beam_type)

        self.label_beam_status.setVisible(False) # disabled until testing

        if beam_is_on:
            self.pushButton_beam_is_on.setText("Beam is ON")
            self.pushButton_beam_is_on.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        else:
            self.pushButton_beam_is_on.setText("Beam is OFF")
            self.pushButton_beam_is_on.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
    
        if beam_is_blanked:
            self.pushButton_beam_blanked.setText("BLANKED")
            self.pushButton_beam_blanked.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        else:
            self.pushButton_beam_blanked.setText("UNBLANKED")
            self.pushButton_beam_blanked.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)

    def apply_beam_settings(self) -> None:
        """Apply the beam settings from the UI to the microscope."""
        beam = self.selected_beam.currentData()
        self.beam_settings = self._get_beam_settings_from_ui()

        # set beam settings
        self.microscope.set_beam_settings(self.beam_settings)

        # # TODO: remove this, once we can confirm beams settings are applied correctly on each element
        self._set_beam_settings_to_ui(self.microscope.get_beam_settings(beam))

        # logging
        logging.debug({"msg": "apply_beam_settings", "beam_settings": self.beam_settings.to_dict()})

        # notifications
        napari.utils.notifications.show_info("Beam Settings Updated")

    def _get_image_settings_from_ui(self) -> ImageSettings:
        """Get the image settings from the UI and return an ImageSettings object."""
        # advanced imaging settings
        line_integration, scan_interlacing, frame_integration = None, None, None
        image_drift_correction = False

        if self.checkBox_image_line_integration.isChecked():
            line_integration = self.spinBox_image_line_integration.value()
        if self.checkBox_image_scan_interlacing.isChecked():
            scan_interlacing = self.spinBox_image_scan_interlacing.value()
        if self.checkBox_image_frame_integration.isChecked():
            frame_integration = self.spinBox_image_frame_integration.value()
            image_drift_correction = self.checkBox_image_drift_correction.isChecked()

        # imaging settings
        self.image_settings = ImageSettings(
            resolution=self.comboBox_image_resolution.currentData(), # (width, height) in pixels
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=self.selected_beam.currentData(),
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            autogamma=self.checkBox_image_use_autogamma.isChecked(),
            save=self.checkBox_image_save_image.isChecked(),
            path=Path(self.lineEdit_image_path.text()),
            filename=self.lineEdit_image_label.text(),
            line_integration=line_integration,
            scan_interlacing=scan_interlacing,
            frame_integration=frame_integration,
            drift_correction=image_drift_correction,
        )

        return self.image_settings

    def _get_beam_settings_from_ui(self) -> BeamSettings:
        """Get the beam settings from the UI and return a BeamSettings object"""
        # beam settings
        self.beam_settings = BeamSettings(
            beam_type=self.selected_beam.currentData(),
            working_distance=self.doubleSpinBox_working_distance.value()*constants.MILLI_TO_SI,
            beam_current=self.doubleSpinBox_beam_current.value()*constants.PICO_TO_SI,
            voltage=self.doubleSpinBox_beam_voltage.value()*constants.KILO_TO_SI,
            hfw = self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            resolution=self.comboBox_image_resolution.currentData(), # (width, height) in pixels,
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            stigmation = Point(x = self.doubleSpinBox_stigmation_x.value(), 
                               y = self.doubleSpinBox_stigmation_y.value()),
            shift = Point(self.doubleSpinBox_shift_x.value() * constants.MICRO_TO_SI, 
                          self.doubleSpinBox_shift_y.value() * constants.MICRO_TO_SI),
            scan_rotation = np.deg2rad(self.spinBox_beam_scan_rotation.value())
        )

        return self.beam_settings

    def _get_detector_settings_from_ui(self) -> FibsemDetectorSettings:
        """Get the detector settings from the UI and return a FibsemDetectorSettings object"""
        # detector settings
        self.detector_settings = FibsemDetectorSettings(
            type=self.detector_type_combobox.currentText(),
            mode=self.detector_mode_combobox.currentText(),
            brightness=self.detector_brightness_slider.value()*constants.FROM_PERCENTAGES,
            contrast=self.detector_contrast_slider.value()*constants.FROM_PERCENTAGES,
        )
        return self.detector_settings

    def set_ui_from_settings(self,
                             image_settings: ImageSettings,
                             beam_type: BeamType,
                             beam_settings: Optional[BeamSettings] = None,
                             detector_settings: Optional[FibsemDetectorSettings] = None) -> None:
        """Update the ui from the image, beam and detector settings"""

        # disconnect beam type combobox
        self.selected_beam.blockSignals(True)
        self.selected_beam.setCurrentText(beam_type.name)
        self.selected_beam.blockSignals(False)

        # apply settings to ui components
        self._set_image_settings_to_ui(image_settings)

        if detector_settings is not None:
            self._set_detector_settings_to_ui(detector_settings)

        if beam_settings is not None:
            self._set_beam_settings_to_ui(beam_settings)

        self.update_ui_saving_settings()

    def _set_image_settings_to_ui(self, image_settings: ImageSettings) -> None:
        """Set the image settings to the UI components."""
        # imaging settings
        res = image_settings.resolution
        res_str = f"{res[0]}x{res[1]}"
        idx = self.comboBox_image_resolution.findText(res_str)
        if idx != -1:
            self.comboBox_image_resolution.setCurrentIndex(idx)
        else:
            self.comboBox_image_resolution.setCurrentText(cfg.DEFAULT_STANDARD_RESOLUTION)

        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)

        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.autogamma)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(str(image_settings.path))
        if self.lineEdit_image_label.text() == "":
            self.lineEdit_image_label.setText(image_settings.filename)

        if image_settings.line_integration is not None:
            self.checkBox_image_line_integration.setChecked(True)
            self.spinBox_image_line_integration.setValue(image_settings.line_integration)
        if image_settings.scan_interlacing is not None:
            self.checkBox_image_scan_interlacing.setChecked(True)
            self.spinBox_image_scan_interlacing.setValue(image_settings.scan_interlacing)
        if image_settings.frame_integration is not None:
            self.checkBox_image_frame_integration.setChecked(True)
            self.spinBox_image_frame_integration.setValue(image_settings.frame_integration)
            self.checkBox_image_drift_correction.setChecked(image_settings.drift_correction)

    def _set_detector_settings_to_ui(self, detector_settings: FibsemDetectorSettings) -> None:
        """Set the detector settings to the UI components."""
        self.detector_type_combobox.blockSignals(True)
        self.detector_mode_combobox.blockSignals(True)
        self.detector_type_combobox.setCurrentText(detector_settings.type)
        self.detector_mode_combobox.setCurrentText(detector_settings.mode)
        self.detector_type_combobox.blockSignals(False)
        self.detector_mode_combobox.blockSignals(False)
        self.detector_brightness_slider.blockSignals(True)
        self.detector_contrast_slider.blockSignals(True)
        self.detector_contrast_slider.setValue(int(detector_settings.contrast * 100))
        self.detector_brightness_slider.setValue(int(detector_settings.brightness * 100))
        self._update_detector_slider_labels()
        self.detector_brightness_slider.blockSignals(False)
        self.detector_contrast_slider.blockSignals(False)

    def _set_beam_settings_to_ui(self, beam_settings: BeamSettings):
        """Set the beam settings to the ui components"""
        # beam settings
        if beam_settings.beam_current is not None:
            self.doubleSpinBox_beam_current.setValue(beam_settings.beam_current * constants.SI_TO_PICO)
        if beam_settings.voltage is not None:
            self.doubleSpinBox_beam_voltage.setValue(beam_settings.voltage * constants.SI_TO_KILO)
        if beam_settings.scan_rotation is not None:
            self.spinBox_beam_scan_rotation.blockSignals(True)
            self.spinBox_beam_scan_rotation.setValue(int(np.degrees(beam_settings.scan_rotation)))
            self.spinBox_beam_scan_rotation.blockSignals(False)

        if beam_settings.working_distance is not None:
            self.doubleSpinBox_working_distance.blockSignals(True)
            self.doubleSpinBox_working_distance.setValue(beam_settings.working_distance * constants.METRE_TO_MILLIMETRE)
            self.doubleSpinBox_working_distance.blockSignals(False)
        if beam_settings.shift is not None:
            self.doubleSpinBox_shift_x.blockSignals(True)
            self.doubleSpinBox_shift_y.blockSignals(True)
            self.doubleSpinBox_shift_x.setValue(beam_settings.shift.x * constants.SI_TO_MICRO)
            self.doubleSpinBox_shift_y.setValue(beam_settings.shift.y * constants.SI_TO_MICRO)
            self.doubleSpinBox_shift_x.blockSignals(False)
            self.doubleSpinBox_shift_y.blockSignals(False)
        if beam_settings.stigmation is not None:
            self.doubleSpinBox_stigmation_x.blockSignals(True)
            self.doubleSpinBox_stigmation_y.blockSignals(True)
            self.doubleSpinBox_stigmation_x.setValue(beam_settings.stigmation.x)
            self.doubleSpinBox_stigmation_y.setValue(beam_settings.stigmation.y)
            self.doubleSpinBox_stigmation_x.blockSignals(False)
            self.doubleSpinBox_stigmation_y.blockSignals(False)

    def update_ui_saving_settings(self) -> None:
        """Toggle the visibility of the imaging saving settings"""
        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.checkBox_save_with_selected_lamella.setEnabled(self.checkBox_image_save_image.isChecked())

        # disable lineedit path if saving with selected lamella
        save_with_selected_lamella = self.checkBox_save_with_selected_lamella.isChecked()
        self.lineEdit_image_path.setEnabled(not save_with_selected_lamella)
        if save_with_selected_lamella:
            try:
                idx = self.parent.comboBox_current_lamella.currentIndex()
                lamella = self.parent.experiment.positions[idx]
                self.lineEdit_image_path.setText(str(lamella.path))
            except Exception as e:
                logging.debug(f"Error setting image path from selected lamella: {e}")
                # add callback to update when lamella selection changes
        else:
            if hasattr(self.parent, "experiment") and self.parent.experiment is not None:
                self.lineEdit_image_path.setText(str(self.parent.experiment.path))

    def _on_current_lamella_changed(self, index: int):
        """Update the image path when the selected lamella changes"""
        try:
            if self.checkBox_save_with_selected_lamella.isChecked():
                lamella = self.parent.experiment.positions[index]
                self.lineEdit_image_path.setText(str(lamella.path))
                self.checkBox_save_with_selected_lamella.setText(F"Save with Selected Lamella ({lamella.name})")
        except Exception as e:
            logging.debug(f"Error updating image path from selected lamella: {e}")

    def update_detector_ui(self):
        """Update the detector ui based on currently selected beam"""
        beam_type = self.selected_beam.currentData()

        is_fib = bool(beam_type is BeamType.ION)
        self.comboBox_presets.setVisible(is_fib and self.IS_TESCAN)
        self.label_presets.setVisible(is_fib and self.IS_TESCAN)

        self.detector_type_combobox.blockSignals(True)
        available_detector_types = self.microscope.get_available_values("detector_type", beam_type=beam_type)
        self.detector_type_combobox.clear()
        self.detector_type_combobox.addItems(available_detector_types)
        self.detector_type_combobox.setCurrentText(self.microscope.get_detector_type(beam_type=beam_type))
        self.detector_type_combobox.blockSignals(False)

        self.detector_mode_combobox.blockSignals(True)
        available_detector_modes = self.microscope.get_available_values("detector_mode", beam_type=beam_type)
        if available_detector_modes is None:
            available_detector_modes = ["N/A"]
        self.detector_mode_combobox.clear()
        self.detector_mode_combobox.addItems(available_detector_modes)
        self.detector_mode_combobox.setCurrentText(self.microscope.get_detector_mode(beam_type=beam_type))
        self.detector_mode_combobox.blockSignals(False)

        # get current settings and update ui
        beam_settings = self.microscope.get_beam_settings(beam_type=beam_type)
        detector_settings = self.microscope.get_detector_settings(beam_type=beam_type)
        self.set_ui_from_settings(image_settings = self.image_settings, beam_type = beam_type, 
                                    beam_settings=beam_settings, detector_settings=detector_settings)

    def _update_beam_settings_ui(self) -> None:
        """Update the beam settings ui components based on the current microscope state"""

        return # TODO: hide until we can test with microscope

        beam_type = self.selected_beam.currentData()

        # set current controls
        self.comboBox_beam_current.blockSignals(True)
        self.comboBox_beam_current.clear()
        print(self.microscope.get_beam_current(beam_type))
        _create_combobox_control(value=self.microscope.get_beam_current(beam_type),   
                        items=self.microscope.get_available_values_cached("current", beam_type),
                        units="A",
                        format_fn=utils.format_value,
                        control= self.comboBox_beam_current)
        self.comboBox_beam_current.blockSignals(False)

        # set voltage controls
        self.comboBox_beam_voltage.blockSignals(True)
        _create_combobox_control(value=self.microscope.get_beam_voltage(beam_type),
                        items=self.microscope.get_available_values_cached("voltage", beam_type),
                        units="V",
                        format_fn=utils.format_value,
                        control=self.comboBox_beam_voltage)
        self.comboBox_beam_voltage.blockSignals(False)

    def _toggle_interactions(self, enable: bool, caller: str = None, imaging: bool = False):
        for btn in self.acquisition_buttons:
            btn.setEnabled(enable)
        for btn in [self.pushButton_run_autocontrast, self.pushButton_run_autofocus]:
            btn.setEnabled(enable)
        self.button_set_beam_settings.setEnabled(enable)
        self.parent.movement_widget._toggle_interactions(enable, caller="ui")
        # if caller != "milling":
            # self.parent.milling_widget._toggle_interactions(enable, caller="ui")
        if enable:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            self.pushButton_take_all_images.setText("Acquire All Images")
            self.pushButton_acquire_sem_image.setText("Acquire SEM Image")
            self.pushButton_acquire_fib_image.setText("Acquire FIB Image")
        elif imaging:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
                btn.setText("Acquiring...")
        else:
            self.pushButton_take_all_images.setText("Acquire All Images")

    def handle_acquisition_progress_update(self, ddict: dict):
        """Handle the acquisition progress update"""
        logging.debug(f"Acquisition Progress Update: {ddict}")

        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            napari.utils.notifications.notification_manager.records.clear()
            napari.utils.notifications.show_info(msg)

        # TODO: implement progress bar for acquisition

    def acquisition_finished(self) -> None:
        """Imaging has finished, update the viewer and re-enable interactions"""
        if self.ib_image is not None:
            self._on_acquire(self.ib_image)
        if self.eb_image is not None:
            self._on_acquire(self.eb_image)
        self._toggle_interactions(True)
        self.is_acquiring = False

    def acquire_sem_image(self) -> None:
        """Acquire an SEM image with the current settings"""
        self.start_acquisition(both=False, beam_type=BeamType.ELECTRON)
    
    def acquire_fib_image(self) -> None:
        """Acquire an FIB image with the current settings"""
        self.start_acquisition(both=False, beam_type=BeamType.ION)

    def acquire_reference_images(self) -> None:
        """Acquire both SEM and FIB images with the current settings."""
        self.start_acquisition(both=True)

    def start_acquisition(self, both: bool = False, beam_type: Optional[BeamType] = None) -> None:
        """Start the image acquisition process"""
        # disable interactions
        self.is_acquiring = True
        self._toggle_interactions(enable=False, imaging=True)
        
        # get imaging settings from ui
        self.image_settings = self._get_image_settings_from_ui()
        if beam_type is not None:
            self.image_settings.beam_type = beam_type

        try:
            filename = self.image_settings.filename
            save_selected_lamella = self.checkBox_save_with_selected_lamella.isChecked()
            if save_selected_lamella:
                self.image_settings.filename = f"ref_{filename}"
        except Exception as e:
            logging.error(f"Error getting selected lamella for image saving: {e}")

        ts = utils.current_timestamp_v3()
        self.image_settings.filename = f"{self.image_settings.filename}-{ts}"

        # start the acquisition worker
        worker = self.acquisition_worker(self.image_settings, both=both)
        worker.finished.connect(self.acquisition_finished)
        worker.start()

    @thread_worker
    def acquisition_worker(self, image_settings: ImageSettings, both: bool = False):
        """Threaded image acquisition worker"""

        msg = "Acquiring Both Images..." if both else "Acquiring Image..."
        self.acquisition_progress_signal.emit({"msg": msg})

        if both:
            self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, image_settings)
        else:
            image = acquire.acquire_image(self.microscope, image_settings)
            if image_settings.beam_type is BeamType.ELECTRON:
                self.eb_image = image
            if image_settings.beam_type is BeamType.ION:
                self.ib_image = image

        self.acquisition_progress_signal.emit({"finished": True})

        logging.debug({"msg": "acquisition_worker", "image_settings": self.image_settings.to_dict()})

    def update_ui_tools(self):
        """Redraw the ui tools (scalebar, crosshair)"""

        # draw scalebar and crosshair
        if self.eb_image is not None and self.ib_image is not None:
            draw_scalebar_in_napari(
                viewer=self.viewer,
                sem_shape=self.eb_image.data.shape,
                fib_shape=self.ib_image.data.shape,
                sem_fov=self.eb_image.metadata.image_settings.hfw,
                fib_fov=self.ib_image.metadata.image_settings.hfw,
                is_checked=self.scalebar_checkbox.isChecked(),
            )
            fm_shape = None
            if self.microscope.fm is not None:
                fm_shape = self.microscope.fm.camera.resolution[::-1]

            draw_crosshair_in_napari(
                viewer=self.viewer,
                sem_shape=self.eb_image.data.shape,
                fib_shape=self.ib_image.data.shape,
                fm_shape=fm_shape,
                is_checked=self.crosshair_checkbox.isChecked(),
            )

        # restore active layer for movement
        self.restore_active_layer_for_movement()

    def _update_layer_positions(self):
        """Update the positions of the image layers in the viewer. Ion beam to the right of electron beam."""
        # translate ion beam layer to the right of electron beam, adjust the camera
        if self.eb_layer and self.ib_layer:
            self.ib_layer.translate = [0.0, self.eb_layer.data.shape[1]]

            # position the image text layer
            points = np.array([[-20, 200], [-20, self.ib_layer.translate[1] + 150]])

            try:
                self.viewer.layers[IMAGE_TEXT_LAYER_PROPERTIES["name"]].data = points
            except KeyError:
                add_points_layer(
                    viewer=self.viewer,
                    data=points,
                    name=IMAGE_TEXT_LAYER_PROPERTIES["name"],
                    size=IMAGE_TEXT_LAYER_PROPERTIES["size"],
                    text=IMAGE_TEXT_LAYER_PROPERTIES["text"],
                    border_width=IMAGE_TEXT_LAYER_PROPERTIES["border_width"],
                    border_width_is_relative=IMAGE_TEXT_LAYER_PROPERTIES[
                        "border_width_is_relative"
                    ],
                    border_color=IMAGE_TEXT_LAYER_PROPERTIES["border_color"],
                    face_color=IMAGE_TEXT_LAYER_PROPERTIES["face_color"],
                )
                self.viewer.reset_view()

    def get_data_from_coord(self, coords: Tuple[float, float]) -> Tuple[Tuple[float, float], BeamType, FibsemImage]:
        
        # TODO: change this to use the image layers, and extent

        # check inside image dimensions, (y, x)
        eb_shape = self.eb_image.data.shape[0], self.eb_image.data.shape[1]
        ib_shape = self.ib_image.data.shape[0], self.ib_image.data.shape[1] + self.eb_image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (
            coords[1] > 0 and coords[1] < eb_shape[1]
        ):
            image = self.eb_image
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (
            coords[1] > eb_shape[0] and coords[1] < ib_shape[1]
        ):
            image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
        else:
            beam_type, image = None, None

        # logging
        logging.debug( {"msg": "get_data_from_coord", "coords": coords, "beam_type": beam_type})

        return coords, beam_type, image
    
    def closeEvent(self, event: QEvent):
        self.viewer.layers.clear()
        event.accept()

    def clear_viewer(self):
        self.viewer.layers.clear()
        self.eb_layer = None
        self.ib_layer = None

    def restore_active_layer_for_movement(self):
        """Restore the active layer to the electron beam for movement"""
        if self.eb_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.eb_layer

    def clear_alignment_area(self):
        """Hide the alignment area layer"""
        if self.alignment_layer is not None:
            self.alignment_layer.mode = "pan_zoom"
            self.alignment_layer.visible = False
        self.restore_active_layer_for_movement()

    def toggle_alignment_area(self, reduced_area: FibsemRectangle, editable: bool = True):
        """Toggle the alignment area layer to selection mode, and display the alignment area."""
        self.set_alignment_layer(reduced_area, editable=editable)
        self.alignment_layer.visible = True

    def set_alignment_layer(self,
                            reduced_area: FibsemRectangle = FibsemRectangle(0.25, 0.25, 0.5, 0.5),
                            editable: bool = True):
        """Set the alignment area layer in napari."""

        # add alignment area to napari
        data = convert_reduced_area_to_napari_shape(reduced_area=reduced_area,
                                                    image_shape=self.ib_image.data.shape,
                                                   )
        if self.alignment_layer is None or ALIGNMENT_LAYER_PROPERTIES["name"] not in self.viewer.layers:
            self.alignment_layer = self.viewer.add_shapes(data=data, 
                                                          name=ALIGNMENT_LAYER_PROPERTIES["name"], 
                        shape_type=ALIGNMENT_LAYER_PROPERTIES["shape_type"], 
                        edge_color=ALIGNMENT_LAYER_PROPERTIES["edge_color"], 
                        edge_width=ALIGNMENT_LAYER_PROPERTIES["edge_width"], 
                        face_color=ALIGNMENT_LAYER_PROPERTIES["face_color"], 
                        opacity=ALIGNMENT_LAYER_PROPERTIES["opacity"], 
                        translate=self.ib_layer.translate) # match the fib layer translation
            self.alignment_layer.metadata = ALIGNMENT_LAYER_PROPERTIES["metadata"]
            self.alignment_layer.events.data.connect(self.update_alignment)
            self.alignment_area_updated.connect(self._on_alignment_area_updated)
        else:
            self.alignment_layer.data = data

        if editable:
            self.viewer.layers.selection.active = self.alignment_layer
            # from napari.layers.shapes.shapes import Mode as NapariShapesMode
            # self.alignment_layer.mode = NapariShapesMode.SELECT
            # self.alignment_layer.selected_data = {0} # select the rectangle
            self.alignment_layer.mode = "select"
            self.alignment_layer.selected_data.clear()
        # TODO: prevent rotation of rectangles?

    def update_alignment(self, event):
        """Validate the alignment area, and update the parent ui."""
        reduced_area = self.get_alignment_area()
        if reduced_area is None:
            return
        self.alignment_area_updated.emit(reduced_area)

    def _on_alignment_area_updated(self, reduced_area: FibsemRectangle):
        """Update the parent ui with the new alignment area. (compatibility for AutoLamellaUI)
        TODO: migrate to AutoLamellaUI once mono-repo is complete.
        Args:
            reduced_area (FibsemRectangle): The new alignment area.
        """
        if self.parent is None:
            return
        try:
            is_valid = reduced_area.is_valid_reduced_area
            msg = "Edit Alignment Area. Press Continue when done."
            if not is_valid:
                msg = "Invalid Alignment Area. Please adjust inside FIB Image."
            self.parent.label_instructions.setText(msg)
            self.parent.pushButton_yes.setEnabled(is_valid)
        except Exception as e:
            logging.info(f"Error updating alignment area: {e}")

    def get_alignment_area(self) -> Optional[FibsemRectangle]:
        """Get the alignment area from the alignment layer."""
        data = self.alignment_layer.data
        if data is None or len(data) == 0:
            return None
        data = data[0]
        return convert_shape_to_image_area(data, self.ib_image.data.shape)