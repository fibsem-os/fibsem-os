import logging
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
from superqt import QIconifyIcon, ensure_main_thread

from fibsem import acquire, constants, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
)
from fibsem.ui import stylesheets as stylesheets
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
from fibsem.ui.napari.utilities import (
    add_points_layer,
    draw_crosshair_in_napari,
    draw_scalebar_in_napari,
)
from fibsem.ui.widgets.custom_widgets import _SpinnerLabel
from fibsem.ui.qtdesigner_files import ImageSettingsWidget as ImageSettingsWidgetUI
from fibsem.ui.widgets.dual_beam_widget import FibsemDualBeamWidget
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget


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

        self._setup_ui()
        self.setup_connections()

        if image_settings is not None:
            self.image_settings = image_settings
            self._set_image_settings_to_ui(image_settings)
            self.update_ui_saving_settings()

        # register initial images
        self.eb_layer = self.viewer.add_image(self.eb_image.data, name=BeamType.ELECTRON.name, blending='additive')
        self.ib_layer = self.viewer.add_image(self.ib_image.data, name=BeamType.ION.name, blending='additive')
        self._on_acquire(self.eb_image)
        self._on_acquire(self.ib_image)

    # ------------------------------------------------------------------
    # Backward-compatibility properties for callers outside this widget
    # ------------------------------------------------------------------

    @property
    def checkBox_image_save_image(self):
        return self.image_settings_widget.save_image_check

    @property
    def lineEdit_image_path(self):
        return self.image_settings_widget.path_edit

    @property
    def lineEdit_image_label(self):
        return self.image_settings_widget.filename_edit

    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Set up the UI components and layout"""

        # --- Spinner / status row (hidden by default) ---
        self._spinner = _SpinnerLabel(parent=self)
        self._status_label = QtWidgets.QLabel("Acquiring...")
        self._status_label.setStyleSheet("color: #4fc3f7;")
        status_row = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 2, 0, 2)
        status_layout.setSpacing(6)
        status_layout.addStretch()
        status_layout.addWidget(self._spinner)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        self._status_row = status_row
        self._status_row.setVisible(False)

        self.pushButton_start_acquisition = QtWidgets.QPushButton("Start Acquisition")
        self.pushButton_acquire_fib_image = QtWidgets.QPushButton("Acquire FIB Image")
        self.pushButton_acquire_sem_image = QtWidgets.QPushButton("Acquire SEM Image")
        self.pushButton_take_all_images = QtWidgets.QPushButton("Acquire All Images")
        self.pushButton_run_autocontrast = QtWidgets.QPushButton("Run AutoContrast")
        self.pushButton_run_autofocus = QtWidgets.QPushButton("Run AutoFocus")
        self.checkBox_save_with_selected_lamella = QtWidgets.QCheckBox("Save with Selected Lamella")

        # add buttons and status row to layout
        # scroll area occupies row 1 in gridLayout; all buttons go below at row 2+
        self.gridLayout.addWidget(self.checkBox_save_with_selected_lamella, 2, 0, 1, 2)
        self.gridLayout.addWidget(self._status_row, 3, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_run_autocontrast, 4, 0)
        self.gridLayout.addWidget(self.pushButton_run_autofocus, 4, 1)
        self.gridLayout.addWidget(self.pushButton_start_acquisition, 5, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_acquire_sem_image, 6, 0)
        self.gridLayout.addWidget(self.pushButton_acquire_fib_image, 6, 1)
        self.gridLayout.addWidget(self.pushButton_take_all_images, 7, 0, 1, 2)

        # --- ImageSettingsWidget (resolution, dwell, hfw, integration, save) ---
        self.image_settings_widget = ImageSettingsWidget(
            parent=self.scrollAreaWidgetContents,
            show_save=True,
            show_advanced=False,
        )
        self.image_group = QtWidgets.QGroupBox("Image")
        self.image_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        image_group_layout = QtWidgets.QVBoxLayout(self.image_group)
        image_group_layout.setContentsMargins(6, 6, 6, 6)
        image_group_layout.addWidget(self.image_settings_widget)

        # --- FibsemDualBeamWidget (SEM + FIB beam & detector settings) ---
        self.dual_beam_widget = FibsemDualBeamWidget(
            microscope=self.microscope,
            initial_beam_type=BeamType.ELECTRON,
            parent=self.scrollAreaWidgetContents,
        )
        self.dual_beam_widget.populate_combos()
        self.dual_beam_widget.sync_from_microscope()

        self.gridLayout_2.addWidget(self.dual_beam_widget, 2, 0, 1, 3)
        self.gridLayout_2.addWidget(self.image_group, 3, 0, 1, 3)

    def setup_connections(self) -> None:
        """Set up the connections for the UI components, signals and initialise the UI"""

        # buttons
        self.pushButton_acquire_sem_image.clicked.connect(self.acquire_sem_image)
        self.pushButton_acquire_fib_image.clicked.connect(self.acquire_fib_image)
        self.pushButton_take_all_images.clicked.connect(self.acquire_reference_images)

        # save image with selected lamella control
        show_lamella_controls = hasattr(self.parent, "experiment")
        self.checkBox_save_with_selected_lamella.setVisible(show_lamella_controls)
        self.checkBox_save_with_selected_lamella.toggled.connect(self.update_ui_saving_settings)
        self.checkBox_save_with_selected_lamella.setToolTip(
            "Save images to the path of the currently selected lamella position in the experiment."
        )
        self.image_settings_widget.save_image_check.toggled.connect(self.update_ui_saving_settings)
        try:
            self.parent.comboBox_current_lamella.currentIndexChanged.connect(self._on_current_lamella_changed)
        except Exception as e:
            logging.debug(f"Error connecting to lamella selection changes: {e}")

        # util
        self.checkBox_enable_ruler.toggled.connect(self.update_ruler)
        self.scalebar_checkbox.toggled.connect(self.update_ui_tools)
        self.crosshair_checkbox.toggled.connect(self.update_ui_tools)

        # signals
        self.acquisition_progress_signal.connect(self.handle_acquisition_progress_update)
        self.viewer_update_signal.connect(self.update_ui_tools)

        # auto functions
        self.pushButton_run_autocontrast.clicked.connect(self.run_autocontrast)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)

        # set ui stylesheets
        self.pushButton_acquire_sem_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_fib_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_take_all_images.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_run_autocontrast.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autofocus.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        self.pushButton_run_autocontrast.setIcon(QIconifyIcon("mdi:contrast-circle", color=stylesheets.GRAY_ICON_COLOR))
        self.pushButton_run_autofocus.setIcon(QIconifyIcon("mdi:image-filter-center-focus", color=stylesheets.GRAY_ICON_COLOR))

        # feature flags
        self.pushButton_show_alignment_area.setVisible(False)

        self.acquisition_buttons: List[QtWidgets.QPushButton] = [
            self.pushButton_take_all_images,
            self.pushButton_acquire_sem_image,
            self.pushButton_acquire_fib_image,
        ]

        # live acquisition
        self.pushButton_start_acquisition.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.microscope.sem_acquisition_signal.connect(self._on_acquire)
        self.microscope.fib_acquisition_signal.connect(self._on_acquire)
        self.pushButton_start_acquisition.clicked.connect(self.toggle_live_acquisition)

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
        if self.microscope.is_acquiring:
            logging.info("Microscope is already acquiring. Stopping acquisition...")
            self.microscope.stop_acquisition()
            self.pushButton_start_acquisition.setText("Start Acquisition")
            self.pushButton_start_acquisition.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            for btn in self.acquisition_buttons:
                btn.setEnabled(True)
                btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            for btn in [self.pushButton_run_autocontrast, self.pushButton_run_autofocus]:
                btn.setEnabled(True)
            self.image_group.setEnabled(True)
            self._spinner.stop()
            self._status_row.setVisible(False)
            return

        # disable other buttons while live acquisition is running
        for btn in self.acquisition_buttons:
            btn.setEnabled(False)
        for btn in [self.pushButton_run_autocontrast, self.pushButton_run_autofocus]:
            btn.setEnabled(False)
        self.image_group.setEnabled(False)
        self._spinner.start()
        self._status_row.setVisible(True)

        beam_type = self.dual_beam_widget.beam_type
        self.microscope.start_acquisition(beam_type)
        self.pushButton_start_acquisition.setText("Stop Acquisition")
        self.pushButton_start_acquisition.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)

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
        self.ruler_layer = add_points_layer(
            self.viewer,
            data=data,
            name=RULER_LAYER_NAME,
            size=IMAGING_RULER_LAYER_PROPERTIES["size"],
            face_color=IMAGING_RULER_LAYER_PROPERTIES["face_color"],
            border_color=IMAGING_RULER_LAYER_PROPERTIES["edge_color"],
            text=text,
        )
        self.viewer.add_shapes(data, shape_type='line', edge_color='lime', name=RULER_LINE_LAYER_NAME, edge_width=5)
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

        def check_point_image_in_eb(point: Tuple[int, int]) -> bool:
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

    def run_autocontrast(self) -> None:
        """Run autocontrast for the selected beam type."""
        beam_type = self.dual_beam_widget.beam_type
        self._toggle_interactions(enable=False)
        worker = self._autocontrast_worker(beam_type)
        worker.finished.connect(lambda: self._on_auto_function_finished("AutoContrast", beam_type=beam_type))
        worker.start()

    def run_autofocus(self) -> None:
        """Run autofocus for the selected beam type."""
        beam_type = self.dual_beam_widget.beam_type
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
        beam_widget = self.dual_beam_widget.sem_widget if beam_type is BeamType.ELECTRON else self.dual_beam_widget.fib_widget
        beam_widget.sync_from_microscope()
        if name == "AutoFocus":
            wd = beam_widget.beam_settings_widget.working_distance_spinbox.value()
            napari.utils.notifications.show_info(f"AutoFocus Complete. Best WD: {wd:.2f}mm")
        if name == "AutoContrast":
            napari.utils.notifications.show_info("AutoContrast Complete.")

    def _get_image_settings_from_ui(self) -> ImageSettings:
        """Get the image settings from the UI and return an ImageSettings object."""
        self.image_settings = self.image_settings_widget.get_settings()
        self.image_settings.beam_type = self.dual_beam_widget.beam_type
        return self.image_settings

    def _get_beam_settings_from_ui(self) -> BeamSettings:
        """Get the beam settings from the UI and return a BeamSettings object"""
        self.beam_settings = self.dual_beam_widget.get_beam_settings()
        return self.beam_settings

    def _get_detector_settings_from_ui(self) -> FibsemDetectorSettings:
        """Get the detector settings from the UI and return a FibsemDetectorSettings object"""
        self.detector_settings = self.dual_beam_widget.get_detector_settings()
        return self.detector_settings

    def set_ui_from_settings(
        self,
        image_settings: ImageSettings,
        beam_type: BeamType,
        beam_settings: Optional[BeamSettings] = None,
        detector_settings: Optional[FibsemDetectorSettings] = None,
    ) -> None:
        """Update the ui from the image, beam and detector settings"""
        self._set_image_settings_to_ui(image_settings)
        beam_widget = self.dual_beam_widget.sem_widget if beam_type is BeamType.ELECTRON else self.dual_beam_widget.fib_widget
        if beam_settings is not None:
            beam_widget.beam_settings_widget.update_from_settings(beam_settings)
        if detector_settings is not None:
            beam_widget.detector_settings_widget.update_from_settings(detector_settings)
        self.update_ui_saving_settings()

    def _set_image_settings_to_ui(self, image_settings: ImageSettings) -> None:
        """Set the image settings to the UI components."""
        self.image_settings_widget.update_from_settings(image_settings)

    def _set_beam_settings_to_ui(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings to the ui components"""
        beam_widget = self.dual_beam_widget.sem_widget if beam_settings.beam_type is BeamType.ELECTRON else self.dual_beam_widget.fib_widget
        beam_widget.beam_settings_widget.update_from_settings(beam_settings)

    def _set_detector_settings_to_ui(self, detector_settings: FibsemDetectorSettings) -> None:
        """Set the detector settings to the UI components."""
        beam_type = self.dual_beam_widget.beam_type
        beam_widget = self.dual_beam_widget.sem_widget if beam_type is BeamType.ELECTRON else self.dual_beam_widget.fib_widget
        beam_widget.detector_settings_widget.update_from_settings(detector_settings)

    def update_ui_saving_settings(self) -> None:
        """Toggle the visibility / state of the image saving settings"""
        save_image = self.image_settings_widget.save_image_check.isChecked()
        self.checkBox_save_with_selected_lamella.setEnabled(save_image)

        save_with_lamella = self.checkBox_save_with_selected_lamella.isChecked()
        self.image_settings_widget.path_edit.setEnabled(save_image and not save_with_lamella)

        if save_with_lamella:
            try:
                idx = self.parent.comboBox_current_lamella.currentIndex()
                lamella = self.parent.experiment.positions[idx]
                self.image_settings_widget.path_edit.setText(str(lamella.path))
            except Exception as e:
                logging.debug(f"Error setting image path from selected lamella: {e}")
        else:
            if hasattr(self.parent, "experiment") and self.parent.experiment is not None:
                self.image_settings_widget.path_edit.setText(str(self.parent.experiment.path))

    def _on_current_lamella_changed(self, index: int):
        """Update the image path when the selected lamella changes"""
        try:
            if self.checkBox_save_with_selected_lamella.isChecked():
                lamella = self.parent.experiment.positions[index]
                self.image_settings_widget.path_edit.setText(str(lamella.path))
                self.checkBox_save_with_selected_lamella.setText(f"Save with Selected Lamella ({lamella.name})")
        except Exception as e:
            logging.debug(f"Error updating image path from selected lamella: {e}")

    def _toggle_interactions(self, enable: bool, caller: str = None, imaging: bool = False):
        for btn in self.acquisition_buttons:
            btn.setEnabled(enable)
        for btn in [self.pushButton_run_autocontrast, self.pushButton_run_autofocus, self.pushButton_start_acquisition]:
            btn.setEnabled(enable)
        self.image_group.setEnabled(enable)
        self.dual_beam_widget.setEnabled(enable)
        self.parent.movement_widget._toggle_interactions(enable, caller="ui")
        if enable:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            self.pushButton_take_all_images.setText("Acquire All Images")
            self.pushButton_acquire_sem_image.setText("Acquire SEM Image")
            self.pushButton_acquire_fib_image.setText("Acquire FIB Image")
            self._spinner.stop()
            self._status_row.setVisible(False)
        elif imaging:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
                btn.setText("Acquiring...")
            self._spinner.start()
            self._status_row.setVisible(True)
        else:
            self.pushButton_take_all_images.setText("Acquire All Images")
            self._spinner.start()
            self._status_row.setVisible(True)

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
        logging.debug({"msg": "get_data_from_coord", "coords": coords, "beam_type": beam_type})

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

    def set_alignment_layer(
        self,
        reduced_area: FibsemRectangle = FibsemRectangle(0.25, 0.25, 0.5, 0.5),
        editable: bool = True,
    ):
        """Set the alignment area layer in napari."""

        # add alignment area to napari
        data = convert_reduced_area_to_napari_shape(
            reduced_area=reduced_area,
            image_shape=self.ib_image.data.shape,
        )
        if self.alignment_layer is None or ALIGNMENT_LAYER_PROPERTIES["name"] not in self.viewer.layers:
            self.alignment_layer = self.viewer.add_shapes(
                data=data,
                name=ALIGNMENT_LAYER_PROPERTIES["name"],
                shape_type=ALIGNMENT_LAYER_PROPERTIES["shape_type"],
                edge_color=ALIGNMENT_LAYER_PROPERTIES["edge_color"],
                edge_width=ALIGNMENT_LAYER_PROPERTIES["edge_width"],
                face_color=ALIGNMENT_LAYER_PROPERTIES["face_color"],
                opacity=ALIGNMENT_LAYER_PROPERTIES["opacity"],
                translate=self.ib_layer.translate,  # match the fib layer translation
            )
            self.alignment_layer.metadata = ALIGNMENT_LAYER_PROPERTIES["metadata"]
            self.alignment_layer.events.data.connect(self.update_alignment)
            self.alignment_area_updated.connect(self._on_alignment_area_updated)
        else:
            self.alignment_layer.data = data

        if editable:
            self.viewer.layers.selection.active = self.alignment_layer
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
