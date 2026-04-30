import logging
import os
import threading
from typing import List, Optional, Union

import napari
from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread
from napari.layers import Image as NapariImageLayer
from fibsem import conversions, utils
from fibsem import config as fcfg
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.calibration import run_autofocus
from fibsem.fm.structures import (
    AutoFocusSettings,
    CameraImageTransform,
    ChannelSettings,
    FluorescenceImage,
    FluorescenceConfiguration,
    OverviewParameters,
    ZParameters,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    FibsemStagePosition,
    Point,
)
from fibsem.ui.fm.widgets import (
    AutofocusWidget,
    CameraWidget,
    FluorescenceMultiChannelWidget,
    HistogramWidget,
    ObjectiveControlWidget,
    ZParametersWidget,
)
from fibsem.ui.napari.utilities import is_position_inside_layer, update_text_overlay
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel


class FMControlWidget(QWidget):
    """Widget for Fluorescence Microscope Control"""

    acquisition_finished_signal = pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.parent_widget = parent
        self.microscope = microscope

        self.viewer = viewer
        self._lock = threading.RLock()
        # if self.parent_widget is not None:
        # self.parent_widget.setEnabled(False)

        if self.microscope.fm is None:
            raise ValueError(
                "FluorescenceMicroscope (fm) must be initialized in the FibsemMicroscope."
            )
        self.fm = self.microscope.fm

        self.fm_resolution = self.fm.camera.resolution
        self.field_of_view = self.fm.camera.field_of_view

        # default settings
        self.channel_settings = ChannelSettings()

        # Consolidated acquisition threading
        self._acquisition_thread: Optional[threading.Thread] = None
        self._acquisition_stop_event = threading.Event()
        self._current_acquisition_type: Optional[str] = None

        self.initUI()
        self.connect_signals()
        self._update_acquisition_button_states()

        update_text_overlay(self.viewer, self.microscope)

    def initUI(self):
        """Initialize the user interface."""
        # objective control widget
        self.objectiveControlWidget = ObjectiveControlWidget(fm=self.fm, parent=self)
        self.btn_refresh_objective = IconToolButton(
            icon="mdi:refresh", tooltip="Refresh objective position"
        )
        self.objectivePanel = TitledPanel(
            "Objective Control", content=self.objectiveControlWidget, collapsible=True
        )
        self.objectivePanel.add_header_widget(self.btn_refresh_objective)
        self.objectivePanel.expand()  # expand objective control by default

        # create camera widget
        self.cameraWidget = CameraWidget(fm=self.fm, parent=self)
        self.cameraPanel = TitledPanel(
            "Camera", content=self.cameraWidget, collapsible=True
        )
        self.cameraPanel.collapse()

        # create channel settings widget
        self.channelSettingsWidget = FluorescenceMultiChannelWidget(
            fm=self.fm, channel_settings=[self.channel_settings], parent=self
        )
        self.channelPanel = TitledPanel(
            "Channel Settings", content=self.channelSettingsWidget, collapsible=True
        )
        self.channelPanel.expand()  # expand channel settings by default

        # create z parameters widget
        self.zParametersWidget = ZParametersWidget(
            z_parameters=ZParameters(), parent=self
        )
        self.zParametersPanel = TitledPanel(
            "Z-Stack Parameters", content=self.zParametersWidget, collapsible=True
        )
        self.zParametersPanel.collapse()

        # Create autofocus widget
        self.autofocusWidget = AutofocusWidget(
            channel_settings=self.channelSettingsWidget.channel_settings, parent=self
        )
        self.autofocusWidget.single_fine_search_mode()
        self.autofocusPanel = TitledPanel(
            "Autofocus Settings", content=self.autofocusWidget, collapsible=True
        )
        self.autofocusPanel.setVisible(False)  # Hide autofocus settings by default

        # image histogram
        self.histogramWidget = HistogramWidget(parent=self)
        self.histogramPanel = TitledPanel(
            "Image Histogram", content=self.histogramWidget, collapsible=True
        )
        self.histogramPanel.setVisible(False)  # Hide histogram by default

        self.pushButton_toggle_acquisition = QPushButton("Start Acquisition", self)
        self.pushButton_acquire_single_image = QPushButton("Acquire Image", self)
        self.pushButton_acquire_zstack = QPushButton("Acquire Z-Stack", self)
        self.pushButton_run_autofocus = QPushButton("Run Auto-Focus", self)
        self.pushButton_cancel_acquisition = QPushButton("Cancel Acquisition", self)

        # Checkbox for lamella association
        self.checkBox_associate_with_lamella = QCheckBox(
            "Save to Selected Lamella", self
        )
        self.checkBox_associate_with_lamella.setChecked(True)
        self.checkBox_associate_with_lamella.setToolTip(
            "When checked, saves files in the selected lamella's directory with lamella name prefix.\n"
            "When unchecked, saves files in the experiment directory."
        )

        self.pushButton_toggle_acquisition.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_single_image.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_zstack.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autofocus.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_cancel_acquisition.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        # Create progress bar (hidden by default)
        self.progressBar_current_acquisition = QProgressBar(self)
        self.progressBar_acquisition_task = QProgressBar(self)
        self.progressText = QLabel("Acquisition Progress", self)
        self.progressBar_current_acquisition.hide()
        self.progressBar_acquisition_task.hide()
        self.progressText.hide()

        # Create scroll area for main content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(self.objectivePanel)
        scroll_layout.addWidget(self.cameraPanel)
        scroll_layout.addWidget(self.channelPanel)
        scroll_layout.addWidget(self.autofocusPanel)
        scroll_layout.addWidget(self.zParametersPanel)
        scroll_layout.addWidget(self.histogramPanel)
        scroll_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(1)  # Qt.ScrollBarAlwaysOff

        # create grid layout for buttons
        button_layout = QGridLayout()
        button_layout.addWidget(self.checkBox_associate_with_lamella, 0, 0, 1, 2)
        button_layout.addWidget(self.pushButton_toggle_acquisition, 1, 0, 1, 2)
        button_layout.addWidget(self.pushButton_run_autofocus, 2, 0, 1, 2)
        button_layout.addWidget(self.pushButton_acquire_single_image, 3, 0)
        button_layout.addWidget(self.pushButton_acquire_zstack, 3, 1)
        button_layout.addWidget(self.pushButton_cancel_acquisition, 4, 0, 1, 2)
        button_layout.addWidget(self.progressText, 5, 0, 1, 2)
        button_layout.addWidget(self.progressBar_current_acquisition, 6, 0, 1, 2)
        button_layout.addWidget(self.progressBar_acquisition_task, 7, 0, 1, 2)

        # Main layout with scroll area and buttons
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    @property
    def is_acquiring(self) -> bool:
        """Check if any acquisition is currently active."""
        return self.fm.is_acquiring or bool(
            self._acquisition_thread and self._acquisition_thread.is_alive()
        )

    @property
    def is_acquisition_active(self) -> bool:
        """Check if any acquisition or operation is currently running.

        Returns:
            True if any acquisition (single image, overview, z-stack, positions) or autofocus is active
        """
        return self._current_acquisition_type is not None

    def _get_current_settings(self):
        """Get current settings from all widgets for acquisition operations.

        Returns:
            Dictionary containing all current settings for acquisitions
        """
        # Get channel settings (always a list now)
        channel_settings = self.channelSettingsWidget.channel_settings

        # Get selected channel for live acquisition
        selected_channel_settings = self.channelSettingsWidget.selected_channel

        return {
            "channel_settings": channel_settings,
            "selected_channel_settings": selected_channel_settings,
            "z_parameters": self.zParametersWidget.z_parameters,
            "autofocus_settings": self.autofocusWidget.autofocus_settings,
            "camera_settings": self.cameraWidget.camera_settings,
        }

    def connect_signals(self):
        """Connect internal signals."""
        # buttons
        self.pushButton_toggle_acquisition.clicked.connect(self.toggle_acquisition)
        self.pushButton_acquire_single_image.clicked.connect(self.acquire_image)
        self.pushButton_acquire_zstack.clicked.connect(self.acquire_image)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)
        self.pushButton_cancel_acquisition.clicked.connect(self.cancel_acquisition)
        self.btn_refresh_objective.clicked.connect(
            lambda: self.objectiveControlWidget.update_objective_position_labels(None)
        )

        # microscope signals
        self.fm.acquisition_signal.connect(self.update_image)
        self.acquisition_finished_signal.connect(self._on_acquisition_finished)
        self.fm.acquisition_progress_signal.connect(self._on_acquisition_progress)

        # live parameter updates from channel list
        self.channelSettingsWidget.channel_field_changed.connect(
            self._on_channel_field_changed
        )

        self.viewer.mouse_double_click_callbacks.append(self.on_mouse_double_click)

        # Update checkbox text when lamella selection changes
        if self.parent_widget is not None and hasattr(
            self.parent_widget, "lamella_list"
        ):
            self.parent_widget.lamella_list.lamella_selected.connect(
                self._update_checkbox_text
            )

        # Initialize checkbox text
        self._update_checkbox_text()

    def close_widget(self):
        """Close the widget."""

        # disconnect signals
        self.fm.acquisition_signal.disconnect(self.update_image)
        self.viewer.mouse_double_click_callbacks.remove(self.on_mouse_double_click)

        # Disconnect lamella list signal if connected
        if self.parent_widget is not None and hasattr(
            self.parent_widget, "lamella_list"
        ):
            try:
                self.parent_widget.lamella_list.lamella_selected.disconnect(
                    self._update_checkbox_text
                )
            except Exception:
                pass

        self.close()

    def on_mouse_double_click(self, viewer, event):
        # only left clicks
        if event.button != 1:
            return

        # Prevent stage movement during acquisitions
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            event.handled = True
            return

        logging.info(f"-" * 50)

        selected_layer: Optional[NapariImageLayer] = None
        movement_type = None
        ALT_MODIFIER = "Alt" in event.modifiers

        fm_image_layers = []
        for layer in self.viewer.layers:
            if isinstance(layer, NapariImageLayer) and "FM Image" in layer.name:
                fm_image_layers.append(layer.name)

        for fm_layer in fm_image_layers:
            if is_position_inside_layer(event.position, self.viewer.layers[fm_layer]):
                selected_layer = self.viewer.layers[fm_layer]
                pixelsize = selected_layer.metadata.get("pixel_size_x", None)
                image_shape = selected_layer.data.shape[-2:]
                movement_type = "FM"
                break

        if selected_layer is None:
            logging.info(f"Position {event.position} is not in any valid layer")
            return

        if pixelsize is None:
            logging.info("Pixelsize is None")
            return

        if (
            self.microscope.get_stage_orientation()
            not in self.microscope.fm.valid_orientations
        ):
            logging.info("Stage must be in FM or SEM orientation to move stage via FM")
            event.handled = True
            return

        # convert from image coordinates to microscope coordinates
        coords = selected_layer.world_to_data(event.position)
        if len(coords) in [3, 4]:
            coords = coords[-2:]

        point_clicked = conversions.image_to_microscope_image_coordinates2(
            coord=Point(x=coords[1], y=coords[0]),  # yx required
            image_shape=image_shape,
            pixelsize=pixelsize,
        )

        logging.info(
            f"Coordinates: {coords} - {point_clicked} - Movement Type {movement_type} - Alt Modifier {ALT_MODIFIER}"
        )
        if movement_type == "FM":
            point_clicked = (
                point_clicked[0],
                -point_clicked[1],
            )  # Y-inverse when t=0, need to make this more robust
            # Apply inverse transform to account for image transformation
            if self.fm._transform is CameraImageTransform.FLIP_X:
                point_clicked = (-point_clicked[0], point_clicked[1])
            elif self.fm._transform is CameraImageTransform.FLIP_Y:
                point_clicked = (point_clicked[0], -point_clicked[1])
            elif self.fm._transform is CameraImageTransform.FLIP_XY:
                point_clicked = (-point_clicked[0], -point_clicked[1])
            elif self.fm._transform is CameraImageTransform.ROTATE_90_CW:
                point_clicked = (point_clicked[1], -point_clicked[0])
            elif self.fm._transform is CameraImageTransform.ROTATE_90_CCW:
                point_clicked = (-point_clicked[1], point_clicked[0])
            elif self.fm._transform is CameraImageTransform.ROTATE_180:
                point_clicked = (-point_clicked[0], -point_clicked[1])

            self.microscope.move_stage_relative(
                FibsemStagePosition(x=point_clicked[0], y=point_clicked[1])
            )
        logging.info(f"-" * 50)

    def _on_channel_field_changed(self, channel, field: str, value) -> None:
        """Update a single microscope parameter during live acquisition."""
        if not self.fm.is_acquiring:
            return
        if channel is not self.channelSettingsWidget.selected_channel:
            return
        logging.info(f"Channel field changed: {field} -> {value}")
        if field == "excitation_wavelength":
            self.fm.filter_set.excitation_wavelength = value
        elif field == "emission_wavelength":
            self.fm.filter_set.emission_wavelength = value
        elif field == "exposure_time":
            self.fm.set_exposure_time(value)  # seconds
        elif field == "gain":
            self.fm.set_gain(value)  # fraction
        elif field == "power":
            self.fm.set_power(value)  # fraction
        elif field == "color":
            self.fm.set_channel_color(value)

    def _on_acquisition_finished(self):
        """Handle consolidated acquisition completion in the main thread."""

        # clear acquisition state
        self._current_acquisition_type = None
        self._last_remaining_time = None

        # hide progress bar when acquisition finishes
        self.progressBar_acquisition_task.hide()
        self.progressBar_current_acquisition.hide()
        self.progressText.setText("")

        # refresh overview bbox, objective widget, and button states
        self.objectiveControlWidget.update_objective_position_labels()

        self._update_acquisition_button_states()

    @ensure_main_thread
    def _on_acquisition_progress(self, progress: dict):
        """Update progress bars on acquisition signal"""

        # Show progress bar when acquisition progress is updated
        if self._current_acquisition_type in ["positions", "overview"]:
            # only show acquisition_task progress bar when acquiring overview/positions
            if not self.progressBar_acquisition_task.isVisible():
                self.progressBar_acquisition_task.show()
        if not self.progressBar_current_acquisition.isVisible():
            self.progressBar_current_acquisition.show()
        if not self.progressText.isVisible():
            self.progressText.show()

        progress_zlevels = progress.get("zlevel", None)
        progress_total_zlevels = progress.get("total_zlevels", None)
        progress_current = progress.get("current", None)
        progress_total = progress.get("total", None)
        channel_name = progress.get("channel", None)
        progress_state = progress.get("state", None)

        if progress_state == "moving":
            self.progressText.setText("Moving stage...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        if progress_state == "autofocus":
            self.progressText.setText("Running Autofocus...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        if progress_state == "finished":
            self.progressBar_acquisition_task.hide()
            self.progressBar_current_acquisition.hide()
            self.progressText.setText("Acquisition complete.")
            self.progressText.hide()
            return

        # set progress message
        if channel_name is not None:
            channel_index = progress.get("channel_index", 1)
            total_channels = progress.get("total_channels", 1)
            msg = f"Acquiring {channel_name} ({channel_index}/{total_channels})..."
            self.progressText.setText(msg)

        # set individual image acquisition progress bar
        if progress_zlevels and progress_total_zlevels:
            percentage_zlevel = (
                int((progress_zlevels / progress_total_zlevels) * 100)
                if progress_total_zlevels > 0
                else 0
            )
            self.progressBar_current_acquisition.setValue(percentage_zlevel)
            self.progressBar_current_acquisition.setFormat(
                f"Z-level {progress_zlevels}/{progress_total_zlevels}"
            )

        # set total acquisition task progress
        if progress_current is not None and progress_total is not None:
            # Format time remaining string if available
            time_remaining_str = ""
            remaining_time = progress.get("estimated_remaining_time", 0)
            if remaining_time > 0:
                self._last_remaining_time = remaining_time
            if self._last_remaining_time is not None and self._last_remaining_time > 0:
                time_remaining_str = f"Remaining Time: {utils.format_duration(self._last_remaining_time)}"

            # Set progress bar/text
            percentage = (
                int((progress_current / progress_total) * 100)
                if progress_total > 0
                else 0
            )
            msg = f"Position {progress_current}/{progress_total} - {time_remaining_str}"
            self.progressBar_acquisition_task.setValue(percentage)
            self.progressBar_acquisition_task.setFormat(msg)

    @ensure_main_thread
    def update_image(self, image: FluorescenceImage):

        if not isinstance(image, FluorescenceImage):
            return

        # Add to napari viewer
        layer_name = f"FM Image {image.metadata.channels[0].name}"
        colormap = (
            image.metadata.channels[0].color if image.metadata.channels else "gray"
        )
        translation = (0, 0)
        data = image.data
        if data.ndim == 4 and data.shape[0] == 1 and data.shape[1] == 1:
            data = data.squeeze(
                axis=(0, 1)
            )  # squish to 2D (required by napari for now)

        # get translation from eb_image layer if it exists
        if "ELECTRON" in self.viewer.layers:
            eb_layer = self.viewer.layers["ELECTRON"]
            translation = (eb_layer.data.shape[0], 0)  # move it below the SEM image...

        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
            self.viewer.layers[layer_name].metadata = image.metadata.to_dict()
            self.viewer.layers[layer_name].translate = translation
            self.viewer.layers[layer_name].colormap = colormap
            self.viewer.layers[layer_name].blending = "additive"
        else:
            self.viewer.add_image(
                data,
                name=layer_name,
                metadata=image.metadata.to_dict(),
                colormap=colormap,
                translate=translation,
                blending="additive",
            )
            self.viewer.reset_view()

    def start_acquisition(self):
        """Start the fluorescence acquisition."""

        if (
            self.microscope.get_stage_orientation()
            not in self.microscope.fm.valid_orientations
        ):
            logging.warning(
                "Stage is not in FM or SEM orientation. Cannot start acquisition."
            )
            return

        if self.is_acquiring:
            logging.warning("Another acquisition is already in progress.")
            return

        # Get current settings
        settings = self._get_current_settings()
        selected_channel_settings = settings["selected_channel_settings"]

        if selected_channel_settings is None:
            logging.warning("No channel selected for live acquisition")
            return

        logging.info(
            f"Starting acquisition with channel settings: {selected_channel_settings}"
        )
        self.fm.start_acquisition(channel_settings=selected_channel_settings)
        self._update_acquisition_button_states()

    def stop_acquisition(self):
        """Stop the fluorescence acquisition."""

        logging.info("Acquisition stopped")

        self.fm.stop_acquisition()
        self._update_acquisition_button_states()

    def cancel_acquisition(self):
        """Cancel all ongoing acquisitions (single image, z-stack, overview, positions, autofocus)."""

        # Cancel consolidated acquisition (single image or future consolidated types)
        if self.is_acquiring:
            logging.info(f"Cancelling {self._current_acquisition_type} acquisition")
            self._acquisition_stop_event.set()
            self._acquisition_thread.join(timeout=5)  # type: ignore[union-attr]

        logging.info("All acquisitions cancelled")

    def toggle_acquisition(self):
        """Toggle acquisition start/stop with F6 key."""
        if self.fm.is_acquiring:
            logging.info("F6 pressed: Stopping acquisition")
            self.stop_acquisition()
        else:
            logging.info("F6 pressed: Starting acquisition")
            self.start_acquisition()

    def _update_checkbox_text(self, _=None):
        """Update the checkbox text and state to include the selected lamella name."""
        base_text = "Save to Selected Lamella"

        if self.parent_widget is not None and hasattr(
            self.parent_widget, "get_selected_lamella"
        ):
            lamella = self.parent_widget.get_selected_lamella()
            if lamella is not None:
                self.checkBox_associate_with_lamella.setText(
                    f"{base_text} ({lamella.name})"
                )
                self.checkBox_associate_with_lamella.setEnabled(True)
                self._update_checkbox_tooltip(lamella)
                return

        # No lamella selected or available - disable checkbox
        self.checkBox_associate_with_lamella.setText(base_text)
        self.checkBox_associate_with_lamella.setEnabled(False)
        self._update_checkbox_tooltip(None)

    def _update_checkbox_tooltip(self, lamella):
        """Update the checkbox tooltip to show actual save paths.

        Args:
            lamella: The selected Lamella object, or None if no lamella is selected
        """
        if lamella is not None:
            experiment_path = "experiment directory"
            if self.parent_widget is not None and hasattr(
                self.parent_widget, "experiment"
            ):
                if self.parent_widget.experiment is not None:
                    experiment_path = str(self.parent_widget.experiment.path)

            tooltip = (
                f"When checked: Saves to lamella directory\n"
                f"  Path: {lamella.path}\n"
                f"  Filename: {lamella.name}-image-HH-MM-SS.ome.tiff\n\n"
                f"When unchecked: Saves to experiment directory\n"
                f"  Path: {experiment_path}\n"
                f"  Filename: image-HH-MM-SS.ome.tiff"
            )
        else:
            tooltip = (
                "No lamella selected.\n\n"
                "When a lamella is selected and this option is checked,\n"
                "files will be saved in the lamella's directory with the lamella name prefix.\n"
                "Otherwise, files are saved in the experiment directory."
            )

        self.checkBox_associate_with_lamella.setToolTip(tooltip)

    def _generate_acquisition_filename(self, name: str) -> str:
        """Generate filename for acquisition, using lamella path and name if available.

        Args:
            name: Base name for the file (e.g., "image", "z-stack")

        Returns:
            Full path to the output file
        """
        path = os.getcwd()
        lamella_prefix = ""

        # Get path and lamella info from parent widget
        if self.parent_widget is not None:
            if (
                hasattr(self.parent_widget, "experiment")
                and self.parent_widget.experiment is not None
            ):
                path = self.parent_widget.experiment.path

            # Get selected lamella if available and checkbox is checked
            if self.checkBox_associate_with_lamella.isChecked() and hasattr(
                self.parent_widget, "get_selected_lamella"
            ):
                lamella = self.parent_widget.get_selected_lamella()
                if lamella is not None:
                    self._validate_lamella_selection(lamella)

                    path = lamella.path
                    lamella_prefix = f"{lamella.name}-"

        # Generate filename with timestamp
        timestamp = utils.current_timestamp_v3(timeonly=True)
        filename = os.path.join(path, f"{lamella_prefix}{name}-{timestamp}.ome.tiff")

        logging.info(f"Generated acquisition filename: {filename}")
        return filename

    def _validate_lamella_selection(self, lamella: "Lamella") -> None:
        # Check if current position is close to lamella fluorescence pose
        if (
            lamella.fluorescence_pose is None
            or lamella.fluorescence_pose.stage_position is None
        ):
            return  # No fluorescence pose defined, skip check

        current_position = self.microscope.get_stage_position()
        lamella_position = lamella.fluorescence_pose.stage_position

        # Check if positions are close using x, y, z axes with 5 micrometer tolerance
        is_close = current_position.is_close2(
            lamella_position, tol=5e-6, axes=["x", "y", "z"]
        )

        if not is_close:
            # Show warning dialog to user
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Position Mismatch Warning")
            msg_box.setText(
                f"Current stage position is not close to selected lamella '{lamella.name}' fluorescence pose."
            )
            msg_box.setInformativeText(
                f"Current Position: {current_position.pretty}\n"
                f"Lamella Position: {lamella_position.pretty}\n"
                f"Tolerance: 5.0 μm\n\n"
                f"Do you want to continue with the acquisition?"
            )

            # Create custom buttons
            msg_box.addButton("Continue", QMessageBox.AcceptRole)
            cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

            if cancel_button is not None:
                msg_box.setDefaultButton(cancel_button)

            msg_box.exec_()

            # Check which button was clicked
            if msg_box.clickedButton() == cancel_button:
                logging.info("Acquisition cancelled by user due to position mismatch")
                raise RuntimeError(
                    "Acquisition cancelled by user due to position mismatch"
                )

    def acquire_image(self):
        """Start threaded image acquisition using the current Z parameters and channel settings."""

        if self.is_acquiring:
            logging.warning("Another acquisition is already in progress.")
            return

        logging.info("Starting image acquisition")
        self._current_acquisition_type = "image"
        self._last_remaining_time = None
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        settings = self._get_current_settings()

        z_parameters = None
        channel_settings = settings["selected_channel_settings"]
        if self.sender() is self.pushButton_acquire_zstack:
            channel_settings = settings["channel_settings"]
            z_parameters = settings["z_parameters"]

        # Generate filename for saving
        name = "z-stack" if z_parameters is not None else "image"
        try:
            filename = self._generate_acquisition_filename(name)
        except RuntimeError as e:
            # User cancelled acquisition due to position mismatch
            logging.info(f"Acquisition cancelled: {e}")
            self._current_acquisition_type = None
            self._update_acquisition_button_states()
            return

        # Start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._image_acquistion_worker,
            args=(channel_settings, z_parameters, filename),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _image_acquistion_worker(
        self,
        channel_settings: List[ChannelSettings],
        z_parameters: ZParameters,
        filename: str,
    ):
        """Worker thread for single image acquisition.

        Args:
            channel_settings: Channel settings for acquisition
            z_parameters: Z-stack parameters (None for single image)
            filename: Full path to save the acquired image
        """
        try:
            image = acquire_image(
                microscope=self.fm,
                channel_settings=channel_settings,
                zparams=z_parameters,
                stop_event=self._acquisition_stop_event,
                filename=filename,
            )

            if self._acquisition_stop_event.is_set() or image is None:
                logging.info("image acquisition was cancelled")
                return

            # Emit the image
            # self.update_persistent_image_signal.emit(image)
            logging.info("Image acquisition completed successfully")

        except Exception as e:
            logging.error(f"Error during image acquisition: {e}")

        finally:
            self.acquisition_finished_signal.emit()

    def _update_acquisition_button_states(self):
        """Update acquisition button states and control widgets based on live acquisition or specific acquisition status."""
        if (
            self.microscope.get_stage_orientation()
            not in self.microscope.fm.valid_orientations
        ):
            # If not in FM or SEM orientation, disable all acquisition buttons
            self.pushButton_toggle_acquisition.setEnabled(False)
            self.pushButton_acquire_zstack.setEnabled(False)
            self.pushButton_cancel_acquisition.setEnabled(False)
            return

        # Check if any acquisition is active (live or specific acquisitions)
        any_acquisition_active = self.is_acquiring or self.is_acquisition_active

        # Special case buttons with unique behavior
        self.pushButton_cancel_acquisition.setVisible(self.is_acquisition_active)

        # Update toggle acquisition button text based on state
        if self.fm.is_acquiring:
            self.pushButton_toggle_acquisition.setText("Stop Acquisition")
        else:
            self.pushButton_toggle_acquisition.setText("Start Acquisition")

        # Enable/disable standard buttons based on acquisition state
        for button in (
            self.pushButton_acquire_single_image,
            self.pushButton_acquire_zstack,
            self.pushButton_run_autofocus,
        ):
            button.setEnabled(not any_acquisition_active)

        # Disable control widgets during acquisition
        self.objectiveControlWidget.setEnabled(not self.is_acquisition_active)
        self.zParametersWidget.setEnabled(not any_acquisition_active)

        # Disable channel settings during specific acquisitions (but allow during live imaging)
        self.channelSettingsWidget.setEnabled(not self.is_acquisition_active)
        self.objectiveControlWidget.pushButton_insert_objective.setEnabled(
            not any_acquisition_active
        )
        self.objectiveControlWidget.pushButton_move_to_focus.setEnabled(
            not any_acquisition_active
        )
        self.objectiveControlWidget.pushButton_retract_objective.setEnabled(
            not any_acquisition_active
        )

        # Disable channel list selection during any acquisition to prevent switching channels mid-acquisition
        # self.channelSettingsWidget.channel_list.setEnabled(not any_acquisition_active)

    def run_autofocus(self):
        """Start threaded auto-focus using the current channel settings and Z parameters."""
        if self.is_acquiring:
            logging.warning(
                "Cannot run auto-focus while another acquisition is in progress. Stop acquisition first."
            )
            return

        logging.info("Starting auto-focus")
        self._current_acquisition_type = "autofocus"
        self._last_remaining_time = None
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings["selected_channel_settings"]
        z_parameters = settings["z_parameters"]
        # autofocus_settings = settings['autofocus_settings']

        # Start auto-focus thread
        self._acquisition_thread = threading.Thread(
            target=self._autofocus_worker,
            args=(channel_settings, z_parameters),
            daemon=True,
        )
        self._acquisition_thread.start()

    def _autofocus_worker(
        self, channel_settings: ChannelSettings, z_parameters: ZParameters
    ):
        """Worker thread for auto-focus."""
        try:
            logging.info("Running auto-focus with laplacian method")

            best_z = run_autofocus(
                microscope=self.fm,
                channel_settings=channel_settings,
                # z_parameters=z_parameters,
                method="laplacian",
                stop_event=self._acquisition_stop_event,
            )

            if best_z is None or self._acquisition_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return

            logging.info(
                f"Auto-focus completed successfully. Best focus: {best_z * 1e6:.1f} μm"
            )

        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")
        finally:
            self.acquisition_finished_signal.emit()

    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info(
            "Closing FluorescenceControlWidget, stopping acquisition if running."
        )

        if self.microscope.fm is None:
            event.accept()
            return

        # Stop live acquisition
        self.microscope.fm.acquisition_signal.disconnect(self.update_image)
        if self.microscope.fm.is_acquiring:
            try:
                self.microscope.fm.stop_acquisition()
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                logging.warning("Acquisition stopped due to widget close.")

        if self.parent_widget:
            self.parent_widget.setEnabled(True)

        event.accept()

    def load_fm_configuration(self):
        """Load FM configuration from a YAML file."""
        try:
            # Get filename from user
            filename, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select Fluorescence Configuration to load",
                directory=os.path.join(fcfg.CONFIG_PATH, "fm"),
                filter="YAML files (*.yaml *.yml);;All files (*.*)",
            )

            if not filename:
                return

            # load configuration and apply to widgets
            config = FluorescenceConfiguration.load(filename)
            self._apply_fluorescence_configuration(config)

        except Exception as e:
            logging.error(f"Failed to load FM configuration: {e}")

    def _apply_fluorescence_configuration(self, config: FluorescenceConfiguration):
        """Apply a FluorescenceConfiguration to the widget settings."""
        self.channelSettingsWidget.channel_settings = config.channel_settings
        self.zParametersWidget.z_parameters = config.z_parameters
        self.cameraWidget.camera_settings = config.camera_settings
        if config.focus_position is not None:
            self.objectiveControlWidget._set_focus_position(config.focus_position)
        if config.limit_position:
            self.objectiveControlWidget._set_limit_position(config.limit_position)

    def save_fm_configuration(self):
        """Save current FM configuration to a YAML file."""
        try:
            # Get filename from user
            filename, _ = QFileDialog.getSaveFileName(
                parent=self,
                caption="Save Fluorescence Configuration",
                directory=os.path.join(fcfg.CONFIG_PATH, "fm", "fm-configuration.yaml"),
                filter="YAML files (*.yaml *.yml);;All files (*.*)",
            )

            if not filename:
                return

            # Gather current settings from UI
            settings = self._get_current_settings()

            # Create FM configuration
            fm_config = FluorescenceConfiguration(
                channel_settings=settings["channel_settings"],
                z_parameters=settings["z_parameters"],
                overview_parameters=OverviewParameters(),
                autofocus_settings=AutoFocusSettings(),
                camera_settings=settings["camera_settings"],
                focus_position=self.fm.objective.focus_position,
            )

            # Export configuration
            fm_config.export(filename)

            QMessageBox.information(
                self,
                "Fluorescence Configuration Saved",
                f"Configuration saved to:\n{filename}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Fluorescence Configuration",
                f"Failed to save FM configuration:\n{str(e)}",
            )
            logging.error(f"Error saving FM configuration: {e}")


def create_widget(
    microscope: FibsemMicroscope,
    viewer: napari.Viewer,
    parent: Optional[QWidget] = None,
) -> FMControlWidget:
    """Create the FMControlWidget with a demo microscope."""

    widget = FMControlWidget(microscope=microscope, viewer=viewer, parent=parent)
    return widget


def main():
    """Main function to run the widget standalone."""
    microscope, settings = utils.setup_session()

    if microscope.fm is None:
        logging.error(
            "FluorescenceMicroscope is not initialized. Cannot create FMControlWidget."
        )
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    # Ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True

    viewer = napari.Viewer()
    widget = create_widget(microscope, viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
    return


if __name__ == "__main__":
    main()
