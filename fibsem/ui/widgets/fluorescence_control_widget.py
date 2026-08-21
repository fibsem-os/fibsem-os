import logging
import os
import threading
from typing import List, Optional, Union

from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from fibsem import config as fcfg
from fibsem import conversions, utils
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.calibration import run_autofocus
from fibsem.fm.config import record_recent_channels
from fibsem.fm.structures import (
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceConfiguration,
    FluorescenceImage,
    OverviewParameters,
    ZParameters,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import Point
from fibsem.ui import notification_service
from fibsem.ui.fm.widgets import (
    AutofocusWidget,
    CameraWidget,
    FluorescenceMultiChannelWidget,
    HistogramWidget,
    ObjectiveControlWidget,
    ZParametersWidget,
)
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueComboBox,
)


class FMControlWidget(QWidget):
    """Widget for Fluorescence Microscope Control"""

    acquisition_finished_signal = pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope,
        viewer=None,  # optional: quad-view hosts run viewer-less
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
        self._acquisition_thread: Optional[FunctionWorker] = None
        self._acquisition_stop_event = threading.Event()
        self._current_acquisition_type: Optional[str] = None

        # FM canvas click context (quad-view): retained from the latest image so a
        # double-click can convert pixel -> stage without napari layer metadata
        self._fm_pixelsize: Optional[float] = None
        self._fm_image_shape: Optional[tuple] = None
        self._fm_canvas = None  # quad-view FM canvas we connect to (for teardown)
        self._fm_move_lock = threading.Lock()  # serialize canvas-driven stage moves

        # guards the debounced working-state autosave while applying a config
        self._loading_config = False

        self.initUI()
        self.connect_signals()
        self._update_acquisition_button_states()

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
        self.autofocusPanel = TitledPanel(
            "Autofocus Settings", content=self.autofocusWidget, collapsible=True
        )
        self.autofocusPanel.collapse()

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

        # Default orientation for fluorescence pose when adding a lamella
        self.label_default_orientation = QLabel("Default Orientation", self)
        self.comboBox_default_orientation = ValueComboBox(parent=self)
        self.comboBox_default_orientation.addItems(["SEM", "FM"])
        self.comboBox_default_orientation.setCurrentText(self.fm.default_orientation)
        self.comboBox_default_orientation.setToolTip(
            "Stage orientation used when computing the fluorescence pose for new lamellas"
        )

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
        self.progressText = QLabel("Acquisition Progress", self)
        self.progressBar_current_acquisition.hide()
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
        button_layout.addWidget(self.label_default_orientation, 0, 0)
        button_layout.addWidget(self.comboBox_default_orientation, 0, 1)
        button_layout.addWidget(self.checkBox_associate_with_lamella, 1, 0, 1, 2)
        button_layout.addWidget(self.pushButton_toggle_acquisition, 2, 0, 1, 2)
        button_layout.addWidget(self.pushButton_run_autofocus, 3, 0, 1, 2)
        button_layout.addWidget(self.pushButton_acquire_single_image, 4, 0)
        button_layout.addWidget(self.pushButton_acquire_zstack, 4, 1)
        button_layout.addWidget(self.pushButton_cancel_acquisition, 5, 0, 1, 2)
        button_layout.addWidget(self.progressText, 6, 0, 1, 2)
        button_layout.addWidget(self.progressBar_current_acquisition, 7, 0, 1, 2)

        # Main layout with scroll area and buttons
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    @property
    def _has_worker(self) -> bool:
        """Whether *this widget's* acquisition thread is still going.

        All that is left of the old `is_acquiring` property, which was
        `fm.is_acquiring or <my own worker>`. The microscope answers that now -- both
        halves of it, since this widget marks its own work through `set_acquiring` --
        and the only caller that wanted the local half is `cancel_acquisition`, which
        joins the thread and would hit `None.join()` on somebody else's run (FIB-513).
        """
        return bool(self._acquisition_thread and self._acquisition_thread.is_alive())

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
        self.comboBox_default_orientation.currentTextChanged.connect(
            lambda orientation: setattr(self.fm, "default_orientation", orientation)
        )

        # microscope signals
        self.fm.acquisition_signal.connect(self.update_image)
        self.acquisition_finished_signal.connect(self._on_acquisition_finished)
        self.fm.acquisition_progress_signal.connect(self._on_acquisition_progress)
        self.fm.acquiring_changed.connect(self._on_fm_acquiring_changed)

        # live parameter updates from channel list
        self.channelSettingsWidget.channel_field_changed.connect(
            self._on_channel_field_changed
        )

        # Debounced auto-save of the FM working state on any settings change, so
        # edits persist even if the app is force-killed before closeEvent runs.
        from superqt.utils import qdebounced

        self._autosave_fm_configuration = qdebounced(
            self.save_fm_configuration, timeout=1000
        )
        self.channelSettingsWidget.settings_changed.connect(
            self._on_fm_settings_changed
        )
        self.cameraWidget.settings_changed.connect(self._on_fm_settings_changed)
        self.autofocusWidget.settings_changed.connect(self._on_fm_settings_changed)
        self.zParametersWidget.settings_changed.connect(self._on_fm_settings_changed)
        self.comboBox_default_orientation.currentTextChanged.connect(
            self._on_fm_settings_changed
        )
        self.objectiveControlWidget.doubleSpinBox_focus_position.valueChanged.connect(
            self._on_fm_settings_changed
        )
        self.objectiveControlWidget.doubleSpinBox_objective_limit.valueChanged.connect(
            self._on_fm_settings_changed
        )

        # Route FM canvas double-click (-> relative move) and Shift+scroll (-> objective)
        # to the matplotlib canvas, replacing the napari viewer's double-click hook.
        controller = self._view_controller()
        if controller is not None:
            # bound methods (not lambdas) so they can be disconnected on teardown; the
            # canvas outlives this widget, so a leaked connection would drive a stale
            # stage move through a destroyed widget
            self._fm_canvas = controller.fm_canvas
            self._fm_canvas.canvas_double_clicked.connect(
                self._on_canvas_fm_double_click
            )
            self._fm_canvas.canvas_scrolled.connect(
                self.objectiveControlWidget._on_canvas_scroll
            )

        # Update checkbox text when lamella selection changes
        if self.parent_widget is not None and hasattr(
            self.parent_widget, "lamella_list"
        ):
            self.parent_widget.lamella_list.lamella_selected.connect(
                self._update_checkbox_text
            )

        # Initialize checkbox text
        self._update_checkbox_text()

    def _teardown_connections(self):
        """Disconnect every external signal/callback wired in connect_signals.

        Idempotent and fully guarded, so it can be called from ``closeEvent`` AND from
        the host's teardown (the ``deleteLater`` path, where ``closeEvent`` never
        fires). The quad-view ``fm_canvas`` outlives this widget, so its connections
        MUST come down or a stale double-click/scroll drives the stage or objective
        through a destroyed widget — and PyQt turns that into qFatal (FIB-329).

        The three ``fm`` signals are psygnal, not Qt, so nothing disconnects them for
        us when the widget dies: psygnal's weak ref is to the *Python* object, and
        ``deleteLater`` destroys the C++ one underneath it while that stays alive. The
        host tears down without stopping the acquisition first
        (``AutoLamellaUI.setup_experiment``), so a disconnect during a live run leaves
        a worker emitting into these slots (FIB-550).
        """
        for signal, slot in (
            (self.fm.acquisition_signal, self.update_image),
            (self.fm.acquisition_progress_signal, self._on_acquisition_progress),
            (self.fm.acquiring_changed, self._on_fm_acquiring_changed),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        if self._fm_canvas is not None:
            for signal, slot in (
                (
                    self._fm_canvas.canvas_double_clicked,
                    self._on_canvas_fm_double_click,
                ),
                (
                    self._fm_canvas.canvas_scrolled,
                    self.objectiveControlWidget._on_canvas_scroll,
                ),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._fm_canvas = None

        # Clear the FM panel's live chrome. The canvas outlives this widget, and the
        # host tears down without stopping the acquisition first
        # (`AutoLamellaUI.setup_experiment`), so the green border and badge would
        # otherwise stay on a panel with nothing left to drive them.
        controller = self._view_controller()
        if controller is not None:
            try:
                controller.set_fm_live(False)
            except RuntimeError:
                pass  # the view went first

        self.objectiveControlWidget.cleanup()

        if self.parent_widget is not None and hasattr(
            self.parent_widget, "lamella_list"
        ):
            try:
                self.parent_widget.lamella_list.lamella_selected.disconnect(
                    self._update_checkbox_text
                )
            except (TypeError, RuntimeError):
                pass

    def close_widget(self):
        """Disconnect external signals and close the widget."""
        self._teardown_connections()
        self.close()

    def _fm_relative_move_from_pixel(self, x, y, image_shape, pixelsize):
        """Convert an FM-image pixel (x=col, y=row) into a relative stage move that
        recentres that point, then move via ``fm_stable_move``.

        ``fm_stable_move`` undoes the display transform itself (``FluorescenceMicroscope
        ._transform``) and projects through the camera's axis tilt, so the move is
        foreshortening-correct and holds focus. Do NOT pre-apply the camera transform
        here — that would double-apply it (see #198 / FIB-133).

        The click keeps the sign convention the beam paths use (y up-positive, straight
        out of ``image_to_microscope_image_coordinates2``). There used to be a
        ``-point_clicked[1]`` here, calibrated at t=0; on a compustage the projection
        already reverses y between t=0 and t=-180, so a constant negation can only ever
        be right at one of them.
        """
        point_clicked = conversions.image_to_microscope_image_coordinates2(
            coord=Point(x=x, y=y),
            image_shape=image_shape,
            pixelsize=pixelsize,
        )
        logging.info(f"FM relative move: {point_clicked}")
        self.microscope.fm_stable_move(dx=point_clicked[0], dy=point_clicked[1])

    def _on_canvas_fm_double_click(self, x, y, modifiers):
        """Quad-view FM canvas double-click -> relative stage move that recentres the
        clicked point. Every guard runs here on the UI thread, so a rejected click never
        starts a thread; the move itself runs off-thread. Modifiers are ignored.
        """
        if self.is_acquisition_active:
            logging.info("Stage movement disabled during acquisition")
            return
        if not self.microscope.fm.has_valid_orientation():
            logging.info(
                "Stage must be in a valid FM orientation to move via FM "
                f"(current: {self.microscope.get_stage_orientation()})"
            )
            return
        shape, pixelsize = self._fm_image_shape, self._fm_pixelsize
        if shape is None or pixelsize is None:
            logging.info("No FM image available to move from")
            return
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            return  # click landed outside the image area
        # serialize: drop the click if a stage move is already running
        if not self._fm_move_lock.acquire(blocking=False):
            logging.info("FM stage move already in progress; ignoring click")
            return
        FunctionWorker(self._canvas_fm_move_worker, x, y, shape, pixelsize).start()

    def _canvas_fm_move_worker(self, x, y, shape, pixelsize):
        try:
            self._fm_relative_move_from_pixel(x, y, shape, pixelsize)
        except Exception as exc:
            logging.exception("FM stage move failed")
            notification_service.show_toast(
                f"FM stage move failed: {exc}", notification_type="error"
            )
        finally:
            self._fm_move_lock.release()

    def _on_channel_field_changed(self, channel, field: str, value) -> None:
        """Update a single microscope parameter during live acquisition."""
        if not self.fm.is_streaming:
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

        # hide progress bar when acquisition finishes
        self.progressBar_current_acquisition.hide()
        self.progressText.setText("")

        # refresh overview bbox, objective widget, and button states
        self.objectiveControlWidget.update_objective_position_labels()

        self._update_acquisition_button_states()

    @ensure_main_thread
    def _on_acquisition_progress(self, progress: dict):
        """Update progress bars on acquisition signal"""

        # Show progress bar when acquisition progress is updated
        if not self.progressBar_current_acquisition.isVisible():
            self.progressBar_current_acquisition.show()
        if not self.progressText.isVisible():
            self.progressText.show()

        progress_zlevels = progress.get("zlevel", None)
        progress_total_zlevels = progress.get("total_zlevels", None)
        channel_name = progress.get("channel", None)
        progress_state = progress.get("state", None)
        progress_task = progress.get("task", None)

        if progress_state == "moving":
            self.progressText.setText("Moving stage...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        if progress_state == "autofocus":
            self.progressText.setText("Running Autofocus...")
            self.progressBar_current_acquisition.setValue(0)
            self.progressBar_current_acquisition.setFormat("")

        if progress_state == "finished":
            self.progressBar_current_acquisition.hide()
            self.progressText.setText("Acquisition complete.")
            self.progressText.hide()
            return

        # set progress message
        if progress_task == "autofocus":
            # Handled before the channel branch, not inside it: a sweep with no channel
            # reports an empty name, which used to render "Acquiring  (1/1)..." and now
            # would leave the previous message sitting there instead.
            self.progressText.setText(
                f"Focusing on {channel_name}..." if channel_name else "Focusing..."
            )
        elif channel_name:
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
            if progress_task == "autofocus":
                # A focus sweep steps the objective through a search range. Calling
                # those positions "Z-level" names the z-stack, which is not running --
                # and says which pass, so a coarse sweep followed by a fine one does
                # not look like the same bar inexplicably starting over.
                total_passes = progress.get("total_passes", 1)
                which = (
                    f" · pass {progress.get('pass_index', 1)}/{total_passes}"
                    if total_passes > 1
                    else ""
                )
                label = f"Focus {progress_zlevels}/{progress_total_zlevels}{which}"
            else:
                label = f"Z-level {progress_zlevels}/{progress_total_zlevels}"
            self.progressBar_current_acquisition.setFormat(label)

    @ensure_main_thread
    def update_image(self, image: FluorescenceImage):

        if not isinstance(image, FluorescenceImage):
            return

        data = image.data
        if data.ndim == 4 and data.shape[0] == 1 and data.shape[1] == 1:
            data = data.squeeze(axis=(0, 1))  # squish to 2D

        # retain click context for the FM canvas double-click -> move. The canvas emits
        # data coords, so this is all the geometry that path needs — no layer metadata.
        self._fm_pixelsize = getattr(image.metadata, "pixel_size_x", None)
        self._fm_image_shape = data.shape[-2:]

        # Composite the acquired image onto the FM canvas via the controller. It handles
        # per-channel colour and blending, which the napari layer stack used to do.
        controller = self._view_controller()
        if controller is not None:
            controller.set_fm_image(image)

    def _view_controller(self):
        """The quad-view ``MicroscopeViewController`` on the main Microscope tab, or
        ``None`` otherwise. Chain: this widget's ``parent_widget`` is the
        ``AutoLamellaUI``, whose ``parent_widget`` is the main UI that owns the
        ``view_controller``."""
        autolamella_ui = self.parent_widget
        main_ui = getattr(autolamella_ui, "parent_widget", None)
        return getattr(main_ui, "view_controller", None)

    def _refuse_to_start(self, what: str) -> bool:
        """Whether *what* may not be started, having logged why. True means "do not".

        Both reasons in one place, asked by all three ways in. The running check used to
        be a union of three assembled here -- the widget's own worker, the live stream,
        and everybody else -- and `fm.is_acquiring` covers all of them now, this widget's
        own work included, since it marks that through `set_acquiring` like anyone else
        (FIB-513).

        The pose check used to live in `start_acquisition` alone, so an image, a z-stack
        or an autofocus sweep could be started from a pose the FM cannot see anything
        from. Nothing but the buttons stopped them, and the buttons are not the guard.
        """
        if self.fm.is_acquiring:
            reason = self.fm.acquiring_reason or "another acquisition"
            logging.warning(
                f"Cannot start {what}: the fluorescence microscope is in use ({reason})."
            )
            return True
        if not self.fm.has_valid_orientation():
            logging.warning(
                f"Cannot start {what}: the stage is not in a valid FM orientation "
                f"(currently {self.microscope.get_stage_orientation()})."
            )
            return True
        return False

    @ensure_main_thread
    def _on_fm_acquiring_changed(self, acquiring: bool) -> None:
        """Something started driving the FM, or stopped -- re-derive what is clickable.

        Marshalled, because the flag is cleared on whichever thread finished with the
        instrument, and this ends in `setEnabled` calls.
        """
        self._update_acquisition_button_states()

    def start_acquisition(self):
        """Start the fluorescence acquisition."""

        if self._refuse_to_start("live acquisition"):
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
        record_recent_channels(selected_channel_settings)
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
        if self._has_worker:
            logging.info(f"Cancelling {self._current_acquisition_type} acquisition")
            self._acquisition_stop_event.set()
            self._acquisition_thread.join(timeout=5)  # type: ignore[union-attr]

        logging.info("All acquisitions cancelled")

    def toggle_acquisition(self):
        """Toggle acquisition start/stop with F6 key."""
        if self.fm.is_streaming:
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

        if self._refuse_to_start("image acquisition"):
            return

        logging.info("Starting image acquisition")
        self._current_acquisition_type = "image"
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

        record_recent_channels(channel_settings)

        # Marked busy as late as possible: everything above can still abort -- the
        # filename step opens a modal -- and an abort leaving the flag set would lock
        # out the overview tab with nothing running.
        self.fm.set_acquiring(True, "z-stack" if z_parameters else "image acquisition")

        # Start acquisition thread
        self._acquisition_thread = FunctionWorker(
            self._image_acquistion_worker, channel_settings, z_parameters, filename
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
            self.fm.set_acquiring(False)
            self.acquisition_finished_signal.emit()

    def _update_acquisition_button_states(self):
        """Update acquisition button states and control widgets based on live acquisition or specific acquisition status."""
        # Every control below answers one of three questions, and the microscope owns
        # all three -- this used to assemble its own union per line, which is how the
        # objective panel came to ask the wrong one (FIB-513).
        acquiring = self.fm.is_acquiring  # anything running, here or anywhere
        streaming = self.fm.is_streaming  # the live view specifically
        interactive = self.fm.is_interactive  # may the user drive the hardware

        # A fourth input, not a fourth question: where the stage is does not change what
        # is running, but it does stop new work being started -- `start_acquisition`
        # refuses on it too -- and stops the hardware being driven by hand.
        #
        # Folded in rather than returned early. The early return here disabled three
        # buttons and skipped every line below, so the objective panel and the channel
        # settings kept whatever they were last set to, including enabled during
        # somebody else's tileset. It also missed two of the buttons it claimed to
        # cover. Dead as shipped -- `ALLOW_UNKNOWN_ORIENTATIONS` answers this yes
        # unconditionally -- which is why nobody noticed.
        posed = self.microscope.fm.has_valid_orientation()
        may_start = posed and not acquiring
        may_touch = posed and interactive

        # Deliberately not gated on the pose, unlike everything else here: taking Cancel
        # away from a run that is under way leaves no way to stop it, and a stage that
        # has wandered somewhere unrecognised is exactly when you want it. The old early
        # return disabled it.
        self.pushButton_cancel_acquisition.setVisible(self.is_acquisition_active)

        # Update toggle acquisition button text based on state
        self.pushButton_toggle_acquisition.setText(
            "Stop Acquisition" if streaming else "Start Acquisition"
        )
        # Clickable while the live stream *is* what is running -- it is the only way to
        # stop one, and stopping is allowed from any pose -- but not while anything else
        # has the microscope. This is the button FIB-441 was about: it stayed live
        # through an overview run, and starting a stream mid-tileset changes the
        # settings the remaining tiles are acquired at.
        self.pushButton_toggle_acquisition.setEnabled(streaming or may_start)

        # Anything that starts new work waits for whatever is running to finish.
        for button in (
            self.pushButton_acquire_single_image,
            self.pushButton_acquire_zstack,
            self.pushButton_run_autofocus,
            self.objectiveControlWidget.pushButton_insert_objective,
            self.objectiveControlWidget.pushButton_move_to_focus,
            self.objectiveControlWidget.pushButton_retract_objective,
        ):
            button.setEnabled(may_start)

        # Parameters for the *next* run, so they follow the same rule.
        self.zParametersWidget.setEnabled(may_start)

        # The FM panel's own green border + "LIVE" badge, the chrome the beam panels have
        # had all along. Driven from `is_streaming` rather than inferred from a button or
        # this widget's own worker: the stream is the microscope's state, and another tab
        # can start or stop one (FIB-513). Costs nothing when there is no quad view --
        # the standalone hosts have no controller (FIB-596).
        controller = self._view_controller()
        if controller is not None:
            controller.set_fm_live(streaming)

        # Hardware the user drives by hand. Live during a stream -- focusing and tuning
        # exposure while watching is the point of them -- and not while a tileset,
        # z-stack or autofocus sweep is driving the same hardware itself. The objective
        # panel used to ask `is_acquisition_active`, which is only *this* widget's work,
        # so another tab's tileset left the position spinbox live against a runner that
        # was stepping the objective per z-level.
        self.objectiveControlWidget.setEnabled(may_touch)
        self.channelSettingsWidget.setEnabled(may_touch)

    def run_autofocus(self):
        """Start threaded auto-focus using the current channel settings and Z parameters."""
        if self._refuse_to_start("auto-focus"):
            return

        logging.info("Starting auto-focus")
        self._current_acquisition_type = "autofocus"
        self._update_acquisition_button_states()
        self._acquisition_stop_event.clear()

        # Get current settings
        settings = self._get_current_settings()
        channel_settings = settings["selected_channel_settings"]
        autofocus_settings = settings["autofocus_settings"]

        self.fm.set_acquiring(True, "auto-focus")

        # Start auto-focus thread
        self._acquisition_thread = FunctionWorker(
            self._autofocus_worker, channel_settings, autofocus_settings
        )
        self._acquisition_thread.start()

    def _autofocus_worker(
        self, channel_settings: ChannelSettings, autofocus_settings: AutoFocusSettings
    ):
        """Worker thread for auto-focus."""
        try:
            ranges = ", ".join(
                f"{p.search_range * 1e6:.1f}µm/{p.step_size * 1e6:.1f}µm"
                for p in autofocus_settings.passes
                if p.enabled
            )
            logging.info(
                f"Running auto-focus: {len(autofocus_settings.passes)} pass(es) [{ranges}], "
                f"method={autofocus_settings.method.value}"
            )

            from fibsem.fm.calibration import plot_autofocus, run_coarse_fine_autofocus

            result = run_coarse_fine_autofocus(
                self.fm,
                autofocus_settings=autofocus_settings,
                channel_settings=channel_settings,
                stop_event=self._acquisition_stop_event,
            )

            if result is not None:
                plot_autofocus(result)

            if result is None or self._acquisition_stop_event.is_set():
                logging.info("Auto-focus was cancelled")
                return

            logging.info(
                f"Auto-focus completed successfully. Best focus: {result.working_distance * 1e6:.1f} μm"
            )

        except Exception as e:
            logging.error(f"Auto-focus failed: {e}")
        finally:
            self.fm.set_acquiring(False)
            self.acquisition_finished_signal.emit()

    def closeEvent(self, event: QEvent):
        """Handle the close event to stop acquisition."""
        logging.info(
            "Closing FluorescenceControlWidget, stopping acquisition if running."
        )

        if self.microscope.fm is None:
            event.accept()
            return

        # Disconnect every external signal/callback (idempotent), then stop live
        self._teardown_connections()
        if self.microscope.fm.is_streaming:
            try:
                self.microscope.fm.stop_acquisition()
            except Exception as e:
                logging.error(f"Error stopping acquisition: {e}")
            finally:
                logging.warning("Acquisition stopped due to widget close.")

        if self.parent_widget:
            self.parent_widget.setEnabled(True)

        event.accept()

    def import_fm_configuration(self):
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

    def _build_fluorescence_configuration(self) -> FluorescenceConfiguration:
        """Build a complete FluorescenceConfiguration from the current UI settings."""
        settings = self._get_current_settings()
        return FluorescenceConfiguration(
            channel_settings=settings["channel_settings"],
            z_parameters=settings["z_parameters"],
            overview_parameters=OverviewParameters(),
            autofocus_settings=settings["autofocus_settings"],
            camera_settings=settings["camera_settings"],
            focus_position=self.fm.objective.focus_position,
            limit_position=self.fm.objective.limit_position,
            default_orientation=self.fm.default_orientation,
        )

    def save_fm_configuration(self) -> None:
        """Persist the current FM configuration as the auto-loaded working state."""
        if self.microscope.fm is None:
            return
        from fibsem.fm.config import save_fm_configuration

        try:
            save_fm_configuration(self._build_fluorescence_configuration())
        except Exception as e:
            logging.warning(f"Could not save FM working state: {e}")

    def _on_fm_settings_changed(self, *args) -> None:
        """Trigger a debounced working-state save when an FM setting changes."""
        if self._loading_config:
            return
        self._autosave_fm_configuration()

    def _apply_fluorescence_configuration(self, config: FluorescenceConfiguration):
        """Apply a FluorescenceConfiguration to the widget settings."""
        self._loading_config = True
        try:
            self.channelSettingsWidget.channel_settings = config.channel_settings
            self.zParametersWidget.z_parameters = config.z_parameters
            if config.camera_settings is not None:
                self.cameraWidget.camera_settings = config.camera_settings
            if config.autofocus_settings is not None:
                self.autofocusWidget.set_autofocus_settings(config.autofocus_settings)
            if config.focus_position is not None:
                self.objectiveControlWidget._set_focus_position(config.focus_position)
            if config.limit_position:
                self.objectiveControlWidget._set_limit_position(config.limit_position)
            self.comboBox_default_orientation.setCurrentText(config.default_orientation)
        finally:
            self._loading_config = False

    def export_fm_configuration(self):
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

            # Export configuration (incl. camera transform + objective limit position)
            self._build_fluorescence_configuration().export(filename)

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
    viewer=None,
    parent: Optional[QWidget] = None,
) -> FMControlWidget:
    """Create the FMControlWidget with a demo microscope."""

    widget = FMControlWidget(microscope=microscope, viewer=viewer, parent=parent)
    return widget


class _DemoHost(QWidget):
    """Minimal stand-in for the AutoLamella chain this widget resolves through.

    ``FMControlWidget._view_controller`` walks parent -> ``parent_widget`` -> the UI
    holding ``view_controller``, so the harness has to present that shape or the FM
    canvas stays empty and the demo looks broken.
    """

    def __init__(self):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        super().__init__()
        self.view_controller = MicroscopeViewController(parent=self)
        self.parent_widget = self


def main():
    """Main function to run the widget standalone."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QSplitter

    from fibsem.ui.stylesheets import NAPARI_STYLE

    microscope, settings = utils.setup_session()

    if microscope.fm is None:
        logging.error(
            "FluorescenceMicroscope is not initialized. Cannot create FMControlWidget."
        )
        raise RuntimeError("FluorescenceMicroscope is not initialized.")

    # Ensure compustage configuration
    assert microscope.system.stage.shuttle_pre_tilt == 0
    assert microscope.stage_is_compustage is True

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")

    host = _DemoHost()
    widget = create_widget(microscope, parent=host)

    window = QSplitter(Qt.Horizontal)
    window.setStyleSheet(NAPARI_STYLE)
    window.addWidget(host.view_controller.widget)
    window.addWidget(widget)
    window.setSizes([720, 460])
    window.resize(1280, 800)
    window.show()
    app.exec_()
    return


if __name__ == "__main__":
    main()
