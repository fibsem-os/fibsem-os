import logging
from typing import List, Optional

from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QEvent, pyqtSignal
from superqt import ensure_main_thread
from fibsem.ui.icon import fibsem_icon

from fibsem import acquire, utils
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
from fibsem.ui import notification_service
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, _SpinnerLabel
from fibsem.ui.widgets.dual_beam_widget import FibsemDualBeamWidget
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget


class FibsemImageSettingsWidget(QtWidgets.QWidget):
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

        self.parent = parent
        self.microscope = microscope
        # viewer-less: the quad-view controller is the display (None when no host viewer)
        self.viewer = getattr(parent, "viewer", None)

        # generate initial blank images
        self.eb_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)
        self.ib_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)

        self._overlay_edited_wired = False  # quad-view: subscribed to controller.overlay_edited
        self.is_acquiring: bool = False

        self._setup_ui()
        self.setup_connections()

        if image_settings is not None:
            self.image_settings = image_settings
            self._set_image_settings_to_ui(image_settings)
            self.update_ui_saving_settings()

        # NOTE: the canvases are intentionally left empty ("No image") on connect — no blank
        # placeholder is seeded. eb_image / ib_image remain as internal fallbacks and appear
        # once a real acquisition arrives (sem/fib_acquisition_signal -> _on_acquire).

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

        # --- Outer layout + scroll area (previously from generated setupUi) ---
        self.gridLayout = QtWidgets.QGridLayout(self)

        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)

        self.gridLayout_2.addItem(
            QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
            10, 0, 1, 2,
        )

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 1, 0, 1, 2)

        self.pushButton_start_acquisition = QtWidgets.QPushButton("Start Acquisition")
        self.pushButton_acquire_fib_image = QtWidgets.QPushButton("Acquire FIB Image")
        self.pushButton_acquire_sem_image = QtWidgets.QPushButton("Acquire SEM Image")
        self.pushButton_take_all_images = QtWidgets.QPushButton("Acquire All Images")
        self.pushButton_run_autocontrast = QtWidgets.QPushButton("Run AutoContrast")
        self.pushButton_run_autofocus = QtWidgets.QPushButton("Run AutoFocus")
        self.checkBox_save_with_selected_lamella = QtWidgets.QCheckBox("Save with Selected Lamella")

        # Spinner (hidden until acquiring / auto-function running)
        self._spinner = _SpinnerLabel(parent=self)
        self._spinner.setVisible(False)

        # Spinner row — right-aligned alongside the lamella checkbox. (Scalebar /
        # crosshair toggles live on the quad-view canvas, not here.)
        tools_row = QtWidgets.QWidget()
        tools_layout = QtWidgets.QHBoxLayout(tools_row)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(4)
        tools_layout.addWidget(self.checkBox_save_with_selected_lamella)
        tools_layout.addStretch()
        tools_layout.addWidget(self._spinner)

        # add buttons to layout — scroll area at row 1, tools row at row 2, buttons below
        self.gridLayout.addWidget(tools_row, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_run_autocontrast, 3, 0)
        self.gridLayout.addWidget(self.pushButton_run_autofocus, 3, 1)
        self.gridLayout.addWidget(self.pushButton_start_acquisition, 4, 0, 1, 2)
        self.gridLayout.addWidget(self.pushButton_acquire_sem_image, 5, 0)
        self.gridLayout.addWidget(self.pushButton_acquire_fib_image, 5, 1)
        self.gridLayout.addWidget(self.pushButton_take_all_images, 6, 0, 1, 2)

        # --- ImageSettingsWidget (resolution, dwell, hfw, integration, save) ---
        self.image_settings_widget = ImageSettingsWidget(
            parent=self.scrollAreaWidgetContents,
            show_save=True,
            show_advanced=False,
        )
        self.image_settings_widget.set_show_advanced_button(False)

        self._btn_advanced_image = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        self.image_group = TitledPanel("Image", content=self.image_settings_widget)
        self.image_group.add_header_widget(self._btn_advanced_image)
        self.image_group._btn_collapse.setChecked(True)  # expanded by default

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
            self.parent.lamella_list.lamella_selected.connect(self._on_current_lamella_changed)
        except Exception as e:
            logging.debug(f"Error connecting to lamella selection changes: {e}")

        # image advanced toggle
        self._btn_advanced_image.toggled.connect(self.image_settings_widget.set_show_advanced)

        # signals
        self.acquisition_progress_signal.connect(self.handle_acquisition_progress_update)

        # auto functions
        self.pushButton_run_autocontrast.clicked.connect(self.run_autocontrast)
        self.pushButton_run_autofocus.clicked.connect(self.run_autofocus)

        # set ui stylesheets
        self.pushButton_acquire_sem_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_acquire_fib_image.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_take_all_images.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_run_autocontrast.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_run_autofocus.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        self.pushButton_run_autocontrast.setIcon(fibsem_icon("mdi:contrast-circle", color=stylesheets.GRAY_ICON_COLOR))
        self.pushButton_run_autofocus.setIcon(fibsem_icon("mdi:image-filter-center-focus", color=stylesheets.GRAY_ICON_COLOR))

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

        # Working-distance Shift+scroll nudge on the SEM/FIB canvases (mirrors the FM
        # objective wheel). Store the bound-method connections for teardown — the canvases
        # outlive this widget, so a leaked connection would drive a dead slot on reconnect.
        self._wd_scroll_connections: List[tuple] = []
        controller = self._view_controller()
        if controller is not None:
            for beam, canvas in (
                (BeamType.ELECTRON, controller.sem_canvas),
                (BeamType.ION, controller.fib_canvas),
            ):
                beam_widget = (
                    self.dual_beam_widget.sem_widget
                    if beam is BeamType.ELECTRON
                    else self.dual_beam_widget.fib_widget
                )
                slot = beam_widget.beam_settings_widget._on_canvas_scroll
                canvas.canvas_scrolled.connect(slot)
                self._wd_scroll_connections.append((canvas, slot))

        # Two-way sync between the quad-view selection and the SEM/FIB beam radios:
        # selecting a canvas checks its radio (revealing that beam's settings), and checking
        # a radio selects its canvas. The controller is persistent, so store the connections
        # for teardown (else a reconnect leaks onto a dead widget). Loop-safe: set_selected
        # no-ops on re-select and setChecked emits no toggled when already set.
        self._view_sync_connections: List[tuple] = []
        if controller is not None:
            controller.view_selected.connect(self._on_view_selected)
            self._view_sync_connections.append((controller.view_selected, self._on_view_selected))
            self._on_view_selected(controller.selected_view)  # align radio to current selection
            self.dual_beam_widget.sem_radio.toggled.connect(self._on_beam_radio_toggled)
            self._view_sync_connections.append(
                (self.dual_beam_widget.sem_radio.toggled, self._on_beam_radio_toggled)
            )

    @ensure_main_thread
    def _on_acquire(self, image: FibsemImage):
        """Update the viewer from the main thread"""
        try:
            if image.metadata is None:
                raise ValueError("Image metadata is None, cannot update the display without beam type information.")

            # update image references
            if self.microscope.is_acquiring:
                if image.metadata.beam_type is BeamType.ELECTRON:
                    self.eb_image = image
                elif image.metadata.beam_type is BeamType.ION:
                    self.ib_image = image
            # Display on the quad-view canvas
            controller = self._view_controller()
            if controller is not None:
                controller.set_image(image.metadata.beam_type, image)
        except Exception as e:
            logging.error(f"Error updating image: {e}")

        self.viewer_update_signal.emit()  # notifies the milling widget of a new image

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None when inactive.

        Two hosts expose it: standalone ``FibsemUI`` (the direct parent holds
        ``view_controller``) and AutoLamella (parent → ``AutoLamellaUI`` →
        ``parent_widget`` → the main window). Check the direct parent first, then
        the AutoLamella chain; ``None`` if neither holds one.
        """
        controller = getattr(self.parent, "view_controller", None)
        if controller is not None:
            return controller
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _on_view_selected(self, key) -> None:
        """Quad-view selection changed -> check the matching beam radio (view -> radio).
        Ignores ``"fm"`` / ``None`` (no beam radio)."""
        if key is BeamType.ELECTRON:
            self.dual_beam_widget.sem_radio.setChecked(True)
        elif key is BeamType.ION:
            self.dual_beam_widget.fib_radio.setChecked(True)

    def _on_beam_radio_toggled(self, sem_checked: bool) -> None:
        """Beam radio toggled -> select the matching canvas in the quad view (radio -> view)."""
        controller = self._view_controller()
        if controller is None:
            return
        widget = getattr(controller, "widget", None)
        if widget is not None and hasattr(widget, "set_selected"):
            widget.set_selected(BeamType.ELECTRON if sem_checked else BeamType.ION)

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
            self._spinner.setVisible(False)
            return

        # disable other buttons while live acquisition is running
        for btn in self.acquisition_buttons:
            btn.setEnabled(False)
        for btn in [self.pushButton_run_autocontrast, self.pushButton_run_autofocus]:
            btn.setEnabled(False)
        self.image_group.setEnabled(False)
        self._spinner.start()
        self._spinner.setVisible(True)

        beam_type = self.dual_beam_widget.beam_type
        self.microscope.start_acquisition(beam_type)
        self.pushButton_start_acquisition.setText("Stop Acquisition")
        self.pushButton_start_acquisition.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)

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
        from fibsem.autofunctions.autofocus import run_auto_focus, AutoFocusSettings, FocusSweepPass
        settings = AutoFocusSettings(
            method="tenengrad",
            passes=[
                FocusSweepPass(search_range=1e-3, step_size=100e-6),
                FocusSweepPass(search_range=100e-6, step_size=10e-6),
            ],
            reduced_area=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
            use_autocontrast=True)
        result = run_auto_focus(
            self.microscope,
            beam_type=beam_type,
            hfw=self.microscope.get_field_of_view(beam_type),
            settings=settings,
        )

    def _on_auto_function_finished(self, name: str, beam_type: BeamType) -> None:
        self._toggle_interactions(enable=True)
        beam_widget = self.dual_beam_widget.sem_widget if beam_type is BeamType.ELECTRON else self.dual_beam_widget.fib_widget
        beam_widget.sync_from_microscope()
        if name == "AutoFocus":
            wd = beam_widget.beam_settings_widget.working_distance_spinbox.value()
            notification_service.show_toast(f"AutoFocus Complete. Best WD: {wd:.2f}mm")
        if name == "AutoContrast":
            notification_service.show_toast("AutoContrast Complete.")

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
        # If save_with_selected_lamella is checked, ensure save_image is also checked
        if self.checkBox_save_with_selected_lamella.isChecked():
            self.image_settings_widget.save_image_check.blockSignals(True)
            self.image_settings_widget.save_image_check.setChecked(True)
            self.image_settings_widget.save_image_check.blockSignals(False)

        save_image = self.image_settings_widget.save_image_check.isChecked()

        save_with_lamella = self.checkBox_save_with_selected_lamella.isChecked()
        self.image_settings_widget.path_edit.setEnabled(save_image and not save_with_lamella)

        if save_with_lamella:
            try:
                lamella = self.parent.lamella_list.selected_lamella
                self.image_settings_widget.path_edit.setText(str(lamella.path))
            except Exception as e:
                logging.debug(f"Error setting image path from selected lamella: {e}")
        else:
            if hasattr(self.parent, "experiment") and self.parent.experiment is not None:
                self.image_settings_widget.path_edit.setText(str(self.parent.experiment.path))

    def _on_current_lamella_changed(self, lamella):
        """Update the image path when the selected lamella changes"""
        try:
            if self.checkBox_save_with_selected_lamella.isChecked() and lamella is not None:
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
            self._spinner.setVisible(False)
        elif imaging:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
                btn.setText("Acquiring...")
            self._spinner.start()
            self._spinner.setVisible(True)
        else:
            self.pushButton_take_all_images.setText("Acquire All Images")
            self._spinner.start()
            self._spinner.setVisible(True)

    def handle_acquisition_progress_update(self, ddict: dict):
        """Handle the acquisition progress update"""
        logging.debug(f"Acquisition Progress Update: {ddict}")

        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            notification_service.show_toast(msg)

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

    def closeEvent(self, event: QEvent):
        self._teardown_connections()
        event.accept()

    def _teardown_connections(self) -> None:
        """Drop the controller subscription (idempotent). Called from ``closeEvent`` AND
        the AutoLamella teardown path — ``deleteLater`` fires neither ``closeEvent`` nor
        ``close``, so without this an alignment edit after a reconnect would drive the
        dead widget's slot on the persistent controller."""
        if self._overlay_edited_wired:
            controller = self._view_controller()
            if controller is not None:
                try:
                    controller.overlay_edited.disconnect(self._on_controller_overlay_edited)
                except (TypeError, RuntimeError):
                    pass
            self._overlay_edited_wired = False

        # WD Shift+scroll: disconnect from the (persistent) canvases and cancel any pending
        # debounced move so it can't fire on a torn-down beam-settings widget.
        for canvas, slot in getattr(self, "_wd_scroll_connections", []):
            try:
                canvas.canvas_scrolled.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._wd_scroll_connections = []
        for beam_widget in (self.dual_beam_widget.sem_widget, self.dual_beam_widget.fib_widget):
            try:
                beam_widget.beam_settings_widget._execute_wd_wheel_move.cancel()
            except Exception:
                pass

        # View <-> beam-radio sync: drop both directions (controller is persistent).
        for signal, slot in getattr(self, "_view_sync_connections", []):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._view_sync_connections = []

    def _ensure_overlay_edited_wiring(self, controller) -> None:
        """Subscribe (once) to the controller's overlay-edit signal so a user drag of
        the alignment area drives the existing validation/workflow path unchanged."""
        if self._overlay_edited_wired:
            return
        controller.overlay_edited.connect(self._on_controller_overlay_edited)
        self.alignment_area_updated.connect(self._on_alignment_area_updated)
        self._overlay_edited_wired = True

    def _on_controller_overlay_edited(self, beam, overlay_id, value) -> None:
        """Route a committed alignment-area edit into the existing signal/workflow.

        The authoritative value lives in the model (``controller.alignment_area``);
        this just drives validation, so it forwards the edit without caching it."""
        if overlay_id == "alignment":
            self.alignment_area_updated.emit(value)

    def hide_alignment_area(self):
        """Hide the alignment area but keep its value.

        The workflow reads ``get_alignment_area()`` straight after sending its
        "clear" (in ``update_alignment_area_ui``), so the rect must survive — the
        overlay is hidden but the model keeps its value.
        """
        controller = self._view_controller()
        if controller is not None:
            controller.set_alignment_edit(BeamType.ION, None, editing=False)

    def clear_alignment_area(self):
        """Remove the alignment area entirely (true teardown; no value retained).

        No caller yet — defined for explicit teardown (e.g. lamella deselect). The
        workflow's mid-flow "clear" uses :meth:`hide_alignment_area` instead.
        """
        controller = self._view_controller()
        if controller is not None:
            controller.arm_overlay(BeamType.ION, None)
            controller.remove_overlay(BeamType.ION, "alignment")

    def toggle_alignment_area(self, reduced_area: FibsemRectangle, editable: bool = True):
        """Toggle the alignment area layer to selection mode, and display the alignment area."""
        self.set_alignment_layer(reduced_area, editable=editable)

    def set_alignment_layer(
        self,
        reduced_area: FibsemRectangle = FibsemRectangle(0.25, 0.25, 0.5, 0.5),
        editable: bool = True,
    ):
        """Display the editable alignment area on the quad-view FIB canvas."""
        controller = self._view_controller()
        if controller is not None:
            self._ensure_overlay_edited_wiring(controller)
            # editing wins over the milling read-only display + owns FIB input
            controller.set_alignment_edit(BeamType.ION, reduced_area, editing=editable)

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
        """Get the alignment area from the model (quad-view)."""
        controller = self._view_controller()
        if controller is not None:
            return controller.alignment_area(BeamType.ION)
        return None
