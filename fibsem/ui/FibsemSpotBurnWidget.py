from __future__ import annotations

import logging
import threading

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.imaging.spot import SpotBurnSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.custom_widgets import ValueComboBox, ValueSpinBox
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate
from fibsem.ui.widgets.spot_burn_coordinates_widget import SpotBurnCoordinatesWidget
from fibsem.utils import format_value

DEFAULT_BEAM_CURRENT = 60e-12  # 60 pA


def build_spot_burn_progress_update(ddict: dict) -> ProgressUpdate:
    """Map a spot-burn progress dict (see run_spot_burn) to a ProgressUpdate.

    Shared by the spot burn widget and the main-window status bar so both render
    progress with identical text and formatting.
    """
    if ddict.get("finished"):
        if ddict.get("error"):
            return ProgressUpdate.failed("Spot burn failed")
        return ProgressUpdate.done()
    return ProgressUpdate.combined(
        current=ddict.get("current_point", 0),
        total=ddict.get("total_points", 0),
        remaining_seconds=ddict.get("total_remaining_time", 0.0),
        total_seconds=ddict.get("total_estimated_time", 0.0),
        message="Burning spots",
    )


class FibsemSpotBurnWidget(QWidget):
    """Live spot-burn tab: place coordinates + set current/exposure + run.

    The *coordinate* half is the shared :class:`SpotBurnCoordinatesWidget` (canvas points
    overlay + list); this widget adds the beam-current/exposure form and the run / cancel
    / progress machinery. Everything is one :class:`SpotBurnSettings`: the editor writes
    ``coordinates``, the form writes current/exposure, and the burn runs the settings.

    Progress is driven by ``microscope.spot_burn_progress_signal`` (shared with the
    status bar). The burn runs directly from the widget's button, or — in a supervised
    workflow — via ``start_spot_burn_signal`` (mirrors milling's task-orchestrated hook).
    """

    # emitted by the workflow task to trigger a burn (mirrors milling's start_milling_signal)
    start_spot_burn_signal = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.parent = parent
        self.microscope: FibsemMicroscope = parent.microscope
        self.worker = None  # FunctionWorker while a burn is running
        self.stop_event: threading.Event = threading.Event()
        self._is_burning = False  # guards the Run/Cancel button while a burn runs
        self._workflow_mode = False  # supervised workflow hides the widget's own button

        self._build_ui()
        self._setup_connections()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # coordinate editor (shared canvas overlay + list)
        self.coord_editor = SpotBurnCoordinatesWidget(
            controller=self._view_controller(), beam=BeamType.ION, parent=self,
        )
        layout.addWidget(self.coord_editor)

        # beam current + exposure
        beam_currents = self.microscope.get_available_values("current", BeamType.ION)
        closest = min(beam_currents, key=lambda x: abs(x - DEFAULT_BEAM_CURRENT))
        self.comboBox_beam_current = ValueComboBox(
            items=beam_currents, value=closest, unit="A", decimals=1,
        )
        self.doubleSpinBox_exposure_time = ValueSpinBox(
            suffix="s", minimum=0.1, maximum=60, decimals=3,
        )
        self.doubleSpinBox_exposure_time.setValue(10)
        form = QFormLayout()
        form.addRow("Beam Current", self.comboBox_beam_current)
        form.addRow("Exposure Time", self.doubleSpinBox_exposure_time)
        layout.addLayout(form)

        # info
        self.label_information = QLabel("")
        self.label_information.setWordWrap(True)
        layout.addWidget(self.label_information)

        # progress — the shared FibsemProgressWidget (renders identically to the status bar)
        self.progress_widget = FibsemProgressWidget(self)
        layout.addWidget(self.progress_widget)

        # run button + workflow hint. Exactly one is visible at a time
        # (_update_run_button_visibility): outside a workflow the button is used directly;
        # in a supervised workflow the burn is run from the workflow control, so the button
        # is hidden (except as "Cancel" mid-burn) and the hint is shown instead.
        self.pushButton_run_spot_burn = QPushButton("Run Spot Burn")
        layout.addWidget(self.pushButton_run_spot_burn)

        self.label_workflow_hint = QLabel(
            "Use the 'Run Spot Burn' button in the workflow controls to burn the selected points."
        )
        self.label_workflow_hint.setWordWrap(True)
        self.label_workflow_hint.setStyleSheet("color: gray; font-style: italic;")
        self.label_workflow_hint.setVisible(False)
        layout.addWidget(self.label_workflow_hint)

        layout.addStretch()

    def _setup_connections(self) -> None:
        self.comboBox_beam_current.currentIndexChanged.connect(self._refresh_info)
        self.doubleSpinBox_exposure_time.valueChanged.connect(self._refresh_info)

        # run button
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_run_spot_burn.setEnabled(False)

        # the workflow task drives the burn via start_spot_burn_signal (mirrors milling).
        # BlockingQueuedConnection: emit() returns only after run_spot_burn_worker has run,
        # so the burn is then either in progress (is_burning=True) or was refused (no
        # in-bounds points), in which case the task re-prompts.
        self.start_spot_burn_signal.connect(
            self.run_spot_burn_worker, Qt.BlockingQueuedConnection  # type: ignore
        )

        # coordinate signal
        self.coord_editor.settings_changed.connect(self._refresh_info)

        # progress is driven by the microscope's spot_burn_progress_signal (shared with the
        # status bar). It fires on the worker thread, so _update_progress_bar is marshalled
        # to the GUI thread via @ensure_main_thread.
        self.microscope.spot_burn_progress_signal.connect(self._update_progress_bar)

        # re-feed the FIB image shape to the editor on each acquisition
        iw = self._image_widget()
        if iw is not None:
            try:
                iw.viewer_update_signal.connect(self._feed_image_shape)
            except Exception:
                pass

        self._refresh_info()

    # ------------------------------------------------------------------
    # Controller / image plumbing
    # ------------------------------------------------------------------

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None."""
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _image_widget(self):
        return getattr(self.parent, "image_widget", None)

    def _feed_image_shape(self, *args) -> None:
        """Push the current FIB image shape to the editor (for 0-1 <-> px conversion)."""
        iw = self._image_widget()
        img = getattr(iw, "ib_image", None) if iw is not None else None
        if img is not None:
            self.coord_editor.set_image_shape(img.data.shape[:2])

    # ------------------------------------------------------------------
    # Workflow mode (button/hint visibility) + teardown
    # ------------------------------------------------------------------

    @property
    def is_burning(self) -> bool:
        """Whether a spot burn is currently running."""
        return self._is_burning

    def set_workflow_mode(self, active: bool) -> None:
        """Toggle workflow mode.

        During a supervised workflow the burn is task-orchestrated (like milling), so the
        widget's own Burn button is hidden and a hint points the user to the workflow
        "Run Spot Burn" control. The button still appears as "Cancel" while a burn is in
        progress so the user can cancel locally. Outside the workflow it is used directly.
        """
        self._workflow_mode = active
        self._update_run_button_visibility()

    def _update_run_button_visibility(self) -> None:
        """Show/hide the run button and workflow hint for the current mode + burn state."""
        if self._workflow_mode:
            # idle: hidden (run via the workflow control); burning: shown as "Cancel"
            self.pushButton_run_spot_burn.setVisible(self._is_burning)
            self.label_workflow_hint.setVisible(not self._is_burning)
        else:
            self.pushButton_run_spot_burn.setVisible(True)
            self.label_workflow_hint.setVisible(False)

    def disconnect_signals(self) -> None:
        """Disconnect from the microscope's progress signal before the widget is destroyed.

        The microscope outlives this widget; without this, the connection would leak the
        widget and fire _update_progress_bar on a deleted object.
        """
        try:
            self.microscope.spot_burn_progress_signal.disconnect(self._update_progress_bar)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Activation lifecycle (called by the workflow via AutoLamellaUI)
    # ------------------------------------------------------------------

    def set_active(self) -> None:
        """Activate: feed the image shape and arm the overlay."""
        self._feed_image_shape()
        self.coord_editor.set_active(True)
        self._refresh_info()

    def set_inactive(self) -> None:
        """Deactivate: disarm the overlay."""
        self.coord_editor.set_active(False)

    def clear_points_layer(self) -> None:
        """Clear the coordinates + overlay (called when the task exits)."""
        self.coord_editor.set_settings(SpotBurnSettings())
        self._refresh_info()

    # ------------------------------------------------------------------
    # Workflow API (SpotBurnSettings in / out)
    # ------------------------------------------------------------------

    def set_settings(self, settings: SpotBurnSettings) -> None:
        """Apply a settings payload (current / exposure / coordinates)."""
        # keep an off-grid (protocol) current exactly selectable so an untouched value
        # round-trips losslessly — otherwise the closest-match snap would rewrite the config
        if self.comboBox_beam_current.findData(settings.milling_current) == -1:
            self.comboBox_beam_current.addItem(
                format_value(settings.milling_current, unit="A", precision=1),
                settings.milling_current,
            )
        self.comboBox_beam_current.set_value(settings.milling_current)
        self.doubleSpinBox_exposure_time.setValue(settings.exposure_time)

        self._feed_image_shape()
        # the editor only owns coordinates; current/exposure live on the form
        self.coord_editor.set_settings(SpotBurnSettings(coordinates=list(settings.coordinates)))
        self._refresh_info()  # set_settings is programmatic (no settings_changed)

    def get_settings(self) -> SpotBurnSettings:
        """The run payload — coordinates from the editor, current/exposure from the form."""
        return SpotBurnSettings(
            coordinates=self.get_coordinates(),
            milling_current=self.comboBox_beam_current.value(),
            exposure_time=self.doubleSpinBox_exposure_time.value(),
        )

    def get_coordinates(self) -> list:
        """Current spot positions (normalised 0-1), filtered to image bounds."""
        coords = self.coord_editor.get_settings().coordinates
        return [pt for pt in coords if 0 <= pt.x <= 1 and 0 <= pt.y <= 1]

    # ------------------------------------------------------------------
    # Info / run
    # ------------------------------------------------------------------

    def _refresh_info(self, *args) -> None:
        n = len(self.coord_editor.get_settings().coordinates)
        exposure = self.doubleSpinBox_exposure_time.value()
        # while a burn runs the button is "Cancel" and must stay enabled — don't let a
        # mid-burn coordinate edit (n -> 0) disable it, or the burn can't be cancelled
        if not self._is_burning:
            self.pushButton_run_spot_burn.setEnabled(n > 0)
        if n > 0:
            self.label_information.setText(
                f"Selected {n} points. Estimated time: {n * exposure:.0f} seconds"
            )
        else:
            self.label_information.setText(
                "No points selected. Right-click the FIB image to add points."
            )

    def run_spot_burn_worker(self) -> None:
        """Run the spot burn worker."""
        settings = self.get_settings()
        if len(settings.coordinates) == 0:
            notification_service.show_toast(
                "No points selected within FIB image bounds.", "warning"
            )
            return

        logging.info(
            f"Running spot burn with {len(settings.coordinates)} points. "
            f"Beam current: {settings.milling_current} A, exposure time: {settings.exposure_time} s"
        )

        self.stop_event.clear()
        self._is_burning = True
        self.pushButton_run_spot_burn.setText("Cancel")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.cancel_spot_burn)
        # in workflow mode the button is hidden when idle — show it as "Cancel" while burning
        self._update_run_button_visibility()

        self.worker = FunctionWorker(self._run_spot_burn, settings)
        self.worker.returned.connect(self.spot_burn_finished)
        self.worker.errored.connect(self.spot_burn_errored)
        self.worker.start()

    def cancel_spot_burn(self) -> None:
        """Cancel the running spot burn."""
        logging.info("Cancelling spot burn...")
        self.stop_event.set()

    def _run_spot_burn(self, settings: SpotBurnSettings) -> None:
        """Worker body (off the GUI thread). Progress flows through
        ``microscope.spot_burn_progress_signal``; completion is delivered on the GUI
        thread via the FunctionWorker's returned / errored signals, so a failure anywhere
        here — including the post-burn acquire — resets the UI via spot_burn_errored."""
        settings.run(
            microscope=self.microscope,
            beam_type=BeamType.ION,
            stop_event=self.stop_event,
        )
        # acquire a post-burn fib image and push it to the view (off the GUI thread)
        image = self.microscope.acquire_image(beam_type=BeamType.ION)
        self.microscope.fib_acquisition_signal.emit(image)

    def spot_burn_finished(self, result) -> None:
        """Called when the spot burn is finished."""
        self._is_burning = False
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setEnabled(True)
        self.pushButton_run_spot_burn.setText("Run Spot Burn")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        # in workflow mode, hide the button again now the burn is done
        self._update_run_button_visibility()

    def spot_burn_errored(self, error) -> None:
        """Called when the spot burn fails."""
        logging.error(f"Spot burn failed: {error}")
        self.microscope.spot_burn_progress_signal.emit({"finished": True, "error": True})
        self.spot_burn_finished(error)

    @ensure_main_thread
    def _update_progress_bar(self, ddict: dict) -> None:
        """Render spot burn progress (identical formatting to the status bar)."""
        self.progress_widget.update_progress(build_spot_burn_progress_update(ddict))

    def closeEvent(self, event) -> None:
        self.disconnect_signals()
        iw = self._image_widget()
        if iw is not None:
            try:
                iw.viewer_update_signal.disconnect(self._feed_image_shape)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)
