import logging
import threading
from typing import List, Optional
import numpy as np

import napari
import napari.utils.notifications
from napari.layers import Image as NapariImageLayer
from napari.layers import Points as NapariPointsLayer
from fibsem.ui.qt.threading import FunctionWorker, thread_worker
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal
from superqt import ensure_main_thread

from fibsem.imaging.spot import run_spot_burn
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point
from fibsem.ui import stylesheets
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate
from fibsem.utils import format_value

SPOT_BURN_POINTS_LAYER_NAME = "spot-burn-points"
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
    # emitted by the workflow task to trigger a burn (mirrors milling's start_milling_signal)
    start_spot_burn_signal = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self._setup_ui()

        self.parent = parent
        self.viewer: napari.Viewer = parent.viewer
        self.microscope: FibsemMicroscope = parent.microscope
        self.worker: FunctionWorker = None
        self.stop_event: threading.Event = threading.Event()
        self._is_burning: bool = False
        self._workflow_mode: bool = False

        # napari layers
        self.pts_layer: Optional[NapariPointsLayer] = None
        self.image_layer: Optional[NapariImageLayer] = None

        self.setup_connections()

    def _setup_ui(self):
        """Hand-built replacement for the former Qt Designer form.

        ``setup_connections`` swaps ``progressBar`` for a ``FibsemProgressWidget`` and
        drops ``label_workflow_hint`` into the run button's cell, so the grid, the
        ``progressBar`` placeholder, and the widget object names are reproduced here.
        """
        self.gridLayout_2 = QGridLayout(self)

        self.label_title = QLabel("Spot Burn")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 2)

        self.label_exposure_time = QLabel("Exposure Time")
        self.doubleSpinBox_exposure_time = QDoubleSpinBox()
        self.gridLayout_2.addWidget(self.label_exposure_time, 1, 0)
        self.gridLayout_2.addWidget(self.doubleSpinBox_exposure_time, 1, 1)

        self.label_beam_current = QLabel("Beam Current")
        self.comboBox_beam_current = QComboBox()
        self.gridLayout_2.addWidget(self.label_beam_current, 2, 0)
        self.gridLayout_2.addWidget(self.comboBox_beam_current, 2, 1)

        self.label_information = QLabel("")
        self.gridLayout_2.addWidget(self.label_information, 3, 0, 1, 2)

        self.progressBar = QProgressBar()
        self.gridLayout_2.addWidget(self.progressBar, 4, 0, 1, 2)

        self.pushButton_run_spot_burn = QPushButton("Run Spot Burn")
        self.gridLayout_2.addWidget(self.pushButton_run_spot_burn, 5, 0, 1, 2)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacer, 6, 0, 1, 2)

        # block accidental scroll-to-change on the input widgets
        for w in (self.comboBox_beam_current, self.doubleSpinBox_exposure_time):
            install_wheel_blocker(w)

    def setup_connections(self):
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)

        beam_currents = self.microscope.get_available_values("current", BeamType.ION)
        for current in beam_currents:
            label = format_value(current, unit="A", precision=1)
            self.comboBox_beam_current.addItem(label, current)

        # find the current closest to 60pA, set index to that
        closest_current = min(beam_currents, key=lambda x: abs(x - DEFAULT_BEAM_CURRENT))
        closest_index = beam_currents.index(closest_current)
        self.comboBox_beam_current.setCurrentIndex(closest_index)

        # set parameters for exposure time
        self.doubleSpinBox_exposure_time.setSuffix(" s")
        self.doubleSpinBox_exposure_time.setRange(0.1, 60)
        self.doubleSpinBox_exposure_time.setValue(10)
        self.doubleSpinBox_exposure_time.valueChanged.connect(self._on_data_changed)

        # initial state (PRIMARY greys out via its :disabled rule while no points are selected)
        self.label_information.setText("No points selected. Please add points to the layer.")
        self.pushButton_run_spot_burn.setEnabled(False)

        # replace the .ui QProgressBar with the shared FibsemProgressWidget (in the same
        # grid cell) so the widget and the status bar render progress identically.
        bar_pos = self.gridLayout_2.getItemPosition(
            self.gridLayout_2.indexOf(self.progressBar)
        )
        self.gridLayout_2.removeWidget(self.progressBar)
        self.progressBar.deleteLater()
        self.progress_widget = FibsemProgressWidget(self)
        self.gridLayout_2.addWidget(self.progress_widget, *bar_pos)

        # progress is driven by the microscope's spot_burn_progress_signal (shared with the
        # status bar). psygnal fires on the worker thread, so _update_progress_bar is
        # marshalled to the GUI thread via @ensure_main_thread.
        self.microscope.spot_burn_progress_signal.connect(self._update_progress_bar)

        # the workflow task drives the burn via start_spot_burn_signal (mirrors milling).
        # BlockingQueuedConnection: emit() returns only after run_spot_burn_worker has
        # run, so the burn is then either in progress (is_burning=True) or was refused
        # (no in-bounds points), in which case the task re-prompts.
        self.start_spot_burn_signal.connect(
            self.run_spot_burn_worker, Qt.BlockingQueuedConnection  # type: ignore
        )

        # workflow-mode hint, shown when the widget's own Burn button is hidden during a
        # supervised workflow (the burn is run from the workflow "Run Spot Burn" control).
        # It shares the Burn button's grid cell on purpose: exactly one of the two is
        # visible at any time (_update_run_button_visibility), so the hint appears in
        # place of the hidden button without the layout shifting.
        self.label_workflow_hint = QLabel(
            "Use the 'Run Spot Burn' button in the workflow controls to burn the selected points."
        )
        self.label_workflow_hint.setWordWrap(True)
        self.label_workflow_hint.setStyleSheet("color: gray; font-style: italic;")
        self.label_workflow_hint.setVisible(False)
        btn_pos = self.gridLayout_2.getItemPosition(
            self.gridLayout_2.indexOf(self.pushButton_run_spot_burn)
        )
        self.gridLayout_2.addWidget(self.label_workflow_hint, *btn_pos)

    @property
    def is_burning(self) -> bool:
        """Whether a spot burn is currently running."""
        return self._is_burning

    def set_workflow_mode(self, active: bool):
        """Toggle workflow mode.

        During a supervised workflow the burn is task-orchestrated (like milling), so the
        widget's own Burn button is hidden and a hint points the user to the workflow
        "Run Spot Burn" control. The button still appears as "Cancel" while a burn is in
        progress so the user can cancel locally. Outside the workflow it is used directly.
        """
        self._workflow_mode = active
        self._update_run_button_visibility()

    def _update_run_button_visibility(self):
        """Show/hide the run button and workflow hint for the current mode + burn state."""
        if self._workflow_mode:
            # idle: hidden (run via the workflow control); burning: shown as "Cancel"
            self.pushButton_run_spot_burn.setVisible(self._is_burning)
            self.label_workflow_hint.setVisible(not self._is_burning)
        else:
            self.pushButton_run_spot_burn.setVisible(True)
            self.label_workflow_hint.setVisible(False)

    def disconnect_signals(self):
        """Disconnect from the microscope's progress signal before the widget is destroyed.

        The microscope outlives this widget; without this, the strong psygnal connection
        would leak the widget and fire _update_progress_bar on a deleted object.
        """
        try:
            self.microscope.spot_burn_progress_signal.disconnect(self._update_progress_bar)
        except Exception:
            pass

    def _add_points_layer(self):
        # check if the points layer exists, if not create it
        if SPOT_BURN_POINTS_LAYER_NAME not in self.viewer.layers:
            self.pts_layer = self.viewer.add_points(data=[],
                                name=SPOT_BURN_POINTS_LAYER_NAME,
                                visible=True,
                                size=20)
            self.pts_layer.events.data.connect(self._on_data_changed)
        else:
            self.pts_layer = self.viewer.layers[SPOT_BURN_POINTS_LAYER_NAME]

    def set_active(self):
        """Called when the widget is activated."""

        # add the points layer if it doesn't exist
        self._add_points_layer()
        if self.pts_layer is None:
            raise RuntimeError("Failed to create points layer for spot burn.")

        self.viewer.layers.selection.active = self.pts_layer
        self.pts_layer.visible = True
        self.pts_layer.mode = "add"

    def set_inactive(self):
        """Called when the widget is deactivated."""

        # hide the points layer
        if SPOT_BURN_POINTS_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[SPOT_BURN_POINTS_LAYER_NAME].visible = False

        self.parent.image_widget.restore_active_layer_for_movement()

    def update_parameters(self, parameters: dict):
        """Update the parameters for the spot burn."""
        milling_current = parameters.get("milling_current", None)
        exposure_time = parameters.get("exposure_time", None)
        coordinates = parameters.get("coordinates", None)

        if self.pts_layer is None:
            self._add_points_layer() # ensure points layer exists
    
        if milling_current is not None:
            index = self.comboBox_beam_current.findData(milling_current)
            if index != -1:
                self.comboBox_beam_current.setCurrentIndex(index)

        if exposure_time is not None:
            self.doubleSpinBox_exposure_time.setValue(exposure_time)

        if coordinates and self.pts_layer is not None:
            self._set_coordinates(coordinates)

    def _set_coordinates(self, coordinates: list):
        """Pre-populate the points layer from normalised image coordinates (0-1)."""

        self.image_layer = self.parent.image_widget.ib_layer
        if self.image_layer is None or not isinstance(self.image_layer, NapariImageLayer):
            return

        image_shape = self.image_layer.data.shape
        translate = self.image_layer.translate

        # convert normalised (0-1) Point coordinates to napari pixel coordinates
        pts = np.array([[pt.y * image_shape[0] + translate[0],
                         pt.x * image_shape[1] + translate[1]]
                        for pt in coordinates])
        self.pts_layer.data = pts
        self._on_data_changed()

    def get_coordinates(self) -> list:
        """Get the current spot burn positions as normalised image coordinates (0-1)."""
        if self.pts_layer is None or len(self.pts_layer.data) == 0:
            return []

        self.image_layer = self.parent.image_widget.ib_layer
        if self.image_layer is None or not isinstance(self.image_layer, NapariImageLayer):
            return []

        layer_translated = self.pts_layer.data - self.image_layer.translate
        image_shape = self.image_layer.data.shape

        coordinates = [Point(float(pt[1] / image_shape[1]), float(pt[0] / image_shape[0]))
                       for pt in layer_translated]
        # exclude points outside of image bounds
        coordinates = [pt for pt in coordinates if 0 <= pt.x <= 1 and 0 <= pt.y <= 1]
        return coordinates

    def clear_points_layer(self):
        """Clear the points layer."""
        if SPOT_BURN_POINTS_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(SPOT_BURN_POINTS_LAYER_NAME)
            self.pts_layer = None

    def _on_data_changed(self, event = None):
        """Called when the data in the points layer changes."""
        if self.pts_layer is None:
            return
        coordinates = self.pts_layer.data
        
        enabled = bool(len(coordinates) > 0)
        self.pushButton_run_spot_burn.setEnabled(enabled)
        if enabled:
            self.label_information.setText(f"Selected {len(coordinates)} points. Estimated time: {len(coordinates) * self.doubleSpinBox_exposure_time.value()} seconds")
        else:
            self.label_information.setText("No points selected. Please add points to the layer.")

    def run_spot_burn_worker(self):
        """Run the spot burn worker."""

        # get the points layer
        if SPOT_BURN_POINTS_LAYER_NAME not in self.viewer.layers:
            napari.utils.notifications.show_warning("No points layer found. Requires 'spot-burn-points' layer.")
            return

        coordinates = self.get_coordinates()  # ensure coordinates are valid and within bounds

        if len(coordinates) == 0:
            napari.utils.notifications.show_warning("No points selected within FIB image bounds.")
            return

        beam_current = self.comboBox_beam_current.currentData()     # amps
        exposure_time = self.doubleSpinBox_exposure_time.value()    # seconds

        logging.info(f"Running spot burn with {len(coordinates)} points. Beam current: {beam_current} A, exposure time: {exposure_time} s")

        self.stop_event.clear()
        self._is_burning = True
        self.pushButton_run_spot_burn.setText("Cancel")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.cancel_spot_burn)
        # in workflow mode the button is hidden when idle — show it as "Cancel" while burning
        self._update_run_button_visibility()

        self.worker = self._spot_burn_worker(microscope=self.microscope,
                                             coordinates=coordinates,
                                             exposure_time=exposure_time,
                                             milling_current=beam_current,)
        self.worker.returned.connect(self.spot_burn_finished)
        self.worker.errored.connect(self.spot_burn_errored)
        self.worker.start()

    def cancel_spot_burn(self):
        """Cancel the running spot burn."""
        logging.info("Cancelling spot burn...")
        self.stop_event.set()

    @thread_worker
    def _spot_burn_worker(self, microscope: FibsemMicroscope, coordinates: List[Point], exposure_time: float, milling_current: float):
        """Worker function to run the spot burn."""
        run_spot_burn(microscope=microscope,
                       coordinates=coordinates,
                       exposure_time=exposure_time,
                       milling_current=milling_current,
                       beam_type=BeamType.ION,
                       stop_event=self.stop_event)

        # acquire a post-burn fib image and update the view
        image = microscope.acquire_image(beam_type=BeamType.ION)
        microscope.fib_acquisition_signal.emit(image)

    def spot_burn_finished(self, result):
        """Called when the spot burn is finished."""
        self._is_burning = False
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setEnabled(True)
        self.pushButton_run_spot_burn.setText("Burn Spot")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        # in workflow mode, hide the button again now the burn is done
        self._update_run_button_visibility()

    def spot_burn_errored(self, error):
        """Called when the spot burn fails."""
        logging.error(f"Spot burn failed: {error}")
        self.microscope.spot_burn_progress_signal.emit({"finished": True, "error": True})
        self.spot_burn_finished(error)

    @ensure_main_thread
    def _update_progress_bar(self, ddict: dict):
        """Render spot burn progress (identical formatting to the status bar)."""
        self.progress_widget.update_progress(build_spot_burn_progress_update(ddict))

