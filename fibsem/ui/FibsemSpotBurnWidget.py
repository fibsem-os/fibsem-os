import logging
import threading
from typing import List, Optional
import numpy as np

from napari.qt.threading import FunctionWorker, thread_worker
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

from fibsem.imaging.spot import run_spot_burn
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.qtdesigner_files import FibsemSpotBurnWidget as FibsemSpotBurnWidgetUI
from fibsem.utils import format_value

SPOT_BURN_POINTS_LAYER_NAME = "spot-burn-points"
DEFAULT_BEAM_CURRENT = 60e-12  # 60 pA

class FibsemSpotBurnWidget(FibsemSpotBurnWidgetUI.Ui_Form, QWidget):
    spot_burn_progress_signal = pyqtSignal(dict)

    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.parent = parent
        self.microscope: FibsemMicroscope = parent.microscope
        self.worker: FunctionWorker = None
        self.stop_event: threading.Event = threading.Event()

        # quad-view: a modal "spot" PointsSpec on the FIB canvas, owned by the controller.
        self._spot_created = False
        self._spot_wired = False

        self.setup_connections()

    # ------------------------------------------------------------------
    # Quad-view overlay (controller-owned)
    # ------------------------------------------------------------------

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController."""
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _fib_image(self):
        """Current FIB image (for normalized<->pixel conversion), or None."""
        iw = getattr(self.parent, "image_widget", None)
        return getattr(iw, "ib_image", None) if iw is not None else None

    def _ensure_spot(self):
        """Ensure the quad-view spot overlay exists in the model, or return None.

        Creates a modal "spot" PointsSpec (right-click to add, Delete to remove) on
        first use and subscribes to the controller's edit signal so the point count
        stays live. Returns the controller, or None on the napari path.
        """
        controller = self._view_controller()
        if controller is None:
            return None
        if not self._spot_wired:
            controller.overlay_edited.connect(self._on_spot_edited)
            self._spot_wired = True
        if not self._spot_created:
            from fibsem.ui.widgets.canvas_state import PointsSpec

            controller.set_overlay(
                BeamType.ION,
                PointsSpec(
                    id="spot", points=[], visible=False,
                    color="white", selected_color="cyan", marker="o", size=6,
                    add_on_right_click=True, removable=True, modal=True,
                ),
            )
            self._spot_created = True
        return controller

    def _on_spot_edited(self, beam, overlay_id, value) -> None:
        if overlay_id == "spot":
            self._on_data_changed()

    def closeEvent(self, event) -> None:
        if self._spot_wired:
            controller = self._view_controller()
            if controller is not None:
                try:
                    controller.overlay_edited.disconnect(self._on_spot_edited)
                except (TypeError, RuntimeError):
                    pass
            self._spot_wired = False
        super().closeEvent(event)

    def setup_connections(self):
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

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

        # initial state
        self.label_information.setText("No points selected. Please add points to the layer.")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)
        self.pushButton_run_spot_burn.setEnabled(False)

        # progress bar
        self.progressBar.setVisible(False)
        self.spot_burn_progress_signal.connect(self._update_progress_bar)

    def set_active(self):
        """Called when the widget is activated."""
        controller = self._ensure_spot()
        if controller is not None:
            controller.set_overlay_visible(BeamType.ION, "spot", True)
            controller.arm_overlay(BeamType.ION, "spot", label="Spot burn", icon="mdi:target")
            controller.fib_canvas.set_hint("drag to move  ·  right-click to add  ·  Delete to remove")
            self._on_data_changed()

    def set_inactive(self):
        """Called when the widget is deactivated."""
        controller = self._view_controller()
        if controller is not None:
            controller.set_overlay_visible(BeamType.ION, "spot", False)
            controller.arm_overlay(BeamType.ION, None)
            controller.fib_canvas.set_hint(None)

    def update_parameters(self, parameters: dict):
        """Update the parameters for the spot burn."""
        milling_current = parameters.get("milling_current", None)
        exposure_time = parameters.get("exposure_time", None)
        coordinates = parameters.get("coordinates", None)

        self._ensure_spot()  # ensure the overlay exists

        if milling_current is not None:
            index = self.comboBox_beam_current.findData(milling_current)
            if index != -1:
                self.comboBox_beam_current.setCurrentIndex(index)

        if exposure_time is not None:
            self.doubleSpinBox_exposure_time.setValue(exposure_time)

        if coordinates:
            self._set_coordinates(coordinates)

    def _set_coordinates(self, coordinates: list):
        """Pre-populate the spots from normalised image coordinates (0-1)."""
        controller = self._ensure_spot()
        if controller is None:
            return
        img = self._fib_image()
        if img is None:
            return
        h, w = img.data.shape[:2]
        controller.set_points(
            BeamType.ION, "spot", [(pt.x * w, pt.y * h) for pt in coordinates]
        )
        self._on_data_changed()

    def get_coordinates(self) -> list:
        """Get the current spot burn positions as normalised image coordinates (0-1)."""
        controller = self._view_controller()
        if controller is None:
            return []
        img = self._fib_image()
        if img is None:
            return []
        h, w = img.data.shape[:2]
        coords = [Point(float(col / w), float(row / h))
                  for col, row in controller.overlay_points(BeamType.ION, "spot")]
        return [pt for pt in coords if 0 <= pt.x <= 1 and 0 <= pt.y <= 1]

    def clear_points_layer(self):
        """Clear the spots and tear the overlay down (called when the task exits)."""
        controller = self._view_controller()
        if controller is not None:
            controller.arm_overlay(BeamType.ION, None)        # restore Move (un-arm)
            controller.remove_overlay(BeamType.ION, "spot")   # drop the overlay
            controller.fib_canvas.set_hint(None)              # clear the hint
            self._spot_created = False                        # recreate on next activate
            self._on_data_changed()

    def _on_data_changed(self, event = None):
        """Called when the spots change (overlay edit signals)."""
        controller = self._view_controller()
        if controller is None:
            return
        n = len(controller.overlay_points(BeamType.ION, "spot"))

        self.pushButton_run_spot_burn.setEnabled(n > 0)
        if n > 0:
            self.label_information.setText(f"Selected {n} points. Estimated time: {n * self.doubleSpinBox_exposure_time.value()} seconds")
            self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        else:
            self.label_information.setText("No points selected. Right-click the FIB image to add points.")
            self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)

    def run_spot_burn_worker(self):
        """Run the spot burn worker."""
        coordinates = self.get_coordinates()  # ensure coordinates are valid and within bounds

        if len(coordinates) == 0:
            notification_service.show_toast("No points selected within FIB image bounds.", "warning")
            return

        beam_current = self.comboBox_beam_current.currentData()     # amps
        exposure_time = self.doubleSpinBox_exposure_time.value()    # seconds

        logging.info(f"Running spot burn with {len(coordinates)} points. Beam current: {beam_current} A, exposure time: {exposure_time} s")

        self.stop_event.clear()
        self.pushButton_run_spot_burn.setText("Cancel")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.cancel_spot_burn)

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
                       parent_ui=self,
                       stop_event=self.stop_event)

    def spot_burn_finished(self, result):
        """Called when the spot burn is finished."""
        self.pushButton_run_spot_burn.clicked.disconnect()
        self.pushButton_run_spot_burn.clicked.connect(self.run_spot_burn_worker)
        self.pushButton_run_spot_burn.setEnabled(True)
        self.pushButton_run_spot_burn.setText("Burn Spot")
        self.pushButton_run_spot_burn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

    def spot_burn_errored(self, error):
        """Called when the spot burn fails."""
        logging.error(f"Spot burn failed: {error}")
        self.spot_burn_progress_signal.emit({"finished": True})
        self.spot_burn_finished(error)

    def _update_progress_bar(self, ddict: dict):
        """Update the progress bar with the current progress.
        
        Parameters
        ----------
        ddict : dict
            Dictionary with the following keys:
            - total_points (int): Total number of points to burn
            - current_point (int): Current point being burned
            - remaining_time (float): Remaining time for current point in seconds
            - total_remaining_time (float): Total remaining time in seconds
            - total_estimated_time (float): Total estimated time in seconds
            - finished (bool): Whether the spot burn is finished
        """        
        total_points = ddict.get("total_points", 0)
        current_point = ddict.get("current_point", 0)
        remaining_time = ddict.get("remaining_time", 0)
        total_remaining_time = ddict.get("total_remaining_time", 0)
        total_estimated_time = ddict.get("total_estimated_time", 0)
        total_elapsed_time = total_estimated_time - total_remaining_time
        finished = ddict.get("finished", False)

        self.progressBar.setVisible(True)
        self.progressBar.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.progressBar.setMinimum(0)
        
        if total_elapsed_time > 0 and total_estimated_time > 0:
            self.progressBar.setMaximum(int(total_estimated_time))
            self.progressBar.setValue(int(total_elapsed_time))
            self.progressBar.setTextVisible(True)
            self.progressBar.setFormat(f"Burning Spot {current_point}/{total_points}... {int(remaining_time)}s remaining")
        else:
            self.progressBar.setValue(0)
            self.progressBar.setFormat("Preparing Spot Burn...")

        if finished:
            self.progressBar.setValue(total_estimated_time)
            self.progressBar.setFormat("Spot Burn Finished")

