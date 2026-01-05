import logging
import threading
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QProgressBar,
    QPushButton,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import MillingState
from fibsem.ui import stylesheets
from fibsem.utils import format_duration

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget


class FibsemMillingWidget2(QWidget):
    """Widget for running a milling task with FibsemMicroscope.
    This widget provides a button to start the milling task and handles
    the threading and progress updates.
    """
    start_milling_signal = pyqtSignal()

    def __init__(self, microscope: FibsemMicroscope, parent: "MillingTaskConfigWidget"):
        super().__init__(parent)
        self.microscope = microscope
        self.parent_widget = parent

        self._milling_thread: Optional[threading.Thread] = None
        self._milling_stop_event = threading.Event()
        layout = QGridLayout()

        # pushbutton for run milling
        self.pushButton_run_milling = QPushButton("Run Milling")
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_run_milling.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)

        self.pushButton_stop_milling = QPushButton("Stop Milling")
        self.pushButton_stop_milling.clicked.connect(self.stop_milling)
        self.pushButton_stop_milling.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_stop_milling.setVisible(False)

        self.pushButton_pause_milling = QPushButton("Pause Milling")
        self.pushButton_pause_milling.clicked.connect(self.pause_resume_milling)
        self.pushButton_pause_milling.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_pause_milling.setVisible(False)

        self.progressBar_milling = QProgressBar(self)
        self.progressBar_milling_stages = QProgressBar(self)
        self.progressBar_milling.setVisible(False)
        self.progressBar_milling_stages.setVisible(False)
        self.progressBar_milling_stages.setStyleSheet(
            stylesheets.PROGRESS_BAR_BLUE_STYLE
        )
        self.progressBar_milling.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)

        self.start_milling_signal.connect(self.run_milling)

        # TODO: milling message display

        layout.addWidget(self.pushButton_run_milling, 0, 0, 1, 2)
        layout.addWidget(self.progressBar_milling_stages, 1, 0, 1, 2)
        layout.addWidget(self.progressBar_milling, 2, 0, 1, 2)
        layout.addWidget(self.pushButton_pause_milling, 3, 0)
        layout.addWidget(self.pushButton_stop_milling, 3, 1)

        self.setLayout(layout)

    @property
    def is_milling(self) -> bool:
        """Check if a milling task is currently running."""
        return self._milling_thread is not None and self._milling_thread.is_alive()

    @ensure_main_thread
    def _on_milling_progress(self, progress: dict):
        logging.info(f"Milling progress: {progress}")

        # update the UI based on the progress information
        self._update_button_states()

        progress_info: dict = progress.get("progress", None)  # type: ignore
        if progress_info is None:
            logging.warning("No progress information provided.")
            return

        state = progress_info.get("state", None)

        # start milling stage progress bar
        if state == "start":
            msg = progress.get("msg", "Preparing Milling Conditions...")
            current_stage = progress_info.get("current_stage", 0)
            total_stages = progress_info.get("total_stages", 1)
            self.progressBar_milling.setVisible(True)
            self.progressBar_milling_stages.setVisible(True)
            self.progressBar_milling.setValue(0)
            self.progressBar_milling.setRange(0, 100)
            self.progressBar_milling.setFormat(msg)
            self.progressBar_milling_stages.setRange(0, 100)
            self.progressBar_milling_stages.setValue(
                int((current_stage + 1) / total_stages * 100)
            )
            self.progressBar_milling_stages.setFormat(
                f"Milling Stage: {current_stage + 1}/{total_stages}"
            )

        # update
        if state == "update":
            logging.debug(progress_info)

            estimated_time = progress_info.get("estimated_time", None)
            remaining_time = progress_info.get("remaining_time", None)
            start_time = progress_info.get("start_time", None)

            if remaining_time is None or estimated_time is None:
                logging.warning(
                    "Remaining time or estimated time not provided in progress info."
                )
                return

            # calculate the percent complete
            percent_complete = int((1 - (remaining_time / estimated_time)) * 100)
            self.progressBar_milling.setValue(percent_complete)
            self.progressBar_milling.setFormat(
                f"Current Stage: {format_duration(remaining_time)} remaining..."
            )

        # finished
        if state == "finished":
            self.progressBar_milling.setVisible(False)
            self.progressBar_milling_stages.setVisible(False)

    def run_milling(self):
        """Start the milling task in a separate thread."""
        # If a milling task is already running, do nothing
        if self.is_milling:
            logging.warning("Milling task is already running.")
            return

        # clear the stop event, disable gui elements
        self._milling_stop_event.clear()
        self.pushButton_run_milling.setEnabled(False)

        # Start the milling task in a separate thread
        self._milling_thread = threading.Thread(
            target=self._milling_worker,
            args=(self.microscope, self.parent_widget.get_config()),
            daemon=True,
        )

        self._milling_thread.start()

    def _milling_worker(
        self, microscope: FibsemMicroscope, milling_task_config: FibsemMillingTaskConfig
    ):
        """Worker function to run the milling task in a separate thread."""
        try:
            if not milling_task_config.stages:
                raise ValueError("No milling stages defined in the configuration.")

            run_milling_task(
                microscope=microscope, config=milling_task_config, parent_ui=self
            )

        except Exception as e:
            logging.error(f"Error occurred while running milling task: {e}")

        finally:
            self._milling_thread = None
            self._update_button_states()

    def stop_milling(self):
        if self.is_milling:
            self._milling_stop_event.set()
            self.microscope.stop_milling()

    def pause_resume_milling(self):
        milling_state = self.microscope.get_milling_state()
        if self.is_milling and milling_state is MillingState.RUNNING:
            self.microscope.pause_milling()
            self.pushButton_pause_milling.setText("Resume Milling")
        else:
            self.microscope.resume_milling()
            self.pushButton_pause_milling.setText("Pause Milling")

    def _update_button_states(self):
        """Update the enabled/disabled state of buttons based on current milling state."""
        if self.is_milling:
            self.pushButton_run_milling.setEnabled(False)
            self.pushButton_stop_milling.setEnabled(True)
            self.pushButton_stop_milling.setVisible(True)
            self.pushButton_pause_milling.setEnabled(True)
            self.pushButton_pause_milling.setVisible(True)
        else:
            self.pushButton_run_milling.setEnabled(True)
            self.pushButton_stop_milling.setEnabled(False)
            self.pushButton_stop_milling.setVisible(False)
            self.pushButton_pause_milling.setEnabled(False)
            self.pushButton_pause_milling.setVisible(False)
