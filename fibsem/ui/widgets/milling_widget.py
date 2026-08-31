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
from fibsem.milling.progress import (
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import MillingState
from fibsem.ui import stylesheets
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.utils import format_duration

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_config_widget2 import MillingTaskConfigWidget2


class FibsemMillingWidget2(QWidget):
    """Widget for running a milling task with FibsemMicroscope.
    This widget provides a button to start the milling task and handles
    the threading and progress updates.
    """

    finished_milling_signal = pyqtSignal()

    def __init__(
        self, microscope: FibsemMicroscope, parent: "MillingTaskConfigWidget2"
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.parent_widget = parent

        self._milling_thread: Optional[FunctionWorker] = None
        self._milling_stop_event = threading.Event()
        self._has_stages = False
        # The last words a producer supplied, so a backend's messageless tick still
        # has a label to show. See `MillingMessageTracker`.
        self._milling_label = MillingMessageTracker()
        # What the instrument last said it was doing. See `pause_resume_milling`.
        self._milling_state: Optional[MillingState] = None
        layout = QGridLayout()

        # pushbutton for run milling
        self.pushButton_run_milling = QPushButton("Run Milling")
        self.pushButton_run_milling.clicked.connect(lambda: self.run_milling(None))
        self.pushButton_run_milling.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)

        self.pushButton_stop_milling = QPushButton("Stop Milling")
        self.pushButton_stop_milling.clicked.connect(self.stop_milling)
        self.pushButton_stop_milling.setStyleSheet(stylesheets.DANGER_BUTTON_STYLESHEET)
        self.pushButton_stop_milling.setVisible(False)

        self.pushButton_pause_milling = QPushButton("Pause Milling")
        self.pushButton_pause_milling.clicked.connect(self.pause_resume_milling)
        self.pushButton_pause_milling.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_pause_milling.setVisible(False)

        self.progressBar_milling = QProgressBar(self)
        self.progressBar_milling_stages = QProgressBar(self)
        self.progressBar_milling.setVisible(False)
        self.progressBar_milling_stages.setVisible(False)
        self.progressBar_milling_stages.setStyleSheet(
            stylesheets.PROGRESS_BAR_STYLESHEET
        )
        self.progressBar_milling.setStyleSheet(stylesheets.PROGRESS_BAR_STYLESHEET)

        if self.parent_widget._milling_enabled:
            self.microscope.milling_progress_signal.connect(self._on_milling_progress)

        # TODO: milling message display

        layout.addWidget(self.pushButton_run_milling, 0, 0, 1, 2)
        layout.addWidget(self.progressBar_milling_stages, 1, 0, 1, 2)
        layout.addWidget(self.progressBar_milling, 2, 0, 1, 2)
        layout.addWidget(self.pushButton_pause_milling, 3, 0)
        layout.addWidget(self.pushButton_stop_milling, 3, 1)

        self.setLayout(layout)

        # disable run button when no stages
        stages_widget = self.parent_widget.config_widget.milling_stages_widget
        stages_widget.stages_changed.connect(self._on_stages_changed)
        self._on_stages_changed(stages_widget.get_stages())

    @property
    def is_milling(self) -> bool:
        """Check if a milling task is currently running."""
        return self._milling_thread is not None and self._milling_thread.is_alive()

    @ensure_main_thread
    def _on_milling_progress(self, payload: object) -> None:
        # Total-by-construction decode. Every in-tree producer emits a
        # `MillingProgress` and this is a no-op for them; it stands because a
        # plugin-loaded strategy is a producer too, and psygnal hands whatever it
        # emits to this slot unchanged (FIB-797).
        report = MillingProgress.from_payload(payload)
        logging.debug("Milling progress: %s", report)

        if report.milling_state is not None:
            self._milling_state = report.milling_state

        # update the UI based on the progress information
        self._update_button_states()

        if report.status.is_terminal:
            self.progressBar_milling.setVisible(False)
            self.progressBar_milling_stages.setVisible(False)
            return

        label = self._milling_label.label(report)

        if report.status is MillingProgressStatus.STAGE_STARTED:
            # `or 1` rather than a `.get` default: a producer that sends 0 total stages
            # is as much a division by zero as one that sends nothing.
            total_stages = report.total_stages or 1
            stage = report.display_stage or 1
            stage_name = report.stage_name or f"Stage {stage}"
            self.progressBar_milling.setVisible(True)
            self.progressBar_milling_stages.setVisible(True)
            self.progressBar_milling.setValue(0)
            self.progressBar_milling.setRange(0, 100)
            self.progressBar_milling.setFormat(label)
            self.progressBar_milling_stages.setRange(0, 100)
            self.progressBar_milling_stages.setValue(int(stage / total_stages * 100))
            self.progressBar_milling_stages.setFormat(
                f"Milling Stage: {stage}/{total_stages} - {stage_name}"
            )

        elif report.status is MillingProgressStatus.STAGE_UPDATE:
            remaining_time = report.remaining_time
            # Falsy rather than `is None`, so an estimate of zero takes this branch
            # instead of dividing by it. That division runs inside a queued Qt slot, and
            # on PyQt5 an exception escaping one is `qFatal` -- the process aborts with
            # nothing in the logfile (FIB-329).
            if remaining_time is None or not report.estimated_time:
                # No countdown to draw, but the producer's words are still worth showing:
                # this is the branch a strategy's own report lands in, and it used to
                # match nothing at all and render nowhere.
                self.progressBar_milling.setFormat(label)
                return

            percent_complete = int((1 - (remaining_time / report.estimated_time)) * 100)
            self.progressBar_milling.setValue(percent_complete)
            self.progressBar_milling.setFormat(
                f"{label} - {format_duration(remaining_time)} remaining"
            )

    def run_milling(self, config: Optional[FibsemMillingTaskConfig] = None):
        """Start the milling task in a separate thread.

        Args:
            config: Optional pre-built config to use. If None, calls
                    ``parent_widget.get_config()`` to build one fresh.
        """
        # If a milling task is already running, do nothing
        if self.is_milling:
            logging.warning("Milling task is already running.")
            return

        # clear the stop event, disable gui elements
        self._milling_stop_event.clear()
        self.pushButton_run_milling.setEnabled(False)

        if config is None:
            config = self.parent_widget.get_config()

        # Start the milling task in a separate thread
        self._milling_thread = FunctionWorker(
            self._milling_worker, self.microscope, config
        )
        # Both of these used to run from inside the worker's `finally`, which put the
        # widget update on the worker thread. Connected here in the order they used to
        # happen in: the controls have to be unlocked before anything acts on
        # `finished_milling_signal`, which listeners take to mean "fully complete".
        self._milling_thread.finished.connect(self._update_button_states)
        self._milling_thread.finished.connect(self.finished_milling_signal.emit)

        self._milling_thread.start()

        # Lock the controls now rather than waiting for the first progress signal to
        # arrive -- preparing the milling conditions takes seconds, and the editor was
        # live for all of them. `config` above is already built, so a late edit could
        # not have reached this run anyway; what it could do is be written back after.
        self._update_button_states()

    def _milling_worker(
        self, microscope: FibsemMicroscope, milling_task_config: FibsemMillingTaskConfig
    ):
        """Run the milling task. Runs off the GUI thread — only signals may cross back.

        Clearing ``_milling_thread`` stays here. ``is_milling`` would answer correctly
        without it — it also asks ``is_alive()``, and the thread has normally exited by
        the time the queued ``finished`` is processed — but that makes the button state
        depend on a race the clear settles outright, and it drops the reference.
        """
        try:
            if not milling_task_config.enabled_stages:
                raise ValueError("No milling stages defined in the configuration.")

            run_milling_task(
                microscope=microscope, config=milling_task_config, parent_ui=self
            )

        except Exception as e:
            logging.error(
                f"Error occurred while running milling task: {e}", exc_info=True
            )

        finally:
            self._milling_thread = None

    def stop_milling(self):
        if self.is_milling:
            self._milling_stop_event.set()
            self.microscope.stop_milling()

    def pause_resume_milling(self):
        # The last state a progress report carried, not a fresh `get_milling_state()`.
        # On ThermoFisher that getter *sets the active view* as a side effect, so a
        # click here during a coincidence mill competes for the view with the
        # fluorescence acquisition the strategy is running. Every producer now carries
        # the value on the report, so the widget already has it.
        if self.is_milling and self._milling_state is MillingState.RUNNING:
            self.microscope.pause_milling()
            self.pushButton_pause_milling.setText("Resume Milling")
        else:
            self.microscope.resume_milling()
            self.pushButton_pause_milling.setText("Pause Milling")

    def _on_stages_changed(self, stages: list):
        """Update internal stage state and refresh button availability."""
        self._has_stages = any(s.enabled for s in stages)
        self._update_button_states()

    @ensure_main_thread
    def _update_button_states(self):
        """Update the enabled/disabled state of the controls for the milling state."""
        milling = self.is_milling
        if milling:
            self.pushButton_run_milling.setEnabled(False)
            self.pushButton_stop_milling.setEnabled(True)
            self.pushButton_stop_milling.setVisible(True)
            self.pushButton_pause_milling.setEnabled(True)
            self.pushButton_pause_milling.setVisible(True)
        else:
            self.pushButton_run_milling.setEnabled(self._has_stages)
            self.pushButton_stop_milling.setEnabled(False)
            self.pushButton_stop_milling.setVisible(False)
            self.pushButton_pause_milling.setEnabled(False)
            self.pushButton_pause_milling.setVisible(False)

        # The stage and pattern editor is part of the milling controls, not scenery:
        # the run holds its own deep copy of the config, so an edit made mid-mill does
        # not reach the beam -- but it is written back as the task's config when the
        # task finishes, leaving the protocol claiming a pattern position that was
        # never milled (FIB-580). Explicitly disabling a child survives the workflow
        # enabling this widget's parent, and the `finally` in _milling_worker means
        # the unlock runs even when the task raises.
        self.parent_widget.config_widget.setEnabled(not milling)
