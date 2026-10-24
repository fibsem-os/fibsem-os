import logging
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
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures.settings import MillingState
from fibsem.ui import stylesheets
from fibsem.ui.control import control_of
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.util.durations import format_duration_precise
from fibsem.workflows.actions import record_operator_action

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_task_config_widget import MillingTaskConfigWidget


class FibsemMillingWidget(QWidget):
    """Widget for running a milling task with FibsemMicroscope.
    This widget provides a button to start the milling task and handles
    the threading and progress updates.
    """

    finished_milling_signal = pyqtSignal()

    def __init__(self, microscope: FibsemMicroscope, parent: "MillingTaskConfigWidget"):
        super().__init__(parent)
        self.microscope = microscope
        self.parent_widget = parent

        self._milling_thread: Optional[FunctionWorker] = None
        self._has_stages = False
        # The last words a producer supplied, so a backend's messageless tick still
        # has a label to show. See `MillingMessageTracker`.
        self._milling_label = MillingMessageTracker()
        # What the instrument last said it was doing. See `pause_resume_milling`.
        self._milling_state: Optional[MillingState] = None
        # A milling task someone else runs on the milling service is in progress:
        # the workflow's mill, which runs on the workflow thread. Set by
        # its task start and cleared by its end, so the controls follow it too.
        self._service_task = False
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

        milling = getattr(self.microscope, "milling", None)
        if self.parent_widget._milling_enabled and milling is not None:
            milling.progress.changed.connect(self._on_milling_progress)

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
        """Check if a milling task is currently running: this widget's own, or
        the workflow's on the milling service."""
        return self._own_run or self._service_task

    @property
    def _own_run(self) -> bool:
        return self._milling_thread is not None and self._milling_thread.is_alive()

    @ensure_main_thread
    def _on_milling_progress(self, payload: object) -> None:
        # Total-by-construction decode. Every in-tree producer emits a
        # `MillingProgress` and this is a no-op for them; it stands because a
        # plugin-loaded strategy is a producer too, and psygnal hands whatever it
        # emits to this slot unchanged.
        report = MillingProgress.from_payload(payload)
        logging.debug("Milling progress: %s", report)

        if report.milling_state is not None:
            self._milling_state = report.milling_state
        if report.status is MillingProgressStatus.TASK_STARTED:
            self._service_task = not self._own_run
        elif report.status.is_terminal:
            self._service_task = False

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
            if self.progressBar_milling.isHidden():
                # A mill with no stage start before it: a service run, the agent's.
                # Its countdown gets the bar; it has no stages to count.
                self.progressBar_milling.setRange(0, 100)
                self.progressBar_milling.setValue(0)
                self.progressBar_milling.setVisible(True)
            remaining_time = report.remaining_time
            # Falsy rather than `is None`, so an estimate of zero takes this branch
            # instead of dividing by it. That division runs inside a queued Qt slot, and
            # on PyQt5 an exception escaping one is `qFatal` -- the process aborts with
            # nothing in the logfile.
            if remaining_time is None or not report.estimated_time:
                # No countdown to draw, but the producer's words are still worth showing:
                # this is the branch a strategy's own report lands in, and it used to
                # match nothing at all and render nowhere.
                self.progressBar_milling.setFormat(label)
                return

            percent_complete = int((1 - (remaining_time / report.estimated_time)) * 100)
            self.progressBar_milling.setValue(percent_complete)
            self.progressBar_milling.setFormat(
                f"{label} - {format_duration_precise(remaining_time)} remaining"
            )

    def run_milling(
        self,
        config: Optional[FibsemMillingTaskConfig] = None,
        output_dir: Optional[str] = None,
    ):
        """Start the milling task in a separate thread.

        Args:
            config: Optional pre-built config to use. If None, calls
                    ``parent_widget.get_config()`` to build one fresh.
            output_dir: The folder the run writes under, when a task asked for
                    the mill; see ``run_milling_task``. None from the Run button.
        """
        # If a milling task is already running, do nothing
        if self.is_milling:
            logging.warning("Milling task is already running.")
            return

        # disable gui elements
        self.pushButton_run_milling.setEnabled(False)

        if config is None:
            config = self.parent_widget.get_config()

        # Start the milling task in a separate thread
        self._milling_thread = FunctionWorker(
            self._milling_worker, self.microscope, config, output_dir
        )
        # A method, not the signal's emit: Qt drops it with the widget, so a run
        # that ends after the widget is gone calls nothing.
        self._milling_thread.finished.connect(self._on_milling_finished)

        self._milling_thread.start(control=control_of(self.microscope), action="mill")

        # Lock the controls now rather than waiting for the first progress signal to
        # arrive -- preparing the milling conditions takes seconds, and the editor was
        # live for all of them. `config` above is already built, so a late edit could
        # not have reached this run anyway; what it could do is be written back after.
        self._update_button_states()

    def _on_milling_finished(self) -> None:
        # Unlock the controls first: listeners take the signal to mean "fully complete".
        self._update_button_states()
        self.finished_milling_signal.emit()

    def _milling_worker(
        self,
        microscope: FibsemMicroscope,
        milling_task_config: FibsemMillingTaskConfig,
        output_dir: Optional[str] = None,
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

            # The milling service's run, the one the agent and the workflow use.
            # Without a task's folder, the run saves where the
            # config's imaging path says, as it always has from this button.
            if output_dir is None:
                output_dir = milling_task_config.acquisition.imaging.path
            microscope.milling.run_task(milling_task_config, path=output_dir)

        except Exception as e:
            logging.error(
                f"Error occurred while running milling task: {e}", exc_info=True
            )

        finally:
            self._milling_thread = None

    def stop_milling(self):
        if self._own_run or self._service_task:
            # This widget's mill or the workflow's, both a task on the milling
            # service: its stop ends the whole task.
            record_operator_action(
                self.microscope, "Stop the mill", stops="milling", via="Stop Milling"
            )
            try:
                self.microscope.milling.stop()
            except Exception as e:  # noqa: BLE001 - already ended
                logging.info(f"Nothing to stop: {e}")

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
        # never milled. Explicitly disabling a child survives the workflow
        # enabling this widget's parent, and the `finally` in _milling_worker means
        # the unlock runs even when the task raises.
        self.parent_widget.config_widget.setEnabled(not milling)
