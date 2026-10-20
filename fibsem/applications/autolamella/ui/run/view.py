"""What the window shows of a run: the status bar's run line and failures, the
supervision chip, Run and Stop, the border, and how the run ended.

It hears every run through the session's runner and the run relay, whoever
started it. The tabs redraw themselves for a run's reports; the window's
`lock` callback is what a run takes from them while it goes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QMessageBox, QWidget

from fibsem.acting import AGENT, OPERATOR
from fibsem.applications.autolamella.workflow_edits import set_supervision
from fibsem.applications.autolamella.workflows.tasks.tasks import get_task_supervision
from fibsem.ui import notification_service
from fibsem.ui.widgets.run_border import RunBorder, run_border_state
from fibsem.ui.widgets.status_bar import FibsemStatusBar
from fibsem.ui.widgets.workflow.controls import AUTOMATED, SUPERVISED, WorkflowControls
from fibsem.ui.widgets.workflow_summary_dialog import (
    WorkflowSummaryDialog,
    run_counts_text,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.runner import GRIDS, RunRequest
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    import pandas as pd

    from fibsem.applications.autolamella.ui.run.attention import AttentionRouter
    from fibsem.ui.qt.run_bridge import RunRelay
    from fibsem.workflows.session import AppSession
    from fibsem.workflows.status import WorkflowStatusEvent, WorkflowStatusUpdate

__all__ = ["RunView"]


class RunView(QObject):
    def __init__(
        self,
        session: AppSession,
        relay: RunRelay,
        *,
        host: QWidget,
        status_bar: FibsemStatusBar,
        controls: WorkflowControls,
        border: RunBorder,
        attention: AttentionRouter,
        flush_editors: Callable[[], None],
        lock: Callable[[bool], None],
        show_failures: Callable[[], None],
        after_supervision: Callable[[], None],
        reset_panel: Callable[[], None],
    ) -> None:
        """*lock* takes from the tabs (True) or gives back (False) what a run
        owns; *show_failures* is the failure line's Show; *after_supervision*
        redraws the workflow list after the chip's toggle; *reset_panel* puts the
        panel back after a run."""
        super().__init__(host)
        self._session = session
        self._relay = relay
        self._host = host
        self._status_bar = status_bar
        self._controls = controls
        self._border = border
        self._attention = attention
        self._flush_editors = flush_editors
        self._lock = lock
        self._after_supervision = after_supervision
        self._reset_panel = reset_panel

        # This run's failures, for the line it leaves when it ends (FIB-1188).
        self._run_active = False
        self._run_failures: List[Tuple[str, str]] = []
        self._run_total = 0
        # The task the chip toggles.
        self.current_task_name: Optional[str] = None
        # The summary already toasted (by identity): the run-end toast is
        # once-per-run, while the runner's last_summary itself must survive as
        # the record remote readers see.
        self._shown_run_summary: Optional[pd.DataFrame] = None

        status_bar.failure_details_requested.connect(show_failures)
        controls.supervision_clicked.connect(self._on_supervised_status_clicked)
        controls.stop_clicked.connect(self._on_stop_workflow_clicked)
        session.runner.starting.connect(self._on_run_starting)
        relay.status.connect(self._on_status)
        relay.finished.connect(self._on_finished)
        # Red while a stop unwinds, whoever asked for it (the agent's too).
        relay.stopping.connect(self._on_stopping)

    def stop(self, actor=None, via: str = "") -> None:
        """Stop the run; ``actor`` and ``via`` say who stopped it and from
        where, for the record (``runner.stop``)."""
        if not self._session.runner.is_running:
            return
        self._session.runner.stop(actor=actor, via=via)
        self._relay.stopping.emit()

    def _on_stop_workflow_clicked(self):
        """Handle stop workflow button click with confirmation."""
        reply = QMessageBox.question(
            self._host,
            "Stop Workflow",
            "Are you sure you want to stop the workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # The border goes red on the run relay's stopping, as for any stop.
            self.stop(actor=OPERATOR, via="Stop")

    def confirm_close(self) -> bool:
        """Ask before closing ends a workflow in progress; True to go ahead.

        Only asked while one is running: any other close loses nothing, since
        pending edits are flushed on the way out. Confirming stops the run, the
        same as Stop Workflow, without waiting for it to wind down -- the worker
        may be waiting on this thread, and it is a daemon thread, so it cannot
        hold the process open.
        """
        if not self._session.runner.is_running:
            return True
        box = QMessageBox(self._host)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Workflow Running")
        box.setText("A workflow is running. Close anyway?")
        box.setInformativeText(
            "Closing stops the workflow and disconnects from the microscope."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec_() != QMessageBox.Yes:
            return False
        self.stop(actor=OPERATOR, via="closing the window")
        return True

    def _on_stopping(self) -> None:
        """A stop was asked of the run, from any source. A bound method rather
        than a lambda, so a stop from the App Server's thread is delivered on
        the GUI thread."""
        self._border.set_state("stopping")

    def _on_run_starting(self, request: RunRequest) -> None:
        """Any run, whoever started it (the Run click, the Grids view, the
        agent), before its thread starts: the border and the Stop button."""
        if request.kind == GRIDS:
            self._border.set_state("automated")
        else:
            self._border.set_state(self._running_border_state(request.task_names[0]))
        # The run writes the experiment from its own thread. Land any edit still
        # waiting in the editors first, so there is only ever one writer (FIB-683).
        self._flush_editors()
        # Show the Stop button immediately so the run is cancellable even while
        # waiting for a scheduled first task (before any task status arrives).
        self._set_running()

    def _set_running(self):
        """Show the stop button, and lock what a run owns.

        Called on every report of a run, so the start is the first call after
        `_set_idle`: that is when the last run's failure line goes."""
        if not self._run_active:
            self._run_active = True
            self._run_failures = []
            self._run_total = 0
            self._status_bar.dismiss_failure()
        self._controls.set_running(True)
        self._lock(True)

    def _set_idle(self):
        """Hide the stop button and show run button."""
        self._run_active = False
        self._status_bar.set_waiting(None)
        self._status_bar.set_run(None)
        self._controls.set_running(False)
        self._controls.set_attention(None)
        self._lock(False)

    def _running_border_state(self, task_name: Optional[str]) -> str:
        """The border for a running workflow: automated or supervised."""
        if task_name is None or not get_task_supervision(
            task_name, self._session.experiment
        ):
            return "automated"
        return "supervised"

    def _update_supervised_status(self) -> bool:
        """Update the supervised status chip for the current task."""
        task_name = self.current_task_name
        if task_name is None:
            return False
        supervised = get_task_supervision(task_name, self._session.experiment)
        mode = SUPERVISED if supervised else AUTOMATED
        agent = AGENT in self._session.deciders()
        self._controls.set_supervision(mode, task_name, agent=agent)
        return supervised

    def _on_supervised_status_clicked(self):
        """Toggle supervision for the current task, as the agent's
        ``set_supervision`` does: saved and on the record."""
        if self.current_task_name is None:
            return
        experiment = self._session.experiment
        protocol = experiment.task_protocol if experiment is not None else None
        if protocol is None:
            return
        supervised = (
            protocol.get_attention(self.current_task_name) is Attention.supervised
        )
        set_supervision(
            experiment,
            self.current_task_name,
            not supervised,
            self._session.record_event,
        )
        self._update_supervised_status()
        if self._session.runner.is_running:
            self._border.set_state(self._running_border_state(self.current_task_name))
        self._after_supervision()

    def _on_status(self, event: WorkflowStatusEvent) -> None:
        """What the run says without needing an answer: its step, its reports,
        and the waiting/supervised/running state, re-read on every event."""
        # What the run is doing now (a scheduled-start countdown, waiting on
        # decisions, loading a grid), under the task it is for; "" clears it.
        if event.status_bar is not None:
            self._status_bar.set_run_step(event.status_bar)
        if event.report is not None:
            self._show_report(event.report)
        self._refresh_indicators()

    def _show_report(self, report: WorkflowStatusUpdate) -> None:
        """One task-lifecycle report on the run line."""
        task_name = report.task_name
        lamella_name = report.item_name
        queue_position = report.queue_position
        queue_total = report.queue_total

        # Position in the live queue, not the launch matrix — stays correct
        # when the queue is added to or reordered mid-run.
        # `queue_total` is a count, so 0 means "nothing to be in the middle of"
        # rather than "unknown" -- truthiness, not `is not None`, or a malformed
        # payload renders "3 of 0".
        where = None
        if queue_position is not None and queue_total:
            where = f"{queue_position} of {queue_total}"

        self._status_bar.set_run(f"{lamella_name} › {task_name}", None, where)
        self._set_running()
        if queue_total:
            self._run_total = queue_total
        if report.status is TaskStatus.Failed:
            # Said for a moment under its own name -- the run moves on to the next
            # item -- and kept for the line the run leaves when it ends.
            what = f"{lamella_name} › {task_name}"
            reason = report.error_message or "failed"
            self._run_failures.append((what, reason))
            self._status_bar.show_outcome(
                what, f"failed: {reason}", failed=True, standalone=True
            )
        self.current_task_name = task_name

    def _refresh_indicators(self) -> None:
        """Re-read the waiting/supervised/running state into the window chrome."""
        self._update_supervised_status()

        hold = self._relay.hold
        self._attention.show_hold(hold)

        self._border.set_state(
            run_border_state(
                self._border.state,
                held=hold is not None,
                pending=self._relay.pending,
                running=self._running_border_state(self.current_task_name)
                if self._session.runner.is_running
                else None,
            )
        )

    def _on_finished(self, cancelled: bool = False) -> None:
        self._workflow_finished()
        self._set_idle()
        self._say_how_the_run_ended()
        self._border.set_state("idle")

    def _workflow_finished(self):
        """The panel's half of a run ending, then the summary. The replay reads
        this function's log line as the run's end."""
        logging.info("Workflow finished.")
        session = self._session
        if session.microscope is None:
            return
        experiment = session.experiment
        if experiment is None or experiment.task_protocol is None:
            return

        self._relay.hold = None
        self._relay.pending = False
        self._reset_panel()

        # show the post-workflow summary of tasks run this session
        self.show_summary()

    def _say_how_the_run_ended(self) -> None:
        """The run's line has gone (`_set_idle`). A run with failures, or one
        that stalled, leaves a line that stays until it is dismissed or the
        next run starts; a clean one says so once, as a toast, and the line goes
        back to the instruction. A run with a summary was already announced by
        the "Run ended" toast (`show_summary`), so this says nothing."""
        runner = self._session.runner
        note = runner.last_note
        failures = self._run_failures
        if failures:
            what, reason = failures[-1]
            n = len(failures)
            count = f"{n} of {self._run_total}" if self._run_total >= n else f"{n}"
            self._status_bar.show_failure(
                "Run finished", f"{count} failed · {what}: {reason}"
            )
        elif note:
            self._status_bar.show_failure("Run stalled", note)
        elif runner.last_summary is None or runner.last_summary.empty:
            notification_service.show_toast("Workflow finished.")

    def show_summary(self) -> None:
        """Toast the end of the last workflow, and open its summary unless the
        workflow's "Show summary after completion" is off.

        The toast always comes, with a Show summary link, and stays in the
        notification bell, so any run's summary is a click away after it
        fades. The dialog opens without blocking the window (FIB-1376): as a
        modal it held the window at the end of every run, which an agent
        starting many short runs, or a run ending overnight, left standing
        over the app. That is also when to turn it off.

        Show-once, but never consume: the runner's ``last_summary`` is also
        the record the App Server's ``run_summary`` endpoint reads after the
        manager is gone. The record stays until the next run's capture
        overwrites it; only the toast is once-only.
        """
        runner = self._session.runner
        summary = runner.last_summary
        if summary is None or summary is self._shown_run_summary:
            return
        self._shown_run_summary = summary
        if summary.empty:
            return
        note = runner.last_note
        failed = (
            "task_status" in summary.columns
            and (summary["task_status"] == "Failed").any()
        )
        notification_service.show(
            f"Run ended: {run_counts_text(summary)}",
            "warning" if note or failed else "success",
            action=(
                "Show summary",
                lambda: self._open_summary(summary, note),
            ),
        )
        experiment = self._session.experiment
        protocol = experiment.task_protocol if experiment is not None else None
        if protocol is None or protocol.options.show_run_summary:
            self._open_summary(summary, note)

    def _open_summary(self, summary: pd.DataFrame, note: str) -> None:
        """Open a run's summary dialog, without blocking the window."""
        try:
            dialog = WorkflowSummaryDialog(summary, note=note, parent=self._host)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.show()
        except Exception as e:
            logging.warning(f"Failed to show workflow summary dialog: {e}")
