"""The Workflow tab: what to run, the run's queue, and its timeline.

Lamella and Grids side by side on the left, one timeline on the right: the queue
is generic over items, and the two runs share the runner's one slot, so there is
one run at a time and one place to watch it. Run, Add to Queue and the
timeline's row actions all go through the session's runner, as the App Server's
do.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.acting import OPERATOR
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui.grid_workflow_widget import (
    GridRunPreflightDialog,
    GridWorkflowWidget,
)
from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
    LamellaWorkflowWidget,
)
from fibsem.applications.autolamella.ui.screen_grids_dialog import (
    ScreenGridsDialog,
    scan_in_progress,
)
from fibsem.applications.autolamella.ui.workflow_preflight_dialog import (
    WorkflowPreflightDialog,
)
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.estimate import (
    estimate_queue_addition,
    step_seconds,
    task_schedule,
)
from fibsem.applications.autolamella.workflows.tasks.attendance import (
    grid_run_attendance,
    run_attendance,
)
from fibsem.structures import BeamType
from fibsem.ui import notification_service
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import SURFACE_COLOR
from fibsem.ui.widgets import preflight
from fibsem.util.durations import format_duration_rounded
from fibsem.workflows.estimate import AdditionEstimate, estimate_run
from fibsem.workflows.grid.estimate import estimate_grid_run
from fibsem.workflows.grid.manager import plan_grid_run
from fibsem.workflows.grid.naming import grid_has_run
from fibsem.workflows.queue import QUEUE_ACTIONS
from fibsem.workflows.runner import (
    GRIDS,
    ITEMS,
    QueueEdit,
    RunRequest,
    item_task_pairs,
)


def confirm_run_workflow_dialog(
    experiment: Experiment,
    request: RunRequest,
    parent=None,
) -> bool:
    """Show the pre-flight estimate before starting the workflow.

    Returns True if confirmed. Replaces two columns of bullets that restated the
    selection the user had just made; see WorkflowPreflightDialog. The estimate
    walks the pairs the run will queue, priced from each lamella's task configs.
    """
    estimate = estimate_run(
        item_task_pairs(request), step_seconds(experiment), task_schedule(experiment)
    )
    attendance = run_attendance(experiment.task_protocol, request.task_names)
    dlg = WorkflowPreflightDialog(estimate, attendance, parent=parent)
    return dlg.exec_() == QDialog.Accepted


def confirm_add_to_queue_dialog(
    lamella_names: list,
    task_names: list,
    run_next: bool,
    already_queued: list,
    estimate: Optional[AdditionEstimate] = None,
    parent=None,
) -> bool:
    """Confirm adding work to a queue that is already running.

    Two figures, and they are allowed to disagree. "Adds" is machine time the addition
    costs; "Expected finish" is what that does to the workflow -- and work slotted in
    ahead of a scheduled hold is absorbed into dead time the workflow was going to spend
    anyway, so the first can be an hour while the second does not move. Quoting only the
    work would overstate the cost; quoting only the finish would hide that the machine
    is now busy for an hour it was not.

    ``already_queued`` is stated rather than acted on: a task that runs twice mills
    twice — the same mechanism that makes "Run again" useful — so the consequence is
    named and the operator decides. Silently adding four of the six pairs asked for
    would be its own surprise.

    ``estimate`` is optional and may be unpriced. A protocol whose configs cannot be
    estimated still has to be able to queue work, so the figures are dropped rather than
    shown as a confident zero.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Add to Queue")
    dlg.setMinimumWidth(560)
    dlg.setStyleSheet(f"QDialog {{ background: {preflight.BACKGROUND}; }}")

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(18, 16, 18, 14)
    layout.setSpacing(12)

    total = len(lamella_names) * len(task_names)
    heading = QLabel(f"Add {total} task(s) to a workflow that is already running?")
    heading.setStyleSheet(f"color: {preflight.TEXT_STRONG}; font-size: 14px;")
    layout.addWidget(heading)

    if estimate is not None and estimate.is_priced:
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        metrics.addWidget(
            preflight.metric(
                "Adds",
                format_duration_rounded(estimate.work_seconds, pad=True),
                _priced_note(estimate),
            ),
            1,
        )
        reference = estimate.finish_before
        metrics.addWidget(
            preflight.metric(
                "Expected finish",
                preflight.format_clock(estimate.finish_after, reference),
                f"was {preflight.format_clock(estimate.finish_before, reference)}",
            ),
            1,
        )
        layout.addLayout(metrics)

    layout.addWidget(
        preflight.detail_block(
            [
                ("Position", "Run next" if run_next else "At the end of the queue"),
                ("Lamella", ", ".join(lamella_names)),
                ("Tasks", ", ".join(task_names)),
            ]
        )
    )

    absorbed = _absorbed_note(estimate)
    if absorbed:
        note = QLabel(absorbed)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {preflight.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(note)

    if already_queued:
        # Body text, no accent colour anywhere. This is a consequence to notice rather
        # than an alarm: the count is bold enough to catch and Cancel is right there,
        # while three lines in a warning colour out-shouted the two figures that are
        # supposed to lead the dialog. Body rather than muted because it is content, not
        # the secondary shade the detail-block labels use.
        warning = QLabel(
            f"<b>{len(already_queued)} already queued</b> and will be added again, "
            f"so those tasks will run twice:<br>"
            + "<br>".join(f"• {n}" for n in already_queued[:6])
            + ("<br>• …" if len(already_queued) > 6 else "")
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {preflight.TEXT}; font-size: 11px;")
        layout.addWidget(warning)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    yes_btn = QPushButton(f"Add {total} task(s)")
    no_btn = QPushButton("Cancel")
    yes_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
    no_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
    yes_btn.clicked.connect(dlg.accept)
    no_btn.clicked.connect(dlg.reject)
    no_btn.setDefault(True)
    btn_row.addWidget(no_btn)
    btn_row.addWidget(yes_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


def _priced_note(estimate: AdditionEstimate) -> str:
    """The task count, and how much of it the figure above actually covers.

    A partly-priced addition would otherwise read as a complete one that happens to be
    quick — a lamella missing a config for one of the tasks contributes nothing, by the
    same rule the pre-flight dialog and the timeline follow.
    """
    tasks = f"{estimate.total_count} task(s)"
    if estimate.priced_count == estimate.total_count:
        return tasks
    return f"{tasks} · {estimate.priced_count} estimated"


def _absorbed_note(estimate: Optional[AdditionEstimate]) -> str:
    """Said only when the two figures disagree, because then they look wrong.

    An hour of work that moves the finish by nothing is the correct answer and reads as
    an error, so the reason goes on screen with it.

    Two guards, and both are needed. **There has to be a scheduled wait**, because that
    is what the sentence claims -- and `delay_seconds` comes back through a datetime
    (microseconds) while `work_seconds` is a difference of two float sums, so for the
    same quantity they can disagree in the twelfth decimal place. That was enough to
    explain a wait that did not exist on a queue with nothing scheduled at all.

    **And the disagreement has to be one the reader can see.** Both figures are rendered
    with `format_duration_rounded`, so a gap that rounds away is not a discrepancy needing an
    explanation -- it is two identical numbers with a paragraph between them.
    """
    if estimate is None or not estimate.is_priced:
        return ""
    if estimate.hold_seconds <= 0 or estimate.work_seconds <= 0:
        return ""
    if format_duration_rounded(
        estimate.delay_seconds, pad=True
    ) == format_duration_rounded(estimate.work_seconds, pad=True):
        return ""
    if estimate.delay_seconds <= 0:
        return (
            "The workflow is already waiting for a scheduled task, and this work "
            "fits inside that wait — so it costs no extra time overall."
        )
    return (
        f"Only {format_duration_rounded(estimate.delay_seconds, pad=True)} of this lands after "
        "the workflow's scheduled wait; the rest fits inside it."
    )


class WorkflowTab(QWidget):
    def __init__(
        self,
        session,
        *,
        controls,
        flush_edits: Callable[[], None],
        on_config_changed: Callable[[object], None],
        stop_task: Callable[..., None],
        after_naming: Callable[[], None],
        say: Callable[[str], None],
        show_toast: Callable[..., None],
        parent: Optional[QWidget] = None,
    ) -> None:
        """*session* holds the experiment, the microscope and the runner;
        *controls* is the status bar's Run button; *flush_edits* lands an edit
        still waiting in the editors before a run writes the experiment;
        *on_config_changed* saves a changed workflow; *after_naming* refreshes
        what shows the grids once a scan has named them; *say* puts what the
        selection allows on the status line."""
        super().__init__(parent)
        self._session = session
        self._controls = controls
        self._flush_edits = flush_edits
        self._on_config_changed = on_config_changed
        self._stop_task = stop_task
        self._after_naming = after_naming
        self._say = say
        self._show_toast = show_toast

        self.timeline = WorkflowProgressWidget()
        self.timeline.queue_action_requested.connect(self._on_queue_action)
        self.timeline.add_to_queue_requested.connect(self._on_add_to_queue)

        self.lamella_workflow_widget = LamellaWorkflowWidget()
        # Each task or workflow change is saved to the experiment's protocol.
        for signal in (
            self.lamella_workflow_widget.task_attention_changed,
            self.lamella_workflow_widget.task_edited,
            self.lamella_workflow_widget.task_remove_requested,
            self.lamella_workflow_widget.task_added,
            self.lamella_workflow_widget.workflow_name_changed,
            self.lamella_workflow_widget.workflow_description_changed,
            self.lamella_workflow_widget.workflow_options_changed,
        ):
            signal.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_order_changed.connect(
            self._on_workflow_task_order_changed
        )
        self.lamella_workflow_widget.lamella_selection_changed.connect(
            self.selection_changed
        )
        self.lamella_workflow_widget.task_selection_changed.connect(
            self.selection_changed
        )

        self.grid_workflow_widget = GridWorkflowWidget()
        self.grid_workflow_widget.selection_changed.connect(self.selection_changed)
        self.grid_workflow_widget.screen_all_requested.connect(
            self._on_screen_all_grids
        )

        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self.lamella_workflow_widget, "Lamella")
        self.left_tabs.addTab(self.grid_workflow_widget, "Grids")
        self.left_tabs.currentChanged.connect(self.selection_changed)

        right = QWidget()
        right.setStyleSheet(f"background: {SURFACE_COLOR};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.timeline)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.left_tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)

        controls.run_clicked.connect(self.run)
        session.runner.starting.connect(self._on_run_starting)

    def set_running(self, running: bool) -> None:
        """A live run is exactly when there is a queue to edit, so the
        timeline's actions are on with it. Whether the front view's selection
        can join a run depends on which kind of run it is, so the Add button is
        read again."""
        self.timeline.set_actions_enabled(running)
        if running:
            self.selection_changed()

    def _on_run_starting(self, request: RunRequest) -> None:
        """Any run, whoever started it: the plan's steps, priced for the
        timeline. Screen all grids builds its queue after the inventory, so it
        is priced for the grids known now."""
        if request.kind == GRIDS:
            known = request.names
            experiment = self._session.experiment
            if known is None and experiment is not None:
                known = [g.name for g in experiment.grids]
            self._push_timeline_estimates(
                plan_grid_run(list(request.task_names), list(known or []))
            )
        else:
            self._push_timeline_estimates(
                [(ln, tn) for tn in request.task_names for ln in request.names or ()]
            )

    def run(self):
        """Run the workflow using the lamella and task selections from the workflow widget."""
        ui = self._session
        if ui.runner.is_running:
            return
        if (
            ui.microscope is None
            or ui.experiment is None
            or ui.experiment.task_protocol is None
        ):
            return

        if self.grid_view_active():
            self._run_grid_workflow()
            return

        selected_tasks = self.lamella_workflow_widget.get_selected_tasks()
        selected_lamella = self.lamella_workflow_widget.get_selected_lamella()

        if not selected_tasks or not selected_lamella:
            return

        task_names = [t.name for t in selected_tasks]
        lamella_names = [lam.name for lam in selected_lamella]
        request = RunRequest(
            kind=ITEMS, task_names=tuple(task_names), names=tuple(lamella_names)
        )

        # Before the estimate and the confirmation: a linked lamella whose grid
        # is in the magazine cannot be run at all, and there is no override.
        checked = ui.runner.check(ui.experiment, request)
        if not checked.started:
            QMessageBox.warning(self, "Cannot run", checked.reason)
            return

        if not confirm_run_workflow_dialog(ui.experiment, request, parent=self):
            return

        started = ui.runner.start(ui.experiment, request)
        if not started.started:
            self._show_toast(started.reason, "warning")
            return
        # Clear selections after starting workflow
        self.lamella_workflow_widget.lamella_list.set_all_selected(False)
        self.lamella_workflow_widget.workflow.set_all_selected(False)

    def grid_view_active(self) -> bool:
        """Whether the Workflow tab's Grids view is the one showing: Run acts on it."""
        return self.left_tabs.currentWidget() is self.grid_workflow_widget

    def _run_grid_workflow(self) -> None:
        """The Run click on the Grids view: confirm, then the grid run."""
        ui = self._session
        grids = self.grid_workflow_widget.get_selected_grids()
        task_names = self.grid_workflow_widget.get_selected_task_names()
        if not grids or not task_names or ui.experiment is None:
            return
        grid_names = [g.name for g in grids]
        exchanges = self.grid_workflow_widget.exchanges_for(grids)
        ending = self._end_of_run()
        dialog = GridRunPreflightDialog(
            task_names,
            grid_names,
            exchanges,
            str(ui.experiment.path),
            beams_off=self._beams_off(),
            first_run=[g.name for g in grids if not grid_has_run(g)],
            estimate=self._grid_run_estimate(task_names, grid_names, exchanges),
            attendance=self._grid_run_attendance(task_names),
            **ending,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._start_grid_run(
            task_names,
            grid_names,
            inventory_first=False,
            **ending,
        )

    def _on_screen_all_grids(self) -> None:
        """One click: read the magazine, scan and name the grids in one dialog,
        then the ticked tasks on every grid it found (FIB-898, FIB-1138)."""
        ui = self._session
        if ui.runner.is_running or ui.experiment is None:
            return
        if ui.microscope is None or ui.experiment.task_protocol is None:
            return
        task_names = self.grid_workflow_widget.get_selected_task_names()
        if not task_names:
            return
        if scan_in_progress():
            notification_service.show_toast(
                "The magazine scan started from Screen all grids is still running.",
                "warning",
            )
            return
        # The run writes the experiment from its own thread, and Screen writes
        # the grid records: land any edit still in the editors first.
        self._flush_edits()
        ending = self._end_of_run()
        dialog = ScreenGridsDialog(
            ui.microscope.sample_handling,
            ui.experiment,
            task_names,
            str(ui.experiment.path),
            beams_off=self._beams_off(),
            # priced on the grids the scan found, not the ones known before it
            estimate=lambda names, exchanges: self._grid_run_estimate(
                task_names, names, exchanges
            ),
            **ending,
            parent=self,
        )
        accepted = dialog.exec_() == QDialog.Accepted
        # A scan or a Screen changes what the slots hold and which grids the
        # experiment records, whether or not a run follows.
        self._after_naming()
        if not accepted or not dialog.grid_names:
            return
        self._start_grid_run(
            task_names,
            dialog.grid_names,
            inventory_first=False,
            **ending,
        )

    def _end_of_run(self) -> dict:
        """How a grid run ends, as the Grids page's ⚙ sets it: unload the last
        grid, write the screening report. Read once at the click, so the run
        ends the way its confirmation said."""
        view = self.grid_workflow_widget
        return {
            "unload_at_end": view.unload_at_end(),
            "report_at_end": view.report_at_end(),
        }

    def _beams_off(self) -> list:
        """The beams that are off now, which the run will turn on: the preflight
        says so. A read at the click, about to commit to a run: the one time the
        GUI asks the hardware."""
        microscope = self._session.microscope
        if microscope is None:
            return []
        off = []
        for beam in (BeamType.ELECTRON, BeamType.ION):
            try:
                if not microscope.is_on(beam):
                    off.append(beam)
            except Exception as e:  # noqa: BLE001 - unknown is not "off"
                logging.warning(
                    f"Could not read whether the {beam.name} beam is on: {e}"
                )
        return off

    def _start_grid_run(
        self,
        task_names: list,
        grid_names,
        inventory_first: bool,
        unload_at_end: bool = False,
        report_at_end: bool = False,
    ) -> None:
        ui = self._session
        started = ui.runner.start(
            ui.experiment,
            RunRequest(
                kind=GRIDS,
                task_names=tuple(task_names),
                names=tuple(grid_names) if grid_names is not None else None,
                options={
                    "inventory_first": inventory_first,
                    "unload_at_end": unload_at_end,
                    "report_at_end": report_at_end,
                },
            ),
        )
        if not started.started:
            self._show_toast(started.reason, "warning")
            return
        # Clear the grid ticks, as a lamella run clears its selection: left ticked,
        # the grids this run is on would be queued again by the next Add to Queue.
        # The task ticks stay; Screen all grids and the next add read them.
        self.grid_workflow_widget.set_all_grids_selected(False)

    def _run_refuses_selection(self) -> str:
        """Why the left panel's selection cannot join the running queue, or "".

        A run has one manager, lamella or grid, and the Add button commits
        whichever view is in front. A grid selection has nothing to join while
        a lamella run is going, and the other way round; the button says so
        rather than offering an add that the handler would refuse.
        """
        runner = self._session.runner
        if runner.manager is None:
            return ""
        return runner.joins(GRIDS if self.grid_view_active() else ITEMS)

    def selection_changed(self, _=None) -> None:
        """Enable the run button only when at least one lamella and one task are selected."""
        refused = self._run_refuses_selection()
        if self.grid_view_active():
            n_grid = len(self.grid_workflow_widget.get_selected_grids())
            n_task = len(self.grid_workflow_widget.get_selected_task_names())
            valid = n_grid > 0 and n_task > 0
            self._controls.set_run_enabled(
                valid,
                f"Run grid workflow: {n_grid} grid{'s' if n_grid != 1 else ''}, "
                f"{n_task} task{'s' if n_task != 1 else ''}"
                if valid
                else "Select a present grid and a task to run the grid workflow",
            )
            self.timeline.set_add_enabled(
                valid and not refused,
                refused
                or (
                    f"Add to the end of the queue: {n_grid} grid"
                    f"{'s' if n_grid != 1 else ''}, {n_task} task"
                    f"{'s' if n_task != 1 else ''}"
                    if valid
                    else "Select a present grid and a task to add to the queue"
                ),
            )
            self._say(
                f"Ready to run: {self.grid_workflow_widget.selection_summary()}."
                if valid
                else "Select a present grid and a task to run the grid workflow."
            )
            return
        n_lam = len(self.lamella_workflow_widget.get_selected_lamella())
        n_task = len(self.lamella_workflow_widget.get_selected_tasks())
        valid = n_lam > 0 and n_task > 0
        if valid:
            tooltip = f"Run workflow: {n_lam} lamella, {n_task} task{'s' if n_task != 1 else ''}"
        else:
            missing = []
            if n_lam == 0:
                missing.append("a lamella")
            if n_task == 0:
                missing.append("a task")
            tooltip = f"Select {' and '.join(missing)} to run the workflow"
        self._controls.set_run_enabled(valid, tooltip)

        # The timeline's Add button commits the same selection, so it follows the
        # same rule — there is nothing to add until something is ticked.
        if valid:
            tip = (
                f"Add to queue: {n_lam} lamella, "
                f"{n_task} task{'s' if n_task != 1 else ''}"
            )
        else:
            tip = f"Select {' and '.join(missing)} to add to the queue"
        self.timeline.set_add_enabled(valid and not refused, refused or tip)
        self._say(
            f"Ready to run: {n_lam} lamella, {n_task} task{'s' if n_task != 1 else ''}."
            if valid
            else f"{tooltip}."
        )

    def _save_workflow_config(self, *_args):
        """Persist the current task list to the experiment protocol after any task change."""
        if self._session.experiment is None:
            return
        protocol = self._session.experiment.task_protocol
        if protocol is None:
            return
        protocol.workflow_config.tasks[:] = self.lamella_workflow_widget.get_tasks()
        self._on_config_changed(protocol.workflow_config)

    def _on_workflow_task_order_changed(self, tasks) -> None:
        """A drag put the workflow's tasks in a new order: into the protocol
        through `AutoLamellaWorkflowConfig.reorder`, as the App Server's reorder
        does. A task dragged above one it requires is put back, and says why."""
        experiment = self._session.experiment
        if experiment is None or experiment.task_protocol is None:
            return
        config = experiment.task_protocol.workflow_config
        why = config.reorder([task.name for task in tasks])
        if why:
            notification_service.show_toast(why, "warning")
            # After the drop has finished, not inside it.
            QTimer.singleShot(
                0, lambda: self.lamella_workflow_widget.set_workflow_config(config)
            )
            return
        self._on_config_changed(config)

    def _on_add_to_queue(self, run_next: bool) -> None:
        """Add the left panel's current selection to the running queue.

        The selection UI is LamellaWorkflowWidget, which stays live during a run
        and is cleared when one is launched — so it is empty and free mid-run.
        Reusing it avoids a second lamella-and-task picker that would have to be
        kept in step with the first. The queue is the runner's, which checks
        the addition as the App Server's is checked.
        """
        runner = self._session.runner
        manager = runner.manager
        if manager is None:
            return
        if self.grid_view_active():
            self._add_grids_to_queue(manager)
            return

        lamellae = self.lamella_workflow_widget.get_selected_lamella()
        tasks = self.lamella_workflow_widget.get_selected_tasks()
        if not lamellae or not tasks:
            self._show_queue_message(
                "Select at least one lamella and one task to add to the queue."
            )
            return

        lamella_names = [lam.name for lam in lamellae]
        task_names = [t.name for t in tasks]
        request = RunRequest(
            kind=ITEMS, task_names=tuple(task_names), names=tuple(lamella_names)
        )
        plan = runner.plan_addition(request, run_next)
        if isinstance(plan, QueueEdit):
            QMessageBox.warning(self, "Cannot add to the queue", plan.reason)
            return

        pending = {(i.item_name, i.task_name) for i in manager.queue.pending}
        already = [f"{t} for {ln}" for ln, t in plan.pairs if (ln, t) in pending]
        if not confirm_add_to_queue_dialog(
            lamella_names,
            task_names,
            run_next,
            already,
            estimate=self._estimate_addition(manager, list(plan.pairs), run_next),
            parent=self,
        ):
            return

        edit = runner.add(request, run_next, actor=OPERATOR)
        if not edit.applied:
            self._show_queue_message(edit.reason)
            return
        where = "to run next" if run_next else "to the queue"
        self._show_queue_message(f"Added {len(edit.items)} task(s) {where}.")

        # Clear the selection, as starting a workflow does — the work is committed,
        # and leaving it ticked invites adding the same batch twice.
        self.lamella_workflow_widget.lamella_list.set_all_selected(False)
        self.lamella_workflow_widget.workflow.set_all_selected(False)

    def _add_grids_to_queue(self, manager) -> None:
        """The Grids view's selection onto the end of the running grid queue:
        each grid's load, then its tasks (see ``plan_grid_addition``)."""
        runner = self._session.runner
        grids = self.grid_workflow_widget.get_selected_grids()
        task_names = self.grid_workflow_widget.get_selected_task_names()
        if not grids or not task_names:
            self._show_queue_message("Select at least one grid and one task to add.")
            return
        grid_names = [g.name for g in grids]
        request = RunRequest(
            kind=GRIDS, task_names=tuple(task_names), names=tuple(grid_names)
        )
        plan = runner.plan_addition(request)
        if isinstance(plan, QueueEdit):
            self._show_queue_message(plan.reason)
            return
        experiment = self._session.experiment
        dialog = GridRunPreflightDialog(
            task_names,
            grid_names,
            self.grid_workflow_widget.exchanges_for(grids),
            str(experiment.path) if experiment is not None else "",
            adding=True,
            first_run=[g.name for g in grids if not grid_has_run(g)],
            addition=self._estimate_addition(manager, list(plan.pairs), run_next=False),
            attendance=self._grid_run_attendance(task_names),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        edit = runner.add(request, actor=OPERATOR)
        if not edit.applied:
            self._show_queue_message(edit.reason)
            return
        self._show_queue_message(f"Added {len(edit.items)} step(s) to the queue.")
        self.grid_workflow_widget.set_all_grids_selected(False)

    def on_queue_changed(self, info: dict) -> None:
        """The queue was edited between tasks — no task lifecycle to hang it off."""
        items = info.get("queue_items", [])
        # An added item can be a (lamella, task) pair the launch matrix never held, so
        # the estimates are recomputed here rather than only at the start.
        self._push_timeline_estimates([(i.item_name, i.task_name) for i in items])
        self.timeline.refresh_queue(items)

    def _estimate_addition(self, manager, pairs: list, run_next: bool):
        """What adding `pairs` would cost the running queue, or None if it cannot
        say: the same walk the pre-flight dialog and the timeline use."""
        experiment = self._session.experiment
        if experiment is None:
            return None
        return estimate_queue_addition(
            experiment, manager.queue, pairs, run_next, self._step_seconds(experiment)
        )

    def _step_seconds(self, experiment):
        """What a step costs, with the loader's exchange and the grids on the
        stage as the Grids view knows them."""
        view = self.grid_workflow_widget
        return step_seconds(
            experiment, view.exchange_seconds(), view.loaded_grid_names()
        )

    def _grid_run_estimate(self, task_names: list, grid_names: list, exchanges: int):
        """What a grid run will take, for its confirmation; None if it cannot say.
        A figure is not worth losing the dialog over (see `_push_timeline_estimates`)."""
        experiment = self._session.experiment
        if experiment is None:
            return None
        try:
            return estimate_grid_run(
                experiment,
                task_names,
                grid_names,
                exchanges,
                self.grid_workflow_widget.exchange_seconds(),
            )
        except Exception:
            logging.warning("Could not estimate the grid run.", exc_info=True)
            return None

    def _grid_run_attendance(self, task_names: list):
        experiment = self._session.experiment
        if experiment is None:
            return None
        return grid_run_attendance(experiment.grid_protocol, task_names)

    def _push_timeline_estimates(self, pairs: list) -> None:
        """Give the timeline the per-item durations, and the schedule they walk past.

        Computed here rather than inside the widget: an estimate comes from a lamella's
        task config, and the timeline is generic — it has no business reaching for an
        Experiment. Recomputed on workflow start and on a queue edit only, never per
        status update: `estimated_duration` walks every milling stage of every task, and
        the status path is the one already watched for latency (FIB-683).
        """
        experiment = self._session.experiment
        if experiment is None:
            return

        # A step with no config for the task, or one that cannot be priced, has
        # no duration to offer, and the column leaves it empty (see step_seconds).
        seconds_for = self._step_seconds(experiment)
        estimates = {}
        for item_name, task_name in pairs:
            seconds = seconds_for(item_name, task_name)
            if seconds is not None:
                estimates[(item_name, task_name)] = seconds
        self.timeline.set_estimates(estimates)
        self.timeline.set_schedule(task_schedule(experiment))

    def _on_queue_action(self, action: str, item_id: str) -> None:
        """Apply a timeline row action to the live queue.

        The timeline emits an intent keyed on WorkItem.id and nothing more; the
        queue is reached from here because the widget is generic and has no
        business knowing what a queue is.
        """
        manager = self._session.runner.manager
        if manager is None:
            return

        queue = manager.queue
        item = next((i for i in queue.items if i.id == item_id), None)
        if item is None:
            self._show_queue_message("That task is no longer in the queue.")
            return
        label = f"{item.task_name} for {item.item_name}"

        if action == "stop_task":
            # Confirmed, unlike Remove: that edits a list, this interrupts an
            # operation already under way on the sample.
            # Built rather than QMessageBox.question(...): the difference from Stop
            # is the point of asking at all, and everything in the text slot renders
            # at question weight. The explanation belongs in the informative slot.
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Stop Task")
            box.setText(f"Stop <b>{label}</b>?")
            box.setInformativeText(
                "It will be marked cancelled and the workflow will carry on with "
                "the next queued task. Use Stop to end the whole run instead."
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)
            if box.exec_() != QMessageBox.Yes:
                return
            self._stop_task(actor=OPERATOR, via="Stop Task", task=label)
            # No queue mutation here: the task unwinds on the worker thread and the
            # status it emits on the way out is what updates the timeline.
            self._show_queue_message(f"Stopping {label}...")
            return

        runner = self._session.runner
        if action == "run_again":
            # A fresh item rather than a rewind: the original attempt still
            # happened and stays in the run record. The same checks as the App
            # Server's requeue: a lamella on a grid not on the stage is refused.
            edit = runner.requeue(
                item.item_name, item.task_name, front=True, actor=OPERATOR
            )
        elif action in QUEUE_ACTIONS:
            edit = runner.edit(item.item_name, item.task_name, action, actor=OPERATOR)
        else:
            logging.warning(f"Unknown queue action from the timeline: {action}")
            return
        self._show_queue_message(self._queue_action_message(action, item, edit))

    @staticmethod
    def _queue_action_message(action: str, item, edit: QueueEdit) -> str:
        """Describe the outcome of a queue action on ``item``, or "" to say nothing.

        The row may have started running between the menu opening and the click,
        which is why the runner answers with why not rather than a bool — say
        so instead of appearing to do nothing.
        """
        if not edit.applied:
            return edit.reason
        if not edit.changed:
            return ""  # already first or last; the user can see that
        label = f"{item.task_name} for {item.item_name}"
        return {
            "run_again": f"Queued {label} to run next.",
            "move_up": f"Moved {label} up.",
            "move_down": f"Moved {label} down.",
            "run_next": f"{label} will run next.",
            "remove": f"Removed {label} from the queue.",
        }.get(action, "")

    def _show_queue_message(self, message: str) -> None:
        """Transient confirmation for a queue edit — no modal, no interruption.

        A toast, not the status bar: there it covered the run's own line."""
        if message:
            notification_service.show_toast(message)
