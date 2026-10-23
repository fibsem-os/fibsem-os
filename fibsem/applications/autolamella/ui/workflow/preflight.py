"""What the app puts in front of a workflow: the lamella run, the grid run (and
Screen Grids' summary of one), and adding to a running queue, from one set of pieces.

It answers the question the user is holding: whether to start this now, and when to
come back.

Built from :mod:`fibsem.ui.widgets.preflight`'s pieces but not from
``OverviewPreflightDialog``: this confirms a different kind of thing, and its rows carry
chips and a right-aligned duration where ``detail_block`` has one left-aligned value.
The coincidence milling dialog does the same, for the same reason.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.workflows.plan import LamellaRunPlan
from fibsem.structures import BeamType
from fibsem.ui import stylesheets
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET, SECONDARY_BUTTON_STYLESHEET
from fibsem.ui.widgets import preflight
from fibsem.ui.widgets.preflight import (
    BACKGROUND,
    BORDER,
    ON_PANEL,
    PANEL,
    TEXT,
    TEXT_MUTED,
    TEXT_STRONG,
    chip,
    detail_block,
    meta_label,
    metric,
    warning_label,
)
from fibsem.ui.widgets.preflight import (
    format_clock as _clock,
)
from fibsem.util.durations import format_duration_rounded
from fibsem.workflows.attendance import (
    RunAttendance,
    run_attendance,
)
from fibsem.workflows.estimate import AdditionEstimate, RunEstimate, TaskEstimate
from fibsem.workflows.grid.manager import LOAD_ENTRY_NAME, plan_grid_run
from fibsem.workflows.grid.naming import beam_name, is_default_grid_name

# The duration column is fixed and right-aligned so the times hold one line down the
# dialog however many chips a row carries. That column is what gets scanned; a ragged
# right edge would break it.
_DURATION_WIDTH = 84

# How many lamella names the expanded list shows before it stops counting them out. A
# hundred-lamella experiment wrapped to a dozen lines and pushed the rest of the dialog
# off the bottom. A scroll area bounded it but put Qt's arrow-button scrollbar in the
# middle of otherwise flat chrome, to reach names nobody hunts for by scrolling -- so
# the list truncates and says how many it did not show instead.
_NAMES_SHOWN = 12

_ON_PANEL = ON_PANEL  # see preflight.ON_PANEL for why every label on a panel needs it


def _task_row(
    task: TaskEstimate,
    reference: Optional[datetime] = None,
    attendance: Optional[RunAttendance] = None,
) -> QWidget:
    """`Rough milling  ×2   [chips]   20m 24s`."""
    attendance = attendance or RunAttendance()
    row = QWidget()
    row.setStyleSheet(_ON_PANEL)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    name = QLabel(
        f"{task.name}  <span style='color: {TEXT_MUTED};'>×{task.count}</span>"
    )
    name.setStyleSheet(_ON_PANEL + f"color: {TEXT}; font-size: 11px;")
    # Task names are user-supplied and can be long. The name is what gives if a row
    # runs out of room -- the duration column is fixed and the chips are the point --
    # so it carries the full name where a clipped one would lose it silently.
    # The full name, and what the task needs from a person while it runs.
    line = attendance.lines.get(task.name, "")
    name.setToolTip(f"{task.name}\n{line}" if line else task.name)
    layout.addWidget(name, stretch=1)

    if task.name in attendance.supervised:
        layout.addWidget(chip("Supervised"))
    if task.scheduled_at is not None:
        layout.addWidget(chip(f"Scheduled {_clock(task.scheduled_at, reference)}"))

    duration = QLabel(format_duration_rounded(task.seconds, pad=True))
    duration.setStyleSheet(_ON_PANEL + f"color: {TEXT}; font-size: 11px;")
    duration.setFixedWidth(_DURATION_WIDTH)
    duration.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    layout.addWidget(duration)
    return row


def task_block(
    tasks: List[TaskEstimate],
    reference: Optional[datetime] = None,
    attendance: Optional[RunAttendance] = None,
) -> QFrame:
    frame = QFrame()
    # Scoped by name: the rows carry chips, and chips are QFrames, so a bare selector
    # would restyle them with this block's fill and border.
    frame.setObjectName("preflightTaskBlock")
    frame.setStyleSheet(
        f"QFrame#preflightTaskBlock {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 4px; }}"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)
    for task in tasks:
        layout.addWidget(_task_row(task, reference, attendance))
    return frame


def estimate_figures(
    estimate: Optional[RunEstimate] = None,
    addition: Optional[AdditionEstimate] = None,
) -> Optional[QWidget]:
    """The duration and the finish: a run's own, or what an addition adds and
    moves the finish to. None when there is nothing priced to quote."""
    if estimate is not None:
        first = metric(
            "Estimated duration",
            format_duration_rounded(estimate.work_seconds, pad=True),
        )
        second = metric(
            "Expected finish", _clock(estimate.expected_finish, estimate.started_at)
        )
    elif addition is not None and addition.is_priced:
        before = addition.finish_before
        first = metric(
            "Adds",
            format_duration_rounded(addition.work_seconds, pad=True),
            _priced_note(addition),
        )
        second = metric(
            "Expected finish",
            _clock(addition.finish_after, before),
            f"was {_clock(before, before)}",
        )
    else:
        return None
    row = QWidget()
    row.setStyleSheet("background: transparent;")
    figures = QHBoxLayout(row)
    figures.setContentsMargins(0, 0, 0, 0)
    figures.setSpacing(10)
    figures.addWidget(first, 1)
    figures.addWidget(second, 1)
    return row


def estimate_footnote(attendance: RunAttendance) -> QLabel:
    note = "Estimates round up."
    if attendance.waits_for_a_human:
        note += " Time spent waiting for you is not included."
    footnote = QLabel(note)
    footnote.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; {ON_PANEL}")
    footnote.setWordWrap(True)
    return footnote


class WorkflowPreflightDialog(QDialog):
    """The estimate, and the two exceptions that stop it being the whole story.

    Takes a finished :class:`RunEstimate` and :class:`RunAttendance` rather than an
    experiment: the arithmetic is tested without Qt, and this lays out what it is
    handed.
    """

    def __init__(
        self,
        estimate: RunEstimate,
        attendance: Optional[RunAttendance] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.estimate = estimate
        self.attendance = attendance or RunAttendance()
        self.setWindowTitle("Run Workflow")
        # Wide enough for a row carrying both chips: "Supervised" plus a scheduled
        # time dated to another day is the widest a row gets before the task name
        # starts giving way.
        self.setMinimumWidth(620)
        self.setStyleSheet(f"QDialog {{ background: {BACKGROUND}; }}")
        self._init_ui()

    def _init_ui(self) -> None:
        est = self.estimate
        att = self.attendance

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # No in-dialog heading: the window title already says "Run Workflow".
        n_tasks, n_lamella = len(est.tasks), len(est.item_names)
        layout.addWidget(
            meta_label(
                f"{n_tasks} task{'s' if n_tasks != 1 else ''} over "
                f"{n_lamella} lamella{'e' if n_lamella != 1 else ''}."
            )
        )

        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(chip(f"{est.step_count} steps"))
        if att.supervised:
            chips.addWidget(chip(f"{len(att.supervised)} supervised"))
        chips.addStretch()
        chips.addWidget(self._lamella_disclosure())
        layout.addLayout(chips)
        layout.addWidget(self._lamella_names_label)

        # Whether the operator has to be there: said before the run, from
        # what each selected task's type asks (attendance.py).
        self.attendance_label = QLabel(att.summary)
        self.attendance_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        self.attendance_label.setWordWrap(True)
        self.attendance_label.setVisible(bool(att.summary))
        layout.addWidget(self.attendance_label)

        layout.addWidget(estimate_figures(est))

        for note in self._exception_notes():
            layout.addWidget(note)

        layout.addWidget(task_block(est.tasks, est.started_at, att))

        layout.addWidget(estimate_footnote(att))

        self.button_run = QPushButton("Run")
        self.button_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_run.setMinimumHeight(30)
        self.button_run.clicked.connect(self.accept)
        button_cancel = QPushButton("Cancel")
        button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        button_cancel.setMinimumHeight(30)
        button_cancel.clicked.connect(self.reject)

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button_cancel)
        footer.addWidget(self.button_run)
        layout.addLayout(footer)

        if est.step_count == 0:
            # Nothing to do. The queue would build empty and the workflow would end
            # immediately; say so here rather than after the dialog is dismissed.
            self.button_run.setEnabled(False)
            self.button_run.setToolTip("No tasks are selected.")

    def _lamella_disclosure(self) -> QWidget:
        """`▸ 2 lamellae`, expanding to their names.

        The per-task breakdown shows `×2`; the names appear here, once, rather than
        repeated under every task.

        Builds the names label as a side effect and leaves it on the instance; the
        caller places it, because it belongs under the chip row rather than inside it.
        """
        names = QLabel(self._names_text())
        names.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        names.setWordWrap(True)
        names.setVisible(False)

        button = QToolButton()
        button.setText(f"{len(self.estimate.item_names)} lamellae")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.RightArrow)
        # Transparent explicitly: NAPARI_STYLE gives every QToolButton a
        # BORDER_COLOR fill and a radius, which reads as a raised control next to the
        # chips it sits beside.
        button.setStyleSheet(
            f"QToolButton {{ color: {TEXT_MUTED}; font-size: 11px; border: none;"
            f" background: transparent; padding: 2px 0px; }}"
        )

        def _toggle(checked: bool) -> None:
            button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            names.setVisible(checked)
            self.adjustSize()

        button.toggled.connect(_toggle)
        self._lamella_names_label = names
        return button

    def _names_text(self) -> str:
        """The names, truncated with a count of what is not shown.

        A selection of a hundred is confirmed by spot-checking a few and trusting the
        count, not by reading all of them -- and the count is what "+88 more" gives
        that a scrollbar hides.
        """
        names = self.estimate.item_names
        shown = "   ".join(names[:_NAMES_SHOWN])
        remaining = len(names) - _NAMES_SHOWN
        if remaining > 0:
            return f"{shown}   + {remaining} more"
        return shown

    def _exception_notes(self) -> List[QLabel]:
        """The two things that stop the headline figure being the whole story.

        Both summarise rather than list once there is more than one, because every row
        already carries its own chip: naming them again grows a sentence without adding
        anything, and three names joined by commas reads worse than a count.
        """
        notes: List[QLabel] = []

        # A hold is invisible in the duration, and someone reading only that figure
        # would be wrong by its length.
        scheduled = self.estimate.scheduled_tasks
        if len(scheduled) == 1:
            task = scheduled[0]
            notes.append(
                self._note(
                    f"{task.name} is scheduled for "
                    f"{_clock(task.scheduled_at, self.estimate.started_at)}. "
                    f"The workflow holds for about "
                    f"{format_duration_rounded(task.hold_seconds, pad=True)} before it."
                )
            )
        elif scheduled:
            notes.append(
                self._note(
                    f"{len(scheduled)} steps are scheduled. The workflow holds for "
                    f"about {format_duration_rounded(self.estimate.hold_seconds, pad=True)} in total, "
                    f"waiting for them."
                )
            )

        supervised = self.attendance.supervised
        if len(supervised) == 1:
            notes.append(
                self._note(
                    f"{supervised[0]} waits for you. The work is counted above; "
                    f"the wait is not."
                )
            )
        elif supervised:
            notes.append(
                self._note(
                    f"{len(supervised)} steps wait for you. Their work is counted "
                    f"above; the waiting is not."
                )
            )
        return notes

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"QLabel {{ color: {TEXT}; font-size: 11px; background: {PANEL};"
            f" border: none; border-left: 2px solid {stylesheets.WARN_COLOR};"
            f" padding: 7px 10px; }}"
        )
        label.setWordWrap(True)
        return label


def confirm_run_workflow_dialog(
    experiment: Experiment,
    plan: LamellaRunPlan,
    parent=None,
) -> bool:
    """Show the pre-flight estimate before starting the workflow.

    Returns True if confirmed. Replaces two columns of bullets that restated the
    selection the user had just made; see WorkflowPreflightDialog. The estimate
    is the plan's (`plan_lamella_run`), as the App Server's plan route gives it.
    """
    attendance = run_attendance(experiment.task_protocol, plan.request.task_names)
    dlg = WorkflowPreflightDialog(plan.estimate, attendance, parent=parent)
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

    figures = estimate_figures(addition=estimate)
    if figures is not None:
        layout.addWidget(figures)

    layout.addWidget(
        detail_block(
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


_STEPS_SHOWN = 12


def grid_run_summary(
    task_names: Sequence[str],
    grid_names: Sequence[str],
    exchanges: int,
    output_root: str,
    estimate: Optional[RunEstimate] = None,
    addition: Optional[AdditionEstimate] = None,
    attendance: Optional[RunAttendance] = None,
    beams_off: Sequence[BeamType] = (),
    first_run: Sequence[str] = (),
    rename_hint: str = "Rename on the Grids tab first if needed.",
    ending: Optional[Sequence[str]] = None,
) -> List[QWidget]:
    """What a grid run is about to do, top to bottom: the counts, how long and
    when it is done, the first steps of the plan, where the files go, and what
    the operator should know first. ``ending``: how the run ends, from the
    Grids page's ⚙; None when adding, which ends the way the run it joins was
    started."""
    attendance = attendance or RunAttendance()
    widgets: List[QWidget] = []

    counts = QWidget()
    counts.setStyleSheet("background: transparent;")
    row = QHBoxLayout(counts)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    row.addWidget(metric("Grids", str(len(grid_names))))
    row.addWidget(metric("Tasks per grid", str(len(task_names))))
    row.addWidget(
        metric("Exchanges", str(exchanges), "" if exchanges else "all loaded")
    )
    widgets.append(counts)

    figures = estimate_figures(estimate, addition)
    if figures is not None:
        widgets.append(figures)
        if estimate is not None and estimate.tasks:
            widgets.append(task_block(estimate.tasks, estimate.started_at, attendance))
        widgets.append(estimate_footnote(attendance))

    plan = plan_grid_run(list(task_names), list(grid_names))
    lines = [
        f"{grid}  ·  {'load' if step == LOAD_ENTRY_NAME else step}"
        for grid, step in plan[:_STEPS_SHOWN]
    ]
    if len(plan) > _STEPS_SHOWN:
        lines.append(f"… and {len(plan) - _STEPS_SHOWN} more steps")
    rows = [("Plan", lines[0])] + [("", line) for line in lines[1:]]
    widgets.append(detail_block(rows if lines else [("Plan", "no grids found")]))
    widgets.append(meta_label(f"Output: {output_root}/grids/<grid>/"))
    widgets.append(
        meta_label(
            "A grid that will not load is skipped and the run continues. "
            "Stop ends the run at the next step; during an exchange, before "
            "the load."
        )
    )

    # Screening starts on an empty stage, so the beams are off more often
    # than not. The run turns them on; said here so it is no surprise, and
    # so an operator who does not want that can cancel.
    if beams_off:
        names = " and ".join(beam_name(b) for b in beams_off)
        plural = len(beams_off) > 1
        widgets.append(
            warning_label(
                f"The {names} beam{'s are' if plural else ' is'} off. The run "
                f"turns {'them' if plural else 'it'} on when it starts."
            )
        )

    # A grid's name is fixed once it has run: its folder and images carry it.
    # This is the last point to change it, so the grids about to run for the
    # first time are named, and nothing is said when there are none.
    if first_run:
        first = ", ".join(
            f"{n} (default name)" if is_default_grid_name(n) else n for n in first_run
        )
        widgets.append(
            warning_label(
                f"Names are fixed once a grid has run. Running for the first "
                f"time: {first}. {rename_hint}"
            )
        )

    if ending is not None:
        widgets.append(
            meta_label(
                f"When the run ends: {' and '.join(ending)}."
                if ending
                else "When the run ends: nothing more. Set on the ⚙."
            )
        )
    return widgets


def run_ending(unload_at_end: bool, report_at_end: bool) -> List[str]:
    """How a grid run ends, as the confirmation says it."""
    return [
        said
        for said, on in (
            ("unload the last grid", unload_at_end),
            ("write the grid screening report", report_at_end),
        )
        if on
    ]


class GridRunPreflightDialog(QDialog):
    """The grid run's confirmation: `grid_run_summary` under a title, with Run
    (or Add, onto the end of a running queue)."""

    def __init__(
        self,
        task_names: List[str],
        grid_names: List[str],
        exchanges: int,
        output_root: str,
        adding: bool = False,
        beams_off: Sequence[BeamType] = (),
        first_run: Sequence[str] = (),
        estimate: Optional[RunEstimate] = None,
        addition: Optional[AdditionEstimate] = None,
        attendance: Optional[RunAttendance] = None,
        unload_at_end: bool = False,
        report_at_end: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        """``first_run``: grids in the plan that have never run, whose names
        this run will fix. ``estimate``: what the run will take, task by task
        (``estimate_grid_run``); ``addition``: what adding it to the running
        queue costs. Without either, no time is quoted. ``attendance``: the
        tasks that wait for a person (``grid_run_attendance``)."""
        super().__init__(parent)
        self.setWindowTitle("Add to the queue" if adding else "Run grid workflow")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"background: {BACKGROUND};")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel(
            "Add the tasks on the selected grids after everything queued"
            if adding
            else "Run the tasks on the selected grids"
        )
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {TEXT_STRONG}; {ON_PANEL}"
        )
        layout.addWidget(title)
        for widget in grid_run_summary(
            task_names,
            grid_names,
            exchanges,
            output_root,
            estimate=estimate,
            addition=addition,
            attendance=attendance,
            beams_off=beams_off,
            first_run=first_run,
            ending=None if adding else run_ending(unload_at_end, report_at_end),
        ):
            layout.addWidget(widget)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_run = QPushButton("Add" if adding else "Run")
        self.btn_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_run.clicked.connect(self.accept)
        self.btn_run.setDefault(True)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_run)
        layout.addLayout(buttons)
