"""What the app puts in front of a workflow.

The dialog that used to ask "Run workflow for 2 lamella with 7 task(s)?" over two
columns of bullets. The lists restated the selection the user had just made; what they
could not answer was the question they were actually holding -- whether to start this
now, and when to come back.

Built from :mod:`fibsem.ui.widgets.preflight`'s pieces but not from
``OverviewPreflightDialog``: this confirms a different kind of thing, and its rows carry
chips and a right-aligned duration where ``detail_block`` has one left-aligned value.
The same call the coincidence milling dialog made, for the same reason.

See FIB-666.
"""

from datetime import datetime
from typing import List, Optional

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

from fibsem.constants import DATETIME_DISPLAY_AMPM, TIME_DISPLAY_AMPM_SHORT
from fibsem.applications.autolamella.workflows.workflow_estimate import (
    TaskEstimate,
    WorkflowEstimate,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.preflight import (
    BACKGROUND,
    BORDER,
    PANEL,
    TEXT,
    TEXT_MUTED,
    TEXT_STRONG,
    chip,
    format_duration,
    meta_label,
)

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

# Anything sitting on a panel has to say it is transparent. The app applies
# NAPARI_STYLE, whose bare `QWidget { background-color: ... }` rule reaches every
# descendant, so a label that only sets `color` paints the *surface* colour onto the
# darker panel behind it -- a lighter block around every word. Only visible with the
# app stylesheet loaded, which is why the offscreen renders looked clean.
_ON_PANEL = "background: transparent; border: none; "


def _clock(when: Optional[datetime], reference: Optional[datetime]) -> str:
    """A time the user can act on: `6:15AM` today, `Tomorrow, 6:15AM` overnight.

    The day is not decoration. An overnight workflow is exactly the case this dialog
    exists for, and a bare clock time for one finishing the next morning is wrong by a
    day in the direction that matters -- `_wait_until_scheduled` dates its countdown for
    the same reason. But a date stamp reads as a filename where a person would just say
    "tomorrow", so the two days either side of today get the word instead.

    Anything further out keeps the dated form: "in 3 days" is arithmetic the reader has
    to do, and a workflow reaching that far is rare enough not to optimise for.
    """
    if when is None:
        return "—"
    # 6:15AM, not 06:15AM -- the leading zero is noise at this size.
    time_str = when.strftime(TIME_DISPLAY_AMPM_SHORT).lstrip("0")
    if reference is None:
        return time_str
    days = (when.date() - reference.date()).days
    if days == 0:
        return time_str
    if days == 1:
        return f"Tomorrow, {time_str}"
    if days == -1:
        return f"Yesterday, {time_str}"
    return when.strftime(DATETIME_DISPLAY_AMPM)


def _metric(label_text: str, value_text: str) -> QWidget:
    """One of the two figures that lead the dialog.

    Two, not one: the duration answers "is this worth starting", the wall-clock finish
    answers "when do I come back", and the second is the one people act on.
    """
    frame = QFrame()
    # Scoped to this frame by name. A bare `QFrame { ... }` selector also matches every
    # QFrame *inside* it, and preflight.chip() is one -- which repainted each chip with
    # the panel's fill on top of its own.
    frame.setObjectName("preflightMetric")
    frame.setStyleSheet(
        f"QFrame#preflightMetric {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 4px; }}"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 9, 12, 9)
    layout.setSpacing(2)

    caption = QLabel(label_text)
    caption.setStyleSheet(_ON_PANEL + f"color: {TEXT_MUTED}; font-size: 11px;")
    value = QLabel(value_text)
    value.setStyleSheet(_ON_PANEL + f"color: {TEXT_STRONG}; font-size: 19px;")
    layout.addWidget(caption)
    layout.addWidget(value)
    return frame


def _task_row(task: TaskEstimate, reference: Optional[datetime] = None) -> QWidget:
    """`Rough milling  ×2   [chips]   20m 24s`."""
    row = QWidget()
    row.setStyleSheet(_ON_PANEL)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    name = QLabel(f"{task.name}  <span style='color: {TEXT_MUTED};'>×{task.lamella_count}</span>")
    name.setStyleSheet(_ON_PANEL + f"color: {TEXT}; font-size: 11px;")
    # Task names are user-supplied and can be long. The name is what gives if a row
    # runs out of room -- the duration column is fixed and the chips are the point --
    # so it carries the full name where a clipped one would lose it silently.
    name.setToolTip(task.name)
    layout.addWidget(name, stretch=1)

    if task.supervised:
        layout.addWidget(chip("Supervised"))
    if task.scheduled_at is not None:
        layout.addWidget(chip(f"Scheduled {_clock(task.scheduled_at, reference)}"))

    duration = QLabel(format_duration(task.seconds))
    duration.setStyleSheet(_ON_PANEL + f"color: {TEXT}; font-size: 11px;")
    duration.setFixedWidth(_DURATION_WIDTH)
    duration.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    layout.addWidget(duration)
    return row


def _task_block(tasks: List[TaskEstimate], reference: Optional[datetime] = None) -> QFrame:
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
        layout.addWidget(_task_row(task, reference))
    return frame


class WorkflowPreflightDialog(QDialog):
    """The estimate, and the two exceptions that stop it being the whole story.

    Takes a finished :class:`WorkflowEstimate` rather than an experiment: the
    arithmetic is tested without Qt, and this lays out what it is handed.
    """

    def __init__(self, estimate: WorkflowEstimate, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.estimate = estimate
        self.setWindowTitle("Run Workflow")
        # Wide enough for a row carrying both chips: "Supervised" plus a scheduled
        # time dated to another day is the widest a row gets before the task name
        # starts giving way.
        self.setMinimumWidth(620)
        self.setStyleSheet(f"QDialog {{ background: {BACKGROUND}; }}")
        self._init_ui()

    def _init_ui(self) -> None:
        est = self.estimate

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # No in-dialog heading: the window title already says "Run Workflow".
        n_tasks, n_lamella = len(est.tasks), len(est.lamella_names)
        layout.addWidget(
            meta_label(
                f"{n_tasks} task{'s' if n_tasks != 1 else ''} over "
                f"{n_lamella} lamella{'e' if n_lamella != 1 else ''}."
            )
        )

        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(chip(f"{est.step_count} steps"))
        if est.supervised_tasks:
            chips.addWidget(chip(f"{len(est.supervised_tasks)} supervised"))
        chips.addStretch()
        chips.addWidget(self._lamella_disclosure())
        layout.addLayout(chips)
        layout.addWidget(self._lamella_names_label)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        metrics.addWidget(_metric("Estimated duration", format_duration(est.work_seconds)))
        metrics.addWidget(
            _metric("Expected finish", _clock(est.expected_finish, est.started_at))
        )
        layout.addLayout(metrics)

        for note in self._exception_notes():
            layout.addWidget(note)

        layout.addWidget(_task_block(est.tasks, est.started_at))

        footnote = QLabel(
            "Estimates round up. Time spent waiting for you is not included."
        )
        footnote.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        footnote.setWordWrap(True)
        layout.addWidget(footnote)

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

        The old dialog's left column was the only place the names appeared, and a
        per-task breakdown replaces them with `×2`. They come back here, once, rather
        than repeated under every task.

        Builds the names label as a side effect and leaves it on the instance; the
        caller places it, because it belongs under the chip row rather than inside it.
        """
        names = QLabel(self._names_text())
        names.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        names.setWordWrap(True)
        names.setVisible(False)

        button = QToolButton()
        button.setText(f"{len(self.estimate.lamella_names)} lamellae")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.RightArrow)
        # Transparent explicitly: with no background rule a QToolButton paints the
        # platform's default button fill, which reads as a raised control next to the
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
        names = self.estimate.lamella_names
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
                    f"{format_duration(task.hold_seconds)} before it."
                )
            )
        elif scheduled:
            notes.append(
                self._note(
                    f"{len(scheduled)} steps are scheduled. The workflow holds for "
                    f"about {format_duration(self.estimate.hold_seconds)} in total, "
                    f"waiting for them."
                )
            )

        supervised = self.estimate.supervised_tasks
        if len(supervised) == 1:
            notes.append(
                self._note(
                    f"{supervised[0].name} waits for you. The work is counted above; "
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
