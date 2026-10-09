"""The main windows' status bar: one line on the left, the host's actions on the right
(FIB-1188).

The line says one thing at a time:

- an **operation** while one runs outside a workflow -- a stage move, the images
  after it: "**Stage move** moving to the SEM orientation…";
- otherwise the **run**, while a workflow runs: "**lamella-02 › Polishing** · 3 of 4";
- otherwise the **instruction**, muted: "Create or load an experiment to begin."

When the operation ends the run comes back, and when the run ends the instruction
does. Nothing is overwritten by whatever spoke last, which is what `showMessage` did:
a queue confirmation replaced the run's own line until the next report. So nothing
here calls it -- its text would cover the line.

No dot: running and waiting already colour the window's border. What is happening
reads bright, with its subject in bold; an instruction reads muted.

Progress that can be measured -- milling, so far -- adds a thin bar and its numbers
to the line: "**Milling: Rough Mill** stage 2 of 3 ▬▬ 42% · 1m 20s left". Inside a
run it sits under the run's task instead. The bar listens for it itself, from the
microscope a host hands it (`set_microscope`), so no window decodes a report.

The right side is the host's: `add_action` places its buttons (AutoLamella's Run,
Stop, Supervised). The bar knows nothing about what they do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QStatusBar, QWidget
from superqt import ensure_main_thread

from fibsem.milling.progress import (
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.ui.stylesheets import STATUS_BAR_STYLESHEET
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    ROW_ALT_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.utils import format_time_remaining

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

# Selector-scoped, with a transparent background: the app sheet gives a QLabel a
# panel of its own, which drew a box behind each part of the line. A bare
# `background: transparent` would reach the labels' tooltips too.
_LABEL = "QLabel {{ background: transparent; color: {color};{extra} }}"
_WHAT_STYLE = _LABEL.format(color=TEXT_STRONG_COLOR, extra=" font-weight: 600;")
_STEP_STYLE = _LABEL.format(color=TEXT_COLOR, extra="")
_MUTED_STYLE = _LABEL.format(color=TEXT_MUTED_COLOR, extra="")
# Thin, rounded, no text inside: the numbers sit beside it, so nothing is clipped in a
# 6 px bar. Blue, as the views' MOVING and ACQUIRING chips are: green in this app says
# OK, automated, supervised.
_BAR_STYLE = (
    f"QProgressBar {{ border: none; border-radius: 3px; background: {ROW_ALT_COLOR}; }}"
    f" QProgressBar::chunk {{ border-radius: 3px; background: {ACCENT_COLOR}; }}"
)

# (what, step, detail): the subject in bold, what it is doing, then where it is.
_Line = Tuple[str, Optional[str], Optional[str]]
# (label, step, fraction, numbers): measurable progress. The label is the subject on
# its own and the start of the step under a run.
_Progress = Tuple[str, Optional[str], Optional[float], Optional[str]]


class FibsemStatusBar(QStatusBar):
    """One line of status on the left -- an operation, else the run, else the
    instruction -- and the host's actions on the right."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(STATUS_BAR_STYLESHEET)

        line = QWidget(self)
        layout = QHBoxLayout(line)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)
        self._what = QLabel()
        self._what.setStyleSheet(_WHAT_STYLE)
        self._step = QLabel()
        self._step.setStyleSheet(_STEP_STYLE)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedSize(140, 6)
        self._bar.setStyleSheet(_BAR_STYLE)
        self._numbers = QLabel()
        self._numbers.setStyleSheet(_STEP_STYLE)
        self._detail = QLabel()
        self._detail.setStyleSheet(_MUTED_STYLE)
        for widget in (self._what, self._step, self._bar, self._numbers, self._detail):
            layout.addWidget(widget)
        layout.addStretch(1)
        self.addWidget(line, 1)

        self._instruction: Optional[str] = None
        self._run: Optional[_Line] = None
        self._operation: Optional[_Line] = None
        self._progress: Optional[_Progress] = None
        self._microscope: Optional["FibsemMicroscope"] = None
        # A delegating strategy names itself once and the backend's ticks carry no
        # words; this keeps the strategy's (FIB-797).
        self._milling_label = MillingMessageTracker()
        self._render()

    # ── the host's side ──────────────────────────────────────────────────
    def add_action(self, widget: QWidget) -> None:
        """Place one of the host's widgets on the right, after those already there."""
        self.addPermanentWidget(widget)

    def follow(self, view_controller: "MicroscopeViewController") -> None:
        """Say what the view controller reports the instrument doing: its stage moves."""
        view_controller.activity_changed.connect(self.set_stage_activity)

    def set_microscope(self, microscope: Optional["FibsemMicroscope"]) -> None:
        """Show *microscope*'s milling progress; None lets the last one go.

        Held for the next call, which disconnects it: a reconnect hands over a new
        client, and the old one's signals must not keep writing here."""
        if self._microscope is not None:
            try:
                self._microscope.milling_progress_signal.disconnect(
                    self._on_milling_progress
                )
            except Exception:
                pass  # already gone with its client
        self._microscope = microscope
        self.set_progress(None)
        if microscope is not None:
            microscope.milling_progress_signal.connect(self._on_milling_progress)

    # ── the three sources ────────────────────────────────────────────────
    def set_instruction(self, text: Optional[str]) -> None:
        """What to do next, shown while nothing is running."""
        self._instruction = text or None
        self._render()

    def set_run(
        self,
        what: Optional[str],
        step: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """The workflow's line while it runs; None when the run is over."""
        self._run = (what, step, detail) if what else None
        self._render()

    def set_run_step(self, step: Optional[str]) -> None:
        """What the run is doing now ("Waiting on 2 decisions…", "Loading grid
        G2..."), under its current subject; "Workflow" before the first task has
        named one. None or "" clears the step and keeps the subject."""
        if self._run is None:
            if step:
                self.set_run("Workflow", step)
            return
        what, _step, detail = self._run
        self.set_run(what, step or None, detail)

    def set_operation(self, what: Optional[str], step: Optional[str] = None) -> None:
        """An operation's line while it runs; None when it is over."""
        self._operation = (what, step, None) if what else None
        self._render()

    def set_progress(
        self,
        label: Optional[str],
        step: Optional[str] = None,
        fraction: Optional[float] = None,
        numbers: Optional[str] = None,
    ) -> None:
        """Measurable progress: *label* with its *step*, and a bar at *fraction* (0-1)
        beside *numbers* when there is a figure to draw. None when it is over."""
        self._progress = (label, step, fraction, numbers) if label else None
        self._render()

    def set_stage_activity(self, text: Optional[str]) -> None:
        """A stage move's words, as the view controller reports them
        (`activity_changed`): "Moving to the SEM orientation…" reads as
        "**Stage move** moving to the SEM orientation…"."""
        if not text:
            self.set_operation(None)
            return
        self.set_operation("Stage move", text[0].lower() + text[1:])

    # ── what it shows ────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """The line as read, its parts joined by spaces."""
        parts = (self._what, self._step, self._numbers, self._detail)
        return " ".join(label.text() for label in parts if label.text())

    @property
    def showing(self) -> str:
        """Which source is on the line: "operation", "progress", "run",
        "instruction" or ""."""
        if self._operation is not None:
            return "operation"
        if self._progress is not None:
            return "progress"
        if self._run is not None:
            return "run"
        return "instruction" if self._instruction else ""

    @property
    def fraction(self) -> Optional[float]:
        """Where the bar is drawn, 0-1, or None when there is no bar."""
        if self._bar.isHidden():
            return None
        return self._bar.value() / self._bar.maximum()

    def _render(self) -> None:
        line = self._operation or self._line_for_progress() or self._run
        fraction = numbers = None
        if self._operation is None and self._progress is not None:
            fraction, numbers = self._progress[2], self._progress[3]
        if line is not None:
            what, step, detail = line
            self._set(self._what, what)
            self._set(self._step, step)
            self._set(self._detail, f"· {detail}" if detail else None)
        else:
            self._set(self._what, None)
            self._set(self._step, None)
            self._set(self._detail, self._instruction)
        self._set(self._numbers, numbers)
        if fraction is not None:
            self._bar.setValue(int(round(min(max(fraction, 0.0), 1.0) * 1000)))
        self._bar.setVisible(fraction is not None)

    def _line_for_progress(self) -> Optional[_Line]:
        """Progress on its own reads "**label** step"; under a run it is the run's
        step, so the task keeps its place: "**task** label · step … · 3 of 5"."""
        if self._progress is None:
            return None
        label, step, _fraction, _numbers = self._progress
        if self._run is None:
            return (label, step, None)
        what, _run_step, detail = self._run
        return (what, " · ".join(p for p in (label, step) if p), detail)

    # ── milling ──────────────────────────────────────────────────────────
    @ensure_main_thread
    def _on_milling_progress(self, payload: object) -> None:
        """One milling report, as the line: the label the producer gave, which stage
        of how many, and the countdown once there is one.

        Total over whatever arrives: this runs in a queued slot, where an exception is
        a process abort on PyQt5 (FIB-329), and a plugin strategy may still emit the
        old dict (`from_payload`)."""
        report = MillingProgress.from_payload(payload)
        if report.status.is_terminal:
            self.set_progress(None)
            return
        label = self._milling_label.label(report)
        step = None
        stage, total = report.display_stage, report.total_stages
        if stage and total and total > 1:
            step = f"stage {stage} of {total}"
        elif self._progress is not None and report.status is not (
            MillingProgressStatus.STAGE_STARTED
        ):
            step = self._progress[1]  # a tick does not repeat the stage count
        fraction = numbers = None
        remaining, estimated = report.remaining_time, report.estimated_time
        if (
            report.status is MillingProgressStatus.STAGE_UPDATE
            and remaining is not None
            and estimated
        ):
            fraction = min(max(1.0 - remaining / estimated, 0.0), 1.0)
            numbers = (
                f"{int(fraction * 100)}% · {format_time_remaining(remaining)} left"
            )
        self.set_progress(label, step, fraction, numbers)

    @staticmethod
    def _set(label: QLabel, text: Optional[str]) -> None:
        label.setText(text or "")
        label.setVisible(bool(text))
