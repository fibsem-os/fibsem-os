"""The main windows' status bar: one line on the left, the host's actions on the right
(FIB-1188).

The line says one thing at a time:

- an **operation** while one runs outside a workflow -- a stage move, the images
  after it: "**Stage move** moving to the SEM orientation…";
- otherwise the **agent's action**, while the agent is in control of the microscope:
  "**Agent** focus sweep · 12s" (FIB-1375);
- otherwise the **run**, while a workflow runs: "**lamella-02 › Polishing** · 3 of 4";
- otherwise the **instruction**, muted: "Create or load an experiment to begin."

When the operation ends the run comes back, and when the run ends the instruction
does. Nothing is overwritten by whatever spoke last, which is what `showMessage` did:
a queue confirmation replaced the run's own line until the next report. So nothing
here calls it -- its text would cover the line.

No dot: running and waiting already colour the window's border. What is happening
reads bright, with its subject in bold; an instruction reads muted.

Progress that can be measured -- milling, an overview's tiles, a spot burn, a
fluorescence acquisition -- adds a thin bar and its numbers to the line: "**Milling: Rough Mill** stage 2 of 3 ▬▬ 42% ·
1m 20s left". Inside a run it sits under the run's task instead, and during the
agent's action under "Agent". The bar listens for
it itself, from the microscope a host hands it (`set_microscope`), so no window
decodes a report. When one ends, the line says how -- done, cancelled, or failed in
red -- for a moment, then gives the line back.

A run that ends with failures leaves a **failure** on the line, in red, with Show and
Dismiss, until it is dismissed or the next run starts: "**Run finished** 2 of 5
failed · 02-civil-cub › Polishing: autofocus found no focus peak". A run held for
someone says what releases it and how long it has waited: "**01-fancy-mite › Rough
Milling** review the milling patterns · waiting 2:14 · 3 of 5".

The right side is the host's: `add_action` places its buttons (AutoLamella's Run,
Stop, Supervised). The bar knows nothing about what they do.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Tuple

from PyQt5.QtCore import QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.fm.progress import FluorescenceAcquisitionProgress
from fibsem.imaging.spot import SpotBurnProgress
from fibsem.imaging.tiling.progress import TiledProgress
from fibsem.ui.stylesheets import SECONDARY_BUTTON_STYLESHEET, STATUS_BAR_STYLESHEET
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    ERROR_COLOR,
    ROW_ALT_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel
from fibsem.util.durations import format_duration_rounded
from fibsem.workflows.progress_line import (
    FM_STALE_S,
    OUTCOME_S,
    ProgressLine,
    ProgressReader,
    progress_sources,
)

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
_FAILED_STYLE = _LABEL.format(color=ERROR_COLOR, extra="")

# How long an outcome stays, and a fluorescence line outlives its last report
# (`fibsem.workflows.progress_line`), as the line's timers count them.
OUTCOME_MS = int(OUTCOME_S * 1000)
FM_STALE_MS = int(FM_STALE_S * 1000)

# Thin, rounded, no text inside: the numbers sit beside it, so nothing is clipped in a
# 6 px bar. Blue, as the views' MOVING and ACQUIRING chips are: green in this app says
# OK, automated, supervised.
_BAR_STYLE = (
    f"QProgressBar {{ border: none; border-radius: 3px; background: {ROW_ALT_COLOR}; }}"
    f" QProgressBar::chunk {{ border-radius: 3px; background: {ACCENT_COLOR}; }}"
)

# (what, step, detail): the subject in bold, what it is doing, then where it is.
_Line = Tuple[str, Optional[str], Optional[str]]
# (label, step, fraction, numbers, failed, standalone): measurable progress. The
# label is the subject on its own and the start of the step under a run -- unless
# *standalone*, when it is the subject whatever is running: a task's failure names
# its own task, which the run's line has already moved past.
_Progress = Tuple[str, Optional[str], Optional[float], Optional[str], bool, bool]


class _LineLabel(ElidedLabel):
    """A part of the line: it asks for its whole text while there is room, and gives
    way to an ellipsis -- the whole text in its tooltip -- when there is not.

    A plain QLabel's minimum is its text, so one long failure set the status bar's
    minimum and the bar set the window's: the whole app grew wider. `ElidedLabel`
    alone asks for nothing, which leaves a short line no room. This asks for its
    text and settles for *floor* pixels.
    """

    def __init__(self, floor: int) -> None:
        super().__init__()
        self._floor = floor
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        width = QFontMetrics(self.font()).horizontalAdvance(self.text()) + 4
        return QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(
            min(self._floor, self.sizeHint().width()),
            super().minimumSizeHint().height(),
        )


class FibsemStatusBar(QStatusBar):
    """One line of status on the left -- an operation, else the run, else the
    instruction -- and the host's actions on the right."""

    # The failure line's Show: where the failures are is the host's to
    # say (AutoLamella's run timeline marks them); the bar only asks.
    failure_details_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(STATUS_BAR_STYLESHEET)

        line = QWidget(self)
        # Transparent, or the app sheet paints the line in the surface colour and
        # the bar's own panel colour shows only in the gaps: a lighter box across
        # the left with a step beside the buttons. By name, so the labels' tooltips
        # keep their panel.
        line.setObjectName("statusLine")
        line.setStyleSheet("#statusLine { background: transparent; }")
        layout = QHBoxLayout(line)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)
        # What gives way first: the step (a failure's reason can run to a paragraph),
        # then the subject and the detail, which keep enough to be recognised.
        self._what = _LineLabel(floor=160)
        self._what.setStyleSheet(_WHAT_STYLE)
        self._step = _LineLabel(floor=40)
        self._step.setStyleSheet(_STEP_STYLE)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedSize(140, 6)
        self._bar.setStyleSheet(_BAR_STYLE)
        self._numbers = QLabel()
        self._numbers.setStyleSheet(_STEP_STYLE)
        self._detail = _LineLabel(floor=120)
        self._detail.setStyleSheet(_MUTED_STYLE)
        self._details_btn = QPushButton("Show")
        self._details_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._details_btn.clicked.connect(self.failure_details_requested)
        self._dismiss_btn = QPushButton("Dismiss")
        self._dismiss_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._dismiss_btn.clicked.connect(self.dismiss_failure)
        for widget in (
            self._what,
            self._step,
            self._bar,
            self._numbers,
            self._detail,
            self._details_btn,
            self._dismiss_btn,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        self.addWidget(line, 1)

        self._instruction: Optional[str] = None
        self._run: Optional[_Line] = None
        self._operation: Optional[_Line] = None
        self._progress: Optional[_Progress] = None
        self._failure: Optional[Tuple[str, str]] = None
        # What releases a held run, and since when (monotonic): the line counts it.
        self._waiting: Optional[Tuple[str, float]] = None
        self._waiting_clock = QTimer(self)
        self._waiting_clock.setInterval(1000)
        self._waiting_clock.timeout.connect(self._render)
        # The agent's action, and since when (monotonic): the line counts it too.
        self._agent: Optional[Tuple[str, float]] = None
        self._agent_clock = QTimer(self)
        self._agent_clock.setInterval(1000)
        self._agent_clock.timeout.connect(self._render)
        self._microscope: Optional["FibsemMicroscope"] = None
        # Each source's reports, read into lines (`ProgressReader`).
        self._reader = ProgressReader()
        self._outcome_timer = QTimer(self)
        self._outcome_timer.setSingleShot(True)
        # A bound method, not a lambda: Qt drops the connection with the bar, where a
        # lambda would still be called after the bar was gone, and crash.
        self._outcome_timer.timeout.connect(self._end_outcome)
        # Whether the line's progress is a fluorescence acquisition's, for the
        # expiry below.
        self._fm_owns_progress = False
        self._fm_stale_timer = QTimer(self)
        self._fm_stale_timer.setSingleShot(True)
        self._fm_stale_timer.timeout.connect(self._end_stale_fm)
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
            for signal, slot in self._subscriptions(self._microscope):
                try:
                    signal.disconnect(slot)
                except Exception:
                    pass  # already gone with its client
        self._microscope = microscope
        self.set_progress(None)
        if microscope is not None:
            for signal, slot in self._subscriptions(microscope):
                signal.connect(slot)

    def _subscriptions(self, microscope: "FibsemMicroscope"):
        slots = {
            "tiled": self._on_tiled_progress,
            "milling": self._on_milling_progress,
            "spot_burn": self._on_spot_burn_progress,
            "fluorescence": self._on_fm_progress,
        }
        return [
            (signal, slots[source]) for signal, source in progress_sources(microscope)
        ]

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

    def set_agent_action(self, action: Optional[str]) -> None:
        """The agent's action on the microscope while it is in control ("focus
        sweep"), counted from when it is first set; None when it lets go."""
        if not action:
            self._agent = None
            self._agent_clock.stop()
        elif self._agent is None or self._agent[0] != action:
            self._agent = (action, time.monotonic())
            self._agent_clock.start()
        self._render()

    def set_progress(
        self,
        label: Optional[str],
        step: Optional[str] = None,
        fraction: Optional[float] = None,
        numbers: Optional[str] = None,
        standalone: bool = False,
    ) -> None:
        """Measurable progress: *label* with its *step*, and a bar at *fraction* (0-1)
        beside *numbers* when there is a figure to draw. None when it is over.
        *standalone* keeps *label* as the subject under a run, as `show_outcome`."""
        self._outcome_timer.stop()
        self._fm_owns_progress = False
        self._fm_stale_timer.stop()
        self._progress = (
            (label, step, fraction, numbers, False, standalone) if label else None
        )
        self._render()

    def _end_outcome(self) -> None:
        self.set_progress(None)

    def show_outcome(
        self, label: str, step: str, failed: bool = False, standalone: bool = False
    ) -> None:
        """How an operation ended -- "done", "cancelled", a failure in red -- for
        `OUTCOME_MS`, then the line goes back to the run or the instruction.
        *standalone* keeps *label* as the subject under a run: a failed task's
        own name, not the task the run has moved on to."""
        self._progress = (label, step, None, None, failed, standalone)
        self._render()
        self._outcome_timer.start(OUTCOME_MS)

    # ── a run's failures, and a run that waits ───────────────────────────
    def show_failure(self, what: str, message: str) -> None:
        """A failure that stays, in red with Show and Dismiss, until
        `dismiss_failure` -- the Dismiss button, or the next run starting."""
        self._failure = (what, message)
        if self._outcome_timer.isActive():
            # A moment's outcome still up -- the failed task's own -- is what this
            # sums up: it gives way rather than covering the summary for its 2 s.
            self._outcome_timer.stop()
            self._progress = None
        self._render()

    def dismiss_failure(self) -> None:
        self._failure = None
        self._render()

    @property
    def failure(self) -> Optional[str]:
        """The lasting failure's message, or None."""
        return self._failure[1] if self._failure else None

    def set_waiting(self, releases: Optional[str]) -> None:
        """The run is held until someone acts: say what releases it, and count how
        long it has waited. None when it is released. The same hold set again keeps
        its clock."""
        if not releases:
            self._waiting = None
            self._waiting_clock.stop()
        elif self._waiting is None or self._waiting[0] != releases:
            since = self._waiting[1] if self._waiting else time.monotonic()
            self._waiting = (releases, since)
            self._waiting_clock.start()
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
        """Which source is on the line: "operation", "progress", "agent", "failure",
        "run", "instruction" or ""."""
        if self._operation is not None:
            return "operation"
        if self._progress is not None:
            return "progress"
        if self._agent is not None:
            return "agent"
        if self._failure is not None:
            return "failure"
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
        showing = self.showing
        line = (
            self._operation
            or self._line_for_progress()
            or self._line_for_agent()
            or self._line_for_failure()
            or self._line_for_run()
        )
        fraction = numbers = None
        failed = showing == "failure"
        if showing == "progress":
            fraction, numbers, failed = self._progress[2:5]
        self._step.setStyleSheet(_FAILED_STYLE if failed else _STEP_STYLE)
        self._details_btn.setVisible(showing == "failure")
        self._dismiss_btn.setVisible(showing == "failure")
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
        step, so the task keeps its place: "**task** label · step … · 3 of 5".
        During the agent's action it is the agent's: "**Agent** label · step …"."""
        if self._progress is None:
            return None
        label, step, standalone = (
            self._progress[0],
            self._progress[1],
            self._progress[5],
        )
        subject = self._line_for_agent() or self._run
        if subject is None or standalone:
            return (label, step, None)
        what, _run_step, detail = subject
        return (what, " · ".join(p for p in (label, step) if p), detail)

    def _line_for_agent(self) -> Optional[_Line]:
        """What the agent is doing, and for how long: "**Agent** focus sweep · 12s"."""
        if self._agent is None:
            return None
        action, since = self._agent
        return ("Agent", action, format_duration_rounded(time.monotonic() - since))

    def _line_for_failure(self) -> Optional[_Line]:
        if self._failure is None:
            return None
        what, message = self._failure
        return (what, message, None)

    def _line_for_run(self) -> Optional[_Line]:
        """The run's line; held, the step is what releases it and the detail how long
        it has waited."""
        if self._run is None or self._waiting is None:
            return self._run
        what, _step, detail = self._run
        releases, since = self._waiting
        minutes, seconds = divmod(int(time.monotonic() - since), 60)
        waited = f"waiting {minutes}:{seconds:02d}"
        return (what, releases, " · ".join(p for p in (waited, detail) if p))

    # ── measurable progress ──────────────────────────────────────────────
    def _show(self, line: ProgressLine) -> None:
        if line.outcome:
            self.show_outcome(line.label, line.step, line.failed, line.standalone)
        else:
            self.set_progress(
                line.label, line.step, line.fraction, line.numbers, line.standalone
            )

    @ensure_main_thread
    def _on_milling_progress(self, payload: object) -> None:
        """One milling report (`ProgressReader.milling`)."""
        self._show(
            self._reader.milling(
                payload, self._agent is not None, self._progress is not None
            )
        )

    @staticmethod
    def _set(label: QLabel, text: Optional[str]) -> None:
        label.setText(text or "")
        label.setVisible(bool(text))

    @ensure_main_thread
    def _on_tiled_progress(self, event: TiledProgress) -> None:
        """One report from a tiled acquisition, either modality: the status bar is the
        one consumer that wants both, while the overview tabs each want theirs
        (FIB-725)."""
        self._show(self._reader.tiled(event))

    @ensure_main_thread
    def _on_spot_burn_progress(self, report: SpotBurnProgress) -> None:
        """One report from a spot burn, supervised or not."""
        self._show(self._reader.spot_burn(report))

    def _end_stale_fm(self) -> None:
        if self._fm_owns_progress:
            self.set_progress(None)

    @ensure_main_thread
    def _on_fm_progress(self, report: FluorescenceAcquisitionProgress) -> None:
        """One report from a fluorescence acquisition. Its line does not outlive its
        last report by more than `FM_STALE_MS`, because most routines never say they
        have finished."""
        line = self._reader.fluorescence(report)
        if line is None:
            return
        if line.outcome:
            if self._fm_owns_progress:
                self._show(line)
            return
        self._show(line)
        self._fm_owns_progress = True
        self._fm_stale_timer.start(FM_STALE_MS)
