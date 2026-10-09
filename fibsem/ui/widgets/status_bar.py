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

The right side is the host's: `add_action` places its buttons (AutoLamella's Run,
Stop, Supervised). The bar knows nothing about what they do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

from fibsem.ui.stylesheets import STATUS_BAR_STYLESHEET
from fibsem.ui.tokens import TEXT_COLOR, TEXT_MUTED_COLOR, TEXT_STRONG_COLOR

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

# Selector-scoped, with a transparent background: the app sheet gives a QLabel a
# panel of its own, which drew a box behind each part of the line. A bare
# `background: transparent` would reach the labels' tooltips too.
_LABEL = "QLabel {{ background: transparent; color: {color};{extra} }}"
_WHAT_STYLE = _LABEL.format(color=TEXT_STRONG_COLOR, extra=" font-weight: 600;")
_STEP_STYLE = _LABEL.format(color=TEXT_COLOR, extra="")
_MUTED_STYLE = _LABEL.format(color=TEXT_MUTED_COLOR, extra="")

# (what, step, detail): the subject in bold, what it is doing, then where it is.
_Line = Tuple[str, Optional[str], Optional[str]]


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
        self._detail = QLabel()
        self._detail.setStyleSheet(_MUTED_STYLE)
        for label in (self._what, self._step, self._detail):
            layout.addWidget(label)
        layout.addStretch(1)
        self.addWidget(line, 1)

        self._instruction: Optional[str] = None
        self._run: Optional[_Line] = None
        self._operation: Optional[_Line] = None
        self._render()

    # ── the host's side ──────────────────────────────────────────────────
    def add_action(self, widget: QWidget) -> None:
        """Place one of the host's widgets on the right, after those already there."""
        self.addPermanentWidget(widget)

    def follow(self, view_controller: "MicroscopeViewController") -> None:
        """Say what the view controller reports the instrument doing: its stage moves."""
        view_controller.activity_changed.connect(self.set_stage_activity)

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
        parts = (self._what, self._step, self._detail)
        return " ".join(label.text() for label in parts if label.text())

    @property
    def showing(self) -> str:
        """Which source is on the line: "operation", "run", "instruction" or ""."""
        if self._operation is not None:
            return "operation"
        if self._run is not None:
            return "run"
        return "instruction" if self._instruction else ""

    def _render(self) -> None:
        line = self._operation or self._run
        if line is not None:
            what, step, detail = line
            self._set(self._what, what)
            self._set(self._step, step)
            self._set(self._detail, f"· {detail}" if detail else None)
        else:
            self._set(self._what, None)
            self._set(self._step, None)
            self._set(self._detail, self._instruction)

    @staticmethod
    def _set(label: QLabel, text: Optional[str]) -> None:
        label.setText(text or "")
        label.setVisible(bool(text))
