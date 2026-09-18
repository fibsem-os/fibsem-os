"""The Review tab: every decision waiting on a person, in one place.

A producing task completes and leaves a proposal on its item (see
``proposals.py``); the consumer of that decision is deferred until someone
confirms or rejects it. This tab lists those proposals and hosts a renderer
per proposal *kind* -- a point of interest is one image and one marker, a
screening review will be a set of candidates with toggles -- so the tab
dispatches to a registered renderer rather than growing an ``if`` per kind.

Two verbs. **Confirm** submits whatever the renderer currently shows; the
delta against the proposal is computed by ``Experiment.decide``, never
declared here. **Reject** means *nothing further here*: the task that was
waiting on the decision is failed, so nothing that requires it runs. There is
no defer button on purpose: items commit independently, so walking away is
deferral.

Every action here is self-contained and never touches hardware. The one write
path is ``Experiment.decide``; the agent server's decide endpoint is the same
client of the same function, so the tab owns no state a decision could be
lost in -- the inbox is re-derived from the experiment on every refresh.
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics, QKeySequence
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    OVERVIEW_POSITIONS,
    POINT_OF_INTEREST,
    TASK_RESULT,
    Author,
    AuthorKind,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import Attention, Experiment, GridRecord
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    DEFECT_RED_COLOR,
    GRAY_ICON_COLOR,
    GRAY_SECONDARY_COLOR,
    GRAY_TEXT_COLOR,
    OK_COLOR,
    ORANGE_COLOR,
    PANEL_COLOR,
    PRIMARY_COLOR,
    SURFACE_COLOR,
)
from fibsem.ui.widgets.overview_widget import MODALITY_CHIP_STYLE

__all__ = [
    "REVIEW_RENDERERS",
    "PointOfInterestReviewRenderer",
    "ReviewRenderer",
    "ReviewTabWidget",
    "register_review_renderer",
    "waiting_on",
]

_KIND_LABELS = {POINT_OF_INTEREST: "Milling positions", TASK_RESULT: "Task results"}

_HEADER_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 10px; font-weight: 600; "
    "letter-spacing: 1px; padding: 8px 6px 2px 6px;"
)
_TITLE_STYLE = f"color: {GRAY_TEXT_COLOR}; font-size: 14px; font-weight: 600;"
_CHIP_STYLE = (
    f"color: {PRIMARY_COLOR}; background: {PANEL_COLOR}; border-radius: 3px; "
    "padding: 2px 7px; font-size: 10px; font-family: monospace;"
)
_MUTED_STYLE = f"color: {GRAY_SECONDARY_COLOR}; font-size: 11px;"
_READOUT_STYLE = f"color: {GRAY_TEXT_COLOR}; font-family: monospace; font-size: 12px;"
_CELL_KEY_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 10px; letter-spacing: 1px; "
    "font-weight: 600; background: transparent;"
)
_CELL_VALUE_STYLE = (
    f"color: {GRAY_TEXT_COLOR}; font-family: monospace; font-size: 12px; "
    "background: transparent;"
)
_ROW_HEIGHT = 30
# The name column: fixed, so the tasks line up down the list and the eye can
# scan either column. Petnames are two words and a number; this fits them.
_ROW_NAME_WIDTH = 132
# Not bold, and a point off the old 13: bold at 13 read as a heading rather
# than a name, which made every row shout. The dot carries the state.
_ROW_NAME_STYLE = f"color: {GRAY_TEXT_COLOR}; font-size: 12px; background: transparent;"
_ROW_TASK_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 11px; background: transparent;"
)
_ROW_RIGHT_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 11px; background: transparent;"
)
_ROW_RIGHT_STRONG = _ROW_RIGHT_STYLE  # the outcome line reads like the line above it


def author_label(author: Author, experiment: Optional[Experiment]) -> str:
    """How a decision's author reads on screen: "you" for the operator this
    experiment names, otherwise the author's own label (the name for another
    person, "agent · model", "auto · proposer")."""
    author = Author.parse(author)
    if experiment is not None and author == experiment.author():
        return "you"
    return author.label


def clock(timestamp: Optional[float]) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%H:%M")


def age(timestamp: Optional[float]) -> str:
    """How long ago, coarsely: "3 min", "2 h", "1 d"."""
    if not timestamp:
        return ""
    seconds = max(0.0, time.time() - timestamp)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} d"


def delta_label(proposal: Proposal, decision: Optional[Decision] = None) -> str:
    """ "moved 2.1 µm", "as proposed", or "" for a value with no delta."""
    delta = proposal.delta(decision).get("poi")
    if not isinstance(delta, Point):
        return ""
    magnitude = (delta.x**2 + delta.y**2) ** 0.5
    if magnitude < 1e-9:
        return "as proposed"
    return f"moved {magnitude * 1e6:.1f} µm"


def _item_tasks(experiment: Experiment, item: Any) -> List[tuple]:
    """(name, requires) for the tasks an item runs, in workflow order: the
    grid protocol's for a grid, the lamella workflow's otherwise."""
    if isinstance(item, GridRecord):
        try:
            protocol = experiment.grid_protocol
        except ValueError:  # no task protocol on this experiment
            return []
        return [
            (name, protocol.requirements(name)) for name in protocol.ordered_task_names
        ]
    config = getattr(
        getattr(experiment, "task_protocol", None), "workflow_config", None
    )
    if config is None:
        return []
    return [(task.name, task.requires) for task in config.tasks]


def waiting_on(experiment: Experiment, task_name: str, item: Any = None) -> List[str]:
    """The tasks deferred until ``task_name``'s proposal is decided: every task
    that requires it, transitively, in workflow order. A grid's by the grid
    protocol, a lamella's by the workflow."""
    gated = {task_name}
    names: List[str] = []
    for name, requires in _item_tasks(experiment, item):
        if any(req in gated for req in requires):
            gated.add(name)
            names.append(name)
    return names


def is_gated(experiment: Experiment, task_name: str, item: Any = None) -> bool:
    """Whether ``task_name`` is set to Review for this kind of item: the grid
    task's attention for a grid, the workflow's for a lamella."""
    if isinstance(item, GridRecord):
        try:
            config = experiment.grid_protocol.task_config.get(task_name)
        except ValueError:
            return False
        return config is not None and config.attention is Attention.review_later
    protocol = getattr(experiment, "task_protocol", None)
    return (
        bool(protocol) and protocol.get_attention(task_name) is Attention.review_later
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class ReviewRenderer(QWidget):
    """One proposal kind's review surface. A subclass shows the proposal and
    answers ``current_values`` with whatever the reviewer has left it as; the
    host turns that into a decision."""

    confirm_requested = pyqtSignal()
    reject_requested = pyqtSignal()
    open_item_requested = pyqtSignal(object)  # the item: go and edit it there

    def set_proposal(
        self, experiment: Experiment, item: Any, task_name: str, proposal: Proposal
    ) -> None:
        raise NotImplementedError

    def current_values(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_running(self, running: bool) -> None:
        """Whether the beam is busy elsewhere -- shown, never acted on."""

    def set_read_only(self, decided: Optional[Decision]) -> None:
        """Show a decided proposal as it was decided: no verbs. None re-arms."""

    def set_to_check(self, applied: Optional[Decision]) -> None:
        """Show a proposal its producer applied itself: the values are not
        editable (they were used), the primary verb is Acknowledge, and
        Reject keeps its meaning. None leaves this state."""

    def set_position(self, text: str) -> None:
        """Where this sits in the list ("3 of 12"); shown, never acted on."""


def decided_proposals(experiment: Experiment) -> List[tuple]:
    """Every decided proposal as (item, task_name, proposal, superseded),
    newest decision first: the other half of the inbox, derived the same way.
    ``superseded`` marks one a re-run replaced. A proposal still to check is
    not here; it has its own group."""
    decided = []
    for item in list(experiment.positions) + list(experiment.grids):
        for task_name, proposal in item.proposals.items():
            if not proposal.pending and not proposal.to_check:
                decided.append((item, task_name, proposal, False))
            for p in proposal.superseded:
                if not p.pending:
                    decided.append((item, task_name, p, True))
    decided.sort(key=lambda e: e[2].current.timestamp, reverse=True)
    return decided


def describe_decision(
    proposal: Proposal, experiment: Optional[Experiment] = None
) -> str:
    """One line: "Confirmed by you at 11:24 · moved (+2.10, −0.40) µm"."""
    d = proposal.current
    if d is None:
        return ""
    who = author_label(d.author, experiment)
    when = clock(d.timestamp)
    if proposal.withdrawn:
        # No author worth naming: nothing decided this, the question was taken
        # back when whatever asked it went away.
        return f"Withdrawn at {when} — {d.reason}"
    if d.outcome is DecisionOutcome.Rejected:
        return f"Rejected by {who} at {when} — {d.reason}"
    # A producer's own decision applied values (or, with none, recorded the
    # result); a person's later empty decision is the look that was owed.
    verb = "Applied" if proposal.values else "Recorded"
    auto = proposal.applied or next(
        (x for x in proposal.decisions if x.author.kind is not AuthorKind.human),
        None,
    )
    if not d.values and auto is not None and auto is not d:
        return (
            f"{verb} by {author_label(auto.author, experiment)} at "
            f"{clock(auto.timestamp)} · checked by {who} at {when}"
        )
    if d.author.kind is not AuthorKind.human:
        return f"{verb} by {who} at {when} · not checked yet"
    delta = proposal.delta(d).get("poi")
    moved = (
        f" · moved ({delta.x * 1e6:+.2f}, {delta.y * 1e6:+.2f}) µm"
        if isinstance(delta, Point)
        else ""
    )
    return f"Confirmed by {who} at {when}{moved}"


REVIEW_RENDERERS: Dict[str, Type[ReviewRenderer]] = {}


def register_review_renderer(
    kind: str,
) -> Callable[[Type[ReviewRenderer]], Type[ReviewRenderer]]:
    def _register(cls: Type[ReviewRenderer]) -> Type[ReviewRenderer]:
        REVIEW_RENDERERS[kind] = cls
        return cls

    return _register


def _via_text(via: str) -> str:
    return {
        "review": "in the Review tab",
        "workflow": "in the workflow",
        "server": "by the agent",
    }.get(via, "")


def _held_text(n: int) -> str:
    return f"{n} task{'s' if n != 1 else ''} held" if n else "nothing is held"


@register_review_renderer(TASK_RESULT)
class TaskResultReviewRenderer(ReviewRenderer):
    """What a task did: its final ion and electron images side by side, one
    line a person can say out loud (the state) with the record -- what ran,
    how it ended, what was held or went ahead, where it was decided -- in
    that line's tooltip, and the two verbs. Nothing to drag, nothing to
    write: confirm says it looks right, reject fails the task.

    Every proposal is a task result, so this is the base. A kind with values
    to decide extends it and puts them on the image: ``_draw_values`` and
    ``_draw_confirmed`` draw them, ``current_values`` reads them back,
    ``_fact`` says what was proposed.
    """

    PENDING_HINT = "Enter — this looks right"
    CONFIRM_LABEL = "Confirm · looks right"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from fibsem.ui.widgets.canvas.quad_view import (
            LamellaEditorView,
            MicroscopeViewController,
        )

        self._experiment: Optional[Experiment] = None
        self._item: Any = None
        self._task_name = ""
        self._proposal: Optional[Proposal] = None
        self._image: Optional[FibsemImage] = None
        self._electron: Optional[FibsemImage] = None
        # a fluorescence result's image: shown on the FM page, never on a beam
        # canvas, and never the image a kind's values are drawn on
        self._fluorescence: Optional[FluorescenceImage] = None
        self._gated: List[str] = []
        self._decided: Optional[Decision] = None
        self._applied: Optional[Decision] = None
        self._running = False
        self._position = ""

        self._controller = MicroscopeViewController(view=LamellaEditorView())
        self._controller.widget.show_beams()
        self._view = self._build_view()

        # No header: the selected inbox row already says the lamella and the
        # task, and the canvas has its own beam label. These two are kept for
        # callers that read them, but not shown.
        self.title = QLabel()
        self.title.hide()
        self.task_chip = QLabel()
        self.task_chip.hide()
        self.btn_open = QPushButton("Go to lamella")
        self.btn_open.setFlat(True)
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.setStyleSheet(
            f"QPushButton {{ color: {GRAY_SECONDARY_COLOR}; background: transparent; "
            "border: none; font-size: 11px; padding: 0 4px; }"
            f"QPushButton:hover {{ color: {GRAY_TEXT_COLOR}; }}"
        )
        self.btn_open.setToolTip(
            "Select this lamella in the Lamella tab, where its settings are edited"
        )
        # the one line, coloured by state; everything else is its tooltip
        self.line = QLabel()
        self.line.setWordWrap(True)
        self.line.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # kept for callers that read the state as text
        self.readout = QLabel()
        self.readout.hide()

        # actions: position on the left, the two verbs on the right
        self.position = QLabel()
        self.position.setStyleSheet(_MUTED_STYLE)
        self.btn_confirm = QPushButton(self.CONFIRM_LABEL)
        self.btn_confirm.setStyleSheet(stylesheets.CONFIRM_BUTTON_STYLESHEET)
        self.btn_confirm.setToolTip(self.PENDING_HINT)
        self.btn_reject = QPushButton("Reject · mark task failed")
        self.btn_reject.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_reject.setToolTip(
            "R — nothing further here; the task is failed and what requires it "
            "does not run"
        )
        actions = QHBoxLayout()
        actions.addWidget(self.position)
        actions.addWidget(self.btn_open)
        actions.addStretch(1)
        actions.addWidget(self.btn_confirm)
        actions.addWidget(self.btn_reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._view, 1)
        # said in place of the image when there is none to show, so the last
        # proposal's image is never left under this one's verbs
        self.no_image = QLabel()
        self.no_image.setAlignment(Qt.AlignCenter)
        self.no_image.setWordWrap(True)
        self.no_image.setStyleSheet(_MUTED_STYLE)
        self.no_image.hide()
        layout.addWidget(self.no_image, 1)
        layout.addWidget(self.line)
        layout.addLayout(actions)

        self.btn_confirm.clicked.connect(self.confirm_requested)
        self.btn_reject.clicked.connect(self.reject_requested)
        self.btn_open.clicked.connect(lambda: self.open_item_requested.emit(self._item))
        self._refresh_line()

    # -- what a kind may put in the middle ------------------------------------

    def _build_view(self) -> QWidget:
        """The widget the panel is built around. A result is its images, so
        this is the beam canvas; a kind whose decision is about something else
        -- where the lamellae go on a grid's overview -- returns its own and
        fills it in ``_show_proposal``."""
        return self._controller.widget

    def _show_proposal(self) -> None:
        """Put this proposal on the view. Called by ``set_proposal`` once the
        record and the chrome are set, so an override has the item, the task
        and the proposal to work from."""
        recorded = str(self._proposal.provenance.get("reference_image") or "")
        shown = self._image is not None or self._fluorescence is not None
        self._view.setVisible(shown)
        self.no_image.setVisible(not shown)
        self.no_image.setText(
            ""
            if shown
            else f"The recorded image could not be read: {os.path.basename(recorded)}"
            if recorded
            else "No image was recorded for this result."
        )
        # The result image goes on the canvas of the beam that took it: a grid's
        # SEM overview is an electron image, and labelled FIB it misleads. A
        # kind's values are drawn on the ion image, which is where every kind
        # with values records them.
        electron_only = (
            self._image is not None
            and self._electron is None
            and _beam_of(self._image) is BeamType.ELECTRON
        )
        if electron_only:
            self._controller.set_image(BeamType.ELECTRON, self._image)
        else:
            if self._image is not None:
                self._controller.set_image(BeamType.ION, self._image)
                self._draw_values()
            if self._electron is not None:
                self._controller.set_image(BeamType.ELECTRON, self._electron)
        self._controller.widget.set_sem_visible(
            electron_only or self._electron is not None
        )
        self._controller.widget.set_fib_visible(not electron_only)

    # -- what a kind adds: its values, on the image ---------------------------

    def _draw_values(self) -> None:
        """Put the proposed values on the image. A plain result has none."""

    def _draw_confirmed(self, decision: Decision) -> None:
        """Put the decided values beside the proposed ones, so the delta shows."""

    def current_values(self) -> Dict[str, Any]:
        return {}

    def _state_words(self) -> tuple:
        """(applied verb, re-run task) for the to-check line and its tooltip."""
        return "Recorded", self._task_name

    def _fact(self) -> str:
        """The record's first sentence: what ran, how it ended, on what."""
        p = self._proposal.provenance if self._proposal else {}
        task = self._task_name
        took = self._took()
        failure = str(p.get("failure") or "")
        if failure:
            head = (
                f"{task} failed" + (f" after {took}" if took else "") + f": {failure}."
            )
        else:
            head = f"{task} completed" + (f" in {took}" if took else "") + "."
        names = [
            os.path.basename(str(p.get(k) or ""))
            for k in ("reference_image", "reference_image_eb")
        ]
        names = [n for n in names if n]
        if not names:
            return head + " No reference images were recorded."
        when = clock(self._proposal.created_at) if self._proposal else ""
        which = (
            "Final images" if any("_final" in n for n in names) else " · ".join(names)
        )
        return f"{head} {which} at {when}."

    def _took(self) -> str:
        """How long the run took, or "" when the record does not say."""
        p = self._proposal.provenance if self._proposal else {}
        started, ended = p.get("started_at"), p.get("ended_at")
        if isinstance(started, (int, float)) and isinstance(ended, (int, float)):
            return _duration(ended - started)
        return ""

    # -- ReviewRenderer ------------------------------------------------------

    def set_proposal(
        self, experiment: Experiment, item: Any, task_name: str, proposal: Proposal
    ) -> None:
        self._experiment = experiment
        self._item = item
        self._task_name = task_name
        self._proposal = proposal
        image = _load_reference_image(experiment, item, proposal)
        self._fluorescence = image if isinstance(image, FluorescenceImage) else None
        self._image = image if isinstance(image, FibsemImage) else None
        electron = _load_reference_image(
            experiment, item, proposal, "reference_image_eb"
        )
        self._electron = electron if isinstance(electron, FibsemImage) else None
        self._gated = waiting_on(experiment, task_name, item)
        self._decided = None
        self._applied = None
        self.title.setText(getattr(item, "name", ""))
        self.task_chip.setText(task_name)
        grid = isinstance(item, GridRecord)
        self.btn_open.setText("Go to grid" if grid else "Go to lamella")
        self.btn_open.setToolTip(
            "Select this grid in the Grids tab, with its overviews"
            if grid
            else "Select this lamella in the Lamella tab, where its settings are edited"
        )
        self.btn_confirm.setText(self.CONFIRM_LABEL)
        self.btn_confirm.setToolTip(self.PENDING_HINT)
        # everything the last proposal put up goes first: images, overlays and
        # the FM composite, so a result without a readable image shows none
        self._controller.clear()
        self._controller.arm_overlay(BeamType.ION, None)
        view = self._controller.widget
        if self._fluorescence is not None:
            view.show_fluorescence()
            self._controller.set_fm_image(self._fluorescence)
        else:
            view.show_beams()
        self._show_proposal()
        self._refresh_line()

    def set_running(self, running: bool) -> None:
        self._running = running

    def set_position(self, text: str) -> None:
        self._position = text
        self._refresh_line()

    def set_to_check(self, applied: Optional[Decision]) -> None:
        self._applied = applied
        if applied is None:
            return
        # the values were used: the marker is not for dragging. The confirmed
        # marker (for centre-of-image, on the proposed one) shows what applied.
        self._draw_confirmed(applied)
        self._controller.arm_overlay(BeamType.ION, None)
        self.btn_confirm.setText("Acknowledge")
        self.btn_confirm.setToolTip(
            "Enter — record that you looked; nothing is written, the values "
            f"were already applied. Re-run {self._task_name} to change them."
        )
        self.btn_confirm.setEnabled(True)
        self.btn_reject.setEnabled(True)
        self._refresh_line()

    def set_read_only(self, decided: Optional[Decision]) -> None:
        self._decided = decided
        self.btn_confirm.setEnabled(decided is None)
        self.btn_reject.setEnabled(decided is None)
        if decided is not None and self._image is not None:
            # the proposed marker stays where it was; the confirmed one is
            # drawn beside it so the delta is visible, and neither is draggable
            self._draw_confirmed(decided)
            self._controller.arm_overlay(BeamType.ION, None)
        self._refresh_line()

    # -- the line ------------------------------------------------------------

    def _refresh_line(self) -> None:
        proposal = self._proposal
        if proposal is None:
            self.line.setText("")
            self.position.setText(self._position)
            return
        experiment = self._experiment
        gated = self._gated
        decided = self._decided
        applied = self._applied
        tip = [self._fact()]
        position = self._position
        failure = str(proposal.provenance.get("failure") or "")

        if decided is not None:
            who = author_label(decided.author, experiment)
            when = clock(decided.timestamp)
            via = _via_text(decided.via)
            if proposal.withdrawn:
                text = f"⊘  Withdrawn at {when} · {decided.reason}"
                colour = GRAY_SECONDARY_COLOR
                tip.append(f"Withdrawn before it was answered: {decided.reason}.")
            elif decided.outcome is DecisionOutcome.Rejected:
                text = f"✗  Rejected by {who} at {when} · {decided.reason}"
                colour = DEFECT_RED_COLOR
                tip.append(f"Rejected {via}: {decided.reason}.".replace("  ", " "))
                if gated:
                    tip.append("Skipped, lamella failed: " + ", ".join(gated) + ".")
            else:
                label = delta_label(proposal, decided)
                if not decided.values:
                    label = "checked"
                text = f"✓  Confirmed by {who} at {when}" + (
                    f" · {label}" if label else ""
                )
                colour = OK_COLOR
                moved = [
                    f"({d.x * 1e6:+.2f}, {d.y * 1e6:+.2f}) µm"
                    for d in proposal.delta(decided).values()
                    if isinstance(d, Point)
                ]
                tip.append(
                    (
                        f"Confirmed {via} at {', '.join(moved)} from the proposal."
                        if moved
                        else f"Confirmed {via}."
                    ).replace("  ", " ")
                )
                if gated:
                    tip.append("Unblocked: " + ", ".join(gated) + ".")
            position = f"{position} · read-only".strip(" ·")
        elif applied is not None:
            verb, rerun = self._state_words()
            text = (
                f"{verb} automatically at {clock(applied.timestamp)} · not checked yet"
            )
            colour = GRAY_SECONDARY_COLOR
            if gated:
                tip.append(", ".join(gated) + " went ahead.")
            tip.append(
                f"Acknowledge records that you looked; re-run {rerun} to change it."
            )
        elif failure:
            # the failure is the line, in the error colour; the tooltip has the rest
            took = self._took()
            text = f"{self._task_name} failed" + (f" after {took}" if took else "")
            text += f" · {failure}"
            colour = DEFECT_RED_COLOR
        else:
            if proposal.asking:
                # The run is stopped on this one, which is a stronger thing
                # than the requires edges below: those defer tasks, this is
                # the task itself waiting to be told.
                text = f"{self._task_name} is parked on this · nothing else is running"
                tip.append(
                    f"{self._task_name} asked this mid-run and is waiting for the "
                    "answer. Confirm hands it back and the task carries on."
                )
            else:
                text = f"Waiting for your decision · {_held_text(len(gated))}"
            colour = ORANGE_COLOR  # waiting on you now: the border's colour
            if gated:
                tip.append("Held until you decide: " + ", ".join(gated) + ".")
        if self._image is None and self._fluorescence is None:
            tip.append(
                "The reference image was not found; confirm uses the proposed values."
            )
        self.line.setText(text)
        self.line.setStyleSheet(f"color: {colour}; font-size: 12px; padding: 2px 0;")
        self.line.setToolTip("\n".join(t for t in tip if t))
        self.position.setText(position)
        self.readout.setText(text + "\n" + "\n".join(tip))


@register_review_renderer(POINT_OF_INTEREST)
class PointOfInterestReviewRenderer(TaskResultReviewRenderer):
    """The task result with the point of interest on it: one draggable marker
    on the ion image, pre-placed where the task proposed it. Same overlay and
    same drag as the inline question; what changes is when it happens."""

    PENDING_HINT = "Enter — this is the answer; drag the marker to correct it first"
    CONFIRM_LABEL = "Confirm"

    def _fact(self) -> str:
        proposal = self._proposal
        if proposal is None:
            return ""
        poi = proposal.values.get("poi")
        where = os.path.basename(str(proposal.provenance.get("reference_image", "")))
        image = "the final image" if "_final_" in where else (where or "the image")
        value = (
            f", {poi.x * 1e6:+.2f}, {poi.y * 1e6:+.2f} µm"
            if isinstance(poi, Point)
            else ""
        )
        return (
            f"Point of interest proposed by {proposal.provenance.get('proposer', '?')} "
            f"at {clock(proposal.created_at)} on {image}{value}."
        )

    def _state_words(self) -> tuple:
        return "Applied", self._task_name

    def _marker(self, id: str, col: float, row: float, color: str, legend) -> None:
        from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

        self._controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id=id,
                points=[(col, row)],
                color=color,
                selected_color=color,
                marker="+",
                size=14,
                edge_width=1.2,
                legend_label=legend,
                add_on_right_click=False,
                removable=False,
            ),
        )

    def _draw_values(self) -> None:
        """The proposed point, magenta: the one that stands, and drags."""
        poi = self._proposal.values.get("poi") if self._proposal else None
        if isinstance(poi, Point):
            px = conversions.microscope_image_to_image_coordinates(
                poi, self._image.data.shape, self._image.metadata.pixel_size.x
            )
            col, row = px.x, px.y
        else:
            row = self._image.data.shape[0] / 2
            col = self._image.data.shape[1] / 2
        self._marker("poi", col, row, "magenta", "Point of Interest")
        self._controller.arm_overlay(
            BeamType.ION, "poi", label="POI", icon="mdi:map-marker"
        )

    def _draw_confirmed(self, decision: Decision) -> None:
        """The decided point takes over as the one that stands (magenta); the
        proposal it replaced stays beside it in orange, so the delta shows."""
        if self._image is None:
            return
        confirmed = decision.values.get("poi")
        if not isinstance(confirmed, Point):
            return
        # an overlay keeps the colour it was made with, so the proposed marker
        # is remade under another id to turn orange
        proposed = self._controller.overlay_points(BeamType.ION, "poi")
        if proposed:
            self._controller.remove_overlay(BeamType.ION, "poi")
            col, row = proposed[0]
            self._marker("proposed", col, row, ORANGE_COLOR, "Proposed")
        px = conversions.microscope_image_to_image_coordinates(
            confirmed, self._image.data.shape, self._image.metadata.pixel_size.x
        )
        self._marker("confirmed", px.x, px.y, "magenta", "Point of Interest")

    def current_values(self) -> Dict[str, Any]:
        proposal = self._proposal
        if proposal is None:
            return {}
        if self._image is None:
            return dict(proposal.values)
        pts = self._controller.overlay_points(BeamType.ION, "poi")
        if not pts:
            return dict(proposal.values)
        col, row = pts[0]
        point = conversions.image_to_microscope_image_coordinates(
            Point(x=col, y=row), self._image.data, self._image.metadata.pixel_size.x
        )
        return {"poi": Point(x=point.x, y=point.y)}

    def set_proposal(
        self, experiment: Experiment, item: Any, task_name: str, proposal: Proposal
    ) -> None:
        super().set_proposal(experiment, item, task_name, proposal)
        # a delta only means something against the one image the values sit on
        self._controller.widget.set_sem_visible(False)


@register_review_renderer(DETECTION)
class DetectionReviewRenderer(TaskResultReviewRenderer):
    """Where the model put the features, for someone to correct: every feature
    on the image it ran on, each in its own colour and labelled with its name.

    Click a marker to select it and drag it where it belongs -- one overlay
    holding every point, which is what ``PointsSpec`` is for ("POI / spot burn
    / detection features", with per-point ``colors`` and ``labels``). Picking
    the feature first, from a list beside the canvas, was a workaround for a
    limit that is not there.

    The one kind asked *during* a task rather than after it (FIB-1025), so the
    run is parked on this answer and the verbs mean what they say right now:
    Confirm hands the points back and milling carries on, Reject fails the
    task. Confirming nothing moved is the commonest answer and the one that
    records the model as right.
    """

    PENDING_HINT = "Enter — these are right; drag a marker to correct one first"
    CONFIRM_LABEL = "Confirm"
    OVERLAY = "features"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.btn_put_back = QPushButton("Put them back")
        self.btn_put_back.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_put_back.setToolTip("Move every feature back where the model put it")
        self.btn_put_back.clicked.connect(self._draw_values)
        # Beside the verbs, because it undoes an edit rather than deciding.
        actions = self.layout().itemAt(self.layout().count() - 1).layout()
        actions.insertWidget(actions.count() - 2, self.btn_put_back)

    def _state_words(self) -> tuple:
        return "Applied", self._task_name

    def _features(self) -> List[Dict[str, Any]]:
        proposal = self._proposal
        values = proposal.values.get("features") if proposal else None
        return [f for f in (values or []) if isinstance(f, dict)]

    def _beam(self) -> BeamType:
        """The canvas the base put the image on: an electron image goes on the
        electron one, anything else on the ion one. Detections run on both
        beams, so this cannot be assumed."""
        if self._electron is None and _beam_of(self._image) is BeamType.ELECTRON:
            return BeamType.ELECTRON
        return BeamType.ION

    @staticmethod
    def _colour(name: str) -> str:
        """The colour the feature carries itself, so a lamella centre looks the
        same here as it does in the detection widget. Magenta for one this
        build does not know, which is a feature from a newer model rather than
        an error."""
        try:
            from fibsem.detection.detection import get_feature

            return str(getattr(get_feature(name), "color", "") or "magenta")
        except Exception:  # noqa: BLE001 - an unknown name is not a failure
            return "magenta"

    def _points(
        self, id: str, features: List[Dict[str, Any]], colour: str = ""
    ) -> None:
        from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

        points, colours, labels = [], [], []
        for feature in features:
            px = feature.get("px")
            if not isinstance(px, Point):
                continue
            name = str(feature.get("name") or "")
            points.append((px.x, px.y))
            colours.append(colour or self._colour(name))
            labels.append(name)
        if not points:
            return
        self._controller.set_overlay(
            self._beam(),
            PointsSpec(
                id=id,
                points=points,
                colors=colours,
                labels=labels,
                color=colours[0],
                selected_color="yellow",
                marker="+",
                size=14,
                edge_width=1.2,
                add_on_right_click=False,
                removable=False,
            ),
        )

    def _show_proposal(self) -> None:
        super()._show_proposal()
        # The base draws values only on the ion branch, because every kind
        # before this one recorded them on an ion image. A detection on an
        # electron image is drawn here instead.
        if self._image is not None and self._beam() is BeamType.ELECTRON:
            self._draw_values()

    def _draw_values(self) -> None:
        """Every feature where the model put it, in its own colour, labelled.

        A feature's point is already in image pixels -- it is where the model
        put it on this image -- so unlike a point of interest there is nothing
        to convert.
        """
        if self._image is None:
            return
        self._points(self.OVERLAY, self._features())
        editable = self._decided is None and self._applied is None
        self.btn_put_back.setVisible(editable)
        self._controller.arm_overlay(
            self._beam(),
            self.OVERLAY if editable else None,
            label="Features",
            icon="mdi:map-marker",
        )

    def _draw_confirmed(self, decision: Decision) -> None:
        """What was decided in each feature's colour; what the model proposed
        stays under it in orange, so the correction is visible and not only
        recorded."""
        if self._image is None:
            return
        self._points("proposed", self._features(), colour=ORANGE_COLOR)
        decided = [
            f for f in decision.values.get("features") or [] if isinstance(f, dict)
        ]
        self._points(self.OVERLAY, decided or self._features())
        self.btn_put_back.setVisible(False)

    def current_values(self) -> Dict[str, Any]:
        """Wherever the markers have been left. Every feature is answered, moved
        or not: the task is waiting for the whole set, and an unmoved one is
        the answer that says the model was right."""
        proposal = self._proposal
        if proposal is None:
            return {}
        features = self._features()
        if self._image is None:
            return dict(proposal.values)
        points = self._controller.overlay_points(self._beam(), self.OVERLAY)
        if len(points) != len(features):
            # The overlay is rebuilt from the features every time, so this is
            # a drawing that never happened rather than an edit to read back.
            return dict(proposal.values)
        return {
            "features": [
                # float(), because the canvas hands back numpy scalars and the
                # record is YAML: safe_dump refuses an np.float64 outright, so
                # a decision made by dragging would record fine and then fail
                # to save.
                {
                    "name": str(f.get("name") or ""),
                    "px": Point(x=float(col), y=float(row)),
                }
                for f, (col, row) in zip(features, points)
            ]
        }

    def _fact(self) -> str:
        proposal = self._proposal
        if proposal is None:
            return ""
        provenance = proposal.provenance
        names = ", ".join(str(f.get("name") or "?") for f in self._features()) or "none"
        model = str(provenance.get("proposer") or "?")
        checkpoint = os.path.basename(str(provenance.get("checkpoint") or ""))
        where = os.path.basename(str(provenance.get("reference_image") or ""))
        return (
            f"{model} found {names} at {clock(proposal.created_at)}"
            f"{f' on {where}' if where else ''}"
            f"{f' · checkpoint {checkpoint}' if checkpoint else ''}."
        )

    def set_proposal(
        self, experiment: Experiment, item: Any, task_name: str, proposal: Proposal
    ) -> None:
        super().set_proposal(experiment, item, task_name, proposal)
        # a delta only means something against the one image the values sit on
        if self._beam() is BeamType.ION:
            self._controller.widget.set_sem_visible(False)


def _beam_of(image: FibsemImage) -> Optional[BeamType]:
    settings = getattr(getattr(image, "metadata", None), "image_settings", None)
    return getattr(settings, "beam_type", None)


def _load_reference_image(
    experiment: Experiment, item: Any, proposal: Proposal, key: str = "reference_image"
) -> Optional[Union[FibsemImage, FluorescenceImage]]:
    """The image the proposal's values sit on, from its provenance: a file
    name relative to the item's folder (``Experiment.item_path``). A
    fluorescence result (an OME-TIFF, channels and planes) loads as a
    ``FluorescenceImage``; a beam image as a ``FibsemImage``."""
    path = proposal.provenance.get(key)
    if not path:
        return None
    path = os.path.join(str(experiment.item_path(item)), path)
    if not os.path.exists(path):
        logging.warning(f"Reference image for review not found: {path}")
        return None
    fluorescence = path.lower().endswith((".ome.tif", ".ome.tiff"))
    try:
        if fluorescence:
            return FluorescenceImage.load(path)
        return FibsemImage.load(path)
    except Exception:
        logging.exception(f"Could not load the reference image for review: {path}")
        return None


def review_preview(image: Any) -> Optional[Any]:
    """What an agent is shown of a proposal's image: the beam image itself,
    or a fluorescence result's channel composite (max over z, tinted per
    channel), the same one the FM canvas and the thumbnail show."""
    if isinstance(image, FluorescenceImage):
        from fibsem.fm.preview import composite_projection

        try:
            return composite_projection(image)
        except Exception:
            logging.exception("Could not project the fluorescence image for review")
            return None
    return image


@register_review_renderer(OVERVIEW_POSITIONS)
class OverviewPositionsReviewRenderer(TaskResultReviewRenderer):
    """Where the lamellae go on this grid: the one review whose confirm makes
    items rather than editing the one it is on.

    The grid's overview, the lamellae it already has drawn locked for context,
    and the positions being placed drawn over them. Confirm creates one lamella
    per placed position; the locked ones are not part of it, and are not
    editable here -- one may already have been milled, and moving a lamella is
    the Lamella tab's job.
    """

    PENDING_HINT = "Enter — create the lamellae you have placed"
    CONFIRM_LABEL = "Confirm"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._microscope: Any = None
        self._drafts: List[Any] = []  # LamellaPoses, in placement order
        # Placements that have not been confirmed, per run of the producing
        # task. One renderer serves every row of this kind, so without this a
        # glance at another grid would throw away what had been placed here.
        # Keyed on the run, so a re-run -- a new image, possibly a new stage
        # position -- never inherits marks made on the old one.
        self._drafts_by_run: Dict[str, List[Any]] = {}
        super().__init__(parent)

    # -- the view -------------------------------------------------------------

    def _build_view(self) -> QWidget:
        from fibsem.ui.widgets.stored_overview_canvas import StoredOverviewCanvas

        self.canvas = StoredOverviewCanvas()
        self.canvas.position_add_requested.connect(self._on_add_requested)
        self.canvas.draft_remove_requested.connect(self._on_remove_requested)
        return self.canvas

    def set_microscope(self, microscope: Any) -> None:
        """Placing a position needs the instrument's geometry; reading this
        review does not. Without one the overview and what is on it still
        show, and the line says why nothing can be placed."""
        self._microscope = microscope
        self._refresh_line()

    # -- what this kind shows -------------------------------------------------

    def _show_proposal(self) -> None:
        run = self._proposal.task_id
        self._drafts = self._drafts_by_run.get(run) or list(
            self._proposal.values.get("positions") or []
        )
        self._drafts_by_run[run] = self._drafts
        self.canvas.clear()
        shown = False
        if self._image is not None:
            self.canvas.set_image(self._image)
            shown = True
        self.canvas.setVisible(shown)
        self.no_image.setVisible(not shown)
        self.no_image.setText("" if shown else "The grid's overview could not be read.")
        self._draw_context()
        self._draw_drafts()

    def _draw_context(self) -> None:
        """The lamellae this grid already has, locked: the operator is placing
        into a populated picture, so they add what is missing instead of a
        second set."""
        experiment, item = self._experiment, self._item
        if experiment is None or item is None:
            return
        existing = []
        for lamella in getattr(experiment, "positions", []):
            if getattr(lamella, "grid_id", None) != getattr(item, "id", None):
                continue
            pose = getattr(
                getattr(lamella, "milling_pose", None), "stage_position", None
            )
            if pose is None:
                continue
            place = deepcopy(pose)
            place.name = lamella.name
            existing.append(place)
        self.canvas.set_positions(existing, movable=False)

    def _placing(self) -> bool:
        """Placing is for a decision that has not been made. A decided
        proposal -- including one its own producer confirmed, which is what a
        to-check row is -- is looked at, not changed; changing what it created
        is a re-run of the overview."""
        return self._decided is None and self._applied is None

    def _draw_drafts(self) -> None:
        places = []
        for poses in self._drafts:
            place = getattr(getattr(poses, "milling", None), "stage_position", None)
            if place is not None:
                places.append(place)
        self.canvas.set_draft_positions(places)
        self.canvas.set_placing_enabled(self._placing())
        self._refresh_line()

    # -- placing --------------------------------------------------------------

    def _on_add_requested(self, position: Any, _record_id: Any = None) -> None:
        """A position marked on the overview, turned into the poses a lamella
        is made from there and then: the geometry is read here, where there is
        an instrument, and never where the decision is applied."""
        if not self._placing():
            return  # decided: the canvas offers nothing here either
        if self._microscope is None:
            notification_service.show_toast(
                "Connect to a microscope to place positions: a lamella's poses "
                "are built with the instrument's geometry.",
                "warning",
            )
            return
        from fibsem.applications.autolamella.poses import build_lamella_poses

        try:
            self._drafts.append(
                build_lamella_poses(microscope=self._microscope, position=position)
            )
        except Exception as e:  # noqa: BLE001 - said to the user, not raised
            logging.error(f"Could not place a position: {e}")
            notification_service.show_toast(str(e), "error")
            return
        self._draw_drafts()

    def _on_remove_requested(self, index: int) -> None:
        if not self._placing():
            return
        if 0 <= index < len(self._drafts):
            self._drafts.pop(index)
            self._draw_drafts()

    # -- the decision ---------------------------------------------------------

    def current_values(self) -> Dict[str, Any]:
        return {"positions": list(self._drafts)}

    def set_to_check(self, applied: Optional[Decision]) -> None:
        super().set_to_check(applied)
        self._draw_drafts()

    def set_read_only(self, decided: Optional[Decision]) -> None:
        super().set_read_only(decided)
        self._draw_drafts()

    def _state_words(self) -> tuple:
        return "Decided", self._task_name

    def _fact(self) -> str:
        proposal = self._proposal
        if proposal is None:
            return ""
        name = getattr(self._item, "name", "this grid")
        proposed = len(proposal.values.get("positions") or [])
        head = (
            f"{proposed} position(s) proposed for {name}"
            if proposed
            else f"Nothing was proposed: place the lamellae for {name}"
        )
        when = clock(proposal.created_at)
        where = os.path.basename(str(proposal.provenance.get("reference_image") or ""))
        return f"{head}, on its overview{f' {where}' if where else ''} at {when}."

    def _refresh_line(self) -> None:
        super()._refresh_line()
        if self._proposal is None:
            return
        if self._applied is not None and self._decided is None:
            # Nobody was asked, so nothing was placed. Saying only "decided
            # automatically" leaves the reader with an overview, an obvious
            # next thought, and no way to act on it here -- placing is an edit
            # to the grid, not a second decision on a record that has one. So
            # the line says where that is done.
            made = len(self._applied.values.get("positions") or [])
            if made:
                self.line.setText(
                    f"Created {made} lamella{'e' if made != 1 else ''} "
                    "automatically · not checked yet"
                )
            else:
                self.line.setText(
                    "Decided automatically · no lamellae were placed · "
                    "add them on the Grids tab"
                )
                self.line.setToolTip(
                    f"{self._fact()}\n"
                    "Nobody was asked where the lamellae go, so none were "
                    "placed. Add them on the Grids tab, or set this task to "
                    "Review and run it again to place them here."
                )
            return
        n = len(self._drafts)
        if self._decided is None and self._applied is None:
            self.btn_confirm.setText(
                f"Confirm · add {n} lamella{'e' if n != 1 else ''}"
                if n
                else "Confirm · add none"
            )
            if self._microscope is None:
                self.line.setText(
                    "Connect a microscope to place positions · "
                    + self.line.text().lower()
                )


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


class _UnknownKindRenderer(ReviewRenderer):
    """What the host shows for a kind nothing has registered a renderer for:
    the facts, and the two verbs on the proposed values as they stand."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._proposal: Optional[Proposal] = None
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setStyleSheet(_READOUT_STYLE)
        self.btn_confirm = QPushButton("Confirm as proposed")
        self.btn_confirm.setStyleSheet(stylesheets.CONFIRM_BUTTON_STYLESHEET)
        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.btn_confirm)
        actions.addWidget(self.btn_reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.btn_confirm.clicked.connect(self.confirm_requested)
        self.btn_reject.clicked.connect(self.reject_requested)

    def set_proposal(self, experiment, item, task_name, proposal) -> None:
        self._proposal = proposal
        self.label.setText(
            f"{getattr(item, 'name', '')} · {task_name}\n"
            f"kind {proposal.kind!r} has no review renderer.\n"
            f"proposed: {proposal.values}"
        )

    def current_values(self) -> Dict[str, Any]:
        return dict(self._proposal.values) if self._proposal is not None else {}

    def set_read_only(self, decided: Optional[Decision]) -> None:
        self.btn_confirm.setEnabled(decided is None)
        self.btn_reject.setEnabled(decided is None)

    def set_to_check(self, applied: Optional[Decision]) -> None:
        self.btn_confirm.setText("Acknowledge" if applied else "Confirm as proposed")


# ---------------------------------------------------------------------------
# Inbox rows
# ---------------------------------------------------------------------------


class _InboxRow(QWidget):
    """dot | name · task | one muted word on the right, on one line.

    The row's job is to let you pick one; the detail is the panel's line and
    its tooltip, and the row's tooltip. The dot's colour is the state: the
    Review colour waiting, grey to check, green confirmed, red rejected, dim
    for a superseded one."""

    def __init__(
        self,
        colour: str,
        name: str,
        task: str,
        right: str,
        dim: bool = False,
        quiet: bool = False,
    ) -> None:
        """``quiet``: a decided row. The filled dot is the "act on me" signal;
        a decided row keeps the colour but hollows the dot, so it reads as
        done. ``dim``: superseded, smaller and in the muted colour."""
        super().__init__()
        # The list paints the row's background and selection; the widget must
        # not paint the app's default one over it.
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(9)
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        self.dot.setStyleSheet(
            f"background: transparent; border-radius: 4px; border: 1.5px solid {colour};"
            if quiet or dim
            else f"background: {colour}; border-radius: 4px; border: none;"
        )
        layout.addWidget(self.dot, 0, Qt.AlignVCenter)
        # elided to the column, never clipped mid-glyph; the full name is the
        # tooltip. The font is set directly so the metrics match the style.
        font = QFont(self.font())
        font.setPixelSize(12 if not dim else 11)
        self.name = QLabel(
            QFontMetrics(font).elidedText(name, Qt.ElideRight, _ROW_NAME_WIDTH - 6)
        )
        self.name.setFont(font)
        self.name.setStyleSheet(_ROW_TASK_STYLE if dim else _ROW_NAME_STYLE)
        self.name.setFixedWidth(_ROW_NAME_WIDTH)
        self.name.setToolTip(name)
        layout.addWidget(self.name, 0, Qt.AlignVCenter)
        self.task = QLabel(task)
        self.task.setStyleSheet(_ROW_TASK_STYLE)
        self.task.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.task.setMinimumWidth(40)
        layout.addWidget(self.task, 1, Qt.AlignVCenter)
        self.right = QLabel(right)
        self.right.setStyleSheet(_ROW_RIGHT_STYLE)
        self.right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.right)


KIND_ALL = "all"
KIND_GRIDS = "grids"
KIND_LAMELLAE = "lamellae"


class _InboxFilterButton(QToolButton):
    """The filter menu: which kind of item to list, and whether to list only
    the proposals something waits on.

    The text field and the Decided chip sit on the filter row because they are
    used constantly; these two are set once and left, so they live behind an
    icon. The icon takes the accent while either is on, so a narrowed inbox is
    never mistaken for the whole one -- the same rule as the lamella list's
    grid filter.
    """

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(26, 26))
        self.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self.setPopupMode(QToolButton.InstantPopup)
        self.setFocusPolicy(Qt.NoFocus)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        # One of three, so they read as radio buttons: a checkable action in a
        # QMenu draws a tick whether or not its group is exclusive, and a
        # ticked "Grids and lamellae" looks like something you could untick.
        # The mark is the action's icon instead, which Qt draws for us.
        self._kind = KIND_ALL
        self._kinds: Dict[str, Any] = {}
        for key, text in (
            (KIND_ALL, "Grids and lamellae"),
            (KIND_GRIDS, "Grids only"),
            (KIND_LAMELLAE, "Lamellae only"),
        ):
            action = self._menu.addAction(text)
            action.triggered.connect(lambda _c=False, k=key: self.set_kind(k))
            self._kinds[key] = action
        self._menu.addSeparator()
        # "Holding a task", not "Waiting": the Waiting group is every pending
        # proposal, and most of those hold nothing. This is the smaller set
        # the run is actually stopped on, the one the renderer's line counts.
        self.held_only = self._menu.addAction("Holding a task")
        self.held_only.setCheckable(True)
        self.held_only.setToolTip("A later task is waiting on this decision")
        self.held_only.triggered.connect(self._on_changed)
        self._paint()

    @property
    def kind(self) -> str:
        return self._kind

    def set_kind(self, key: str) -> None:
        self._kind = key if key in self._kinds else KIND_ALL
        self._on_changed()

    @property
    def narrowing(self) -> bool:
        return self.kind != KIND_ALL or self.held_only.isChecked()

    def _on_changed(self, _checked: bool = False) -> None:
        self._paint()
        self.changed.emit()

    def _paint(self) -> None:
        for key, action in self._kinds.items():
            action.setIcon(
                fibsem_icon(
                    "mdi:radiobox-marked"
                    if key == self._kind
                    else "mdi:radiobox-blank",
                    color=ACCENT_COLOR if key == self._kind else GRAY_ICON_COLOR,
                )
            )
        on = self.narrowing
        self.setIcon(
            fibsem_icon(
                "mdi:filter-variant", color=ACCENT_COLOR if on else GRAY_ICON_COLOR
            )
        )
        words = []
        if self.kind != KIND_ALL:
            words.append(self._kinds[self.kind].text().lower())
        if self.held_only.isChecked():
            words.append("what holds a task")
        self.setToolTip(
            f"Showing {', '.join(words)}" if words else "Filter by item or hold"
        )


def _group_header(text: str) -> QListWidgetItem:
    header = QListWidgetItem(text)
    header.setFlags(Qt.NoItemFlags)
    header.setData(Qt.UserRole, None)
    return header


class _GroupHeaderRow(QWidget):
    """A group header, with room on its right for an action: the to-check
    group's "Mark all as checked"."""

    def __init__(self, text: str, action: str = "", slot=None) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(3, 0, 8, 0)
        self.layout_.setSpacing(12)
        # smaller and quieter than the rows it labels
        self.label = QLabel(text)
        self.label.setStyleSheet(
            f"color: {GRAY_SECONDARY_COLOR}; font-size: 12px; background: transparent;"
        )
        self.layout_.addWidget(self.label)
        self.layout_.addStretch(1)
        self.button: Optional[QPushButton] = None
        if action:
            self.button = QPushButton(action)
            self.button.setFlat(True)
            self.button.setCursor(Qt.PointingHandCursor)
            self.button.setFocusPolicy(Qt.NoFocus)
            self.button.setStyleSheet(
                f"QPushButton {{ color: {ACCENT_COLOR}; background: transparent; "
                "border: none; font-size: 11px; padding: 0 2px; }"
                f"QPushButton:hover {{ color: {GRAY_TEXT_COLOR}; }}"
            )
            self.button.clicked.connect(slot)
            self.layout_.addWidget(self.button)


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


class ReviewTabWidget(QWidget):
    """Host: the inbox on the left, the current proposal's renderer on the right."""

    decided = pyqtSignal(str, str)  # item_id, task_name -- after it was applied
    pending_changed = pyqtSignal(int)  # how many are waiting, for the tab badge
    # waiting, to check: only waiting means the run is stalled on someone
    counts_changed = pyqtSignal(int, int)
    open_item_requested = pyqtSignal(object)  # go to this item where it is edited

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        # (item, task_name, proposal, state) per list row, state one of
        # "waiting" (a decision gates a consumer), "check" (the producer applied
        # it; a look is owed) or "decided" (read-only). Only waiting is pending.
        self._entries: List[tuple] = []
        self._renderers: Dict[str, ReviewRenderer] = {}
        self._microscope: Any = None
        self._running = False

        self.list = QListWidget()
        self.list.setMinimumWidth(300)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # No focus rectangle on the current row (the workflow list does the
        # same); the selection fill is the only mark.
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setStyleSheet(
            "QListWidget { outline: none; } "
            "QListWidget::item { border: none; } "
            "QListWidget::item:selected { border: none; }"
        )
        self.list.currentRowChanged.connect(self._on_row_changed)
        # The filter row: a field for the common case, a chip for the one
        # group that grows without bound, and a menu for what is set once.
        # It sits above the list rather than on the first group header, which
        # is where the Decided toggle used to be re-homed every refresh.
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("Filter by name or task")
        self.filter_text.setClearButtonEnabled(True)
        self.filter_text.setMinimumHeight(26)
        # Qt has no placeholder-only selector, so this sizes the typed text
        # with it -- which is right: the field is chrome, not content.
        self.filter_text.setStyleSheet("QLineEdit { font-size: 11px; }")
        self.filter_text.setToolTip(
            "Every word you type must appear in the item's name or the task's."
        )
        self.filter_text.textChanged.connect(lambda _t: self.refresh())
        self.show_decided = QPushButton("Decided")
        self.show_decided.setCheckable(True)
        self.show_decided.setCursor(Qt.PointingHandCursor)
        self.show_decided.setStyleSheet(MODALITY_CHIP_STYLE)
        self.show_decided.setFocusPolicy(Qt.NoFocus)
        self.show_decided.setToolTip(
            "List proposals that have been confirmed or rejected, including "
            "ones a re-run superseded, read-only."
        )
        self.show_decided.toggled.connect(lambda _on: self.refresh())
        self.filters = _InboxFilterButton()
        self.filters.changed.connect(self.refresh)
        filter_row = QWidget()
        row_layout = QHBoxLayout(filter_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(self.filter_text, 1)
        row_layout.addWidget(self.show_decided)
        row_layout.addWidget(self.filters)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(filter_row)
        left_layout.addWidget(self.list, 1)

        self.empty = QLabel("Nothing is waiting for a decision.")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet(_MUTED_STYLE)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 99999])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Qt dispatches a shortcut before the focused widget sees the key, so
        # these two would fire while someone types in the filter field: every
        # term with an "r" in it would reject the current proposal, and Enter
        # would confirm it. The field's keys are the field's.
        confirm = QShortcut(QKeySequence(Qt.Key_Return), self)
        confirm.setContext(Qt.WidgetWithChildrenShortcut)
        confirm.activated.connect(self._on_confirm_shortcut)
        reject = QShortcut(QKeySequence("R"), self)
        reject.setContext(Qt.WidgetWithChildrenShortcut)
        reject.activated.connect(self._on_reject_shortcut)

    # -- wiring --------------------------------------------------------------

    def set_microscope(self, microscope: Any) -> None:
        """The instrument, for the one renderer that needs its geometry.

        Reading a review needs nothing but the record, and rejecting one needs
        nothing either. *Placing* a position does: a lamella's poses are built
        from the instrument's geometry, so a review that creates lamellae can
        only be answered at a connected microscope. Every other kind ignores
        this, and this one says so rather than failing at the confirm.
        """
        self._microscope = microscope
        for renderer in self._renderers.values():
            setter = getattr(renderer, "set_microscope", None)
            if setter is not None:
                setter(microscope)

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        if self._experiment is not None:
            try:
                self._experiment.decided.disconnect(self._on_experiment_decided)
                self._experiment.asked.disconnect(self._on_experiment_decided)
            except Exception:
                pass
        self._experiment = experiment
        if experiment is not None:
            # Fires on the thread decide() ran on, which is this one (main).
            experiment.decided.connect(self._on_experiment_decided)
            # A question recorded mid-task. Without this the inbox only
            # re-derives when a task *finishes*, and an in-run question is
            # raised halfway through one -- so it would not appear until
            # something else happened to refresh the tab.
            experiment.asked.connect(self._on_experiment_decided)
        self.refresh()

    def set_running(self, running: bool) -> None:
        self._running = running
        for renderer in self._renderers.values():
            renderer.set_running(running)

    @property
    def pending_count(self) -> int:
        return getattr(self, "_pending", 0)

    @property
    def check_count(self) -> int:
        return getattr(self, "_to_check", 0)

    # -- filtering -----------------------------------------------------------

    def _passes(self, item: Any, task_name: str, proposal: Proposal) -> bool:
        """Whether one proposal's row is listed under the current filter.

        The text is matched against the item's name and the task's together,
        as one string: every word you type has to appear somewhere in it, so
        "rough" finds every rough result, "whale rough" finds one lamella's.
        Plain case-insensitive substrings -- no globs, no fuzzy matching, so a
        near miss is always explainable by reading the row.

        Held asks what the run is waiting on you for, so it wants both halves:
        a later task requires this one, *and* this proposal is still pending. A
        result the producer already applied holds nothing -- the tasks after it
        have run -- so listing it under "held only" would be a lie the row
        itself cannot correct.
        """
        kind = self.filters.kind
        if kind != KIND_ALL:
            grid = isinstance(item, GridRecord)
            if grid != (kind == KIND_GRIDS):
                return False
        if self.filters.held_only.isChecked():
            if not proposal.pending:
                return False
            if not waiting_on(self._experiment, task_name, item):
                return False
        terms = self.filter_text.text().lower().split()
        if terms:
            haystack = f"{item.name} {task_name}".lower()
            if not all(term in haystack for term in terms):
                return False
        return True

    def _shown(self, proposals: List[tuple]) -> List[tuple]:
        return [p for p in proposals if self._passes(p[0], p[1], p[2])]

    def _count(self, shown: int, total: int) -> str:
        """ "2 of 7" while filtering, so a short list is never read as a short
        queue."""
        return f"{shown} of {total}" if shown != total else str(total)

    def refresh(self) -> None:
        """Re-derive the inbox from the experiment. Keeps the selection on the
        same (item, task) while it is listed; when it has left the list (just
        decided, just acknowledged) the row that took its place is selected,
        so a run of acknowledgements is a run of Returns."""
        current = self._current_key()
        previous_index = self._current_index()
        self._entries = []
        self.list.blockSignals(True)
        self.list.clear()
        self._headers: List[_GroupHeaderRow] = []
        hidden = 0
        if self._experiment is not None:
            experiment = self._experiment
            all_pending = experiment.pending_proposals()
            # A question the run is parked on, asked mid-task and waiting to be
            # told the answer (FIB-1025). Not a new urgency -- it is the
            # existing "holds a task" distinction with the stakes raised, since
            # nothing else runs while it is up -- but it is worth its own group
            # above the rest, because everything under Waiting can be left.
            all_holding = [e for e in all_pending if e[2].asking]
            all_waiting = [e for e in all_pending if not e[2].asking]
            holding = self._shown(all_holding)
            hidden += len(all_holding) - len(holding)
            if holding:
                self._add_header(
                    f"Holding the workflow · "
                    f"{self._count(len(holding), len(all_holding))}"
                )
            for item, task_name, proposal in holding:
                self._add_row(
                    summary=f"{item.name} · {task_name} · asking now",
                    widget=_InboxRow(ORANGE_COLOR, item.name, task_name, "asking now"),
                    entry=(item, task_name, proposal, "waiting"),
                    tooltip=f"{task_name} is parked on this answer; "
                    "nothing else runs until you give it.",
                )
            waiting = self._shown(all_waiting)
            hidden += len(all_waiting) - len(waiting)
            if waiting:
                self._add_header(
                    f"Waiting · {self._count(len(waiting), len(all_waiting))}"
                )
            for item, task_name, proposal in waiting:
                held = waiting_on(experiment, task_name, item)
                self._add_row(
                    summary=f"{item.name} · {task_name} · waiting",
                    widget=_InboxRow(
                        ORANGE_COLOR, item.name, task_name, age(proposal.created_at)
                    ),
                    entry=(item, task_name, proposal, "waiting"),
                    tooltip="Waiting for your decision"
                    + (f" · held: {', '.join(held)}" if held else ""),
                )
            # What is pending is a fact about the experiment, not about what
            # the filter is showing: the tab badge and the stall check read
            # these, so a narrowed list must not shrink them.
            pending = len(all_pending)
            all_to_check = experiment.proposals_to_check()
            to_check = self._shown(all_to_check)
            hidden += len(all_to_check) - len(to_check)
            if to_check:
                self._add_header(
                    f"To check · {self._count(len(to_check), len(all_to_check))}",
                    "Mark all as checked",
                    self.acknowledge_all,
                )
            for item, task_name, proposal in to_check:
                applied = proposal.applied or proposal.current
                failed = bool(proposal.provenance.get("failure"))
                self._add_row(
                    summary=f"{item.name} · {task_name} · to check",
                    widget=_InboxRow(
                        DEFECT_RED_COLOR if failed else GRAY_SECONDARY_COLOR,
                        item.name,
                        task_name,
                        age(applied.timestamp),
                    ),
                    entry=(item, task_name, proposal, "check"),
                    tooltip=describe_decision(proposal, experiment),
                )
            if self.show_decided.isChecked():
                all_decided = decided_proposals(experiment)
                decided = [p for p in all_decided if self._passes(p[0], p[1], p[2])]
                hidden += len(all_decided) - len(decided)
                if decided:
                    self._add_header(
                        f"Decided · {self._count(len(decided), len(all_decided))}"
                    )
                for item, task_name, proposal, superseded in decided:
                    d = proposal.current
                    rejected = d.outcome is DecisionOutcome.Rejected
                    # Withdrawn is not an answer, so it must not read as one:
                    # the green tick beside a question nobody answered says
                    # somebody agreed with it.
                    withdrawn = proposal.withdrawn
                    if superseded or withdrawn:
                        colour = GRAY_SECONDARY_COLOR
                    elif rejected:
                        colour = DEFECT_RED_COLOR
                    else:
                        colour = OK_COLOR
                    word = (
                        "withdrawn"
                        if withdrawn
                        else "rejected"
                        if rejected
                        else "confirmed"
                    )
                    self._add_row(
                        summary=f"{item.name} · {task_name} · {word}"
                        + (" · superseded" if superseded else ""),
                        widget=_InboxRow(
                            colour,
                            item.name,
                            task_name,
                            clock(d.timestamp),
                            dim=superseded,
                            quiet=True,
                        ),
                        entry=(item, task_name, proposal, "decided"),
                        tooltip=describe_decision(proposal, experiment)
                        + ("\nSuperseded by a re-run." if superseded else ""),
                    )
        else:
            pending = 0
            all_to_check = []
        if not self._headers:
            self._add_header(
                f"Nothing matches · {hidden} hidden" if hidden else "Nothing waiting"
            )
        self.list.blockSignals(False)
        self._pending = pending
        self._to_check = len(all_to_check) if self._experiment is not None else 0
        self.pending_changed.emit(pending)
        self.counts_changed.emit(pending, self._to_check)

        select = None
        if current is not None:
            for i, (item, task_name, _p, _s) in enumerate(self._entries):
                if (item.id, task_name) == current:
                    select = i
                    break
        if select is None:
            # the row that took the decided one's place, or the last row
            select = min(previous_index or 0, max(len(self._entries) - 1, 0))
        if self._entries:
            self._select_entry(select)
        else:
            self.empty.setText(
                "No proposals match the filter."
                if hidden
                else "Nothing is waiting for a decision."
            )
            self.stack.setCurrentWidget(self.empty)

    def _add_header(self, text: str, action: str = "", slot=None) -> _GroupHeaderRow:
        header = _group_header(text)
        header.setText("")  # the widget draws it
        header.setData(Qt.UserRole + 1, text)
        self.list.addItem(header)
        widget = _GroupHeaderRow(text, action, slot)
        header.setSizeHint(QSize(0, max(widget.sizeHint().height(), 24)))
        self.list.setItemWidget(header, widget)
        self._headers.append(widget)
        return widget

    def _add_row(
        self, summary: str, widget: QWidget, entry: tuple, tooltip: str = ""
    ) -> None:
        # The widget draws the row; the summary rides on a data role rather
        # than as the item's text, which would paint through it.
        row = QListWidgetItem()
        row.setData(Qt.UserRole, len(self._entries))
        row.setData(Qt.UserRole + 1, summary)
        # A fixed height: the widget's own size hint is taken before its
        # stylesheets apply, and came out a line short.
        widget.setFixedHeight(_ROW_HEIGHT)
        row.setSizeHint(QSize(0, _ROW_HEIGHT))
        if tooltip:
            row.setToolTip(tooltip)
        self.list.addItem(row)
        self.list.setItemWidget(row, widget)
        self._entries.append(entry)

    # -- selection -----------------------------------------------------------

    def row_summaries(self) -> List[str]:
        """The list as text: group headers and one line per row."""
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            out.append(item.data(Qt.UserRole + 1) or item.text())
        return out

    def _current_key(self):
        index = self._current_index()
        if index is None:
            return None
        item, task_name, _p, _s = self._entries[index]
        return (item.id, task_name)

    def _current_index(self) -> Optional[int]:
        row = self.list.currentItem()
        if row is None:
            return None
        index = row.data(Qt.UserRole)
        return index if isinstance(index, int) else None

    def _select_entry(self, index: int) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == index:
                self.list.setCurrentRow(i)
                return

    def _on_row_changed(self, _row: int) -> None:
        index = self._current_index()
        if index is None or self._experiment is None:
            return
        item, task_name, proposal, state = self._entries[index]
        renderer = self._renderer_for(proposal.kind)
        renderer.set_proposal(self._experiment, item, task_name, proposal)
        renderer.set_running(self._running)
        renderer.set_read_only(proposal.current if state == "decided" else None)
        renderer.set_to_check(
            (proposal.applied or proposal.current) if state == "check" else None
        )
        same = [i for i, e in enumerate(self._entries) if e[3] == state]
        nth = same.index(index) + 1 if index in same else 0
        prefix = {"decided": "decided ", "check": "to check "}.get(state, "")
        renderer.set_position(f"{prefix}{nth} of {len(same)}")
        self.stack.setCurrentWidget(renderer)

    def _renderer_for(self, kind: str) -> ReviewRenderer:
        renderer = self._renderers.get(kind)
        if renderer is None:
            cls = REVIEW_RENDERERS.get(kind, _UnknownKindRenderer)
            renderer = cls()
            renderer.confirm_requested.connect(self.confirm_current)
            renderer.reject_requested.connect(self.reject_current)
            renderer.open_item_requested.connect(self.open_item_requested)
            setter = getattr(renderer, "set_microscope", None)
            if setter is not None:
                setter(self._microscope)
            self.stack.addWidget(renderer)
            self._renderers[kind] = renderer
        return renderer

    # -- the two verbs -------------------------------------------------------

    def _typing(self) -> bool:
        """Whether the filter field has the keyboard."""
        return self.filter_text.hasFocus()

    def _on_confirm_shortcut(self) -> None:
        if not self._typing():
            self.confirm_current()

    def _on_reject_shortcut(self) -> None:
        if not self._typing():
            self.reject_current()

    def confirm_current(self) -> None:
        """Confirm a waiting proposal with the values as the reviewer left
        them; acknowledge a to-check one with no values at all. Empty values
        write nothing through, so the acknowledgement is record-only by
        construction: it says someone looked, and that is all it says."""
        index = self._current_index()
        if index is None or self._experiment is None:
            return
        item, task_name, proposal, state = self._entries[index]
        if state == "decided":
            return
        if state == "check":
            values: Dict[str, Any] = {}
        else:
            values = self._renderer_for(proposal.kind).current_values()
        decision = Decision(
            outcome=DecisionOutcome.Confirmed,
            author=self._experiment.author(),
            via="review",
            values=values,
            task_id=proposal.task_id,  # the run shown, refused if it re-ran
        )
        self._apply(item, task_name, decision)

    def acknowledge_all(self) -> None:
        """Record a look on every to-check proposal listed, one decision each
        with no values (writes nothing), then one save. What was waiting is
        untouched. The list is the one shown, each by the run it showed: a task
        that re-ran since is refused, and stays to check."""
        experiment = self._experiment
        if experiment is None:
            return
        author = experiment.author()
        done = 0
        listed = [e for e in self._entries if e[3] == "check"]
        for item, task_name, proposal, _state in listed:
            result = experiment.decide(
                item.id,
                task_name,
                Decision(
                    outcome=DecisionOutcome.Confirmed,
                    author=author,
                    values={},
                    via="review",
                    task_id=proposal.task_id,
                ),
            )
            if result.applied:
                done += 1
        if not done:
            return
        try:
            experiment.save()
        except Exception:
            logging.exception("saving the experiment after acknowledging failed")
        self.decided.emit("", "")
        self.refresh()

    def reject_current(self) -> None:
        index = self._current_index()
        if index is None or self._experiment is None:
            return
        item, task_name, proposal, state = self._entries[index]
        if state == "decided":
            return
        reason, ok = QInputDialog.getText(
            self,
            "Reject",
            f"Why is there nothing further here for {item.name}? "
            f"{task_name} is marked failed; nothing that requires it runs.",
        )
        reason = reason.strip()
        if not ok or not reason:
            return
        decision = Decision(
            outcome=DecisionOutcome.Rejected,
            author=self._experiment.author(),
            via="review",
            reason=reason,
            task_id=proposal.task_id,
        )
        self._apply(item, task_name, decision)

    def _apply(self, item: Any, task_name: str, decision: Decision) -> None:
        assert self._experiment is not None
        result = self._experiment.decide(item.id, task_name, decision)
        if not result.applied:
            QMessageBox.warning(
                self,
                "Not applied",
                result.reason
                + ("\n\nStop the running task first." if result.running else ""),
            )
            return
        try:
            self._experiment.save()
        except Exception:
            logging.exception("saving the experiment after a decision failed")
        self.decided.emit(item.id, task_name)
        self.refresh()

    def _on_experiment_decided(self, _item_id: str, _task_name: str) -> None:
        # A decision landed through the other client of decide() (the agent
        # server); the inbox is re-derived either way.
        self.refresh()


def review_tab_icon():
    return fibsem_icon("mdi:clipboard-check-outline", color=GRAY_ICON_COLOR)
