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
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics, QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions
from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    TASK_RESULT,
    Author,
    AuthorKind,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import Experiment
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui import stylesheets
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
_ROW_NAME_STYLE = f"color: {GRAY_TEXT_COLOR}; font-size: 13px; font-weight: 600; background: transparent;"
_ROW_NAME_QUIET_STYLE = (
    f"color: {GRAY_TEXT_COLOR}; font-size: 13px; background: transparent;"
)
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


def waiting_on(experiment: Experiment, task_name: str) -> List[str]:
    """The tasks deferred until ``task_name``'s proposal is decided: every task
    that requires it, transitively, in workflow order."""
    protocol = getattr(experiment, "task_protocol", None)
    config = getattr(protocol, "workflow_config", None)
    if config is None:
        return []
    gated = {task_name}
    names: List[str] = []
    for task in config.tasks:
        if any(req in gated for req in task.requires):
            gated.add(task.name)
            names.append(task.name)
    return names


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
        self._gated: List[str] = []
        self._decided: Optional[Decision] = None
        self._applied: Optional[Decision] = None
        self._running = False
        self._position = ""

        self._controller = MicroscopeViewController(view=LamellaEditorView())
        self._controller.widget.show_beams()

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
        layout.addWidget(self._controller.widget, 1)
        layout.addWidget(self.line)
        layout.addLayout(actions)

        self.btn_confirm.clicked.connect(self.confirm_requested)
        self.btn_reject.clicked.connect(self.reject_requested)
        self.btn_open.clicked.connect(lambda: self.open_item_requested.emit(self._item))
        self._refresh_line()

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
        self._image = _load_reference_image(item, proposal)
        self._electron = _load_reference_image(item, proposal, "reference_image_eb")
        self._gated = waiting_on(experiment, task_name)
        self._decided = None
        self._applied = None
        self.title.setText(getattr(item, "name", ""))
        self.task_chip.setText(task_name)
        self.btn_confirm.setText(self.CONFIRM_LABEL)
        self.btn_confirm.setToolTip(self.PENDING_HINT)
        for overlay in ("poi", "proposed", "confirmed"):
            self._controller.remove_overlay(BeamType.ION, overlay)
        self._controller.arm_overlay(BeamType.ION, None)
        if self._image is not None:
            self._controller.set_image(BeamType.ION, self._image)
            self._draw_values()
        if self._electron is not None:
            self._controller.set_image(BeamType.ELECTRON, self._electron)
        self._controller.widget.set_sem_visible(self._electron is not None)
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
            if decided.outcome is DecisionOutcome.Rejected:
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
            text = f"Waiting for your decision · {_held_text(len(gated))}"
            colour = ORANGE_COLOR  # waiting on you now: the border's colour
            if gated:
                tip.append("Held until you decide: " + ", ".join(gated) + ".")
        if self._image is None:
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


def _load_reference_image(
    item: Any, proposal: Proposal, key: str = "reference_image"
) -> Optional[FibsemImage]:
    """The image the proposal's values sit on, from its provenance: a file
    name relative to the item's folder."""
    path = proposal.provenance.get(key)
    if not path:
        return None
    path = os.path.join(str(getattr(item, "path", "")), path)
    if not os.path.exists(path):
        logging.warning(f"Reference image for review not found: {path}")
        return None
    try:
        return FibsemImage.load(path)
    except Exception:
        logging.exception(f"Could not load the reference image for review: {path}")
        return None


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
        """``quiet``: a decided row. A filled dot and a bold name are the
        "act on me" signal; a decided row keeps the colour but hollows the
        dot and drops the bold, so it reads as done. ``dim``: superseded."""
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
        font.setPixelSize(13 if not dim else 11)
        font.setBold(not (dim or quiet))
        self.name = QLabel(
            QFontMetrics(font).elidedText(name, Qt.ElideRight, _ROW_NAME_WIDTH - 6)
        )
        self.name.setFont(font)
        self.name.setStyleSheet(
            _ROW_TASK_STYLE
            if dim
            else (_ROW_NAME_QUIET_STYLE if quiet else _ROW_NAME_STYLE)
        )
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


def _group_header(text: str) -> QListWidgetItem:
    header = QListWidgetItem(text)
    header.setFlags(Qt.NoItemFlags)
    header.setData(Qt.UserRole, None)
    return header


class _GroupHeaderRow(QWidget):
    """A group header, with room on its right for an action (the to-check
    group's "Mark all as checked") and, on the first header in the list, the
    Show decided toggle, so the list needs no row of its own above it."""

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

    def adopt(self, widget: QWidget) -> None:
        """Put a persistent widget (the Show decided toggle) at the right."""
        self.layout_.addWidget(widget)
        widget.show()


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
        # One persistent toggle, re-homed into the first group header on
        # every refresh (and taken back before the list is cleared, or the
        # clear would delete it with the header's widget).
        self.show_decided = QCheckBox("Show decided")
        self.show_decided.setStyleSheet(
            f"QCheckBox {{ color: {GRAY_SECONDARY_COLOR}; font-size: 11px; "
            "background: transparent; spacing: 4px; }"
        )
        self.show_decided.setFocusPolicy(Qt.NoFocus)
        self.show_decided.setToolTip(
            "List proposals that have been confirmed or rejected, including "
            "ones a re-run superseded, read-only."
        )
        self.show_decided.toggled.connect(lambda _on: self.refresh())
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
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

        confirm = QShortcut(QKeySequence(Qt.Key_Return), self)
        confirm.setContext(Qt.WidgetWithChildrenShortcut)
        confirm.activated.connect(self.confirm_current)
        reject = QShortcut(QKeySequence("R"), self)
        reject.setContext(Qt.WidgetWithChildrenShortcut)
        reject.activated.connect(self.reject_current)

    # -- wiring --------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        if self._experiment is not None:
            try:
                self._experiment.decided.disconnect(self._on_experiment_decided)
            except Exception:
                pass
        self._experiment = experiment
        if experiment is not None:
            # Fires on the thread decide() ran on, which is this one (main).
            experiment.decided.connect(self._on_experiment_decided)
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

    def refresh(self) -> None:
        """Re-derive the inbox from the experiment. Keeps the selection on the
        same (item, task) while it is listed; when it has left the list (just
        decided, just acknowledged) the row that took its place is selected,
        so a run of acknowledgements is a run of Returns."""
        current = self._current_key()
        previous_index = self._current_index()
        self._entries = []
        self.list.blockSignals(True)
        self.show_decided.setParent(self)  # before clear(): keep the toggle
        self.show_decided.hide()
        self.list.clear()
        self._headers: List[_GroupHeaderRow] = []
        if self._experiment is not None:
            experiment = self._experiment
            waiting = experiment.pending_proposals()
            if waiting:
                self._add_header(f"Waiting · {len(waiting)}")
            for item, task_name, proposal in waiting:
                held = waiting_on(experiment, task_name)
                self._add_row(
                    summary=f"{item.name} · {task_name} · waiting",
                    widget=_InboxRow(
                        ORANGE_COLOR, item.name, task_name, age(proposal.created_at)
                    ),
                    entry=(item, task_name, proposal, "waiting"),
                    tooltip="Waiting for your decision"
                    + (f" · held: {', '.join(held)}" if held else ""),
                )
            pending = len(self._entries)
            to_check = experiment.proposals_to_check()
            if to_check:
                self._add_header(
                    f"To check · {len(to_check)}",
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
                decided = decided_proposals(experiment)
                if decided:
                    self._add_header(f"Decided · {len(decided)}")
                for item, task_name, proposal, superseded in decided:
                    d = proposal.current
                    rejected = d.outcome is DecisionOutcome.Rejected
                    if superseded:
                        colour = GRAY_SECONDARY_COLOR
                    elif rejected:
                        colour = DEFECT_RED_COLOR
                    else:
                        colour = OK_COLOR
                    self._add_row(
                        summary=f"{item.name} · {task_name} · "
                        + ("rejected" if rejected else "confirmed")
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
            to_check = []
        if not self._headers:
            self._add_header("Nothing waiting")
        self._headers[0].adopt(self.show_decided)
        self.list.blockSignals(False)
        self._pending = pending
        self._to_check = len(to_check)
        self.pending_changed.emit(pending)
        self.counts_changed.emit(pending, len(to_check))

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
            self.stack.addWidget(renderer)
            self._renderers[kind] = renderer
        return renderer

    # -- the two verbs -------------------------------------------------------

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
        )
        self._apply(item, task_name, decision)

    def acknowledge_all(self) -> None:
        """Record a look on every to-check proposal, one decision each with no
        values (writes nothing), then one save. What was waiting is untouched."""
        experiment = self._experiment
        if experiment is None:
            return
        author = experiment.author()
        done = 0
        for item, task_name, _proposal in experiment.proposals_to_check():
            result = experiment.decide(
                item.id,
                task_name,
                Decision(
                    outcome=DecisionOutcome.Confirmed,
                    author=author,
                    values={},
                    via="review",
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
