"""The Review tab: every decision waiting on a person, in one place.

A producing task completes and leaves a proposal on its item (see
``proposals.py``); the consumer of that decision is deferred until someone
confirms or rejects it. This tab lists those proposals and hosts a renderer
per proposal *kind* -- a point of interest is one image and one marker, a
screening review will be a set of candidates with toggles -- so the tab
dispatches to a registered renderer rather than growing an ``if`` per kind.

Two verbs. **Confirm** submits whatever the renderer currently shows; the
delta against the proposal is computed by ``Experiment.decide``, never
declared here. **Reject** means *nothing further here*, and on a gating kind
that retires the item, which the button says out loud. There is no defer
button on purpose: items commit independently, so walking away is deferral.

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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
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
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions
from fibsem.applications.autolamella.proposals import (
    MILLING_SETUP,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import Experiment
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
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
    "MillingSetupReviewRenderer",
    "ReviewRenderer",
    "ReviewTabWidget",
    "register_review_renderer",
    "waiting_on",
]

_KIND_LABELS = {MILLING_SETUP: "Milling positions"}

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
_ROW_NAME_STYLE = f"color: {GRAY_TEXT_COLOR}; font-size: 13px; font-weight: 600; background: transparent;"
_ROW_TASK_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 11px; background: transparent;"
)
_ROW_RIGHT_STYLE = (
    f"color: {GRAY_SECONDARY_COLOR}; font-size: 11px; background: transparent;"
)
_ROW_RIGHT_STRONG = (
    f"color: {GRAY_TEXT_COLOR}; font-size: 11px; background: transparent;"
)


def author_label(author: str, experiment: Optional[Experiment]) -> str:
    """How a decision's author reads on screen: "you" for the operator this
    experiment names, the name for another person, "agent · model" for an
    agent, and the raw string when it fits none of those."""
    if experiment is not None and author == experiment.author():
        return "you"
    if author.startswith("human:"):
        return author[len("human:") :] or "someone"
    if author.startswith("agent:"):
        return "agent · " + (author[len("agent:") :] or "unknown")
    return author or "unknown"


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

    def set_position(self, text: str) -> None:
        """Where this sits in the list ("3 of 12"); shown, never acted on."""


def decided_proposals(experiment: Experiment) -> List[tuple]:
    """Every decided proposal as (item, task_name, proposal, superseded),
    newest decision first: the other half of the inbox, derived the same way.
    ``superseded`` marks one a re-run replaced."""
    decided = []
    for item in list(experiment.positions) + list(experiment.grids):
        for task_name, proposal in item.proposals.items():
            if not proposal.pending:
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


@register_review_renderer(MILLING_SETUP)
class MillingSetupReviewRenderer(ReviewRenderer):
    """One reference image, one draggable marker: the point of interest as the
    task proposed it, pre-placed. Same overlay and same drag as the inline
    question; what changes is when it happens."""

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

        self._controller = MicroscopeViewController(view=LamellaEditorView())
        self._controller.widget.show_beams()

        self.title = QLabel()
        self.title.setStyleSheet(_TITLE_STYLE)
        self.task_chip = QLabel()
        self.task_chip.setStyleSheet(_CHIP_STYLE)
        self.position = QLabel()
        self.position.setStyleSheet(_MUTED_STYLE)
        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addWidget(self.task_chip)
        head.addStretch(1)
        head.addWidget(self.position)

        # proposer · confidence · proposed · image, as one strip of cells
        self.cells = QFrame()
        self.cells.setObjectName("review_cells")
        # by object name: a bare QFrame selector would restyle the QLabels too
        self.cells.setStyleSheet(
            f"#review_cells {{ background: {PANEL_COLOR}; border-radius: 3px; }}"
        )
        grid = QGridLayout(self.cells)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(1)
        self._cell_values: Dict[str, QLabel] = {}
        for col, key in enumerate(("proposer", "confidence", "proposed", "image")):
            k = QLabel(key.upper())
            k.setStyleSheet(_CELL_KEY_STYLE)
            v = QLabel("—")
            v.setStyleSheet(_CELL_VALUE_STYLE)
            grid.addWidget(k, 0, col)
            grid.addWidget(v, 1, col)
            self._cell_values[key] = v
        grid.setColumnStretch(4, 1)
        # kept for callers that read the readout as text
        self.readout = QLabel()
        self.readout.hide()

        self.decision = QLabel()
        self.decision.setWordWrap(True)
        self.decision.hide()
        self.waiting = QLabel()
        self.waiting.setStyleSheet(_MUTED_STYLE)
        self.waiting.setWordWrap(True)

        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setStyleSheet(stylesheets.CONFIRM_BUTTON_STYLESHEET)
        self.btn_confirm.setToolTip("Enter — this is the answer; the delta is computed")
        self.btn_reject = QPushButton("Reject · mark lamella failed")
        self.btn_reject.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_reject.setToolTip("R — nothing further here; retires the lamella")
        self.status = QLabel()
        self.status.setStyleSheet(_MUTED_STYLE)
        actions = QHBoxLayout()
        actions.addWidget(self.btn_confirm)
        actions.addWidget(self.btn_reject)
        actions.addStretch(1)
        actions.addWidget(self.status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addWidget(self._controller.widget, 1)
        layout.addWidget(self.cells)
        layout.addWidget(self.decision)
        layout.addWidget(self.waiting)
        layout.addLayout(actions)

        self.btn_confirm.clicked.connect(self.confirm_requested)
        self.btn_reject.clicked.connect(self.reject_requested)
        self._running = False
        self._decided: Optional[Decision] = None
        self._refresh_status()

    # -- ReviewRenderer ------------------------------------------------------

    def set_proposal(
        self, experiment: Experiment, item: Any, task_name: str, proposal: Proposal
    ) -> None:
        from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

        self._experiment = experiment
        self._item = item
        self._task_name = task_name
        self._proposal = proposal
        self._image = _load_reference_image(item, proposal)

        self.title.setText(getattr(item, "name", ""))
        self.task_chip.setText(task_name)
        poi = proposal.values.get("poi")
        self._cell_values["proposer"].setText(
            str(proposal.provenance.get("proposer", "?"))
        )
        self._cell_values["confidence"].setText(
            "—" if proposal.confidence is None else f"{proposal.confidence:.2f}"
        )
        self._cell_values["proposed"].setText(
            f"{poi.x * 1e6:+.2f}, {poi.y * 1e6:+.2f} µm"
            if isinstance(poi, Point)
            else "—"
        )
        image_name = os.path.basename(
            str(proposal.provenance.get("reference_image", ""))
        )
        which = "final" if "_final_" in image_name else image_name or "—"
        self._cell_values["image"].setText(
            f"{which} · {clock(proposal.created_at)}".strip(" ·")
        )
        self.readout.setText(
            "\n".join(f"{k} {v.text()}" for k, v in self._cell_values.items())
        )
        self._gated = waiting_on(experiment, task_name)
        self._decided = None
        self._refresh_waiting()
        self._controller.remove_overlay(BeamType.ION, "confirmed")

        if self._image is None:
            self.status.setText(
                "Reference image not found — confirm uses the proposed point."
            )
            return
        self._controller.set_image(BeamType.ION, self._image)
        if isinstance(poi, Point):
            px = conversions.microscope_image_to_image_coordinates(
                poi, self._image.data.shape, self._image.metadata.pixel_size.x
            )
            col, row = px.x, px.y
        else:
            row = self._image.data.shape[0] / 2
            col = self._image.data.shape[1] / 2
        self._controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id="poi",
                points=[(col, row)],
                color="magenta",
                selected_color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                legend_label="Point of Interest",
                add_on_right_click=False,
                removable=False,
            ),
        )
        self._controller.arm_overlay(
            BeamType.ION, "poi", label="POI", icon="mdi:map-marker"
        )
        self._refresh_status()

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

    def set_running(self, running: bool) -> None:
        self._running = running
        self._refresh_status()

    def set_position(self, text: str) -> None:
        self.position.setText(text)

    def set_read_only(self, decided: Optional[Decision]) -> None:
        from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

        self._decided = decided
        self.btn_confirm.setEnabled(decided is None)
        self.btn_reject.setEnabled(decided is None)
        if decided is not None and self._image is not None:
            # the proposed marker stays where it was; the confirmed one is drawn
            # beside it so the delta is visible, and neither is draggable
            confirmed = decided.values.get("poi")
            if isinstance(confirmed, Point):
                px = conversions.microscope_image_to_image_coordinates(
                    confirmed, self._image.data.shape, self._image.metadata.pixel_size.x
                )
                self._controller.set_overlay(
                    BeamType.ION,
                    PointsSpec(
                        id="confirmed",
                        points=[(px.x, px.y)],
                        color=ORANGE_COLOR,
                        selected_color=ORANGE_COLOR,
                        marker="+",
                        size=14,
                        edge_width=1.2,
                        legend_label=None,  # the decision line says it
                        add_on_right_click=False,
                        removable=False,
                    ),
                )
            self._controller.arm_overlay(BeamType.ION, None)
        self._refresh_waiting()
        self._refresh_status()

    def _refresh_waiting(self) -> None:
        gated = getattr(self, "_gated", [])
        decided = getattr(self, "_decided", None)
        if not gated:
            self.waiting.setText("Nothing is waiting on this.")
        elif decided is None:
            self.waiting.setText("Waiting on this: " + ", ".join(gated))
        elif decided.outcome is DecisionOutcome.Rejected:
            self.waiting.setText("Skipped, lamella failed: " + ", ".join(gated))
        else:
            self.waiting.setText("Unblocked: " + ", ".join(gated))

    def _refresh_status(self) -> None:
        decided = getattr(self, "_decided", None)
        if decided is not None and self._proposal is not None:
            rejected = decided.outcome is DecisionOutcome.Rejected
            colour = DEFECT_RED_COLOR if rejected else OK_COLOR
            icon = "✗" if rejected else "✓"
            self.decision.setText(
                f"{icon}  {describe_decision(self._proposal, self._experiment)}"
            )
            self.decision.setStyleSheet(
                f"color: {colour}; background: {SURFACE_COLOR}; border-left: 3px solid "
                f"{colour}; padding: 6px 10px; font-size: 12px;"
            )
            self.decision.show()
            self.status.setText("Read-only · re-run the task to propose again")
            return
        self.decision.hide()
        beam = "beam is busy elsewhere" if self._running else "beam is idle"
        self.status.setText(f"Drag the marker to correct it · no deadline · {beam}")


def _load_reference_image(item: Any, proposal: Proposal) -> Optional[FibsemImage]:
    """The image the proposal's values sit on, from its provenance. A delta only
    means something against the same image, so nothing else is shown."""
    path = proposal.provenance.get("reference_image")
    if not path:
        return None
    item_dir = str(getattr(item, "path", ""))
    if not os.path.isabs(path):
        path = os.path.join(item_dir, path)
    if not os.path.exists(path):
        # An early proposal recorded the experiment folder rather than the
        # lamella's; the file is the lamella's by name.
        fallback = os.path.join(item_dir, os.path.basename(path))
        if os.path.exists(fallback):
            path = fallback
        else:
            logging.warning(f"Reference image for review not found: {path}")
            return None
    try:
        return FibsemImage.load(path)
    except Exception:
        logging.exception(f"Could not load the reference image for review: {path}")
        return None


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
        actions.addWidget(self.btn_confirm)
        actions.addWidget(self.btn_reject)
        actions.addStretch(1)
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


# ---------------------------------------------------------------------------
# Inbox rows
# ---------------------------------------------------------------------------


class _InboxRow(QWidget):
    """icon | name over task | right-aligned two-line detail."""

    def __init__(
        self,
        icon: str,
        colour: str,
        name: str,
        task: str,
        top_right: str,
        bottom_right: str,
        dim: bool = False,
    ) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(9)
        ic = QLabel()
        ic.setPixmap(fibsem_icon(icon, color=colour).pixmap(16, 16))
        ic.setFixedWidth(18)
        layout.addWidget(ic, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(0)
        nm = QLabel(name)
        nm.setStyleSheet(_ROW_TASK_STYLE if dim else _ROW_NAME_STYLE)
        tk = QLabel(task)
        tk.setStyleSheet(_ROW_TASK_STYLE)
        text.addWidget(nm)
        text.addWidget(tk)
        layout.addLayout(text, 1)
        right = QVBoxLayout()
        right.setSpacing(0)
        tr = QLabel(top_right)
        tr.setStyleSheet(_ROW_RIGHT_STYLE)
        tr.setAlignment(Qt.AlignRight)
        br = QLabel(bottom_right)
        br.setStyleSheet(_ROW_RIGHT_STRONG)
        br.setAlignment(Qt.AlignRight)
        right.addWidget(tr)
        right.addWidget(br)
        layout.addLayout(right)


def _group_header(text: str) -> QListWidgetItem:
    header = QListWidgetItem(text)
    header.setFlags(Qt.NoItemFlags)
    header.setData(Qt.UserRole, None)
    return header


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


class ReviewTabWidget(QWidget):
    """Host: the inbox on the left, the current proposal's renderer on the right."""

    decided = pyqtSignal(str, str)  # item_id, task_name -- after it was applied
    pending_changed = pyqtSignal(int)  # how many are waiting, for the tab badge

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        # (item, task_name, proposal, decided) per list row; decided rows are
        # read-only and do not count as pending
        self._entries: List[tuple] = []
        self._renderers: Dict[str, ReviewRenderer] = {}
        self._running = False

        self.list = QListWidget()
        self.list.setMinimumWidth(300)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.show_decided = QCheckBox("Show decided")
        self.show_decided.setToolTip(
            "List proposals that have been confirmed or rejected, including "
            "ones a re-run superseded, read-only."
        )
        self.show_decided.toggled.connect(lambda _on: self.refresh())
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self.show_decided)
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

    def refresh(self) -> None:
        """Re-derive the inbox from the experiment. Keeps the selection on the
        same (item, task) when it is still pending."""
        current = self._current_key()
        self._entries = []
        self.list.blockSignals(True)
        self.list.clear()
        if self._experiment is not None:
            experiment = self._experiment
            waiting = experiment.pending_proposals()
            if waiting:
                self.list.addItem(_group_header(f"Waiting · {len(waiting)}"))
            for item, task_name, proposal in waiting:
                blocks = len(waiting_on(experiment, task_name))
                self._add_row(
                    summary=f"{item.name} · {task_name} · waiting",
                    widget=_InboxRow(
                        "mdi:circle-medium",
                        PRIMARY_COLOR,
                        item.name,
                        task_name,
                        f"waiting {age(proposal.created_at)}",
                        f"blocks {blocks} task{'s' if blocks != 1 else ''}"
                        if blocks
                        else "blocks nothing",
                    ),
                    entry=(item, task_name, proposal, False),
                )
            pending = len(self._entries)
            if self.show_decided.isChecked():
                decided = decided_proposals(experiment)
                if decided:
                    self.list.addItem(_group_header(f"Decided · {len(decided)}"))
                for item, task_name, proposal, superseded in decided:
                    d = proposal.current
                    rejected = d.outcome is DecisionOutcome.Rejected
                    who = author_label(d.author, experiment)
                    if superseded:
                        icon, colour = "mdi:history", GRAY_SECONDARY_COLOR
                    elif rejected:
                        icon, colour = "mdi:close-circle-outline", DEFECT_RED_COLOR
                    else:
                        icon, colour = "mdi:check-circle-outline", OK_COLOR
                    outcome = (
                        f"rejected · {d.reason}"
                        if rejected
                        else delta_label(proposal, d)
                    )
                    self._add_row(
                        summary=f"{item.name} · {task_name} · "
                        + ("rejected" if rejected else "confirmed")
                        + (" · superseded" if superseded else ""),
                        widget=_InboxRow(
                            icon,
                            colour,
                            item.name,
                            task_name + (" · superseded" if superseded else ""),
                            f"{clock(d.timestamp)} · {who}",
                            outcome,
                            dim=superseded,
                        ),
                        entry=(item, task_name, proposal, True),
                        tooltip=describe_decision(proposal, experiment),
                    )
        else:
            pending = 0
        self.list.blockSignals(False)
        self._pending = pending
        self.pending_changed.emit(pending)

        select = 0
        if current is not None:
            for i, (item, task_name, _p, _d) in enumerate(self._entries):
                if (item.id, task_name) == current:
                    select = i
                    break
        if self._entries:
            self._select_entry(select)
        else:
            self.stack.setCurrentWidget(self.empty)

    def _add_row(
        self, summary: str, widget: QWidget, entry: tuple, tooltip: str = ""
    ) -> None:
        # The widget draws the row; the summary rides on a data role rather
        # than as the item's text, which would paint through it.
        row = QListWidgetItem()
        row.setData(Qt.UserRole, len(self._entries))
        row.setData(Qt.UserRole + 1, summary)
        row.setSizeHint(widget.sizeHint())
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
        item, task_name, _p, _d = self._entries[index]
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
        item, task_name, proposal, decided = self._entries[index]
        renderer = self._renderer_for(proposal.kind)
        renderer.set_proposal(self._experiment, item, task_name, proposal)
        renderer.set_running(self._running)
        renderer.set_read_only(proposal.current if decided else None)
        same = [i for i, e in enumerate(self._entries) if e[3] == decided]
        nth = same.index(index) + 1 if index in same else 0
        renderer.set_position(
            f"decided {nth} of {len(same)}" if decided else f"{nth} of {len(same)}"
        )
        self.stack.setCurrentWidget(renderer)

    def _renderer_for(self, kind: str) -> ReviewRenderer:
        renderer = self._renderers.get(kind)
        if renderer is None:
            cls = REVIEW_RENDERERS.get(kind, _UnknownKindRenderer)
            renderer = cls()
            renderer.confirm_requested.connect(self.confirm_current)
            renderer.reject_requested.connect(self.reject_current)
            self.stack.addWidget(renderer)
            self._renderers[kind] = renderer
        return renderer

    # -- the two verbs -------------------------------------------------------

    def confirm_current(self) -> None:
        index = self._current_index()
        if index is None or self._experiment is None:
            return
        item, task_name, proposal, decided = self._entries[index]
        if decided:
            return
        renderer = self._renderer_for(proposal.kind)
        decision = Decision(
            outcome=DecisionOutcome.Confirmed,
            author=self._experiment.author(),
            values=renderer.current_values(),
        )
        self._apply(item, task_name, decision)

    def reject_current(self) -> None:
        index = self._current_index()
        if index is None or self._experiment is None:
            return
        item, task_name, proposal, decided = self._entries[index]
        if decided:
            return
        retires = "This retires the lamella." if proposal.gating else ""
        reason, ok = QInputDialog.getText(
            self,
            "Reject",
            f"Why is there nothing further here for {item.name}? {retires}".strip(),
        )
        reason = reason.strip()
        if not ok or not reason:
            return
        decision = Decision(
            outcome=DecisionOutcome.Rejected,
            author=self._experiment.author(),
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
