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
from typing import Any, Callable, Dict, List, Optional, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
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
    GRAY_ICON_COLOR,
    GRAY_SECONDARY_COLOR,
    GRAY_TEXT_COLOR,
    PANEL_COLOR,
    PRIMARY_COLOR,
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
        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addWidget(self.task_chip)
        head.addStretch(1)

        self.readout = QLabel()
        self.readout.setStyleSheet(_READOUT_STYLE)
        self.readout.setWordWrap(True)
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
        layout.addWidget(self.readout)
        layout.addWidget(self.waiting)
        layout.addLayout(actions)

        self.btn_confirm.clicked.connect(self.confirm_requested)
        self.btn_reject.clicked.connect(self.reject_requested)
        self._running = False
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
        lines = [
            f"proposer   {proposal.provenance.get('proposer', '?')}",
            "confidence "
            + ("—" if proposal.confidence is None else f"{proposal.confidence:.2f}"),
        ]
        if isinstance(poi, Point):
            lines.append(f"proposed   ({poi.x * 1e6:+.2f}, {poi.y * 1e6:+.2f}) µm")
        self.readout.setText("\n".join(lines))
        gated = waiting_on(experiment, task_name)
        self.waiting.setText(
            "Waiting on this: " + ", ".join(gated)
            if gated
            else "Nothing is waiting on this."
        )

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

    def _refresh_status(self) -> None:
        beam = "beam is busy elsewhere" if self._running else "beam is idle"
        self.status.setText(f"Drag the marker to correct it · no deadline · {beam}")


def _load_reference_image(item: Any, proposal: Proposal) -> Optional[FibsemImage]:
    """The image the proposal's values sit on, from its provenance. A delta only
    means something against the same image, so nothing else is shown."""
    path = proposal.provenance.get("reference_image")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(str(getattr(item, "path", "")), path)
    if not os.path.exists(path):
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
        self._entries: List[tuple] = []  # (item, task_name, proposal) per list row
        self._renderers: Dict[str, ReviewRenderer] = {}
        self._running = False

        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.setMaximumWidth(340)
        self.list.currentRowChanged.connect(self._on_row_changed)

        self.empty = QLabel("Nothing is waiting for a decision.")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet(_MUTED_STYLE)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.list)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
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
        return len(self._entries)

    def refresh(self) -> None:
        """Re-derive the inbox from the experiment. Keeps the selection on the
        same (item, task) when it is still pending."""
        current = self._current_key()
        self._entries = []
        self.list.blockSignals(True)
        self.list.clear()
        if self._experiment is not None:
            by_kind: Dict[str, List[tuple]] = {}
            for item, task_name, proposal in self._experiment.pending_proposals():
                by_kind.setdefault(proposal.kind, []).append(
                    (item, task_name, proposal)
                )
            for kind, entries in by_kind.items():
                header = QListWidgetItem(
                    f"{_KIND_LABELS.get(kind, kind).upper()}  {len(entries)}"
                )
                header.setFlags(Qt.NoItemFlags)
                header.setData(Qt.UserRole, None)
                self.list.addItem(header)
                for entry in entries:
                    item, task_name, _p = entry
                    row = QListWidgetItem(f"{item.name}   {task_name}")
                    row.setData(Qt.UserRole, len(self._entries))
                    row.setIcon(fibsem_icon("mdi:circle-medium", color=PRIMARY_COLOR))
                    self.list.addItem(row)
                    self._entries.append(entry)
        self.list.blockSignals(False)
        self.pending_changed.emit(len(self._entries))

        select = 0
        if current is not None:
            for i, (item, task_name, _p) in enumerate(self._entries):
                if (item.id, task_name) == current:
                    select = i
                    break
        if self._entries:
            self._select_entry(select)
        else:
            self.stack.setCurrentWidget(self.empty)

    # -- selection -----------------------------------------------------------

    def _current_key(self):
        index = self._current_index()
        if index is None:
            return None
        item, task_name, _p = self._entries[index]
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
        item, task_name, proposal = self._entries[index]
        renderer = self._renderer_for(proposal.kind)
        renderer.set_proposal(self._experiment, item, task_name, proposal)
        renderer.set_running(self._running)
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
        item, task_name, proposal = self._entries[index]
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
        item, task_name, proposal = self._entries[index]
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
