"""The Lamella tab: one card per lamella on the left; the selected lamella's
protocol and history on the right.

The window tells it the experiment's lamellae and the run's progress; a card
chosen here is said with `lamella_selected`, and the window keeps the other
lamella lists in step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardContainer
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    LamellaTaskImageWidget,
)
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession


class LamellaTab(QWidget):
    # A card chosen by the operator, or None.
    lamella_selected = pyqtSignal(object)

    def __init__(
        self, session: "AppSession", *, card_mode: str, parent: Optional[QWidget] = None
    ) -> None:
        """*session* is what the protocol editor reads its microscope and
        experiment from; *card_mode* is the cards' density from the preferences."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.setChildrenCollapsible(True)

        # ── Left: 1-column card strip ──────────────────────────────────────
        self.cards = LamellaCardContainer(columns=1, mode=card_mode)
        self.cards.lamella_selected.connect(self._on_card_selected)

        card_scroll = QScrollArea()
        card_scroll.setWidget(self.cards)
        card_scroll.setWidgetResizable(True)
        card_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        card_scroll.setMaximumWidth(340)

        outer_splitter.addWidget(card_scroll)
        outer_splitter.setStretchFactor(0, 0)

        # ── Right: sub-tab widget ──────────────────────────────────────────
        right_tabs = QTabWidget()

        # History tab: what was done to this lamella, task by task, with each
        # task's images.
        self.history = LamellaTaskImageWidget()

        # Protocol tab: matplotlib canvas (left) + editor (right). The editor owns its
        # own controller, built eagerly so the splitter can embed the canvas before a
        # microscope connects.
        self.editor = AutoLamellaProtocolEditorWidget(session=session)
        self.editor.setMinimumWidth(550)

        protocol_splitter = QSplitter(Qt.Horizontal)
        protocol_splitter.setChildrenCollapsible(False)
        protocol_splitter.addWidget(self.editor.view_controller.widget)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.editor)
        scroll_area.setWidgetResizable(True)
        protocol_splitter.addWidget(scroll_area)
        protocol_splitter.setSizes([700, 550])

        # With no card selected the editor has no lamella, so the page says so
        # rather than leaving the last one up to be edited (FIB-1370).
        protocol_empty = QLabel("Select a lamella card to edit its protocol.")
        protocol_empty.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 12px;")
        protocol_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._protocol_page = QStackedWidget()
        self._protocol_page.addWidget(protocol_empty)
        self._protocol_page.addWidget(protocol_splitter)

        right_tabs.addTab(self._protocol_page, "Protocol")
        right_tabs.addTab(self.history, "History")

        outer_splitter.addWidget(right_tabs)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([340, 99999])

        layout.addWidget(outer_splitter)

        self.selected: Optional[Lamella] = None
        self.show_lamella(None)

    def _on_card_selected(self, lamella: Optional[Lamella]) -> None:
        self.show_lamella(lamella)
        self.lamella_selected.emit(lamella)

    def show_lamella(self, lamella: Optional[Lamella]) -> None:
        """The right side shows *lamella*'s protocol and history, or says to
        choose a card."""
        self.selected = lamella
        self.history.set_lamella(lamella)
        self.editor.select_lamella(lamella.name if lamella else None)
        self._protocol_page.setCurrentIndex(0 if lamella is None else 1)

    def select(self, lamella: Lamella) -> None:
        """Select *lamella*'s card, as a choice made elsewhere."""
        self.cards.select_lamella(lamella.name)

    def set_lamellae(self, lamellae) -> Optional[Lamella]:
        """One card per lamella. The selected card stays selected if its lamella
        is still there (FIB-966); returns it, or None."""
        selected = self.selected
        self.cards.clear()
        for lamella in lamellae:
            self.cards.add_lamella(lamella)
        still_there = next(
            (
                lamella
                for lamella in lamellae
                if selected is not None and lamella.id == selected.id
            ),
            None,
        )
        if still_there is not None:
            self.cards.select_lamella(still_there.name)
        self.show_lamella(still_there)
        return still_there

    def task_reported(self, lamella: Optional[Lamella], status: TaskStatus) -> None:
        """A run's report on *lamella* (None: on no one lamella): its card is
        redrawn, and so is the history it shows when the task has ended."""
        if lamella is not None:
            self.cards.refresh_lamella(lamella)
        else:
            self.cards.refresh_all()
        selected = self.selected
        self.show_lamella(selected)
        # Re-selecting does not redraw the History panel: it skips the lamella it
        # already shows. A task that has finished on that lamella has added to its
        # history, and this report comes after the task has returned -- its images
        # written and its history entry appended -- so rebuild it here (FIB-1111).
        if (
            selected is not None
            and lamella is not None
            and selected.id == lamella.id
            and status
            in (TaskStatus.Completed, TaskStatus.Failed, TaskStatus.Cancelled)
        ):
            self.history.refresh()
