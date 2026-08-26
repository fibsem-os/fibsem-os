"""Dialog for exporting the handoff map.

Deliberately thin. Everything about *what the document says* lives in
`tools.handoff_map`, which has no Qt in it, because the artifact is most wanted at the
end of a run nobody stayed for -- so it has to be reachable from a workflow hook and not
only from a dialog someone remembered to open. This is the half that collects what the
operator knows and the record does not: which grid, which slot, and any note for whoever
opens the box.

Behind the `handoff_map` feature flag while it sits beside Generate Overview Plot rather
than replacing it.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import DefectType, Experiment
from fibsem.applications.autolamella.tools.handoff_map import (
    DEFECT_LABELS,
    HandoffOptions,
    generate_handoff_map,
    lamella_row,
)
from fibsem.ui import notification_service
from fibsem.ui import utils as ui_utils
from fibsem.ui.stylesheets import (
    CONFIRM_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import TitledPanel

logger = logging.getLogger(__name__)

# Where the operator's answers are kept between exports. On the experiment rather than in
# preferences: which slot a grid is in is a fact about that grid, not a setting.
GRID_KEY = "grid"
SLOT_KEY = "slot"
NOTE_KEY = "handoff_note"


class HandoffMapDialog(QDialog):
    """Collect what only the operator knows, then write the document."""

    def __init__(self, experiment: Experiment, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if experiment is None:
            raise ValueError("HandoffMapDialog requires an experiment.")
        self.experiment = experiment
        self.setWindowTitle(f"Export Handoff Map - {experiment.name}")
        self.setMinimumWidth(560)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── what the record cannot know ──────────────────────────────────
        about = QFormLayout()
        self.edit_title = QLineEdit(f"Handoff Map: {self.experiment.name}")
        self.edit_grid = QLineEdit(self.experiment.metadata.get(GRID_KEY, ""))
        self.edit_grid.setPlaceholderText("e.g. A")
        self.edit_slot = QLineEdit(self.experiment.metadata.get(SLOT_KEY, ""))
        self.edit_slot.setPlaceholderText("e.g. 3")
        self.edit_note = QLineEdit(self.experiment.metadata.get(NOTE_KEY, ""))
        self.edit_note.setPlaceholderText("Anything the recipient should know")

        about.addRow("Title", self.edit_title)
        about.addRow("Grid", self.edit_grid)
        about.addRow("Cassette slot", self.edit_slot)
        about.addRow("Note", self.edit_note)

        about_panel = QWidget()
        about_panel.setLayout(about)
        layout.addWidget(
            TitledPanel("This grid", content=about_panel, collapsible=False)
        )

        # ── which lamellae ───────────────────────────────────────────────
        self.lamella_list = QListWidget()
        self.lamella_list.setAlternatingRowColors(True)
        for lamella in self.experiment.positions:
            row = lamella_row(lamella)
            label = f"{lamella.name}"
            detail = DEFECT_LABELS.get(lamella.defect.state, "-")
            if detail != "-":
                label = f"{label}   [{detail}]"
            if row["Thickness"] != "-":
                label = f"{label}   {row['Thickness']}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Everything ticked, including the flagged ones: a map that silently omits
            # the failures tells the recipient a grid is better than it is, and they
            # will find the empty positions anyway.
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, lamella.name)
            if lamella.defect.state is DefectType.FAILURE:
                item.setToolTip(lamella.defect.description or "Flagged as failed")
            self.lamella_list.addItem(item)
        layout.addWidget(
            TitledPanel("Lamellae", content=self.lamella_list, collapsible=False)
        )

        # ── what goes in it ──────────────────────────────────────────────
        sections = QWidget()
        sections_layout = QVBoxLayout(sections)
        self.chk_map = QCheckBox("Map pages (one per overview)")
        self.chk_table = QCheckBox("Lamella table")
        self.chk_cards = QCheckBox("Lamella detail cards (with final images)")
        for box in (self.chk_map, self.chk_table, self.chk_cards):
            box.setChecked(True)
            sections_layout.addWidget(box)
        layout.addWidget(TitledPanel("Pages", content=sections, collapsible=False))

        overviews = len(self.experiment.find_overview_images())
        fm = len(self.experiment.find_fluorescence_overview_images())
        hint = f"{overviews} overview(s) found"
        if fm:
            # Said plainly rather than left as an absence. They cannot be marked on a
            # beam overview's axes -- different stage tilt, different instrument -- so
            # their omission is a fact about the geometry, not an oversight.
            hint += (
                f"; {fm} fluorescence overview(s) will be listed but not drawn "
                "(different view, does not register)"
            )
        self.label_hint = QLabel(hint)
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(self.label_hint)

        buttons = QDialogButtonBox()
        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet(CONFIRM_BUTTON_STYLESHEET)
        self.btn_export.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        buttons.addButton(self.btn_export, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self.btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def selected_names(self) -> List[str]:
        names = []
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.data(Qt.ItemDataRole.UserRole))
        return names

    def options(self) -> HandoffOptions:
        return HandoffOptions(
            title=self.edit_title.text().strip(),
            note=self.edit_note.text().strip(),
            grid=self.edit_grid.text().strip(),
            slot=self.edit_slot.text().strip(),
            include_map=self.chk_map.isChecked(),
            include_table=self.chk_table.isChecked(),
            include_cards=self.chk_cards.isChecked(),
            lamella_names=self.selected_names(),
        )

    def _remember_answers(self) -> None:
        """Keep grid, slot and note on the experiment so the next export starts there."""
        self.experiment.metadata[GRID_KEY] = self.edit_grid.text().strip()
        self.experiment.metadata[SLOT_KEY] = self.edit_slot.text().strip()
        self.experiment.metadata[NOTE_KEY] = self.edit_note.text().strip()
        try:
            self.experiment.save()
        except Exception as e:
            # Not fatal: the document is what was asked for, and it has already been
            # written by the time this runs.
            logger.warning(f"Could not save the grid details onto the experiment: {e}")

    def _on_export(self) -> None:
        if not self.selected_names():
            QMessageBox.warning(
                self, "Nothing to map", "Select at least one lamella to include."
            )
            return

        default = os.path.join(
            str(self.experiment.path), f"{self.experiment.name}-handoff-map.pdf"
        )
        path = ui_utils.open_save_file_dialog(
            msg="Save the handoff map",
            path=default,
            _filter="PDF Document (*.pdf)",
            parent=self,
        )
        if not path:
            return

        try:
            generate_handoff_map(self.experiment, path, self.options())
        except Exception as e:
            logger.error(f"Could not write the handoff map: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Export failed", f"Could not write the handoff map:\n{e}"
            )
            return

        self._remember_answers()
        notification_service.show_toast(f"Saved {os.path.basename(path)}", "success")
        self.accept()


def create_handoff_map_dialog(
    experiment: Experiment, parent: Optional[QWidget] = None
) -> HandoffMapDialog:
    """Build the dialog for *experiment*."""
    return HandoffMapDialog(experiment=experiment, parent=parent)
