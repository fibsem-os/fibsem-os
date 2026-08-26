"""Dialog for exporting the handoff map.

Everything about *what the document says* lives in `tools.handoff_map`, which has no Qt
in it, because the artifact is most wanted at the end of a run nobody stayed for -- so it
has to be reachable from a workflow hook and not only from a dialog someone remembered to
open. This is the half that collects what the operator knows and the record does not
(which grid, which slot, a note for whoever opens the box), and shows them what they are
about to send.

That last part is the substance. The first version exported blind: you ticked boxes and
got a PDF, and the only way to find out what was on it was to open it. A map is a
picture, and a picture is the one thing you cannot check from a list of options.

Behind the `handoff_map` feature flag while it sits beside Generate Overview Plot rather
than replacing it.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import DefectType, Experiment
from fibsem.applications.autolamella.tools.handoff_map import (
    DEFECT_LABELS,
    HandoffOptions,
    fluorescence_stacks,
    generate_handoff_map,
    lamella_row,
    latest_output,
    view_label,
)
from fibsem.ui import notification_service
from fibsem.ui import utils as ui_utils
from fibsem.ui.stylesheets import (
    CONFIRM_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    ERROR_COLOR,
    GRAY_CANVAS_COLOR,
    OK_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel

logger = logging.getLogger(__name__)

# Where the operator's answers are kept between exports. On the experiment rather than in
# preferences: which slot a grid is in is a fact about that grid, not a setting.
GRID_KEY = "grid"
SLOT_KEY = "slot"
NOTE_KEY = "handoff_note"

# The application's own defect colours, so a lamella flagged in the lamella list is the
# same colour here and on the exported page.
_CHIP_COLOURS = {
    DefectType.NONE: OK_COLOR,
    DefectType.REWORK: WARN_COLOR,
    DefectType.FAILURE: ERROR_COLOR,
}

# How long after a tick the preview re-renders. Compositing a mosaic takes long enough to
# be felt, and ticking four lamellae in a row should cost one render rather than four.
_PREVIEW_DEBOUNCE_MS = 250


class HandoffMapDialog(QDialog):
    """Collect what only the operator knows, show the page, then write it."""

    def __init__(self, experiment: Experiment, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if experiment is None:
            raise ValueError("HandoffMapDialog requires an experiment.")
        self.experiment = experiment
        self.setWindowTitle(f"Export Handoff Map - {experiment.name}")
        self.setMinimumSize(980, 620)

        self._canvas: Optional[QWidget] = None
        self._figure: Optional[Figure] = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)

        self._setup_ui()
        self._schedule_preview()

    # ── construction ─────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        columns = QHBoxLayout(self)

        options = QWidget()
        options.setMaximumWidth(340)
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.addWidget(self._grid_panel())
        options_layout.addWidget(self._overview_panel())
        options_layout.addWidget(self._lamella_panel(), stretch=1)
        options_layout.addWidget(self._pages_panel())
        options_layout.addWidget(self._images_panel())

        right = QVBoxLayout()
        right.addWidget(self._preview_panel(), stretch=1)
        right.addWidget(self._buttons())

        columns.addWidget(options)
        columns.addLayout(right, stretch=1)

    def _grid_panel(self) -> TitledPanel:
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(0, 0, 0, 0)

        self.edit_title = QLineEdit(self.experiment.name)
        self.edit_grid = QLineEdit(self.experiment.metadata.get(GRID_KEY, ""))
        self.edit_grid.setPlaceholderText("e.g. A")
        self.edit_slot = QLineEdit(self.experiment.metadata.get(SLOT_KEY, ""))
        self.edit_slot.setPlaceholderText("e.g. 3")
        self.edit_note = QLineEdit(self.experiment.metadata.get(NOTE_KEY, ""))
        self.edit_note.setPlaceholderText("Anything the recipient should know")

        # Grid and slot share a row: they are one answer -- where is this sample -- and
        # two full-width fields made the panel taller than the question deserves.
        where = QWidget()
        where_layout = QHBoxLayout(where)
        where_layout.setContentsMargins(0, 0, 0, 0)
        where_layout.addWidget(self.edit_grid)
        where_layout.addWidget(self.edit_slot)

        form.addRow("Title", self.edit_title)
        form.addRow("Grid / slot", where)
        form.addRow("Note", self.edit_note)

        for edit in (self.edit_title, self.edit_grid, self.edit_slot, self.edit_note):
            edit.editingFinished.connect(self._schedule_preview)

        return TitledPanel("This grid", content=content, collapsible=False)

    def _overview_panel(self) -> TitledPanel:
        """Every overview, ticked. Ticking is how a view gets a page."""
        self.overview_list = QListWidget()
        self.overview_list.setMaximumHeight(88)
        self.overview_list.itemChanged.connect(self._schedule_preview)

        for path in self.experiment.find_overview_images():
            item = QListWidgetItem(os.path.basename(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.overview_list.addItem(item)

        return TitledPanel("Overviews", content=self.overview_list, collapsible=False)

    def _lamella_panel(self) -> TitledPanel:
        self.lamella_list = QListWidget()
        self.lamella_list.itemChanged.connect(self._schedule_preview)

        for lamella in self.experiment.positions:
            row = lamella_row(lamella)
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Everything ticked, flagged ones included: a map that silently omits the
            # failures tells the recipient the grid is better than it is, and they will
            # find the empty positions anyway.
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, lamella.name)
            self.lamella_list.addItem(item)
            widget = _LamellaRow(lamella, row)
            self.lamella_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

        return TitledPanel("Lamellae", content=self.lamella_list, collapsible=False)

    def _pages_panel(self) -> TitledPanel:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chk_map = QCheckBox("Map - one page per view")
        self.chk_table = QCheckBox("Lamella table")
        self.chk_cards = QCheckBox("Lamella detail cards")
        for box in (self.chk_map, self.chk_table, self.chk_cards):
            box.setChecked(True)
            box.stateChanged.connect(self._schedule_preview)
            layout.addWidget(box)
        return TitledPanel("Pages", content=content, collapsible=False)

    def _images_panel(self) -> TitledPanel:
        """Which images go on a card, and how many lamellae actually have each.

        The counts are the point: ticking a modality nothing recorded produces a page of
        empty slots, and knowing that before exporting is cheaper than finding out
        afterwards.
        """
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        total = len(self.experiment.positions)
        counts = self._image_counts()
        self.chk_ion = QCheckBox(f"Final ion beam  ({counts['ion']} of {total})")
        self.chk_electron = QCheckBox(
            f"Final electron beam  ({counts['electron']} of {total})"
        )
        self.chk_fluorescence = QCheckBox(
            f"Fluorescence, max projection  ({counts['fm']} of {total})"
        )
        for box in (self.chk_ion, self.chk_electron, self.chk_fluorescence):
            box.setChecked(True)
            layout.addWidget(box)

        return TitledPanel("Images on each card", content=content, collapsible=False)

    def _image_counts(self) -> Dict[str, int]:
        counts = {"ion": 0, "electron": 0, "fm": 0}
        for lamella in self.experiment.positions:
            if latest_output(lamella, "final_fib"):
                counts["ion"] += 1
            if latest_output(lamella, "final_sem"):
                counts["electron"] += 1
            if fluorescence_stacks(lamella):
                counts["fm"] += 1
        return counts

    def _preview_panel(self) -> TitledPanel:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas_holder, stretch=1)

        self.label_preview = QLabel("")
        self.label_preview.setWordWrap(True)
        self.label_preview.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(self.label_preview)

        return TitledPanel("Map preview", content=content, collapsible=False)

    def _buttons(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox()
        self.btn_export = QPushButton("Export PDF")
        self.btn_export.setStyleSheet(CONFIRM_BUTTON_STYLESHEET)
        self.btn_export.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        buttons.addButton(self.btn_export, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self.btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cancel.clicked.connect(self.reject)
        return buttons

    # ── preview ──────────────────────────────────────────────────────────

    def _schedule_preview(self, *_) -> None:
        self._preview_timer.start(_PREVIEW_DEBOUNCE_MS)

    def _render_preview(self) -> None:
        """Draw the first map page, as the document would.

        Through the same `plot_overview_composite` the PDF uses, so this *is* the page
        rather than a picture of roughly the same thing. If the two ever diverge the
        preview stops being worth having.
        """
        try:
            figure, caption = self._build_preview(self.options())
        except Exception as e:
            logger.warning(f"Could not render the handoff preview: {e}")
            figure, caption = None, f"Preview unavailable: {e}"

        self._replace_canvas(figure)
        self.label_preview.setText(caption)

    def _build_preview(self, options: HandoffOptions):
        from fibsem.applications.autolamella.tools.handoff_map import (
            count_marked,
            view_key,
        )
        from fibsem.imaging.tiled import (
            DEFECT_FAILURE_COLOUR,
            DEFECT_REWORK_COLOUR,
            plot_overview_composite,
        )
        from fibsem.structures import FibsemImage

        if not options.include_map:
            return None, "Map pages are turned off."
        paths = options.selected_overviews(self.experiment)
        if not paths:
            return None, "No overview is selected, so the document has no map."

        lamellae = options.selected(self.experiment)
        selected = {lam.name for lam in lamellae}
        positions = [
            p for p in self.experiment.get_milling_positions() if p.name in selected
        ]
        colours = {}
        for lam in lamellae:
            if lam.defect.state is DefectType.FAILURE:
                colours[lam.name] = DEFECT_FAILURE_COLOUR
            elif lam.defect.state is DefectType.REWORK:
                colours[lam.name] = DEFECT_REWORK_COLOUR

        # Only the first view is previewed. The rest are pages of the same kind, and
        # rendering every one on every keystroke would cost seconds for a picture nobody
        # is looking at yet; the caption says how many there are.
        groups: Dict[tuple, list] = {}
        for path in paths:
            image = FibsemImage.load(path)
            groups.setdefault(view_key(image), []).append(image)

        key = next(iter(groups))
        images = groups[key]
        figure = plot_overview_composite(
            images,
            positions,
            color=options.marker_color,
            colors=colours,
            descriptions={lam.name: lam.description for lam in lamellae},
            show_names=True,
            show_descriptions=options.show_descriptions,
            show_scalebar=True,
            figsize=None,
        )
        # The canvas background, not the page's white. This is a map on screen, in a
        # window full of canvases that all use it -- `FibsemRealSpaceCanvas` sets both
        # figure and axes to exactly this. The *document* is white; the preview is not
        # pretending to be paper, it is showing you the map.
        figure.patch.set_facecolor(GRAY_CANVAS_COLOR)
        for axes in figure.axes:
            axes.set_facecolor(GRAY_CANVAS_COLOR)

        marked = count_marked(images, positions)
        caption = (
            f"{view_label(key)}  -  {len(images)} overview(s) composited  -  "
            f"{marked} of {len(positions)} selected lamellae fall here"
        )
        if len(groups) > 1:
            caption += f"  -  page 1 of {len(groups)} map pages"
        return figure, caption

    def _replace_canvas(self, figure: Optional[Figure]) -> None:
        if self._canvas is not None:
            self._canvas_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if self._figure is not None and self._figure is not figure:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
        self._figure = figure

        if figure is None:
            placeholder = QLabel("Nothing to preview")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                f"color: {TEXT_MUTED_COLOR}; background: {SURFACE_COLOR};"
            )
            self._canvas = placeholder
            self._canvas_layout.addWidget(placeholder)
            return

        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(240)
        self._canvas = canvas
        self._canvas_layout.addWidget(canvas)
        canvas.draw_idle()

    # ── state ────────────────────────────────────────────────────────────

    def selected_names(self) -> List[str]:
        return _checked(self.lamella_list)

    def selected_overview_paths(self) -> List[str]:
        return _checked(self.overview_list)

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
            overview_paths=self.selected_overview_paths(),
            include_ion_image=self.chk_ion.isChecked(),
            include_electron_image=self.chk_electron.isChecked(),
            include_fluorescence_image=self.chk_fluorescence.isChecked(),
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


class _LamellaRow(QWidget):
    """A lamella in the list: name, defect chip, and what is known about it.

    A chip rather than "[failed]" in the item's text, and no alternating row tint --
    which was making a flagged lamella read as a *disabled* one, the opposite of the
    attention it should draw.
    """

    def __init__(self, lamella, row: Dict[str, str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 3, 4, 4)
        layout.setSpacing(1)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        name = QLabel(lamella.name)
        name.setStyleSheet("font-size: 12px;")
        top.addWidget(name)
        top.addStretch()

        state = lamella.defect.state
        if state is not DefectType.NONE:
            chip = QLabel(DEFECT_LABELS.get(state, "").upper())
            colour = _CHIP_COLOURS.get(state, TEXT_MUTED_COLOR)
            chip.setStyleSheet(
                f"color: {colour}; border: 1px solid {colour}; border-radius: 7px;"
                " padding: 0px 5px; font-size: 9px;"
            )
            top.addWidget(chip)
        layout.addLayout(top)

        thickness = row["Thickness"]
        detail = " - ".join(
            part
            for part in (
                thickness if thickness not in ("-", "not milled") else "",
                lamella.description or lamella.defect.description or "",
            )
            if part
        )
        if detail:
            sub = QLabel(detail)
            sub.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 10px;")
            layout.addWidget(sub)


def _checked(widget: QListWidget) -> List[str]:
    out = []
    for i in range(widget.count()):
        item = widget.item(i)
        if item.checkState() == Qt.CheckState.Checked:
            out.append(item.data(Qt.ItemDataRole.UserRole))
    return out


def create_handoff_map_dialog(
    experiment: Experiment, parent: Optional[QWidget] = None
) -> HandoffMapDialog:
    """Build the dialog for *experiment*."""
    return HandoffMapDialog(experiment=experiment, parent=parent)
