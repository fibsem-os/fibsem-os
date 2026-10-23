from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions
from fibsem.applications.autolamella.config_edits import (
    apply_correlation,
    copy_task_configs,
    edit_item_fields,
    item_fields_version,
    poi_patch,
)
from fibsem.applications.autolamella.reference_images import (
    fm_stack_label,
    reference_image_label,
)
from fibsem.applications.autolamella.structures import (
    Lamella,
)
from fibsem.applications.autolamella.task_outputs import (
    lamella_images,
    latest_task_image,
)
from fibsem.applications.autolamella.ui.protocol.editor import (
    AutoLamellaTaskConfigEditor,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
    SpotBurnFiducialTaskConfig,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
)
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.edit_recording import PendingEdits
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    NEUTRAL_200,
    SEMANTIC_WARNING_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.canvas.canvas_state import AlignmentSpec, PointsSpec
from fibsem.ui.widgets.canvas.overlay_controls import (
    CanvasOverlayControls,
    CanvasPopover,
)
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.checklist import ApplyLamellaConfigDialog
from fibsem.ui.widgets.custom_widgets import (
    ContextMenuConfig,
    IconToolButton,
    TaskNameListWidget,
    ValueComboBox,
)
from fibsem.ui.widgets.spot_burn_coordinates_widget import (
    SpotBurnCoordinatesWidget,
)
from fibsem.utils import format_value
from fibsem.workflows.image_positions import (
    ImagePosition,
    PositionComparison,
    PositionMatch,
    compare_positions,
    latest_image_at,
    read_image_position,
)

if TYPE_CHECKING:
    from fibsem.correlation.structures import CorrelationResult
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.workflows.session import AppSession

# Reducer overlay ids on the FIB (ION) canvas
POI_OVERLAY_ID = "poi"
ALIGNMENT_OVERLAY_ID = "alignment_area"

_WARNING_LABEL_STYLE = f"color: {SEMANTIC_WARNING_COLOR}; font-size: 11px;"
_DESCRIPTION_STYLE = (
    f"QLineEdit {{ background: transparent; border: none; color: {TEXT_MUTED_COLOR};"
    " font-style: italic; padding: 2px 4px; }"
    f"QLineEdit:focus {{ background: {CANVAS_BG}; border: 1px solid {BORDER_COLOR};"
    f" border-radius: 2px; color: {NEUTRAL_200}; font-style: normal; }}"
)

# The view toggles in the FIB canvas's overlay popover, by key.
OVERLAY_SEM = "sem"
OVERLAY_RELATED = "related"
OVERLAY_ALIGNMENT = "alignment"
OVERLAY_EDIT_ALIGNMENT = "edit_alignment"


def _picker_label(relpath: str, task_names: Iterable[str]) -> str:
    """A reference image's picker label, with its run's time when it sits in a
    run's own folder."""
    return _with_run_time(
        reference_image_label(os.path.basename(relpath), task_names), relpath
    )


def _with_run_time(label: str, relpath: str) -> str:
    """``label`` with the time of the run whose folder holds ``relpath``
    (``<task>/<YYYYMMDD-HHMMSS>/...``), so two runs' images of the same
    name read apart. A file in the lamella's own folder keeps its label."""
    for part in reversed(os.path.normpath(os.path.dirname(relpath)).split(os.sep)):
        try:
            when = datetime.datetime.strptime(part, "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        return f"{label} · {when:%d %b %H:%M}"
    return label


def _fill_picker(combo: QComboBox, filenames: List[str], labels: List[str]) -> None:
    """Offer *filenames* under *labels*; the filename rides as item data and tooltip."""
    combo.clear()
    for filename, label in zip(filenames, labels):
        combo.addItem(label, filename)
        combo.setItemData(combo.count() - 1, filename, Qt.ToolTipRole)


def _select_filename(combo: QComboBox, filename: str) -> None:
    """Land the picker on *filename*, if it offers it."""
    index = combo.findData(filename)
    if index >= 0:
        combo.setCurrentIndex(index)
    combo.setToolTip(combo.currentData() or "")


class _PositionNotice(QWidget):
    """The strip under the image pickers that says where the FIB image was taken.

    Lit with the warning tint when the lamella has moved since the image was taken;
    muted for what is only worth knowing (no position recorded, or the editor switched
    images itself); hidden when the image was taken where the lamella is. The button
    switches to the newest image taken at the lamella's current position, and is shown
    only when there is one.
    """

    switch_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("positionNotice")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(8)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.switch_button = QPushButton("Show current image")
        self.switch_button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.switch_button.setToolTip(
            "Show the newest image taken at the lamella's current position."
        )
        self.switch_button.clicked.connect(self.switch_clicked)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.switch_button)
        self.clear()

    def show_warning(self, text: str, can_switch: bool) -> None:
        rgb = QColor(SEMANTIC_WARNING_COLOR)
        self.label.setText(text)
        self.label.setStyleSheet(
            f"color: {NEUTRAL_200}; background: transparent; border: none;"
        )
        tint = f"{rgb.red()}, {rgb.green()}, {rgb.blue()}"
        self.setStyleSheet(
            f"#positionNotice {{ background: rgba({tint}, 0.12);"
            f" border: 1px solid rgba({tint}, 0.5); border-radius: 2px; }}"
        )
        self.switch_button.setVisible(can_switch)
        self.setVisible(True)

    def show_note(self, text: str, can_switch: bool) -> None:
        self.label.setText(text)
        self.label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"
            " background: transparent; border: none;"
        )
        self.setStyleSheet("#positionNotice { background: transparent; border: none; }")
        self.switch_button.setVisible(can_switch)
        self.setVisible(True)

    def clear(self) -> None:
        self.label.setText("")
        self.switch_button.setVisible(False)
        self.setVisible(False)


class AutoLamellaProtocolEditorWidget(QWidget):
    """A widget to edit the AutoLamella protocol."""

    def __init__(self, session: "AppSession", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session

        # Canvas display: own a controller driving a task-driven stacked view (FIB +
        # SEM beside it + FM page). Created eagerly (no microscope needed) so the host
        # tab can embed ``view_controller.widget`` before the microscope connects.
        self.view_controller = MicroscopeViewController(view=LamellaEditorView())

        self.image: FibsemImage
        self.fm_image: Optional[FluorescenceImage] = None
        self.show_related_milling_tasks = True
        self.show_sem_image = False
        self.show_alignment_area = True
        self.alignment_area_editable = False
        self._active_lamella_name: Optional[str] = None
        self._active_task_name: Optional[str] = None
        self._selected_lamella: Optional[Lamella] = None
        self._overlay_wired = False  # subscribed to controller.overlay_edited
        # Where each of the selected lamella's FIB/SEM images was taken, by filename,
        # read from their headers on selection.
        self._image_positions: Dict[str, ImagePosition] = {}
        # The FIB image the editor switched to itself, and the lamella (by id) it did
        # so for, to say so while it is shown.
        self._switched_to: Optional[str] = None
        self._switched_for: Optional[str] = None

        # Each edit, for the experiment's record, and the write a burst
        # of them makes -- see `_save_experiment`. Built before the microscope
        # check below: the window flushes on close, and an editor that never saw
        # a microscope must still answer that rather than raise.
        self._edits = PendingEdits(
            lambda: self.session.microscope,
            via="lamella editor",
            parent=self,
        )

        if self.session.microscope is None:
            return

        self._on_microscope_connected()

    def _on_microscope_connected(self):
        """Callback when the microscope is connected."""
        if self.session.microscope is None:
            raise ValueError("No microscope connected, cannot proceed.")
        self.microscope = self.session.microscope
        if hasattr(self, "task_editor"):
            # Reconnect: built once, so the forms still hold the old microscope.
            self.task_editor.set_microscope(self.microscope)
        else:
            # First connect: build the full UI
            self._create_widgets()
            self._initialise_widgets()

    def set_experiment(self):
        """Set the experiment for the protocol editor."""
        # The parent has already swapped `.experiment` by the time this runs, so this
        # is the last moment the previous one's pending edit can be written. It is
        # written to the right file because the pending write carries its own
        # experiment rather than looking one up -- see `_save_experiment`.
        self.flush_pending_save()
        self._refresh_experiment_positions()

    def set_active_lamella_name(
        self, lamella_name: Optional[str], task_name: Optional[str] = None
    ) -> None:
        """Track which lamella/task is actively being processed and lock/unlock editing accordingly."""
        self._active_lamella_name = lamella_name
        self._active_task_name = task_name
        self._apply_editing_lock(self._is_editing_locked())
        if hasattr(self, "listWidget_selected_task"):
            self._refresh_task_states()

    def _refresh_task_states(self) -> None:
        """Chip each task row with what the lamella says about it.

        Completed from the lamella's own record, in progress from what the workflow
        told this editor it is running. A task with neither shows nothing.
        """
        lamella = self._selected_lamella
        if lamella is None:
            self.listWidget_selected_task.set_task_states({})
            return
        states = {
            name: ("Completed", stylesheets.GREEN_COLOR)
            for name in lamella.completed_tasks
        }
        if (
            self._active_lamella_name == lamella.name
            and self._active_task_name is not None
        ):
            states[self._active_task_name] = ("In Progress", stylesheets.ACCENT_COLOR)
        self.listWidget_selected_task.set_task_states(states)

    def _is_editing_locked(self) -> bool:
        if self._active_lamella_name is None:
            return False
        selected_name = self._selected_lamella.name if self._selected_lamella else ""
        selected_task = self.listWidget_selected_task.selected_task
        return selected_name == self._active_lamella_name and (
            self._active_task_name is None or selected_task == self._active_task_name
        )

    def _apply_editing_lock(self, locked: bool) -> None:
        self.task_editor.set_editable(not locked)
        self.spot_burn_coordinates_widget.setEnabled(not locked)
        if locked:
            self.label_lamella_warning.setText(
                "This lamella is currently being processed and cannot be edited."
            )
        else:
            self.label_lamella_warning.setText("")

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        self.task_editor = AutoLamellaTaskConfigEditor(
            self.session, self.microscope, self._edits, "item", parent=self
        )
        # drive patterns/reposition on the FIB canvas
        self.task_editor.milling.set_controller(self.view_controller)
        self.task_editor.milling.set_alignment_area_visible(False)

        self.spot_burn_coordinates_widget = SpotBurnCoordinatesWidget(
            controller=self.view_controller,
            beam=BeamType.ION,
            parent=self,
        )

        self.pushButton_refresh_positions = IconToolButton(
            icon="mdi:refresh",
            tooltip="Reload task configs and images for the current lamella.",
        )
        self.pushButton_refresh_positions.clicked.connect(
            self._refresh_experiment_positions
        )
        self.pushButton_apply_to_other = IconToolButton(
            icon="mdi:file-transfer",
            tooltip="Apply this lamella's task configurations to other lamella in the experiment.",
        )
        self.pushButton_apply_to_other.clicked.connect(self._on_apply_to_other_clicked)
        self.pushButton_open_correlation = IconToolButton(
            icon="mdi:target",
            tooltip="Open the 3D correlation tool to align the FIB and FM images.",
        )
        self.pushButton_open_correlation.clicked.connect(self._open_correlation_dialog)

        # View toggles live on the FIB canvas, not in the editor header: they change
        # what is drawn over the image, so they sit with the canvas tools, in the same
        # popover the overview canvases use. Listing "Edit alignment area" under
        # "Alignment area" also makes the dependency between them visible, where a
        # greyed pencil six buttons along did not.
        self.overlay_controls = CanvasOverlayControls(
            [
                (OVERLAY_SEM, "SEM image", self.show_sem_image),
                (
                    OVERLAY_RELATED,
                    "Related milling tasks",
                    self.show_related_milling_tasks,
                ),
                (OVERLAY_ALIGNMENT, "Alignment area", self.show_alignment_area),
                (
                    OVERLAY_EDIT_ALIGNMENT,
                    "Edit alignment area",
                    self.alignment_area_editable,
                ),
            ]
        )
        self.overlay_controls.set_enabled(
            OVERLAY_EDIT_ALIGNMENT, self.show_alignment_area
        )
        self.overlay_controls.toggled.connect(self._on_overlay_toggled)
        fib_canvas = self.view_controller.get_canvas(BeamType.ION)
        # The eye, as on the overview canvases' overlay control; layers is the FM
        # channel control's icon.
        self.btn_overlays = fib_canvas.add_toolbar_button(
            "mdi:eye-outline", "Overlays", self._toggle_overlays_popover, checkable=True
        )
        fib_canvas._reposition_overlay_buttons()
        self.overlay_popover = CanvasPopover(self.overlay_controls, parent=fib_canvas)

        self.label_lamella_name = QLabel("")
        self.label_lamella_name.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200}; background: transparent;"
        )

        # free-text lamella description (this editor is the source of truth for it).
        # Beside the name, because it is a note about the lamella and not a task
        # setting: borderless and muted until clicked, so it reads as a caption.
        self.line_edit_description = QLineEdit()
        self.line_edit_description.setPlaceholderText("Add a description…")
        self.line_edit_description.setToolTip("Free-text note about this lamella")
        self.line_edit_description.setStyleSheet(_DESCRIPTION_STYLE)
        self.line_edit_description.editingFinished.connect(self._on_description_edited)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.label_lamella_name)
        self.button_layout.addWidget(self.line_edit_description, 1)
        self.button_layout.addWidget(self.pushButton_refresh_positions)
        self.button_layout.addWidget(self.pushButton_apply_to_other)
        self.button_layout.addWidget(self.pushButton_open_correlation)
        self.listWidget_selected_task = TaskNameListWidget()
        self.listWidget_selected_task.set_buttons_visible(add=False, remove=False)

        # Reference image pickers. Rows are named by beam; items by task and stage,
        # with the filename in the tooltip. The FM row is only there for the
        # fluorescence task, and the SEM row is muted while the SEM image is off in
        # the canvas overlays.
        self.combobox_fm_filenames = ValueComboBox()
        self.combobox_fm_filenames_label = QLabel("FM")
        self.combobox_fm_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_fib_filenames = ValueComboBox()
        self.combobox_fib_filenames_label = QLabel("FIB")
        self.combobox_fib_filenames.currentIndexChanged.connect(self._on_image_selected)

        self.combobox_sem_filenames = ValueComboBox()
        self.combobox_sem_filenames_label = QLabel("SEM")
        self.combobox_sem_filenames.currentIndexChanged.connect(self._on_image_selected)
        self.combobox_sem_filenames.setEnabled(self.show_sem_image)

        # One warning style: the theme's warning token, small. Whether a task has
        # completed is a chip on its row in the list, not a sentence here.
        # Says when the FIB image was taken somewhere the lamella has since moved from.
        self.position_notice = _PositionNotice()
        self.position_notice.switch_clicked.connect(self._on_show_current_image)

        self.label_lamella_warning = QLabel("")
        self.label_lamella_warning.setStyleSheet(_WARNING_LABEL_STYLE)
        self.label_lamella_warning.setWordWrap(True)
        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet(_WARNING_LABEL_STYLE)
        self.label_warning.setWordWrap(True)

        self.grid_layout = QGridLayout()
        # The beam labels are short; the pickers take the width.
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.addLayout(self.button_layout, 0, 0, 1, 2)
        self.grid_layout.addWidget(self.listWidget_selected_task, 1, 0, 1, 2)
        self.grid_layout.addWidget(self.combobox_fib_filenames_label, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fib_filenames, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames_label, 3, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_sem_filenames, 3, 1, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames_label, 4, 0, 1, 1)
        self.grid_layout.addWidget(self.combobox_fm_filenames, 4, 1, 1, 1)
        self.grid_layout.addWidget(self.position_notice, 5, 0, 1, 2)
        self.grid_layout.addWidget(self.label_lamella_warning, 6, 0, 1, 2)
        self.grid_layout.addWidget(self.label_warning, 7, 0, 1, 2)

        # main layout. No scroll area here: the window wraps this editor in one
        # already, and a second one nested inside it only ever added a second bar.
        self.main_layout = QVBoxLayout(self)
        self.task_editor.add_form(self.spot_burn_coordinates_widget)
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addWidget(self.task_editor)
        self.main_layout.addWidget(self.task_editor.milling)
        self.main_layout.addStretch()

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""
        self.listWidget_selected_task.task_selected.connect(
            lambda _: self._on_selected_task_changed()
        )
        self.task_editor.edited.connect(lambda _: self._refresh_milling_fov_warning())
        self.task_editor.milling.set_right_click_menu_actions(
            self._add_poi_context_menu_action
        )
        self._correlation_open = False
        self.spot_burn_coordinates_widget.settings_changed.connect(
            self._on_spot_burn_coordinates_changed
        )
        if not self._overlay_wired:
            self.view_controller.overlay_edited.connect(
                self._on_controller_overlay_edited
            )
            self._overlay_wired = True

        # Built after the host selected a lamella: show it now.
        if self._selected_lamella is not None:
            self._on_selected_lamella_changed()

    def refresh_if_showing(self, item_name: str) -> None:
        """Rebuild the panel if it is displaying ``item_name``.

        Called (on the GUI thread) after an outside writer — the agent's
        config patch — changed the item's state under an open form. The
        rebuild is what prevents the stale form writing old values back on
        the operator's next edit; pending operator edits were flushed before
        the outside write was allowed to apply.
        """
        selected = self._selected_lamella
        if selected is not None and selected.name == item_name:
            self._on_selected_lamella_changed()

    def _refresh_experiment_positions(self):
        """Refresh the editor from the current experiment positions."""
        experiment = self.session.experiment
        if experiment is None:
            return
        # Only the lamella already shown, never a stand-in: the editor shows what the
        # host selected, and with nothing selected an edit here must have nowhere to
        # land.
        if self._selected_lamella is None:
            return
        self._selected_lamella = experiment.get_lamella_by_name(
            self._selected_lamella.name
        )
        self._on_selected_lamella_changed()

    def select_lamella(self, name: Optional[str]) -> None:
        """Select a lamella by name and refresh the editor. Called externally (e.g. card click).

        ``None`` clears the selection, so the editor holds no lamella to write to.
        """
        if name is None:
            self.flush_pending_save()
            self._selected_lamella = None
            return
        if self.session.experiment is None:
            return
        lamella = self.session.experiment.get_lamella_by_name(name)
        if lamella is None:
            return
        self._selected_lamella = lamella
        # Before the microscope connects there are no widgets yet; the selection
        # is shown when they are built.
        if hasattr(self, "label_lamella_name"):
            self._on_selected_lamella_changed()

    def _on_selected_lamella_changed(self):
        """Callback when the selected lamella changes."""
        # Whatever is pending belongs to the lamella being left behind.
        self.flush_pending_save()
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return
        self.label_lamella_name.setText(selected_lamella.name)

        # populate description without re-emitting editingFinished
        self.line_edit_description.blockSignals(True)
        self.line_edit_description.setText(selected_lamella.description)
        self.line_edit_description.blockSignals(False)

        task_names = self._sort_task_names_by_workflow(
            list(selected_lamella.task_config.keys())
        )
        self.listWidget_selected_task.set_tasks(task_names)
        self._refresh_task_states()

        # load fluorescence image: the lamella folder's, then each run's own
        # folder's, as paths relative to the lamella folder
        filenames = lamella_images(selected_lamella, ".ome.tiff")
        task_names = list(selected_lamella.task_config.keys())
        fm_basenames = filenames
        self.combobox_fm_filenames.blockSignals(True)
        _fill_picker(
            self.combobox_fm_filenames,
            fm_basenames,
            [fm_stack_label(os.path.basename(f)) for f in fm_basenames],
        )
        self.combobox_fm_filenames.setCurrentIndex(
            len(filenames) - 1
        )  # Select latest by default
        self.combobox_fm_filenames.setToolTip(
            self.combobox_fm_filenames.currentData() or ""
        )
        self.combobox_fm_filenames.blockSignals(False)

        # Each reference image's header, read once -- a fraction of a millisecond
        # each -- for where it was taken and when. The pickers list them in the
        # order they were taken, so a run's images sit together and in order; the
        # default below still falls back to the last image found, as it did.
        fib_filenames = [
            f
            for f in lamella_images(selected_lamella, "_ib.tif")
            if "alignment" not in f
        ]  # show all non-alignment images
        sem_filenames = [
            f
            for f in lamella_images(selected_lamella, "_eb.tif")
            if "alignment" not in f
        ]
        positions = {
            f: read_image_position(self._image_path(selected_lamella, f))
            for f in fib_filenames + sem_filenames
        }
        fib_listed = sorted(fib_filenames, key=lambda f: positions[f].timestamp)
        sem_listed = sorted(sem_filenames, key=lambda f: positions[f].timestamp)

        # load fib reference image
        self.combobox_fib_filenames.blockSignals(True)

        # remember selected fib filename
        selected_fib_filename = self.combobox_fib_filenames.currentData() or ""
        base_filenames = fib_filenames
        _fill_picker(
            self.combobox_fib_filenames,
            fib_listed,
            [_picker_label(f, task_names) for f in fib_listed],
        )

        latest_task_filename = self._latest_task_image(selected_lamella, "_ib.tif")

        _select_filename(
            self.combobox_fib_filenames,
            self._default_fib_filename(
                base_filenames, latest_task_filename, selected_fib_filename
            ),
        )

        # load sem reference image
        self.combobox_sem_filenames.blockSignals(True)

        selected_sem_filename = self.combobox_sem_filenames.currentData() or ""
        sem_base_filenames = sem_filenames
        _fill_picker(
            self.combobox_sem_filenames,
            sem_listed,
            [_picker_label(f, task_names) for f in sem_listed],
        )

        latest_sem_task_filename = self._latest_task_image(selected_lamella, "_eb.tif")

        default_sem_filename = sem_filenames[-1] if sem_filenames else ""
        if latest_sem_task_filename in sem_base_filenames:
            default_sem_filename = latest_sem_task_filename
        elif selected_sem_filename in sem_base_filenames:
            default_sem_filename = selected_sem_filename

        _select_filename(self.combobox_sem_filenames, default_sem_filename)

        # The default above is the last task's image, else the last found; unless it
        # was taken where the lamella is now, it gives way to the newest image that
        # was, if there is one. Only here, where the editor chose the image: a pick
        # from the picker is never replaced, only described.
        self._image_positions = positions
        switched_before = (
            self._switched_to if self._switched_for == selected_lamella.id else None
        )
        self._switched_to = None
        if self._fib_image_position().match is not PositionMatch.SAME:
            self._switched_to = self._select_current_images()
        elif self.combobox_fib_filenames.currentData() == switched_before:
            # Selected again and still on the image the editor switched to (now the
            # previous selection): keep saying why it is shown.
            self._switched_to = switched_before
        self._switched_for = selected_lamella.id
        self.combobox_fib_filenames.blockSignals(False)
        self.combobox_sem_filenames.blockSignals(False)

        # hide if no filenames (the FM row also follows the task, see
        # _on_selected_task_changed)
        self.combobox_fib_filenames.setVisible(len(fib_filenames) > 0)
        self.combobox_fib_filenames_label.setVisible(len(fib_filenames) > 0)
        self.combobox_sem_filenames.setVisible(len(sem_filenames) > 0)
        self.combobox_sem_filenames_label.setVisible(len(sem_filenames) > 0)
        self._sync_sem_picker()
        self.overlay_controls.set_enabled(OVERLAY_SEM, len(sem_filenames) > 0)

        # warnings and widget enablement
        lamella_warnings = []
        if not fib_filenames:
            lamella_warnings.append("No FIB reference images found.")
        if selected_lamella.last_completed_task is None:
            lamella_warnings.append("No tasks have been completed for this lamella.")
        self.label_lamella_warning.setText("  ".join(lamella_warnings))

        # Re-apply lock if this lamella/task is the one currently being processed
        self._apply_editing_lock(self._is_editing_locked())

        self._on_image_selected(0)
        self._draw_point_of_interest(selected_lamella.poi)

    def _toggle_overlays_popover(self) -> None:
        """Show or hide the overlays popover, anchored under its button."""
        self.overlay_popover.set_open(self.btn_overlays.isChecked(), self.btn_overlays)

    def _on_overlay_toggled(self, key: str, checked: bool) -> None:
        handler = {
            OVERLAY_SEM: self._on_toggle_sem_image,
            OVERLAY_RELATED: self._on_toggle_related_tasks,
            OVERLAY_ALIGNMENT: self._on_toggle_alignment_area,
            OVERLAY_EDIT_ALIGNMENT: self._on_toggle_alignment_area_editable,
        }.get(key)
        if handler is not None:
            handler(checked)

    def _on_toggle_sem_image(self, checked: bool):
        """Toggle displaying the sem image and enablement of sem controls."""
        self.show_sem_image = checked
        self._sync_sem_picker()
        self._on_image_selected(0)

    def _sync_sem_picker(self) -> None:
        """The SEM row is muted while the SEM image is off in the canvas overlays."""
        available = self.combobox_sem_filenames.count() > 0
        self.combobox_sem_filenames.setEnabled(self.show_sem_image and available)
        self.combobox_sem_filenames_label.setEnabled(self.show_sem_image and available)
        if available and not self.show_sem_image:
            self.combobox_sem_filenames.setToolTip(
                "Turn on 'SEM image' in the canvas overlays to show it."
            )
        else:
            self.combobox_sem_filenames.setToolTip(
                self.combobox_sem_filenames.currentData() or ""
            )

    def _on_toggle_related_tasks(self, checked: bool):
        """Toggle displaying related milling tasks."""
        self.show_related_milling_tasks = checked
        self._on_selected_task_changed()

    def _on_image_selected(self, index):
        """Callback when an image is selected — load the images and push them to the canvases."""
        p = self._selected_lamella
        if p is None:
            return
        fm_filename = self.combobox_fm_filenames.currentData() or ""
        fib_filename = self.combobox_fib_filenames.currentData() or ""
        sem_filename = self.combobox_sem_filenames.currentData() or ""
        self.combobox_fib_filenames.setToolTip(fib_filename)

        # load the fib reference image
        reference_image_path = os.path.join(p.path, fib_filename)
        if os.path.exists(reference_image_path) and os.path.isfile(
            reference_image_path
        ):
            self.image = FibsemImage.load(reference_image_path)
        else:
            self.image = FibsemImage.generate_blank_image(hfw=100e-6, random=False)

        # load the sem reference image
        sem_image = None
        if self.show_sem_image and sem_filename:
            sem_image_path = os.path.join(p.path, sem_filename)
            if os.path.exists(sem_image_path) and os.path.isfile(sem_image_path):
                sem_image = FibsemImage.load(sem_image_path)

        # load the fluorescence image
        self.fm_image = None
        fm_image_path = os.path.join(p.path, fm_filename) if fm_filename else None
        if fm_image_path and os.path.exists(fm_image_path):
            self.fm_image = FluorescenceImage.load(fm_image_path)

        # push the FIB image to the ION canvas (patterns / POI / alignment / spot-burn host)
        self.view_controller.set_image(BeamType.ION, self.image)
        # keep the spot-burn editor's 0-1 <-> pixel conversion in step with the FIB image
        self.spot_burn_coordinates_widget.set_image_shape(self.image.data.shape[:2])

        # SEM reference: shown beside FIB when toggled on and available
        self._update_sem_display(sem_image)

        self._refresh_position_notice()

        self._on_selected_task_changed()

    def _update_sem_display(self, sem_image: Optional[FibsemImage]) -> None:
        """Show/hide the SEM canvas beside FIB and feed it the reference image."""
        show = sem_image is not None
        self.view_controller.widget.set_sem_visible(show)
        if show:
            self.view_controller.set_image(BeamType.ELECTRON, sem_image)

    # --- where the shown image was taken --------------------------------

    def _lamella_position(self) -> Optional[FibsemStagePosition]:
        """Where the selected lamella is now: its milling pose.

        One pose for every image the pickers offer, whichever task is selected:
        every beam task images from the milling pose, and the fluorescence task shows
        the FM page rather than these images.
        """
        lamella = self._selected_lamella
        if lamella is None or lamella.milling_pose is None:
            return None
        return lamella.milling_pose.stage_position

    def _image_position(self, filename: str) -> ImagePosition:
        position = self._image_positions.get(filename)
        if position is None and self._selected_lamella is not None and filename:
            position = read_image_position(
                self._image_path(self._selected_lamella, filename)
            )
            self._image_positions[filename] = position
        return position or ImagePosition(filename, None, 0.0)

    def _fib_image_position(self) -> PositionComparison:
        """Where the FIB image in the picker was taken, against the lamella's pose."""
        filename = self.combobox_fib_filenames.currentData() or ""
        return compare_positions(
            self._image_position(filename).stage_position, self._lamella_position()
        )

    def _current_image(self, suffix: str) -> Optional[ImagePosition]:
        """The newest of the lamella's images ending *suffix* taken where it is now."""
        return latest_image_at(
            (p for f, p in self._image_positions.items() if f.endswith(suffix)),
            self._lamella_position(),
        )

    def _select_current_images(self) -> Optional[str]:
        """Put both pickers on images taken at the lamella's current position.

        The SEM picker takes the FIB image's partner from the same acquisition when it
        was taken there too, else the newest SEM image that was. Returns the FIB
        filename selected, or None if no FIB image was taken there. Selecting fires
        the pickers' signals unless the caller has blocked them.
        """
        fib = self._current_image("_ib.tif")
        if fib is None:
            return None
        _select_filename(self.combobox_fib_filenames, fib.filename)
        partner = self._image_positions.get(fib.filename[: -len("_ib.tif")] + "_eb.tif")
        if (
            partner is not None
            and compare_positions(
                partner.stage_position, self._lamella_position()
            ).match
            is PositionMatch.SAME
        ):
            _select_filename(self.combobox_sem_filenames, partner.filename)
        else:
            sem = self._current_image("_eb.tif")
            if sem is not None:
                _select_filename(self.combobox_sem_filenames, sem.filename)
        return fib.filename

    def _on_show_current_image(self) -> None:
        """The notice's button: show the images taken at the lamella's position."""
        self.combobox_fib_filenames.blockSignals(True)
        self.combobox_sem_filenames.blockSignals(True)
        try:
            self._select_current_images()
        finally:
            self.combobox_fib_filenames.blockSignals(False)
            self.combobox_sem_filenames.blockSignals(False)
        self._switched_to = None
        self._on_image_selected(0)

    def _refresh_position_notice(self) -> None:
        """Say whether the FIB image shown was taken where the lamella is now."""
        canvas = self.view_controller.get_canvas(BeamType.ION)
        filename = self.combobox_fib_filenames.currentData() or ""
        lamella_position = self._lamella_position()
        if (
            not filename
            or lamella_position is None
            or lamella_position.x is None
            or lamella_position.y is None
        ):
            # No image, or nowhere to compare it with.
            self.position_notice.clear()
            canvas.set_hint(None)
            return

        comparison = self._fib_image_position()
        current = self._current_image("_ib.tif")
        can_switch = current is not None and current.filename != filename
        if comparison.match is PositionMatch.MOVED:
            moved = comparison.describe()
            text = f"The lamella has moved since this image was taken: {moved}."
            if current is None:
                text += " No image has been taken at its current position."
            self.position_notice.show_warning(text, can_switch)
            canvas.set_hint(f"Lamella moved since this image: {moved}")
            return

        canvas.set_hint(None)
        if comparison.match is PositionMatch.UNKNOWN:
            if self._image_position(filename).stage_position is None:
                text = "This image has no stage position recorded."
            else:
                text = (
                    "This image's stage position is in a different coordinate "
                    "system from the lamella's."
                )
            self.position_notice.show_note(text, can_switch)
        elif filename == self._switched_to:
            label = self.combobox_fib_filenames.currentText()
            self.position_notice.show_note(
                f"Showing {label}, the image taken at the lamella's current position.",
                False,
            )
        else:
            self.position_notice.clear()

    def _sort_task_names_by_workflow(self, task_names: List[str]) -> List[str]:
        """Sort task names by experiment workflow order; keep unknown tasks in original order."""
        experiment = self.session.experiment
        if experiment is None or experiment.task_protocol is None:
            return task_names

        workflow_names = experiment.task_protocol.workflow_config.workflow
        if not workflow_names:
            return task_names

        workflow_order = {name: i for i, name in enumerate(workflow_names)}
        original_order = {name: i for i, name in enumerate(task_names)}
        default_order = len(workflow_order)

        return sorted(
            task_names,
            key=lambda name: (
                workflow_order.get(name, default_order),
                original_order[name],
            ),
        )

    def _on_selected_task_changed(self):
        """Show the selected task's config, and what goes with it on the canvas."""
        selected_stage_name = self.listWidget_selected_task.selected_task
        if not selected_stage_name:
            return
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return

        task_config = self.task_editor.show_task(selected_stage_name, selected_lamella)

        # display milling stages from other tasks as background in the milling task editor for context
        if task_config.milling:
            background_milling_stages = []
            if self.show_related_milling_tasks and (
                related_configs := task_config.related_tasks
            ):
                for cfg in selected_lamella.task_config.values():
                    if isinstance(cfg, tuple(related_configs)) and cfg.milling:
                        for mcfg in cfg.milling.values():
                            background_milling_stages.extend(mcfg.enabled_stages)
            self.task_editor.milling.set_fib_image(self.image)
            self.task_editor.milling.set_background_milling_stages(
                background_milling_stages
            )
        self._refresh_milling_fov_warning()

        # Re-apply lock if this lamella/task is currently being processed
        self._apply_editing_lock(self._is_editing_locked())

        # special handling for fluorescence acquisition task
        is_fluorescence_task = isinstance(task_config, AcquireFluorescenceImageConfig)
        show_fm_row = is_fluorescence_task and self.combobox_fm_filenames.count() > 0
        self.combobox_fm_filenames.setVisible(show_fm_row)
        self.combobox_fm_filenames_label.setVisible(show_fm_row)
        if is_fluorescence_task:
            # drop the previous lamella/task's FM channels before compositing this one
            # (set_fm_image upserts by name and never removes stale channels)
            self.view_controller.clear_fm()
            if self.fm_image is not None:
                self.view_controller.set_fm_image(self.fm_image)

        # special handling for spot burn fiducial task
        is_spot_burn_task = isinstance(task_config, SpotBurnFiducialTaskConfig)
        self.spot_burn_coordinates_widget.setVisible(is_spot_burn_task)
        if is_spot_burn_task:
            self.spot_burn_coordinates_widget.set_settings(task_config.to_settings())

        self._draw_point_of_interest(selected_lamella.poi)
        self._draw_alignment_area()

        # task type drives which canvas page is visible: the FM composite for
        # fluorescence, else FIB (+ SEM)
        if is_fluorescence_task:
            self.view_controller.widget.show_fluorescence()
        else:
            self.view_controller.widget.show_beams()

    def _refresh_milling_fov_warning(self) -> None:
        """Warn when the shown task's milling FoV does not match the image FoV."""
        lamella = self._selected_lamella
        task_name = self.task_editor.task_name
        config = lamella.task_config.get(task_name) if lamella and task_name else None
        if config is not None and config.milling:
            try:
                milling_fov = next(iter(config.milling.values())).field_of_view
                image_hfw = self.image.metadata.image_settings.hfw  # type: ignore
                if not np.isclose(milling_fov, image_hfw):
                    milling_fov_um = format_value(milling_fov, unit="m", precision=0)
                    image_fov_um = format_value(image_hfw, unit="m", precision=0)
                    self.label_warning.setText(
                        f"Milling Task FoV ({milling_fov_um}) does not match image FoV ({image_fov_um})."
                    )
                    self.label_warning.setVisible(True)
                    return
            except Exception as e:
                logging.warning(f"Could not compare milling FoV and image FoV: {e}")

        self.label_warning.setVisible(False)

    def _on_spot_burn_coordinates_changed(self, settings: SpotBurnSettings):
        """The spot burn coordinates were edited on the canvas.

        Only the coordinates are taken from the editor's settings — milling current and
        exposure stay owned by the task parameters form.
        """
        self.task_editor.set_values({"coordinates": list(settings.coordinates)})

    def _on_point_of_interest_updated(
        self, point: Point, via: Optional[str] = None
    ) -> Dict[str, Any]:
        """Callback when the point of interest is updated. *via* names where the
        point came from, for the record, when not from the editor itself."""
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            return {"applied": False}

        logging.info(f"Updated {selected_lamella.name}, Point of Interest: {point}")
        self._edits.touch(
            selected_lamella, None, "poi", lambda: selected_lamella.poi, via
        )
        self._edits.touch_patterns(selected_lamella, via=via or "point of interest")
        result = self._edit_item_fields(selected_lamella, poi_patch(point))
        if result["applied"]:
            self._draw_point_of_interest(selected_lamella.poi)
        return result

    def _edit_item_fields(
        self, lamella: Lamella, patch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Patch *lamella*'s own fields as the App Server's item patch does, then
        refresh the patterns that moved with them and save."""
        result = edit_item_fields(lamella, patch, item_fields_version(lamella))
        if not result["applied"]:
            logging.warning(f"{lamella.name}: edit refused: {result}")
            return result
        if result["synced_tasks"]:
            self._on_selected_task_changed()  # show the patterns where they now are
        self._save_experiment()
        return result

    def _draw_point_of_interest(self, point: Point):
        """Draw the point-of-interest marker on the FIB canvas.

        Display-only (``modal`` + never armed → inert): the POI is moved via the milling
        editor's right-click "Move Point of Interest Here", not by dragging.
        """
        if getattr(self, "image", None) is None or self.image.metadata is None:
            return
        image_coords = conversions.microscope_image_to_image_coordinates(
            point=point,
            image_shape=self.image.data.shape,
            pixel_size=self.image.metadata.pixel_size.x,
        )
        self.view_controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id=POI_OVERLAY_ID,
                points=[(image_coords.x, image_coords.y)],
                color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                add_on_right_click=False,
                removable=False,
                modal=True,
                legend_label="Point of Interest",
            ),
        )

    def _on_toggle_alignment_area(self, checked: bool):
        self.show_alignment_area = checked
        if not checked:
            # Unchecking the edit switch runs its handler, which disarms the overlay.
            self.alignment_area_editable = False
            self.overlay_controls.set_visible(OVERLAY_EDIT_ALIGNMENT, False)
        self.overlay_controls.set_enabled(OVERLAY_EDIT_ALIGNMENT, checked)
        self._draw_alignment_area()
        if not checked:
            self.view_controller.arm_overlay(BeamType.ION, None)

    def _on_toggle_alignment_area_editable(self, checked: bool):
        self.alignment_area_editable = checked
        self._draw_alignment_area()
        # arming is an explicit user action here (redraws never touch arming, so they
        # can't steal input focus from the spot-burn overlay)
        self.view_controller.arm_overlay(
            BeamType.ION,
            ALIGNMENT_OVERLAY_ID if checked else None,
            label="Alignment",
            icon="mdi:vector-rectangle",
        )

    def _draw_alignment_area(self):
        """Draw (or remove) the editable alignment-area rectangle on the FIB canvas.

        Overlay only — arming is handled by the edit toggle so a task-change redraw
        never disarms another overlay. A separate id (not the milling widget's shared
        ``alignment``) keeps the milling read-only display from clobbering it.
        """
        if not self.show_alignment_area or self._selected_lamella is None:
            self.view_controller.remove_overlay(BeamType.ION, ALIGNMENT_OVERLAY_ID)
            return
        self.view_controller.set_overlay(
            BeamType.ION,
            AlignmentSpec(
                id=ALIGNMENT_OVERLAY_ID,
                rect=self._selected_lamella.alignment_area,
                display=True,
                editing=self.alignment_area_editable,
            ),
        )

    def _on_controller_overlay_edited(self, beam, overlay_id: str, value) -> None:
        """Fold a committed alignment-area edit on the FIB canvas back to the lamella."""
        if beam == BeamType.ION and overlay_id == ALIGNMENT_OVERLAY_ID:
            self._on_alignment_area_updated(value)

    def _on_alignment_area_updated(self, rect: FibsemRectangle):
        """Callback when the user drags/resizes the alignment area."""
        lamella = self._selected_lamella
        if lamella is None or not self.alignment_area_editable:
            return
        self._edits.touch(
            lamella, None, "alignment_area", lambda: lamella.alignment_area
        )
        self._edits.touch_patterns(lamella, via="alignment area")
        result = self._edit_item_fields(
            lamella,
            {
                f"alignment_area.{side}": float(getattr(rect, side))
                for side in ("left", "top", "width", "height")
            },
        )
        if not result["applied"]:
            # refused and put back: show the area the lamella still has
            self._draw_alignment_area()
            notification_service.show_toast(
                "Alignment area not changed: it must lie inside the image.",
                "warning",
            )

    def _add_poi_context_menu_action(
        self, config: ContextMenuConfig, point: Point
    ) -> None:
        """Add POI movement action to the right-click context menu."""
        if self._correlation_open:
            return
        config.add_action(
            "Move Point of Interest Here",
            callback=lambda: self._on_point_of_interest_updated(point),
        )

    def _open_correlation_dialog(self) -> None:
        """Open CorrelationTabWidget in a modal dialog."""
        selected_lamella = self._selected_lamella
        if selected_lamella is None:
            logging.error("No lamella selected, cannot open correlation dialog.")
            return

        from fibsem.correlation.history import (
            LamellaCorrelation,
            correlation_dir,
            new_run_dir,
        )
        from fibsem.ui.correlation.widgets.correlation_tab_widget import (
            CorrelationTabDialog,
        )

        history = LamellaCorrelation.discover(correlation_dir(selected_lamella.path))

        experiment = self.session.experiment
        protocol = experiment.task_protocol if experiment is not None else None
        spot_burns = self._spot_burn_coordinates(selected_lamella)

        # Named now, created by the widget only if this session records a
        # result: a folder for every open would offer an abandoned session as a
        # previous correlation to seed from.
        project_path = new_run_dir(selected_lamella.path)

        fib_current = self._image_path(
            selected_lamella, self.combobox_fib_filenames.currentData() or ""
        )
        fm_current = self._image_path(
            selected_lamella, self.combobox_fm_filenames.currentData() or ""
        )

        # The app's correlation, started afresh: one correlation is open at a
        # time, and whatever reads the session sees this one.
        session = self.session.correlation
        session.reset()
        dialog = CorrelationTabDialog(parent=self, session=session)
        dialog.set_project_dir(project_path)
        if protocol is not None:
            dialog.set_correlation_config(protocol.correlation)
        # Only a real image: with no reference image on disk self.image is a blank
        # placeholder generated for the editor's own canvas, and correlating
        # against it would scale the spot burns and the resulting POI by
        # placeholder dimensions.
        if self.image is not None and os.path.isfile(fib_current):
            dialog.set_fib_image(self.image)
        if self.fm_image is not None:
            dialog.set_fm_image(self.fm_image)

        # Lamella setup lives in the window's Images tab: the images in
        # this lamella's folder, and what to seed from — spot burns for a first
        # correlation, a previous run for a re-correlation.
        # Installed after the images are set, so seeding can rescale FM z.
        section = dialog.add_lamella_setup(
            spot_burns=spot_burns,
            spot_burn_field_width=self._spot_burn_field_width(selected_lamella),
            history=history,
            config=protocol.correlation if protocol is not None else None,
            fib_options=self._image_paths(
                selected_lamella, self.combobox_fib_filenames
            ),
            fib_current=fib_current,
            fm_options=self._image_paths(selected_lamella, self.combobox_fm_filenames),
            fm_current=fm_current,
        )
        section.emit_current_seed()  # apply the default source, live on the canvas
        # Predicted fiducials take their rotation and scale from a
        # previous run -- this lamella's first, then any other's on this system.
        if experiment is not None and getattr(experiment, "path", None):
            from fibsem.correlation.prior import experiment_runs

            dialog.set_prior_runs(
                experiment_runs(str(experiment.path), selected_lamella.path)
            )

        try:
            if dialog.exec_() != QDialog.Accepted:
                return

            # Persist the edited config to the protocol so every lamella inherits it.
            if protocol is not None:
                protocol.correlation = dialog.correlation_config
                self._save_experiment()

            if dialog.result is not None:
                self._handle_correlation_dialog_result(dialog.result, project_path)
        finally:
            # Gone before the next open: a view left behind would still listen
            # to the session's points.
            dialog.deleteLater()

    @staticmethod
    def _default_fib_filename(
        base_filenames: List[str],
        latest_task_filename: str,
        previous_selection: str,
    ) -> str:
        """Which FIB image the combo lands on: the last completed task's reference
        image, else whatever was selected before, else the newest.

        Everything here is a *basename*, because that is what the combo holds:
        a full path would make ``setCurrentText`` a silent no-op.
        """
        if latest_task_filename in base_filenames:
            return latest_task_filename
        if previous_selection in base_filenames:
            return previous_selection
        return base_filenames[-1] if base_filenames else ""

    @staticmethod
    def _latest_task_image(lamella: Lamella, suffix: str) -> str:
        """The last completed task's most zoomed final image ending in
        ``suffix``; empty when it left none (`latest_task_image`)."""
        return latest_task_image(lamella, suffix)

    @staticmethod
    def _image_path(lamella: Lamella, filename: str) -> str:
        """Absolute path for one of the lamella's discovered image filenames."""
        return os.path.join(lamella.path, filename) if filename else ""

    @classmethod
    def _image_paths(cls, lamella: Lamella, combo) -> List[str]:
        """Absolute paths for every filename offered by one of the editor combos.

        The correlation window's picker loads by path, so it handles this
        lamella's images and anything browsed to with the same control.
        """
        return [
            cls._image_path(lamella, combo.itemData(i)) for i in range(combo.count())
        ]

    @staticmethod
    def _spot_burn_coordinates(lamella: Lamella) -> List[Point]:
        """The lamella's spot-burn fiducial coordinates, or [] if it has none.

        Matched by config type — there's no fixed task-name key. Coordinates are
        normalised (0-1) to the lamella's FIB reference image.
        """
        for cfg in lamella.task_config.values():
            if isinstance(cfg, SpotBurnFiducialTaskConfig):
                return list(cfg.coordinates)
        return []

    @staticmethod
    def _spot_burn_field_width(lamella: Lamella) -> Optional[float]:
        """The field width (m) of the image the spot burns were placed on: the
        task's reference image, at ``reference_imaging.field_of_view1``."""
        for cfg in lamella.task_config.values():
            if isinstance(cfg, SpotBurnFiducialTaskConfig):
                return cfg.reference_imaging.field_of_view1 or None
        return None

    def _handle_correlation_dialog_result(
        self, result: "CorrelationResult", run_folder: Optional[str] = None
    ) -> None:
        """Handle the CorrelationResult returned from CorrelationTabDialog."""
        if result is None or not result.poi:
            logging.warning("Correlation dialog closed with no POI result.")
            return
        logging.info(
            f"correlation-result: rms={result.rms_error:.3f}, poi={result.poi[0].px_m}"
        )
        lamella = self._selected_lamella
        experiment = self.session.experiment
        apply_correlation(
            result,
            lamella,
            run_folder,
            getattr(experiment, "path", None),
            lambda poi: self._on_point_of_interest_updated(poi, via="correlation"),
            self._record_event,
        )

    def _record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put an event on the experiment's record, when a microscope keeps one."""
        microscope = self.session.microscope
        if microscope is not None:
            microscope.record_event(kind, payload)

    def _on_apply_to_other_clicked(self):
        """Open dialog to apply this lamella's config to other lamella."""
        source_lamella = self._selected_lamella
        if source_lamella is None:
            return

        experiment = self.session.experiment
        if experiment is None or len(experiment.positions) < 2:
            QMessageBox.information(
                self,
                "Not Enough Lamella",
                "There must be at least 2 lamella in the experiment to apply configurations.",
            )
            return

        other_names = [
            p.name for p in experiment.positions if p.id != source_lamella.id
        ]
        task_names = self._sort_task_names_by_workflow(
            list(source_lamella.task_config.keys())
        )

        dialog = ApplyLamellaConfigDialog(
            source_name=source_lamella.name,
            other_lamella_names=other_names,
            task_names=task_names,
            parent=self,
        )

        if dialog.exec_() != QDialog.Accepted:
            return

        selected_lamella_names = dialog.get_selected_lamella_names()
        selected_tasks = dialog.get_selected_tasks()
        update_base_protocol = dialog.get_update_base_protocol()

        if not selected_lamella_names or not selected_tasks:
            return

        via = "apply to other lamellae"
        targets = [p for p in experiment.positions if p.name in selected_lamella_names]
        self._edits.touch_task_configs(targets, selected_tasks, via=via)
        if update_base_protocol:
            protocol = experiment.task_protocol.task_config
            for task in selected_tasks:
                self._edits.touch(
                    None,
                    task,
                    "protocol.task_config",
                    lambda t=task: protocol.get(t),
                    via,
                )
        updated_count, _ = copy_task_configs(
            experiment,
            selected_lamella_names,
            selected_tasks,
            source_name=source_lamella.name,
            update_base_protocol=update_base_protocol,
        )

        # Show success message
        task_list = ", ".join(f"'{t}'" for t in selected_tasks)
        names_str = ", ".join(selected_lamella_names)
        msg = (
            f"Applied {len(selected_tasks)} task(s) ({task_list}) "
            f"to {updated_count} lamella ({names_str})."
        )
        if update_base_protocol:
            msg += "\nBase protocol was also updated."
        QMessageBox.information(self, "Apply Complete", msg)
        logging.info(msg)

    def _on_description_edited(self):
        """Persist the free-text description for the selected lamella."""
        lamella = self._selected_lamella
        if lamella is None:
            return
        text = self.line_edit_description.text()
        if lamella.description == text:
            return
        self._edit_item_fields(lamella, {"description": text})

    def _save_experiment(self):
        """Persist the edit, coalescing a burst of them into one write.

        `Experiment.save()` serialises the whole experiment -- 0.84 s at 100 lamella,
        on the GUI thread -- and every committed field in this editor called it. Editing
        a task means several fields in a row, and each paid the full price.

        Only *when* the write happens moves; what is written does not. The edit is
        already applied to the lamella, so a flush at any later point writes the
        complete current state. The experiment is held with the pending write
        (`PendingEdits.save_later`), so a write scheduled against one experiment
        cannot land on whichever one is loaded when the timer fires.
        """
        experiment = self.session.experiment
        if experiment is None:
            return
        self._edits.save_later(experiment, "item")

    def hideEvent(self, event) -> None:
        """Flush when the panel goes away -- a tab switch, or the window closing."""
        self.flush_pending_save()
        super().hideEvent(event)

    def flush_pending_save(self) -> None:
        """Record and write a coalesced edit now, if one is waiting. Safe to call
        at any time.

        Called wherever waiting any longer would risk the edit: the editor moving to
        another lamella or task, the experiment being swapped, a workflow starting, the
        panel being hidden, and the window closing. Cheap and idempotent when nothing
        is pending, so callers do not have to know whether there is.
        """
        self._edits.flush()
