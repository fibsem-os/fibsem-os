import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.config_edits import set_reference_imaging
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.applications.autolamella.ui.common.run_refusal import refused_during_run
from fibsem.applications.autolamella.ui.grids.protocol import GridProtocolWidget
from fibsem.applications.autolamella.ui.protocol.dialogs import (
    AutoLamellaGlobalTaskEditDialog,
    ProtocolDetailsDialog,
    ProtocolHeaderWidget,
)
from fibsem.applications.autolamella.ui.protocol.edit_recording import PendingEdits
from fibsem.applications.autolamella.ui.protocol.editor import TaskConfigEditor
from fibsem.applications.autolamella.ui.protocol.lamella_template import (
    LamellaDefaultConfigWidget,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.structures import BeamType, FibsemImage
from fibsem.ui import stylesheets
from fibsem.ui.tokens import (
    BORDER_COLOR,
    NEUTRAL_200,
    SEMANTIC_WARNING_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.custom_widgets import TaskNameListWidget
from fibsem.ui.widgets.spot_burn_coordinates_widget import SpotBurnCoordinatesWidget
from fibsem.ui.widgets.workflow.add_task_dialog import AddTaskDialog

# Frame used by the spot-burn coordinate dialog when the task's stored reference
# imaging is unusable (missing/zero). Matches ImageSettings' own defaults.
_FALLBACK_RESOLUTION = (1536, 1024)
_FALLBACK_HFW = 150e-6


def _valid_resolution(resolution) -> Tuple[int, int]:
    """A usable (x, y) pixel resolution, falling back when the protocol's is not."""
    try:
        x, y = (int(v) for v in resolution)
    except (TypeError, ValueError):
        return _FALLBACK_RESOLUTION
    return (x, y) if x > 0 and y > 0 else _FALLBACK_RESOLUTION


if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.workflows.session import AppSession


def _task_tooltips(protocol: AutoLamellaTaskProtocol) -> Dict[str, str]:
    """What each task is, and how the workflow runs it -- read here, edited elsewhere.

    The Protocol tab edits task definitions; the Workflow tab composes them into a
    run. The list rows stay plain so nothing on them looks editable that is not,
    and the tooltip says which tab owns each line.
    """
    workflow = {t.name: t for t in protocol.workflow_config.tasks}
    tips: Dict[str, str] = {}
    for name, config in protocol.task_config.items():
        # The base config class carries no task_type; a bare one reads as its class.
        raw = getattr(config, "task_type", None) or type(config).__name__
        kind = " ".join(w.capitalize() for w in str(raw).lower().split("_"))
        task = workflow.get(name)
        if task is None:
            run = "not included"
        else:
            parts = [config.attention.value]
            parts.append("required" if task.required else "optional")
            if config.requires:
                parts.append("after " + ", ".join(config.requires))
            run = ", ".join(parts)
        tips[name] = f"{name}\nType: {kind}\nWorkflow: {run}"
    return tips


class _DirtyBanner(QWidget):
    """The strip above the editor: quiet while the lamellae carry the current
    settings, lit with an Apply button once an edited task has not reached them.

    One fixed height in both states, so switching between them moves nothing
    underneath; and a widget that is always laid out, rather than one that floats,
    so it covers nothing either.
    """

    apply_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dirtyBanner")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(10)
        self.label = QLabel("")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.apply_button.setFixedHeight(24)
        self.apply_button.clicked.connect(self.apply_clicked)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.apply_button)
        self.show_for("", 0)

    def show_for(self, task_name: str, lamellae: int) -> None:
        """Lit for an edited *task_name* that has not been applied; quiet otherwise.

        Says only what the edit flag knows: that this task was edited here and
        Apply has not run since. Whether each lamella actually matches is not
        tracked, and the wording does not claim it.
        """
        rgb = QColor(SEMANTIC_WARNING_COLOR)
        plural = "lamella" if lamellae == 1 else "lamellae"
        if task_name and lamellae:
            self.label.setText(
                f"'{task_name}' was edited but has not been applied to the "
                f"{lamellae} existing {plural} yet."
            )
            self.label.setStyleSheet(
                f"color: {NEUTRAL_200}; background: transparent; border: none;"
            )
            self.apply_button.setText("Apply now")
            self.apply_button.setToolTip(
                f"Overwrite '{task_name}' on all {lamellae} existing {plural} with "
                "the protocol's settings, including any edited individually."
            )
            self.apply_button.setVisible(True)
            self.setStyleSheet(
                # Not an f-string: every literal brace would need doubling.
                "#dirtyBanner { background: rgba(%d, %d, %d, 0.12);"  # noqa: UP031
                " border-top: 1px solid rgba(%d, %d, %d, 0.5); }"
                % (
                    rgb.red(),
                    rgb.green(),
                    rgb.blue(),
                    rgb.red(),
                    rgb.green(),
                    rgb.blue(),
                )
            )
            return
        self.label.setText(
            "Task edits reach existing lamellae only when applied; new lamellae "
            "take the current settings."
        )
        self.label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"
            " background: transparent; border: none;"
        )
        self.apply_button.setVisible(False)
        self.setStyleSheet(
            f"#dirtyBanner {{ background: transparent;"
            f" border-top: 1px solid {BORDER_COLOR}; }}"
        )


class AutoLamellaProtocolTaskConfigEditor(QWidget):
    """A widget to edit the AutoLamella protocol."""

    workflow_config_changed = pyqtSignal(object)  # AutoLamellaWorkflowConfig
    # The grid protocol changed on this editor's Grid page (a task added,
    # removed, or edited and saved). Re-emitted here because the Grid page is
    # built inside this editor, after the window has wired its signals.
    grid_protocol_changed = pyqtSignal()

    def __init__(self, session: "AppSession", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        # Each edit, for the experiment's record (FIB-1034).
        self._edits = PendingEdits(
            lambda: self.session.microscope,
            via="protocol editor",
            parent=self,
        )
        self.setStyleSheet(stylesheets.NAPARI_STYLE)

        self.task_editor: Optional[TaskConfigEditor] = None
        self.microscope = self.session.microscope  # type: ignore[assignment]
        self.experiment: "Experiment" = self.session.experiment  # type: ignore[assignment]

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._try_initialize()

    def _try_initialize(self) -> None:
        """Create and populate widgets once both microscope and experiment are available."""
        if self.task_editor is not None:
            return  # already initialized
        if self.microscope is None:
            return
        if self.experiment is None or self.experiment.task_protocol is None:
            return
        self._create_widgets()
        self._setup_connections()
        self._initialise_widgets()
        self._on_selected_task_changed()

    def _on_microscope_connected(self):
        if self.session.microscope is None:
            raise ValueError("Microscope is None, cannot open protocol editor.")
        self.microscope = self.session.microscope
        if self.task_editor is not None:
            # Reconnect: built once, so every widget under it still holds the
            # microscope it was built with, and each one has to be told.
            self.task_editor.set_microscope(self.microscope)
            self.grid_protocol.set_microscope(self.microscope)
        else:
            self._try_initialize()

    def set_experiment(self, experiment: "Experiment"):
        """Set the experiment for the protocol editor."""
        self.experiment = experiment
        if self.task_editor is None:
            self._try_initialize()
        else:
            self._initialise_widgets()
            self._on_selected_task_changed()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""

        # Protocol identity (Column 1, top): one line, and a pencil for the
        # name, description and version. They are set when the experiment is
        # created and rarely touched after; a form for them took the top of the
        # only column that has to fit two task lists.
        self.protocol_header = ProtocolHeaderWidget(parent=self)
        # The grid protocol: its task list and settings panel are laid into this
        # editor's columns under the Lamella | Grid selector.
        self.grid_protocol = GridProtocolWidget(parent=self, embedded=True)
        self.grid_protocol.set_experiment(self.experiment)
        self.grid_protocol.set_microscope(self.microscope)
        self.grid_protocol.protocol_changed.connect(self.grid_protocol_changed)

        # Task forms (Column 2), and their milling editor (Column 3 — no
        # controller, config panels only)
        self.task_editor = TaskConfigEditor(
            self.session, self.microscope, self._edits, "protocol", parent=self
        )

        # lamella, milling controls (Column 1)
        self.task_list_widget = TaskNameListWidget()

        self.pushButton_sync_to_lamella = QPushButton(
            "Apply Config to Existing Lamella"
        )
        self.pushButton_sync_to_lamella.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_sync_to_lamella.setToolTip(
            "Update all existing lamella with the current task configuration"
        )
        self._protocol_dirty = False

        self.pushButton_open_global_editor = QPushButton("Global Edit")
        self.pushButton_open_global_editor.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_open_global_editor.setToolTip(
            "Globally edit reference imaging settings and milling FoV across multiple tasks."
        )

        self.pushButton_open_lamella_defaults = QPushButton("Lamella Template")
        self.pushButton_open_lamella_defaults.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_open_lamella_defaults.setToolTip(
            "Edit the initial state applied to every new lamella created from this protocol."
        )

        # only meaningful for a spot-burn task; shown/hidden in _on_selected_task_changed
        self.pushButton_edit_spot_burn = QPushButton("Spot Burn Coordinates")
        self.pushButton_edit_spot_burn.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_edit_spot_burn.setToolTip(
            "Place this task's default spot-burn coordinates on a reference frame, "
            "instead of hand-editing them in the protocol file."
        )
        self.pushButton_edit_spot_burn.setVisible(False)

        self.label_warning = QLabel("")
        self.label_warning.setStyleSheet("color: orange;")
        self.label_warning.setVisible(False)
        self.label_warning.setWordWrap(True)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.pushButton_sync_to_lamella)
        self.button_layout.addWidget(self.pushButton_open_global_editor)
        self.button_layout.addWidget(self.pushButton_open_lamella_defaults)
        self.button_layout.addWidget(self.pushButton_edit_spot_burn)

        lamella_page = QWidget()
        lamella_layout = QVBoxLayout(lamella_page)
        lamella_layout.setContentsMargins(0, 6, 0, 0)
        lamella_layout.addWidget(self.task_list_widget)
        lamella_layout.addLayout(self.button_layout)
        grid_page = QWidget()
        grid_layout = QVBoxLayout(grid_page)
        grid_layout.setContentsMargins(0, 6, 0, 0)
        grid_layout.addWidget(self.grid_protocol.task_list)
        # Lamella | Grid: one list at a time, and the columns to the right follow.
        self.protocol_tabs = QTabWidget()
        self.protocol_tabs.addTab(lamella_page, "Lamella")
        self.protocol_tabs.addTab(grid_page, "Grid")
        self.protocol_tabs.currentChanged.connect(
            lambda _i: self._on_protocol_kind_changed()
        )

        # --- Column 1: Protocol info + task selector ---
        col1_content = QWidget()
        col1_layout = QVBoxLayout(col1_content)
        col1_layout.setContentsMargins(4, 4, 4, 4)
        col1_layout.addWidget(self.protocol_header)
        col1_layout.addWidget(self.protocol_tabs)
        col1_layout.addWidget(self.label_warning)
        col1_layout.addStretch()
        col1_scroll = QScrollArea()
        col1_scroll.setWidgetResizable(True)
        col1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col1_scroll.setWidget(col1_content)

        # --- Column 2: Task parameters ---
        col2_content = QWidget()
        col2_layout = QVBoxLayout(col2_content)
        col2_layout.setContentsMargins(4, 4, 4, 4)
        col2_layout.addWidget(self.task_editor)
        col2_layout.addWidget(self.grid_protocol.editor_panel)
        self.grid_protocol.editor_panel.setVisible(False)
        col2_layout.addStretch()
        col2_scroll = QScrollArea()
        col2_scroll.setWidgetResizable(True)
        col2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col2_scroll.setWidget(col2_content)

        # --- Column 3: Milling parameters (config panels, no napari canvas) ---
        col3_scroll = QScrollArea()
        col3_scroll.setWidgetResizable(True)
        col3_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        col3_scroll.setWidget(self.task_editor.milling)

        # --- 3-column splitter ---
        splitter = QSplitter(Qt.Horizontal)  # type: ignore
        splitter.addWidget(col1_scroll)
        splitter.addWidget(col2_scroll)
        splitter.addWidget(col3_scroll)
        # The middle column wide enough that the grid task's settings (the canvas
        # Overview tab's settings column) and the lamella parameters keep their
        # labels; the third wide enough that a milling stage row shows every cell
        # at its readable minimum without running off the edge.
        splitter.setSizes([280, 580, 580])

        self._main_layout.addWidget(splitter)
        # The status strip below the columns is always there, at one height, so
        # nothing moves when it changes state: quiet while the lamellae carry the
        # current settings, lit with the Apply button once a task is edited. At the
        # bottom, where status lives and where the existing Apply button already
        # is, rather than at the top where a quiet sentence would be the first
        # thing read. That button stays, for now, as the second route to the action.
        self.dirty_banner = _DirtyBanner(parent=self)
        self.dirty_banner.apply_clicked.connect(self._on_sync_to_lamella_clicked)
        self._main_layout.addWidget(self.dirty_banner)

    def _set_protocol_dirty(self, dirty: bool):
        self._protocol_dirty = dirty
        self._refresh_status_strip()
        if dirty and self.pushButton_sync_to_lamella.isEnabled():
            self.pushButton_sync_to_lamella.setStyleSheet(
                stylesheets.PRIMARY_BUTTON_STYLESHEET
            )
            self.pushButton_sync_to_lamella.setToolTip(
                "Protocol edited — click to apply current task configuration to existing lamella."
            )
        else:
            self.pushButton_sync_to_lamella.setStyleSheet(
                stylesheets.SECONDARY_BUTTON_STYLESHEET
            )
            self.pushButton_sync_to_lamella.setToolTip(
                "Update all existing lamella with the current task configuration"
            )

    def _refresh_status_strip(self) -> None:
        banner = getattr(self, "dirty_banner", None)
        if banner is None or self.experiment is None:
            return  # the widgets are still being built
        lamellae = len(self.experiment.positions)
        banner.show_for(
            self.task_list_widget.selected_task if self._protocol_dirty else "",
            lamellae,
        )

    def _setup_connections(self):
        """Setup signal connections - called once during initialization."""
        self.task_list_widget.task_selected.connect(
            lambda _: self._on_selected_task_changed()
        )
        self.task_list_widget.add_clicked.connect(self._on_add_task_clicked)
        self.task_list_widget.remove_clicked.connect(self._on_remove_task_clicked)
        self.task_editor.edited.connect(lambda _: self._set_protocol_dirty(True))
        self.pushButton_edit_spot_burn.clicked.connect(
            self._on_spot_burn_coordinates_clicked
        )
        self.pushButton_sync_to_lamella.clicked.connect(
            self._on_sync_to_lamella_clicked
        )
        self.pushButton_open_global_editor.clicked.connect(self._on_global_edit_clicked)
        self.pushButton_open_lamella_defaults.clicked.connect(
            self._on_lamella_defaults_clicked
        )
        self.protocol_header.edit_clicked.connect(self._on_edit_protocol_details)

    def _initialise_widgets(self):
        """Initialise the widgets based on the current experiment protocol."""

        if self.experiment is None or self.experiment.task_protocol is None:
            raise ValueError(
                "Experiment or task protocol is None, cannot initialise protocol editor."
            )

        self.protocol_header.update_from_protocol(self.experiment.task_protocol)
        self.grid_protocol.set_experiment(self.experiment)

        task_names = list(self.experiment.task_protocol.task_config.keys())
        self.task_list_widget.set_tasks(task_names)
        self.task_list_widget.set_task_tooltips(
            _task_tooltips(self.experiment.task_protocol)
        )

    def refresh_if_showing_task(self, task_name: str) -> None:
        """Rebuild the panel if it is displaying ``task_name``.

        Called (on the GUI thread) after an outside writer — the agent's
        protocol-level config patch — changed the config under an open form,
        so no stale form survives to write old values back.
        """
        try:
            selected = self.task_list_widget.selected_task
        except Exception:
            return
        if selected == task_name:
            self._on_selected_task_changed()

    def _on_selected_task_changed(self):
        """Show the selected task's config."""
        task_config = self.task_editor.show_task(self.task_list_widget.selected_task)
        if task_config is None:
            # No lamella task to show: a protocol with none yet (a grid-only
            # experiment, or one being built from an empty protocol). The
            # columns wait for a task rather than the editor failing to open.
            return

        self._set_protocol_dirty(False)

        # spot-burn coordinates live in their own dialog (they need a canvas to place
        # points on, which this tab has no room for) — offer the button only for that task
        self.pushButton_edit_spot_burn.setVisible(
            isinstance(task_config, SpotBurnFiducialTaskConfig)
        )

    def flush_pending_save(self) -> None:
        """Record and write the edits still settling. Safe to call at any time."""
        self._edits.flush()

    def _on_spot_burn_coordinates_clicked(self):
        """Edit this task's default spot-burn coordinates on a reference frame.

        A dialog rather than a column in this tab: placing points needs a canvas, and
        this editor's third column is the milling viewer built with ``viewer=None``.
        The widget brings its own canvas via a standalone ``MicroscopeViewController``,
        so no microscope and no acquired image are involved.
        """
        task_name = self.task_list_widget.selected_task
        config = self.experiment.task_protocol.task_config[task_name]
        if not isinstance(config, SpotBurnFiducialTaskConfig):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Spot Burn Coordinates — {task_name}")
        dialog.setModal(True)

        # The frame the coordinates are relative to. Coordinates are normalised 0-1 in x
        # and y *independently*, so the aspect ratio has to match the real reference image
        # or every point lands somewhere else — hence the task's own reference-imaging
        # resolution rather than an arbitrary square. field_of_view1 gives the scalebar
        # its scale. Deliberately flat, not noise: this is a coordinate frame, not a
        # picture of anyone's sample.
        # Fall back rather than crash on a malformed protocol: this feature exists
        # because people hand-edit these files, so a null or zero resolution is a real
        # input, and generate_blank_image would raise TypeError / ZeroDivisionError.
        ref = config.reference_imaging
        resolution = _valid_resolution(getattr(ref.imaging, "resolution", None))
        hfw = ref.field_of_view1 if (ref.field_of_view1 or 0) > 0 else _FALLBACK_HFW
        image = FibsemImage.generate_blank_image(
            resolution=resolution, hfw=hfw, random=False
        )

        # Kept on the dialog: the controller must outlive the coordinate widget, and
        # LamellaEditorView must outlive the controller (it is the controller's widget).
        dialog._view = LamellaEditorView()
        dialog._controller = MicroscopeViewController(view=dialog._view)
        dialog._controller.set_image(BeamType.ION, image)

        coord_widget = SpotBurnCoordinatesWidget(
            controller=dialog._controller,
            beam=BeamType.ION,
            settings=config.to_settings(),
            parent=dialog,
        )
        coord_widget.set_image_shape(image.data.shape)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        body = QHBoxLayout()
        body.setSpacing(8)
        coord_widget.setMinimumWidth(360)
        body.addWidget(coord_widget, 0)
        body.addWidget(dialog._controller.widget, 1)
        layout.addLayout(body)

        hint = QLabel(
            f"Coordinates are fractions of the reference image ({resolution[0]}×{resolution[1]} px, "
            f"{hfw * 1e6:.0f} µm wide): 0-1 across, 0-1 down, origin top-left. "
            "Right-click the frame to add a point, drag to move, Delete to remove."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        ok_btn = QPushButton("OK")
        cancel_btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        ok_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        for b in (cancel_btn, ok_btn):
            b.setDefault(False)
            b.setAutoDefault(False)
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        dialog.resize(1000, 560)
        try:
            if dialog.exec_() == QDialog.Accepted:
                # the fields the dialog edits (``apply_settings``), so the task's
                # others (milling, reference imaging, autofocus) survive the edit
                settings = coord_widget.get_settings()
                self.task_editor.set_values(
                    {
                        "coordinates": list(settings.coordinates),
                        "milling_current": settings.milling_current,
                        "exposure_time": settings.exposure_time,
                    }
                )
        finally:
            # QDialog(self) parents the dialog to this editor, so Qt keeps it alive after
            # exec_ returns and every open would strand another canvas + controller for
            # the session. Delete after reading the settings, never before.
            dialog.deleteLater()

    def _touch_protocol(self, task_name: str, via: Optional[str] = None) -> None:
        """Before the protocol's config for *task_name* is replaced, added or
        removed: for the experiment's record (FIB-1034)."""
        configs = self.experiment.task_protocol.task_config
        self._edits.touch(
            None, task_name, "protocol.task_config", lambda: configs.get(task_name), via
        )

    def _save_experiment(self):
        """Save the experiment if available."""
        if self.session.experiment is not None:
            self.session.experiment.save(save_protocol=True)

    def _on_add_task_clicked(self):
        """Show dialog to add a new task."""
        if refused_during_run(self.session.runner, "add tasks"):
            return
        dialog = AddTaskDialog(
            {task_type: cls.config_cls for task_type, cls in get_tasks().items()},
            self.experiment.task_protocol.task_config,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted or dialog.chosen is None:
            return
        task_type, task_name = dialog.chosen
        via = "add task"
        self._touch_protocol(task_name, via)
        self._edits.touch_task_configs(self.experiment.positions, [task_name], via)
        # To the protocol, the workflow and every lamella
        self.experiment.add_task(task_type, task_name)

        # Save experiment
        self._save_experiment()

        # Refresh widgets
        self._initialise_widgets()
        self.task_list_widget.select(task_name)
        self.workflow_config_changed.emit(self.experiment.task_protocol.workflow_config)
        logging.info(f"Added new task: {task_name} ({task_type})")

    def _on_remove_task_clicked(self):
        """Remove the selected task with confirmation dialog."""
        if refused_during_run(self.session.runner, "remove tasks"):
            return
        selected_task_name = self.task_list_widget.selected_task

        if not selected_task_name:
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the task '{selected_task_name}'?\n\n"
            "It is removed from the protocol, the workflow and every lamella, "
            "including any changes made to a lamella's copy. Task history is kept.\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Remove from experiment
            if selected_task_name in self.experiment.task_protocol.task_config:
                via = "remove task"
                self._touch_protocol(selected_task_name, via)
                self._edits.touch(
                    None,
                    selected_task_name,
                    "protocol.workflow_config",
                    lambda: self.experiment.task_protocol.workflow_config,
                    via,
                )
                self._edits.touch_task_configs(
                    self.experiment.positions, [selected_task_name], via
                )
                self.experiment.remove_task(selected_task_name)

                # Save experiment
                self._save_experiment()

                # Refresh widgets
                self._initialise_widgets()
                self.workflow_config_changed.emit(
                    self.experiment.task_protocol.workflow_config
                )

                logging.info(f"Removed task: {selected_task_name}")

    def _on_global_edit_clicked(self):
        """Show dialog for globally editing reference imaging and milling FoV."""
        dialog = AutoLamellaGlobalTaskEditDialog(self.experiment, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            via = "global edit"
            update_lamella = dialog.checkbox_update_existing.isChecked()
            selected_task_names = dialog.get_selected_tasks()
            for task_name in selected_task_names:
                self._touch_protocol(task_name, via)
            if update_lamella:
                self._edits.touch_task_configs(
                    self.experiment.positions, selected_task_names, via
                )

            updated_count, skipped = set_reference_imaging(
                self.experiment,
                selected_task_names,
                dialog.ref_image_params_widget.get_settings(),
                dialog.milling_fov(),
                self.experiment.positions if update_lamella else (),
            )

            # Refresh the current task view to reflect changes
            self._on_selected_task_changed()

            # Show success message
            selected_count = len(selected_task_names)
            msg = f"Successfully updated reference imaging for {selected_count} task(s) and milling FoV for {updated_count} task(s) with milling."
            if update_lamella:
                msg += "\nExisting lamella configurations were also updated."
            if skipped:
                names = ", ".join(f"{lamella} ({task})" for lamella, task in skipped)
                msg += f"\nNot changed, because the task is running: {names}."
            QMessageBox.information(
                self,
                "Global Edit Applied",
                msg,
            )

    def _on_lamella_defaults_clicked(self):
        """Show dialog for editing the protocol's LamellaDefaultConfig."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Lamella Template")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        template_widget = LamellaDefaultConfigWidget()
        template_widget.set_template(self.experiment.task_protocol.lamella_defaults)
        layout.addWidget(template_widget)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        ok_btn = QPushButton("OK")
        cancel_btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        ok_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        cancel_btn.setDefault(False)
        cancel_btn.setAutoDefault(False)
        ok_btn.setDefault(False)
        ok_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(dialog.reject)

        def accept_if_valid():
            why = template_widget.get_template().why_invalid()
            if why:
                QMessageBox.warning(dialog, "Lamella Template", why)
                return
            dialog.accept()

        ok_btn.clicked.connect(accept_if_valid)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        try:
            if dialog.exec_() == QDialog.Accepted:
                self.experiment.task_protocol.lamella_defaults = (
                    template_widget.get_template()
                )
                self._save_experiment()
        finally:
            dialog.deleteLater()  # parented to this editor, so Qt keeps it otherwise

    def _on_edit_protocol_details(self) -> None:
        """The pencil on the header line: name, description and version."""
        protocol = self.experiment.task_protocol if self.experiment else None
        if protocol is None:
            return
        dialog = ProtocolDetailsDialog(protocol, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        for field, value in dialog.values().items():
            if getattr(protocol, field, None) != value:
                setattr(protocol, field, value)
                logging.info(f"Updated protocol {field}: {value}")
        self._save_experiment()
        self.protocol_header.update_from_protocol(protocol)

    @property
    def grid_protocol_active(self) -> bool:
        tabs = getattr(self, "protocol_tabs", None)
        return tabs is not None and tabs.currentIndex() == 1

    def _on_protocol_kind_changed(self) -> None:
        """Lamella or Grid: the columns to the right show that kind's settings."""
        grid = self.grid_protocol_active
        self.grid_protocol.editor_panel.setVisible(grid)
        self.task_editor.setVisible(not grid)
        if grid:
            self.task_editor.milling.setVisible(False)
            self.pushButton_edit_spot_burn.setVisible(False)
            self.grid_protocol.refresh()
        else:
            self._on_selected_task_changed()

    def _on_sync_to_lamella_clicked(self):
        """Sync the current task configuration to all existing lamella."""
        selected_task_name = self.task_list_widget.selected_task

        if not selected_task_name:
            return

        # Check if there are any positions
        if not self.experiment.positions:
            QMessageBox.information(
                self,
                "No Lamella",
                "There are no existing lamella to sync the configuration to.",
            )
            return

        # Show confirmation dialog
        num_lamella = len(self.experiment.positions)
        reply = QMessageBox.question(
            self,
            "Confirm Sync",
            (
                f"Are you sure you want to sync the task configuration '{selected_task_name}' to all "
                f"{num_lamella} existing lamella?\n\nThis will overwrite their current task configurations, while "
                f"keeping any existing milling pattern positions unchanged."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Perform the sync (from base protocol to all lamella)
            all_lamella_names = [p.name for p in self.experiment.positions]
            self._edits.touch_task_configs(
                self.experiment.positions, [selected_task_name], via="sync to lamellae"
            )
            updated_count = self.experiment.apply_lamella_config(
                lamella_names=all_lamella_names,
                task_names=[selected_task_name],
            )

            # Save the experiment
            self._save_experiment()
            self._set_protocol_dirty(False)

            # Show success message
            QMessageBox.information(
                self,
                "Sync Complete",
                f"Successfully synced task configuration '{selected_task_name}' to {updated_count} lamella.",
            )

            logging.info(
                f"Synced task configuration '{selected_task_name}' to {updated_count} lamella"
            )
