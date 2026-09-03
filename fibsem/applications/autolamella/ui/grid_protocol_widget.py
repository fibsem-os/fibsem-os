"""The grid protocol's editor: the tasks in it, and the selected task's settings.

Hosted by the Protocol tab, under its Lamella | Grid selector: the task list
(the same list widget the lamella tasks use, so the two read alike) goes in the
first column and the settings panel in the second. One protocol shared by every
grid; which tasks run, and in what order, is chosen on Workflow → Grids.

Every edit is saved as it is made, like the rest of the app: there is no Save.
Reset to defaults is the one deliberate action. No form of its own: the beam task
shows the canvas Overview tab's settings column, the fluorescence task the FM
Overview tab's channel list and settings, so a setting means the same thing on
both tabs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment, GridTaskProtocol
from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
    GridTaskConfig,
)
from fibsem.ui.tokens import NEUTRAL_200, TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TaskNameListWidget,
    TitledPanel,
)
from fibsem.ui.widgets.fibsem_overview_settings_widget import (
    FibsemOverviewSettingsWidget,
)

# The poses a beam overview can be taken at. The stage knows more (MILLING), but
# an overview of a grid is taken flat to one beam or the other.
_ORIENTATIONS = ["SEM", "FIB"]

# The name a new task gets, by type, before the person renames it. The role the
# task records under follows the beam, so the default name says which.
_DEFAULT_NAMES = {
    BeamOverviewGridTaskConfig.task_type: "overview_sem",
    FluorescenceOverviewGridTaskConfig.task_type: "overview_fm",
}
_BTN_SIZE = 28


def _config_classes() -> Dict[str, Type[GridTaskConfig]]:
    """task_type -> config class, from the registry, so a new task type shows up
    here without a change."""
    return {
        task_type: task_cls.config_cls
        for task_type, task_cls in GRID_TASK_REGISTRY.items()
    }


class AddGridTaskDialog(QDialog):
    """Pick a task type and a name."""

    def __init__(self, protocol: GridTaskProtocol, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add grid task")
        self.setMinimumWidth(380)
        self._protocol = protocol
        self.result_: Optional[Tuple[str, str]] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.type_combo = QComboBox()
        for task_type, config_cls in _config_classes().items():
            self.type_combo.addItem(config_cls.display_name, task_type)
        self.type_combo.currentIndexChanged.connect(self._suggest_name)
        form.addRow("Task", self.type_combo)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "Unique within the protocol; the task's outputs are filed under it"
        )
        form.addRow("Name", self.name_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._suggest_name()

    def _suggest_name(self) -> None:
        base = _DEFAULT_NAMES.get(self.type_combo.currentData(), "grid_task")
        name, n = base, 2
        while name in self._protocol.task_config:
            name, n = f"{base}_{n}", n + 1
        self.name_edit.setText(name)

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Add grid task", "Give the task a name.")
            return
        if name in self._protocol.task_config:
            QMessageBox.warning(
                self, "Add grid task", f"There is already a task named {name}."
            )
            return
        self.result_ = (self.type_combo.currentData(), name)
        self.accept()


# ---------------------------------------------------------------------------
# Editors, one per config type: load(config) fills the form, apply(config) reads it
# ---------------------------------------------------------------------------


class _BeamOverviewEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.orientation = QComboBox()
        self.orientation.setEditable(True)
        self.orientation.addItems(_ORIENTATIONS)
        self.orientation.setToolTip(
            "The pose to acquire at; the grid's slot position is re-expressed for it"
        )
        form.addRow("Orientation", self.orientation)
        self.filename = QLineEdit()
        self.filename.setToolTip(
            "The stem of the stitched image's name; a time stamp is added per run"
        )
        form.addRow("Filename", self.filename)
        layout.addLayout(form)
        # The canvas Overview tab's own column (Imaging, Focus, Stack, Grid), so a
        # setting here is the setting there. Its Output panel is hidden: the task
        # files under the grid and stamps the name itself.
        self.settings = FibsemOverviewSettingsWidget()
        output = getattr(self.settings, "output_panel", None)
        if output is not None:
            output.hide()
        layout.addWidget(self.settings)
        layout.addStretch(1)
        self.orientation.currentTextChanged.connect(lambda _t: self.changed.emit())
        self.filename.editingFinished.connect(self.changed)
        self.settings.settings_changed.connect(self.changed)

    def load(self, config: BeamOverviewGridTaskConfig) -> None:
        self.orientation.setCurrentText(config.orientation)
        self.filename.setText(config.filename)
        self.settings.update_from_settings(config.settings)

    def apply(self, config: BeamOverviewGridTaskConfig) -> None:
        config.orientation = self.orientation.currentText().strip() or "SEM"
        config.filename = self.filename.text().strip() or "overview"
        config.settings = self.settings.get_settings()


class _FluorescenceOverviewEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Imported here: the FM widgets pull in the fluorescence stack, which a
        # system without an FM never needs to load until this editor is shown.
        from fibsem.ui.fm.widgets.channel_list_widget import ChannelListWidget
        from fibsem.ui.fm.widgets.fm_overview_settings_widget import (
            FMOverviewSettingsWidget,
        )

        self._fm = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.filename = QLineEdit()
        self.filename.setToolTip("The stem of the mosaic's name; a time stamp is added")
        form.addRow("Filename", self.filename)
        layout.addLayout(form)
        self.channels = ChannelListWidget(fm=None, channel_settings=[])
        # Four 40 px rows and the header. The list is Expanding, and left to it
        # the Channels panel took the column and pushed the settings down.
        self.channels.setMaximumHeight(220)
        layout.addWidget(TitledPanel("Channels", content=self.channels))
        self.settings = FMOverviewSettingsWidget()
        # The task files its output under the grid; the widget's own output panel
        # would say somewhere else.
        self.settings.output_panel.hide()
        layout.addWidget(self.settings)
        layout.addStretch(1)
        self.channels.settings_changed.connect(self.settings.set_channel_settings)
        self.filename.editingFinished.connect(self.changed)
        for signal in (
            self.channels.settings_changed,
            self.channels.channel_changed,
            self.channels.enabled_changed,
            self.channels.order_changed,
            self.channels.channel_added,
            self.channels.channel_removed,
            self.settings.changed,
            self.settings.z_widget.settings_changed,
            self.settings.autofocus_widget.settings_changed,
        ):
            signal.connect(lambda *_: self.changed.emit())

    def set_fm(self, fm) -> None:
        self._fm = fm
        self.channels.set_fm(fm)

    def load(self, config: FluorescenceOverviewGridTaskConfig) -> None:
        self.filename.setText(config.filename)
        self.channels.channel_settings = list(config.channels)
        self.settings.set_channel_settings(list(config.channels))
        self.settings.parameters = config.overview
        self.settings.z_widget.z_parameters = config.zparams
        if config.autofocus_settings is not None:
            self.settings.autofocus_widget.set_autofocus_settings(
                config.autofocus_settings
            )

    def apply(self, config: FluorescenceOverviewGridTaskConfig) -> None:
        config.filename = self.filename.text().strip() or "overview"
        config.channels = list(self.channels.channel_settings)
        config.overview = self.settings.parameters
        config.zparams = self.settings.z_widget.z_parameters
        config.autofocus_settings = self.settings.autofocus_settings


_EDITORS: Dict[str, Type[QWidget]] = {
    BeamOverviewGridTaskConfig.task_type: _BeamOverviewEditor,
    FluorescenceOverviewGridTaskConfig.task_type: _FluorescenceOverviewEditor,
}


class GridTaskEditorPanel(QWidget):
    """The selected task's settings: a title, a hint, reset, and the form."""

    changed = pyqtSignal()  # an edit in the showing form
    reset_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._editors: Dict[str, QWidget] = {}
        self._microscope = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        header.addWidget(self.title, 1)
        self.btn_reset = IconToolButton(
            icon="mdi:restore", tooltip="Reset to defaults", size=_BTN_SIZE
        )
        self.btn_reset.clicked.connect(self.reset_clicked)
        header.addWidget(self.btn_reset)
        layout.addLayout(header)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.hint)
        self.stack = QStackedWidget()
        self._blank = QWidget()
        self.stack.addWidget(self._blank)
        layout.addWidget(self.stack, 1)

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        fm = getattr(microscope, "fm", None)
        editor = self._editors.get(FluorescenceOverviewGridTaskConfig.task_type)
        if editor is not None:
            editor.set_fm(fm)

    def editor_for(self, task_type: str) -> Optional[QWidget]:
        editor = self._editors.get(task_type)
        if editor is None:
            editor_cls = _EDITORS.get(task_type)
            if editor_cls is None:
                return None
            editor = editor_cls()
            editor.changed.connect(self.changed)
            if hasattr(editor, "set_fm"):
                editor.set_fm(getattr(self._microscope, "fm", None))
            self._editors[task_type] = editor
            self.stack.addWidget(editor)
        return editor

    def show_nothing(self, hint: str) -> None:
        self.title.setText("")
        self.hint.setText(hint)
        self.btn_reset.setEnabled(False)
        self.stack.setCurrentWidget(self._blank)

    def show_config(self, config: GridTaskConfig) -> bool:
        """Fill the form for `config`. Returns whether there is one for its type."""
        editor = self.editor_for(config.task_type)
        if editor is None:
            self.title.setText(config.display_name)
            self.hint.setText(f"No settings editor for {config.task_type}.")
            self.btn_reset.setEnabled(False)
            self.stack.setCurrentWidget(self._blank)
            return False
        self.title.setText(f"{config.task_name} · {config.display_name}")
        self.hint.setText(
            f"Saved as grids/<grid>/{config.task_name}/ under the experiment. "
            "Same settings form as the Overview tab."
        )
        self.btn_reset.setEnabled(True)
        editor.load(config)
        self.stack.setCurrentWidget(editor)
        return True

    def apply_to(self, config: GridTaskConfig) -> None:
        editor = self._editors.get(config.task_type)
        if editor is not None:
            editor.apply(config)


class GridProtocolWidget(QWidget):
    """The grid protocol's task list and settings panel, and what ties them.

    ``embedded=True`` builds the two panels without placing them, for a host
    (the Protocol tab) that lays them into its own columns as ``task_list`` and
    ``editor_panel``. Standalone, they sit either side of a splitter.
    """

    protocol_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, embedded: bool = False):
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._microscope = None
        self._loading = False
        self.task_list = TaskNameListWidget()
        self.task_list.btn_add.setToolTip("Add grid task")
        self.task_list.btn_remove.setToolTip("Remove grid task")
        self.task_list.task_selected.connect(lambda _n: self._show_selected())
        self.task_list.add_clicked.connect(self._on_add)
        self.task_list.remove_clicked.connect(self._on_remove)
        self.editor_panel = GridTaskEditorPanel()
        self.editor_panel.changed.connect(self._on_editor_changed)
        self.editor_panel.reset_clicked.connect(self.reset_selected)
        if not embedded:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            splitter = QSplitter(Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(8, 8, 4, 8)
            left_layout.addWidget(self.task_list, 1)
            left.setMinimumWidth(220)
            left.setMaximumWidth(320)
            splitter.addWidget(left)
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(4, 8, 8, 8)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(self.editor_panel)
            right_layout.addWidget(scroll)
            splitter.addWidget(right)
            splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter)
        self.refresh()

    # -- model -----------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        self.refresh()

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        self.editor_panel.set_microscope(microscope)

    @property
    def protocol(self) -> Optional[GridTaskProtocol]:
        if self._experiment is None or self._experiment.task_protocol is None:
            return None
        return self._experiment.grid_protocol

    @property
    def selected_task_name(self) -> Optional[str]:
        return self.task_list.selected_task or None

    def selected_config(self) -> Optional[GridTaskConfig]:
        protocol, name = self.protocol, self.selected_task_name
        if protocol is None or name is None:
            return None
        return protocol.task_config.get(name)

    @property
    def task_names(self) -> List[str]:
        protocol = self.protocol
        return list(protocol.ordered_task_names) if protocol is not None else []

    def refresh(self) -> None:
        protocol = self.protocol
        self.task_list.blockSignals(True)
        self.task_list.set_tasks(self.task_names)
        self.task_list.blockSignals(False)
        self.task_list.set_buttons_visible(protocol is not None, protocol is not None)
        self._show_selected()

    def _show_selected(self) -> None:
        protocol = self.protocol
        config = self.selected_config()
        if protocol is None:
            self.editor_panel.show_nothing(
                "Load or create a task protocol first; the grid tasks live in it."
            )
            return
        if config is None:
            self.editor_panel.show_nothing(
                "No grid tasks yet. Add one with +: an SEM or FIB overview, or a "
                "fluorescence overview."
            )
            return
        self._loading = True
        try:
            self.editor_panel.show_config(config)
        finally:
            self._loading = False

    # -- edits -----------------------------------------------------------------

    def _on_add(self) -> None:
        protocol = self.protocol
        if protocol is None:
            return
        dialog = AddGridTaskDialog(protocol, parent=self)
        if dialog.exec_() != QDialog.Accepted or dialog.result_ is None:
            return
        self.add_task(*dialog.result_)

    def add_task(self, task_type: str, name: str) -> GridTaskConfig:
        """Add a task of `task_type` named `name` with default settings, and save."""
        protocol = self.protocol
        if protocol is None:
            raise ValueError("No task protocol to add a grid task to.")
        config_cls = _config_classes()[task_type]
        config = protocol.add(config_cls(task_name=name))
        self._save()
        self.task_list.blockSignals(True)
        self.task_list.set_tasks(self.task_names, preferred=name)
        self.task_list.select(name)
        self.task_list.blockSignals(False)
        self._show_selected()
        return config

    def _on_remove(self) -> None:
        name = self.selected_task_name
        if name is None:
            return
        if (
            QMessageBox.question(
                self,
                "Remove grid task",
                f"Remove {name} from the protocol? Its recorded outputs are kept.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.remove_task(name)

    def remove_task(self, name: str) -> None:
        protocol = self.protocol
        if protocol is None:
            return
        protocol.remove(name)
        self._save()
        self.refresh()

    def _on_editor_changed(self) -> None:
        """An edit in the form: into the config and onto disk, as it is made.
        Not while a form is being filled from a config."""
        if self._loading:
            return
        self.apply_selected()

    def apply_selected(self) -> Optional[GridTaskConfig]:
        """Read the form into the selected config and write the protocol."""
        config = self.selected_config()
        if config is None:
            return None
        self.editor_panel.apply_to(config)
        self._save()
        return config

    def reset_selected(self) -> Optional[GridTaskConfig]:
        """Replace the selected task's settings with the type's defaults, and save."""
        protocol, config = self.protocol, self.selected_config()
        if protocol is None or config is None:
            return None
        fresh = type(config)(task_name=config.task_name)
        protocol.task_config[config.task_name] = fresh
        self._save()
        self._show_selected()
        return fresh

    def _save(self) -> None:
        if self._experiment is None:
            return
        try:
            self._experiment.save(save_protocol=True)
        except Exception as e:  # noqa: BLE001 - a failed save is worth a line
            logging.warning(f"Could not save the grid protocol: {e}")
        self.protocol_changed.emit()
