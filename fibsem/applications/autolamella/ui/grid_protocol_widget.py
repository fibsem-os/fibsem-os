"""Grids tab · Protocol: the tasks in the experiment's grid protocol, and their settings.

One protocol shared by every grid, so this does not follow card selection.
Which tasks run, and in what order, is chosen on Workflow → Grids; this is only
what each task is set to. No form of its own: the beam task shows the canvas
Overview tab's settings column, the fluorescence task the FM Overview tab's
channel list and settings, so a setting means the same thing on both tabs.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Type

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QToolButton,
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
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import NEUTRAL_200, TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton, TitledPanel
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


_ROW_HEIGHT = 32
_BTN_SIZE = 28


class _TaskRow(QWidget):
    """One task in the list: its name, its kind, and a trash icon."""

    clicked = pyqtSignal(str)
    remove_clicked = pyqtSignal(str)

    def __init__(self, config: GridTaskConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.task_name = config.task_name
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(_ROW_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 4, 0)
        layout.setSpacing(8)
        # The name first: it is what the workflow, the outputs and the history
        # call the task. The kind is the muted second word.
        self.name_label = QLabel(config.task_name)
        self.name_label.setTextFormat(Qt.PlainText)
        self.name_label.setStyleSheet("background: transparent; font-weight: bold;")
        layout.addWidget(self.name_label)
        self.task_label = ElidedLabel(config.display_name)
        self.task_label.setStyleSheet(
            f"background: transparent; color: {TEXT_MUTED_COLOR}; font-size: 11px;"
        )
        layout.addWidget(self.task_label, 1)
        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE
        )
        self.btn_remove.clicked.connect(
            lambda: self.remove_clicked.emit(self.task_name)
        )
        layout.addWidget(self.btn_remove)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.task_name)
        super().mousePressEvent(event)


class GridProtocolWidget(QWidget):
    """The task list, and the selected task's settings.

    Every edit is saved as it is made, like the rest of the app: there is no
    Save. Reset to defaults is the one deliberate action.
    """

    protocol_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._microscope = None
        self._editors: Dict[str, QWidget] = {}
        self._rows: Dict[str, _TaskRow] = {}
        self._loading = False
        self._setup_ui()
        self.refresh()

    # -- layout ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        header = QHBoxLayout()
        header.setContentsMargins(6, 0, 4, 0)
        title = QLabel("Grid tasks")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        header.addWidget(title)
        header.addStretch(1)
        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add grid task", size=_BTN_SIZE
        )
        self.btn_add.clicked.connect(self._on_add)
        header.addWidget(self.btn_add)
        left_layout.addLayout(header)
        subtitle = QLabel(
            "Shared by every grid. Which tasks run, and in what order, is chosen "
            "on the Workflow tab; this is only their settings."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        left_layout.addWidget(subtitle)
        self.task_list = QListWidget()
        self.task_list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self.task_list.setSelectionMode(QListWidget.SingleSelection)
        self.task_list.setResizeMode(QListWidget.Adjust)
        self.task_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.task_list.currentItemChanged.connect(lambda *_: self._show_selected())
        left_layout.addWidget(self.task_list, 1)
        left.setMinimumWidth(220)
        left.setMaximumWidth(320)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 8, 8, 8)
        header = QHBoxLayout()
        self.editor_title = QLabel()
        self.editor_title.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        header.addWidget(self.editor_title, 1)
        self.btn_reset = IconToolButton(
            icon="mdi:restore", tooltip="Reset to defaults", size=_BTN_SIZE
        )
        self.btn_reset.clicked.connect(self._on_reset)
        header.addWidget(self.btn_reset)
        right_layout.addLayout(header)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        right_layout.addWidget(self.hint_label)
        self.editor_stack = QStackedWidget()
        self._blank = QWidget()
        self.editor_stack.addWidget(self._blank)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.editor_stack)
        right_layout.addWidget(scroll, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # -- model -----------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        self.refresh()

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        fm = getattr(microscope, "fm", None)
        editor = self._editors.get(FluorescenceOverviewGridTaskConfig.task_type)
        if editor is not None:
            editor.set_fm(fm)

    @property
    def protocol(self) -> Optional[GridTaskProtocol]:
        if self._experiment is None or self._experiment.task_protocol is None:
            return None
        return self._experiment.grid_protocol

    @property
    def selected_task_name(self) -> Optional[str]:
        item = self.task_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def selected_config(self) -> Optional[GridTaskConfig]:
        protocol, name = self.protocol, self.selected_task_name
        if protocol is None or name is None:
            return None
        return protocol.task_config.get(name)

    def refresh(self) -> None:
        protocol = self.protocol
        current = self.selected_task_name
        self.task_list.blockSignals(True)
        self.task_list.clear()
        self._rows = {}
        if protocol is not None:
            for name in protocol.ordered_task_names:
                config = protocol.task_config[name]
                row = _TaskRow(config)
                row.clicked.connect(self._select)
                row.remove_clicked.connect(self._on_remove)
                item = QListWidgetItem(self.task_list)
                item.setData(Qt.UserRole, name)
                item.setSizeHint(QSize(0, _ROW_HEIGHT))
                self.task_list.addItem(item)
                self.task_list.setItemWidget(item, row)
                self._rows[name] = row
                if name == current:
                    self.task_list.setCurrentItem(item)
        self.task_list.blockSignals(False)
        if protocol is not None and self.task_list.currentItem() is None:
            if self.task_list.count():
                self.task_list.setCurrentRow(0)
        self.btn_add.setEnabled(protocol is not None)
        self._show_selected()

    def _show_selected(self) -> None:
        protocol = self.protocol
        config = self.selected_config()
        self.btn_reset.setEnabled(config is not None)
        if protocol is None:
            self.editor_title.setText("")
            self.hint_label.setText(
                "Load or create a task protocol first; the grid tasks live in it."
            )
            self.editor_stack.setCurrentWidget(self._blank)
            return
        if config is None:
            self.editor_title.setText("")
            self.hint_label.setText(
                "No grid tasks yet. Add one: an SEM or FIB overview, or a "
                "fluorescence overview."
            )
            self.editor_stack.setCurrentWidget(self._blank)
            return
        editor = self._editor_for(config.task_type)
        if editor is None:
            self.editor_title.setText(config.display_name)
            self.hint_label.setText(f"No settings editor for {config.task_type}.")
            self.editor_stack.setCurrentWidget(self._blank)
            return
        self.editor_title.setText(f"{config.task_name} · {config.display_name}")
        self.hint_label.setText(
            f"Saved as grids/<grid>/{config.task_name}/ under the experiment. "
            "Same settings form as the Overview tab."
        )
        self._loading = True
        try:
            editor.load(config)
        finally:
            self._loading = False
        self.editor_stack.setCurrentWidget(editor)

    def _editor_for(self, task_type: str) -> Optional[QWidget]:
        editor = self._editors.get(task_type)
        if editor is None:
            editor_cls = _EDITORS.get(task_type)
            if editor_cls is None:
                return None
            editor = editor_cls()
            editor.changed.connect(self._on_editor_changed)
            if hasattr(editor, "set_fm"):
                editor.set_fm(getattr(self._microscope, "fm", None))
            self._editors[task_type] = editor
            self.editor_stack.addWidget(editor)
        return editor

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
        self.refresh()
        self._select(name)
        return config

    def _on_remove(self, name: str) -> None:
        protocol = self.protocol
        if protocol is None or name not in protocol.task_config:
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
        editor = self._editor_for(config.task_type)
        if editor is not None:
            editor.apply(config)
        self._save()
        return config

    def _on_reset(self) -> None:
        self.reset_selected()

    def reset_selected(self) -> Optional[GridTaskConfig]:
        """Replace the selected task's settings with the type's defaults, and save."""
        protocol, config = self.protocol, self.selected_config()
        if protocol is None or config is None:
            return None
        fresh = type(config)(task_name=config.task_name)
        protocol.task_config[config.task_name] = fresh
        self._save()
        self.refresh()
        return fresh

    def _select(self, name: str) -> None:
        for i in range(self.task_list.count()):
            if self.task_list.item(i).data(Qt.UserRole) == name:
                self.task_list.setCurrentRow(i)
                return

    def _save(self) -> None:
        if self._experiment is None:
            return
        try:
            self._experiment.save(save_protocol=True)
        except Exception as e:  # noqa: BLE001 - a failed save is worth a line
            logging.warning(f"Could not save the grid protocol: {e}")
        self.protocol_changed.emit()
