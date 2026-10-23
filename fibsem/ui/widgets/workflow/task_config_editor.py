"""The task editor: one task config's forms, for the protocol's config or an
item's own copy.

Every edit is written by ``task_configs.set_task_config_values``, beside the
App Server's config edits, then recorded and saved once the host's pending
edits settle (``fibsem/ui/edit_recording.py``). The forms are handed copies:
each edits what it was given in place, so the config itself only changes
through that write.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.edit_recording import PendingEdits
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)
from fibsem.ui.widgets.task_config_form import AutoLamellaTaskParametersConfigWidget
from fibsem.workflows.task_configs import set_task_config_values

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import ReferenceImageParameters
    from fibsem.workflows.session import AppSession
    from fibsem.workflows.task_config import TaskConfig

__all__ = ["TaskConfigEditor", "milling_paths"]


def milling_paths(config: "TaskConfig") -> Dict[str, FibsemMillingTaskConfig]:
    """A task config's milling configs, by the path a write names each: one of
    a ``milling`` map as ``milling.<key>``, a field that is one by its name."""
    paths: Dict[str, FibsemMillingTaskConfig] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if isinstance(value, FibsemMillingTaskConfig):
            paths[f.name] = value
        elif f.name == "milling" and isinstance(value, dict):
            paths.update({f"milling.{key}": v for key, v in value.items()})
    return paths


class TaskConfigEditor(QWidget):
    """The task parameters and reference imaging forms, stacked, and the
    milling editor, which the host lays out. A task with more than one milling
    config picks the one shown under the forms.

    *level* is ``protocol`` or ``item``, as in ``task_configs``.
    """

    edited = pyqtSignal(str)  # the task name, after an edit changed its config

    def __init__(
        self,
        session: "AppSession",
        microscope: "FibsemMicroscope",
        edits: PendingEdits,
        level: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.session = session
        self._edits = edits
        self._level = level
        self._item: Any = None
        self._task_name: Optional[str] = None
        self._milling: Dict[str, FibsemMillingTaskConfig] = {}
        self._milling_path: Optional[str] = None

        self.parameters = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.reference_imaging = ReferenceImageParametersWidget(parent=self)
        self.milling = MillingTaskViewerWidget(
            microscope=microscope, milling_enabled=False, parent=self
        )
        self.milling_choice = QWidget(self)
        self._milling_combo = QComboBox()
        row = QHBoxLayout(self.milling_choice)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Milling"))
        row.addWidget(self._milling_combo, 1)
        self.milling_choice.setVisible(False)
        self._forms: List[QWidget] = [
            self.parameters,
            self.reference_imaging,
            self.milling,
        ]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.parameters)
        layout.addWidget(self.reference_imaging)
        layout.addWidget(self.milling_choice)

        self.parameters.parameter_changed.connect(
            lambda name, value: self.set_values({name: value})
        )
        self.reference_imaging.settings_changed.connect(self._on_reference_imaging)
        self.milling.settings_changed.connect(self._on_milling)
        self._milling_combo.currentIndexChanged.connect(self._on_milling_chosen)

    @property
    def task_name(self) -> Optional[str]:
        return self._task_name

    def set_microscope(self, microscope: "FibsemMicroscope") -> None:
        """After a reconnect: the forms keep the microscope they were built with
        otherwise. The milling editor's own children are not updated (FIB-530)."""
        self.milling.microscope = microscope

    def show_task(
        self, task_name: Optional[str], item: Any = None
    ) -> Optional["TaskConfig"]:
        """Show *task_name*'s config: the protocol's, or *item*'s at the item
        level. Returns it, or None (every form hidden) when there is none."""
        self._edits.flush()
        self._task_name = task_name
        self._item = item
        config = self._config()
        if config is None:
            self._milling = {}
            self._milling_path = None
            self.milling.clear()
            for widget in (*self._forms, self.milling_choice):
                widget.setVisible(False)
            return None
        self._show_config(config)
        return config

    def _show_config(self, config: "TaskConfig") -> None:
        """Fill the forms from *config*, and show the ones it has."""
        self.parameters.set_task_config(copy.deepcopy(config))
        self.parameters.setVisible(True)
        imaging = getattr(config, "reference_imaging", None)
        if imaging is not None:
            self.reference_imaging.update_from_settings(copy.deepcopy(imaging))
        self.reference_imaging.setVisible(imaging is not None)

        self._milling = milling_paths(config)
        self._milling_combo.blockSignals(True)
        self._milling_combo.clear()
        for path, milling in self._milling.items():
            self._milling_combo.addItem(milling.name or path, path)
        self._milling_combo.blockSignals(False)
        self.milling_choice.setVisible(len(self._milling) > 1)
        self._show_milling(next(iter(self._milling), None))

    def _show_milling(self, path: Optional[str]) -> None:
        self._milling_path = path
        if path is None:
            self.milling.clear()
            self.milling.setVisible(False)
            return
        # the config's own, now: an edit replaces the one it was shown from
        self.milling.set_config(copy.deepcopy(milling_paths(self._config())[path]))
        self.milling.setVisible(True)

    def add_form(self, widget: QWidget) -> None:
        """A form of the host's own, laid under the task parameters: the Lamella
        tab's spot-burn coordinates, which are placed on its canvas."""
        self.layout().insertWidget(1, widget)

    def set_editable(self, editable: bool) -> None:
        for widget in (*self._forms, self.milling_choice):
            widget.setEnabled(editable)

    def set_values(self, values: Dict[str, Any]) -> None:
        """Write *values* to the shown config (``set_task_config_values``),
        recorded and saved once the edits settle."""
        if self._task_name is None:
            return
        experiment = self.session.experiment
        result = set_task_config_values(
            experiment,
            self._level,
            self._item.name if self._item is not None else "",
            self._task_name,
            values,
            self._edits,
        )
        if not result["applied"]:
            logging.warning(f"task config edit not applied: {result.get('error')}")
            return
        if result["changes"]:
            self._edits.save_later(experiment, self._level)
            self.edited.emit(self._task_name)

    def _config(self) -> Optional["TaskConfig"]:
        if self._task_name is None:
            return None
        if self._level == "item":
            if self._item is None:
                return None
            return self._item.task_config.get(self._task_name)
        protocol = self.session.experiment.task_protocol
        return protocol.task_config.get(self._task_name)

    def _on_reference_imaging(self, settings: "ReferenceImageParameters") -> None:
        self.set_values({"reference_imaging": settings})

    def _on_milling(self, config: FibsemMillingTaskConfig) -> None:
        if self._milling_path is not None:
            self.set_values({self._milling_path: config})

    def _on_milling_chosen(self, index: int) -> None:
        self._edits.flush()
        self._show_milling(self._milling_combo.itemData(index))

    def hideEvent(self, event) -> None:
        """Flush when the forms go away: a tab switch, or the window closing."""
        self._edits.flush()
        super().hideEvent(event)
