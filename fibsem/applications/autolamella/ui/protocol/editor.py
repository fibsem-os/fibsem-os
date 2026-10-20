"""The task editor: one task config's forms, for the protocol's config on the
Protocol tab and for a lamella's own copy on the Lamella tab.

Every edit is written by ``config_edits.set_task_config_values``, beside the
App Server's config edits, then recorded and saved once the host's pending
edits settle (``edit_recording``). The forms are handed copies: each edits what
it was given in place, so the config itself only changes through that write.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.applications.autolamella.config_edits import set_task_config_values
from fibsem.applications.autolamella.ui.protocol.edit_recording import PendingEdits
from fibsem.applications.autolamella.ui.protocol.forms import (
    AutoLamellaFluorescenceAcquisitionTaskConfigWidget,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
)
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.reference_image_parameters_widget import (
    ReferenceImageParametersWidget,
)
from fibsem.ui.widgets.task_config_form import AutoLamellaTaskParametersConfigWidget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskConfig,
        Lamella,
    )
    from fibsem.microscope import FibsemMicroscope
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.structures import ReferenceImageParameters
    from fibsem.workflows.session import AppSession

# The fluorescence form's fields: what it sends is a whole config, and only
# these are its to write.
_FLUORESCENCE_FIELDS = (
    "orientation",
    "zparams",
    "channel_settings",
    "autofocus_settings",
)


class TaskConfigEditor(QWidget):
    """The task parameters, reference imaging and fluorescence forms, stacked,
    and the milling editor, which the host lays out: beside the forms on the
    Protocol tab, under them on the Lamella tab.

    *level* is ``protocol`` or ``item``, as in ``config_edits``.
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
        self._lamella: Optional["Lamella"] = None
        self._task_name: Optional[str] = None
        self._milling_key: Optional[str] = None

        self.parameters = AutoLamellaTaskParametersConfigWidget(parent=self)
        self.reference_imaging = ReferenceImageParametersWidget(parent=self)
        self.fluorescence = AutoLamellaFluorescenceAcquisitionTaskConfigWidget(
            microscope=microscope, config=None, parent=self
        )
        self.milling = MillingTaskViewerWidget(
            microscope=microscope, milling_enabled=False, parent=self
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.parameters)
        layout.addWidget(self.reference_imaging)
        layout.addWidget(self.fluorescence)

        self.parameters.parameter_changed.connect(
            lambda name, value: self.set_values({name: value})
        )
        self.reference_imaging.settings_changed.connect(self._on_reference_imaging)
        self.milling.settings_changed.connect(self._on_milling)
        self.fluorescence.settings_changed.connect(self._on_fluorescence)

    @property
    def task_name(self) -> Optional[str]:
        return self._task_name

    def set_microscope(self, microscope: "FibsemMicroscope") -> None:
        """After a reconnect: the forms keep the microscope they were built with
        otherwise, and the fluorescence channels its filters (FIB-525). The
        milling editor's own children are not updated (FIB-530)."""
        self.milling.microscope = microscope
        self.fluorescence.set_microscope(microscope)

    def show_task(
        self, task_name: Optional[str], lamella: Optional["Lamella"] = None
    ) -> Optional["AutoLamellaTaskConfig"]:
        """Show *task_name*'s config: the protocol's, or *lamella*'s at the item
        level. Returns it, or None (every form hidden) when there is none."""
        self._edits.flush()
        self._task_name = task_name
        self._lamella = lamella
        config = self._config()
        if config is None:
            self._milling_key = None
            self.milling.clear()
            for widget in (
                self.parameters,
                self.reference_imaging,
                self.fluorescence,
                self.milling,
            ):
                widget.setVisible(False)
            return None

        self.parameters.set_task_config(copy.deepcopy(config))
        self.reference_imaging.update_from_settings(
            copy.deepcopy(config.reference_imaging)
        )
        if config.milling:
            self._milling_key = next(iter(config.milling))
            self.milling.set_config(copy.deepcopy(config.milling[self._milling_key]))
            self.milling.setVisible(True)
        else:
            self._milling_key = None
            self.milling.clear()
            self.milling.setVisible(False)

        fluorescence = isinstance(config, AcquireFluorescenceImageConfig)
        self.fluorescence.setVisible(fluorescence)
        self.parameters.setVisible(not fluorescence)
        self.reference_imaging.setVisible(not fluorescence)
        if fluorescence:
            self.fluorescence.set_task_config(config)
        return config

    def add_form(self, widget: QWidget) -> None:
        """A form of the host's own, laid under the task parameters: the Lamella
        tab's spot-burn coordinates, which are placed on its canvas."""
        self.layout().insertWidget(1, widget)

    def set_editable(self, editable: bool) -> None:
        for widget in (
            self.parameters,
            self.reference_imaging,
            self.fluorescence,
            self.milling,
        ):
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
            self._lamella.name if self._lamella is not None else "",
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

    def _config(self) -> Optional["AutoLamellaTaskConfig"]:
        if self._task_name is None:
            return None
        if self._level == "item":
            if self._lamella is None:
                return None
            return self._lamella.task_config.get(self._task_name)
        protocol = self.session.experiment.task_protocol
        return protocol.task_config.get(self._task_name)

    def _on_reference_imaging(self, settings: "ReferenceImageParameters") -> None:
        self.set_values({"reference_imaging": settings})

    def _on_milling(self, config: "FibsemMillingTaskConfig") -> None:
        if self._milling_key is not None:
            self.set_values({f"milling.{self._milling_key}": config})

    def _on_fluorescence(self, config: AcquireFluorescenceImageConfig) -> None:
        self.set_values({name: getattr(config, name) for name in _FLUORESCENCE_FIELDS})

    def hideEvent(self, event) -> None:
        """Flush when the forms go away: a tab switch, or the window closing."""
        self._edits.flush()
        super().hideEvent(event)
