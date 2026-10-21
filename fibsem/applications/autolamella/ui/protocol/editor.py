"""AutoLamella's task editor: the shared one, with the fluorescence form for a
fluorescence acquisition task, for the protocol's config on the Protocol tab
and for a lamella's own copy on the Lamella tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.ui.protocol.forms import (
    AutoLamellaFluorescenceAcquisitionTaskConfigWidget,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import (
    AcquireFluorescenceImageConfig,
)
from fibsem.ui.edit_recording import PendingEdits
from fibsem.ui.widgets.workflow.task_config_editor import TaskConfigEditor

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.workflows.session import AppSession
    from fibsem.workflows.task_config import TaskConfig

# The fluorescence form's fields: what it sends is a whole config, and only
# these are its to write.
_FLUORESCENCE_FIELDS = (
    "orientation",
    "zparams",
    "channel_settings",
    "autofocus_settings",
)


class AutoLamellaTaskConfigEditor(TaskConfigEditor):
    """:class:`TaskConfigEditor`, with the fluorescence form in place of the
    task parameters and reference imaging for a fluorescence task."""

    def __init__(
        self,
        session: "AppSession",
        microscope: "FibsemMicroscope",
        edits: PendingEdits,
        level: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(session, microscope, edits, level, parent)
        self.fluorescence = AutoLamellaFluorescenceAcquisitionTaskConfigWidget(
            microscope=microscope, config=None, parent=self
        )
        self.layout().insertWidget(2, self.fluorescence)
        self._forms.append(self.fluorescence)
        self.fluorescence.settings_changed.connect(self._on_fluorescence)

    def set_microscope(self, microscope: "FibsemMicroscope") -> None:
        """And the fluorescence channels its filters (FIB-525)."""
        super().set_microscope(microscope)
        self.fluorescence.set_microscope(microscope)

    def _show_config(self, config: "TaskConfig") -> None:
        super()._show_config(config)
        fluorescence = isinstance(config, AcquireFluorescenceImageConfig)
        self.fluorescence.setVisible(fluorescence)
        if fluorescence:
            self.parameters.setVisible(False)
            self.reference_imaging.setVisible(False)
            self.fluorescence.set_task_config(config)

    def _on_fluorescence(self, config: AcquireFluorescenceImageConfig) -> None:
        self.set_values({name: getattr(config, name) for name in _FLUORESCENCE_FIELDS})
