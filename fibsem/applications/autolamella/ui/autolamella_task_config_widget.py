"""The task-parameters form.

Renders an AutoLamellaTaskConfig's own fields -- the ones past the base class --
as a grid of controls, using the shared builder that the pattern, strategy and
milling-settings forms use (FIB-526).

This form used to read raw ``dataclasses.fields(...).metadata`` and understood
four keys, so most of the vocabulary was simply unavailable to it: an int field
got the full 32-bit range because nothing could say otherwise, floats got Qt's
defaults, and there was no way to mark a field advanced or hidden. Going through
``config.field_specs`` means a task config now declares those the same way a
pattern does.

The container stays separate from the milling ones on purpose: it has milling
sub-configs and reference imaging to show, and no type selector.
"""

from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.ui.widgets.custom_widgets import TitledPanel, align_form
from fibsem.ui.widgets.form_builder import Control, FormDefaults, build_control

# What this form falls back to for what a field does not declare. These are the
# bounds and precision it already had hardcoded; passing them keeps the form
# looking the same, so declaring real `limits` is a visible change rather than a
# side effect of moving onto the shared builder.
TASK_FORM_DEFAULTS = FormDefaults(
    float_range=(-1e10, 1e10),
    int_range=(-2147483648, 2147483647),
    step=1.0,
    decimals=2,
)


@dataclass
class _Row:
    """One built form row."""

    label: QLabel
    control: Control
    field: str
    advanced: bool


def build_parameter_rows(
    config: AutoLamellaTaskConfig, grid: QGridLayout
) -> List[_Row]:
    """Fill *grid* with a control per configurable parameter, and return the rows.

    Shared by both containers in this module, which each carried their own copy
    of this loop -- and had already drifted: only one of them honoured `label`.
    """
    specs = config.field_specs
    rows: List[_Row] = []

    for name in config.parameters:
        spec = specs[name]
        if spec.hidden:
            continue

        control = build_control(
            spec, getattr(config, name), defaults=TASK_FORM_DEFAULTS
        )
        if control is None:
            continue

        label = QLabel(spec.label or name.replace("_", " ").title())
        if spec.doc:
            label.setToolTip(spec.doc)

        row_index = len(rows)
        grid.addWidget(label, row_index, 0)
        grid.addWidget(control.widget, row_index, 1)
        rows.append(
            _Row(
                label=label,
                control=control,
                field=name,
                advanced=spec.advanced,
            )
        )

    return rows


class AutoLamellaTaskParametersConfigWidget(QWidget):
    """The task parameters on their own, in a titled panel.

    The milling and reference imaging sections are separate widgets, laid out
    beside this one by the Protocol and Lamella tabs.
    """

    config_changed = pyqtSignal(AutoLamellaTaskConfig)
    parameter_changed = pyqtSignal(str, object)  # field name, new value

    def __init__(
        self,
        task_config: Optional[AutoLamellaTaskConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.task_config = task_config
        self._rows: List[_Row] = []
        self._advanced_visible = False

        self._setup_ui()
        if self.task_config:
            self._update_from_config()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

        self.params_widget = QWidget()
        self.grid_layout = QGridLayout(self.params_widget)
        align_form(self.grid_layout)
        self.params_panel = TitledPanel("Task Parameters", content=self.params_widget)
        self.params_panel._btn_collapse.setChecked(True)

        self.main_layout.addWidget(self.params_panel)

    def set_task_config(self, task_config: Optional[AutoLamellaTaskConfig]):
        """Set the task configuration to edit."""
        self.task_config = task_config
        self._update_from_config()

    def _update_from_config(self):
        if not self.task_config:
            return

        self._clear_form()

        if not self.task_config.parameters:
            self.hide()
            return

        self._rows = build_parameter_rows(self.task_config, self.grid_layout)
        for row in self._rows:
            # A read-only row displays a value it cannot reconstruct; wiring it
            # up would let a focus-out write the displayed text back.
            if row.control.editable:
                row.control.connect(
                    lambda name=row.field: self._on_parameter_changed(name)
                )
        self._update_visibility()
        self.show()

    def _update_visibility(self) -> None:
        for row in self._rows:
            visible = (not row.advanced) or self._advanced_visible
            row.label.setVisible(visible)
            row.control.widget.setVisible(visible)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()

    def _on_parameter_changed(self, field_name: str):
        row = next((r for r in self._rows if r.field == field_name), None)
        if row is None or not row.control.editable:
            return
        value = row.control.read()
        setattr(self.task_config, field_name, value)
        self.parameter_changed.emit(field_name, value)

    def _clear_form(self):
        """Clear the grid. Rows are dropped first -- see the sibling container."""
        self._rows = []
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child and child.widget():
                child.widget().setParent(None)

    def get_task_config(self) -> Optional[AutoLamellaTaskConfig]:
        """Get the current task configuration."""
        return self.task_config
