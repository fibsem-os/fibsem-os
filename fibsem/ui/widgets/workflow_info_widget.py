from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
)

_LABEL_STYLE = "color: #a0a0a0; min-width: 80px; font-size: 11px;"
_SECTION_STYLE = (
    "font-size: 10px; font-weight: bold; color: #707070;"
    " padding: 4px 0px 2px 0px; letter-spacing: 0.5px;"
)
_LINE_EDIT_STYLE = """
QLineEdit {
    background: #2b2d31;
    color: #d6d6d6;
    border: 1px solid #3a3d42;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 11px;
}
QLineEdit:focus { border-color: #007ACC; }
"""


class WorkflowInfoWidget(QWidget):
    """Compact editor for workflow name, description, and run options."""

    name_changed = pyqtSignal(str)
    description_changed = pyqtSignal(str)
    options_changed = pyqtSignal(object)  # AutoLamellaWorkflowOptions

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #23252a;")

        self._config: Optional[AutoLamellaWorkflowConfig] = None
        self._options: Optional[AutoLamellaWorkflowOptions] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # ── name row ─────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_lbl = QLabel("Name")
        name_lbl.setStyleSheet(_LABEL_STYLE)
        name_row.addWidget(name_lbl)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Workflow name…")
        self.name_edit.setStyleSheet(_LINE_EDIT_STYLE)
        name_row.addWidget(self.name_edit)
        root.addLayout(name_row)

        # ── description row ───────────────────────────────────────────────
        desc_row = QHBoxLayout()
        desc_row.setSpacing(6)
        desc_lbl = QLabel("Description")
        desc_lbl.setStyleSheet(_LABEL_STYLE)
        desc_row.addWidget(desc_lbl)
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Workflow description…")
        self.desc_edit.setStyleSheet(_LINE_EDIT_STYLE)
        desc_row.addWidget(self.desc_edit)
        root.addLayout(desc_row)

        # ── options section ───────────────────────────────────────────────
        opt_lbl = QLabel("OPTIONS")
        opt_lbl.setStyleSheet(_SECTION_STYLE)
        root.addWidget(opt_lbl)

        self.turn_beams_off_cb = QCheckBox("Turn beams off after completion")
        self.turn_beams_off_cb.setStyleSheet("color: #d6d6d6; font-size: 11px;")
        root.addWidget(self.turn_beams_off_cb)

        # ── signals ───────────────────────────────────────────────────────
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.desc_edit.editingFinished.connect(self._on_desc_changed)
        self.turn_beams_off_cb.toggled.connect(self._on_options_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config: AutoLamellaWorkflowConfig) -> None:
        self._config = config
        self.name_edit.blockSignals(True)
        self.desc_edit.blockSignals(True)
        self.name_edit.setText(config.name or "")
        self.desc_edit.setText(config.description or "")
        self.name_edit.blockSignals(False)
        self.desc_edit.blockSignals(False)

    def set_options(self, options: AutoLamellaWorkflowOptions) -> None:
        self._options = options
        self.turn_beams_off_cb.blockSignals(True)
        self.turn_beams_off_cb.setChecked(options.turn_beams_off)
        self.turn_beams_off_cb.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        text = self.name_edit.text()
        if self._config is not None:
            self._config.name = text
        self.name_changed.emit(text)

    def _on_desc_changed(self) -> None:
        text = self.desc_edit.text()
        if self._config is not None:
            self._config.description = text
        self.description_changed.emit(text)

    def _on_options_changed(self) -> None:
        if self._options is not None:
            self._options.turn_beams_off = self.turn_beams_off_cb.isChecked()
        self.options_changed.emit(self._options)
