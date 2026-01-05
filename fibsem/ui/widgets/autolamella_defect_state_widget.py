"""Widget helpers for editing the :class:`DefectState` associated with a lamella."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from fibsem.applications.autolamella.structures import DefectState
from fibsem.ui import stylesheets


class AutoLamellaDefectStateWidget(QtWidgets.QGroupBox):
    """Provide simple controls to edit a lamella's :class:`DefectState`."""

    defect_state_changed = QtCore.pyqtSignal(DefectState)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._defect_state: DefectState = DefectState()
        self._updating: bool = False

        self.setTitle("Defect")
        self._setup_ui()
        self._connect_signals()
        self._update_inputs()
        self.setEnabled(True)

    # --------------------------------------------------------------------- #
    # UI helpers
    def _setup_ui(self) -> None:
        layout = QtWidgets.QGridLayout(self)

        self.checkbox_has_defect = QtWidgets.QCheckBox("Has Defect")
        self.checkbox_has_defect.setStyleSheet(stylesheets.CHECKBOX_STYLE)

        self.checkbox_requires_rework = QtWidgets.QCheckBox("Requires Rework")
        self.checkbox_requires_rework.setStyleSheet(stylesheets.CHECKBOX_STYLE)

        self.line_description = QtWidgets.QLineEdit()
        self.line_description.setPlaceholderText("Enter defect reason or notes")
        layout.addWidget(self.checkbox_has_defect, 0, 0)
        layout.addWidget(self.checkbox_requires_rework, 0, 1)
        layout.addWidget(self.line_description, 1, 0, 1, 2)

        layout.setColumnStretch(1, 1)

    def _connect_signals(self) -> None:
        self.checkbox_has_defect.toggled.connect(self._handle_input_changed)
        self.checkbox_requires_rework.toggled.connect(self._handle_input_changed)
        self.line_description.editingFinished.connect(self._handle_input_changed)

    # --------------------------------------------------------------------- #
    # Public API
    def set_defect_state(self, defect_state: Optional[DefectState]) -> None:
        """Update the widget to reflect a new :class:`DefectState` instance."""
        self._defect_state = defect_state or DefectState()
        self._update_inputs()

    def get_defect_state(self) -> DefectState:
        """Return the currently bound :class:`DefectState`."""
        return self._defect_state

    # --------------------------------------------------------------------- #
    # Internal helpers
    def _handle_input_changed(self) -> None:
        if self._updating or self._defect_state is None:
            return

        has_defect = self.checkbox_has_defect.isChecked()
        requires_rework = bool(self.checkbox_requires_rework.isChecked()) if has_defect else False
        description = self.line_description.text().strip() if has_defect else ""

        self._defect_state.has_defect = has_defect
        self._defect_state.requires_rework = requires_rework
        self._defect_state.description = description

        if has_defect:
            self._defect_state.updated_at = datetime.timestamp(datetime.now())
        else:
            self._defect_state.updated_at = None

        self._update_inputs()
        self.defect_state_changed.emit(self._defect_state)

    def _update_inputs(self) -> None:
        self._updating = True

        state = self._defect_state or DefectState()
        self.checkbox_has_defect.setChecked(state.has_defect)
        self.checkbox_requires_rework.setChecked(state.requires_rework)
        self.checkbox_requires_rework.setEnabled(state.has_defect)
        self.line_description.setEnabled(state.has_defect)
        self.line_description.setText(state.description or "")

        self._updating = False



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = AutoLamellaDefectStateWidget()
    widget.set_defect_state(DefectState())

    def _on_defect_state_changed(defect_state: DefectState) -> None:
        print("Defect state changed:", defect_state)

    widget.defect_state_changed.connect(_on_defect_state_changed)

    widget.show()
    sys.exit(app.exec_())
