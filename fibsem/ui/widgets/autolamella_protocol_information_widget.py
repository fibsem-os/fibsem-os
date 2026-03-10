from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFormLayout, QLineEdit, QWidget

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol


class ProtocolInformationWidget(QWidget):
    """Displays and edits protocol name, description, and version."""

    field_changed = pyqtSignal(str, str)  # (field_name, new_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lineEdit_name = QLineEdit()
        self.lineEdit_description = QLineEdit()
        self.lineEdit_version = QLineEdit()

        layout.addRow("Name", self.lineEdit_name)
        layout.addRow("Description", self.lineEdit_description)
        layout.addRow("Version", self.lineEdit_version)

    def _connect_signals(self):
        self.lineEdit_name.editingFinished.connect(
            lambda: self.field_changed.emit("name", self.lineEdit_name.text())
        )
        self.lineEdit_description.editingFinished.connect(
            lambda: self.field_changed.emit("description", self.lineEdit_description.text())
        )
        self.lineEdit_version.editingFinished.connect(
            lambda: self.field_changed.emit("version", self.lineEdit_version.text())
        )

    def update_from_protocol(self, protocol: AutoLamellaTaskProtocol) -> None:
        """Populate fields from protocol, suppressing signals."""
        for widget, value in [
            (self.lineEdit_name, protocol.name or ""),
            (self.lineEdit_description, protocol.description or ""),
            (self.lineEdit_version, protocol.version or ""),
        ]:
            widget.blockSignals(True)
            widget.setText(value)
            widget.blockSignals(False)
