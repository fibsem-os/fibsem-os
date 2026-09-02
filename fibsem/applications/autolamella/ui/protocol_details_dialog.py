"""The protocol's identity on the Protocol tab: one line, and a pencil.

Name, description and version are set when the experiment is created and rarely
touched after; the form they used to occupy took the top of the only column that
has to fit two task lists. The line says which protocol is loaded, which is the
one thing that panel earned; the dialog behind the pencil does the rest.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.ui.tokens import NEUTRAL_200, TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import ElidedLabel, IconToolButton

_FIELDS = ("name", "description", "version")


class ProtocolHeaderWidget(QWidget):
    """`AutoLamella Protocol · v1.0`, and a pencil."""

    edit_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 0)
        layout.setSpacing(6)
        self.label = ElidedLabel()
        self.label.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        layout.addWidget(self.label, 1)
        # A plain label: an eliding one beside a stretched name has no width left.
        self.version_label = QLabel()
        self.version_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.version_label)
        self.btn_edit = IconToolButton(
            icon="mdi:pencil-outline",
            tooltip="Edit the protocol's name, description and version",
            size=24,
        )
        self.btn_edit.clicked.connect(self.edit_clicked)
        layout.addWidget(self.btn_edit)

    def update_from_protocol(self, protocol: AutoLamellaTaskProtocol) -> None:
        self.label.setText(protocol.name or "Untitled protocol")
        version = protocol.version or ""
        self.version_label.setText(f"v{version}" if version else "")
        self.setToolTip(protocol.description or "")


class ProtocolDetailsDialog(QDialog):
    """Name, description and version, for the pencil."""

    def __init__(
        self, protocol: AutoLamellaTaskProtocol, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Protocol details")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edits: Dict[str, QLineEdit] = {}
        for field in _FIELDS:
            edit = QLineEdit(getattr(protocol, field, "") or "")
            form.addRow(field.capitalize(), edit)
            self.edits[field] = edit
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, str]:
        return {field: edit.text().strip() for field, edit in self.edits.items()}
