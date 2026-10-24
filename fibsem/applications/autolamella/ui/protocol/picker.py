"""What the protocol, new-experiment and load-experiment dialogs share: the
read-only summary of the chosen protocol, and the file pick that loads one."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PyQt5 import QtWidgets

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.ui import utils as fui

ERROR_INVALID_PROTOCOL_TITLE = "Invalid Protocol"
ERROR_INVALID_PROTOCOL_MSG = (
    "The selected protocol file is not valid. It may be corrupted, incorrectly formatted, or missing required fields.\n\n"
    "Error: {error}\n\n"
    "Please select a valid protocol file (*.yaml)."
)
_NONE = "No protocol loaded"


def pick_protocol(
    parent: QtWidgets.QWidget, start: str
) -> Optional[Tuple[str, AutoLamellaTaskProtocol]]:
    """Ask for a protocol file from *start* and load it. None when the operator
    cancels, or when the file does not load (after saying why)."""
    path = fui.open_existing_file_dialog(
        msg="Select a task protocol file (*.yaml)", path=start, parent=parent
    )
    if not path:
        return None
    try:
        protocol = AutoLamellaTaskProtocol.load(path)
    except Exception as e:
        logging.error(f"Failed to load protocol: {e}")
        QtWidgets.QMessageBox.critical(
            parent,
            ERROR_INVALID_PROTOCOL_TITLE,
            ERROR_INVALID_PROTOCOL_MSG.format(error=e),
        )
        return None
    logging.info(f"Protocol loaded from {path}")
    return path, protocol


class ProtocolSummary(QtWidgets.QWidget):
    """The chosen protocol's name, description, path and task count, read-only."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = self._field(_NONE)
        self.description = self._field(_NONE)
        self.path = self._field(_NONE)
        self.tasks = self._field("0")
        form.addRow("Name", self.name)
        form.addRow("Description", self.description)
        form.addRow("Path", self.path)
        form.addRow("Tasks", self.tasks)

    @staticmethod
    def _field(placeholder: str) -> QtWidgets.QLineEdit:
        field = QtWidgets.QLineEdit()
        field.setEnabled(False)
        field.setPlaceholderText(placeholder)
        return field

    def show_protocol(
        self, protocol: Optional[AutoLamellaTaskProtocol], path: str = ""
    ) -> None:
        """Fill the fields from *protocol*, or empty them (placeholders show)."""
        if protocol is None:
            for field in (self.name, self.description, self.path, self.tasks):
                field.setText("")
            return
        self.name.setText(protocol.name or "")
        self.description.setText(protocol.description or "")
        self.path.setText(path)
        self.path.setCursorPosition(0)
        self.tasks.setText(str(len(protocol.task_config)))
