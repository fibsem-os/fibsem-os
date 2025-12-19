import logging
from collections.abc import Callable
from enum import Enum
from typing import Optional, Union

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QWidget,
)

from fibsem.utils import format_value


class QFilePathLineEdit(QWidget):
    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.button_browse = QToolButton(self)
        self.button_browse.setText("...")
        self.button_browse.setMaximumWidth(80)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button_browse)

        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.button_browse.clicked.connect(self.browse_file)
        self.lineEdit.textChanged.connect(self.textChanged.emit)
        self.lineEdit.editingFinished.connect(self.editingFinished.emit)

    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.lineEdit.setText(selected_files[0])
                self.textChanged.emit(selected_files[0])
                self.editingFinished.emit()

    def text(self) -> str:
        return self.lineEdit.text()
    
    def setText(self, text: str) -> None:
        self.lineEdit.setText(text)


def _create_combobox_control(value: Union[str, int, float, Enum], 
                             items: list, 
                             units: Optional[str], 
                             format_fn: Optional[Callable] = None) -> QComboBox:
    """Create a QComboBox control for selecting from a list of items."""
    control = QComboBox()
    for item in items:
        if isinstance(item, (float, int)):
            item_str = format_value(val=item, unit=units, precision=1)
        elif isinstance(item, Enum):
            item_str = item.name # TODO: migrate to QEnumComboBox
        elif format_fn is not None:
            item_str = format_fn(item)
        else:
            item_str = str(item)
        control.addItem(item_str, item)

    if isinstance(value, tuple) and len(value) == 2:
        value = list(value)  # Convert tuple to list for easier handling

    # find the closest match to the current value (should only be used for numerical values)
    idx = control.findData(value)
    if idx == -1:
        # get the closest value
        closest_value = min(items, key=lambda x: abs(x - value))
        idx = control.findData(closest_value)
    if idx == -1:
        logging.debug(f"Warning: No matching item or nearest found for {items} with value {value}. Using first item.")
        idx = 0
    control.setCurrentIndex(idx)

    return control
