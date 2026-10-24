from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel


class TaskChecklist(TitledPanel):
    """A titled list of names to tick, every one ticked to start, with buttons
    that tick or untick them all."""

    changed = pyqtSignal()

    def __init__(
        self,
        title: str,
        select_label: str = "Select All",
        deselect_label: str = "Deselect All",
        parent: QWidget | None = None,
    ):
        super().__init__(title, collapsible=False, parent=parent)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        self.list_widget.itemChanged.connect(lambda _: self.changed.emit())
        layout.addWidget(self.list_widget)

        buttons = QHBoxLayout()
        self.pushButton_select_all = QPushButton(select_label)
        self.pushButton_deselect_all = QPushButton(deselect_label)
        self.pushButton_select_all.clicked.connect(lambda: self.set_all_checked(True))
        self.pushButton_deselect_all.clicked.connect(
            lambda: self.set_all_checked(False)
        )
        buttons.addWidget(self.pushButton_select_all)
        buttons.addWidget(self.pushButton_deselect_all)
        layout.addLayout(buttons)

        self.set_content(content)

    def set_items(self, names: list[str]) -> None:
        """List *names*, every one ticked."""
        self.list_widget.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.list_widget.addItem(item)

    def set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked  # type: ignore
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def checked(self) -> list[str]:
        """The ticked names, in list order."""
        items = (self.list_widget.item(i) for i in range(self.list_widget.count()))
        return [item.text() for item in items if item.checkState() == Qt.Checked]  # type: ignore


class ApplyDialog(QDialog):
    """A dialog that only its Apply button accepts: Enter neither accepts it nor
    presses a button, so a value being typed is never applied half-done."""

    def apply_buttons(self, apply_label: str) -> QDialogButtonBox:
        """Apply and Cancel, as `pushButton_apply` and `pushButton_cancel`."""
        box = QDialogButtonBox(self)
        self.pushButton_apply = QPushButton(apply_label)
        self.pushButton_apply.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_cancel = QPushButton("Cancel")
        for button in (self.pushButton_apply, self.pushButton_cancel):
            button.setAutoDefault(False)
            button.setDefault(False)
        box.addButton(self.pushButton_apply, QDialogButtonBox.AcceptRole)
        box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)
        self.pushButton_apply.clicked.connect(self.accept)
        self.pushButton_cancel.clicked.connect(self.reject)
        return box

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)
