from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
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


class ApplyLamellaConfigDialog(ApplyDialog):
    """Dialog for applying a lamella's task configuration to other lamella."""

    def __init__(
        self,
        source_name: str,
        other_lamella_names: list[str],
        task_names: list[str],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.source_name = source_name

        self.setWindowTitle(f"Apply Config from '{source_name}'")
        self.setModal(True)
        self.setMinimumWidth(450)

        self._other_lamella_names = other_lamella_names
        self._task_names = task_names

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        """Create the dialog widgets."""
        # Description
        self.label_description = QLabel(
            f"Apply task configurations from <b>{self.source_name}</b> "
            f"to other lamella. Existing milling pattern positions will be preserved."
        )
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet("font-style: italic; margin-bottom: 10px;")

        self.lamella_checklist = TaskChecklist("Target Lamella")
        self.lamella_checklist.set_items(self._other_lamella_names)
        self.task_checklist = TaskChecklist("Tasks to Apply")
        self.task_checklist.set_items(self._task_names)

        # --- Base protocol checkbox ---
        self.checkbox_update_base_protocol = QCheckBox("Also update the base protocol")
        self.checkbox_update_base_protocol.setToolTip(
            "When enabled, also updates the base protocol's task configurations "
            "so that new lamella created in the future will use these settings."
        )

        # --- Info label ---
        self.label_info = QLabel()
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.label_info.setWordWrap(True)

        self.button_box = self.apply_buttons("Apply")

        # Update info when selections change (must be after buttons are created)
        self.lamella_checklist.changed.connect(self._update_info_label)
        self.task_checklist.changed.connect(self._update_info_label)
        self.checkbox_update_base_protocol.stateChanged.connect(
            lambda _: self._update_info_label()
        )
        self._update_info_label()

    def _setup_layout(self):
        """Setup the dialog layout."""
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_description)
        main_layout.addWidget(self.lamella_checklist)
        main_layout.addWidget(self.task_checklist)
        main_layout.addWidget(self.checkbox_update_base_protocol)
        main_layout.addWidget(self.label_info)
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def get_selected_lamella_names(self) -> list[str]:
        """Get list of selected target lamella names."""
        return self.lamella_checklist.checked()

    def get_selected_tasks(self) -> list[str]:
        """Get list of selected task names."""
        return self.task_checklist.checked()

    def get_update_base_protocol(self) -> bool:
        """Whether to also update the base protocol."""
        return self.checkbox_update_base_protocol.isChecked()

    def _update_info_label(self):
        """Update the info label with summary of what will be applied."""
        selected_lamella_names = self.get_selected_lamella_names()
        selected_tasks = self.get_selected_tasks()

        if not selected_lamella_names or not selected_tasks:
            missing = []
            if not selected_lamella_names:
                missing.append("target lamella")
            if not selected_tasks:
                missing.append("tasks")
            self.label_info.setText(
                f"Please select at least one {' and '.join(missing)}."
            )
            self.label_info.setStyleSheet("color: orange; font-style: italic;")
            self.pushButton_apply.setEnabled(False)
            return

        parts = []
        task_list = ", ".join(f"'{t}'" for t in selected_tasks)
        lamella_list = ", ".join(selected_lamella_names)
        parts.append(
            f"{len(selected_tasks)} task(s) ({task_list}) will be applied to "
            f"{len(selected_lamella_names)} lamella ({lamella_list})."
        )
        if self.get_update_base_protocol():
            parts.append("The base protocol will also be updated.")

        self.label_info.setText(" ".join(parts))
        self.label_info.setStyleSheet("color: gray; font-style: italic;")
        self.pushButton_apply.setEnabled(True)
