from __future__ import annotations

from PyQt5.QtCore import Qt
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

class ApplyLamellaConfigDialog(QDialog):
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

        # --- Target lamella selection ---
        lamella_content = QWidget()
        lamella_layout = QVBoxLayout(lamella_content)
        lamella_layout.setContentsMargins(0, 0, 0, 0)

        self.lamella_list = QListWidget()
        self.lamella_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lamella_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        for name in self._other_lamella_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.lamella_list.addItem(item)

        lamella_buttons_layout = QHBoxLayout()
        self.pushButton_select_all_lamella = QPushButton("Select All")
        self.pushButton_deselect_all_lamella = QPushButton("Deselect All")
        self.pushButton_select_all_lamella.clicked.connect(
            lambda: self._set_all_check_state(self.lamella_list, Qt.Checked))  # type: ignore
        self.pushButton_deselect_all_lamella.clicked.connect(
            lambda: self._set_all_check_state(self.lamella_list, Qt.Unchecked))  # type: ignore
        lamella_buttons_layout.addWidget(self.pushButton_select_all_lamella)
        lamella_buttons_layout.addWidget(self.pushButton_deselect_all_lamella)

        lamella_layout.addWidget(self.lamella_list)
        lamella_layout.addLayout(lamella_buttons_layout)

        self.lamella_group = TitledPanel("Target Lamella", content=lamella_content, collapsible=False)

        # --- Task selection ---
        task_content = QWidget()
        task_layout = QVBoxLayout(task_content)
        task_layout.setContentsMargins(0, 0, 0, 0)

        self.task_list = QListWidget()
        self.task_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore

        for task_name in self._task_names:
            item = QListWidgetItem(task_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)  # type: ignore
            item.setCheckState(Qt.Checked)  # type: ignore
            self.task_list.addItem(item)

        task_buttons_layout = QHBoxLayout()
        self.pushButton_select_all_tasks = QPushButton("Select All")
        self.pushButton_deselect_all_tasks = QPushButton("Deselect All")
        self.pushButton_select_all_tasks.clicked.connect(
            lambda: self._set_all_check_state(self.task_list, Qt.Checked))  # type: ignore
        self.pushButton_deselect_all_tasks.clicked.connect(
            lambda: self._set_all_check_state(self.task_list, Qt.Unchecked))  # type: ignore
        task_buttons_layout.addWidget(self.pushButton_select_all_tasks)
        task_buttons_layout.addWidget(self.pushButton_deselect_all_tasks)

        task_layout.addWidget(self.task_list)
        task_layout.addLayout(task_buttons_layout)

        self.task_group = TitledPanel("Tasks to Apply", content=task_content, collapsible=False)

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

        # --- Buttons ---
        self.button_box = QDialogButtonBox(self)
        self.pushButton_apply = QPushButton("Apply")
        self.pushButton_apply.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_apply.setAutoDefault(False)
        self.pushButton_cancel = QPushButton("Cancel")
        self.pushButton_cancel.setAutoDefault(False)

        self.button_box.addButton(self.pushButton_apply, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)
        self.pushButton_apply.clicked.connect(self.accept)
        self.pushButton_cancel.clicked.connect(self.reject)
        self.pushButton_apply.setDefault(False)
        self.pushButton_cancel.setDefault(False)

        # Update info when selections change (must be after buttons are created)
        self.lamella_list.itemChanged.connect(lambda _: self._update_info_label())
        self.task_list.itemChanged.connect(lambda _: self._update_info_label())
        self.checkbox_update_base_protocol.stateChanged.connect(lambda _: self._update_info_label())
        self._update_info_label()

    def _setup_layout(self):
        """Setup the dialog layout."""
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_description)
        main_layout.addWidget(self.lamella_group)
        main_layout.addWidget(self.task_group)
        main_layout.addWidget(self.checkbox_update_base_protocol)
        main_layout.addWidget(self.label_info)
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def _set_all_check_state(self, list_widget: QListWidget, state):
        """Set the check state for all items in a list widget."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item:
                item.setCheckState(state)

    def get_selected_lamella_names(self) -> list[str]:
        """Get list of selected target lamella names."""
        selected = []
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item and item.checkState() == Qt.Checked:  # type: ignore
                selected.append(item.text())
        return selected

    def get_selected_tasks(self) -> list[str]:
        """Get list of selected task names."""
        selected = []
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            if item and item.checkState() == Qt.Checked:  # type: ignore
                selected.append(item.text())
        return selected

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
            self.label_info.setText(f"Please select at least one {' and '.join(missing)}.")
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

    def keyPressEvent(self, event):
        """Prevent Enter/Return from accepting the dialog."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)