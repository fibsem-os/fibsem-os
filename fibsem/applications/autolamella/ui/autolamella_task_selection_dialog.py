from typing import List, Optional, Tuple
import logging
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QDialogButtonBox,
    QWidget,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSizePolicy,
    QAbstractScrollArea,
    QLabel,
    QPushButton,
)
from PyQt5.QtCore import Qt

from fibsem.ui import stylesheets

class AutoLamellaTaskSelectionDialog(QDialog):
    def __init__(self, tasks: List[str], lamella: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tasks to Run")
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setMinimumSize(500, 500)

        self.lamella_list = QListWidget()
        self.tasks_list = QListWidget()
        self.tasks_list.setDragEnabled(True)
        self.tasks_list.setAcceptDrops(True)
        self.tasks_list.setDropIndicatorShown(True)
        self.tasks_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.tasks_list.setDefaultDropAction(Qt.MoveAction)

        for list_widget in (self.lamella_list, self.tasks_list):
            list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            list_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
            list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        lamella_group = QGroupBox("Lamella")
        lamella_layout = QVBoxLayout()
        lamella_group.setLayout(lamella_layout)

        for lamella_name in lamella:
            item = QListWidgetItem(lamella_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked)
            self.lamella_list.addItem(item)

        lamella_layout.addWidget(self.lamella_list)

        # Add Select All / Deselect All buttons for lamella
        lamella_buttons_layout = QHBoxLayout()
        self.select_all_lamella_button = QPushButton("Select All Lamella")
        self.deselect_all_lamella_button = QPushButton("Deselect All Lamella")

        self.select_all_lamella_button.clicked.connect(self._select_all_lamella)
        self.deselect_all_lamella_button.clicked.connect(self._deselect_all_lamella)

        lamella_buttons_layout.addWidget(self.select_all_lamella_button)
        lamella_buttons_layout.addWidget(self.deselect_all_lamella_button)
        lamella_layout.addLayout(lamella_buttons_layout)

        tasks_group = QGroupBox("Tasks")
        tasks_layout = QVBoxLayout()
        tasks_group.setLayout(tasks_layout)

        for task in tasks:
            item = QListWidgetItem(task)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            item.setCheckState(Qt.Checked)
            self.tasks_list.addItem(item)

        tasks_layout.addWidget(self.tasks_list)

        # Add Select All / Deselect All buttons for tasks
        tasks_buttons_layout = QHBoxLayout()
        self.select_all_tasks_button = QPushButton("Select All Tasks")
        self.deselect_all_tasks_button = QPushButton("Deselect All Tasks")

        self.select_all_tasks_button.clicked.connect(self._select_all_tasks)
        self.deselect_all_tasks_button.clicked.connect(self._deselect_all_tasks)

        tasks_buttons_layout.addWidget(self.select_all_tasks_button)
        tasks_buttons_layout.addWidget(self.deselect_all_tasks_button)
        tasks_layout.addLayout(tasks_buttons_layout)

        self.lamella_list.itemSelectionChanged.connect(
            lambda: self._on_item_selected(self.lamella_list, "Lamella")
        )
        self.tasks_list.itemSelectionChanged.connect(
            lambda: self._on_item_selected(self.tasks_list, "Task")
        )
        self.lamella_list.itemChanged.connect(lambda _: self._update_status_message())
        self.tasks_list.itemChanged.connect(lambda _: self._update_status_message())

        if self.lamella_list.count() > 0:
            self.lamella_list.setCurrentRow(0)

        if self.tasks_list.count() > 0:
            self.tasks_list.setCurrentRow(0)


        self.status_label = QLabel()
        self.status_label.setVisible(False)

        button_box = QDialogButtonBox()
        self.accept_button = QPushButton("Run Workflow")
        self.accept_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        self.accept_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        button_box.addButton(self.accept_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(self.cancel_button, QDialogButtonBox.RejectRole)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(lamella_group)
        main_layout.addWidget(tasks_group)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(button_box)

        self._update_status_message()

    def get_selected_tasks(self) -> List[str]:
        return [self.tasks_list.item(i).text() for i in range(self.tasks_list.count()) if self.tasks_list.item(i).checkState() == Qt.Checked] # type: ignore

    def get_selected_lamella(self) -> List[str]:
        return [self.lamella_list.item(i).text() for i in range(self.lamella_list.count()) if self.lamella_list.item(i).checkState() == Qt.Checked] # type: ignore

    def _on_item_selected(self, list_widget: QListWidget, label: str) -> None:
        selected_items = list_widget.selectedItems()
        if selected_items:
            logging.info(f"{label} selected: {selected_items[0].text()}")

    def _select_all_lamella(self) -> None:
        """Select all lamella in the lamella list."""
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item:
                item.setCheckState(Qt.Checked)  # type: ignore
        self._update_status_message()

    def _deselect_all_lamella(self) -> None:
        """Deselect all lamella in the lamella list."""
        for i in range(self.lamella_list.count()):
            item = self.lamella_list.item(i)
            if item:
                item.setCheckState(Qt.Unchecked)  # type: ignore
        self._update_status_message()

    def _select_all_tasks(self) -> None:
        """Select all tasks in the tasks list."""
        for i in range(self.tasks_list.count()):
            item = self.tasks_list.item(i)
            if item:
                item.setCheckState(Qt.Checked)  # type: ignore
        self._update_status_message()

    def _deselect_all_tasks(self) -> None:
        """Deselect all tasks in the tasks list."""
        for i in range(self.tasks_list.count()):
            item = self.tasks_list.item(i)
            if item:
                item.setCheckState(Qt.Unchecked)  # type: ignore
        self._update_status_message()

    def _update_status_message(self) -> None:
        lamella_checked_count = sum(
            1
            for i in range(self.lamella_list.count())
            if self.lamella_list.item(i).checkState() == Qt.Checked # type: ignore
        )
        tasks_checked_count = sum(
            1
            for i in range(self.tasks_list.count())
            if self.tasks_list.item(i).checkState() == Qt.Checked # type: ignore
        )

        if lamella_checked_count == 0 or tasks_checked_count == 0:
            self.status_label.setText("At least one lamella / task must be selected")
            self.status_label.setStyleSheet("color: orange;")
            self.status_label.setVisible(True)
            self.accept_button.setEnabled(False)
            return

        lamella_label = "lamella" if lamella_checked_count == 1 else "lamellae"
        task_label = "task" if tasks_checked_count == 1 else "tasks"
        self.status_label.setText(
            f"{lamella_checked_count} {lamella_label} and {tasks_checked_count} {task_label} selected"
        )
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setVisible(True)
        self.accept_button.setEnabled(True)


# TODO: support lamella status, last completed task
def open_task_selection_dialog(lamella_names: List[str],
                               task_names: List[str],
                               parent_ui: Optional[QWidget] = None) -> Tuple[bool, List[str], List[str]]:

    if len(lamella_names) == 0:
        return False, [], []

    task_dialog = AutoLamellaTaskSelectionDialog(tasks=task_names,
                                                 lamella=lamella_names,
                                                 parent=parent_ui)
    task_dialog.exec_()

    if task_dialog.result() == QDialog.Rejected:
        logging.info("Task selection dialog cancelled.")
        return False, [], []

    # get selected tasks and lamella
    selected_tasks = task_dialog.get_selected_tasks()
    selected_lamella = task_dialog.get_selected_lamella()
    return True, selected_tasks, selected_lamella
    

def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    accepted, selected_tasks, selected_lamella = open_task_selection_dialog(["Lamella 1", "Lamella 2"], ["Task 1", "Task 2", "Task 3"])
    print("Accepted:", accepted)
    print("Selected tasks:", selected_tasks)
    print("Selected lamella:", selected_lamella)
    app.exec_()
    

if __name__ == "__main__":
    main()
