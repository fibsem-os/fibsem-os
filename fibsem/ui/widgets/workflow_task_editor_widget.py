from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription

_BTN_SIZE = QSize(26, 26)

_SECTION_LABEL_STYLE = "font-weight: bold; font-size: 11px; color: #a0a0a0; padding: 2px 0px;"
_TASK_NAME_STYLE = "font-size: 13px; font-weight: bold; color: #e0e0e0;"

_LIST_STYLE = """
    QListWidget {
        background: #2b2d31;
        border: 1px solid #3a3d42;
        border-radius: 4px;
        outline: none;
    }
    QListWidget::item {
        background: #2b2d31;
        border-bottom: 1px solid #3a3d42;
    }
    QListWidget::item:alternate {
        background: #303338;
    }
    QListWidget::item:selected {
        background: transparent;
    }
"""

_APPLY_BTN_STYLE = """
QPushButton {
    background: #2196f3;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 16px;
    font-weight: bold;
}
QPushButton:hover { background: #1976d2; }
QPushButton:pressed { background: #1565c0; }
"""

_CANCEL_BTN_STYLE = """
QPushButton {
    background: #3a3d42;
    color: #c0c0c0;
    border: none;
    border-radius: 4px;
    padding: 5px 16px;
}
QPushButton:hover { background: #4a4d52; }
QPushButton:pressed { background: #2a2d32; }
"""


class _RequirementRowWidget(QWidget):
    """Simple checkbox row for a single available task."""

    def __init__(self, task_name: str, checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        layout.addWidget(self.checkbox)

        self.name_label = QLabel(task_name)
        layout.addWidget(self.name_label, 1)


class WorkflowTaskEditorWidget(QWidget):
    """
    Inline editor for a single AutoLamellaTaskDescription.

    Pass ``available_tasks`` as the names of all other tasks in the workflow
    so the user can select which ones this task depends on.

    Signals
    -------
    apply_clicked  : emits a *copy* of the task with the edited values applied.
    cancel_clicked : emits when the user clicks Cancel.
    """

    apply_clicked = pyqtSignal(object)   # AutoLamellaTaskDescription
    cancel_clicked = pyqtSignal()

    def __init__(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._available = [t for t in (available_tasks or []) if t != task.name]

        self.setStyleSheet("background: #2b2d31;")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ── header ──────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(QIconifyIcon("mdi:pencil-box-outline", color="#a0a0a0").pixmap(18, 18))
        header_row.addWidget(icon_lbl)

        title = QLabel("Edit Task")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #c0c0c0;")
        header_row.addWidget(title, 1)
        root.addLayout(header_row)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.HLine)
        sep0.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep0)

        # ── task name ───────────────────────────────────────────────────
        self._name_label = QLabel(task.name)
        self._name_label.setStyleSheet(_TASK_NAME_STYLE)
        root.addWidget(self._name_label)

        # ── properties ──────────────────────────────────────────────────
        props_lbl = QLabel("PROPERTIES")
        props_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(props_lbl)

        self._optional_cb = QCheckBox("Optional  (uncheck to make required)")
        self._optional_cb.setChecked(not task.required)
        root.addWidget(self._optional_cb)

        # ── requirements ────────────────────────────────────────────────
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep1)

        req_lbl = QLabel("REQUIREMENTS")
        req_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        root.addWidget(req_lbl)

        self._req_list = QListWidget()
        self._req_list.setSpacing(1)
        self._req_list.setStyleSheet(_LIST_STYLE)
        self._req_list.setAlternatingRowColors(True)
        self._req_list.setFocusPolicy(Qt.NoFocus)
        self._req_list.setMinimumHeight(120)
        root.addWidget(self._req_list, 1)

        self._req_rows: List[_RequirementRowWidget] = []
        if self._available:
            for name in self._available:
                row = _RequirementRowWidget(name, checked=(name in task.requires))
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, 36))
                self._req_list.addItem(item)
                self._req_list.setItemWidget(item, row)
                self._req_rows.append(row)
        else:
            empty = QLabel("No other tasks available")
            empty.setStyleSheet("color: #606060; font-size: 11px; padding: 6px;")
            empty.setAlignment(Qt.AlignCenter)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 36))
            self._req_list.addItem(item)
            self._req_list.setItemWidget(item, empty)

        # ── buttons ─────────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep2)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_CANCEL_BTN_STYLE)
        btn_row.addWidget(self._cancel_btn)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setStyleSheet(_APPLY_BTN_STYLE)
        btn_row.addWidget(self._apply_btn)

        root.addLayout(btn_row)

        self._apply_btn.clicked.connect(self._on_apply)
        self._cancel_btn.clicked.connect(self.cancel_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_task(
        self,
        task: AutoLamellaTaskDescription,
        available_tasks: Optional[List[str]] = None,
    ) -> None:
        """Reload the editor with a different task."""
        self._task = task
        self._available = [t for t in (available_tasks or []) if t != task.name]
        self._name_label.setText(task.name)
        self._optional_cb.setChecked(not task.required)

        self._req_list.clear()
        self._req_rows.clear()

        if self._available:
            for name in self._available:
                row = _RequirementRowWidget(name, checked=(name in task.requires))
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, 36))
                self._req_list.addItem(item)
                self._req_list.setItemWidget(item, row)
                self._req_rows.append(row)
        else:
            empty = QLabel("No other tasks available")
            empty.setStyleSheet("color: #606060; font-size: 11px; padding: 6px;")
            empty.setAlignment(Qt.AlignCenter)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 36))
            self._req_list.addItem(item)
            self._req_list.setItemWidget(item, empty)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        self._task.required = not self._optional_cb.isChecked()
        self._task.requires = [
            row.name_label.text()
            for row in self._req_rows
            if row.checkbox.isChecked()
        ]
        self.apply_clicked.emit(self._task)
