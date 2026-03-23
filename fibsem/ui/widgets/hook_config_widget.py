"""Widget for viewing and editing the AutoLamella hook configuration."""

from __future__ import annotations

from typing import List, Optional, Type

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.iconify import QIconifyIcon

from fibsem.applications.autolamella.workflows.tasks.hooks import (
    FunctionHook,
    Hook,
    HookEvent,
    HookManager,
    LoggingHook,
    NotificationHook,
    WebhookHook,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_EVENTS = [
    HookEvent.TASK_STARTED,
    HookEvent.TASK_COMPLETED,
    HookEvent.TASK_FAILED,
    HookEvent.WORKFLOW_STARTED,
    HookEvent.WORKFLOW_COMPLETED,
]

_HOOK_DISPLAY_NAMES = {
    "LoggingHook": "Logging",
    "NotificationHook": "Notification",
    "WebhookHook": "Webhook",
}

_HOOK_CLASSES: dict[str, Type[Hook]] = {
    "LoggingHook": LoggingHook,
    "NotificationHook": NotificationHook,
    "WebhookHook": WebhookHook,
}

_TYPE_COLORS = {
    "LoggingHook":      "#50a6ff",
    "NotificationHook": stylesheets.GREEN_COLOR,
    "WebhookHook":      stylesheets.ORANGE_COLOR,
}


# ---------------------------------------------------------------------------
# HookEditDialog
# ---------------------------------------------------------------------------

class HookEditDialog(QDialog):
    """Dialog for creating or editing a single hook.

    Pass an existing hook to pre-populate. Pass a hook class to create a new one.
    Returns the configured hook via get_hook().
    """

    def __init__(self, hook: Optional[Hook] = None,
                 hook_cls: Optional[Type[Hook]] = None,
                 parent=None):
        super().__init__(parent)
        self._hook_cls = hook_cls or (type(hook) if hook else LoggingHook)
        self._hook = hook
        self.setWindowTitle(f"{'Edit' if hook else 'Add'} {_HOOK_DISPLAY_NAMES.get(self._hook_cls.__name__, self._hook_cls.__name__)} Hook")
        self.setMinimumWidth(420)
        self._setup_ui()
        if hook:
            self._load(hook)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Common fields ---
        common_w = QWidget()
        common_form = QFormLayout(common_w)
        common_form.setContentsMargins(4, 4, 4, 4)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. my_hook")
        common_form.addRow("Name", self._name_edit)

        self._enabled_chk = QCheckBox()
        self._enabled_chk.setChecked(True)
        common_form.addRow("Enabled", self._enabled_chk)

        layout.addWidget(TitledPanel("General", content=common_w, collapsible=False))

        # --- Events ---
        events_w = QWidget()
        events_layout = QVBoxLayout(events_w)
        events_layout.setContentsMargins(4, 4, 4, 4)
        events_layout.setSpacing(2)
        self._event_checks: dict[str, QCheckBox] = {}
        for event in _ALL_EVENTS:
            chk = QCheckBox(event.value)
            self._event_checks[event.value] = chk
            events_layout.addWidget(chk)
        layout.addWidget(TitledPanel("Events", content=events_w, collapsible=False))

        # --- Type-specific fields ---
        specific_w = QWidget()
        specific_form = QFormLayout(specific_w)
        specific_form.setContentsMargins(4, 4, 4, 4)
        self._specific_widgets: dict = {}

        cls_name = self._hook_cls.__name__

        if cls_name == "LoggingHook":
            combo = QComboBox()
            combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
            combo.setCurrentText("INFO")
            specific_form.addRow("Level", combo)
            self._specific_widgets["level"] = combo

        elif cls_name == "NotificationHook":
            type_combo = QComboBox()
            type_combo.addItems(["info", "success", "warning", "error"])
            specific_form.addRow("Notification type", type_combo)
            self._specific_widgets["notification_type"] = type_combo

            tmpl = QLineEdit()
            tmpl.setPlaceholderText("Task {task_name} {event} for {lamella_name}")
            specific_form.addRow("Message template", tmpl)
            self._specific_widgets["message_template"] = tmpl

        elif cls_name == "WebhookHook":
            url_edit = QLineEdit()
            url_edit.setPlaceholderText("https://hooks.example.com/...")
            specific_form.addRow("URL", url_edit)
            self._specific_widgets["url"] = url_edit

            method_combo = QComboBox()
            method_combo.addItems(["POST", "GET"])
            specific_form.addRow("Method", method_combo)
            self._specific_widgets["method"] = method_combo

            timeout_spin = QSpinBox()
            timeout_spin.setRange(1, 60)
            timeout_spin.setValue(5)
            timeout_spin.setSuffix(" s")
            specific_form.addRow("Timeout", timeout_spin)
            self._specific_widgets["timeout"] = timeout_spin

        if self._specific_widgets:
            layout.addWidget(TitledPanel(
                _HOOK_DISPLAY_NAMES.get(cls_name, cls_name),
                content=specific_w,
                collapsible=False,
            ))

        # --- Buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load(self, hook: Hook):
        self._name_edit.setText(hook.name)
        self._enabled_chk.setChecked(hook.enabled)
        for event_val, chk in self._event_checks.items():
            chk.setChecked(event_val in hook.events)

        cls_name = type(hook).__name__
        if cls_name == "LoggingHook":
            self._specific_widgets["level"].setCurrentText(hook.level)
        elif cls_name == "NotificationHook":
            self._specific_widgets["notification_type"].setCurrentText(hook.notification_type)
            self._specific_widgets["message_template"].setText(hook.message_template)
        elif cls_name == "WebhookHook":
            self._specific_widgets["url"].setText(hook.url)
            self._specific_widgets["method"].setCurrentText(hook.method)
            self._specific_widgets["timeout"].setValue(hook.timeout)

    def get_hook(self) -> Hook:
        """Return a new hook instance with the current dialog values."""
        name = self._name_edit.text().strip()
        enabled = self._enabled_chk.isChecked()
        events = [v for v, chk in self._event_checks.items() if chk.isChecked()]

        cls_name = self._hook_cls.__name__

        if cls_name == "LoggingHook":
            return LoggingHook(
                name=name, enabled=enabled, events=events,
                level=self._specific_widgets["level"].currentText(),
            )
        elif cls_name == "NotificationHook":
            return NotificationHook(
                name=name, enabled=enabled, events=events,
                notification_type=self._specific_widgets["notification_type"].currentText(),
                message_template=self._specific_widgets["message_template"].text()
                    or "Task {task_name} {event} for {lamella_name}",
            )
        elif cls_name == "WebhookHook":
            return WebhookHook(
                name=name, enabled=enabled, events=events,
                url=self._specific_widgets["url"].text().strip(),
                method=self._specific_widgets["method"].currentText(),
                timeout=self._specific_widgets["timeout"].value(),
            )
        return LoggingHook(name=name, enabled=enabled, events=events)


# ---------------------------------------------------------------------------
# Hook row widget
# ---------------------------------------------------------------------------

class _HookRowWidget(QWidget):
    """Single row in the hook list: enable toggle · name · type badge · events."""

    enabled_changed = pyqtSignal(bool)

    def __init__(self, hook: Hook, parent=None):
        super().__init__(parent)
        self.hook = hook
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self._chk_enabled = QCheckBox()
        self._chk_enabled.setChecked(hook.enabled)
        self._chk_enabled.setToolTip("Enable / disable hook")
        self._chk_enabled.toggled.connect(self._on_enabled_changed)
        layout.addWidget(self._chk_enabled)

        name_lbl = QLabel(hook.name or "<unnamed>")
        name_lbl.setStyleSheet("color: #d6d6d6; font-weight: bold;")
        layout.addWidget(name_lbl)

        cls_name = type(hook).__name__
        color = _TYPE_COLORS.get(cls_name, "#888")
        display = _HOOK_DISPLAY_NAMES.get(cls_name, cls_name)
        type_lbl = QLabel(display)
        type_lbl.setStyleSheet(
            f"color: {color}; font-size: 11px; border: 1px solid {color};"
            " border-radius: 3px; padding: 1px 5px;"
        )
        layout.addWidget(type_lbl)

        events_str = ", ".join(e.replace("task_", "").replace("workflow_", "wf_") for e in hook.events) or "no events"
        events_lbl = QLabel(events_str)
        events_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(events_lbl, 1)

    def _on_enabled_changed(self, checked: bool):
        self.hook.enabled = checked
        self.enabled_changed.emit(checked)


# ---------------------------------------------------------------------------
# HookConfigWidget
# ---------------------------------------------------------------------------

class HookConfigWidget(QWidget):
    """List + add/edit/remove controls for a HookManager.

    Usage::

        widget = HookConfigWidget(manager)
        # or embed in a layout and call get_manager() when done.
    """

    hooks_changed = pyqtSignal(list)  # emits List[Hook] on any change

    def __init__(self, manager: Optional[HookManager] = None, parent=None):
        super().__init__(parent)
        self._manager = manager or HookManager()
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Button bar
        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(4, 4, 4, 4)
        bar_layout.setSpacing(4)

        self._btn_add = IconToolButton("mdi:plus", tooltip="Add hook", size=24)
        self._btn_edit = IconToolButton("mdi:pencil-outline", tooltip="Edit selected hook", size=24)
        self._btn_remove = IconToolButton("mdi:trash-can-outline", tooltip="Remove selected hook", size=24)

        bar_layout.addWidget(self._btn_add)
        bar_layout.addWidget(self._btn_edit)
        bar_layout.addWidget(self._btn_remove)
        bar_layout.addStretch()

        # List
        self._list = QListWidget()
        self._list.setSpacing(2)
        self._list.setStyleSheet("QListWidget { background: #1e2124; border: none; }"
                                  "QListWidget::item { border-radius: 3px; }"
                                  "QListWidget::item:selected { background: #3d4251; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 0, 4, 4)
        content_layout.setSpacing(4)
        content_layout.addWidget(bar)
        content_layout.addWidget(self._list)

        outer.addWidget(TitledPanel("Hooks", content=content, collapsible=False))

        # Connections
        self._btn_add.clicked.connect(self._on_add)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_remove.clicked.connect(self._on_remove)
        self._list.itemDoubleClicked.connect(lambda _: self._on_edit())

    def _refresh_list(self):
        self._list.clear()
        for hook in self._manager._hooks:
            if isinstance(hook, FunctionHook):
                continue  # FunctionHooks are code-only, not shown
            row = _HookRowWidget(hook)
            row.enabled_changed.connect(lambda _: self.hooks_changed.emit(self._manager._hooks))
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

    def _selected_index(self) -> Optional[int]:
        """Return the list index of the selected item, adjusted for any FunctionHooks."""
        row = self._list.currentRow()
        if row < 0:
            return None
        # Map visible index → _hooks index (skip FunctionHooks)
        visible = 0
        for i, hook in enumerate(self._manager._hooks):
            if isinstance(hook, FunctionHook):
                continue
            if visible == row:
                return i
            visible += 1
        return None

    def _on_add(self):
        menu = QMenu(self)
        for cls_name, display in _HOOK_DISPLAY_NAMES.items():
            action = menu.addAction(display)
            action.setData(cls_name)
        chosen = menu.exec_(self._btn_add.mapToGlobal(self._btn_add.rect().bottomLeft()))
        if chosen is None:
            return
        cls = _HOOK_CLASSES[chosen.data()]
        dlg = HookEditDialog(hook_cls=cls, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._manager.register(dlg.get_hook())
            self._refresh_list()
            self.hooks_changed.emit(self._manager._hooks)

    def _on_edit(self):
        idx = self._selected_index()
        if idx is None:
            return
        hook = self._manager._hooks[idx]
        dlg = HookEditDialog(hook=hook, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new_hook = dlg.get_hook()
            # Preserve runtime-injected _notify if present
            if isinstance(hook, NotificationHook) and isinstance(new_hook, NotificationHook):
                new_hook._notify = hook._notify
            self._manager._hooks[idx] = new_hook
            self._refresh_list()
            self.hooks_changed.emit(self._manager._hooks)

    def _on_remove(self):
        idx = self._selected_index()
        if idx is None:
            return
        del self._manager._hooks[idx]
        self._refresh_list()
        self.hooks_changed.emit(self._manager._hooks)

    def get_manager(self) -> HookManager:
        """Return the current HookManager (reflects all edits made in the widget)."""
        return self._manager
