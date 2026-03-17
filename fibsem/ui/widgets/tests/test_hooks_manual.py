"""Manual test for the hook system with toast notifications.

Usage:
    python fibsem/ui/widgets/tests/test_hooks_manual.py

Buttons fire hook events directly into a HookManager wired to a real
ToastManager — no microscope or experiment required.
"""

import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.widgets.notifications import NotificationBell, ToastManager
from fibsem.applications.autolamella.workflows.tasks.hooks import (
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    LoggingHook,
    NotificationHook,
)


class HookTestWindow(QMainWindow):
    _hook_toast_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hook System — Manual Test")
        self.resize(460, 380)
        self.setStyleSheet("background-color: #262930;")

        # Toast infrastructure
        self.toast_manager = ToastManager(self)
        self.bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(self.bell)

        # Wire signal → toast (mirrors AutoLamellaMainUI)
        self._hook_toast_signal.connect(self.toast_manager.show_toast)

        # Build HookManager (mirrors AutoLamellaUI.setup_hooks)
        self.manager = HookManager()
        self.manager.register(LoggingHook(
            name="task_logger",
            events=[HookEvent.TASK_STARTED, HookEvent.TASK_COMPLETED, HookEvent.TASK_FAILED,
                    HookEvent.WORKFLOW_STARTED, HookEvent.WORKFLOW_COMPLETED],
        ))
        self.manager.register(NotificationHook(
            name="completion_toast",
            events=[HookEvent.TASK_COMPLETED],
            notification_type="success",
            message_template="Task {task_name} complete for {lamella_name}",
        ))
        self.manager.register(NotificationHook(
            name="failure_toast",
            events=[HookEvent.TASK_FAILED],
            notification_type="error",
            message_template="Task {task_name} FAILED: {error}",
        ))
        self.manager.register(NotificationHook(
            name="workflow_toast",
            events=[HookEvent.WORKFLOW_STARTED, HookEvent.WORKFLOW_COMPLETED],
            notification_type="info",
            message_template="Workflow {event}",
        ))
        self.manager.register(FunctionHook(
            name="console_logger",
            events=[HookEvent.TASK_COMPLETED, HookEvent.TASK_FAILED],
            callback=lambda ctx: print(f"[FunctionHook] {ctx.event} — {ctx.task_name} / {ctx.lamella_name}"),
        ))
        self.manager.wire(self)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        label_style = "color: #888; font-size: 11px; padding-top: 8px;"
        btn_style = "color: #d6d6d6; background-color: #3d4251; padding: 8px; border-radius: 3px;"

        layout.addWidget(self.bell)

        layout.addWidget(self._section("Task events"))
        for label, ctx in [
            ("task_started  (no toast — logging only)",
             HookContext(event=HookEvent.TASK_STARTED,    task_name="MillTrench",  lamella_name="Lamella-1", task_type="MILL_TRENCH")),
            ("task_completed → success toast",
             HookContext(event=HookEvent.TASK_COMPLETED,  task_name="MillTrench",  lamella_name="Lamella-1", task_type="MILL_TRENCH")),
            ("task_failed    → error toast",
             HookContext(event=HookEvent.TASK_FAILED,     task_name="MillTrench",  lamella_name="Lamella-2", task_type="MILL_TRENCH", error="Stage timeout")),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _, c=ctx: self.manager.fire(c))
            layout.addWidget(btn)

        layout.addWidget(self._section("Workflow events"))
        for label, ctx in [
            ("workflow_started  → info toast",
             HookContext(event=HookEvent.WORKFLOW_STARTED)),
            ("workflow_completed → info toast",
             HookContext(event=HookEvent.WORKFLOW_COMPLETED)),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _, c=ctx: self.manager.fire(c))
            layout.addWidget(btn)

        layout.addWidget(self._section("task_type filter (IMAGING hook only fires for IMAGING tasks)"))
        imaging_hook = NotificationHook(
            name="imaging_only",
            events=[HookEvent.TASK_COMPLETED],
            notification_type="warning",
            message_template="Imaging done: {lamella_name}",
            task_types=["IMAGING"],
            _notify=lambda msg, typ: self._hook_toast_signal.emit(msg, typ),
        )
        self.manager.register(imaging_hook)

        for label, ctx in [
            ("task_completed  task_type=IMAGING  → warning toast",
             HookContext(event=HookEvent.TASK_COMPLETED, task_name="Acquire", lamella_name="Lamella-3", task_type="IMAGING")),
            ("task_completed  task_type=MILL_TRENCH → no extra toast",
             HookContext(event=HookEvent.TASK_COMPLETED, task_name="MillTrench", lamella_name="Lamella-3", task_type="MILL_TRENCH")),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _, c=ctx: self.manager.fire(c))
            layout.addWidget(btn)

        layout.addStretch()

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888; font-size: 11px; padding-top: 8px;")
        return lbl


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HookTestWindow()
    w.show()
    sys.exit(app.exec_())
