"""AutoLamella's hook set.

Outside the UI package and free of Qt, so a headless run gets the same hooks a GUI
run does. Not a user setting: which run events reach a person is the notifications
design's question, not a hooks list in the preferences file.
"""

from typing import List

from fibsem.hooks import Hook, HookEvent, HookManager, LoggingHook, NotificationHook


def default_hooks() -> List[Hook]:
    """The hooks every AutoLamella run registers.

    Returns **fresh instances on every call**, deliberately. `HookManager.set_notifier`
    injects a callable into each `NotificationHook`, so a shared module-level list
    would leak one run's notifier into the next. `lamella_run_hooks()` runs once per workflow run.

    No toast for a task that completes: one per lamella per task is noise, and the
    events page and the record carry every one. Failures and cancellations are
    problems, so they pop up.
    """
    return [
        LoggingHook(
            name="task_logger",
            events=[
                HookEvent.TASK_STARTED,
                HookEvent.TASK_COMPLETED,
                HookEvent.TASK_FAILED,
                HookEvent.TASK_CANCELLED,
            ],
        ),
        NotificationHook(
            name="failure_toast",
            events=[HookEvent.TASK_FAILED],
            notification_type="error",
            # A failure does not stop the run, so say what happens next: without the
            # remaining count this reads as a dead workflow to someone at the
            # instrument who is not watching the timeline.
            message_template=(
                "Task {task_name} FAILED for {item_name}: {error} "
                "— continuing, {tasks_remaining} tasks remaining"
            ),
        ),
        NotificationHook(
            name="cancellation_toast",
            events=[HookEvent.TASK_CANCELLED],
            notification_type="warning",
            message_template="Task {task_name} cancelled for {item_name}",
        ),
    ]


def build_hook_manager() -> HookManager:
    """A manager holding the default hooks, for one run."""
    manager = HookManager()
    for hook in default_hooks():
        manager.register(hook)
    return manager
