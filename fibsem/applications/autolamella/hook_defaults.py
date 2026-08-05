"""AutoLamella's default hook set, expressed as data rather than as UI code.

Deliberately outside the UI package and free of Qt, so the same hooks a GUI run gets
can be built headlessly, and so the defaults can be serialized and compared against a
user's saved configuration without a QApplication.

First step of FIB-497: before a user can override the defaults, the defaults have to
be something you can round-trip through YAML and diff. They were previously four
`manager.register(...)` calls inlined in `AutoLamellaUI.setup_hooks()`, reachable only
by reading the source.
"""

import logging
from typing import List, Optional

from fibsem.hooks import Hook, HookEvent, HookManager, LoggingHook, NotificationHook


def default_hooks() -> List[Hook]:
    """The hooks AutoLamella registers when the user has not configured any.

    Returns **fresh instances on every call**, deliberately. `HookManager.set_notifier`
    injects a callable into each `NotificationHook` and `HookConfigWidget` edits hooks
    in place, so a shared module-level list would leak one run's notifier -- and one
    session's edits -- into the next. `setup_hooks()` runs once per workflow run.

    Every hook here must be a type in `HOOK_TYPES`. `HookManager.to_dict` silently
    drops anything that is not, so a `FunctionHook` default would vanish the first time
    a user saved their configuration. Pinned by a test.
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
            name="completion_toast",
            events=[HookEvent.TASK_COMPLETED],
            notification_type="success",
            message_template="Task {task_name} complete for {item_name}",
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


def build_hook_manager(configured: Optional[List[dict]]) -> HookManager:
    """Build the manager for a run from a user's saved hooks, or from the defaults.

    ``configured`` is ``UserPreferences.hooks``. The three states are distinct:

    * ``None`` -- never configured. Use the defaults, and note they are *not* written
      to disk on first run: materialising them would freeze every user who never opens
      the dialog on this version's defaults forever, so improving them later would
      reach nobody.
    * a list -- **replaces** the defaults rather than merging with them. Someone who
      deleted the failure toast must get no failure toast; merging would put it back.
    * ``[]`` -- everything turned off. A real answer, not an empty file.

    A configuration that cannot be read falls back to the defaults with a warning
    rather than raising: a bad hooks section must not stop the application starting,
    and running with no notifications at all is worse than running with the standard
    ones. The whole section falls back rather than the offending entry, so the warning
    describes something a user can actually find and fix.
    """
    if configured is None:
        manager = HookManager()
        for hook in default_hooks():
            manager.register(hook)
        return manager

    try:
        return HookManager.from_dict({"hooks": configured})
    except Exception:
        logging.exception(
            "Could not build hooks from your saved configuration; falling back to the "
            "defaults. Fix or remove the 'hooks' section of user-preferences.yaml."
        )
        manager = HookManager()
        for hook in default_hooks():
            manager.register(hook)
        return manager
