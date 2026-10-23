"""The hook set every AutoLamella run registers.

Pinned here: changing it should be a deliberate edit to this file, not a side effect.
"""

from typing import List

from fibsem.applications.autolamella.hook_defaults import default_hooks
from fibsem.hooks import (
    HookEvent,
    HookManager,
    LoggingHook,
    NotificationHook,
)


def _by_name(hooks: List) -> dict:
    return {hook.name: hook for hook in hooks}


def test_the_default_set_is_what_it_is():
    """Pins the shipped defaults. If this fails, the change was intended or it wasn't."""
    hooks = _by_name(default_hooks())

    # No completion toast: one per lamella per task is noise (FIB-1538).
    assert set(hooks) == {
        "task_logger",
        "failure_toast",
        "cancellation_toast",
    }
    assert isinstance(hooks["task_logger"], LoggingHook)
    assert hooks["task_logger"].events == [
        HookEvent.TASK_STARTED,
        HookEvent.TASK_COMPLETED,
        HookEvent.TASK_FAILED,
        HookEvent.TASK_CANCELLED,
    ]
    assert [
        hooks[n].notification_type for n in ("failure_toast", "cancellation_toast")
    ] == [
        "error",
        "warning",
    ]


def test_the_failure_toast_says_the_run_continues():
    """A failure does not stop the run, and without the remaining count the toast reads
    as a dead workflow to someone at the instrument."""
    template = _by_name(default_hooks())["failure_toast"].message_template

    assert "{error}" in template
    assert "{tasks_remaining}" in template


def test_each_call_returns_fresh_hooks():
    """set_notifier injects a callable into every NotificationHook, so a shared
    module-level list would leak one run's state into the next. lamella_run_hooks()
    runs once per workflow run."""
    first, second = default_hooks(), default_hooks()

    assert all(a is not b for a, b in zip(first, second))

    manager = HookManager()
    for hook in first:
        manager.register(hook)
    manager.set_notifier(lambda message, kind: None)

    injected = [h for h in first if isinstance(h, NotificationHook)]
    untouched = [h for h in second if isinstance(h, NotificationHook)]
    assert all(h._notify is not None for h in injected)
    assert all(h._notify is None for h in untouched)


def test_a_notifier_reaches_every_default_notification_hook():
    """What the runner does after building the hooks: the toasts must actually deliver."""
    delivered = []
    manager = HookManager()
    for hook in default_hooks():
        manager.register(hook)
    manager.set_notifier(lambda message, kind: delivered.append((message, kind)))

    from fibsem.hooks import HookContext

    manager.fire(
        HookContext(
            event=HookEvent.TASK_FAILED,
            task_name="MillTrench",
            item_name="lam-1",
            error="stage timeout",
            tasks_remaining=3,
        )
    )

    assert len(delivered) == 1
    message, kind = delivered[0]
    assert kind == "error"
    assert "MillTrench" in message and "lam-1" in message
    assert "stage timeout" in message and "3" in message
