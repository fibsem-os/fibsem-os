"""Unit tests for the AutoLamella hook system."""

import yaml
import io
import pytest

from fibsem.applications.autolamella.workflows.tasks.hooks import (
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    LoggingHook,
    NotificationHook,
    WebhookHook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**kwargs) -> HookContext:
    defaults = dict(
        event=HookEvent.TASK_COMPLETED,
        task_name="MillTrench",
        task_type="MILL_TRENCH",
        lamella_name="Lamella-1",
    )
    defaults.update(kwargs)
    return HookContext(**defaults)


# ---------------------------------------------------------------------------
# LoggingHook
# ---------------------------------------------------------------------------

def test_logging_hook_fires(caplog):
    import logging
    manager = HookManager()
    manager.register(LoggingHook(name="l", events=[HookEvent.TASK_COMPLETED], level="INFO"))
    with caplog.at_level(logging.INFO):
        manager.fire(_ctx())
    assert any("task_completed" in r.message for r in caplog.records)


def test_logging_hook_respects_level(caplog):
    import logging
    manager = HookManager()
    manager.register(LoggingHook(name="l", events=[HookEvent.TASK_COMPLETED], level="WARNING"))
    with caplog.at_level(logging.WARNING):
        manager.fire(_ctx())
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# NotificationHook
# ---------------------------------------------------------------------------

def test_notification_hook_calls_notify():
    received = []
    hook = NotificationHook(
        name="n",
        events=[HookEvent.TASK_COMPLETED],
        notification_type="success",
        message_template="Task {task_name} {event} for {lamella_name}",
        _notify=lambda msg, typ: received.append((msg, typ)),
    )
    hook.run(_ctx())
    assert len(received) == 1
    msg, typ = received[0]
    assert "MillTrench" in msg
    assert "Lamella-1" in msg
    assert typ == "success"


def test_notification_hook_formats_error():
    received = []
    hook = NotificationHook(
        name="n",
        events=[HookEvent.TASK_FAILED],
        notification_type="error",
        message_template="FAILED: {error}",
        _notify=lambda msg, typ: received.append((msg, typ)),
    )
    hook.run(_ctx(event=HookEvent.TASK_FAILED, error="Stage timeout"))
    assert "Stage timeout" in received[0][0]


def test_notification_hook_falls_back_to_logging_without_notify(caplog):
    import logging
    hook = NotificationHook(
        name="n",
        events=[HookEvent.TASK_COMPLETED],
        notification_type="success",
    )
    with caplog.at_level(logging.INFO):
        hook.run(_ctx())
    assert any("NotificationHook" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FunctionHook
# ---------------------------------------------------------------------------

def test_function_hook_calls_callback():
    calls = []
    manager = HookManager()
    manager.register(FunctionHook(
        name="f",
        events=[HookEvent.TASK_COMPLETED],
        callback=lambda ctx: calls.append(ctx.task_name),
    ))
    manager.fire(_ctx())
    assert calls == ["MillTrench"]


# ---------------------------------------------------------------------------
# HookManager — filtering
# ---------------------------------------------------------------------------

def test_disabled_hook_does_not_fire():
    received = []
    manager = HookManager()
    manager.register(NotificationHook(
        name="n",
        events=[HookEvent.TASK_COMPLETED],
        enabled=False,
        _notify=lambda msg, typ: received.append(msg),
    ))
    manager.fire(_ctx())
    assert received == []


def test_wrong_event_does_not_fire():
    received = []
    manager = HookManager()
    manager.register(NotificationHook(
        name="n",
        events=[HookEvent.TASK_FAILED],
        _notify=lambda msg, typ: received.append(msg),
    ))
    manager.fire(_ctx(event=HookEvent.TASK_COMPLETED))
    assert received == []


def test_task_type_filter_matches():
    received = []
    manager = HookManager()
    manager.register(NotificationHook(
        name="n",
        events=[HookEvent.TASK_COMPLETED],
        task_types=["MILL_TRENCH"],
        _notify=lambda msg, typ: received.append(msg),
    ))
    manager.fire(_ctx(task_type="MILL_TRENCH"))
    assert len(received) == 1


def test_task_type_filter_excludes():
    received = []
    manager = HookManager()
    manager.register(NotificationHook(
        name="n",
        events=[HookEvent.TASK_COMPLETED],
        task_types=["MILL_TRENCH"],
        _notify=lambda msg, typ: received.append(msg),
    ))
    manager.fire(_ctx(task_type="IMAGING"))
    assert received == []


def test_hook_exception_is_isolated():
    """A failing hook must not prevent subsequent hooks from firing."""
    calls = []

    def bad_hook(ctx):
        raise RuntimeError("boom")

    manager = HookManager()
    manager.register(FunctionHook(name="bad", events=[HookEvent.TASK_COMPLETED], callback=bad_hook))
    manager.register(FunctionHook(name="good", events=[HookEvent.TASK_COMPLETED], callback=lambda ctx: calls.append("ok")))
    manager.fire(_ctx())
    assert calls == ["ok"]


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def test_logging_hook_yaml_roundtrip():
    original = LoggingHook(name="logger", events=[HookEvent.TASK_FAILED], level="WARNING", task_types=["MILL_TRENCH"])
    restored = LoggingHook.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.events == original.events
    assert restored.level == original.level
    assert restored.task_types == original.task_types
    assert restored.enabled == original.enabled


def test_notification_hook_yaml_roundtrip():
    original = NotificationHook(
        name="toast",
        events=[HookEvent.TASK_COMPLETED],
        notification_type="success",
        message_template="Done: {task_name}",
        enabled=False,
    )
    restored = NotificationHook.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.notification_type == original.notification_type
    assert restored.message_template == original.message_template
    assert restored.enabled == original.enabled
    assert restored._notify is None  # not serialized


def test_hook_manager_yaml_roundtrip():
    manager = HookManager()
    manager.register(LoggingHook(name="l", events=[HookEvent.TASK_STARTED, HookEvent.TASK_COMPLETED], level="INFO"))
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_FAILED], notification_type="error"))

    buf = io.StringIO()
    yaml.dump(manager.to_dict(), buf)
    buf.seek(0)
    restored = HookManager.from_dict(yaml.safe_load(buf))

    assert len(restored._hooks) == 2
    assert isinstance(restored._hooks[0], LoggingHook)
    assert isinstance(restored._hooks[1], NotificationHook)
    assert restored._hooks[0].name == "l"
    assert restored._hooks[1].notification_type == "error"


def test_function_hook_excluded_from_yaml():
    """FunctionHook should be silently excluded from serialization."""
    manager = HookManager()
    manager.register(FunctionHook(name="f", events=[HookEvent.TASK_COMPLETED], callback=lambda ctx: None))
    d = manager.to_dict()
    assert d["hooks"] == []


def test_unknown_hook_type_skipped(caplog):
    import logging
    d = {"hooks": [{"type": "NonExistentHook", "name": "x", "events": []}]}
    with caplog.at_level(logging.WARNING):
        manager = HookManager.from_dict(d)
    assert len(manager._hooks) == 0
    assert any("NonExistentHook" in r.message for r in caplog.records)


def test_save_and_load_yaml(tmp_path):
    path = str(tmp_path / "hooks.yaml")
    manager = HookManager()
    manager.register(LoggingHook(name="l", events=[HookEvent.TASK_COMPLETED]))
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_FAILED], notification_type="error"))
    manager.save_yaml(path)

    restored = HookManager.load_yaml(path)
    assert len(restored._hooks) == 2
    assert restored._hooks[1].notification_type == "error"


# ---------------------------------------------------------------------------
# wire()
# ---------------------------------------------------------------------------

def test_wire_injects_notify():
    """wire() should set _notify on NotificationHooks."""
    manager = HookManager()
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_COMPLETED]))
    assert manager._hooks[0]._notify is None

    calls = []

    class FakeUI:
        class _hook_toast_signal:
            @staticmethod
            def emit(msg, typ):
                calls.append((msg, typ))

    manager.wire(FakeUI())
    assert manager._hooks[0]._notify is not None

    manager._hooks[0].run(_ctx())
    assert len(calls) == 1


def test_wire_none_parent_is_safe():
    manager = HookManager()
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_COMPLETED]))
    manager.wire(None)  # must not raise
    assert manager._hooks[0]._notify is None


# ---------------------------------------------------------------------------
# WebhookHook
# ---------------------------------------------------------------------------

# def test_webhook_hook_rejects_invalid_scheme():
#     hook = WebhookHook(name="w", events=[HookEvent.TASK_COMPLETED], url="ftp://example.com")
#     with pytest.raises(ValueError, match="scheme"):
#         hook.run(_ctx())


# def test_webhook_hook_rejects_empty_url():
#     hook = WebhookHook(name="w", events=[HookEvent.TASK_COMPLETED], url="")
#     with pytest.raises(ValueError, match="empty"):
#         hook.run(_ctx())


# def test_webhook_hook_rejects_invalid_method():
#     with pytest.raises(ValueError, match="method"):
#         WebhookHook(name="w", events=[], url="https://x.com", method="DELETE")


# def test_webhook_hook_from_dict_sanitises_bad_method(caplog):
#     import logging
#     with caplog.at_level(logging.WARNING):
#         hook = WebhookHook.from_dict({"name": "w", "events": [], "url": "https://x.com", "method": "DELETE"})
#     assert hook.method == "POST"
#     assert any("invalid method" in r.message for r in caplog.records)

# def test_webhook_hook_yaml_roundtrip():
#     original = WebhookHook(
#         name="slack",
#         events=[HookEvent.TASK_FAILED],
#         url="https://hooks.example.com/abc",
#         method="POST",
#         timeout=10,
#     )
#     restored = WebhookHook.from_dict(original.to_dict())
#     assert restored.url == original.url
#     assert restored.method == original.method
#     assert restored.timeout == original.timeout
#     assert restored.events == original.events


# def test_webhook_hook_in_manager_yaml_roundtrip(tmp_path):
#     manager = HookManager()
#     manager.register(WebhookHook(name="w", events=[HookEvent.WORKFLOW_COMPLETED], url="https://x.com"))
#     path = str(tmp_path / "hooks.yaml")
#     manager.save_yaml(path)
#     restored = HookManager.load_yaml(path)
#     assert len(restored._hooks) == 1
#     assert isinstance(restored._hooks[0], WebhookHook)
#     assert restored._hooks[0].url == "https://x.com"
