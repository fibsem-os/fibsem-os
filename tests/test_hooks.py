"""Unit tests for the fibsem hook system."""

from fibsem.hooks import (
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    LoggingHook,
    NotificationHook,
    fire_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(**kwargs) -> HookContext:
    defaults = dict(
        event=HookEvent.TASK_COMPLETED,
        task_name="MillTrench",
        task_type="MILL_TRENCH",
        item_name="Lamella-1",
    )
    defaults.update(kwargs)
    return HookContext(**defaults)


# ---------------------------------------------------------------------------
# LoggingHook
# ---------------------------------------------------------------------------


def test_logging_hook_fires(caplog):
    import logging

    manager = HookManager()
    manager.register(
        LoggingHook(name="l", events=[HookEvent.TASK_COMPLETED], level="INFO")
    )
    with caplog.at_level(logging.INFO):
        manager.fire(_ctx())
    assert any("task_completed" in r.message for r in caplog.records)


def test_logging_hook_respects_level(caplog):
    import logging

    manager = HookManager()
    manager.register(
        LoggingHook(name="l", events=[HookEvent.TASK_COMPLETED], level="WARNING")
    )
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
        message_template="Task {task_name} {event} for {item_name}",
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


def test_notification_hook_cancelled_is_warning():
    """A user cancel notifies as a 'warning' (amber), not an 'error' failure."""
    received = []
    hook = NotificationHook(
        name="cancel",
        events=[HookEvent.TASK_CANCELLED],
        notification_type="warning",
        message_template="Task {task_name} cancelled for {item_name}",
        _notify=lambda msg, typ: received.append((msg, typ)),
    )
    # only fires for the cancelled event, not for a failure
    manager = HookManager()
    manager.register(hook)
    manager.fire(_ctx(event=HookEvent.TASK_FAILED, error="boom"))
    assert received == []
    manager.fire(
        _ctx(event=HookEvent.TASK_CANCELLED, task_name="Mill", item_name="01-elk")
    )
    assert received == [("Task Mill cancelled for 01-elk", "warning")]


def test_notification_hook_unknown_placeholder_does_not_drop_message():
    """One bad placeholder must not swallow the whole notification."""
    received = []
    hook = NotificationHook(
        name="typo",
        events=[HookEvent.TASK_COMPLETED],
        message_template="{task_name} on {itemname}",
        _notify=lambda msg, typ: received.append(msg),
    )
    hook.run(_ctx())
    assert received == ["MillTrench on {itemname}"]


def test_notification_hook_renders_skip_reason():
    received = []
    hook = NotificationHook(
        name="skip",
        events=[HookEvent.TASK_SKIPPED],
        message_template="Skipped {task_name} on {item_name}: {skip_reason}",
        _notify=lambda msg, typ: received.append(msg),
    )
    hook.run(_ctx(event=HookEvent.TASK_SKIPPED, skip_reason="missing_prereqs"))
    assert received == ["Skipped MillTrench on Lamella-1: missing_prereqs"]


def test_skip_reason_renders_empty_when_absent():
    """Every other event leaves skip_reason unset; the placeholder must not print None."""
    received = []
    hook = NotificationHook(
        name="skip",
        events=[HookEvent.TASK_COMPLETED],
        message_template="{task_name}[{skip_reason}]",
        _notify=lambda msg, typ: received.append(msg),
    )
    hook.run(_ctx())
    assert received == ["MillTrench[]"]


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
    manager.register(
        FunctionHook(
            name="f",
            events=[HookEvent.TASK_COMPLETED],
            callback=lambda ctx: calls.append(ctx.task_name),
        )
    )
    manager.fire(_ctx())
    assert calls == ["MillTrench"]


# ---------------------------------------------------------------------------
# HookManager — filtering
# ---------------------------------------------------------------------------


def test_disabled_hook_does_not_fire():
    received = []
    manager = HookManager()
    manager.register(
        NotificationHook(
            name="n",
            events=[HookEvent.TASK_COMPLETED],
            enabled=False,
            _notify=lambda msg, typ: received.append(msg),
        )
    )
    manager.fire(_ctx())
    assert received == []


def test_wrong_event_does_not_fire():
    received = []
    manager = HookManager()
    manager.register(
        NotificationHook(
            name="n",
            events=[HookEvent.TASK_FAILED],
            _notify=lambda msg, typ: received.append(msg),
        )
    )
    manager.fire(_ctx(event=HookEvent.TASK_COMPLETED))
    assert received == []


def test_task_type_filter_matches():
    received = []
    manager = HookManager()
    manager.register(
        NotificationHook(
            name="n",
            events=[HookEvent.TASK_COMPLETED],
            task_types=["MILL_TRENCH"],
            _notify=lambda msg, typ: received.append(msg),
        )
    )
    manager.fire(_ctx(task_type="MILL_TRENCH"))
    assert len(received) == 1


def test_task_type_filter_excludes():
    received = []
    manager = HookManager()
    manager.register(
        NotificationHook(
            name="n",
            events=[HookEvent.TASK_COMPLETED],
            task_types=["MILL_TRENCH"],
            _notify=lambda msg, typ: received.append(msg),
        )
    )
    manager.fire(_ctx(task_type="IMAGING"))
    assert received == []


def test_hook_exception_is_isolated():
    """A failing hook must not prevent subsequent hooks from firing."""
    calls = []

    def bad_hook(ctx):
        raise RuntimeError("boom")

    manager = HookManager()
    manager.register(
        FunctionHook(name="bad", events=[HookEvent.TASK_COMPLETED], callback=bad_hook)
    )
    manager.register(
        FunctionHook(
            name="good",
            events=[HookEvent.TASK_COMPLETED],
            callback=lambda ctx: calls.append("ok"),
        )
    )
    manager.fire(_ctx())
    assert calls == ["ok"]


# ---------------------------------------------------------------------------
# set_notifier()
# ---------------------------------------------------------------------------


def test_set_notifier_applies_to_already_registered_hooks():
    calls = []
    manager = HookManager()
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_COMPLETED]))
    assert manager._hooks[0]._notify is None

    manager.set_notifier(lambda msg, typ: calls.append((msg, typ)))
    manager.fire(_ctx())
    assert len(calls) == 1


def test_set_notifier_applies_to_hooks_registered_later():
    """A hook added after the notifier — e.g. from the config widget — must notify too,
    not silently fall back to logging."""
    calls = []
    manager = HookManager()
    manager.set_notifier(lambda msg, typ: calls.append((msg, typ)))
    manager.register(NotificationHook(name="late", events=[HookEvent.TASK_COMPLETED]))

    manager.fire(_ctx())
    assert len(calls) == 1


def test_set_notifier_does_not_override_an_explicit_notify():
    manager_calls, own_calls = [], []
    manager = HookManager()
    manager.register(
        NotificationHook(
            name="own",
            events=[HookEvent.TASK_COMPLETED],
            _notify=lambda msg, typ: own_calls.append(msg),
        )
    )
    manager.set_notifier(lambda msg, typ: manager_calls.append(msg))

    manager.fire(_ctx())
    assert len(own_calls) == 1 and manager_calls == []


def test_no_notifier_is_safe():
    manager = HookManager()
    manager.register(NotificationHook(name="n", events=[HookEvent.TASK_COMPLETED]))
    manager.set_notifier(None)  # headless: must not raise
    manager.fire(_ctx())
    assert manager._hooks[0]._notify is None


# ---------------------------------------------------------------------------
# fire_event() — the producer-side API
# ---------------------------------------------------------------------------


def test_fire_event_builds_the_context():
    received = []
    manager = HookManager()
    manager.register(
        FunctionHook(
            name="f",
            events=[HookEvent.TASK_COMPLETED],
            callback=received.append,
        )
    )

    fire_event(manager, HookEvent.TASK_COMPLETED, task_name="Mill", item_name="grid-1")

    assert len(received) == 1
    assert (received[0].task_name, received[0].item_name) == ("Mill", "grid-1")


def test_fire_event_without_a_manager_is_a_no_op():
    fire_event(None, HookEvent.TASK_COMPLETED, task_name="Mill")  # must not raise


def test_any_producer_can_fire_events():
    """The producer contract is fire_event() alone — no AutoLamella, no Qt, no
    microscope. This stands in for a future milling/acquisition producer."""
    seen = []
    manager = HookManager()
    manager.register(
        FunctionHook(
            name="f",
            events=[
                HookEvent.TASK_STARTED,
                HookEvent.TASK_COMPLETED,
                HookEvent.TASK_FAILED,
            ],
            callback=lambda ctx: seen.append((ctx.event, ctx.item_name)),
        )
    )

    def run_some_work(hook_manager, item_name: str, fail: bool = False):
        fire_event(
            hook_manager,
            HookEvent.TASK_STARTED,
            task_name="Polish",
            item_name=item_name,
        )
        if fail:
            fire_event(
                hook_manager,
                HookEvent.TASK_FAILED,
                task_name="Polish",
                item_name=item_name,
                error="beam off",
            )
            return
        fire_event(
            hook_manager,
            HookEvent.TASK_COMPLETED,
            task_name="Polish",
            item_name=item_name,
        )

    run_some_work(manager, "grid-1")
    run_some_work(manager, "grid-2", fail=True)
    run_some_work(None, "grid-3")  # no manager: silent

    assert seen == [
        ("task_started", "grid-1"),
        ("task_completed", "grid-1"),
        ("task_started", "grid-2"),
        ("task_failed", "grid-2"),
    ]


def test_notify_reaches_the_notifier_and_is_quiet_without_one():
    """A run's own notices (a grid run's report) go where hooks' do: the app's
    notifications when it listens, nowhere when headless."""
    from fibsem.hooks import HookManager

    manager = HookManager()
    manager.notify("nobody is listening", "info")

    said = []
    manager.set_notifier(lambda message, kind: said.append((message, kind)))
    manager.notify("Grid screening report written", "info")
    assert said == [("Grid screening report written", "info")]
