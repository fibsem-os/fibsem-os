"""Hooks come from the user's saved configuration, or from the defaults.

The step that makes hooks configurable without editing Python — the whole claim the
hook layer has over a plain signal connection. See FIB-497.

The three states of the `hooks` key are distinct and the distinction is the feature:
absent means never configured, a list replaces the defaults, and [] means the user
turned everything off.
"""

import logging

import pytest
import yaml

from fibsem import config as fcfg
from fibsem.applications.autolamella.hook_defaults import (
    build_hook_manager,
    default_hooks,
)
from fibsem.hooks import (
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    NotificationHook,
    SlackHook,
)


def _names(manager: HookManager):
    return [hook.name for hook in manager._hooks]


# ---------------------------------------------------------------------------
# the three states
# ---------------------------------------------------------------------------


def test_never_configured_uses_the_defaults():
    manager = build_hook_manager(None)

    assert _names(manager) == [hook.name for hook in default_hooks()]


def test_a_saved_configuration_replaces_the_defaults():
    """Not merged. Someone who deleted the failure toast must get no failure toast."""
    saved = [
        {
            "type": "SlackHook",
            "name": "slack",
            "events": ["any_failure"],
            "enabled": True,
            "task_types": [],
            "url": "https://hooks.slack.com/x",
        }
    ]

    manager = build_hook_manager(saved)

    assert _names(manager) == ["slack"]
    assert "failure_toast" not in _names(manager)


def test_an_empty_list_means_no_hooks_at_all():
    """[] is an answer -- everything turned off -- not an empty file."""
    manager = build_hook_manager([])

    assert manager._hooks == []


def test_empty_and_absent_are_not_the_same_thing():
    """Conflating them either resurrects hooks a user deleted or silences a fresh
    install. This is why the field is Optional rather than a plain list."""
    assert build_hook_manager([])._hooks == []
    assert build_hook_manager(None)._hooks != []


# ---------------------------------------------------------------------------
# what survives a round trip
# ---------------------------------------------------------------------------


def test_the_defaults_round_trip_through_the_preferences_field():
    """Save the defaults, reload them, get the defaults."""
    manager = build_hook_manager(None)

    restored = build_hook_manager(manager.to_dict()["hooks"])

    assert _names(restored) == _names(manager)


def test_a_group_subscription_survives_the_file():
    """Stored as the group name, so it keeps covering events added later."""
    saved = build_hook_manager(
        [
            {
                "type": "SlackHook",
                "name": "slack",
                "events": ["any_failure"],
                "url": "https://hooks.slack.com/x",
            }
        ]
    ).to_dict()["hooks"]

    assert saved[0]["events"] == ["any_failure"]
    assert isinstance(build_hook_manager(saved)._hooks[0], SlackHook)


def test_the_whole_thing_survives_yaml():
    """yaml.safe_dump refuses callables and arbitrary objects, so this is not
    hypothetical -- NotificationHook holds a _notify callable at runtime."""
    manager = build_hook_manager(None)
    manager.set_notifier(lambda message, kind: None)

    written = yaml.safe_dump({"hooks": manager.to_dict()["hooks"]})
    reloaded = build_hook_manager(yaml.safe_load(written)["hooks"])

    assert _names(reloaded) == _names(manager)


# ---------------------------------------------------------------------------
# a configuration that is wrong
# ---------------------------------------------------------------------------


def test_an_unknown_hook_type_is_skipped_not_fatal():
    saved = [
        {"type": "SomeFutureHook", "name": "future", "events": ["any_failure"]},
        {"type": "LoggingHook", "name": "keep", "events": ["any_failure"]},
    ]

    manager = build_hook_manager(saved)

    assert _names(manager) == ["keep"]


def test_a_broken_configuration_falls_back_to_the_defaults(caplog):
    """A bad hooks section must not stop the application starting, and running with no
    notifications is worse than running with the standard ones."""
    with caplog.at_level(logging.ERROR):
        manager = build_hook_manager(["not a hook dict"])

    assert _names(manager) == [hook.name for hook in default_hooks()]
    assert any("user-preferences" in r.message for r in caplog.records)


def test_a_configured_hook_actually_fires():
    """Through the manager, not just constructed."""
    manager = build_hook_manager(
        [
            {
                "type": "NotificationHook",
                "name": "mine",
                "events": ["any_failure"],
                "notification_type": "error",
                "message_template": "{item_name} broke",
            }
        ]
    )
    delivered = []
    manager.set_notifier(lambda message, kind: delivered.append((message, kind)))

    manager.fire(HookContext(event=HookEvent.TASK_FAILED, item_name="lam-01"))
    manager.fire(HookContext(event=HookEvent.TASK_COMPLETED, item_name="lam-01"))

    assert delivered == [("lam-01 broke", "error")]


# ---------------------------------------------------------------------------
# the preferences field itself
# ---------------------------------------------------------------------------


def test_an_absent_hooks_key_loads_as_none():
    prefs = fcfg.UserPreferences.from_dict({"display": {}})

    assert prefs.hooks is None


def test_an_empty_hooks_key_loads_as_empty_not_none():
    prefs = fcfg.UserPreferences.from_dict({"display": {}, "hooks": []})

    assert prefs.hooks == []


def test_a_file_containing_only_hooks_is_not_mistaken_for_the_legacy_format():
    """The nested-format check is by key; without "hooks" in that list a file with no
    other section would fall through to the flat legacy branch and lose them."""
    prefs = fcfg.UserPreferences.from_dict(
        {"hooks": [{"type": "LoggingHook", "name": "only", "events": ["any_failure"]}]}
    )

    assert prefs.hooks is not None
    assert prefs.hooks[0]["name"] == "only"


def test_preferences_round_trip_keeps_hooks(tmp_path, monkeypatch):
    """Through the real save/load pair, including the yaml on disk."""
    monkeypatch.setattr(fcfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(fcfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))

    prefs = fcfg.UserPreferences()
    prefs.hooks = build_hook_manager(None).to_dict()["hooks"]
    fcfg.save_user_preferences(prefs)

    reloaded = fcfg.load_user_preferences()

    assert [h["name"] for h in reloaded.hooks] == [h.name for h in default_hooks()]


def test_a_function_hook_is_not_written_to_the_file():
    """FunctionHooks are code-only. to_dict drops them, which is what keeps a callable
    out of the yaml -- and means a config load can never remove one."""
    manager = build_hook_manager(None)
    manager.register(FunctionHook(name="code_only", events=["any_completion"]))

    assert "code_only" not in [h["name"] for h in manager.to_dict()["hooks"]]
    assert "code_only" in _names(manager)
