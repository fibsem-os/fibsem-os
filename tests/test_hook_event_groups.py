"""Subscribing to a group of events rather than an enumerated list.

A hook that lists every terminal event silently stops covering the set the moment a
new event is added -- HookEvent went from six members to ten in a single day, and any
configuration written the day before missed four of them with no warning. A group is
maintained in one place instead of in every config file on every machine. See FIB-494.
"""

import logging

import pytest

from fibsem.hooks import (
    EVENT_CATEGORIES,
    EVENT_GROUPS,
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    covers_event,
    resolve_events,
    unknown_subscriptions,
)


# ---------------------------------------------------------------------------
# the property that makes groups worth having
# ---------------------------------------------------------------------------

def test_every_event_is_classified():
    """The requirement, not a nice-to-have: adding an event without deciding what it
    is must fail here rather than quietly break someone's notifications months later.

    If this fails, put the new event in the category that matches its outcome.
    """
    classified = {e for events in EVENT_CATEGORIES.values() for e in events}

    assert set(HookEvent) - classified == set()


def test_the_categories_do_not_overlap():
    """A partition, not loose grouping. If an event could sit in two categories, a new
    one could be filed under a broad category only, leaving a narrow group silently
    not covering it -- which is the original bug wearing a new hat."""
    seen = set()
    for name, events in EVENT_CATEGORIES.items():
        overlap = seen & set(events)
        assert not overlap, f"{name} overlaps an earlier category on {overlap}"
        seen |= set(events)


def test_broader_groups_are_derived_not_hand_listed():
    """any_terminal is the union of the terminal categories, so classifying a new
    event once puts it in every group that should contain it."""
    expected = set()
    for category in ("any_completion", "any_failure", "any_cancellation", "any_skip"):
        expected |= set(EVENT_CATEGORIES[category])

    assert set(EVENT_GROUPS["any_terminal"]) == expected
    assert HookEvent.TASK_STARTED not in EVENT_GROUPS["any_terminal"]


def test_cancellation_is_not_a_failure():
    """Someone pressed Stop. That is not something to wake anyone up about -- if you
    were there to cancel it, you already know."""
    assert HookEvent.TASK_CANCELLED not in EVENT_GROUPS["any_failure"]
    assert HookEvent.WORKFLOW_CANCELLED not in EVENT_GROUPS["any_failure"]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def test_a_group_covers_its_events():
    assert covers_event(["any_failure"], HookEvent.TASK_FAILED)
    assert covers_event(["any_completion"], HookEvent.EXPERIMENT_COMPLETED)
    assert not covers_event(["any_failure"], HookEvent.TASK_COMPLETED)


def test_exact_event_names_still_work():
    """Every configuration written before groups existed uses them."""
    assert covers_event([HookEvent.TASK_FAILED], HookEvent.TASK_FAILED)
    assert covers_event(["task_failed"], "task_failed")
    assert not covers_event(["task_failed"], "task_completed")


def test_groups_and_exact_names_mix():
    subscriptions = ["any_failure", HookEvent.WORKFLOW_STARTED]

    assert covers_event(subscriptions, HookEvent.TASK_FAILED)
    assert covers_event(subscriptions, HookEvent.WORKFLOW_STARTED)
    assert not covers_event(subscriptions, HookEvent.TASK_STARTED)


def test_a_hook_subscribed_to_a_group_actually_fires():
    """Through HookManager.fire, not just the matcher."""
    fired = []
    manager = HookManager()
    manager.register(
        FunctionHook(name="any", events=["any_terminal"], callback=fired.append)
    )

    for event in (HookEvent.TASK_STARTED, HookEvent.TASK_FAILED,
                  HookEvent.EXPERIMENT_COMPLETED):
        manager.fire(HookContext(event=event))

    assert [c.event for c in fired] == ["task_failed", "experiment_completed"]


def test_a_new_event_joins_its_group_without_touching_any_config(monkeypatch):
    """The whole point. A hook subscribed to a group covers an event that did not
    exist when the subscription was written."""
    from fibsem import hooks

    patched = dict(hooks.EVENT_GROUPS)
    patched["any_failure"] = list(patched["any_failure"]) + ["task_timed_out"]
    monkeypatch.setattr(hooks, "EVENT_GROUPS", patched)

    assert hooks.covers_event(["any_failure"], "task_timed_out")


# ---------------------------------------------------------------------------
# display vs storage
# ---------------------------------------------------------------------------

def test_resolve_expands_a_group_for_display():
    resolved = resolve_events(["any_cancellation"])

    assert resolved == ["task_cancelled", "workflow_cancelled"]


def test_resolve_does_not_repeat_an_event_named_twice():
    resolved = resolve_events(["any_failure", "task_failed"])

    assert resolved == ["task_failed"]


def test_the_group_survives_serialization_unexpanded():
    """The rule the whole issue turns on: store the group, not its expansion. A saved
    expansion stops covering events added later, which is the drift groups exist to
    prevent, reintroduced at save time."""
    hook = FunctionHook(name="x", events=["any_failure"])

    assert hook.to_dict()["events"] == ["any_failure"]


# ---------------------------------------------------------------------------
# a name that will never match
# ---------------------------------------------------------------------------

def test_a_misspelled_subscription_warns(caplog):
    """It would otherwise match nothing, forever, without a word."""
    with caplog.at_level(logging.WARNING):
        FunctionHook(name="typo", events=["any_failuer"])

    assert any("any_failuer" in r.message for r in caplog.records)


def test_a_valid_subscription_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        FunctionHook(name="fine", events=["any_failure", HookEvent.TASK_STARTED])

    assert not caplog.records


def test_unknown_subscriptions_reports_only_the_bad_ones():
    assert unknown_subscriptions(["any_failure", "nope", HookEvent.TASK_FAILED]) == ["nope"]


def test_a_webhook_still_validates_its_own_fields():
    """WebhookHook has its own __post_init__; it must not lose the base one, or gain
    an exception from it."""
    from fibsem.hooks import WebhookHook

    with pytest.raises(ValueError):
        WebhookHook(name="w", events=["any_failure"], url="http://x", method="DELETE")
