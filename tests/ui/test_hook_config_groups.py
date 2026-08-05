"""The hook editor must not destroy a group subscription.

Before FIB-494 the dialog rendered one checkbox per HookEvent and rebuilt `events`
from whatever was ticked. A hook subscribed to `any_failure` matched no checkbox, so
it loaded with everything unticked and saved as `events=[]` -- the editor silently
unsubscribing a hook from everything. Groups are worthless if the only tool for
editing them deletes them.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.hooks import EVENT_GROUPS, HookEvent, LoggingHook, WebhookHook
from fibsem.ui.widgets.hook_config_widget import HookEditDialog


def test_a_group_subscription_survives_an_edit(qapp):
    """Open and accept without changing anything: the group must come back out."""
    hook = LoggingHook(name="failures", events=["any_failure"])

    dialog = HookEditDialog(hook=hook)
    result = dialog.get_hook()

    assert result.events == ["any_failure"]


def test_a_group_is_not_saved_as_its_expansion(qapp):
    """The rule the issue turns on. An expansion stops covering events added later."""
    hook = LoggingHook(name="terminal", events=["any_terminal"])

    result = HookEditDialog(hook=hook).get_hook()

    assert result.events == ["any_terminal"]
    assert "task_failed" not in result.events


def test_exact_events_still_round_trip(qapp):
    hook = LoggingHook(
        name="two", events=[HookEvent.TASK_STARTED, HookEvent.TASK_FAILED]
    )

    result = HookEditDialog(hook=hook).get_hook()

    assert set(result.events) == {"task_started", "task_failed"}


def test_a_group_and_an_exact_event_both_survive(qapp):
    hook = LoggingHook(name="mixed", events=["any_failure", HookEvent.WORKFLOW_STARTED])

    result = HookEditDialog(hook=hook).get_hook()

    assert set(result.events) == {"any_failure", "workflow_started"}


def test_every_group_is_offered(qapp):
    """A group the dialog cannot show is a group a user cannot keep."""
    dialog = HookEditDialog(hook_cls=LoggingHook)

    assert set(dialog._group_checks) == set(EVENT_GROUPS)


def test_a_group_checkbox_says_what_it_covers(qapp):
    dialog = HookEditDialog(hook_cls=LoggingHook)

    tooltip = dialog._group_checks["any_cancellation"].toolTip()

    assert "task_cancelled" in tooltip and "workflow_cancelled" in tooltip
    assert "future" in tooltip


def test_ticking_a_group_produces_the_group(qapp):
    dialog = HookEditDialog(hook_cls=WebhookHook)
    dialog._name_edit.setText("slack")
    dialog._group_checks["any_failure"].setChecked(True)
    dialog._specific_widgets["url"].setText("https://hooks.example.com/x")

    result = dialog.get_hook()

    assert result.events == ["any_failure"]
    assert result.url == "https://hooks.example.com/x"


# ---------------------------------------------------------------------------
# a new hook type has to be editable, or it can only be written by hand
# ---------------------------------------------------------------------------

def test_a_slack_hook_round_trips_through_the_dialog(qapp):
    from fibsem.hooks import SlackHook

    hook = SlackHook(
        name="slack", events=["any_failure"],
        url="https://hooks.slack.com/services/x",
        message_template="{item_name} broke",
    )

    result = HookEditDialog(hook=hook).get_hook()

    assert isinstance(result, SlackHook)
    assert result.url == "https://hooks.slack.com/services/x"
    assert result.message_template == "{item_name} broke"
    assert result.events == ["any_failure"]


def test_slack_is_offered_when_adding_a_hook(qapp):
    """A type missing from the editor's list can only be configured by hand."""
    from fibsem.hooks import HOOK_TYPES
    from fibsem.ui.widgets.hook_config_widget import _HOOK_CLASSES

    assert set(_HOOK_CLASSES) == set(HOOK_TYPES)
