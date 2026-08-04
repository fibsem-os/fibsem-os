"""What actually reaches the other end, and what happens when it is refused.

WebhookHook posted its body and never looked at the reply, so an endpoint that
rejected it -- Slack answering invalid_payload, an expired URL, a typo in the path --
failed in complete silence: the hook sat in the configuration looking connected and
delivered nothing. A rejected request is not an exception, so the try/except around it
caught nothing.

SlackHook exists for the same reason: Slack refuses an arbitrary JSON body, so
pointing a plain WebhookHook at a Slack URL could never have worked.
"""

import logging

import pytest

from fibsem.hooks import (
    HOOK_TYPES,
    DEFAULT_MESSAGE_TEMPLATE,
    HookContext,
    HookEvent,
    SlackHook,
    WebhookHook,
)

requests = pytest.importorskip("requests")


class _Response:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400


@pytest.fixture
def posted(monkeypatch):
    """Capture what would have been sent, and control what comes back."""
    calls = []

    def fake_request(method, url, json=None, timeout=None):
        calls.append({"method": method, "url": url, "json": json, "timeout": timeout})
        return calls[-1].get("_response") or _Response()

    monkeypatch.setattr(requests, "request", fake_request)
    return calls


def _context(**kwargs) -> HookContext:
    fields = dict(
        event=HookEvent.TASK_FAILED,
        task_name="MillTrench",
        item_name="lam-01",
        error="stage timeout",
        tasks_remaining=3,
    )
    fields.update(kwargs)
    return HookContext(**fields)


# ---------------------------------------------------------------------------
# the generic webhook
# ---------------------------------------------------------------------------

def test_a_generic_webhook_still_posts_the_structured_event(posted):
    hook = WebhookHook(name="w", events=["any_failure"], url="https://example.com/x")

    hook._post(_context())

    body = posted[0]["json"]
    assert body["event"] == "task_failed"
    assert body["item_name"] == "lam-01"
    assert body["error"] == "stage timeout"
    assert body["tasks_remaining"] == 3


def test_a_rejected_post_warns_with_the_reason(monkeypatch, caplog):
    """The failure that was previously silent. Slack's own reply is one word, and it
    is the whole diagnosis, so the body has to be in the message."""
    def fake_request(method, url, json=None, timeout=None):
        return _Response(400, "invalid_payload")

    monkeypatch.setattr(requests, "request", fake_request)
    hook = WebhookHook(name="slack-ish", events=["any_failure"], url="https://x/y")

    with caplog.at_level(logging.WARNING):
        hook._post(_context())

    message = " ".join(r.message for r in caplog.records)
    assert "slack-ish" in message and "400" in message and "invalid_payload" in message


def test_a_long_error_body_is_truncated(monkeypatch, caplog):
    """An endpoint can answer with an arbitrarily long HTML error page."""
    def fake_request(method, url, json=None, timeout=None):
        return _Response(500, "x" * 5000)

    monkeypatch.setattr(requests, "request", fake_request)
    hook = WebhookHook(name="w", events=["any_failure"], url="https://x/y")

    with caplog.at_level(logging.WARNING):
        hook._post(_context())

    assert len(caplog.records[0].message) < 400


def test_a_successful_post_is_quiet(posted, caplog):
    hook = WebhookHook(name="w", events=["any_failure"], url="https://x/y")

    with caplog.at_level(logging.WARNING):
        hook._post(_context())

    assert not caplog.records


def test_a_connection_failure_is_still_logged(monkeypatch, caplog):
    """The exception path must survive the response check being added around it."""
    def boom(method, url, json=None, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(requests, "request", boom)
    hook = WebhookHook(name="w", events=["any_failure"], url="https://x/y")

    with caplog.at_level(logging.ERROR):
        hook._post(_context())  # must not raise into the workflow

    assert any("no route to host" in r.exc_text for r in caplog.records if r.exc_text)


# ---------------------------------------------------------------------------
# slack
# ---------------------------------------------------------------------------

def test_slack_posts_the_shape_slack_accepts(posted):
    """An incoming webhook wants {"text": ...}; the structured event gets a 400."""
    hook = SlackHook(
        name="slack",
        events=["any_failure"],
        url="https://hooks.slack.com/services/x",
        message_template="{task_name} failed for {item_name}: {error}",
    )

    hook._post(_context())

    assert posted[0]["json"] == {"text": "MillTrench failed for lam-01: stage timeout"}


def test_slack_inherits_the_webhook_plumbing(posted):
    hook = SlackHook(name="s", events=["any_failure"], url="https://hooks.slack.com/x")

    hook._post(_context())

    assert posted[0]["method"] == "POST"
    assert posted[0]["timeout"] == 5


def test_slack_validates_its_url_like_any_webhook():
    hook = SlackHook(name="s", events=["any_failure"], url="ftp://nope")

    with pytest.raises(ValueError):
        hook.run(_context())


def test_slack_round_trips_including_the_message(posted):
    hook = SlackHook(
        name="s", events=["any_failure"], url="https://hooks.slack.com/x",
        message_template="{item_name} broke",
    )

    restored = SlackHook.from_dict(hook.to_dict())

    assert restored.message_template == "{item_name} broke"
    assert restored.url == "https://hooks.slack.com/x"
    assert restored.events == ["any_failure"]
    assert isinstance(restored, SlackHook)


def test_slack_without_a_message_uses_the_default():
    restored = SlackHook.from_dict(
        {"name": "s", "events": ["any_failure"], "url": "https://hooks.slack.com/x"}
    )

    assert restored.message_template == DEFAULT_MESSAGE_TEMPLATE


def test_slack_is_a_configurable_type():
    """In HOOK_TYPES, so it survives a save and appears in the editor's type list."""
    assert HOOK_TYPES["SlackHook"] is SlackHook
