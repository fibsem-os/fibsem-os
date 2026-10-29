"""Slack notifications (fibsem.alerts): what is told, when, and how it is
posted, against a fake Slack. Nothing here reaches the network."""

import json
import logging

import httpx
import pytest

from fibsem.acting import AGENT
from fibsem.alerts import Alerts, send_test
from fibsem.alerts.rules import ANSWER, Alert
from fibsem.alerts.slack import SlackChannel, SlackError, blocks
from fibsem.config import NotificationPreferences, UserPreferences
from fibsem.workflows.events import EventBuffer

TOKEN = "xoxb-test-token-not-real"


class FakeRequest:
    def __init__(self, request_id, what, status="pending"):
        self.status = status
        self._view = {
            "request_id": request_id,
            "reason": "to see the lamella",
            "actions": [{"what": what}],
        }

    def to_dict(self):
        return self._view


class FakeAgent:
    def __init__(self):
        self.request = None
        self.suggested = None
        self.asks = None
        self.pictures = {}

    def current(self):
        return self.request

    def suggestion(self):
        return self.suggested

    def asking(self):
        return self.asks

    def picture(self, kind):
        return self.pictures.get(kind)


class FakeSession:
    def __init__(self):
        self.event_buffer = EventBuffer()
        self.agent = FakeAgent()
        self.experiment = None


class FakeSlack:
    """Slack's Web API as far as posting goes: every call is kept."""

    def __init__(self, refuse_blocks=0, error=None):
        self.calls = []
        self.refuse_blocks = refuse_blocks
        self.error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        if request.url.host == "files.example":
            self.calls.append(("upload", request.content))
            return httpx.Response(200, text="OK")
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        body = (
            json.loads(request.content)
            if method == "chat.postMessage"
            else dict(httpx.QueryParams(request.content.decode()))
        )
        self.calls.append((method, body))
        if self.error:
            return httpx.Response(200, json={"ok": False, "error": self.error})
        if method == "files.getUploadURLExternal":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "upload_url": "https://files.example/up",
                    "file_id": "F1",
                },
            )
        if method == "chat.postMessage":
            if any(b["type"] == "image" for b in body["blocks"]) and self.refuse_blocks:
                self.refuse_blocks -= 1
                return httpx.Response(
                    200, json={"ok": False, "error": "invalid_blocks"}
                )
            return httpx.Response(
                200, json={"ok": True, "channel": "C123", "ts": "1.2"}
            )
        return httpx.Response(200, json={"ok": True})

    def posts(self):
        return [body for method, body in self.calls if method == "chat.postMessage"]


def _channel(slack):
    def make(token, channel, link=""):
        return SlackChannel(
            token,
            channel,
            link,
            client=httpx.Client(transport=httpx.MockTransport(slack)),
            sleep=lambda s: None,
        )

    return make


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _settings(**changes):
    settings = NotificationPreferences(
        enabled=True, channel="#microscope-alerts", link="http://localhost:8001/ui"
    )
    for key, value in changes.items():
        setattr(settings, key, value)
    return settings


def _alerts(session, slack, settings=None, clock=None, token=TOKEN):
    settings = settings or _settings()
    return Alerts(
        session,
        settings=lambda: settings,
        token=token,
        channel=_channel(slack),
        clock=clock or Clock(),
    )


def _message(session, text="I can't find the fiducial on 01. Can you look?"):
    session.event_buffer.append("note", {"text": text, "to": "operator"}, actor=AGENT)


def _failed(session, what="Mill fiducial"):
    session.event_buffer.append(
        "agent_called",
        {"action": what, "status": 500, "ok": False, "level": "drive"},
        actor=AGENT,
    )


def _headlines(slack):
    return [post["text"] for post in slack.posts()]


def test_the_agents_message_and_a_failure_are_sent_at_once():
    session, slack = FakeSession(), FakeSlack()
    alerts = _alerts(session, slack)
    _message(session)
    _failed(session)
    alerts.tick()
    assert _headlines(slack) == [
        "Message from the agent",
        "The agent's action failed: Mill fiducial",
    ]
    post = slack.posts()[0]
    assert post["channel"] == "#microscope-alerts"
    assert "fiducial on 01" in json.dumps(post["blocks"])
    button = post["blocks"][-1]["elements"][0]
    assert button["url"] == "http://localhost:8001/ui/agent"
    # Told once.
    alerts.tick()
    assert len(slack.posts()) == 2


def test_what_happened_before_it_started_is_not_sent():
    session, slack = FakeSession(), FakeSlack()
    _message(session)
    alerts = _alerts(session, slack)
    alerts.tick()
    assert slack.posts() == []


def test_an_ask_waits_and_is_dropped_when_answered():
    session, slack, clock = FakeSession(), FakeSlack(), Clock()
    alerts = _alerts(session, slack, clock=clock)
    session.agent.request = FakeRequest("r1", "Move the stage 2 µm in x")
    alerts.tick()
    clock.now += 4 * 60
    alerts.tick()
    assert slack.posts() == []
    session.agent.request = None  # answered at the PC
    clock.now += 2 * 60
    alerts.tick()
    assert slack.posts() == []

    session.agent.request = FakeRequest("r2", "Mill the fiducial")
    alerts.tick()
    clock.now += 5 * 60
    alerts.tick()
    alerts.tick()
    assert _headlines(slack) == ["The agent asks: Mill the fiducial"]
    assert "Waiting 5 min." in json.dumps(slack.posts()[0]["blocks"])


def test_a_run_held_on_a_decision_is_an_ask():
    session, slack, clock = FakeSession(), FakeSlack(), Clock()
    alerts = _alerts(session, slack, clock=clock, settings=_settings(wait_minutes=1))
    session.event_buffer.append(
        "proposal_held", {"item_name": "02-prime-yeti", "task_name": "Setup_Lamella"}
    )
    alerts.tick()
    clock.now += 61
    alerts.tick()
    assert _headlines(slack) == [
        "The run is waiting for an answer: Setup Lamella on 02-prime-yeti"
    ]
    assert slack.posts()[0]["blocks"][-1]["elements"][0]["url"].endswith("/experiment")


def test_a_finished_run_only_when_asked_for():
    session, slack = FakeSession(), FakeSlack()
    settings = _settings()
    alerts = _alerts(session, slack, settings=settings)
    session.event_buffer.append(
        "workflow_completed",
        {"experiment_name": "AutoLamella-1", "task_counts": {"completed": 7}},
    )
    alerts.tick()
    assert slack.posts() == []
    settings.run_finished = True
    session.event_buffer.append(
        "workflow_completed",
        {"experiment_name": "AutoLamella-1", "task_counts": {"completed": 7}},
    )
    alerts.tick()
    assert _headlines(slack) == ["Run finished: AutoLamella-1"]


def test_a_run_that_failed_is_sent():
    session, slack = FakeSession(), FakeSlack()
    alerts = _alerts(session, slack)
    session.event_buffer.append(
        "workflow_failed", {"error": "stage did not move", "tasks_remaining": 3}
    )
    alerts.tick()
    assert _headlines(slack) == ["The run stopped with an error"]


@pytest.mark.parametrize(
    "settings, token",
    [
        (_settings(enabled=False), TOKEN),
        (_settings(), ""),
        (_settings(channel=""), TOKEN),
        (_settings(messages=False), TOKEN),
    ],
)
def test_nothing_is_sent_unless_set_up(settings, token):
    session, slack = FakeSession(), FakeSlack()
    alerts = _alerts(session, slack, settings=settings, token=token)
    _message(session)
    alerts.tick()
    assert slack.posts() == []


def test_settings_that_cant_be_read_send_nothing():
    session, slack = FakeSession(), FakeSlack()

    def unreadable():
        raise OSError("no file")

    alerts = Alerts(session, settings=unreadable, token=TOKEN, channel=_channel(slack))
    _message(session)
    alerts.tick()
    assert slack.posts() == []


def test_turned_on_later_does_not_send_what_happened_meanwhile():
    session, slack = FakeSession(), FakeSlack()
    settings = _settings(enabled=False)
    alerts = _alerts(session, slack, settings=settings)
    _message(session, "first")
    alerts.tick()
    settings.enabled = True
    alerts.tick()
    _message(session, "second")
    alerts.tick()
    assert ["second" in json.dumps(p["blocks"]) for p in slack.posts()] == [True]


def test_it_reads_on_across_a_new_stream():
    session, slack = FakeSession(), FakeSlack()
    alerts = _alerts(session, slack)
    _message(session, "before the microscope connected")
    alerts.tick()
    session.event_buffer = EventBuffer()  # a microscope connected
    _message(session, "after")
    alerts.tick()
    assert len(slack.posts()) == 2


def test_an_image_is_uploaded_and_shown_when_asked_for():
    session, slack, clock = FakeSession(), FakeSlack(), Clock()
    alerts = _alerts(
        session, slack, clock=clock, settings=_settings(images=True, wait_minutes=0)
    )
    session.agent.request = FakeRequest("r1", "Mill the fiducial")
    session.agent.pictures["request"] = {"png": b"\x89PNG..."}
    alerts.tick()
    assert ("upload", b"\x89PNG...") in slack.calls
    image = [b for b in slack.posts()[0]["blocks"] if b["type"] == "image"]
    assert image == [
        {
            "type": "image",
            "slack_file": {"id": "F1"},
            "alt_text": "The agent asks: Mill the fiducial",
        }
    ]


def test_images_are_off_by_default():
    assert NotificationPreferences().images is False
    session, slack = FakeSession(), FakeSlack()
    alerts = _alerts(session, slack, settings=_settings(images=False, wait_minutes=0))
    session.agent.request = FakeRequest("r1", "Mill the fiducial")
    session.agent.pictures["request"] = {"png": b"\x89PNG..."}
    alerts.tick()
    assert [m for m, _ in slack.calls] == ["chat.postMessage"]


def test_an_image_slack_wont_show_goes_under_the_message():
    slack = FakeSlack(refuse_blocks=99)
    channel = _channel(slack)(TOKEN, "#microscope-alerts")
    assert channel.send(Alert("k", ANSWER, "Look"), b"png") == "C123"
    complete = [
        body for method, body in slack.calls if method == "files.completeUploadExternal"
    ]
    assert complete[-1]["channel_id"] == "C123" and complete[-1]["thread_ts"] == "1.2"


def test_the_agents_words_cant_make_links_or_mentions():
    text = json.dumps(
        blocks(Alert("k", ANSWER, "<!channel> look", ["<http://x|here> & there"]))
    )
    assert (
        "<!channel>" not in text
        and "&lt;!channel&gt;" in text
        and "&amp; there" in text
    )


def test_send_test_returns_the_channels_id():
    slack = FakeSlack()
    assert (
        send_test(_settings(enabled=False), token=TOKEN, channel=_channel(slack))
        == "C123"
    )
    assert _headlines(slack) == ["Test from fibsemOS"]
    with pytest.raises(ValueError, match="FIBSEM_SLACK_TOKEN"):
        send_test(_settings(), token="", channel=_channel(slack))
    with pytest.raises(SlackError, match="not_in_channel"):
        send_test(
            _settings(),
            token=TOKEN,
            channel=_channel(FakeSlack(error="not_in_channel")),
        )


def test_the_token_is_never_logged(caplog):
    session, slack = FakeSession(), FakeSlack(error="not_in_channel")
    alerts = _alerts(session, slack)
    _message(session)
    with caplog.at_level(logging.DEBUG):
        alerts.tick()
    assert "not_in_channel" in caplog.text
    assert TOKEN not in caplog.text


def test_the_settings_round_trip_and_never_hold_the_token():
    preferences = UserPreferences()
    preferences.install.notifications.channel = "#microscope-alerts"
    again = UserPreferences.from_dict(preferences.to_dict())
    assert again.install.notifications.channel == "#microscope-alerts"
    assert again.install.notifications.enabled is False
    assert "token" not in json.dumps(preferences.to_dict()["install"]["notifications"])
