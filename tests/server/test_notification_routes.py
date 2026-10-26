"""The operator's routes for Slack notifications, and ``fibsem-cli notify``:
settings are this PC's, the token is never served or taken, and the agent
can't reach them. Slack is faked; nothing reaches the network."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fibsem import cli  # noqa: E402
from fibsem import config as cfg  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.alerts import service  # noqa: E402
from fibsem.alerts.slack import SlackError  # noqa: E402
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server import notifications as routes  # noqa: E402

AGENT_TOKEN = "agent-token"
OPERATOR_TOKEN = "operator-token"
SLACK_TOKEN = "xoxb-test-token-not-real"


@pytest.fixture(autouse=True)
def preferences(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))
    monkeypatch.setenv(service.TOKEN_ENV, SLACK_TOKEN)


class FakeChannel:
    sent = []
    error = None

    def __init__(self, token, channel, link=""):
        self.token, self.channel, self.link = token, channel, link

    def send(self, alert, image=None):
        if FakeChannel.error:
            raise SlackError("chat.postMessage", FakeChannel.error)
        FakeChannel.sent.append((self.token, self.channel, self.link, alert.headline))
        return "C123"

    def close(self):
        pass


@pytest.fixture(autouse=True)
def slack(monkeypatch):
    FakeChannel.sent, FakeChannel.error = [], None
    real = service.send_test

    def send_test(settings, token=None, channel=None, link=""):
        return real(settings, token, FakeChannel, link)

    monkeypatch.setattr(routes, "send_test", send_test)
    monkeypatch.setattr(service, "SlackChannel", FakeChannel)
    return FakeChannel


@pytest.fixture
def client():
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token=AGENT_TOKEN)
    auth.operator_token = OPERATOR_TOKEN
    app = FastAPI()
    app.state.auth = auth
    app.include_router(routes.build_notifications_router())
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {OPERATOR_TOKEN}"
        yield client


def _saved():
    return cfg.load_user_preferences().install.notifications


def test_the_settings_say_whether_the_token_is_set_never_what_it_is(client):
    answer = client.get("/app/notifications").json()
    assert answer["token_set"] is True and answer["token_from"] == "FIBSEM_SLACK_TOKEN"
    assert answer["settings"]["enabled"] is False
    assert SLACK_TOKEN not in str(answer)


def test_the_operator_changes_a_setting_and_the_rest_stay(client):
    answer = client.put(
        "/app/notifications", json={"channel": " #alerts ", "enabled": True}
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["settings"]["channel"] == "#alerts"
    client.put("/app/notifications", json={"wait_minutes": 10})
    saved = _saved()
    assert (saved.channel, saved.enabled, saved.wait_minutes) == ("#alerts", True, 10)


@pytest.mark.parametrize(
    "body, error",
    [
        ({"token": "xoxb-1"}, "unknown_field"),
        ({"enabled": "yes"}, "wrong_type"),
        ({"wait_minutes": True}, "wrong_type"),
        ({"wait_minutes": -1}, "out_of_range"),
    ],
)
def test_a_bad_setting_is_refused(client, body, error):
    answer = client.put("/app/notifications", json=body)
    assert answer.status_code == 422
    assert answer.json()["detail"]["error_type"] == error
    assert _saved() == cfg.NotificationPreferences()


def test_send_test_keeps_the_channels_id_and_a_new_name_drops_it(client, slack):
    client.put("/app/notifications", json={"channel": "#alerts"})
    answer = client.post("/app/notifications/test")
    assert answer.json() == {"channel_id": "C123"}
    assert slack.sent == [
        (SLACK_TOKEN, "#alerts", "http://testserver/ui", "Test from fibsemOS")
    ]
    assert _saved().channel_id == "C123"
    # The ID is used from now on, and a channel named anew is found again.
    client.post("/app/notifications/test")
    assert slack.sent[-1][1] == "C123"
    client.put("/app/notifications", json={"channel": "#other"})
    assert _saved().channel_id == ""


def test_send_test_says_what_slack_refused(client, slack):
    client.put("/app/notifications", json={"channel": "#alerts"})
    slack.error = "not_in_channel"
    answer = client.post("/app/notifications/test")
    assert answer.status_code == 502
    assert answer.json()["detail"]["slack_error"] == "not_in_channel"


def test_send_test_without_a_token_says_how_to_set_it(client, monkeypatch):
    monkeypatch.delenv(service.TOKEN_ENV)
    client.put("/app/notifications", json={"channel": "#alerts"})
    answer = client.post("/app/notifications/test")
    assert answer.status_code == 422
    assert "FIBSEM_SLACK_TOKEN" in answer.json()["detail"]["message"]


@pytest.mark.parametrize("token", [AGENT_TOKEN, "nobody"])
def test_only_the_operator_reaches_them(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/app/notifications", headers=headers).status_code in (401, 403)
    assert client.put(
        "/app/notifications", json={"enabled": True}, headers=headers
    ).status_code in (401, 403)
    assert client.post("/app/notifications/test", headers=headers).status_code in (
        401,
        403,
    )
    assert _saved().enabled is False


def _cli(*argv):
    with pytest.raises(SystemExit) as ended:
        cli.main(["notify", *argv])
    return ended.value.code


def test_the_cli_sets_up_tests_and_turns_off(capsys, slack):
    assert _cli("setup", "#alerts") == 0
    saved = _saved()
    assert (saved.enabled, saved.channel, saved.channel_id) == (True, "#alerts", "C123")
    assert _cli("status") == 0
    assert (
        "on. Channel: #alerts (C123). FIBSEM_SLACK_TOKEN: set."
        in capsys.readouterr().out
    )
    assert _cli("off") == 0
    assert _saved().enabled is False
    assert SLACK_TOKEN not in capsys.readouterr().out
