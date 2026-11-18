"""A time limit on the agent level: Assist or Drive set until a time drops to
Watch then, says so on the record and in the refusal, and a late timer never
lets a call through."""

import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent import watch
from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig, build_server
from fibsem.server import auth as auth_module
from fibsem.workflows.events import payloads
from tests.server.test_operator_routes import server  # noqa: F401

TOKEN = "agent"
HFW = "/microscope/devices/electron/hfw"


@pytest.fixture
def clock(monkeypatch):
    """The auth module's clock, moved by hand."""
    now = [time.time()]
    monkeypatch.setattr(auth_module, "time", _Clock(now))
    return now


class _Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now[0]

    def monotonic(self):
        return time.monotonic()


def test_drive_until_a_time_drops_to_watch_then(clock):
    heard = []
    auth = AuthConfig(token=TOKEN)
    auth.set_level(
        AgentLevel.DRIVE, until=clock[0] + 60, on_expired=lambda: heard.append(1)
    )
    assert auth.level is AgentLevel.DRIVE and auth.level_until == clock[0] + 60

    assert not auth.expire_if_due()
    clock[0] += 61
    assert auth.expire_if_due()
    assert auth.level is AgentLevel.WATCH and auth.expired and auth.level_until is None
    assert heard == [1]
    assert not auth.expire_if_due()  # once


def test_setting_the_level_again_or_stopping_clears_the_limit(clock):
    auth = AuthConfig(token=TOKEN)
    auth.set_level(AgentLevel.DRIVE, until=clock[0] + 60)
    auth.set_level(AgentLevel.DRIVE)
    clock[0] += 61
    assert not auth.expire_if_due() and auth.level is AgentLevel.DRIVE

    auth.set_level(AgentLevel.DRIVE, until=clock[0] + 60)
    auth.stop_agent()
    assert auth.level_until is None and auth.stopped and not auth.expired


def test_a_limit_is_only_for_an_acting_level_in_the_future(clock):
    auth = AuthConfig(token=TOKEN)
    with pytest.raises(ValueError, match="only for Assist or Drive"):
        auth.set_level(AgentLevel.WATCH, until=clock[0] + 60)
    with pytest.raises(ValueError, match="already passed"):
        auth.set_level(AgentLevel.DRIVE, until=clock[0] - 1)
    assert auth.level is AgentLevel.WATCH


def test_a_call_after_the_limit_is_refused_though_the_timer_has_not_fired(
    clock, microscope
):
    auth = AuthConfig(token=TOKEN)
    auth.set_level(AgentLevel.DRIVE, until=clock[0] + 60)
    app = build_server(microscope, auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        hfw = client.get(HFW).json()["value"]
        assert client.put(HFW, json={"value": hfw}).status_code == 200

        clock[0] += 61
        refused = client.put(HFW, json={"value": hfw})
        assert refused.status_code == 403
        detail = refused.json()["detail"]
        assert detail["level"] == "watch"
        assert "time limit" in detail["message"]
        assert client.get(HFW).status_code == 200  # still sees


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def _level_events(server):  # noqa: F811
    events = server.get("/app/events", headers=server.operator).json()["events"]
    return [e["payload"] for e in events if e["kind"] == "agent_level_set"]


def test_the_operator_gives_drive_a_time_limit_and_it_runs_out(server):  # noqa: F811
    until = _iso(time.time() + 2)
    answer = server.put(
        "/app/agent/level",
        headers=server.operator,
        json={"level": "drive", "until": until},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"level": "drive", "until": until}
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["level"] == "drive" and state["level_until"] == until
    assert _level_events(server)[-1] == {
        "previous": "watch",
        "level": "drive",
        "until": until,
    }

    deadline = time.monotonic() + 10
    while server.config.level is not AgentLevel.WATCH:
        assert time.monotonic() < deadline, "the level never dropped"
        time.sleep(0.05)
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["level"] == "watch" and state["level_expired"]
    assert state["level_until"] is None
    assert _level_events(server)[-1] == {
        "previous": "drive",
        "level": "watch",
        "expired": True,
    }


@pytest.mark.parametrize(
    "body, status, error",
    [
        ({"level": "drive", "until": "after lunch"}, 422, "bad_until"),
        ({"level": "watch", "until": "2099-01-01T00:00:00Z"}, 409, "level_refused"),
        ({"level": "drive", "until": "2000-01-01T00:00:00Z"}, 409, "level_refused"),
    ],
)
def test_a_limit_that_cant_be_kept_is_refused(server, body, status, error):  # noqa: F811
    answer = server.put("/app/agent/level", headers=server.operator, json=body)
    assert answer.status_code == status, answer.text
    assert answer.json()["detail"]["error_type"] == error
    assert server.config.level is AgentLevel.WATCH


def test_the_watch_says_the_limit_and_that_it_ran_out():
    given = payloads.LevelSet(level="drive", previous="watch", until="14:30")
    assert watch._level_line(given).endswith("until 14:30")
    ran_out = payloads.LevelSet(level="watch", previous="drive", expired=True)
    assert "ran out" in watch._level_line(ran_out)
    assert watch._lowered(ran_out)
