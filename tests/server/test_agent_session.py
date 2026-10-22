"""The agent's session as the agent reports it: ``/app/agent/session``."""

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server.agent_session import (  # noqa: E402
    SESSION_BUDGET,
    SESSION_ENDED,
    SESSION_GOAL,
    SESSION_STARTED,
    build_agent_session_router,
    report,
)


@pytest.fixture
def server():
    """The route on its own, with what it puts on the record."""
    recorded = []
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.include_router(
        build_agent_session_router(lambda kind, data: recorded.append((kind, data)))
    )
    client = TestClient(
        app, headers={"Authorization": f"Bearer {app.state.auth.token}"}
    )
    return app.state.auth, client, recorded


def _kinds(recorded):
    return [kind for kind, _ in recorded]


def test_a_session_starts_changes_goal_and_ends_on_the_record(server):
    auth, client, recorded = server
    first = {"session_id": "s1", "model": "m", "goal": "mill 4", "budget_usd": 5}
    assert client.post("/app/agent/session", json=first).status_code == 200
    client.post("/app/agent/session", json={"spent_usd": 0.5, "turns": 1})
    client.post("/app/agent/session", json={"goal": "mill 6"})
    response = client.post(
        "/app/agent/session", json={"ended": True, "reason": "budget spent"}
    )
    assert _kinds(recorded) == [SESSION_STARTED, SESSION_GOAL, SESSION_ENDED]
    session = response.json()["session"]
    assert session["goal"] == "mill 6" and session["spent_usd"] == 0.5
    assert session["ended"] and session["reason"] == "budget spent"
    assert client.get("/app/agent/session").json()["session"] == session
    assert auth.agent_session == session


def test_the_spend_reaching_80_percent_of_the_budget_goes_on_the_record_once(
    server,
):
    _, client, recorded = server
    client.post("/app/agent/session", json={"session_id": "s1", "budget_usd": 5})
    for spent in (3.9, 4.0, 4.5):
        client.post("/app/agent/session", json={"spent_usd": spent})
    assert _kinds(recorded) == [SESSION_STARTED, SESSION_BUDGET]
    assert recorded[1][1]["spent_usd"] == 4.0
    # A new session warns again.
    again = {"session_id": "s2", "budget_usd": 5, "spent_usd": 4.5}
    client.post("/app/agent/session", json=again)
    assert _kinds(recorded)[2:] == [SESSION_STARTED, SESSION_BUDGET]


def test_a_new_session_id_or_a_report_after_the_end_starts_a_new_session(server):
    _, client, recorded = server
    client.post("/app/agent/session", json={"session_id": "s1", "goal": "a"})
    client.post("/app/agent/session", json={"session_id": "s2"})
    session = client.get("/app/agent/session").json()["session"]
    assert session["session_id"] == "s2" and "goal" not in session
    client.post("/app/agent/session", json={"ended": True})
    client.post("/app/agent/session", json={"session_id": "s2"})
    assert _kinds(recorded) == [SESSION_STARTED] * 2 + [SESSION_ENDED, SESSION_STARTED]


@pytest.mark.parametrize(
    "body",
    [
        {"cost": 1},
        {"budget_usd": -1},
        {"spent_usd": True},
        {"turns": 1.5},
        {"ended": "yes"},
        {"goal": "x" * 501},
        {"model": 3},
    ],
)
def test_a_bad_report_is_refused_and_records_nothing(server, body):
    auth, client, recorded = server
    response = client.post("/app/agent/session", json=body)
    assert response.status_code == 422
    assert auth.agent_session is None and recorded == []


def test_reading_and_reporting_need_the_agent_token_and_work_at_watch(server):
    auth, client, _ = server
    anonymous = TestClient(client.app)
    assert anonymous.get("/app/agent/session").status_code == 401
    auth.set_level(AgentLevel.OFF)
    assert client.post("/app/agent/session", json={}).status_code == 403
    auth.set_level(AgentLevel.WATCH)
    assert client.post("/app/agent/session", json={}).status_code == 200


def test_a_spend_that_is_not_a_number_is_refused():
    from fastapi import HTTPException

    auth = AuthConfig.generate(level=AgentLevel.WATCH)
    with pytest.raises(HTTPException):
        report(auth, {"spent_usd": float("inf")})
    assert auth.agent_session is None
