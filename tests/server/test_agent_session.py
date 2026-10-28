"""The agent's session as the agent reports it: ``/app/agent/session``."""

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig
from fibsem.server.agent_session import (
    SESSION_BUDGET,
    SESSION_CONSTRAINTS,
    SESSION_ENDED,
    SESSION_GOAL,
    SESSION_STARTED,
    build_agent_session_router,
    deadline,
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
        {"tokens": {"input": -1}},
        {"tokens": {"prompt": 1}},
        {"context": {"used": 1.5}},
        {"context": 412000},
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


def test_the_checklist_is_state_and_the_constraints_go_on_the_record(server):
    _, client, recorded = server
    client.post("/app/agent/session", json={"session_id": "s1"})
    checklist = [
        {"text": "Polish 01", "state": "done"},
        {"text": "Image each lamella after polishing"},
        {"text": "Keep the ion beam off between lamellae", "state": "always"},
    ]
    session = client.post("/app/agent/session", json={"checklist": checklist})
    assert session.json()["session"]["checklist"][1]["state"] == "open"
    constraints = [{"text": "Finish", "by": "2026-10-24T07:00+11:00"}, {"text": "x"}]
    session = client.post("/app/agent/session", json={"constraints": constraints})
    client.post("/app/agent/session", json={"constraints": constraints})
    assert _kinds(recorded) == [SESSION_STARTED, SESSION_CONSTRAINTS]
    kept = session.json()["session"]
    assert kept["constraints"][0] == {"text": "Finish", "by": "2026-10-24T07:00+11:00"}
    assert len(kept["checklist"]) == 3
    # A new session starts with neither.
    fresh = client.post("/app/agent/session", json={"session_id": "s2"}).json()
    assert "checklist" not in fresh["session"] and "constraints" not in fresh["session"]


def test_a_clock_time_is_its_next_occurrence():
    now = datetime(2026, 10, 23, 23, 30).astimezone()
    assert deadline("07:00", now) == now.replace(hour=7, minute=0) + timedelta(days=1)
    assert deadline("23:45", now) == now.replace(minute=45)


@pytest.mark.parametrize(
    "body",
    [
        {"checklist": [{"text": "x", "state": "maybe"}]},
        {"checklist": [{"text": ""}]},
        {"checklist": "x"},
        {"constraints": [{"text": "x", "by": "tomorrow"}]},
        {"constraints": [{"text": "x" * 201}]},
        {"hooks": [{"when": "site", "say": "x"}]},
        {"hooks": "x"},
    ],
)
def test_a_bad_checklist_constraint_or_reminder_is_refused(server, body):
    _, client, recorded = server
    assert client.post("/app/agent/session", json=body).status_code == 422
    assert recorded == []


def test_a_session_without_an_id_takes_the_first_one_reported(server):
    _, client, recorded = server
    client.post("/app/agent/session", json={"checklist": [{"text": "Polish 01"}]})
    session = client.post("/app/agent/session", json={"session_id": "s1"}).json()
    assert session["session"]["checklist"] == [{"text": "Polish 01", "state": "open"}]
    assert _kinds(recorded) == [SESSION_STARTED]


def test_the_profile_and_its_reminders_are_kept(server):
    _, client, _ = server
    hooks = [{"when": "site", "label": "At a site", "say": "Check the film."}]
    body = {"profile": "night", "profile_hash": "abc", "hooks": hooks}
    session = client.post("/app/agent/session", json=body).json()["session"]
    assert (session["profile"], session["profile_hash"]) == ("night", "abc")
    assert session["hooks"] == hooks


def test_the_tokens_and_the_context_are_kept(server):
    auth, client, _ = server
    tokens = {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}
    context = {"used": 412000, "window": 1000000, "compact_at": 970000}
    client.post("/app/agent/session", json={"tokens": tokens, "context": context})
    session = client.get("/app/agent/session").json()["session"]
    assert (session["tokens"], session["context"]) == (tokens, context)
