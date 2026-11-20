"""The agent's session as the agent reports it: ``/app/agent/session``."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.instrument_id import instrument_id
from fibsem.server import AuthConfig
from fibsem.server.agent.session import (
    CONTEXT_RESET_ASKED,
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
        {"tokens": {"input": -1}},
        {"tokens": {"prompt": 1}},
        {"context": {"used": 1.5}},
        {"context": 412000},
        {"playbook": {"hash": "abc"}},
        {"playbook": {"name": "x", "skills": [{"skill": "y", "state": "half"}]}},
        {"playbook": {"name": "x", "extra": 1}},
    ],
)
def test_a_bad_report_is_refused_and_records_nothing(server, body):
    auth, client, recorded = server
    response = client.post("/app/agent/session", json=body)
    assert response.status_code == 422
    assert auth.agent_session is None and recorded == []


def test_the_sessions_playbook_is_reported_and_none_clears_it(server):
    _, client, _ = server
    playbook = {
        "name": "autolamella",
        "hash": "abc",
        "chosen_by": "the operator",
        "skills": [
            {"skill": "autolamella-polishing", "state": "done", "summary": "180 nm"}
        ],
    }
    hooks = [{"when": "setup_placement", "label": "Placing", "say": "Check."}]
    body = {"session_id": "s1", "playbook": playbook, "hooks": hooks}
    client.post("/app/agent/session", json=body)
    session = client.get("/app/agent/session").json()["session"]
    assert session["playbook"] == playbook
    client.post("/app/agent/session", json={"playbook": {}})
    assert client.get("/app/agent/session").json()["session"]["playbook"] == {}


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


def test_the_checklist_is_state_and_a_new_session_starts_without_one(server):
    _, client, recorded = server
    client.post("/app/agent/session", json={"session_id": "s1"})
    checklist = [
        {"text": "Polish 01", "state": "done"},
        {"text": "Image each lamella after polishing"},
        {"text": "Keep the ion beam off between lamellae", "state": "always"},
    ]
    session = client.post("/app/agent/session", json={"checklist": checklist})
    client.post("/app/agent/session", json={"checklist": checklist})
    assert _kinds(recorded) == [SESSION_STARTED]
    kept = session.json()["session"]
    assert kept["checklist"][1]["state"] == "open" and len(kept["checklist"]) == 3
    fresh = client.post("/app/agent/session", json={"session_id": "s2"}).json()
    assert "checklist" not in fresh["session"]


@pytest.mark.parametrize(
    "body",
    [
        {"checklist": [{"text": "x", "state": "maybe"}]},
        {"checklist": [{"text": ""}]},
        {"checklist": "x"},
        # Constraints are the plan's Rules now.
        {"constraints": [{"text": "x"}]},
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


def test_the_reminders_are_kept(server):
    _, client, _ = server
    hooks = [{"when": "site", "label": "At a site", "say": "Check the film."}]
    session = client.post("/app/agent/session", json={"hooks": hooks}).json()["session"]
    assert session["hooks"] == hooks


def test_the_tokens_and_the_context_are_kept(server):
    auth, client, _ = server
    tokens = {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}
    context = {"used": 412000, "window": 1000000, "compact_at": 970000}
    client.post("/app/agent/session", json={"tokens": tokens, "context": context})
    session = client.get("/app/agent/session").json()["session"]
    assert (session["tokens"], session["context"]) == (tokens, context)


def test_a_report_the_harness_writes_reads_and_unset_fields_stay_out(server):
    from fibsem.ai.agent.report import SessionReport, Tokens, plain

    _, client, _ = server
    body = plain(SessionReport(session_id="s1", tokens=Tokens(input=3)))
    assert body == {"session_id": "s1", "tokens": {"input": 3}}
    session = client.post("/app/agent/session", json=body).json()["session"]
    assert session["tokens"] == {"input": 3} and "goal" not in session


def test_the_session_names_the_instrument_and_the_open_experiment(server):
    from types import SimpleNamespace

    auth, client, recorded = server
    microscope = SimpleNamespace(
        system=SimpleNamespace(info=SimpleNamespace(serial_number="SN-1")),
        experiment=SimpleNamespace(id="exp-1"),
    )
    client.app.state.served = SimpleNamespace(current=microscope)
    session = client.post("/app/agent/session", json={"session_id": "s1"}).json()
    assert session["session"]["instrument_serial"] == "SN-1"
    assert session["session"]["instrument_id"] == instrument_id()
    assert session["session"]["experiment_id"] == "exp-1"
    assert recorded[0][1]["experiment_id"] == "exp-1"

    microscope.experiment = None
    microscope.system.info.serial_number = "Unknown"
    session = client.post("/app/agent/session", json={"turns": 1}).json()["session"]
    assert "experiment_id" not in session and "instrument_serial" not in session
    # The id is this PC's, whatever the microscope reports.
    assert session["instrument_id"] == instrument_id()


def test_an_instrument_id_is_made_once_and_kept(tmp_path):
    path = tmp_path / "state" / "instrument.json"
    made = instrument_id(path)
    assert made and instrument_id(path) == made
    path.write_text("not json")
    # An unreadable file is left alone, and no id is made up for one run.
    assert instrument_id(path) is None
    assert path.read_text() == "not json"


def test_an_agent_cannot_report_the_ids_itself(server):
    _, client, _ = server
    body = {"session_id": "s1", "experiment_id": "made-up"}
    assert client.post("/app/agent/session", json=body).status_code == 422


def test_the_operator_sets_the_budget_and_the_agents_reports_keep_it(server):
    auth, client, recorded = server
    auth.operator_token = "op-token"
    operator = TestClient(client.app, headers={"Authorization": "Bearer op-token"})
    put = operator.put
    assert put("/app/agent/budget", json={"budget_usd": 600}).status_code == 409
    client.post("/app/agent/session", json={"session_id": "s1", "budget_usd": 500})
    # The agent can't set its own.
    assert client.put("/app/agent/budget", json={"budget_usd": 900}).status_code == 403
    for bad in ({}, {"budget_usd": 0}, {"budget_usd": True}, {"budget_usd": "9"}):
        assert put("/app/agent/budget", json=bad).status_code == 422
    response = put("/app/agent/budget", json={"budget_usd": 600})
    assert response.status_code == 200
    assert response.json()["session"]["budget_set_usd"] == 600
    # The harness's next report, with the budget it had, keeps the operator's.
    body = {"budget_usd": 500, "spent_usd": 1}
    session = client.post("/app/agent/session", json=body).json()["session"]
    assert session["budget_set_usd"] == 600 and session["budget_usd"] == 500
    # A report can't set it, and a new session starts without it.
    body = {"budget_set_usd": 900}
    assert client.post("/app/agent/session", json=body).status_code == 422
    fresh = client.post("/app/agent/session", json={"session_id": "s2"}).json()
    assert "budget_set_usd" not in fresh["session"]
    assert _kinds(recorded) == [SESSION_STARTED, SESSION_STARTED]


def test_the_operator_asks_for_a_fresh_context_on_the_record(server):
    auth, client, recorded = server
    auth.operator_token = "op-token"
    operator = TestClient(client.app, headers={"Authorization": "Bearer op-token"})
    assert operator.post("/app/agent/context-reset").status_code == 409
    body = {"session_id": "s1", "contexts": 2}
    client.post("/app/agent/session", json=body)
    # The agent can't ask for itself here: it has context_reset.
    assert client.post("/app/agent/context-reset").status_code == 403
    response = operator.post("/app/agent/context-reset")
    assert response.status_code == 200
    session = response.json()["session"]
    assert session["reset_asked_in"] == 2 and session["reset_asked_at"]
    assert recorded[-1] == (CONTEXT_RESET_ASKED, {"session_id": "s1", "context": 2})
    # The harness's next report keeps it, and can't set it.
    session = client.post("/app/agent/session", json={"contexts": 3}).json()
    assert session["session"]["reset_asked_in"] == 2
    body = {"reset_asked_in": 3}
    assert client.post("/app/agent/session", json=body).status_code == 422


def test_the_context_reports_its_parts():
    body = {
        "session_id": "s1",
        "context": {
            "used": 50000,
            "parts": [{"name": "Messages", "tokens": 30000}],
        },
    }
    session, _ = report(AuthConfig.generate(level=AgentLevel.WATCH), body)
    assert session["context"]["parts"] == [{"name": "Messages", "tokens": 30000}]


def test_the_playbook_file_says_when_it_changed_since_the_session_started(
    server, tmp_path, monkeypatch
):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    folder = tmp_path / "home" / "knowledge" / "playbooks"
    folder.mkdir(parents=True)
    (folder / "own.yaml").write_text("extends: autolamella\n")
    from fibsem.ai.agent import playbooks, sources

    loaded = playbooks.load("own", sources.find_root())
    _, client, _ = server
    assert client.get("/app/agent/playbook-file").json() == {
        "name": None,
        "hash": None,
        "edited": False,
    }
    client.post(
        "/app/agent/session",
        json={"session_id": "s1", "playbook_about": loaded.about()},
    )
    assert client.get("/app/agent/playbook-file").json()["edited"] is False
    (folder / "own.yaml").write_text("extends: autolamella\nrules: [Stop at 150 nm.]\n")
    answer = client.get("/app/agent/playbook-file").json()
    assert answer["edited"] is True and answer["hash"] != loaded.hash
    (folder / "own.yaml").unlink()
    assert client.get("/app/agent/playbook-file").json()["hash"] is None
