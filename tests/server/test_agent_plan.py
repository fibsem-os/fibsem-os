"""The agent's plan for its session, plan.md: ``/app/agent/plan``."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent import watch
from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig
from fibsem.server.agent_plan import (
    PLAN_EDITED,
    SAMPLE_SET,
    PlanStore,
    build_agent_plan_router,
    parse,
    render,
)
from fibsem.server.agent_session import build_agent_session_router

OPERATOR = "operator-token-for-tests"


@pytest.fixture
def server(tmp_path):
    """The plan and session routes, an open experiment, what goes on the record."""
    recorded = []
    experiment = tmp_path / "grid-1"
    experiment.mkdir()
    opened = {"path": str(experiment)}
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.state.auth.operator_token = OPERATOR
    app.state.agent_plan = PlanStore(lambda: opened["path"])

    def emit(kind, payload, actor=None):
        recorded.append((kind, payload, actor))

    app.include_router(build_agent_session_router(lambda kind, data: None))
    app.include_router(build_agent_plan_router(lambda: opened["path"], emit))
    agent = TestClient(app, headers={"Authorization": f"Bearer {app.state.auth.token}"})
    operator = TestClient(app, headers={"Authorization": f"Bearer {OPERATOR}"})
    agent.post("/app/agent/session", json={"session_id": "s1"})
    return agent, operator, recorded, experiment, opened


def section(client, name, text):
    return client.post("/app/agent/plan/section", json={"section": name, "text": text})


def test_the_agent_writes_its_plan_a_section_at_a_time_beside_the_experiment(server):
    agent, _, recorded, experiment, _ = server
    assert agent.get("/app/agent/plan").json()["markdown"] == ""
    section(
        agent,
        "steps",
        "### Sites\n- [x] Pick sites\n- [ ] Setup\n\n### Polish\n- [ ] 01",
    )
    answer = section(agent, "Brief", "Sample: yeast\nWhere do the granules sit?").json()
    assert answer["sample"] == "yeast"
    assert [s["name"] for s in answer["sections"]] == ["Steps", "Brief"]
    assert answer["steps"] == [
        {"group": "Sites", "text": "Pick sites", "done": True},
        {"group": "Sites", "text": "Setup", "done": False},
        {"group": "Polish", "text": "01", "done": False},
    ]
    assert answer["done"] == 1 and answer["updated_by"] == "agent"
    kept = (experiment / "agent" / "plan-s1.md").read_text(encoding="utf-8")
    assert kept.index("## Brief") < kept.index("## Steps"), "the default order"
    assert kept == answer["markdown"]
    assert [(k, p["sample"], a) for k, p, a in recorded] == [
        (SAMPLE_SET, "yeast", "agent")
    ]
    # Empty text removes a section; a heading inside one is refused.
    assert "Steps" not in [
        s["name"] for s in section(agent, "Steps", "").json()["sections"]
    ]
    assert section(agent, "Rules", "## Rules\n- x").status_code == 422


def test_the_sample_is_written_once_then_only_the_operator_changes_it(server):
    agent, operator, recorded, _, _ = server
    section(agent, "Brief", "Sample: yeast\nThe question.")
    refused = section(agent, "Brief", "Sample: HeLa\nThe question.")
    assert (
        refused.status_code == 409
        and refused.json()["detail"]["error_type"] == "sample_set"
    )
    assert section(agent, "Brief", "The question, no sample line.").status_code == 409
    assert (
        section(agent, "Brief", "Sample: yeast\nA better question.").status_code == 200
    )

    edited = operator.put(
        "/app/agent/plan",
        json={"markdown": "# Plan\n\n## Brief\n\nSample: HeLa\nA better question.\n"},
    ).json()
    assert edited["sample"] == "HeLa" and edited["updated_by"] == "operator"
    kinds = [(k, a) for k, _, a in recorded]
    assert kinds == [
        (SAMPLE_SET, "agent"),
        (SAMPLE_SET, "operator"),
        (PLAN_EDITED, "operator"),
    ]
    assert recorded[-1][1]["sections"] == ["Brief"]


def test_an_operators_edit_wakes_the_agent_and_the_agents_own_do_not(server):
    agent, operator, recorded, _, _ = server
    section(agent, "Rules", "- Keep the ion beam off between lamellae")
    assert not [k for k, _, _ in recorded if k == PLAN_EDITED]
    section(
        operator, "Rules", "- Keep the ion beam off between lamellae\n- Stop at 180 nm"
    )
    event = {"kind": PLAN_EDITED, "payload": recorded[-1][1], "seq": 1}
    printed = []
    watch._print_events([event], 0, printed.append)
    assert printed == ["PLAN EDITED by the operator: Rules"]


def test_the_sessions_checklist_is_read_from_the_plan(server):
    agent, _, _, _, _ = server
    agent.post(
        "/app/agent/session",
        json={"checklist": [{"text": "old way", "state": "open"}]},
    )
    assert (
        agent.get("/app/agent/session").json()["session"]["checklist"][0]["text"]
        == "old way"
    )
    section(agent, "Rules", "- Stop at 180 nm")
    section(agent, "Steps", "- [x] Sites\n- [ ] Polish")
    assert agent.get("/app/agent/session").json()["session"]["checklist"] == [
        {"text": "Stop at 180 nm", "state": "always"},
        {"text": "Sites", "state": "done"},
        {"text": "Polish", "state": "open"},
    ]


def test_a_restarted_app_reads_the_plan_back_and_none_is_kept_without_a_session(
    server, tmp_path
):
    agent, _, _, experiment, opened = server
    section(agent, "Brief", "Sample: yeast")
    again = PlanStore(lambda: opened["path"])
    assert again.current("s1") == {"Brief": "Sample: yeast"}
    assert again.current("s2") == {}
    agent.post("/app/agent/session", json={"ended": True})
    assert section(agent, "Brief", "x").json()["detail"]["error_type"] == "no_session"

    # No experiment open: held in memory only.
    opened["path"] = None
    agent.post("/app/agent/session", json={"session_id": "s3"})
    assert section(agent, "Progress", "Started.").status_code == 200
    assert agent.get("/app/agent/plan").json()["sections"] == [
        {"name": "Progress", "text": "Started."}
    ]
    assert not list(experiment.glob("agent/plan-s3.md"))


def test_a_plan_reads_back_as_it_was_written():
    sections = {"Brief": "Sample: yeast\nQ", "Steps": "### A\n- [ ] one", "Notes": "x"}
    assert parse(render(sections)) == sections
    assert list(parse("# Plan\n\n## notes\nx\n## brief\ny\n")) == ["notes", "Brief"]
    own = render({"Summary": "s", "Results": "r", "Brief": "b"})
    assert list(parse(own)) == ["Brief", "Results", "Summary"]
