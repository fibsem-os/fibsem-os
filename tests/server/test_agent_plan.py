"""The agent's plan for its session, plan.md: ``/app/agent/plan``."""

from datetime import datetime, timedelta

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
    dated,
    deadline,
    parse,
    render,
    rule_time,
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


def test_a_rule_with_a_time_is_a_deadline_saved_with_its_date(server, experiment_s1):
    agent, operator, _, _, _ = server
    section(
        agent,
        "Rules",
        "- Finish by 07:00\n- Don't change the milling currents\n"
        "- Coat by **2026-11-06 21:30**",
    )
    when = deadline("07:00")
    finish = f"Finish by {when:%Y-%m-%d %H:%M}"
    # The bare time is written out with its date, in the plan and its file.
    assert (
        agent.get("/app/agent/plan")
        .json()["sections"][0]["text"]
        .startswith(f"- {finish}\n")
    )
    assert finish in experiment_s1.read_text()
    found = agent.get("/app/agent/session").json()["session"]["deadlines"]
    assert found == [
        {"text": finish, "by": when.isoformat(timespec="minutes")},
        {"text": "Coat by **2026-11-06 21:30**", "by": found[1]["by"]},
    ]
    assert found[1]["by"].startswith("2026-11-06T21:30")
    # The report the harness posts answers with them too, and the operator's
    # edits are dated the same way.
    reported = agent.post("/app/agent/session", json={"turns": 1}).json()["session"]
    assert reported["deadlines"] == found
    operator.put("/app/agent/plan", json={"markdown": "## Rules\n\n- Done by 23:59"})
    last = deadline("23:59")
    assert agent.get("/app/agent/session").json()["session"]["deadlines"] == [
        {
            "text": f"Done by {last:%Y-%m-%d %H:%M}",
            "by": last.isoformat(timespec="minutes"),
        }
    ]
    # Rules without a time give none.
    section(agent, "Rules", "- Stop at 180 nm")
    assert "deadlines" not in agent.get("/app/agent/session").json()["session"]


@pytest.fixture
def experiment_s1(server):
    return server[3] / "agent" / "plan-s1.md"


def test_a_clock_time_is_its_next_occurrence():
    now = datetime(2026, 10, 23, 23, 30).astimezone()
    assert deadline("07:00", now) == now.replace(hour=7, minute=0) + timedelta(days=1)
    assert deadline("23:45", now) == now.replace(minute=45)
    assert rule_time("Finish by 7:00", now) == (
        now.replace(hour=7, minute=0) + timedelta(days=1)
    ).isoformat(timespec="minutes")
    assert dated("- Finish by 7:00\n- Coat by **25:00**", now) == (
        "- Finish by 2026-10-24 07:00\n- Coat by **25:00**"
    )
    assert dated("Done by 2026-11-06 07:00", now) == "Done by 2026-11-06 07:00"
    for rule in (
        "Stop at 180 nm",
        "Finish by 25:00",
        "Done by 12:30:15",
        "nearby 07:00",
    ):
        assert rule_time(rule, now) is None


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

    # No experiment open: kept in the session's own folder, so a restarted
    # app reads it back, and moved into the first experiment that opens.
    opened["path"] = None
    agent.post("/app/agent/session", json={"session_id": "s3"})
    assert section(agent, "Progress", "Started.").status_code == 200
    assert agent.get("/app/agent/plan").json()["sections"] == [
        {"name": "Progress", "text": "Started."}
    ]
    assert not list(experiment.glob("agent/plan-s3.md"))
    assert PlanStore(lambda: None).current("s3") == {"Progress": "Started."}
    later = tmp_path / "grid-2"
    later.mkdir()
    opened["path"] = str(later)
    section(agent, "Brief", "Sample: yeast")
    kept = (later / "agent" / "plan-s3.md").read_text(encoding="utf-8")
    assert "Started." in kept and "Sample: yeast" in kept
    assert PlanStore(lambda: None).current("s3") == {}, "moved, not copied"
    # The session keeps its experiment when the operator opens another.
    opened["path"] = str(experiment)
    section(agent, "Progress", "Milled 01.")
    assert "Milled 01." in (later / "agent" / "plan-s3.md").read_text()
    assert not (experiment / "agent" / "plan-s3.md").exists()


def test_a_plan_reads_back_as_it_was_written():
    sections = {"Brief": "Sample: yeast\nQ", "Steps": "### A\n- [ ] one", "Notes": "x"}
    assert parse(render(sections)) == sections
    assert list(parse("# Plan\n\n## notes\nx\n## brief\ny\n")) == ["notes", "Brief"]
    own = render({"Summary": "s", "Results": "r", "Brief": "b"})
    assert list(parse(own)) == ["Brief", "Results", "Summary"]


def test_the_experiments_sample_is_the_sessions_and_the_agents_is_saved_on_it(
    tmp_path,
):
    """A sample named when the experiment was made holds from the start; one the
    agent writes is saved on the experiment."""
    kept = {"sample": None}

    def set_sample(value):
        kept["sample"] = value
        return True

    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.state.agent_plan = PlanStore(
        lambda: str(tmp_path), lambda: kept["sample"], set_sample
    )
    app.include_router(build_agent_session_router(lambda kind, data: None))
    app.include_router(
        build_agent_plan_router(lambda: str(tmp_path), lambda *a, **k: None)
    )
    agent = TestClient(app, headers={"Authorization": f"Bearer {app.state.auth.token}"})
    agent.post("/app/agent/session", json={"session_id": "s1"})

    section(agent, "Brief", "Sample: yeast\nThe question.")
    assert kept["sample"] == "yeast"

    # A new session over the same experiment starts with its sample.
    agent.post("/app/agent/session", json={"session_id": "s2"})
    assert agent.get("/app/agent/plan").json()["sample"] == "yeast"
    assert section(agent, "Steps", "- [ ] Setup").status_code == 200
    refused = section(agent, "Brief", "Sample: HeLa\nThe question.")
    assert refused.status_code == 409 and "Sample: yeast" in str(refused.json())
    assert section(agent, "Brief", "Sample: yeast\nThe question.").status_code == 200
    assert kept["sample"] == "yeast"
