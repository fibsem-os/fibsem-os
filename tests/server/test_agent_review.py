"""The agent's review list: ``/app/agent/review``, and the operator's answer."""

import json

import numpy as np
import pytest
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig
from fibsem.server.agent.operator import build_operator_router
from fibsem.server.agent.review import (
    ADDED,
    CLOSED,
    ReviewStore,
    build_agent_review_router,
)
from fibsem.server.agent.session import build_agent_session_router

OPERATOR = "operator-token-for-tests"


class FakeAgent:
    """What the answer route uses of the operator's agent: where the
    experiment is, and the message it puts on the record."""

    def __init__(self, path):
        self.path = path
        self.messages = []

    def experiment_path(self):
        return self.path

    def message(self, text, evidence=None):
        self.messages.append((text, evidence))
        return f"m{len(self.messages)}"


@pytest.fixture
def server(tmp_path):
    recorded = []
    experiment = tmp_path / "grid-1"
    (experiment / "lamella-01").mkdir(parents=True)
    tifffile.imwrite(
        experiment / "lamella-01" / "polish.tif", np.zeros((8, 12), np.uint8)
    )
    app = FastAPI()
    app.state.auth = AuthConfig.generate(level=AgentLevel.WATCH)
    app.state.auth.operator_token = OPERATOR
    fake = FakeAgent(str(experiment))

    def emit(kind, payload, actor=None):
        recorded.append((kind, payload, actor))

    app.include_router(build_agent_session_router(lambda kind, data: None))
    app.include_router(build_agent_review_router(lambda: str(experiment), emit))
    app.include_router(build_operator_router(fake))
    agent = TestClient(app, headers={"Authorization": f"Bearer {app.state.auth.token}"})
    operator = TestClient(app, headers={"Authorization": f"Bearer {OPERATOR}"})
    agent.post("/app/agent/session", json={"session_id": "s1"})
    return agent, operator, recorded, experiment, fake


def add(client, **body):
    return client.post("/app/agent/review", json=body)


def test_the_agent_adds_finished_work_checks_and_waits_beside_the_plan(server):
    agent, operator, recorded, experiment, _ = server
    done = add(agent, title="Rough mill done", about="Lamella 01").json()
    check = add(
        agent,
        title="Polished to 290 nm",
        level="check",
        about="Lamella 01",
        images=["lamella-01/polish.tif"],
        facts=[{"label": "Thickness", "value": "290 nm (target 300 ± 20)"}],
    ).json()
    wait = add(
        agent,
        title="Keep site 3 or move it 1 µm in?",
        level="wait",
        about="Lamella 03",
        options=["Keep it", "Move 1 µm in"],
    ).json()
    assert [done["id"], check["id"], wait["id"]] == ["r1", "r2", "r3"]
    assert [done["state"], check["state"], wait["state"]] == [
        "done",
        "to_check",
        "waiting",
    ]
    assert check["options"] == ["Accept", "Reject"]
    assert wait["options"] == ["Keep it", "Move 1 µm in"]
    # The operator reads it all; the agent a slice.
    listed = operator.get("/app/agent/review").json()
    assert [i["id"] for i in listed["items"]] == ["r1", "r2", "r3"]
    assert listed["counts"] == {"waiting": 1, "to_check": 1, "rejected": 0, "all": 3}
    waiting = agent.get("/app/agent/review", params={"show": "waiting"}).json()
    assert [i["id"] for i in waiting["items"]] == ["r3"]
    about = agent.get("/app/agent/review", params={"about": "lamella 01"}).json()
    assert [i["id"] for i in about["items"]] == ["r1", "r2"]
    kept = experiment / "agent" / "review-s1.jsonl"
    assert [json.loads(line)["op"] for line in kept.read_text().splitlines()] == [
        "add",
        "add",
        "add",
    ]
    assert [(k, p["id"], a) for k, p, a in recorded] == [
        (ADDED, "r1", "agent"),
        (ADDED, "r2", "agent"),
        (ADDED, "r3", "agent"),
    ]


def test_what_an_item_takes_is_checked(server):
    agent, operator, *_ = server
    assert add(agent, title="").status_code == 422
    assert add(agent, title="x", level="maybe").status_code == 422
    assert add(agent, title="x", images=["../outside.png"]).status_code == 422
    assert add(agent, title="x", images=["/abs.png"]).status_code == 422
    assert add(agent, title="x", options=["Only one"]).status_code == 422
    # The operator answers items; adding them is the agent's.
    assert add(operator, title="x").status_code == 403


def test_the_operators_answer_reaches_the_agent_as_a_message_with_its_marks(
    server,
):
    agent, operator, _, experiment, fake = server
    add(
        agent,
        title="Polished to 290 nm",
        level="check",
        images=["lamella-01/polish.tif"],
    )
    answer = operator.post(
        "/app/agent/review/r1/answer",
        json={
            "choice": "Reject",
            "note": "Curtaining at the lower right.",
            "evidence": {
                "image": "lamella-01/polish.tif",
                "marks": {
                    "areas": [
                        {"x": 2, "y": 2, "width": 4, "height": 3, "verdict": "bad"}
                    ]
                },
            },
        },
    )
    assert answer.status_code == 200, answer.text
    item = answer.json()
    assert item["state"] == "answered"
    assert item["told"] is True
    assert item["answer"]["choice"] == "Reject"
    assert item["answer"]["evidence"]
    text, evidence = fake.messages[0]
    assert (
        text == 'Review r1 "Polished to 290 nm": Reject. Curtaining at the lower right.'
    )
    assert evidence["evidence_id"] == item["answer"]["evidence"]
    assert operator.get("/app/agent/review").json()["counts"]["rejected"] == 1
    # Once answered it is not answered again, and the agent's token can't answer.
    again = operator.post("/app/agent/review/r1/answer", json={"choice": "Accept"})
    assert again.status_code == 409
    add(agent, title="Thinned", level="check")
    refused = agent.post("/app/agent/review/r2/answer", json={"choice": "Accept"})
    assert refused.status_code == 403
    assert (
        operator.post(
            "/app/agent/review/r2/answer", json={"choice": "Maybe"}
        ).status_code
        == 422
    )


def test_the_agent_closes_an_item_it_no_longer_needs_answered(server):
    agent, operator, recorded, *_ = server
    add(agent, title="Keep site 3?", level="wait")
    closed = agent.post(
        "/app/agent/review/r1/close", json={"reason": "Answered in the chat."}
    ).json()
    assert closed["state"] == "closed"
    assert recorded[-1][0] == CLOSED
    assert operator.get("/app/agent/review").json()["counts"]["waiting"] == 0
    assert (
        agent.post("/app/agent/review/r1/close", json={"reason": "again"}).status_code
        == 409
    )
    assert (
        agent.post("/app/agent/review/r9/close", json={"reason": "x"}).status_code
        == 404
    )


def test_a_restarted_app_reads_the_list_back(server, tmp_path):
    agent, operator, _, experiment, _ = server
    add(agent, title="Rough mill done")
    add(agent, title="Keep site 3?", level="wait")
    operator.post("/app/agent/review/r2/answer", json={"choice": "Accept"})
    store = ReviewStore(lambda: str(experiment))
    items = store.current("s1")
    assert [(i["id"], i.get("answer", {}).get("choice")) for i in items] == [
        ("r1", None),
        ("r2", "Accept"),
    ]
