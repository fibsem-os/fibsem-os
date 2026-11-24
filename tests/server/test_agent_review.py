"""The agent's review list: ``/app/agent/review``, and the operator's answer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server import AuthConfig
from fibsem.server.agent.files import session_folder
from fibsem.server.agent.operator import build_operator_router
from fibsem.server.agent.review import (
    ADDED,
    CLOSED,
    ReviewStore,
    build_agent_review_router,
    reached_in,
    resolve_review_item,
)
from fibsem.server.agent.session import build_agent_session_router

OPERATOR = "operator-token-for-tests"


class FakeAgent:
    """What the answer route uses of the operator's agent: where the
    experiment is, and the message it puts on the record."""

    def __init__(self, path):
        self.path = path
        self.messages = []
        self.reached = {}
        run = SimpleNamespace(
            task_id="t-polish-1", name="Polishing", task_type="MILL_POLISHING"
        )
        self.experiment = SimpleNamespace(
            positions=[
                SimpleNamespace(id="id-01", name="Lamella 01", task_history=[run]),
                SimpleNamespace(id="id-02", name="Lamella 02", task_history=[]),
            ]
        )
        self.since = []

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
    app.include_router(
        build_agent_review_router(
            lambda: str(experiment),
            emit,
            reached=lambda since: fake.since.append(since) or fake.reached,
            resolve=lambda item, task, score: resolve_review_item(
                fake.experiment, item, task, score
            ),
        )
    )
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
    assert listed["counts"] == {
        "waiting": 1,
        "to_check": 1,
        "rejected": 0,
        "all": 3,
        "scores": {},
    }
    waiting = agent.get("/app/agent/review", params={"show": "waiting"}).json()
    assert [i["id"] for i in waiting["items"]] == ["r3"]
    about = agent.get("/app/agent/review", params={"about": "lamella 01"}).json()
    assert [i["id"] for i in about["items"]] == ["r1", "r2"]
    kept = session_folder("s1") / "review-s1.jsonl"
    assert not (experiment / "agent").exists()
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


def test_accepting_done_work_with_nothing_to_add_is_not_told(server):
    agent, operator, _, _, fake = server
    add(agent, title="Rough mill done")
    add(agent, title="Thinned")
    quiet = operator.post("/app/agent/review/r1/answer", json={"choice": "Accept"})
    assert quiet.json()["state"] == "answered"
    assert quiet.json()["told"] is False
    assert fake.messages == []
    # A Reject, or an Accept with a note, still reaches the agent.
    told = operator.post(
        "/app/agent/review/r2/answer", json={"choice": "Accept", "note": "Nice."}
    )
    assert told.json()["told"] is True
    assert fake.messages[0][0] == 'Review r2 "Thinned": Accept. Nice.'


def test_scored_items_count_a_pass_or_a_fail_per_kind(server):
    agent, operator, recorded, *_ = server
    sites = add(agent, title="Sites on Grid 1", about="Grid 1", score="sites").json()
    assert sites["score"] == "sites"
    assert sites["options"] == ["Accept", "Reject"]
    add(agent, title="Target found", item_name="Lamella 01", score="target")
    add(agent, title="Target found", item_name="Lamella 02", score="target")
    add(agent, title="Polished to 190 nm", item_name="Lamella 01", score="polishing")
    add(agent, title="Rough mill done", about="Lamella 01")
    assert recorded[0][1]["score"] == "sites"
    operator.post("/app/agent/review/r1/answer", json={"choice": "Accept"})
    operator.post(
        "/app/agent/review/r2/answer",
        json={"choice": "Reject", "note": "The other cilium."},
    )
    operator.post("/app/agent/review/r3/answer", json={"choice": "Accept"})
    counts = agent.get("/app/agent/review").json()["counts"]
    assert counts["scores"] == {
        "sites": {
            "passed": 1,
            "failed": 0,
            "open": 0,
            "after_help": 0,
            "missing": [],
            "closed": [],
        },
        "target": {
            "passed": 1,
            "failed": 1,
            "open": 0,
            "after_help": 0,
            "missing": [],
            "closed": [],
        },
        "polishing": {
            "passed": 0,
            "failed": 0,
            "open": 1,
            "after_help": 0,
            "missing": [],
            "closed": [],
        },
    }


def test_a_redo_after_a_reject_keeps_one_score_per_lamella(server):
    agent, operator, *_ = server
    add(agent, title="Target found", item_name="Lamella 01", score="target")
    operator.post("/app/agent/review/r1/answer", json={"choice": "Reject"})
    # The agent tries again on the same lamella, with its own label.
    add(
        agent,
        title="Target found again",
        item_name="Lamella 01",
        about="L01 redo",
        score="target",
    )
    operator.post("/app/agent/review/r2/answer", json={"choice": "Accept"})
    target = agent.get("/app/agent/review").json()["counts"]["scores"]["target"]
    assert (target["passed"], target["failed"], target["after_help"]) == (0, 1, 1)


def test_a_lamella_polished_with_no_scored_item_counts_as_a_fail(server):
    agent, operator, _, _, fake = server
    fake.reached = {
        "polishing": [
            {"item_id": "id-01", "item_name": "Lamella 01"},
            {"item_id": "id-02", "item_name": "Lamella 02"},
        ]
    }
    add(agent, title="Polished to 190 nm", item_name="Lamella 01", score="polishing")
    operator.post("/app/agent/review/r1/answer", json={"choice": "Accept"})
    polishing = agent.get("/app/agent/review").json()["counts"]["scores"]["polishing"]
    assert polishing["passed"] == 1
    assert polishing["failed"] == 1
    assert polishing["missing"] == ["Lamella 02"]


def test_which_lamellae_reached_polishing_is_read_from_their_tasks():
    from types import SimpleNamespace as NS

    def run(task_type, status, at="2026-11-24T10:00:00+11:00"):
        return NS(task_type=task_type, status=NS(name=status), start_timestamp=at)

    positions = [
        NS(name="L1", task_history=[run("MILL_POLISHING", "Completed")]),
        NS(name="L2", task_history=[run("MILL_COINCIDENT", "Failed")]),
        NS(name="L3", task_history=[run("MILL_POLISHING", "InProgress")]),
        NS(name="L4", task_history=[run("MILL_ROUGH", "Completed")]),
        NS(name="L5", task_history=None),
        # Polished the day before: an earlier session's.
        NS(
            name="L6",
            task_history=[
                run("MILL_POLISHING", "Completed", "2026-11-23T10:00:00+11:00")
            ],
        ),
    ]
    since = "2026-11-24T09:00:00+11:00"

    def names(found):
        return [r["item_name"] for r in found["polishing"]]

    assert names(reached_in(positions, since)) == ["L1", "L2"]
    assert names(reached_in(positions)) == ["L1", "L2", "L6"]


def test_missing_lamellae_are_counted_from_the_session_start(server):
    agent, _, _, _, fake = server
    agent.get("/app/agent/review")
    assert fake.since[-1] is not None


def test_a_scored_item_names_a_lamella_the_experiment_has(server):
    agent, _, recorded, *_ = server
    refused = add(agent, title="x", score="polishing", item_name="Lamella 01 polish")
    assert refused.status_code == 404
    assert refused.json()["detail"]["item_names"] == ["Lamella 01", "Lamella 02"]
    # The server fills in the ids, and the latest polishing run.
    added = add(agent, title="x", score="polishing", item_name="Lamella 01").json()
    assert (added["item_id"], added["item_name"], added["about"]) == (
        "id-01",
        "Lamella 01",
        "Lamella 01",
    )
    assert (added["task_id"], added["task_name"]) == ("t-polish-1", "Polishing")
    assert recorded[-1][1]["item_id"] == "id-01"
    assert recorded[-1][1]["task_id"] == "t-polish-1"
    # Grids aren't positions: sites go by about.
    assert add(agent, title="x", score="sites", about="Grid 1").is_success


def test_item_and_task_names_are_optional_but_checked(server):
    agent, *_ = server
    plain = add(agent, title="Rough mill done").json()
    assert "item_id" not in plain
    named = add(agent, title="Rough mill done", item_name="Lamella 02").json()
    assert named["item_id"] == "id-02" and "task_id" not in named
    task = add(agent, title="x", item_name="Lamella 01", task_name="polishing").json()
    assert task["task_id"] == "t-polish-1"
    wrong = add(agent, title="x", item_name="Lamella 01", task_name="Setup")
    assert wrong.status_code == 404
    assert wrong.json()["detail"]["task_names"] == ["Polishing"]
    assert add(agent, title="x", task_name="Polishing").status_code == 422


def test_a_scored_item_the_agent_closed_unanswered_is_a_closed_fail(server):
    agent, _, _, _, fake = server
    fake.reached = {"polishing": [{"item_id": "id-01", "item_name": "Lamella 01"}]}
    add(agent, title="Polished", item_name="Lamella 01", score="polishing")
    agent.post("/app/agent/review/r1/close", json={"reason": "Asked in the chat."})
    polishing = agent.get("/app/agent/review").json()["counts"]["scores"]["polishing"]
    assert polishing["failed"] == 1
    assert polishing["closed"] == ["Lamella 01"]
    assert polishing["missing"] == []


def test_an_answer_kept_when_the_agent_cannot_be_told(server, monkeypatch):
    agent, operator, _, _, fake = server
    add(agent, title="Thinned", level="check")

    def broken(text, evidence=None):
        raise RuntimeError("no agent")

    monkeypatch.setattr(fake, "message", broken)
    answer = operator.post(
        "/app/agent/review/r1/answer", json={"choice": "Reject", "note": "Too thick."}
    )
    assert answer.status_code == 200
    assert answer.json()["told"] is False
    assert answer.json()["state"] == "answered"


def test_a_list_that_cannot_be_saved_refuses_and_tells_nobody(
    server, tmp_path, monkeypatch
):
    agent, operator, _, _, fake = server
    add(agent, title="Thinned", level="check")
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder")
    monkeypatch.setattr(ReviewStore, "_file", lambda self, sid: blocker / "r.jsonl")
    refused = add(agent, title="Rough mill done")
    assert refused.status_code == 503
    assert refused.json()["detail"]["error_type"] == "not_saved"
    answer = operator.post(
        "/app/agent/review/r1/answer", json={"choice": "Reject", "note": "Too thick."}
    )
    assert answer.status_code == 503
    # Nothing held that the file doesn't have, and the agent wasn't told.
    assert fake.messages == []
    monkeypatch.undo()
    items = agent.get("/app/agent/review").json()["items"]
    assert [(i["id"], i["state"]) for i in items] == [("r1", "to_check")]


def test_a_scored_item_names_what_it_scores_and_keeps_accept_or_reject(server):
    agent, *_ = server
    assert add(agent, title="x", score="speed", about="L1").status_code == 422
    # The lamella it scores goes in item_name; the grid in about.
    assert add(agent, title="x", score="target", about="L1").status_code == 422
    assert add(agent, title="x", score="sites").status_code == 422
    assert (
        add(
            agent,
            title="x",
            score="target",
            item_name="Lamella 01",
            options=["Good", "Bad"],
        ).status_code
        == 422
    )


def test_review_add_with_its_optional_fields_left_out_sends_none_of_them():
    """The MCP server hands a left-out optional argument over as None; sent as
    null, ReviewAddBody (level: str = "done") refused the call (rehearsal,
    24 Nov)."""
    from fibsem.ai.agent.catalog import CATALOG, route_call

    spec = next(s for s in CATALOG if s.name == "review_add")
    left_out = {name: None for name in spec.optional}
    method, path, body = route_call(spec, {"title": "Sites", **left_out})
    assert (method, path, body) == ("POST", "/app/agent/review", {"title": "Sites"})
