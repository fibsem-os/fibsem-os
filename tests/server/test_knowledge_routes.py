"""``/app/knowledge`` and the knowledge tools: the agent and the operator write,
only the operator removes, and nothing answers the agent at Off."""

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fibsem import knowledge
from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import Experiment
from fibsem.client import tools
from fibsem.server import AuthConfig, build_server
from tests.run_state import stub_session

AGENT, OPERATOR = "agent-token", "operator-token"


@pytest.fixture
def server(tmp_path, monkeypatch):
    """The App Server over an open experiment holding one large figure, its
    knowledge base under a state root of its own."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "state"))
    host = stub_session()
    host.experiment = Experiment(path=tmp_path, name="Grid-03")
    folder = tmp_path / "Grid-03"
    folder.mkdir()
    Image.fromarray(np.zeros((1200, 2400, 3), np.uint8)).save(folder / "figure.png")
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR, level=AgentLevel.WATCH)
    app = build_server(
        None,
        app_context=AutoLamellaAppContext(host),
        auth=auth,
        serve_microscope=False,
    )
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {AGENT}"
    return client, host, auth


def _as_operator():
    return {"Authorization": f"Bearer {OPERATOR}"}


def _error(response):
    return response.status_code, response.json()["detail"]["error_type"]


def test_the_agent_writes_an_entry_and_reads_it_back(server):
    client, _, _ = server
    written = client.post(
        "/app/knowledge",
        json={"section": "instrument", "title": "10 pA worn", "text": "Use 30 pA."},
    )
    assert written.status_code == 200, written.text
    entry = written.json()["entry"]
    assert (entry["by"], entry["experiment"], entry["edited"]) == (
        "agent",
        "Grid-03",
        False,
    )

    index = client.get("/app/knowledge").json()
    assert [row["title"] for row in index["entries"]] == ["10 pA worn"]
    assert index["entries"][0]["by"] == "agent"
    read = client.get(f"/app/knowledge/instrument/{entry['id']}").json()
    assert read["entry"]["text"] == "Use 30 pA." and read["pictures"] is None


def test_the_operator_s_write_says_so_and_an_edit_is_marked(server):
    client, _, _ = server
    entry = client.post(
        "/app/knowledge",
        json={"section": "sample", "sample": "yeast", "title": "Fast", "text": "1.4x"},
    ).json()["entry"]
    edited = client.post(
        "/app/knowledge",
        headers=_as_operator(),
        json={
            "section": "sample",
            "sample": "yeast",
            "id": entry["id"],
            "title": "Mills fast",
            "text": "About 1.4x the estimate.",
        },
    ).json()["entry"]
    assert (edited["title"], edited["by"]) == ("Mills fast", "agent")
    assert edited["edited"] is True and edited["edited_by"] == "operator"

    own = client.post(
        "/app/knowledge",
        headers=_as_operator(),
        json={"section": "instrument", "title": "Stage drifts", "text": "x"},
    ).json()["entry"]
    assert own["by"] == "operator" and own["edited"] is False


def test_only_the_operator_removes_an_entry(server):
    client, _, _ = server
    entry = client.post(
        "/app/knowledge", json={"section": "instrument", "title": "Old", "text": "x"}
    ).json()["entry"]
    path = f"/app/knowledge/instrument/{entry['id']}"
    assert _error(client.delete(path)) == (403, "operator_only")
    removed = client.delete(path, headers=_as_operator())
    assert removed.status_code == 200
    assert removed.json()["entry"]["title"] == "Old"
    assert client.get("/app/knowledge").json()["entries"] == []
    assert _error(client.delete(path, headers=_as_operator())) == (404, "not_found")


def test_at_off_the_agent_is_refused_and_the_operator_is_not(server):
    client, _, auth = server
    auth.set_level(AgentLevel.OFF)
    body = {"section": "instrument", "title": "x", "text": "x"}
    assert _error(client.get("/app/knowledge")) == (403, "agent_off")
    assert _error(client.post("/app/knowledge", json=body)) == (403, "agent_off")
    operator = client.post("/app/knowledge", json=body, headers=_as_operator())
    assert operator.status_code == 200


def test_images_are_copied_in_from_the_experiment_downsized(server):
    client, _, _ = server
    entry = client.post(
        "/app/knowledge",
        json={
            "section": "instrument",
            "title": "Wide spot",
            "text": "x",
            "images": ["figure.png"],
        },
    ).json()["entry"]
    assert entry["images"] == ["figure.png"]
    picture = client.get(f"/app/knowledge/instrument/{entry['id']}/images/figure.png")
    assert picture.headers["content-type"] == "image/png"
    with Image.open(knowledge.image("instrument", entry["id"], "figure.png")) as kept:
        assert kept.size == (1024, 512)
    shown = client.get(
        f"/app/knowledge/instrument/{entry['id']}", params={"pictures": True}
    ).json()["pictures"]
    assert shown[0]["name"] == "figure.png"
    assert shown[0]["preview"]["width"] == 768


@pytest.mark.parametrize(
    "images, status, error_type",
    [
        (["../outside.png"], 403, "not_allowed"),
        (["missing.png"], 404, "not_found"),
    ],
)
def test_an_image_outside_the_experiment_is_refused(server, images, status, error_type):
    client, _, _ = server
    body = {"section": "instrument", "title": "x", "text": "x", "images": images}
    assert _error(client.post("/app/knowledge", json=body)) == (status, error_type)
    assert client.get("/app/knowledge").json()["entries"] == []


def test_with_no_experiment_open_it_works_but_images_are_refused(server):
    client, host, _ = server
    host.experiment = None
    body = {"section": "instrument", "title": "x", "text": "x"}
    assert (
        client.post("/app/knowledge", json=body).json()["entry"]["experiment"] is None
    )
    with_image = dict(body, images=["figure.png"])
    assert _error(client.post("/app/knowledge", json=with_image)) == (
        409,
        "no_experiment",
    )
    summary = {"section": "experiment", "title": "x", "text": "x"}
    assert _error(client.post("/app/knowledge", json=summary)) == (422, "invalid")


def test_an_experiment_summary_is_the_open_experiment_s(server):
    client, _, _ = server
    entry = client.post(
        "/app/knowledge",
        json={"section": "experiment", "title": "Grid-03, 27 Oct", "text": "3 of 7."},
    ).json()["entry"]
    assert entry["id"] == "Grid-03" and entry["path"].endswith("summary.md")


def test_the_tools_call_their_routes(server):
    client, _, _ = server
    written = tools.send(
        client,
        "knowledge_write",
        {
            "section": "sample",
            "sample": "yeast",
            "title": "Mills fast",
            "text": "1.4x",
            "images": ["figure.png"],
        },
    )
    assert written.status_code == 200, written.text
    entry = written.json()["entry"]
    listed = tools.send(client, "knowledge_list", {"section": "sample"}).json()
    assert [row["id"] for row in listed["entries"]] == [entry["id"]]
    assert listed["samples"] == ["yeast"]
    got = tools.send(
        client,
        "knowledge_get",
        {"section": "sample", "id": entry["id"], "sample": "yeast"},
    ).json()
    assert got["entry"]["text"] == "1.4x"
    assert "image_b64_jpeg" in got["pictures"][0]["preview"]


def test_knowledge_get_answers_its_images_as_pictures(server):
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    client, _, _ = server
    entry = client.post(
        "/app/knowledge",
        json={
            "section": "instrument",
            "title": "Spot",
            "text": "x",
            "images": ["figure.png"],
        },
    ).json()["entry"]
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(
        sidecar.call_tool("knowledge_get", {"section": "instrument", "id": entry["id"]})
    )
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    kinds = [getattr(c, "type", None) for c in contents]
    assert kinds == ["text", "image"]
    answer = json.loads(contents[0].text)
    assert answer["pictures"][0] == {
        "name": "figure.png",
        "preview": {"width": 768, "height": 384, "image_index": 0},
    }
