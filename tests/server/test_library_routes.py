"""``/app/library``: the shipped profiles, playbooks and skills beside the
lab's own; only the operator edits, an edit to a shipped one saving the lab's
copy, and removes the lab's."""

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent import library, sources
from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.server import AuthConfig, build_server
from tests.run_state import stub_session

AGENT, OPERATOR = "agent-token", "operator-token"
OPERATOR_HEADERS = {"Authorization": f"Bearer {OPERATOR}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "state"))
    auth = AuthConfig(token=AGENT, operator_token=OPERATOR, level=AgentLevel.WATCH)
    app = build_server(
        None,
        app_context=AutoLamellaAppContext(stub_session()),
        auth=auth,
        serve_microscope=False,
    )
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {AGENT}"
    return client


def _error(response):
    return response.status_code, response.json()["detail"]["error_type"]


def test_the_shipped_ones_are_listed_and_only_the_agents_skills(client):
    profiles = client.get("/app/library/profiles").json()["items"]
    assert "fibsem-agent" in [p["name"] for p in profiles]
    assert all(p["shipped"] and not p["lab"] for p in profiles)
    assert all(p["description"] for p in profiles)
    rows = client.get("/app/library/skills").json()["items"]
    # front matter Claude Code reads, with a colon in a value, still describes
    assert all(s["description"] for s in rows)
    skills = [s["name"] for s in rows]
    assert "autolamella-lead" in skills
    # the checkout's skills for developers aren't the agent's
    assert "review-changes" not in skills
    playbooks = client.get("/app/library/playbooks").json()["items"]
    assert "autolamella-correlative" in [p["name"] for p in playbooks]
    assert _error(client.get("/app/library/hooks")) == (422, "invalid")


def test_an_edit_to_a_shipped_one_saves_the_labs_copy(client):
    shipped = client.get("/app/library/playbooks/autolamella")
    assert shipped.json()["source"] == "shipped"
    # the agent can read but not edit
    assert (
        client.put(
            "/app/library/playbooks/autolamella", json={"text": "description: x\n"}
        ).status_code
        == 403
    )
    saved = client.put(
        "/app/library/playbooks/autolamella",
        json={"text": "description: Ours.\nskills: [autolamella]\n"},
        headers=OPERATOR_HEADERS,
    )
    assert saved.status_code == 200, saved.text
    assert (saved.json()["source"], saved.json()["lab"]) == ("lab", True)
    assert str(sources.lab_folder("playbooks")) in saved.json()["path"]
    listed = client.get("/app/library/playbooks").json()["items"]
    row = next(p for p in listed if p["name"] == "autolamella")
    assert (row["description"], row["lab"], row["shipped"]) == ("Ours.", True, True)
    # the shipped one still reads as shipped
    assert (
        "Ours."
        not in client.get("/app/library/playbooks/autolamella?source=shipped").json()[
            "text"
        ]
    )

    removed = client.delete(
        "/app/library/playbooks/autolamella", headers=OPERATOR_HEADERS
    )
    assert removed.json() == {
        "name": "autolamella",
        "description": None,
        "lab": False,
        "shipped": True,
    }
    assert client.get("/app/library/playbooks/autolamella").json()["source"] == (
        "shipped"
    )


def test_a_file_that_would_not_load_is_refused_and_the_old_one_kept(client):
    before = client.get("/app/library/profiles/fibsem-agent").json()["text"]
    for text in ("budget: 5\n", "mode: everything\n", "skills: [no-such-skill]\n"):
        refused = client.put(
            "/app/library/profiles/fibsem-agent",
            json={"text": text},
            headers=OPERATOR_HEADERS,
        )
        assert _error(refused) == (422, "invalid"), text
    after = client.get("/app/library/profiles/fibsem-agent").json()
    # refused, nothing was copied: the shipped one is still the one used
    assert (after["text"], after["source"]) == (before, "shipped")

    refused = client.put(
        "/app/library/playbooks/autolamella",
        json={"text": "extends: no-such-playbook\n"},
        headers=OPERATOR_HEADERS,
    )
    assert _error(refused) == (422, "invalid")
    assert library.read("playbooks", "autolamella").source == "shipped"


def test_a_skill_is_copied_as_its_folder_and_needs_front_matter(client):
    refused = client.put(
        "/app/library/skills/screen-grids",
        json={"text": "No front matter."},
        headers=OPERATOR_HEADERS,
    )
    assert _error(refused) == (422, "invalid")
    assert not (sources.lab_folder("skills") / "screen-grids").exists()
    saved = client.put(
        "/app/library/skills/screen-grids",
        json={"text": "---\nname: screen-grids\ndescription: Ours.\n---\n\nBody.\n"},
        headers=OPERATOR_HEADERS,
    )
    assert saved.json()["text"].endswith("Body.\n")
    assert (sources.lab_folder("skills") / "screen-grids" / "SKILL.md").is_file()
    # the shipped base skill's own text saves as it is
    text = library.read("skills", "fibsem-agent").text
    resaved = client.put(
        "/app/library/skills/fibsem-agent",
        json={"text": text},
        headers=OPERATOR_HEADERS,
    )
    assert resaved.status_code == 200, resaved.text
    assert client.delete(
        "/app/library/skills/screen-grids", headers=OPERATOR_HEADERS
    ).json()["shipped"]
    assert not (sources.lab_folder("skills") / "screen-grids").exists()


def test_names_are_checked(client):
    for name in ("../etc", "a/b", "x.", ""):
        with pytest.raises(library.LibraryError):
            library.read("skills", name)
    assert _error(client.get("/app/library/skills/x.")) == (422, "invalid")
    # nothing shipped and nothing of the lab's by that name
    assert _error(
        client.put(
            "/app/library/profiles/nothing",
            json={"text": "mode: files\n"},
            headers=OPERATOR_HEADERS,
        )
    ) == (404, "not_found")
    assert _error(
        client.delete("/app/library/profiles/fibsem-agent", headers=OPERATOR_HEADERS)
    ) == (404, "not_found")
