"""The Hub serves this PC's playbooks, skills and subagents, the ones an
agent on the hub PC runs with: read by anyone signed in to the hub, changed
only at the hub PC itself."""

import pytest
from fastapi.testclient import TestClient

from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.server.pairing import OPERATOR


@pytest.fixture
def page(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text('{"token": "operator-token"}')
    auth = hub_auth(token_file, tmp_path / "paired.json")
    _, paired = auth.pairing.tokens.add("laptop", role=OPERATOR)
    hub = Hub(file=tmp_path / "hub.yaml", tokens=tmp_path / "hub")
    client = TestClient(build_hub_app(hub, auth))
    here = {"Authorization": "Bearer operator-token"}
    away = {"Authorization": f"Bearer {paired}"}
    return client, here, away


@pytest.mark.parametrize("kind", ["playbooks", "skills", "agents"])
def test_the_library_is_read_by_anyone_signed_in(page, kind):
    client, here, away = page
    assert client.get(f"/app/library/{kind}").status_code == 401
    for headers in (here, away):
        answer = client.get(f"/app/library/{kind}", headers=headers)
        assert answer.status_code == 200
        assert answer.json()["items"]


def test_the_library_is_changed_only_at_the_hub_pc(page):
    client, here, away = page
    route = "/app/library/skills/hub-test-skill"
    assert client.post(route).status_code == 401
    refused = client.post(route, headers=away)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "at_this_pc_only"
    assert client.put(route, json={"text": "x"}, headers=away).status_code == 403
    assert client.delete(route, headers=away).status_code == 403

    made = client.post(route, headers=here)
    assert made.status_code == 200
    names = [
        item["name"]
        for item in client.get("/app/library/skills", headers=away).json()["items"]
    ]
    assert "hub-test-skill" in names
    assert client.delete(route, headers=here).status_code == 200
