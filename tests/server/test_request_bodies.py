"""Typed request bodies (`fibsem.server.bodies`): a route's body is in
/openapi.json, and a body that doesn't read is refused as every refusal is."""

import pytest
from fastapi.testclient import TestClient

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.server import AuthConfig, build_server
from tests.run_state import stub_session

# The routes with a typed body, by method and path (the operator's are
# mounted only with an operator).
TYPED = [
    ("post", "/app/agent/requests"),
    ("post", "/app/agent/requests/{request_id}/withdraw"),
    ("post", "/app/agent/plan/section"),
    ("put", "/app/agent/plan"),
    ("post", "/app/agent/transcript"),
    ("post", "/app/agent/read"),
    ("put", "/app/library/{kind}/{name}"),
    ("post", "/app/workflow/plan"),
    ("post", "/app/workflow/grids/plan"),
    ("post", "/app/workflow/decide"),
    ("post", "/app/workflow/start"),
    ("post", "/app/workflow/grids/start"),
    ("post", "/app/experiment/protocol/supervision"),
    ("post", "/app/workflow/queue/requeue"),
    ("post", "/app/workflow/queue/edit"),
    ("post", "/app/workflow/queue/grids"),
    ("post", "/app/workflow/proposals/suggest"),
    ("post", "/app/workflow/stop"),
    ("post", "/app/workflow/stop_task"),
]


@pytest.fixture
def client(microscope):
    session = stub_session()
    session.microscope = microscope
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(session),
        auth=AuthConfig(token="agent"),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = "Bearer agent"
        yield client


def test_each_typed_route_describes_its_body(client):
    paths = client.get("/openapi.json").json()["paths"]
    bodies = {
        (method, path): paths[path][method]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        for method, path in TYPED
    }
    assert bodies[("post", "/app/agent/plan/section")] == {
        "$ref": "#/components/schemas/PlanSectionBody"
    }


def test_a_key_the_route_does_not_take_is_refused(client):
    refused = client.post(
        "/app/agent/plan/section",
        json={"section": "Brief", "text": "Sample: yeast", "colour": "red"},
    )
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["error_type"] == "invalid"
    assert detail["message"] == "colour is not a field this route takes."


def test_a_missing_field_is_refused_as_missing(client):
    refused = client.post("/app/agent/requests/abc/withdraw", json={})
    assert refused.status_code == 422
    assert refused.json()["detail"] == {
        "error_type": "missing_field",
        "message": "reason is required.",
    }


def test_no_body_is_refused_as_missing(client):
    refused = client.post("/app/agent/read")
    assert refused.json()["detail"] == {
        "error_type": "missing_field",
        "message": "Send a JSON body.",
    }


def test_a_wrong_type_names_the_field(client):
    refused = client.post("/app/agent/read", json={"ids": "m1"})
    detail = refused.json()["detail"]
    assert detail["error_type"] == "invalid"
    assert detail["message"].startswith("ids: ")


def test_a_bad_query_is_refused_the_same_way(client):
    # Not a body, but the same refusal: a query that doesn't read.
    refused = client.get("/app/events", params={"since": "lots"})
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["error_type"] == "invalid"
    assert detail["message"].startswith("query.since: ")
