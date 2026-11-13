"""Typed request bodies (`fibsem.server.bodies`): a route's body is in
/openapi.json, and a body that doesn't read is refused as every refusal is."""

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent.catalog import CATALOG
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.codec import decode
from fibsem.server import AuthConfig, build_server
from fibsem.server.bodies import ConnectBody, PairBody
from fibsem.server.device_auth import DeviceAuth
from fibsem.server.devices import build_device_app
from fibsem.server.pairing import PairedTokens, Pairing
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
    ("post", "/app/experiment/protocol/task_config/{task_name}"),
    ("post", "/app/workflow/schedule"),
    ("post", "/app/workflow/options"),
    ("post", "/app/experiment/protocol/tasks"),
    ("post", "/app/experiment/protocol/save"),
    ("post", "/app/experiment/create"),
    ("post", "/app/experiment/load"),
    ("post", "/app/experiment/protocol/load"),
    ("post", "/app/experiment/items"),
    ("post", "/app/experiment/items/{item_name}"),
    ("post", "/app/experiment/grids/{grid_name}/note"),
    ("post", "/app/experiment/images/markers"),
    ("post", "/app/experiment/images/drawn"),
    ("post", "/app/experiment/images/pair"),
    ("post", "/app/experiment/images/pixels"),
    ("post", "/app/experiment/analysis/intensity"),
    ("post", "/app/experiment/analysis/profile"),
    ("post", "/app/experiment/export/image"),
    ("post", "/app/experiment/notes"),
    ("post", "/app/experiment/messages"),
    ("post", "/app/experiment/evidence"),
    ("post", "/app/knowledge"),
    ("post", "/pair"),
    ("post", "/connect"),
    ("post", "/pairing"),
    ("put", "/pairing/always_allow_local"),
    ("patch", "/pairing/computers/{computer_id}"),
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


def test_each_tool_sends_only_what_its_route_takes(client):
    # The catalog's parameters and a typed body are two copies of one
    # contract: a parameter the body lacks would be refused on every call.
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    wrong = {}
    for tool in CATALOG:
        operation = spec["paths"].get(tool.path, {}).get(tool.method.lower())
        body = (operation or {}).get("requestBody")
        if body is None:
            continue
        ref = body["content"]["application/json"]["schema"]
        ref = ref.get("$ref") or next(
            (r["$ref"] for r in ref.get("anyOf", ()) if "$ref" in r), None
        )
        if ref is None:
            continue
        fields = set(schemas[ref.rsplit("/", 1)[1]].get("properties", {}))
        in_path = {p["name"] for p in operation.get("parameters", ())}
        extra = set(tool.params or {}) - fields - in_path
        if extra:
            wrong[tool.name] = sorted(extra)
    assert wrong == {}


def test_a_pairing_ask_from_another_version_is_not_refused_for_its_keys():
    # /pair and /connect are where a client of another version first meets
    # this one: a key this version doesn't know is ignored there.
    assert decode(PairBody, {"code": "123456", "colour": "red"}).code == "123456"
    assert decode(ConnectBody, {"wait_s": 1, "later": True}).wait_s == 1.0


def test_the_device_server_refuses_the_same_way(tmp_path):
    # It shares the pairing routes, so it answers their refusals alike.
    auth = DeviceAuth("install", Pairing(PairedTokens(tmp_path / "paired.json")))
    with TestClient(build_device_app([], auth)) as http:
        refused = http.put(
            "/pairing/always_allow_local",
            json={"on": "yes"},
            headers={"Authorization": f"Bearer {auth.token}"},
        )
    assert refused.status_code == 422
    assert refused.json()["detail"]["error_type"] == "invalid"


def test_no_body_shares_a_name_with_an_answer():
    # /openapi.json names its schemas by class name: two classes of one name
    # describe one of them wrongly.
    import dataclasses

    from fibsem.server import bodies, responses

    def names(module):
        return {
            name
            for name, value in vars(module).items()
            if dataclasses.is_dataclass(value) and value.__module__ == module.__name__
        }

    assert names(bodies) & names(responses) == set()
