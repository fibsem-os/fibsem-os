"""The microscope, device and service routes answer what /openapi.json says they
do: each answer on the Demo is checked against its route's declared schema."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

import jsonschema  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.devices import build_device_app, demo_devices  # noqa: E402
from fibsem.server.leases import LEASE_HEADER  # noqa: E402

TOKEN = "test-token"


@pytest.fixture(scope="module")
def client():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


def _matches(client, method, path, route, **kwargs):
    """The answer to *method* on *path*, checked against *route*'s schema."""
    response = client.request(method, path, **kwargs)
    assert response.status_code == 200, response.text
    openapi = client.get("/openapi.json").json()
    declared = openapi["paths"][route][method.lower()]["responses"]["200"]
    schema = declared["content"]["application/json"]["schema"]
    assert schema, f"{method} {route} declares no response schema"
    jsonschema.validate(
        response.json(), {**schema, "components": openapi["components"]}
    )
    return response.json()


def test_the_microscope_routes_answer_their_schema(client):
    _matches(client, "GET", "/microscope", "/microscope")
    _matches(client, "GET", "/microscope/state", "/microscope/state")
    _matches(client, "GET", "/microscope/control", "/microscope/control")
    lease = _matches(
        client, "POST", "/microscope/control", "/microscope/control", json={}
    )["lease"]
    route = "/microscope/control/{lease}"
    headers = {LEASE_HEADER: lease}
    _matches(client, "PUT", f"/microscope/control/{lease}", route, headers=headers)
    assert _matches(client, "DELETE", f"/microscope/control/{lease}", route)["released"]


def test_the_device_and_service_routes_answer_their_schema(client):
    devices = _matches(client, "GET", "/microscope/devices", "/microscope/devices")
    base = "/microscope/devices/{device}"
    for name, device in devices.items():
        _matches(client, "GET", f"/microscope/devices/{name}", base)
        _matches(client, "GET", f"/microscope/devices/{name}/values", base + "/values")
        for parameter in list(device["parameters"])[:3]:
            path = f"/microscope/devices/{name}/{parameter}"
            _matches(client, "GET", path, base + "/{parameter}")
            _matches(client, "GET", path + "/metadata", base + "/{parameter}/metadata")
    _matches(client, "GET", "/microscope/services", "/microscope/services")


def test_the_device_servers_health_answers_its_schema():
    app = build_device_app(demo_devices())
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.auth.token}"
        assert _matches(client, "GET", "/health", "/health")["ok"]
