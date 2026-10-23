"""Instrument reads an agent needs at Watch (agent surface map, step 5): whether
a beam is live, and where the stage is among the holder's slots.

Skipped when the [server] extra is not installed.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.server import AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def watching(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="watch", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def test_a_beam_says_whether_it_is_live(watching, microscope):
    assert watching.get("/microscope/devices/electron/live").json() == {"value": False}
    described = watching.get("/microscope/devices/electron").json()
    assert described["parameters"]["live"]["settable"] is False
    microscope.devices["electron"].start_live()
    try:
        assert watching.get("/microscope/devices/electron/live").json()["value"]
    finally:
        microscope.devices["electron"].stop_live()


def test_where_the_stage_is_among_the_slots_is_a_query(watching):
    answer = watching.post(
        "/microscope/services/sample_handling/queries/current", json={"kwargs": {}}
    )
    assert answer.status_code == 200, answer.text
    assert set(answer.json()["result"]) == {"slot", "grid"}

    slots = watching.post(
        "/microscope/services/sample_handling/queries/holder_slots",
        json={"kwargs": {}},
    ).json()["result"]
    assert [slot["index"] for slot in slots] == sorted(slot["index"] for slot in slots)
    assert all("position" in slot for slot in slots)

    # Reading the inventory moves nothing, so a watching agent may.
    inventory = watching.post(
        "/microscope/services/sample_handling/queries/get_inventory",
        json={"kwargs": {}},
    )
    assert inventory.status_code == 200, inventory.text
