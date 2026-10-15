"""The agent's grid tools: the sample handling service's exchange, unload, moves
and inventory, by name, and the lines an operator reads when the agent asks for
them at Assist.

The Demo is an Arctis here: a compustage whose autoloader holds G1, G2 and an
unnamed third grid.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.server.catalog import CATALOG, route_call  # noqa: E402
from fibsem.server.request_text import describe_action  # noqa: E402
from tests.fixtures.demo_stage import demo_grid_loader, demo_session  # noqa: E402
from tests.server.test_agent_requests import Host, RecordingContext  # noqa: E402

TOKEN = "agent"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TOOLS = {spec.name: spec for spec in CATALOG}
RUN_STATE = "/microscope/services/sample_handling/run"


@pytest.fixture
def arctis(monkeypatch):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = demo_session(compustage=True)
    microscope.sample_handling.magazine = demo_grid_loader(
        microscope, occupied=(1, 2, 3), names={1: "G1", 2: "G2"}
    )
    return microscope


@pytest.fixture
def driving(arctis):
    app = build_server(arctis, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def _call(client, tool, **arguments):
    method, path, body = route_call(TOOLS[tool], arguments)
    return client.request(method, path, json=body)


def _ran(client, tool, **arguments):
    """A long grid command, to its end: the run as ``GET .../run`` gives it."""
    answer = _call(client, tool, **arguments)
    assert answer.status_code == 200, answer.text
    run = answer.json()["result"]
    assert run["service"] == "sample_handling"
    ended = client.get(RUN_STATE, params={"run_id": run["run_id"], "wait": 10})
    return ended.json()["run"]


def _states(client):
    answer = _call(client, "grid_get_inventory")
    assert answer.status_code == 200, answer.text
    return {e["name"]: e["state"] for e in answer.json()["result"] if e["name"]}


def test_the_agent_exchanges_grids_by_name(driving):
    assert _states(driving) == {
        "G1": "occupied",
        "G2": "occupied",
        "Grid-03": "occupied",
    }

    loaded = _ran(driving, "grid_load", grid_name="G1")
    assert loaded["state"] == "done", loaded
    assert loaded["result"]["loaded_grid"]["name"] == "G1"
    assert _states(driving)["G1"] == "loaded"

    # The exchange: G1 goes back to the magazine, G2 comes in.
    assert _ran(driving, "grid_load", grid_name="G2")["state"] == "done"
    assert _states(driving) == {
        "G1": "occupied",
        "G2": "loaded",
        "Grid-03": "occupied",
    }

    moved = _call(driving, "grid_move_to", grid_name="G2")
    assert moved.status_code == 200, moved.text
    here = _call(driving, "grid_get_current").json()["result"]
    assert here["grid"] == "G2"
    slots = _call(driving, "grid_get_holder_slots").json()["result"]
    assert here["slot"] == slots[0]["name"]

    assert _ran(driving, "grid_unload")["state"] == "done"
    assert "loaded" not in _states(driving).values()

    scanned = _ran(driving, "grid_run_inventory")
    assert {e["name"] for e in scanned["result"] if e["name"]} == {
        "G1",
        "G2",
        "Grid-03",
    }


def test_a_grid_not_in_the_magazine_ends_refused_and_says_what_to_do(driving):
    ended = _ran(driving, "grid_load", grid_name="G9")
    assert ended["state"] == "failed"
    assert ended["error"]["error_type"] == "GridExchangeError"
    assert "Run an inventory" in ended["error"]["message"]


def test_a_grid_not_on_the_stage_is_not_moved_to(driving):
    refused = _call(driving, "grid_move_to", grid_name="G2")
    assert refused.status_code == 422
    assert "not in any holder slot" in refused.json()["detail"]["message"]


def test_the_grid_tools_need_drive_and_the_reads_watch(arctis):
    auth = AuthConfig.generate(level="watch", token=TOKEN)
    with TestClient(build_server(arctis, auth=auth)) as watching:
        watching.headers.update(AUTH)
        assert _call(watching, "grid_get_inventory").status_code == 200
        assert _call(watching, "grid_load", grid_name="G1").status_code == 403
        assert _call(watching, "grid_unload").status_code == 403


# --- what the operator reads ------------------------------------------------------


def _inventory(*rows, source="magazine"):
    return {
        "/microscope/services/sample_handling/queries/get_inventory": {
            "result": [
                {"slot_name": slot, "name": name, "state": state, "source": source}
                for slot, name, state in rows
            ]
        }
    }


EXCHANGE = _inventory(("Slot-01", "G1", "loaded"), ("Slot-02", "G2", "occupied"))
EMPTY_STAGE = _inventory(("Slot-01", "G1", "occupied"), ("Slot-02", "G2", "occupied"))
FIXED = _inventory(("Slot-01", "G1", "loaded"), source="holder")
SLOTS = {
    "/microscope/services/sample_handling/queries/holder_slots": {
        "result": [
            {
                "name": "Shuttle",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0},
                "loaded_grid": {"name": "G2"},
            }
        ]
    },
    "/microscope/devices/stage/position": {
        "value": {"x": 1e-3, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0}
    },
}


@pytest.mark.parametrize(
    "tool, arguments, answers, change",
    [
        (
            "grid_load",
            {"grid_name": "G2"},
            EXCHANGE,
            "Load grid 'G2' from Slot-02 (unloads 'G1' first)",
        ),
        ("grid_load", {"grid_name": "G2"}, EMPTY_STAGE, "Load grid 'G2' from Slot-02"),
        (
            "grid_load",
            {"grid_name": "G1"},
            EXCHANGE,
            "Load grid 'G1': it is on the stage already, nothing moves",
        ),
        (
            "grid_load",
            {"grid_name": "G1"},
            FIXED,
            "Load grid 'G1': it is in Slot-01 already, nothing moves",
        ),
        (
            "grid_load",
            {"grid_name": "G9"},
            EXCHANGE,
            "Load grid 'G9': it is not in the inventory",
        ),
        ("grid_load", {"grid_name": "G2"}, {}, "Load grid 'G2'"),
        ("grid_unload", {}, EXCHANGE, "Return grid 'G1' to the magazine"),
        ("grid_unload", {}, EMPTY_STAGE, "Unload: no grid is on the stage"),
        ("grid_unload", {}, FIXED, "Unload: nothing to do on a fixed holder"),
        (
            "grid_move_to",
            {"grid_name": "G2"},
            SLOTS,
            "Move to grid 'G2', in Shuttle: x 0 µm, y 0 µm, z 0 µm, tilt 0°, "
            "rotation 0°\nfrom x 1 mm, y 0 µm, z 0 µm, tilt 0°, rotation 0°",
        ),
        (
            "grid_move_to",
            {"grid_name": "G1"},
            SLOTS,
            "Move to grid 'G1': it is not on the stage, so it will be refused",
        ),
    ],
)
def test_a_grid_action_reads_as_what_the_loader_will_do(
    tool, arguments, answers, change
):
    described = describe_action(tool, arguments, "/x", read=answers.get)
    assert described["change"] == change
    # Called through the generic service command, it reads the same.
    command = TOOLS[tool].path.rsplit("/", 1)[1]
    generic = describe_action(
        "service_call_command",
        {"service": "sample_handling", "command": command, "kwargs": arguments},
        "/x",
        read=answers.get,
    )
    assert generic == described


def test_assist_asks_for_an_exchange_and_the_operator_reads_it(arctis, tmp_path):
    host = Host()
    host.microscope = arctis
    auth = AuthConfig(token=TOKEN, dashboard_token="dashboard")
    auth.set_level(AgentLevel.ASSIST)
    app = build_server(arctis, app_context=RecordingContext(host), auth=auth)
    arctis.sample_handling.ensure_loaded("G1")
    with TestClient(app, raise_server_exceptions=False) as client:
        filed = client.post(
            "/app/agent/requests",
            headers=AUTH,
            json={
                "actions": [{"tool": "grid_load", "arguments": {"grid_name": "G2"}}],
                "reason": "G2 is next on the screening list",
            },
        )
        assert filed.status_code == 202, filed.text
        (action,) = filed.json()["actions"]
        assert (action["what"], action["change"]) == (
            "Grids",
            "Load grid 'G2' from Slot-02 (unloads 'G1' first)",
        )

        store = app.state.requests
        store.approve(filed.json()["request_id"])
        assert store.wait(filed.json()["request_id"], timeout=10).status == "applied"
        run = client.get(RUN_STATE, headers=AUTH, params={"wait": 10}).json()["run"]
        assert run["state"] == "done"
    assert [g.name for g in arctis.sample_handling.loaded_grids] == ["G2"]
