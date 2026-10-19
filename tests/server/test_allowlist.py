"""The App Server's allowlist: what the agent may never touch, at any level
(Patrick, 2026-10-14). The level decides how it acts; the allowlist whether it
ever may.

Skipped when the [server] extra is not installed.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ai.agent.allowlist import blocked, path_outside  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


@pytest.fixture(scope="module")
def client(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def _error(response) -> str:
    return response.json()["detail"]["error_type"]


def test_a_blocked_entry_is_refused_at_drive_and_stays_readable(client):
    refused = client.put("/microscope/devices/electron/on", json={"value": False})
    assert refused.status_code == 403
    assert _error(refused) == "not_allowed"
    message = refused.json()["detail"]["message"]
    assert "may not set electron.on" in message
    assert "stays readable" in message
    assert client.get("/microscope/devices/electron/on").status_code == 200

    for device, command in (
        ("stage", "home"),
        ("stage", "link"),
        ("stage", "move_absolute"),
        ("manipulator", "insert"),
    ):
        refused = client.post(
            f"/microscope/devices/{device}/commands/{command}", json={"kwargs": {}}
        )
        assert refused.status_code == 403, (device, command)
        assert _error(refused) == "not_allowed"
        message = refused.json()["detail"]["message"]
        assert f"may not run {device}.{command}" in message
        assert "readable" not in message

    # A mill started bare returns at once, holding no control while it mills.
    refused = client.post(
        "/microscope/services/milling/commands/start", json={"kwargs": {}}
    )
    assert refused.status_code == 403
    assert "milling_run_stage" in refused.json()["detail"]["message"]


def test_risky_but_legitimate_entries_are_left_to_the_level(client, microscope):
    voltage = microscope.devices["electron"].voltage.get_value()
    allowed = client.put(
        "/microscope/devices/electron/voltage", json={"value": voltage}
    )
    assert allowed.status_code == 200, allowed.text


def test_the_listings_mark_what_the_agent_may_not_use(client):
    devices = client.get("/microscope/devices").json()
    electron = devices["electron"]
    assert electron["parameters"]["on"]["agent"] is False
    assert electron["parameters"]["voltage"]["agent"] is True
    assert devices["stage"]["commands"]["home"]["agent"] is False
    assert devices["stage"]["commands"]["move_absolute"]["agent"] is False
    assert all(
        not info["agent"] for info in devices["manipulator"]["commands"].values()
    )
    # Nobody can set the chamber's state; reading it is never refused.
    assert devices["chamber"]["parameters"]["state"]["agent"] is True
    services = client.get("/microscope/services").json()
    assert services["stage_movement"]["commands"]["move_to_device"]["agent"] is True
    # The raw absolute move is blocked; the safe one is the agent's.
    commands = services["stage_movement"]["commands"]
    assert commands["move_absolute"]["agent"] is False
    assert commands["safe_absolute_stage_movement"]["agent"] is True


def test_a_path_is_refused_outside_the_open_experiment(client, tmp_path):
    # No experiment is open here, so no path is allowed at all.
    refused = client.post(
        "/microscope/services/imaging/commands/focus_sweep",
        json={"kwargs": {"path": str(tmp_path)}},
    )
    assert refused.status_code == 403
    assert _error(refused) == "not_allowed"
    # Refused for its path, not for being the focus sweep.
    message = refused.json()["detail"]["message"]
    assert message.startswith("imaging.focus_sweep refused:")
    assert "may not" not in message

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    inside = str(experiment / "focus")
    assert path_outside({"path": inside}, lambda: str(experiment)) is None
    assert path_outside({}, lambda: None) is None
    assert path_outside({"path": str(tmp_path / "elsewhere")}, lambda: str(experiment))
    assert path_outside(
        {"path": str(experiment / ".." / "elsewhere")}, lambda: str(experiment)
    )


def test_an_image_saves_inside_the_open_experiment_only(client, tmp_path):
    from fibsem.structures import ImageSettings

    refused = client.post(
        "/microscope/services/imaging/commands/acquire_image",
        json={
            "kwargs": {
                "image_settings": {"save": True, "path": str(tmp_path), "filename": "x"}
            }
        },
    )
    assert refused.status_code == 403
    assert _error(refused) == "not_allowed"

    experiment = str(tmp_path)
    saving = ImageSettings(save=True, path=str(tmp_path / "elsewhere" / ".." / "in"))
    assert path_outside({"image_settings": saving}, lambda: experiment) is None
    saving.path = "/"
    assert path_outside({"image_settings": saving}, lambda: experiment)
    saving.save = False  # nothing is written, so the path doesn't matter
    assert path_outside({"image_settings": saving}, lambda: experiment) is None


def test_blocks_go_by_the_kind_of_device_not_its_name(microscope):
    assert blocked(microscope.devices["ion"], "write", "on")
    assert blocked(microscope.devices["ion"], "write", "hfw") is None
    assert blocked(microscope.devices["stage"], "command", "move_relative") is None
    assert blocked(microscope.devices["manipulator"], "command", "stop")
