"""The simulated standalone cryo-FM: the shipped Demo configuration connects an
FM and an x, y, z stage with no microscope, the App Server serves them, and an
agent on the cryo-fm playbook is offered its tools."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fibsem.config as cfg
from fibsem.ai.agent import instruments, playbooks
from fibsem.ai.agent.catalog import tools_for_capabilities
from fibsem.ai.agent.levels import AgentLevel
from fibsem.instrument import DeviceInstrument
from fibsem.server import AuthConfig, build_server
from fibsem.workflows.session import AppSession

ROOT = Path(__file__).resolve().parents[2]
CRYO_FM = "cryo-fm"
CONFIGURATION = os.path.join(cfg.SHIPPED_CONFIG_PATH, "cryo-fm-demo-configuration.yaml")
AGENT = {"Authorization": "Bearer agent-token"}
UNMOVED = {"y": None, "z": None, "r": None, "t": None, "coordinate_system": None}


@pytest.fixture
def client():
    session = AppSession()
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token="agent-token")
    app = build_server(session=session, auth=auth, operator=session.agent)
    with TestClient(app) as client:
        client.instrument = session.connect(CONFIGURATION)
        yield client
    session.disconnect()


def test_the_demo_cryo_fm_is_an_fm_and_an_xyz_stage(client):
    assert isinstance(client.instrument, DeviceInstrument)
    devices = client.get("/microscope/devices", headers=AGENT).json()
    assert {"fm", "camera", "objective", "stage"} <= set(devices)
    stage = client.instrument.devices["stage"]
    assert list(stage.axes) == ["x", "y", "z"]

    moved = client.post(
        "/microscope/devices/stage/commands/move_relative",
        headers=AGENT,
        json={"kwargs": {"delta": {"x": 1e-4, **UNMOVED}}},
    )
    assert moved.status_code == 200, moved.text
    assert stage.position.get_value().x == pytest.approx(1e-4)

    image = client.post(
        "/microscope/devices/fm/commands/acquire_frame",
        headers=AGENT,
        json={"kwargs": {}, "answer": "preview"},
    )
    assert image.status_code == 200, image.text
    assert image.json()["result"]["summary"]["kind"] == "frame"


def test_an_agent_on_the_cryo_fm_playbook_gets_its_tools():
    playbook = playbooks.load("cryo-fm", ROOT)
    assert playbook.instrument == CRYO_FM
    assert playbook.skills == ["fibsem-agent", "cryo-fm"]
    # The base's session settings hold, not its FIB-SEM rules.
    assert playbook.session.max_turns
    assert not any("coincidence" in rule.lower() for rule in playbook.rules)
    instruments.load(CRYO_FM, ROOT)

    capabilities = {"routers": {"services": True, "app": True}}
    serves = instruments.serves(CRYO_FM, ROOT)
    tools = {t.name for t in tools_for_capabilities(capabilities, serves)}
    assert {"device_call_command", "stage_get_position", "plan_get"} <= tools
    assert not {"milling_run_stage", "beam_acquire_image", "fm_acquire_image"} & tools


def test_a_session_keeps_to_its_instrument_kinds_playbooks():
    from fibsem.ai.agent import harness

    # A FIB-SEM session loads the FIB-SEM playbooks' skills, not the cryo-FM's.
    assert "cryo-fm" not in harness.session_names(ROOT).skills
    cryo = playbooks.load("cryo-fm", ROOT)
    assert harness.session_names(ROOT, cryo).skills[:2] == ["fibsem-agent", "cryo-fm"]

    # Its tools are fixed at its start: it can't switch to another kind's playbook.
    reminders = playbooks.Reminders()
    reminders.set(cryo, "launch")
    control = harness.PlaybookControl(reminders, ROOT)
    listed = [row["name"] for row in json.loads(control.list())["playbooks"]]
    assert listed == ["cryo-fm"]
    assert control.set("autolamella", "agent").startswith("Refused: the autolamella")
    assert reminders.playbook.name == "cryo-fm"
