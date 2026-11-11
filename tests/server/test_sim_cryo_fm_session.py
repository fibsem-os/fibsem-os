"""A scripted agent session on the simulated cryo-FM, end to end: the MCP
tools a cryo-fm session is offered, called the way its skill says, through the
App Server to the Demo devices. Only the model is stood in, by the script, so
nothing here spends model usage."""

import asyncio
import json
import os
import threading
import time

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.ai.agent.mcp.sidecar import build_sidecar  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402

CONFIGURATION = os.path.join(cfg.SHIPPED_CONFIG_PATH, "cryo-fm-demo-configuration.yaml")
TOKEN = "agent-token"
CRYO_FM = "cryo-fm"


class Agent:
    """The stand-in model: calls a tool by name and reads its answer."""

    def __init__(self, sidecar):
        self.sidecar = sidecar

    def tools(self) -> set:
        listed = asyncio.run(self.sidecar.list_tools())
        return {t.name for t in getattr(listed, "tools", listed)}

    def call(self, name: str, **arguments):
        """(the answer's JSON, the images it carries)."""
        try:
            result = asyncio.run(self.sidecar.call_tool(name, arguments))
        except Exception as exc:  # a refusal reaches the model as an error result
            return {"error": str(exc)}, []
        if isinstance(result, tuple):
            result = result[0]
        contents = list(getattr(result, "content", result))
        text = "".join(getattr(c, "text", "") for c in contents)
        images = [c for c in contents if getattr(c, "type", "") == "image"]
        try:
            return json.loads(text), images
        except ValueError:
            return text, images


@pytest.fixture
def session():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    session = AppSession()
    yield session
    session.disconnect()


def _agent(session, level) -> Agent:
    auth = AuthConfig.generate(level=level, token=TOKEN)
    app = build_server(session=session, auth=auth, operator=session.agent)
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    client.__enter__()
    session.connect(CONFIGURATION)
    capabilities = client.get("/capabilities").json()
    return Agent(build_sidecar(client, capabilities, CRYO_FM))


def _delta(**axes) -> dict:
    return {
        "delta": {
            axis: axes.get(axis)
            for axis in ("x", "y", "z", "r", "t", "coordinate_system")
        }
    }


def test_a_cryo_fm_session_images_focuses_and_moves(session):
    agent = _agent(session, AgentLevel.DRIVE)

    # Offered the device tools, none of a FIB-SEM's.
    tools = agent.tools()
    assert {"device_list", "device_call_command", "stage_get_position"} <= tools
    assert not {"beam_acquire_image", "milling_run_stage", "fm_acquire_image"} & tools

    devices, _ = agent.call("device_list")
    assert {"fm", "camera", "light_source", "filter_set", "objective", "stage"} <= set(
        devices
    )

    # The channel, then one image: its summary and a preview to look at.
    for device, parameter, value in [
        ("light_source", "power", 0.2),
        ("camera", "exposure_time", 0.05),
    ]:
        answer, _ = agent.call(
            "device_set_parameter", device=device, parameter=parameter, value=value
        )
        assert answer["value"] == pytest.approx(value), answer
    frame, images = agent.call(
        "device_call_command", device="fm", command="acquire_frame"
    )
    assert frame["result"]["summary"]["kind"] == "frame", frame
    assert len(images) == 1

    # Focus: a step of the objective, read back.
    before, _ = agent.call(
        "device_get_parameter", device="objective", parameter="position"
    )
    agent.call(
        "device_call_command",
        device="objective",
        command="move_relative",
        kwargs={"delta": 2e-6},
    )
    after, _ = agent.call(
        "device_get_parameter", device="objective", parameter="position"
    )
    assert after["value"] - before["value"] == pytest.approx(2e-6)

    # A z-stack of one channel through focus; the objective comes back after.
    focus = after["value"]
    channel = {"name": "green", "excitation_wavelength": 450.0, "power": 0.2}
    stack, _ = agent.call(
        "device_call_command",
        device="fm",
        command="acquire_z_stack",
        kwargs={
            "channels": [channel],
            "positions": [focus - 1e-6, focus, focus + 1e-6],
            "restore_position": focus,
        },
    )
    assert stack["result"]["summary"]["count"] == 3, stack
    back, _ = agent.call(
        "device_get_parameter", device="objective", parameter="position"
    )
    assert back["value"] == pytest.approx(focus)

    # The stage: a relative move by x and z, read back.
    start, _ = agent.call("stage_get_position")
    moved, _ = agent.call(
        "device_call_command",
        device="stage",
        command="move_relative",
        kwargs=_delta(x=5e-5, z=-1e-6),
    )
    assert "error" not in moved, moved
    end, _ = agent.call("stage_get_position")
    assert end["position"]["x"] - start["position"]["x"] == pytest.approx(5e-5)
    assert end["position"]["z"] - start["position"]["z"] == pytest.approx(-1e-6)

    # An absolute move is the operator's.
    refused, _ = agent.call(
        "device_call_command",
        device="stage",
        command="move_absolute",
        kwargs={"position": _delta(x=0, y=0, z=0)["delta"]},
    )
    assert "not_allowed" in refused["error"], refused
    assert agent.call("stage_get_position")[0]["position"] == end["position"]


def test_at_watch_it_reads_but_moves_nothing(session):
    agent = _agent(session, AgentLevel.WATCH)
    start, _ = agent.call("stage_get_position")
    refused, _ = agent.call(
        "device_call_command",
        device="stage",
        command="move_relative",
        kwargs=_delta(x=5e-5),
    )
    assert '"error_type": "level"' in refused["error"], refused
    assert agent.call("stage_get_position")[0]["position"] == start["position"]


def test_the_agent_cancels_its_running_z_stack(session):
    agent = _agent(session, AgentLevel.DRIVE)
    fm = session.instrument.devices["fm"]
    focus = session.instrument.devices["objective"].position.get_value()
    started = threading.Event()
    frames = []

    def counted(*args, **kwargs):
        started.set()
        frames.append(1)
        time.sleep(0.02)
        return acquire(*args, **kwargs)

    acquire = fm.acquire_frame
    fm.acquire_frame = counted
    stack = {}
    running = threading.Thread(
        target=lambda: stack.update(
            answer=agent.call(
                "device_call_command",
                device="fm",
                command="acquire_z_stack",
                kwargs={
                    "channels": [{"name": "green", "excitation_wavelength": 450.0}],
                    "positions": [focus + i * 1e-7 for i in range(500)],
                    "restore_position": focus,
                },
            )[0]
        )
    )
    running.start()
    assert started.wait(10)

    # A stop needs its reason, and takes no control from the running stack.
    unsaid, _ = agent.call("device_call_command", device="fm", command="cancel")
    assert "missing_reason" in unsaid["error"], unsaid
    cancelled, _ = agent.call(
        "device_call_command", device="fm", command="cancel", reason="wrong channel"
    )
    assert "error" not in cancelled, cancelled
    running.join(30)
    assert not running.is_alive()
    assert len(frames) < 500
    assert stack["answer"]["result"] == [], stack
