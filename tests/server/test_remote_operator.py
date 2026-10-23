"""The operator off the PC: the operator token reaches the agent's routes as the
operator, whatever the agent's level. The blocklist still applies.

Skipped when the [server] extra is not installed.
"""

import threading

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fibsem.acting import OPERATOR, current_actor  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.client.remote_microscope import RemoteMicroscope  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.devices import WS_PROTOCOL  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402
from fibsem.workflows.actions import ACTION  # noqa: E402

TOKEN = "agent-token"
OPERATOR_TOKEN = "operator-token"


@pytest.fixture(scope="module")
def auth():
    auth = AuthConfig.generate(level=AgentLevel.OFF, token=TOKEN)
    auth.operator_token = OPERATOR_TOKEN
    return auth


@pytest.fixture(scope="module")
def url(microscope, auth):
    app = build_server(microscope, auth=auth)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", ws=WS_PROTOCOL
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "App Server failed to start"
        threading.Event().wait(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def operator(url):
    remote = RemoteMicroscope.connect(url, OPERATOR_TOKEN)
    yield remote
    remote.disconnect()


@pytest.fixture(autouse=True)
def agent_off(auth):
    yield
    auth.set_level(AgentLevel.OFF)


def _recording(microscope):
    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload, current_actor()))

    microscope.session.record_signal.connect(record)
    return recorded, lambda: microscope.session.record_signal.disconnect(record)


def test_the_operator_moves_the_stage_with_the_agent_off(
    operator, microscope, auth, monkeypatch
):
    taken = []
    take = microscope.control.take

    def taking(by, action):
        taken.append(by)
        return take(by, action)

    monkeypatch.setattr(microscope.control, "take", taking)
    start = microscope.get_stage_position()
    try:
        operator.move_stage_relative(FibsemStagePosition(x=5e-6))
        assert microscope.get_stage_position().x == pytest.approx(start.x + 5e-6)
    finally:
        microscope.move_stage_absolute(start)
    assert taken and all(by is OPERATOR for by in taken)
    # Not the agent being heard from.
    assert auth.seconds_since_seen() is None


def test_the_blocklist_holds_for_the_operator_off_the_pc(operator):
    with pytest.raises(Exception, match="403"):
        operator.devices["manipulator"].insert()


def test_the_operators_stop_needs_no_reason_and_is_theirs(operator, microscope):
    recorded, done = _recording(microscope)
    try:
        answer = operator.client.transport.request(
            "POST",
            "/microscope/devices/electron/commands/stop_live",
            json={"kwargs": {}},
        )
    finally:
        done()
    assert answer.status_code == 200, answer.text
    (stop,) = [(p, actor) for kind, p, actor in recorded if kind == ACTION]
    payload, actor = stop
    assert actor is OPERATOR
    assert payload["stops"] == "stop live" and payload["via"] == "remote"
    assert "level" not in payload
