"""The client (`fibsem.client`) over the App Server: devices and services reached by
name from the server's descriptions, values and results read as their types, and
refusals raised as the exceptions they name.

Skipped when the [server] extra is not installed.
"""

import inspect
import os
import subprocess
import sys
import threading

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.cancellation import NothingToStop  # noqa: E402
from fibsem.client import (  # noqa: E402
    FibsemClient,
    Remote,
    ServerError,
    ServerUnreachable,
    Transport,
)
from fibsem.devices.core import _Controllable  # noqa: E402
from fibsem.devices.stage import StageLimitError  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures.image import FibsemImage  # noqa: E402
from fibsem.structures.settings import (  # noqa: E402
    BeamType,
    FibsemStagePosition,
    MicroscopeState,
    MillingState,
    Point,
    ScanMode,
)

TOKEN = "test-token"


class _App(TestClient):
    def request(self, *args, timeout=None, **kwargs):  # it refuses a timeout
        return super().request(*args, **kwargs)


class InProcess(Transport):
    """A transport that sends to *app* in process."""

    def __init__(self, app, token=TOKEN, **kwargs):
        self.app = app
        super().__init__("http://testserver", token, **kwargs)

    def _client(self, url, headers):
        return _App(
            self.app, base_url=url, headers=headers, raise_server_exceptions=False
        )


def in_process(app, token=TOKEN):
    """A client of *app*, in process."""
    return FibsemClient(transport=InProcess(app, token))


@pytest.fixture(scope="module")
def app(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app):  # its startup
        yield app


@pytest.fixture
def client(app):
    with in_process(app) as client:
        yield client


def test_the_devices_and_services_are_the_servers(client, microscope):
    assert set(client.devices) == set(microscope.devices)
    assert set(client.services) == set(microscope.services)
    assert "correlation" in client.analysis
    with pytest.raises(KeyError, match="there are electron"):
        client.devices["nothing"]


def test_a_read_is_live_and_typed(client, microscope):
    electron = client.devices["electron"]
    microscope.set_field_of_view(140e-6, BeamType.ELECTRON)
    assert electron.get("hfw") == pytest.approx(140e-6)
    assert isinstance(electron.get("stigmation"), Point)
    assert electron.get("resolution") == tuple(
        microscope.get_resolution(BeamType.ELECTRON)
    )
    assert isinstance(electron.get("scanning_mode"), ScanMode)
    assert isinstance(client.devices["stage"].get("position"), FibsemStagePosition)
    assert client.services["milling"].get("state") is MillingState.IDLE


def test_a_write_reaches_the_microscope_and_answers_the_value(client, microscope):
    written = client.devices["ion"].set("stigmation", Point(0.01, -0.01))
    assert isinstance(written, Point)
    assert microscope.get_stigmation(BeamType.ION).x == pytest.approx(0.01)
    client.devices["electron"].set("hfw", 150e-6)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(150e-6)


def test_values_and_metadata(client, microscope):
    values = client.devices["electron"].values()
    assert isinstance(values["shift"], Point)
    assert values["voltage"] == microscope.get_beam_voltage(BeamType.ELECTRON)
    assert "choices" in client.devices["ion"].metadata("current")


def test_a_beam_image_comes_back_as_an_image(client):
    image = client.devices["electron"].acquire()
    assert isinstance(image, FibsemImage)
    assert image.data.ndim == 2


def test_a_query_and_a_command_are_typed_by_what_they_return(client, microscope):
    stage_movement = client.services["stage_movement"]
    assert stage_movement.stage_orientation() == microscope.get_stage_orientation()
    here = microscope.get_stage_position()
    projected = stage_movement.project_stable_move(
        dx=2e-6, dy=0.0, beam_type=BeamType.ELECTRON, base_position=here
    )
    expected = microscope.project_stable_move(2e-6, 0.0, BeamType.ELECTRON, here)
    assert isinstance(projected, FibsemStagePosition)
    assert projected.x == pytest.approx(expected.x)
    moved = stage_movement.move_relative(delta=FibsemStagePosition(x=1e-6))
    assert isinstance(moved, FibsemStagePosition)


def test_a_service_parameter_is_written(client, microscope):
    stored = microscope.system.stage.milling_angle
    try:
        client.services["stage_movement"].set("milling_angle", 22.0)
        assert microscope.system.stage.milling_angle == pytest.approx(22.0)
    finally:
        microscope.set_milling_angle(stored)


def test_the_microscope_as_a_whole(client, microscope):
    fitted = microscope.is_available("manipulator")
    assert client.system.manipulator.enabled is fitted
    assert client.info()["manufacturer"] == microscope.manufacturer
    state = client.state()
    assert isinstance(state, MicroscopeState)
    client.set_state(state)
    assert client.capabilities()["level"] == "drive"


def test_a_refusal_raises_the_exception_it_names(client, microscope):
    before = microscope.stage.position.get_value()
    target = FibsemStagePosition(x=before.x, y=1.0, z=before.z, r=before.r, t=before.t)
    with pytest.raises(StageLimitError):
        client.services["stage_movement"].safe_absolute_stage_movement(
            stage_position=target
        )
    assert microscope.stage.position.get_value() == before


def test_a_refusal_with_no_local_exception_is_a_server_error(client):
    with pytest.raises(ServerError) as raised:
        client.services["milling"].stop()
    assert raised.value.status == 422
    assert raised.value.error_type == "missing_reason"


def test_a_stop_carries_its_reason(client):
    # Nothing is milling, so the stop is refused for that, not for its reason.
    with pytest.raises(NothingToStop):
        client.services["milling"].stop(reason="a test stops it")
    with pytest.raises(NothingToStop):
        client.services["milling"].end(reason="a test ends it")


def test_the_level_refuses_a_write_at_watch(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="watch", token=TOKEN))
    with TestClient(app), in_process(app) as client:
        assert client.devices["electron"].get("hfw") > 0
        with pytest.raises(ServerError) as raised:
            client.devices["electron"].set("hfw", 150e-6)
    assert raised.value.status == 403


def test_a_wrong_token_is_refused(app):
    with pytest.raises(ServerError) as raised:
        in_process(app, token="wrong").capabilities()
    assert raised.value.status == 401


def test_a_name_that_is_neither_command_nor_query_says_what_there_is(client):
    with pytest.raises(AttributeError, match="move_relative"):
        client.services["stage_movement"].fly()


def test_no_served_command_or_query_is_hidden_by_the_clients_own_names():
    """``remote.get`` must stay the read, not a command a device calls ``get``."""
    own = {name for name in dir(Remote) if not name.startswith("_")}
    own |= {"name", "path", "description"}
    pending, hidden = list(_Controllable.__subclasses__()), set()
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            served = hasattr(method, "__device_command__") or hasattr(
                method, "__device_query__"
            )
            if served and name in own:
                hidden.add(f"{cls.__name__}.{name}")
    assert not hidden


def test_importing_the_client_needs_no_server_and_no_drivers():
    code = (
        "import sys, fibsem.client; "
        "print([m for m in sys.modules if m.split('.')[0] in "
        "('fastapi', 'starlette', 'uvicorn', 'requests') "
        "or m.startswith('fibsem.drivers')])"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}  # this fibsem
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.stdout.strip() == "[]", out.stderr


# -- the transport -----------------------------------------------------------------


def test_with_nothing_given_the_transport_finds_no_server(monkeypatch):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr("fibsem.client.transport.read_discovery_file", lambda: None)
    with pytest.raises(ServerUnreachable, match="No App Server found"):
        Transport()


def test_a_restarted_server_is_followed(app, monkeypatch):
    """A 401 from a server that came back with a new token: the transport reads the
    discovery file again and retries there once."""
    found = {"url": "http://testserver", "token": "old", "pid": os.getpid()}
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr("fibsem.client.transport.read_discovery_file", lambda: found)

    class Discovered(InProcess):
        def __init__(self):
            self.app = app
            Transport.__init__(self)

    transport = Discovered()
    found["token"] = TOKEN
    assert transport.fetch("GET", "/capabilities")["level"] == "drive"


class _Mocked(Transport):
    """A transport whose servers are *answers*, by url."""

    def __init__(self, answers, url=None, token=None):
        self.answers = answers
        super().__init__(url, token)

    def _client(self, url, headers):
        return httpx.Client(
            transport=httpx.MockTransport(self.answers[url]),
            base_url=url,
            headers=headers,
        )


def _discovered(monkeypatch, found):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setattr("fibsem.client.transport.read_discovery_file", lambda: found)


def test_resolve_takes_what_is_given_then_the_environment_then_discovery(
    monkeypatch,
):
    from fibsem.client.transport import resolve

    _discovered(monkeypatch, None)
    assert resolve() is None
    assert resolve("http://x", None) is None  # a url without a token
    assert resolve("http://x", "t") == ("http://x", "t")
    monkeypatch.setenv("FIBSEM_SERVER_TOKEN", "e")
    assert resolve("http://x") == ("http://x", "e")
    _discovered(monkeypatch, {"url": "http://d", "token": "d"})
    assert resolve() == ("http://d", "d")


def test_a_401_with_nothing_new_to_find_is_answered_as_is(monkeypatch):
    _discovered(monkeypatch, {"url": "http://x", "token": "t"})
    transport = _Mocked({"http://x": lambda request: httpx.Response(401)})
    assert transport.request("GET", "/capabilities").status_code == 401


def test_a_refused_connection_follows_a_moved_server_with_its_user_agent(
    monkeypatch,
):
    found = {"url": "http://x", "token": "t"}
    _discovered(monkeypatch, found)
    seen = []

    def gone(request):
        raise httpx.ConnectError("refused")

    def moved(request):
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, json={"ok": True})

    transport = _Mocked({"http://x": gone, "http://y": moved})
    transport.user_agent = "fibsem-mcp/0.6 claude-code/2.1.3"
    found["url"] = "http://y"
    assert transport.fetch("GET", "/app/workflow/status") == {"ok": True}
    assert transport.url == "http://y"
    assert seen == ["fibsem-mcp/0.6 claude-code/2.1.3"]


def test_a_request_that_timed_out_is_not_sent_again(monkeypatch):
    found = {"url": "http://x", "token": "t"}
    _discovered(monkeypatch, found)
    sent = []

    def slow(request):
        sent.append(request.url.path)
        raise httpx.ReadTimeout("no answer")

    transport = _Mocked({"http://x": slow, "http://y": slow})
    found["url"] = "http://y"
    with pytest.raises(ServerUnreachable, match="no answer in time"):
        transport.request("POST", "/microscope/devices/stage/commands/move")
    assert sent == ["/microscope/devices/stage/commands/move"]


@pytest.fixture(scope="module")
def port(microscope):
    """The App Server on a free port."""
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "App Server failed to start"
        threading.Event().wait(0.01)
    yield server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(timeout=5)


def test_over_the_network_from_the_environment(port, monkeypatch):
    monkeypatch.setenv("FIBSEM_SERVER_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("FIBSEM_SERVER_TOKEN", TOKEN)
    seen = []
    with FibsemClient(user_agent="a-test/1") as client:
        client.transport._http.event_hooks["request"].append(seen.append)
        assert isinstance(client.devices["ion"].acquire(), FibsemImage)
    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert seen[0].headers["User-Agent"] == "a-test/1"


def test_a_server_that_is_not_there_is_unreachable():
    with FibsemClient("http://127.0.0.1:9", TOKEN) as client:
        with pytest.raises(ServerUnreachable, match="127.0.0.1:9"):
            client.capabilities()
