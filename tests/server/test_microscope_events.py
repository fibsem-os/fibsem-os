"""The App Server's ``WS /microscope/events``: devices' and services' changes as
they happen, to the remote driver's devices and to the client, behind the agent
token at Watch, and never to a browser page.

Skipped when the [server] extra is not installed.
"""

import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
websockets = pytest.importorskip("websockets")

from websockets.exceptions import ConnectionClosed, InvalidStatus  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from fibsem.client import FibsemClient  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    DeviceClient,
    connect_remote_beams,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.devices import WS_PROTOCOL  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402

TOKEN = "test-token"


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture(scope="module")
def auth():
    return AuthConfig.generate(level="drive", token=TOKEN)


@pytest.fixture(scope="module")
def port(microscope, auth):
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
    yield server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def at_drive(auth):
    yield
    auth.set_level("drive")


def _url(port):
    return f"ws://127.0.0.1:{port}/microscope/events"


def test_a_remote_device_cache_follows_a_change_made_on_the_microscope(
    port, microscope
):
    client = DeviceClient("127.0.0.1", port, token=TOKEN, root="/microscope")
    try:
        beams = connect_remote_beams("127.0.0.1", port, client=client)
        hfw = beams[BeamType.ELECTRON].parameters["hfw"]
        assert client.connected
        seen = []
        hfw.changed.connect(seen.append)
        microscope.set_field_of_view(175e-6, BeamType.ELECTRON)
        assert wait_for(lambda: seen == [pytest.approx(175e-6)])
        assert hfw.cached == pytest.approx(175e-6)
    finally:
        client.close()


def test_the_client_signals_device_changes_and_service_signals(port, microscope):
    client = FibsemClient(f"http://127.0.0.1:{port}", TOKEN)
    try:
        client.follow()
        changed, signalled = [], []
        client.devices["electron"].changed.connect(
            lambda name, value: changed.append((name, value))
        )
        client.services["sample_handling"].signalled.connect(
            lambda name, value: signalled.append((name, value))
        )
        microscope.set_field_of_view(185e-6, BeamType.ELECTRON)
        assert wait_for(lambda: ("hfw", pytest.approx(185e-6)) in changed)
        holder = microscope.sample_handling.holder
        microscope.sample_handling.holder_changed.emit(holder)
        assert wait_for(lambda: len(signalled) == 1)
        name, value = signalled[0]
        assert name == "holder_changed"
        assert value["name"] == holder.name
    finally:
        client.close()


def test_no_token_is_refused(port):
    with pytest.raises(InvalidStatus) as refused:
        connect(_url(port), open_timeout=5)
    assert refused.value.response.status_code == 401


def test_a_browser_page_is_refused(port):
    with pytest.raises(InvalidStatus) as refused:
        connect(
            _url(port),
            additional_headers={"Authorization": f"Bearer {TOKEN}"},
            origin="http://example.com",
            open_timeout=5,
        )
    assert refused.value.response.status_code == 403


def test_off_is_refused(port, auth):
    auth.set_level("off")
    with pytest.raises(InvalidStatus) as refused:
        connect(
            _url(port),
            additional_headers={"Authorization": f"Bearer {TOKEN}"},
            open_timeout=5,
        )
    assert refused.value.response.status_code == 403


def test_the_stream_closes_when_the_agent_is_turned_off(port, auth):
    with connect(
        _url(port),
        additional_headers={"Authorization": f"Bearer {TOKEN}"},
        open_timeout=5,
    ) as socket:
        auth.set_level("off")
        with pytest.raises(ConnectionClosed) as closed:
            socket.recv(timeout=5)
    assert closed.value.rcvd.code == 1008


def test_a_client_too_far_behind_is_closed(microscope):
    """Events wait for a client that isn't reading only up to the backlog."""
    import asyncio

    from fibsem.server.devices import stream_events

    class Stuck:
        """A websocket whose client never reads: a send never finishes."""

        def __init__(self):
            self.code = None

        async def accept(self):
            pass

        async def receive_text(self):
            await asyncio.Event().wait()

        async def send_text(self, text):
            await asyncio.Event().wait()

        async def close(self, code):
            self.code = code

    stuck = Stuck()
    electron = microscope.devices["electron"]

    async def run():
        streaming = asyncio.ensure_future(
            stream_events(stuck, {"device": {"electron": electron}}, backlog=2)
        )
        await asyncio.sleep(0.05)
        for hfw in (150e-6, 160e-6, 170e-6, 180e-6, 190e-6):
            electron.hfw.set_value(hfw)
        await asyncio.wait_for(streaming, timeout=5)

    asyncio.run(run())
    assert stuck.code == 1013
