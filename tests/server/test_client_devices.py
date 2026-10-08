"""FibsemClient.devices: the remote driver's devices over a running agent server,
with the client's token.

Skipped when the [server] extra is not installed.
"""

import os
import threading

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")

from fibsem import utils  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    DeviceClient,
    RemoteBeam,
    RemoteDeviceError,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.client import FibsemClient  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402

TOKEN = "test-token"


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
def port(microscope):
    """The agent server over the Demo microscope, hardware armed, on a free port."""
    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=True, token=TOKEN)
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "agent server failed to start"
        threading.Event().wait(0.01)
    yield server.servers[0].sockets[0].getsockname()[1]
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(port):
    with FibsemClient("127.0.0.1", port, token=TOKEN) as client:
        yield client


def test_the_client_connects_and_reads_what_is_fitted(client, microscope):
    fitted = microscope.is_available("manipulator")
    assert client.system.manipulator.enabled is fitted


def test_devices_are_the_remote_beams(client):
    assert set(client.devices) == {"electron", "ion"}
    assert isinstance(client.devices["electron"], RemoteBeam)
    assert client.devices is not client.devices  # read-only views...
    assert client.devices["ion"] is client.devices["ion"]  # ...of one build


def test_a_read_is_live(client, microscope):
    microscope.set_field_of_view(150e-6, BeamType.ELECTRON)
    hfw = client.devices["electron"].parameters["hfw"]
    assert hfw.get_value() == pytest.approx(150e-6)
    microscope.set_field_of_view(160e-6, BeamType.ELECTRON)
    assert hfw.get_value() == pytest.approx(160e-6)


def test_a_write_reaches_the_microscope(client, microscope):
    written = client.devices["electron"].parameters["hfw"].set_value(120e-6)
    assert written == pytest.approx(120e-6)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(120e-6)


def test_a_command_runs_on_the_microscope(client, microscope):
    ion = client.devices["ion"]
    ion.call_command("blank")
    assert microscope.beams[BeamType.ION].blanked.get_value() is True
    ion.call_command("unblank")
    assert microscope.beams[BeamType.ION].blanked.get_value() is False


def test_the_devices_send_the_token(client, port):
    client.devices["electron"].parameters["current"].get_value()
    session = client._device_client._session
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"
    url = f"http://127.0.0.1:{port}/devices/electron/current"
    assert requests.get(url, timeout=5).status_code == 401
    tokenless = DeviceClient("127.0.0.1", port, events=False)
    try:
        with pytest.raises(RemoteDeviceError, match="401"):
            tokenless.describe()
    finally:
        tokenless.close()


def test_no_event_stream_is_opened(client):
    client.devices["electron"].parameters["hfw"].get_value()
    assert client._device_client._events is None
