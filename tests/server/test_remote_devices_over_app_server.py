"""The remote driver's devices over a running App Server's ``/microscope`` routes,
with its token and no event stream: what a drop-in microscope off the PC builds on.

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
    connect_remote_beams,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
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
    """The App Server over the Demo microscope, hardware armed, on a free port."""
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


@pytest.fixture
def remote(port):
    remote = DeviceClient(
        "127.0.0.1", port, token=TOKEN, events=False, root="/microscope"
    )
    yield remote
    remote.close()


@pytest.fixture
def beams(remote, port):
    beams = connect_remote_beams("127.0.0.1", port, client=remote)
    return {beam.name: beam for beam in beams.values()}


def test_the_beams_are_remote_beams(beams):
    assert set(beams) == {"electron", "ion"}
    assert isinstance(beams["electron"], RemoteBeam)


def test_a_read_is_live(beams, microscope):
    microscope.set_field_of_view(150e-6, BeamType.ELECTRON)
    hfw = beams["electron"].parameters["hfw"]
    assert hfw.get_value() == pytest.approx(150e-6)
    microscope.set_field_of_view(160e-6, BeamType.ELECTRON)
    assert hfw.get_value() == pytest.approx(160e-6)


def test_a_write_reaches_the_microscope(beams, microscope):
    written = beams["electron"].parameters["hfw"].set_value(120e-6)
    assert written == pytest.approx(120e-6)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(120e-6)


def test_a_command_runs_on_the_microscope(beams, microscope):
    ion = beams["ion"]
    ion.call_command("blank")
    assert microscope.beams[BeamType.ION].blanked.get_value() is True
    ion.call_command("unblank")
    assert microscope.beams[BeamType.ION].blanked.get_value() is False


def test_the_devices_send_the_token(beams, remote, port):
    beams["electron"].parameters["current"].get_value()
    session = remote._session
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"
    url = f"http://127.0.0.1:{port}/microscope/devices/electron/current"
    assert requests.get(url, timeout=5).status_code == 401
    tokenless = DeviceClient("127.0.0.1", port, events=False, root="/microscope")
    try:
        with pytest.raises(RemoteDeviceError, match="401"):
            tokenless.describe()
    finally:
        tokenless.close()


def test_no_event_stream_is_opened(beams, remote):
    beams["electron"].parameters["hfw"].get_value()
    assert remote._events is None
