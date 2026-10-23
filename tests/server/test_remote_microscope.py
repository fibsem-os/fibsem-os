"""`RemoteMicroscope`: a `FibsemMicroscope` whose devices and services are an App
Server's, over the Demo. The facade's calls run on the server.

Skipped when the [server] extra is not installed.
"""

import threading
import time

import pytest

from tests.fixtures.app_server import serving

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fibsem.cancellation import OperationCancelledError  # noqa: E402
from fibsem.client.remote_microscope import RemoteMicroscope  # noqa: E402
from fibsem.drivers.remote.devices import RemoteBeam, RemoteStage  # noqa: E402
from fibsem.drivers.remote.services import RemoteService  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.services.milling import Milling  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    MillingState,
)

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
def url(microscope, auth):
    with serving(build_server(microscope, auth=auth)) as url:
        yield url


@pytest.fixture(scope="module")
def remote(url):
    remote = RemoteMicroscope.connect(url, TOKEN)
    yield remote
    remote.disconnect()


@pytest.fixture(autouse=True)
def at_drive(auth):
    yield
    auth.set_level("drive")


def test_it_has_the_servers_devices_and_services(remote, microscope):
    assert set(remote.devices) == set(microscope.devices)
    assert set(remote.services) == set(microscope.services)
    assert isinstance(remote.beams[BeamType.ELECTRON], RemoteBeam)
    assert isinstance(remote.stage, RemoteStage)
    assert isinstance(remote.milling, RemoteService)
    assert remote.system.info.model == "DemoMicroscope"
    # a service keeps its own signals and the signatures of what it serves
    assert remote.milling.patterns_changed is not Milling.patterns_changed
    assert "draw" in remote.milling.commands and "run" in remote.milling.commands
    assert remote.milling.commands["run"].long


def test_the_facade_reads_and_acquires_on_the_server(remote, microscope):
    assert remote.get_stage_position() == microscope.get_stage_position()
    image = remote.acquire_image(
        ImageSettings(beam_type=BeamType.ELECTRON, resolution=(64, 48))
    )
    assert isinstance(image, FibsemImage) and image.data.shape == (48, 64)
    state = remote.get_microscope_state()
    assert state.stage_position == microscope.get_stage_position()


def test_a_move_moves_the_servers_stage(remote, microscope):
    start = microscope.get_stage_position()
    try:
        moved = remote.move_stage_relative(FibsemStagePosition(x=10e-6))
        assert moved.x == pytest.approx(start.x + 10e-6)
        assert microscope.get_stage_position().x == pytest.approx(start.x + 10e-6)
    finally:
        microscope.move_stage_absolute(start)


def test_a_query_takes_its_arguments_positionally(remote, microscope):
    position = microscope.get_stage_position()
    assert remote.stage_movement.current_milling_angle() == pytest.approx(
        microscope.stage_movement.current_milling_angle()
    )
    assert remote.stage_movement.stage_orientation(position) == (
        microscope.stage_movement.stage_orientation(position)
    )


def _square():
    return FibsemRectangleSettings(
        width=5e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
    )


def test_a_mill_runs_on_the_server_and_its_signals_cross(remote):
    drawn = []
    remote.milling.patterns_changed.connect(drawn.append)
    try:
        remote.setup_milling(FibsemMillingSettings())
        remote.draw_patterns([_square()])
        remote.run_milling()  # waits on the server's run
        remote.finish_milling()
        assert remote.get_milling_state() is MillingState.IDLE
        # drawn, then cleared at the end of the run
        assert wait_for(lambda: len(drawn) >= 2)
        assert len(drawn[0]) == 1 and drawn[-1] == []
    finally:
        remote.milling.patterns_changed.disconnect(drawn.append)


def test_a_set_stop_event_stops_the_mill(remote):
    stop = threading.Event()
    stop.set()
    remote.setup_milling(FibsemMillingSettings())
    remote.draw_patterns([_square()])
    try:
        with pytest.raises(OperationCancelledError):
            remote.run_milling(stop_event=stop)
    finally:
        remote.finish_milling()


def test_a_parameter_change_on_the_server_reaches_the_remote(remote, microscope):
    hfw = remote.beams[BeamType.ELECTRON].hfw
    microscope.set_field_of_view(165e-6, BeamType.ELECTRON)
    assert wait_for(lambda: hfw.cached == pytest.approx(165e-6))


def test_a_refused_command_raises(remote, auth):
    auth.set_level("watch")
    with pytest.raises(Exception, match="403"):
        remote.move_stage_relative(FibsemStagePosition(x=1e-6))


def test_disconnect_closes_its_connection(url):
    remote = RemoteMicroscope.connect(url, TOKEN)
    remote.disconnect()
    assert remote._client is None


def test_a_stop_event_set_while_milling_stops_the_servers_run(
    remote, microscope, monkeypatch
):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "0")  # a mill that takes a while
    stop = threading.Event()
    remote.setup_milling(FibsemMillingSettings())
    remote.draw_patterns([_square()])
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(OperationCancelledError):
            remote.run_milling(stop_event=stop)
        assert time.monotonic() - started < 5
        assert microscope.milling.runs.latest.state == "cancelled"
    finally:
        remote.finish_milling()


def test_back_to_back_commands_never_find_the_last_one_in_control(remote):
    """Control is given up before a command's answer is sent."""
    for _ in range(200):
        remote.milling.clear()
