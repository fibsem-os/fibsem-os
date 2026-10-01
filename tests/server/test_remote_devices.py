"""Remote devices over localhost: the Demo beams behind a device server, used from a
coordinator through the remote driver. A remote device must answer exactly as the
device it stands for, and fail closed when the server goes away."""

import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import requests  # noqa: E402

from fibsem.devices.core import (  # noqa: E402
    ParameterReadOnly,
    ParameterUnavailable,
)
from fibsem.devices.drivers.remote import (  # noqa: E402
    DeviceClient,
    RemoteBeam,
    RemoteDeviceUnreachable,
    connect_remote_beams,
)
from fibsem.server.devices import DeviceServer, demo_devices  # noqa: E402
from fibsem.structures import BeamType, FibsemRectangle, Point, ScanMode  # noqa: E402

BEAMS = (BeamType.ELECTRON, BeamType.ION)

# A value for each free parameter; parameters with choices take their last choice.
SETS = {
    "hfw": 123e-6,
    "working_distance": 4e-3,
    "scan_rotation": 1.5,
    "blanked": True,
    "voltage": None,
    "current": None,
    "detector_type": None,
    "detector_mode": None,
}


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def served():
    """(local devices by beam type, remote beams, server), all on 127.0.0.1."""
    local = {d.beam_type: d for d in demo_devices()}
    server = DeviceServer(local.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    remote = connect_remote_beams("127.0.0.1", server.port, client=client)
    yield local, remote, server
    for beam in remote.values():
        beam.client.close()
    server.stop()


@pytest.mark.parametrize("beam_type", BEAMS)
def test_a_remote_beam_describes_itself_as_the_local_one(served, beam_type):
    local, remote, _ = served
    assert remote[beam_type].describe() == local[beam_type].describe()


@pytest.mark.parametrize("beam_type", BEAMS)
def test_remote_reads_match_local_reads(served, beam_type):
    local, remote, _ = served
    for name, param in local[beam_type].parameters.items():
        assert remote[beam_type].parameters[name].get_value() == param.get_value()


@pytest.mark.parametrize("beam_type", BEAMS)
@pytest.mark.parametrize("name", SETS)
def test_a_remote_set_leaves_the_same_state_as_a_local_one(served, beam_type, name):
    local, remote, _ = served
    choices = local[beam_type].parameters[name].choices
    value = SETS[name] if choices is None else choices[-1]
    written = remote[beam_type].parameters[name].set_value(value)
    assert local[beam_type].parameters[name].get_value() == written == value


def test_the_new_api_checks_the_same_way_remotely(served):
    local, remote, _ = served
    sem = remote[BeamType.ELECTRON]
    assert (
        sem.scan_rotation.set_value(7.0)
        == local[BeamType.ELECTRON].scan_rotation.limits.max
    )
    with pytest.raises(ValueError):
        sem.current.set_value(12345.0)
    with pytest.raises(TypeError):
        sem.hfw.set_value("wide")


def test_the_server_checks_values_itself(served):
    """A client that skips the checks still can't write a bad value."""
    _, remote, _ = served
    sem = remote[BeamType.ELECTRON]
    with pytest.raises(ValueError):
        sem.current.write_through(12345.0)


def test_errors_keep_their_meaning_across_the_wire(served):
    local, remote, _ = served
    client = remote[BeamType.ELECTRON].client
    with pytest.raises(ParameterUnavailable):
        client.request("GET", "devices/electron/preset", 5)  # Demo has no presets
    scan_rotation = local[BeamType.ELECTRON].parameters["scan_rotation"]
    scan_rotation._write = None  # as a backend with no write_scan_rotation
    scan_rotation.refresh_metadata(emit=False)
    with pytest.raises(ParameterReadOnly):
        client.request("PUT", "devices/electron/scan_rotation", 5, json={"value": 1})


def test_a_change_on_the_far_side_reaches_the_coordinator(served):
    local, remote, _ = served
    sem = remote[BeamType.ELECTRON]
    sem.hfw.get_value()
    seen = []
    sem.hfw.changed.connect(seen.append)
    local[BeamType.ELECTRON].hfw.set_value(321e-6)  # e.g. the far computer's own UI
    assert wait_for(lambda: seen == [321e-6])
    assert sem.hfw.cached == 321e-6


def test_the_coordinators_own_set_is_signalled_once(served):
    _, remote, _ = served
    sem = remote[BeamType.ELECTRON]
    sem.hfw.get_value()
    seen = []
    sem.hfw.changed.connect(seen.append)
    sem.hfw.set_value(222e-6)
    time.sleep(0.2)  # let the server's echo arrive
    assert seen == [222e-6]


def test_commands_run_on_the_server(served):
    local, remote, _ = served
    sem = remote[BeamType.ELECTRON]
    sem.blank()
    assert local[BeamType.ELECTRON].blanked.get_value() is True
    sem.call_command("unblank")
    assert local[BeamType.ELECTRON].blanked.get_value() is False
    assert sem.commands["acquire"].available is False


def test_a_stopped_server_fails_closed(served):
    _, remote, server = served
    sem = remote[BeamType.ELECTRON]
    last = sem.hfw.get_value()
    dropped = []
    sem.client.disconnected.connect(lambda: dropped.append(True))
    server.stop()
    assert wait_for(lambda: dropped == [True])
    assert not sem.client.connected
    with pytest.raises(RemoteDeviceUnreachable):
        sem.hfw.get_value()  # a guard read never falls back to a stale value
    with pytest.raises(RemoteDeviceUnreachable):
        sem.hfw.set_value(100e-6)
    assert sem.hfw.cached == last  # displays keep the last known value


def test_a_parameter_that_means_something_else_on_the_server_refuses_to_connect(
    served,
):
    _, remote, _ = served
    client = remote[BeamType.ELECTRON].client
    description = client.describe()["electron"]
    description["parameters"]["voltage"]["unit"] = "kV"
    with pytest.raises(TypeError, match="voltage"):
        RemoteBeam(BeamType.ELECTRON, client=client).connect(description)


def test_health_reports_each_device(served):
    local, remote, _ = served
    client = remote[BeamType.ELECTRON].client
    assert client.health() == {
        "ok": True,
        "devices": {
            "electron": {"ok": True, "detail": None},
            "ion": {"ok": True, "detail": None},
        },
    }
    local[BeamType.ION].check_health = lambda: "plasma source not responding"
    health = client.health()
    assert health["ok"] is False
    assert health["devices"]["ion"] == {
        "ok": False,
        "detail": "plasma source not responding",
    }


def test_a_frozen_server_is_noticed_without_a_read(served):
    """No clean close, no failed read: only the heartbeat can tell."""
    _, _, server = served

    @server.app.get("/freeze")
    async def freeze() -> None:
        time.sleep(3.0)  # blocks the server's event loop, as a hung PC would

    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.2)
    beams = connect_remote_beams("127.0.0.1", server.port, client=client)
    dropped = []
    client.disconnected.connect(lambda: dropped.append(True))
    threading.Thread(
        target=requests.get, args=(f"{client.base_url}/freeze",), daemon=True
    ).start()
    try:
        assert wait_for(lambda: dropped == [True], timeout=2.5)
        assert not client.connected
    finally:
        for beam in beams.values():
            beam.client.close()


@pytest.mark.parametrize("beam_type", BEAMS)
def test_a_connected_remote_beam_starts_with_every_value_cached(served, beam_type):
    local, remote, _ = served
    for name, param in remote[beam_type].parameters.items():
        assert param._cached == local[beam_type].parameters[name].get_value()


def test_the_client_reconnects_and_catches_up_on_what_it_missed(served):
    local, remote, server = served
    sem = remote[BeamType.ELECTRON]
    events = []
    sem.client.disconnected.connect(lambda: events.append("down"))
    sem.client.reconnected.connect(lambda: events.append("up"))
    seen = []
    sem.hfw.changed.connect(seen.append)

    port = server.port
    server.stop()
    assert wait_for(lambda: events == ["down"])
    local[BeamType.ELECTRON].hfw.set_value(456e-6)  # changed while nobody listened

    restarted = DeviceServer(local.values(), port=port).start()
    try:
        assert wait_for(lambda: events == ["down", "up"], timeout=10)
        assert sem.client.connected
        assert seen == [456e-6]  # the missed change, signalled on resync
        assert sem.hfw.cached == 456e-6
        local[BeamType.ELECTRON].hfw.set_value(789e-6)  # and events flow again
        assert wait_for(lambda: seen == [456e-6, 789e-6])
    finally:
        restarted.stop()


def test_a_structured_value_crosses_the_wire_as_itself(served):
    """A ``Point`` parameter reads back as a ``Point``, and a remote set of one is
    written on the far side and signalled once."""
    local, remote, _ = served
    shift = remote[BeamType.ELECTRON].shift
    assert isinstance(shift.get_value(), Point)
    seen = []
    shift.changed.connect(seen.append)
    shift.set_value(Point(1e-6, -2e-6))
    far = local[BeamType.ELECTRON].shift.get_value()
    assert (far.x, far.y) == (1e-6, -2e-6)
    time.sleep(0.2)  # time for the server's event to arrive and be matched
    assert len(seen) == 1 and (seen[0].x, seen[0].y) == (1e-6, -2e-6)


def test_an_enum_value_crosses_the_wire_as_itself(served):
    """The scan mode is a ``ScanMode`` on both sides of the wire."""
    local, remote, _ = served
    mode = remote[BeamType.ELECTRON].scanning_mode
    assert mode.get_value() is ScanMode.FULL_FRAME
    local[BeamType.ELECTRON].spot(Point(0.5, 0.5))
    assert local[BeamType.ELECTRON].scanning_mode.get_value() is ScanMode.SPOT
    assert mode.get_value() is ScanMode.SPOT


def test_the_scan_area_commands_run_on_the_far_side(served):
    """spot, reduced_area and full_frame run on the server, with their structured
    arguments arriving as themselves, and the mode reads back on both sides."""
    local, remote, _ = served
    beam, far = remote[BeamType.ELECTRON], local[BeamType.ELECTRON]
    beam.spot(Point(0.25, 0.75))
    assert far._system.scanning_mode_value == Point(0.25, 0.75)
    assert beam.scanning_mode.cached is ScanMode.SPOT
    area = FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    beam.reduced_area(area)
    assert far._system.scanning_mode_value == area
    assert beam.scanning_mode.cached is ScanMode.REDUCED_AREA
    beam.full_frame()
    assert far.scanning_mode.get_value() is ScanMode.FULL_FRAME
    assert beam.scanning_mode.cached is ScanMode.FULL_FRAME
