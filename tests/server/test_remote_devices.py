"""Remote devices over localhost: the Demo beams behind a device server, used from a
coordinator through the remote driver. A remote device must answer exactly as the
device it stands for, and fail closed when the server goes away."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.devices.core import (  # noqa: E402
    ParameterReadOnly,
    ParameterUnavailable,
)
from fibsem.devices.drivers.remote import (  # noqa: E402
    RemoteBeam,
    RemoteDeviceUnreachable,
    connect_remote_beams,
)
from fibsem.server.devices import DeviceServer, demo_devices  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402

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
    remote = connect_remote_beams("127.0.0.1", server.port)
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
