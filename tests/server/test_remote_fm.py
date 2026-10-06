"""A simulated FM served over localhost, as the METEOR PC would serve its FM: the
remote parts answer as the local ones do, and frames cross the wire intact."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.devices.drivers.remote import (  # noqa: E402
    DeviceClient,
    RemoteCamera,
    RemoteFM,
    RemoteObjective,
    connect_remote_fm,
)
from fibsem.fm.structures import (
    ChannelSettings,  # noqa: E402
    EmissionFilter,  # noqa: E402
)
from fibsem.server.devices import DeviceServer, demo_fm_devices  # noqa: E402
from fibsem.structures import InsertableDeviceState  # noqa: E402


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def served():
    local = {d.name: d for d in demo_fm_devices()}
    server = DeviceServer(local.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    remote = connect_remote_fm("127.0.0.1", server.port, client=client)
    yield local, remote
    client.close()
    server.stop()


def test_the_remote_fm_has_the_same_parts(served):
    local, remote = served
    assert sorted(remote) == sorted(local)
    assert isinstance(remote["fm"], RemoteFM)
    assert isinstance(remote["camera"], RemoteCamera)
    assert isinstance(remote["objective"], RemoteObjective)
    for name in local:
        assert remote[name].describe() == local[name].describe()


def test_remote_parameters_write_the_far_side(served):
    local, remote = served
    remote["camera"].exposure_time.set_value(0.25)
    assert local["camera"].exposure_time.get_value() == 0.25
    fluorescence = remote["filter_set"].emission_filter.choices[1]
    assert isinstance(fluorescence, EmissionFilter)  # a structure, not a dict
    remote["filter_set"].emission_filter.set_value(fluorescence)
    assert local["filter_set"].emission_filter.get_value() == fluorescence
    with pytest.raises(ValueError):
        remote["camera"].binning.set_value(3)


def test_a_frame_crosses_the_wire_intact(served):
    local, remote = served
    np.random.seed(0)
    expected = local["camera"].acquire()
    # The same numbered frame over the same noise: any corruption shows.
    local["camera"].sim_index -= 1
    np.random.seed(0)
    frame = remote["camera"].acquire()
    assert frame.dtype == expected.dtype == np.uint16
    assert np.array_equal(frame, expected)


def test_a_channel_is_one_call_and_its_changes_reach_the_coordinator(served):
    _, remote = served
    seen = []
    remote["light_source"].power.changed.connect(seen.append)
    channel = ChannelSettings(excitation_wavelength=450, power=0.2, exposure_time=0.01)
    frame = remote["fm"].acquire_channel(channel.to_dict())
    assert isinstance(frame, np.ndarray) and frame.ndim == 2
    assert wait_for(lambda: seen == [0.2])
    assert wait_for(lambda: remote["filter_set"].excitation_wavelength.cached == 450)


def test_an_objective_move_runs_on_the_server_and_its_state_follows(served):
    local, remote = served
    objective = remote["objective"]
    assert objective.state.cached is InsertableDeviceState.RETRACTED
    objective.insert()
    assert local["objective"].state.get_value() is InsertableDeviceState.INSERTED
    assert wait_for(lambda: objective.state.cached is InsertableDeviceState.INSERTED)
    # the guard's live read, an enum again after the wire
    assert objective.state.get_value() is InsertableDeviceState.INSERTED


def test_a_guard_read_fails_closed_when_the_fm_computer_is_gone(served):
    from fibsem.devices.drivers.remote import RemoteDeviceUnreachable

    _, remote = served
    remote["fm"].client._session.close()
    remote["fm"].client.base_url = "http://127.0.0.1:1"  # nothing listens there
    with pytest.raises(RemoteDeviceUnreachable):
        remote["objective"].state.get_value()
