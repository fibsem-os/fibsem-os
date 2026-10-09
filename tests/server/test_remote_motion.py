"""The Demo stage, chamber and manipulator behind a device server, used from a
coordinator through the remote driver over localhost. Each must answer as the device
it stands for: moves, pumping and venting run on the far side, and what the device
says about itself beyond its parameters (its facts) comes across with it."""

import math

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

import yaml  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.core import ParameterMetadata  # noqa: E402
from fibsem.devices.entries import (  # noqa: E402
    DeviceBuildError,
    build_device_entries,
    resolve_device_entries,
)
from fibsem.devices.manipulator import Manipulator  # noqa: E402
from fibsem.devices.stage import UNLIMITED, StageLimitError  # noqa: E402
from fibsem.drivers.remote.devices import (  # noqa: E402
    DeviceClient,
    RemoteChamber,
    RemoteManipulator,
    RemoteStage,
)
from fibsem.server.devices import (  # noqa: E402
    DeviceServer,
    demo_devices,
    describe_device,
)
from fibsem.structures import (  # noqa: E402
    ChamberState,
    DeviceEntry,
    FibsemManipulatorPosition,
    FibsemStagePosition,
    InsertableDeviceState,
)


@pytest.fixture
def served():
    """(local devices by name, remote devices by name), all on 127.0.0.1."""
    local = {d.name: d for d in demo_devices(["stage", "chamber", "manipulator"])}
    server = DeviceServer(local.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    descriptions = client.describe()
    remote = {
        "stage": RemoteStage(client=client),
        "chamber": RemoteChamber(client=client),
        "manipulator": RemoteManipulator(client=client),
    }
    for name, device in remote.items():
        device.connect(descriptions[name])
    yield local, remote
    client.close()
    server.stop()


@pytest.mark.parametrize("name", ["stage", "chamber", "manipulator"])
def test_a_remote_device_describes_itself_as_the_local_one(served, name):
    local, remote = served
    assert remote[name].describe() == local[name].describe()
    assert remote[name].commands.keys() == local[name].commands.keys()


def test_a_remote_stage_has_the_local_axes_poses_and_frame(served):
    local, remote = served
    stage, here = remote["stage"], local["stage"]
    assert list(stage.axes) == list(here.axes)
    assert stage.axes.t.limits == here.axes.t.limits
    assert stage.frame == here.frame
    assert stage.compustage == here.compustage
    assert stage.poses(0, 35, 52) == here.poses(0, 35, 52)
    assert stage.has_builtin_shuttle() == here.has_builtin_shuttle()


def test_a_remote_stage_moves_on_the_server(served):
    local, remote = served
    target = FibsemStagePosition(x=1e-3, y=-2e-3, z=None, r=None, t=None)
    moved = remote["stage"].move_absolute(target)
    assert (moved.x, moved.y) == pytest.approx((1e-3, -2e-3))
    here = local["stage"].position.get_value()
    assert (here.x, here.y) == pytest.approx((1e-3, -2e-3))
    moved = remote["stage"].move_relative(FibsemStagePosition(x=1e-3))
    assert moved.x == pytest.approx(2e-3)
    assert remote["stage"].position.cached.x == pytest.approx(2e-3)


def test_a_remote_stage_refuses_a_move_outside_the_limits(served):
    local, remote = served
    before = local["stage"].position.get_value()
    with pytest.raises(StageLimitError):
        remote["stage"].move_absolute(FibsemStagePosition(x=1.0))
    # the server checks too: the old API's unchecked path is refused there
    with pytest.raises(StageLimitError):
        remote["stage"].move_through(FibsemStagePosition(x=1.0))
    assert local["stage"].position.get_value() == before


def test_home_and_link_run_on_the_server(served):
    _, remote = served
    assert remote["stage"].home() is True
    assert remote["stage"].link() is True


def test_a_remote_chamber_pumps_and_vents_on_the_server(served):
    local, remote = served
    assert remote["chamber"].vent() == ChamberState.VENTED
    assert local["chamber"].state.get_value() == ChamberState.VENTED
    assert remote["chamber"].pump() == ChamberState.PUMPED
    assert remote["chamber"].state.cached == ChamberState.PUMPED
    assert local["chamber"].state.get_value() == ChamberState.PUMPED


def test_a_remote_needle_has_the_local_axes_and_named_positions(served):
    local, remote = served
    needle, here = remote["manipulator"], local["manipulator"]
    assert needle.axes() == here.axes()
    assert needle.named_positions() == here.named_positions()
    for name in here.named_positions():
        assert needle.saved_position(name) == here.saved_position(name)
    with pytest.raises(ValueError, match="no saved position"):
        needle.saved_position("NOWHERE")


def test_a_remote_needle_moves_on_the_server(served):
    local, remote = served
    needle = remote["manipulator"]
    needle.insert("PARK")
    assert local["manipulator"].state.get_value() == InsertableDeviceState.INSERTED
    assert needle.state.cached == InsertableDeviceState.INSERTED
    offset = FibsemManipulatorPosition(x=1e-6, y=0, z=0, r=0, t=0)
    at = needle.move_to_offset(offset, "EUCENTRIC")
    assert at.x == pytest.approx(needle.saved_position("EUCENTRIC").x + 1e-6)
    moved = needle.move_relative(offset)
    assert moved.x == pytest.approx(at.x + 1e-6)
    assert local["manipulator"].position.get_value().x == pytest.approx(moved.x)
    needle.retract()
    assert local["manipulator"].state.get_value() == InsertableDeviceState.RETRACTED


def test_a_remote_needle_stops_on_the_server(served):
    local, remote = served
    assert remote["manipulator"].stop() == local["manipulator"].position.get_value()


def test_a_needle_that_cannot_stop_says_so_across_the_wire(served, monkeypatch):
    local, remote = served
    here = local["manipulator"]
    monkeypatch.setattr(here, "_stop", lambda: Manipulator._stop(here))
    with pytest.raises(NotImplementedError, match="cannot stop"):
        remote["manipulator"].stop()


def test_an_unbounded_axis_crosses_the_wire_as_unbounded(served):
    local, remote = served
    here = local["stage"]
    limits = dict(here.position.limits, r=UNLIMITED)
    here.position._metadata_source = lambda: ParameterMetadata(limits=limits)
    here.position.refresh_metadata()
    description = describe_device(here)
    assert description["parameters"]["position"]["limits"]["r"] == {
        "min": None,
        "max": None,
    }
    remote["stage"].position.refresh_metadata()
    assert remote["stage"].position.limits["r"].max == math.inf
    assert remote["stage"].position.limits["r"].min == -math.inf


def test_a_remote_stage_entry_is_named_stage(served):
    _, remote = served
    port = int(remote["stage"].client.base_url.rsplit(":", 1)[1])
    entry = DeviceEntry.from_dict(
        {
            "name": "stage2",
            "type": "stage",
            "driver": "remote",
            "address": "127.0.0.1",
            "port": port,
        }
    )
    resolved = resolve_device_entries([], {"stage2": entry}, "Demo")
    with pytest.raises(DeviceBuildError, match="'stage2' was not built"):
        build_device_entries(resolved, None)


def test_the_demo_takes_its_stage_chamber_and_needle_from_a_server(tmp_path):
    server = DeviceServer(demo_devices(["stage", "chamber", "manipulator"])).start()
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    at = {"driver": "remote", "address": "127.0.0.1", "port": server.port}
    devices = config["hardware"]["devices"]
    for name in ("stage", "chamber", "manipulator"):
        entries = [e for e in devices if e["name"] == name]
        if entries:
            entries[0].update(at)
        else:
            devices.append({"name": name, **at})
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    try:
        assert isinstance(microscope.stage, RemoteStage)
        assert isinstance(microscope.chamber_device, RemoteChamber)
        assert isinstance(microscope.manipulator_device, RemoteManipulator)
        start = microscope.get_stage_position()
        microscope.move_stage_relative(FibsemStagePosition(x=1e-4))
        assert microscope.get_stage_position().x == pytest.approx(start.x + 1e-4)
        assert microscope.manipulator_named_positions() == ["PARK", "EUCENTRIC"]
    finally:
        microscope.stage.client.close()
        server.stop()
