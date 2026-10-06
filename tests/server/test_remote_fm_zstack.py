"""A z-stack on a remote FM is one command, run on the FM's computer.

The FM API sends the positions and channels once (``FM.acquire_z_stack``); the far
side moves the objective and takes each frame, reports where it is as it goes, and
answers with every frame together. Cancelling stops it between frames. A server from
before the command still runs the stack, one request per step, as before.
"""

import threading

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.devices.drivers.remote import DeviceClient  # noqa: E402
from fibsem.fm.acquisition import acquire_z_stack  # noqa: E402
from fibsem.fm.api import DeviceFluorescenceMicroscope  # noqa: E402
from fibsem.fm.progress import FluorescenceAcquisitionStatus  # noqa: E402
from fibsem.fm.remote import RemoteFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import (  # noqa: E402
    ChannelSettings,
    ZParameters,
    ZStackOrder,
)
from fibsem.server.devices import DeviceServer, demo_fm_devices  # noqa: E402

CHANNELS = [
    ChannelSettings(name="green", excitation_wavelength=488, exposure_time=0.01),
    ChannelSettings(
        name="far-red", excitation_wavelength=640, exposure_time=0.02, color="red"
    ),
]
ZPARAMS = ZParameters(zmin=-1e-6, zmax=1e-6, zstep=1e-6)  # 3 planes


@pytest.fixture
def served():
    far = {d.name: d for d in demo_fm_devices()}
    far["fm"]._fm.camera.binning = 8  # small frames, so each stack is quick
    server = DeviceServer(far.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    fm = RemoteFluorescenceMicroscope.connect("127.0.0.1", server.port, client=client)
    yield far, fm, client
    client.close()
    server.stop()


def _commands(client, monkeypatch):
    """Every command the client sends, by name."""
    sent = []
    request = client.request

    def recording(method, path, timeout, **kwargs):
        if "/commands/" in path:
            sent.append(path.rsplit("/", 1)[1])
        return request(method, path, timeout, **kwargs)

    monkeypatch.setattr(client, "request", recording)
    return sent


def _far_moves(far, monkeypatch):
    moves = []
    objective = far["fm"]._fm.objective
    move = objective.move_absolute

    def recording(position):
        moves.append(round(position, 12))
        return move(position)

    monkeypatch.setattr(objective, "move_absolute", recording)
    return moves


@pytest.mark.parametrize("order", [ZStackOrder.CHANNEL, ZStackOrder.Z_LEVEL])
def test_a_remote_z_stack_is_one_command(served, monkeypatch, order):
    far, fm, client = served
    assert fm.runs_z_stack_on_device
    sent = _commands(client, monkeypatch)
    moves = _far_moves(far, monkeypatch)
    z_init = fm.objective.position
    zparams = ZParameters(zmin=-1e-6, zmax=1e-6, zstep=1e-6, order=order)
    positions = [round(z, 12) for z in zparams.generate_positions(z_init=z_init)]

    image = acquire_z_stack(fm, CHANNELS, zparams)

    assert sent.count("acquire_z_stack") == 1
    assert "move_absolute" not in sent and "acquire_frame" not in sent
    assert image.data.shape[:2] == (2, 3)
    # The far side moved as the step-by-step stack moves, then went back.
    if order == ZStackOrder.Z_LEVEL:
        assert moves == positions + [round(z_init, 12)]
    else:
        assert moves == positions * 2 + [round(z_init, 12)]
    assert fm.objective.position == pytest.approx(z_init)
    assert [c.name for c in image.metadata.channels] == ["green", "far-red"]


def test_each_plane_says_where_it_was_taken(served):
    far, fm, _ = served
    z_init = fm.objective.position
    positions = ZPARAMS.generate_positions(z_init=z_init)

    stack = fm.acquire_z_stack_on_device(CHANNELS[:1], ZPARAMS)

    assert stack.data.shape[:2] == (1, 3)
    assert stack.metadata.channels[0].exposure_time == pytest.approx(0.01)
    assert stack.metadata.channels[0].excitation_wavelength == pytest.approx(488)
    assert np.isfinite(stack.data).all()
    assert len(positions) == 3


def test_progress_arrives_for_every_frame(served):
    _, fm, _ = served
    seen = []
    fm.acquisition_progress_signal.connect(seen.append)

    acquire_z_stack(fm, CHANNELS, ZPARAMS)

    deadline = threading.Event()
    for _ in range(50):  # events cross the websocket after the answer, at worst
        if len(seen) >= 6:
            break
        deadline.wait(0.05)
    assert [(p.channel, p.channel_index, p.zlevel) for p in seen] == [
        ("green", 1, 1),
        ("green", 1, 2),
        ("green", 1, 3),
        ("far-red", 2, 1),
        ("far-red", 2, 2),
        ("far-red", 2, 3),
    ]
    assert all(
        p.status is FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK
        and (p.total_channels, p.total_zlevels) == (2, 3)
        for p in seen
    )


def test_cancelling_stops_it_on_the_far_side(served, monkeypatch):
    far, fm, _ = served
    moves = _far_moves(far, monkeypatch)
    z_init = fm.objective.position
    stop = threading.Event()
    group = far["fm"]
    acquire = group.acquire_frame

    def slow_frame(channel=None):
        stop.set()  # asked to stop during the first frame
        threading.Event().wait(0.5)  # long enough for the cancel to arrive
        return acquire(channel)

    monkeypatch.setattr(group, "acquire_frame", slow_frame)

    result = acquire_z_stack(fm, CHANNELS, ZPARAMS, stop_event=stop)

    assert result is None
    assert len(moves) == 2  # the first plane, then back
    assert moves[-1] == pytest.approx(z_init)


def test_a_server_without_the_command_still_runs_the_stack(served, monkeypatch):
    _, fm, client = served
    group = fm.devices["fm"]
    monkeypatch.setattr(
        group, "server_commands", group.server_commands - {"acquire_z_stack"}
    )
    assert not fm.runs_z_stack_on_device
    sent = _commands(client, monkeypatch)

    image = acquire_z_stack(fm, CHANNELS, ZPARAMS)

    assert image.data.shape[:2] == (2, 3)
    assert "acquire_z_stack" not in sent
    assert sent.count("acquire_frame") == 6


def test_the_group_command_falls_back_to_steps_on_a_server_without_it(
    served, monkeypatch
):
    """Called on the remote group itself, the stack runs here through the remote
    parts: the group has them in its roles, so it finds the objective to move."""
    far, fm, client = served
    group = fm.devices["fm"]
    assert group.objective is fm.devices["objective"]
    monkeypatch.setattr(
        group, "server_commands", group.server_commands - {"acquire_z_stack"}
    )
    sent = _commands(client, monkeypatch)
    z = far["objective"].position.get_value()

    frames = group.acquire_z_stack([CHANNELS[0].to_dict()], [z, z + 1e-6])

    assert len(frames) == 2
    assert "acquire_z_stack" not in sent
    assert sent.count("move_absolute") >= 2


def test_a_local_fm_still_runs_the_stack_step_by_step():
    """Each slice is shown as it arrives, as before: no command for a local FM."""
    from fibsem.devices.drivers.fm import bind_fm_devices
    from fibsem.microscopes.simulator import SimulatedFluorescenceMicroscope

    fm = DeviceFluorescenceMicroscope(
        bind_fm_devices(SimulatedFluorescenceMicroscope())
    )
    assert not fm.runs_z_stack_on_device


def test_the_group_command_takes_the_steps_the_api_takes():
    """The same moves and frames, in the same order, as the step-by-step stack."""
    from fibsem.devices.drivers.fm import bind_fm_devices
    from fibsem.microscopes.simulator import SimulatedFluorescenceMicroscope

    def record(fm_class_run):
        sim = SimulatedFluorescenceMicroscope()
        sim.camera.binning = 8  # small frames, so each stack is quick
        devices = bind_fm_devices(sim)
        steps = []
        move, acquire = sim.objective.move_absolute, sim.acquire_image
        sim.objective.move_absolute = lambda z: (
            steps.append(("move", round(z, 12))),
            move(z),
        )[1]
        sim.acquire_image = lambda ch=None: (
            steps.append(("frame", ch.name if ch else None)),
            acquire(ch),
        )[1]
        fm_class_run(devices)
        return steps

    for order in (ZStackOrder.CHANNEL, ZStackOrder.Z_LEVEL):
        zparams = ZParameters(zmin=-1e-6, zmax=1e-6, zstep=1e-6, order=order)

        def step_by_step(devices):
            acquire_z_stack(DeviceFluorescenceMicroscope(devices), CHANNELS, zparams)

        def one_command(devices):
            fm = DeviceFluorescenceMicroscope(devices)
            z_init = fm.objective.position
            devices["fm"].acquire_z_stack(
                channels=[c.to_dict() for c in CHANNELS],
                positions=list(zparams.generate_positions(z_init=z_init)),
                order="z" if order == ZStackOrder.Z_LEVEL else "channel",
                restore_position=z_init,
            )

        assert record(one_command) == record(step_by_step), order
