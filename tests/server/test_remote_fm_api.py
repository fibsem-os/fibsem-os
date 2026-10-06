"""Today's FM API over the Demo FM devices served on localhost: the calls the FM UI and
workflows make answer as they do on a local FM, and read the FM's computer each time."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from fibsem.devices.drivers.remote import (  # noqa: E402
    DeviceClient,
    RemoteDeviceUnreachable,
)
from fibsem.fm.microscope import FluorescenceMicroscope  # noqa: E402  # noqa: E402
from fibsem.fm.remote import RemoteFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings, FluorescenceImage  # noqa: E402
from fibsem.server.devices import DeviceServer, demo_fm_devices  # noqa: E402


def _far_group(fm):
    """The FM group on the far side, behind the remote one."""
    return fm._served["fm"]


@pytest.fixture
def served():
    local = {d.name: d for d in demo_fm_devices()}
    server = DeviceServer(local.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    fm = RemoteFluorescenceMicroscope.connect("127.0.0.1", server.port, client=client)
    fm._served = local  # for the tests that check the far side's devices
    # The FM API over the served devices, as the FM's computer would hold it.
    far = FluorescenceMicroscope(local)
    yield far, fm
    client.close()
    server.stop()


def test_it_is_a_fluorescence_microscope(served):
    _, fm = served
    assert isinstance(fm, FluorescenceMicroscope)
    assert fm.client.health()["ok"] is True


def test_camera_properties_read_and_write_the_far_side(served):
    far, fm = served
    fm.camera.exposure_time = 0.25
    assert far.camera.exposure_time == 0.25
    fm.set_binning(2)  # the old setter path
    assert far.camera.binning == 2
    far.camera.gain = 3.0  # changed on the FM's computer
    assert fm.camera.gain == 3.0
    assert fm.camera.available_binnings == tuple(far.camera.available_binnings)
    assert fm.camera.exposure_time_limits == tuple(far.camera.exposure_time_limits)
    assert fm.camera.pixel_size == tuple(far.camera.pixel_size)
    assert fm.camera.resolution == tuple(far.camera.resolution)
    assert fm.camera.field_of_view == far.camera.field_of_view


def test_light_and_filters_read_and_write_the_far_side(served):
    far, fm = served
    fm.set_power(0.3)
    assert far.light_source.power == 0.3
    assert fm.light_source.power_limits == tuple(far.light_source.power_limits)
    fm.filter_set.excitation_wavelength = 450
    assert far.filter_set.excitation_wavelength == 450
    assert fm.filter_set.available_excitation_wavelengths == tuple(
        far.filter_set.available_excitation_wavelengths
    )
    fm.filter_set.emission_wavelength = "Fluorescence"
    assert far.filter_set.emission_wavelength == "Fluorescence"
    assert fm.filter_set.emission_wavelength == "Fluorescence"
    fm.filter_set.emission_wavelength = None
    assert far.filter_set.emission_wavelength is None


def test_an_excitation_between_bands_selects_the_nearest_remotely(served):
    """FIB-1094: the server used to refuse what the local FM accepts."""
    far, fm = served
    fm.filter_set.excitation_wavelength = 488
    assert far.filter_set.excitation_wavelength == 450
    assert fm.filter_set.excitation_wavelength == 450

    fm.set_channel(
        ChannelSettings(excitation_wavelength=600, power=0.2, exposure_time=0.01)
    )
    assert far.filter_set.excitation_wavelength == 635
    assert fm.filter_set.emission_wavelength is None
    assert fm.filter_set.available_emission_wavelengths == tuple(
        far.filter_set.available_emission_wavelengths
    )


def test_a_bad_value_is_refused_by_the_far_side(served):
    far, fm = served
    before = far.camera.binning
    with pytest.raises(ValueError):
        fm.camera.binning = 3
    assert far.camera.binning == before


def test_the_objective_moves_and_announces_where_it_ended_up(served):
    far, fm = served
    moves = []
    fm.objective.position_changed.connect(lambda p, s: moves.append((p, s)))
    assert fm.objective.state == "Retracted"
    fm.objective.insert()
    assert far.objective.state == "Inserted" == fm.objective.state
    assert moves == [(far.objective.position, "Inserted")]
    fm.objective.move_relative(1e-6)
    assert fm.objective.position == pytest.approx(far.objective.position)
    assert len(moves) == 2
    low, high = fm.objective.limits
    assert (low, high) == tuple(far.objective.limits)
    fm.objective.focus_position = fm.objective.position  # a session setting
    assert fm.objective.focus_position == fm.objective.position
    fm.objective.retract()
    assert far.objective.state == "Retracted"


def test_acquire_image_with_the_current_settings(served):
    far, fm = served
    images = []
    fm.acquisition_signal.connect(images.append)
    fm._rate_limit = 0
    image = fm.acquire_image()
    assert isinstance(image, FluorescenceImage)
    assert image.data.shape == tuple(far.camera.resolution)[::-1]
    md = image.metadata
    assert md.channels[0].exposure_time == far.camera.exposure_time
    assert (md.pixel_size_x, md.pixel_size_y) == tuple(far.camera.pixel_size)
    assert images == [image]


def test_acquire_image_sets_up_the_channel_on_the_far_side(served):
    far, fm = served
    channel = ChannelSettings(
        name="GFP",
        color="green",
        excitation_wavelength=450,
        emission_wavelength=None,
        power=0.2,
        exposure_time=0.01,
    )
    image = fm.acquire_image(channel)
    assert far.light_source.power == 0.2
    assert far.filter_set.excitation_wavelength == 450
    assert far.camera.exposure_time == 0.01
    md = image.metadata.channels[0]
    assert (md.name, md.color) == ("GFP", "green")  # this session's labels
    assert (md.power, md.excitation_wavelength, md.exposure_time) == (0.2, 450, 0.01)
    assert isinstance(image.data, np.ndarray) and image.data.ndim == 2


def _counting_requests(fm):
    """Record each request the FM API sends to the FM's computer."""
    sent = []
    request = fm.client.request

    def counted(method, path, *args, **kwargs):
        sent.append((method, path))
        return request(method, path, *args, **kwargs)

    fm.client.request = counted
    return sent


def test_a_frame_is_one_request_with_its_metadata(served):
    far, fm = served
    channel = ChannelSettings(
        name="GFP", excitation_wavelength=450, power=0.2, exposure_time=0.01
    )
    sent = _counting_requests(fm)
    image = fm.acquire_image(channel)
    assert sent == [("POST", "devices/fm/commands/acquire_frame")]
    md = image.metadata
    ch = md.channels[0]
    assert (ch.power, ch.excitation_wavelength, ch.exposure_time) == (0.2, 450, 0.01)
    assert ch.emission_wavelength == far.filter_set.emission_wavelength
    assert (ch.gain, ch.offset, ch.binning) == (
        far.camera.gain,
        far.camera.offset,
        far.camera.binning,
    )
    assert ch.objective_position == far.objective.position
    assert (md.pixel_size_x, md.pixel_size_y) == tuple(far.camera.pixel_size)
    assert md.resolution == tuple(far.camera.resolution)

    sent.clear()
    fm.acquire_image()  # the current settings, as live view takes each frame
    assert sent == [("POST", "devices/fm/commands/acquire_frame")]


def test_a_server_without_acquire_frame_still_acquires(served):
    """A METEOR PC on an older fibsem: the frame alone, metadata read as before."""
    far, fm = served
    fm.devices["fm"].server_commands -= {"acquire_frame"}
    sent = _counting_requests(fm)
    image = fm.acquire_image()
    assert sent[0] == ("POST", "devices/fm/commands/acquire_channel")
    assert len(sent) > 1  # the metadata, read live
    assert image.metadata.channels[0].exposure_time == far.camera.exposure_time


def test_live_acquisition_streams_frames(served):
    _, fm = served
    fm._rate_limit = 0
    images = []
    fm.acquisition_signal.connect(images.append)
    fm.start_acquisition()
    end = time.monotonic() + 5
    while len(images) < 3 and time.monotonic() < end:
        time.sleep(0.01)
    fm.stop_acquisition()
    assert len(images) >= 3


def test_live_view_is_pulled_from_the_far_side(served):
    """Live view runs on the FM's computer; each frame is one request for the next."""
    far, fm = served
    fm._rate_limit = 0
    images = []
    fm.acquisition_signal.connect(images.append)
    sent = _counting_requests(fm)
    channel = ChannelSettings(excitation_wavelength=450, power=0.3, exposure_time=0.01)
    fm.start_acquisition(channel)
    end = time.monotonic() + 5
    while len(images) < 3 and time.monotonic() < end:
        time.sleep(0.01)
    fm.stop_acquisition()
    assert len(images) >= 3
    assert far.light_source.power == 0.3  # set up on the far side
    commands = [path.rsplit("/", 1)[-1] for _, path in sent if "/commands/" in path]
    assert commands[0] == "start_live" and commands[-1] == "stop_live"
    assert set(commands[1:-1]) == {"acquire_frame"}


def test_far_side_live_view_stops_when_the_viewer_goes(served):
    """A client that disappears mid-live (a crash, a dropped link) leaves the light
    on only until the FM's own watchdog notices nobody is asking."""
    _, fm = served
    group = fm.devices["fm"]
    far_group = _far_group(fm)
    far_group.live_timeout = 0.2
    group.start_live()
    assert far_group.is_live
    deadline = time.monotonic() + 3
    while far_group.is_live and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not far_group.is_live


def test_a_server_without_live_view_is_still_pulled(served):
    _, fm = served
    group = fm.devices["fm"]
    group.server_commands -= {"start_live", "stop_live"}
    fm._rate_limit = 0
    images = []
    fm.acquisition_signal.connect(images.append)
    fm.start_acquisition()
    end = time.monotonic() + 5
    while len(images) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    fm.stop_acquisition()
    assert len(images) >= 2
    assert not _far_group(fm).is_live


def test_a_guard_read_fails_closed_when_the_fm_computer_is_gone(served):
    _, fm = served
    fm.client._session.close()
    fm.client.base_url = "http://127.0.0.1:1"  # nothing listens there
    with pytest.raises(RemoteDeviceUnreachable):
        fm.objective.state
    with pytest.raises(RemoteDeviceUnreachable):
        fm.acquire_image()


def test_connect_names_a_server_that_serves_no_fm():
    from fibsem.server.devices import demo_devices

    server = DeviceServer(demo_devices()).start()
    try:
        with pytest.raises(RuntimeError, match="serves no FM"):
            RemoteFluorescenceMicroscope.connect("127.0.0.1", server.port)
    finally:
        server.stop()


def test_a_configuration_naming_a_remote_fm_connects_to_it(tmp_path):
    """`fm.driver: remote` on a METEOR-style offset system: the microscope's FM is
    the served one, and its images carry the stage position like a local FM's."""
    import os

    import fibsem.config as cfg
    from fibsem import utils

    server = DeviceServer(demo_fm_devices()).start()
    try:
        settings = utils.load_yaml(
            os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
        )
        utils.configuration_device(settings, "fm").update(
            driver="remote", address="127.0.0.1", port=server.port
        )
        settings["sim"]["has_fm"] = False  # no FM on the beams' connection
        path = tmp_path / "remote-fm-configuration.yaml"
        utils.save_yaml(path, settings)

        microscope, _ = utils.setup_session(config_path=str(path))
        assert isinstance(microscope.fm, RemoteFluorescenceMicroscope)
        assert microscope.fm.parent is microscope
        microscope.fm.objective.insert()
        image = microscope.fm.acquire_image()
        assert image.metadata.stage_position == microscope.get_stage_position()
        microscope.fm.client.close()
    finally:
        server.stop()


# ── a server that starts after the microscope (FIB-1086) ─────────────


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(condition, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_without_offline_an_unserved_fm_refuses_to_connect():
    with pytest.raises(RemoteDeviceUnreachable):
        RemoteFluorescenceMicroscope.connect("127.0.0.1", _free_port())


def test_an_fm_connected_offline_fails_closed_then_comes_online_by_itself():
    port = _free_port()
    client = DeviceClient("127.0.0.1", port, heartbeat=0.5)
    fm = RemoteFluorescenceMicroscope.connect(
        "127.0.0.1", port, client=client, offline=True
    )
    came_online = []
    client.reconnected.connect(lambda: came_online.append(True))
    server = None
    try:
        assert not fm.online
        assert fm.devices["objective"].parameters == {}  # absent until it answers
        with pytest.raises(RemoteDeviceUnreachable):
            fm.objective.state  # the guard's read fails closed
        with pytest.raises(RemoteDeviceUnreachable):
            fm.acquire_image()

        local = {d.name: d for d in demo_fm_devices()}
        server = DeviceServer(local.values(), port=port).start()

        assert _wait_for(lambda: fm.online and client.connected, timeout=5)
        assert came_online == [True]
        assert fm.objective.state == "Retracted"
        power = fm.devices["light_source"].power
        assert power.cached == local["light_source"].power.get_value()  # primed
        fm.set_power(0.3)
        assert local["light_source"].power.get_value() == 0.3
        assert isinstance(fm.acquire_image(), FluorescenceImage)
    finally:
        client.close()
        if server is not None:
            server.stop()


def test_a_configured_fm_comes_online_after_the_microscope_with_its_calibration(
    tmp_path,
):
    """The beams connect without waiting for the FM's PC. When its server starts,
    the FM binds, and the configured objective limit is pushed to it."""
    import os

    import fibsem.config as cfg
    from fibsem import utils

    port = _free_port()
    settings = utils.load_yaml(
        os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    utils.configuration_device(settings, "fm").update(
        driver="remote", address="127.0.0.1", port=port
    )
    settings.setdefault("calibration", {})["objective"] = {"limit_position": 0.004}
    path = tmp_path / "remote-fm-configuration.yaml"
    utils.save_yaml(path, settings)

    microscope, _ = utils.setup_session(config_path=str(path))
    fm = microscope.fm
    server = None
    try:
        assert isinstance(fm, RemoteFluorescenceMicroscope)
        assert not fm.online
        microscope.get_stage_position()  # the beams' side works meanwhile

        local = {d.name: d for d in demo_fm_devices()}
        server = DeviceServer(local.values(), port=port).start()

        far = FluorescenceMicroscope(local)
        assert _wait_for(lambda: far.objective.limit_position == 0.004, timeout=10)
        assert fm.online
        assert fm.objective.limit_position == 0.004
    finally:
        fm.client.close()
        if server is not None:
            server.stop()
