"""Today's FM API over a simulated FM served on localhost: the calls the FM UI and
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
from fibsem.fm.microscope import FluorescenceMicroscope  # noqa: E402
from fibsem.fm.remote import RemoteFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import ChannelSettings, FluorescenceImage  # noqa: E402
from fibsem.server.devices import DeviceServer, demo_fm_devices  # noqa: E402


@pytest.fixture
def served():
    local = {d.name: d for d in demo_fm_devices()}
    server = DeviceServer(local.values()).start()
    client = DeviceClient("127.0.0.1", server.port, heartbeat=0.5)
    fm = RemoteFluorescenceMicroscope.connect("127.0.0.1", server.port, client=client)
    # The simulated FM behind the served devices, as its computer would hold it.
    far = local["fm"]._fm
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
    fm.filter_set.emission_wavelength = None
    assert far.filter_set.emission_wavelength is None


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
        settings["hardware"]["fm"].update(
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
