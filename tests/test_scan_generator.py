"""A scan generator: a Scanner a beam images through when its `scanner` role is bound.

The Demo has a simulated one (`type: scan_generator`, `driver: demo`), and binds a
beam's role from the beam's entry (`roles: {scanner: scan_generator}`), so the binding
is tested without hardware.
"""

import numpy as np
import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.drivers.demo import DemoScanGenerator
from fibsem.devices.entries import ResolvedEntry, RoleBindingError, bind_device_roles
from fibsem.devices.scanner import Scanner
from fibsem.structures import BeamType, DeviceEntry, ImageSettings

SCAN_GENERATOR = {"name": "scan_generator", "type": "scan_generator", "driver": "demo"}


def _demo_with(tmp_path, *entries, roles=None):
    """The Demo, with *entries* added and *roles* on the electron beam's entry."""
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    devices = config["hardware"]["devices"]
    if roles is not None:
        electron = next((e for e in devices if e.get("name") == "electron"), None)
        if electron is None:
            electron = {"name": "electron"}
            devices.append(electron)
        electron["roles"] = roles
    devices.extend(entries)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    return microscope


@pytest.fixture
def bound(tmp_path):
    microscope = _demo_with(
        tmp_path, SCAN_GENERATOR, roles={"scanner": "scan_generator"}
    )
    yield microscope
    microscope.disconnect()


def test_the_demo_builds_a_scan_generator_from_its_entry(tmp_path):
    microscope = _demo_with(tmp_path, SCAN_GENERATOR)

    device = microscope.devices["scan_generator"]
    assert isinstance(device, Scanner)
    assert {"acquire", "stop"} <= set(device.commands)
    # Unbound, the beam images through the vendor's scan as before.
    assert "scanner" not in microscope.beams[BeamType.ELECTRON].roles


def test_a_bound_beam_images_through_the_scan_generator(bound):
    scanner = bound.devices["scan_generator"]
    sem = bound.beams[BeamType.ELECTRON]
    assert sem.scanner is scanner
    assert "scanner" not in bound.beams[BeamType.ION].roles

    settings = ImageSettings(
        resolution=(64, 48), dwell_time=1e-7, hfw=80e-6, beam_type=BeamType.ELECTRON
    )
    image = bound.acquire_image(settings)

    assert scanner.sim_frames == 1
    assert image.data.shape == (48, 64)
    assert image.metadata.image_settings.resolution == (64, 48)
    assert image.metadata.pixel_size.x == pytest.approx(80e-6 / 64)
    assert sem.hfw.get_value() == pytest.approx(80e-6)


def test_without_settings_it_scans_with_the_beams_own(bound):
    sem = bound.beams[BeamType.ELECTRON]
    resolution = tuple(sem.resolution.get_value())

    image = sem.acquire()

    assert image.data.shape == (resolution[1], resolution[0])
    assert image.metadata.image_settings.dwell_time == sem.dwell_time.get_value()


def test_live_view_runs_on_the_scan_generator_and_stops_it(bound):
    sem = bound.beams[BeamType.ELECTRON]
    scanner = bound.devices["scan_generator"]
    frames = []
    sem.live_frame.connect(frames.append)

    sem.start_live()
    deadline = 200
    while scanner.sim_frames < 2 and deadline:
        deadline -= 1
        sem._live_stop.wait(0.01)
    sem.stop_live()

    assert scanner.sim_frames >= 2
    assert frames
    assert not sem.is_live


def test_a_scanner_refuses_a_frame_it_cannot_scan():
    scanner = DemoScanGenerator("sg").connect()

    with pytest.raises(ValueError, match="resolution"):
        scanner.acquire((0, 10), 1e-6)
    with pytest.raises(ValueError, match="dwell"):
        scanner.acquire((10, 10), 0)
    assert scanner.acquire((10, 8), 1e-9).shape == (8, 10)


# -- binding --------------------------------------------------------------------


def _resolved(*entries):
    return [ResolvedEntry(DeviceEntry.from_dict(e), "Demo") for e in entries]


def test_binding_to_a_device_that_is_not_a_scanner_fails(tmp_path):
    with pytest.raises(RoleBindingError, match="takes a Scanner"):
        _demo_with(tmp_path, roles={"scanner": "stage"})


def test_binding_to_a_name_no_entry_has_fails(tmp_path):
    with pytest.raises(RoleBindingError, match="no device is named 'nope'"):
        _demo_with(tmp_path, roles={"scanner": "nope"})


def test_binding_a_role_the_beam_does_not_have_fails(tmp_path):
    with pytest.raises(RoleBindingError, match="no 'detector' role"):
        _demo_with(tmp_path, SCAN_GENERATOR, roles={"detector": "scan_generator"})


def test_a_switched_off_scan_generator_leaves_the_beam_on_its_own_scan(tmp_path):
    microscope = _demo_with(
        tmp_path,
        {**SCAN_GENERATOR, "enabled": False},
        roles={"scanner": "scan_generator"},
    )

    assert "scan_generator" not in microscope.devices
    sem = microscope.beams[BeamType.ELECTRON]
    assert "scanner" not in sem.roles
    assert sem.acquire().data.size


def test_bindings_that_form_a_cycle_are_refused():
    resolved = _resolved(
        {"name": "a", "type": "chamber", "roles": {"scanner": "b"}},
        {"name": "b", "type": "scan_generator", "roles": {"x": "a"}},
    )
    with pytest.raises(RoleBindingError, match="cycle: a -> b -> a"):
        bind_device_roles(resolved, {})


def test_a_device_cannot_fill_its_own_role():
    resolved = _resolved({"name": "a", "type": "chamber", "roles": {"scanner": "a"}})
    with pytest.raises(RoleBindingError, match="cycle"):
        bind_device_roles(resolved, {})


def test_the_bound_role_survives_a_save(tmp_path):
    microscope = _demo_with(
        tmp_path, SCAN_GENERATOR, roles={"scanner": "scan_generator"}
    )
    saved = microscope.system.to_dict()["hardware"]["devices"]

    electron = next(e for e in saved if e["name"] == "electron")
    assert electron["roles"] == {"scanner": "scan_generator"}
    assert any(e["name"] == "scan_generator" for e in saved)
    assert np.asarray(microscope.beams[BeamType.ELECTRON].acquire().data).ndim == 2
