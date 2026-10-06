"""The Tescan backend builds its devices from its device entries (``fibsem.devices.entries``).

Its beams and stage are the devices it builds by itself; the configuration's
``hardware.devices`` list is an overlay on them, as on the Demo and Thermo. Runs over the
fake SharkSEM connection (``tests/fixtures/tescan_sdk.py``).
"""

import logging
import os

import pytest

import fibsem.config as cfg
from fibsem import manufacturers, utils
from fibsem.devices.drivers.tescan import TescanBeam, TescanStage
from fibsem.devices.entries import DeviceBuildError
from fibsem.microscopes.registry import device_builder
from fibsem.structures import BeamType, DeviceEntry
from tests.fixtures.tescan_sdk import connect


def _system(*entries):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.other_devices = [DeviceEntry.from_dict(e) for e in entries]
    return system


def test_the_tescan_driver_has_builders_for_its_beams_and_stage():
    for device_type in ("beam", "stage"):
        assert device_builder(manufacturers.TESCAN, device_type) is not None
    assert device_builder(manufacturers.TESCAN, "chamber") is None


def test_connect_builds_both_beams_and_the_stage(monkeypatch):
    microscope, _ = connect(monkeypatch, _system())

    assert list(microscope.devices) == ["electron", "ion", "stage"]
    assert isinstance(microscope.beams[BeamType.ION], TescanBeam)
    assert isinstance(microscope.stage, TescanStage)
    assert microscope.stage.name == "stage"


def test_a_disabled_column_or_stage_is_not_built(monkeypatch):
    system = _system()
    system.electron.enabled = False
    system.stage.enabled = False
    microscope, _ = connect(monkeypatch, system)

    assert list(microscope.devices) == ["ion"]
    assert microscope.stage is None
    assert set(microscope.beams) == {BeamType.ION}


def test_a_device_tescan_has_no_builder_for_is_skipped_with_a_warning(
    monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING):
        microscope, _ = connect(
            monkeypatch, _system({"name": "laser", "type": "laser"})
        )

    assert "laser" not in microscope.devices
    assert "driver 'Tescan' has no builder for a 'laser' device" in caplog.text


def test_a_required_device_it_cannot_build_fails_connect(monkeypatch):
    with pytest.raises(DeviceBuildError, match="'laser'"):
        connect(
            monkeypatch,
            _system({"name": "laser", "type": "laser", "required": True}),
        )
