"""A manipulator on its own controller, built by a driver that only builds devices.

The vendor backend does not know it is there, so it would report no manipulator. A
`hardware.devices` entry naming the device's driver builds it next to the vendor's
devices, and the microscope then reports a manipulator, and drives it, whatever the
backend's own answer was. Runs over the Tescan fake SDK (``tests/fixtures/tescan_sdk``),
whose backend reports no manipulator, and on the Demo, whose own one it replaces.
"""

import os
from copy import deepcopy

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.manipulator import Manipulator
from fibsem.microscopes import registry
from fibsem.microscopes.registry import DeviceBuilder, DriverEntry, register_driver
from fibsem.structures import (
    DeviceEntry,
    FibsemManipulatorPosition,
    InsertableDeviceState,
)
from tests.fixtures.tescan_sdk import connect

DRIVER = "acme-needles"
ENTRY = {"name": "manipulator", "type": "manipulator", "driver": DRIVER}
PARK = FibsemManipulatorPosition(x=1e-6, y=2e-6, z=3e-6)


class ToyManipulator(Manipulator):
    """A needle that keeps its position in memory, as a controller would."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.sim_position = FibsemManipulatorPosition()
        self.sim_inserted = False

    def read_position(self):
        return deepcopy(self.sim_position)

    def read_state(self):
        if self.sim_inserted:
            return InsertableDeviceState.INSERTED
        return InsertableDeviceState.RETRACTED

    def named_positions(self):
        return ["PARK"]

    def saved_position(self, name):
        if name != "PARK":
            raise ValueError(name)
        return deepcopy(PARK)

    def _insert(self, name):
        self.sim_position = self.saved_position(name)
        self.sim_inserted = True

    def _retract(self):
        self.sim_position = FibsemManipulatorPosition()
        self.sim_inserted = False

    def _move_absolute(self, position):
        self.sim_position = deepcopy(position)

    def _move_relative(self, delta):
        self.sim_position = self.sim_position + delta


def build_toy_manipulator(entry, context):
    device = ToyManipulator(context.microscope)
    device.name = entry.name
    return device.connect()


@pytest.fixture(autouse=True)
def needle_driver():
    saved = dict(registry.DRIVER_PLUGINS.registered)
    register_driver(
        DriverEntry(
            DRIVER,
            devices={"manipulator": DeviceBuilder(f"{__name__}:build_toy_manipulator")},
        )
    )
    yield
    registry.DRIVER_PLUGINS.registered.clear()
    registry.DRIVER_PLUGINS.registered.update(saved)


def _tescan(monkeypatch, *entries):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.other_devices = [DeviceEntry.from_dict(e) for e in entries]
    return connect(monkeypatch, system)


def test_a_backend_without_a_manipulator_reports_the_one_configured(monkeypatch):
    microscope, _ = _tescan(monkeypatch, ENTRY)

    assert isinstance(microscope.manipulator_device, ToyManipulator)
    assert microscope.is_available("manipulator") is True
    assert microscope.capability_sources["manipulator"] == "device"
    assert microscope.manipulator_named_positions() == ["PARK"]


def test_the_configured_manipulator_is_the_one_driven(monkeypatch):
    microscope, fake = _tescan(monkeypatch, ENTRY)

    assert microscope.insert_manipulator("PARK") == PARK
    assert microscope.get_manipulator_state() is True
    delta = FibsemManipulatorPosition(x=1e-6)
    assert microscope.move_manipulator_relative(delta) == PARK + delta
    microscope.retract_manipulator()
    assert microscope.get_manipulator_state() is False
    assert not [path for path, _, _ in fake.log if "Nanomanipulator" in path]


def test_without_the_entry_the_backend_still_reports_none(monkeypatch):
    microscope, _ = _tescan(monkeypatch)

    assert microscope.manipulator_device is None
    assert microscope.is_available("manipulator") is False


def test_the_entry_replaces_the_demo_manipulator():
    system = utils.load_microscope_configuration(None, None).system
    system.info.manufacturer = "Demo"
    system.other_devices = [DeviceEntry.from_dict(ENTRY)]
    microscope = registry.connect_microscope(system)

    assert isinstance(microscope.manipulator_device, ToyManipulator)
    assert microscope.is_available("manipulator") is True


def test_it_is_fitted_where_the_instrument_says_there_is_none():
    """sim-arctis's simulated instrument has no manipulator, as a Thermo whose xT
    reports none; the configured one is fitted all the same."""
    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    system = utils.load_microscope_configuration(path).system
    system.info.manufacturer = "Demo"
    system.other_devices = [DeviceEntry.from_dict(ENTRY)]
    microscope = registry.connect_microscope(system)
    try:
        assert microscope.capability_sources["manipulator"] == "device"
        assert microscope.is_available("manipulator") is True
    finally:
        microscope.disconnect()
