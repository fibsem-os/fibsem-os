"""A device-only driver's guided setup: the devices it offers, the keys its entries
take, and what the wizard writes for them (gaps 8 and 10 of the plugin walkthrough)."""

import pytest

from fibsem import guided_setup as wizard
from fibsem.devices.entries import DeviceBuildError, ResolvedEntry, build_device_entries
from fibsem.drivers import registry
from fibsem.drivers.registry import (
    DeviceBuilder,
    DriverEntry,
    device_setup,
    device_setups,
    driver_setups,
    register_driver,
)
from fibsem.drivers.setup import SetupDevice, SetupKey
from fibsem.structures import DeviceEntry
from fibsem.utils import configuration_device

UNIT = SetupDevice(
    type="lock_in",
    label="Test lock-in",
    summary="A lock-in amplifier.",
    name="lockin",
    keys=(
        SetupKey("address", "127.0.0.1", "Address"),
        SetupKey("gain", 1.0),
        SetupKey("channels", 2),
        SetupKey("simulate", False),
    ),
    role=("electron", "detector"),
)

BUILT = []


def build_unit(entry, context):
    BUILT.append(dict(entry.options))
    return None


@pytest.fixture
def unit_driver():
    saved = dict(registry.DRIVER_PLUGINS.registered)
    register_driver(
        DriverEntry(
            "TestUnit",
            devices={"lock_in": DeviceBuilder(f"{__name__}:build_unit")},
            setup=f"{__name__}:UNIT",
        )
    )
    BUILT.clear()
    yield
    registry.DRIVER_PLUGINS.registered.clear()
    registry.DRIVER_PLUGINS.registered.update(saved)


def test_the_keys_fill_their_defaults_and_refuse_what_they_do_not_take():
    assert UNIT.options({"gain": 3}) == {
        "address": "127.0.0.1",
        "gain": 3.0,
        "channels": 2,
        "simulate": False,
    }
    with pytest.raises(ValueError, match=r"unknown keys .*\['port'\]"):
        UNIT.options({"port": 1})
    with pytest.raises(ValueError, match="'channels' takes a whole number"):
        UNIT.options({"channels": "two"})
    with pytest.raises(ValueError, match="'channels' takes a whole number"):
        UNIT.options({"channels": True})
    with pytest.raises(ValueError, match="'simulate' takes true or false"):
        UNIT.options({"simulate": 1})


def test_a_device_only_driver_offers_its_device_and_no_manufacturer(unit_driver):
    assert ("TestUnit", UNIT) in device_setups()
    assert "TestUnit" not in [name for name, _ in driver_setups()]
    assert device_setup("TestUnit", "lock_in") is UNIT
    assert device_setup("TestUnit", "stage") is None
    assert device_setup("NotInstalled", "lock_in") is None
    assert device_setup("Demo", "beam") is None


def _resolved(**options):
    entry = DeviceEntry.from_dict(
        {"name": "lockin", "type": "lock_in", "driver": "TestUnit", **options}
    )
    return ResolvedEntry(entry, "TestUnit")


def test_building_an_entry_checks_its_keys_first(unit_driver):
    build_device_entries([_resolved(gain=2)], None)
    assert BUILT == [
        {"address": "127.0.0.1", "gain": 2.0, "channels": 2, "simulate": False}
    ]
    with pytest.raises(DeviceBuildError, match="its keys are wrong: unknown keys"):
        build_device_entries([_resolved(required=True, port=7)], None)
    assert len(BUILT) == 1


def test_the_wizard_writes_a_chosen_device_and_its_role(unit_driver):
    offered = wizard.get_offered_device("TestUnit:lock_in")
    assert offered is not None and offered.setup is UNIT

    choices = wizard.SetupChoices()
    choices.devices = {offered.key: wizard.DeviceChoice({"gain": 4.0})}
    config = wizard.build_configuration(choices)
    assert configuration_device(config, "lockin") == {
        "name": "lockin",
        "type": "lock_in",
        "driver": "TestUnit",
        "address": "127.0.0.1",
        "gain": 4.0,
        "channels": 2,
        "simulate": False,
    }
    assert configuration_device(config, "electron")["roles"] == {"detector": "lockin"}
    assert wizard.chosen_device_rows(choices) == [
        (
            "Test lock-in",
            "address: 127.0.0.1, gain: 4.0, channels: 2, simulate: False; "
            "electron's detector",
        )
    ]

    choices.devices = {offered.key: wizard.DeviceChoice(bind_role=False)}
    config = wizard.build_configuration(choices)
    assert "roles" not in configuration_device(config, "electron")


def test_a_device_no_longer_installed_is_left_out(unit_driver):
    choices = wizard.SetupChoices()
    choices.devices = {"Gone:lock_in": wizard.DeviceChoice()}
    config = wizard.build_configuration(choices)
    names = [entry["name"] for entry in config["hardware"]["devices"]]
    assert "lockin" not in names
