"""Fixture microscope drivers.

Each function returns the record fibsem registers. The fixture driver reuses the
demo microscope class, since what is under test is the registration, not a driver.
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry
from fibsem_test_plugin import DEVICE_DRIVER, DRIVER_MANUFACTURER


def fixture_driver() -> DriverEntry:
    return DriverEntry(
        DRIVER_MANUFACTURER,
        "fibsem.drivers.demo.microscope:DemoMicroscope",
    )


def fixture_device_driver() -> DriverEntry:
    """A driver that builds one device and no microscope, as a scan generator's
    plugin does. It builds the Demo's scan generator."""
    return DriverEntry(
        DEVICE_DRIVER,
        devices={
            "scan_generator": DeviceBuilder(
                "fibsem.drivers.demo.devices:build_demo_scan_generator"
            )
        },
        setup="fibsem_test_plugin.setup:SCANNER",
    )


def clashing_driver() -> DriverEntry:
    return DriverEntry("Demo", "fibsem_test_plugin.drivers:NotADriver")


def not_a_record() -> str:
    return "fibsem.drivers.demo.microscope:DemoMicroscope"
