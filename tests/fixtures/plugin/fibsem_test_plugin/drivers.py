"""Fixture microscope drivers.

Each function returns the record fibsem registers. The fixture driver reuses the
demo microscope class, since what is under test is the registration, not a driver.
"""

from fibsem.drivers.registry import DriverEntry
from fibsem_test_plugin import DRIVER_MANUFACTURER


def fixture_driver() -> DriverEntry:
    return DriverEntry(
        DRIVER_MANUFACTURER,
        "fibsem.drivers.demo.microscope:DemoMicroscope",
        config={"ion-column-tilt": 54, "electron-column-tilt": 0},
    )


def clashing_driver() -> DriverEntry:
    return DriverEntry("Demo", "fibsem_test_plugin.drivers:NotADriver")


def not_a_record() -> str:
    return "fibsem.drivers.demo.microscope:DemoMicroscope"
